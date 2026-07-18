"""
Utility functions for UGRC centerline cleaning and attribute transfer.

Design contract
---------------
- Filtering / data selection lives in the calling notebook.
- This module only accepts pre-filtered GeoDataFrames and implements the
  mechanics of dissolve, split, and attribute transfer.
- All functions are pure (no file I/O, no side effects).

Public API
----------
    build_junction_split_points(all_segments_gdf) -> GeoDataFrame
        Points at branch/junction locations (join_count >= 3), matched by a
        1 m search radius -- arcpy's SpatialJoin DOES honour search_radius
        under the INTERSECT match option (verified against Esri docs), so
        this is a distance-based join, not exact coincidence.
        Input must be the COMBINED freeway + surface GeoDataFrame across all
        VERT_LEVELs — matching arcpy FL_All which includes both classes.

    dissolve_and_singlepart(gdf) -> GeoDataFrame
        Topologically merge connected lines into maximal chains, then explode
        to singlepart LineStrings. Must be called per VERT_LEVEL slice only.

    split_lines_at_points(lines_gdf, points_gdf, search_radius_m) -> GeoDataFrame
        Split each line wherever a point falls within search_radius_m.

    rcl_merge(fwy_levels, surf_levels) -> GeoDataFrame
        Full RCL merge pipeline. Both freeway and surface levels are dissolved
        and split at junction points independently, then concatenated.
        Output includes MIDX, MIDY, and MidPointID columns.
            fwy_levels: dict with keys 'lvl0', 'lvl1', 'lvl2', 'lvl3', 'all'
            surf_levels: dict with keys 'lvl0', 'lvl1', 'lvl2', 'lvl3'

    transfer_attributes_by_midpoint(cleaned_gdf, original_gdf, cols) -> GeoDataFrame
        For each cleaned geometry compute its midpoint, nearest-join to
        original_gdf, and append cols. Returns cleaned_gdf with cols appended.

    restitch_vert_level_transitions(gdf_fulldissolve, chain_level_col,
            gdf_centerlines, source_level_col, identity_cols, exclude_points,
            point_tolerance=0.05) -> GeoDataFrame
        Reconnect chains dissolved separately per VERT_LEVEL across their own
        level boundary when they are the same physical route continuing
        through an elevation change (e.g. a freeway climbing onto its own
        flyover), without reintroducing the GEOS cross-line-crossing noding
        that dissolving across all levels in one pass would cause. Returns
        geometry only; result.attrs carries match/merge counts.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import linemerge, snap, split, unary_union
from shapely.strtree import STRtree


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_endpoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of start/end points for every LineString in gdf."""
    pts: list[Point] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            if coords:
                pts += [Point(coords[0]), Point(coords[-1])]
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                coords = list(part.coords)
                if coords:
                    pts += [Point(coords[0]), Point(coords[-1])]
    return gpd.GeoDataFrame(geometry=pts, crs=gdf.crs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_junction_split_points(all_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract both-end vertices from all segments (fwy + surface, all levels)
    and return every vertex occurrence whose Join_Count (self-count of
    vertices within a 1 m search radius, inclusive of itself) is >= 3.

    Equivalent to arcpy FeatureVerticesToPoints(BOTH_ENDS) + SpatialJoin(self,
    search_radius="1 Meters") + filter Join_Count >= 3. SpatialJoin's
    match_option is left at its default (INTERSECT), and per Esri's Spatial
    Join documentation, search_radius IS honoured under INTERSECT: "A search
    radius is only valid when Match Option is Intersect, Within a Distance,
    Within a Distance (Geodesic), Have Their Center In, Closest, or Closest
    (Geodesic)." (An earlier iteration of this function assumed INTERSECT
    ignored search_radius and used exact-coincidence matching instead; that
    was incorrect and has been reverted in favour of the documented
    distance-based behaviour.)

    Output intentionally is NOT deduplicated to one point per location: arcpy's
    join_operation defaults to JOIN_ONE_TO_ONE, so FWY_FV2PEndpointsCount
    retains one row per input vertex occurrence (with a Join_Count field),
    and the Join_Count >= 3 filter is applied on that per-occurrence table.
    split_lines_at_points already deduplicates by on-line projected position,
    so passing multiple near-duplicate points at/near the same junction is
    harmless downstream.
    """
    vertices = _extract_endpoints(all_segments_gdf)
    if vertices.empty:
        return vertices

    geoms = vertices.geometry.values
    tree = STRtree(geoms)
    query_idx, _ = tree.query(geoms, predicate="dwithin", distance=1.0)
    join_counts = np.bincount(query_idx, minlength=len(geoms))

    return vertices[join_counts >= 3].reset_index(drop=True)


def dissolve_and_singlepart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Topologically merge all connected lines into maximal chains, then explode
    to singlepart LineStrings.

    Equivalent to arcpy Dissolve(MULTI_PART) + MultipartToSinglepart.
    arcpy's Dissolve without dissolve fields performs a full topological merge
    of all touching/connected line segments into maximal chains (not merely a
    multipart collection), which linemerge(unary_union(...)) replicates exactly.

    IMPORTANT: call this only on a single VERT_LEVEL slice at a time.
    Applying unary_union across multiple vertical levels would incorrectly merge
    at-grade roads with elevated structures sharing endpoint coordinates.
    Per-level isolation is enforced by rcl_merge, which calls this function
    once per level dict entry.
    """
    union = unary_union(gdf.geometry.tolist())
    # linemerge requires a collection; unary_union may return a bare LineString
    lines = list(union.geoms) if hasattr(union, "geoms") else [union]
    merged = linemerge(lines)
    result = gpd.GeoDataFrame(geometry=[merged], crs=gdf.crs)
    return result.explode(index_parts=False).reset_index(drop=True)


def split_lines_at_points(
    lines_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    search_radius_m: float,
) -> gpd.GeoDataFrame:
    """
    Split each line in lines_gdf wherever a point in points_gdf falls within
    search_radius_m metres.

    Strategy:
      1. For each line, collect all split-points within the search radius
         (inclusive — uses intersects on buffer, matching arcpy search_radius).
      2. Project each point onto the line; deduplicate by on-line position to
         prevent coincident projected points from breaking shapely split.
      3. Use shapely snap + split to cut the line at those locations.

    Equivalent to arcpy SplitLineAtPoint.
    """
    result_geoms: list = []
    n_failures = 0

    for line_geom in lines_gdf.geometry:
        nearby = points_gdf[
            points_gdf.geometry.intersects(line_geom.buffer(search_radius_m))
        ].geometry

        if nearby.empty:
            result_geoms.append(line_geom)
            continue

        # Project onto line and deduplicate by on-line distance (mm precision)
        seen: set[int] = set()
        splitter_pts: list[Point] = []
        for pt in nearby:
            dist_mm = round(line_geom.project(pt) * 1000)
            if dist_mm not in seen:
                seen.add(dist_mm)
                splitter_pts.append(line_geom.interpolate(line_geom.project(pt)))

        splitter = MultiPoint(splitter_pts)
        snapped_line = snap(line_geom, splitter, tolerance=search_radius_m)

        try:
            pieces = split(snapped_line, splitter)
            result_geoms.extend(list(pieces.geoms))
        except Exception:
            n_failures += 1
            result_geoms.append(line_geom)

    if n_failures:
        warnings.warn(
            f"split_lines_at_points: {n_failures} line(s) could not be split "
            "and were kept unsplit.",
            stacklevel=2,
        )

    result = gpd.GeoDataFrame(geometry=result_geoms, crs=lines_gdf.crs)
    return result[~result.is_empty].reset_index(drop=True)


def rcl_merge(
    fwy_levels: dict[str, gpd.GeoDataFrame],
    surf_levels: dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """
    Full RCL merge pipeline converting raw per-level centerline GDFs into a
    clean CandidateTDMRoadLinks dataset.

    Parameters
    ----------
    fwy_levels : dict
        Keys: 'lvl0', 'lvl1', 'lvl2', 'lvl3' (optional), 'all'
        'all' must contain ALL freeway/ramp/CD features across every level.
        Per-level keys hold pre-filtered GeoDataFrames for that level only.
        Levels with an empty GeoDataFrame are silently skipped.
    surf_levels : dict
        Keys: 'lvl0', 'lvl1', 'lvl2', 'lvl3' (optional)
        Each value is a pre-filtered GeoDataFrame of surface street lines
        for that level. Levels with an empty GeoDataFrame are silently skipped.

    Returns
    -------
    GeoDataFrame
        Merged and split centerlines with MIDX, MIDY, and MidPointID columns,
        suitable for subsequent attribute transfer and snapping.
        Matches the arcpy RCL2TDMGeometry output schema.

    Pipeline (mirrors arcpy RCLmerge exactly)
    ------------------------------------------
    1. Build junction split-points from the combined fwy + surface pool
       (all VERT_LEVELs) via a 1 m search-radius vertex self-join.
    2. Freeway levels: dissolve_and_singlepart -> split_lines_at_points
       Radii: lvl0=0.1 m, lvl1=1.0 m, lvl2=0.1 m, lvl3=0.1 m
    3. Surface levels: dissolve_and_singlepart -> split_lines_at_points
       Radii: lvl0=1.0 m, lvl1=0.1 m, lvl2=0.1 m, lvl3=0.1 m
    4. Concatenate all split parts.
    5. Compute MIDX, MIDY, MidPointID on the merged output.
    """
    base_crs = fwy_levels["all"].crs

    # Build split-points from the combined fwy + surface pool (all levels)
    all_surf = [v for v in surf_levels.values() if v is not None and not v.empty]
    all_segments = gpd.GeoDataFrame(
        pd.concat([fwy_levels["all"]] + all_surf, ignore_index=True),
        geometry="geometry",
        crs=base_crs,
    )
    split_points = build_junction_split_points(all_segments)

    _FWY_RADII  = {"lvl0": 0.1, "lvl1": 1.0, "lvl2": 0.1, "lvl3": 0.1}
    _SURF_RADII = {"lvl0": 1.0, "lvl1": 0.1, "lvl2": 0.1, "lvl3": 0.1}

    # Freeway levels: per-level dissolve -> split
    split_fwy_parts: list[gpd.GeoDataFrame] = []
    for lvl in ("lvl0", "lvl1", "lvl2", "lvl3"):
        gdf = fwy_levels.get(lvl)
        if gdf is None or gdf.empty:
            continue
        singlepart = dissolve_and_singlepart(gdf)
        split_fwy_parts.append(
            split_lines_at_points(singlepart, split_points, _FWY_RADII[lvl])
        )

    # Surface levels: per-level dissolve -> split (not merged before processing)
    split_surf_parts: list[gpd.GeoDataFrame] = []
    for lvl in ("lvl0", "lvl1", "lvl2", "lvl3"):
        gdf = surf_levels.get(lvl)
        if gdf is None or gdf.empty:
            continue
        singlepart = dissolve_and_singlepart(gdf)
        split_surf_parts.append(
            split_lines_at_points(singlepart, split_points, _SURF_RADII[lvl])
        )

    merged = pd.concat(split_fwy_parts + split_surf_parts, ignore_index=True)
    result = gpd.GeoDataFrame(merged, geometry="geometry", crs=base_crs)

    # Compute midpoint coordinates and MidPointID matching arcpy output
    midpts = result.geometry.interpolate(0.5, normalized=True)
    result["MIDX"] = midpts.x
    result["MIDY"] = midpts.y
    result["MidPointID"] = (
        result["MIDX"].astype(int).astype(str)
        + "|"
        + result["MIDY"].astype(int).astype(str)
    )

    return result


def transfer_attributes_by_midpoint(
    cleaned_gdf: gpd.GeoDataFrame,
    original_gdf: gpd.GeoDataFrame,
    cols: list[str],
) -> gpd.GeoDataFrame:
    """
    For each cleaned geometry compute its midpoint, find the nearest geometry
    in original_gdf (by sjoin_nearest), and append cols to cleaned_gdf.

    Because cleaned segments are derived from original lines, a midpoint of a
    cleaned segment reliably maps back to the one original segment whose geometry
    was used to generate that section.

    Parameters
    ----------
    cleaned_gdf : GeoDataFrame
        Output of rcl_merge — geometry + MIDX/MIDY/MidPointID columns.
    original_gdf : GeoDataFrame
        Raw source GeoDataFrame containing cols to transfer.
    cols : list[str]
        Columns from original_gdf to append to cleaned_gdf.

    Returns
    -------
    GeoDataFrame
        cleaned_gdf with cols appended.
    """
    midpoints = cleaned_gdf.copy()
    midpoints["geometry"] = midpoints.geometry.interpolate(0.5, normalized=True)

    orig_subset = original_gdf[["geometry"] + cols].copy()

    joined = gpd.sjoin_nearest(
        midpoints[["geometry"]],
        orig_subset,
        how="left",
        distance_col="_join_dist_m",
    )
    # sjoin_nearest can produce duplicates when equidistant; keep the first
    n_ties = int(joined.index.duplicated(keep="first").sum())
    if n_ties:
        warnings.warn(
            f"transfer_attributes_by_midpoint: {n_ties} midpoint(s) had an exact "
            "distance tie between two source features; kept the first match "
            "arbitrarily.",
            stacklevel=2,
        )
    joined = joined[~joined.index.duplicated(keep="first")]

    result = cleaned_gdf.copy()
    for col in cols:
        result[col] = joined[col].values
    return result


def restitch_vert_level_transitions(
    gdf_fulldissolve: gpd.GeoDataFrame,
    chain_level_col: str,
    gdf_centerlines: gpd.GeoDataFrame,
    source_level_col: str,
    identity_cols: tuple[str, ...],
    exclude_points: list[Point],
    point_tolerance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Reconnect per-VERT_LEVEL dissolved chains across their own level boundary
    when they are the same physical route continuing through an elevation
    change, e.g. a freeway mainline climbing onto its own flyover.

    dissolve_and_singlepart must be run once per VERT_LEVEL, never across all
    levels in one pass -- unary_union on LineStrings nodes every crossing
    point, not just shared endpoints, so a single global dissolve spuriously
    splits every elevated road against every at-grade street it merely
    passes over (verified empirically: it produces MORE chains than the raw
    ungrouped input, not fewer). That per-level isolation is what leaves a
    same-route elevation change looking like two separate chains touching at
    a single point -- this function undoes only that specific case.

    A restitch candidate is a coordinate shared by exactly two chain
    endpoints from two *different* `chain_level_col` values (a same-level
    2-way touch would already have been merged inside dissolve_and_singlepart,
    so this case only arises at a level boundary by construction; groups of
    size != 2, or where both entries share a level, are left alone). Each
    side's nearest gdf_centerlines row -- searched within its own
    `source_level_col` value only -- must agree, non-null, on every column in
    `identity_cols` (route identity, e.g. DOT_RTNAME); otherwise the touch is
    treated as an incidental different-route crossing and left split. Points
    within `point_tolerance` of any `exclude_points` entry (snapped
    model-node positions) are never restitched -- those must remain hard
    split boundaries regardless of identity.

    Chains connected through more than one level transition (e.g. a stacked
    interchange climbing through levels 0 -> 1 -> 2) are grouped via
    connected components and re-dissolved together.

    Parameters
    ----------
    gdf_fulldissolve : GeoDataFrame
        Output of dissolve_and_singlepart, concatenated across levels, with
        an added `chain_level_col` column recording which level each row was
        dissolved within.
    chain_level_col : str
        Column on gdf_fulldissolve holding each chain's dissolve level.
    gdf_centerlines : GeoDataFrame
        Source rows (e.g. CandidateTDMRoadLinks) used to look up route
        identity at each candidate point; must include source_level_col and
        every column in identity_cols.
    source_level_col : str
        Column on gdf_centerlines holding each row's VERT_LEVEL.
    identity_cols : tuple[str, ...]
        Columns that must agree (non-null) on both sides for a restitch.
    exclude_points : list[Point]
        Coordinates that must remain hard boundaries (snapped model nodes).
    point_tolerance : float
        Matching tolerance in metres for exclude_points and identity lookup.

    Returns
    -------
    GeoDataFrame
        geometry only, one row per (possibly restitched) chain.
        result.attrs: n_candidates, n_excluded, n_identity_mismatch, n_merged
        (count of restitch edges actually applied), n_components_merged
        (count of resulting multi-chain groups, <= n_merged when a group
        spans more than one transition).
    """
    n = len(gdf_fulldissolve)
    geoms = gdf_fulldissolve.geometry.tolist()
    levels = gdf_fulldissolve[chain_level_col].tolist()

    # ── Group chain endpoints by rounded coordinate ──────────────────────────
    endpoint_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for i, geom in enumerate(geoms):
        coords = list(geom.coords)
        start_key = (round(coords[0][0], 4), round(coords[0][1], 4))
        end_key = (round(coords[-1][0], 4), round(coords[-1][1], 4))
        endpoint_groups[start_key].append(i)
        endpoint_groups[end_key].append(i)

    candidates: list[tuple[tuple[float, float], int, int]] = []
    for key, idxs in endpoint_groups.items():
        if len(idxs) != 2:
            continue
        a, b = idxs
        if a == b or levels[a] == levels[b]:
            continue
        candidates.append((key, a, b))

    result_attrs = {
        "n_candidates": len(candidates),
        "n_excluded": 0,
        "n_identity_mismatch": 0,
        "n_merged": 0,
        "n_components_merged": 0,
    }

    if not candidates:
        result = gpd.GeoDataFrame(geometry=geoms, crs=gdf_fulldissolve.crs)
        result.attrs.update(result_attrs)
        return result

    # ── Exclude snapped model-node positions ─────────────────────────────────
    excl_tree = STRtree(exclude_points) if exclude_points else None

    def _is_excluded(pt: Point) -> bool:
        if excl_tree is None:
            return False
        for idx in excl_tree.query(pt.buffer(point_tolerance)):
            if exclude_points[idx].distance(pt) <= point_tolerance:
                return True
        return False

    kept: list[tuple[tuple[float, float], int, int]] = []
    for key, a, b in candidates:
        if _is_excluded(Point(key)):
            result_attrs["n_excluded"] += 1
        else:
            kept.append((key, a, b))

    # ── Look up route identity at each candidate point, per source level ─────
    merge_edges: list[tuple[int, int]] = []
    if kept:
        from_pts = gpd.GeoDataFrame(
            geometry=[Point(k) for k, a, b in kept], crs=gdf_fulldissolve.crs
        )
        nearest_by_level: dict = {}
        for lvl in pd.unique(gdf_centerlines[source_level_col]):
            sub = gdf_centerlines.loc[
                gdf_centerlines[source_level_col] == lvl,
                ["geometry", *identity_cols],
            ]
            if sub.empty:
                continue
            joined = gpd.sjoin_nearest(
                from_pts, sub, how="left", distance_col="_restitch_d"
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            nearest_by_level[lvl] = joined.reset_index(drop=True)

        for i, (key, a, b) in enumerate(kept):
            lvl_a, lvl_b = levels[a], levels[b]
            ra = nearest_by_level.get(lvl_a)
            rb = nearest_by_level.get(lvl_b)
            if ra is None or rb is None:
                result_attrs["n_identity_mismatch"] += 1
                continue
            ra_row, rb_row = ra.iloc[i], rb.iloc[i]
            same = all(
                pd.notna(ra_row[c]) and pd.notna(rb_row[c]) and ra_row[c] == rb_row[c]
                for c in identity_cols
            )
            if same:
                merge_edges.append((a, b))
            else:
                result_attrs["n_identity_mismatch"] += 1

    result_attrs["n_merged"] = len(merge_edges)

    # ── Group via connected components and re-dissolve each group ────────────
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(merge_edges)

    out_geoms = []
    for comp in nx.connected_components(G):
        if len(comp) == 1:
            out_geoms.append(geoms[next(iter(comp))])
        else:
            result_attrs["n_components_merged"] += 1
            subset = gdf_fulldissolve.iloc[list(comp)]
            merged = dissolve_and_singlepart(subset)
            out_geoms.extend(merged.geometry.tolist())

    result = gpd.GeoDataFrame(geometry=out_geoms, crs=gdf_fulldissolve.crs)
    result.attrs.update(result_attrs)
    return result
