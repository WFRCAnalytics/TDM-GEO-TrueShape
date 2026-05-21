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
    build_freeway_split_points(fwy_all_gdf) -> GeoDataFrame
        Points at freeway branch/junction locations (join_count >= 3).
        Used as split locations for per-level freeway processing.

    dissolve_and_singlepart(gdf) -> GeoDataFrame
        Dissolve all features into one geometry, then explode to singlepart
        LineStrings.

    split_lines_at_points(lines_gdf, points_gdf, search_radius_m) -> GeoDataFrame
        Split each line wherever a point falls within search_radius_m.

    rcl_merge(fwy_levels, surf_levels) -> GeoDataFrame
        Full RCL merge pipeline. Freeway levels are dissolved, split at junction
        points, then merged with dissolved surface roads.
            fwy_levels: dict with keys 'lvl0', 'lvl1', 'lvl2', 'lvl3', 'all'
            surf_levels: dict with keys 'lvl0', 'lvl1', 'lvl2', 'lvl3'

    transfer_attributes_by_midpoint(cleaned_gdf, original_gdf, cols) -> GeoDataFrame
        For each cleaned geometry compute its midpoint, nearest-join to
        original_gdf, and append cols. Returns cleaned_gdf with cols appended.
"""

from __future__ import annotations

import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from shapely.ops import snap, split


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

def build_freeway_split_points(fwy_all_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract both-end vertices from all freeway lines, self-join to count
    co-located vertices, and return those with join_count >= 3.

    These are the branch/junction nodes that force segment boundaries so that
    each dissolved freeway segment terminates at every interchange or merge.
    Equivalent to arcpy FeatureVerticesToPoints(BOTH_ENDS) + SpatialJoin(self,
    intersects, join_count >= 3).
    """
    vertices = _extract_endpoints(fwy_all_gdf)

    joined = gpd.sjoin(vertices, vertices, how="left", predicate="intersects")
    join_counts = joined.groupby(joined.index).size().rename("join_count")
    vertices = vertices.join(join_counts)

    return vertices[vertices["join_count"] >= 3].drop(columns="join_count").copy()


def dissolve_and_singlepart(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explode any MultiLineString features to singlepart LineStrings.

    Equivalent to arcpy Dissolve(MULTI_PART) + MultipartToSinglepart.
    arcpy's Dissolve with no dissolve fields and MULTI_PART=True groups all
    input features into one multipart record without changing geometry, then
    MultipartToSinglepart explodes them back.  The net effect is an identity
    operation that only flattens MultiLineStrings → LineStrings.

    NOTE: unary_union is intentionally NOT used here.  unary_union computes
    the true geometric union (GEOS noding + merging), which collapses interior
    pseudonodes and reduces the endpoint count by ~50%, breaking the snapping
    pool.  arcpy's Dissolve makes no such geometric change.
    """
    return gdf.explode(index_parts=False).reset_index(drop=True)


def split_lines_at_points(
    lines_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    search_radius_m: float,
) -> gpd.GeoDataFrame:
    """
    Split each line in lines_gdf wherever a point in points_gdf falls within
    search_radius_m metres.

    Strategy:
      1. For each line, collect all split-points within the search radius.
      2. Project each point onto the line to get the nearest on-line location.
      3. Use shapely snap + split to cut the line at those locations.

    Equivalent to arcpy SplitLineAtPoint.
    """
    result_geoms: list = []

    for line_geom in lines_gdf.geometry:
        nearby = points_gdf[
            points_gdf.geometry.within(line_geom.buffer(search_radius_m))
        ].geometry

        if nearby.empty:
            result_geoms.append(line_geom)
            continue

        splitter_pts = [
            line_geom.interpolate(line_geom.project(pt)) for pt in nearby
        ]
        splitter = MultiPoint(splitter_pts)
        snapped_line = snap(line_geom, splitter, tolerance=search_radius_m)

        try:
            pieces = split(snapped_line, splitter)
            result_geoms.extend(list(pieces.geoms))
        except Exception:
            result_geoms.append(line_geom)

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
        Each value is a pre-filtered GeoDataFrame of freeway/ramp/CD lines.
        Levels with an empty GeoDataFrame are silently skipped.
    surf_levels : dict
        Keys: 'lvl0', 'lvl1', 'lvl2', 'lvl3' (optional)
        Each value is a pre-filtered GeoDataFrame of surface street lines.
        Levels with an empty GeoDataFrame are silently skipped.

    Returns
    -------
    GeoDataFrame
        Geometry-only (no original attributes) merged and split centerlines
        suitable for subsequent attribute transfer and snapping.
    """
    split_points = build_freeway_split_points(fwy_levels["all"])

    # Split radii match original arcpy values per level
    _RADII = {"lvl0": 0.1, "lvl1": 1.0, "lvl2": 0.1, "lvl3": 0.1}

    split_fwy_parts: list[gpd.GeoDataFrame] = []
    base_crs = fwy_levels["all"].crs

    for lvl in ("lvl0", "lvl1", "lvl2", "lvl3"):
        gdf = fwy_levels.get(lvl)
        if gdf is None or gdf.empty:
            continue
        singlepart = dissolve_and_singlepart(gdf)
        split_fwy_parts.append(
            split_lines_at_points(singlepart, split_points, _RADII[lvl])
        )

    surf_parts: list[gpd.GeoDataFrame] = []
    for lvl in ("lvl0", "lvl1", "lvl2", "lvl3"):
        gdf = surf_levels.get(lvl)
        if gdf is not None and not gdf.empty:
            surf_parts.append(gdf)

    surface_singlepart = dissolve_and_singlepart(
        gpd.GeoDataFrame(
            pd.concat(surf_parts, ignore_index=True),
            geometry="geometry",
            crs=surf_parts[0].crs,
        )
    )

    merged = pd.concat(
        split_fwy_parts + [surface_singlepart],
        ignore_index=True,
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=base_crs)


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
        Output of rcl_merge — geometry only.
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
    joined = joined[~joined.index.duplicated(keep="first")]

    result = cleaned_gdf.copy()
    for col in cols:
        result[col] = joined[col].values
    return result
