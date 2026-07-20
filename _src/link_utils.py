"""
Utility functions for WFRC true-shape link generation.

Design contract
---------------
- Filtering / topology definitions live in the calling notebook.
- This module only accepts pre-filtered data and implements the mechanics
  of geometric operations, splitting, and graph assembly.
- All heavy-lifting uses vectorised Shapely 2.x APIs (shapely.get_coordinates,
  shapely.points, STRtree bulk queries); row-level Python loops are limited to
  the piece-splitting step where sequential state is unavoidable.

Stage 2 — Public API
--------------------
    resolve_snap_coords(gdf_nodes_snapped, gdf_centerlines, tolerance=0.05)
        For each snapped node, resolve the rounded snap coordinate to the
        exact centerline vertex coordinate using STRtree nearest-vertex lookup.
        Adds x_exact, y_exact, snap_resolved columns.

    split_candidate_links(gdf_components, snap_points_exact, id_cols)
        Split each pre-dissolved centerline component (e.g. CandidateTDMRoadLinks
        rows) at all snapped node positions. Each row is treated independently —
        no linemerge/unary_union is applied — so components from different
        VERT_LEVELs are never combined. Uses M-value projection + substring to
        handle multiple split points per line in one pass. id_cols values are
        copied from each parent component onto its resulting sub-pieces.
        Returns a GeoDataFrame of id_cols + geometry, one row per piece.

Stage 3 — Public API
--------------------
    dissolve_pseudonodes(gdf_links, pass_through_ids, invariant_cols=("FT_2027", "LN_2027"))
        Collapse chains of model links connected through pass-through nodes
        (originally strict pseudonodes; generalizable to any non-split node
        set, e.g. all unsnapped nodes) into single dissolved links spanning
        from one real/split node to another. Every gdf_links column (besides
        A, B, geometry) rides along, taken from each chain's first
        constituent link; non-invariant columns are checked for constituent
        agreement and disagreements are tallied in
        result.attrs["disagreement_counts"].
        Returns a DataFrame with A, B, <every other gdf_links column>,
        n_constituents, constituent_ab_pairs.

    Matching a physical link piece to the dissolved chain it belongs to is a
    direct coordinate join (piece endpoint coordinate -> node id -> chain
    A/B), not a graph search -- see 03_transfer_attributes.qmd Part B/C. No
    piece-graph traversal is needed anywhere in this pipeline.

Centroid Connectors — Public API
---------------------------------
    build_centroid_connector_links(gdf_cc_links, gdf_nodes, gdf_nodes_snapped,
                                    gdf_cc_snapped, gdf_ext_snapped)
        Rebuild each FT_2027==1 model link as a straight line between its two
        nodes' resolved coordinates (main network snap -> CC/external
        nearest-point snap -> raw model coordinate). No dissolve, split, or
        chain matching involved -- every row already maps 1:1 to its source
        model link.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

# =============================================================================
# Stage 2 helpers
# =============================================================================


def resolve_snap_coords(
    gdf_nodes_snapped: gpd.GeoDataFrame,
    gdf_centerlines: gpd.GeoDataFrame,
    tolerance: float = 0.05,
) -> gpd.GeoDataFrame:
    """
    Resolve rounded snap coordinates to exact centerline vertex coordinates.

    nodes_snapped.gpkg stores snapped_x_round / snapped_y_round rounded to
    2 decimal places (cm precision). Shapely split/substring operations require
    coordinates that match the actual vertices stored in the centerline geometry.
    This function finds, for each snapped node, the nearest unique centerline
    vertex within `tolerance` metres and records its exact coordinate.

    Parameters
    ----------
    gdf_nodes_snapped : GeoDataFrame
        Output of 01_node_classification.qmd. Must have columns snapped (bool),
        snapped_x_round, snapped_y_round.
    gdf_centerlines : GeoDataFrame
        Centerline layer in the same CRS. Used only for vertex extraction.
    tolerance : float
        Maximum distance in metres between the rounded snap coord and the
        nearest exact vertex. Nodes exceeding this emit a warning.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes_snapped with three new columns:
        - x_exact, y_exact : float  — exact vertex coordinate (NaN if unresolved)
        - snap_resolved     : bool  — True when a vertex was found within tolerance
    """
    # ── Extract all unique centerline vertices ────────────────────────────────
    raw_coords = shapely.get_coordinates(gdf_centerlines.geometry.values)
    # Round to 4dp before dedup to avoid floating-point noise creating false dupes
    unique_coords = np.unique(np.round(raw_coords, 4), axis=0)
    vertex_geoms = shapely.points(unique_coords[:, 0], unique_coords[:, 1])

    tree = STRtree(vertex_geoms)

    # ── Build query points from snapped nodes ────────────────────────────────
    snapped_mask = gdf_nodes_snapped["snapped"].fillna(False).astype(bool)
    df_snapped = gdf_nodes_snapped[snapped_mask]

    x_exact = np.full(len(gdf_nodes_snapped), np.nan)
    y_exact = np.full(len(gdf_nodes_snapped), np.nan)
    snap_resolved = np.zeros(len(gdf_nodes_snapped), dtype=bool)

    if df_snapped.empty:
        result = gdf_nodes_snapped.copy()
        result["x_exact"] = x_exact
        result["y_exact"] = y_exact
        result["snap_resolved"] = snap_resolved
        return result

    query_pts = shapely.points(
        df_snapped["snapped_x_round"].to_numpy(dtype=float),
        df_snapped["snapped_y_round"].to_numpy(dtype=float),
    )

    # nearest() returns one closest vertex index per query point
    nearest_idxs = tree.nearest(query_pts)
    nearest_verts = vertex_geoms[nearest_idxs]
    dists = shapely.distance(query_pts, nearest_verts)
    resolved = dists <= tolerance

    n_unresolved = int((~resolved).sum())
    if n_unresolved:
        warnings.warn(
            f"resolve_snap_coords: {n_unresolved} snapped node(s) had no "
            f"centerline vertex within {tolerance} m — x_exact/y_exact left NaN."
        )

    nearest_xy = shapely.get_coordinates(nearest_verts)

    # Map results back to the full-GDF positional index
    snapped_positions = np.where(snapped_mask)[0]
    x_exact[snapped_positions] = np.where(resolved, nearest_xy[:, 0], np.nan)
    y_exact[snapped_positions] = np.where(resolved, nearest_xy[:, 1], np.nan)
    snap_resolved[snapped_positions] = resolved

    result = gdf_nodes_snapped.copy()
    result["x_exact"] = x_exact
    result["y_exact"] = y_exact
    result["snap_resolved"] = snap_resolved
    return result


def split_candidate_links(
    gdf_components: gpd.GeoDataFrame,
    snap_points_exact: list[Point],
    id_cols: tuple[str, ...] = ("OBJECTID", "UNIQUE_ID"),
) -> gpd.GeoDataFrame:
    """
    Split each pre-dissolved centerline component at all snapped node positions.

    Each row of gdf_components (e.g. CandidateTDMRoadLinks) is already a
    maximal-chain component bounded only by physical junctions — see
    centerline_utils.rcl_merge. This function treats rows independently and
    applies no linemerge/unary_union, so components from different VERT_LEVELs
    are never combined.

    For each component, finds all snap points that lie on it (within 1 cm
    tolerance), projects them to M-values, and uses shapely.ops.substring to
    extract the sub-segments between consecutive cut points. Points at or
    beyond a line's endpoints (M ≈ 0 or M ≈ total length) are silently
    skipped — those lines are already bounded there.

    Parameters
    ----------
    gdf_components : GeoDataFrame
        One row per maximal-chain centerline component. Must include geometry
        and every column named in id_cols.
    snap_points_exact : list[Point]
        Exact-coordinate snap points, one per snapped node (from x_exact/y_exact
        columns of resolve_snap_coords output). Points that do not lie on any
        component are silently ignored.
    id_cols : tuple[str, ...]
        Columns to copy from each parent component onto its resulting
        sub-pieces (lineage keys — see CLAUDE.md GERS roadmap).

    Returns
    -------
    GeoDataFrame
        One row per output piece: id_cols + geometry. Every piece's endpoints
        are either a snapped node position or a component boundary inherited
        from gdf_components.
    """
    components = gdf_components.geometry.tolist()
    n_components = len(components)

    # ── Assign each snap point to every component it lies on ─────────────────
    # Use a 1 cm buffer for the query, then verify with distance check. A point
    # can legitimately lie on more than one component: components are dissolved
    # per VERT_LEVEL (see 02_create_link.qmd Part A), and at a freeway gore the
    # mainline and the ramp/CD road are digitized to share the exact same XY
    # coordinate even though they live in different VERT_LEVEL components.
    # Assigning to every match (not just the first STRtree hit) is safe: for
    # components where the point is actually one of that component's own
    # endpoints, _split_line_at_points' _EPS skip makes it a no-op.
    _QUERY_TOL = 0.01  # metres
    tree = STRtree(components)

    piece_to_pts: defaultdict[int, list[Point]] = defaultdict(list)
    for pt in snap_points_exact:
        if pt is None:
            continue
        candidate_idxs = tree.query(pt.buffer(_QUERY_TOL))
        for idx in candidate_idxs:
            if idx < n_components and components[idx].distance(pt) <= _QUERY_TOL:
                piece_to_pts[idx].append(pt)

    # ── Split each component at its assigned points, carrying id_cols ────────
    id_values = gdf_components[list(id_cols)].to_numpy()

    result_rows: list[dict] = []
    for i, line in enumerate(components):
        pts = piece_to_pts.get(i)
        sub_geoms = _split_line_at_points(line, pts) if pts else [line]
        for geom in sub_geoms:
            row = {col: id_values[i, j] for j, col in enumerate(id_cols)}
            row["geometry"] = geom
            result_rows.append(row)

    return gpd.GeoDataFrame(result_rows, geometry="geometry", crs=gdf_components.crs)


# =============================================================================
# Internal helpers
# =============================================================================


def _split_line_at_points(line: LineString, points: list[Point]) -> list[LineString]:
    """
    Split a single LineString at one or more interior points using M-values.

    Points that project to M <= eps or M >= (length - eps) are treated as
    endpoint coincidences and do not generate a split.
    """
    total_len = line.length
    if total_len < 1e-9:
        return [line]

    _EPS = 0.50  # 50 cm — skip splits within 50 cm of an endpoint; drop sub-segments shorter than 50 cm

    # Project each point to an M-value along the line; deduplicate
    m_set: set[float] = set()
    for pt in points:
        m = line.project(pt)
        if _EPS < m < total_len - _EPS:
            m_set.add(m)

    if not m_set:
        return [line]

    cuts = [0.0] + sorted(m_set) + [total_len]

    segments: list[LineString] = []
    for i in range(len(cuts) - 1):
        seg = substring(line, cuts[i], cuts[i + 1])
        if seg is not None and not seg.is_empty and seg.length > _EPS:
            segments.append(seg)

    return segments if segments else [line]


# =============================================================================
# Stage 3 helpers
# =============================================================================


def dissolve_pseudonodes(
    gdf_links: gpd.GeoDataFrame,
    pass_through_ids: set,
    invariant_cols: tuple[str, ...] = ("FT_2027", "LN_2027"),
) -> pd.DataFrame:
    """
    Collapse chains of model links connected through pass-through nodes into
    dissolved links.

    A pass-through node is any node the caller designates as a non-split
    point -- originally this meant strictly topological pseudonodes
    (degree-2, is_pseudo=True, see 01_node_classification.qmd), but the set
    can be broadened to any node that is not itself a split point (e.g. every
    currently-unsnapped node, a strict superset of is_pseudo). This function
    merges consecutive links into a single dissolved link spanning from one
    real/split node to another.

    Every column in gdf_links other than A, B, and geometry rides along onto
    the dissolved output, taken from the chain's first constituent link.
    `invariant_cols` names columns already guaranteed uniform across every
    constituent in a chain (no check performed there) -- this guarantee holds
    for strict pseudonode chains but not necessarily for a broadened
    pass-through set, where an unsnapped-but-real node could see an FT/LN
    change; pass `invariant_cols=()` in that case. Every other column is
    checked for agreement across constituents -- but only in multi-constituent
    chains, since a single-constituent chain trivially agrees with itself.
    Disagreements are tallied per column and surfaced via one aggregate
    warning plus `result.attrs["disagreement_counts"]`.

    Parameters
    ----------
    gdf_links : GeoDataFrame
        Filtered model links to dissolve. Must have columns A, B.
    pass_through_ids : set
        Set of node N-values to treat as non-split, pass-through points.
    invariant_cols : tuple[str, ...]
        Columns already known to be uniform across every constituent of a
        pass-through chain -- skips the disagreement check for these.

    Returns
    -------
    DataFrame with one row per dissolved link:
        A, B                  : int  — real start/end node N-values
        <every other gdf_links column, minus geometry> — taken from the
            chain's first constituent link
        n_constituents        : int  — number of original links collapsed
        constituent_ab_pairs  : str  — JSON list of [A, B] pairs in traversal order

        result.attrs["disagreement_counts"] : dict[str, int] — for each
        non-invariant column, the number of multi-constituent chains where at
        least one constituent's value differed from the chain's first.

    Notes
    -----
    Uses a directed MultiDiGraph so that opposite-direction links (A→P and P→A)
    on the same road segment produce two separate chains (A→B and B→A) rather
    than collapsing into a single self-loop.

    At a pass-through node with more than one valid forward continuation (only
    possible when `pass_through_ids` is broadened beyond strict degree-2
    pseudonodes), the walk picks the geometrically straightest continuation
    (smallest turn angle from the incoming bearing) rather than an arbitrary
    graph-iteration-order edge. This is a no-op for ordinary degree-2
    pseudonodes, which never have more than one non-backtrack candidate.
    """
    transfer_cols = [c for c in gdf_links.columns if c not in ("A", "B", "geometry")]
    col_pos = {c: i for i, c in enumerate(transfer_cols)}
    values = gdf_links[transfer_cols].to_numpy(dtype=object)
    invariant_set = set(invariant_cols)
    geoms = gdf_links["geometry"].to_numpy()

    def _bearing(p0: tuple[float, float], p1: tuple[float, float]) -> float:
        return np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 360

    def _in_bearing(row_idx: int) -> float:
        # Direction of travel arriving at the edge's B endpoint (last segment).
        coords = shapely.get_coordinates(geoms[row_idx])
        return _bearing(tuple(coords[-2]), tuple(coords[-1]))

    def _out_bearing(row_idx: int) -> float:
        # Direction of travel departing the edge's A endpoint (first segment).
        coords = shapely.get_coordinates(geoms[row_idx])
        return _bearing(tuple(coords[0]), tuple(coords[1]))

    def _turn_angle(in_bearing: float, out_bearing: float) -> float:
        d = abs(out_bearing - in_bearing) % 360
        return min(d, 360 - d)

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for row_idx, (_, row) in enumerate(gdf_links.iterrows()):
        a, b = int(row["A"]), int(row["B"])
        G.add_edge(a, b, orig_A=a, orig_B=b, row_idx=row_idx)

    real_nodes = set(G.nodes()) - pass_through_ids
    dissolved: list[dict] = []
    visited_edges: set[tuple] = set()
    disagreement_counts: dict[str, int] = defaultdict(int)
    n_affected_chains = 0

    for start in sorted(real_nodes):
        # out_edges: only edges leaving `start` (directed)
        for _, nbr, key, edata in G.out_edges(start, keys=True, data=True):
            ek = (start, nbr, key)
            if ek in visited_edges:
                continue
            visited_edges.add(ek)

            chain_pairs: list[list[int]] = [[edata["orig_A"], edata["orig_B"]]]
            chain_row_idxs: list[int] = [edata["row_idx"]]
            prev = start
            curr = nbr

            while curr in pass_through_ids:
                # Prefer edges not returning to the node just visited (`prev`);
                # fall back to any unvisited edge. This prevents a pseudonode's
                # back-edge from being chosen when a forward edge is also
                # available -- checking against `prev` (not just the chain's
                # original `start`) is required on a two-way street modeled as
                # two independent one-way rows: the reverse-direction sibling
                # of the edge just traversed is a distinct edge to a node other
                # than `start`, so it would otherwise look like valid forward
                # progress and send the walk straight back the way it came.
                #
                # A true degree-2 pseudonode only ever has one non-backtrack
                # candidate, so the tie-break below is a no-op for every
                # pass-through node this function has ever been called on
                # historically. It only activates for a pass_through_ids set
                # broadened to real multi-neighbor junctions (e.g. rail wyes/
                # crossovers where the physical GTFS shape runs straight
                # through with no corresponding split) -- there, the straightest
                # continuation (smallest turn angle from the incoming bearing)
                # is preferred over an arbitrary graph-iteration-order pick.
                candidates = []
                fallback = None
                for _, cnbr, ckey, cedata in G.out_edges(curr, keys=True, data=True):
                    cek = (curr, cnbr, ckey)
                    if cek in visited_edges:
                        continue
                    if cnbr != prev:
                        candidates.append((cnbr, cedata, cek))
                    elif fallback is None:
                        fallback = (cnbr, cedata, cek)

                if len(candidates) > 1:
                    in_bearing = _in_bearing(chain_row_idxs[-1])
                    next_found = min(
                        candidates,
                        key=lambda c: _turn_angle(in_bearing, _out_bearing(c[1]["row_idx"])),
                    )
                elif candidates:
                    next_found = candidates[0]
                else:
                    next_found = fallback
                if next_found is None:
                    break
                next_nbr, next_edata, next_ek = next_found
                visited_edges.add(next_ek)
                chain_pairs.append([next_edata["orig_A"], next_edata["orig_B"]])
                chain_row_idxs.append(next_edata["row_idx"])
                prev = curr
                curr = next_nbr

            first_vals = values[chain_row_idxs[0]]
            row_out: dict = {"A": start, "B": curr}
            for col in transfer_cols:
                row_out[col] = first_vals[col_pos[col]]

            if len(chain_row_idxs) > 1:
                chain_disagreed = False
                for col in transfer_cols:
                    if col in invariant_set:
                        continue
                    pos = col_pos[col]
                    fv = first_vals[pos]
                    fv_na = pd.isna(fv)
                    for idx in chain_row_idxs[1:]:
                        v = values[idx, pos]
                        if fv_na and pd.isna(v):
                            continue
                        if v != fv:
                            disagreement_counts[col] += 1
                            chain_disagreed = True
                            break
                if chain_disagreed:
                    n_affected_chains += 1

            row_out["n_constituents"] = len(chain_pairs)
            row_out["constituent_ab_pairs"] = json.dumps(chain_pairs)
            dissolved.append(row_out)

    result = pd.DataFrame(dissolved)
    result.attrs["disagreement_counts"] = dict(disagreement_counts)

    if disagreement_counts:
        warnings.warn(
            f"dissolve_pseudonodes: {n_affected_chains} multi-constituent chain(s) "
            f"had at least one non-uniform attribute column across "
            f"{len(disagreement_counts)} column(s); each chain's first constituent "
            "value was kept. See result.attrs['disagreement_counts'] for a "
            "per-column breakdown.",
            stacklevel=2,
        )

    return result


# =============================================================================
# Centroid connector helpers
# =============================================================================


def build_centroid_connector_links(
    gdf_cc_links: gpd.GeoDataFrame,
    gdf_nodes: gpd.GeoDataFrame,
    gdf_nodes_snapped: gpd.GeoDataFrame,
    gdf_cc_snapped: gpd.GeoDataFrame,
    gdf_ext_snapped: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Rebuild centroid connector link geometry as a straight line between each
    link's two resolved node coordinates.

    Every FT_2027 == 1 model link is already a straight two-vertex tether
    between a zone-side node (TAZ centroid, external station, or transit
    station -- never snapped anywhere, so its model coordinate is fixed and
    must not move) and a network-side attachment node. Rather than branching
    on node-ID ranges to tell the two apart, both endpoints of every link are
    resolved independently through the same coordinate lookup: a true
    centroid node never appears in any snapped-node table and so always
    falls through to its raw coordinate unchanged; a network-side node
    resolves to wherever it was actually snapped.

    Resolution cascade per node N (first match wins, highest priority last so
    it can overwrite a lower tier):
      1. `gdf_nodes_snapped` where `snapped == True` -- the main greedy
         road-network snap (also covers rail-station nodes snapped via
         `FixedTransit_Rail`).
      2. `gdf_cc_snapped` where `snap_resolved == True` -- nearest-point CC
         attachment snap (01_node_classification.qmd).
      3. `gdf_ext_snapped` where `snap_resolved == True` -- nearest-point
         external-station snap.
      4. Raw `gdf_nodes` coordinate -- true centroid nodes always land here;
         an unresolved network-side node (beyond either snap's distance cap)
         also falls back here rather than fabricating a position.

    Parameters
    ----------
    gdf_cc_links : GeoDataFrame
        FT_2027 == 1 model links (A, B, + all other model attribute
        columns). Only A and B are read; every other column rides through
        unchanged onto the output.
    gdf_nodes : GeoDataFrame
        Full raw node table (N, geometry) -- source of the always-available
        fallback coordinate.
    gdf_nodes_snapped : GeoDataFrame
        `nodes_snapped` layer from 01_node_classification.qmd. Must have N,
        snapped, snapped_x_round, snapped_y_round.
    gdf_cc_snapped, gdf_ext_snapped : GeoDataFrame
        `cc_nodes_snapped` / `ext_nodes_snapped` layers. Must have N,
        snap_resolved, x_snap, y_snap.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_cc_links with geometry replaced by the resolved straight
        line, plus:
        - A_resolution, B_resolution : str -- which tier resolved each
          endpoint ("network_snapped", "cc_snapped", "ext_snapped", "raw").
        - trueshape_method : "centroid_connector" (constant).
    """
    coord_lookup: dict[int, tuple[float, float]] = {}
    tier_lookup: dict[int, str] = {}

    def _apply_tier(node_ids, xs, ys, tier: str) -> None:
        for n, x, y in zip(node_ids, xs, ys):
            coord_lookup[int(n)] = (float(x), float(y))
            tier_lookup[int(n)] = tier

    # Tier 4 (lowest priority -- applied first so higher tiers overwrite it).
    raw_xy = shapely.get_coordinates(gdf_nodes.geometry.values)
    _apply_tier(gdf_nodes["N"], raw_xy[:, 0], raw_xy[:, 1], "raw")

    # Tier 3.
    ext_resolved = gdf_ext_snapped[gdf_ext_snapped["snap_resolved"]]
    _apply_tier(ext_resolved["N"], ext_resolved["x_snap"], ext_resolved["y_snap"], "ext_snapped")

    # Tier 2.
    cc_resolved = gdf_cc_snapped[gdf_cc_snapped["snap_resolved"]]
    _apply_tier(cc_resolved["N"], cc_resolved["x_snap"], cc_resolved["y_snap"], "cc_snapped")

    # Tier 1 (highest priority).
    main_snapped = gdf_nodes_snapped[gdf_nodes_snapped["snapped"].fillna(False).astype(bool)]
    _apply_tier(main_snapped["N"], main_snapped["snapped_x_round"], main_snapped["snapped_y_round"], "network_snapped")

    a_ids = gdf_cc_links["A"].astype(int).to_numpy()
    b_ids = gdf_cc_links["B"].astype(int).to_numpy()

    missing = (set(a_ids) | set(b_ids)) - coord_lookup.keys()
    if missing:
        raise ValueError(
            f"build_centroid_connector_links: {len(missing)} node id(s) referenced by "
            f"gdf_cc_links have no coordinate in gdf_nodes (sample: {sorted(missing)[:5]})."
        )

    geoms = [
        LineString([coord_lookup[a], coord_lookup[b]])
        for a, b in zip(a_ids, b_ids)
    ]

    result = gdf_cc_links.copy()
    result["geometry"] = geoms
    result["A_resolution"] = [tier_lookup[n] for n in a_ids]
    result["B_resolution"] = [tier_lookup[n] for n in b_ids]
    result["trueshape_method"] = "centroid_connector"
    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf_cc_links.crs)

