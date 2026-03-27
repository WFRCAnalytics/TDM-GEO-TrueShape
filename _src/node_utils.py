"""
Utility functions for network node classification and snapping.

Exports:

    nodes_on(gdf_nodes, gdf_links, query)
        Boolean Series — True if node N appears in A or B of any link
        matching the pandas .query() string.

        e.g. gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2023 in [20, 22, 23]")

    count_links(gdf_nodes, gdf_links)
    snap_nodes(gdf_nodes, gdf_centerlines_filtered, node_mask, max_distance_m, label, ...)
    snap_transit(gdf_nodes, gdf_stops, node_mask, max_distance_m, ...)

All filtering logic lives entirely in the calling notebook — not here.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Classification helpers — monkey-patched onto GeoDataFrame
# ---------------------------------------------------------------------------


def nodes_on(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame, query: str) -> pd.Series:
    """
    Return a boolean Series: True if the node's N appears in A or B of
    any link matching the pandas .query() string.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Full links layer. Must contain columns A and B, plus any columns
        referenced in query.
    query : str
        pandas .query() string to filter links.
        e.g. "FT_2023 in [20, 22, 23]"
        e.g. "FT_2023 == 1"

    Returns
    -------
    pd.Series
        Boolean Series aligned to gdf_nodes index.

    Example
    -------
    gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2023 in [20, 22, 23]")
    gdf_nodes["FixedTransit"] = (
        nodes_on(gdf_nodes, gdf_links, "FT_2023 in [70, 80]")
        | gdf_nodes["N"].between(10_000, 19_999)
    )

    """
    filtered = gdf_links.query(query)
    connected = set(filtered["A"]).union(filtered["B"])
    return gdf_nodes["N"].isin(connected)


def count_links(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a LinkCount column to the nodes GeoDataFrame, counting how many
    link endpoints (A or B) match each node's N value.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A and B.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with LinkCount column appended.

    """
    result = gdf_nodes.copy()
    link_counts = pd.concat([gdf_links["A"], gdf_links["B"]]).value_counts()
    result["LinkCount"] = result["N"].map(link_counts).fillna(0).astype(int)
    return result


def assign_directions(gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add a 'link_directions' column to the nodes GeoDataFrame, containing a
    comma-separated string of all unique directions (from link 'DIRECTION' col)
    connected to that node.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, and DIRECTION.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with 'link_directions' column appended.

    """
    result = gdf_nodes.copy()

    # Ensure DIRECTION column exists to avoid KeyError
    if "DIRECTION" not in gdf_links.columns:
        print("Warning: 'DIRECTION' column not found in links. Skipping direction assignment.")
        result["link_directions"] = ""
        return result

    # Stack A and B nodes so we have a flat list of (Node, Direction)
    links_a = gdf_links[["A", "DIRECTION"]].rename(columns={"A": "N"})
    links_b = gdf_links[["B", "DIRECTION"]].rename(columns={"B": "N"})

    # Combine, drop null directions, and strip whitespace just in case
    node_dirs = pd.concat([links_a, links_b]).dropna(subset=["DIRECTION"])
    node_dirs["DIRECTION"] = node_dirs["DIRECTION"].astype(str).str.strip()

    # Group by Node 'N', get unique directions, and join as a comma-separated string
    # e.g., {'NB', 'WB'} becomes "NB,WB"
    dir_strings = (
        node_dirs.groupby("N")["DIRECTION"]
        .unique()
        .apply(lambda x: ",".join(sorted([d for d in x if d and d.lower() != "nan"])))
    )

    # Map back to the nodes dataframe; fill missing with empty string
    result["link_directions"] = result["N"].map(dir_strings).fillna("")

    return result


# ---------------------------------------------------------------------------
# Snapping helpers (private)
# ---------------------------------------------------------------------------


def _assign_line_directions(gdf_lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract allowed directions from FULLNAME or DOT_RTNAME for LineStrings."""
    lines = gdf_lines.copy()
    lines["allowed_dirs"] = ""

    if "FULLNAME" in lines.columns:
        extracted = lines["FULLNAME"].astype(str).str.extract(r"\b(NB|SB|EB|WB)\b", expand=False)
        lines["allowed_dirs"] = extracted.fillna("")

    if "DOT_RTNAME" in lines.columns:
        mask_empty = lines["allowed_dirs"] == ""
        # The 5th character (index 4) is the direction (P or N)
        lrs_dir = lines.loc[mask_empty, "DOT_RTNAME"].astype(str).str[4:5]

        # P = Positive (Northbound or Eastbound)
        lines.loc[mask_empty & (lrs_dir == "P"), "allowed_dirs"] = "NB,EB"
        # N = Negative (Southbound or Westbound)
        lines.loc[mask_empty & (lrs_dir == "N"), "allowed_dirs"] = "SB,WB"

    return lines


def _spatial_snap(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoDataFrame | gpd.GeoSeries,
    max_distance_m: float,
    crs_projected: str,
) -> tuple[list, list, list]:
    """
    Snap nodes using Segment-First (Point-to-Line-to-Point) logic.
    Uses a Global Greedy Assignment to resolve collisions.
    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)

    # 1. Adapt to LineStrings (Roads) vs Points (GTFS)
    is_lines = False
    if isinstance(snap_targets, gpd.GeoDataFrame):
        geom_types = snap_targets.geometry.type.unique()
        if any("LineString" in t for t in geom_types):
            is_lines = True
            snap_targets = _assign_line_directions(snap_targets)
            targets_proj = snap_targets.to_crs(crs_projected)
            has_dirs = True
        else:
            targets_proj = snap_targets.to_crs(crs_projected)
            has_dirs = "allowed_dirs" in snap_targets.columns
    else:
        targets_proj = snap_targets.to_crs(crs_projected)
        has_dirs = False

    target_geoms_proj = targets_proj.geometry.values
    target_geoms_orig = snap_targets.geometry.values

    tree = STRtree(target_geoms_proj)
    DIR_GROUP = {"NB": "P", "EB": "P", "P": "P", "SB": "N", "WB": "N", "N": "N"}

    # ==========================================
    # PHASE 1: Bulk Spatial Query (Vectorized)
    # ==========================================
    query_pairs = tree.query(
        nodes_proj.geometry.values, predicate="dwithin", distance=max_distance_m
    )
    node_indices = query_pairs[0]
    target_indices = query_pairs[1]

    # Calculate distance to the physical TARGET (Line or Point)
    distances_to_target = shapely.distance(
        nodes_proj.geometry.values[node_indices], target_geoms_proj[target_indices]
    )

    node_dir_sets = [
        set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
        for d_str in gdf_nodes.get("link_directions", pd.Series([""] * len(gdf_nodes)))
    ]

    if has_dirs:
        target_dir_sets = [
            set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
            for d_str in targets_proj.get("allowed_dirs", pd.Series([""] * len(targets_proj)))
        ]

    # ==========================================
    # PHASE 2: Generate Segment-First Bids
    # ==========================================
    all_candidates = []
    for i in range(len(node_indices)):
        n_idx = node_indices[i]
        t_idx = target_indices[i]
        dist_target = distances_to_target[i]
        proj_node = nodes_proj.geometry.values[n_idx]

        is_compatible = True
        if has_dirs:
            n_dirs = node_dir_sets[n_idx]
            t_dirs = target_dir_sets[t_idx]
            if t_dirs and n_dirs and not n_dirs.intersection(t_dirs):
                is_compatible = False

        if is_compatible:
            if is_lines:
                # Extract specific endpoints of this valid physical line
                proj_line = target_geoms_proj[t_idx]
                orig_line = target_geoms_orig[t_idx]

                if proj_line.geom_type == "MultiLineString":
                    p_start, p_end = (
                        Point(proj_line.geoms[0].coords[0]),
                        Point(proj_line.geoms[-1].coords[-1]),
                    )
                    o_start, o_end = (
                        Point(orig_line.geoms[0].coords[0]),
                        Point(orig_line.geoms[-1].coords[-1]),
                    )
                else:
                    p_start, p_end = Point(proj_line.coords[0]), Point(proj_line.coords[-1])
                    o_start, o_end = Point(orig_line.coords[0]), Point(orig_line.coords[-1])

                dist_start = proj_node.distance(p_start)
                dist_end = proj_node.distance(p_end)

                # Global Endpoint IDs (rounded to 1mm to prevent floating point misses)
                id_start = (round(p_start.x, 3), round(p_start.y, 3))
                id_end = (round(p_end.x, 3), round(p_end.y, 3))

                if dist_start <= max_distance_m:
                    all_candidates.append((dist_target, dist_start, n_idx, id_start, o_start))
                if dist_end <= max_distance_m:
                    all_candidates.append((dist_target, dist_end, n_idx, id_end, o_end))
            else:
                # GTFS Point logic remains the same
                orig_pt = target_geoms_orig[t_idx]
                dist_pt = proj_node.distance(target_geoms_proj[t_idx])
                id_pt = (round(target_geoms_proj[t_idx].x, 3), round(target_geoms_proj[t_idx].y, 3))
                if dist_pt <= max_distance_m:
                    all_candidates.append((dist_target, dist_pt, n_idx, id_pt, orig_pt))

    # ==========================================
    # PHASE 3: Global Sort (Topology > Topography)
    # ==========================================
    # Sort primarily by distance to the LineString, secondarily by distance to Endpoint
    all_candidates.sort(key=lambda x: (x[0], x[1]))

    # ==========================================
    # PHASE 4: The Claiming Process
    # ==========================================
    claimed_nodes = set()
    claimed_endpoints = set()

    num_nodes = len(gdf_nodes)
    snapped_geoms = [None] * num_nodes
    snap_distances_m = [None] * num_nodes
    snapped_flags = [False] * num_nodes

    # Unpack bid: (dist_target, dist_endpoint, node_idx, endpoint_id, endpoint_geom_orig)
    for _, dist_endpoint, node_idx, endpoint_id, endpoint_geom_orig in all_candidates:
        if node_idx not in claimed_nodes and endpoint_id not in claimed_endpoints:
            claimed_nodes.add(node_idx)
            claimed_endpoints.add(endpoint_id)

            snapped_geoms[node_idx] = endpoint_geom_orig
            snap_distances_m[node_idx] = round(dist_endpoint, 2)
            snapped_flags[node_idx] = True

    # Handle Leftovers
    for i, (orig_geom, proj_geom) in enumerate(zip(gdf_nodes.geometry, nodes_proj.geometry)):
        if i not in claimed_nodes:
            nearest_idx = tree.nearest(proj_geom)
            abs_distance_m = (
                proj_geom.distance(target_geoms_proj[nearest_idx])
                if nearest_idx is not None
                else np.nan
            )

            snapped_geoms[i] = orig_geom
            snap_distances_m[i] = round(abs_distance_m, 2) if pd.notna(abs_distance_m) else np.nan
            snapped_flags[i] = False

    return snapped_geoms, snap_distances_m, snapped_flags


# ---------------------------------------------------------------------------
# Public snapping functions
# ---------------------------------------------------------------------------


def snap_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_centerlines_filtered: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float,
    label: str,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """Snap nodes to the endpoints of pre-filtered centerlines."""
    result = gdf_nodes.copy()

    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    if len(gdf_centerlines_filtered) == 0:
        return result

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[result.loc[candidate_idx, "snap_rule"] == "none"]

    if len(candidate_idx) == 0:
        return result

    print(
        f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_centerlines_filtered):,} centerlines"
    )

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags = _spatial_snap(
        candidate_nodes, gdf_centerlines_filtered, max_distance_m, crs_projected
    )

    for idx, geom, dist, snapped_flag in zip(candidate_idx, snapped_geoms, distances, flags):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result


def snap_transit(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float = 200,
    label: str = "FixedTransit_GTFS",
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """
    Snap a subset of nodes to the nearest GTFS stop point.

    Follows the same first-call-wins pattern as snap_nodes() — nodes already
    snapped by a previous call are automatically skipped.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Pass the result of previous snap calls to chain.
    gdf_stops : GeoDataFrame
        GTFS stops layer with Point geometry.
    node_mask : pd.Series
        Boolean Series aligned to gdf_nodes.index selecting candidate nodes.
        e.g. gdf_nodes["FixedTransit"]
    max_distance_m : float, optional
        Nodes further than this threshold are not moved. Default 200m.
    label : str, optional
        Written to the snap_rule audit column. Default 'FixedTransit_GTFS'.
    crs_projected : str, optional
        Projected CRS for metric distances. Default EPSG:26912 (UTM 12N).

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with updated geometry and audit columns.

    """
    result = gdf_nodes.copy()

    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[result.loc[candidate_idx, "snap_rule"] == "none"]

    if len(candidate_idx) == 0:
        print(f"  [{label}] No unsnapped candidate nodes — skipping.")
        return result

    print(f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_stops):,} stops")

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags = _spatial_snap(
        candidate_nodes, gdf_stops.geometry, max_distance_m, crs_projected
    )

    for idx, geom, dist, snapped_flag in zip(candidate_idx, snapped_geoms, distances, flags):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result
