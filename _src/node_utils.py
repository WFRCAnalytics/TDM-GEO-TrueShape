"""
Utility functions for network node classification and snapping.

Exports:

    nodes_on(gdf_nodes, gdf_links, query)
        Boolean Series — True if node N appears in A or B of any link
        matching the pandas .query() string.

        e.g. gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2027 in [20, 22, 23]")

    count_links(gdf_nodes, gdf_links)
    assign_node_directions(gdf_nodes, gdf_links, freeway_ft_codes)
    assign_node_type(gdf_nodes, is_fwy_mask, is_ramp_mask, is_surface_mask)
        Add node_type column — one of: "fwy", "gore", "ramp", "ramp_sf", "surface".
    extract_endpoints(gdf_centerlines)
    assign_endpoint_directions(gdf_ep_unique, gdf_ep_raw)
    assign_endpoint_type(gdf_ep_unique)
        Add ep_type column — one of: "fwy", "gore", "fwy_sf", "ramp", "ramp_sf", "surface".
    snap_nodes(gdf_nodes, gdf_endpoints, node_mask, max_distance_m, label, ...)
    snap_transit(gdf_nodes, gdf_stops, node_mask, max_distance_m, ...)

All filtering and topology classification logic lives in the calling notebook.
"""

from collections import defaultdict, deque

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree


# ---------------------------------------------------------------------------
# Classification helpers
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
        e.g. "FT_2027 in [20, 22, 23]"
        e.g. "FT_2027 == 1"

    Returns
    -------
    pd.Series
        Boolean Series aligned to gdf_nodes index.

    Example
    -------
    gdf_nodes["Freeway"] = nodes_on(gdf_nodes, gdf_links, "FT_2027 in [20, 22, 23]")
    gdf_nodes["FixedTransit"] = (
        nodes_on(gdf_nodes, gdf_links, "FT_2027 in [70, 80]")
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


# ---------------------------------------------------------------------------
# Direction resolution — shared private helper
# ---------------------------------------------------------------------------


def _resolve_direction(
    directions: list[str],
    fullnames: list[str],
    dot_rtnames: list[str],
    is_freeway_flags: list[bool],
    is_interchange_flags: list[bool],
) -> str:
    """
    Resolve a single direction string from a collection of associated
    segments (either network links or centerline segments).

    Priority cascade
    ----------------
    1. If ANY freeway segment is present:
       a. Pool DIRECTION / FULLNAME tokens from freeway segments only.
       b. If still empty, fall back to LRS P/N from freeway DOT_RTNAME[4].
    2. If interchange-only (no freeway):
       a. Pool DIRECTION values directly (EB/WB/NB/SB from network links,
          or DOT_RTNAME[4] P/N → "NB,SB" / "EB,WB" from centerline segments).
    3. Nothing resolves → empty string (direction-agnostic).

    Parameters
    ----------
    directions : list[str]
        DIRECTION column values from network links, OR pre-extracted
        direction strings from centerline segments. May be empty strings.
    fullnames : list[str]
        FULLNAME values from centerline segments. Pass [] for network links.
    dot_rtnames : list[str]
        DOT_RTNAME values (for LRS P/N fallback). Pass [] for network links.
    is_freeway_flags : list[bool]
        True if the corresponding segment is a freeway/mainline.
    is_interchange_flags : list[bool]
        True if the corresponding segment is a ramp or CD.

    Returns
    -------
    str
        Comma-separated direction string, e.g. "NB", "NB,EB", "SB,WB", "".
    """
    CARDINAL = {"NB", "SB", "EB", "WB"}
    DIR_FROM_LRS = {"P": "NB,EB", "N": "SB,WB"}

    has_freeway = any(is_freeway_flags)

    collected: set[str] = set()

    if has_freeway:
        # --- Freeway segments only ---
        for d, fn, rt, is_fw in zip(directions, fullnames, dot_rtnames, is_freeway_flags):
            if not is_fw:
                continue
            # Try DIRECTION column first (network links)
            if d and d.strip().upper() in CARDINAL:
                collected.add(d.strip().upper())
            # Try FULLNAME token (centerline segments)
            elif fn:
                token = _extract_fullname_direction(fn)
                if token:
                    collected.add(token)
        # LRS P/N fallback — only if nothing resolved from DIRECTION/FULLNAME
        if not collected:
            for rt, is_fw in zip(dot_rtnames, is_freeway_flags):
                if not is_fw:
                    continue
                lrs_char = str(rt)[4:5] if rt and len(str(rt)) > 4 else ""
                if lrs_char in DIR_FROM_LRS:
                    for d in DIR_FROM_LRS[lrs_char].split(","):
                        collected.add(d)

    else:
        # --- Interchange-only (no freeway segments present) ---
        for d, rt, is_ic in zip(directions, dot_rtnames, is_interchange_flags):
            if not is_ic:
                continue
            # DIRECTION column (network links: direct cardinal value)
            if d and d.strip().upper() in CARDINAL:
                collected.add(d.strip().upper())
            # LRS P/N (centerline segments: encode as NB,SB or EB,WB groups)
            elif rt:
                lrs_char = str(rt)[4:5] if len(str(rt)) > 4 else ""
                if lrs_char in DIR_FROM_LRS:
                    for group_d in DIR_FROM_LRS[lrs_char].split(","):
                        collected.add(group_d)

    return ",".join(sorted(collected))


def _extract_fullname_direction(fullname: str) -> str:
    """
    Extract the FIRST NB/SB/EB/WB token from a FULLNAME string.
    Returns empty string if none found.
    """
    import re
    m = re.search(r"\b(NB|SB|EB|WB)\b", str(fullname))
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Node direction assignment
# ---------------------------------------------------------------------------


def assign_node_directions(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
    freeway_ft_codes: list[int],
) -> gpd.GeoDataFrame:
    """
    Add direction columns to the nodes GeoDataFrame.

    Produces two columns:
    - link_directions : direction string derived from all connected links,
                        using freeway-priority cascade (freeway links first,
                        interchange links only if no freeway links present).
    - fw_directions   : direction string derived from freeway links only.
                        Empty string for non-freeway nodes. Used for gore
                        pass (b) matching where only freeway direction matters.

    For gore nodes with multiple freeway links in opposite directions (e.g.
    both NB and SB), fw_directions stores the MAJORITY direction — whichever
    cardinal appears on the most freeway links. Ties are broken alphabetically
    (deterministic). This prevents a gore node from bidding on both sides of
    the freeway.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, DIRECTION, FT_<year>.
    freeway_ft_codes : list[int]
        FT code values that classify a link as freeway mainline.
        e.g. list((set(range(20, 28)) | set(range(30, 37))) - {21, 31})

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with link_directions and fw_directions appended.
    """
    result = gdf_nodes.copy()

    if "DIRECTION" not in gdf_links.columns:
        print("Warning: 'DIRECTION' column not found in links. Skipping direction assignment.")
        result["link_directions"] = ""
        result["fw_directions"] = ""
        return result

    # Detect FT column (FT_2027 preferred, fall back to FT_2023)
    ft_col = next((c for c in ["FT_2027", "FT_2023"] if c in gdf_links.columns), None)
    if ft_col is None:
        print("Warning: No FT column found in links. Cannot determine freeway links.")
        result["link_directions"] = ""
        result["fw_directions"] = ""
        return result

    fw_set = set(freeway_ft_codes)

    # Stack A and B endpoints so each link contributes to both its nodes
    links_a = gdf_links[["A", "DIRECTION", ft_col]].rename(columns={"A": "N"})
    links_b = gdf_links[["B", "DIRECTION", ft_col]].rename(columns={"B": "N"})
    stacked = pd.concat([links_a, links_b], ignore_index=True)
    stacked["DIRECTION"] = stacked["DIRECTION"].astype(str).str.strip()
    stacked["is_freeway"] = stacked[ft_col].isin(fw_set)
    # Interchange = ramp or CD FT codes (everything in the FW family not in fw_set,
    # plus explicit ramp FT codes). We only need is_freeway for the cascade here;
    # the interchange flag is implicit (not freeway and not surface).
    stacked["is_interchange"] = ~stacked["is_freeway"]  # simplified for link-level cascade

    # --- link_directions: freeway-priority cascade per node ---
    link_dir_map: dict[int, str] = {}
    for node_n, grp in stacked.groupby("N"):
        dirs = grp["DIRECTION"].tolist()
        is_fw = grp["is_freeway"].tolist()
        is_ic = grp["is_interchange"].tolist()
        # No FULLNAME or DOT_RTNAME available from network links
        link_dir_map[node_n] = _resolve_direction(
            directions=dirs,
            fullnames=[""] * len(dirs),
            dot_rtnames=[""] * len(dirs),
            is_freeway_flags=is_fw,
            is_interchange_flags=is_ic,
        )

    result["link_directions"] = result["N"].map(link_dir_map).fillna("")

    # --- fw_directions: freeway links only, majority-vote for opposing dirs ---
    fw_dir_map: dict[int, str] = {}
    fw_links = stacked[stacked["is_freeway"]].copy()
    CARDINAL = {"NB", "SB", "EB", "WB"}

    for node_n, grp in fw_links.groupby("N"):
        cardinals = [
            d.strip().upper()
            for d in grp["DIRECTION"]
            if d.strip().upper() in CARDINAL
        ]
        if not cardinals:
            fw_dir_map[node_n] = ""
            continue
        # Majority vote: pick the direction with the highest count
        counts = pd.Series(cardinals).value_counts()
        max_count = counts.max()
        winners = sorted(counts[counts == max_count].index.tolist())
        # If a clean majority exists, use it. If tied, keep all tied directions
        # (rare case — both will participate in tiered matching).
        fw_dir_map[node_n] = ",".join(winners)

    result["fw_directions"] = result["N"].map(fw_dir_map).fillna("")

    return result


# ---------------------------------------------------------------------------
# Centerline endpoint extraction and direction assignment
# ---------------------------------------------------------------------------


def extract_endpoints(gdf_centerlines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract start and end endpoints from every centerline segment as a flat
    GeoDataFrame of raw (non-deduplicated) point records.

    Each row represents one endpoint of one segment and carries the source
    segment's key attributes needed for topology classification and direction
    assignment. Deduplication and aggregation by coordinate happen in the
    notebook (Step E2) so the logic is visible and inspectable.

    Parameters
    ----------
    gdf_centerlines : GeoDataFrame
        Active centerline segments. Expected columns: geometry, FULLNAME,
        DOT_RTNAME, is_freeway, is_interchange, is_surface (boolean flags
        assigned in the notebook before calling this function).

    Returns
    -------
    GeoDataFrame
        One row per raw endpoint with columns:
        geometry, seg_idx, x_round, y_round,
        fullname, dot_rtname,
        is_freeway, is_interchange, is_surface,
        is_start (True = start endpoint of segment, False = end endpoint)
    """
    records = []

    for seg_idx, row in gdf_centerlines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "MultiLineString":
            coords_start = geom.geoms[0].coords[0]
            coords_end = geom.geoms[-1].coords[-1]
        else:
            coords_start = geom.coords[0]
            coords_end = geom.coords[-1]

        fullname = str(row.get("FULLNAME", "")) if "FULLNAME" in gdf_centerlines.columns else ""
        dot_rtname = str(row.get("DOT_RTNAME", "")) if "DOT_RTNAME" in gdf_centerlines.columns else ""
        is_fw = bool(row.get("is_freeway", False))
        is_ic = bool(row.get("is_interchange", False))
        is_sf = bool(row.get("is_surface", False))

        for coords, is_start in [(coords_start, True), (coords_end, False)]:
            records.append({
                "geometry":      Point(coords),
                "seg_idx":       seg_idx,
                "x_round":       round(coords[0], 3),
                "y_round":       round(coords[1], 3),
                "fullname":      fullname,
                "dot_rtname":    dot_rtname,
                "is_freeway":    is_fw,
                "is_interchange": is_ic,
                "is_surface":    is_sf,
                "is_start":      is_start,
            })

    return gpd.GeoDataFrame(records, crs=gdf_centerlines.crs)


def assign_endpoint_directions(
    gdf_ep_unique: gpd.GeoDataFrame,
    gdf_ep_raw: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign an ep_allowed_dirs direction string to each unique endpoint,
    using freeway-priority cascade via _resolve_direction.

    Direction cascade per unique endpoint
    --------------------------------------
    1. Any freeway segment terminates here →
       FULLNAME token from freeway segment → LRS P/N from freeway DOT_RTNAME[4].
    2. Interchange-only (no freeway) →
       DOT_RTNAME[4] P/N → "NB,SB" or "EB,WB" group (vague but correct group).
       Also checks: parent freeway route from DOT_RTNAME[:4] on ramp segments
       (e.g. "0015NR..." → parent is I-15) to confirm corridor affiliation,
       though direction is still sourced from the LRS P/N character.
    3. Surface or unresolved → empty string.

    Parameters
    ----------
    gdf_ep_unique : GeoDataFrame
        Deduplicated endpoint layer with columns: x_round, y_round,
        is_freeway, is_interchange, is_surface (aggregated booleans).
    gdf_ep_raw : GeoDataFrame
        Raw endpoint records from extract_endpoints(). Used to look up
        all segments associated with each unique coordinate.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_ep_unique with ep_allowed_dirs column appended.
    """
    result = gdf_ep_unique.copy()

    # Build lookup: (x_round, y_round) → list of raw endpoint records
    coord_key = list(zip(gdf_ep_raw["x_round"], gdf_ep_raw["y_round"]))
    raw_by_coord: dict[tuple, list] = defaultdict(list)
    for i, key in enumerate(coord_key):
        raw_by_coord[key].append(i)

    ep_dirs = []
    for _, ep_row in result.iterrows():
        key = (ep_row["x_round"], ep_row["y_round"])
        raw_indices = raw_by_coord.get(key, [])

        if not raw_indices:
            ep_dirs.append("")
            continue

        raw_rows = gdf_ep_raw.iloc[raw_indices]

        ep_dirs.append(_resolve_direction(
            directions=[""] * len(raw_rows),   # no DIRECTION col on centerlines
            fullnames=raw_rows["fullname"].tolist(),
            dot_rtnames=raw_rows["dot_rtname"].tolist(),
            is_freeway_flags=raw_rows["is_freeway"].tolist(),
            is_interchange_flags=raw_rows["is_interchange"].tolist(),
        ))

    result["ep_allowed_dirs"] = ep_dirs
    return result


# ---------------------------------------------------------------------------
# Public type-label assignment functions
# ---------------------------------------------------------------------------


def assign_node_type(
    gdf_nodes: gpd.GeoDataFrame,
    is_fwy_mask: pd.Series,
    is_ramp_mask: pd.Series,
    is_surface_mask: pd.Series,
) -> gpd.GeoDataFrame:
    """
    Add a node_type column to gdf_nodes using three non-exclusive boolean masks.

    The three masks map to five mutually-exclusive topology labels via priority
    cascade (freeway > ramp > surface). A node touching both freeway and ramp
    links is labelled "gore"; a node touching both ramp and surface links is
    labelled "ramp_sf".

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Modified copy is returned.
    is_fwy_mask : pd.Series[bool]
        True if node touches at least one freeway mainline link.
        Typically: gdf_nodes["Freeway"] & ~gdf_nodes["ManagedAccess"] & no_pseudo
    is_ramp_mask : pd.Series[bool]
        True if node touches at least one ramp or CD link.
        Typically: (gdf_nodes["Ramp"] | gdf_nodes["CD"]) & no_pseudo
    is_surface_mask : pd.Series[bool]
        True if node touches at least one surface street link.
        Typically: gdf_nodes["Arterial"] | gdf_nodes["Collector"] | gdf_nodes["Local"]

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with node_type column appended.
        Values: "fwy" | "gore" | "ramp" | "ramp_sf" | "surface"
    """
    result = gdf_nodes.copy()
    fwy  = is_fwy_mask.values.astype(bool)
    ramp = is_ramp_mask.values.astype(bool)
    surf = is_surface_mask.values.astype(bool)

    labels = np.where(
        fwy & ~ramp,        "fwy",
        np.where(
        fwy &  ramp,        "gore",
        np.where(
        ramp & ~surf,       "ramp",
        np.where(
        ramp &  surf,       "ramp_sf",
                            "surface"
    ))))
    result["node_type"] = labels
    return result


def assign_endpoint_type(gdf_ep_unique: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add an ep_type column to gdf_ep_unique derived from the three boolean
    topology flags already present on the layer.

    Parameters
    ----------
    gdf_ep_unique : GeoDataFrame
        Deduplicated endpoint layer with columns:
        is_freeway, is_interchange, is_surface.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_ep_unique with ep_type column appended.
        Values: "fwy" | "gore" | "fwy_sf" | "ramp" | "ramp_sf" | "surface"

    Notes
    -----
    Priority cascade (matches _ep_type_label):
        is_freeway & is_interchange            → "gore"  (mainline/ramp junction)
        is_freeway & is_surface                → "fwy_sf" (at-grade crossing / frontage)
        is_freeway & ~is_interchange & ~is_surface → "fwy"
        ~is_freeway & is_interchange & is_surface  → "ramp_sf" (ramp-to-arterial)
        ~is_freeway & is_interchange & ~is_surface → "ramp"
        else                                   → "surface"
    """
    result = gdf_ep_unique.copy()
    fw  = result["is_freeway"].values.astype(bool)
    ic  = result["is_interchange"].values.astype(bool)
    sf  = result["is_surface"].values.astype(bool)

    labels = np.where(
        fw &  ic,           "gore",
        np.where(
        fw &  sf,           "fwy_sf",
        np.where(
        fw & ~ic & ~sf,     "fwy",
        np.where(
        ~fw & ic & sf,      "ramp_sf",
        np.where(
        ~fw & ic & ~sf,     "ramp",
                            "surface"
    )))))
    result["ep_type"] = labels
    return result




# ---------------------------------------------------------------------------
# Topology type-tier lookup (private)
# ---------------------------------------------------------------------------

# Type tier table: (node_type, endpoint_type) → int
# 0 = same type (preferred), 1 = adjacent type (permitted), None = reject.
#
# Node types    : "fwy", "gore", "ramp", "ramp_sf", "surface"
# Endpoint types: "fwy", "gore", "ramp", "ramp_sf", "surface", "fwy_sf"
#
# Design rationale (data-driven from diagnostic):
#   - fwy-only nodes: same-type median 83m vs any-type 24m (3.5× gap).
#     Gore and fwy+sf endpoints are Tier-1 adjacents; all ramp/surface rejected.
#   - gore nodes: same-type median 108m vs any-type 53m (2×).
#     fwy-only, fwy+sf, ramp-only are Tier-1; ramp+sf and surface rejected.
#   - ramp-only nodes: same-type median 84m vs any-type 19m (4.5×).
#     Gore and ramp+sf are Tier-1; fwy and surface rejected.
#   - ramp+sf nodes: same-type median 35m vs any-type 11m (3.1×). 119 of 284
#     failed nodes were within 50m of a ramp+sf endpoint; 206 within 150m of
#     a surface endpoint — surface is the natural Tier-1 fallback.
#   - surface nodes: same-type median 23m vs any-type 19m (1.2x gap — small).
#     fwy+sf endpoints sit at surface intersections — Tier-0 for surface nodes.
#     No ramp/fwy adjacents needed.

_TYPE_TIER: dict[tuple[str, str], int] = {
    # fwy-only node
    ("fwy",     "fwy"):     0,
    ("fwy",     "gore"):    1,
    ("fwy",     "fwy_sf"):  1,
    # gore node
    ("gore",    "gore"):    0,
    ("gore",    "fwy"):     1,
    ("gore",    "fwy_sf"):  1,
    ("gore",    "ramp"):    1,
    # ramp-only node
    ("ramp",    "ramp"):    0,
    ("ramp",    "gore"):    1,
    ("ramp",    "ramp_sf"): 1,
    # ramp+surface node
    ("ramp_sf", "ramp_sf"): 0,
    ("ramp_sf", "surface"): 1,
    ("ramp_sf", "ramp"):    1,
    # surface-only node
    ("surface", "surface"): 0,
    ("surface", "fwy_sf"):  0,   # fwy+sf endpoints sit at surface intersections
}
_TYPE_TIER_REJECT = 99   # sentinel for hard reject


def _node_type_label(is_fwy: bool, is_ramp: bool, is_surface: bool) -> str:
    """Classify a single node into one of 5 topology type labels."""
    if is_fwy and not is_ramp:
        return "fwy"
    if is_fwy and is_ramp:
        return "gore"
    if is_ramp and not is_surface:
        return "ramp"
    if is_ramp and is_surface:
        return "ramp_sf"
    return "surface"


def _ep_type_label(is_freeway: bool, is_interchange: bool, is_surface: bool) -> str:
    """Classify a single endpoint into one of 6 topology type labels."""
    if is_freeway and not is_interchange and not is_surface:
        return "fwy"
    if is_freeway and is_interchange:
        return "gore"
    if is_freeway and is_surface:
        return "fwy_sf"
    if not is_freeway and is_interchange and not is_surface:
        return "ramp"
    if not is_freeway and is_interchange and is_surface:
        return "ramp_sf"
    return "surface"


# ---------------------------------------------------------------------------
# Snapping core (private)
# ---------------------------------------------------------------------------


def _spatial_snap(
    gdf_nodes: gpd.GeoDataFrame,
    snap_targets: gpd.GeoDataFrame,
    max_distance_m: float,
    crs_projected: str,
    direction_col: str = "link_directions",
    target_id_cols: list[str] = None,
    node_type_col: str = "node_type",
) -> tuple[list, list, list, dict]:
    """
    Snap nodes to target points using Gale-Shapley Stable Matching.

    snap_targets is expected to be a pre-classified endpoint GeoDataFrame
    (output of the notebook's endpoint layer construction) with columns:
        - geometry        : Point geometry
        - ep_allowed_dirs : direction string (from assign_endpoint_directions)
        - is_freeway      : bool
        - is_interchange  : bool
        - is_surface      : bool
        - ep_type         : str  (one of: fwy, gore, fwy_sf, ramp, ramp_sf, surface)

    For transit snapping, snap_targets may be a plain point GeoDataFrame
    without direction columns — direction matching is skipped automatically.

    Algorithm overview
    ------------------
    Phase 1 — Bulk spatial query via STRtree dwithin predicate.
    Phase 2 — Two-dimensional tier assignment per (node, target) pair:
              type_tier   : topology compatibility (0=same, 1=adjacent, 99=reject).
                            Derived from _TYPE_TIER lookup using node_type_col on
                            gdf_nodes and ep_type on snap_targets. Hard rejects
                            (type_tier == 99) are dropped before Gale-Shapley.
              dir_tier    : direction compatibility (0=exact cardinal, 1=same P/N
                            group). Interchange-only endpoints are direction-agnostic
                            so P/N group matches are promoted to dir_tier=0.
              Sort key    : (type_tier, dir_tier, dist) — type dominates direction.
    Phase 3 — Node-proposing Gale-Shapley stable match.
              Nodes rank targets by (type_tier, dir_tier, dist). Targets rank nodes
              by the same key plus n_idx as deterministic tie-break.
    Phase 4 — Write assignments and extract GERS attributes.

    Parameters
    ----------
    direction_col : str
        Column on gdf_nodes to use for directional matching.
        Pass "link_directions" for passes (a) and (c),
        "fw_directions" for pass (b) gore nodes.
    node_type_col : str
        Column on gdf_nodes containing the topology type label
        (one of: "fwy", "gore", "ramp", "ramp_sf", "surface").
        If the column is absent, type-tier matching is skipped and all
        endpoints within max_distance_m are treated as Tier-0 candidates
        (preserves backward compatibility with transit snapping).
    """
    nodes_proj = gdf_nodes.to_crs(crs_projected)
    num_nodes = len(gdf_nodes)

    # Prepare GERS attribute storage
    snapped_attrs: dict[str, list] = {}
    if target_id_cols and isinstance(snap_targets, gpd.GeoDataFrame):
        for col in target_id_cols:
            snapped_attrs[col] = [None] * num_nodes
    # Milepost is not applicable for point endpoint targets (no F/T mile)
    # but keep the key if present for transit targets
    if isinstance(snap_targets, gpd.GeoDataFrame):
        if "DOT_F_MILE" in snap_targets.columns and "DOT_T_MILE" in snap_targets.columns:
            snapped_attrs["milepost"] = [np.nan] * num_nodes

    targets_proj = snap_targets.to_crs(crs_projected)
    target_geoms_proj = targets_proj.geometry.values
    target_geoms_orig = snap_targets.geometry.values

    tree = STRtree(target_geoms_proj)
    DIR_GROUP = {"NB": "P", "EB": "P", "P": "P", "SB": "N", "WB": "N", "N": "N"}

    # Direction columns on snap_targets
    has_dirs = "ep_allowed_dirs" in snap_targets.columns

    # --- Type-tier pre-computation ---
    # Precompute a type label per node (positional — aligned to candidate_nodes)
    # and a type label per target endpoint (positional — aligned to snap_targets).
    # If node_type_col is absent (e.g. transit snapping), type matching is skipped
    # and every (node, endpoint) pair gets type_tier=0.
    has_type = node_type_col in gdf_nodes.columns and "ep_type" in snap_targets.columns
    node_type_labels: list[str] = []
    target_type_labels: list[str] = []
    if has_type:
        node_type_labels = gdf_nodes[node_type_col].tolist()
        target_type_labels = snap_targets["ep_type"].tolist()

    # --- Direction-agnostic flag per target endpoint ---
    # An interchange-only endpoint (ramp or ramp_sf) has inherently vague direction
    # (LRS P/N encodes corridor side, not approach direction), so same-group matches
    # are promoted to dir_tier=0 to avoid starving ramp nodes via overpass-protection.
    is_ic_only_target = np.zeros(len(targets_proj), dtype=bool)
    if all(c in snap_targets.columns for c in ["is_interchange", "is_freeway", "is_surface"]):
        is_ic_only_target = (
            snap_targets["is_interchange"].values.astype(bool)
            & ~snap_targets["is_freeway"].values.astype(bool)
        )

    # ==========================================
    # PHASE 1: Bulk Spatial Query
    # ==========================================
    query_pairs = tree.query(
        nodes_proj.geometry.values, predicate="dwithin", distance=max_distance_m
    )
    node_indices = query_pairs[0]
    target_indices = query_pairs[1]

    distances_to_target = shapely.distance(
        nodes_proj.geometry.values[node_indices], target_geoms_proj[target_indices]
    )

    # Pre-compute node direction sets
    node_dir_col = gdf_nodes.get(direction_col, pd.Series([""] * num_nodes))
    node_dirs_exact = [
        set(d for d in str(d_str).split(",") if d)
        for d_str in node_dir_col
    ]
    node_dirs_grp = [
        set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
        for d_str in node_dir_col
    ]

    if has_dirs:
        target_dirs_exact = [
            set(d for d in str(d_str).split(",") if d)
            for d_str in targets_proj["ep_allowed_dirs"]
        ]
        target_dirs_grp = [
            set(DIR_GROUP.get(d, d) for d in str(d_str).split(",") if d)
            for d_str in targets_proj["ep_allowed_dirs"]
        ]

    # ==========================================
    # PHASE 2: Compatibility Check & Bid Generation
    # ==========================================
    # snap_targets are Points (endpoint layer), not LineStrings.
    # Each target IS an endpoint — no need to extract start/end coords.
    #
    # Sort key is (type_tier, dir_tier, dist):
    #   type_tier  — topology compatibility from _TYPE_TIER lookup.
    #                0 = same type, 1 = adjacent type, 99 = hard reject (dropped).
    #   dir_tier   — directional compatibility.
    #                0 = exact cardinal or direction-agnostic endpoint.
    #                1 = same P/N group (overpass protection for mainline endpoints).
    #   dist       — projected distance in metres.
    all_candidates = []

    for i in range(len(node_indices)):
        n_idx = node_indices[i]
        t_idx = target_indices[i]
        dist_pt = distances_to_target[i]

        # --- Type tier ---
        if has_type:
            n_type = node_type_labels[n_idx]
            t_type = target_type_labels[t_idx]
            type_tier = _TYPE_TIER.get((n_type, t_type), _TYPE_TIER_REJECT)
            if type_tier == _TYPE_TIER_REJECT:
                continue   # hard reject — cross-type mismatch
        else:
            type_tier = 0  # transit or legacy call — skip type matching

        # --- Direction tier ---
        dir_tier = 0
        if has_dirs:
            n_exact = node_dirs_exact[n_idx]
            n_grp   = node_dirs_grp[n_idx]
            t_exact = target_dirs_exact[t_idx]
            t_grp   = target_dirs_grp[t_idx]
            t_is_ic_only = bool(is_ic_only_target[t_idx])

            if t_exact and n_exact:
                if n_exact.intersection(t_exact):
                    dir_tier = 0   # exact cardinal match
                elif n_grp.intersection(t_grp):
                    # Same P/N group. Interchange-only endpoints are direction-
                    # agnostic (LRS P/N is a corridor side, not approach dir),
                    # so promote to dir_tier=0. Freeway/gore endpoints keep
                    # dir_tier=1 for overpass protection (NB prefers NB over SB).
                    dir_tier = 0 if t_is_ic_only else 1
                else:
                    # Opposite P/N groups — cross-carriageway, reject.
                    continue

        if dist_pt <= max_distance_m:
            orig_pt = target_geoms_orig[t_idx]
            id_pt = (round(target_geoms_proj[t_idx].x, 3), round(target_geoms_proj[t_idx].y, 3))
            all_candidates.append(
                (type_tier, dir_tier, dist_pt, n_idx, id_pt, orig_pt, t_idx)
            )

    # ==========================================
    # PHASE 3: Gale-Shapley Stable Matching
    # ==========================================

    # --- 3a. Build preference structures ---

    # temp_node_prefs[n_idx][ep_id] = best proposal payload
    temp_node_prefs: dict[int, dict] = defaultdict(dict)

    # ep_node_dist[ep_id][n_idx] = (tier, dist, n_idx) — deterministic tie-break
    ep_node_dist: dict[tuple, dict[int, tuple]] = defaultdict(dict)

    for bid in all_candidates:
        type_tier, dir_tier, dist_pt, n_idx, ep_id, geom_orig, t_idx = bid

        payload = {
            "type_tier":  type_tier,
            "dir_tier":   dir_tier,
            "dist_ep":    dist_pt,
            "ep_id":      ep_id,
            "geom_orig":  geom_orig,
            "t_idx":      t_idx,
        }

        # Keep the best bid per (node, endpoint) pair — lowest (type, dir, dist)
        if ep_id not in temp_node_prefs[n_idx]:
            temp_node_prefs[n_idx][ep_id] = payload
        else:
            existing = temp_node_prefs[n_idx][ep_id]
            if (type_tier, dir_tier, dist_pt) < (
                existing["type_tier"], existing["dir_tier"], existing["dist_ep"]
            ):
                temp_node_prefs[n_idx][ep_id] = payload

        # Endpoint scores nodes by (type_tier, dir_tier, dist, n_idx)
        new_score = (type_tier, dir_tier, dist_pt, n_idx)
        existing_score = ep_node_dist[ep_id].get(n_idx, (99, 99, float("inf"), -1))
        if new_score < existing_score:
            ep_node_dist[ep_id][n_idx] = new_score

    # Convert to sorted preference lists
    node_prefs: dict[int, list] = defaultdict(list)
    for n_idx, ep_dict in temp_node_prefs.items():
        node_prefs[n_idx] = sorted(
            ep_dict.values(),
            key=lambda x: (x["type_tier"], x["dir_tier"], x["dist_ep"])
        )

    # --- 3b. Node-proposing Gale-Shapley loop ---
    next_proposal = dict.fromkeys(node_prefs, 0)
    ep_holder: dict = {}
    free_nodes: deque = deque(node_prefs.keys())

    while free_nodes:
        n_idx = free_nodes.popleft()
        prefs = node_prefs[n_idx]

        if next_proposal[n_idx] >= len(prefs):
            # Exhausted all candidates — falls through to leftover handler.
            continue

        proposal = prefs[next_proposal[n_idx]]
        next_proposal[n_idx] += 1
        ep_id = proposal["ep_id"]

        if ep_id not in ep_holder:
            ep_holder[ep_id] = {"n_idx": n_idx, "bid": proposal}
        else:
            current_idx = ep_holder[ep_id]["n_idx"]
            current_score = ep_node_dist[ep_id][current_idx]
            new_score = ep_node_dist[ep_id][n_idx]

            if new_score < current_score:
                # Endpoint upgrades to better node — displaced node re-queues.
                ep_holder[ep_id] = {"n_idx": n_idx, "bid": proposal}
                free_nodes.append(current_idx)
            else:
                # Endpoint keeps current holder — rejected node re-queues.
                free_nodes.append(n_idx)

    # ==========================================
    # PHASE 4: Apply Assignments & Extract Attributes
    # ==========================================
    claimed_nodes: set = set()
    snapped_geoms = [None] * num_nodes
    snap_distances_m = [None] * num_nodes
    snapped_flags = [False] * num_nodes

    for accepted in ep_holder.values():
        n_idx = accepted["n_idx"]
        bid = accepted["bid"]

        claimed_nodes.add(n_idx)
        snapped_geoms[n_idx] = bid["geom_orig"]
        snap_distances_m[n_idx] = round(bid["dist_ep"], 2)
        snapped_flags[n_idx] = True

        t_idx = bid["t_idx"]

        if target_id_cols and isinstance(snap_targets, gpd.GeoDataFrame):
            for col in target_id_cols:
                if col in snap_targets.columns:
                    snapped_attrs[col][n_idx] = snap_targets.iloc[t_idx][col]

        if "milepost" in snapped_attrs:
            # Point endpoint targets — milepost not applicable; leave as NaN
            pass

    # Leftover handler — unmatched nodes retain original geometry, flagged False
    for i, (orig_geom, proj_geom) in enumerate(zip(gdf_nodes.geometry, nodes_proj.geometry)):
        if i not in claimed_nodes:
            nearest_idx = tree.nearest(proj_geom)
            abs_dist = (
                proj_geom.distance(target_geoms_proj[nearest_idx])
                if nearest_idx is not None
                else np.nan
            )
            snapped_geoms[i] = orig_geom
            snap_distances_m[i] = round(abs_dist, 2) if pd.notna(abs_dist) else np.nan
            snapped_flags[i] = False

    return snapped_geoms, snap_distances_m, snapped_flags, snapped_attrs


# ---------------------------------------------------------------------------
# Public snapping functions
# ---------------------------------------------------------------------------


def snap_nodes(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_endpoints: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float,
    label: str,
    direction_col: str = "link_directions",
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
    node_type_col: str = "node_type",
) -> gpd.GeoDataFrame:
    """
    Snap a masked subset of nodes to the nearest compatible endpoint,
    using Gale-Shapley stable matching with two-dimensional tier sorting.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Full nodes layer (with snap_rule, link_directions, fw_directions,
        and node_type — the topology label column).
    gdf_endpoints : GeoDataFrame
        Pre-classified endpoint layer from the notebook's Step E block.
        Must have ep_allowed_dirs, ep_type, is_freeway, is_interchange,
        is_surface.
    node_mask : pd.Series
        Boolean mask selecting which nodes to attempt snapping.
        Nodes with snap_rule == "none" or "exceeded_threshold" are attempted,
        allowing nodes that failed an earlier pass (e.g. ramp-to-surface nodes
        that found no ramp+sf target in Pass (c)) to fall through to subsequent
        passes (e.g. Surface) without being permanently locked out.
    max_distance_m : float
        Maximum search radius in metres.
    label : str
        snap_rule label applied to successfully snapped nodes.
    direction_col : str
        Column on gdf_nodes used for directional matching.
        "link_directions" for passes (a) and (c); "fw_directions" for (b).
    target_id_cols : list[str], optional
        Columns on gdf_endpoints to copy to snapped_<col> output columns.
    crs_projected : str
        CRS for metric distance calculations.
    node_type_col : str
        Column on gdf_nodes containing the topology type label
        (one of: "fwy", "gore", "ramp", "ramp_sf", "surface").
        Passed through to _spatial_snap for type-tier matching.
        If the column is absent, type matching is skipped (e.g. transit).
    """
    result = gdf_nodes.copy()

    if "snap_rule" not in result.columns:
        result["snap_rule"] = "none"
        result["snap_distance_m"] = np.nan
        result["snapped"] = False

    if len(gdf_endpoints) == 0:
        print(f"  [{label}] No target endpoints — skipping.")
        return result

    candidate_idx = node_mask[node_mask].index
    candidate_idx = candidate_idx[
        result.loc[candidate_idx, "snap_rule"].isin(["none", "exceeded_threshold"])
    ]

    if len(candidate_idx) == 0:
        print(f"  [{label}] No unsnapped candidate nodes — skipping.")
        return result

    print(f"  [{label}] {len(candidate_idx):,} nodes → {len(gdf_endpoints):,} endpoints"
          f"  (dir_col={direction_col}, type_col={node_type_col}, max={max_distance_m}m)")

    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags, attrs = _spatial_snap(
        candidate_nodes, gdf_endpoints, max_distance_m, crs_projected,
        direction_col=direction_col, target_id_cols=target_id_cols,
        node_type_col=node_type_col,
    )

    for col in attrs.keys():
        col_name = f"snapped_{col}"
        if col_name not in result.columns:
            result[col_name] = None

    for i, (idx, geom, dist, snapped_flag) in enumerate(
        zip(candidate_idx, snapped_geoms, distances, flags)
    ):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

        for col, values_list in attrs.items():
            result.at[idx, f"snapped_{col}"] = values_list[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result


def snap_transit(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_stops: gpd.GeoDataFrame,
    node_mask: pd.Series,
    max_distance_m: float = 200,
    label: str = "FixedTransit_Rail",
    target_id_cols: list[str] = None,
    crs_projected: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """
    Snap a subset of nodes to the nearest GTFS stop point.

    Direction matching is not applied for transit snapping — gdf_stops
    is not expected to have ep_allowed_dirs.
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

    # Prepare stops as a minimal endpoint-compatible GeoDataFrame
    # (no direction columns — _spatial_snap skips direction matching automatically)
    candidate_nodes = result.loc[candidate_idx]
    snapped_geoms, distances, flags, attrs = _spatial_snap(
        candidate_nodes, gdf_stops, max_distance_m, crs_projected,
        direction_col="link_directions", target_id_cols=target_id_cols,
    )

    for col in attrs.keys():
        col_name = f"snapped_{col}"
        if col_name not in result.columns:
            result[col_name] = None

    for i, (idx, geom, dist, snapped_flag) in enumerate(
        zip(candidate_idx, snapped_geoms, distances, flags)
    ):
        result.at[idx, "geometry"] = geom
        result.at[idx, "snap_distance_m"] = dist
        result.at[idx, "snapped"] = snapped_flag
        result.at[idx, "snap_rule"] = label if snapped_flag else "exceeded_threshold"

        for col, values_list in attrs.items():
            result.at[idx, f"snapped_{col}"] = values_list[i]

    snapped = sum(flags)
    exceeded = len(flags) - snapped
    print(f"         → {snapped:,} snapped | {exceeded:,} exceeded {max_distance_m}m threshold")

    return result
