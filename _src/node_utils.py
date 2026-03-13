"""
Utility functions for network node classification.
"""

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Classification config
# Maps each road type label to the FT_<year> values that define it.
# Update ft_year argument in classify_nodes() to switch between years.
# ---------------------------------------------------------------------------
ROAD_TYPE_FT = {
    "Freeway": list(range(20, 28)) + list(range(30, 40)),
    "Expressway": [12, 13, 14, 15],
    "Arterial": [2, 3],
    "Collector": [4, 5],
    "Local": [6, 7],
    "Ramp": [28, 29, 41, 42],
    "CentroidConnector": [1],
    "FixedTransit": [70, 80],
}

# N value ranges that independently indicate a FixedTransit node (inclusive)
FIXED_TRANSIT_N_RANGES = [(10_000, 19_999), (50_000, 59_999)]


def _nodes_connected_to(links: pd.DataFrame, ft_col: str, ft_values: list[int]) -> set:
    """
    Return the set of node IDs (from columns A and B) connected to links
    whose ft_col value is in ft_values.
    """
    mask = links[ft_col].isin(ft_values)
    filtered = links.loc[mask, ["A", "B"]]
    return set(filtered["A"]).union(filtered["B"])


def _fixed_transit_by_n_range(n_series: pd.Series) -> pd.Series:
    """
    Return boolean Series: True if N falls within any FixedTransit N range.
    pandas .between() is inclusive on both ends by default.
    """
    mask = pd.Series(False, index=n_series.index)
    for lo, hi in FIXED_TRANSIT_N_RANGES:
        mask |= n_series.between(lo, hi)
    return mask


def classify_nodes(
    gdf_nodes: gpd.GeoDataFrame, gdf_links: gpd.GeoDataFrame, ft_year: int = 2023
) -> gpd.GeoDataFrame:
    """
    Add road-type flag columns and a LinkCount column to the nodes GeoDataFrame.

    Road-type flag columns (Boolean):
        Freeway, Expressway, Arterial, Collector, Local,
        Ramp, CentroidConnector

    FixedTransit columns (Boolean, two independent methods):
        FixedTransit_byFT  — True if node appears in FT_<year>=70/80 links
        FixedTransit_byN   — True if N falls within transit N ranges
                             (10,000–19,999 or 50,000–59,999, inclusive)

    LinkCount column (int):
        Number of links whose A or B matches a given node N.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, and FT_<ft_year>.
    ft_year : int, optional
        Year suffix for the functional type column. Default is 2023,
        which resolves to column FT_2023.

    Returns
    -------
    GeoDataFrame
        Copy of gdf_nodes with new classification columns appended.

    """
    ft_col = f"FT_{ft_year}"
    if ft_col not in gdf_links.columns:
        ft_cols = [c for c in gdf_links.columns if c.startswith("FT_")]
        raise ValueError(f"Column '{ft_col}' not found in links. Available FT_ columns: {ft_cols}")

    result = gdf_nodes.copy()
    n = result["N"]
    links = gdf_links[["A", "B", ft_col]]

    # -- Standard road-type flags -------------------------------------------
    for label, ft_values in ROAD_TYPE_FT.items():
        if label == "FixedTransit":
            continue  # handled separately below
        connected = _nodes_connected_to(links, ft_col, ft_values)
        result[label] = n.isin(connected)

    # -- FixedTransit (two independent methods) -----------------------------
    result["FixedTransit_byFT"] = n.isin(
        _nodes_connected_to(links, ft_col, ROAD_TYPE_FT["FixedTransit"])
    )
    result["FixedTransit_byN"] = _fixed_transit_by_n_range(n)

    # -- LinkCount ----------------------------------------------------------
    all_nodes_in_links = pd.concat([links["A"], links["B"]])
    link_counts = all_nodes_in_links.value_counts()
    result["LinkCount"] = n.map(link_counts).fillna(0).astype(int)

    return result
