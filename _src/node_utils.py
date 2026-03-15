"""
Utility functions for network node classification.

Exports two mask helpers and one counting function:

    ft_mask(gdf_nodes, gdf_links, ft_col, ft_values)
        Boolean Series — True if a node's N appears in A or B of any link
        whose ft_col value is in ft_values.

    n_range_mask(gdf_nodes, *ranges)
        Boolean Series — True if a node's N falls within any of the given
        (lo, hi) ranges, inclusive on both ends.

    count_node_links(gdf_nodes, gdf_links)
        Returns a copy of gdf_nodes with a LinkCount column appended.

Classification logic (which FT codes map to which road type, etc.) lives
entirely in the calling notebook — not here.
"""

import geopandas as gpd
import pandas as pd


def ft_mask(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
    ft_col: str,
    ft_values: list[int],
) -> pd.Series:
    """
    Return a boolean Series: True if the node's N appears in A or B of any
    link whose ft_col value is in ft_values.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    gdf_links : GeoDataFrame
        Links layer. Must contain columns A, B, and ft_col.
    ft_col : str
        Name of the functional type column in gdf_links (e.g. 'FT_2023').
    ft_values : list of int
        FT values that define this road type.

    Returns
    -------
    pd.Series
        Boolean Series aligned to gdf_nodes index.
    """
    if ft_col not in gdf_links.columns:
        ft_cols = [c for c in gdf_links.columns if c.startswith("FT_")]
        raise ValueError(
            f"Column '{ft_col}' not found in links. "
            f"Available FT_ columns: {ft_cols}"
        )
    mask = gdf_links[ft_col].isin(ft_values)
    connected = set(gdf_links.loc[mask, "A"]).union(gdf_links.loc[mask, "B"])
    return gdf_nodes["N"].isin(connected)


def n_range_mask(
    gdf_nodes: gpd.GeoDataFrame,
    *ranges: tuple[int, int],
) -> pd.Series:
    """
    Return a boolean Series: True if the node's N falls within any of the
    given ranges. Ranges are inclusive on both ends.

    Parameters
    ----------
    gdf_nodes : GeoDataFrame
        Nodes layer. Must contain column N.
    *ranges : tuple of (int, int)
        One or more (lo, hi) range tuples.

    Returns
    -------
    pd.Series
        Boolean Series aligned to gdf_nodes index.

    Example
    -------
    n_range_mask(gdf_nodes, (10_000, 19_999), (50_000, 59_999))
    """
    mask = pd.Series(False, index=gdf_nodes.index)
    for lo, hi in ranges:
        mask |= gdf_nodes["N"].between(lo, hi)
    return mask


def count_node_links(
    gdf_nodes: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
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
