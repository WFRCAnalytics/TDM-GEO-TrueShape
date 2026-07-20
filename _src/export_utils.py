"""
Utility functions for exporting the final true-shape network as a drop-in
replacement for the raw WFv1000 MasterNet Node/Link shapefiles.

Design contract
---------------
- Which rows/columns feed the final export (qa_status filters, rename maps)
  is a notebook-level decision -- this module only reshapes and writes
  already-decided data.
- Schema fidelity (field names, order, DBF width/precision) is enforced by
  copying the schema straight from the original .shp rather than letting
  geopandas infer one, so the output can be swapped in for the original
  file without surprising downstream TDM software.

Public API
----------
    conform_to_reference_schema(gdf, reference_props, rename_map=None) -> GeoDataFrame
        Rename columns per rename_map, then select exactly reference_props
        (+ geometry) in that order. Raises if any reference column is still
        missing afterwards.

    write_matching_shapefile(gdf, out_path, schema_source_path)
        Write gdf as an ESRI Shapefile using the field schema (names, types,
        DBF widths/precision) and CRS read directly from schema_source_path,
        so the output is byte-schema-compatible with the original source
        file rather than whatever geopandas would auto-infer.
"""

from __future__ import annotations

from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


def conform_to_reference_schema(
    gdf: gpd.GeoDataFrame,
    reference_props: list[str],
    rename_map: dict[str, str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Reshape gdf to exactly reference_props (+ geometry), in that order.

    Parameters
    ----------
    gdf : GeoDataFrame
    reference_props : list[str]
        Target column names, in the order they must appear in the output
        (excluding geometry). Typically read straight from a reference
        schema, e.g. `list(fiona.open(shp_path).schema["properties"])`.
    rename_map : dict[str, str], optional
        Applied to gdf's columns before selection, e.g. to undo an
        ArcGIS-REST-introduced rename such as EXTERNAL_ -> EXTERNAL.

    Returns
    -------
    GeoDataFrame with columns exactly [*reference_props, "geometry"].
    """
    result = gdf.rename(columns=rename_map or {})
    missing = [c for c in reference_props if c not in result.columns]
    if missing:
        raise ValueError(
            f"conform_to_reference_schema: {len(missing)} reference column(s) "
            f"not found on gdf after renaming (sample: {missing[:10]})."
        )
    return result[[*reference_props, "geometry"]].copy()


def _coerce_value(value, ftype: str):
    """Cast a single property value to match a fiona schema field type string."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    if ftype.startswith("int"):
        return int(value)
    if ftype.startswith("float"):
        return float(value)
    if ftype.startswith("str"):
        return str(value)
    return value


def write_matching_shapefile(
    gdf: gpd.GeoDataFrame,
    out_path: Path,
    schema_source_path: Path,
) -> None:
    """
    Write gdf as an ESRI Shapefile whose field schema (names, order, DBF
    widths/precision) and CRS are copied directly from schema_source_path.

    geopandas' own to_file() infers a schema from gdf's dtypes, which can
    silently drift from the original file's DBF field widths/precision
    (e.g. a float column re-inferred with far more decimal places than the
    source ever had). Copying the schema from the source guarantees the
    output is a faithful drop-in replacement.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must already have exactly the columns named in the source schema's
        properties (see conform_to_reference_schema).
    out_path : Path
        Destination .shp path. Parent directory is created if missing.
    schema_source_path : Path
        Path to the original .shp file to copy the field schema and CRS from.
    """
    with fiona.open(schema_source_path) as src:
        schema = src.schema
        crs = src.crs

    prop_names = list(schema["properties"].keys())
    missing = [c for c in prop_names if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"write_matching_shapefile: gdf is missing column(s) required by "
            f"{Path(schema_source_path).name}'s schema: {missing}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prop_dicts = gdf[prop_names].to_dict(orient="records")
    geoms = gdf.geometry.values

    records = [
        {
            "geometry": shapely.geometry.mapping(geom),
            "properties": {
                name: _coerce_value(raw_props[name], schema["properties"][name])
                for name in prop_names
            },
        }
        for geom, raw_props in zip(geoms, prop_dicts)
    ]

    with fiona.open(out_path, "w", driver="ESRI Shapefile", schema=schema, crs=crs) as dst:
        dst.writerecords(records)
