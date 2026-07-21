"""
Utility functions for downloading ArcGIS feature layers.
"""

import geopandas as gpd
from arcgis.features import FeatureLayer


def fetch_feature_layer(
    service_url: str, gis, out_sr: int = 26912, where: str = "1=1", out_fields: str = "*"
) -> gpd.GeoDataFrame:
    """
    Fetch an ArcGIS feature layer and return as a GeoDataFrame.

    Parameters
    ----------
    service_url : str
        Full URL to the ArcGIS FeatureServer layer (must end with layer index, e.g. /0).
    gis : arcgis.gis.GIS
        Authenticated or anonymous GIS connection.
    out_sr : int, optional
        Output spatial reference EPSG code. Default is 26912 (NAD83 UTM Zone 12N).
        WFRC and UGRC services store data natively in 26912; requesting it here
        means the server returns coordinates as-is with no server-side reprojection
        and no client-side datum transformation.
    where : str, optional
        SQL where clause to filter features. Default is '1=1' (all features).
    out_fields : str, optional
        Comma-separated field names to return. Default is '*' (all fields).

    Returns
    -------
    geopandas.GeoDataFrame
        The fetched data as a GeoDataFrame.

    """
    print(f"Fetching: {service_url}")
    layer = FeatureLayer(service_url, gis=gis)

    # Large hosted layers (e.g. UtahRoads, 400k+ features) are fetched by the
    # ArcGIS API in pages of `maxRecordCount` via resultOffset/resultRecordCount.
    # Without an explicit, stable sort order, the server does not guarantee a
    # consistent row ordering across separate paged requests -- page windows can
    # drift, silently duplicating some features across two pages while dropping
    # others in the gap between pages. Sorting by the OID field makes paging
    # deterministic and gap-free.
    oid_field = layer.properties.objectIdField

    try:
        feature_set = layer.query(
            where=where,
            out_fields=out_fields,
            return_geometry=True,
            out_sr=out_sr,
            order_by_fields=f"{oid_field} ASC",
        )
        print(f"Success! Fetched {len(feature_set.features)} features.")
    except Exception as e:
        print(f"Error querying layer: {e}")
        raise

    gdf = gpd.GeoDataFrame(feature_set.sdf, geometry="SHAPE").set_crs(epsg=out_sr)

    # Verify no records were dropped or duplicated during pagination.
    server_count = layer.query(where=where, return_count_only=True)
    n_unique_oids = gdf[oid_field].nunique()
    if n_unique_oids != server_count:
        raise RuntimeError(
            f"Pagination mismatch for {service_url}: server reports "
            f"{server_count} features matching '{where}', but fetched "
            f"{len(gdf)} rows with {n_unique_oids} unique {oid_field} values. "
            "Records may have been dropped or duplicated during paging."
        )
    n_dupe_rows = len(gdf) - n_unique_oids
    if n_dupe_rows:
        print(f"Note: dropping {n_dupe_rows} duplicate rows (same {oid_field}).")
        gdf = gdf.drop_duplicates(subset=[oid_field]).reset_index(drop=True)

    return gdf
