"""
CRS constants following the UGRC recommendation for Utah spatial data.

Background
----------
UGRC recommends NAD_1983_To_WGS_1984_5 when converting between WGS 84 and
NAD 83 for Utah data.
Reference: https://gis.utah.gov/blog/2021-06-23-choosing-right-transformation/

This transformation applies a 7-parameter Helmert shift (ESRI WKID 1515,
EPSG operation 1515 — "NAD83 to WGS 84 (5)"). It is NOT a null shift.

The alternative that ArcGIS assigns to all standard CONUS projected systems by
default — WGS_1984_(ITRF00)_To_NAD_1983 (ESRI code 108190) — uses slightly
different Helmert parameters. That ~0.03 ft (~9 mm) mismatch between the two
exceeds default geodatabase cluster tolerances and produces slivers in dissolve
and topology operations. The _5 method avoids this.

ESRI projection engine database (github.com/Esri/projection-engine-db-doc):
    NAD_1983_To_WGS_1984_5   ESRI WKID 1515
    Parameters (coordinate-frame convention, from EPSG:1515):
        X = -0.991 m   Y = 1.9072 m   Z = 0.5129 m
        rX = -0.0257899"  rY = -0.0096501"  rZ = -0.0116599"  scale = 0

pyproj / geopandas usage
------------------------
Pass PROJ_UTM12N directly to GeoDataFrame.to_crs(). The +towgs84 clause on
the target CRS encodes the _5 datum shift so pyproj applies exactly the UGRC-
recommended parameters when transforming from WGS 84 (EPSG:4326):

    from _src.crs_utils import PROJ_UTM12N
    gdf_utm = gdf_wgs84.to_crs(PROJ_UTM12N)

Convention note: PROJ4 +towgs84 uses the position-vector (Bursa-Wolf)
convention; EPSG:1515 uses coordinate-frame. The rotation signs are negated
between the two conventions. The translations (X, Y, Z) are identical.

    EPSG:1515 coordinate-frame:  rX = -0.0257899  rY = -0.0096501  rZ = -0.01166
    PROJ4 position-vector:       rX = +0.0257899  rY = +0.0096501  rZ = +0.01166

GDAL/OSR equivalent (requires: pip install gdal or conda install gdal)
----------------------------------------------------------------------
Place the same +towgs84 clause on the target SRS. OSR resolves the datum
shift in the same direction (local datum → WGS84) as PROJ4:

    from osgeo import osr, ogr
    import shapely.wkt

    tgt = osr.SpatialReference()
    tgt.SetFromUserInput(
        "+proj=utm +zone=12 +ellps=GRS80 "
        "+towgs84=-0.991,1.9072,0.5129,0.0257899,0.0096501,0.01166,0 "
        "+units=m +no_defs"
    )
    tgt.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    ct = osr.CoordinateTransformation(src, tgt)

    # Apply to a Shapely geometry via OGR round-trip:
    ogr_geom = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
    ogr_geom.Transform(ct)
    result = shapely.wkt.loads(ogr_geom.ExportToWkt())
"""

# UGRC-recommended target CRS: WGS 84 → NAD 83 UTM Zone 12N via NAD_1983_To_WGS_1984_5.
# +towgs84 encodes the ESRI WKID 1515 / EPSG:1515 Helmert shift in position-vector
# convention (rotations negated vs EPSG:1515's coordinate-frame values).
# Use with: gdf.to_crs(PROJ_UTM12N)
PROJ_UTM12N: str = (
    "+proj=utm +zone=12 +ellps=GRS80 "
    "+towgs84=-0.991,1.9072,0.5129,0.0257899,0.0096501,0.01166,0 "
    "+units=m +no_defs"
)
