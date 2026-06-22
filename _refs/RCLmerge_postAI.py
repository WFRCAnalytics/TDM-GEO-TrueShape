Here’s a much cleaner, modular, and more maintainable rewrite of your ArcPy script. It removes most of the duplicated code by using reusable helper functions and configuration dictionaries.

Key improvements:

* Eliminates repetitive blocks
* Centralizes SQL generation
* Uses loops for all level processing
* Easier to modify counties, levels, and classes
* More readable and maintainable
* Keeps the exact same workflow logic

```python
# -*- coding: utf-8 -*-
"""
Refactored ArcPy workflow for RCL merge processing
"""

import arcpy
from pathlib import Path

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

GDB = Path(
    r"C:\Users\bgranberg\Documents\ArcGIS\Projects\FunctionalClassEditsApril\Default.gdb"
)

arcpy.env.workspace = str(GDB)
arcpy.env.scratchWorkspace = str(GDB)
arcpy.env.overwriteOutput = True

ROADS = "UTRANS.Roads_Edit.Simple"

COUNTIES = (
    "'49003', '49011', '49035', "
    "'49049', '49057', '49045'"
)

FREEWAY_CLASSES = (
    "DOT_FCLASS IN ('Interstate', 'Other Freeway') "
    "OR DOT_RTNAME LIKE '%C%' "
    "OR DOT_RTNAME LIKE '%R%'"
)

SURFACE_CLASSES = (
    "DOT_FCLASS IN ("
    "'Local', "
    "'Major Collector', "
    "'Minor Arterial', "
    "'Minor Collector', "
    "'Principal Arterial'"
    ")"
)

COUNTY_SQL = (
    f"(COUNTY_L IN ({COUNTIES}) "
    f"OR COUNTY_R IN ({COUNTIES}))"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def fc(name):
    """Return feature class path inside GDB."""
    return str(GDB / name)


def make_layer(name, where_clause):
    """Create feature layer."""
    arcpy.management.MakeFeatureLayer(
        in_features=ROADS,
        out_layer=name,
        where_clause=where_clause
    )
    return name


def dissolve_to_singlepart(in_fc, out_prefix):
    """Dissolve and multipart-to-singlepart."""
    dissolved = fc(f"{out_prefix}_Dissolve")
    singlepart = fc(f"{out_prefix}_MP2SP")

    arcpy.management.Dissolve(
        in_features=in_fc,
        out_feature_class=dissolved,
        multi_part="MULTI_PART"
    )

    arcpy.management.MultipartToSinglepart(
        in_features=dissolved,
        out_feature_class=singlepart
    )

    return singlepart


def split_at_points(in_fc, split_points, out_name, radius="0.1 Meters"):
    """Split line at points."""
    out_fc = fc(out_name)

    arcpy.management.SplitLineAtPoint(
        in_features=in_fc,
        point_features=split_points,
        out_feature_class=out_fc,
        search_radius=radius
    )

    return out_fc


def process_network(
    level,
    feature_type,
    where_clause,
    split_points,
    radius="0.1 Meters"
):
    """
    Generic workflow:
    Make Layer -> Dissolve -> MultipartToSinglepart -> SplitLineAtPoint
    """

    layer_name = f"FL_{feature_type}Lvl{level}"

    layer = make_layer(layer_name, where_clause)

    singlepart = dissolve_to_singlepart(
        layer,
        f"{feature_type}Lvl{level}"
    )

    result = split_at_points(
        singlepart,
        split_points,
        f"{feature_type}Lvl{level}_Resplit",
        radius
    )

    return result


def build_vert_clause(level, inclusive=True):
    """Build VERT_LEVEL SQL clause."""

    if inclusive:
        return f"VERT_LEVEL IN ('{level}')"

    excluded = "', '".join(str(v) for v in level)
    return f"VERT_LEVEL NOT IN ('{excluded}')"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def RCLmerge():

    arcpy.ImportToolbox(
        r"c:\program files\arcgis\pro\Resources\ArcToolbox\toolboxes\Data Management Tools.tbx"
    )

    # -------------------------------------------------------------------------
    # Build endpoint split points
    # -------------------------------------------------------------------------

    all_sql = f"""
        VERT_LEVEL IN ('0', '1', '2', '3')
        AND ({FREEWAY_CLASSES}
             OR {SURFACE_CLASSES})
        AND {COUNTY_SQL}
    """

    all_layer = make_layer("FL_FwyALL", all_sql)

    endpoints = fc("FWYAllLevels_FV2PEndpoints")

    arcpy.management.FeatureVerticesToPoints(
        in_features=all_layer,
        out_feature_class=endpoints,
        point_location="BOTH_ENDS"
    )

    endpoint_counts = fc("FWY_FV2PEndpointsCount")

    arcpy.analysis.SpatialJoin(
        target_features=endpoints,
        join_features=endpoints,
        out_feature_class=endpoint_counts,
        search_radius="1 Meters"
    )

    split_points = "FWY_FV2PEndpointsCount_3Plus"

    arcpy.management.MakeFeatureLayer(
        in_features=endpoint_counts,
        out_layer=split_points,
        where_clause="Join_Count >= 3"
    )

    # -------------------------------------------------------------------------
    # Process freeway levels
    # -------------------------------------------------------------------------

    freeway_results = []

    freeway_config = {
        0: ("NOT IN ('1', '2', '3')", "0.1 Meters"),
        1: ("IN ('1')", "1 Meters"),
        2: ("IN ('2')", "0.1 Meters"),
        3: ("IN ('3')", "0.1 Meters"),
    }

    for level, (vert_sql, radius) in freeway_config.items():

        sql = f"""
            VERT_LEVEL {vert_sql}
            AND ({FREEWAY_CLASSES})
            AND {COUNTY_SQL}
        """

        result = process_network(
            level=level,
            feature_type="FWY",
            where_clause=sql,
            split_points=split_points,
            radius=radius
        )

        freeway_results.append(result)

    # -------------------------------------------------------------------------
    # Process surface levels
    # -------------------------------------------------------------------------

    surface_results = []

    surface_config = {
        0: ("NOT IN ('1', '2', '3')", "1 Meters"),
        1: ("NOT IN ('0', '2', '3')", "0.1 Meters"),
        2: ("NOT IN ('0', '1', '3')", "0.1 Meters"),
        3: ("NOT IN ('0', '1', '2')", "0.1 Meters"),
    }

    for level, (vert_sql, radius) in surface_config.items():

        sql = f"""
            VERT_LEVEL {vert_sql}
            AND ({SURFACE_CLASSES})
            AND {COUNTY_SQL}
        """

        result = process_network(
            level=level,
            feature_type="Surf",
            where_clause=sql,
            split_points=split_points,
            radius=radius
        )

        surface_results.append(result)

    # -------------------------------------------------------------------------
    # Create merged geometry dataset
    # -------------------------------------------------------------------------

    output_fc = fc("RCL2TDMGeometry")

    arcpy.management.CopyFeatures(
        in_features=freeway_results[0],
        out_feature_class=output_fc
    )

    append_inputs = freeway_results[1:] + surface_results

    arcpy.management.Append(
        inputs=append_inputs,
        target=output_fc,
        schema_type="NO_TEST"
    )

    # -------------------------------------------------------------------------
    # Geometry attributes
    # -------------------------------------------------------------------------

    arcpy.management.CalculateGeometryAttributes(
        in_features=output_fc,
        geometry_property=[
            ["MIDX", "INSIDE_X"],
            ["MIDY", "CENTROID_Y"]
        ],
        coordinate_system=(
            'PROJCS["NAD_1983_UTM_Zone_12N",'
            'GEOGCS["GCS_North_American_1983",'
            'DATUM["D_North_American_1983",'
            'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
            'PRIMEM["Greenwich",0.0],'
            'UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],'
            'PARAMETER["False_Easting",500000.0],'
            'PARAMETER["False_Northing",0.0],'
            'PARAMETER["Central_Meridian",-111.0],'
            'PARAMETER["Scale_Factor",0.9996],'
            'PARAMETER["Latitude_Of_Origin",0.0],'
            'UNIT["Meter",1.0]]'
        )
    )

    arcpy.management.AddField(
        in_table=output_fc,
        field_name="MidPointID",
        field_type="TEXT",
        field_length=30
    )

    arcpy.management.CalculateField(
        in_table=output_fc,
        field="MidPointID",
        expression='str(int(!MIDX!)) + "|" + str(int(!MIDY!))'
    )

    print("RCL merge processing complete.")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    RCLmerge()
```
