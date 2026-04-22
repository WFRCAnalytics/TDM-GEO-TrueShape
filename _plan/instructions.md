**Role & Context:**
Act as an expert Geospatial Data Scientist and Travel Demand Modeling (TDM) Engineer.

I am working on a network conflation project to snap abstract TDM network nodes to physical UGRC/UDOT roadway centerline endpoints. I have attached my current workflow in two files: `01_node_classification.qmd` (which orchestrates the logic) and `node_utils.py` (which handles the spatial matching using a Gale-Shapley stable matching algorithm).

**The Breakthrough (New SQL Definitions):**
Previously, the classification of links and segments was messy. I have just finalized a strict, mutually exclusive categorization for both our TDM links and physical centerlines using SQL. Every link/segment now falls strictly into one of three buckets.

Here are the 3 classes for the **Roadway Centerlines** (Physical Geography):
1. Freeway Mainline:
```sql
(
  "DOT_FCLASS" IN ('Interstate', 'Other Freeway')
  OR "DOT_RTNAME" IN ('0015', '0067', '0080', '0084', '0089', '0092', '0154', '0177', '0201', '0215')
  OR (
    "DOT_FCLASS" = 'Principal Arterial'
    AND ("NAME" LIKE '%NB%' OR "NAME" LIKE '%SB%' OR "NAME" LIKE '%EB%' OR "NAME" LIKE '%WB%')
  )
)
AND "DOT_RTNAME" NOT LIKE '%R%'
AND "DOT_RTNAME" NOT LIKE '%C%'
AND "DOT_RTNAME" <> 'DupRoute'
```
2. Ramps and CDs:
```sql
("DOT_RTNAME" LIKE '%R%' OR "DOT_RTNAME" LIKE '%C%') AND "DOT_RTNAME" <> 'DupRoute'
```
3. Surface Streets:
```sql
"DOT_FCLASS" NOT IN ('Interstate', 'Other Freeway', 'Institutional', '')
AND "DOT_FCLASS" IS NOT NULL
AND "DOT_RTNAME" NOT IN ('0015', '0067', '0080', '0084', '0089', '0092', '0154', '0177', '0201', '0215')
AND NOT (
  "DOT_FCLASS" = 'Principal Arterial'
  AND ("NAME" LIKE '%NB%' OR "NAME" LIKE '%SB%' OR "NAME" LIKE '%EB%' OR "NAME" LIKE '%WB%')
)
AND "DOT_RTNAME" NOT LIKE '%R%'
AND "DOT_RTNAME" NOT LIKE '%C%'
AND "DOT_RTNAME" <> 'DupRoute'
```

And here are the exact corresponding 3 classes for the **TDM Network Links** (Abstract Graph):
1. Freeway Mainline (No HOV/Managed Lanes):
```sql
"LN_2027" > 0 AND (
  "FT_2027" IN (12, 13, 14, 15, 22, 23, 24, 25, 26, 32, 33, 34, 35, 36, 40)
  OR ("FT_2027" = 2 AND "DIRECTION" = 1)
)
```
2. Ramps and CDs:
```sql
"LN_2027" > 0 AND "FT_2027" IN (20, 21, 28, 29, 30, 31, 41, 42)
```
3. Surface Streets (No centroid connectors, FT=1):
```sql
"LN_2027" > 0 AND "FT_2027" IN (2, 3, 4, 5, 6, 7)
AND NOT ("FT_2027" = 2 AND "DIRECTION" = 1)
```

**The Challenge:**
A node (or centerline endpoint) connects to one or more of these links. Therefore, the topology of a node is defined by the *combination* of link types it touches. For example:
* Pure Freeway Node (touches only Freeway links)
* Pure Ramp Node (touches only Ramp links)
* Pure Surface Node (touches only Surface links)
* Gore Node (touches Freeway + Ramp)
* Ramp Terminal (touches Ramp + Surface)
* Freeway-Surface Crossing (touches Freeway + Surface)

Currently, my code relies heavily on inheriting string-based cardinal directions ("NB", "SB") to guide the Gale-Shapley matching. This works for freeways, but completely falls apart for complex system interchanges (loops and directional ramps), causing mid-ramp nodes to snap to the wrong geometries just because the strings matched.

**The Request:**
I want to **keep the Gale-Shapley algorithm** (`_spatial_snap`), but I need to completely refactor how we define the tiers and generate the bids. Please review the attached code and provide a revised approach and updated Python code:

1. **Refactor Classification (QMD Part A & B)**: Update the logic so that `node_type` and `ep_type` strictly rely on the combinatorics of my three new mutually exclusive link/segment definitions outlined in the SQL above.
2. **Revamp the Tiering System (`node_utils.py`)**: Redefine the `_TYPE_TIER` dictionary to perfectly match these new topological combinations.
3. **Solve the Directional Bid for Ramps**: Since string-based direction ("NB", "SB") is useless for a 270-degree loop ramp, how should we adapt the Gale-Shapley bid generation (`Phase 2: Compatibility Check & Bid Generation`) for the Ramp tier? Should we calculate and compare the geometric azimuth/heading vectors of the links entering/exiting the node versus the centerline endpoints to generate the `dir_tier` score, instead of comparing strings?
4. **Provide the updated code blocks** for both the notebook classification step and the `node_utils.py` matching logic.
