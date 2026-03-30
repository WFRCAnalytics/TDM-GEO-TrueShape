# WFRC Network Conflation & True-Shape Routing Project

## 🎯 Project Objective

The goal of this project is to convert the Wasatch Front Regional Council (WFRC) skeleton travel demand model network (WFv1000 MasterNet) into a "true-shape" geographic network. This is achieved by conflating the abstract model nodes and links to high-fidelity, physical roadway centerlines provided by the Utah Geospatial Resource Center (UGRC) and Utah DOT, establishing a GERS-like (Global Entity Reference System) relational lineage.

## 🗂️ Data Sources

- **Model Links:** WFRC MasterNet (ArcGIS REST). Attributes: `A`, `B`, `DIRECTION`, `FT_2023`, `LN_2023`.
- **Model Nodes:** WFRC MasterNet (ArcGIS REST). Attributes: `N`.
- **Roadway Centerlines:** UGRC/UDOT (ArcGIS REST). Attributes: `DOT_FCLASS`, `POSTTYPE`, `DOT_RTNAME`, `FULLNAME`, `CARTOCODE`.
- **Transit Stops:** UTA GTFS Feed.

## 🏗️ Architecture & Separation of Concerns

The project strictly separates notebook execution from algorithmic heavy lifting:

1. **`00_data_preparation.qmd`:** Data fetching, caching to local `_data/raw/` GeoPackages.
2. **`01_node_classification.qmd`:** Topological node classification, mask generation, snapping execution, and metric generation.
3. **`_src/node_utils.py`:** Pure Python backend containing the spatial math (`shapely`), indexing (`STRtree`), and algorithm implementations. **Rule:** Filtering logic belongs in the `.qmd`; `node_utils.py` only accepts pre-filtered data.

## 🧠 Core Methodologies

### 1. Topological Node Classification

Nodes are classified based on the functional types (`FT_2023`) of the links connected to them using `nodes_on()`.

- **Strict Mutually Exclusive Flags:** Pure mainline freeway nodes (`FT 30-36`), pure HOV nodes (`FT 37-38`), and Managed Access nodes (`FT 39`) are strictly separated. Non-physical geometries (like parallel HOT lanes) are excluded from spatial snapping to prevent geometric distortion.
- **Junction vs. Mainline:** Freeways are snapped in a tiered approach. "Pure Mainline" nodes snap strictly to `POSTTYPE == "FWY"`. "Junction/Gore" nodes (touching both freeways and ramps) can snap to `FWY` or `RAMP`.

### 2. Direction-Aware Spatial Indexing

To prevent the "divided highway collapse" (where NB and SB nodes both snap to the closest physical centerline), the spatial index enforces directionality.

- **Node Direction:** Inferred from the `DIRECTION` column of connected links.
- **Centerline Direction:** Inferred via regex on `FULLNAME` (e.g., "I-15 NB FWY") or the 5th character of the UDOT LRS `DOT_RTNAME` (P = Positive/NB/EB, N = Negative/SB/WB).
- **Alias Mapping:** Directions are grouped into "P" and "N" groups prior to set intersection to handle micro-geometry curves (e.g., an EB model link matching a NB physical centerline where the highway bends).

### 3. Snapping Algorithm: Global Greedy Point-to-Point

The core snapping engine (`_spatial_snap` in `node_utils.py`) uses a highly optimized Global Greedy Assignment to resolve "bidding wars" (endpoint collisions).

- **Vectorized Querying:** `shapely.STRtree.query()` and `shapely.distance()` are used to bulk-calculate every legally compatible (direction-matched) Node-to-Endpoint pair within the `max_distance_m` threshold.
- **Global Sorting:** Pairs are flattened and sorted globally from shortest absolute distance to longest.
- **Greedy Claiming:** A sequential `for` loop claims endpoints. A node claims the absolute closest endpoint *if* neither the node nor the endpoint has already been claimed.
- *NOTE FOR AI AGENTS:* Do **NOT** attempt to "optimize" this by switching to a "Segment-First" or "Point-to-Line-to-Point" nearest-neighbor approach. We tested this. It fails catastrophically at complex interchanges due to the Pigeonhole Principle (restricting a segment to only two endpoints causes mass rejections when \>2 model nodes map to the same interchange geometry). **The Point-to-Point Greedy pool is mandatory.**

### 4. Hierarchical Execution

Snapping must occur in a strict hierarchical order. First-call-wins logic ensures nodes snapped in Step 1 are skipped in Step 2.

1. **Freeways** (Mainline pass, then Junction pass).
2. **Collector-Distributor (CD) & Ramps**.
3. **Surface Streets** (Arterials, Collectors, Locals).
4. **Fixed Transit** (Rail stops).

## 📊 Evaluation & Metrics

Algorithm performance is quantitatively evaluated using Empirical Cumulative Distribution Function (ECDF) plots and KPI tables (Median, P90, Max displacement).

- **Visual Debugging:** Failed or displaced nodes are exported to `debug_freeway_snapping.gpkg` containing `displacement_lines` (LineStrings connecting original to snapped coordinates) for QGIS rendering.

## 🚀 Future Roadmap: GERS Lineage & True-Shape Links

- **Target IDs:** The snapping algorithm must be updated to pass the persistent physical ID (e.g., `OBJECTID`) of the claimed centerline back to the model node (`snapped_target_id`).
- **Link Routing:** Once `A` and `B` nodes possess physical Target IDs, link true-shape conflation will be solved deterministically via shortest-path routing between the two known physical centerlines along the UGRC network, storing the complete array of traversed segments to build a true GERS linkage.
