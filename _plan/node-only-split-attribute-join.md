# True-Shape Link Generation: Node-Only Splitting + Direct Attribute Join

## Context

The current `02_create_link.qmd` → `03_transfer_attributes.qmd` pipeline builds true-shape
links by: (1) splitting `CandidateTDMRoadLinks` (already pre-cut at arcpy-style junction
points) at snapped node positions, then (2) collapsing model-link pseudonode chains and
running a constrained BFS through the resulting piece graph to assemble one merged
geometry per dissolved model link — falling back to a **synthetic straight line** between
node points whenever the BFS fails.

This has two problems the user wants fixed:
1. The split points on the true-shape geometry are partly inherited from
   `CandidateTDMRoadLinks`'s pre-existing junction cuts, not purely from where model nodes
   snapped. The user wants splitting defined **only** by snapping.
2. Failed BFS assembly produces fabricated straight-line geometry. The user wants **every**
   physical centerline piece retained regardless of whether attribute transfer succeeds —
   attributes should be left blank rather than inventing geometry.

Investigation confirmed a key empirical fact that shapes the design: pseudonodes
(`is_pseudo=True`, 8,885 nodes) are **never** snapped (`01_node_classification.qmd:1002-1007`,
verified: 0 of 8,885 pseudonodes have `snapped=True`). But snapping coverage overall is
sparse — only 3,670 of 21,953 nodes (~17%) are currently snapped — so many **non-pseudo**
real nodes are also unsnapped (`snap_rule` = `none` or `exceeded_threshold`). This matters:
the set of nodes that act as "pass-through" (no split occurs there) for the new pipeline is
**all unsnapped nodes**, a superset of `is_pseudo`, not `is_pseudo` itself.

Also confirmed: the filtered model link table (`LN_2027>0 AND FT_2027 NOT IN (1,37,38,39)`,
28,101 rows) has zero duplicate directed `(A,B)` pairs, and 24,718/28,101 links have their
reverse `(B,A)` also present — so a direction-sensitive dictionary join is sound and simple.

## Approach

### Stage 2 (`02_create_link.qmd`) — split geometry only at snap points

Replace "split each pre-cut `CandidateTDMRoadLinks` row independently" with "fully
re-dissolve the network per `VERT_LEVEL`, then split only at snapped positions":

1. Load `CandidateTDMRoadLinks.gpkg` as today (still the snap-target layer used unchanged
   by `01_node_classification.qmd` — that notebook is untouched).
2. **New**: for each `VERT_LEVEL` present (0, 1, 2), reuse
   `centerline_utils.dissolve_and_singlepart` (existing function, already does
   `unary_union` → `linemerge` → explode) across **all** rows of that level — freeway and
   surface rows combined, not kept separate as `00_data_preparation.qmd`'s `rcl_merge` does.
   Concatenate levels → `gdf_fulldissolve`. VERT_LEVEL separation is kept (existing
   documented rule: merging elevations would fuse overpasses with roads beneath them).
   Mixing freeway+surface within a level is intentional: it lets a ramp that ends inside an
   arterial intersection (today an artificial `CandidateTDMRoadLinks` boundary, since
   `rcl_merge` dissolves fwy/surface pools separately before splitting) merge into one
   chain, exactly matching "splits defined only by snapping."
3. Run `resolve_snap_coords` (unchanged function) against `gdf_fulldissolve` instead of
   `CandidateTDMRoadLinks` directly.
4. Run `split_candidate_links` (unchanged function) against `gdf_fulldissolve`, passing
   `id_cols=()` — a re-dissolved row no longer maps 1:1 to a single original `OBJECTID`, so
   lineage keys aren't carried through the split anymore.
5. **New**: after building the piece table, reattach lineage/display attributes
   (`OBJECTID`, `UNIQUE_ID`, `DOT_FCLASS`, `FULLNAME`, `VERT_LEVEL`, etc.) via
   `centerline_utils.transfer_attributes_by_midpoint` (existing function, already used for
   exactly this purpose in `00_data_preparation.qmd`) against `CandidateTDMRoadLinks`.
6. Export `link_pieces.gpkg` — same file, same downstream column set, sourced differently.

CC-attachment-node splitting stays as-is (still split points, still produce one-sided
pieces there where appropriate).

### Stage 3 (`03_transfer_attributes.qmd`) — direct chain-aware attribute join, no BFS

This replaces the pseudonode-dissolve-then-piece-graph-BFS mechanism with a much simpler
direct join, because Stage 2 now guarantees every piece boundary is already a snap point:

1. Drop the `nodes_classified.gpkg` input entirely — `is_pseudo` is no longer the relevant
   concept. `nodes_snapped.gpkg` alone (already has `snapped`, `snap_rule`,
   `snapped_x_round/y_round`) is sufficient.
2. Build `snapped_node_ids`: nodes where `snapped==True` **and** `snap_rule !=
   "FixedTransit_Rail"` (rail-snapped nodes attach to GTFS infrastructure, not road
   centerlines — same exclusion `02_create_link.qmd` already applies). Build the reverse
   lookup `coord_to_node: (x_round, y_round) -> N`.
3. Build `pass_through_ids = (all A/B node ids in filtered gdf_links) - snapped_node_ids`.
   This is the generalized replacement for the old `is_pseudo` set — a strict superset.
4. Call the **existing** `dissolve_pseudonodes(gdf_links, pass_through_ids,
   invariant_cols=())` unchanged in logic — only the input set changes name/meaning
   (rename the parameter to `pass_through_ids` for clarity) and `invariant_cols` becomes
   empty, because uniform `FT_2027`/`LN_2027` is no longer guaranteed for this broader
   pass-through set. **Only its `A`, `B`, `n_constituents`, `constituent_ab_pairs` output
   columns are used** — this call's job is purely to answer "does a real-to-real chain
   exist between these two nodes, and which original link hops make it up." Its own
   per-chain attribute columns (picked via "first constituent wins") are **not** used for
   the final output — see step 6. `disagreement_counts` is still surfaced in the summary
   as a QA diagnostic. A chain with `n_constituents==1` is a direct single-hop match;
   multi-hop chains fall out of the same call. **No BFS through piece geometry anywhere.**
5. Load `link_pieces.gpkg`. Vectorized (no per-row Python loop):
   - `A_piece = pieces[['x_start','y_start']].map(coord_to_node)`,  same for `B_piece`.
   - `merge1 = pieces.merge(gdf_dissolved, left_on=['A_piece','B_piece'], right_on=['A','B'])`
     (piece drawn in the same order as the matched chain — "forward").
   - `merge2 = pieces.merge(gdf_dissolved, left_on=['B_piece','A_piece'], right_on=['A','B'])`
     (piece drawn opposite to the matched chain — "reverse"). This is the direction-sensitive
     A→B vs B→A case: the TDM network usually codes a two-way street as two separate
     directed rows, so a single split-line piece can legitimately match **both** a forward
     and a reverse chain, producing two output rows from one piece. Not all roads are
     two-way this way — e.g. freeways are digitized as two independent one-way lines to
     begin with, each already its own `(A,B)`, so a freeway piece typically only matches
     one direction and stays a single row.
   - Pieces matching neither merge keep exactly one row with blank attribute columns and
     `trueshape_method="unattributed"` — geometry always retained.
6. **Attribute resolution for matched pieces — nearest constituent to piece centroid.**
   A chain's `constituent_ab_pairs` may list more than one original hop (e.g. piece
   `A=30,B=40` resolves to a chain walking through unsnapped node 35, with constituents
   `(30,35)` and `(35,40)`). Rather than arbitrarily using whichever hop the graph walk
   visited first, resolve it geometrically:
   - Explode each matched (piece, chain) row's `constituent_ab_pairs` into one row per
     constituent hop.
   - Join each exploded hop back to `gdf_links` on exact `(A,B)` to fetch that original
     link's full attribute row and geometry (safe: confirmed zero duplicate directed pairs).
   - Compute `piece.geometry.centroid.distance(constituent.geometry)` for every candidate
     (vectorized via shapely/geopandas).
   - Group by the matched (piece_id, chain match) and keep the minimum-distance candidate
     (`idxmin`; keep-first on an exact tie, consistent with `transfer_attributes_by_midpoint`'s
     existing tie policy) — that candidate's attributes become the piece's final attributes.
   - This one rule uniformly covers `n_constituents==1` too (only one candidate, trivially
     "nearest") — no separate code path needed for the single-hop case.
   - `pd.concat`(attributed matches + blank unmatched) → final export.
7. Export `links_trueshape.gpkg` with the same name, new schema: one row per piece per
   claiming chain (duplicated only when both directions match), attributes from the
   nearest-constituent resolution above (blank when unmatched), `n_constituents`,
   `constituent_ab_pairs`, `trueshape_method`, `piece_direction` (forward/reverse), plus an
   endpoint-status QA column (`both_snapped`/`a_only`/`b_only`/`neither`).
8. Summary section reports: total pieces, matched vs. unattributed (count and by
   `length_m`, since coverage-by-length is the meaningful inspection metric), how many
   matches required multi-constituent nearest-centroid resolution vs. a trivial single-hop
   match, and `dissolve_pseudonodes`'s `disagreement_counts` (expect this to be non-trivial
   now, since it's no longer guarded by the `is_pseudo` uniformity guarantee — informational
   only, since step 6 no longer relies on that carried value).

### `_src/link_utils.py` cleanup

- Delete `build_piece_graph`, `assemble_chain`, `assemble_chain_relaxed`, `_assemble_loop`
  — no longer called anywhere once Stage 3 is rewritten (verified only `03_transfer_attributes.qmd`
  used them).
- Rename `dissolve_pseudonodes`'s `pseudo_node_ids` parameter to `pass_through_ids` and
  update its docstring to describe the broadened meaning (pass-through = not snapped, a
  superset of topological pseudonodes). No change to the function body/algorithm.
- Update the module docstring's "Stage 3 — Public API" section accordingly.

### `CLAUDE.md` roadmap note

The "Link Routing" bullet under Future Roadmap currently says true-shape conflation "will
be solved deterministically via shortest-path routing." That's no longer accurate once this
lands (it becomes a direct chain/coordinate join, not routing/BFS) — update that sentence
for accuracy.

## Known limitation (flagged, not fixed now)

`dissolve_pseudonodes`'s walk picks one "preferred, else first-found" onward edge at each
pass-through node. That heuristic was designed for degree-2 pseudonodes, where there's only
one real choice. With the broadened pass-through set, an unsnapped node that happens to be a
genuine branch point (3+ connections, just not yet snapped) could have more than one valid
onward path, and the walk will pick one arbitrarily rather than enumerating all of them. This
is an accepted simplification — as snapping coverage grows in later pipeline iterations, the
pass-through set shrinks and this edge case becomes rarer. Not fixing it now; flagging for
awareness.

## Files touched

- `02_create_link.qmd` — rewritten (full-redissolve-then-split, per above)
- `03_transfer_attributes.qmd` — rewritten (direct chain join, per above)
- `_src/link_utils.py` — remove 4 now-dead functions, rename one parameter, docstring updates
- `CLAUDE.md` — one-paragraph accuracy update to the roadmap note
- No changes to `00_data_preparation.qmd`, `01_node_classification.qmd`, `_src/node_utils.py`,
  `_src/centerline_utils.py`, or `_src/arcgis_utils.py`

## Verification

1. Run `02_create_link.qmd` end-to-end (`quarto render 02_create_link.qmd` or execute cells),
   confirm `link_pieces.gpkg` is produced, spot-check that piece count differs from today's
   (expect fewer, longer pieces since arcpy-junction pre-cuts no longer force extra splits),
   and that every piece still has non-null lineage columns after the midpoint join.
2. Run `03_transfer_attributes.qmd` end-to-end, confirm `links_trueshape.gpkg` row count ==
   `link_pieces.gpkg` piece count + (extra rows for dual-direction matches), confirm summary
   shows a plausible matched/unattributed split, and manually inspect a handful of
   `unattributed` rows to confirm they retain real (non-straight-line) geometry.
3. Open `links_trueshape.gpkg` in QGIS (or `.explore()` in the notebook) and visually confirm
   full gapless coverage of the road network, with attributed vs. unattributed pieces
   distinguishable by `trueshape_method`.
