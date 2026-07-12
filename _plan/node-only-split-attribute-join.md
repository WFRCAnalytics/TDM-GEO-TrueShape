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

## Implementation Tasks

Ordered so each task is independently runnable/checkable before starting the next.
Stage 2 (T1-T4) must land before Stage 3 (T5-T9) since T5+ reads `link_pieces.gpkg`.

- [x] **T1 — Full network redissolve per `VERT_LEVEL`** (`02_create_link.qmd`)
  Loop over `VERT_LEVEL` values in `CandidateTDMRoadLinks`, call `dissolve_and_singlepart`
  per level across all rows (fwy+surface combined), concat → `gdf_fulldissolve`.
  *Verify:* print row count per level before/after; assert total `length_m` of
  `gdf_fulldissolve` ≈ total length of `CandidateTDMRoadLinks` (dissolve only merges, never
  adds/drops length); confirm no cross-VERT_LEVEL merging occurred.
  **Verified:** 15,566 candidate rows → 6,586 (lvl0) + 526 (lvl1) + 25 (lvl2) = 7,137
  dissolved chains; total length before/after both 6,420,888.7 m (diff 0.000 m). Fixed an
  unrelated pre-existing bug blocking notebook execution: `_src/link_utils.py` used PEP 604
  `X | None` annotations without `from __future__ import annotations`, which the
  `wftdm-docs` kernel's Python couldn't evaluate at import time — added the import
  (mirrors `_src/centerline_utils.py`).

- [x] **T2 — Re-point snap resolution + split onto the redissolved layer** (`02_create_link.qmd`)
  Change `resolve_snap_coords` and `split_candidate_links` to target `gdf_fulldissolve`
  instead of `CandidateTDMRoadLinks`; pass `id_cols=()`.
  *Verify:* split runs without error; print new piece count vs. today's 19,424 (expect
  fewer/longer pieces); spot-check a sample of pieces for valid, non-zero-length geometry.
  **Verified:** 7,137 chains split at 9,548 points (3,600 model-node + 5,948 CC) → 12,252
  pieces (down from 19,424, as expected). `link_pieces.gpkg` round-trip: 0 null/invalid
  geometries; one piece at ~8 micron length (float split artifact, negligible).

- [x] **T3 — Reattach lineage attributes via midpoint join** (`02_create_link.qmd`)
  Call `transfer_attributes_by_midpoint(gdf_pieces, gdf_centerlines, TRANSFER_COLS)` after
  building the piece geometry table.
  *Verify:* null-count check on `OBJECTID`/`UNIQUE_ID`/`DOT_FCLASS`/etc. post-join — should
  be near-zero, consistent with `CandidateTDMRoadLinks`'s own null rates.
  **Verified:** 1,390/12,252 pieces (11.3%) have a null transferred attribute vs.
  1,516/15,566 (9.7%) any-null rate in `CandidateTDMRoadLinks` itself — consistent, driven
  almost entirely by `DOT_FCLASS`'s inherent 9.6% null rate, not a join defect.

- [x] **T4 — Export `link_pieces.gpkg` + update notebook prose** (`02_create_link.qmd`)
  Update title/description/markdown to describe the new methodology; refresh the summary
  stats section.
  *Verify:* file round-trips via `gpd.read_file` with expected columns; full notebook
  renders end-to-end without error.
  **Verified:** frontmatter, Load Data, and Part A/C/D prose rewritten to describe the
  redissolve-then-split methodology (A/B/C/D heading numbering fixed, was duplicated).
  `link_pieces.gpkg` round-trips with 12,252 rows and all `TRANSFER_COLS` + geometry
  columns present; full notebook renders end-to-end without error.

- [x] **T5 — Snapped-node lookup, drop `nodes_classified.gpkg`** (`03_transfer_attributes.qmd`)
  Remove the `nodes_classified.gpkg` load. Build `snapped_node_ids` (`snapped==True` and
  `snap_rule != "FixedTransit_Rail"`) and the reverse `coord_to_node` dict from
  `nodes_snapped.gpkg`.
  *Verify:* print `len(snapped_node_ids)` (~3,600, i.e. 3,670 minus rail-snapped count);
  spot-check 2-3 known `(x,y) → N` lookups by hand against `nodes_snapped.gpkg`.
  **Verified:** 3,670 total snapped − 70 rail-snapped = 3,600 road-snapped nodes exactly as
  predicted; `coord_to_node` also has 3,600 entries (no rounding collisions); 3 random
  round-trip spot-checks all OK. Notebook renders cleanly through this section; next cell
  (old pseudonode dissolve, deferred to T6) fails on `pseudo_node_ids` as expected.

- [x] **T6 — Generalized pass-through chain dissolve** (`_src/link_utils.py`, `03_transfer_attributes.qmd`)
  Rename `dissolve_pseudonodes`'s `pseudo_node_ids` param to `pass_through_ids` (docstring
  update only, no logic change). In the notebook, build
  `pass_through_ids = (A∪B node ids in gdf_links) - snapped_node_ids` and call
  `dissolve_pseudonodes(gdf_links, pass_through_ids, invariant_cols=())`.
  *Verify:* print `len(gdf_dissolved)`, `n_constituents` distribution, and
  `disagreement_counts`; sanity-check counts are plausible against `len(gdf_links)`.
  **Verified:** 28,101 links -> 10,447 dissolved chains (17,654 fewer); 9,569/13,169 node
  ids are pass-through (unsnapped); 72.3% multi-constituent chains, consistent with ~16%
  snap coverage. `FT_2027`/`LN_2027` disagreements now surfaced (303/197 chains) instead of
  hidden by the old `invariant_cols` default — confirms the design rationale.

- [x] **T7 — Vectorized forward/reverse piece-to-chain match** (`03_transfer_attributes.qmd`)
  Compute `A_piece`/`B_piece` via `coord_to_node` map on `link_pieces.gpkg`; build `merge1`
  (forward, `(A_piece,B_piece)==(A,B)`) and `merge2` (reverse, `(B_piece,A_piece)==(A,B)`)
  against `gdf_dissolved`.
  *Verify:* print matched vs. total piece counts; find one known two-way street's `piece_id`
  and confirm it appears in both `merge1` and `merge2` with opposite `A`/`B`.
  **Verified:** 8,653/12,252 pieces matched (7,148 single-direction + 1,505 dual-direction =
  10,158 matched rows). Spot-check on piece_id 7951 confirmed correct forward/reverse dual-row
  behavior for a two-way street. Investigated an anomaly found during verification: a handful
  of pieces (4) had >2 matches (up to 8). Root-caused via standalone diagnostic
  (`gdf_dissolved.duplicated(subset=['A','B'], keep=False)`): 2,085/10,447 dissolved chains
  share an `(A,B)` pair with at least one other chain — 2,041 of those (865 unique node ids)
  are **self-loop** chains (multiple distinct loop/branch roads that both start and end at the
  same real snapped node), and 44 (22 unique pairs) are **genuine parallel paths** between the
  same two snapped nodes (e.g. a direct 1-hop link coexisting with a separate multi-hop
  alternate route between the same `A,B`). All 4 offending pieces are self-loop pieces
  (`A_piece==B_piece`) or the one parallel-path case — confirmed benign, real topology, not a
  join bug. Node-IDs alone can't disambiguate which candidate chain such a piece belongs to;
  this is deferred to T8, which already plans a nearest-geometry resolution step — broadened
  below to resolve across candidate *chains* (not just within one chain's constituents).

- [ ] **T8 — Nearest-centroid resolution across candidate chains and constituents** (`03_transfer_attributes.qmd`)
  T7 confirmed `matched` can legitimately contain more than one candidate chain for a single
  `(piece_id, piece_direction)` (duplicate `(A,B)` chains from self-loops or parallel paths,
  not just multi-hop `constituent_ab_pairs` within one chain) — so resolution must pick the
  best candidate **across all matched rows for a (piece_id, piece_direction), not just within
  one already-chosen chain**. Explode every matched row's `constituent_ab_pairs` into one row
  per constituent hop (carrying its parent chain's match-row identity), join back to
  `gdf_links` on exact `(A,B)` for geometry + attributes, compute distance from the piece's
  centroid to each candidate hop, then group by `(piece_id, piece_direction)` and keep the
  global `idxmin` — this single rule simultaneously selects the correct chain among
  duplicates *and* the correct constituent's attributes within it.
  *Verify:* filter to `n_constituents > 1` matches and manually confirm the chosen
  constituent is geometrically nearest for a handful of cases; confirm `n_constituents==1`
  rows pass through unchanged; specifically re-check the 4 pieces identified in T7
  (piece_id 1443, 336, 11353, 1264) now resolve to exactly one row each per direction.
  **Verified:** 2,760 raw matched rows -> 3,933 exploded constituent-hop candidates ->
  resolved to 2,742 unique `(piece_id, piece_direction)` groups (matches distinct-pair count
  exactly, one row each). 878/2,742 groups were genuinely ambiguous (>1 candidate hop/chain);
  the other 1,864 solo-candidate groups all passed through unchanged (verified via index
  membership). All 4 pieces flagged in T7 (1443, 336, 11353, 1264) now resolve to exactly 2
  rows each (forward + reverse), e.g. piece 1264 (the genuine-parallel-path case) correctly
  picks the direct 1-hop `(20800,20777)`/`(20777,20800)` chain over the 15-hop alternate,
  ~2.5m from centroid vs. far more for the alternate. Of 1,865 single-hop-chain
  `(piece,direction)` groups, only 1 was actually contested against a competing chain — this
  correctly matches the "22 unique parallel-path pairs" finding from T7 (most are theoretical
  network topology that doesn't happen to have a matched piece touching them). Final
  `gdf_matched_resolved`: 2,742 rows, 0 null geometries, 208 columns. Fixed a column-name
  collision surfaced mid-implementation: `gdf_links` (TDM) and `gdf_pieces` (UGRC) both carry
  a differently-sourced `ONEWAY` column, plus the hop geometry vs. piece geometry both named
  `geometry` — resolved by dropping the spent hop geometry before the final join and adding
  explicit `_tdm`/`_ugrc` suffixes.

- [x] **T9 — Assemble final output + export `links_trueshape.gpkg`** (`03_transfer_attributes.qmd`)
  Concat attributed matches + blank-attribute unmatched pieces; add `trueshape_method`,
  `piece_direction`, and endpoint-status QA columns.
  *Verify:* row count == piece count + dual-direction extra rows; zero null geometries;
  grep the codebase to confirm no straight-line-fallback code path remains anywhere.
  **Verified:** full end-to-end render succeeded (22/22 cells). `gdf_trueshape`: 13,327 rows =
  12,252 pieces + 1,075 dual-direction extras exactly, 0 null geometries (assertions passed).
  Method breakdown: 2,742 `matched` + 10,585 `unattributed` (all with real retained geometry —
  no synthetic/straight-line path exists anywhere in the new notebook). Endpoint-status
  cross-check confirms design integrity: piece-level `both_snapped`=2,009 vs. row-level
  `both_snapped`=3,084 in the final output, and 2,009+1,075=3,084 exactly — i.e. every one of
  the 1,075 extra dual-direction rows originates from an originally both-snapped piece, as it
  must. `a_only`/`b_only`/`neither` counts (3,526/3,503/3,214) identical piece-level vs.
  row-level, confirming those pieces (which can never match) are never duplicated. Coverage by
  length: 19.2% (1,315,433 m / 6,864,949 m) — low but expected and not a bug, directly tracking
  today's sparse ~16.4% node-snap coverage; will rise as snapping coverage improves in later
  pipeline iterations. Added an `endpoint_status` QA column (`both_snapped`/`a_only`/`b_only`/
  `neither`) to `gdf_pieces` in Part B, carried through to the final export.

- [x] **T10 — Remove dead BFS/piece-graph code** (`_src/link_utils.py`)
  Delete `build_piece_graph`, `assemble_chain`, `assemble_chain_relaxed`, `_assemble_loop`;
  update the module docstring's Stage 3 Public API section.
  *Verify:* `grep -r` confirms zero remaining references anywhere in the repo;
  `python -c "import _src.link_utils"` succeeds.
  **Verified:** all 4 functions deleted (file trimmed 659 -> 427 lines); module docstring's
  Stage 3 section rewritten to describe the direct coordinate-join match instead of graph
  traversal; removed the now-unused `deque` import (`LineString`/`Point`/`nx` all still used
  elsewhere, kept). `python -c "import _src.link_utils"` succeeds; repo-wide grep for all 4
  function names finds zero remaining references outside this plan file's historical notes
  and the generated `docs/search.json` index.

- [x] **T11 — Update `CLAUDE.md` roadmap note**
  Correct the "Link Routing" bullet — no longer shortest-path routing, now a direct
  chain/coordinate join.
  *Verify:* diff review, single paragraph, no other content touched.
  **Verified:** `git diff CLAUDE.md` shows exactly one line changed (the "Link Routing"
  bullet), rewritten to describe the redissolve-then-split + direct `(A,B)` coordinate-join
  match, explicitly noting no BFS/pathfinding is involved and geometry is never fabricated.

- [x] **T12 — End-to-end run + visual verification**
  Run `02_create_link.qmd` then `03_transfer_attributes.qmd` fully. Open
  `links_trueshape.gpkg` in QGIS or `.explore()`.
  *Verify:* full gapless coverage of the road network; attributed vs. unattributed pieces
  visually distinguishable by `trueshape_method`; matches the plan's Verification section.
  **Verified:** both notebooks rendered fully from a clean state after all T1-T11 edits
  (including T10's dead-code removal), zero errors. Also fixed `03_transfer_attributes.qmd`'s
  stale frontmatter (title/subtitle/description still described the old pseudonode/BFS/piece-
  graph method) to accurately describe the new pass-through-dissolve + coordinate-join
  design. Final `links_trueshape.gpkg`: 13,327 rows, 0 null/empty/invalid/zero-length
  geometries, 100% `LineString` geom_type, 2,742 `matched` (1,315 km) + 10,585 `unattributed`
  (5,550 km) by `trueshape_method`. Static map render (matched=red, unattributed=gray) over
  the full Wasatch Front confirms gapless coverage — red pieces connect seamlessly into gray
  with no visible gaps and no anomalous straight-line/synthetic-geometry artifacts; matched
  coverage concentrates in denser already-snapped areas (SLC downtown grid, Ogden, Provo), as
  expected given today's ~16% node-snap coverage.

**All 12 tasks complete. Plan fully implemented and verified.**
