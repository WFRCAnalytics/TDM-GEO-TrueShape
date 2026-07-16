# Freeway Node-Snap / True-Shape Match Diagnosis

## Context

The user observed that freeway/interchange `qa_status` in `links_trueshape.gpkg`
skewed overwhelmingly toward `endpoints_only` / `a_only` / `b_only` instead of
`matched` — bad specifically for freeways, the project's stated priority.
Two distinct root causes were found. One is fixed here; the other is a
structural limitation that is documented, not fixed, so a future agent
doesn't waste time re-diagnosing or re-attempting the same non-solution.

## Root cause #1 — endpoint-pool / `_TYPE_TIER` drift (FIXED)

`node_utils.py`'s `_TYPE_TIER` table declares which `ep_type`s a `node_type`
may fall back to (tier 1 = "adjacent, permitted"), and `_spatial_snap` scores
candidates against it — but `01_node_classification.qmd`'s Part D1 built each
pass's candidate pool by hand-typing an `ep_type` list, and those lists had
drifted narrower than `_TYPE_TIER` (and the passes' own inline comments)
already declared permitted:

- Pass (a), `fwy` nodes: pool was `{fwy}` only. `_TYPE_TIER` (and the
  existing comment: *"gore/fwy_sf ep = Tier-1"*) permits `{fwy, gore, fwy_sf}`.
- Pass (b), `gore`/`gore_sf` nodes: pool was `{gore, fwy(route≥2), ramp_sf}`.
  `_TYPE_TIER` permits `{gore, fwy, ramp, fwy_sf, ramp_sf}` — `ramp` and
  `fwy_sf` were silently missing despite the pass's own comment claiming
  *"fwy/ramp/fwy_sf ep = Tier-1"*.
- Pass (c), `ramp` nodes: pool was `{ramp, ramp_sf}`. `_TYPE_TIER` permits
  `{ramp, gore, ramp_sf}` — `gore` was missing.

Empirically, this mattered because arcpy only cuts physical centerlines at
real interchanges, not at the FT/lane-count transition points that generate
many freeway model nodes — so the nearest real vertex to a starved node was
disproportionately one class over (median distance to *any* compatible
endpoint ~100–150m, well inside the 500m threshold, vs. a median of 1.3km to
a same-type-only endpoint). The Tier-1 fallback existed on paper and in the
scoring function; it just never received a candidate to score.

**Fix**: added `node_utils.allowed_ep_types(node_type, max_tier=1)`, which
reads directly off `_TYPE_TIER`, and rebuilt each pass's pool in
`01_node_classification.qmd` from it (keeping the handful of deliberate,
separately-justified deviations — e.g. pass (a2) still excludes `surface`
because D2 handles it with better direction logic; pass (b)'s `fwy` is still
restricted to `fw_route_count >= 2` to avoid over-matching isolated
single-route freeway endpoints). This closes the drift permanently: a future
change to `_TYPE_TIER` now propagates automatically instead of requiring
someone to remember to update N hand-typed lists in the notebook.

**Measured effect (node-level snap rate, before → after)**:

| node_type | before | after |
|---|---|---|
| fwy (pure mainline) | 18/70 = 25.7% | 52/70 = **74.3%** |
| gore | 571/666 = 85.7% | 622/666 = **93.4%** |
| gore_sf | 11/14 = 78.6% | 13/14 = 92.9% |
| ramp | 134/204 = 65.7% | 127/204 = 62.3% (regression) |
| fwy_sf | 109/128 = 85.2% | 102/128 = 79.7% (regression) |

The `ramp`/`fwy_sf` regressions are an expected side effect of the
first-call-wins hierarchical pass order (CLAUDE.md §4): pass (a)/(b) now run
with wider pools and can claim an endpoint (e.g. a `gore` or `ramp` vertex)
that a later pass's node previously had exclusive access to. Net effect
across all freeway/interchange node types: 1,103 → 1,176 nodes snapped
(+6.6%), with the worst-performing category (`fwy`) nearly tripling its
success rate — a clear, targeted win for the specific complaint.

## Root cause #2 — model/physical granularity mismatch (NOT fixed, by design)

**This is the more important finding and the reason root cause #1's fix did
not meaningfully move the freeway `matched` piece rate downstream** (242/919
→ 242/923 matched pieces — statistically flat; `endpoints_only` absorbed
almost all of the newly-snapped nodes instead of `matched`).

The WFRC model places a node wherever `FT_2027` or `LN_2027` changes (a
posted-speed boundary, a lane add/drop) — this happens often on a freeway
mainline. UGRC's physical centerlines only carry a digitized vertex at
genuine topological breaks (interchanges/ramps cut by arcpy). There are
**56 freeway-adjacent nodes network-wide** (282 total) that are
`NeighborCount==2` with a real FT or lane-count transition but no
`is_pseudo` status — i.e. "must snap" by the model's own rules, but with no
corresponding real vertex anywhere near their true corridor position.

Root cause #1's fix makes these nodes *snap* successfully now (to the
nearest Tier-1-compatible vertex, typically the neighboring interchange —
which is a real vertex, just not *this* node's true location). But
`03_transfer_attributes.qmd` Part A still treats these nodes as hard chain
boundaries (they fail the `is_pseudo` test, so `dissolve_pseudonodes` does
not walk through them), which means the model network is conceptually split
into two hops around a point that the physical network was never actually
split at. The result: the physical piece spanning that stretch doesn't
align 1:1 with either hop's `(A,B)` — both hop endpoints now have *a* valid
snapped coordinate (`endpoints_only`), but no dissolved chain matches either
one, so the piece still can't resolve to `matched`. This is why the fix
moved failures from `a_only`/`b_only` into `endpoints_only` rather than into
`matched`.

**Why this isn't "fixed" here**: there is no snapping-side fix for a node
whose true physical location genuinely doesn't exist as a vertex in the
source data — moving it to a real but different-location vertex just
relocates the mismatch rather than resolving it, which is exactly what was
just measured. A real fix would have to live in Stage 3's chain-dissolution
logic (e.g., treat FT/LN-transition-only nodes as *physically* pass-through
for the purposes of geometry matching while still preserving their
attribute-boundary role in the reported model attributes) — a materially
different and riskier change than a pool-alignment fix, out of scope for
this pass, and not something to re-attempt via snapping-threshold or
pool-widening tweaks. **Any future agent investigating remaining freeway
`endpoints_only` should start from this section, not from node_utils.py.**
