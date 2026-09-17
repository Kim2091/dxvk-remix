# Three SHARC presets, and Balanced as the shipped default (2026-09-16)

Worktree `wsn3g`, branch `revised-9-10`, starting from HEAD `e6052749f`, tree clean. Statements
are labelled **source** (read from the tree, `file:line` after this change unless marked HEAD),
**inference** (a conclusion from source facts plus stated assumptions) or **judgement** (a choice
of value that the evidence narrows but does not determine). **Every in-game impact below is
unmeasured.** The only measurements quoted are the user's, attributed where used. Neither game was
run for this work.

Abbreviations: `host` = `src/dxvk/rtx_render/rtx_sharc.cpp`, `opt` =
`src/dxvk/rtx_render/rtx_sharc.h`, `indirect.cpp` =
`src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `grid` =
`src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`.

## Verdict

Four options trade along the quality/speed axis. Ten do not, and all three presets write the same
value for those ten. The four that move are `updateTileSize` and `updateBounces` (the update
pass's budget), `updateSkyRetries` (how that budget is spent outdoors) and `capacityLog2` (what
the resolve pass pays for every frame whether or not the slots hold anything).

`capacityLog2` is the question the brief singled out, and the answer has two halves:

- **For occupancy, 22 is still very probably inert**, even with `updatePrimaryVertex` on. The
  primary deposit does not create cells the old estimate had not already counted; it changes which
  of them are *resident every frame*. Section 3 puts the outdoor live set at 1.5-3 x 10^5 against
  2^20 = 1,048,576 slots, a load of 0.15-0.3, and insertion needs sixteen consecutive occupied
  slots to fail. Unmeasured, and there is still no counter for insertion failures.
- **For cost, capacity is not inert at all.** The resolve dispatch is one thread per slot per
  frame (`host:183`), so 22 runs four times the threads of 20 regardless of occupancy. On 13
  September the resolve measured 0.02 ms (user screenshots, `docs/GPU-stage-profiling-2026-09-13.md`,
  capacity of that session not recorded), so the cost is small but real and it is pure overhead if
  the slots are empty.

Balanced keeps 22 because it is the user's tested anchor and there is no reading to overturn it.
Performance drops to 20 and reclaims the resolve threads. Quality keeps 22 as headroom against an
*estimated* load, since a failed insert is a silently dropped cell that nothing counts
(`update:42-45`; `hooks:449` ignores the return in the non-deferred path).

Honest size of the prize: the whole cache measured about 0.1 ms net in Portal RTX
(`docs/SHARC-implementation-status.md:520-522`, user). Performance trims the cost side and the
benefit side together, so it should be read as a few tenths of a millisecond at most, not a
frame-time change anyone will feel. Say so plainly rather than selling it.

## 1. The three presets

Applied at `host:376-427`. Values marked **same** are identical in all three by design.

| option | Quality | **Balanced (default)** | Performance | why |
| :-- | :-- | :-- | :-- | :-- |
| `updateTileSize` | 4 | **8** | 12 | §2. The honest cost lever: one update path per NxN tile, so cost goes as 1/N^2. Quality launches ~4x Balanced's update paths, Performance ~0.44x. |
| `updateBounces` | 8 | **4** | 3 | §2. Depth of the update path. 3 also drops the update shader back to its compact four-slot propagation variant (`indirect.cpp:833-834`). |
| `updateSkyRetries` | 2 | **1** | 0 | §4. One extra ray per retry, only on paths whose first bounce missed. Costs nothing indoors (no sky exits); buys nothing indoors either. |
| `capacityLog2` | 22 | **22** | 20 | Verdict above and §3. A cost knob, not an occupancy one, on present estimates. |
| `accumulationFrames` | 8 | **8** | 8 | same. §5 - trades noise against lag, not against speed. |
| `staleFrames` | 32 | **32** | 32 | same. §5. |
| `gridScale` | 50 | **50** | 50 | same. §6. |
| `minSampleCount` | 2 | **2** | 2 | same. §7. |
| `minRoughness` | 0.05 | **0.05** | 0.05 | same. §7. |
| `minRoughnessSpecular` | 0.7 | **0.7** | 0.7 | same, and inert while `footprintGate` is on (`opt:34`). §7. |
| `maxEmissiveLuminance` | 0.1 | **0.1** | 0.1 | same. §7. |
| `footprintGate` | on | **on** | on | same. §7. |
| `allowSpecularPaths` | on | **on** | on | same. §7. |
| `updatePrimaryVertex` | on | **on** | **off** | **Superseded 2026-09-16 (see the correction below).** It is a lever after all: the user measured it at about 0.1 ms in game, the same order as the whole cache's net benefit, so Performance declines it. Quality and Balanced keep it for the confirmed quality gain. |

Not written by any preset, deliberately: `measureGpuTime`, `collectQueryStats` and
`logFallbackStats` (diagnostics, about a millisecond, and they say nothing about quality);
`allowRayPortals`, `deferredUpdates`, `queryTraceRay`, `queryRayGeneration`, `updateRayGeneration`
(backend and A/B switches, not quality settings).

Four of fourteen move. That is the finding, not a shortfall: most of the SHARC option surface is
correctness and coverage, and the presets should not pretend otherwise.

> **Correction, 2026-09-16, after this document was written.** The user measured
> `rtx.sharc.updatePrimaryVertex` at **about 0.1 ms** in game. That falsifies the cost half of the
> reasoning in this row and in §8: the option *is* a performance lever, and it is the only preset
> value with a frame-time measurement behind it rather than an estimate. Performance now sets it
> **off**; Quality and Balanced keep it **on**, because the quality gain is confirmed and 0.1 ms is
> under one percent of a frame. **Five of fourteen move, not four.** The coverage half of the
> original reasoning still stands, and the consequence is stated in the Performance tooltip: with
> the primary deposit off, under open sky most update paths store nothing, so that preset thins the
> cache to near nothing outdoors while still paying for the update and resolve passes.
> Reasoning, and why no adaptive controller was built to decide this per scene:
> [SHARC-adaptive-2026-09-16.md](SHARC-adaptive-2026-09-16.md) §6 and §8.

## 2. Why the tile size and bounce count are the real levers

**Source.** The update dispatch is `ceil(resolution / tile)` wide (`indirect.cpp:788-793`); each
invocation drives one rotating pixel of its tile and re-runs that pixel's stored first segment.
Update paths run `min(updateBounces, SHARC_PROPAGATION_DEPTH)` bounces
(`path_state.slangh:309-310`) with Russian roulette forced off (`:544-548`). `updateTileSize` is
clamped to [1, 16] and `updateBounces` to [1, 8] (`host:154`, `:102`).

**Inference.** Path count goes as 1/N^2 and segments per path go roughly as the bounce limit, so
the update pass's traced work is proportional to `bounces / tile^2`. Relative to Balanced
(8, 4 = 0.0625): Quality (4, 8) is 0.5, i.e. **8x**; Performance (12, 3) is 0.0208, i.e.
**0.33x**. Those are segment-count ratios, not frame-time ratios - the pass has fixed per-path
setup that does not scale, and the deepest bounces of a path frequently terminate early.

**What that is worth in milliseconds.** The one measurement in the tree is user screenshots from
13 September: SHARC update / resolve / query at 0.31 / 0.02 / 0.82 ms (lean) and 0.83 / 0.02 /
1.62 ms (full), against a ~16-18 ms frame (`docs/GPU-stage-profiling-2026-09-13.md`). The settings
behind those numbers are not recorded and the build predates both sky options, so they cannot be
scaled to today's configuration. They do establish the shape: the update pass is a few tenths of a
millisecond, not a few milliseconds. **Judgement:** Performance should be expected to return a
fraction of a few tenths of a millisecond, partly given back through lost terminations. Quality
costs the same quantity multiplied by roughly eight, which is the largest single number in this
document and the reason Quality is labelled as it is.

**The four-slot cliff is the one discrete effect.** **Source:** with `updatePrimaryVertex` on the
primary occupies propagation slot 0, and the host selects the compact `deferred4` update shader
only when `updateBounces + 1 <= 4` (`indirect.cpp:833-841`); the arrays are
`SHARC_PROPAGATION_DEPTH`-sized per lane (`update:10-12`, depth 8 by default,
`sharc_sdk.slangh:32-33`). **Inference:** Performance's `updateBounces` 3 therefore halves the
per-lane propagation arrays as well as shortening the path, which is a register-pressure saving
on top of the ray saving. Balanced's 4 sits one past that cliff. Unmeasured; it is the one place
where 3 is qualitatively different from 4 rather than just one bounce shallower.

**What Quality's depth buys.** **Source:** update paths are truncated at `updateBounces` with
roulette off, which the path-correctness audit identified as a dark-tail bias
(`docs/SHARC-path-correctness-audit-2026-09-15.md` §7); at depth 8 the deepest vertex is not
inserted and its lighting is credited to the previous one (`update:36-38`), which the same audit
showed is correct. **Inference:** raising the limit to 8 both reduces that dark bias and puts more
vertices per path into more cells, which is a coverage gain that costs rays rather than latency.
Indoors that matters; outdoors it is moot, because update paths end at the sky at the first bounce
(`docs/SHARC-open-world-diagnosis-2026-09-16.md` §6.5).

## 3. Is capacity 22 load-bearing now that the primary vertex is deposited?

The measurement the brief refers to - 20 versus 22 not moving the hit rate - predates
`updatePrimaryVertex`. The question is whether depositing at primary vertices populates enough new
cells to make capacity bind. **Inference, and it is the least certain claim here:**

**Source.** Capacity is `1 << clamp(capacityLog2, 18, 22)` (`host:93`). The table is open-addressed
with a 16-slot linear probe window: the base slot is `hash % (capacity - 15)` (`grid:146-150`) and
an insert fails only when all sixteen consecutive slots from the base are held by *other* keys
(`grid:279-297`). Entries are cleared `staleFrames` frames after their last sample (`SDK:853-868`).

**Inference, how many cells exist.** The open-world diagnosis §2 counted the ground cells a wide
FNV vista touches: a 90-degree wedge from 2 m to 2 km at ~0.8 degrees per voxel is ~53k voxels,
and terrain normals straddle the axis planes so the three sign bits in the key (`grid:197-200`)
split each into up to four cells - order 10^5, plus rock and building faces.

**Inference, what the primary deposit changes.** It does not add cells outside that count: every
primary vertex lies on a camera-visible surface, and camera-visible surfaces are a subset of what
the count already enumerates. What it changes is *residency*. Before, a cell fed once every
`k >= 32` frames was evicted and its slot freed; now every camera-visible eligible cell is fed by
every update tile that lands on it, every frame, so the visible set is fully resident and a
panning camera leaves a 32-frame tail of recently visible cells behind it. Taking the visible set
as most of the 10^5-2x10^5 ground-and-face count and the tail as 1.5-2x on top, the live set is
**1.5-3 x 10^5**: load 0.15-0.3 at 2^20, 0.04-0.07 at 2^22.

**Inference, does that load fail inserts.** Sixteen consecutive occupied slots at load 0.3 is
about 0.3^16 = 4e-9 under uniform hashing, 1.5e-5 at load 0.5, 3e-3 at load 0.7. Clustering makes
the real figure worse than uniform, but not by four orders of magnitude at these loads. **So 22 is
very probably still inert for occupancy, and 20 should hold outdoors.** Interiors are smaller
still (the quality/interior profile doc reached the same conclusion for FNV's 80% case).

**One title where capacity really did move:** Portal RTX, because portal space is part of the cell
key since `7c337eb39`, so through-portal geometry occupies its own cells and the same room is
stored twice (`docs/SHARC-implementation-status.md:405-413`, which records the user seeing "about
the same, possibly a touch worse" performance and names `capacityLog2` as the knob). Portal's
config is at 20 today. That is a reason to leave the *Quality* preset at 22 rather than trim it,
and it is the one scenario in which the preset's capacity choice could be wrong in the generous
direction rather than the mean one.

**What would settle it:** the panel's "Of misses: no cell" share, read at 22 and again at 20 in the
same view (it is under "Include cache reuse statistics"). A share that does not move exonerates
capacity, exactly as the open-world doc's §7 test prescribed; a share that moves means the estimate
above is wrong and Performance's 20 should be reverted to 22. The principled alternative - counting
insertion failures directly - is still unbuilt (open-world doc §8.2).

## 4. Sky retries: an exterior-only line item

**Source.** A first-bounce sky miss is re-sampled from a cosine lobe about the primary normal and
re-traced, up to `updateSkyRetries` times, 0..4 (`hooks:344-390`, `opt:36`). Retries do not consume
the bounce budget.

**Inference** (from `docs/SHARC-sky-budget-2026-09-16.md` §4, whose arithmetic I re-derived): with
a 65.9% first-bounce sky share, measured by the user in FNV open desert, `N` retries recover
`1 - 0.659^N` of the zero-vertex paths as secondary samples - 34% at N=1, 57% at N=2 - at an
expected 0.66 extra segments per update path at N=1 and about 1.2 at N=2. Indoors the sky share is
near zero, so the option costs nothing and buys nothing there.

**Judgement.** Quality takes 2 because the extra coverage lands precisely on the cells that stay
sparse - hidden faces and distant relief - and it is paid for in rays, not in latency. Performance
takes 0 on a specific argument rather than a general "less is faster": outdoors the cache's product
is the stability of a 1.01-segment tail and the frame-time payoff is near zero (sky-budget doc §9),
so declining to spend extra rays there is the cheapest quality the Performance preset can give up.
This is the preset value I would change first if the user disagrees, because the sky options have
**no recorded measurement at all** - the user's live config has run `updateSkyRetries = 1` but no
reading of it exists in any document.

## 5. Accumulation and eviction: identical in all three, on purpose

**Source.** Per entry, if a cell's `accumulatedFrameNum` exceeds `accumulationFrames` (`A`) when a
sample arrives, its history is scaled by `A / accumulatedFrameNum` before the new samples are added
(`SDK:925-930`, `:947-949`); idle cells are untouched until eviction at `staleFrames`
(`SDK:853-868`). A query reads a cell only if its sample count exceeds `minSampleCount`
(`SDK:741`). Host clamps: `A` in [1, 64], stale in [8, 128] (`host:152-153`).

**Inference.** Closed forms (open-world doc §3, re-derived): fed every frame, `N -> s(A+1)`; fed
every `k` frames, `N_inf = 1 + A/k`, readable iff `k < A` at `minSampleCount` 2. So `A` sets two
things - the cell's own sample count, which is its noise floor, and the time constant with which
the cell follows a lighting change. It does not change how much work any pass does.

**Why Quality does not raise it.** This is the brief's trap and it deserves a direct answer. With
`updatePrimaryVertex` on and tile 8, a camera-visible cell receives roughly `(16..32)^2 / 64` =
4-16 samples a frame (sky-budget doc §2), so at `A` = 8 it already holds `s(A+1)` = 36-144 samples.
At Quality's tile 4 that is four times as many: 144-576. **Quality buys its samples with rays, not
with time.** Raising `A` to 32 would add a half-second time constant to the indirect channel to buy
a quantity Quality has already bought at four times the density - and the user has repeatedly said
snappiness matters, which makes lag a quality defect, not a quality feature. 8 stays.

**Why Performance does not lower it.** Lowering `A` saves no work at all: the resolve does the same
arithmetic per slot either way. It would only make cells noisier. 8 stays.

**Why `staleFrames` is identical.** With the primary deposit on, a cell coming back into view is
re-fed within one frame, and at `A` = 8 a cell returning after a 60-frame absence comes back at
`N * 8/68 + 1`, i.e. near-fresh regardless. So the retention window has little left to do, and
eviction costs nothing to perform. 32 in all three.

## 6. Grid density: identical in all three

**Source and inference.** `gridScale` is the SDK's `sceneScale`; a cell's edge is the vertex's
camera distance rounded down to a power of two divided by the value, which makes it an angle -
0.57 to 1.15 degrees at 50 - and therefore unit-free (`docs/SHARC-grid-scale-2026-09-16.md` §3).
It is not a speed lever in either direction: the per-path cost of an update or a query does not
depend on it, and the resolve's cost is set by capacity, not by occupancy. What it moves is a pair
of quality quantities *in opposition* - finer cells resolve lighting better and clear the too-close
gate more easily (the gate wants a segment longer than `1.732 / gridScale` radians, `hooks:455-459`)
but are quadratically more numerous and correspondingly harder to populate. There is no reading
that says which side of 50 either title wants, so moving it per preset would be inventing a
direction. 50 in all three.

## 7. Correctness and coverage controls: identical in all three

Each of these was asked "what would a preset buy by moving it", and each answer was "artefacts, or
nothing".

- **`minRoughness` 0.05.** Raising it *shrinks* eligibility, which means fewer cache hits, fewer
  terminations and therefore a slightly *slower* frame, plus less cached indirect light. It is not
  a performance lever in the direction anyone would expect. Portal RTX measured roughness as the
  remaining 15.8% reject share with the option already at its 0.05 floor
  (`docs/SHARC-implementation-status.md:524-526`, user).
- **`footprintGate` on.** NVIDIA's prescribed test, and the user reported "an extreme improvement"
  from it - far more data in the cache at no quality cost (`docs/SHARC-implementation-status.md:544-546`).
  Turning it off for a preset would trade coverage for nothing.
- **`minSampleCount` 2.** Lowering to 1 re-admits the two-sample glow on freshly revealed geometry
  that this floor was added to remove (same section, user-confirmed); raising to 3 costs coverage.
  Neither changes cost.
- **`maxEmissiveLuminance` 0.1.** A coverage control that took Portal RTX from "almost entirely
  magenta" in the reject view to 0.8% emissive rejects (`docs/SHARC-implementation-status.md:516-518`).
  Lower costs coverage; higher risks emissive surfaces bleeding their own light into the cache.
  No speed either way.
- **`allowSpecularPaths` on.** With it off, Portal RTX's eligible-surface share was 0.2%; on, 73.8%
  (`docs/SHARC-implementation-status.md:497-504`, user, 2.54M paths). Turning it off would not make
  a Performance preset fast, it would make the cache absent.
- **`minRoughnessSpecular` 0.7.** Inert while `footprintGate` is on (`opt:34`, `host:98-99`); kept
  at the anchor's value so that an A/B with the gate off lands somewhere sane.

## 8. What each preset actually buys and costs

All unmeasured.

**Quality (tile 4, bounces 8, capacity 22, retries 2).** Buys: roughly four times the update paths
and twice the sky retries, which lands on the cells the Balanced budget leaves sparse - hidden
faces, surfaces just off screen, distant relief, interior corners no bounce ray reaches - plus a
lower cell noise floor everywhere (144-576 samples per visible cell against 36-144, §5) and less
dark-tail bias from the deeper update paths (§2). Costs: an update pass with roughly eight times
Balanced's traced segments. Against a 13 September reading of 0.31-0.83 ms for that pass on an
older build, this is the preset's whole price and it is a substantial one. If it is too much, tile
6 is the halfway house (1.8x Balanced instead of 4x) and the rest of the preset is unaffected.

**Performance (tile 12, bounces 3, capacity 20, retries 0).** Buys: about 2.25x fewer update paths,
one bounce less depth plus the four-slot propagation variant (§2), no retry rays outdoors, and a
quarter of the resolve threads (§3). Costs: coverage on the sparsest cells first - hidden faces,
freshly revealed geometry, outdoor relief - which shows up as a higher "no cell" and "below sample
floor" miss share and a lower termination rate, i.e. slightly noisier indirect light in the places
that were already marginal. It does not touch any correctness control, so it should not introduce
an artefact class, only thin the cache. **Do not expect frame time to move much:** the whole cache
measured ~0.1 ms net, so this cuts the cost and the benefit together.

**Balanced.** The anchor. Nothing was changed from the user's tested configuration.

## 9. Balanced as the shipped default

Nine `RTX_OPTION` defaults in `opt` were moved so that a user who sets nothing gets the Balanced
column:

| option | was | now | `opt` line |
| :-- | :-- | :-- | :-- |
| `allowSpecularPaths` | false | true | 33 |
| `updatePrimaryVertex` | false | true | 35 |
| `updateSkyRetries` | 0 | 1 | 36 |
| `capacityLog2` | 21 | 22 | 40 |
| `updateTileSize` | 5 | 8 | 41 |
| `updateBounces` | 8 | 4 | 42 |
| `maxEmissiveLuminance` | 0.0 | 0.1 | 46 |
| `minRoughnessSpecular` | 0.5 | 0.7 | 48 |
| `minRoughness` | 0.8 | 0.05 | 49 |

Already correct and untouched: `footprintGate` true (`opt:34`), `accumulationFrames` 8 (`:43`),
`staleFrames` 32 (`:44`), `gridScale` 50 (`:45`), `minSampleCount` 2 (`:47`).

Two consequences worth stating. First, `updatePrimaryVertex` and `updateSkyRetries` were shipped
opt-in three commits ago precisely because they were unvalidated; defaulting them on is a
deliberate promotion on the strength of the user running them, not on a measurement. Second, both
games' `rtx.conf` files set most of these explicitly, so neither install changes behaviour from
this commit - the defaults now simply agree with what FNV's conf already says. Portal RTX's conf
still differs on `accumulationFrames` (4) and `capacityLog2` (20) and keeps those.

## 10. Should the preset be an `rtx.conf` option?

**Recommendation: no. Keep it a UI action.** Reasons, in order of weight:

1. **Every value a preset writes is already a conf key**, so a `rtx.sharc.preset` option would be a
   second way to say the same thing, and the precedence question has no good answer. The option
   system resolves per key through layers; a preset option would have to cross-write nine to
   fourteen other options at load, and a user who wrote both `preset = Performance` and
   `updateTileSize = 4` would get a result neither of them can predict from the file. The one
   historical bug of this shape in SHARC - `minRoughness` silently clamped to 0.5 while the slider
   showed 0.05 (`7dcd68206`, `docs/SHARC-implementation-status.md:427-431`) - cost a whole
   measurement session, and it was a single clamp rather than fourteen cross-writes.
2. **"Set nothing" is now Balanced** (§9), which is the case a conf option would most often be used
   for. Quality and Performance are four lines in a conf, and those four lines say what they do; a
   `preset = 2` line does not, and it would silently change meaning whenever a preset is retuned.
3. **The button is a starting point, not a lock.** It writes through `setDeferred`, so the sliders
   immediately show what was applied and the user can adjust from there and save. An option would
   fight that every frame.

The case that would change this: if presets ever have to be *shipped per title* by something that
does not open the UI - a launcher, a bridge, a per-game config generator. Then the right shape is
an enum option whose `onChange` writes the member options into a layer *below* the user's own conf,
so an explicit key always wins, and the UI combo becomes a writer to that same enum. That is a
larger change to the option system than this task warrants, and nothing today needs it.

## 11. Implementation

- `host:367-372` at HEAD, the single "Performance preset (fewer updates)" button that set
  `updateTileSize` 8, `updateBounces` 4, `capacityLog2` 20 - the current FNV configuration, chosen
  for Portal's cost when the cache was over-served there - is replaced by
  `ImGui::BeginCombo("SHARC preset", "Choose to apply...")` with three `ImGui::Selectable` entries
  (`host:376-427`). This is the panel's existing preset idiom, taken from
  `RtxAtmosphere::showSkyAppearance` (`rtx_atmosphere_ui.cpp:578-675`), including
  `RemixGui::SetTooltipToLastWidgetOnHover` on each entry.
- Each entry applies the ten shared values then the four that differ, both through `setDeferred`,
  and sets `m_resetRequested` so the cache clears - the same mechanism the old button used.
- The diagnostic and backend options are untouched by every path through the combo.

## 12. Three stale texts corrected in the same files

Behaviour-neutral; verified against source before changing.

1. **`opt:28`, `allowRayPortals`.** Said vertices reached through a portal "are never inserted into
   the cache, so portal-only geometry stays uncached" and that the feature was "untested against a
   real portal scene". Both described the insertion gate `f2b7dd366` added, which `7c337eb39`
   replaced with portal space in the cell key - see the comment at `hooks:433-436`, which says in
   as many words that portal-only geometry can now be cached, "which the earlier insertion gate
   could not do". And it has been exercised: `docs/SHARC-implementation-status.md:405-418` records
   the keyed build running in Portal RTX with the user reporting through-portal content looked
   better than the gate build. (The 21-minute, 59,400-frame zero-fallback session at `:376-401` is a
   weaker citation for this claim, because it ran at 18:38-18:59 and `7c337eb39` landed at 19:46 -
   that session tested the gate, not the key.) Text now describes the key, cites the confirmation,
   and keeps the one real consequence: split cells raise occupancy against a fixed capacity.
2. **`host:313`,** the same claim as the panel tooltip. Corrected the same way.
3. **`rtx_debug_view.cpp:497`,** debug view 581's legend said "Magenta - emissive (any non-zero
   emissive radiance)". Since `9c464b31a` the test is
   `calcBt709Luminance(emissiveLight) > cb.sharcArgs.maxEmissiveLuminance` (`hooks:46`), a
   threshold. Legend now names the option.

Not taken: the identical stale wording on the `kSharcRejectEmissive` comment
(`hooks:22`, "any emissive radiance at the vertex"). It is a shader-source comment, so touching it
recompiles every SHARC stage and forces a fresh blob-identity comparison for a line that never
reaches SPIR-V. Worth doing in a change that is already rebuilding shaders.

## 13. Build, validation and deployment

**Build.** Release, `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`, synchronously,
two passes after each of the two rounds of source edits. Pass A relinked (`d3d9.dll` 20:30:50,
280,177,664 bytes, was 280,174,080 at 20:09:01) and pass B changed nothing. A tooltip wording fix
then produced pass C (relinked 20:38:06, same size) and pass D, which again changed nothing. All
four exit 0. The only lines matching "error" in any log are pre-existing `LNK4099` warnings on
`mini_chromium.lib`, which match on `scoped_clear_last_error_win.obj`. Logs
`_Comp64Release/presets-build{A,B,C,D}.log`. This change touches only C++, so no shader was
recompiled; the deployed `d3d9.dll` is the 20:38:06 build, MD5 `17b9dc3495774e82aca581cd39ff2d20`.

**Validator.** `validate_sharc_integration.py --shader-dir _Comp64Release/src/dxvk/rtx_shaders`:
all PASS, exit 0, including "52 declared SHARC stages omit bindings 10/51 and the stealing flag"
and "legacy TraceRay closest hit retains the reservoir binding and sample stealing".

**Options doc.** The 22 `rtx.sharc.*` rows in `RtxOptions.md` were regenerated from `opt` using the
writer's own escaping table (`rtx_option_manager.cpp:421-444`) and default formatting
(`Config::generateOptionString`, `config.cpp:1091-1095`). 10 rows changed; the other 12 came out
byte-identical to the committed file, which is the check that the regeneration matches the writer.
`test_documentation` was not run - it needs a device-free factory probe - but the file it compares
against is now consistent with the header.

**Deployment.** With neither game running (process check first): `d3d9.dll` + `d3d9.pdb` to
`Fallout New Vegas/.trex`, `d3d9.dll` to `PortalRTX/bin/.trex`, each `cmp`-verified against the
build. Backups kept as **`backup-pre-sharc-presets-20260916-203330`** - taken before the first
deployment, so they hold the pre-change HEAD build (280,174,080 bytes), not an intermediate.
The pass C/D build was copied over the pass A/B one without a second backup for that reason.
Not pushed.

## 14. What only the game can settle

All unmeasured; each is an A/B in the same spot with the stats window open
(`measureGpuTime` + `collectQueryStats`), each reading taken after the cache age passes ~120 frames.

1. **Balanced, as a control.** Confirm the panel reads what the user's current conf reads. The
   defaults now equal that conf, so nothing should have moved. Anything that did move is a defect
   in §9, not a preset result.
2. **Capacity, the one open question (§3).** Balanced at 22, then `capacityLog2 = 20` in the same
   view, watching "Of misses: no cell" and "GPU ms: resolve". If the no-cell share does not move,
   capacity is exonerated with the primary deposit on and Performance's 20 is right - and Balanced
   could be trimmed to 21 or 20 too. If it moves, revert Performance to 22 and say so.
3. **Performance.** Read "GPU ms: update" and "GPU ms: resolve" against Balanced in an interior,
   then the hit rate, "Cache terminates" and the miss split. The pass times are the honest
   measurement of what the preset buys; the miss split is what it costs. Then look at a dim
   interior for indirect noise on surfaces the camera has not faced for a second.
4. **Quality.** Read "GPU ms: update" first - if it has not risen sharply, the preset did not
   apply. Then the miss split in the hardest case for the cache: an interior with deep alcoves, or
   an exterior with relief. `DEBUG_VIEW_SHARC_CACHED_RADIANCE` (583) is the view that shows whether
   the extra density arrived.
5. **Sky retries specifically (§4)**, since nothing about them has ever been measured: outdoors, at
   0, 1 and 2, watching the miss split and "GPU ms: update". If 1 to 2 moves nothing, Quality's 2
   should come back to 1.
6. **Portal RTX regression.** All three presets in the same chamber. Portal's conf sets
   `accumulationFrames` 4 and `capacityLog2` 20; a preset overwrites the second of those, which is
   the one place a preset will change an existing Portal session's behaviour.
