# SHARC path correctness audit (2026-09-15)

Worktree `wsn3g`, branch `revised-9-10`, HEAD `5667f1667`, tree clean. Source reading only:
nothing built, deployed, pushed or run; no source file modified. Every statement is
labelled **source** (read directly from the tree, with file:line) or **inference** (a
conclusion drawn from several source facts). Where only a running scene could settle a
point, it is marked **unknown** and what would settle it is stated. Numbers quoted from
`docs/SHARC-implementation-status.md` are the user's in-game readings, not mine.

Line numbers refer to HEAD. Abbreviations: `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `integrator` =
`src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `grid` =
`src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `state` =
`src/dxvk/shaders/rtx/algorithm/path_state.slangh`, `resolve` =
`src/dxvk/shaders/rtx/algorithm/resolve.slangh`.

## Verdicts

| # | Question | Verdict |
|---|---|---|
| 1 | Update and query eligibility agree? | **Yes, by construction.** One function, one call, before the update/query split. The only asymmetry is the query-side distance guard, which is not a contamination risk. |
| 2 | `skyGatherEligible` means "arrived by a diffuse lobe"? | **Yes at the hook**, with two conservative edges: opaque diffuse *transmission* at the primary counts as specular (bounce 1 only), and an opacity-transmission lobe resets it to false although the arrival lobe is unchanged. |
| 3 | Faint-emissive caching double-counts? | **No.** Own emission never enters a vertex's own cell on either update path; the query adds emission through the path before reading. The threshold trades coverage for nothing in the transport. The commit message's stated risk does not occur. |
| 4 | Split roughness floor coherent? | **Coherent as a read policy.** The insertion-side split is inert with respect to the glow. It selects by arrival lobe *type*, not lobe width; the design doc asked for the launching roughness, which the hook cannot see. |
| 5 | Portal space in the key | **Correct on insert, query, first vertex and mid-trace; 7-bit level is bounded; no other key construction exists.** Residual: the ray mask is not a pure function of portal space (view-model / player-model bits), so the key is necessary, not sufficient. |
| 6 | `farEnough` with per-leg `segmentHitDistance` | **Still conservative.** Non-teleport legs are collinear, so the last leg is a lower bound on vertex spacing. Post-teleport the two vertices are in different portal spaces except for the COMBINED-to-COMBINED case. It is a quality guard, not a correctness invariant, because cache resampling is off. |
| 7 | Anything else | Nothing in today's changes is wrong. Pre-existing: update paths are truncated without roulette (dark tail bias, magnitude unknown); `minRoughnessSpecular` is not clamped to be at least `minRoughness`. |

## 1. Update and query eligibility

**Source.** `sharcRejectReason` (`hooks:28-62`) is the only eligibility test. It is called
once at `hooks:229-236`, and `sharcEligible` (`hooks:238`) gates the single `if` at
`hooks:259`. The `#if SHARC_UPDATE && SHARC_DEFERRED_UPDATE` / `#elif SHARC_UPDATE` /
`#elif SHARC_QUERY` split is *inside* that `if` (`hooks:265-313`). So the deferred update,
the non-deferred update and the query all see the same predicate on the same inputs.

**Source.** Both query backends run the same code. The TraceRay closest-hit shader calls
`integratePathVertex` on the `PathState` payload
(`integrate_indirect_closesthit.rchit.slang:380-393`); the RayQuery compute and ray-generation
variants call the same function from `RESOLVE_RAY_QUERY` (`resolve_expanded.slangh:165`). The
hook is invoked from `integratePathVertex` at `integrator:576`. The `PathState` *is* the
TraceRay payload, so every field the hook reads (`portalSpace`, `skyGatherEligible`,
`segmentHitDistance`, `mediumMaterialIndex`, `insideMedium`) survives the trace.

**Source.** Inputs to the predicate and where each stage gets them:

| Input | Source | Same in update and query? |
|---|---|---|
| `materialType`, `sharcMaterial`, `emissiveLight` | `integrator:501-504`, `hooks:228` | yes, same code |
| `mediumMaterialIndex`, `insideMedium` | `integrator:1379-1381` (first vertex, from G-buffer), `integrator:646-648` (later) | yes |
| `skyGatherEligible` | `integrator:1393`, `638-641` | yes |
| `portalSpace` | `integrator:74`, `resolve:679` | yes |
| `cb.sharcArgs.*` | one constant buffer | yes |

**Source.** Stage differences that exist but do not touch eligibility:

- Path length: update stages use `min(updateBounces, SHARC_PROPAGATION_DEPTH)` (`state:305-315`);
  query stages use `cb.pathMaxBounces`.
- Russian roulette is forced off in update stages (`state:532-534`).
- NEE-cache task insertion is compiled out of update stages (`integrator:296-298`, `521-523`,
  `863-865`); both stages still *read* the NEE cache.
- RTXDI sample stealing is compiled out of every SHARC stage: the closest-hit shader defines
  `RAB_HAS_RTXDI_RESERVOIRS` only under `#if !ENABLE_SHARC`
  (`integrate_indirect_closesthit.rchit.slang:357-364`), and `integrator:672-673` turns the
  branch into `if (false)`. So the direct-light estimator at a vertex (NEE cache or RIS,
  `integrator:728-747`) is the same in both stages.

**Source.** The one real asymmetry: the query refuses a lookup when the last resolve leg is
shorter than the voxel diagonal (`hooks:277-282`); the update inserts with no distance test
(`hooks:265-271`). A cell can therefore be written from a short-segment update vertex and
read only by paths arriving from farther away.

**Inference.** That asymmetry is not a contamination risk: a short-segment update vertex is
still a valid sample of the cell's outgoing radiance. See §6 for what the guard is for.

**Source.** Two more insertion-only refusals, both harmless: the deferred path refuses when
`state.count >= SHARC_PROPAGATION_DEPTH` (`update:36-38`), unreachable because the update
bounce cap is at most the depth (`state:310`); and `HashGridInsertEntry` failing on a full
bucket (`update:42-45`, `SDK:591-592`) skips the vertex without terminating the path. In the
deferred chain a skipped vertex's NEE and downstream light flow into the previous inserted
vertex's slot through `sharcAccumulateRadiance` (`update:16-23`, which targets
`radiance[count-1]`) with the throughput accumulated since that vertex, which is exactly the
incoming light the previous vertex receives. Correct, and the comment at `hooks:268-270`
says so.

## 2. `skyGatherEligible`

**Source.** Three assignments:

1. `integrator:79` in `pathStateCreateEmpty`: `false`. Overwritten unconditionally at
   `integrator:1393` before the bounce loop starts, so it never reaches the hook.
2. `integrator:1393`: `!geometryFlags.firstSampledLobeIsSpecular`. The flag comes from the
   direct integrator: `firstSampledLobeIsSpecular = true`, then for opaque primaries
   `= firstRayPathSampledLobe != opaqueLobeTypeDiffuseReflection`
   (`integrator_direct.slangh:326-332`), written back to `SharedFlags` at `:398-403`.
3. `integrator:638-641`: after sampling the continuation at vertex k,
   `opaque && (lobe == DiffuseReflection || lobe == DiffuseTransmission)`.

**Source.** The hook reads it at vertex k+1 (`hooks:235-236`); nothing between the sampling at
k and the hook at k+1 writes it. `resolveVertex` and its continuations touch `origin`,
`direction`, `portalSpace`, `rayMask`, `segmentHitDistance`, `continueResolving`,
`decalEncountered` (`resolve:285-306`, `54-123`, `325`, `404`, `677-679`) and nothing else in
the payload. So at the hook it means "the lobe that launched the segment that just arrived",
for every re-trace kind (cutout, clip, inactive portal quad, teleport).

Per case:

- **First indirect vertex.** Meaning: the lobe sampled at the G-buffer integration surface.
  Correct for diffuse reflection and for translucent primaries (specular). **Edge:** an opaque
  diffuse-*transmission* primary lobe reads as specular here (`integrator_direct.slangh:331`)
  but as diffuse at every later vertex (`integrator:640-641`). Conservative (stricter floor,
  or a lobe reject with `allowSpecularPaths` off), not wrong. **Inference**: the direct
  integrator's flag was designed for denoiser lobe routing ("opacity ... should be filtered like
  a reflection", `:330`), not for this use.
- **After a portal crossing.** Unchanged; correct, the launching lobe is what matters.
- **After PSR.** The G-buffer integration surface for a PSR pixel is the replaced surface
  (`geometry_resolver.slangh:3776-3788`), the direct integrator samples the lobe *there*, and
  the indirect path starts there. So "diffuse arrival" at bounce 1 means the PSR surface's
  own lobe was diffuse; the mirror or glass hop before it is invisible to the cache.
  **Inference**: that is the right meaning for a world-space outgoing-radiance cache; the
  camera-side prefix does not change what the vertex needs.
- **After a re-trace.** Unchanged; correct.
- **Transmission.** Opaque diffuse transmission is diffuse (bounce ≥ 2); translucent
  specular transmission is specular (`materialType` guard at `:639` prevents the translucent
  lobe enum, whose value 0 collides with `opaqueLobeTypeDiffuseReflection`,
  `surface_material.h:27-33`, from reading as diffuse). **Edge:** an
  `opaqueLobeTypeOpacityTransmission` sample (`opaque_surface_material_interaction.slangh:1160-1164`)
  is a straight-through continuation, so the true arrival lobe at k+1 is whatever launched the
  segment into k; the code sets `false`. Conservative.

**Verdict.** The signal means what commit `5667f1667` assumes at the point the hook reads
it. The two edges make specific arrivals *stricter* than necessary; neither can loosen the
floor.

**Source.** In query stages the flag shares its byte with the low seven bits of
`sharcSegmentDistance` (`state:415-424`, `432-445`); both setters mask the other's bits, so
the packing is sound.

## 3. Caching faintly emissive surfaces

**Source.** The fork compiles the SDK with `SHARC_SEPARATE_EMISSIVE 1`
(`sharc_sdk.slangh:25-27`) and always passes `SharcHitData` with `emissive = 0`
(`hooks:262`, `(SharcHitData)0`). Under `SHARC_SEPARATE_EMISSIVE`:

- `SharcUpdateHit` adds `directLighting` to the *new* vertex's cell (`SDK:637-647`) and adds
  `sharcHitData.emissive` only to `sharcRadiance`, which is propagated to *prior* vertices
  (`SDK:649-651`, `653-685`). Own emission is never written to the own cell.
- `SharcGetCachedRadianceFromHash` adds `sharcHitData.emissive` to the returned value
  (`SDK:760-762`), i.e. zero here.

The fork's emission bookkeeping in the integrator:

- Vertex k's material emission is accumulated at `integrator:563-573`
  (`emissiveLight * misWeight`), **before** the hook call at `:576`. Segment emission
  (particles, decals, the translucent transparency layer, sky on a miss) is accumulated at
  `:432`, also before the hook.
- `accumulateRadiance` (`integrator:116-125`) routes both into the cache: deferred stages call
  `sharcAccumulateRadiance` (`update:16-23`), which adds `weights[count-1] * radiance` to the
  **most recently inserted** vertex; non-deferred stages call `SharcUpdateMiss`
  (`SDK:545-579`), which adds `radiance * sampleWeights[i]` for every vertex already in the
  list. At the moment of `:572`, vertex k has not been inserted (that happens at `hooks:266`
  or `:271`). So k's emission lands in the cells of vertices before k, weighted by the
  throughput from them to k.
- Vertex k's own cell then receives NEE at k (`:870`, weight 1 since `update:51` /
  `SDK:706`) and, through the chain, everything downstream after `accumulateThroughput` at
  `:1095` scales the weight.

**Inference.** The cell at k therefore holds an estimate of
`NEE_k + T_k * L_in(k+1) + ...` and never `E_k`. A query path terminating at k has already
added `E_k * misWeight` at `:572` and then adds `throughput_k * cell_k` at `hooks:309`. A full
path would add `E_k * misWeight + throughput_k * (NEE_k + T_k * (E_{k+1} * mis + ...))`. Same
decomposition, each term once. **There is no double counting**, on either update path, for
any value of the threshold. The design doc's own rule for this case is satisfied: "pass zero
emission only in the Query hit data; Update still propagates it"
(`docs/SHARC-design-revised-9-10.md:100`), with propagation done through the path's
accumulator rather than `SharcHitData.emissive`, which is equivalent.

**Inference.** The flush order confirms it for the deferred chain: `sharcFlushVertices`
(`update:57-67`) computes `L_i = radiance[i] + weights[i] * L_j` from the last inserted vertex
backwards, where `radiance[i]` already holds `E_j * weights[i]` (recorded when j was hit) and
`L_j` excludes `E_j`. No term repeats.

**Source, on what the threshold actually changed.** Before `9c464b31a` the test was
`any(emissiveLight > 0)`; it is now `calcBt709Luminance(emissiveLight) > maxEmissiveLuminance`
(`hooks:46`). Since own emission is never cached, the old gate excluded surfaces for no
transport reason; the only thing it guaranteed was that `E_k` is exactly zero when the query
terminates at k, which is not needed because the path adds `E_k` itself.

**Verdict.** Caching an emissive surface is sound in this implementation. The threshold
trades coverage against nothing measurable in the estimator; raising it to infinity would not
introduce double counting. The commit message of `9c464b31a` ("a cached cell on an emissive
surface folds some of that surface's own emission into what later paths read back") describes
a failure mode that does not occur in the code as written. What *would* produce it: setting
`sharcHit.emissive = emissiveLight` on the query, or moving the hook above `:572`. The comment
at `hooks:215-216` guards the second; nothing guards the first except this document and the
design doc.

**Unknown.** Whether emission reaches the path by any route other than
`polymorphicSurfaceMaterialInteraction.emissiveRadiance` (`:504`) and the resolve's
`emissiveRadiance` (`:326-334`, `432`). I found none in `integratePathVertex`; a material
class with a different emission path would need to be checked separately if one is added.

## 4. The split roughness floor

**Source.** `roughnessFloor = diffusePath ? minRoughness : minRoughnessSpecular`
(`hooks:56-57`), compared with the **hit material's** `isotropicRoughness` (`hooks:57`).
`diffusePath` is `pathState.skyGatherEligible` (`hooks:236`). The same predicate runs on
insertion and on query. The cell key contains position, level, normal octant and portal space
(`grid:188-208`) and nothing about the lobe that arrived.

**Inference, per cell class**, writing `r` for the hit roughness, `m` for `minRoughness`,
`s` for `minRoughnessSpecular` (default 0.5), with `allowSpecularPaths` on:

| Hit roughness | Written by | Read by |
|---|---|---|
| `r < m` | nobody | nobody |
| `m <= r < s` | diffuse arrivals only | diffuse arrivals only |
| `r >= s` | both | both |

So a specular arrival never reads a cell at a surface below `s`, regardless of who wrote it.
The query-side floor is the whole of the fix for the glow: a narrow arrival lobe at a
directional surface is never served the isotropic average. That holds because the check is
on the *reading* path.

The insertion-side split changes only which update paths contribute to cells in the middle
band. A specular-arrival sample at such a cell is `NEE_k` evaluated toward that arrival's
incoming direction plus downstream; a diffuse-arrival sample at the same cell has the same
form with a different incoming direction. The samples differ in kind not at all. Excluding
the specular ones from insertion neither protects a diffuse reader nor harms it; it reduces
sample count in the middle band. **The floor belongs on query. On insertion it is inert.**

**Source.** The design doc's instruction for glossy support was to use "the lobe and
roughness that *launched* the segment, not the roughness of the hit material"
(`docs/SHARC-design-revised-9-10.md:90`). The hook has the launching lobe's *type* but not its
roughness; `PathState._roughness` holds the G-buffer surface's perceptual roughness
(`integrator:1388`), valid at bounce 1 only. **Inference**: the current split keys on the
type as a proxy for width. A rough-specular arrival (say launching roughness 0.9) is nearly
diffuse but is held to `s`; a mirror-like arrival at a surface of roughness exactly `s` is
accepted. Whether the second case still glows is a visual question. What would settle it: a
smooth wall reflecting a 0.5-roughness floor in Portal RTX with `allowSpecularPaths` on,
compared against `integrateIndirectMode` 0. Carrying the previous vertex's roughness in the
payload (one byte) would let the floor follow the design.

**Source.** `rtx_sharc.cpp:93-94` clamps both floors to `[0.05, 1]` independently. Nothing
enforces `minRoughnessSpecular >= minRoughness`; the option text only asks for it
(`rtx_sharc.h:44`). Setting the specular floor *below* the diffuse one gives specular arrivals
the looser test. Fix: `roughnessSpecular = std::max(roughnessSpecular, roughness)` at
`rtx_sharc.cpp:94`, or refuse the combination in the UI.

## 5. Portal space in the key

**Source, insertion.** Deferred: `hooks:266` passes `pathState.portalSpace` to
`sharcInsertVertex`, which builds `makeSharcParameters(portalSpace)` (`update:39`) and calls
`HashGridInsertEntry` with those parameters (`update:42`). `HashGridComputeSpatialHash` reads
`gridParameters.portalSpace` (`grid:210-214`) and ORs it into bits 61-62 (`grid:205`).
Non-deferred: `hooks:261` builds the parameters from `pathState.portalSpace` and `hooks:271`
passes them to `SharcUpdateHit`, which inserts with `sharcParameters.hashGridParameters`
(`SDK:591`).

**Source, query.** Same `sharcParameters` (`hooks:261`) feed
`HashGridComputeSpatialHashWithVoxelSize` (`hooks:274`) and the lookup (`hooks:283`). The
debug views use `makeSharcParameters(pathState.portalSpace)` too (`hooks:119`, `132`).

**Source, first vertex.** `pathState.portalSpace = geometryFlags.portalSpace`
(`integrator:74`). The G-buffer pass writes it from the primary resolver
(`geometry_resolver.slangh:3142`) and overrides it with the PSR resolver's value when a PSR
surface is the integration surface (`:3776-3788`). The `rayMask` at `integrator:82` comes from
the same flags. If the first segment crosses a portal, `resolve:679` updates the payload before
the hook runs, so the first cached vertex already carries its post-crossing space.

**Source, mid-trace.** `updateStateOnPortalCrossing` (`resolve:54-123`) rewrites
`portalSpace` and `rayMask` in the payload at `resolve:679`; the hook runs only after
`continueResolving` is false (`integrator:545-548`), so it sees the final space for the
vertex. The deferred chain then credits the post-portal vertex's outgoing light to the
pre-portal vertex's cell (a different space, hence a different cell) through
`weights[count-1]`, which is exactly the light that vertex receives through the portal.
Correct.

**Source, level bound.** Level = `0.5 * log2(|camera - p|^2)` clamped to `[1, 127]`
(`grid:153-161`); `sceneScale` enters only the voxel size (`grid:163-168`), not the level.
`SharcGetAdjacentLevelHashKey` clamps to the same range (`SDK:806`, `811`). A level of 127
needs a camera distance of 2^127; a squared distance that overflows to infinity still clamps
to 127. The field is bounded for every finite position. Layout: position bits 0-50, level
51-57, normal 58-60, portal 61-62, bit 63 free (`grid:64-83`); no overlap.

**Source, other key constructions.** A repo-wide search for `HashGridComputeSpatialHash`,
`HashGridInsertEntry`, `HashGridFindEntry`, `HashGridFind(`, `HashGridInsert(` and
`HashGridKey(` outside the SDK finds only `hooks:132`, `hooks:135`, `hooks:274` and
`update:42`, all with portal-aware parameters. Inside the SDK: `SharcGetAdjacentLevelHashKey`
rebuilds the key and now carries the portal bits (`SDK:823-825`); `SharcResolveEntry`
compares whole 64-bit keys (`SDK:910`) and blends only through the adjacent-level key
(`SDK:956`); `HashGridGetPositionFromKey` (`grid:222-237`) is debug-only and ignores normal
and portal bits alike. Nothing else.

**Source.** With no active portal `portalSpace` is the constant `PORTAL_SPACE_NONE` (value 2,
`ray_portal.slangh:35`), so the cell count is unchanged, as `7c337eb39` claims. A key can never
equal `HASH_GRID_INVALID_HASH_KEY` (0): level is at least 1 (bit 51) and main space sets bit 62.

**Residual, inference.** The premise of the key change is that radiance depends on the ray
mask and the mask is a function of portal space. The second half is only approximately true.
The initial dynamic bits come from the primary surface's object mask (`integrator:82`) and
from view-model origin (`:84-87`, `instance_definitions.slangh:33-48`); crossing rules read
the current mask (`resolve:72-73`, `89-92`) and `cb.virtualInstancePortalIndex` (`:68`).
Two paths in the same portal space can therefore carry different `OBJECT_MASK_ALL_DYNAMIC`
bits and write the same cell. This is the view-model / player-model residual the portal
investigation already bounded; the key change narrows it but does not close it.
**Unknown**: whether it is visible. It needs a Portal RTX scene with the player model near
an open portal and an A/B against `integrateIndirectMode` 0.

**Source.** `rtx_sharc.cpp:78, 97` still clears the cache whenever `numActiveRayPortals`
changes. With portal space in the key the clear is no longer needed for correctness.
**Inference**: it costs a cold cache each time a chamber's portal pair becomes active. Not
measured.

## 6. The `farEnough` guard

**Source.** `farEnough = pathState.segmentHitDistance > 1.732051 * voxelSize` (`hooks:277-281`),
where `voxelSize` is the *current* vertex's (`hooks:274`). `segmentHitDistance` is the last
resolve leg: reset at `integrator:312` on every `integratePathVertex` call and at
`resolve:1732` in the RayQuery loop, set to the leg's hit distance at `resolve:404`, set to
`kMissHitDistance` (-1, `ray_helper.slangh:27`) on a miss (`resolve:325`). Legs restart at:
clipped geometry (`resolve:424-428`), cutout / opacity / translucent approximations (the other
`resolveVertexFinalContinue` calls), an inactive portal quad (`resolve:660-666`) and a
teleport (`resolve:689-714`).

**Source.** Every non-teleport continuation keeps the direction and only offsets the origin
past the surface (`resolve:298-305`). **Inference**: those legs are collinear, so the Euclidean
distance from the previous vertex to this one is the sum of the legs and is never less than
the last leg. The test is therefore a lower bound: it can refuse a lookup whose true spacing
is fine, and cannot admit one where the previous vertex sits inside this vertex's voxel. That
is the conservative direction. The status doc says the same; this confirms it.

**Source.** Teleport legs are not collinear, and the exit portal can put this vertex
physically close to the previous one. But the crossing changes `portalSpace`
(`resolve:64-94`): NONE to PORTAL_x, PORTAL_x to COMBINED or back to NONE. Since
`7c337eb39` the two vertices are then in different cells whatever their spacing.
**Edge:** COMBINED stays COMBINED on a further crossing (`resolve:96-103`), so a third
crossing in one path can leave two consecutive vertices in the same space with a short
Euclidean gap the guard cannot see. That also needs the same normal octant and voxel. Rare,
and its consequence is only the one below.

**What the guard is for, inference.** `SHARC_ENABLE_CACHE_RESAMPLING` is 0
(`sharc_sdk.slangh:29-31`), so the update never reads the cache and there is no feedback
loop for the guard to break. Reading a cell the path's previous vertex contributed to is a
*quality* concern (a coarse cell's average served across a corner), which is what the design
doc's "adequate segment footprint" wording (`docs/SHARC-design-revised-9-10.md:88, 90`) and the
SDK's usage guidance address. The guard is sound in the sense that matters: it never lets the
cache be read where the design says not to. Its cost is the refused lookups the status doc
measured (4.1% of eligible surfaces, 88.5% of those at bounce 1), which are legitimate
too-close cases, not leg artefacts (0.0% "last leg only").

## 7. Everything else examined

Nothing in the five commits under audit is wrong beyond the two edges in §2 and the missing
clamp in §4. Items found on the way, all pre-existing:

- **Update paths are truncated without compensation.** Roulette is forced off in update
  stages (`state:532-534`) and the bounce cap is `min(updateBounces, 8)` (`state:310`). A cell
  therefore averages estimates that stop after 8 - b further bounces for the bounce index b at
  which each update path reached it. The reference the query replaces runs with roulette
  (`integrator:923-959`), which is unbiased in expectation. **Inference**: cells are biased
  dark by the missing tail, most for cells reached late in update paths. **Unknown**:
  magnitude; it scales with scene albedo and with the mix of bounce indices at each cell. A
  constant-albedo enclosure with a known analytic answer would measure it. This is the
  "finite update paths" status string (`rtx_sharc.cpp:159`) and the SDK's resampling is the
  feature that would remove it.
- **NEE-cache starvation is unbiased.** Update stages never feed the NEE cache, and once the
  cache serves 78.5% of query paths at bounce 1 few query paths reach deep vertices to feed
  those cells. An empty cell falls back to uniform light selection with probability 1
  (`nee_cache.h:543-549`), and an empty triangle list skips the triangle RIS
  (`integrator:779`) leaving the BSDF-hit emission at full weight (`:565-573`). Higher
  variance in deep cells, no bias. Checked clean.
- **`firstBounceHitDistance` records the last leg, not the segment** (`integrator:384-387`
  runs per leg). Upstream behaviour, unrelated to SHARC.
- **Option clears.** Changing `maxEmissiveLuminance` or `minRoughnessSpecular` resets the
  cache (`rtx_sharc.cpp:97-101`). Not required: a cell at a surface that becomes ineligible is
  simply never read again, and a newly eligible surface starts empty. Harmless.
- **Debug view range check** relies on indices 580..587 being contiguous
  (`hooks:69-72`); they are (`debug_view_indices.h:258-265`).

## What could not be settled from source

1. Whether the residual ray-mask dependence (§5) produces a visible artefact.
2. Whether a mirror-like arrival at a surface of roughness exactly `minRoughnessSpecular`
   still glows (§4); i.e. whether the type-of-lobe proxy is enough or the launching roughness
   is needed.
3. The size of the truncation bias in update paths (§7) for a given scene.
4. The frame cost of the portal-count cache clear (§5) in Portal RTX.

Each needs a running scene and an A/B against `integrateIndirectMode` 0; none changes the
verdicts above, which follow from the code alone.
