# SHARC: recovering the update budget that exits to the sky (2026-09-16)

Worktree `wsn3g`, branch `revised-9-10`, starting from HEAD `b74a97ede`, tree clean. Statements
are labelled **source** (read from the tree, with `file:line`; line numbers refer to the tree
*after* this change unless marked HEAD) or **inference** (a conclusion from source facts plus
stated assumptions). Every in-game impact is **unmeasured**. The measurement being answered is
the user's, FNV open desert: "Path ends: sky 65.9%", 1.01 segments/path, cache terminates 23.3%,
lookup hit rate 88.2% (`accumulationFrames` 32, `minSampleCount` 2, `capacityLog2` 20,
`updateTileSize` 8, `updateBounces` 4, `gridScale` 50). Portal RTX, enclosed, on the same build:
~95% hit rate, ~78% termination.

Abbreviations: `pass` = `src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slangh`,
`integrator` = `src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`, `direct` =
`src/dxvk/shaders/rtx/algorithm/integrator_direct.slangh`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `host` = `src/dxvk/rtx_render/rtx_sharc.cpp`,
`indirect.cpp` = `src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp`, `design` =
`docs/SHARC-design-revised-9-10.md`, `audit` = `docs/SHARC-path-correctness-audit-2026-09-15.md`,
`diagnosis` = `docs/SHARC-open-world-diagnosis-2026-09-16.md`.

## Verdict

- **The wasted share is the sky share, exactly.** An update path and the query path of the same
  pixel trace the *same* first segment: both stages load `RayOriginDirection` for the pixel
  (`pass:171-177`), the update stage only chooses which pixel (`pass:81-93`). So the 65.9% the
  query panel prints is also the fraction of update paths whose first bounce misses, and with
  vertices inserted only at resolved secondary hits (`hooks:443-449` from `integrator:576`) those
  paths write nothing: `sharcAccumulateRadiance` credits `radiance[count-1]` and with `count` 0
  the unrolled loop matches no slot (`update:16-23`). At 2560x1440 native and tile 8 that is
  ~38k of ~58k paths a frame (inference; the DLSS internal resolution scales both numbers).
- **It is recoverable, without biasing anything.** Two independent mechanisms, both implemented
  behind options that default off and clear the cache when changed:
  1. `rtx.sharc.updatePrimaryVertex` (option 2 in the brief): the update path deposits its
     primary vertex, valued as the direct pass's lighting plus the sampled continuation. This
     recovers **every** sky-bound path (each writes one sample) and turns every camera-visible
     eligible surface into a cell fed by every update tile that lands on it. The sample is the
     same random variable a secondary vertex produces, with RTXDI standing in for the
     secondary NEE. Sections 2 and 3 give the estimator argument and the two places where it is
     not identical (RTXDI reuse and the firefly clamp on the direct term; the camera as the
     only outgoing direction).
  2. `rtx.sharc.updateSkyRetries` (option 1): a first bounce that misses is re-sampled from a
     cosine lobe about the primary normal and traced again, up to N times. **No pdf correction
     is needed and none is applied**, because a cell is a plain average of outgoing-radiance
     samples (`update:64`, `SDK:479`), never an integral over the arrival distribution; the
     arrival pdf enters only the *previous* vertex's estimate, and the retry never credits a
     previous vertex (section 4). It recovers `1 - 0.659^N` of the zero-vertex paths as
     *secondary* samples: 34% at N=1, 57% at N=2, 71% at N=3, 81% at N=4 (inference: the cosine
     lobe's sky probability taken equal to the sampled lobe's).
- **Ranking:** primary deposit > sky retry > spend less > tile selection; folding the
  zero-vertex case into `SharcUpdateMiss` *is* the primary deposit (section 5).
- **Interiors:** neither option is compiled into the query stages or the legacy stages, and
  both are off by default, so Portal RTX at the current config runs byte-identical shaders
  (section 8 lists what was verified). With the primary deposit *on*, interior cells become
  dominated by camera-direction, RTXDI-lit samples; that can only raise readability, but it is
  the one quality risk this change carries (section 3.3), and the reason it is opt-in.

## 1. Where the budget goes today

**Source.** The update dispatch is `ceil(resolution / tile)` wide (`indirect.cpp:788-793`); each invocation maps to one rotating pixel of its tile (`pass:81-93`), reads the
primary's launch state (`pass:171-182`) and runs `integrateIndirectPath` (`pass:201-205`).
Update paths run `min(updateBounces, SHARC_PROPAGATION_DEPTH)` bounces
(`path_state.slangh:309-310`) with roulette forced off (`:544-548`). Vertices are inserted only
inside `sharcOnResolvedVertex` (`hooks:437-449`), which `integratePathVertex` reaches only on a
hit (`integrator:551-561` returns on a miss, after decrementing `bounceIteration`). Sky radiance
on a miss goes through `accumulateRadiance` (`integrator:432`) to `sharcAccumulateRadiance`
(deferred, `update:16-23`) or `SharcUpdateMiss` (`SDK:545-579`); both loop over vertices already
listed.

**Source.** Query and update paths of one pixel share origin, direction and RNG: the direct pass
writes the sampled continuation once (`direct:375-378`), both stages read it (`pass:171-177`), and
`createRNG(pixelCoordinate, frameIdx, offset)` (`integrator:598`) is keyed on the same pixel and
frame in both. The update path for a sampled pixel is therefore a replica of that pixel's query
path, continued past where the query terminates.

**Inference.** So the query panel's "Path ends: sky" is the update stage's first-bounce miss rate
to within the tile sampling. At 65.9%, two thirds of update paths insert nothing. The remaining
third insert one vertex at bounce 1 (if eligible) and whatever later bounces reach before their
own sky exit; the diagnosis's estimate of 0.5-1 vertices per update path outdoors stands.

## 2. Option 2, deposit at the primary vertex: what the cell would hold

**Source, what a cell holds.** For a vertex inserted at bounce `k`, the deferred update stores
`radiance[k] = NEE_k + weights[k] * (E_{k+1} * mis + sky ...)` and `weights[k]` is the product
of every `accumulateThroughput` since insertion (`update:25-32`, `integrator:93-101`, `:1102`);
the flush chains `L_k = radiance[k] + weights[k] * L_{k+1}` from the deepest vertex back
(`update:70-80`) and adds each `L_k` with weight 1 and sample count 1 (`update:77`). Own emission
never enters the own cell (audit section 3). Queries add `throughput * cell` and stop
(`hooks:504-509`).

**Source, what the primary already has.** The direct pass computes the primary's NEE with RTXDI
(or RIS when RTXDI is off), splits it by lobe, firefly-clamps it and writes it to
`PrimaryDirectDiffuseLobeRadianceHitDistance` / `...Specular...` (`direct:488-527`); emission is
explicitly *not* included ("Emission not added in for the primary vertex as it is handled
already by the G-Buffer pass", `direct:418-419`). The same pass stores the sampled continuation's
weight -- BSDF times cosine over lobe and direction pdfs, starting from `1 / integrationSurfacePdf`
(`direct:416`, `:363-366`, `:378`) -- which the indirect stage loads as `throughput` and installs
as `pathState.throughput` (`pass:172, 178`; `integrator:1387`); POM attenuation multiplies it
before the first trace (`integrator:1474`). Both textures are already bound to the update stage
(`indirect.cpp:730-731`, base class slots `:236-237`), and NRC's training path already treats
the primary as vertex 0 using exactly these textures (`integrator:1211-1235`,
`geometry_resolver.slangh:1677-1686`).

**Inference, the estimator.** Insert the primary as slot 0 with `radiance[0] = direct_RTXDI` and
`weights[0] = T_1` before the first segment is traced. The existing chain then does the rest:
segment emission and sky on a miss land in slot 0 weighted by `T_1` (`integrator:432`), bounce
1's emission with its MIS weight likewise (`:572`), bounce 1's NEE goes to slot 1 after it is
inserted (`:877`), and the flush yields
`L_0 = direct_RTXDI + T_1 * (E_1 * mis + NEE_1 + T_2 * ...)`, or `direct_RTXDI + T_1 * L_sky` on
a sky miss. That is a sample of the primary's outgoing radiance minus its own emission, of the
same form as a secondary vertex's `NEE_k + T_k * L_{k+1}`, and the sky-lit ground term
`T_1 * L_sky` is precisely the term the desert needs most. A query that later terminates at
this surface adds `throughput * cell` and had already added the surface's emission itself
(`:572`), so nothing is counted twice.

**Inference, where it differs from a secondary sample.**

1. *Direct-light estimator.* RTXDI (temporal + spatial reuse) instead of RIS / NEE-cache
   sampling (`integrator:735-754`). Lower variance; whatever bias the reuse carries into the
   screen's direct lighting is carried into the cell, so the cache agrees with what the pixel
   itself shows rather than with the secondary estimator. NRC accepts the same trade.
2. *Firefly clamp.* The textures are `sanitizeRadianceHitDistance` outputs
   (`integrator_helpers.slangh:26-38`, `fireflyFiltering` at `cb.fireflyFilteringLuminanceThreshold`);
   secondary NEE is not clamped. Slight dark bias on very bright direct samples, bounded by the
   threshold. Unmeasured.
3. *Outgoing direction.* Every primary sample of a cell looks toward the camera; secondary
   samples look toward whichever bounce ray arrived. The cache is isotropic by design and the
   roughness floor bounds the view dependence for both; but a cell fed mostly by primaries at a
   glossy-yet-eligible surface (low `minRoughness`) averages the *camera's* view of its specular
   direct light. This is the quality risk of the option and the reason it is opt-in. The
   query-side footprint gate (`hooks:464-465`) protects readers from narrow *arrival* lobes; it
   does nothing about writer view dependence.
4. *Roughness source.* `FirstHitPerceptualRoughness` is an r8 texture
   (`integrate_indirect_bindings.slangh:81-82`); the floor comparison can differ from the
   secondary path's by one 8-bit step. Negligible.

**Inference, would the cell be read.** Yes. Queries look up at every eligible secondary vertex
whose segment passes the distance and footprint gates (`hooks:451-468`); in open terrain the
bounce rays that hit anything land largely on camera-visible surfaces (a rock face's rays on the
ground in front of it, the ground's rays on rock faces and, at grazing angles, on distant
relief). Hidden faces (the far side of a rock) are the exception; they stay secondary-fed and
are what the retry option is for. A cell of 16-32 px across (diagnosis section 1) then receives
one sample per update tile that lands in it, `(16..32)^2 / 64` = 4-16 samples a frame at tile 8,
against the 0.1-0.3 the diagnosis estimated for secondary-fed cells. With `minSampleCount` 2
such a cell is readable in its first frame. No feedback loop exists: resampling is compiled out
(`sharc_sdk.slangh:109-111`), so the update never reads cells, and a query never reads its own
pixel's primary (lookups happen at secondary vertices only).

**Design-doc check.** The design says primary surfaces are not *query* eligible ("Primary/
visibility surfaces, including replacement primary surfaces, are not", `design` section Query)
and "Never invent a cache position for the sky". Both hold: nothing reads at the primary, and the
sky still deposits nowhere -- it is credited to the real surface the ray left, which is what the
existing chain does for every deeper vertex already.

## 3. Option 2 as implemented

**Source.** `hooks:295-342`, called from `integrator:1489-1498` once per update path after the
launch state is final (after the POM attenuation, so `pathState.throughput` is the whole
primary-to-bounce-1 weight), whether or not the path will launch. A path that will not launch
deposits the direct light with a zero segment weight (`hooks:326`). That is deliberate: a cell
is a *mean* of its samples, and the direct pass's first-bounce roulette (`direct:220-262`,
active when `pathMinBounces` is 0) zeroes the throughput of terminated paths and divides the
survivors' by `p`; the mean comes out at `direct + T * L` only if the terminated paths are
counted as zero-indirect samples. Skipping them would bias the cell *bright* by `1/p`. The
same holds for invalid samples (`solidAnglePdf` 0), where the pixel itself adds nothing.

*Exclusions before the material test* (`hooks:301-307`): `primarySelectedIntegrationSurface`
false (a translucent primary integrates the surface behind it), `performedAnyPSR()` (the
G-buffer surface is the replaced one and the throughput carries the mirror or glass in front of
it -- inserting it would weight the primary's sample by the mirror's reflectance), `isViewModel`
(moves with the camera).

*Material eligibility* (`hooks:263-286`): the primary is judged by the same predicate as every
secondary hit, `sharcRejectReason` (`hooks:28-66`), fed from the G-buffer words the direct pass
itself decodes (`opaque_surface_material_interaction.slangh:975-1005`): material type from bits
30-31 of `SharedMaterialData0`, opacity from its byte 1 (1 when the thin-film flag is set, as
there), emissive colour and intensity from the packed bytes, subsurface radius from
`SharedSubsurfaceData` (`surface_material_helper.slangh:161`), isotropic roughness via
`calcRoughness(perceptual)` (`brdf.slangh:133-142`, the same function
`opaqueSurfaceMaterialInteractionCreate` uses; anisotropy does not enter the isotropic value,
`brdf.slangh:120-128`). Medium state comes from the G-buffer flags exactly as the path state's
does (`integrator:1386-1388`). The lobe term is passed as eligible with the diffuse floor: the
camera ray is the arrival and has no lobe to classify. The two material words were not bound to
the update stage before; they now are, for update stages only (`integrate_indirect_bindings.slangh:56-64`,
`indirect.cpp:322-323`, `:692-695`).

*Key and value* (`hooks:311-326`): position and geometry normal from
`PrimaryWorldPositionWorldTriangleNormal` through `minimalSurfaceInteractionReadFromGBuffer`
(`gbuffer_helpers.slangh:63-73`, decoding `minimal_surface_interaction.slangh:28-42`), which
holds the `surfaceInteraction.geometryNormal` the G-buffer pass computed for that hit
(`geometry_resolver.slangh:374-379`) -- the same quantity a secondary vertex hands to the hash
(`hooks:442`), in fp32 world units (rgba32f), so the primary lands in the cell a bounce ray
arriving there would key. Direct light is the sum of both lobe textures, as secondary NEE sums
`diffuseLight + specularLight` (`integrator:848, 877`); NaN/inf rejects the deposit.

*Insertion* (`hooks:327-337`): deferred build -- `sharcInsertVertex` then `sharcSeedVertex`
(`update:57-68`) sets `radiance[0] = direct`, `weights[0] = vec3(pathState.throughput)`, only
if the insert succeeded (a full bucket leaves the path exactly as before). Original build --
`SharcUpdateHit(..., direct, 0)` then `SharcSetThroughput(..., throughput)`. Portal space is
the path's (`integrator:74`).

*Propagation depth.* The primary takes slot 0 of `SHARC_PROPAGATION_DEPTH`; the host now picks
the 4-slot variant only when `updateBounces + 1 <= 4` (`indirect.cpp:833-834`, `:1342`). At
`updateBounces` 8 the deepest vertex is not inserted (`update:36-38`) and its lighting is
credited to the previous vertex, which the audit (section 1) showed is correct.

## 4. Option 1, re-sample a first bounce that missed

**Correctness.** The brief asks how the pdf weighting follows if direction sampling is biased
toward geometry. It does not need to, for this cache: a cell accumulates `L_k` with weight 1
and count 1 per arriving update path (`update:77`, `SDK:479`), so a cell's value is the mean of
outgoing-radiance samples at that surface, and each sample's value is computed from that
surface onward (its own NEE and continuation, `integrator:598-1102`) with no dependence on how
the ray got there beyond the view-dependent part of its BSDF, which the isotropic cache ignores
for every sample already. Changing the distribution of arrival directions changes *which* cells
receive samples and how many, not their expected value. The arrival pdf matters only to the
*previous* vertex, whose estimate `NEE + T * L_next` uses `T = f cos / pdf`; the retry never
credits a previous vertex (below). The diagnosis reached the same conclusion for the same idea
(section 8.3) and rejected it on cost, assuming the primary material had to be rebuilt to
re-sample its BSDF; it does not -- a cosine lobe about the primary normal is a valid diffuse
gather and needs only the normal, which the G-buffer has.

**Source, as implemented.** `hooks:344-390`, called after every traced segment
(`integrator:1542-1544`). The trigger is a first-bounce miss: `continuePath` false with
`bounceIteration` 0, which only the miss path produces (`integrator:558`). Skipped when the path
teleported before missing (`portalSpace` changed; the payload no longer describes the primary).
Then: the slot list is flushed and reset (deferred: `sharcFlushVertices` + `count = 0`; original:
`SharcInit`) so that a deposited primary keeps exactly one incoming sample, `direct + T_1 * sky`,
and the retry chain starts empty; a cosine direction is drawn from an RNG stream at offset 240+
(the per-bounce closest-hit offsets stop far below, `path_state.slangh:40-46`, `:502-505`) about
the G-buffer geometry normal, oriented toward the viewer by construction
(`surface_interaction.slangh:360-373`); origin, cone radius and medium state are restored from
the launch values (the miss may have re-traced through cutouts or media and moved them);
`solidAnglePdf` becomes `cos / pi` so an emissive hit's MIS weight (`integrator:568`) sees the
pdf actually used; `skyGatherEligible` is set (a cosine lobe is a diffuse gather, and the cell
at the landing vertex gates on it, `hooks:413-414`); `continuePath` is set and the loop's
increment returns `bounceIteration` to 1, so retries do not consume the bounce budget. The retry
count lives in a loop-local, not in the payload (`integrator:1492`).

**Cost.** One extra ray per retry on the 66% of paths that missed: expected extra segments per
update path `0.659 * (1 + 0.659 + ... )`, about 0.66 at N=1 and 1.2 at N=2, on a pass that traces
a few percent of the query's segments. Unmeasured.

**With both options on.** The primary's sample uses the original bounce (sky or hit) and is
complete before any retry; retry vertices form their own chains. Neither sample is credited
twice.

## 5. Options 3, 4 and 5

- **Option 5, `SharcUpdateMiss`.** Covers every vertex already listed (`SDK:552-578`, the loop
  over `pathLength`); the deferred equivalent covers only the most recent one (`update:16-23`),
  which the audit showed is equivalent because deeper vertices chain at flush. Neither has
  anything to credit when the list is empty, and there is no honest vertex to invent for the sky
  (the design forbids it). Giving the path a listed vertex before its first miss *is* option 2.
- **Option 3, choose tiles by expected occlusion.** Any tile weighting is unbiased for the cache
  by the argument of section 4, but it cannot be computed without tracing (a ground tile next to
  a rock bounces onto it far more often than one in the open, and the normal alone does not say
  which is which), and it would starve exactly the open-ground cells the query reads most. With
  the primary deposit no tile is wasted, so its motivation is gone. The rotating in-tile offset
  (`pass:87-93`) is untouched by this change.
- **Option 4, spend less.** The honest fallback if 1 and 2 had been unsound. They are not. The
  live lever it leaves is `updateTileSize`: with the primary deposit on, camera-visible cells are
  fed 4-16 samples a frame at tile 8, more than Portal's cells needed, so a larger tile may hold
  the hit rate while cutting the update pass -- but it also thins the secondary samples that
  hidden faces depend on. A tile sweep with the primary deposit on is the measurement.

## 6. What the user should measure

FNV, the same open-desert view, the same settings as the quoted reading, stats window open
(`measureGpuTime` + `collectQueryStats`), each reading after the cache age passes ~120 frames.

1. **Baseline** (both options off): note "Path ends: sky", lookup hit rate, "Cache terminates",
   the "Of misses: no cell | below sample floor" split, and "GPU ms: update | resolve | query".
2. **`rtx.sharc.updatePrimaryVertex = True`.** Expected, unmeasured: the no-cell and
   below-floor shares both fall (camera-visible cells are now fed every frame); hit rate rises
   toward the high 90s; "Cache terminates" rises toward its ceiling
   `(1 - 0.659) x hit rate` = 31-33% -- "Path ends: sky" is a query statistic and should not
   move; "GPU ms: update" rises by one hash insert per path (small). Debug view 587 (cell age,
   B = sample count) on open ground should show counts climbing within a frame of the toggle;
   view 583 (cached radiance) should cover the ground. Brightness parity: A/B the terminated
   ground against `integrateIndirectMode = 0`; a visible level difference at ground pixels would
   point at section 2's RTXDI-vs-RIS or firefly-clamp differences.
3. **`rtx.sharc.updateSkyRetries = 2`** (first alone, then with 2). Expected, unmeasured: the
   misses that remain after step 2 -- hidden faces -- shrink further; "GPU ms: update" rises by
   roughly the extra-segment estimate of section 4. If step 2 moved the hit rate and step 3 does
   not, the query's misses were on camera-visible surfaces; if step 3 helps beyond step 2, they
   were on hidden ones. That split is itself a result worth recording.
4. **Portal RTX regression** (interiors are the target content): same chamber, primary deposit
   on. Hit rate should not fall below the ~95% baseline and terminations should hold ~78%;
   "GPU ms: update" is the cost. The quality check is section 3.3: with `allowSpecularPaths` on
   and `minRoughness` low, look at glossy-but-cached panels while moving the camera for a tint
   or brightness that follows the camera. If it appears, the option stays off for Portal and the
   fix is a roughness floor for primary deposits specifically (a one-line change in
   `sharcPrimaryRejectReason`).
5. **Tile sweep** (optional, after 2): `updateTileSize` 8 -> 12 -> 16 with the primary deposit on,
   watching hit rate against "GPU ms: update".

## 7. Options and defaults

| Option | Default | Effect | Cache reset on change |
|---|---|---|---|
| `rtx.sharc.updatePrimaryVertex` | false | update paths deposit their primary vertex | yes (`host:80`) |
| `rtx.sharc.updateSkyRetries` | 0 | re-sample a first-bounce sky miss up to N times, 0..4 | yes (`host:80`) |

Both ride in the upper bits of `sharcArgs.enabled`, a word the host wrote and nothing read
(`sharc_args.h:16-21`, `:41-43`; `host:157-158`). The member keeps its name because SPIR-V
carries member names as debug strings: a rename alone made every indirect blob, legacy ones
included, differ from the baseline by exactly one `OpMemberName` (checked with `spirv-dis`),
which the validator's byte-identity contract rightly refuses. `SharcArgs` stays 80 bytes and
`RaytraceArgs` keeps its layout. UI: two controls under the update settings (`host:320-321`). Every new shader line is
under `SHARC_UPDATE` (`integrator:1489-1497`, `:1542-1544`; `hooks:257-391`;
`integrate_indirect_bindings.slangh:56-64`), so query and legacy stages are unaffected.

## 8. Build, validation and deployment

**Build.** Release, `meson compile -C _Comp64Release`, three synchronous passes after the last
source edit: pass A recompiled the shader phase only and returned 0 without relinking (DLL
mtime unchanged, the behaviour the brief warns about); pass B, after touching the two edited
shader sources, recompiled the 36 update blobs and relinked (DLL 19:24:45); pass C changed
nothing (DLL mtime identical). All three exit 0, no error lines. Logs
`_Comp64Release/sky-budget-build{A,B,C}.log`. `d3d9.dll` 280,173,568 bytes, SHA-256
`d0ec48af92afcbaa0dff8304e156a005976f8d0fd63979e5cdfeb3d95e52f563`; `d3d9.pdb` 139,849,728
bytes, SHA-256 `581d88dc99367cf7dc91b0fd11dddd382ef8afe140a91476b5a891a91aea9446`. The only
shader warnings from the edited files are the pre-existing 15205 class (undefined identifier in
a preprocessor expression), which every SHARC-gated line in the tree already emits.

**Blobs against the HEAD `b74a97ede` build**, byte comparison of every `integrate_indirect_sharc_*`
stage plus the legacy `integrate_indirect_rayquery_neeCache`: the 12 update stages differ, as
they must; all 41 query stages and the legacy stage are identical. An earlier attempt that
renamed `SharcArgs::enabled` made every blob differ by one `OpMemberName`; the name was kept
for that reason (section 7).

**Validator.** `validate_sharc_integration.py --baseline-dll <HEAD dll>`: 62 PASS, exit 0,
including "legacy stages byte-identical to the baseline DLL". The `--baseline-dir` contract
(every SHARC blob identical) was not used: it is the contract for a refactor, and this change
alters the update stages on purpose.

**Options doc.** `RtxOptions.md` regenerated by the headless factory probe (D3D9 factory
created and released with `DXVK_DOCUMENTATION_WRITE_RTX_OPTIONS_MD=1`, fresh `LOCALAPPDATA`,
no device): 2,078 rows; the two new rows, and seven stale `rtx.sharc` rows the committed file
still carried from before the emissive, footprint and sample-floor work.

**Deployment.** With neither game running: `d3d9.dll` + `d3d9.pdb` to
`Fallout New Vegas/.trex`, `d3d9.dll` to `PortalRTX/bin/.trex`, each `cmp`-verified against the
build; previous files kept as `d3d9.dll.backup-pre-sky-budget-20260916-193518` (and the FNV
`.pdb` likewise). The Portal file replaced was byte-identical to the HEAD `b74a97ede` build.
The FNV file replaced was not: it was the 280,171,520-byte DLL from the 05:34 build of this
change (options present, both off, and possibly without the roulette-consistent deposit of
section 3), which was deployed to FNV during the gap between the two halves of this session by
something other than this session. With both options off it behaves as HEAD.

## 9. Why a working cache looks like plain importance sampling

The user's observation: even where SHARC is working, the image is nearly identical to plain
importance sampling, only steadier in motion. The source says that is what a correct cache
must look like here, and the sky share is why the difference is so small outdoors.

**Source.** A query path always traces its first segment; the lookup happens only at a resolved
secondary vertex (`hooks:437-449`, reached from `integrator:576`), never at the primary. A hit
replaces `NEE_k + T_k * L_{k+1} + ...` -- the tail beyond that vertex -- with the cell's mean
and stops (`hooks:504-509`). The cell is a mean over `accumulationFrames` (32 here) of update
samples of the same estimator (section 2), so the expectation of the pixel is unchanged; the
cache changes variance and cost, not the level. What does differ in expectation is
second-order: update paths are truncated at `updateBounces` without roulette (dark tail bias,
`audit` section 7), and the primary deposit swaps RIS for RTXDI at one vertex (section 2).

**Inference, outdoors.** With 65.9% of paths ending at the sky at bounce 1, two thirds of the
pixels' indirect term is `T_1 * L_sky`, one traced ray, computed identically in both modes; the
cache cannot touch it. For the other third the cache replaces a tail that is itself short in
open terrain (1.01 segments per path at 23% termination), so the replaced quantity is small
and the visible effect is only its variance. "Identical but steadier" is the signature of an
unbiased, temporally accumulated cache applied to shallow transport. It also bounds what this
change can buy: even at a 100% hit rate, terminations cannot exceed the 34% of paths that hit
anything, and no first-bounce ray is ever saved. Outdoors the cache's product is the
stability of the tail, not frame time; the primary deposit makes that tail *available* on
open ground, it does not make the ground cheaper to render. Indoors (Portal, 78% termination,
1.12 segments) the same argument gives the 0.1 ms the user measured: the tail was never the
frame's cost.

## 10. What only the game can settle

1. How far the hit rate and the termination share move with the primary deposit on, and how much
   of the remaining miss share the retries recover (section 6, steps 2-3).
2. Whether the RTXDI-lit, firefly-clamped primary samples change the level of terminated ground
   against `integrateIndirectMode = 0` (section 2, differences 1-2).
3. Whether camera-direction samples tint glossy cached surfaces in Portal RTX (section 3.3).
4. The update-pass cost of one extra insert per path and of the retries ("GPU ms: update").
5. Whether a larger `updateTileSize` holds the hit rate once visible cells are over-fed
   (section 5, option 4).
