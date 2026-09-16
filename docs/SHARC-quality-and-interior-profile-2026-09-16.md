# SHARC quality without the cache, and a settings profile for an interior-dominant title (2026-09-16)

Worktree `wsn3g`, branch `revised-9-10`, HEAD `cd61e67fd` plus one uncommitted source change
(§5). Statements are labelled **source** (read from the tree, with `file:line` at HEAD) or
**inference** (a conclusion from source facts plus stated assumptions). Every in-game impact
below is **unmeasured**; the only measurements quoted are the user's earlier panel readings,
attributed where used. The scope correction this doc answers: Fallout New Vegas is 80%+
interiors, so yesterday's open-desert diagnosis (`SHARC-open-world-diagnosis-2026-09-16.md`)
described the minority case.

Abbreviations: `integrator` = `src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`,
`hooks` = `src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `path` =
`src/dxvk/shaders/rtx/algorithm/path_state.slangh`, `pass` =
`src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slang`, `rchit` =
`src/dxvk/shaders/rtx/pass/integrate/integrate_indirect_closesthit.rchit.slang`, `bridge` =
`src/dxvk/shaders/rtx/algorithm/rtxdi/RtxdiApplicationBridge.slangh`, `host` =
`src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp`, `opts` =
`src/dxvk/rtx_render/rtx_options.h`, `gi` = `src/dxvk/rtx_render/rtx_restir_gi_rayquery.h`,
`SDK` = `src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `grid` =
`src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`.

## Verdicts

1. **Question 1 — confirmed, with one refinement.** With zero cache hits the SHARC query stage
   executes the same integrator as `rtx.integrateIndirectMode = 0` at every site that can
   touch a pixel, except one: RTXDI sample stealing at camera-visible secondary vertices,
   which SHARC compiles out and mode 0 performs *if* `rtx.di.enableSampleStealing` is on.
   In the user's configuration it is off: DLSS-RR applies the RayReconstruction path-tracer
   preset, which clears that option (`rtx_options.cpp:313-322`, `385-388`). So on this
   machine SHARC-with-no-hits is mode 0, bit for bit at every image-affecting site (§2).
   Everything the user has seen as "SHARC beats NRC and ReSTIR GI even when the cache does
   nothing" is therefore "plain path tracing beats NRC and ReSTIR GI", and the source lists
   what those two add: ReSTIR GI a spatiotemporally reused, clamped, boiling-filtered
   estimator whose sample coherency NVIDIA's own RR preset exists to break; NRC a neural
   approximation that replaces the path after a spread threshold, including direct light at
   the terminal vertex, with an acknowledged weakness on sharp features (§3). That the
   unbiased estimator wins *through RR* is the one step source cannot prove; §4 says what
   settles it in game.
2. **The compute / RayQuery stealing defect is real, upstream, and affects every non-SHARC
   mode on those backends.** `pass` never defined `RAB_HAS_RTXDI_RESERVOIRS` or
   `RAB_HAS_RESTIR_GI_RESERVOIRS` (upstream `158dc61e4` is identical), so the 32 legacy
   compute and RayQuery-raygen variants load empty reservoirs: an RTXDI steal fails and skips
   the NEE cache / RIS fallback (no shadow ray, no direct light at that vertex), and a ReSTIR
   GI steal in `StealSample` mode adds zero and still terminates the path on rough surfaces.
   Fixed here by adding the two defines under `#if !ENABLE_SHARC` in `pass`, mirroring the
   closest-hit and miss stages; built twice, validated, deployed. The user's RTX 5080 runs
   the TraceRay backend and is unaffected either way (§5).
3. **Question 2 — one profile, tuned for interiors; the outdoor case degrades to mode 0 at
   ~0.1 ms plus the update pass.** The rtx.conf the user already runs (`accumulationFrames`
   4, `minSampleCount` 2, `staleFrames` 32, tile 8, bounces 4, capacity 20, grid 50) *is*
   the interior profile. None of yesterday's outdoor remedies (A 32, stale 128, tile 4, grid
   25) should be applied globally: each costs the interior something and buys the exterior
   nothing visible, because paths that exit to the sky never consult the cache and the
   remaining outdoor paths have no tail to cut (1.01 segments/path). No adaptive mechanism
   is warranted on present evidence; the one that would be principled is described and not
   built (§6).
4. **"Too close" does not bite indoors.** The threshold scales with camera distance
   (`d/58` to `d/29` of the vertex's own distance), so it rejects metre-scale legs at 100 m
   and centimetre-scale legs at 3 m; Portal RTX measured 4.1% of eligible against 19-30%
   outdoors. Another reason the desert numbers mislead (§7).
5. **Cheap outdoor help without an interior cost: essentially none in settings** (§8); the
   honest statement is that outdoors SHARC costs 0.1 ms and does nothing, which is
   acceptable.
6. What to sample indoors is in §9.

## 1. What is being compared

**Source.** `IntegrateIndirectMode`: 0 ImportanceSampled ("reference integration mode"),
1 ReSTIR GI, 2 NRC (the default), 3 SHARC (`opts:193-199`, `534-544`). Backend for the
indirect pass: `rtx.renderPassIntegrateIndirectRaytraceMode` defaults to TraceRay
(`opts:618`) and the automatic preset picks TraceRay on NVIDIA, RADV and AMD proprietary,
RayQuery (compute) on anything else (`rtx_options.cpp:610-631`). Neither game's rtx.conf sets
the mode, the backend, the preset, any `rtx.restirGI.*`, `rtx.neuralRadianceCache.*` or
`rtx.di.*` option (grep of `Fallout New Vegas/rtx.conf` and `PortalRTX/rtx.conf`; both set
only `rtx.sharc.*`, atmosphere, sparse rendering and texture lists). So the legacy modes on
this machine run: raygen from `pass` + closest-hit / miss from `rchit` and the `.rmiss`,
which define both reservoir macros under `#if !ENABLE_SHARC` (`rchit:358-364`,
`integrate_indirect_miss.rmiss.slang:137-141`).

SHARC: `rtx.sharc.queryTraceRay` defaults on (`rtx_sharc.h:30`), so the query runs
`integrate_indirect_sharc_query_trace*` raygen with `integrate_indirect_sharc_query_closesthit*`
(`host:791-793`); all SHARC variants carry `ENABLE_SHARC=1` and therefore no reservoir
macros, and the stealing branch is compiled out by the guard at `integrator:679-680`.

Host differences once SHARC is active: NRC is off for the pass (`nrcEnabled = nrc.isActive()
&& !sharc.isActive()`, `host:763`), ReSTIR GI resources are not bound (`host:750-753`), SHARC
buffers replace NRC's (`host:743-747`), the update dispatch is tile-sized (`host:782-786`),
and the frame runs update → resolve → query (`host:937-955`). The direct pass has no
`ENABLE_SHARC` or `enableReSTIRGI` branch (grep of `integrator_direct.slangh` and
`rtx_pathtracer_integrate_direct.cpp`), so every mode receives identical inputs. Downstream,
the ReSTIR GI passes early-out unless mode 1 (`rtx_restir_gi_rayquery.cpp:174`) and the
demodulate / composite / RR constants branch on `useReSTIRGI()` only
(`demodulate.comp.slang:292-346`, `composite.comp.slang:316-356`, `rtx_composite.cpp:489-498`,
`rtx_ray_reconstruction.cpp:183`), so mode 0 and mode 3 share the whole post-integrator path.

## 2. Question 1: SHARC query with zero hits versus mode 0, every difference

Every `ENABLE_SHARC` / `SHARC_QUERY` site reachable by a query path, and whether it can change
the image. Update-stage-only sites (`SHARC_UPDATE`) are listed once at the end.

| # | site | SHARC query | mode 0 (TraceRay) | image effect |
|---|---|---|---|---|
| 1 | ReSTIR GI hit-geometry store `integrator:340-378` | compiled out | compiled in, inside `cb.enableReSTIRGI` | none: `enableReSTIRGI = restirGI.isActive()` is mode == 1 only (`rtx_context.cpp:1319`, `opts:1482`) |
| 2 | ReSTIR GI sample stealing `integrator:969-1088`, hit-distance bookkeeping `1110-1122`, `storeRestirGIRadiance` `1168-1192`, `initReSTIRGI` `1341-1360` | compiled out | compiled in, all under `cb.enableReSTIRGI` | none |
| 3 | NRC (`ENABLE_NRC`) sites `integrator:441-451`, `585-589`, `893-897`, `initNrc`, `writeToNrcOutputsAfterIntegration` | compiled out | compiled out | none |
| 4 | **RTXDI sample stealing** `integrator:675-731` | `if (false)` → always the fallback: NEE-cache analytical light (`735-741`, `rtx.neeCache.enableAnalyticalLight` default on, `rtx_nee_cache.h:58`) or RIS (`749`) | `if (cb.enableRtxdi && cb.enableRtxdiSampleStealing && rough)`: project the vertex, take the light index, UV and `weightSum` from that pixel's spatial RTXDI reservoir (`lighting.slangh:43-90`), skip the fallback | **the only site that can differ.** In the user's config it does not: RR applies the RayReconstruction path-tracer preset, which sets `rtx.di.enableSampleStealing` false (`rtx_options.cpp:313-322`, invoked from `updateLightingSetting`, `385-388`; RR's own default preset is `RayReconstruction`, `rtx_ray_reconstruction.h:79`). If the option were on, the stolen sample is a reservoir resampled for the *primary* surface at that pixel applied at a different point (approximate, low noise, spatially correlated with the direct signal) while SHARC draws an independent fresh sample; NVIDIA's own text: "No visible IQ gains, but exhibits considerable perf drop (8% in integrate pass)" (`rtx_rtxdi_rayquery.h:57`). Either way the shadow ray (`evalNEESecondary`, `integrator:841`) and the NEE-cache triangle RIS (`767-820`) are shared |
| 5 | `NEE_CACHE_ENABLE` | always 1 in SHARC variants (`pass:44-106`) | variant chosen by `neeCache.isActive()` (`host:778-780`) | none: `rtx.neeCache.enable` defaults on (`rtx_nee_cache.h:50`) and is not overridden; every NEE-cache action is also runtime-gated on `cb.neeCacheArgs.enable` (`integrator:260`, `311`, `445`, `743`, `767`) |
| 6 | `sharcOnResolvedVertex` `integrator:576` → `hooks:262-378` | reject-reason evaluation, one `HashGridFind` + resolved-buffer read (`SDK:733-742`), radiance accumulated **only on a hit** (`hooks:368-373`) | absent | none on a miss; stats writes only with `SHARC_QUERY_STATS`, debug writes only when a SHARC debug view is selected (`hooks:292-296`, `364-367`) |
| 7 | `sharcOnSegmentBegin` `integrator:1501`, `sharcOnResolveLeg` `539-541`, `sharcOnContinuationSampled` `659-661` | payload bookkeeping (`hooks:214-255`) | absent | none. `sharcLaunchPerceptualRoughness` aliases `_roughness` (`path:449-457`); its only other reader, `getSeparateUnorderedApproximationsActive`, runs at the top of the vertex (`integrator:234`) and only for bounce ≤ 1 (`path:552-556`), before the byte is overwritten at that vertex's continuation sample |
| 8 | `skyGatherEligible` | set at `integrator:638-641` and `1400` in both stages; in SHARC the byte is shared with `sharcSegmentDistance` bits 1-7, and getter/setters mask bit 0 correctly both ways (`path:414-447`) | same assignments | none: the atmosphere's `applySkyIndirectRadianceScale` argument (`integrator:404-410`) sees the same value |
| 9 | payload layout | one extra byte in former padding (`path:103-107`) | — | none |
| 10 | roulette and bounce limit | `calculateUseRussianRoulette` differs only for `SHARC_UPDATE` (`path:544-548`); `pathMaxBounces` differs only for `SHARC_UPDATE` (`path:305-313`) | `cb.pathMaxBounces`, `cb.enableRussianRoulette` | none for the query |
| 11 | compute thread group 8×8 vs 16×8 (`pass:449-453`) | — | — | scheduling only, and not the backend in use |
| 12 | output write `integrator:1544-1560` (`#if !SHARC_UPDATE`) | identical | identical | none |
| U | update stage (`SHARC_UPDATE`): NEE-cache insertions compiled out (`integrator:314`, `445`, `863-866`), no screen output (`integrate_indirect.slangh:154-158`), debug views gated `!SHARC_UPDATE`, roulette off, bounces `min(updateBounces, 8)` | runs before the query each frame | absent | none except through cache hits; cost only |

**Inference.** With no hits, row 6 is a read that changes nothing and rows 7-9 are bookkeeping.
So `SHARC(no hits) ≡ mode 0` at every pixel-affecting site, given the RR preset; with
`rtx.di.enableSampleStealing` forced on, the two differ only in *which* light sample the
camera-visible secondary vertices take, both unbiased in expectation up to the steal's
approximation. Note the history: before `9901d7e99` (14 Sep) the SHARC steal *failed* and
those vertices got no NEE at all — darker indirect — and the user still preferred SHARC; the
0.2%-eligibility Portal readings postdate the fix (status doc, 15 Sep). Nothing the cache does
can explain the preference at 0.2% eligibility; the estimator can.

## 3. What ReSTIR GI and NRC add that mode 0 and SHARC-with-no-hits do not

### ReSTIR GI (mode 1)

**Source**, defaults from `gi:91-143` and the shaders:

- Temporal reuse of the first indirect vertex with history `max(500 ms / frame time, 20)`
  frames (`useAdaptiveTemporalHistory`, `temporalAdaptiveHistoryLengthMs` 500, `gi:134-136`,
  `rtx_restir_gi_rayquery.h:69-75`): 30 frames at 60 fps, 20 at 120. Permutation sampling on
  (`gi:118`, "This will improve results in DLSS"); temporal Jacobian and temporal bias
  correction on; discard-enlarged-pixels on; `historyDiscardStrength` 0.
- Spatial reuse with pairwise ray-traced bias correction (`gi:113`) and a final visibility
  ray (`useFinalVisibility`, `restir_gi_final_shading.comp.slang:176`).
- Boiling filter (thresholds 10 / 20, reservoir removal at 62×, `gi:130-133`,
  `final_shading:111-125`): it *rescales* diffuse and specular luminance down to the group
  average when they exceed it — an explicit energy clamp.
- Specular firefly clamp `fireflyThreshold` 50 (`gi:138`), `roughnessClamp` 0.01, virtual
  samples for mirrors (`gi:106-110`), lighting-change validation against RTXDI gradients
  (`gi:140-142`).
- MIS: `Parallax` mode mixes the resampled result with the initial path-traced sample for
  **specular only**; diffuse is 100% the resampled signal (`useDiffuseMIS = false`,
  `final_shading:383-397`). The result "includes radiance often even if the Hit T value is 0
  due to not being selected for integration … a more 'solid' signal" (`final_shading:398-401`),
  which is why the demodulate pass relaxes its consistency checks for it
  (`demodulate.comp.slang:292-300`).
- Sample stealing inside the integrator at rough camera-visible secondary vertices:
  `StealPixel` reads **last frame's composited colour** (`LastComposite`,
  `integrator:1033-1038`) and re-injects `colour × albedo + w × F0` as indirect light, then
  terminates the path (`1061-1064`) — a one-frame recursive feedback of the denoised output
  into the estimator; `StealSample` reads last frame's spatial reservoir (`1041-1058`).
  Default `StealPixel` (`gi:121`), downgraded to `StealSample` unless direct and indirect are
  denoised separately (`rtx_context.cpp:1312-1318`); the RR preset selects `StealSample` with
  3 px jitter (`rtx_restir_gi_rayquery.cpp:273-274`). `stealBoundaryPixelSamplesWhenOutsideOfScreen`
  is documented "at the cost of some bias" (`gi:123`).
- Composite post filter (3× average luminance clamp) runs only with ReSTIR GI
  (`rtx_composite.cpp:489-492`, `rtx_composite.h:122-124`).
- **RR interaction, in NVIDIA's words.** `setToRayReconstructionPreset`
  (`rtx_restir_gi_rayquery.cpp:254-281`): "More aggressive boiling filter to reduce sample
  coherency", "Reduce sample coherency … when stealing samples", "Randomize temporal
  reprojection to reduce coherency" (`useDLSSRRCompatibilityMode`, `gi:119-120`), fixed
  30-frame history. In the temporal shader the compatibility mode replaces the reprojected
  diffuse sample with one from a random 80 px radius (scaled by resolution/960, so ~213 px at
  2560) and inflates its M by 1/0.26 (`restir_gi_temporal_reuse.comp.slang:294-303`). This is
  source evidence that ReSTIR GI's reuse pattern is a known problem for RR. Whether the preset
  is active in the user's build is not knowable from the conf: it is applied only if ReSTIR
  GI is the mode at the moment the path-tracer preset runs (`rtx_options.cpp:325-327`), and
  the default mode is NRC, so switching to ReSTIR GI later in the UI most likely leaves
  `useDLSSRRCompatibilityMode` at its default false. Readable in the ReSTIR GI panel.

**Inference.** Every item above is a departure from the unbiased estimator: temporal history
is lag and correlation, the boiling filter and firefly clamp are energy loss, stealing is
feedback of the previous frame, and diffuse GI is entirely the resampled signal. The user's
"boiling and lag" description matches the mechanisms the source names.

### NRC (mode 2, the default)

**Source.** Paths terminate once the accumulated spread exceeds `terminationHeuristicThreshold`
(0.1 on Ultra, 0.03 High, 0.001 Medium, `rtx_neural_radiance_cache.cpp:389-412`) and the
network's prediction replaces the remainder (`integrator:441-451`, `585-589`, `893-897`;
resolve adds the query result into the indirect buffers, `nrc_resolve.comp.slang:43-70`).
`includeDirectLighting` is on (`rtx_neural_radiance_cache.h:55`), so the cache also stands in
for direct light at the terminal vertex; `boostEmissives` exists because "NRC's irradiance
cache poorly captures sharp emissive features" (`integrator:581-584`,
`rtx_neural_radiance_cache.h:56`). Encoding resolution `smallestResolvableFeatureSizeMeters`
0.01 (Ultra); scene bounds a 200 m box reset on camera cuts (`sceneBoundsWidthMeters`,
`rtx_neural_radiance_cache.h:116-127`); radiance normalisation `maxExpectedAverageRadianceValue`
2.5 ("set the value to an average radiance that you see in your bright scene", `156-160`);
6% unbiased training paths, 2% of primary segments trained on, 4 target training iterations
per frame (`130-134`, `63-72`). Inputs are NaN-sanitised (`integrator:1327`).

**Inference.** NRC's error is approximation error: a small network fitted online, queried at
a hash-grid resolution, standing in for everything past the second-ish vertex including direct
light there. It is smooth and low-noise, which is exactly what a denoiser cannot repair — RR
removes noise, not blur or the wrong mean. Two FNV-specific risks, unmeasured: the 200 m
bounds box against a worldspace of kilometres (the option text says it "must be large enough
to contain any geometry that can be loaded during gameplay"), and the 2.5 normalisation
against a physical-sky desert. Neither is needed to explain the observation; both are worth
one look at the NRC panel (training loss with `enableCalculateTrainingLoss`).

### Why the comparison comes out the way it does

**Inference, and the part source cannot prove.** Mode 0's only disadvantage is variance. RR is
a denoiser: it is the tool that removes variance, and it is fed per-pixel guide buffers and a
stochastic-lobe signal that mode 0 produces exactly as designed (the demodulate consistency
checks assume that shape, `demodulate.comp.slang:292-346`). ReSTIR GI hands RR a signal with
less variance but with correlation, lag and clamping baked in — and NVIDIA's RR preset is a
list of measures to undo the correlation. NRC hands RR a signal with almost no variance and an
approximation error RR cannot see. So with RR in the loop, the unbiased estimator's weakness
is the one thing being fixed downstream, and the alternatives' weaknesses are not. That is the
user's hypothesis, and every source fact above is consistent with it; §4 lists the in-game
tests that would falsify it.

## 4. What settles question 1 in game

All unmeasured; each is a same-interior A/B under RR.

1. `rtx.integrateIndirectMode` **0 vs 3** with the SHARC panel showing a normal hit rate. If
   the two are indistinguishable, the cache contributes nothing visible and SHARC's value is
   the terminations (frame time); if 3 looks different, that difference is the cache's, not
   the estimator's. With `rtx.di.enableSampleStealing` confirmed false in the panel, mode 0 is
   the exact SHARC-no-hit control (§2 row 4).
2. **0 vs 1**, then 1 with `rtx.restirGI.useDLSSRRCompatibilityMode` toggled. If RR
   compatibility mode closes most of the gap, the gap was sample coherency.
3. **0 vs 2**, then 2 with `rtx.neuralRadianceCache.terminationHeuristicThreshold` at 0.001
   (paths run much longer before the cache takes over). If the gap shrinks, it was the
   network's approximation.
4. A still camera for 2 s and then a 90° pan, in each mode: lag and boiling are visible only
   in motion, and the ReSTIR GI history length is 20-30 frames.

## 5. The compute / RayQuery reservoir defect

**Source.** `pass` (60 variants: compute `.comp` and RayQuery-raygen `.rgen`, with and
without NRC, NEE cache and WBOIT, plus the SHARC set) defined neither `RAB_HAS_RTXDI_RESERVOIRS`
nor `RAB_HAS_RESTIR_GI_RESERVOIRS` (`pass:424-430` at HEAD; `git log -S` finds no history;
upstream `158dc61e4` lines 211-217 are the same). The closest-hit and miss stages define both
(`rchit:358-364`). Without the macros `RAB_LoadReservoir` returns `RTXDI_EmptyReservoir()`
(`bridge:583-587`), whose `lightIdx` is invalid (`submodules/rtxdi/rtxdi-sdk/include/rtxdi/Reservoir.slangh:45-53`),
so `RTXDI_IsValidReservoir` is false (`82-85`) and `sampleLightRTXDI` returns false
(`lighting.slangh:51-56`); `isWithinGbuffer` stays true, the fallback at `integrator:735` is
skipped, and unless the NEE-cache triangle RIS rescues it (`767-820`: only for specular-lobe
primaries with roughness < 0.1 under the default `enableModeAfterFirstBounce = SpecularOnly`,
`nee_cache.h:895-902`) `lightSampleValid` is false and `evalNEESecondary` (`841`) never runs:
no shadow ray, no direct light at every camera-visible rough secondary vertex, in modes 0, 1
and 2, whenever `rtx.di.enableSampleStealing` is on (its default). The ReSTIR GI branch is
worse: `RAB_LoadGIReservoir` returns `createEmpty()` (`bridge:590-596`; radiance 0,
`avgWeight` 0, flags 0, `Reservoir.slangh:97-107`), the occluded flag reads false, zero is
accumulated, and rough surfaces still take `continuePath = false` (`integrator:1041-1064`) —
zero indirect and early termination. That branch is reached in `StealSample` mode (enum None
0, StealSample 1, StealPixel 2, `rtx_restir_gi_rayquery.h:41-45`), which the RR preset selects
(`rtx_restir_gi_rayquery.cpp:273`) and which `StealPixel` degrades to unless direct and
indirect are denoised separately (`rtx_context.cpp:1312-1318`); `StealPixel` reads a texture
and is unaffected.

Who is exposed: non-NVIDIA/AMD/RADV hardware by the automatic preset (`rtx_options.cpp:627-631`),
and anyone setting `rtx.renderPassIntegrateIndirectRaytraceMode` to RayQuery or
RayQueryRayGen. Not this machine.

**Fix applied** (`pass:427-437`): the two defines under `#if !ENABLE_SHARC`, with a comment,
mirroring `rchit`. Restoring the intended behaviour rather than disabling the steal is the
right shape because the host binds the reservoir buffers identically for every backend
(`host:700`, `750-753`) and the TraceRay stage has run with them for years. The fork's guard
at `integrator:679` keeps its `ENABLE_SHARC &&` clause: it is now redundant for the legacy
variants (the macro is defined there) but `validate_sharc_integration.py:190-198` pins the
exact text, and SHARC stages still never bind the reservoirs.

Built with `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4` twice,
synchronously (pass 1 rc 0, shaders only, DLL mtime unchanged at 03:44:39; pass 2 rc 0, DLL
relinked 04:35:31, 279 917 056 bytes, was 279 885 312; 0 errors in either log).
`validate_sharc_integration.py --shader-dir … --spirv-dis … --spirv-val …` passes: "52
declared SHARC stages omit bindings 10/51 and the stealing flag" and "legacy TraceRay closest
hit retains the reservoir binding and sample stealing". Byte containment against the pre-fix
DLL: every SHARC blob and every legacy TraceRay blob (raygen, SER raygen, closest hit) is
found unchanged in both DLLs; `integrate_indirect_rayquery_neeCache`,
`_rayquery_raygen_neeCache`, `_rayquery_nrc_neeCache` and `_rayquery` are new. Disassembly:
the legacy compute and RayQuery-raygen blobs now carry `Binding 51` and reference
`RtxdiReservoirBuffer` (and `RestirGIReservoirBuffer` in the non-NRC ones); the SHARC blobs
reference neither. So the deployed DLL differs from the previous one only in variants this
machine does not run. The validator's optional `--baseline-dll` check for
`integrate_indirect_rayquery_neeCache*` will now need a post-fix baseline; that is by design.

Deployed with neither game running: FNV `.trex` dll + pdb and Portal RTX `bin/.trex` dll,
backups `backup-pre-rayquery-reservoirs-20260916-043921`, all three copies `cmp`-verified.
Not pushed; not committed.

**Upstream report, one paragraph.** `integrate_indirect.slang` (the compute and RayQuery
ray-generation variants) never defines `RAB_HAS_RTXDI_RESERVOIRS` /
`RAB_HAS_RESTIR_GI_RESERVOIRS`, unlike `integrate_indirect_closesthit.rchit.slang` and the
miss shader. `RAB_LoadReservoir` / `RAB_LoadGIReservoir` therefore return empty reservoirs in
those backends: with `rtx.di.enableSampleStealing` (default true) a steal at a camera-visible
secondary vertex fails, `isWithinGbuffer` stays true and the NEE-cache / RIS fallback is
skipped, so no NEE and no shadow ray at that vertex; with ReSTIR GI in `StealSample` mode
(the Ray Reconstruction preset) the steal accumulates zero and still terminates the path on
rough surfaces. Affects `renderPassIntegrateIndirectRaytraceMode = RayQuery /
RayQueryRayGen`, which the automatic preset selects on non-NVIDIA/AMD/RADV hardware. Fix:
define both macros in `integrate_indirect.slang` as the closest-hit stage does.

## 6. Question 2: a settings profile for an interior-dominant title

### The arithmetic that matters indoors

**Source.** A cell is readable when `accumulatedSampleNum > minSampleCount` (`SDK:741`,
`sharc_sdk.slangh:42-43`). Each frame with new samples, if the cell's frame count exceeds
`accumulationFrames` (`A`) the history is scaled by `A / accumulatedFrameNum` before the new
samples are added (`SDK:925-929`, `949`); idle cells are untouched until evicted at
`staleFrames` (`853-868`). Host clamps: `A` 1..64, stale 8..128, tile 1..16
(`rtx_sharc.cpp:150-152`).

**Inference** (closed forms from the open-world doc §3, unchanged): fed `s` samples every
frame, `N → s(A+1)` with time constant `A` frames; fed every `k` frames, `N∞ = 1 + A/k`,
readable iff `k < A/(m−1)`, i.e. iff `k < A` at `m = 2`.

Interiors are Portal's regime, not the desert's: every wall illuminates every other, update
paths of 4 bounces put 3-4 vertices into the room, and cells are fed every frame (`k = 1`,
`s ≥ 1`; Portal measured 99.2% hit rate at tile 8, status doc 15 Sep). Then:

- `A` sets responsiveness (`A` frames: 67 ms at 60 fps for `A = 4`, 133 ms for 8) and the
  cache's own per-cell sample count `s(A+1)` — 5-15 samples at `A = 4` for `s` 1-3. That
  count is the cache's noise floor: a cell reads back the same value for every query in the
  frame, so its noise is spatially structured at voxel scale and temporally smoothed over
  `A` frames. It is visible directly in `DEBUG_VIEW_SHARC_CACHED_RADIANCE` (583,
  `debug_view_indices.h:261`) and that view, not the panel, is how to judge whether 4 is too
  low.
- `minSampleCount` 2 is reached in 3 frames at `k = 1` and never matters again.
- `staleFrames` 32 never fires on a cell in view.
- `updateTileSize` 8 over-served Portal (99% hit); FNV interiors are smaller and darker than
  test chambers, so 8 should hold; the panel says if not.

### The outdoor case at those settings

**Inference.** Outdoors cells with `k ≥ 4` are never readable and paths fall through to mode 0
— no bias, same image as mode 0 at those vertices, at the cost of one hash probe. The
measured outdoor cost of the whole cache was ~0.1 ms (user), and the update pass is cheap
there because its paths exit to the sky at the first bounce (66% measured). The cache also
cannot *help* outdoors: sky-exit paths never consult it, and with 1.01 segments/path there is
no tail for a hit to cut. So the degradation is total and free, which is the acceptable kind.

Yesterday's outdoor remedies, and what each would cost the 80%:

| remedy (open-world doc §7) | interior cost |
|---|---|
| `accumulationFrames` 32 | 0.5 s indirect lag on every light change indoors — the opposite of what the user wants |
| `staleFrames` 128 | nothing indoors; keeps outdoor cells that `A = 4` then crushes anyway |
| `updateTileSize` 4 | 4× update paths, and indoors they are the *long* ones (4 bounces, no roulette): roughly 16% of the query's segment count instead of 4% |
| `gridScale` 25 | 16 cm voxels at 4 m; averages across inside corners and under furniture |
| `minSampleCount` 1 | re-admits the two-sample glow on newly revealed geometry, indoors too |

None should be applied globally.

### Recommended profile

The user's current rtx.conf (both games) already is it. Listed with the fork default for
contrast (`rtx_sharc.h:38-47`), and the one conditional change per row:

| option | in use | fork default | keep / change | condition |
|---|---|---|---|---|
| `accumulationFrames` | 4 | 8 | keep 4 | raise to 6 only if the cached-radiance view is splotchy in a dim interior at the user's frame rate |
| `minSampleCount` | 2 | 2 | keep | — |
| `staleFrames` | 32 | 32 | keep | — |
| `updateTileSize` | 8 | 5 | keep 8 | 6 if the indoor hit rate reads below ~90% with "below sample floor" as the dominant miss |
| `updateBounces` | 4 | 8 | keep 4 | 6 only if the indoor "cache terminates" share is well below the hit rate (cells exist but the update vertices stop short of where queries land) |
| `capacityLog2` | 20 | 21 | keep 20 | outdoors load is 0.1-0.3 (open-world doc §2); interiors are far smaller |
| `gridScale` | 50 | 50 | keep | — |
| `minRoughness` / `minRoughnessSpecular` / `footprintGate` / `allowSpecularPaths` / `maxEmissiveLuminance` | 0.05 / 0.7 / on / on / 0.1 | 0.8 / 0.5 / on / off / 0 | keep | these were set on Portal evidence and interiors are where specular arrivals matter |

Plainly: tune for the 80%, let the 20% fall back to mode 0.

### Adaptive accumulation: not now

**Inference.** Nothing measured warrants it: the only regime where `A = 4` is wrong is one
where the cache is idle anyway. If an indoor reading ever shows a large "below sample floor"
share (cells fed less than every 4 frames *inside a room*, which the density argument says
should not happen), the principled mechanism is not a per-frame global rule but a per-cell
one: window the accumulator by **samples** instead of frames — rescale history when
`accumulatedSampleNum` exceeds a cap rather than when `accumulatedFrameNum` exceeds `A`
(the branch at `SDK:925-929`). A cell fed 3 samples a frame then has a time constant of
`cap/3` frames (responsive), a cell fed once every 10 frames has `10·cap` frames (lags where
lag is invisible), no cell is ever crushed below the floor by sparse feeding, and there is no
frame-to-frame state to oscillate because the rule reads only the cell's own counters. It is
a resolve-only change with one option. Not built; not warranted until the indoor split is
read.

## 7. "Too close" indoors

**Source.** The guard rejects when the last resolve leg is shorter than `1.732 × voxelSize`
(`hooks:318-323`); `voxelSize = 2^floor(log2 d) / gridScale` with `d` the camera distance of
the *queried vertex* (`grid:153-168`, `levelBias` 0).

**Inference.** `2^floor(log2 d) ∈ (d/2, d]`, so the threshold lies in `(d/57.7, d/28.9]` at
`gridScale` 50. It is a fraction of the vertex's own camera distance, not a length:

| vertex at | threshold | typical first-bounce leg | rejected? |
|---|---|---|---|
| 3 m (room) | 5-10 cm | 0.5-5 m | only contact geometry: floor at a wall base, clutter on a table |
| 8 m (hall) | 14-28 cm | 1-8 m | rarely |
| 60 m (desert) | 1-2 m | ground → nearby rock or bush, 0.5-2 m | usually |
| 200 m | 3.5-7 m | ground → relief | usually |

That is why 97% of outdoor rejections sit at the first bounce and why Portal measured 4.1% of
eligible (88.5% of it at bounce 1, 0.0% last-leg artefact; status doc 15 Sep) against 19-30%
in the desert. Indoors the guard is doing what it is for — refusing a cell the path may still
be inside — at a rate that does not matter. Nothing to change for interiors, and this is a
second reason the outdoor numbers mislead about the cache's reach.

## 8. Cheap help for the open-sky minority without an interior cost

**Inference.** There is nothing in the settings that changes the sky-exit share or the
segment count, which are what make the cache idle outdoors. The only knobs that touch
outdoor readability at all:

- `minSampleCount` 1: readable after the second sample regardless of `k`; indoor steady state
  is untouched (`N ≫ 2`) but newly revealed geometry reads back two-sample estimates for a
  frame or two, indoors too — the exact glow the floor was added against (commit
  `d94cfd786`). Not recommended; only worth trying if the outdoor "below sample floor" bucket
  is large *and* the user wants outdoor terminations, which buy nothing at 1.01 segments/path.
- Whole-segment distance in the too-close guard (`sharcSegmentDistance` instead of the last
  leg, `hooks:320-323`): indoor-neutral (Portal 0.0% last-leg artefact), possibly meaningful
  outdoors through vegetation cutouts, and already instrumented — counter 19 ("too close by
  the last leg only") on the panel is the number to read before touching it.
- Everything else (tile, `A`, stale, grid) trades interior quality or cost for outdoor
  readability that has no visible payoff.

The honest summary for the exterior is: SHARC costs ~0.1 ms and does nothing, and mode 0
through RR is what the user is looking at there already.

## 9. What to sample indoors

The panel (`rtx.sharc.collectQueryStats` and `measureGpuTime`, both already on in the FNV
conf) in three interior sizes — a shack, a corridor, a casino floor — standing still for two
seconds, then walking, at the profile above. Expected values are inferences.

1. Eligible %, hit rate, cache terminates %, "Path ends: sky" (expect ≈ 0), segments/path
   (expect > 1.5), roulette ends. Hit rate ≥ 90% and terminates within a few points of it
   means the profile is right and nothing in §6 needs to move.
2. The miss split. "Below sample floor" above ~5% indoors would mean cells fed less than
   every 4 frames inside a room — the one result that would justify the sample-windowed
   resolve in §6 — and is the first thing that would move under `updateTileSize` 6.
3. Too close % of eligible and its first-bounce share (expect ≈ Portal's 4%, §7).
4. GPU ms update / resolve / query, against mode 0's frame time in the same spot: SHARC's
   whole cost, and the only number that says whether interiors pay for the terminations.
5. `DEBUG_VIEW_SHARC_CACHED_RADIANCE` (583) at `accumulationFrames` 4 and 8 in the dimmest
   of the three interiors: the splotchiness comparison that decides `A`.
6. The §4 A/Bs, starting with 0 vs 3, plus a read of `rtx.di.enableSampleStealing` and
   `rtx.restirGI.useDLSSRRCompatibilityMode` in their panels so the control is known.

## Files touched

- `src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slang:427-437` — the two reservoir
  defines under `#if !ENABLE_SHARC`, with a comment (§5). Uncommitted.
- This document.
