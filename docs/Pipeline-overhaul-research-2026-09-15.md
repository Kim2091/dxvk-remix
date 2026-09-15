# Pipeline overhaul research (2026-09-15)

Research only. Worktree `wsn3g`, branch `revised-9-10`, HEAD `b29b2ce0a`, tree clean. No code was changed, nothing was built, deployed or measured for this document. Every timing quoted below is either taken from an earlier record in `docs/` (labelled as such) or is a build-log or git statistic, never a new GPU measurement. "Structural" means cheaper by construction; nothing here is a claimed speedup.

Question: can the rendering pipeline be substantially simplified while holding quality, and specifically should SHARC become the sole indirect path with NRC, ReSTIR GI and the importance-sampled path deleted behind a swappable interface?

## Verdict

1. **The hypothesis is half right.** SHARC is already the only indirect path the user's configuration executes; NRC and ReSTIR GI cost nothing at runtime there beyond a few null bindings and one 2-layer gradient texture. What they cost is build time, shipped bytes, code surface and mode-combination bugs. The importance-sampled path **cannot** be deleted: the SHARC query and update shaders *are* the importance-sampled integrator with `ENABLE_SHARC=1`, and its plain variants are SHARC's own fallback (unsupported device features, raytraced render targets, allocation failure). Deleting "legacy" means deleting SHARC.
2. **Single highest-value simplification is fork-owned, not upstream-owned:** the SHARC backend matrix. Of 56 SHARC blobs, the shipped FNV configuration executes about six (one TraceRay query raygen, one deferred4 raygen update, one closest hit, one miss, each in its WBOIT form). The compute query, RayGen-RayQuery query, non-deferred update, compute update, `fuseAssembly`, and `SHARC_QUERY_STATS` variants are diagnostic backends and rejected experiments. Removing them touches no upstream line, carries zero merge tax, and cuts the fork's own dead-path surface by roughly 40 blobs and ~600 lines of C++ selection code.
3. **Biggest hazard is not a shared producer, it is the sync mechanism.** Upstream is re-planted under the fork as a squash every 5-8 weeks (`c5740aae5`: "Re-plant the fork's work as a single squash on top of NVIDIA dxvk-remix main"). A deletion is a permanent diff that re-conflicts on every sync. Second hazard: the graphics presets (`rtx_options.cpp:505-516`) write `integrateIndirectMode = NRC` or `ReSTIRGI` and never SHARC, so any preset change silently switches the user off SHARC today.
4. **Merge-tax verdict:** hard-deleting NRC costs about 6-7 conflicting hunks per month in files the fork must keep (plus about 5 free modify/delete conflicts in files it would remove); ReSTIR GI about 4 per month. For NRC that tax buys a real reduction (35% of shader blobs, 45% of blob bytes, 115 MB of shipped DLLs, half the G-buffer variant matrix, the sparse-rendering subsystem). For ReSTIR GI it buys almost nothing (four compute blobs, 1.9k lines, already `isActive()`-gated). Recommendation: **compile NRC out at build time rather than deleting it** (near-zero tax, most of the build and shipping gain, none of the code-surface gain), **keep ReSTIR GI as the fallback**, and **delete only fork-owned complexity**. Hard deletion is worth it only if the fork stops syncing upstream, and this document says so plainly rather than pretending an abstraction removes the cost.

## 1. The pipeline as it actually runs

Source: `RtxContext::injectRTX` (`rtx_context.cpp:496-900`), `dispatchPathTracing`/`dispatchIntegrate` (`:1773-1794`), `dispatchDenoise` (`:1807-1957`), `DxvkPathtracerIntegrateIndirect::dispatchLighting`/`dispatch`/`dispatchNEE` (`rtx_pathtracer_integrate_indirect.cpp:697-1120`).

User configuration, from the docs and the FNV snapshot `_Comp64Release/fused-assembly-first-capture-rtx.conf` (mode itself is not in that snapshot; the status log records FNV on full-feature SHARC, `leanSecondary=False`, `fuseAssembly=False`): `integrateIndirectMode=3` (SHARC), TraceRay query backend, deferred ray-generation updates with `updateBounces=4`, `updateTileSize=8`, `capacityLog2=20`, `allowWboit`/`allowOpacityMicromap`/`allowSpecularPaths` on, `skyMode=1` (Numos atmosphere), `opacityMicromap.enable=True`, `raytracedRenderTarget.enable=True`, sparse rendering off, `sceneScale=0.1`. DLSS-RR is assumed on (the 2026-09-13 profile shows an `UpscalingOrRayReconstruction` interval and a 0.117 ms `Denoising` interval, which is consistent with RR plus secondary-only NRD; see below). `wboitEnabled` defaults true, `useRTXDI` true, NEE cache on, `enablePSRR`/`enablePSTR` true.

| Order | Pass | Owner | Dispatched in the user's configuration? | Gate in source |
|---|---|---|---|---|
| 1 | Scene preparation, precipitation, TLAS/BLAS | `SceneManager` | yes | always |
| 2 | Raytrace args, atmosphere LUTs, cloud grids | `updateRaytraceArgsConstantBuffer`, `fork_hooks::updateAtmosphereConstants` | yes (3.8 ms median in the 2026-09-13 profile) | `skyMode` |
| 3 | Volumetrics | `RtxGlobalVolumetrics` | yes | volumetrics enable |
| 4 | G-buffer (primary + PSR) | `DxvkPathtracerGbuffer` | yes | always |
| 5 | Sparse rendering (mask, rates, compaction) | `SparseRendering` | **no** | requires RR **and** NRC active (`rtx_sparse_rendering.cpp:110-131`) |
| 6 | RTXDI initial/temporal/spatial | `DxvkRtxdiRayQuery::dispatch` | yes | `useRTXDI` |
| 7 | NEE cache update | `NeeCachePass` | yes | `neeCache.enable && enableUpdate` |
| 8 | Direct integration | `DxvkPathtracerIntegrateDirect` | yes | always |
| 9 | RTXDI gradients + filter | `dispatchGradient` | **no** | needs NRD as primary denoiser or ReSTIR GI lighting validation (`rtx_rtxdi_rayquery.cpp:364-389`); both false under RR + SHARC |
| 10 | SHARC update (sparse), resolve, query | `dispatchLighting` | yes | `metaSharc().isActive()` |
| 10' | Legacy / ReSTIR GI / NRC indirect trace | `dispatch()` switch | **no** (fallback only) | `!sharc.isActive()` |
| 11 | NEE assembly (`integrate_nee_plain`) | `dispatchNEE` | yes | always; plain variant when neither NRC nor ReSTIR GI is active |
| 12 | Cloud screen pass | `RtxAtmosphere::dispatchCloudScreenPass` | yes | reads `m_primaryLinearViewZ` from step 4 |
| 13 | NRC train + resolve | `NeuralRadianceCache::dispatchTrainingAndResolve` | **no** | `isActive()` |
| 14 | RTXDI confidence | `dispatchConfidence` | **no** | needs NRD primary or ReSTIR GI (`:391-401`) |
| 15 | ReSTIR GI temporal/spatial/final shading | `DxvkReSTIRGIRayQuery::dispatch` | **no** | `useReSTIRGI()` |
| 16 | Demodulate | `DemodulatePass` | yes | always |
| 17 | NRD primary direct / primary indirect | `dispatchDenoise` | **no** under RR | `isSecondaryOnly` (`rtx_context.cpp:1867`) |
| 17' | NRD secondary combined | `dispatchDenoise` | yes when PSR possible | `enablePSRR || enablePSTR || RaytracedRenderTarget::enable` (`:1897-1900`), and `preprocessSecondarySignal` (default true) |
| 18 | Composite | `CompositePass` | yes | always |
| 19 | DLSS-RR | `DxvkRayReconstruction` | yes | `m_currentUpscaler` |
| 20 | Bloom, motion blur, tonemap, lens, sRGB/dither, neural uplift, overlay, DLFG/FSR-FG | various | per option | |

Three consequences of this map:

- Steps 5, 9, 13, 14, 15 and 17 are dead in the user's configuration. They are not dead in the codebase: every one has a live consumer under RR-off (NRD primary), NRC, or ReSTIR GI. Their runtime cost when inactive is a predicate check, so none of them is a performance target.
- The dead-path found on 2026-09-14 (RTXDI sample stealing in SHARC stages, `integrator_indirect.slangh:709`) was a *shader* dead path: a runtime flag reaching code compiled without the binding it needs. The remaining risks of that class are listed in section 3.
- `m_rtxdiGradients` (`rtx_resources.cpp:1175`) is allocated and bound to the indirect pass every frame regardless of whether step 9 runs. That is the only mode-independent allocation found that is unread in the user's configuration; it is a 2-layer R16G16 texture at gradient resolution (small), so memory only.

### 1.1 Indirect integration in detail

The indirect integrator is one shader source family (`integrator_indirect.slangh`, 1621 lines) compiled into 128 blobs (77 MB of the 259 MB total, `_Comp64Release/src/dxvk/rtx_shaders`):

| Family | Blobs | Bytes | Selected by |
|---|---:|---:|---|
| Legacy / ReSTIR GI (`!ENABLE_NRC && !ENABLE_SHARC`) | 36 | 20 MB | mode 0 or 1, and SHARC fallback |
| NRC (`ENABLE_NRC=1`) | 36 | 23 MB | mode 2 |
| SHARC (`ENABLE_SHARC=1`, update/query x backend x WBOIT x stats x portals x POM) | 56 | 34 MB | mode 3 |

Modes 0 and 1 share the same 36 blobs; ReSTIR GI is a runtime flag (`cb.enableReSTIRGI`) inside them plus three separate compute passes. NRC and SHARC are compile-time axes. The mode-specific shader hooks are, by line in `integrator_indirect.slangh`:

| Hook | Lines | NRC | ReSTIR GI | SHARC |
|---|---|---|---|---|
| Path init | 1271-1391 | `initNrc` (reload progress state, clear direct outputs on terminate) | `initReSTIRGI` (radiance factor to texture) | none (state in payload) |
| Per-vertex, before material resolve | 339-377, 420-455 | `updateNrcOnHit` -> terminate decisions | store hit geometry for reservoir | none |
| Per-vertex, after emission | 567-616 | | | eligibility test, insert (update) or lookup + terminate (query) |
| NEE at vertex | 700-720, 999-1118 | | reservoir sample stealing, virtual hit distance | stealing branch compiled out |
| Path end | 1565-1598 | `writeToNrcOutputsAfterIntegration` | `storeRestirGIRadiance` | `sharcFlushVertices` (deferred update) |
| Payload | `path_state.slangh:55-130` | 5 extra fields under `#if ENABLE_NRC` | 3 flag bits, always present | `sharcState` under `SHARC_UPDATE` |

Post-trace, `dispatchNEE` selects one of four `integrate_nee` variants. The base variant (`RAB_HAS_RESTIR_GI_RESERVOIRS=1` **and** `ENABLE_NRC=1`) is selected only when both are active, which the single-valued mode enum makes impossible: it is a dead variant and a dead prewarm (`rtx_pathtracer_integrate_indirect.cpp:588, 1102-1104`).

### 1.2 G-buffer

291 blobs, 168 MB, the largest family. Axes (`gbuffer.slang:23-39`): features {none, debug, lean_psr, lean_no_psr} x psr {-, psr} x pipeline {raygen, raygen_ser, rayquery_raygen} x nrc {-, nrc} x wboit {-, wboit}, plus closest-hit/miss x material x nrc x wboit. **144 of the 291 blobs (93 MB) exist only for `ENABLE_NRC=1`.** NRC's G-buffer coupling is real code, not a define: `geometry_resolver.slangh:1600-1695` writes NRC query/training path data on primary hit and miss for both the primary and PSR resolvers, and the G-buffer dispatch binds four NRC buffers every frame (`rtx_pathtracer_gbuffer.cpp:728-729`; the bind helper binds null slices when inactive).

The `lean_psr` / `lean_no_psr` feature variants are an upstream axis (unrelated to the removed SHARC lean profile); no redundant *pass* was found in the G-buffer stage: PSR is a second dispatch only when `psrEnabled` and is one pipeline otherwise.

### 1.3 Denoiser and reconstruction axis (separate from the integrator axis)

Under DLSS-RR the only NRD work is the secondary-combined instance, and only when a PSR surface can exist (`rtx_context.cpp:1897-1929`). Primary direct and primary indirect NRD instances release their resources every frame under RR. The profile of 2026-09-13 recorded `Denoising` at 0.117 ms median, consistent with that. So on the user's path NRD is already close to free, and the remaining NRD cost is the secondary instance the fork itself justified keeping (reflections through PSR).

What NRD still serves in the codebase: every non-DLSS upscaler. The fork added FSR upscaling and frame generation (`rtx_fork_fsr.cpp`, `rtx_fork_fsr_framegen.cpp`) and keeps XeSS, NIS and TAAU. Removing NRD would remove the denoiser for all of them. RTXDI gradients and confidence exist mainly for NRD (`rtx_rtxdi_rayquery.cpp:372-401`). The denoiser axis is therefore a product-scope decision (NVIDIA-only vs. multi-vendor), not a code-simplification decision, and it is not pursued further here beyond the inventory in section 2.11.

## 2. Removal candidates

For each: cost to keep, what breaks if removed, quality change, and who consumes its outputs. Consumers are the hazard; each entry lists them explicitly.

### 2.1 NRC (`IntegrateIndirectMode::NeuralRadianceCache`)

Footprint: 3,175 lines (`rtx_neural_radiance_cache.*`, `rtx_nrc_context.*`, `shaders/rtx/pass/nrc/*`, `nrc_args.h`), 35 `rtx.nrc.*` options, 21 shader files outside `pass/nrc/` reference NRC (largest: `integrator_indirect.slangh` 52 references, `path_state.slangh` 27, `geometry_resolver.slangh` 15). Shader blobs: 36 indirect + 144 G-buffer + 2 `integrate_nee` + 1 `nrc_resolve` + 3 sparse-rendering blobs that only run with NRC = 186 of 526 blobs; roughly 116 MB of 259 MB of SPIR-V. Shipped binaries: `submodules/nrc/Bin` is 115 MB (`nvrtc64_130_0.dll` alone is 101 MB) and imposes a 610.47 minimum driver (`upstream 3789b7f6e`). Runtime cost when inactive: null buffer bindings in the G-buffer and indirect passes, `isActive()` checks, and the `enableNrc` branch in direct-pass Russian roulette (`integrator_direct.slangh:222-230`); no dispatch.

Cost to keep: the build and shipping weight above, and the mode-combination surface: sparse rendering exists only for NRC + RR; `TRUE_OR_CHECK_WHEN_NRC_ENABLED` guards 6 debug-view writes in the integrator; the NRC variant axis doubles the G-buffer matrix; the aliasing rules in `rtx_resources.cpp:417-442` special-case NRC + RR. Upstream investment is concentrated here (Boost Emissives, driver bumps, the new `remixinternal_GetNrcStatus` export for HdRemix in the pending sync).

What breaks if removed: `SparseRendering` (dead by construction; delete with it), `rtx_user_menu.cpp:559` NRC preset combo, graphics presets `enableNrcPreset` (`rtx_options.cpp:505-516`; must be retargeted to SHARC or ReSTIR GI), `RtxContext::checkNeuralRadianceCacheSupport` (`:1641-1651`), debug views 550-590, the `m_indirectRadianceHitDistance` aliasing branch, the G-buffer NRC axis and bind helper, `integrate_nee_nrc`, `remixinternal_GetNrcStatus` after the next sync (HdRemix calls it; it needs a stub returning "inactive"). The NRC build dependency (`meson.build:499-541`) and packaging.

Quality change: none for the user (mode unused). For other users: NRC is upstream's default and is the mode upstream tunes RR presets against; removing it forces them to ReSTIR GI or SHARC.

Consumers of NRC outputs: `nrc_resolve` writes `m_primaryIndirectDiffuse/Specular` and `IndirectRadianceHitDistance` (`nrc_resolve.comp.slang:305-327`); `integrate_nee` adds first-bounce NEE to training vertices. Nothing outside the NRC path reads NRC buffers. Safe to remove as a unit **if** the sparse-rendering subsystem goes with it.

### 2.2 ReSTIR GI (`IntegrateIndirectMode::ReSTIRGI`)

Footprint: 1,871 lines (`rtx_restir_gi_rayquery.*`, `restir_gi_*` shaders, `algorithm/rtxdi/*`), 40 `rtx.restirGI.*` options, 33 fields in `RaytraceArgs`, 36 references in `integrator_indirect.slangh`, 17 in `RtxdiApplicationBridge.slangh` (shared with RTXDI DI), 10 in `integrate_nee.slangh`, 5 in `demodulate`, 4 in `composite`, 2 in `rtxdi_spatial_reuse`, 1 in `ray_reconstruction.h`. Blobs: 4 (`restir_gi_temporal_reuse`, `spatial_reuse`, `final_shading`, `final_shading_occlusion`) + `integrate_nee_restir_gi`. Its per-mode resources are created only when active (`RtxPass::createDownscaledResource`, `rtx_restir_gi_rayquery.cpp:318-348`).

Cost to keep: three runtime flags compiled into the 36 legacy blobs (which SHARC blobs already exclude via `#if !ENABLE_NRC && !ENABLE_SHARC`), the `enableReSTIRGI` branches in demodulate/composite/RTXDI spatial reuse/RR prepare, the option surface, and `enablePreviousTLAS = !RR || useReSTIRGI` coupling (`rtx_options.h:678`).

What breaks if removed: `SparseRendering` comment-level dependency only; `enablePreviousTLAS` simplifies to `!RR`; `demodulate.comp.slang:640` (`isIndirectActive` predicate) and `composite.comp.slang:316-351` (BSDF factor 2 and post filter) lose their ReSTIR GI branch; `m_gbufferLast`, `m_bsdfFactor`, `m_rtxdiGradients` remain because RTXDI DI reads them (`rtx_rtxdi_rayquery.cpp:451-507`); graphics presets that select ReSTIR GI when NRC is unsupported.

Quality change: for the user none. For everyone else ReSTIR GI is the only path that is (a) supported on GPUs without `shaderBufferInt64Atomics`/`shaderFloat16` (SHARC's `isSupported`, `rtx_sharc.cpp:31-36`), (b) tuned for NRD denoising, and (c) upstream's fallback when NRC is unsupported. SHARC's own fallback today is the importance-sampled path (`rtx_sharc.cpp:45-72` sets `m_active=false`; `restirGI.isActive()` stays false because the mode is still `Sharc`), which is the noisiest option. Removing ReSTIR GI removes the only good fallback. **Recommendation: keep it, and make SHARC's unsupported-device fallback select it explicitly** (a one-line policy change in `prepareFrame` or the option-change handler; a quality decision, so flagged rather than assumed).

Consumers of ReSTIR GI outputs: `restir_gi_final_shading` writes `m_primaryIndirectDiffuse/Specular` in place; `m_lastCompositeOutput` is written by composite (`rtx_composite.cpp:391`) and read by the indirect pass for `StealPixel`; `m_bsdfFactor2` is read by composite. All are inside the mode's own `isActive()` gate.

### 2.3 Importance-sampled path (`IntegrateIndirectMode::ImportanceSampled`)

Not removable. It is the substrate: `integrate_indirect*` legacy variants are the SHARC query/update variants without `ENABLE_SHARC`, and they are dispatched whenever `RtxSharc::prepareFrame` declines (device features, raytraced render target camera valid, WBOIT/portals/OMM without the allow flags, allocation failure). The FNV configuration has `raytracedRenderTarget.enable=True`; `enableRaytracedRenderTarget` is true only when a render-target camera is valid this frame (`rtx_context.cpp:1138`), so in scenes with a captured render target SHARC silently falls back to the importance-sampled shaders. The status string in the SHARC panel is the only indication. This is a mode-combination gap worth a log line and possibly a ReSTIR GI fallback (see 2.2).

The mode enum value itself could go (fold into "SHARC with cache disabled"), but that changes nothing compiled and adds conflict surface in `rtx_options.h`, the UI and presets. Not worth it.

### 2.4 SHARC backend matrix (fork-owned)

`rtx_sharc.h` exposes `queryTraceRay`, `queryRayGeneration`, `updateRayGeneration`, `deferredUpdates`, `collectQueryStats`, `measureGpuTime`, `fuseAssembly` and the three `allow*` overrides. Each is a compiled axis:

| Variant group | Blobs | Used by the shipped configuration |
|---|---:|---|
| TraceRay query raygen: {plain, SER} x {WBOIT} x {stats} x {fused} | 12 | 2 (`sharc_query_trace[_ser]` and `_wboit`) |
| TraceRay query closest hit: {portals} x {POM} x {WBOIT} x {stats} | 16 | 2 (`no_portals_no_pom[_wboit]`) |
| TraceRay query miss: {portals} x {WBOIT} x {stats} | 8 | 1-2 |
| RayGen-RayQuery query: {WBOIT} x {stats} | 4 | 0 |
| Compute query: {WBOIT} x {stats} | 4 | 0 |
| Update raygen: {plain, deferred, deferred4} x {WBOIT} | 6 | 2 (`deferred4_raygen[_wboit]`) |
| Update compute: {plain, deferred, deferred4} x {WBOIT} | 6 | 0 |

Cost to keep: 56 blobs (34 MB), the 220-line selection tree in `dispatch()` and `getSharcTracePipelineShaders`, three shader layout classes, 12 options, and the `compatibilityFlags` cache-reset bit set (`rtx_sharc.cpp:60-66`) that must enumerate every backend. Every one of these was a measurement instrument; the log records their results (TraceRay adopted; fusion measured slower, 2.956 vs 2.339 ms; stats are diagnostic).

What breaks if removed: nothing in the render path. `validate_sharc_*.py` scripts pin blob names and would need updating. `measureGpuTime`/`collectQueryStats` could stay behind `REMIX_DEVELOPMENT` if the timing panel is still wanted.

Quality change: none; the retained blobs are byte-identical.

Merge tax: zero. Every line is fork-owned (`rtx_sharc.*`, `sharc/` shaders, the SHARC classes in `rtx_pathtracer_integrate_indirect.cpp`, the `//!variant` blocks in `integrate_indirect*.slang`).

### 2.5 Fused query/assembly (`rtx.sharc.fuseAssembly`)

Rejected on measurement (2026-09-13). Still present: 4 raygen variants, `IntegrateIndirectSharcFusedShader`, `integrate_fused_binding_indices.h` / `integrate_fused_bindings.slangh`, `SHARC_FUSED_ASSEMBLY` blocks in `integrator_indirect.slangh:1586-1589` and `integrate_nee.slangh:84-88`, the throughput allocation branch in `rtx_resources.cpp:1154-1157`, the seven-term predicate in `dispatchLighting` (`:983-989`). Fork-owned; zero tax; delete.

### 2.6 Sparse rendering

Dead by construction in this configuration and dead globally if NRC is compiled out (`isEnabled` requires NRC active, `rtx_sparse_rendering.cpp:125-131`). 11 options, 3 blobs, five `m_sparseRendering*` textures bound in indirect, NEE, demodulate and composite. Upstream just refactored it (`ee655a978`, unified sampling rate, 4 NRC/GI-touching hunks in the pending sync) so it is an active upstream surface: deleting it is the same tax class as deleting NRC. Recommendation: leave the code, stop shipping its 3 blobs with NRC (they are only prewarmed when `isEnabledByOptions()`).

### 2.7 Legacy indirect backends (`RaytraceMode::RayQuery`, `RayQueryRayGen`)

Upstream-owned axis; the user runs `TraceRay`. Blobs: 8 + 8 of the 36 legacy, and the same again for NRC. Deleting them removes a supported upstream configuration (compute-only GPUs without ray pipelines) for no runtime gain. Leave.

### 2.8 Dead `integrate_nee` combined variant

`integrate_nee.comp` (both defines) is unreachable (section 1.1). Fork-owned split (`b063c9691`); remove the variant and the `IntegrateNEEShader::getShader()` fallback branch, or turn the branch into an assert. Zero tax.

### 2.9 Resources live regardless of mode

| Resource | Allocated | Read in user config by | Verdict |
|---|---|---|---|
| `m_rtxdiGradients` (`rtx_resources.cpp:1175`) | always | nobody (gradient pass is off under RR+SHARC; indirect binds it but only reads it under `enableReSTIRGILightingValidation`) | small; could be lazily created by `dispatchGradient`; upstream-owned, low value |
| `m_bsdfFactor`, `m_gbufferLast` | always | RTXDI DI temporal/spatial (`rtx_rtxdi_rayquery.cpp:451-507`), composite | shared producer, keep |
| `m_rtxdiReservoirBuffer` | always | RTXDI DI, direct pass; legacy indirect (stealing) | shared producer, keep |
| NEE cache buffers | always | indirect (all modes), NEE assembly, RTXDI | shared, keep |
| Unordered TLAS | always | G-buffer/primary particles, indirect unordered resolve | shared, keep (the trap from the lean work) |
| Previous TLAS + BLAS keep-alive | `enablePreviousTLAS` | RTXDI visibility, gradients | already gated; under RR+SHARC it is off |
| NRC buffers | NrcContext when active | NRC only | already conditional |
| ReSTIR GI buffers | `RtxPass` when active | ReSTIR GI only | already conditional |
| SHARC buffers (`capacity` x 40 B: 40 MB at 2^20) | `prepareFrame` when selected | SHARC | correct |
| 7 `DxvkDenoise`/`NRDContext` objects (`dxvk_objects.h:427-435`) | constructed always; GPU resources created on first dispatch and released when unused | NRD path | already lazy |

No producer was found that could be gated on the indirect mode without breaking a consumer outside the indirect path. This confirms the 2026-09-13 dependency audit.

### 2.10 Duplicated math between direct and indirect

- `evalNEESecondary`, `evaluateUnshadowedLight`, `sampleDirection`, Russian roulette (`integrator.slangh:123-320`) are already shared.
- RTXDI sample stealing exists in both integrators (`integrator_direct.slangh:452`, `integrator_indirect.slangh:700-760`); the indirect copy is dead for SHARC and, per the option's own text, buys no visible quality for an 8% integrate cost in the legacy path. Deleting it from the indirect integrator entirely is an upstream-line edit (`~60 lines`, upstream touched this region in 0cad9edcb and f33678769) so it has tax; the guard already gives SHARC the benefit.
- The fork-touchpoints index (`docs/fork-touchpoints.md:905-934`) still lists `evalAtmosphereSunNEESecondary` / `evalAtmosphereMoonNEESecondary` (~200 duplicated lines) in `integrator_indirect.slangh`; those functions no longer exist in the shader tree (the atmosphere sun is now a regular light with a cloud-shadow term at `:789-801`). The index is stale for this file and should be refreshed before any refactor uses it as the conflict map.

### 2.11 Denoiser / reconstruction axis inventory (no recommendation)

Removable only by dropping non-DLSS support: `rtx_nrd_context.*` (1,037 lines), `rtx_denoise.*`, `rtx_nrd_settings.*`, `external/nrd`, the three NRD instances plus three reference-mode second-lobe instances, `denoiseDirectAndIndirectLightingSeparately` (combined-denoiser branch in demodulate/composite), RTXDI gradient/confidence passes (only NRD and ReSTIR GI consume them), `useDenoiserReferenceMode` and the accumulation options, NRD-specific `NrdArgs` in composite. Under RR the only live piece is the secondary-combined instance. The fork's FSR/XeSS/TAAU support makes NRD load-bearing; this axis is a product decision and not part of the recommended plan.

## 3. Hazards

### 3.1 Shared producers (the trap)

Listed in 2.9. The rule that survived the lean work still holds: nothing that RTXDI, the G-buffer, NEE assembly, demodulate or composite reads may be gated on the indirect mode, and nothing may be gated on `sharc.isActive()` because the fallback frames run the legacy shaders against the same producers.

### 3.2 Atmosphere, clouds, volumetrics

Checked explicitly (`rtx_atmosphere.cpp`, `rtx_global_volumetrics.cpp`, `rtx_weather.cpp`, `rtx_fork_*.cpp`, `pass/atmosphere/*`, `pass/volumetrics/*`, `algorithm/volume_integrator*`): **no atmosphere, cloud or volumetric code reads NRC, ReSTIR GI or SHARC state.** Volumetrics uses only the RTXDI random sampler (`volume_integrate.slangh:59-98`). The cloud screen pass reads `m_primaryLinearViewZ` (G-buffer). Composite reads `atmosphereArgs` and the indirect radiance textures, which every mode produces in the same format.

The dependency runs the other way, and it is inside the indirect integrator, so any variant that survives must keep it:

- `#define ATMOSPHERE_AVAILABLE` in `integrate_indirect.slang`, `integrate_indirect_closesthit.rchit.slang`, `integrate_indirect_miss.rmiss.slang`; sky radiance on miss via `evalSkyRadiance(..., applySkyIndirectRadianceScale = pathState.skyGatherEligible)` (`integrator_indirect.slangh:399-411`).
- `skyGatherEligible` is set from the first sampled lobe (`:1430`) and updated per vertex (`:675`); SHARC's `sharcLobeEligible` reuses it (`:576`). With `allowSpecularPaths=True` (user config) glossy paths are cached in an isotropic cache; the SHARC design doc warned about this and it is a quality choice, not a dead path.
- Cloud ground shadow on secondary NEE for the atmosphere sun (`:789-801`).
- The SHARC cache stores sky-lit radiance; weather and time-of-day changes are handled by `accumulationFrames`/`staleFrames`, not by reset. Any change to the sky-scale policy (`skyIndirectRadianceScale`, weather blending) changes cached values for up to `staleFrames` frames. Not a removal hazard; a policy-change hazard.

### 3.3 Presets and UI

`rtx_options.cpp:505-516` selects NRC (or ReSTIR GI if unsupported) for the high presets; nothing selects SHARC. `rtx_user_menu.cpp:559` and `dxvk_imgui.cpp:352-365` list all modes. Any consolidation that leaves the preset code untouched leaves the user one preset click away from NRC. Fixing this is fork-owned (the preset lambda is upstream code, but the SHARC branch in it would be a ~4-line inline tweak).

### 3.4 Validators and the touchpoint index

`scripts-common/validate_sharc_integration.py`, `validate_sharc_resources.py`, `validate_sharc.py`, `validate_sharc_deferred.py` and `validate_sharc_estimator.py` pin blob names, bindings 10/51, the reservoir guard and the gate expressions in `rtx_context.cpp`. Every stage below must update them in the same commit or they become false alarms. `docs/fork-touchpoints.md` is stale for `integrator_indirect.slangh` (3.2) and does not yet list the SHARC edits to upstream files; the fridge-list invariant in `docs/CONTRIBUTING.md` requires an entry per upstream-file edit.

### 3.5 Upstream API that will land at the next sync

The 50 pending upstream commits include `ee655a978` (unified sampling rate: renames `m_sparseRenderingIndirectActiveLocalPixelCoords` to `m_sparseRenderingActiveLocalPixelCoords`, touching the indirect, NEE, NRC and demodulate bindings that SHARC's `AssemblyResources` now references), `673dc2af9` (`remixinternal_GetNrcStatus` export, atomics in `NeuralRadianceCache`), and `7294f6856` (DLSS SR/RR 4.5). These are the concrete next conflicts regardless of what is decided here.

## 4. Interface proposal for the swappable indirect boundary

### 4.1 Where the seam actually is

Three seams exist today, and they are not equally abstractable.

1. **Scheduling and resources (C++).** `IntegrateIndirectMode` enum; per-mode owners (`NeuralRadianceCache : RtxPass`, `DxvkReSTIRGIRayQuery : RtxPass`, `RtxSharc : CommonDeviceObject`); `RtxPass` already gives `isEnabled()/onActivation()/onDeactivation()/createDownscaledResource()` (`rtx_resources.h:527-560`); `dispatchLighting` decides pre-passes (SHARC update/resolve), the trace, and NEE assembly; `injectRTX` runs post-passes (NRC train/resolve, ReSTIR GI reuse) between the cloud pass and demodulate; `updateRaytraceArgsConstantBuffer` fills per-mode constants; `rtx_resources.cpp:417-442` picks aliasing per mode.
2. **Descriptor layouts.** `IntegrateIndirectRayGenShader` (all slots), `IntegrateIndirectSharcBaseShader` (minus NRC/ReSTIR GI/reservoir/previous lights), `IntegrateIndirectSharcShader/UpdateShader/StatsShader/FusedShader`, `IntegrateNEEShader/IntegrateNEEPlainShader`. Already per-backend.
3. **Shader hooks and variants.** The six hook sites in section 1.1, the payload fields, the `//!variant` matrices in four `.slang` files, and the `#if !ENABLE_NRC && !ENABLE_SHARC` "everything else" blocks.

Seams 1 and 2 are cheap to formalize and are mostly formalized already. Seam 3 is where 80% of the coupling lives, and no C++ interface reaches it.

### 4.2 What a C++ interface would look like, and what it would cost

```
class IndirectBackend {              // one instance per mode, owned by DxvkObjects
  virtual bool isSupported(const DxvkDevice&) const = 0;
  virtual void prepareFrame(RtxContext&, RaytraceArgs&, bool resetHistory) = 0;   // constants, allocation, fallback decision
  virtual void dispatchPrePasses(RtxContext&, const RaytracingOutput&) = 0;      // SHARC update+resolve; NRC: none; ReSTIR GI: none
  virtual DxvkRaytracingPipelineShaders selectTrace(const TraceKey&) = 0;        // backend x SER x OMM x WBOIT x portals x POM
  virtual void bindTraceResources(RtxContext&) = 0;
  virtual Rc<DxvkShader> selectAssembly() = 0;                                   // integrate_nee variant
  virtual void dispatchPostPasses(RtxContext&, const RaytracingOutput&) = 0;     // NRC train/resolve; ReSTIR GI reuse+final shading
  virtual AliasingPolicy aliasing() const = 0;                                   // IndirectRadianceHitDistance rules
};
```

Crossing the boundary: `RaytraceArgs` (per-mode sub-structs already exist: `nrcArgs`, `sharcArgs`, the 33 ReSTIR GI fields), the shared G-buffer and lighting textures, the NEE cache, `IndirectRadianceHitDistance` (the contract output every backend must write per pixel, with `firstBounceHitDistance` in `.w`), and `m_primaryIndirectDiffuse/Specular` after assembly.

Cost: one virtual call per frame per method (nothing), but the real costs are (a) it changes upstream lines in `rtx_context.cpp`, `rtx_pathtracer_integrate_indirect.cpp` and `rtx_resources.cpp` that upstream edits monthly (58, 14 and 19 commits in 18 months respectively), so it *raises* the merge tax for as long as NRC and ReSTIR GI remain; (b) it cannot express the shader side: a new technique still needs its own `//!variant` rows, its own payload fields, and edits at the same six hook sites; (c) the `TraceKey` explodes exactly like `getPipelineShaders` does now because the SER/OMM/WBOIT/portal/POM axes are orthogonal to the backend.

### 4.3 Recommendation on the interface

Do not build the virtual interface. The honest swap mechanism is: git history for the deleted backends, and **the SHARC hook blocks as the documented shader contract**. Concretely:

- On the shader side, the only refactor worth doing is naming: move the three SHARC blocks in `integrator_indirect.slangh` (`:567-616`, `:1561-1563`, the stealing guard) into `sharc_update.slangh`/a new `sharc_integrator_hooks.slangh` as `sharcOnResolvedVertex(...) -> bool terminate`, `sharcFinishPath(...)`, so the integrator body calls named hooks instead of carrying `#if` blocks. This is fork-owned text moving between fork-owned files plus three call sites in an upstream file; the call sites are already fork lines. It makes the contract legible (a future technique implements the same three functions and its own variant rows) without pretending the variant matrix is abstracted away. Net tax: none beyond today's.
- On the C++ side, the seam consolidates by itself once the SHARC backend matrix is pruned (2.4): `dispatch()` collapses to "SHARC update / SHARC query / legacy fallback", the three SHARC layout classes become one, and `dispatchLighting` loses the fusion predicate. Do that as the by-product of deletion, not as an abstraction exercise.
- Keep `IntegrateIndirectMode` and `RtxPass` as they are. They are the interface upstream already maintains; matching it costs nothing at sync time.

## 5. Upstream divergence cost

### 5.1 Mechanism and cadence

Syncs are replants: the fork's cumulative diff is squashed onto the new upstream tip (`c5740aae5`, parent `59affb700` = upstream 2026-07-28; message: "Re-plant the fork's work as a single squash ... Conflicts resolved in favour of the fork where it owns the subsystem ... and upstream where it refactored shared infra"). Observed cadence: 2026-06-21 (76 upstream commits absorbed), 2026-07-29, and 50 upstream commits pending as of 2026-08-24. Upstream rate: 189, 152 and 154 non-merge commits in the last three 6-month windows. The fork's diff against its base is large already: 439 files, +61,158/-3,364 lines; +41,083/-2,893 in `rtx_render` + `shaders` alone. Every sync already resolves conflicts in `rtx_context.cpp` (+306 fork lines), `rtx_pathtracer_integrate_indirect.cpp` (+640), `integrator_indirect.slangh` (+204), `composite.comp.slang` (+579).

### 5.2 Churn on the files a deletion would touch

Upstream commits per file (last 18 months / last 6 months / since the current base):

| File | 18 mo | 6 mo | pending |
|---|---:|---:|---:|
| `rtx_context.cpp` | 58 | 21 | 4 |
| `rtx_options.h` | 65 | 16 | 0 |
| `rtx_neural_radiance_cache.cpp` | 26 | 6 | 2 |
| `rtx_pathtracer_integrate_indirect.cpp` | 14 | 6 | 1 |
| `rtx_restir_gi_rayquery.cpp` | 11 | 2 | 0 |
| `rtx_resources.cpp` | 19 | 3 | 0 |
| `integrator_indirect.slangh` | 16 | 7 | 0 |
| `geometry_resolver.slangh` | 29 | (n/a) | 1 |
| `raytrace_args.h` | 19 | 8 | 3 |
| `composite.comp.slang` | 12 | 7 | 1 |
| `demodulate.comp.slang` | 7 | 6 | 1 |
| `pass/rtxdi/` (ReSTIR GI shaders live here) | 12 | (n/a) | 4 |

### 5.3 Conflict surface, counted

A throwaway script (scratchpad, not in the repo) split every upstream diff hunk in the candidate file set by whether its added/removed lines mention NRC or ReSTIR GI identifiers, and whether the file would be wholly deleted (free modify/delete conflict) or is shared (real three-way conflict).

| Window | NRC hunks, deleted files | NRC hunks, shared files | ReSTIR GI hunks, deleted files | ReSTIR GI hunks, shared files | All hunks in the file set |
|---|---:|---:|---:|---:|---:|
| Pending sync (4 weeks) | 6 | 0 | 0 | 2 | 129 |
| Last 6 months | 33 | 39 | 8 | 23 | 1,080 |
| Last 18 months | 110 | 98 | 29 | 43 | 2,729 |

Rates from the 6-month window: NRC about 6.5 shared-file hunks per month (12 total), ReSTIR GI about 4 shared-file hunks per month (5 total). The shared-file hunks come predominantly from sweeps (`f33678769` uniform downscaling: 25 hunks; `f900f2c1b` GPU/CPU optimizations: 21; `3afe79184` "Code refactor": 20; `230d7db6e` WBOIT: 27), which is the worst kind: a fix to shared code lands inside a region the fork deleted, and the resolution has to separate the two by hand.

Interpretation:

- Deleted-file hunks are free. `git rebase`/replant reports modify/delete, and "keep deleted" is the answer every time.
- Shared-file hunks are the tax. At the observed cadence that is roughly 25-50 extra conflicting hunks per sync for NRC and 15-30 for ReSTIR GI, on top of the fork's existing conflict load in the same files. Each is typically a few minutes if it is a rename sweep and much longer if it interleaves with a fix. The risk is not time; it is silently dropping an upstream fix that shared lines with a deleted block.
- NRC is also where upstream is heading (default mode, RR presets, HdRemix stats API). Deleting it means diverging from the direction of every future indirect-lighting change upstream makes.

### 5.4 Merge-tax verdict

- **ReSTIR GI hard deletion: not worth it.** Runtime cost when inactive is zero by construction; code removed is small; it is the correct fallback for SHARC-unsupported devices and NRD users; tax ~4 shared hunks/month forever.
- **NRC hard deletion: worth it only if the fork intends to stop syncing.** The gain is the largest available (build time, 115 MB of DLLs, driver floor, G-buffer matrix, sparse rendering, 3.2k lines, the RR/NRC aliasing special cases); the tax is the highest in the tree (~6.5 shared hunks/month plus the `remixinternal_GetNrcStatus` export to stub) and it compounds with every sync. The replant workflow makes it *permanent*: there is no point at which the deletion stops conflicting.
- **NRC build-time exclusion: worth it now.** Keep every source line; add a build option that (a) makes `compile_shaders.py` skip variants whose define list contains `ENABLE_NRC=1` and emit empty blob headers for them so the C++ `#include` lines stay valid, (b) makes `NrcContext::checkIsSupported` return false under the option (one line; the UI already removes the combo entry and presets already fall through to ReSTIR GI when NRC is unsupported), (c) stops linking `NRC_Vulkan` and shipping `submodules/nrc/Bin` (`meson.build:499-541`, ~10 upstream lines). Tax: three small upstream-file edits that upstream rarely touches, versus the 200+ lines of shared-file NRC code left exactly as upstream has them. Gain: ~186 fewer blobs to compile and embed, 115 MB less to ship, no NRC driver floor. Not gained: code-surface reduction; the mode-combination surface stays, but it is provably unreachable when the mode cannot activate.

## 6. Staged plan

Each stage is independently buildable, validated by byte-identity of the retained blobs plus the existing validators, and revertible by a single `git revert`. Riskiest and least reversible last. No stage promises a speedup; stages 1-4 are structurally cheaper and should be image-identical.

| Stage | Change | Owner of the lines | Merge tax | Verification | Reversible |
|---|---|---|---|---|---|
| 0 | Paired same-session measurement harness (already requested in `GPU-stage-profiling-2026-09-13.md`); a script that inventories blobs by axis from the build directory; refresh `fork-touchpoints.md` for `integrator_indirect.slangh` | fork | none | harness output on the unchanged build | n/a |
| 1 | Delete `fuseAssembly` (2.5) and the dead combined `integrate_nee` variant (2.8) | fork | none | validators; retained blobs byte-identical; FNV image identical | trivially |
| 2 | Prune the SHARC backend matrix (2.4) to TraceRay query {plain, SER} x {WBOIT} + closest-hit {portals, POM} x {WBOIT} + miss {portals} x {WBOIT} + deferred4 raygen update {WBOIT}; move stats/timing behind `REMIX_DEVELOPMENT` or delete; collapse the three layout classes to one; collapse `dispatch()`; update `compatibilityFlags`; update validators | fork | none | 56 -> 18 blobs (4 query raygen, 8 closest hit, 4 miss, 2 update); retained blobs byte-identical; FNV A/B identical (SER on/off, WBOIT on/off) | trivially |
| 3 | Fix the preset hazard (3.3): presets that would select NRC/ReSTIR GI keep the user's SHARC selection when SHARC is supported; SHARC fallback selects ReSTIR GI instead of importance-sampled (2.2, 2.3); log the render-target fallback | ~10 lines in `rtx_options.cpp`, `rtx_sharc.cpp` | small (`rtx_options.cpp` sees ~1 upstream commit/month) | preset switch in FNV keeps SHARC; force a fallback (disable allowWboit) and confirm ReSTIR GI passes appear in the profile | trivially |
| 4 | Name the shader hooks (4.3): move the SHARC blocks into a hooks include with three named functions | fork text; three fork call sites | none | all 14 SHARC blobs byte-identical (the compiler inlines) | trivially |
| 5 | NRC build-time exclusion (5.4) as a meson option defaulting on for fork packages | 3 upstream files, ~15 lines | small | NRC absent from the UI; blob count -186; DLL size; SHARC and legacy blobs byte-identical; ReSTIR GI still selectable | option flip |
| 6 | Optional: lazy `m_rtxdiGradients` creation; `enablePreviousTLAS` unchanged | upstream lines | small | memory counter; RTXDI gradient path with RR off still works | trivially |
| 7 | Optional and only if syncing is abandoned: hard-delete NRC + sparse rendering, then ReSTIR GI | upstream lines everywhere | high, permanent | full mode-matrix retest on RR and NRD paths | practically irreversible |

Stages 1, 2 and 4 remove roughly 45 blobs, three shader layout classes, ~600 lines of selection code and 8 options without touching a line upstream owns. Stage 5 removes the build and shipping weight of NRC without touching the shared NRC code. Stage 7 is listed so that the decision is explicit, not so that it is taken.

## 7. What could not be determined without measurement

- **Runtime residue of the inactive modes.** Structurally near zero (null bindings, predicates, one small texture). The 2026-09-13 resource-specialization build already reported no measurable change from stripping NRC/ReSTIR GI descriptors, which is consistent. Confirming it needs the stage-0 paired harness on the pre- and post-stage-2 builds.
- **Shader build time by axis.** The last full shader compile step took 122.1 s wall-clock (`_Comp64Release/.ninja_log`, `_built_shaders.txt`; parallel across cores), but per-variant durations are not logged. Measuring the share attributable to the NRC axis and the SHARC diagnostic axis needs a per-variant timing run of `compile_shaders.py`.
- **Startup and pipeline-cache effect of fewer prewarmed pipelines.** Structurally fewer; unmeasured.
- **Whether NRC's DLLs cost anything when NRC is inactive.** `NrcContext::checkIsSupported` is called at G-buffer and indirect prewarm; whether it loads `NRC_Vulkan.dll`/`nvrtc` was not traced. Measure process working set with and without stage 5.
- **Device population.** Whether any target user's GPU lacks `shaderBufferInt64Atomics`/`shaderFloat16` (SHARC unsupported) decides how much ReSTIR GI matters as a fallback. Not knowable from source.
- **SHARC on the NRD path.** SHARC has only been evaluated under RR. Whether its noise level is acceptable through ReBLUR/ReLAX (FSR/XeSS users) is a quality question that decides whether ReSTIR GI is still needed for non-RR users.
- **Stage-3 quality.** Switching the SHARC fallback from importance-sampled to ReSTIR GI is a visible change in fallback frames (render-target scenes in FNV); it needs an A/B.
- **`allowSpecularPaths=True`.** The isotropic cache stores glossy bounces; the visual cost in reflections of bright ground was never isolated. Toggle A/B.

## 8. Files consulted

`docs/SHARC-implementation-status.md`, `docs/Lean-SHARC-parity-2026-09-14.md`, `docs/Lean-SHARC-dependency-audit-2026-09-13.md`, `docs/CoreRenderingOptimizations.md`, `docs/GPU-stage-profiling-2026-09-13.md`, `docs/NrcExecutionPathResearch.md`, `docs/SHARC-design-revised-9-10.md`, `docs/ShaderVariants.md`, `docs/fork-touchpoints.md`, `docs/CONTRIBUTING.md`, `AGENTS.md`; `src/dxvk/rtx_render/{rtx_context, rtx_pathtracer_integrate_indirect, rtx_pathtracer_gbuffer, rtx_restir_gi_rayquery, rtx_neural_radiance_cache, rtx_nrc_context, rtx_sharc, rtx_rtxdi_rayquery, rtx_nee_cache, rtx_resources, rtx_denoise, rtx_nrd_context, rtx_ray_reconstruction, rtx_composite, rtx_demodulate, rtx_sparse_rendering, rtx_options, rtx_atmosphere, rtx_global_volumetrics}.{cpp,h}`, `src/dxvk/dxvk_objects.h`, `src/dxvk/imgui/{dxvk_imgui, rtx_user_menu}.cpp`; `src/dxvk/shaders/rtx/algorithm/{integrator_indirect, integrator_direct, integrator, path_state, geometry_resolver}.slangh`, `pass/integrate/*`, `pass/sharc/*`, `pass/nrc/*`, `pass/rtxdi/*`, `pass/gbuffer/*.slang`, `pass/composite/composite.comp.slang`, `pass/demodulate/demodulate.comp.slang`, `pass/volumetrics/volume_integrate.slangh`, `pass/atmosphere/atmosphere_sky.slangh`, `pass/raytrace_args.h`, `utility/debug_view_indices.h`; `_Comp64Release/src/dxvk/rtx_shaders/*.spv` (blob inventory), `_Comp64Release/.ninja_log`, `_Comp64Release/fused-assembly-first-capture-rtx.conf`; git history of `upstream/main` against merge-base `59affb700`.
