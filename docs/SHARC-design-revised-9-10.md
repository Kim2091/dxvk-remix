# SHARC integration design for revised-9-10

Research date: 2026-09-11. Implementation update: 2026-09-12. Experimental renderer integration is implemented; runtime image quality and performance are not yet validated. See `SHARC-implementation-status.md` for the implemented scope and validation.

Target: `revised-9-10`, commit `fd4cc524ffd8a2da61794f89e05768aa187c171a`, inspected in the `wsn3g` worktree. The implementation and this design are in the `wsn3g` worktree; the user-supplied `dxvk-remix` directory is on a different branch. The original research below describes the target commit. Implementation decisions in the status document supersede proposals below.

Three Luna research agents investigated upstream SHARC, Remix's host integration, and Remix's path estimator. Their findings were cross-checked against the pinned sources. Recommendations below are intentionally distinct from existing behavior.

## Recommendation

Implement SHARC as a fourth indirect integration choice alongside the existing modes: a world-space surface-radiance cache with a sparse update pass, compute resolve, and queries during normal indirect tracing. Retain current defaults during development. NRC and ReSTIR GI remain alternative indirect modes; RTXDI direct lighting, NEE sampling, denoising, and the branch's sky/cloud renderer remain part of the rendering pipeline.

Start with diffuse transport, ordinary 64-bit hash keys, non-directional cache storage, and explicit emissive handling. Query the first eligible indirect hit. Do not replace the primary surface with cached lighting. Reuse geometry, material, ray traversal, and secondary lighting routines through shader variants; avoid a separate material/path-tracing implementation.

The important Remix-specific simplification is to start cache insertion at the first indirect hit. The primary surface seeds an update ray but is not itself inserted. Its deferred NEE contribution consequently stays in the existing output pipeline. Secondary-hit direct lighting is already evaluated inline, where SHARC needs it.

## Research baseline

| Source | Revision and findings |
| --- | --- |
| [SHARC](https://github.com/NVIDIA-RTX/SHARC/tree/4e21b585c33c83d723ca9a1e11bbb1090d145793) | Pinned commit `4e21b585c33c83d723ca9a1e11bbb1090d145793`; changelog identifies 1.8.3. Shader-only implementation, no inference runtime or host SDK queue submission. |
| [Integration guide](https://github.com/NVIDIA-RTX/SHARC/blob/4e21b585c33c83d723ca9a1e11bbb1090d145793/docs/Integration.md) | Update -> Resolve -> Render/Query; sparse example is one pixel per 5x5 block. |
| [Current shader API](https://github.com/NVIDIA-RTX/SHARC/blob/4e21b585c33c83d723ca9a1e11bbb1090d145793/include/SharcCommon.h) | Authoritative API, temporal rules, propagation, and query behavior. |
| [RTXGI path tracer](https://github.com/NVIDIA-RTX/RTXGI/tree/10b5770b8eaddfc1faab82b65f799ac6f47dcc44/Samples/Pathtracer) | Behavioral reference. Uses older field names; copying it verbatim would not match current SHARC. |
| Remix target | `rtx_context.cpp`, `rtx_pathtracer_integrate_indirect.cpp`, `integrator_indirect.slangh`, `integrate_nee.comp.slang`, NRC lifecycle, adapter/device features, atmosphere/weather history, and shader build/variant rules. |

Current `SharcParameters` uses `hashGridParameters` and `hashGridData`; older sample names are `gridParameters` and `hashMapData`. Current `SharcResolveParameters` contains `cameraPositionPrev`, `accumulationFrameNum`, `responsiveFrameNum`, `staleFrameNumMax`, and `frameIndex`. The final boolean in `SharcGetCachedRadiance` is `skipResponsiveLighting`, not a debug flag. Implement against the pinned headers.

SHARC is an approximate radiance cache, with spatial and temporal bias. It can reduce secondary tracing and noise; neither a speedup nor parity with NRC is established for this branch. Sparse update cost and a capacity-wide resolve may outweigh saved rays in short-path scenes.

## Frame integration and ownership

Proposed sequence inside `RtxContext::dispatchIntegrate`:

```text
existing scene, atmosphere, volumetrics, G-buffer preparation
existing sparse-rendering setup, RTXDI and NEE-cache maintenance
existing direct integration and RTXDI gradient

if SHARC selected and supported:
    prepare frame constants; ensure buffers; process cache reset
    sparse SHARC Update: trace root continuations and cache indirect surfaces
    resource dependency: update -> resolve
    SHARC Resolve: process every cache entry
    resource dependency: resolve -> query
    existing indirect integration using SHARC Query variants
else:
    existing indirect integration

existing dispatchNEE: primary lobe output assembly and primary NEE
existing downstream demodulation, denoising and composition
```

The query pass uses the renderer's normal resolution and active-pixel mapping. Its cache-update sampling budget is separate from the branch's sparse-rendering budget. First establish correctness with sparse rendering disabled, then explicitly integrate its active-pixel behavior so two masks do not accidentally starve cache updates.

Do not place SHARC resolve in NRC's existing post-path-tracing training slot. NRC records queries and resolves later; SHARC queries read radiance immediately and need the resolved cache first. SHARC's intentional update resampling reads prior resolved data, then resolve publishes this frame's data for rendering.

Add `RtxSharc` as a device-owned object through `DxvkObjects::metaSharc()`, following `CommonDeviceObject`/`RtxPass` conventions. It owns buffers, support status, effective settings, prior camera position, initialization state, reset reasons, and statistics. Proposed methods are `prepareFrame`, `dispatchUpdate`, `dispatchResolve`, and binding helpers. Shader dispatch may remain in the indirect pass owner where that best preserves existing pipeline selection.

Use ordinary DXVK-managed compute/ray dispatches and `Rc<DxvkBuffer>`. There is no reason to copy NRC's external queue locking, SDK buffer enum, or inference/readback lifecycle. All constants for Update, Resolve, and Query must come from one frame snapshot.

## Estimator contract

### Update

Map one rotating/jittered sample per configurable screen tile across the complete view; start experiments at 5x5, approximately 4% of pixels. Dispatch with ceiling division and bounds checks. Use an independent RNG stream and full-resolution source pixel coordinates. Do not launch a reduced top-left viewport or reuse NRC's expanded training dimensions.

Initialize `SharcState` once per sampled path. Reconstruct the selected primary/root surface and launch its continuation using existing material sampling. The root selects the distribution of update rays; its camera-prefix throughput is not part of the local radiance cached at the next surface.

At each completed indirect surface hit:

1. Resolve actual geometry/material and finish incoming segment processing. Never insert an intermediate `continueResolving` event or a portal/render-target control surface.
2. Evaluate local secondary direct lighting with the existing light selection, shadowing, cloud-ground shadow, BRDF/PDF, and MIS rules. Keep this local term separate from camera-weighted accumulated radiance and incoming segment emission.
3. Call `SharcUpdateHit(parameters, state, hitData, localDirectLighting, random)`. Respect its false return, including allocation failure and supported cache resampling.
4. For continuation, call `SharcSetThroughput` with the new segment's BRDF/lobe/PDF weight, including any roulette survival compensation. Reset local accumulation as required; SHARC propagates contributions internally.
5. On an actual miss, pass terminal environment radiance through `SharcUpdateMiss` and terminate. Never invent a cache position for the sky.

Preserve transport accounting when extending beyond the initial opaque path: incoming segment transmittance must affect earlier SHARC sample weights, not dim the local lighting stored at the current surface. Incoming segment emission contributes to previous vertices, not to the current surface's outgoing radiance. The adapter must submit that terminal-style contribution to the existing state before inserting the current surface (the propagation operation in `SharcUpdateMiss` can serve this purpose without making the renderer terminate). Apply attenuation in the same ordering as the resolver's emission convention. Treat this as an explicit adapter contract, not a reset of arbitrary `PathState` fields. Complex transmission/SSS/portal transport remains outside the first validated subset.

Keep update payload state exclusive to `SHARC_UPDATE` variants. Do not enlarge all existing path payloads. Query needs eligibility state but not SHARC's propagation arrays. Benchmark register pressure and TraceRay payload size before enabling all traversal backends.

Suppress update writes to final radiance textures, ReSTIR reservoirs, sparse-rendering statistics whose meaning is rendered samples, and normal NEE-cache feedback/task slots. Continue reading the NEE cache and evaluating secondary lighting. Update and render must not overwrite one another's per-pixel tasks.

### Query

Use `surfaceInteraction.position` in stable world coordinates and the oriented geometric normal. The cache's default normal key stores only sign bits; it does not provide exact surface/material identity. Thin walls, nearby surfaces, and moving objects require explicit validation.

The first continuation hit is `bounceIteration == 1` on this branch and is eligible in principle. Primary/visibility surfaces, including replacement primary surfaces, are not. Gate queries on completed surface resolution, valid opaque diffuse material/transport, and adequate segment footprint. Skip delta transmission, sharp specular, SSS, emissive surfaces, and portal/render-target boundaries in the initial implementation. Maintain matching insertion/query eligibility where mixing incompatible surface classes would contaminate a cell.

Use `HashGridGetLevel` and `HashGridGetVoxelSize`. A conservative initial distance test is `segmentLength > sqrt(3) * voxelSize`. For later glossy support, use the lobe and roughness that launched the segment, not the roughness of the hit material. Upstream prose and sample differ by a factor of two in the cone-footprint formula; choose, document, and validate one convention rather than combining them accidentally.

Place the query after geometry-resolve attenuation, medium attenuation, and special traversal decisions, but before the current surface's secondary NEE and continuation. A successful query adds cached outgoing radiance through the existing throughput-weighted accumulation helper exactly once, then stops continuation. Do not also add that surface's direct light: it is already in cached outgoing radiance. On a failed lookup, execute the normal path tracer.

Preserve explicit incoming segment emission, the first indirect hit distance, lobe identity, and downstream output initialization. Set path termination state so control reaches normal `IndirectRadianceHitDistance` writes and `dispatchNEE` lobe distribution; an early return from the shader entry point would skip these outputs. Do not repurpose hit distance as voxel distance or cache age.

### Emission, demodulation, and sky

Use `SHARC_SEPARATE_EMISSIVE=1` as the proposed integration policy. Keep current-hit emission out of the cached local reflected-light term. Update must supply surface emission through `SharcHitData.emissive` so it reaches prior cached vertices; do not zero this update field because the rendered path has its own explicit emission accumulator. Begin from raw material radiance, never camera-prefix-weighted radiance. Unlike the analytic-light sample, Remix can also sample emissive triangles through NEE: the propagated BSDF-hit contribution needs the matching incoming-edge MIS weight when that competing estimator exists. That weight applies to propagation toward prior vertices, not to a current surface's intrinsic emission or to the whole cached outgoing-radiance estimate. Validate this separately with triangle NEE toggled to rule out double counting.

Query must evaluate emissive identity before deciding eligibility. Initially skip cache queries at emissive hits and run Remix's explicit MIS-weighted emission and normal continuation/termination. At eligible non-emissive hits, a successful query replaces secondary NEE and continuation. If later permitting emissive queries while retaining explicit emission, pass zero emission only in the Query hit data; Update still propagates it. Never apply emission both through SHARC reconstruction and through Remix's current emissive accumulator.

Material demodulation inside SHARC is separate from Remix's final denoiser demodulation. Start with it disabled for the estimator proof. Then add a clamped, documented albedo/reflectance factor supplied consistently to update and query; changing that convention invalidates the cache. Do not copy NRC's irradiance conventions into a radiance cache.

Keep cache radiance scene-linear and independent of display exposure. SHARC's `radianceScale` is integer accumulation quantization, not display pre-exposure. Upstream recommends starting at 1000; select the production value from measured accumulation range and firefly tests.

Update misses must use the same dome/sky/atmosphere code and `skyGatherEligible` policy as rendered paths. This branch intentionally gives diffuse sky gathers a different scale from reflections/refractions. An isotropic cache cannot encode arbitrary path-class differences; the initial diffuse-only subset avoids promising correctness for those mixed cases. Cache surface lighting, not screen-space cloud color or view-dependent aerial-perspective composition.

## GPU resources, synchronization, and capability gates

| Buffer | Default stride | Directional SH stride | Access |
| --- | ---: | ---: | --- |
| Hash entries | 8 B | 8 B | Update/Resolve read-write; Query read |
| Accumulation | 16 B | 32 B | Update accumulation; Resolve read-write |
| Resolved | 16 B | 24 B | Resolve read-write; Update resampling and Query read |
| Optional lock | 4 B | 4 B | Only non-compact key fallback without 64-bit atomics |

At 2^20, 2^21, and 2^22 entries, the default three buffers cost 40, 80, and 160 MiB respectively. Directional encoding at 2^22 costs 256 MiB. The lock fallback adds 16 MiB at that capacity. These totals exclude allocator overhead, profiling buffers, and transient overlap during replacement. Start with upstream's 2^22 reference configuration and measure 2^21 as a lower-memory option.

Zero all allocated cache buffers before first use and after hard reset. Ordinary resolve clears accumulation as it consumes it. Responsive lighting changes this rule: clear accumulation before every Update because responsive resolve does not clear it. Resolve covers every slot, for example `ceil(capacity / 256)` groups with a bounds guard.

Use DXVK resource tracking and verify the resulting dependencies: transfer clear -> shader accesses; Update ray-tracing or compute writes -> Resolve compute reads/writes; Resolve writes -> Query ray-tracing or compute reads; Query reads -> the next Update/Resolve writes. Include prior-frame resolved reads in the hazard analysis. Keep the passes on the same command stream initially. No in-place reuse across overlapping frames without dependencies; resource replacement uses normal deferred lifetime tracking.

The default 64-bit key path requires shader Int64 and storage-buffer 64-bit atomic compare/exchange. On the target branch, adapter setup enables supported fp16/storage16 features, but the reviewed code does not enable `shaderBufferInt64Atomics`; the D3D9 feature selection does not explicitly enable `shaderInt64`. Query supported features, enable required optional features at device creation, and gate SHARC on the enabled features, not GPU vendor alone. Do not make SHARC requirements mandatory for starting Remix.

The lock fallback avoids 64-bit atomics but retains 64-bit key arithmetic and adds contention/storage. Treat it as a separately validated fallback; do not silently substitute the compact 32-bit hash representation, which changes collision behavior.

All SHARC variants need native fp16 storage even with `SHARC_USE_FP16=0`. Remix compiles through Slang to SPIR-V with scalar block layout. Test the actual bundled Slang frontend, structured-buffer strides, fp16 types, and atomic emission; the guide's DXC `-enable-16bit-types` switch is not a universal Slang integration recipe. Keep SHARC macros consistent across Update, Resolve, Query, and debug shaders. Retain default shaders with no SHARC descriptors or propagation payload.

## History and changing scenes

Keep SHARC history decisions separate from screen-space denoiser reset. Resolution changes and ordinary camera motion need not clear a world-space cache; recreate sampling state without reallocating fixed-capacity cache buffers. Current and previous cache camera positions must advance exactly once per rendered update/resolve frame.

Hard-reset on scene replacement, coordinate-origin convention changes, capacity/layout/key changes, cache scale changes, demodulation changes, and explicit reset. A large teleport can conservatively clear or start a short warm-up policy. Ordinary camera cuts can retain valid spatial entries if geometry and coordinates are unchanged.

Use stable world coordinates; never hash camera-relative froxel coordinates. SHARC `sceneScale` is its own grid-density parameter, not `RtxOptions::sceneScale`. The existing Remix meter conversion is `100 * sceneScale`; grid voxel size also depends on distance-selected level and `levelBias`. Define a documented cache coordinate convention and expose effective voxel size in debug views. Do not inherit atmosphere's separately calibrated cloud/AP scale implicitly.

Sun/moon changes, cloud shadows, lightning, transient lights, emissive edits, and weather transitions change cached surface lighting. Feed a typed frame snapshot from atmosphere/weather/scene owners to `RtxSharc`. Use short accumulation or selective policy for continuous changes and a reset for abrupt large discontinuities; do not clear every frame because cloud time changed. Consume actual effective settings, not per-frame writes through `RTX_OPTION`.

Moving/deforming objects and disoccluded backgrounds remain a quality risk: the upstream key contains position/level/normal, not instance ID or motion history. Prototype query/insertion exclusions for known dynamic surfaces and measure stale lighting on surfaces they uncover. These exclusions cannot by themselves invalidate previously cached static surfaces. Local invalidation or more responsive history may be needed before broad scene support.

Responsive lighting is a later compile-time mode, not a cost-free per-light flag. Upstream treats a marked entry's signal as responsive and uses extra entries in the same table. It changes clear behavior and effective occupancy. Directional SH storage is another later mode for glossy leakage, not a replacement for eligibility and visibility checks.

## Source change map

All existing locations refer to target commit `fd4cc524`.

| Existing or proposed file | Responsibility |
| --- | --- |
| `src/dxvk/rtx_render/rtx_sharc.{h,cpp}` (new) | Device owner, buffers, settings, frame state, support, resolve, diagnostics. |
| `src/dxvk/dxvk_objects.{h,cpp}` | Common-object ownership and accessor; follow NV-DXVK wrapper rules. |
| `src/dxvk/rtx_render/rtx_context.cpp:1640` | `dispatchIntegrate`: order sparse Update/Resolve before indirect Query. |
| `src/dxvk/rtx_render/rtx_context.cpp:413` | Separate cache reset policy from existing global history reset. |
| `src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp:397` | Bind resources, dispatch purpose/resolution, select supported variants. |
| `src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh:213` | Vertex update/query adapters; local transport accounting. |
| `src/dxvk/shaders/rtx/algorithm/path_state.slangh` | Update-only propagation state and query eligibility information. |
| `src/dxvk/shaders/rtx/pass/integrate/integrate_indirect*` | Ray-query/raygen/closest-hit/miss variants and binding declarations. |
| `src/dxvk/shaders/rtx/pass/integrate/integrate_nee.comp.slang:279` | Existing NRC-only primary training patch: preserve it; SHARC does not consume it. |
| `src/dxvk/shaders/rtx/pass/sharc/` (new) | Shared args, bindings, resolve, debug shaders; upstream adapter outside vendor headers. |
| `src/dxvk/shaders/rtx/pass/raytrace_args.h` | Typed frame-local SHARC constants. |
| `src/dxvk/rtx_render/rtx_options.h:193` | Extend existing mode enum without renumbering current values. Feature tuning stays in `RtxSharc`. |
| `src/dxvk/imgui/dxvk_imgui.cpp` | Mode selector, support reason, cache settings, reset and counters. |
| `src/dxvk/dxvk_adapter.cpp:500`, `src/d3d9/d3d9_device.cpp:4343` | Optional device-feature enablement and reporting. |
| `src/dxvk/rtx_render/rtx_atmosphere.cpp:3496`, `rtx_weather.cpp:963` | Effective lighting/weather snapshot for temporal policy. |
| `src/dxvk/meson.build`, shader include configuration | Register host files and pinned dependency include path. |
| `docs/ShaderVariants.md`, `scripts-common/compile_shaders.py:232` | Follow existing compile-time variants and Slang/SPIR-V build. |

Append a new mode value rather than changing existing config serialization. Audit `#if !ENABLE_NRC` blocks: those often contain ReSTIR-specific assumptions and must distinguish SHARC as well. Enumerate supported combinations rather than multiplying NRC and SHARC axes into impossible variants. Begin with one traversal backend for proof, then cover the user's normal TraceRay path, ray-query path, and any SER variants advertised in the UI. Unsupported combinations must visibly fall back.

Proposed `rtx.sharc` settings: capacity preset, update tile size, grid scale and level bias, accumulation frames, stale frames, maximum update bounces, query eligibility threshold, and debug mode. Keep counters, current camera, effective radiance scale, reset reasons, and support status in runtime state. Use deferred option writes and regenerate option documentation when implemented.

Pin the dependency and preserve NVIDIA's license/notices. SHARC uses the NVIDIA RTX SDK license rather than Remix's MIT license; dependency packaging must preserve that distinction and follow its source/object distribution terms. No large RTXGI framework dependency is needed.

## Implementation milestones and evidence required

1. **Compiler and capabilities.** Compile minimal Update/Resolve/Query wrappers against the pin using bundled Slang; validate SPIR-V layouts and capabilities. Verify D3D9-created device feature enablement and supported/unsupported fallback. This settles compatibility before modifying path estimation.
2. **Cache mechanics.** Allocate, zero, insert known radiance, resolve, and query synthetic positions. Verify stride, empty/full table behavior, overflow failure, aging, prior-frame dependencies, and resource retirement. Keep rendering unchanged.
3. **Diffuse estimator proof.** Sparse updates from root continuations; cache only validated indirect diffuse surfaces. Add cache-debug output and compare with ordinary path tracing. Test constant environment, diffuse enclosure, colored bounce, analytic light, emissive triangle, thin wall, and empty cache.
4. **Rendering integration.** Enable query termination, preserve first-hit/lobe outputs, deferred NEE and denoiser inputs. Validate all supported traversal variants, NEE enabled/disabled, and secondary/replacement primary surfaces. Add shader variants rather than unconditional SHARC code.
5. **Branch behavior.** Exercise day/night and weather blends, moving cloud shadows, lightning, emissive animation, motion/disocclusion, camera cuts/teleports, resolution changes, toggles, game-scale differences, and sparse rendering. Expand material/transport support only with reference comparisons.
6. **Performance and optional modes.** Measure Update, Resolve, Query, total indirect and total frame time; report VRAM, occupancy/collisions, fallback/termination rate, rays/bounces, overflow, and image error. Sweep tile sizes and capacities. Evaluate material demodulation, responsive lighting and directional encoding separately.

Use high-sample uncached path tracing as the image reference, with identical lighting/material policy. Compare raw radiance before denoising as well as final images so denoisers cannot hide estimator errors. Check darkening after lights switch off, not only convergence after lights switch on. Synthetic propagation cases must catch double throughput, lost emission, and direct-light double counting.

Acceptance requires no validation errors, no SHARC resource/payload overhead in disabled shader variants, a documented fallback on unsupported settings/devices, bounded history behavior, and measured quality/performance tradeoffs. Set numerical image-error and time budgets from representative game captures rather than claiming a universal target before measurement.

### Compile-only evidence collected

Minimal compute wrappers for Update (`SharcInit`, `SharcUpdateHit`, `SharcSetThroughput`, `SharcUpdateMiss`), Resolve (`SharcResolveEntry`), and Query (`SharcGetCachedRadiance`) compiled successfully using Vulkan SDK Slang `2026.13.1-1-g84792eb15`. All three passed `spirv-val --target-env vulkan1.2 --scalar-block-layout` after using explicit Vulkan bindings and fully initialized grid parameters.

The probe used `HASH_GRID_COMPACT=0` and `HASH_GRID_ENABLE_64_BIT_ATOMICS=0`: ordinary 64-bit keys with the lock fallback, default non-directional storage, no responsive lighting or material/emissive extensions. Disassembly confirmed Int64, Float16, and UniformAndStorageBuffer16BitAccess capabilities, with 8-byte hash, 16-byte accumulation/resolved, and 4-byte lock strides. This demonstrates that current SHARC can compile through this Slang/SPIR-V route; it does not validate default 64-bit atomics, ray-tracing payloads, the proposed separate-emissive variant, or GPU execution.

Reproducible command shape, with paths supplied by the local checkout:

```text
slangc <pass>.slang -entry main -stage compute -target spirv -emit-spirv-directly -fvk-use-scalar-layout -I <pinned-sharc>/include -o <pass>.spv
spirv-val --target-env vulkan1.2 --scalar-block-layout <pass>.spv
```

The target worktree's expected `external/slang/slangc.exe` is absent. The smoke compiler came from `C:/VulkanSDK/1.4.357.0/Bin`, so testing the project's actual bundled compiler remains necessary. Temporary wrappers, logs, SPIR-V, and disassembly were retained in `%TEMP%/codex-sharc-smoke-20260911`.

Open gates: bundled compiler compatibility, enabled Int64 atomic support, payload/register cost, final grid calibration across games, dynamic-scene response, emissive/MIS correctness, and full-feature traversal parity. This design does not claim any of those have passed runtime validation.
