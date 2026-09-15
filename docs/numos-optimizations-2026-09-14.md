# Numos cloud optimizations - 2026-09-14

Base commit: 3c3d2a1ce. Changes are uncommitted.

## Implemented

1. Sun and ambient density-grid bakes share type/coverage controls across each workgroup's eight vertical lanes. XZ positions are independent of the vertical index, even though the world-space Y coordinate follows the curved shell. Each 8x8x4 group evaluates 32 columns instead of 256 voxels: 87.5% fewer control-field evaluations, not an 87.5% reduction in total bake time. Storage is 256 bytes per group with one workgroup barrier. Padding lanes reach the barrier before returning. Grid dimensions, dispatch dimensions, sample locations, integration counts, and texture bindings remain unchanged.
2. Both view-march layers bypass type/coverage noise when the corresponding spread is exactly zero. The second primary slab crossing is skipped once transmittance is below the existing exit threshold; previously its sample loop immediately exited after paying setup costs. Sample budgets, extinction, detail/shadow quality, and empty-space stepping are unchanged. Compiler specialization may already remove some zero-spread work for the echo layer, so incremental savings there may be small.
3. Depth-aware limits were already implemented in the current source. cloud_render passes the primary surface's slant distance in km to marchCloudLayers; both marchCloudSlab and marchEchoDeck clip to it. loadValidatedCloudHistoryTap rejects mismatched surface depths. No additional depth-clamp change was needed. The earlier scout's claim that the screen pass supplied zero clamps was stale.

## Validation

- Release Meson compilation and subsequent relink succeeded. The second compile was necessary because shader generation updates headers after the first dependency scan.
- All four affected SPIR-V modules passed spirv-val with target-env vulkan1.3: cloud_sun_density_grid, cloud_ambient_density_grid, cloud_render, cloud_secondary_lut.
- Disassembly confirms an 8x8x4 workgroup, a 32-element float2 Workgroup array, and an OpControlBarrier with Workgroup scope and acquire/release shared-memory semantics.
- All four generated SPIR-V byte sequences were found exactly in _Comp64Release/src/d3d9/d3d9.dll.
- DLL SHA256: ac70f9a3cbac639efa46d2985b79866b6c554a06b5853b43986c64ca427de46e
- git diff --check passed.
- Build emitted a nonfatal vswhere PATH diagnostic and external-library missing-PDB warnings, but MSVC activation, compilation, and linking succeeded.

## Runtime validation still needed

No game deployment or GPU benchmark was performed. Use the existing AtmosphereCloudSunGrid / AtmosphereCloudAmbientGrid timing stages and cloud-render GPU zone. Compare the baseline and new build with identical camera, weather, resolution, and sample settings. Include noon, sunset, moving camera/foreground silhouettes, inside-cloud views, and layer-2-only views. Include zero and nonzero type/coverage spreads. Record bake-frame and non-bake-frame timings separately and compare total frame time. Shared-memory synchronization and branch costs may offset some arithmetic savings; no measured speedup or visual equivalence is claimed yet.

## Profiling follow-up (deployed 2026-09-14)

User measured approximately 2.5 ms at 96 samples with no noticeable visual or timing change from the initial optimization.

Added Cloud Profiling beside Max Cloud Samples in the atmosphere UI, backed by rtx.atmosphere.cloudProfilingMode (default 0):
- 0 Normal: production shader, verified byte-identical to the previously deployed module.
- 1 No moon shadows: removes the live moon-shadow density marches in both layers, retaining unshadowed moon lighting and the normal view density. Expected to have little or no impact in daylight.
- 2 Density only (unlit): removes lighting and lightning in both layers, retaining view density/detail evaluation, density gates, attenuation, aerial factors, depth accumulation, adaptive stepping, SDF skipping, and opacity termination. White unlit output is intentional.

These are separate compile-time shader variants, selected only for the cloud screen pass. They do not modify voxel-grid or secondary-LUT quality. Mode changes can briefly blend prior lighting through temporal history; wait several seconds and let pipeline compilation settle before reading GPU pass timings. Compare with identical camera, weather, sample settings, and render scale. Timing differences include compiler scheduling/register-pressure effects; they are not additive exact per-instruction costs.

Release build succeeded. Four affected SPIR-V modules passed Vulkan 1.3 validation and were verified embedded byte-for-byte in the DLL. git diff --check passed. Deployed matching DLL/PDB to Fallout New Vegas/.trex with hash verification. Backup suffix: .backup-pre-numos-profiling-20260914-160238. DLL SHA256: c5e5d87c2928aa4ffc6bb90776acac695138d74fdc9d6f199c8bc94416807d37.

Next measurement: Normal versus Density only at the user's 96-sample scene. A large reduction points toward lighting; little reduction points toward density/detail sampling and traversal. Use No moon shadows to separate lunar shadow cost in night scenes. No diagnostic runtime timing has been collected yet.

## Automatic log capture correction

The user's first profiling test was logged by the existing rtx.profile.gpuStages=True setting, but every entry was labelled CloudScreen without its diagnostic mode. Preserved that capture at C:/Users/sparkles/Projects/vapourkit/temp/numos-unlabelled-capture-20260914.log. There are 31 CloudScreen records from 16:25:39 to 16:26:35, mostly 1.05-1.42 ms, with a 2.02 ms spike. User kept the camera still, enabled density-only, disabled it, changed to night and enabled a moon, then enabled the moon-shadow diagnostic. Without exact mode transition frames and with changed lighting, do not treat the chronological clusters as a controlled per-mode benchmark.

Added rtx.atmosphere.cloudProfilingLog and Log Cloud Timings in the UI. Changing Cloud Profiling automatically turns this on, including when returning to Normal. Uses the existing asynchronous timestamp sampler every 120 rendered frames. Each Cloud profile record includes mode, base samples, maximum samples, frame ID and milliseconds. Metadata is captured with the submitted frame and retained until the timestamp queries become available, rather than reading the current mode at readback. Existing GPU-stage logging still works; cloud-only logging does not require enabling the full-stage log. Turn Log Cloud Timings off after testing (and disable rtx.profile.gpuStages separately if that was already enabled).

Next capture: keep camera, sample settings and lighting fixed for Normal -> DensityOnly -> Normal, at least 10 seconds each. For lunar cost use a separate fixed night scene and Normal -> NoMoonShadows -> Normal. After the user says done, read rtx-remix/logs/remix-dxvk.log directly and summarize stable per-mode records; do not ask the user to transcribe timings.

Automatic logging build compiled successfully and deployed with matching PDB and verified hashes. DLL SHA256: 2d86cf6010fdb27d131338d2495f71b232ee683571535b51925d21cca239351d. Backup suffix: .backup-pre-numos-logging-20260914-163155. Profiling shaders and log marker verified embedded. Runtime mode-labelled capture pending.

## Labelled capture results and workgroup experiment

Preserved the 16:39:52-16:40:49 capture at C:/Users/sparkles/Projects/vapourkit/temp/numos-labelled-capture-20260914.log. All records used base samples=32 and maxSamples=96. Medians (all samples, no outlier filtering): Normal 1.475580 ms, n=16; DensityOnly 1.219585 ms, n=6; NoMoonShadows 1.432580 ms, n=9. Normal ranged 1.0025-2.53235 ms; DensityOnly was tightly clustered at 1.21037-1.23187 ms; NoMoonShadows ranged 1.40083-2.53386 ms. Removing lighting saved about 0.256 ms / 17.35%; removing moon shadow marches about 0.043 ms / 2.91%. These are indicative, not exact additive component costs: shader scheduling changes and lighting transitions/frame spikes confound precision. Density, detail sampling, and traversal account for most of the remaining work.

The active erosion and micro-AO settings use the detail taps, so removing them changes appearance. Added two full-quality layout experiments to the existing profiling dropdown: mode 3 Full quality (16x4 threads), mode 4 Full quality (8x4 threads). Normal remains 8x8. Host dispatch dimensions match each variant, including ceil division and the existing per-pixel bounds guard. These affect only the cloud screen pass and keep sample counts, quality and shader arithmetic unchanged. Automatic log labels are FullQuality16x4 / FullQuality8x4. Both modules passed spirv-val Vulkan 1.3; a binary instruction comparison verifies they are identical to Normal after normalising only OpExecutionMode LocalSize. No speedup is claimed until measured.

Next test: same view, weather and sample settings, Normal -> Full quality (16x4 threads) -> Full quality (8x4 threads) -> Normal. At least 10 seconds each after initial pipeline compilation. User need only say done; read the labelled log directly and compare medians and spread.

Workgroup experiment built and deployed with matching PDB, backups and hash verification. DLL SHA256: 4d6a93ca222ee062fc58e5049db7177cc7e54000d58b4c04a32b8b8406c0454c. Backup suffix: .backup-pre-numos-workgroups-20260914-164442. All five cloud variants verified embedded.

## Workgroup capture result (16:45:22-16:46:12)

Preserved log: C:/Users/sparkles/Projects/vapourkit/temp/numos-workgroups-capture-20260914.log. Recorded order was Normal (10 records), FullQuality8x4 (5), FullQuality16x4 (4), then DensityOnly (7); there was no closing Normal segment. All records: samples=32, maxSamples=96.

Medians: Normal 1.20320 ms; FullQuality8x4 1.31379 ms; FullQuality16x4 1.33427 ms; DensityOnly 1.11002 ms. Means: 1.36520, 1.38547, 1.33632, 1.13020 ms respectively. Normal and 16x4 contain distinct roughly 1.2 and 1.47 ms timing clusters; the small 16x4 sample count puts its median between them. Do not interpret a small mean change as a speedup. Neither full-quality workgroup variant establishes an improvement. Keep the production default 8x8 (Normal); no default or quality change was made. The capture ended in DensityOnly, so the user should return the dropdown to Normal.

The inconsistent baselines across captures and bimodal readings limit precise component attribution. DensityOnly is still lower, but the earlier 17% figure is specific to the earlier capture, not a universal lighting percentage. Further work should target density/detail sampling rather than workgroup shape; avoid more layout variants without GPU profiler evidence.

## Density sampler optimization prepared

Added optional compile-time CLOUD_TIGHT_DENSITY_BOUNDS variants: mode 5 Full quality (tighter density bounds), mode 6 Density only (tighter bounds). The current sampler bounds every saturated noise signal by +/-0.5, but its billowy displacement is bounded by +/-0.15. After the existing global early-out, compute the existing local typeShaped and use lerp(0.5,0.15,typeShaped) to reject samples before mid/detail reads. Tighten the post-mid gate using the same bound for the remaining 0.6 base contribution. A relative float margin protects cancellation near the boundary. Both original rejection checks remain in place. The conservative sdfStepKmOut is unchanged: a local type bound cannot safely be used to step through positions with different type/height.

No authored detail, erosion, opacity, sampling budget, shading or texture content was reduced. The expected gain is less work on provably empty samples; warp divergence and extra bound arithmetic can offset the savings. This is an unmeasured experiment, not the new production default. Other shader consumers and Normal compile with the feature disabled.

Validation: scripts-common/validate_numos_density_bounds.py checks 37,680 float32 cases (texture-channel extrema and random cases) against the original displacement; no newly rejected sample could contain material. Both new modules passed spirv-val with Vulkan 1.3. Normal is byte-identical to the deployed Normal shader. Final DLL embedding/deployment verification follows the release link.

Single-session user test after deployment: keep camera, lighting/weather and sample settings fixed; Cloud Profiling Normal -> Full quality (tighter density bounds) -> Normal, 20 seconds each after compilation settles. Inspect for disappearing cloud edges while switching. Logs automatically identify the modes. Return to Normal, leave FNV running, and say done. Mode 6 is already in the same build if a density-only follow-up is useful, so it needs no additional deployment. Do not ask the user to close FNV after a measurement; closure is needed only when an already-validated replacement is ready to deploy.

Density-bound build deployed while FNV and bridge were absent. Matching PDB and DLL hashes verified, all seven shader blobs embedded. DLL SHA256: 32cd6fd04792424caabfed66fc1194e89f68725a5f069b261089f03f326bafdb. Backup suffix: .backup-pre-numos-density-bounds-20260914-184918.

## Tighter density bounds capture result

Preserved capture: C:/Users/sparkles/Projects/vapourkit/temp/numos-density-bounds-capture-20260914.log. All records use samples=32, maxSamples=96. Normal before (frames 8052-10239): n=19, median 1.251330 ms, mean 1.393258 ms. TightDensityBounds (10376-11816): n=13, median 1.203200 ms, mean 1.338978 ms. Normal after (11936-13136): n=11, median 1.230850 ms, mean 1.259705 ms.

Combined Normal: n=30, median 1.235455 ms, mean 1.344289 ms. Tight bounds lowers the combined median by 0.032255 ms (2.61%), but the mean by only 0.005311 ms (0.40%). Relative to the closing Normal segment, the optimized median is 0.027650 ms lower but its mean is 0.079273 ms higher. Timing clusters and spikes prevent claiming a reliable improvement. No runtime visual-equivalence feedback was provided. Keep Normal as the production default; no further user test or restart is needed for this result.

Source inspection confirms both primary and echo fixed-step marches already skip empty samples using the sampler's conservative radius; primary also has adaptive traversal. Merely adding empty-space skipping would duplicate existing work. The tighter local bound intentionally leaves that radius unchanged, so it reduces some detail reads without reducing march iterations. A stronger next investigation is a conservative bound valid across the ray interval (including coverage/type variation and lattice boundaries), or a coarse occupancy hierarchy that can skip whole empty intervals. Before another deployment, establish conservativeness and measure skipped iterations/detail reads in a batched diagnostic build; the present sparse GPU timing alone cannot distinguish sample-count savings from divergence or GPU scheduling noise. Do not increase the current local-type radius without proving it remains valid at subsequent positions.

## Current-frame cloud rendering test package (2026-09-14)

User abandoned the density-bound optimization investigation and requested cloud controls/presets, compositing fixes, smoothing removal, and reflection optimization. First testable batch implements render-size bookkeeping correction and removal of cloud temporal accumulation. Other requested features remain pending.

ensureCloudRenderRT now records scaledExtent rather than downscaleExtent. This fixes repeated allocations below native resolution and the erroneous reuse of a small texture when returning to scale 1.0. Each allocation logs [Cloud render] extent=WxH internal=WxH scale=S, enabling live verification of actual dimensions.

Removed composite cloud EMA and its reconstruction/clipping helpers, history texture allocation/swap, composite/debug bindings, and smoothing/clamp GUI controls. Retired history debug view 883 returns black if selected through an old config and is removed from the GUI list. Old historyWeight/historyClampGamma config keys remain inert for compatibility; cloudHistoryDepthTolerance still controls spatial upsampling, so it remains. Current-frame cloud depth and surface-aware spatial upsampling are retained. No material-depth, layer-order, slab-lighting, preset/seed, or reflection changes are claimed in this batch.

Validation: release Meson build and explicit successful incremental recheck; composite, debug_view, and debug_view_using_optional_extensions SPIR-V passed Vulkan 1.3 validation with scalar-block-layout (enabled by the renderer on supported hardware). Their compiled bytes were verified embedded in the DLL. git diff --check passed. Initial validation without the layout flag rejected the existing CompositeArgs layout; validation with the engine feature passed.

Deployed DLL and matching PDB while game/bridge were absent; backed up and verified hashes. DLL SHA256 949E0BC226719E08E1215A45F95C5B83BF55017381F3F7C2C44CA6A96754E15F. Backup suffix .backup-pre-numos-current-frame-20260914-202336.

User test: launch FNV, Cloud Profiling Normal; same outdoor view with foreground geometry, set render scale 1.0 -> 0.5 -> 0.25 -> 1.0, wait about five seconds each. Pan/walk at final 1.0 to inspect smudging, then at 0.5 inspect foreground silhouettes. Report visual issues; say done and leave game running. Read live log for actual extents; no shutdown required after testing. Smoothing controls should be absent. Look for restored native detail on returning to 1.0 and any newly exposed jitter without temporal accumulation.

## Reduced-scale silhouette follow-up

User confirmed the current-frame package was a major fix, with stair stepping remaining on geometry/cloud boundaries. Runtime resize logs confirm 1280x800 -> 320x200 -> 1280x800 recovery; capture ended at scale 0.25. Whether native scale also exhibits the issue is not yet confirmed.

Checkpoint f8544fa82 records the previously validated work. User now requests local commits for changes, never push.

The reduced-scale composite previously reconstructed color with depth-compatible taps but read entry/mean depth from an unrelated nearest texel. That can give sky cloud color foreground/no-cloud depth, changing aerial in-scatter and alpha-surface visibility at the coarse texel boundary. Reconstruction now keeps color and cloud depth together: entry is the minimum contributing cloud entry, mean depth is opacity-weighted over the same taps. If the four bilinear taps contain no matching surface, search their outer 4x4 ring; explicitly separate sky from geometry. A foreground surface before the fallback cloud entry gets no cloud. Native scale retains exact texel loads.

This is a targeted reconstruction fix, not proof of the screenshot's complete cause. A reduced-resolution texture cannot reconstruct arbitrary thin surfaces absent from every tap; unmatched fallback and surfaces inside a cloud remain approximate. No temporal filtering added. Validation: composite SPIR-V passes Vulkan 1.3 scalar-block-layout validation; release shader build succeeded; relink required to embed fresh generated header. Runtime test should compare the same ridge at 1.0, 0.5, and 0.25 while stationary and moving, and check for halos or missing clouds.

## Reduced-scale shimmer follow-up

User reports shimmer across clouds while walking, especially scale 0.25, not only at geometry silhouettes. The screen-pass comment claiming static far-field jitter was stale: adaptive marching uses nubis3JitterAnimateKm and the code deliberately animated offsets expecting cloud EMA/DLSS reconstruction. With the cloud EMA removed, coarse pixels can magnify this noise.

At reduced cloud resolution, use the existing static per-pixel march offset for both jitter inputs. Keep primary-camera projection jitter and all density, sampling budgets, and native-scale behavior unchanged. This removes frame-index noise from the reduced-resolution march, but cannot guarantee stability while walking: screen-space static noise can still move over the volume, and coarse spatial undersampling remains. Watch for fixed grain/banding replacing shimmer. This is a targeted test candidate, not a claim of runtime success.

All seven cloud-render SPIR-V variants passed Vulkan 1.3 validation; release relink includes these shaders together with the prior silhouette reconstruction fix. Local commits only; no push.

## Alpha foliage / atmosphere overlay test package

User confirmed the silhouette and stable-sampling packages fixed the main issues. New screenshots show foliage fading under cloud/haze while solid trunks remain visible. Source inspection found the geometry resolver attenuates background sky/radiance by accumulated foreground transparency (PrimaryAttenuation), but the later cloud and aerial in-scatter additions ignored that throughput. Added radiance now carries PrimaryAttenuation, including local aerial light contributions; the final stochastic alpha coverage multiplication remains separate.

The separate stochastic alpha layer also lacked aerial perspective entirely. It now samples aerial perspective at its own hit distance, using its existing near-volume attenuation and premultiplied coverage for in-scatter. No material alpha mode, texture opacity, temporal smoothing, or depth routing changed.

Validation: composite shader compiled and passed SPIR-V Vulkan 1.3 scalar-block-layout validation; diff whitespace check passed. Release relink and embedded-shader verification precede deployment. Runtime confirmation remains pending. This fixes a demonstrated additive-light weighting mismatch; it does not establish that every transparent-material path or foreground emissive contribution is correctly ordered. Inspect the pictured foliage against both sky/clouds and distant terrain, plus distant foliage that should still receive haze.


## Transparent foreground depth ordering

The additive-light patch did not resolve the reported foliage overlay. The user reproduced it with Ray Reconstruction disabled and with stochastic alpha blending disabled. In debug view 880, trunks/branches appear as foreground depth but foliage does not. That establishes that the atmosphere's primary depth is insufficient for this view; it does not by itself prove the exact material classification.

Source inspection identifies another incorrect operation: transparent/emissive approximations accumulate into SharedRadiance before the background sky is added, and the composite subsequently attenuates that entire sum by the background cloud/aerial transmittance. The prior throughput fix only weighted new light, leaving this extinction mismatch intact.

Add an RGBA16F foreground resource at internal resolution (8 bytes/pixel). The primary resolver records accumulated foreground radiance before sky/final opaque emission and tracks the nearest contributing transparency distance in kilometres. Unordered traversal returns the nearest contributor in a register, avoiding per-intersection image access. Fully transparent, non-emitting texels do not establish foreground depth. Every primary pixel initializes the resource, including non-Numos views. The existing reflected/refracted PSR path is excluded from the new composite separation.

Composite removes the recorded foreground contribution from SharedRadiance, applies background atmosphere to the remainder, applies aerial perspective/cloud visibility at the foreground distance, and recombines them. Foreground in-scatter uses the coverage complementary to PrimaryAttenuation. Existing RR particle emission stays on its current path and is excluded from the recorded shared foreground, avoiding re-adding it. No material classification, primary geometry depth, cloud reconstruction, cloud jitter, or temporal smoothing changes.

This remains a test candidate. A single nearest depth approximates multiple transparent layers: widely separated overlapping particles/foliage can be under-fogged, and PSR transparency is not addressed here. Runtime confirmation is required; shader compilation alone cannot establish that this is the material path used by the pictured foliage. Test the original nearby foliage against clouds and hazy terrain, and check more distant foliage still receives haze. Disable debug view 880 for the visual test. Keep the current RR setting for the first comparison; no additional toggle sequence is required.

Validation: release Meson compilation succeeded. All 145 affected SPIR-V modules passed Vulkan 1.3 validation with scalar block layout. An initial embedding check caught a stale generated header; the final relink contains the exact bytes of all 145 validated modules. Whitespace checks passed. DLL SHA256: 624b06a4dab3a16d8885b58fa19f5081823d5a0296ea6eb00ee0c2d8af7f3a89. Matching PDB SHA256: 68f4a16e0b98d9f7195a3f3e3a40d88e212169ff3c9c5b9f9723486b39a3c0df. Runtime visual confirmation is pending.
