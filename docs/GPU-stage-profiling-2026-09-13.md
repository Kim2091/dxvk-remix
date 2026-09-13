# GPU stage profiling (2026-09-13)

## Observed baseline

Older DLL, user screenshots: lean update/resolve/query 0.31/0.02/0.82 ms (1.15 ms); full 0.83/0.02/1.62 ms (2.47 ms). Full total frame time reported 18.5 ms; earlier lean result 16.6 ms. Resource-specialization A/B reported identical performance. These are user snapshots, not averaged benchmark captures.

## Local diagnostic build

Temporary opt-in `rtx.profile.gpuStages` logs one sampled rendering sequence per 120 injections. Four query sets are reused only after results become available. The CPU never waits for query results; pending sets skip measurement. Each sequence has adjacent graphics-queue timestamp boundaries from scene preparation through final output/frame-generation dispatch. Logs identify original frame numbers; stages include G-buffer, RTXDI, direct/indirect integration, NEE, volumetrics/clouds, NRD, upscaling/RR and post-processing. No rendering algorithm or quality setting is changed by instrumentation itself.

Interpretation: durations are elapsed queue intervals, including stalls and overlap effects, not isolated shader busy-time or exclusive CPU/GPU costs. MeasuredSequence is the sum span of listed intervals and must not be added again. Game CPU, texture uploads preceding the first marker, frame-end housekeeping after the last marker, other queues and presentation are not fully covered. This is a first broad breakdown, not a full-system profiler. Empty/menu scenes are excluded by the summarizer. Intervals include adjacent setup commands and diagnostic screenshots if enabled. Only stable scene captures with no screenshot actions should be compared.

Validation: Release build passed; 66 current SHARC shader stages satisfy resource contracts and are embedded in the DLL. Factory-only smoke test succeeded and regenerated RtxOptions.md in an isolated process with fresh LOCALAPPDATA and the automation consent flag cleared. Log summarizer checks passed for complete, partial and invalid samples. Runtime measurement awaits a loaded game scene.

## Deployment

DLL SHA-256: 8E23B8295E39D8BEA243FEA2905BF16C490AE3A3F98BAFBFECF27CF3235C1753
PDB SHA-256: 818F803B3AEE8452A155A07F03F89C999AEF8737DBA7D51DCF1BD4ACB636C250

Installed to FNV .trex with verified DLL/PDB/config backups using suffix `.backup-pre-gpu-profile-20260913-015313`. Full SHARC selected (`leanSecondary=False`), separate SHARC timing/statistics disabled, stage logging enabled. Bridge binaries unchanged. Launched via nvse_loader.exe. This profiling binary also includes the prior local resource/scheduling specialization; it is not bit-identical to the older baseline build. No commits/pushes made.

Log: `C:/Users/sparkles/Projects/Games/Fallout New Vegas/rtx-remix/logs/remix-dxvk.log`

Summarize after a stable scene capture:

```powershell
python scripts-common/summarize_gpu_stages.py 'C:/Users/sparkles/Projects/Games/Fallout New Vegas/rtx-remix/logs/remix-dxvk.log' --last 20
```

To disable logging, set `rtx.profile.gpuStages=False` with the game closed. Restore backed-up rtx.conf for exact previous settings, and DLL/PDB together to return to the older binary. Keep this diagnostic separate from final performance comparisons.

## First capture and refinement

Last 12 completed sampled frames (2466..3786): measured sequence median 16.344 ms; frame preparation 3.671; upscaling/RR 2.887; indirect integration 2.677; cloud screen 2.016; G-buffer 0.988; volumetrics 0.882; RTXDI 0.765. Raw capture and summary preserved in _Comp64Release/gpu-stage-first-capture.log and gpu-stage-first-summary.txt. Frame preparation includes atmosphere.updateFrame/computeLuts; it is not pure allocation cost.

Refined timestamps separate frame resource callbacks, argument setup before atmosphere, atmosphere preparation, and argument setup afterward. Release build and 66-stage shader resource/embedding checks passed. Refined DLL deployed with verified backup suffix .backup-pre-profile-detail-20260913-015724; DLL A7474041990A2E61A32112F2BDD3B85C5DA376EE34BE6688844C1740E321E7F8; PDB 4D3049F3BB240C7FECFD0A079BC6ADE554778D4A01455626087EFBDA70E91D08. Config unchanged.

## Confirmed second capture

User confirmed the same scene was ready. Preserved raw log: `_Comp64Release/gpu-stage-second-capture.log`; summary: `_Comp64Release/gpu-stage-second-summary.txt`. Last 12 complete sampled frames: 4909..6229.

| Interval | Median ms | Min ms | Max ms |
|---|---:|---:|---:|
| MeasuredSequence | 16.183 | 15.234 | 17.581 |
| AtmospherePreparation | 3.811 | 3.467 | 4.826 |
| UpscalingOrRayReconstruction | 2.992 | 2.714 | 4.521 |
| IndirectIntegration | 2.409 | 2.097 | 3.480 |
| CloudScreen | 1.471 | 1.388 | 1.745 |
| Volumetrics | 1.229 | 0.911 | 1.987 |
| GBuffer | 1.040 | 1.019 | 1.229 |
| RTXDI | 0.735 | 0.686 | 1.067 |
| Denoising | 0.117 | 0.115 | 0.121 |
| FrameResourcePreparation | 0.002 | 0.002 | 0.002 |

The broad preparation cost is atmosphere GPU work, not resource callbacks. Individual stage medians are not an additive frame-time decomposition. The 16.183 ms measured span is not directly comparable to the user's 18.5 ms whole-frame display; instrumentation coverage differs and this is not an optimization A/B.

### Next optimization priority

Inspect atmosphere preparation before expanding the SHARC rewrite. Split its elapsed interval into sky/NVDF preparation, aerial-perspective light culling, visibility and integration, cloud sky transmittance, sun and ambient density grids, and secondary cloud LUT. Current evidence cannot assign the 3.811 ms to a specific one of those passes.

Source inspection confirms that sky LUTs already have dirty-key caching, aerial perspective rebuilds each frame, and cloud density grids deliberately update every frame when cloud ground shadows are enabled. Reducing update frequency or resolution would change quality and must not be presented as a free optimization. Existing debug skip switches freeze data and are diagnostic only.

Potential feature-preserving investigations after subpass timing: redundant atmosphere constant-buffer uploads and associated synchronization; prove dependency requirements before changing broad barriers; share density evaluation only where shader inputs and sample positions agree. None has a measured gain yet. The upscaling interval also includes exposure resource setup and RCAS sharpening, so it is not an exclusive DLSS/RR kernel measurement. Ordinary denoising is only 0.117 ms here, making NRD SH a poor first performance target for this capture.

No new rendering changes or deployment were made for this capture analysis. Profiling remains enabled in the installed diagnostic build, full SHARC remains selected, and no GitHub push was performed.

## Atmosphere subpass diagnostic

Added adjacent sampled timestamp boundaries for AtmosphereArgs, CloudShapeAndNvdf, SkyLuts, LightCull, FogSetup, FogVisibility, FogIntegration, CloudSkyTransmittance, CloudSunGrid, CloudAmbientGrid, CloudSecondaryLut, and Finish (all use the Atmosphere prefix in the log). These replace the broad AtmospherePreparation interval. Sum the atmosphere intervals within each frame before computing aggregate statistics; do not sum their medians. Optional skipped passes may omit markers, so each reported interval includes commands since the preceding marker. FogVisibility separates tracing only when the existing separate-visibility mode is active; otherwise tracing remains part of FogIntegration.

The context's existing query ring was expanded to 64 slots per sampled frame to accommodate the added boundaries. Subsystems use an explicit RtxContext reference to record timings, with query state still owned by RtxContext. Rendering shaders, sample counts, resolutions, update cadence, and barriers are unchanged. This is measurement infrastructure, not a performance optimization.

Atmosphere diagnostic Release build and 66-stage resource/embedding validation passed; git diff --check passed. DLL/PDB deployed with verified backups .backup-pre-atmosphere-profile-20260913-021206. DLL SHA256: 79AA93882179BE13714C6A27E36431C63928013AD9EA4765D11067FF6914BA6A. Config unchanged. Launched via NVSE. Runtime subpass capture pending.

## Atmosphere baseline and first layout experiment

User-confirmed baseline preserved in _Comp64Release/gpu-atmosphere-baseline.log and gpu-atmosphere-baseline-summary.txt. Twelve frames 2720..4054: secondary cloud LUT median 1.068 ms, fog visibility 1.006 ms, cloud sun grid 0.985 ms, ambient grid 0.323 ms, fog integration 0.171 ms. Summing atmosphere intervals per frame yields median 3.873917 ms (range 3.402394..5.045816). MeasuredSequence median 16.709 ms. The secondary LUT interval includes mip generation and dependencies.

First experiment: cloud_secondary_lut workgroup layout changed from 8x8 to 32x2, with matching CPU dispatch dimensions. Same 64 invocations per group and all 65536 texels updated each frame. Rationale: on 32-lane warps, keep each warp at one elevation to reduce march divergence. Wider azimuth coverage may hurt texture locality; performance is unproven until A/B. No shader-body calculations, resolutions, ray samples, jitter, barriers, or update rates changed. Source comparison against HEAD (explicit UTF-8 and normalized line endings) confirmed the only shader change is the workgroup declaration/comment; exhaustive coordinate coverage check passed for both layouts. Initial comparison attempts failed due to text decoding/line-ending handling, corrected without modifying shader logic.

Release build, 66-stage SHARC resource/DLL validation, and diff whitespace checks passed. Runtime visual comparison and timing remain pending. Config unchanged; profiling stays enabled. No commit or push.

Experiment deployed with verified DLL/PDB backups .backup-pre-cloud-lut-coherence-20260913-021716. DLL SHA256: 4AC55EBCBFC43DD9CE4E4AF6331A491296786F9DDD1A2827FD33E6267085A670.

## Cloud layout result and fog visibility experiment

User-confirmed cloud layout capture saved to _Comp64Release/gpu-cloud-layout-capture.log and gpu-cloud-layout-summary.txt. Last 12 frames 5185..6505: cloud secondary LUT 1.051 ms versus 1.068 baseline (0.017 ms, 1.6%). Ranges overlap; no convincing gain. Whole measured sequence 16.112 versus 16.709 ms cannot be attributed to this change: unmodified cloud screen and indirect intervals also moved substantially. Layout experiment reverted in source and the next build.

Next candidate: dispatch the existing AP_VISIBILITY_PASS over depth as well as screen coordinates. Each invocation traces one slice instead of the whole column. It replays preceding slices' depth-only arithmetic to retain the exact reference bounds, including comparisons on repeated/collapsed distances, without replaying their rays. Invalid camera rays clear only the invocation's output slice; depth bounds guard prevents out-of-range writes. Each output froxel has one writer. Cached integration remains sequential and unchanged because its light/transmittance accumulation has real dependencies.

Tradeoff: more independent ray work, but repeated setup and depth arithmetic. No ray-count, sample-position, resolution, cadence, or shadow-range reduction. The visibility tracing/sampling body matches baseline source; fused, cached-integration, and unshadowed SPIR-V hashes are unchanged. Release build, resource embedding validation for 66 SHARC stages, and diff whitespace checks passed. GPU output equality and performance remain unverified pending runtime capture. No commit/push; profiling stays enabled, game configuration unchanged.

Fog parallel experiment deployed with verified DLL/PDB backups .backup-pre-fog-parallel-20260913-022145. DLL SHA256: D68491B26CA4B7F2299182A73713D1C0447C865876DD2E14AA74EF4CF64FE08B.

## Parallel fog visibility result

User-confirmed capture preserved in _Comp64Release/gpu-fog-parallel-capture.log and gpu-fog-parallel-summary.txt. Last 12 frames 23147..24467: fog visibility median 0.977 ms (0.949..1.339), original baseline 1.006 ms (0.970..1.607). Difference 0.029 ms / 2.9%, with overlapping ranges and changes in unmodified stages. Not a demonstrated optimization. Whole measured sequence 15.766 ms is not evidence that this patch saved the difference from baseline.

Reverted only the parallel-fog shader and dispatch changes. Verified shader source restored to HEAD; atmosphere profiler retained. Restored the original atmosphere diagnostic DLL/PDB from .backup-pre-cloud-lut-coherence-20260913-021716, with hash verification. Tested candidate preserved as .backup-fog-parallel-tested-20260913-022621. Game was closed and was not relaunched. Configuration unchanged; diagnostic profiling still enabled. The _Comp64Release outputs still contain the last candidate until rebuilt; do not redeploy them as baseline.

### Concrete next investigation

The sun and ambient density-grid shaders both evaluate cloud type and coverage via cloudControlField2D at each voxel's XZ position. cloudVoxelUVWToWorld computes X and Z independently of vertical coordinate, so the same column repeats those control-field evaluations through all 32 vertical voxels, in both passes. This is actual repeated arithmetic, unlike workgroup scheduling. Candidate: share exact per-column control values within a group or precompute an unfiltered full-precision 2D control grid for both passes. Verify identical input coordinates/args, preserve all optical-depth sample positions and full-rate updates, and compare output before claiming quality preservation. Synchronization, extra memory traffic, and existing compiler optimization can erase any savings; 32 repetitions does not imply a 32x speedup for the grid. No implementation or measured gain yet.

Prefer investigating this reuse before another user-facing scheduling A/B. Neither scheduling experiment earned retention. No commit or GitHub push.

## Shared cloud controls experiment

Both sun and ambient density-grid shaders retain their 8x8x4 workgroups and original dispatch dimensions. Each group's 32 XZ columns has one Y=0 lane compute the full-precision cloud type/coverage pair, stored in a float2 groupshared array (256 bytes total). All eight vertical lanes reuse it after GroupMemoryBarrierWithGroupSync. Bounds return occurs after the barrier so every invocation participates, including hypothetical partial edge groups. No subgroup-size assumptions, new bindings, textures, sampling changes, or update-cadence changes.

Extracted the original type/coverage expressions verbatim into computeCloudBakeControls. Optical-depth helpers now receive that pair; all integration code is preserved. This reduces control evaluation from 256 to 32 per full workgroup, but not all shading cost by eight. Previously rejected sun/ambient rays could skip control evaluation; now producer lanes compute it before the ray rejection, so unfavorable scenes can regress. Shared-memory synchronization is another cost. Runtime timing and image equivalence remain to be tested.

Validation: Release build passed. Source reconstruction verified that replacing the passed controls with the original expressions restores the entire shared header to baseline, and helper arithmetic matches the extracted original. Exhaustive workgroup index check verified exactly one writer and eight matching-XZ readers per column, and no return before synchronization. Cloud render, cloud secondary LUT, cached fog integration and unshadowed fog SPIR-V hashes are unchanged. 66 SHARC stage resource/embedding validation and diff whitespace checks passed. Prior two scheduling experiments remain reverted. No commit/push; config unchanged, profiling enabled.

Shared-control candidate deployed with verified DLL/PDB backups .backup-pre-cloud-controls-20260913-023135. DLL SHA256: 8219DE7F8B803AF082E2B1CCC167739DFF9A76D93204C437003B99228392535D. Launched via NVSE; capture pending.

## Shared cloud controls result

User-confirmed capture preserved in _Comp64Release/gpu-cloud-controls-capture.log and gpu-cloud-controls-summary.txt. Last 12 frames 3292..4612: sun grid median 0.935 ms versus baseline 0.985; ambient grid 0.403 versus 0.323. Per-frame sum of the two grids: median 1.342974 ms versus baseline 1.308193 ms, ranges 1.192960..2.800672 versus 1.144832..2.659324. No demonstrated net benefit. Separate-run variation prevents attributing the small difference confidently; unchanged passes also varied.

Saved source experiment patch to _Comp64Release/cloud-controls-experiment.patch. Reversed only the shared-control edits and verified all three affected shader files match HEAD. Restored baseline DLL/PDB from .backup-pre-cloud-controls-20260913-023135, with hash checks and tested-candidate backup .backup-cloud-controls-tested-20260913-023454. Game was closed and not relaunched. Profiling remains enabled; settings unchanged. Build outputs still contain the rejected candidate until rebuilt; deployed binaries are the restored baseline. No commit/push.

Three small atmosphere experiments have not earned retention. Before further small candidates, prefer same-session paired baseline/candidate sampling with shader variants, warm both paths and report paired per-pass deltas. This reduces restart drift but does not eliminate evolving weather, GPU clocks, overlap effects or the need for image comparison. More separated launches with tiny deltas are not a sound basis for claiming progress. No additional capture requested now.

## Main pipeline: plain indirect assembly variant

Dependency review confirms that IndirectRadianceHitDistance is also consumed by secondary demodulation; it cannot simply be removed after SHARC query. integrate_nee also samples emissive triangle lighting, performs visibility/MIS, feeds cache tasks and writes primary diffuse/specular signals. Removing the assembly pass would remove functionality.

Concrete missing specialization: dispatchNEE previously selected the combined NRC + ReSTIR GI shader when neither backend was active. Added integrate_nee_plain (ENABLE_NRC=0; no ReSTIR GI define), prewarmed and selected only when both effective backend states are false. This covers SHARC and ordinary integration. Both-active and single-active routes retain their original variants. No shader body, pass order, output format, shared allocation, rendering settings, or material effects changed. CPU binding helpers and managed descriptor layout are still shared; no binding/allocation savings are claimed.

Release build and diff whitespace checks passed. All three original NEE variants retain their SPIR-V hashes. Plain variant omits NRC training / ReSTIR reservoir / BSDF factor bindings 71,82,83; retains output bindings 80,81; verified embedded in DLL. SPIR-V size 298456 bytes versus combined 333340 bytes (10.47% smaller), not a performance result. Existing 66-stage SHARC resource checks passed. Game image/mode-switch validation remains pending. Profiling stays enabled; separate-run small timing changes must not be claimed as wins. No commit/push.

Background reviews are read-only and deferred while main pipeline work is prioritized. Cloud candidates: fuse sun/ambient bakes without group-shared synchronization; hoist frame-uniform cloud shading context; inspect compiler elimination of unused secondary-LUT depth bookkeeping; verify unused final mip. Atmosphere candidates: check authored versus weather-resolved volumetrics/AP handoff; skip local-light LUT writes when there are no consumers; prove identical args before consolidating uploads. All require independent verification. In particular the handoff finding is a potential correctness issue, not a proven optimization; cloud sun integration uses its existing adaptive/near/far sampling and must not be reduced to a nominal tap count.

Plain NEE candidate deployed with verified backup .backup-pre-nee-plain-20260913-024741. DLL SHA256: 5FBF737CC5B8DDCD14CD90BEF766FE9885F7B9FF92090E1307D73D4482AC1A7F. Launched via NVSE.

## Plain assembly variant runtime capture

User reported ready; no explicit visual-comparison result supplied. Preserved _Comp64Release/gpu-nee-plain-capture.log and gpu-nee-plain-summary.txt. Last 12 frames 3126..4446: IndirectAssembly median 0.337 ms (0.317..0.517); MeasuredSequence 15.869 ms (15.188..16.881). Original second broad capture assembly median was 0.332 ms, original detailed capture 0.430 ms, and subsequent unrelated-candidate runs ranged from 0.306 to 0.453 ms. Thus selecting only the 0.430 ms reference would overstate the evidence. No demonstrated assembly speedup; no causal whole-frame speedup claim.

Retain the explicit neither-backend specialization as a structural correction with smaller compiled code, not a measured optimization. Installed DLL and configuration unchanged after capture; profiling remains enabled. Runtime timing was collected, but image equivalence and live backend-transition validation are still pending. No new launch or user capture requested. Before evaluating more small variants, implement paired same-session measurements with both paths warmed and no rendering-settings changes, rather than relying on separate launches. No commit/push.
