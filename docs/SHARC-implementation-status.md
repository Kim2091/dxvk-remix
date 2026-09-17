# SHARC experimental integration — 2026-09-12

Implemented in the `wsn3g` worktree on `revised-9-10`. Initial integration is committed/pushed as `061f13bb7`; subsequent compatibility and performance changes are uncommitted. The original requested `dxvk-remix` checkout is a different branch; its unrelated changes were preserved.

## Use and implemented scope

Select **SHARC** in the indirect integration mode UI, or set `rtx.integrateIndirectMode = 3`. Existing defaults are unchanged. Settings are under `rtx.sharc`: `capacityLog2` (21 = 80 MiB), `updateTileSize` (5), `updateBounces` (8), `accumulationFrames` (8), `staleFrames` (32), `gridScale` (50), and `minRoughness` (0.8). The settings UI reports support/fallback status and offers Reset SHARC.

The device-owned cache uses pinned SHARC 1.8.3 (`4e21b585c33c83d723ca9a1e11bbb1090d145793`), 64-bit atomic hashes, and native half-float resolved storage. Required features include shader Int64, buffer Int64 atomics, float16/storage16, and RayQuery. Unsupported devices trace ordinary paths. WBOIT, ray portals, and opacity micromaps fall back by default. The SHARC settings expose independent compatibility overrides: `rtx.sharc.allowWboit`, `rtx.sharc.allowRayPortals`, and `rtx.sharc.allowOpacityMicromap` (all false by default). These bypass the corresponding fallback without enabling the underlying feature. Changes to the feature/override combination clear the cache. Active raytraced render targets still fall back. The initial backend is explicitly compute RayQuery.

Frame sequence: direct integration and gradient -> sparse cache update -> cache resolve -> full-resolution indirect query -> existing primary NEE/output assembly. Dedicated shader layouts use reserved bindings **230–232**; the branch's atmosphere owns 200–217. Explicit read/write cache access metadata uses DXVK's normal compute hazard tracking, including configurations that skip pure write-after-write barriers. The sparse update does not write the aliased indirect output, ReSTIR outputs, NEE feedback tasks, or debug image.

The primary surface only seeds rays. Cache insertion starts at eligible indirect surfaces: incoming diffuse ray, rough opaque material, full opacity, no medium, no diffusion-profile SSS, and no current emission. Queries additionally require a segment longer than the cache voxel diagonal. Cache misses continue normal path tracing. Query hits add cached reflected lighting through the ordinary path throughput and terminate before local NEE, preserving final output writing.

Each inserted cache vertex starts with weight one. Local NEE and later radiance contributions propagate through SHARC's stored segment weights, independent of the camera prefix. Current emission/MIS is propagated before insertion and remains explicit in rendering. Attenuation and continuation weights multiply retained cache weights. With cache resampling and responsive lighting disabled, hash insertion overflow preserves existing vertices and continues their estimate. Updates use finite paths of at most eight bounces and disable roulette; propagation depth is eight. This remains a spatial/temporal approximation with finite-depth bias.

History clears on renderer history reset, activation/frame discontinuity, capacity/grid/roughness changes, update-bounce changes, and explicit Reset. Moving geometry/light/weather changes rely on temporal replacement and stale eviction unless they trigger the renderer's history reset. There is no new automatic complete scene-change detector.

## Verification and limitations

Final Release compilation and linking succeeded (Meson exit code 0). Final build log: `_Comp64Release/sharc-build-complete.log`; artifact: `_Comp64Release/src/d3d9/d3d9.dll`. Only trailing whitespace was cleaned after shader compilation; no executable behavior changed. Use the repository-required `meson compile -C _Comp64Release`, never invoke ninja directly. Packman tools and builds require elevated workspace access in the current environment.

`validate_sharc.py` passed all nine SDK compile/SPIR-V/layout contracts using the repository's Packman Slang 2025.10.4 compiler. `validate_sharc_estimator.py` passes five numerical propagation tests including independent backward recurrence, skipped surfaces, and finite propagation depth. `validate_sharc_integration.py` passes all four actual built-binary checks: update, query, resolve, and ordinary baseline. It verifies SPIR-V validity, cache bindings, update atomic capability, absence of update image writes, presence of query image output, and no SHARC resources in the baseline. `git diff --check` passes.

The Release DLL and matching symbols were deployed to Fallout New Vegas (.trex) on 2026-09-12, with backups and SHA-256 verification. No GPU execution, image comparison, timing, or visual quality validation has been performed. Do not claim a speedup or production readiness. Next runtime work is a reference comparison with cache disabled, constant-light diffuse scenes, emissive/sky cases, camera and light changes, overflow stress, and GPU validation for synchronization. Sharp reflected lighting, material detail within a cell, and dynamic lighting can exhibit cache bias or lag. Lock-buffer, directional SH, material demodulation, and responsive-lighting variants are not exposed production options.

The September 11 continuation note is historical: its missing estimator/scheduler and enum issues have since been implemented/fixed. Prefer this document and current source when continuing.

## Performance pass ? September 12

User reports FNV runs with all three compatibility overrides enabled; observed
frame time was approximately 24 ms with ReSTIR GI versus 31 ms with SHARC before
this pass. This is user-reported scene validation, not comprehensive compatibility
or a controlled benchmark.

- `rtx.sharc.deferredUpdates` defaults to true. The update shader records local
  radiance and segment weights, then propagates backward and writes each inserted
  vertex once at path completion. This replaces repeated writes to every preceding
  vertex for each lighting contribution. Camera-prefix throughput is excluded;
  failed insertions continue contributing to earlier vertices. This is for the
  current finite-depth, non-resampling, non-demodulated RGB estimator. Floating
  point ordering and integer quantization differ slightly from the original.
- The original update shader remains selectable via **Batch SHARC cache writes**,
  enabling same-settings comparison. Changing it resets the cache.
- Updates of at most four bounces select a dedicated four-vertex shader; five to
  eight bounces use the eight-vertex shader. Maximum bounces never exceed storage.
- Query computes the spatial key and voxel size together, removing a duplicate
  grid-level calculation. Cache descriptor access matches actual shader usage:
  query reads hash/resolved data, update reads/writes hash/accumulation data.
- **Measure SHARC GPU time** (`rtx.sharc.measureGpuTime`, default false) displays
  smoothed update/resolve/query milliseconds using asynchronous timestamp queries.
  Pending queries skip measurement; there is no CPU wait for GPU results. Timing
  boundaries include synchronization and can affect overlap; disable the option
  for final total-frame comparisons. Profiler ranges also identify the three passes.
- **Performance preset (fewer updates)** explicitly changes tile size to 8, bounces
  to 4, and capacity exponent to 20. It reduces coverage/path depth and may affect
  quality. Existing defaults and scene settings remain unchanged until clicked.

Validation: Release build succeeded; six built-SPIR-V contracts passed (original
update, deferred update, deferred four-vertex update, query, resolve, baseline).
Five deferred numerical tests include 2,000 randomized comparisons against an
independent forward-propagation reference, failed insertions, repeated cache cells,
zero channels, and emission ordering. All five original estimator tests passed.
In-game timings for this pass remain to be measured; no speedup is claimed yet.

## Query pass ? September 12

User measured update 1.42 ms, resolve 0.02 ms, query 12.33 ms with the performance
preset (20/8/4) and roughness 0.8. Lowering roughness to 0.5 raised query time to
approximately 14 ms with no reported visual change. These measurements implicate
the full-resolution query/tracing pass; they do not establish the cache hit rate.

`rtx.sharc.queryRayGeneration` defaults to true and runs the same inline RayQuery
integrator as a dedicated ray-generation shader. This does not introduce TraceRay
hit/miss shaders or SER. Disable **Ray-generation SHARC query** to compare the
compute backend, now dispatched in 8x8 groups instead of 16x8. Cache resources
include ray-tracing access stages and switching backend resets the cache.
No quality parameters or path limits are changed by selecting the backend.

With GPU timing enabled, **Include cache reuse statistics** collects a rotating
1/64 pixel sample in separately compiled diagnostic query variants. The panel
reports paths terminated by cache hits, mean traced ray segments per path, lookup
hit rate, eligible-surface fraction, roughness/non-diffuse/other surface rejects,
and near-distance rejects among eligible surfaces. It also shows cache age to
identify repeated resets. Counters use binding 235 (230?234 are used by the separate
sparse-compaction pass), an asynchronous readback ring, a transfer-to-host barrier,
and completion events; pending frames skip diagnostics instead of blocking.
Disable statistics for timing comparisons without diagnostic atomics.

The minimum-roughness label now explicitly says squared; 0.8 corresponds to
approximately 0.89 perceptual roughness. The performance preset does not change it.

Validation: Release build succeeded and all nine built shader contracts passed,
including both compute/ray-generation query variants with and without statistics.
The ray-generation variants contain inline RayQuery and no TraceRay instruction.
The next in-game comparison should use the same roughness and stationary view,
then compare query backends and inspect measured cache reuse. No performance gain
for this query pass has been measured yet.


## Reuse and update experiment - September 12

Latest user measurement from the ray-generation query build: update 1.48 ms,
resolve 0.02 ms, query 7.75 ms, cache age 290 frames. Cache terminates 4.9% of
paths, with 1.05 main ray segments/path and 66.7% lookup hit rate. Eligible
surfaces are 19.6%; rejects are 0.0% roughness, 56.8% incoming non-diffuse, and
23.6% other. Of eligible surfaces, 23.5% are too close. Settings: capacity 20,
tile 8, update bounces 4, accumulation 8, stale 32, grid 50, roughness 0.8.
WBOIT override was on; portal/OMM overrides off. The query timing is about 37%
lower than the preceding 12.33 ms report, but these are user measurements rather
than a controlled benchmark. The segment counter excludes NEE visibility rays
and repeated traversal inside material resolving.

The next experiment implements:

- **Ray-generation SHARC updates** (`rtx.sharc.updateRayGeneration`, default
  true). The original, deferred eight-vertex, and deferred four-vertex update
  shaders each have a dedicated inline-RayQuery ray-generation variant. Both
  backends dispatch the same logical tile grid and use the same source-pixel
  selection, path depth, estimator, and resource access. Switching backend clears
  the cache. The compute variants remain available for comparison.
- **Reuse rough surfaces in specular paths** (`rtx.sharc.allowSpecularPaths`,
  default false). Optional insertion/query policy accepts rough receiving surfaces
  even when the incoming ray was non-diffuse. This is an approximation: removing
  the preceding diffuse filter can expose spatial/directional smoothing in
  reflections. Existing receiving-material, medium, emission, roughness and
  query-distance checks remain. Update and query use the same policy, and toggling
  it clears the cache. `skyGatherEligible` retains its original sky-scaling meaning.
  The previously unused final uint in the 64-byte SharcArgs now carries this policy.
- Sampled query diagnostics add sky-miss, bounce-limit, zero-weight and roulette
  path endings (counters 10..13, 14 uint readback within the existing 256-byte
  slot). UI shows authored global path limits/roulette and calls the existing
  ray counter **segments/path**. Cache termination is reported separately; rare
  material-resolver exits can leave the displayed categories below 100%.
- Removed a redundant cache-age assignment in the reset branch.

Validation: Release build completed via `meson compile` with exit 0; log:
`_Comp64Release/sharc-reuse-build.log`. All twelve compiled-SPIR-V contracts passed
(six updates, four queries, resolve, baseline). Ray-generation updates are verified
as inline RayQuery without TraceRay or image output writes. Both five-test
estimator suites passed, and `git diff --check` passed. No runtime speedup or
image-quality result for this new experiment is available yet.

Deployed the new renderer and matching PDB to FNV `.trex`; both destination hashes
match their build sources. Backups use suffix
`.backup-pre-sharc-reuse-20260912-080256`. DLL SHA256:
`D7A61CB467AF5D60E780DC834F562F15A0F887F5BD981FB31BAF2C9A8C4D72FA`.
PDB SHA256:
`18122B220CED11FAF438DA1322087685430F816E3FF49226FA4531D2C6FC4A9F`.
The game was closed, and configuration was not modified. The bridge is unchanged.
Normal-layout package completed using
`C:/Users/sparkles/AppData/Local/Temp/package_sharc_reuse.py`:
`_packages/rtx-remix-revised-9-10-sharc-reuse.zip` (270,181,821 bytes), plus
`rtx-remix-revised-9-10-sharc-reuse-symbols.zip` (42,247,243 bytes). ZIP CRC,
core PE architectures, and source/destination hashes passed; archive SHA256
values are in `rtx-remix-revised-9-10-sharc-reuse-archives.sha256`.

Next in-game test: keep the same scene/settings, compare update timing with
Ray-generation SHARC updates on/off, then enable Reuse rough surfaces in specular
paths and compare query time, cache termination percentage, and reflected
lighting after a few seconds. Read Path ends to explain the short paths. Keep
roughness at 0.8; lowering it previously worsened query timing without a visible
benefit. Disable sampled statistics for final timing comparisons.


## WBOIT resolver correction - September 12

The user corrected the earlier interpretation: enabling rough-surface reuse did
help performance. The next paired captures with reuse enabled show statistics
off at update 1.60 / resolve 0.04 / query 8.82 ms and statistics on at 1.30 / 0.03 /
9.78 ms. With statistics on: 23.7% cache termination, 1.04 segments/path, 85.5%
lookup hit rate, 67.6% eligible surfaces, 49.2% sky misses and 26.8% roulette
termination. Earlier capture had 22.3% termination, 57.5% sky misses and 1.02
segments/path. Do not treat the cross-capture delta as a controlled exact overhead
measurement; the user reports a modest benefit from wider reuse.

Inspection found that the SHARC compatibility override allowed WBOIT but all ten
update/query variants lacked UNORDERED_RESOLVE_WBOIT. Thus SHARC still used the
sorted-bin particle resolver while the other renderer passes followed WBOIT.
This can increase temporary state/sorting work and produce different particle
lighting. Ten dedicated WBOIT variants now mirror all existing original/deferred
update and compute/ray-generation query combinations, including statistics.
Dispatch selects them using the existing WBOIT setting. Cache reset on a WBOIT
change already exists. No new compatibility permission or quality control is
needed. Particle transparency in indirect lighting may change to match the
selected WBOIT behavior.

Statistics now default to false to avoid their overhead in ordinary timing tests;
explicit saved settings retain priority. The timing panel reports the active
particle resolver. Rough-surface reuse and all quality settings retain their
existing defaults/settings. TraceRay/SER was investigated but is not implemented
in this pass; resolve the concrete shader-selection mismatch first.

Release build succeeded via Meson (exit 0); log `_Comp64Release/sharc-wboit-build.log`.
All 23 compiled shader checks passed: twelve updates, eight queries, resolve,
and both ordinary baselines. They also verify actual compiled uniform access to
WBOIT energy compensation, rather than relying on filenames to identify the
resolver. `git diff --check` also passed. Runtime measurements are pending.
User confirmed closing FNV; process check verified both game and bridge had
exited. Deployed renderer DLL and matching PDB to FNV `.trex`, with verified
source/destination SHA256 matches and backup suffix
`.backup-pre-sharc-wboit-20260912-081911`.
DLL: `39F61EE070006309A3EA8088CCEA4D35951296D681ADB3F148CD9FDB256B16E0`.
PDB: `8A193D8A9A8296C8A732F0720333D512089B053BBDC1A4540D6635F91F0275E8`.
No game config changes or automatic launch. Next test: leave reuse enabled,
statistics off, confirm Particle transparency: WBOIT, and compare update/query
against the same scene (previous statistics-off query 8.82 ms). Inspect particles
as the selected transparency behavior is now actually used by indirect lighting.
Normal-layout package completed via
`C:/Users/sparkles/AppData/Local/Temp/package_sharc_wboit.py`:
`_packages/rtx-remix-revised-9-10-sharc-wboit.zip` (272,842,267 bytes), plus
`rtx-remix-revised-9-10-sharc-wboit-symbols.zip` (42,288,029 bytes). ZIP integrity,
core PE architectures, and source hashes passed. Archive checksums are in
`rtx-remix-revised-9-10-sharc-wboit-archives.sha256`.


## TraceRay query experiment - September 12

Latest WBOIT-corrected capture: update 1.55 / resolve 0.03 / query 8.08 ms,
statistics off, cache age 265. WBOIT, batched writes, ray-generation updates/query,
and rough-surface reuse enabled. Capacity 20, tile 8, bounces 4, accumulation 8,
stale 32, density 50, minimum squared roughness 0.8. This is 0.74 ms below the
previous statistics-off capture; these snapshots are not a controlled benchmark.

Adding a TraceRay query backend using the existing indirect hit/miss integration
and device-gated indirect-pass SER setting. Separate SHARC variants retain WBOIT
and sampled-statistics selection; updates and cache eligibility are unchanged.
rtx.sharc.queryTraceRay defaults true for this experiment; disabling it restores
the existing ray-generation/compute choice. Backend changes reset cache history.
The timing panel identifies the backend actually selected for the current frame.
Completed: release build and all 43 compiled shader contracts passed, including
16 split-stage variants and four additional legacy hit/miss checks. Payload
scalar/vector types match across stages; query stages retain the inline SHARC
reservoir policy. All 37 current SHARC SPIR-V blobs were located byte-for-byte
inside the final DLL. The shader generator updated headers after C++ dependencies
were considered, so an additional Meson compile was necessary to relink; always
verify final shader embedding after shader-only changes.
Logs: sharc-trace-build.log, sharc-trace-final-build.log,
sharc-trace-link-build.log, sharc-trace-validation.log in _Comp64Release.

Deployed to FNV .trex with game/bridge closed and matching DLL/PDB hashes.
Backup suffix: .backup-pre-sharc-trace-20260912-085525.
DLL SHA256: 7CE801E35A4A8C5FC903CBCD1D90747D7021138C0B7B891035A8F5EFADBE0D47.
PDB SHA256: 1555A026E5D8E06268D88B81E88EFD81862DC43613199A4FB2C6525DE83AFDF8.
Package: _packages/rtx-remix-revised-9-10-sharc-trace.zip (275047193 bytes),
plus -symbols.zip (42374137 bytes) and -archives.sha256. Normal bridge/renderer
layout, architecture, source hashes and archive CRC verified by
C:/Users/sparkles/AppData/Local/Temp/package_sharc_trace.py.

The user now reports SHARC is almost on par with ReSTIR GI and requests further
optimization. Exact new update/query/total timings are pending; do not infer
an exact speedup from this qualitative report.


## Material specialization and fixed-index updates - September 12

User estimates the TraceRay build was about 1 ms slower than ReSTIR GI in total
frame time. No new per-pass measurements were supplied.

TraceRay closest-hit shaders now select POM and portal support using the existing
renderer feature flags. Miss shaders specialize portal support. Added 16 shader
variants, preserving full-feature variants for compatible scenes. SHARC retains
its existing inline-query reservoir policy. Relevant permutations are prewarmed.
The WBOIT closest-hit SPIR-V falls from 779624 bytes to 728744 bytes with both
features inactive (about 6.5%); binary size is not a GPU-time measurement.

Deferred update arrays use unrolled constant-index accesses, including reverse
propagation, to allow compiler scalarization and reduce potential register
spilling. Update density, bounce limits, radiance recurrence, cache eligibility
and all quality settings are unchanged. Original non-batched updates remain
available through the existing option; the previous DLL is backed up for exact
A/B testing. Performance impact is not measured yet.

Release compilation and final relink passed. Logs in _Comp64Release:
sharc-specialized-build.log, sharc-specialized-link.log,
sharc-specialized-final-link.log, sharc-specialized-validation.log.
59 SPIR-V contracts passed, including the 16 additional specialized stages.
Six estimator tests passed, including 2000 random paths, overflow, unwritten
slots, repeated cells, failed insertion and zero color channels. All 53 current
SHARC shader blobs are embedded byte-for-byte in the renderer DLL.
Git diff whitespace check passed.

Deployed with FNV and bridge closed; verified source/destination SHA256 matches.
d3d9.dll: 8567AE125AF396DC5781FC65E3D543F23250761EFF044122F8C526078A2C773F
Backup: C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\d3d9.dll.backup-pre-sharc-specialized-20260912-210320
d3d9.pdb: BEDE72FD97AABE5CF64A5774860F80B9E15504362C627AA99A6A1BF9FC444B43
Backup: C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex\d3d9.pdb.backup-pre-sharc-specialized-20260912-210320

Packaging script: C:/Users/sparkles/AppData/Local/Temp/package_sharc_specialized.py.
Packaging completed: _packages/rtx-remix-revised-9-10-sharc-specialized.zip
(278969822 bytes), corresponding -symbols.zip (42312011 bytes), and
-archives.sha256. Archive CRC, core PE architecture and source hashes passed.

The user supplied C:/Users/sparkles/Downloads/SHARC openremix and explicitly
requested Luna research. Agent /root/sharc_custom_research is performing read-only
comparison and ranking transferable optimizations. No downloaded code has been
executed or incorporated. Research completed; see SHARC-openremix-research-2026-09-12.md. Only two GLSL
integration files were supplied, not a full path tracer. Best performance lead is
opt-in update cache resampling, which requires deliberate integration in both
update backends. Resolve cross-entry history switches merit a correctness audit
but are not a plausible source of a ~1 ms gain. No resampling or downloaded-code
changes have been implemented. Current specialized build runtime timing is pending.


User validation of specialized build: the apparent FNV hang was a false alarm.
SHARC is now very fast and, in the user's assessment, visually outperforms
ReSTIR GI and NRC. Rough-surface reuse no longer has a noticeable effect in
their test. User requested committing this baseline and trying cache resampling.

## FNV bridge hang fix and resampling removal (2026-09-12)

Confirmed IPC deadlock with both bridge command rings full; renderer workers were idle. Built and deployed shared recursive response/device serialization, fail-fast response identity/timeout handling, and reset/readback corrections. Paired-queue regression passed 1,000 request/response UIDs. User reports the deployed fix worked; original initiating event remains unproven. See FNV-hang-diagnosis-2026-09-12.md for evidence and deployment records.

The experimental SHARC update cache resampling commit was removed from branch history at user request after showing no performance benefit. SHARC and its earlier performance improvements remain. This source-history change does not replace the currently deployed game binaries; the diagnosis records describe the historical deployment.


## Lean secondary profile — September 13

Opt-in full-resolution lean SHARC shader profile implemented and built. See [first-milestone details](Lean-SHARC-first-milestone-2026-09-13.md) for scope, artifacts, validation and A/B instructions. No game deployment or runtime performance/quality validation yet.

## 2026-09-13: Feature-preserving SHARC resource specialization

Full-feature SHARC now uses a resource layout without NRC/ReSTIR-GI descriptors and skips their indirect binding helpers while active. Release build and compiled contracts passed; 66 current stages verified in the DLL. Local only, no deployment or performance measurement. See [SHARC-first renderer](SHARC-first-renderer-2026-09-13.md).

### Test build deployed to FNV

The feature-preserving resource specialization and dedicated indirect-lighting scheduling entry point were built and deployed to FNV on 2026-09-13. DLL/PDB backup suffix: .backup-pre-sharc-first-20260913-012241. Existing settings and bridge hashes unchanged; lean remains enabled in the user's configuration. No in-game test performed. See [test build and boundary](SHARC-first-test-build-2026-09-13.md).

### Whole-sequence profiling prepared

User reported no measurable resource-cleanup gain. Full/lean SHARC stage snapshots were 2.47/1.15 ms; full frame time 18.5 ms. Opt-in sampled GPU stage logging is now built and deployed reversibly to FNV, with local log analysis. See [profiling record](GPU-stage-profiling-2026-09-13.md). No GitHub push.

## Assembly boundary test candidate (2026-09-13)

Built and deployed the indirect subsystem ownership refactor plus an explicit borrowed AssemblyResources interface. Plain NEE assembly now omits three unused backend descriptors and skips their CPU preparation; legacy and NEE visualization bindings remain intact. No shader algorithm or quality setting changed. Release build and compiled resource/DLL embedding validation passed. FNV and the bridge launched; in-game compatibility remains to be checked. No measured speedup is claimed, and the complete backend resource boundary remains unfinished. Deployment hashes, rollback suffix and test instructions: [assembly test](SHARC-assembly-test-2026-09-13.md).

Assembly candidate follow-up: user found appearance correct but performance worse. Capture preserved and installed DLL/PDB reverted to pre-assembly-boundary backup. Main measured increase is ScenePreparation; causality is unresolved. See SHARC-assembly-test-2026-09-13.md. Do not redeploy current build outputs as a known-good candidate.

Retest requested: the assembly-boundary candidate is installed again, restored from verified backup. Performance verdict remains pending retest.

Assembly retest result: user reports properly measured performance largely matches the old build. Latest 12-sample median sequence is 15.387 ms; scene preparation returned to 0.117 ms and assembly is 0.312 ms. Earlier slowdown did not reproduce. Keep candidate installed; count this as structural progress with no demonstrated speedup, not a confirmed performance regression. See SHARC-assembly-test-2026-09-13.md for capture details and remaining coverage limits.

Query-probe experiment 0ba50562d built, validated and deployed. Only the SHARC query lookup-loop unroll attribute changes; performance pending game test. See SHARC-query-probe-test-2026-09-13.md.

Query-probe first capture: user confirms correct lighting and perceives a slight improvement. IndirectIntegration median 2.366 vs 2.390 ms, but total measured sequence 15.932 vs 15.387 ms. Result inconclusive across separate runs; no demonstrated overall speedup. Capture retained and candidate left installed for comparison. See SHARC-query-probe-test-2026-09-13.md.

Fusion prerequisite: extracted shared per-pixel NEE assembly helper; source calculation unchanged, release and embedded resource checks passed. No fused mode exists yet and game install unchanged. Audit identifies throughput/diffuse aliasing, raw secondary consumer, half-precision handoff, sparse/miss exits and NEE feedback interleaving as requirements. See SHARC-query-assembly-fusion-2026-09-13.md.


## 2026-09-13: selectable combined query/assembly prototype

The next test candidate introduces opt-in `rtx.sharc.fuseAssembly` (default false), merging full-feature TraceRay SHARC query and primary NEE assembly. It preserves update/resolve, raw secondary output, primary lighting math, and existing split fallback. It needs one independent throughput allocation (8 bytes per render pixel). Build, shader-resource/embedding validation, and DLL factory smoke test passed; runtime quality and speed remain unmeasured. See [fusion audit and test procedure](SHARC-query-assembly-fusion-2026-09-13.md).

### Fused assembly runtime decision

The fresh same-DLL split comparison measured median query+assembly at 2.339 ms versus 2.956 ms fused; measured GPU sequence was 15.492 ms split versus 15.628 ms fused. Fusion remains disabled in FNV and defaults off in source. Separate-capture variability limits causal attribution, but current evidence does not justify adoption. See SHARC-fused-assembly-test-2026-09-13.md for preserved captures and limitations.

## Work paused by user - 2026-09-13

Stop the renderer overhaul here. Preserve the current implementation and experimental options without further optimization or deployment work.

Fusion produced no useful demonstrated performance benefit. In the fresh comparison, query plus assembly was slower with fusion (2.956 ms versus 2.339 ms split), while the overall measured GPU sequence was similar (15.628 ms versus 15.492 ms). Separate captures limit causal conclusions. Keep fusion disabled; it remains an opt-in research prototype, not an accepted optimization. FNV remains on full-feature SHARC with leanSecondary=False and fuseAssembly=False.

## Lean parity pass resumed - 2026-09-14

User lifted the pause. Analysis of the five lean gaps found that the full SHARC RTXDI sample stealing branch was non-functional and dropped NEE at camera-visible secondary vertices (no SHARC stage binds the reservoir); it is now compiled out for SHARC stages, matching lean. Lean and legacy blobs unchanged. See [parity record](Lean-SHARC-parity-2026-09-14.md) for the per-gap analysis, ranking and validation. Local only; no deployment, no push.

### Lean feature tiers built (2026-09-14)

Added opt-in `rtx.sharc.leanFeatureLevel` (1 restores alpha-blended indirect shadows, 2 also restores the unordered particle/decal resolve; each tier used only in frames containing that geometry) and `rtx.sharc.leanIndirectPom`. 28 new lean variants; level-2 query stages are byte-identical to the full stages after the stealing guard. All validators pass; RtxOptions.md regenerated. See [parity record](Lean-SHARC-parity-2026-09-14.md) for sizes, validation and the FNV A/B procedure. Not deployed; no timing measured.

## Consolidated into one SHARC path - 2026-09-15

User decision after the parity analysis: the lean profile is removed. Its scene gate was ported into the main path first (commit a4c057777): the direct/indirect alpha-blend and translucent shadow-ray mask bits and the indirect unordered resolve are now skipped per frame when the TLAS contains no such instances, which is output-identical; shader blobs unchanged. Then the lean options, 42 shader variants, selection code, UI and validator were deleted; four shader sources are byte-identical to the pre-lean tree and all 56 SHARC stages are byte-identical to the stealing-guard build. The SHARC reservoir guard from 9901d7e99 is kept and pinned by validate_sharc_integration.py. RtxOptions.md regenerated. Local only; nothing deployed or measured. Record: Lean-SHARC-parity-2026-09-14.md.

## Ray portals confirmed working in Portal RTX - 2026-09-15

SHARC ran for a full Portal RTX session with ray portals active and **fell back zero
times**: 59,400 selected frames, 0 on a raytraced render target, 0 on other conditions,
across a 21-minute session (18:38 to 18:59). The runtime log carries 0 errors and 4,318
warnings, all benign -- replacement textures without mip-maps, two unknown D3D9 formats,
USD skel primvar types, and Sentry declining to initialise. No CUDA, NRC or USD load
failure. NRC v0.15 initialised cleanly, confirming the dependency deployment.

User reports portal content looked better than the other indirect samplers.

This exercises the 2026-09-15 portal work: insertion is gated on
portalSpace == PORTAL_SPACE_NONE so portal-reached vertices never pollute a cell that
main-space paths read, and the gate predicate was narrowed to numActiveRayPortals > 0
rather than firing whenever portal texture hashes are merely configured -- the latter
alone would have disabled SHARC for the entire Portal RTX session.

Caveats. allowRayPortals was True, so the portal gate could not have fired regardless;
the zero on "other conditions" proves WBOIT, micromaps and allocation never fired, not
that the portal gate was exercised. The insertion gate is shader-side and not logged, so
its engagement is inferred from portals being active, not observed. No frame time was
measured. The two known portal artifacts remain unfixed: the farEnough self-reference
guard weakens after a teleport, and grid level uses true world distance so through-portal
geometry caches coarsely for its screen size.

Deployment: PortalRTX/bin/.trex, our d3d9.dll plus 32 dependency DLLs and the usd/ plugin
tree from the FNV runtime (CUDA 13, newer NRC_Vulkan, unbundled USD stack, upscalers).
Backups retained with suffix backup-pre-sharc-20260915-*.

## Portal-space hash key confirmed in Portal RTX - 2026-09-15

User reports through-portal content looks better than the insertion-gate build, with
performance about the same, possibly a touch worse. That matches the predicted trade:
portal-only geometry is now cached instead of brute-forced, which improves the image,
while cells separated by portal space raise occupancy against the fixed capacity.

capacityLog2 is the knob if the slight cost is worth chasing. It is 20 (1M entries) in
the Portal RTX config; 21 halves collision pressure for twice the memory. Untested.

No frame time was measured on either build; "about the same, maybe a tad worse" is the
user's in-game impression, not a capture. The gate build is retained at
PortalRTX/bin/.trex/d3d9.dll.backup-portal-gate-20260915-1950 and its source is tagged
sharc-portals-verified-20260915, so an A/B remains possible without a rebuild.

## Eligibility diagnosis and per-pixel debug views - 2026-09-15

Portal RTX readings with the stats window (roulette on, bounces 1..4, allowSpecularPaths on,
~2.56M paths): eligible surfaces 0.2%, roughness rejects 62.0%, "other" rejects 37.8%, too close
30.1% of eligible, hit rate 69.7%. Roulette off lengthened paths (1.71 to 3.96 segments) without
moving eligibility (0.3%), so path length is not the constraint.

**Clamp bug.** All of those numbers were taken at an effective `minRoughness` of 0.5: `prepareFrame`
clamped the option to [0.5, 1] before writing `sharcArgs`, while the slider showed the user's 0.05.
Fixed in 7dcd68206 (floor 0.05), built and deployed to both installs, backup suffix
`backup-pre-minrough-fix-20260915-203424` (Portal RTX: dll only; FNV: dll and pdb). Not yet
re-measured; the 62% roughness figure is a reading of the wrong threshold, and how much of it
survives at a real 0.05 is the first thing to read off the panel.

**What "other" is.** Slot 9 is `!sharcSurfaceEligible`, five terms in one number. The
integrator's own resolve loop says which are plausible in Portal RTX: translucent materials
(glass panels) fail the opaque test; any vertex whose `emissiveLight` has a non-zero channel
fails the emissive test, and the test is strict (`> 0`), so a panel with a faint emissive map
counts the same as a light fixture. Which of the two dominates cannot be inferred from source,
which is why slot 9 is now split (counters 14..18: non-opaque, medium, opacity < 1, subsurface,
emissive) and the panel prints the split under the "Surface rejects" line.

**Too close.** `farEnough` compares `segmentHitDistance` with the voxel diagonal, and
`segmentHitDistance` is the *last resolve leg*: `RESOLVE_RAY_TRACE` zeroes it before every
re-trace, and `resolveVertexFinalContinue` re-traces for clipped geometry, opacity below the
transparency threshold / stochastic alpha (cutouts such as grates and catwalks), the opaque and
translucent approximations, a miss on a portal quad, and a portal teleport. A wall behind a grate
therefore measures the grate-to-wall gap, and a wall reached through a portal measures the
post-portal leg. Both are conservative (the true vertex spacing is never shorter than the last
leg), so the guard cannot admit a self-referencing lookup; it rejects lookups it should allow.
The portal case is now moot for self-reference anyway: since 7c337eb39 the cell key carries
portal space, so the pre- and post-portal vertices can never share a cell. (The 2026-09-15
portal investigation reasoned the other way, "the guard weakens"; that predates the key change.)

Query stages now track the whole segment length in `PathState.sharcSegmentDistance` (a sign-less
float16 in the spare bits of `_skyGatherEligible` plus the padding byte before
`_pixelCoordinate`, so the payload does not grow) via two new hooks, `sharcOnSegmentBegin` and
`sharcOnResolveLeg`. The guard itself is unchanged: counters 19..21 split too-close into
"last leg only" (the whole segment passes), "post-portal vertex" and "first indirect bounce", so
the measurement artefact can be sized before the guard is switched to the whole-segment
distance, which is a one-line change in `sharcOnResolvedVertex`.

`gridScale` scales the threshold directly (`voxel = 2^level / gridScale`), so it is a lever on
the genuine too-close share and on nothing else; it has no effect on the last-leg artefact and
the split says which of the two dominates.

**Debug views** (`rtx.debugView.debugViewIdx` 580..587, listed under "SHARC" in the Debug View
panel; query stages only, so SHARC must be the active indirect mode). The vertex shown is
bounce ROUND(Debug Knob [0]) + 1, knob 0 = first indirect hit, the NRC convention:

| Index | View | Reading |
|---|---|---|
| 580 | SHARC Query: Outcome | green hit, red miss, blue too close, grey rejected, black no vertex |
| 581 | SHARC Query: Rejection Reason | red non-opaque, green medium, blue opacity, yellow subsurface, magenta emissive, cyan lobe, white roughness, black eligible |
| 582 | SHARC Query: Too Close Guard | green passed, red genuinely close, yellow last-leg artefact, magenta/cyan the same two through a portal |
| 583 | SHARC Query: Cached Radiance | radiance read where the path ended on the cache (HDR) |
| 584 | SHARC Query: Termination Bounce | bounce at which the cache ended the path, 0 never |
| 585 | SHARC Grid: Cells | hash-coloured cell at the vertex |
| 586 | SHARC Grid: Level / Voxel Size / Last Leg | R level, G voxel size, B last leg (raw values) |
| 587 | SHARC Grid: Cell Age | R accumulated frames, G stale frames, B sample count (raw values) |

The rejection reason, stats split and eligibility test all derive from one function,
`sharcRejectReason`, so they cannot disagree. Cost when no SHARC view is selected: one uniform
range compare per resolved vertex in the query stages, plus the segment-length bookkeeping
(two float16 pack/unpack operations per resolve leg). Not measured.

**Legacy blobs.** The first build of the views left the non-SHARC stages one instruction
different from the pre-change DLL: the `inout PathState` copy-back of the two new hook calls
survives as a redundant byte store even though their non-SHARC bodies are empty, and the
`_sharcSegmentDistanceHi` byte was declared for every variant. Both are now under
`ENABLE_SHARC` / `SHARC_QUERY`, and `validate_sharc_integration.py --baseline-dll` (the
pre-change stats build) passes all 62 contracts including "legacy stages byte-identical to the
baseline DLL". Deployed to both installs with backup suffix
`backup-pre-legacy-gate-20260915-211222`; no view has been looked at in-game yet.

## Portal RTX: the cache was gated off, not underperforming - 2026-09-15

Three gates, none of them tuning, were keeping SHARC at 0.2% eligibility in Portal RTX.
With all three cleared, measured over 2.54M paths in 120 frames at cache age 1012:

| | before | after |
|---|---|---|
| eligible surfaces | 0.2% | 73.8% |
| cache terminates | 0.2% of paths | 78.5% |
| lookup hit rate | 58.6% | 99.2% |
| roulette path ends | 89.6% | 21.0% |
| segments/path | 1.71 | 1.12 |

The gates, in the order they were found:

1. **allowSpecularPaths was absent from the Portal RTX rtx.conf**, so it defaulted false and
   lobe eligibility collapsed to skyGatherEligible, which is false throughout a windowless
   test chamber. Config, not code. The rejection-reason view was flooded cyan.
2. **minRoughness was clamped to 0.5** on the way into sharcArgs while the option and slider
   accepted lower values, so every earlier attempt to widen eligibility measured an unchanged
   threshold (7dcd68206).
3. **The emissive test was any(emissiveLight > 0)**, which disqualified every surface carrying
   a faint emissive map, most of the level. Now a luminance threshold, default 0 for the old
   behaviour (9c464b31a). At 0.1 the view goes from almost entirely magenta to 0.8% emissive.

User reports 0.1 ms consistently at maxEmissiveLuminance 0.1, and cached radiance now covering
most of the image. **0.1 ms is the honest figure**: 78.5% of paths terminating on the cache buys
little because the indirect pass was never the frame's bottleneck.

Remaining rejects are roughness 15.8%, medium 5.8%, non-opaque 3.9%, emissive 0.8%. Roughness is
the quality wall rather than a defect: Portal's panels are smoother than an isotropic diffuse
cache can represent, and minRoughness is already at its 0.05 floor. Too-close is 4.1% of eligible
surfaces, 88.5% of that at the first bounce, with 0.0% last-leg artefact and 0.0% post-portal --
so the whole-segment guard change is not needed.

Next lead: a 99.2% hit rate means the cache is over-served and its upkeep can be cut. Raise
updateTileSize, lower updateBounces, lower capacityLog2, each until the hit rate leaves 99%.

## Footprint gate and sample floor confirmed in Portal RTX - 2026-09-16

Both land, and between them they replace the whole hit-roughness approach to specular reuse.

**Sample floor.** minSampleCount = 2 removes the glow that appeared on camera movement,
worst on geometry previously cut off by the screen edge. SHARC_SAMPLE_NUM_THRESHOLD is 0 in
the SDK, so a cell answered a query from a single high-variance sample; cells for off-screen
geometry were never updated, so the first path to land in one was read back as converged.
The roughness split could only ever mask this, which is why minRoughnessSpecular had to go
to 0.68 and still did not fully clear it.

**Footprint gate.** User reports an extreme improvement: far more data in the cache, holding
even with minRoughnessSpecular at 0.05. That is the expected shape of the fix. The hit-material
roughness test was a blunt proxy that refused every specular arrival at a surface below the
floor regardless of the arriving lobe's width; NVIDIA's footprint test measures the lobe that
launched the segment, so tight lobes are refused and broad ones admitted on their merits. The
coverage the old threshold was spending is recovered at no quality cost.

Both defaults now reflect this: footprintGate true, minSampleCount 2. minRoughnessSpecular is
inert while the gate is on and is kept only so the previous behaviour can still be A/B'd.

Not measured: frame time for either change. The cache holding more data is a coverage result,
not a speed one, and the indirect pass was never this frame's bottleneck.

## Sky-bound update paths: two opt-in recoveries - 2026-09-16

In FNV open desert the panel reads "Path ends: sky 65.9%" and 1.01 segments/path. Update and
query paths of one pixel share the same stored first segment, so that is also the share of
update paths that insert nothing: vertices are cached only at resolved secondary hits, and a
path whose first bounce misses has no vertex for the sky to be credited to. Two thirds of the
update budget outdoors bought no cache sample.

Both recoveries are options under `rtx.sharc`, default off, cache-clearing on change, and
compiled only into the update stages (query and legacy blobs are byte-identical to the
previous build):

- `updatePrimaryVertex`: the update path also deposits its primary vertex, valued as the
  direct pass's RTXDI lighting plus the sampled continuation's weight times whatever the path
  gathers. Every sky-bound path then writes one sample (`direct + T * L_sky`), and every
  camera-visible eligible surface is fed by every update tile that lands on it. Excluded:
  PSR pixels, translucent primaries, view models; eligibility is the same predicate secondary
  hits use, fed from the G-buffer material words.
- `updateSkyRetries` (0..4): a first-bounce sky miss is re-sampled from a cosine lobe about the
  primary normal and traced again. No pdf correction is needed: a cell averages outgoing
  radiance samples, whose values do not depend on how the ray arrived.

Full reasoning, ranking of the five directions in the brief, and the measurement plan:
[SHARC-sky-budget-2026-09-16.md](SHARC-sky-budget-2026-09-16.md). Nothing measured in game.

Release build (three passes, DLL 280,173,568 bytes, SHA-256 `d0ec48af...2f563`), integration
validator 62 PASS with legacy stages byte-identical to the `b74a97ede` DLL, all 41 query blobs
byte-identical, the 12 update blobs changed. Deployed to both installs, backup suffix
`backup-pre-sky-budget-20260916-193518`; the FNV file replaced was an intermediate build of this
same change that had been deployed during the session gap, the Portal file was HEAD.

## Three settings presets, and Balanced as the shipped default - 2026-09-16

The single "Performance preset (fewer updates)" button is replaced by a Quality / Balanced /
Performance combo in the panel's existing preset idiom. Only four options trade along the
quality-speed axis and move between presets - `updateTileSize` (4 / 8 / 12), `updateBounces`
(8 / 4 / 3), `updateSkyRetries` (2 / 1 / 0) and `capacityLog2` (22 / 22 / 20). The other ten a
preset writes are correctness and coverage controls that buy artefacts rather than speed when
loosened, so all three write the same value for them; the diagnostics and the backend A/B
toggles are never written.

Nine `RTX_OPTION` defaults were moved so a user who sets nothing gets Balanced, which is the
user's tested FNV configuration: `allowSpecularPaths` and `updatePrimaryVertex` to true,
`updateSkyRetries` 0 to 1, `capacityLog2` 21 to 22, `updateTileSize` 5 to 8, `updateBounces`
8 to 4, `maxEmissiveLuminance` 0 to 0.1, `minRoughnessSpecular` 0.5 to 0.7, `minRoughness`
0.8 to 0.05. Both games' confs already set most of these, so neither install changes behaviour.

On the capacity question: with the primary deposit on, the outdoor live set is estimated at
1.5-3 x 10^5 cells against 2^20 slots, and an insert fails only when sixteen consecutive slots
are held, so 22 is very probably still inert for occupancy - but it is not inert for cost,
because the resolve dispatches one thread per slot every frame. Performance therefore drops to
20. Unmeasured; the panel's "no cell" miss share at 22 versus 20 is what settles it.

Three stale texts corrected in the same files: the `allowRayPortals` description and its panel
tooltip still described the insertion gate that `7c337eb39` replaced with portal-space keying,
and called the feature untested when `:405-418` above records it confirmed in Portal RTX; debug
view 581's legend still called the emissive test "any non-zero emissive radiance" when
`9c464b31a` made it a luminance threshold. The identical stale comment at
`sharc_integrator_hooks.slangh:22` was left alone rather than recompile every SHARC stage.

Reasoning, per-value justification and the measurement plan:
[SHARC-presets-2026-09-16.md](SHARC-presets-2026-09-16.md). Release build (four synchronous
passes, `d3d9.dll` 280,177,664 bytes), integration validator all PASS, `RtxOptions.md`
regenerated. Deployed to both installs, backup suffix `backup-pre-sharc-presets-20260916-203330`.
Nothing measured in game.

## Can SHARC tune itself? No, and here is the constraint that decides it - 2026-09-16

The question was whether a modder could get a good result without knowing which options to touch:
either a controller that retunes SHARC from the stats counters, or better defaults. The answer is
better defaults, and three findings rule out the controller independently.

**Only three of the twenty-two `rtx.sharc` options can change without clearing the cache** -
`accumulationFrames`, `staleFrames` and `updateTileSize` (`rtx_sharc.cpp:152-154`, absent from the
clear condition at `:103-108` and from `compatibilityFlags` at `:79-86`). Everything else, including
both sky options, zeroes all three buffers on change. A controller that retuned them would empty
1M-4M cells per decision, and a cleared cache needs frames the sparse tail may never get back.

**The signal costs more than the feature and arrives a second late.** The counters are a separate
shader permutation (60 `*sharc*stats*` blobs), gated behind `measureGpuTime` + `collectQueryStats`
(about 1 ms, user), read back through an 8-slot GPU-event rotation that silently drops frames
whose event has not signalled (`rtx_sharc.cpp:254-257`), and published only as a sum over 120
collected frames (`:263-268`) - mean sample age about a second, fully replaced about every two.
There is no cheap always-on subset: the one counter that would be nearly free, the sky-exit share,
says what the cache cannot do rather than what to change, and the miss split - the only line that
names a remedy - is the one that costs an extra `HashGridFind` on every miss.

**Both movable knobs change their own measurement,** with a settling time no shorter than the
sampling period, and `staleFrames` and `accumulationFrames` trade the two miss buckets against each
other. Damping that means acting less than once every few seconds, which is slower than a person
with the panel open.

**What is already self-limiting, and is the right pattern:** `updateSkyRetries` fires only on a
first-bounce sky miss (`hooks:352-354`), so it is free indoors with no detection; `footprintGate`
tests the real lobe per lookup rather than a threshold per title; `minSampleCount` and `staleFrames`
only bind on cells that are actually starved or idle; and `gridScale` is an angle by construction,
so it is scene-invariant rather than scene-adapted. Every future "make it adapt" idea should first
be checked against "can this be a per-sample test in the shader?".

**`updatePrimaryVertex` measured at about 0.1 ms** (user) - the same order as the whole cache's net
benefit, and the only preset value with a frame-time measurement rather than an estimate. Its cost
is not purely per-path: the per-path insert shows on `GPU ms: update`, but it also makes every
camera-visible cell permanently resident, and a live slot runs the whole resolve body where a dead
one returns after one load (`SDK:838-840`), so part of it should show on `GPU ms: resolve` and that
part scales with scene openness. Its benefit is likewise two mechanisms, only one of which is about
the sky: recovering sky-bound paths (outdoors only) and feeding camera-visible surfaces densely
every frame (everywhere). So there is no clean scene signal that says when to turn it off.

**One change made: the Performance preset now turns `updatePrimaryVertex` off**
(`rtx_sharc.cpp:395-426`); Quality and Balanced keep it. Performance already declines
`updateSkyRetries` on the argument that the outdoor frame-time payoff is near zero, and the same
argument applies with more force to a measured cost - if the preset cannot decline the one item
with a measurement behind it, it has nothing left to decline. The tooltip states the consequence:
without the deposit, under open sky most update paths store nothing, so that preset thins the cache
to near nothing outdoors and `rtx.integrateIndirectMode = 0` is the honest setting there instead.

**The primary-deposit roughness guard was considered and declined.** Applying the footprint gate's
own criterion to the primary - `segmentLength` is the camera distance, so it cancels against the
voxel size - gives `sqrt(0.5*alpha^2/(1-alpha^2)) > 1/gridScale`, i.e. alpha > 0.028 at
`gridScale` 50, which is *below* the `minRoughness` floor of 0.05 the deposit already applies. So
the spatial criterion calls for no extra guard, the residual worry is angular and that test does not address it, the
artefact has not been seen by anyone, and `minRoughness` is already the control for it. A
primary-only floor would be a judgement number of exactly the kind `minRoughnessSpecular` was.

Reasoning, with the option-by-option reset table, the staleness arithmetic and the five alternatives
to a controller: [SHARC-adaptive-2026-09-16.md](SHARC-adaptive-2026-09-16.md). C++ only, no shader
recompiled; release build (two synchronous passes, `d3d9.dll` 280,178,176 bytes), integration
validator 61 PASS. Deployed to Fallout New Vegas, backup suffix
`backup-pre-perf-preset-primary-20260916-210436`. **Not deployed to Portal RTX** - it was running
throughout (`NvRemixBridge.exe` holding the DLL), so that install is still on the `ed39787a9` build
and needs the copy repeating once it is closed.

## Specular fireflies: the update budget scales with render resolution - 2026-09-16

The report was that `allowSpecularPaths` shows far more detail than the other samplers at low DLSS
presets but produces a lot of fireflies on reflective materials, and that only `updateTileSize`
helped, at a price. **Those are one fact, not two.**

**The update budget is resolution-dependent, from source.** The update dispatch is
`ceil(rayDims / tile)` (`indirect.cpp:788-793`) and `rayDims` is `m_compositeOutputExtent`, which is
`m_downscaledExtent` - the pre-upscale **render** resolution (`rtx_resources.cpp:1161`); the shader's
launch mapping uses the same `cb.camera.resolution` (`integrate_indirect.slangh:81-86`). So the
number of update paths falls with the DLSS preset, quadratically in the linear scale factor.

**And the path count is the divisor on every firefly.** The resolve blend pins `accumulatedFrameNum`
at `accumulationFrames` and renormalises (`SDK:925-948`), whose fixed point is
`accumulatedSampleNum = (N+1)k` for a cell fed `k` times a frame - so **one deposit of luminance `L`
moves a cell by `L/((N+1)k)`**. At 1440p, tile 8: 57,600 paths at DLAA against 6,420 at Ultra
Performance, so **an outlier is worth about 9x more in a cell at Ultra Performance**. Halving the
tile quadruples `k`, which is exactly why that worked. Both knobs move one denominator.

**`footprintGate` and `minSampleCount` were never tests of this.** `minSampleCount` bounds the sample
*count*; a cell holding an outlier has `9k` samples and sails past it - a count gate cannot see an
outlier in a mean. And `footprintGate`'s threshold collapses to a roughness test: voxel size is
`2^floor(log2 d)/gridScale` (`HashGridCommon.h:153-168`), so for a reflection whose segment length is
comparable to the hit's camera distance the distance cancels and the gate reduces to
`alpha > 1/gridScale`, i.e. **a launching perceptual roughness of about 0.17 at gridScale 50** - most
reflective materials pass. It is a directionality gate, not a variance gate. A third gate of that
family would not be one either, which is why the fix is not a gate.

**The SHARC SDK ships no outlier handling at all** - nothing disabled by this fork, nothing to
enable. The only radiance clamps in it are the float16 range clamps at pack time (`SDK:425-427`);
`SharcAddVoxelData` scales and atomically adds unbounded (`SDK:494-497`). Responsive lighting would
make it *worse* (a shorter accumulation window is a larger per-frame weight) and cache resampling
would propagate hot cells.

**One option added: `rtx.sharc.maxDepositLuminance`, default 0 (off)**, clamping the luminance of a
single deposit in `sharcFlushVertices` (`sharc_update.slangh:82-92`), the one deposit site on the
shipping deferred backend. It is the only remedy on the list that **costs no coverage** - it refuses
no lookup, rejects no surface and loses no cell, so it cannot take back the detail the option exists
for; the price is bias bounded by the threshold. Three ALU on a pass dispatched at 1/64 of the render
resolution. It is deliberately **not** in the clear condition, so it can be dragged live.

**A resolution-independent update budget was considered and declined.** Feasible in one line
(`updateTileSize` is one of the three options that do not clear the cache) and sound in principle - a
world-space cache's fill requirement is set by the scene, not the pixel count. But matching DLAA at
Ultra Performance means tile 3, about 7x the update pass, which is exactly the cost the user already
rejected, and it takes frame time away from whoever chose a low preset to get it. The honest form is
a per-preset `updateTileSize` table in `rtx.conf`, which is in the doc.

Also declined: a specular-only `minSampleCount` (its coverage cost lands precisely on the detail
being protected, and it bounds `n` rather than variance), a read-side clamp (64x the invocations,
leaves the cell hot so the artefact becomes a stable patch), an update-pass stats counter (needs a
new shader permutation family) and a relative clamp against the cell's own mean (a dependent read
into a 160 MiB buffer per deposit, against a ~0.1 ms budget).

Reasoning, the arithmetic and the ranked alternatives:
[SHARC-specular-fireflies-2026-09-16.md](SHARC-specular-fireflies-2026-09-16.md). Release build (two
synchronous passes, `d3d9.dll` 280,213,504 bytes), integration validator 61 PASS, resources validator
PASS, deferred and estimator validators OK, `RtxOptions.md` regenerated. (`test_graph_documentation`
fails for missing `GameValueRead*` golden docs - pre-existing and unrelated.) Deployed to **both**
installs, neither running, backup suffix `backup-pre-deposit-clamp-20260916-213514` - which also
clears the Portal RTX copy left outstanding by the previous section. **Nothing measured in game.**
