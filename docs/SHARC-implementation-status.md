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
