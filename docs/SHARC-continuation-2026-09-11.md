> Superseded on 2026-09-12: experimental implementation and Release build are complete in `wsn3g`. See `C:/Users/sparkles/Projects/Fable_5_testing/wsn3g/docs/SHARC-implementation-status.md` for current state. Runtime GPU/image/performance validation remains unperformed. The checkpoint below is historical.

# SHARC implementation checkpoint — 2026-09-11

User requested pausing because usage was almost exhausted. Resume from this checkpoint; **implementation is incomplete and has not been built or run in game**. Do not treat the SDK smoke tests as full renderer validation.

## Task, authorization, and working directory

User requested extensive research by Luna agents into NVIDIA SHARC, a design for `revised-9-10`, then authorized implementation with “proceed.” Primary should own estimator/scheduling decisions and use Luna for bounded implementation/audits. Latest instruction is to pause and save progress. Resume implementation only when user returns.

- Actual target: `C:\Users\sparkles\Projects\Fable_5_testing\wsn3g`, branch `revised-9-10`, starting commit `fd4cc524ffd8a2da61794f89e05768aa187c171a`.
- Requested `dxvk-remix` directory is actually on `volumetrics/ap-local-lights` with unrelated dirty scene-manager/UI changes. Do not overwrite those changes. Only the design, validator, and this checkpoint were intentionally placed there.
- Target `wsn3g` started clean; current implementation edits are uncommitted. No deployment or commit performed.
- `wsn3g` is outside writable roots. Escalated `exec_command` calls have been auto-approved for this worktree. Use temporary Python/PowerShell scripts in a writable location, then execute with `sandbox_permissions: require_escalated` to edit/copy into target. Do not bypass sandbox.
- Original host implementation script: `%TEMP%\implement_sharc_host.py`. It is one-shot with assertions; **do not rerun**.

Read target `AGENTS.md`, `.agents/skills/build-remix/SKILL.md`, `docs/ShaderVariants.md`, shader README, and contributing style. Announced build skill previously. Mandatory build command is **meson compile**, never ninja directly. Subsystems derive CommonDeviceObject, owned by DxvkObjects; options belong to subsystem; significant shader features use compile-time variants. Core DXVK divergence outside rtx_render needs NV-DXVK comment blocks. No renderer globals.

## Research artifacts and decisions

Full design: `docs/SHARC-design-revised-9-10.md` in both repositories. It still needs updating to actual implementation choices; some sections reflect research proposals rather than final decisions.

- Upstream SHARC 1.8.3 pinned at `4e21b585c33c83d723ca9a1e11bbb1090d145793`.
- Downloaded research: `%TEMP%\codex-sharc-research-20260911\source`; RTXGI sample at sibling `rtxgi`, pin `10b5770b8eaddfc1faab82b65f799ac6f47dcc44`.
- RTXGI sample uses obsolete SHARC field names. Follow vendored SDK, not sample blindly.
- SDK is shader-only. Frame order **Update -> Resolve -> Query**, same frame, not NRC's post-render training slot.
- Use first indirect hit (bounce index 1) as first inserted surface. G-buffer root only seeds update rays. Avoid primary deferred NEE dependency.
- Cache stores local outgoing radiance, not camera-throughput-weighted lighting. Preserve inline secondary NEE, resolver attenuation/emission, sky contribution, BSDF continuation, and emissive MIS.
- Initial chosen backend: dedicated compute RayQuery update/query. UI says experimental compute path. Normal configured backends remain for non-SHARC/fallback.
- Initial chosen hash synchronization: **64-bit buffer atomics**, optionally enabled device capabilities; unsupported devices fall back. Lock-buffer alternative was tested but is not production configuration. Do not add a lock buffer without changing shader/host layout together.
- SHARC config: separate emission on, no resampling, propagation depth 8, finite update paths at most 8 bounces. Initial opaque/rough/diffuse eligibility, no querying emissive/glass/delta/SSS/portal paths.
- Hash/accumulation/resolved strides: 8/16/16 bytes. 2^21 entries = 80 MiB; 2^22 = 160 MiB. Native fp16 storage required.
- Current temporal proposal: 8 accumulated frames, stale eviction 32; global history/config/activation reset. Weather/light/material change policy still needs review.

## Completed target changes

Added files:

- `src/dxvk/shaders/rtx/external/sharc/`: five unmodified upstream headers, License.md, pin README.
- `pass/sharc/sharc_args.h`: 64-byte shared SharcArgs (camera/current previous, capacity, gridScale, accumulation/stale/update tile/update bounces, radianceScale, minRoughness, enabled, debugMode).
- `pass/sharc/sharc_binding_indices.h`: hash 200, accumulation 201, resolved 202.
- `pass/sharc/sharc_sdk.slangh`: atomics1, separate emissive1, resampling0, depth8.
- `pass/sharc/sharc_bindings.slangh`: three RWStructuredBuffers, makeSharcParameters from cb.sharcArgs.
- `pass/sharc/sharc_resolve.comp.slang`: compute256, bounds check, current SDK resolve parameters.
- `rtx_render/rtx_sharc.cpp/.h`: CommonDeviceObject, prepareFrame, bindResources, dispatchResolve, UI, support gate, buffers and reset logic. Support requires int64+buffer atomics+fp16+storage16+rayquery. Allocation failure and unsupported WBOIT/portals/OMM fall back. Verify all APIs by building.
- `scripts-common/validate_sharc.py`: standalone SDK compile/SPIR-V/layout contract test.

Modified:

- `dxvk_objects.h`: RtxSharc include/accessor/Lazy member.
- `dxvk_device.cpp`: DxvkObjects constructor initializes m_sharc(device); constructor is here, not dxvk_objects.cpp.
- `dxvk_adapter.cpp`: optionally copies shaderInt64 and shaderBufferInt64Atomics support into enabled features.
- `rtx_options.h`: SHARC enum3/documentation.
- `imgui/dxvk_imgui.cpp`: combo option and subsystem settings.
- `meson.build`: new cpp/h registration.
- `pass/raytrace_args.h`: include and sharcArgs after nrcArgs.
- `rtx_context.cpp`: one prepareFrame call after NRC constants setup. **No SHARC dispatch scheduling yet.**

Luna shader entry implementation completed immediately before pause:

- `pass/integrate/integrate_indirect.slang`: dedicated variants `integrate_indirect_sharc_update.comp` (ENABLE_SHARC=1 SHARC_UPDATE=1 NEE_CACHE_ENABLE=1) and `integrate_indirect_sharc_query.comp` (ENABLE_SHARC=1 SHARC_QUERY=1 NEE_CACHE_ENABLE=1).
- `integrate_indirect_bindings.slangh`: conditional SHARC binding include.
- `integrate_indirect.slangh`: update launch ceil tile mapping, rotating source pixel, bypass NRC/sparse remapping, root miss returns without output write. Root ray/sample/throughput data preserved.
- Agent reports `git diff --check` passed. **Full shaders not compiled. Successful update paths still reach ordinary integrator output writer; must fix before use.**

## Known concrete bugs / unfinished work

1. **Enum Count collision exists.** Direct read at pause shows:
   ```cpp
   ImportanceSampled = 0,
   ReSTIRGI = 1,
   Sharc = 3,
   NeuralRadianceCache = 2,
   Count
   ```
   Count implicitly becomes 3. Move Sharc after NRC or explicitly Count=4. Host Luna incorrectly reported no collision; trust file, not that report.
2. `rtx_pathtracer_integrate_indirect.cpp/.h` remain unchanged. Need dedicated managed compute shaders, bindings metadata, update/query dispatch selection and dimensions.
3. Context dispatchIntegrate is still direct -> gradient -> indirect -> NEE. Add active SHARC update -> resolve -> query between gradient and NEE.
4. `algorithm/path_state.slangh` and `algorithm/integrator_indirect.slangh` remain unchanged. No estimator/cache lookup/update integrated yet.
5. SHARC update must compile out all indirect output writes, ReSTIR writes/stealing, NEE feedback task writes, debug writes (including RESOLVE_RAY_TRACE debug argument), and NRC outputs. Retain NEE sampling and emissive candidate lookup for MIS.
6. Root throughput texture aliases final indirect output. Update dispatch must not bind/mark aliased indirect output Write; preserve root data for query. Current legacy dispatch binds throughput Read then indirect radiance Write LAST.
7. SDK adapter needs explicit defaults (notably SHARC_ENABLE_RESPONSIVE_LIGHTING=0) to suppress undefined macro warnings; ensure update/query macros aren't accidentally active in other variants.
8. UI exposes debug modes 1/2 but no debug implementation yet. Implement them or remove unsupported choices.
9. Render-target traversal, unsupported material transitions, finite depth, temporal invalidation, shader device capabilities and synchronization need review.
10. No integrated shader/C++ build or game validation. Current tree is a partial implementation, not a usable feature.

## Estimator implementation map / candidate approach

Primary had not written estimator code at pause. Last reasoning explored two equivalent schemes; choose deliberately and verify numerically rather than claiming already implemented.

SDK behavior inspected directly:

- SharcInit sets pathLength=0.
- SharcSetThroughput multiplies **all existing per-vertex sampleWeights** by supplied segment weight.
- SharcUpdateHit inserts cell, adds local direct with sample count1, propagates direct+separate emission to previous vertices, pushes new cell with weight1. With resampling off, insertion failure is the main false-return case; upstream documentation says terminate on false.
- SharcUpdateMiss adds radiance*stored weight to each existing vertex with sample count0; it does not change/clear path state. Thus it can represent additive contributions, not only terminal environment.

Possible clean adaptation: hook accumulateThroughput to additionally call SharcSetThroughput with each local attenuation/continuation factor, and hook accumulateRadiance(PathState, raw radiance) to call SharcUpdateMiss. Keep full path throughput only for ordinary logic, never multiply SHARC input by it. After current surface emission/MIS is accumulated (therefore propagated only to previous vertices), insert eligible current cell with direct0/emissive0 before local NEE; subsequent NEE through the radiance hook contributes to current and previous vertices, once. Sample count is incremented only by insertion. This is algebraically equivalent to passing local direct to UpdateHit, avoids camera-prefix contamination, and handles skipped/ineligible surfaces as contributions to older cells. Must review insertion-failure policy and all special accumulation paths carefully.

Alternative: collect local NEE in a separate variable, exclude it from generic SHARC propagation, call UpdateHit after NEE with that direct value; for ineligible surfaces propagate the collected NEE via UpdateMiss. Current emission already propagated separately must then use hit.emissive=0 to avoid double count. Do not combine both schemes.

Query: after resolver emission/attenuation, medium handling, render-target handling, continueResolving/miss checks and current emissive MIS, but before local NEE/direction sampling. On eligible nonemissive rough opaque incoming diffuse ray, query cell only when segment distance > sqrt(3)*voxelSize. Cache success accumulates cache radiance through normal path throughput, sets continuePath=false, returns vertex function; final output writer still executes. Never terminate entire shader before output.

Relevant locations in `algorithm/integrator_indirect.slangh` (line numbers approximate before edits):

- pathStateCreateEmpty ~50; accumulateThroughput ~87; accumulateRadiance overloads ~99.
- integratePathVertex ~213; unordered NEE task insert ~279; resolver ~300.
- ReSTIR geometry #if !ENABLE_NRC ~321; also gate SHARC.
- Sky miss ~372; resolver radiance then throughput ~413.
- Medium attenuation ~435; render-target special traversal ~467.
- materialType/emissiveLight ~487; task insert ~502, **retain cell.searchCandidate for MIS**.
- continueResolving ~520; miss return ~526; current emission MIS ~535.
- sampleDirection ~570; overwrites skyGatherEligible ~604 (capture incoming eligibility before this).
- opaque interaction ~615; NEE ~620..833; feedback insert ~818; accumulateRadiance(neeLight) ~823.
- termination ~850; roulette ~870 (disable for initial finite updates).
- ReSTIR stealing ~906; continuation throughput accumulation ~1039.
- ReSTIR helpers also guarded !ENABLE_NRC ~1049,1107,1285; include !ENABLE_SHARC.
- integrateIndirectPath ~1302: initializes G-buffer data; validates root PDF/throughput; initNrc/initReSTIR; loop ~1420; output block ~1460; debug output after it.

PathState: SHARC_UPDATE-only SharcState member; bounded pathMaxBounces uses cb.sharcArgs.updateBounces, disable roulette for update. Empty SHARC state makes camera-prefix weight irrelevant, but preserve root validity checks. Avoid introducing zero/full-path-throughput termination biases into cached local targets. Separate unordered approximations need explicit treatment.

Host map: managed IntegrateIndirectRayGenShader parameter block reused for legacy compute. Prefer dedicated SHARC subclass/metadata so non-SHARC variants have no cache descriptor overhead. bindResources exists. dispatch currently ~397 and mode switch ~500. Update dimensions ceil(render/tile); normal query full size. Existing shader compiler may require `-capability spvInt64Atomics` for update/resolve; prove with actual compilation. `RtxOptions::useReSTIRGI()` only true for ReSTIR mode, but still compile-gate shader side effects.

## Validation and available toolchain

Configured target builds **do exist**: `_Comp64Release`, `_Comp64UnitTest`. Packman junctions looked missing without elevation; they work elevated.

- `external/slang` -> `C:\packman-repo\chk\rtx-remix-slang\slang-2025.10.4-39-g496fab196-win64-remix`.
- Actual `external\slang\slangc.exe -version`: **2025.10.4-39-g496fab196**.
- `external/spirv_tools` available too; inspect paths for executables.
- Meson: `C:\Users\sparkles\AppData\Roaming\Python\Python314\Scripts\meson.exe`.
- Build elevated from target: `& <meson path> compile -C _Comp64Release`; skill lists shader target `rtx_shaders` (verify actual target if needed).
- VS2022 Professional installed. meson compile activates MSVC environment.
- Fallback SDK tools: `C:\VulkanSDK\1.4.357.0\Bin`, newer Slang2026.13.1. Prefer actual Packman tools.

Validator copied into both repositories. Fixed relative output path resolution (compilation uses temp cwd), uses spirv-val `--scalar-block-layout`, checks lock stride only in lock update because other passes don't reference lock resource.

**All nine SDK contracts passed using actual Packman compiler and pinned headers:** atomic64 update/resolve/query; lock update/resolve/query; separate-emissive update/resolve/query. These validate SDK API/SPIR-V/layout only. Earlier newer-toolchain tests also passed. Full renderer compilation has not been attempted.

Validator CLI requires `--slangc`, `--spirv-val`, `--spirv-dis`, `--include`, `--output`; use script help. Last output location `_Comp64Release\sharc-validation`. Test meaningful estimator energy accounting, disabled-mode shader isolation, resource layouts/barriers, and real image behavior separately.

## Agent checkpoints

- `/root/remix_host_integration` (Luna): research complete; last host implementation assignment paused **without edits or build**. Its enum no-collision report is wrong (verified above). Its renewed preference for locks is an old recommendation, not current decision.
- `/root/remix_path_integration` (Luna): entry/variant files implemented as described; paused after diff check. No algorithm integrator edits.
- `/root/sharc_upstream` (Luna): validator complete, copied to target, nine actual-toolchain passes. Asked to audit/fix only rtx_sharc cpp/h and SDK adapter; stop requested during audit. Check its final message for late changes before resuming.

Recommended resumption: inspect git diff and latest agent notes -> fix enum -> implement core estimator and host scheduling -> compile full SHARC shaders -> full meson build -> audits/tests -> update design and report runtime limitations. Do not redeploy or launch games implicitly from this checkpoint.

Final upstream-agent checkpoint received: no further source edits during audit. Confirmed missing scheduler/estimator, unused updateBounces/minRoughness/debugMode, correct 8/16/16 layout, and need explicit SHARC_ENABLE_RESPONSIVE_LIGHTING=0 and SHARC_ENABLE_SH_ENCODING=0 in adapter. All three agents are paused.
