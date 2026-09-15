# Lean SHARC parity with the full profile (2026-09-14)

Worktree `wsn3g`, branch `revised-9-10`, starting from `fb2d09e13`. Goal: close the five gaps listed in the dependency audit while keeping the lean profile faster than the full profile. No game deployment and no push were performed; all timing statements below are structural unless labelled as user measurements.

## What lean actually removes

Source comparison of the variant defines shows the lean stages are the full SHARC stages with `SHARC_LEAN_SECONDARY=1` and nothing else (the lean raygen also sets the opaque/translucent material mask, but the non-SER TraceRay raygen is byte-identical to the full one, so the raygen carries no feature code). The define changes exactly five sites:

| Gap | Site | Full-profile control |
|---|---|---|
| Unordered particles/decals | `path_state.slangh` (activation and SER hint), `nee_cache_light.slangh` | `rtx.enableSeparateUnorderedApproximations`, `rtx.enableUnorderedResolveInIndirectRays`, probabilistic selection (all default on) |
| Alpha-blended indirect shadows | `integrator.slangh` shadow mask | `rtx.enableIndirectAlphaBlendShadows` (default on) |
| RTXDI sample stealing | `integrator_indirect.slangh` | `rtx.useRTXDI`, `rtx.di.enableSampleStealing` (default on) |
| Secondary POM | closest-hit variant selection (`no_pom`), raygen `OPAQUE_MATERIAL_USE_POM` | `rtx.displacement.enableIndirectHit` (default off) and `getActivePOMCount() > 0` |
| Ray portals | hit/miss variant selection (`no_portals`), lean disabled when portals exist | portal texture hashes / active portals |

The FNV configuration snapshot (`_Comp64Release/fused-assembly-first-capture-rtx.conf`) leaves every one of those controls at its default, and `rtx.wboitEnabled` defaults to true, so the full profile runs the WBOIT unordered resolver there.

## Per-gap analysis

**RTXDI sample stealing (gap 4).** No SHARC stage binds the primary reservoir (binding 51): the closest-hit and miss sources define `RAB_HAS_RTXDI_RESERVOIRS` only when `!ENABLE_SHARC` (introduced by `e4158e0d5` to match the inline query, which never defined it because upstream `integrate_indirect.slang` does not). Without the define `RAB_LoadReservoir` returns `RTXDI_EmptyReservoir()`, so `sampleLightRTXDI` always returns false. The stealing branch sets `isWithinGbuffer = true` before that call, and the NEE-cache/RIS fallback runs only `if (!isWithinGbuffer)`. Consequence in the full SHARC profile with default options: an opaque secondary vertex whose reflectivity/roughness test passes, that projects on screen and matches the primary G-buffer surface, receives no NEE light sample and no shadow ray, in both query and update paths. Lean compiles the branch out and always falls through to the NEE cache or RIS. The cheapest correct restoration is therefore to make the full profile behave like lean: the branch is now compiled out wherever `ENABLE_SHARC` is set and reservoirs are unavailable. Legacy TraceRay stages still define the reservoirs and are untouched; the legacy inline RayQuery stages retain the upstream structure and are also untouched. Real stealing would need binding 51 in every SHARC stage; the option text for `rtx.di.enableSampleStealing` reports no visible quality gain and an 8% integrate-pass cost, so it is not restored.

**Secondary POM (gap 3).** The full profile already gates the POM closest-hit variant on `pomMode != Off && enableIndirectHit`, and `pomMode` is `Off` unless the scene has displaced materials. `enableIndirectHit` defaults to false, so in the default configuration this is not a difference. Restoration: lean POM closest-hit variants selected under the same gate, behind an opt-in option. Zero cost when the gate is off.

**Ray portals (gap 5).** Never present in FNV. Lean keeps its existing fallback to the full profile when portals exist (correct output, no speedup). Not restored in this pass; it would add 24 more hit/miss variants for a scene class the user does not run.

**Alpha-blended indirect shadows (gap 2).** The compiled difference is one mask bit on the secondary shadow ray; the cost is the any-hit work on `OBJECT_MASK_ALPHA_BLEND` instances the shadow ray crosses. Restoration: a lean tier with the bit compiled in, selected only when the option is on and the opaque TLAS contains alpha-blended instances this frame (counted on the CPU from the merged instance masks, including point-instancer batches). Zero cost in frames without such geometry.

**Unordered particles/decals (gap 1).** Restoration needs the separate unordered TLAS traversal (`resolveVertexUnordered`) at bounce <= 1 with the existing probabilistic selection, plus the SER coherence hint and, with WBOIT enabled, the WBOIT resolver. Restoration: a lean tier compiled with those sites, selected only when the unordered TLAS has instances this frame. That gate rarely fires in FNV outdoors, so the tier costs whatever the traversal costs; this is the one gap that may not be recoverable without giving up part of the speedup, and only a measurement can settle it.

**Ranking by visual impact over expected cost:** stealing (cost zero, fixes a full-profile defect), POM (zero when gated off), alpha-blended shadows (near-zero code, scene-gated), unordered (real traversal cost, weak gate in FNV), portals (no FNV impact, deferred).

## Step A: stealing branch guard (built and validated)

Change: `integrator_indirect.slangh` compiles the stealing branch out when `ENABLE_SHARC && !defined(RAB_HAS_RTXDI_RESERVOIRS)`, in addition to the lean define.

Compiled effect, `_Comp64Release/parity-snapshot-pre` (HEAD build) versus the new build:

| Stage | Full before | Full after | Lean |
|---|---:|---:|---:|
| query closest-hit, no portals, no POM | 736,736 | 702,112 | 700,016 |
| query miss, no portals | 736,028 | 701,404 | 699,308 |
| update deferred4 raygen | 816,588 | 782,228 | 779,828 |
| query TraceRay raygen | 81,124 | 81,124 | 81,124 (identical) |

The non-functional stealing branch accounted for about 94% of the code-size difference between the full and lean hit/miss/update stages. After step A every SHARC stage omits bindings 10 and 51 and no longer reads `enableRtxdiSampleStealing`. All 14 lean blobs, the legacy TraceRay closest-hit stages (which still bind 51 and read the flag), and the legacy inline RayQuery stages are byte-identical to the pre-change snapshot. Existing validators passed on the final DLL: `validate_lean_sharc.py` (14 lean variants, DLL embedding), `validate_sharc_integration.py`, `validate_sharc_resources.py` (70 stages embedded). Two compiles were needed to relink after the shader header regeneration. Logs: `_Comp64Release/parity-a-build.log`, `parity-a-link.log`.

Artifacts: `_Comp64Release/d3d9.parity-a.dll` (SHA-256 `5F996FCD0A50F8FE1EB08EB408F9A63AC5D1B8235A41F85752344B591A2B6B10`), post-A blobs in `_Comp64Release/parity-snapshot-a`.

Expected visible change in the full SHARC profile: bounced light from camera-visible secondary surfaces regains its direct-light (NEE) term. This is a code-path conclusion; it has not been observed in game. Lean output is unchanged by step A.

## Step B: lean feature tiers (built and validated)

Two opt-in options, both default off:

- `rtx.sharc.leanFeatureLevel` (0..2). Level 1 compiles the alpha-blended shadow mask bit back into the lean stages; level 2 additionally compiles the unordered particle/decal resolve back in (bounce <= 1, probabilistic selection and SER coherence hint as in the full profile, WBOIT resolver when `rtx.wboitEnabled`). The level is resolved per frame: 1 needs `rtx.enableIndirectAlphaBlendShadows` and at least one alpha-blended instance in the opaque TLAS; 2 needs the unordered runtime flags and at least one instance in the unordered TLAS. Counts come from the accel manager after the TLAS build (entries with mask 0 are ignored; point-instancer batches count their input size). Queries always follow the level; sparse updates follow it only with ray-generation deferred updates (the default backend) and otherwise stay at level 0. Changing the option resets the cache; scene-driven tier changes do not.
- `rtx.sharc.leanIndirectPom`. Selects displacement-aware lean closest-hit stages under the full profile's existing gate (`rtx.displacement.enableIndirectHit` and displaced materials in the scene).

Portal scenes still hand over to the full profile. The SHARC panel shows the requested level and the effective per-frame selection; the runtime logs `[SHARC] Lean profile: query level N, update level M...` once per distinct selection.

Shader variants: 28 added, 42 lean in total. Closest-hit 16 (four tiers x POM x statistics), miss 8, ray generation 2 (SER with the unordered hint), update 6 (deferred and deferred4 ray generation x three tiers). Selection lives in `rtx_pathtracer_integrate_indirect.cpp` (`LEAN_SHARC_TIERED`), the profile in `RtxSharc::resolveLeanProfile`, the counters in `AccelManager::buildTlas`.

Compiled sizes in bytes (closest hit without POM / miss / deferred4 ray-generation update):

| Tier | Closest hit | Miss | Update |
|---|---:|---:|---:|
| 0 (lean) | 700,016 | 699,308 | 779,828 |
| 1 (+ alpha-blended shadows) | 700,232 | 699,524 | 780,044 |
| 2 (+ unordered resolve) | 702,112 | 701,404 | 781,924 |
| 2 with WBOIT resolver | 694,120 | 693,412 | 773,932 |
| Full profile after step A | 702,112 | 701,404 | 782,228 |

Level-2 query closest-hit, miss and SER ray-generation stages are byte-identical to the full stages after step A; level-2 update stages differ from the full update only by the portal material mask (304 bytes), which lean deliberately keeps. After step A, therefore, the whole lean/full difference in FNV is the shadow mask bit (216 bytes of code) plus the unordered resolve (1,880 bytes) and the per-frame variant selection. Code size is no longer a credible source of the lean speedup; if level 0 stays faster than level 2 in game, the difference is the runtime traversal work of those two features, and the levels isolate each one.

Validation on the final DLL (`_Comp64Release/parity-b-build.log`, `parity-b-lean-validation.log`, `parity-b-integration-validation.log`, `parity-b-resources-validation.log`):

- `validate_lean_sharc.py`: all 42 lean variants pass SPIR-V validation, per-tier uniform use (stealing flag absent everywhere; shadow flag used from level 1; unordered flag from level 2; WBOIT compensation only in `_wboit`), displacement marker only in POM closest-hit stages, descriptors a subset of the full counterpart, bindings 10/51 absent, level-2 identity for query stages, statistics binding, a single ray-payload signature, and DLL embedding. Raygen parity holds for all four pairs. The 56 declared full SHARC stages are byte-identical to the step-A snapshot and read neither the stealing flag nor bindings 10/51. The legacy TraceRay closest-hit stage still binds the reservoir and reads the flag, and four legacy blobs are byte-identical to the pre-lean DLL.
- `validate_sharc_integration.py`: 59 contracts pass. `validate_sharc_resources.py`: 98 stages embedded, foreign descriptors absent.
- Twelve stale `_resample` blobs from the removed resampling experiment remain in the build directory; they are not declared, not embedded, and excluded by name.
- `RtxOptions.md` regenerated by a headless factory probe in `_Comp64Release/parity-options-probe` (fresh `LOCALAPPDATA`, automation flag cleared, repository `public/bin` on PATH for dependencies, no device or game created). The regeneration also refreshed seven atmosphere rows that were stale in the committed file.

Artifacts: `_Comp64Release/d3d9.parity-b.dll` and `.pdb` (copies of `src/d3d9/`), SHA-256 `AB8AE8EE418AEF4198127EEDDD493AD19D3D3D5C1044E859630E3EE05C9889C8` / `2608773EC190C8D2B6433F677BAFA6C2A0160EF8C1615CAB98E61207E35BDBC5`. Not deployed. No runtime, image or timing test was performed; every performance statement above is structural.

## Deliberately left

- Ray portals (gap 5): 24 more hit/miss variants for a scene class absent from FNV; the existing full-profile fallback is correct.
- Real RTXDI sample stealing in SHARC: would need binding 51 in every SHARC stage; the option's own text reports no visible gain and an 8% integrate-pass cost.
- Upstream inline RayQuery stages (`integrate_indirect_rayquery*`) keep the same stealing fallback structure; untouched, byte-identical.
- Restored tiers for compute and non-deferred update backends (diagnostic comparison backends only).

## A/B procedure for FNV

Same save and view for every run; hold still 10 s or more after each change so the cache and history settle; `rtx.sharc.measureGpuTime = False` for final numbers; `rtx.profile.gpuStages` optional for the IndirectIntegration interval.

1. Step A on the full profile: `rtx.sharc.leanSecondary = False`, new DLL versus the currently installed one. Expect bounced light from camera-visible surfaces (sunlit ground onto walls, floor onto walls) to be brighter, not darker; no speed change is claimed.
2. Tiers: `rtx.sharc.leanSecondary = True`, then `rtx.sharc.leanFeatureLevel` 0, 1, 2 in turn (restart between runs, or use the SHARC panel; the cache resets). Confirm the panel line "Effective: query level N" matches the requested level in that scene. Level 2 runs the full profile's shaders; 0 to 1 isolates the alpha-blended shadow cost, 1 to 2 the unordered resolve cost.
3. Visual checks: level 1 against level 0 in rooms lit through blended glass or effects; level 2 against level 1 in reflections of smoke, sparks or decals (puddles, glass, wet floors). Level 2 against full should be indistinguishable.
4. Cross-check without lean: full profile with `rtx.enableUnorderedResolveInIndirectRays = False` and `rtx.enableIndirectAlphaBlendShadows = False` should perform like lean level 0; if it does not, the remaining difference is variant selection rather than shader content.
5. POM only if `rtx.displacement.enableIndirectHit = True` and displaced materials are present: add `rtx.sharc.leanIndirectPom = True`.
