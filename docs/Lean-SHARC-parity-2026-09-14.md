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
