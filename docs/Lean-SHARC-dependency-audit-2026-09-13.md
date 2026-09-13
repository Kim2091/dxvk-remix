# Lean SHARC: dependency audit and visual limits (2026-09-13)

Baseline implementation committed/pushed as `2274a1247` on `personal/revised-9-10`. User reports 3–4 ms improvement in FNV, scene dependent, with no apparent quality regression.

## First dependency cleanup

Inspected all 14 compiled lean shader stages. None declares binding 10 (previous-frame light data) or 51 (primary RTXDI reservoir). Lean query/update pipeline classes now filter these two descriptors out of their inherited resource declarations. Full-profile declarations remain unchanged. The validator asserts their absence in every lean stage, including update stages. This removes unnecessary declared dependencies; runtime timing is needed to determine any actual descriptor/synchronization benefit.

Do not remove the underlying allocations or their producers: primary RTXDI/direct lighting still consumes its reservoir and light history. The unordered scene is shared with primary particle/decal rendering. RTXDI gradient dispatch already skips when neither denoiser gradients nor best-light sampling needs it. NEE cache maintenance and indirect NEE assembly still have live consumers. No whole frame pass has been established as redundant merely because lean mode is active.

The next broader audit should map full-frame resource consumers for the actual FNV denoiser/cache configuration before gating any producer or allocation. Disabling a shared producer based solely on the selected lean option would also be wrong during SHARC fallback.

## Exact code conditions versus observable symptoms

These are source-established differences, not a claim that every example below has been reproduced in FNV. There is no exhaustive known list of scenes that will crash or visibly fail.

- **Unordered particles/decals:** full mode must have separate unordered approximations and indirect unordered resolve enabled, and a secondary segment must encounter an affected object (full mode normally considers this on bounce iteration <= 1, optionally probabilistically). Lean skips that entire extra resolver. Objects routed only into the unordered acceleration structure are not recovered by ordinary ordered tracing. A reflected smoke puff, spark, or decal can therefore disappear or lose its attenuation/material/emissive contribution through this path. Other light-sampling paths can still contribute illumination; this is not a guarantee that all particle light vanishes. Objects still handled by ordered resolving are not categorically removed.
- **Alpha-blended indirect shadows:** when indirect alpha-blend shadows would otherwise be enabled, lean excludes OBJECT_MASK_ALPHA_BLEND from the secondary next-event-estimation shadow ray. An occluder classified into that mask between a secondary surface and a sampled light no longer blocks that sample. A reflected surface or bounced illumination may be too bright. Opaque/cutout masks and the separate translucent-shadow option remain; this is not removal of all glass shadows or foliage.
- **Secondary displacement:** when the full path would evaluate POM on a material hit during secondary tracing, lean uses the non-POM material variant. Height-based texture-coordinate/depth detail can disagree between the main view and a reflection, particularly at grazing angles. Normal maps and actual geometric displacement are not removed by this change.
- **RTXDI sample reuse:** when RTXDI and its sample-stealing option are enabled and the roughness/reflectivity and screen-space correspondence tests pass, the full path can reuse a primary reservoir sample. Lean instead follows the existing NEE-cache/RIS selection fallback. Difficult lighting may converge differently or exhibit more noise, but the light is not deliberately deleted. Visible worsening is not guaranteed.

## Fallbacks, not deliberately broken scenes

Ray portal presence (active portals or configured portal texture hashes) always disables lean. Without the SHARC portal override, SHARC itself falls back; with it, full SHARC is used. Active raytraced render targets and unsupported SHARC device features use ordinary importance-sampled paths. WBOIT and opacity micromaps block SHARC unless their respective experimental overrides are enabled. Allocation failure also prevents lean activation. These conditions can lose the speedup, not prove a rendering failure. Enabling an experimental override does not establish exhaustive compatibility.

No lean-specific in-game crash is known from the testing reported so far. Cache lag, temporal reconstruction artifacts, and finite-update bias can also occur in full SHARC and should not automatically be attributed to this lean change.

## Validation

Release build succeeded. All 14 lean shader contracts passed against the final DLL, including absence of bindings 10 and 51; four representative baseline shader blobs remain byte-identical to the pre-lean DLL. Git diff whitespace check passed. Logs: _Comp64Release/lean-sharc-dependency-build.log and _Comp64Release/lean-sharc-dependency-validation.log. This follow-up is local and not deployed; no performance gain has been measured for it.
