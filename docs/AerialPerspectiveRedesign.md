# Aerial perspective redesign investigation

Date: 2026-09-08. Scope: the current working tree's Numos aerial perspective path. Source inspection and analytical work counts only; no GPU capture, benchmark, shader build, or runtime changes were performed.

## Implementation follow-up: 2026-09-09

The investigation below describes the starting implementation. The following changes have now been implemented and checked:

- A separate visibility pass writes six sun visibility bits and a sky-probe count to an R32_UINT volume. Integration reads the mask without filtering or temporal reuse and retains the existing RGBA16F output and composite. The extra allocation is 4.5 MiB at 192 x 192 x 32, created only when this path is selected and resized with the AP dimensions.
- The split remains experimental and disabled by default. Enable `rtx.atmosphere.aerialPerspectiveSeparateVisibility = True`, or use **Separate Visibility Pass (Experimental)** under the atmosphere's aerial-perspective scene-shadow controls. Disabling it restores the fused tracing path.
- A dedicated no-query shader is selected when scene shadows are disabled, the opaque TLAS is missing, the no-trace diagnostic is selected, or the scene-shadow range is zero. The diagnostic's separate sun/sky behavior is retained. This shader has no ray-query capability or acceleration-structure descriptor.
- Dome-light frames skip AP generation because its only consumer, composite, already bypasses AP under the same condition.
- The AP sampler is retained instead of requesting it each frame. Separate visibility and integration GPU zones sit inside the existing AP timing zone.
- Shader declarations, the UI, and the two affected generated option rows were updated. Unrelated local option-documentation edits were preserved.

The initially implemented dispatch with one thread per depth interval changed some shadow-edge samples. Keeping the original loop-carried depth arithmetic in the visibility pass resolved those differences. The retained split therefore uses one thread per column in each pass. Sample-step calculation also retains the original loop-carried fraction rather than recomputing the preceding fraction independently. Neither a parallel prefix scan nor temporal/spatial visibility approximation is enabled.

### Verification

The release runtime built successfully with `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`. Four build jobs avoid the paging-space exhaustion encountered by the initial unrestricted parallel build. All dependent shaders compiled, and all four AP SPIR-V modules passed Vulkan 1.2 validation with scalar block layout.

`python scripts-common/validate_aerial_perspective.py` checks compiled SPIR-V validity, descriptor bindings, workgroup dimensions, and isolation of ray queries to the tracing variants. It requires the repository's existing external SPIR-V validator.

A copied and adapted standalone Vulkan harness is in `_Comp64Release/aerial-perspective-redesign/`. It compares actual GPU output against a shader compiled from the pre-change Git HEAD, using freshly generated constant-buffer offsets from that shader. It does not modify a game installation.

- Fused fallback: **155/155 bit-identical active-volume comparisons**.
- Split/no-query selection: **155/155 bit-identical active-volume comparisons**.
- Final zero-range selection, including inverted shadows: **10/10 additional bit-identical comparisons**.
- No nonfinite output values or Vulkan validation messages in those runs.

The 31-case suite covers day/night/sunset, disabled and diagnostic shadows, absent multiple scattering, zero illumination, zero/range-boundary shadows, 1 m and 20 m handoffs, moved/degenerate/wide frustums, Y-up/Z-up/flipped axes, 1/4/7/32/128 depth slices, odd XY size, altitude, and dense absorption. Each case runs against opaque triangles, non-opaque triangles, forced-non-opaque instances, triangles plus procedural geometry, and procedural-only geometry. Results are in `fused-final.txt`, `split-final.txt`, and the zero-range result files. These are synthetic shader tests; they do not establish game-wide visual parity or validate every runtime toggle sequence.

### Performance decision

On the tested NVIDIA GeForce RTX 5080 Laptop GPU, warmed 192 x 192 x 32 synthetic daytime measurements made the retained split approximately **18-26% slower**; night fixtures were approximately **21-32% slower**. It stays off by default. The no-query variant with shadows disabled was generally 3-7% faster, with one roughly 8% slower result, so its end-to-end speedup is not established. The dome-light skip eliminates this bake for eligible frames, but game-frame savings were not measured.

The timing logs are `benchmark_192.args.timings.txt`, `benchmark_night.args.timings.txt`, and `benchmark_shadows_off.args.timings.txt`. They use alternating warmed batches, include both split dispatches and their dependency barrier, and exclude readback. These small synthetic scenes are not a basis for claiming a general AP speedup. Keep the fused default and benchmark any experimental split on the target GPU and scene.

Consumer-depth pruning, visibility caches, and atmospheric transport caches remain future work. They require frame-order changes or additional validity/error handling and were not added merely to increase the amount of changed code.

## Recommendation

Prototype separating scene visibility from atmospheric integration while retaining the existing sample positions, volume resolution, and composite. Measure that against the current fused shader before adopting it. Follow with conservative elimination of work that cannot reach any consumer. These are the strongest candidates under a requirement to preserve the current image.

For larger ray-count reductions, investigate a persistent world-space visibility cache with fresh tracing on invalidation. This can preserve the feature set, but interpolated or temporally stale visibility cannot guarantee the same image. Treat it as a separately validated design, not an automatically equivalent replacement.

## What the current code actually does

Relevant files:

- `src/dxvk/rtx_render/rtx_atmosphere.cpp`: parameter setup around 989, volumetrics handoff around 1039, unconditional per-frame AP dispatch around 1498, resource creation and dispatch at 1741-1823.
- `src/dxvk/rtx_render/rtx_atmosphere.h`: authored AP defaults and controls at 154-245.
- `src/dxvk/shaders/rtx/pass/atmosphere/aerial_perspective_lut.comp.slang`: scene queries, six-direction sky visibility, and column integration.
- `src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_common.slangh`: ray setup at 2181, segment integration at 2235, frustum/depth mapping and fade around 2360-2440.
- `src/dxvk/shaders/rtx/pass/composite/composite.comp.slang`: application at 615-662.

The output is a camera-fitted 192 x 192 x 32 RGBA16F volume at source defaults. RGB stores cumulative in-scatter; alpha stores the mean of RGB transmittance. The camera frustum is square in texture coordinates, with its actual aspect encoded in the camera basis. Depth is exponential from the global volumetrics handoff (normally 20 m, floored to 1 m when necessary) to 32 km.

One thread already marches an entire column and writes successive cumulative slices. This avoids repeatedly integrating the whole prefix for every froxel. Recommending that conversion as a new optimization would miss the implementation already present.

Each nonempty interval uses six power-distributed atmospheric samples. Each sample can trace one sun ray. A separate set of six sky rays is evaluated once at the original interval midpoint, shared by the interval's six samples. Thus the interval costs up to twelve rays, not forty-two. Sky queries are skipped when multiple-scattering strength is zero; sun queries are skipped when the atmospheric solar term is zero. Both stop querying beyond their camera-distance limit, subject to the existing diagnostic-mode behavior.

The same range parameter also sets each visibility ray's maximum travel distance. The scene query uses opaque-mask geometry, forces triangle opacity, accepts the first hit, and skips procedural primitives. Alpha-tested triangles consequently behave as solid triangles here; glass and alpha-blended particles are excluded by mask. Matching generic surface shadow behavior would change this pass's current result.

The integrator preserves separate direct-sun and sky visibility. This matters: shadowed air can still receive multiple scattering outdoors, while a roof can suppress that ambient contribution indoors. The atmospheric phase function includes the AP-specific Mie anisotropy cap. It uses atmospheric transmittance and multiple-scattering LUTs, with planet/surface handling and numerical guards.

The composite takes one trilinear volume sample for eligible geometry pixels, applies the full-resolution near fade (50-250 m by default), and attenuates AP in-scatter by the near volumetrics transmittance. Sky misses and an active dome light bypass AP. The AP segment itself has no moon lighting or ground bounce. Cloud-shadow sampling is not called by this segment integrator. Those effects elsewhere in the renderer are not evidence that this volume already integrates them.

## Work and likely costs

At source defaults:

| Quantity | Analytical count |
|---|---:|
| Screen columns | 36,864 |
| Output froxels | 1,179,648 |
| Atmospheric samples, before empty-segment clipping | 7,077,888 |
| Output storage | 9 MiB |
| Absolute query ceiling if all intervals are in range | 14,155,776 |

For a center ray, a 20 m near bound, a 32 km far bound, and a 1 km trace range, the actual sample distribution permits 106 sun queries and 102 sky queries per column, assuming nonempty segments, nonzero solar terms, and enabled multiple scattering. Multiplying this representative column by all columns gives 7,667,712 queries; this is an illustration, not the actual frame count. Off-axis ray lengths, sphere clipping, lighting, user configuration, and modes change the count.

Likely costs are traversal, dependent work across 32 intervals, shader register pressure around ray queries, and repeated medium/lighting evaluation. The output's storage size alone does not establish a bandwidth bottleneck. Inspect occupancy, spills, texture throughput, and traversal timing before selecting an architecture. A dispatch with millions of rays is not proof that traversal dominates on every scene/GPU.

Some comments are stale: a header still refers to a 4 x 4 x 4 dispatch, descriptions mention a 32 x 32 grid, and the UI says scene shadows are off by default even though the option initializer is true. The executable path is 8 x 8 x 1 with one thread per column.

## Candidate 1: separate visibility from integration

Create a visibility kernel indexed by column and interval. Reconstruct exactly the current six sun sample locations and the original sky midpoint, retaining clipped integration bounds, query flags, masks, range tests, diagnostics, and sky direction orientation. Store six sun visibility bits and a sky unoccluded count from 0 through 6. Nine bits suffice for the production data; a 16-bit representation is 2.25 MiB at the full default grid, if supported efficiently, or use a 32-bit word for simpler storage.

The integration kernel remains one thread per column, reads visibility, and evaluates the same atmospheric equations in the same order. It writes the existing texture and leaves the composite unchanged. Explicitly order visibility writes before integration reads. Avoid accidentally tracing out-of-range intervals simply because the pass now has more parallel work.

This preserves the mathematical samples and can expose more independent traversal work while removing ray-query state from the integration kernel. It does not reduce the baseline number of rays. Extra dispatches, reads/writes, duplicated sample setup, and less favorable traversal coherence could outweigh the gain. The current code already hoists per-ray preparation and limits medium state across traversal, so the gain is uncertain.

Use proper shader variants for substantial optional tracing paths, following `docs/ShaderVariants.md`. Keep the reference fused implementation for comparisons. Verify equivalent numerical behavior rather than promising bitwise equality from a compiler/kernel refactor.

## Candidate 2: parallel interval integration plus cumulative scan

Compute each interval's local RGB in-scatter L and RGB transmittance T independently, then combine intervals in depth order:

`(La, Ta) composed with (Lb, Tb) = (La + Ta * Lb, Ta * Tb)`

This operation is associative in real arithmetic, enabling a parallel prefix scan. It avoids reintegrating prefixes while exposing depth parallelism. It does not reduce ray or medium-sample counts. Retain all three transmittance channels until the final output; averaging transmittance before composition changes colored extinction.

A separate full-grid scratch buffer containing six float32 values per interval is 27 MiB before alignment. A workgroup-per-column approach can keep intermediates in shared memory, but must handle the supported 4-128 depth slices and hardware subgroup widths. Benchmark both scheduling and memory consequences. Changed accumulation order introduces floating-point differences; half-precision intermediates introduce additional approximation.

This is a second experiment if profiling shows serial dependency or occupancy costs remain after visibility separation. It is not inherently faster merely because it is parallel.

## Candidate 3: conservative consumer-driven work reduction

The inspected path only samples AP for eligible primary geometry pixels. Build a conservative map of the deepest required AP slice per column from eligible surface depths. Propagate each pixel's demand to every XY volume column and both depth slices touched by trilinear interpolation. Integrate all preceding intervals for those columns, but skip the unused tails and columns.

This can remove both ray queries and atmospheric samples in interiors, close views, and sky-heavy views without reducing image quality. It needs a dispatch-order redesign because the current bake runs during ray-tracing argument setup, before the final surface-depth consumer data. Keep it before composite, use the same jitter/frustum/depth conventions, and audit every consumer before enabling pruning. Avoid stale texels being sampled after changes in camera or demand.

Full near-fade exclusion can eliminate output demand for nearby surfaces, but cannot remove the near integration prefix for farther surfaces: those surfaces still need all accumulated air in front of them. Likewise, a column depth cutoff cannot ignore neighboring columns sampled by the filter.

Also inspect a conservative whole-pass skip when no consumer can use AP, such as the dome-light bypass. Prove that the CPU predicate matches the composite predicate and that no other consumer needs the texture.

## Candidate 4: persistent visibility cache

Separate geometry-dependent visibility from atmospheric coefficients. Cache sun visibility and sky openness in a world-space grid, potentially cascaded, and evaluate atmospheric lighting from current parameters. Sky directions are fixed in world space for a fixed up-axis convention, so sky visibility is more reusable than sun visibility when time of day changes.

The cache needs invalidation for geometry, instance masks/opacity policy, moving doors/objects, sun direction (sun channel), up-axis changes, ray-range changes, scene resets, and newly exposed regions. Query reach matters: changing an occluder can invalidate probes far away along its shadow, not just probes inside the object's bounds. Without reliable scene-change information, conservative global invalidation may be necessary and can erase savings.

Fresh tracing at the requested location is the reliable fallback. Interpolating cached visibility across a thin wall can leak light even when every stored sample is valid. A few agreeing neighbors do not prove there is no thin occluder between them. Temporal reuse adds possible lag and ghosting; surface motion vectors alone do not describe the participating air along the ray.

This is the best candidate for reducing query counts across frames, but requires an explicit visual tolerance. A coarse cache or rotating one of the six sky directions per frame does not preserve the current deterministic instantaneous result.

## Candidate 5: cached atmospheric transport plus local shadow correction

The medium and atmospheric LUTs vary smoothly compared with geometry visibility. A more ambitious design stores unshadowed transport separately and computes a local correction where scene visibility is active. Keep directional single scattering and sky-gated multiple scattering separate; multiplying the final haze by one visibility scalar loses the current behavior.

At the existing quadrature points, linearity permits splitting unshadowed radiance and visibility corrections while keeping extinction unchanged. Resampling that transport from a lower-dimensional or lower-resolution cache introduces interpolation error. Parameterize altitude, view/sun geometry, interval distances, and all active atmosphere settings; do not assume a camera-fitted integral can be reprojected like a surface image.

Past the trace range, production scene visibility is one, so a separate unshadowed far path is possible. Preserve the current sky-midpoint rule in intervals straddling the range boundary and preserve the diagnostic modes. Splitting an interval at the range boundary and changing its samples would change numerical results. Never obtain a far segment by subtracting nearly opaque cumulative integrals and dividing by tiny transmittance without stability analysis.

This direction is appropriate if atmospheric evaluation remains expensive after traversal improvements. Hillaire's [reference project](https://github.com/sebh/UnrealEngineSkyAtmosphere) accompanies the EGSR 2020 LUT-based atmosphere technique and includes a path-tracing comparison. It is useful as an atmospheric reference, but the fork's indoor scene visibility must be preserved independently.

## Approaches that do not satisfy strict preservation by themselves

- Lower XY resolution or fewer depth slices: changes angular/detail resolution or depth interpolation.
- Fewer integration samples or one sun query for an entire column: loses depth-dependent visibility, including doorways.
- Drop sky rays or use one sun visibility factor for both terms: changes indoor haze and outdoor ambient fill.
- Replace ray queries with shadow maps: adds bias, resolution limits, and a different occlusion representation; also requires six-direction sky handling.
- Extend the near volumetrics grid and delete AP: the medium model, lighting, resolution, and compositing differ. Sharing infrastructure is possible, but this is a larger feature-parity project.
- Reproject the final cumulative AP volume with surface motion vectors: history refers to different air segments and can retain stale shadows.

## Validation and implementation order

1. Capture the existing `Atmosphere Aerial Perspective LUT` GPU zone on fixed camera paths; record GPU, render resolution, AP settings, TLAS complexity, and median/tail frame timings. Measure AP-enabled versus disabled frame cost as well as the zone itself.
2. Profile production tracing, no-trace diagnostics, and disabled scene shadowing to isolate costs. These are diagnostic measurements, not proposed feature removals. Inspect sun-only and sky-only cost in temporary instrumentation if needed.
3. Prototype visibility separation behind a variant. Compare its visibility data and output LUT against the reference at identical sample positions. Include total dispatch and barrier costs.
4. If worthwhile, prototype depth-demand pruning after confirming frame ordering and all consumers. Separately evaluate interval parallelism if shader latency still dominates.
5. Consider persistent visibility or atmospheric transport caches only after the exact-sample experiments establish the remaining cost and an acceptable error budget.

Test static and moving cameras; doors opening and closing; thin walls and alpha-tested foliage; roofed interiors; bright sun behind terrain; horizon views; dense Mie haze; sunset/planet shadow; Y-up, Z-up, and flipped up; changed FOV and aspect; dynamic atmosphere settings; volume resizing; zero multiple scattering; all scene-shadow diagnostics; missing TLAS; dome lighting; sky misses; disabled global volumetrics; and surfaces near handoff, fade, shadow-range, and far-volume boundaries.

Compare HDR in-scatter and transmittance separately before tone mapping, then final images and motion sequences. Set numerical/image tolerances before judging prototypes. Do not report a speedup or full visual parity until these measurements exist.
