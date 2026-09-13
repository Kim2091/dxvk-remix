# SHARC-first renderer: feature-preserving first step

## Contract

Full-feature SHARC is the quality baseline. The existing lean-secondary profile remains a separate approximation experiment. Removing particle/decal resolve, POM, or indirect shadows is not part of this redesign. SHARC is the preferred indirect-lighting architecture, not a replacement for visibility, direct lighting, materials, scene preparation, or reconstruction. NRD spherical-harmonics integration remains a separate research item; no setting or denoiser was changed.

## Implemented

Added a SHARC-specific base resource layout in rtx_pathtracer_integrate_indirect.cpp. Query and update variants no longer inherit 11 NRC descriptors or three ReSTIR-GI descriptors. While SHARC is effectively active, its indirect dispatch skips the NRC and ReSTIR-GI resource-binding helpers. When SHARC falls back, the original helpers still execute. This uses effective activation, not the configured option.

No shader source, sampling rule, material behavior, resolution, cache update rate, or pass ordering changed. Existing uncommitted lean-only resource filtering remains in place as a separate layer. No producer allocation was removed. The update class previously removed three ReSTIR-GI outputs already; its new base additionally removes the 11 NRC slots.

## Audit findings

- Current renderer includes reference 66 SHARC stages. All omit all 14 foreign resource bindings and are embedded in the final DLL. The build folder contains 78 matching blobs, including abandoned variants; the persistent validator selects current source includes rather than accepting stale files as deployed shaders.
- NRC and ReSTIR-GI dispatches already have inactive guards. Their presence in source does not imply their rendering passes run alongside SHARC.
- SHARC update -> cache resolve -> query is a real data dependency; merging/removing the resolve requires a separate algorithm and synchronization analysis.
- Primary RTXDI still needs lighting reservoirs. Unordered geometry still serves primary particles/decals. Those shared producers cannot be removed merely because SHARC is selected.
- The final indirect NEE pass still assembles lighting outputs. It is not established as redundant.

## Validation

Release build passed (sharc-resource-build.log). Existing SHARC compiled contracts passed (sharc-resource-contracts.log). scripts-common/validate_sharc_resources.py verifies all 66 current stages against the removed resource set and checks byte-for-byte embedding in the linked DLL. Git diff whitespace check passed. No game frame or live mode transition has been tested, and no performance improvement is claimed. Build may retain existing third-party symbol/delay-load linker warnings.

## Next decision gate

Measure full-feature SHARC with leanSecondary disabled, holding scene/camera, resolution, reconstruction, and cache settings fixed. Collect total frame time and existing SHARC Update/Resolve/Query timings; use GPU pass timings for G-buffer, RTXDI, NEE cache, indirect assembly, reconstruction, and scene preparation. Compare current resource specialization to the previous full-feature build. Check motion, glass, particles, decals, displacement, dynamic lights, and fallback transitions.

Select the next producer/consumer change based on the dominant measured cost: eliminate a proven-unused write or copy first, then consider pass fusion or narrower buffer formats only with a complete live-consumer map. Do not delete other integrators for an assumed GPU gain. NRD SH needs directional signal production, matching resource bindings and reconstruction support before it can be evaluated fairly.

This first step is built locally, not committed, pushed, or deployed. The FNV install retains the previously deployed lean renderer.

## Subsequent test deployment

The local-only statement above describes the first build. The follow-up adds a dedicated indirect-lighting scheduling entry point and is now deployed to FNV. See [test build and boundary](SHARC-first-test-build-2026-09-13.md) for current validation, backup information, and comparison instructions.

## Indirect subsystem ownership (2026-09-13 follow-up)

The main context now calls `DxvkPathtracerIntegrateIndirect::dispatchLighting` once after direct integration and RTXDI gradients. The subsystem owns effective SHARC selection, sparse update, cache publication, full-resolution query, and NEE assembly. Low-level dispatch and assembly methods are private, preventing external callers from bypassing their required ordering. Legacy fallback remains intact. GPU timing labels and their boundaries are preserved.

Entry prerequisites are current-frame GBuffer/material data, the sparse active-pixel mask, direct-generated ray/lobe inputs, scene acceleration structures and the prepared NEE cache. Exit outputs include assembled primary indirect diffuse/specular signals and raw indirect radiance/hit distance consumed by secondary demodulation. Cache training, query statistics and timing hooks remain inside the sequence.

This is a scheduling boundary, not yet an isolated resource interface: the existing RaytracingOutput aggregate and common scene bindings still cross it. No allocations, shader dispatches, samples or material features were removed. No performance gain is expected from this ownership change alone. SHARC is still an indirect-lighting implementation, not the entire renderer.

Remaining overhaul work:

1. Narrow the resource interface using actual shader bindings and consumers; distinguish shared inputs, lighting outputs, scratch storage and backend history. Preserve raw secondary radiance until its consumer is explicitly migrated.
2. Use that interface to evaluate a SHARC-specific query/assembly implementation. Preserve emissive-triangle NEE, visibility/MIS, lobe splitting, feedback and sparse/secondary behavior; assembly is not a copy pass.
3. Optimize measured query work before cache termination only where material eligibility and special-geometry handling remain equivalent. Update/query merging remains unproven (see SHARC-update-query-reuse-2026-09-13.md).
4. Validate image behavior and warmed paired timings before describing any resulting change as a quality-preserving speedup. Keep the full-feature comparison path.

No new game deployment is needed solely to measure this scheduling refactor; the installed plain-NEE candidate remains the comparison build.
