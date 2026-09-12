# SHARC experimental integration — 2026-09-12

Implemented in the `wsn3g` worktree on `revised-9-10`. Changes are uncommitted. The original requested `dxvk-remix` checkout is a different branch; its unrelated changes were preserved.

## Use and implemented scope

Select **SHARC** in the indirect integration mode UI, or set `rtx.integrateIndirectMode = 3`. Existing defaults are unchanged. Settings are under `rtx.sharc`: `capacityLog2` (21 = 80 MiB), `updateTileSize` (5), `updateBounces` (8), `accumulationFrames` (8), `staleFrames` (32), `gridScale` (50), and `minRoughness` (0.8). The settings UI reports support/fallback status and offers Reset SHARC.

The device-owned cache uses pinned SHARC 1.8.3 (`4e21b585c33c83d723ca9a1e11bbb1090d145793`), 64-bit atomic hashes, and native half-float resolved storage. Required features include shader Int64, buffer Int64 atomics, float16/storage16, and RayQuery. Unsupported devices trace ordinary paths. WBOIT, ray portals, opacity micromaps, and active raytraced render targets also fall back. The initial backend is explicitly compute RayQuery.

Frame sequence: direct integration and gradient -> sparse cache update -> cache resolve -> full-resolution indirect query -> existing primary NEE/output assembly. Dedicated shader layouts use reserved bindings **230–232**; the branch's atmosphere owns 200–217. Explicit read/write cache access metadata uses DXVK's normal compute hazard tracking, including configurations that skip pure write-after-write barriers. The sparse update does not write the aliased indirect output, ReSTIR outputs, NEE feedback tasks, or debug image.

The primary surface only seeds rays. Cache insertion starts at eligible indirect surfaces: incoming diffuse ray, rough opaque material, full opacity, no medium, no diffusion-profile SSS, and no current emission. Queries additionally require a segment longer than the cache voxel diagonal. Cache misses continue normal path tracing. Query hits add cached reflected lighting through the ordinary path throughput and terminate before local NEE, preserving final output writing.

Each inserted cache vertex starts with weight one. Local NEE and later radiance contributions propagate through SHARC's stored segment weights, independent of the camera prefix. Current emission/MIS is propagated before insertion and remains explicit in rendering. Attenuation and continuation weights multiply retained cache weights. With cache resampling and responsive lighting disabled, hash insertion overflow preserves existing vertices and continues their estimate. Updates use finite paths of at most eight bounces and disable roulette; propagation depth is eight. This remains a spatial/temporal approximation with finite-depth bias.

History clears on renderer history reset, activation/frame discontinuity, capacity/grid/roughness changes, update-bounce changes, and explicit Reset. Moving geometry/light/weather changes rely on temporal replacement and stale eviction unless they trigger the renderer's history reset. There is no new automatic complete scene-change detector.

## Verification and limitations

Final Release compilation and linking succeeded (Meson exit code 0). Final build log: `_Comp64Release/sharc-build-complete.log`; artifact: `_Comp64Release/src/d3d9/d3d9.dll`. Only trailing whitespace was cleaned after shader compilation; no executable behavior changed. Use the repository-required `meson compile -C _Comp64Release`, never invoke ninja directly. Packman tools and builds require elevated workspace access in the current environment.

`validate_sharc.py` passed all nine SDK compile/SPIR-V/layout contracts using the repository's Packman Slang 2025.10.4 compiler. `validate_sharc_estimator.py` passes five numerical propagation tests including independent backward recurrence, skipped surfaces, and finite propagation depth. `validate_sharc_integration.py` passes all four actual built-binary checks: update, query, resolve, and ordinary baseline. It verifies SPIR-V validity, cache bindings, update atomic capability, absence of update image writes, presence of query image output, and no SHARC resources in the baseline. `git diff --check` passes.

The Release DLL and matching symbols were deployed to Fallout New Vegas (.trex) on 2026-09-12, with backups and SHA-256 verification. No GPU execution, image comparison, timing, or visual quality validation has been performed. Do not claim a speedup or production readiness. Next runtime work is a reference comparison with cache disabled, constant-light diffuse scenes, emissive/sky cases, camera and light changes, overflow stress, and GPU validation for synchronization. Sharp reflected lighting, material detail within a cell, and dynamic lighting can exhibit cache bias or lag. Lock-buffer, directional SH, material demodulation, and responsive-lighting variants are not exposed production options.

The September 11 continuation note is historical: its missing estimator/scheduler and enum issues have since been implemented/fixed. Prefer this document and current source when continuing.
