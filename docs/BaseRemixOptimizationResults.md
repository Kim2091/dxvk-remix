# Base Remix optimization results — 2026-09-11

Branch: `revised-9-10`. Baseline: `4840ae5c99276d57c797d77682aad138fc79fb6f`.
Numos was excluded. These are implementation checks and targeted measurements, not an end-to-end FPS claim.

## Implemented changes

1. Bindless descriptor writes are cached independently for each of the four existing frame slots. Descriptor fields and resource identity determine dirty ranges; more than 32 ranges fall back to one full write. Per-frame command-list resource tracking remains active even when no descriptor write is needed. Resource references prevent false cache hits from recycled Vulkan handles. Buffer offset and range remain part of the comparison.
2. Texture budget priorities are computed once per collection, preserving the existing near-equality comparison and feedback-stamp tie policy. Each managed texture lazily caches aligned mip suffix sizes across collections. Format, extent, layers, and mip count invalidate the cache, including after hot reload or asset replacement. No asset pointer needs to be retained. Low-memory mip limits and load/demotion scheduling are preserved.
3. ReSTIR GI final shading uses a separate early-occlusion shader when visibility validation is disabled. Validation retains the original shader and accurate-distance behavior. POM, alpha handling, portal traversal, and volume attenuation still use the existing shared visibility implementation. An opaque hit clears attenuation; unblocked rays retain their full segment length. This change benefits ReSTIR GI, not NRC.
4. NEE integration has NRC and ReSTIR GI variants that exclude the other mode's code. Dispatch uses the same subsystem active-state checks as the shader constants. Neither-mode and combined-mode cases retain the generic fallback. Output clears, sparse dispatch, training updates, and GI reservoir writes retain their existing behavior.

## Validation

- Full release build succeeded using `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`.
- Production descriptor-template probe: 895,083 descriptor comparisons. Exercises ring reuse, unchanged calls with continued resource tracking, empty/shrinking/growing tables, buffer offset/range changes, distinct resources sharing a handle value, fragmented dirty ranges, and allocation retry. The probe mocks command tracking and descriptor submission; it does not replace Vulkan lifetime validation.
- Production mip-cache probe: 184,193 ranges, including block alignment, layers, arbitrary extents, metadata changes, zero-length ranges, truncated chains, and oversized fallback.
- Both new NEE variants and the GI occlusion variant passed SPIR-V validation and actual Vulkan compute-pipeline creation on an NVIDIA GeForce RTX 5080 Laptop GPU (raw driver version 2559967232).
- Generic NEE SPIR-V SHA-256 remains `63199B2B11E3A0999717669F4AA099758289F5BC74E34A81F259C73807DA4001`.
- GI validation SPIR-V SHA-256 remains `9F9E76D256E0ADBF269F1267D07401189AFA501D05897CB6380939E1541F1DC9`.
- No game was launched, stopped, or modified for these checks. In-game image comparisons, scene-transition/streaming checks, and representative GPU frame timings remain outstanding.

The committed probe generators extract the current production implementations rather than maintain a second copy:

```powershell
python scripts-common/validate_bindless_descriptors.py <output-directory>
python scripts-common/validate_texture_mip_cache.py <output-directory>
```

Compile the generated `validate_descriptors.cpp` and `validate_texture_cache.cpp` with MSVC `/O2 /EHsc /std:c++17`, assertions enabled, and the Vulkan include directory. Run the resulting executables. These are standalone probes, not tests to run from a game build directory.

## Measurements and limits

| Measurement | Baseline | Changed |
|---|---:|---:|
| 8,192 stable buffer descriptors, driver microbenchmark, CPU microseconds/batch | 26.850 | 6.396 |
| Mip size helper, 8,192 textures x 200 batches, milliseconds | 47.478 | 4.371 |
| GI final shading registers | 168 | 168 |
| GI final shading shared memory, bytes | 2,080 | 1,568 |
| GI final shading native executable, bytes | 98,176 | 96,896 |
| NEE NRC registers | 168 | 128 |
| NEE NRC shared memory, bytes | 7,168 | 12,288 |
| NEE NRC native executable, bytes | 115,200 | 104,576 |
| NEE GI registers | 168 | 168 |
| NEE GI shared memory, bytes | 7,168 | 7,168 |
| NEE GI native executable, bytes | 115,200 | 110,080 |

Descriptor timing excludes Remix resource lookups and tracking; it measures buffer descriptor comparison versus actual driver writes. Highly dynamic tables still incur comparisons before writing, so their CPU benefit is not established. Caching retains resource references until a frame slot is updated again; this can delay reclamation, particularly when rendering pauses. Descriptor vectors add CPU memory. The mip cache adds approximately 296 bytes per managed texture on x64. The shader statistics come from `VK_KHR_pipeline_executable_properties`, not a timed dispatch. NRC's lower register count trades against higher shared-memory use; actual GPU benefit is not established. The driver's implausible local-memory statistic was excluded rather than interpreted as a valid occupancy metric.

Probe sources, SPIR-V baselines, reflection data, and build logs are retained locally under the game workspace's `numos-investigation/optimization-work` directory.

## NEE maintenance investigation (candidate 5)

The existing maintenance shader uses 64 registers and 5,696 bytes of shared memory on this GPU; its native executable is 120,192 bytes. Dispatch covers 32 x 4 x 32 groups of 128 threads, updating eight cells per group. These compiler statistics do not demonstrate a scheduling or occupancy bottleneck.

The pass remaps geometry/light identifiers, merges and ages candidates, refreshes samples, and expires retained entries. Group barriers span multiple cells. A cell receiving no new task is not necessarily idle. Scheduling and update frequency are therefore unchanged.

Before changing this pass, capture `UpdateNEECacheShader` timings and active-warps/barrier-stall metrics in representative static, moving-camera, geometry-reshuffle, and light-change scenes. Any compaction or batching experiment must preserve aging, refresh, expiration, reset behavior, and cell identity. Accept it only after matching output/history behavior and demonstrating a GPU timing improvement.
