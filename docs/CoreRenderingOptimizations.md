# Core rendering optimization audit — 2026-09-09

Implemented an exact-input cache for acceleration-structure size queries in the dynamic BLAS and TLAS preparation paths. This reduces CPU/driver overhead; it does not reduce GPU ray counts or change lighting quality.

## Implemented

- `rtx_accel_size_cache.h`: one bounded cache entry per owner, with no heap allocation inside the cache.
- `BlasEntry::buildSizeCache`: reuses sizing results for single-geometry dynamic BLAS preparation, including animated geometry whose buffer addresses change but sizing inputs do not.
- `AccelManager::m_tlasSizeCache`: independent entries for opaque, unordered, and SSS TLAS. Instance count includes GPU PointInstancer slots.

Keys include the device, AS type, build flags, primitive count, geometry type/flags, vertex format/stride/maxVertex/index type, transform presence, and instance pointer layout. Comparisons use typed fields, not structure padding or pointer hashes. Changes to these inputs trigger a new query. All three returned sizes are cached.

Any build, geometry, or geometry-data extension chain bypasses the cache and invalidates its entry. This includes opacity micromaps, whose contents can change without their pointer changing. Multiple geometries, `ppGeometries`, and procedural AABBs also use the original query path. There is no new scene-generation dependency.

Vulkan explicitly ignores buffer addresses for sizing except whether triangle transform data is present; build mode and source/destination AS handles are also ignored. The cache conservatively requires exact primitive counts and maximum vertex values rather than reusing oversized results. See the [Vulkan size-query contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetAccelerationStructureBuildSizesKHR.html).

The size-query cache leaves AS build/update decisions, flags, geometry uploads, barriers, resource lifetimes, and GPU rendering passes unchanged. The subsequent memory changes are described below.

## Measurement and validation

Standalone probe: `scripts-common/validate_accel_size_cache.cpp`.

- Cache invalidation/fallback checks passed, including 35 expected driver misses.
- 112 comparisons against actual Vulkan driver queries passed for BLAS/TLAS, changing counts, changing addresses, and fast-build/fast-trace flags.
- Vulkan validation output was empty.
- Full release runtime build passed with `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`; final log: `_Comp64Release/core-render-build-final.txt`. Existing missing-PDB linker warnings remain. Unit-test executables were built as dependencies but were not run from the game build directory.
- Benchmark alternates direct/cached batches, discards two warmup batches, and takes medians from 20 batches of 20,000 calls. Cache hits cycle through 1,024 separate entries. Validation is disabled for timing.

NVIDIA GeForce RTX 5080 Laptop GPU, installed Windows driver, release MSVC probe:

| Path | Direct driver query | Cached query |
| --- | ---: | ---: |
| Dynamic BLAS sizing | 555.78 ns | 4.37 ns |
| TLAS sizing | 466.88 ns | 3.85 ns |

This is an isolated CPU measurement, not a GPU or game FPS result. At those measured costs, 1,000 eligible repeated BLAS queries would save approximately 0.55 ms of CPU work. Actual savings depend on how many geometries reach this path, cache-hit rate, and CPU/GPU overlap. The existing unchanged-scene fast path already skips BLAS preparation entirely; this change saves nothing for geometry that never reaches the query. The two or three TLAS queries alone provide only a small saving.

Logs are in `_Comp64Release/core-render-validation/`. No game deployment or visual capture comparison has been performed for this change.

From a PowerShell session at the repository root, compile the standalone probe with the installed SDK:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
. .\build_common.ps1
SetupBuild -BuildArch x64
New-Item -ItemType Directory -Force _Comp64Release/core-render-validation | Out-Null
cl /nologo /std:c++17 /O2 /EHsc /I include/vulkan/include scripts-common/validate_accel_size_cache.cpp /Fe:_Comp64Release/core-render-validation/validate_accel_size_cache.exe /Fo:_Comp64Release/core-render-validation/validate_accel_size_cache.obj /link /LIBPATH:C:/VulkanSDK/1.4.357.0/Lib vulkan-1.lib
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
.\_Comp64Release\core-render-validation\validate_accel_size_cache.exe
Remove-Item Env:VK_INSTANCE_LAYERS
.\_Comp64Release\core-render-validation\validate_accel_size_cache.exe
```

The probe needs an acceleration-structure-capable first enumerated Vulkan device. Its timings with validation enabled are not performance results.

The runtime build required two small fixes to existing optional-private-module guards: check for the actual `meson.build` file, not merely the containing directory. An empty/private directory can exist in this checkout without its build definition. The changes preserve the pre-existing local build customizations.

## GPU candidates requiring scene measurements

The build-quality candidate has now been measured below. The other items remain investigation priorities, not implemented or measured speedups:

1. **Acceleration-structure build quality.** Dynamic BLAS, merged BLAS, and TLAS currently request `PREFER_FAST_BUILD`. Measure total build plus GBuffer/direct/indirect traversal time when selectively using fast-trace builds for long-lived geometry. A universal flag switch can trade lower traversal time for higher build time and memory use. Animated geometry and frequently rebuilt buckets need separate measurements.
2. **Visibility traversal specialization.** `algorithm/visibility.slangh` rejects procedural candidates without evaluating them and uses visibility-mode template flags. Hardware primitive culling or compile-time ray flags may avoid candidate handling. Validate translucent attenuation, accurate shadow hit distances, portals, alpha tests, micromaps, and traversal-order effects before changing this shared path. Forcing opaque rays would discard supported material behavior.
3. **Direct integration register and bandwidth pressure.** The direct pass loads material data for both lighting and indirect-ray preparation. Measure its live register count and occupancy before splitting it: separation would reload material data and add synchronization, similar to the AP split that benchmarked slower. Sparse direct-inactive pixels can still require indirect preparation, so skipping the entire pass for them is unsafe.

The existing scene-generation/bucket caches, sparse lighting paths, indirect NRC/NEE/trace-mode shader variants, and denoiser gating already cover several obvious optimizations. In particular, Ray Reconstruction can still require secondary-signal preprocessing or NRD training data; disabling all NRD work solely because RR is enabled would change supported behavior.

## Follow-up: acceleration-structure memory and build quality

Implemented two additional changes in `rtx_accel_manager.cpp`:

1. Dynamic and merged BLAS updates reserve `updateScratchSize`, selected after the final build/update decision. Full builds retain `buildScratchSize`. Existing slice alignment, non-overlap, barriers, and resource tracking are preserved. This also handles drivers where update scratch could exceed build scratch: each mode now uses its own requirement.
2. TLAS builds no longer request `ALLOW_UPDATE`. `internalBuildTlas` always issues a full build, including the opaque current/previous TLAS pair, unordered TLAS, and optional SSS TLAS. No TLAS update operation exists in this path. BLAS update support remains enabled.

The [Vulkan size-query contract](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetAccelerationStructureBuildSizesKHR.html) defines separate scratch requirements for builds and updates. Using the appropriate returned value avoids reserving full-build scratch for updates.

### Measured allocation changes

On the RTX 5080 Laptop GPU with fast-build preference:

| Workload | Previous requirement | New requirement |
| --- | ---: | ---: |
| Update scratch, 24,576-triangle merged mesh | 105,728 bytes | 1,280 bytes |
| TLAS, 2,048 box instances | 384,384 bytes | 356,736 bytes |
| TLAS, one instance | 2,304 bytes | 2,048 bytes |

For the large-mesh update, Remix's aligned slice shrinks from 105,984 to 1,536 bytes on this device (98.6%). The 2,048-instance TLAS requirement shrinks by 7.2%. These are synthetic allocation measurements, not percentages of total game VRAM. Small-mesh update scratch was unchanged. Scratch pooling, other builds in the frame, and allocator behavior affect the realized memory footprint. GPU timing differences from removing TLAS update support were small/noisy; no FPS improvement is claimed.

### Build quality result

The benchmark tests fast-build/fast-trace independently for BLAS and TLAS, then separately tests TLAS update support. It uses merged boxes, instanced boxes, and an empty TLAS. Each ray dispatch contains 1,048,576 rays. Workloads cover coherent closest hits, incoherent shadow/closest hits, and a deterministic cutout surrogate.

For the merged mesh, switching BLAS to fast-trace gave these medians:

| Operation | Fast build | Fast trace |
| --- | ---: | ---: |
| Full BLAS build | 0.3154 ms | 0.3478 ms |
| BLAS update after deformation | 0.0372 ms | 0.1276 ms |
| Primary-like traversal | 0.0723 ms | 0.0704 ms |
| Shadow traversal | 0.0832 ms | 0.0786 ms |
| Indirect-like traversal | 0.0908 ms | 0.0826 ms |
| Cutout traversal | 0.1131 ms | 0.1116 ms |

The four traversal workloads together saved about 0.0162 ms, while the BLAS update cost increased about 0.0904 ms. This does not support changing the default for frequently updated geometry. A long-lived static mesh could amortize its higher initial build cost, but a representative game capture is needed before choosing an automatic policy. For the instanced boxes, fast-trace TLAS with update support increased the allocation from 384,384 to 712,064 bytes without a meaningful traversal improvement. Fast-build preference remains unchanged.

This measured dependence on geometry lifetime is consistent with [NVIDIA's build-flag guidance](https://developer.nvidia.com/blog/effectively-integrating-rtx-ray-tracing-real-time-rendering-engine/), which distinguishes static, high-detail deformable, and frequently rebuilt geometry.

### Validation and reproduction

- Full release runtime build passed; log: `_Comp64Release/core-render-memory-build.txt`.
- 96 GPU result-set comparisons passed with zero hit/distance differences (48 per flag sweep, including baseline self-checks). Each set contains 1,048,576 rays.
- Validation uses actual BLAS deformation between build and update, separate scratch buffers sized to each operation's driver requirement, and includes empty TLAS builds.
- Both final Vulkan validation logs are empty. These are standalone GPU checks, not full Remix material/portal/OMM/SSS scene tests. No game deployment has been performed.
- Timings are GPU timestamps with device-local output. Readback is outside the timed interval and only performed for correctness checks. Four warmup iterations are discarded, followed by 20 measured samples per variant with rotating variant order.

Probe sources: `scripts-common/benchmark_accel_build.cpp` and `scripts-common/benchmark_accel_build.comp`. In the MSVC PowerShell environment initialized above:

```powershell
& 'C:/VulkanSDK/1.4.357.0/Bin/glslangValidator.exe' -V --target-env vulkan1.2 scripts-common/benchmark_accel_build.comp -o _Comp64Release/core-render-validation/benchmark_accel_build.spv
cl /nologo /std:c++17 /O2 /EHsc /I include/vulkan/include scripts-common/benchmark_accel_build.cpp /Fe:_Comp64Release/core-render-validation/benchmark_accel_build.exe /Fo:_Comp64Release/core-render-validation/benchmark_accel_build.obj /link /LIBPATH:C:/VulkanSDK/1.4.357.0/Lib vulkan-1.lib
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
.\_Comp64Release\core-render-validation\benchmark_accel_build.exe _Comp64Release/core-render-validation/benchmark_accel_build.spv --tlas-only
.\_Comp64Release\core-render-validation\benchmark_accel_build.exe _Comp64Release/core-render-validation/benchmark_accel_build.spv
Remove-Item Env:VK_INSTANCE_LAYERS
# Repeat both commands for timings without validation overhead.
```

Final logs in `_Comp64Release/core-render-validation/`: `tlas-final-results.txt`, `build-final-results.txt`, `tlas-final-validation-results.txt`, `build-final-validation-results.txt`, and the corresponding validation/error logs. Earlier files without `final` were exploratory and are superseded.
