# NuMos thunderstorm refresh fix - 2026-09-17

The FNV log showed the sun-shadow grid taking roughly 4 ms with resolved bake
cadence 1/1 despite saved Quarter/Quarter settings. Weather variation continuously
changes cloudCoverageMean. Exact float comparison in the voxel cache key made
that alone sufficient to force full sun-grid and dome refreshes every frame.

Round coverage only in the voxel cache key to 1/1024 increments. The maximum
rounding error is 1/2048; values sharing a bin differ by less than 1/1024.
Shaders still use continuous coverage. All other key comparisons remain intact,
including slab mapping and published NVDF state. Weather transitions that change
other inputs may still force full refreshes. This does not eliminate storm cost.

Log Cloud Timings now reports BakeInterleave requested/resolved periods and
sunFullFrames, domeFullFrames, inputChanges, and windowFrames over 120 frames.
A stable-camera storm that still reports inputChanges=120 needs further diagnosis;
a single resolvedPeriod snapshot is not sufficient evidence of sustained cadence.
Compare sun-grid GPU timing and moving-cloud lighting after warmup. Visible stale
lighting or no GPU-time reduction would invalidate the practical benefit.

Validation: release build passed; 16 tests passed with only the known existing
graph-documentation failure. A standalone C++ harness compiled source-extracted
production normalizers and the actual weather drift function. With default storm
coverage variation isolated for 300 seconds at 60 Hz, full-refresh requests fell
from 17,964 to 154 across 17,999 comparisons. Sub-bin changes retained the key;
coverage-bin crossings, abrupt coverage changes, slab altitude/thickness, grid
remapping, sun direction, cloud enable, and published NVDF changes invalidated it.
Frame phase, time, and lightning retained the existing normalization behavior.
This is a CPU decision regression, not a GPU performance measurement. Harness:
_Comp64Release/cloud_coverage_cache_regression.cpp (generated, not tracked).
In-game performance and image quality remain unmeasured for this build.
