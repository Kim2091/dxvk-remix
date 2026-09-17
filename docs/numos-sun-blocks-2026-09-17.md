# NuMos sun-block experiment and sample diagnostics ? 2026-09-17

User reports screen interleave adds shimmer/grain for little performance benefit. Keep the
Every frame screen default and pursue a different candidate; no new accumulation or scaler.

## Sample-slider investigation

Observed in the existing FNV log: maxSamples changes among 32, 222 and 256. This proves the
CPU option changes; it does not by itself prove the GPU loop bound. Source inspection shows
both main-layer paths use cloudViewSamplesMax, with no forced 256 main-layer loop bound.
The cap is floored by the base count (normally 32); at zero spacing the fixed base count wins.
Adaptive rays stop at the slab exit or opacity threshold, so increasing a nonbinding cap has
no effect on those rays. Budget exhaustion adds up to four coarse tail evaluations. A second
layer and a second shell crossing can add evaluations beyond one main-layer budget.

Recent sampled screen times include 1.46227 and 1.43766 ms at cap 32, and 1.54112 ms at cap 222.
These are observations from a live slider sweep, not a controlled same-view A/B or proof of
speedup. The screenshot uses 0.5 km spacing; actual counts must be measured before attributing
its weak cap response to a specific cause. Fixed-at-256 is not supported by the current evidence.

Added a GPU reduction of the existing density-evaluation diagnostic in cloudDepth.w. While
Log Cloud Timings is on, once per 120 eligible frames it reads every internal pixel and reports
meanAll, meanEvaluated, maximum and evaluatedPercent. Reused/no-march pixels contribute zero.
Counts include view-density calls across layers and tails, not lighting shadow-ray taps.
All 8x8 lanes participate in barriers, with zero-weight padding at odd extents. CPU consumes
coherent readback only after a timestamp query confirms the GPU copy has completed; it never
waits synchronously. Buffer resize waits for that same completion. UI shows the last completed
measurement and its captured cap/spacing/screen period. Timing records now capture submitted
shader arguments rather than only the earlier CPU option snapshot. Statistics have their own
GPU stage, outside CloudScreen timing.

Falsifier: if a no-second-layer, single-crossing base-32/cap-32 frame consistently shows counts well
above 36, the expected bound or diagnostic is wrong. If measured counts finish below the cap,
a higher cap should leave them unchanged. Counts at/above the cap with tail work support
budget exhaustion, but mean counts alone do not identify it. Debug view 912 localizes the work.

## New candidate: Coherent Sun Shadow Blocks

Recent FNV sun-grid timings are approximately 0.48?0.51 ms. The previous Quarter mapping put
adjacent X lanes four columns apart. The new compile-time variant schedules adjacent blocks of
eight columns, seeking better texture-cache locality. It retains the same voxel integrals,
number of updates, shared weather controls, and full-refresh rules. Every frame mapping is
identical. Half/Quarter have a different spatial age pattern, so temporal lighting quality
still requires review; no claim of identical moving images is made.

The live Coherent Sun Shadow Blocks checkbox defaults on for this experiment. Off selects the
original variant. CloudConfig and Cloud profile logs capture sunBlocks alongside the resolved
bake cadence. Compare AtmosphereCloudSunGrid GPU time at a fixed view/weather/cadence after
warmup. Equal or worse time, or visible block-age artifacts, falsifies the candidate's benefit.
No new measured GPU speedup or in-game visual validation is claimed.

## Verification and deployment

- Release build passed. All 531 SPIR-V modules validated for Vulkan 1.3 scalar layout.
- All three new/changed shader modules are embedded byte-for-byte in the built DLL.
- Exhaustive 256-column schedule check passes for periods 1/2/4: no holes or duplicates,
  eight contiguous columns per group, and identity mapping at period 1.
- Generated option docs updated; release suite: 16 passed, only the known graph-documentation failure.
- FNV DLL/PDB replaced after two stopped-process checks; deployed hashes match; three backup pairs retained.
- Backup suffix: .backup-pre-sun-blocks-20260917-044731.
- DLL SHA256: FFDB3D16B3BF6F0518732847B0F4873814B9F83956829147C64C2ADB5A84C434.
- PDB SHA256: 4870D5A9249951E8BE025BCDE1BB6A984EF95F20A51C530A979DD16F75E9D710.
- Config unchanged: screen interleave remains default Every frame. New sun-block experiment defaults on.
- Actual GPU sample readout and sun-block performance/visual comparison still await an in-game run.
