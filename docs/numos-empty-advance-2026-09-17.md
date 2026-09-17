# NuMos empty-space cursor experiment - 2026-09-17

Latest storm screen timings were roughly 2.2-3.2 ms after the bake-refresh fix,
with about 13-14 density evaluations per evaluated pixel. These are live-session
observations, not a controlled baseline for the new experiment.

The previous adaptive march used the remaining known-empty distance as the next
sample segment length. Its jittered sample could land inside already-known empty
air, pay another density evaluation, and consume another iteration. The experiment
advances the cursor directly through that remaining interval after finishing the
current segment. The next evaluation uses the regular adaptive step length.

The skipped interval starts at the completed segment end, not the jittered sample:
advance = max(samplePosition + conservativeRadius - cursor, 0) * saturate(stepScale).
It is clamped to the slab exit. This retains the existing sample-anchored bound,
including the sampler's displacement margin and lattice-boundary limit. Zero step
scale disables it. It does not extend the bound or remove cloud material within it.
Correctness still depends on that existing conservative-radius contract.

Empty-Space Advance is a live checkbox, default on for comparison. Only the Normal
profiling mode's adaptive primary-layer screen march selects the new compile-time
variant. Off, fixed-count marching, special profiling modes, the second layer,
reflection dome and lighting bakes retain their previous paths. Both Every frame
and screen-interleave variants are provided. No GPU argument layout changed.

Spacing, cap and density definition are unchanged. Sample positions, budget reach,
and which segments use the coarse tail can change, so identical pixels are not
claimed. Compare cloud silhouettes, gaps and motion; unexpected holes or increased
edge noise reject the experiment. Fewer evaluations without lower CloudScreen GPU
time also fail its performance goal. Logs capture emptyAdvance in GPU sample and
timing records so asynchronous observations identify the dispatched variant.

Validation: release build passed, generated options updated, 16 tests passed with
only the known graph-documentation failure. All four baseline/experimental screen
shader blobs were verified byte-for-byte inside the DLL. A source-extracted C++
harness tested 200,000 randomized cursor cases: monotonic movement, slab clipping,
skips confined to the supplied empty interval, and zero-scale identity. This tests
the cursor contract, not the fidelity of the underlying GPU density bound.
No in-game speedup or visual validation has yet been measured for this experiment.

All 533 SPIR-V modules validated. Deployed DLL/PDB hashes match the build; three
backup pairs retained. Deployment backup suffix: .backup-pre-empty-advance-20260917-170431.
