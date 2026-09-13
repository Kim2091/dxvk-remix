# SHARC update/query reuse feasibility — 2026-09-13

## Finding

Do not replace training with ordinary query results. The two paths share initial screen-derived ray state at selected pixels, but deliberately compute different estimators. This investigation corrects the earlier suggestion that their duplicate code necessarily implies a large removable duplicate workload.

## Source evidence

- `pass/integrate/integrate_indirect.slangh:initPixelCoordinateOnLaunch` maps one update invocation to a rotating pixel per update tile. At the configured tile size 8, selected pixels are approximately 1/64 of the full-resolution query population (edge tiles and sparse query mapping affect exact overlap).
- The same file's `integrate_indirect_pass` loads initial origin/direction, throughput/cone radius and first sampled lobe from the direct pass. The first segment can start from the same state for matching pixels. This does not prove identical later paths or side effects.
- `algorithm/path_state.slangh:pathMaxBounces` uses the capped SHARC update depth for training and the regular path depth for query. `calculateUseRussianRoulette` returns false for update; query can use stochastic termination and probability reweighting.
- `algorithm/integrator_indirect.slangh` inserts eligible training vertices in the update branch. The query branch instead looks up resolved radiance and terminates the path on a usable hit. Emission is accumulated before that lookup. Eligibility excludes medium/translucency/SSS/emissive receivers and considers roughness/lobe state.
- `pass/sharc/sharc_update.slangh` records vertex indices, segment radiance and weights and propagates contributions backward. A query that terminates on cached lighting does not supply the same training tail.
- `rtx_context.cpp:dispatchIndirectLighting` orders update -> resolve -> query. Query observes this frame's published update. Ordinary simultaneous fusion would change that dependency unless publication and query completion remain separate.
- Update/query also differ in NEE task feedback, debug writes and screen output guards. Sharing material resolution requires auditing these side effects, portals, POM, unordered particles/decals, opacity micromaps and selected secondary surfaces.

## Designs and tradeoffs

1. **Replay a shared first-segment record at update-selected pixels.** Update can record geometric hit data for replay by query after resolve. This preserves cache publication order in principle. Start with geometry-only records, not a wholesale material/lighting snapshot: the latter can depend on path state and has side effects. Records must represent misses and special traversal correctly or explicitly fall back to original query. Preserve resource lifetime and current-frame scene identity. Cost is records, extra query lookup/branch and replay machinery to save traversal for only roughly 1.6% of pixels at tile size 8. That percentage describes pixel overlap, not a measured time bound. No basis yet for a substantial gain.

2. **Use query paths to train the cache.** This can potentially remove more training work, but cached termination hides the tail needed for training; roulette, bounce limits and sample selection differ. Continuing selected query paths with independent training state can restore missing work, but requires two estimator states, careful RNG separation and cache publication design. Updating next frame changes latency, while reading and training the same cache without separation introduces ordering hazards. This is an algorithm experiment, not a feature-preserving cleanup.

3. **Reusable path records / wavefront architecture.** A backend-private path producer could export compact hit records to training and rendering consumers. Keep cache state and training storage inside the indirect backend; expose only common lighting/hit-distance outputs to reconstruction. This is a larger architecture project with memory, compaction, divergence and register-pressure tradeoffs. It needs output/estimator tests, not just a frame-time comparison.

## Decision

No reuse shader is implemented or deployed: the read-only feasibility task found no justified drop-in reuse optimization. The existing plain NEE specialization remains installed. The preceding rejected atmosphere experiments remain reverted.

Before implementing option 1, measure matching-pixel query cost and first-hit eligibility; before option 2, define acceptable estimator/history changes explicitly. Preserve exact current-frame cache publication if claiming current semantics. Never equate shared source code with removable full-screen duplicate traversal. Further micro-candidate timings require warmed, paired same-session comparisons and image/denoiser-input checks.

A direct query-path optimization remains a better near-term target than training/query fusion if quality must remain unchanged. Its next audit should locate work done before cache termination that is unnecessary on a hit, while preserving eligibility, emission, distance tests, material evaluation and particle/decal effects. Moving a lookup earlier without those prerequisites is unsafe.
