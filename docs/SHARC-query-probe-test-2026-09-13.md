# SHARC query probe-loop test

## Change

Allow driver unrolling of the bounded hash-grid lookup loop in SHARC_QUERY shader variants. A HASH_GRID_FIND_LOOP_ATTR hook changes only Find; insertion retains its original loop attribute. The Remix SHARC wrapper sets the hook to [unroll] only for query variants. Update/resolve and non-SHARC shaders retain their original compilation behavior.

This is a compiler scheduling experiment, not a new estimator. Probe order, probe range, empty-slot stopping policy, sample validity checks, radiance decoding and output expressions remain unchanged. Larger driver-generated machine code and register pressure may outweigh reduced loop overhead. No speedup is established.

## Evidence

The suspected eager evaluation of the distance guard was ruled out: baseline SPIR-V already branches around cache lookup when the receiving hit is too close. No change was made to that logic.

Compared all existing shader hashes before/after compilation. Exactly 36 query shader binaries changed. For each changed blob, replacing exactly one OpLoopMerge control word from Unroll (1) back to DontUnroll (2) reproduces its original SHA256 hash. Every other shader binary is identical. This proves that the intermediate shader code change is limited to the loop attribute; driver machine code and runtime performance still require testing.

Baseline hashes: _Comp64Release/sharc-probe-baseline-hashes.json. Baseline renderer: commit 96e98a505, assembly-boundary candidate. Final build and embedded-resource checks must pass before deployment.

## Test

Keep the existing full-feature SHARC settings. Load the same save/view as the properly measured assembly-boundary retest. Check the image, then allow 45-60 seconds stationary for a capture. Compare IndirectIntegration and total timings; the prior single-run medians were 2.390 and 15.387 ms. Separate-run differences alone do not establish a gain. If promising, verify warmed A/B again and overall performance with profiling disabled equally on both builds. GPU profiling remains enabled for this diagnostic deployment.

Release build and all 66 SHARC compiled-resource/DLL embedding checks passed, including plain assembly validation. Existing shader/linker warnings remain. Runtime result pending.

Deployed query probe experiment 0ba50562d to FNV. Backup suffix: .backup-pre-query-probe-20260913-055157. DLL SHA256: A62077CD661F8F812FFB2129015A61E95DB7F5A9DAFE29B2B84712B00EFAD2AA. PDB SHA256: 83C320F48259C58F48CA60D996E4F32B68A5473AB09BA1B7F2917FDD1556F503. Both verified. Configuration unchanged; profiling remains enabled.
