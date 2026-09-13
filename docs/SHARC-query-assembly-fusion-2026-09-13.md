# SHARC query / assembly fusion audit

## Implemented prerequisite

Extracted the existing NEE assembly calculation and helper functions to pass/integrate/integrate_nee.slangh as assembleIndirectLighting(pixelCoordinate). The standalone compute entry retains dispatch bounds checking and sparse pixel mapping. All four assembly variants use the same helper. A source comparison verifies that the calculation and supporting function bodies match the prior committed implementation apart from trailing whitespace cleanup. No fused dispatch or runtime option exists yet; this extraction does not remove GPU work.

## Producer / consumer contract

| Signal | Producer | Consumer / constraint |
|---|---|---|
| Raw indirect radiance + hit distance | Query; sanitized only for primary-selected paths | Assembly for primary-selected paths; secondary demodulation reads it separately at demodulate.comp.slang:784. Retain the raw output. |
| Primary indirect diffuse/specular | Assembly, including primary emissive-triangle NEE | Primary demodulation reads at lines 652-653, then denoising/reconstruction. Both outputs must be written even when only one lobe receives the query result. |
| Throughput/cone radius | Direct integration | Query input shares physical storage with primary indirect diffuse output (rtx_resources.cpp:1152). Concurrent fused input/output ownership requires explicit handling; ordinary AliasedResource read/write binding is not a completed solution. |
| Primary material and geometry | GBuffer | Assembly reconstructs the primary surface regardless of whether the query selected primary or secondary. Query bindings do not contain the full primary-material set. |
| NEE cache candidates/samples | Prepared NEE cache | Assembly samples emissive triangles, traces visibility, calculates MIS and inserts task feedback. Preserve candidate order, RNG stream 7 and feedback conditions. |
| Sparse pixel coordinates | Sparse preparation | Query and assembly map launch coordinates; fused execution must map exactly once and keep one invocation per active pixel. |

## Constraints on the first selectable prototype

- Compile a dedicated SHARC query+assembly variant. Keep standalone query+assembly selectable; do not globally replace the entry point.
- Reuse the helper, with an explicit radiance argument for register handoff, rather than copying the NEE algorithm. A fused helper cannot retain the separate Texture2D raw-input declaration alongside the query RWTexture2D declaration.
- Preserve the current rgba16f storage conversion before lobe selection and clamping when passing radiance directly. Full-precision register handoff would otherwise change numerical behavior. Verify conversion behavior including overflow/nonfinite handling against the texture path before claiming equivalence.
- Retain raw radiance writes for secondary demodulation. Buffer deletion is outside this prototype.
- Start with a separate throughput input allocation or a proven explicit alias binding strategy. The former costs memory/bandwidth and must be included in the performance comparison; the latter needs proof that no invocation reads another invocation's overwritten data.
- Run primary assembly for secondary-selected pixels too. It contributes primary NEE even if the query used the secondary surface.
- Preserve primary-miss clearing, secondary-selected misses, sparse inactive-pixel behavior, and normal query termination. Do not attach assembly only to the successful path tail.
- Allocate noncolliding additional shader bindings for the primary material inputs and diffuse/specular outputs. Keep backend-specific NRC/GI buffers out of the fused SHARC layout.
- Initially fall back for unsupported shader backends, lean-secondary, query statistics and debug modes until their variants and exit coverage are explicitly supported. Effective activation must govern pass skipping, not just the requested option.
- Preserve the SHARC update -> resolve -> query dependency. Only skip the separate assembly dispatch when the fused shader was actually selected.
- Account for NEE task insertion interleaving after fusion. Do not assume identical temporal cache feedback merely because the per-pixel math matches; inspect cache read/write behavior and compare convergence.

## Measurement and status

Assembly is approximately 0.3 ms in recent captures. Its visibility/lighting calculation remains necessary, so deleting the dispatch cannot be equated with saving the whole interval. Extra registers, descriptors or scratch allocation may erase a bandwidth/dispatch saving. No fused-path gain is claimed.

The probe-loop experiment remains unchanged. Compare split versus fused with the same loop attribute on both sides, or revert that experiment symmetrically; do not mix it into a fusion comparison.

Release build passed for extraction. Only the four assembly shader blobs changed; source bodies are identical but byte-for-byte SPIR-V equivalence is not claimed. Game installation was not modified for this prerequisite. Fused variant, selection, resource ownership, runtime correctness and performance testing remain unfinished.
