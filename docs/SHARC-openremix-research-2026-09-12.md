# SHARC OpenRemix comparison - September 12, 2026

Requested by the user; read-only comparison performed by Luna agent
/root/sharc_custom_research and checked by the primary agent.

## Evidence and scope

The supplied folder C:/Users/sparkles/Downloads/SHARC openremix contains only
sharc.glsl (1462 bytes) and sharc-resolve.comp (1259 bytes). They reference pinned
SHARC 1.8.3.0. The actual ray tracing/update integration, scene.glsl, SDK files,
benchmark results and material-factor construction are not included. The findings
below describe configuration differences and their behavior in our pinned SDK;
they cannot establish the other renderer's speed or image quality.
No downloaded code was executed or incorporated.

Our current deployed specialized build is documented in SHARC-implementation-status.md.
The user estimates the preceding TraceRay build was about 1 ms slower than
ReSTIR GI overall. Exact current pass timings are unavailable. Earlier captures
placed updates around 1.3-1.6 ms and resolve around 0.02-0.04 ms; these predate
the latest specialization and are not a controlled current benchmark.

## Ranked findings

1. Cache resampling is the strongest performance experiment. sharc.glsl:9 enables
   SHARC_ENABLE_CACHE_RESAMPLING. Our SDK SharcCommon.h:615-635 chooses a random
   resampling depth, reads a sufficiently mature resolved entry, and can terminate
   update tracing. This targets update rays, not the first query segment. Our
   custom deferred updater never calls SharcUpdateHit, so switching the macro
   alone does not implement it. The original updater also passes random=0 and
   does not consume the function's continuation result; that path needs explicit
   integration too. Preserve reflected-radiance/emission ordering, use a real
   random sample, avoid adding the same vertex estimate twice, and test long-run
   feedback, dynamic lighting, darkening and self-reinforcement. Existing resolved
   data must be bound for reads during updates, with the correct stage access.
   A full finite-path fallback is needed for cold/missing cache data and A/B tests.
   Speedup is a hypothesis bounded by the update cost, not established by these files.

2. Disabled resolve cross-entry history reads merit a correctness review, but are
   a weak performance target. sharc.glsl:13-14 disables adjacent-level blending and
   linear probing. Its comment specifically cites in-place resolve ownership.
   In our SDK, linear probing reads neighboring hash/resolved slots at
   SharcCommon.h:897-916; adjacent-level history reads occur at :947-972 while
   each invocation writes its own resolved entry in the same dispatch. Our current
   implementation uses the SDK's defaults with an in-place resolved buffer. These
   accesses deserve an explicit audit of ordering/coherent history assumptions.
   Disabling them changes collision history recovery and moving-camera warmup;
   a separate immutable history buffer is another design option with memory and
   bandwidth costs. Do not describe disabling them as a quality-neutral speedup.
   The entire measured resolve pass was only tens of microseconds.

3. Material demodulation is quality/material-detail work, not a direct speed knob.
   sharc.glsl:8 enables SHARC_MATERIAL_DEMODULATION. SDK :606-613 takes a reciprocal
   of the hit's material factor; query :757-759 multiplies by it. Our zero-initialized
   SharcHitData does not supply that factor and our custom updater assumes cached
   reflected radiance. Enabling this directly could cause invalid or black results.
   A consistent positive material factor and matching update/query estimators are
   required. The supplied files do not show their factor choice.

4. SHARC_SAMPLE_NUM_THRESHOLD=3 (sharc.glsl:10) changes cache maturity acceptance
   in update resampling and queries. It may reduce early unstable estimates, but
   stricter acceptance can decrease cache termination and increase tracing. There
   is no evidence here that it accelerates our steady-state workload.

5. Their 128-thread resolve group and occupancy counter are not compelling ports.
   sharc-resolve.comp:10 uses 128 threads (ours uses 256). Lines 13-28 add shared
   occupancy counting, barriers and one global atomic per nonempty group. This is
   diagnostic overhead, and its GLSL buffer-reference ABI differs from our Vulkan
   descriptor layout. The resolve pass is already small.

## Next experiment

Implement opt-in update-cache resampling with explicit policy for both original
and deferred update paths, separate compile variants/resource slots, valid random
depth selection and a finite-path fallback. Preserve emission-before-cache semantics
and backward propagation to prior vertices. Compare update and total frame time
against resampling off in the same stationary view, then test camera movement and
lighting changes. Keep the recently deployed material specialization/fixed-index
update build as the baseline. Do not claim the 1 ms gap is closed without timings.
