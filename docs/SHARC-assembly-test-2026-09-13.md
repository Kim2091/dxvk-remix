# Indirect assembly boundary test

This candidate includes the indirect subsystem scheduling boundary and an explicit, borrowed AssemblyResources interface. Assembly can no longer access the full RaytracingOutput aggregate. It still receives shared scene state through RtxContext and common bindings; this is not yet a fully independent backend API.

The plain NEE assembly variant now uses a descriptor layout omitting NRC training vertices (71), ReSTIR GI reservoirs (82), and BSDF factor (83). Its CPU path skips their binding preparation. NRC/ReSTIR modes retain their existing bindings, as do NEE visualization modes. Shader algorithms and primary diffuse/specular outputs are unchanged. Raw indirect radiance remains available to secondary demodulation.

Aliased resources are passed by reference rather than resolving views in the interface constructor. This preserves the binding-time read/write ownership transitions. The existing SHARC update, resolve, query, and assembly order remains intact. The original fallback still runs when SHARC is inactive.

## Validation and limits

Compiled shader contract validation checks all 66 source-referenced SHARC stages and the plain assembly stage. The current shader blobs also occur byte-for-byte in the previously installed DLL. Runtime visual equivalence, debug-view transitions and backend switching require game testing. No performance improvement is claimed; this milestone establishes a narrower assembly boundary and removes inactive descriptor preparation, not a new lighting estimator or pass fusion.

## Test

Use the existing full-feature SHARC configuration (leanSecondary=False). Load the familiar save and inspect stationary lighting first, then rotate/move the camera. Check reflective surfaces, glass, particles, decals and dynamic lights available in that save. Look for missing indirect light, flicker or changes during motion. A 45?60 second stationary interval can provide timings from the existing GPU stage log, but the primary purpose of this build is compatibility. Compare only the same warmed scene/view for timing.

If convenient, test an NEE visualization view and return to normal rendering. Backend switching is a separate check; return to SHARC afterward. No NRD, resolution, cache sample rate or material setting is changed by this deployment.

Deployment details and final validation follow below.

## Verified deployment

Release build passed; SHARC and plain assembly compiled contracts passed. Existing third-party linker warnings remain. DLL and PDB backed up with suffix .backup-pre-assembly-boundary-20260913-051140 and deployed with SHA256 verification. No game configuration was edited.

DLL SHA256: 917FD981762A647356DC0DD98DAA16023F60D1D87C66E6673E36186286168A46
PDB SHA256: 8161536B71037A946FF5CEA7AF06BE6443FDF41E36BCBC03C4BE3FC6466569E0


## Runtime result and rollback

User reports correct appearance but much slower performance. Last 12 complete samples (1480..2800): measured sequence median 17.537 ms vs previous plain-NEE capture 15.869 ms. ScenePreparation is 1.893 vs 0.132 ms; IndirectAssembly 0.351 vs 0.337 ms; IndirectIntegration 2.582 vs 2.486 ms. Separate-run results do not establish causality. GPU intervals include stalls, and independent stage medians cannot be summed to explain the total exactly. The large difference is in scene preparation, not an observed large assembly interval increase.

Preserved capture and summary in _Comp64Release/gpu-assembly-boundary-regression*. Restored the previous DLL/PDB pair from .backup-pre-assembly-boundary-20260913-051140 with hash verification. Source changes remain available for investigation; candidate is not accepted as a performance improvement. Next diagnostic is the same save/view on restored baseline, before further renderer changes.

User requested a retest: restored the exact assembly-boundary candidate DLL/PDB from .backup-assembly-regression-20260913-053558 and verified both against original deployment SHA256 hashes. Prior performance report is pending retest; no conclusion withdrawn or confirmed yet. Configuration unchanged.

## Properly measured retest

User reports performance largely the same as the older build when measured properly; prior visual check found correct appearance. Retest last 12 samples (1874..3194): measured sequence median 15.387 ms (15.068..17.535), ScenePreparation 0.117 ms, IndirectIntegration 2.390 ms, IndirectAssembly 0.312 ms. Earlier plain-NEE capture medians were 15.869, 0.132, 2.486 and 0.337 ms respectively. The previously observed large scene-preparation interval does not reproduce. These separate captures do not establish a speedup; classify the change as structural with no demonstrated performance regression in this retest. Exact cause of the initial slow capture is unresolved. Debug/backend transition coverage remains unverified.

Capture preserved at _Comp64Release/gpu-assembly-boundary-retest.log with companion summary. Candidate remains installed. GPU profiling remains enabled. No settings or source code changed for this retest.
