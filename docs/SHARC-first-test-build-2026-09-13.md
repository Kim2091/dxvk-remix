# SHARC-first test build and backend boundary

## Boundary in this milestone

`RtxContext::dispatchIntegrate` owns shared direct lighting, RTXDI gradient generation, and final indirect NEE assembly. It delegates indirect tracing to `dispatchIndirectLighting`. That entry point selects the effective backend and owns SHARC update/resolve/query ordering and its timing/statistics calls. An inactive SHARC still invokes the existing indirect path.

SHARC-specific descriptors live in its shader classes. Its cache storage and invalidation remain owned by RtxSharc. Shared scene/material/G-buffer resources and reconstruction remain shared. No SHARC cache fields were added to denoiser inputs. Shader calculations and pass order are unchanged by this milestone.

This is an initial boundary, not a finished pluggable-backend interface: the entry point still receives RaytracingOutput, and integration still uses existing common objects. Future ReSTIR PT integration must own its path history and resampling resources, explicitly declare shared input requirements, and produce compatible lighting/hit-distance outputs (or provide an adapter). Adding a backend must not require unrelated backends to allocate its resources. Do not mistake NRD SH signal buffers for SHARC cache buffers.

## FNV test procedure

1. Use the same save, camera, resolution, denoiser, and SHARC settings as before. Let shaders and lighting history settle after loading or switching modes.
2. First leave Lean secondary rendering enabled, matching the current FNV configuration. Check stability and compare frame time with the previously deployed build under matching settings.
3. Disable Lean secondary rendering for the feature-preserving SHARC test. This restores the previously omitted secondary features; it can cost more than lean mode. That difference is not evidence that resource specialization regressed performance.
4. With lean disabled, inspect smoke/fire/sparks, decals in reflections, glass, displaced materials at grazing angles, and moving lights. Move the camera, then switch SHARC -> ordinary importance sampling -> SHARC and check for stale images or crashes.
5. If measuring SHARC Update/Resolve/Query, enable the existing SHARC GPU timing control for that diagnostic run; keep query statistics disabled for timing comparisons. Record resolution, reconstruction mode, frame time and settings with the result.

The correct quality-preserving performance comparison is OLD full SHARC versus NEW full SHARC, not OLD lean versus NEW full. The backup made at deployment supports a matched before/after comparison. Restore DLL and PDB together while FNV and its bridge are closed.

## Scope

No material effects were removed by resource specialization. The separate existing lean profile still omits some secondary effects. No denoiser change, NRD SH integration, default-mode switch, or new graphics configuration is included. A large speedup is not assumed; this is the first testable structural milestone.

## Verified FNV deployment

Deployed at 2026-09-13 01:22:43. Backup suffix: .backup-pre-sharc-first-20260913-012241. DLL SHA-256: DE7462418EEC483C0341DDDBAD9DA2986DEA0892830B745699286FF800BF8148. PDB SHA-256: 5759C2D28D0CBF26C13D377EA4610AD586F09B3DFBF43D91B9623EF3D660209C. DLL/PDB copies and backups verified. Root bridge, bridge server, rtx.conf and dxvk.conf hashes unchanged. Game not launched. Release build, 66-stage resource/embedding checks, 14 lean contracts and four baseline shader checks passed. The extracted schedule matches the previous code apart from indentation. Source remains uncommitted.
