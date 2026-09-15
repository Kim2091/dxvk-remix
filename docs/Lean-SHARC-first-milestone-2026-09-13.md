# Lean SHARC: first full-resolution implementation

> **Superseded (2026-09-15):** the lean profile and its options no longer exist; see docs/Lean-SHARC-parity-2026-09-14.md. This file is kept as the historical record of the experiment.

Implemented in `C:/Users/sparkles/Projects/Fable_5_testing/wsn3g`, branch `revised-9-10`, over HEAD `bb69836fa`. This is the first secondary-shader milestone, not the completed streamlined renderer.

## Enable and compare

Select SHARC, then enable **Lean secondary rendering** in its settings:

```ini
rtx.integrateIndirectMode = 3
rtx.sharc.leanSecondary = True
```

The option defaults to false. The UI reports whether the lean profile is actually active. Existing SHARC hardware and compatibility checks still apply: a selected but unsupported/fallback SHARC mode does not activate lean shaders. Existing WBOIT/OMM override settings are not changed.

Lean mode uses a dedicated full-resolution TraceRay query, with SER when the existing device-gated setting enables it. Compute/ray-generation query preferences remain stored and apply again when lean mode is disabled. Updates retain the chosen compute/ray-generation backend, original/deferred accumulation, four/eight-vertex storage selection, tile size and path limits. Ray portal scenes use the full profile; active raytraced render targets retain the existing ordinary-path fallback.

Switching the effective profile invalidates the SHARC cache and resets subsequent denoiser/upscaler history. Warm up both profiles before measuring. Keep the full baseline on TraceRay too for a comparison that isolates material simplification.

## What changed

Fourteen dedicated shader variants cover six update combinations, four TraceRay query ray-generation combinations, and two each of closest-hit and miss stages. In these shaders:

- Indirect RTXDI sample stealing is compiled out.
- The separate unordered particle/decal resolve path is compiled out.
- Alpha-blended shadow geometry is excluded from the indirect NEE shadow mask.
- Secondary POM is disabled and portal material handling is omitted.

Primary G-buffer/material shaders, cutout alpha testing, primary transparency, direct lighting settings, atmospheric passes, reconstruction selection and rendering resolution retain their existing configuration. Indirect lighting can still alter the final appearance of primary surfaces. Ordered material handling, glass/refraction and SSS have not been removed wholesale. NEE cache maintenance/output assembly remains intact.

Update and query use the same simplified policy. The full SHARC and other integrator variants remain available. Resource descriptor layouts remain conservative; this milestone does not claim to remove every unused binding or allocation.

Expected visual tradeoffs: less accurate particles/decals in indirect lighting, different reflected displacement, and missing alpha-blended indirect shadows. Exact effects depend on scene classification and the existing separate-unordered setting.

## Validation completed

- Release build and final incremental relink succeeded via `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`.
- Existing SHARC integration validator: 59 compiled SPIR-V contracts passed.
- New `scripts-common/validate_lean_sharc.py`: all 14 lean variants passed SPIR-V validation, resource/interface checks, query/update isolation and cross-stage ray-payload checks. Compiled uniform accesses to the three removed runtime controls are absent. All 14 blobs are present byte-for-byte in the final DLL.
- Four representative full/ordinary shaders remain byte-identical to blobs in the saved pre-lean DLL and are present in the final DLL.
- Five existing estimator tests and six deferred-estimator tests passed. These validate the preserved propagation math, not the visual approximation.
- Headless D3D9 factory smoke test created/released the interface and enumerated the GPU; no rendering device, game or frame was created. `RtxOptions.md` was regenerated from the built renderer.
- An initial documentation-probe request was rejected because the automation flag grants Sentry upload consent. The approved replacement explicitly clears that flag and uses fresh isolated executable/log and LOCALAPPDATA directories, retaining false crash/usage consent defaults. The rejected probe did not run.

Build logs: `_Comp64Release/lean-sharc-build.log`, `lean-sharc-final-build.log`. Validation logs: `lean-sharc-baseline-validation.log`, `lean-sharc-validation.log`. Factory probe: `_Comp64Release/lean-options-ub3wbrni/options-probe.log`.

Only trailing whitespace and documentation line endings were normalized after the final executable build; no executable behavior changed.

## Artifacts

- Renderer: `_Comp64Release/src/d3d9/d3d9.dll`
- Symbols: `_Comp64Release/src/d3d9/d3d9.pdb`
- Pre-change renderer retained at `_Comp64Release/d3d9.pre-lean.dll`

SHA-256:

```text
d3d9.dll  CF18F001B1E93A4496B7E538DB457ED2A57FFACB7D5B28172D80E4711659A5EE
d3d9.pdb  ADA5B1C259833B92517748B8BE3FA1A05D8256524E5B06E32D15F44FDDE9976A
pre-lean  0CCC2F648BE153C2D5E5B26BC3612A24EFC6B1014E0D750E89B2DC77660D3DD3
```

No game deployment or configuration change was performed. The installed FNV SDK dependency DLLs were read by the factory probe without replacing them.

## Runtime acceptance and remaining milestones

Compare full versus lean with the same TraceRay/SER mode, resolution, cache settings and camera. Measure total rendered-frame time plus SHARC update/query time, with statistics disabled for final timing. Inspect motion, particles, foliage, mirrors, glass, contact shadows and moving lights. Include profile toggles and portal/fallback transitions. User FNV testing reports a 3–4 ms improvement depending on scene, with no apparent quality regression. This is user-observed acceptance, not an exhaustive scene validation.

This implementation precedes unused-pass elimination, conditional G-buffer/resource allocation, SSS/glass simplification and sparse SHARC reconstruction. Those remain separate milestones. Use the first runtime comparison to decide whether secondary shader work warrants further specialization or the next effort should target pass scheduling/query coverage.

## FNV deployment � 2026-09-13 00:43 local

Deployed the Release renderer and matching PDB to `C:\Users\sparkles\Projects\Games\Fallout New Vegas\.trex`. No FalloutNV or NvRemixBridge process was running at deployment. Both installed files were backed up with suffix `.backup-pre-lean-sharc-20260913-004311`; backup SHA-256 hashes matched the originals. Installed files matched the validated build:

- DLL SHA-256: `CF18F001B1E93A4496B7E538DB457ED2A57FFACB7D5B28172D80E4711659A5EE`
- PDB SHA-256: `ADA5B1C259833B92517748B8BE3FA1A05D8256524E5B06E32D15F44FDDE9976A`

Bridge binaries and configuration were preserved. Lean secondary rendering remains opt-in through `rtx.sharc.leanSecondary = True`. Game was not launched; in-game visual and performance validation remains outstanding.
