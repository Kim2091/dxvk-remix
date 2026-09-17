# NuMos native-resolution cleanup and detail LOD evaluation

Supersedes the screen-resolution and temporal-reuse instructions in the September 14/17
handoffs and the earlier world-space design. Those documents retain historical experiments.

The screen pass now marches every pixel every frame at the internal render extent. Removed:
cloud resolution scaling, sample boost, spatial reconstruction, temporal accumulation and its
clamp/depth tolerance, screen interleave, history cache keys/counters, ping-pong targets,
lightning history suppression, and the upscaler checkerboard probe. Their options and UI
controls are gone. The composite uses exact loads of the current color and depth targets.
The shared atmosphere tail shrinks by one 16-byte row; partial rows keep explicit padding.

Preserved: Quarter sun-grid and reflection-dome interleave, the anchor-motion outlier cut test,
budget-exhaustion tail integration, primary-surface march clamp, animated native march jitter,
and density-evaluation debug view 912. No SHARC changes.

## Detail LOD evaluation

- MEASURED (prior session/user report): mip filtering had no measured frame-time cost;
  bias zero looked wrong. Mode 1 was inactive at 1x, so the preferred historical native look
  was mip zero. No new GPU timing or visual result is claimed here.
- SOURCE VERIFIED: enabled LOD is clamp(log2(max(2 * texelsPerStep, 1)) + bias, 0, 7).
  Bias -3 selects mip zero through four texels per step, but still filters above that.
  It is an approximation to OFF, not an equivalent setting.
- SOURCE VERIFIED: filtering also affects mid-frequency displacement, followed by nonlinear
  surface gates and density shaping. Preserving the noise mean does not guarantee unchanged
  coverage or silhouettes. The lighting-grid taps continue to use mip zero.
- Decision: OFF is the native default. Keep the mip chain and live Off / Always control for
  a direct comparison; mode 1 remains off for old configs. Bias -3 is retained only as the
  previous comparison setting. The chain has not earned default-on status at 1x.
- Falsifier: in the same native-resolution view, Always/-3 consistently reduces visible
  shimmer relative to Off without losing the preferred cloud structure. That would establish
  a reason to keep filtering enabled. Similar appearance alone does not establish a benefit.
  Compare using the existing live controls; no synthetic probe is needed. Timing logs record
  both the resolved LOD enable and bias so a control change can be verified.

Removing one RGBA16F plus RGBA32F history pair saves 24 bytes per internal pixel of image payload
(CALCULATED, excluding allocation overhead). No frame-time improvement is asserted.

## Verification and deployment

- Release build passed, including the shared-layout assertions.
- All 522 SPIR-V modules passed `spirv-val --target-env vulkan1.3 --scalar-block-layout`.
- All 487 freshly generated shader modules were found byte-for-byte in the built DLL.
- `RtxOptions.md` regenerated from the built DLL. Release tests: 16 passed; only the
  pre-existing `test_graph_documentation` failure noted in the handoff remains.
- Deployed DLL and PDB after checking FalloutNV and NvRemixBridge were stopped, checking again
  immediately before replacement, and backing up both files and `rtx.conf`. Deployed hashes
  match the build. The stale reduced-scale sample-boost override was removed; three paired
  runtime backups remain.
- Backup suffix: `.backup-pre-native-cleanup-20260917-041221`.
- DLL SHA256: `E805C1058875E51003958FEDA0ECAA22421E7D99882AF8EF933A1BD59EF64145`.
- PDB SHA256: `C1A20C08E3AC75B91CF6E0D172DD7B04C75BC6353753BF377EBEF205A1ED2556`.
- In-game visual comparison and new GPU timings have not been performed in this cleanup.
