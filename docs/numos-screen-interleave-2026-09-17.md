# NuMos screen interleave restoration — 2026-09-17

User clarified that Half/Quarter screen marching must remain. Restores only that component
of f72f60672's parent, at native internal resolution. Spatial scaling, upsampling, temporal
accumulation blending and clamping remain removed. Detail LOD remains Off by default.

- Live Screen Cloud Interleave: Every frame (original default), Half, Quarter.
- Live Screen Reuse Depth Tolerance: original default 0.1, relative surface distance.
- Each thread owns a 1x1, 2x1 or 2x2 cell; one pixel marches and the others reproject.
  Failed reprojection marches fresh. Fresh results never blend with history.
- Native jittered current rays project through the previous jittered camera rotation.
  Surface classification/distance validation is restored, without variance or clamp logic.
  Rotation-only reprojection retains the old limitation under rapid camera translation.
- Paired color/depth targets return. This restores 24 bytes/internal pixel of image payload
  (CALCULATED, excluding allocation overhead), cancelling the cleanup's history-memory saving.
- Full updates on changed voxel inputs, camera/anchor cuts, lightning, new extent or a frame gap.
  A disabled debug dispatch invalidates history. Existing sun-grid/dome behavior is unchanged.
- Separate interleave variants for all seven screen profiling modes; Every frame does not compile
  the history sampling path. Shared arguments add one explicitly padded 16-byte row.

## Falsifiable verification

With profiling logs on, ScreenInterleave reports requested/resolved periods and fullFrames/120.
If Half/Quarter shows fullFrames=120/120 in a stable scene, the requested cadence is being
overridden and cannot deliver its intended march reduction. The counter can report either
extreme and intermediate values. Existing debug view 912 shows density evaluations: carried
history writes zero, while both scheduled and rejection-fallback marches report actual work.
Neither counter alone proves a frame-time improvement; compare the existing GPU stage timings.

No new in-game visual comparison or GPU performance measurement has been performed.

## Verification and deployment

- Release build passed, including the shared-argument layout assertion.
- All 529 SPIR-V modules validated for Vulkan 1.3 with scalar block layout.
- All 14 screen variants have history bindings exactly when interleave is compiled in.
- All 490 freshly generated SPIR-V modules are present byte-for-byte in the built DLL.
- Generated RtxOptions.md updated. Full release tests: 16 passed; only the known
  pre-existing test_graph_documentation failure remains.
- Deployed DLL/PDB to FNV after two stopped-process checks, with paired backups and
  matching deployed hashes. Three complete backup pairs retained. Game config unchanged;
  Screen Cloud Interleave remains Every frame until Half or Quarter is selected live.
- Backup suffix: .backup-pre-screen-interleave-20260917-043058.
- DLL SHA256: 59B5688ED23CACCFFDECFC3B076B8DDCE82CCE32A324D0B1AEC39F8383F0F991.
- PDB SHA256: 5F18EDC7F15A3F5BF7B2E1256EB6A09CC638FCB2266F784D3E86905456768EC8.
