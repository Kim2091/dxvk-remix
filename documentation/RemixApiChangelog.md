## Remix API Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- 2026-10-28: `remixapi_GetMemoryPressure` reports the runtime's own pressure
  level, which now follows `usageBytes` (driver usage minus the free space inside
  the runtime's memory chunks) instead of the driver's usage. The struct is
  unchanged.
- 2026-10-20: `remixapi_GetMemoryPressure` reports a `level` that never exceeds
  the runtime's own pressure level, which now follows the driver's usage. The
  reported `level`, `usageBytes`, `bytesToRelease` and `requestHostRelease` follow
  usage minus the free space inside the runtime's memory chunks.
  `requestHostRelease` is set from level 2 (High) by default.

### Added
- 2026-10-27: the bridge forwards `remixapi_InstanceInfoGpuInstancingEXT` on
  `DrawInstance` and the retained-instance calls (the client logs "Forwarding
  InstanceInfoGpuInstancingEXT"), so 32-bit hosts can GPU-instance a mesh.
- 2026-10-27: colour grading, as named exports: `remixapi_SetColorGradePreset`,
  `remixapi_SetColorGrade`, `remixapi_MixColorGradePresets`, `remixapi_GetColorGrade` and
  `remixapi_SaveColorGradePreset`, with `remixapi_ColorGradeEXT`
  (`REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT` = 0x4E560004, `REMIXAPI_COLOR_GRADE_VERSION` 1).
  Besides the grade (matrix, balance, saturation, contrast), the struct carries the tone curve's
  `anchor` (default 0.18) and `shoulder` (default 0.01). A 0 in either takes the default.
  The bridge exports and forwards them. `remixapi_Interface` is unchanged. docs/RemixApi.md,
  "Colour grading".
- 2026-10-19: the bridge forwards `remixapi_TextureInfoHostImportEXT` on
  `CreateTexture` (the client logs "Forwarding
  remixapi_TextureInfoHostImportEXT"), so 32-bit hosts can upload textures in
  any `vkFormat`, such as R16G16B16A16_SFLOAT. The terrain-layers blend
  (`remixapi_MaterialInfoTerrainLayersEXT`) clamps the blended albedo at 0, so
  layers and tint may be signed float textures.
- `remixapi_DrawScreenOverlayD3D11` / `PFN_remixapi_DrawScreenOverlayD3D11`
  and `remixapi_ScreenOverlayLayerD3D11`: a native D3D11 host passes its UI
  render targets (up to 8 layers, with a premultiplied-alpha flag and a
  destination rectangle). Remix copies them on the GPU into persistent images.
  This is a named export like the other D3D11 host extensions, so
  `remixapi_Interface` and `REMIXAPI_VERSION` are unchanged.
- Runtime option `rtx.enableGpuScreenOverlay` (Developer Options, "GPU Screen
  Overlay (D3D11 Hosts)"), default on.
- 2026-10-03: `remixapi_TextureInfoHostImportEXT`
  (`REMIXAPI_STRUCT_TYPE_TEXTURE_INFO_HOST_IMPORT_EXT` = 33), chained on
  `remixapi_TextureInfo`. `CreateTexture` then carries what a D3D11 import
  carries: `encoding` (`REMIXAPI_TEXTURE_ENCODING_*`) and an optional
  `vkFormat` that overrides `format`. The API recorder's replayer uses it for
  recorded imports. Commit `2f2a32165`.
- 2026-10-03: `REMIXAPI_TEXTURE_HOST_IMPORT_FLAG_REPLACE` in
  `remixapi_TextureInfoHostImportEXT.flags`. It swaps the image of the live
  texture registered under `hash` in place and keeps its texture-table slot,
  as `ReplaceD3D11TextureEx` does. Without a live texture under that hash, the
  texture is created normally. Commit `1066c5580`.
- 2026-10-03: `remixapi_CameraInfoFrameEventsEXT`
  (`REMIXAPI_STRUCT_TYPE_CAMERA_INFO_FRAME_EVENTS_EXT` = `0x4E560001`) and
  the flags `REMIXAPI_CAMERA_FRAME_EVENT_CAMERA_CUT`, `_HISTORY_INVALID` and
  `_ORIGIN_SHIFT`, chained on the main camera's `remixapi_CameraInfo`.
  A cut or history invalidation restarts temporal history without clearing
  the scene. An origin shift moves the camera history by `originShift`, so the
  host's origin move is neither camera motion nor a cut. Fork sTypes start at
  `0x4E560000`, a range upstream never uses. Commit `e3e70114e`.
- 2026-10-04: `remixapi_AssetKeyEXT`
  (`REMIXAPI_STRUCT_TYPE_ASSET_KEY_EXT` = `0x4E560002`), chained on
  `remixapi_MeshInfo` or `remixapi_MaterialInfo`. It gives the asset the key
  that mods, captures and caches use instead of the mesh hash or material
  content hash, plus aliases (keys older host versions produced) and an
  optional name. Replacements are looked up by key, then by each alias in
  order. It is honoured while `rtx.scene.assetKeys` is on (default off).
  Commit `0b31bedd5`.
- 2026-10-06: named export `remixapi_fork_ReplayIdentityTrace(const wchar_t*
  path, remixapi_fork_IdentityReplayResult* result)`, a diagnostics entry
  point for `remix_replay --identity-trace`. It replays a D3D9 identity trace
  (`rtx.d3d9Identity.traceFrames`) through a private draw-call tracker on the
  calling thread and presents nothing. It is declared in
  `src/dxvk/rtx_render/rtx_fork_d3d9_identity_trace_format.h`, not in
  `remix_c.h`, so `remixapi_Interface` and `REMIXAPI_VERSION` are unchanged.
  Commit `4408886d5`.
- 2026-10-20: a D3D9 point light can carry an asset key: `Falloff` =
  4932953.0 marks it, `Direction.x/y/z` hold bits 0-21, 22-42 and 43-63 as
  exact integers. With `rtx.scene.assetKeys`, replacements, the light picker
  and captures use the key instead of the position hash
  ([RemixApi.md](../docs/RemixApi.md#asset-keys-remixapi_assetkeyext)).

### Changed
- 2026-10-20: the bridge forwards `remixapi_AssetKeyEXT` on
  `remixapi_MeshInfo` (key and aliases). The bridge client and server must
  come from the same build.
- 2026-10-20: captures write an API material whose albedo was imported from
  the D3D9 device as `mat_<D3D9 image hash>` unless it has an asset key.

## [0.1000.1]

Remix Plus rebased onto NVIDIA dxvk-remix `0.6.5`, which grew
`remixapi_MaterialInfoOpaqueEXT` by four DLSS control-mask fields (112 -> 128
bytes on x64, 92 -> 108 on x86). The runtime and the bridge client read those
fields only from callers reporting API version `0.1000.1` or newer; binaries
built against `0.1000.0` keep working and receive upstream's defaults.

### Added
- remixapi_MaterialInfoOpaqueEXT.enableDlssControlMask (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskIntensity (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskToneStrength (inherited from upstream 0.6.5)
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskStructuralStrength (inherited from upstream 0.6.5)

### Changed
- `REMIXAPI_VERSION` is now `0.1000.1` (was `0.1000.0`).

## [0.1000.0]

Remix Plus adopts its own ABI version line (reserved MINOR `1000`), distinct
from stock NVIDIA dxvk-remix `0.6.x`. Because the runtime treats every minor as
breaking while MAJOR is 0, this version makes the runtime reject binaries built
against stock Remix `0.6.x` or against older Remix Plus `0.6.x` — the
`remixapi_Interface` layout and `remixapi_InstanceCategoryBit` ABI differ
between them. Rebuild plugins/hosts against this header.

### Added
- `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS` (bit 24) — the upstream name
  for the category previously exposed (under the fork) only as `LEGACY_EMISSIVE`.
  Use `SMOOTH_NORMALS`; the old alias has been removed (see below).

### Changed
- `remixapi_InstanceCategoryBit` bit values now match upstream NVIDIA exactly.
  An earlier fork build had shifted `IGNORE_ALPHA_CHANNEL` to bit 8 (cascading
  bits 8–20 up by one) to mirror the internal `InstanceCategories` order; this
  is reverted. The C↔internal mapping in `toRtCategories()` is by-name, so the
  public bit values are free to — and now do — match upstream. **Breaking** for
  any consumer that had serialized or hard-coded the shifted bit values.
- `remixapi_Interface.SetCameraMediumMaterial` moved from the middle of the
  struct (between `SetupCamera` and `DrawInstance`) to immediately after
  `Present`, mirroring upstream's canonical layout (upstream `2bac8874`). Fork
  extension functions remain appended after it. **Breaking** struct-offset
  change; rebuild consumers. The `sizeof(remixapi_Interface)` sentinel is
  unchanged (move, not add/remove).
- `REMIXAPI_VERSION` is now `0.1000.0` (was `0.6.4`).

### Fixed
- `remixapi_AutoInstancePersistentLights` no longer emits a per-frame
  `LockDevice` + empty `EmitCs` on the native-D3D9 present path when no external
  (C-API) light has ever been registered and no C-API scene work is queued.
  This empty per-frame dispatch disturbed the light pipeline for native-only
  consumers, manifesting as "persistent lights break all lights / heavy
  flicker." Genuine C-API light consumers are unaffected (the persistent
  re-instancing path is preserved).

### Removed
- `REMIXAPI_INSTANCE_CATEGORY_BIT_LEGACY_EMISSIVE` is **removed**. Its name
  implied emissive behavior, but it routed to bit 24 / `InstanceCategories::SmoothNormals`
  — so any caller using it got a silent wrong-category result. Removing it turns
  that into a compile error; use `REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS`.
  Source-only break — the bit layout, enum size, and struct offsets are unchanged
  (bit 24 still exists as `SMOOTH_NORMALS`), so no binary/ABI change and the
  `0.1000.0` version line is unaffected.

## [0.6.5]

### Added
- remixapi_MaterialInfoOpaqueEXT.enableDlssControlMask
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskIntensity
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskToneStrength
- remixapi_MaterialInfoOpaqueEXT.dlssControlMaskStructuralStrength

## [0.4.2]

### Added
- MaterialInfoOpaqueEXT.displaceOut

### Changed
- renamed MaterialInfoOpaqueEXT.heightTextureStrength to MaterialInfoOpaqueEXT.displaceIn

### Fixed

### Removed


## [0.4.3]

### Added
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceDiffusionProfile
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceRadius
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceRadiusScale
- remixapi_MaterialInfoOpaqueSubsurfaceEXT.subsurfaceMaxSampleRadius

### Changed

### Fixed

### Removed
