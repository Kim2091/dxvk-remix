## Remix API Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- GameStateStore keys `__weather.drift_speed` and `__weather.drift_intensity` — plugin-controlled cloud-drift speed and intensity multipliers. Both default to 1.0 when unset. Smoothed inside the renderer with tau = 1.0s. See [`docs/integrators/weather-presets.md`](integrators/weather-presets.md) section 8 for the recommended per-preset values and integration pattern.

### Changed

### Fixed

### Removed
- `rtx.atmosphere.sunDisc` (GameStateStore/config key) — removed. The option had no consumer (the sun disc is rendered via the sun-as-distant-light / NEE path); setting it had no effect.


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


## [0.1000.1] - 2026-08-04

### Added
- `remixapi_Interface.UpdateMeshBatched` — batched, vertex-data-only in-place
  update of an already-registered mesh (handle = `info->hash`). Rewrites the
  mesh's vertex bytes (positions/normals/texcoords/colors) while keeping the
  handle, so per-frame-regenerated geometry keeps temporal identity: the BLAS
  is refit instead of rebuilt and the renderer produces real per-vertex motion
  vectors (fixes duplicate-trail ghosting under DLSS/RR for game-CPU-skinned
  meshes). Surface count, per-surface vertex count, index count, and skinning
  presence must match the registered mesh or the whole update is dropped with
  a WARN; indices/materials/skinning are never changed through this entry
  point. Applied at the next render-thread flush point in call order with
  queued mesh creates. Appended at the end of `remixapi_Interface`
  (`sizeof` sentinel 328 → 336); older runtimes leave the slot `NULL`, so
  callers should feature-detect and fall back to destroy+create. No
  `REMIXAPI_VERSION` bump (append-only slot addition).
- `REMIXAPI_INSTANCE_CATEGORY_BIT_MAKE_EMISSIVE` (bit 27, append-only — no
  ABI change) and the matching `InstanceCategories::MakeEmissive`. Patches the
  draw's opaque material to emit light from its albedo (texture when present,
  else the albedo colour), scaled by the new `rtx.emissiveTexturesIntensity`
  and the global `rtx.emissiveIntensity`. Also reachable without the API bit
  by tagging texture/mesh hashes into the new `rtx.emissiveTextures` list —
  exposed as "Make Emissive" in the dev-menu texture grid and as USD
  `remix_category:make_emissive`. Unlike the removed
  `REMIXAPI_INSTANCE_CATEGORY_BIT_LEGACY_EMISSIVE` (which silently routed to
  SmoothNormals), this bit has a real consumer on both draw paths.

### Changed

### Fixed
- "Ignore Alpha Channel of Textures" (`rtx.ignoreAlphaOnTextures`, dev-menu
  grid, and `REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ALPHA_CHANNEL`) now works
  for API-submitted draws. Its effect is a material-level flag that only
  D3D9's legacy→opaque material conversion ever set; API-created materials
  never pass through that conversion, so the tag was inert on the API path.
  The flag is now applied from the instance category at instance update, which
  serves both paths.

### Removed
