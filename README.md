# Remix Plus

[![Build Status](https://github.com/RemixProjGroup/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/RemixProjGroup/dxvk-remix/actions/workflows/build.yml)

**Remix Plus** is a community-maintained fork of NVIDIA's
[`dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix) — created
and led by [Kim2091](https://github.com/Kim2091) — that extends the
Remix SDK API for modern-game plugin integrations. It
brings the SDK extensions developed in the gmod-rtx community fork —
batched mesh and light creation, plugin-injected game state, UI state
plumbing, VRAM control, additional tonemap operators, the Numos
sky system, and more — onto a clean, NVIDIA-rebase-friendly
base, so plugin authors and game integrations can build on a
maintained codebase that's API-compatible with the broader Remix
ecosystem.

Like upstream `dxvk-remix`, Remix Plus is a fork of
[DXVK](https://github.com/doitsujin/dxvk) that overhauls the D3D9
fixed-function pipeline for path-traced remastering. The `bridge`
subfolder enables 32-bit games to communicate with the 64-bit
runtime.

> Bugs encountered with Remix Plus belong in this repo's issue
> tracker, not in upstream DXVK or NVIDIA `dxvk-remix`.

## What's new vs upstream `dxvk-remix`

### Remix SDK API extensions

- **Batched mesh creation** — `CreateMeshBatched` for high-throughput
  geometry submission paths.
- **Batched mesh updates** — `UpdateMeshBatched` rewrites an existing
  external mesh's vertex data in place, keeping temporal identity (and
  therefore motion vectors) for geometry the game regenerates every
  frame. See [Dolphin and resident-host support](#dolphin-and-resident-host-support).
- **Batched light creation + deferred updates** — `CreateLightBatched`,
  `UpdateLightDefinition` for per-frame light churn.
- **UI state query/set** — `GetUIState` / `SetUIState` so plugins can
  observe and drive Remix's developer UI from outside the runtime.
- **Texture-hash category mutation and readback** — `AddTextureHash`,
  `RemoveTextureHash`, `dxvk_GetTextureHash` for plugin-driven texture
  classification at runtime, plus `GetTextureHashList` to read back a
  snapshot of any `HashSet` RtxOption by full option name.
- **D3D11 shared-texture handles** — `dxvk_GetSharedD3D11TextureHandle`
  for interop with D3D11-side rendering paths.
- **VRAM control** — `RequestTextureVramFree`, `RequestVramCompaction`,
  `GetVramStats` give plugins a driver-view handle on memory pressure.
- **Plugin-injected game state** — `SetGameValue` writes named values
  into a fork-owned store; `GameValueReadBool` and `GameValueReadNumber`
  graph (Sense) components read them back inside replacement logic.
- **`externalMesh` field** on `RasterGeometry` for capture/replacement
  parity when geometry comes in via the Remix API path.
- **`InstanceCategoryBit` ABI** synced to the gmod/plugin layout so
  category bits round-trip correctly across the API boundary.

### Tonemapping & auto-exposure

- **Eight tonemap operators** in the UI dropdown: Hill ACES, Narkowicz
  ACES, Hable Filmic, AgX Minimal, Lottes 2016, PsychoV17_Beta, Gran
  Turismo 7 (SDR), and Neutwo. Each operator has its own parameter
  panel — controls are visible at a glance instead of buried.
- **AgX Minimal** (Benjamin Wrensch / MIT) replaces the older
  multi-knob AgX surface. Look presets: None / Golden / Punchy.
- **PsychoV17_Beta** — Slang port of renodx Psycho Test 17 (Carlos
  Lopez Jr. / MIT). Stockman-Sharpe LMS + Naka-Rushton cone response
  + gamut compression.
- **Gran Turismo 7 reference** — Slang port of Polyphony Digital's
  SIGGRAPH 2025 GT7 tone-mapping reference (MIT). SDR mode, ICtCp UCS.
- **Hable presets** (Hejl, Uncharted 2) with the original parameters.
- **Perceptual auto-exposure** — Stockman-Sharpe Yf histogram +
  geometric-mean adaptation + first-site cone-contrast law. Asymmetric
  in log-exposure space: cone-bleach is fast (~0.10–0.20 s),
  rod-recovery is slow (~0.50–1.50 s). Two tau sliders replace the old
  Adaptation Speed / EV-Min / EV-Max / Average Mode controls.

The legacy Tonemapping Mode (Global / Local / Direct) combo, the local
tonemapper, the dynamic tone curve / Tuning Mode sliders, the User
Brightness slider, and the exposure-compensation curve are removed —
the apply pass always runs in operator-only mode.

### Numos sky system

- **Numos atmosphere (Hillaire scattering)** — physically-based atmospheric scattering
  ported from the gmod-rtx community fork. Daylight, sunset, and
  twilight all behave correctly without manual fog tuning.
- **Volumetric clouds** — procedural FBM cloud layer with
  weather-driven coverage and Nubis-style spatial variation. Anvil,
  shear, and vertical-profile shaping are artist-tunable. Renders
  through a sky-dome curvature with sample-seam jitter to hide
  stepping artifacts. Sun and any number of moons cast shadows
  through the volume; twilight and night cloud lighting are
  physically correct rather than tuned-by-eye. Shadow-tap cost is
  heavily reduced via multi-octave density approximation,
  cadence-decoupled shadow caching, combined-moon marching, and
  density-gated skipping.
- **Night sky** — stars, milky way, shooting stars, and airglow,
  with sidereal rotation so the celestial sphere actually moves.
  Multi-moon support: independent elevation / rotation / phase per
  moon, unified moon-disk eval with surface-style presets (Rocky,
  Volcanic), soft radial glow/halo, and physically-scaled lunar
  illumination on the cloud volume.

### Hardware skinning

- **HW skinning** with capture and replacement parity, so skinned
  meshes injected via the Remix API path participate in capture and
  asset replacement the same as fixed-pipeline geometry.

### Dolphin and resident-host support

The runtime half of the Dolphin Remix backend (client:
[`Kim2091/dolphin`](https://github.com/Kim2091/dolphin) branch
`remix-backend`, which vendors `remix_c.h` from
`feat/numos-panel-curation` @ `08cccb4c`). Dolphin drives Remix
entirely through the C API and keeps one process alive across many
games — two things that broke assumptions the runtime had never had
to question.

**Running several games from one process**

- **Per-game paths and config re-resolved** — a host that points the
  path and config env vars at a different folder before each game was
  ignored: both were resolved once under run-once guards, so a
  resident runtime kept reading and saving into the first game's
  files. `CreateD3D9` now refreshes `RtxFileSys` and the log when the
  path vars change, and `RtxOptions::Create`'s repeat-call path
  rebuilds the system option layers when the config vars change.
  Unchanged environment runs none of it.
- **`DXVK_USER_CONFIG_FILE`** — `user.conf` was the one config layer
  whose path could not be redirected, resolved as a bare filename
  against the working directory. It now goes through the same
  `createLayersFromEnvVar` path as `dxvk.conf` and `rtx.conf`. Every
  edit made in the Remix UI targets the user layer, so without this
  one game's tagged hashes landed in another game's settings.
- **Window subclasses detached when the device is destroyed** —
  `remixapi_Shutdown` force-releases the device to a zero refcount
  regardless of who else holds one, so the swapchain outlived its
  parent and `D3D9WindowProc` walked into freed memory on the host
  application's UI thread.
- **Sentry shut down before runtime unload**, so a later exception
  cannot dispatch into unmapped module code.
- **Number formatting pinned to `std::locale::classic()`** in
  `dxvk::str::format` and `Config::generateOptionString`. A host can
  replace the global locale out from under the runtime — Dolphin does,
  at startup — and every hash printed or written to `rtx.conf` came
  out digit-grouped (`1,AC8,CA7,5E4,0AA,123`). The read side is
  `stoull`/`strtoull`, which is C-locale and stops at the first
  separator, so none of it round-tripped.

**Texture categorization on the API path**

- **Dev-menu categories now reach API-submitted draws** — Sky, Ignore
  and the rest of the grid-assigned categories never applied to
  API geometry. The hook resolved identity from the albedo hash
  alone, so untextured draws had none at all — and untextured is
  exactly what the occluders turned out to be. Identity now falls
  back to the client's own `remixapi_MeshInfo::hash`, and the hook
  resets to the client-supplied categories and camera type before
  applying, since `submitExternalDraw` reuses one `DrawCallState`
  across every submesh while `setCategory` is add-only.
- **`GetTextureHashList`** — categories could be mutated from the
  client (`AddTextureHash` / `RemoveTextureHash`) but never read
  back, so a client routing its own draws — Dolphin's screen-overlay
  UI path, where `rtx.uiTextures` has no runtime consumer — could not
  learn what the user had tagged. Returns a snapshot of any `HashSet`
  RtxOption by full option name.
- **Make Emissive category** (`rtx.emissiveTextures` +
  `rtx.emissiveTexturesIntensity`, USD `remix_category:make_emissive`,
  API bit `MAKE_EMISSIVE`) patches the tagged draw's opaque material
  to emit from its albedo. **Ignore Alpha Channel** is fixed on the
  API path in the same change: its real effect was a material flag
  set only in D3D9's legacy→opaque conversion, so the tag was inert
  for API-created materials. Both are now applied by one fork hook at
  instance update, covering both draw paths.
- **Tagging view** — an untextured element has no thumbnail, so it
  never appears in the grid and clicking it in the world is its only
  entry point; routing the 2D layer into the world is something only
  the host can do, since overlay draws never become runtime draw
  calls. The dev menu publishes the request through the existing
  game-state store as `__remix.tagging.worldView`, with an automatic
  mode under `__remix.tagging.worldViewFollowsMenu` that follows the
  menu's open state. Host-seeded, so a title with no host listening
  shows nothing rather than a control that silently does nothing.
- **Emissive tag strength surfaced** —
  `rtx.emissiveTexturesIntensity` had no widget anywhere and was
  reachable only by hand-editing `rtx.conf`. It now appears both in
  the Make Emissive category and in Lighting under the global
  Emissive Intensity it multiplies against.
- **Ignore takes precedence over Sky**, and a category change
  bypasses the preserve path for one dynamic update. The preserve
  path is **off by default** — not broadly compatible yet.

**External mesh correctness**

- **`UpdateMeshBatched`** — games that CPU-skin characters resubmit
  new vertex bytes every frame, and recreating the mesh each time
  destroys temporal identity, so that geometry rendered with zero
  motion vectors and ghosted under DLSS/RR. The batched-only entry
  point rewrites an existing external mesh's vertex data in place, on
  a fresh buffer rather than the live mapped one, and bumps
  `hashes[VertexShader]` so the BLAS is refit with history buffers
  populated and the mesh gains real per-vertex motion vectors.
- **Capture crash on skinned external meshes fixed** — on the API
  path bones-per-vertex is a mesh property while the bone matrices
  arrive per-instance, and `toRtDrawState` parses the EXT before the
  mesh handle is resolved, so `numBonesPerVertex` stayed 0 with
  `numBones > 0`. The skinning dispatch reads the geometry's copy and
  worked; the capturer reads the instance's copy, and its
  `(bonesPerVertex - 1)` weight loop underflowed `size_t` and read off
  the end of the blend-weight buffer, killing the process on every
  capture of a scene containing an API-skinned mesh.
- **External mesh replacement lifetime fixed** — replaced external
  draws were stale-on-arrival (`frameLastSeen == 0`), so their
  `ReplacementInstance` was destroyed in the same present it was
  created, before the TLAS build; what rendered historically was
  zombie instances. Adds the `rtx.logReplacementInstanceGC`
  diagnostics that measured the mechanism.
- **`includeOriginal` replacement prims resolve their client
  material** — the external replacement branch passed a blank
  `LegacyMaterialData` default, so an enhanced anchor mesh rendered
  solid white. Latent until the lifetime fix above, because that prim
  never previously survived to the TLAS.

### Capture and overlay quality-of-life

- **Overwrite-existing-capture** checkbox in the capture dialog.
- **Null-image / null-map / dimension guards** on capture export
  paths — eliminates a class of crashes when capturing edge-case
  resources.
- **Keyboard and mouse events** forwarded to ImGui on the legacy
  `WndProc` fallback path, so plugin-API-driven overlays receive
  input even when a game menu captures raw input.
- **Quieter logs** — spammy swapchain-recreate throws and repeated
  mesh-registration warnings silenced.

### Engineering

- **Fork-touchpoint pattern** — fork logic is extracted into
  dedicated `rtx_fork_*.cpp` modules, with one-line dispatches in
  upstream files. Reduces NVIDIA-rebase pain by ~54% (measured) and
  makes the fork's surface area auditable. See
  [`docs/fork-touchpoints.md`](docs/fork-touchpoints.md) for the
  authoritative inventory.
- **PR template fridge-list reminder** keeps the discipline honest.

## Contributing

Contributions are welcome. Whether you write Remix plugins, ship a
game integration, or want to make this fork better — start with the
contribution guide:

**[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** covers setup,
build, fork-touchpoint discipline, code style, and PR submission.

The short version:

1. Fork [`RemixProjGroup/dxvk-remix`](https://github.com/RemixProjGroup/dxvk-remix).
2. Branch on your fork — any name is fine.
3. Keep PRs small and focused.
4. Build clean (release flavor, exit code 0, zero errors).
5. Open a PR against canonical's `main` branch.
6. Add yourself to `src/dxvk/imgui/dxvk_imgui_about.cpp` under
   "Github Contributors".

If you touch any upstream file, update
[`docs/fork-touchpoints.md`](docs/fork-touchpoints.md) in the same
commit — that's the one rigid rule.

Questions? File an issue or ask on the
[RTX Remix Discord](https://discord.gg/c7J6gUhXMk).

## Quick build

Detailed requirements and walkthrough live in
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). The compressed
version, assuming you have Visual Studio 2019 (with the v142
toolchain), the Windows SDK, Meson 1.8.2+, the Vulkan SDK
1.4.313.2+, and Python 3.9+:

```powershell
git clone --recursive https://github.com/<your-fork>/dxvk-remix.git
cd dxvk-remix
.\scripts\build.ps1
```

`scripts/build.ps1` is the fork-side runtime build entry point — it
discovers Visual Studio via vswhere, runs `meson setup`/`compile`/
`install`, and verifies artifacts. It defaults to the `release`
flavor; pass `-Flavor debug` or `-Flavor debugoptimized` for the
instrumented flavors, `-Clean` for a fresh build dir, or
`-EnableTracy` for the Tracy profiler.

Output `d3d9.dll` lands in `_Comp64Release/src/d3d9/` and is
installed to `_output/`. Configure game targets via
`gametargets.conf` (copy `gametargets.example.conf`) and the build
will deploy automatically.

To build the 32-bit-to-64-bit bridge separately, use the fork-side
bridge wrapper `.\bridge\scripts\build.ps1` (builds the x64 server
and x86 client + launcher into `bridge/_output/`).

## Remix API

If you're integrating Remix into a game with available source, you
can either use the D3D9 surface directly (Remix's `d3d9.dll`
implements D3D9) or program against the Remix C API to push game
data into the renderer. Start with
[`docs/RemixSDK.md`](docs/RemixSDK.md) for setup and the mental
model, then see [`docs/RemixApi.md`](docs/RemixApi.md) for the full
API reference. The C header is
[`public/include/remix/remix_c.h`](public/include/remix/remix_c.h),
with a type-safe C++ wrapper at
[`public/include/remix/remix.h`](public/include/remix/remix.h).

## Project documentation

- [Anti-Culling System](docs/AntiCullingSystem.md)
- [Cloud System](docs/CloudSystem.md)
- [Contributing Guide](docs/CONTRIBUTING.md)
- [Contributing Style Guide](docs/CONTRIBUTING-style-guide.md)
- [Foliage System](docs/FoliageSystem.md)
- [Fork Touchpoints](docs/fork-touchpoints.md)
- [GPU Print](docs/GpuPrint.md)
- [Opacity Micromap](docs/OpacityMicromap.md)
- [Remix API (hub reference)](docs/RemixApi.md)
- [Remix API Changelog](docs/RemixApiChangelog.md)
- [Remix API Surface (auto-generated)](RemixApiSurface.md)
- [Remix Config](docs/RemixConfig.md)
- [Remix Logic](docs/RemixLogic.md)
- [Remix SDK Setup](docs/RemixSDK.md)
- [Remix Sky API](docs/RemixSkyAPI.md)
- [Rtx Options](RtxOptions.md)
- [Terrain System](docs/TerrainSystem.md)
- [Unit Test](docs/UnitTest.md)

## Team

- [Kim2091](https://github.com/Kim2091) — project lead and lead maintainer
- [CR](https://github.com/sambow23) — maintainer
- [TheGreatHMMMM](https://github.com/TheGreatHMMMM) — contributor
- [Gokuwashere](https://github.com/BrunchyChineapple) — contributor

## Credits

Remix Plus stands on the work of:

- [DXVK](https://github.com/doitsujin/dxvk) — D3D9 → Vulkan
  translation layer.
- [NVIDIA `dxvk-remix`](https://github.com/NVIDIAGameWorks/dxvk-remix) —
  path-traced remastering fork of DXVK.
- The **gmod-rtx community fork** — origin of most of the SDK
  extensions Remix Plus carries.

Thanks to all the contributors whose work makes this possible.
