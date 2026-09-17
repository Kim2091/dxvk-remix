# Numos cloud optimizations - 2026-09-16

Base commit: 02e1cd9fe (branch `revised-9-10`). Follows `numos-handoff-2026-09-14.md`.

## Baseline, measured

Source: the game's own `rtx.profile.gpuStages = True` log (`rtx-remix/logs/remix-dxvk.log`), session of
2026-09-16 19:51-20:08, Fallout: New Vegas, RTX 5080 Laptop, internal resolution 1707x1067, base samples 32,
`cloudViewSamplesMax` 64. All figures below are GPU timestamps, milliseconds per frame.

The "current config vs 100% scale" discrepancy in the brief is resolved by the log: that session
STARTED at `cloudRenderResolutionScale = 0.25` (logged at load; cloud RT 427x267), the user live-toggled
0.75 / 1.0 / 0.25 during it, switched to 1.0 at 20:07:55, swept the scale down and back, and exited at
20:08 with rtx.conf saved at 1.0. So the on-disk config (scale 1) is the live config now, and both of the
user's numbers are visible in the same log.

| Stage (log label)              | scale 0.25 (frames 27000-28400) | scale 1.0 (frames 28520-29600) |
|--------------------------------|---------------------------------|--------------------------------|
| CloudScreen (screen march)     | 0.31-0.37                       | 2.58-2.95                      |
| AtmosphereCloudSunGrid         | 0.49-0.52 (one 1.35 spike)      | 0.49-0.52                      |
| AtmosphereCloudSecondaryLut    | 0.39-0.48                       | 0.39-0.48                      |
| AtmosphereCloudAmbientGrid     | 0.17-0.20                       | 0.17-0.20                      |
| AtmosphereCloudSkyTransmittance| 0.004                           | 0.004                          |
| AtmosphereCloudShapeAndNvdf    | 0 (amortized; no re-bake)       | 0                              |
| Cloud total                    | ~1.4                            | ~3.8-4.1                       |

Attribution at scale 1.0: the screen march is ~70% of the cloud cost; the three every-frame bakes that
do not depend on the screen at all (sun grid, reflection dome, ambient grid) are ~1.1 ms, ~28%. The
dome is the odd one: 65k texels cost 0.4 ms while 1.8M screen pixels cost 2.8 ms, i.e. a dome texel is
~4x a screen pixel -- it has no surface clamp, and its horizon rows are the 50+ km grazing marches.

The earlier labelled captures (2026-09-14, in the previous notes) put lighting at ~17% of the screen
pass and moon shadows at ~3%; density/detail sampling and traversal are the rest. Nothing cheap is
left inside a sample, so the work here reduces how many samples run per frame, not what a sample does.

## Change: temporal interleave, three places

All three are the same idea with the same safety rule: do 1/2 (or 1/4) of the work per frame, and
make the result complete every frame anyway. The rule that keeps them lossless is that the interleave
only ever spans frames whose full results would have agreed -- on any frame the voxel-grid cache key
changes (sun/moon direction past `skyViewRebakeGranularityDeg`, camera past a 47 m voxel, wind or
evolution past `cloudVoxelGridRebakeGranularityKm`, any cloud parameter, an NVDF publish, a clouds
toggle) every one of them runs a full update. That key is now compared every frame, not only under the
granularity gate, and is the single "cloud inputs changed" signal (`RtxAtmosphere::resolveCloudInterleave`).

1. **Screen pass** (`cloudScreenInterleaveMode`, default half). Each thread of `cloud_render.comp.slang`
   owns a 2x1 cell (2x2 at quarter) and marches exactly one pixel of it; the fresh pixel alternates
   per row and per frame, a checkerboard. The other pixel is reprojected from the previous frame's
   RT: the pixel's unjittered world direction is projected through the previous frame's unjittered
   world-to-projection, so sky reprojects exactly under rotation and a camera at rest maps every pixel
   onto itself (a sub-1% offset is snapped). Every bilinear tap must resolve the same surface class
   (sky vs geometry, via the depth companion's `.z`) at a matching distance (`cloudHistoryDepthTolerance`,
   0.1 relative) or the pixel is marched fresh -- silhouettes and moving objects are never blended
   across, and nothing on screen is older than one period. Full marches also run on camera cuts
   (`isCameraCut` and the anchor-delta cut, since Gamebryo keeps translation out of the view matrix)
   and while a lightning flash is live (a transient would otherwise reach only half the pixels).
   Warp-uniform by construction: the fresh/reprojected split is per THREAD, not per lane, because this
   pass is warp-synchronous and a per-lane skip idles lanes without shortening the warp. The RT and
   depth companion are now a ping-pong pair; `getCloudRenderRT()` follows the index, so composite and
   the debug view are untouched. The composite's own reduced-scale reconstruction is unchanged.
2. **Sun grid** (`cloudSunGridInterleaveMode`, default half). `cloud_sun_density_grid.comp.slang` bakes
   X columns `period * x + phase`; the dispatch covers 1/period of the columns. Adjacent columns are
   at most one period old and the trilinear read blends across them. This is materially different
   from the every-8-frames stagger removed on 2026-05-19: that updated the whole grid at once every
   8th frame (a 2 Hz jump at 16 fps); this moves every frame with a bounded one-frame lag on half
   the columns, and any key change forces a full bake. The ambient grid is left at full rate
   (0.18 ms; the column scan already made it cheap).
3. **Reflection dome** (`cloudSecondaryLutInterleaveMode`, default half). `cloud_secondary_lut.comp.slang`
   marches rows `period * y + phase`; the mip chain the Sky Cloud Bleed reads still rebuilds from the
   whole mip 0 every frame. Rows rather than a checkerboard for the same warp reason as above. Full
   bakes on key change and camera cuts.

CB layout: one new 16-byte row at the tail of `AtmosphereArgs` (three packed interleave words plus the
reprojection tolerance), zeroed in `normalizeForSkyLutCache` so the per-frame phase never keys a bake,
with a `static_assert(sizeof(AtmosphereArgs) % 16 == 0)` added in `rtx_atmosphere.cpp`.

## Expected cost, estimated (not yet measured in game)

At the default half settings and the scale-1.0 baseline above: screen 2.8 -> ~1.5 ms (half the marches
plus the re-marched silhouette band and the reprojection reads), sun grid 0.50 -> ~0.26, dome 0.41 ->
~0.22 (the mip chain is not interleaved). Cloud total ~3.9 -> ~2.2 ms. Frames on which the key changes
(roughly once a second of game time at default sun speed, plus once per 47 m walked) cost the full
baseline. These are estimates from the work fraction; the measured numbers go in the section below.

## Validation

- Release Meson compile succeeded (one relink pass; the second invocation only regenerated version.h).
- All 521 SPIR-V modules in `_Comp64Release/src/dxvk/rtx_shaders` pass `spirv-val --target-env vulkan1.3
  --scalar-block-layout` (the shared `atmosphere_args.h` grew, so every atmosphere consumer rebuilt).
- The 12 directly changed modules (seven `cloud_render*` variants, `cloud_sun_density_grid`,
  `cloud_secondary_lut`, both `cloud_ambient_density_grid*`, `composite`) were found byte-for-byte in the
  final `d3d9.dll` with `mmap.find`.
- `static_assert(sizeof(AtmosphereArgs) % 16 == 0)` holds; `git diff --check` clean.
- DLL SHA256 `F8106BDAA6ABD064A3DB9D292A6375101E6071D8AF4D51D76799320B3A968404` (280,542,208 bytes);
  PDB SHA256 `91BC7DA07E5A831C83086D35E0F0262231A614AEBEDF58B65A14883F563EA8C7`.
- In-game timings and the visual check are pending: the `[GPU stages]` log lines (`rtx.profile.gpuStages`
  is on in the game's rtx.conf) give CloudScreen / AtmosphereCloudSunGrid / AtmosphereCloudSecondaryLut
  per 120 frames; each of the three modes can be flipped live between Every frame and Half per frame in
  Atmosphere -> Clouds -> Quality & performance for an A/B in one session.
