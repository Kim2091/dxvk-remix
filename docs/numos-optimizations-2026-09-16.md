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

## Measured in game (2026-09-17 00:00 session, 38 sampled frames, scale 1.0, 32/64 samples)

Same GPU clock as the baseline session: PostProcessing 0.399 / Upscaling 4.19 / ambient grid 0.172 ms all
match. Sun static (sunElevation 38.01, time cycle off, wind 0), so key-forced full updates were rare; the
interleave settings themselves are not logged, so clusters are attributed by their bimodal timing.

| Pass | Every frame (measured) | Half per frame (measured) | Ratio | Estimate was |
|---|---|---|---|---|
| Reflection dome | 0.371-0.383 (n=4) | 0.209-0.240, median 0.223 (n=31) | 0.59x | 0.41 -> 0.22 (met) |
| Sun grid | 0.850-0.894 (n=5) | 0.582-0.719, median 0.688 (n=29) | 0.78x | 0.50 -> 0.26 (missed) |
| Screen march, camera still | 3.11-3.17 (n=2, full-march frames) | 1.51-1.53 (n=7) | 0.48x | 0.5x (met) |
| Screen march, camera moving | not isolated | 0.55-1.40 and a 1.85-2.19 cluster | - | view-dependent |

Cloud total at the defaults: 1.8-3.1 ms across the session (2.6-2.8 with the camera still), against the
3.8-4.1 ms baseline in a different view yesterday; the sun grid's every-frame cost is 1.7x yesterday's in
this location, so only within-session ratios are trustworthy for it.

The sun grid under-delivers: half the threads, but each bakes columns two voxels apart, so a warp's taps
span twice the texels and the texture-cache coherence the full bake enjoyed is halved. Interleaving
whole 8-column blocks (or Z slices) would keep the warp footprint contiguous; untested. Quarter per frame
was not exercised in this capture. Water reflections were not looked at, so the dome interleave is
measured but not visually checked. Geometry edges and terrain shadows were reported clean, no shimmer.

## Reduced-scale shimmer (2026-09-17)

Diagnosis, from source and the live config rather than a capture (the shimmer is not in any log):

- `cloudDetailScale = 12` makes the detail volume repeat every 1 km, so the two live consumers of its
  base taps -- erosion at 0.58 (density) and micro-AO at 0.6 (up to +-27% shading) -- carry content from
  167 m down to 31 m (16 m vertically, the wispy squeeze). The march steps 49 m at the deck base and
  75-140 m at 3-10 km. Nothing in the sampler prefilters by step; `cameraDistKm` only gates the near HF
  fold and the (off) fine band. That is 2-4x under-sampled at the fine end, at every render scale.
- At native scale the per-pixel march jitter is animated and DLSS averages the resulting 1-px noise.
  `599ad5818` froze the jitter below native scale because animated jitter there flickers whole 4x4
  blocks; frozen, the same error becomes a screen-locked block pattern that crawls over moving cloud
  (the walking shimmer), and the DLSS projection jitter still wobbles every texel's ray by 1/8 texel
  through the Halton sequence (a still-camera flicker). Both are the same under-sampling seen twice.
- The composite reconstruction (`a79cac79d`) and the temporal smoother's removal (`f8544fa82`) are
  not the cause and are untouched; no accumulation is reintroduced.

Change, one build:

1. **Detail LOD** (`cloudDetailLodMode`, default 1 = reduced scale only; `cloudDetailLodBias`). The
   detail volume now has an 8-level box mip chain (`cloud_detail_noise_mip.comp.slang`, built once
   at init). Each detail tap in the march samples mip `log2(2 * texelsPerStep) + bias`, where
   texelsPerStep is the current step in that tap's texels -- the Nyquist rule -- so a step that cannot
   integrate a band gets it filtered toward the channel mean instead of aliased. Bakes and the shadow
   taps pass lod 0 (grids unchanged). Mode 2 applies it at native scale and to the dome as an A/B.
2. **Reduced-scale sample boost** (`cloudReducedScaleSampleBoost`, default 1 = off): optionally shrinks
   the screen pass's step target and adaptive floor and grows its cap by the boost when its RT is below
   native. The dome and bakes keep their spacing. (Shipped first as `cloudReducedScaleStepScale`, a
   spacing multiplier defaulting to 0.5 -- see the regression below.)
3. **Unjittered rays below native scale**: the reduced pass marches the unjittered camera ray (and
   reprojects unjittered), so a still camera renders bit-identical frames. The surface clamp still
   reads the jittered depth at the texel centre. At native scale the reprojection lookup now uses the
   jittered matrices at both ends, matching what DLSS assumes a pixel's ray is (yesterday's build
   reprojected unjittered directions into jittered pixels: a half-pixel inconsistency on the
   reprojected half).
4. **Labelled timing log**: `[Cloud profile]` and a new `[GPU stages] stage=CloudConfig` line carry
   `interleave=S/G/D` (resolved screen / sun-grid / dome periods), the cloud RT `extent`, `detailLod`
   and `stepScale`, so the next capture needs no cluster guessing.

Held: the sun-grid block interleave (cache-coherence fix for its 0.78x) is deferred, not dropped.

Expected, estimated: 25% scale with the defaults ~0.35-0.5 ms screen march (the 0.31-0.37 measured
plus the mip-sampling cost of the LOD, unmeasured); native scale unchanged in cost and appearance (LOD
off there by default). Whether the LOD look at 25% reads as acceptable is the user's call; the bias
slider trades residual crawl against softness live.

### Regression on the first deploy (2026-09-17 00:44 session), measured

The user set the Quality & performance controls to their minimums with the scale at 0.5 and saw ~3.5 ms
of clouds. The labelled log (`stage=CloudConfig`) shows why: `stepScale=0.25`, i.e. the new spacing
slider at its minimum, which meant FOUR times the samples per ray (step 37 m, floor 6 m, cap 256), on
every reduced-scale frame. The numbers agree with that and with nothing else:

| Segment (this session) | CloudScreen | cloud total |
|---|---|---|
| native, Half, sky-filled view (GBuffer 0.6 / Indirect 0.35 ms) | 3.3-3.6 | 4.4-4.6 |
| native, Quarter, same view | 2.12-2.16 | 2.85-3.1 |
| 25%, Quarter, 4x samples, LOD on | 1.29-1.74 | 2.0-2.5 |
| 50%, Quarter, 4x samples, LOD on, mixed view | 2.27-2.68 (one 3.87) | 3.0-3.5 |

25% at 4x samples is 0.31-0.37 x 4 = 1.3-1.5, which is what was measured, so the LOD's own cost is
small against it but not isolated (no `detailLod=0` reduced-scale frame in the capture). The native
3.5 ms is not a regression: that segment looked almost entirely at sky, the heaviest case for the
march, and the same view at Quarter cost 2.1. The control was wrong, not the march: its minimum was
the most expensive setting. Replaced by `cloudReducedScaleSampleBoost` (1 = native rate = cheapest =
default, up to 4), the stale key was removed from the game's rtx.conf, and the log now prints
`sampleBoost=` instead of `stepScale=`.

Native Quarter at 0.61x of Half (2.14 vs 3.5 in the same view) is the first clean measurement of that
mode; it is the full-resolution alternative to a reduced scale, at roughly the cost 50% scale would
have at Half.
