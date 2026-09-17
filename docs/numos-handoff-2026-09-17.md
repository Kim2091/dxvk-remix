# Numos handoff - 2026-09-17

> Screen scaling and cloud temporal reuse were removed on 2026-09-17. See
> [the native cleanup and LOD evaluation](numos-native-cleanup-2026-09-17.md) for current behavior;
> the descriptions of those paths below are historical.

Supersedes `numos-handoff-2026-09-14.md` for the cloud performance work; that file still holds the
alpha-foliage history, the preserved shimmer-fix commits, and the working rules, which are unchanged.


## DECISION 2026-09-17 (late): back to native scale. Read this before proposing reduced scale again.

The user's call, after seven sessions: *"the spatial resolution scaling led to literally all of these
issues, so we just need to find a way to optimize it while keeping that at 1x. temporal accumulation
isn't even needed with that. we're wasting performance while chasing performance in the wrong way."*

It is the right call, and the evidence is this run's own findings. **One decision produced three
layers of compensation, and each layer produced the next round's bug:**

1. `5543992b4` froze the march jitter's frame index below native scale (`cloud_render.comp.slang`).
   `fastJitter` taps a 128x128 **blue-noise** tile -- salt-and-pepper by construction -- so freezing
   its index stamps the same tile into the sky every frame, magnified by the upsample.
2. A frozen pattern is exactly what a temporal filter **preserves**, so DLSS-RR reconstructed it
   faithfully. That is the "powder" at 0.5 and the static 4x4 blocks at 0.25: the same defect at two
   magnifications, not two bugs.
3. Animating the jitter needs an averager, and `f8544fa82` had deleted the composite EMA the jitter
   policy still cites (`cloud_march_common.slangh`, "we have two [temporal filters]"). So the cloud
   pass grew its own accumulator (`cloudHistoryWeight`).
4. An unbounded accumulator smears when the camera translates, because cloud reprojection is
   rotation-only. So it grew a neighbourhood clamp (`cloudHistoryClampGamma`).
5. The clamp measured its box from **history** taps, which are already temporally smoothed, so in
   smooth cloud the box collapsed and switched the accumulator off -- leaving raw animated march
   noise. The user: *"it shimmers greatly when accumulation clamp is on at all."*

Cutting at the root removes all of it. Nothing was deleted -- every path is still reachable from its
slider -- but the defaults are now native scale with no screen-space temporal reuse.

### Defaults as of `fc1906291`

| Option | Now | Why |
|---|---|---|
| `cloudRenderResolutionScale` | 1.0 | the root cause |
| `cloudScreenInterleaveMode` | every frame | no screen-space temporal reuse |
| `cloudHistoryWeight` | 0 | not needed at 1x |
| `cloudHistoryClampGamma` | 0 | only existed to bound the accumulator |
| `cloudDetailLodMode` | always | mode 1 is inert at 1x; band-limiting is still correct |
| `cloudDetailLodBias` | 0 | -3 was justified by pairing with the accumulator; that is gone |
| `cloudSunGridInterleaveMode` | 2 (quarter) | **kept** -- a lighting bake, never caused an artifact, ~0.4 ms |
| `cloudSecondaryLutInterleaveMode` | 2 (quarter) | **kept**, same reasoning |

The game's `rtx.conf` had five stale overrides that would have beaten these; they were removed
(backup `rtx.conf.backup-pre-native-pivot-20260917-0500`).

### The price, measured

This hands back roughly **2.5 ms**. Native scale, every frame: screen march **2.58-2.95 ms**, bakes
**~1.05 ms**, total **3.8-4.1 ms**, against the **1.2-1.5 ms** the reduced-scale configuration ran
at. Recovering that algorithmically is hard and is not guaranteed. Optimisations from here must be
real reductions in work, not quality traded away spatially or temporally -- that trade is what this
whole arc was.

### Where to look, and the rule

Lighting measured only **~17%** of the march, so the march is dominated by **density evaluation**.
That points at fewer evaluations, not cheaper ones: empty-space skipping against the NVDF SDF
(`nvdfStepScale`, the conservative empty radius), where the adaptive step actually goes, early
termination, coarse-to-fine slab entry/exit, and thread-group shape (the 16x4 / 8x4
`cloudProfilingMode` variants have not been compared since the march changed).

**Instrument before optimising.** `DEBUG_VIEW_CLOUD_SAMPLE_COUNT` (912) maps density evaluations per
ray. This is not a style preference -- reasoning lost to counters twice in one run: the stale RR
normal guide, and the anchor cut I had argued could not fire and which the `stage=CloudHistory`
counters showed firing on ~8% of frames.

### Parked, closed by this decision -- not unresolved

- **The upscaler passthrough probe** (`cloudDebugInjectPattern`, static + animated). Two iterations
  both measured their own construction: the static pattern is temporally stable so a temporal filter
  preserves it by design, and the animated one survived perfectly even on **geometry** under DLSS-SR,
  which should be impossible and means the control failed. Confirmed along the way, from the log
  rather than inference: the live upscaler really is DLSS-RR, and both SR and RR read
  `m_compositeOutput` as `pUnresolvedColor`, so the injection does land in the upscaler's input.
  Unresolved and now moot.
- **Depth-aware cloud reprojection.** Entirely moot: there is no reprojection at 1x every-frame.

## Start here

- Renderer repo: `C:\Users\sparkles\Projects\Fable_5_testing\wsn3g`, branch `revised-9-10`. All work is
  committed locally; nothing is pushed. SHARC commits in the history are someone else's and untouched.
- Game: `C:\Users\sparkles\Projects\Games\Fallout New Vegas`. Deployed runtime is `.trex\d3d9.dll` +
  `d3d9.pdb`, DLL SHA256 `97C465AB1F37E1A2B11C06601B36DAD43522A1A84672D08F0E69C34DD1E4FE74`, PDB
  `DC9C30000188183F4EFB493DE541ADDD4E852FE1E4206AA5BC0BFB13376917A3` (commit `7d0817e67`). Backups in
  `.trex` are pruned to the newest three pairs (`backup-pre-cloud-interleave-20260916-235119`,
  `backup-pre-detail-lod-20260917-003650`, `backup-pre-boost-fix-20260917-005910`).
- Timing log: `rtx-remix/logs/remix-dxvk.log` with `rtx.profile.gpuStages = True` (on in the game's
  rtx.conf). Every 120 frames it prints the cloud stages plus a `stage=CloudConfig` line carrying
  `interleave=S/G/D` (resolved screen / sun-grid / dome periods; 1 = a forced full update),
  `extent=WxH`, `detailLod=`, `sampleBoost=`. `[Cloud profile]` lines carry the same. Read those, never
  infer settings from timing clusters again.
- Detailed notes with every measurement: `docs/numos-optimizations-2026-09-16.md`.

## Commits (all local, newest first)

- `7d0817e67` reduced-scale sample control became a boost (1 = cheapest, default); fixes the 4x regression.
- `7bbf6aa98` / `8d0279d33` regenerated `RtxOptions.md` (test_documentation needs it).
- `5543992b4` detail LOD (detail-volume mip chain sampled by march step), unjittered reduced-scale rays,
  jitter-consistent native reprojection, labelled timing log.
- `ed27a5611` measured interleave numbers recorded.
- `3ea7bf661` temporal interleave: screen pass (checkerboard / 2x2 with depth-validated reprojection),
  sun grid (columns), reflection dome (rows), with full updates on any key change / camera cut / lightning.

## The arc, measured (RTX 5080 Laptop, FNV, 1707x1067 internal, 32 base / 64 max samples)

| Pass | 2026-09-16 baseline, scale 1.0, everything every frame | 2026-09-17 01:02, the user's config (below) |
|---|---|---|
| Screen march | 2.58-2.95 | 0.47-0.77, median 0.65 (forced full frames 0.80) |
| Sun grid | 0.49-0.52 | 0.37-0.43, median 0.41 |
| Reflection dome | 0.39-0.48 | 0.16-0.18, median 0.17 |
| Ambient grid | 0.17-0.20 | 0.15-0.17, median 0.17 |
| Cloud total | 3.8-4.1 | 1.2-1.5, median 1.41 |

Different views on different days; within-session ratios (in the notes) are the trustworthy ones. Native
scale, same session yesterday: Half 3.3-3.6 / Quarter 2.12-2.16 in a sky-filled view.

The config that produced the right-hand column is the game's rtx.conf, NOT the code defaults:
`cloudRenderResolutionScale = 0.5` (default 1.0), all three `*InterleaveMode = 2` = Quarter (default 1 =
Half), `cloudDetailLodMode = 1` (default), `cloudDetailLodBias = -3` (default 0 -- see unverified),
`cloudReducedScaleSampleBoost = 1` (default). The retired `cloudReducedScaleStepScale` key was removed
from rtx.conf; `rtx.conf.backup-pre-boost-fix-20260917-005116` is the copy before that edit.

Detail LOD cost, measured: the two `detailLod=0` frames marched at 0.68-0.71 ms against 0.46-0.67 on the
neighbouring `detailLod=1` frames -- within noise, no measurable cost either way.

## Unverified -- do not write these up as confirmed

1. **The look at reduced scale.** The user reported the frame time ("about 1 ms") and nothing about the
   image. The session ran at 0.5, never 0.25, and rtx.conf carries `cloudDetailLodBias = -3` (the slider
   minimum, most likely from the same "everything to lowest" sweep), which at 0.5 drops the base-tap mip
   from ~4-5 to ~1-2, i.e. mostly turns the anti-shimmer off. Whether 0.25 (or 0.5) crawls or flickers
   with the LOD at bias 0 is therefore untested. One question settles it: at scale 0.25 and bias 0, turn
   and walk for 15 s -- do blocks crawl or flicker over the cloud?
2. **Reflections.** The dome's row interleave has been measured (0.41 -> 0.17 at Quarter) but never looked
   at; the user could not find water. Any wet surface with cloudy sky, flip Cloud Reflection Interleave
   Every frame / Quarter.
3. Native Quarter's look was never separately reviewed either; the user has run it for two sessions
   without complaint, which is not the same thing.

## Deferred

- Sun-grid block interleave (contiguous 8-column blocks instead of stride-P columns, to recover the
  texture-cache coherence the column stagger loses). Measured Half 0.69 vs Every 0.88 (0.78x, expected
  0.5x); Quarter 0.41. Estimated worth: ~0.1-0.15 ms/frame at the current 1.4 ms total. Not worth a user
  session on its own; land it bundled with the next cloud change.
- Water-reflection check and the 0.25 look check above.

## What each option does (new since 2026-09-14)

- `cloudScreenInterleaveMode` 0/1/2: every pixel / half (checkerboard) / quarter (2x2) marched per frame;
  the rest reprojected along the camera rotation, re-marched unless every tap has the same surface class
  at a matching distance (`cloudHistoryDepthTolerance`). Full march on key change, camera/anchor cut,
  lightning. RT + depth companion are a ping-pong pair (`getCloudRenderRT()` follows the index).
- `cloudSunGridInterleaveMode`, `cloudSecondaryLutInterleaveMode`: same periods for the sun grid's X
  columns and the dome's rows. The dome's mip chain still rebuilds every frame (Sky Cloud Bleed reads it).
- `cloudDetailLodMode` 0/1/2: off / below native scale only / always. Each detail tap samples mip
  `log2(2 * texels per step) + cloudDetailLodBias` of the detail volume's new 8-level box chain
  (`cloud_detail_noise_mip.comp.slang`, built once at init). Bakes and shadow taps stay at level 0.
- `cloudReducedScaleSampleBoost` 1..4: extra samples per ray below native scale; 1 = native rate = cheapest.
- Below native scale the pass marches and reprojects the UNJITTERED camera ray (a still camera is
  bit-identical frame to frame); the surface clamp still reads the jittered depth at the texel centre.

## Rules that bit this week

- A slider whose minimum is the most expensive setting will be dragged there in an "everything to lowest"
  sweep. Make the cheapest setting the minimum, and say so in the label.
- `AtmosphereArgs` grows only in whole 16-byte rows; `static_assert(sizeof(AtmosphereArgs) % 16 == 0)` is
  in `rtx_atmosphere.cpp`. Zero new per-frame fields in `normalizeForSkyLutCache`.
- `meson compile` twice (the second relinks the regenerated shader headers), then `spirv-val` every module
  and `mmap.find` the changed ones in the DLL before deploying. Regenerate `RtxOptions.md` from
  `_Comp64Release/tests/rtx/unit/` after any option change or `test_documentation` fails.
