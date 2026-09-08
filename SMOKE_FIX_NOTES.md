# Smoke fix: decouple the lobe wavelength from Detail Scale (2026-09-07)

Baseline `565c09ef3` on `clouds/world-space-stage4a`. Five files, one new option, no default
changes to anything that renders at the default config. Not built, not deployed, not committed.

## The mechanism

Two separate things, both read from the code and the live FNV `rtx.conf`.

**The smoke is geometric, not erosion.** The conf runs `nubis3ErosionStrength 0`,
`nubis3EdgeErosion 0` and `cloudDetailStrength` at its default 0, so the only sub-kilometre
structure in the density field is the mid-tap level-set displacement in
`sampleCloudDensityNubis3` (`sdfLive -= wobbleMid * nubis3ShapeVarietyKm`). That tap's frequency
was `0.193 x detailFreq`, i.e. the lobe wavelength was
`cloudNoiseTileKm / (cloudDetailScale x 0.193 x 6)`:

| | default (`cloudDetailScale 4.3`) | live (`cloudDetailScale 12`) |
|---|---|---|
| coarsest lobe wavelength, horizontal | 2.41 km | 0.86 km |
| same, vertical, wispy channel (baker's `kWispSqueeze` doubles the vertical cycles) | 1.20 km | 0.43 km |
| amplitude (`nubis3ShapeVarietyKm`; the conf's 2 is clamped to 1.5 CPU-side) | 1.11 km | 1.5 km |

At the deck base `typeShaped` drops to ~0.48 (`cloudTypeMean 1` with the default 0.54 spread, minus
the 0.25 altitude swing), so about half of `wobbleMid` is the wispy channel, whose centred range is
[-0.5, +0.28]. Displacement of the underside at the live conf: wispy [-0.39, +0.22] km, billowy
[-0.04, +0.11] km, about 0.6 km peak-to-peak against a 0.43 km vertical wavelength. A displaced
level set only bulges while the displacement gradient is below the SDF's unit gradient; for a
sinusoid that is pk-pk < wavelength / pi (~32%). The live conf sits at ~140% (gradient ~4.4), so the
underside folds into hanging sheets and curly-alligator web strands. Those strands are the "long
stringy wisps, almost like smoke". At the default config the same ratio is ~37% worst case, ~18%
at type 1, which is the bulging regime the amplitude was tuned in.

**Profile depth is the coverage knob only because `cloudDensity` is capped.** Extinction is
`cloudDensity x profile^0.6` per km (`slabDensityScale = 1` for layer 1, so the sharpen exponent is
0.6) and nothing else. In a 1.5 km deck at a 1 km profile depth no sample reaches profile 1. With
the UI capping `cloudDensity` at 4 (the conf is at 4), the only opacity lever left was
`nvdfProfileDepthKm`. Lowering it makes the deck denser (a half-height column: 65% opaque at zenith
at depth 1.0, 74% at 0.3, 87% at 0.15) but makes the strands opaque at the same time (a 300 m
strand: 32% at depth 1.0, 55% at 0.3, 70% at 0.15). That is the bind: the strands are invisible
ghosts at a deep profile and solid smoke at a shallow one, and the deep profile also leaves the thin
parts of the deck translucent.

## The change

1. `nubis3ShapeVarietyWavelengthKm` (new, default 2.4 km, `padRetired5` CB slot). The mid tap's
   frequency is now `1 / (6 x wavelength)` instead of `0.193 x detailFreq`. At the default detail
   scale that is the old frequency to within 0.4%, so the default look is unchanged. At the live
   conf the lobes go from 0.86 km back to 2.4 km, which puts the 1.5 km amplitude in the bulging
   regime (pk-pk ~25% horizontal / ~50% vertical worst case, ~12% / ~24% at type 1). This is the
   shape-variety half of `4aaeb6ec6` on `backup/cloud-audit-rebuild-20260907`, re-derived; the
   other two-thirds of that commit (composite outline fix, type-gating the wisp cut) are NOT taken
   (the wisp-cut gate changes the default look; the outline bug is a separate issue that IS present
   in this tree, `d5a1e4c02` is an ancestor).
2. Amplitude guard, CPU-side in `getAtmosphereArgs`: effective `nubis3ShapeVarietyKm` is capped at
   0.65 x the wavelength. Cap is 1.56 km at the default wavelength, so neither the 1.11 default nor
   the conf's 1.5 is touched; it only bites if the wavelength is dragged down, and then it shallows
   the lobes instead of tearing them. CPU-side so the shader's conservative step bound
   (`maxOutwardKm`) and the interior-texture mid mix see the same number.
3. `Density` slider max 4 -> 8 in `rtx_atmosphere_ui.cpp`. Default unchanged. This is the lever
   for sky coverage now. Tooltip warns about far-step coarseness above ~6.
4. `Lobe Wavelength` slider (0.5..6 km) next to `Shape Variety` in the Shape tree.

Files: `src/dxvk/shaders/rtx/pass/atmosphere/cloud_nubis3_common.slangh`,
`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`, `src/dxvk/rtx_render/rtx_atmosphere.h`,
`src/dxvk/rtx_render/rtx_atmosphere.cpp`, `src/dxvk/rtx_render/rtx_atmosphere_ui.cpp`.

## What to use for sky coverage now

* `cloudDensity` (up to 8 in the UI) for how opaque the deck is. Keep `nvdfProfileDepthKm` where
  it looks right (1 km in the conf) and raise density instead of lowering depth.
* `cloudCoverageMean` for how much of the sky has bodies at all. NOTE (pre-existing, not changed):
  the NVDF nominal auto-tracks coverage in 0.25 steps and the live level-set offset is only
  `nvdfCoverageOffsetKm 0.2` km per unit, so between rebake steps the slider moves the bodies by
  at most +-25 m and then jumps at 0.625 / 0.875. This is likely why the coverage slider did not
  feel like a coverage control. Raising `nvdfCoverageOffsetKm` (conf-only) makes it continuous but
  also shifts the current bodies (-22 m -> -110 m at 1.0 with coverage 0.39 / nominal 0.5).
* `nvdfProfileDepthKm` goes back to being edge softness / lighting (it feeds the MS and ambient
  terms) and can now be lowered without producing smoke, because there are no strands to expose.

## Look risk

At the live conf the base relief changes from a 0.86 km torn web to 2.4 km smooth lobes. The smoke
and that fine relief were the same feature, so this cannot be avoided; if the ghost texture at a
deep profile is missed, lower `Lobe Wavelength` toward ~1.5 km (the cap then trims amplitude to
~1 km) rather than raising `Detail Scale`. Nothing changes at the default config.

`nubis3LobeFineKm 0.83` in the conf is not an option at this commit (it was added later on the
backup branch) and is ignored here.

# Painted shading fix: split the lighting profile from the density ramp (2026-09-08)

Baseline: the smoke fix above, still uncommitted on `565c09ef3`. Six files, one new option, bit-identical
at its default. Not built, not deployed, not committed. The parent's hypothesis (profile does double duty)
checked out; nothing else in the evaluator was found to dominate.

## The mechanism, verified

`profile = saturate(-sdfLive / nvdfProfileDepthKm)` is computed once in `sampleCloudDensityNubis3` and has
two unrelated customers:

* density: `valueErosion(profile, erosion)` then the `pow(., 0.6)` sharpen; the erosion frequency blends
  `lerp(noise.r, noise.g, profileProv)` / `pow(profileProv, 0.25)`; the wisp cut's `edgeBand = 1 - profile`.
* lighting: `profileOut` -> `dim_profile` in `evalNubisCubedSampleCore`, where
  `M = dim_profile * exp(-sigma_ms * D_sun) * verticalLight` (the body lobe B) and
  `ambient_shape = pow(1 - dim_profile, 0.5)` scales BOTH ambient terms (topAmbient and domeFill).

Numbers at the live state (deck 1.5 km, `cloudDensity` 6.7, sharpen exponent 0.6; erosion ignored, since at
type 1 the 0.3-scaled billowy composite x 0.58 carves at most the outer 17% of the ramp). Contribution-weighted
along a ray entering a face at normal incidence, weight = sigma x T, exactly the factor each sample's colour
lands in the pixel with. Chord transmittance is a chord entering and leaving through the ramp:

| | ramp OD | light from profile >= 0.99 | ambient_shape 10% / mean / 90% | M 10% / mean / 90% | 300 m chord T | 600 m chord T |
|---|---|---|---|---|---|---|
| depth 0.10 (current) | 0.42 | 66% | 0.00 / 0.21 / 0.76 | 0.42 / 0.82 / 0.98 | 22% | 3% |
| depth 0.50 (current) | 2.09 | 13% | 0.00 / 0.62 / 0.92 | 0.15 / 0.50 / 0.89 | 54% | 16% |
| density 0.10 / lighting 0.50 (new) | 0.42 | 5% | 0.49 / 0.77 / 0.96 | 0.08 / 0.33 / 0.70 | 22% | 3% |
| density 0.10 / lighting 1.00 (new) | 0.42 | 0.2% | 0.79 / 0.90 / 0.98 | 0.04 / 0.17 / 0.35 | 22% | 3% |

Volume of a 1.5 x 2.5 x 2.5 km body sitting at profile 1: 73% at depth 0.10, 12% at 0.50.

So at 0.10 two-thirds of what the eye receives from any face comes from samples where the ambient is exactly
zero and M is 1 x (0.92..1.00). What is left to vary is `verticalLight` (a pure top-to-base gradient,
1.00 -> 0.13 over the deck at `cloudUndersideLightSigma` 0.2, identical at both depths) and micro-AO (confined
to the outer `cloudMsSdfDepth` 128 m, +-27% at strength 0.6, never reaching its clamp). A bright 100 m
ambient rim around a flat body-lobe fill with one smooth vertical gradient is a poster. At 0.50 the whole
visible shell sits inside the ramp, so M and the ambient vary continuously, but that same ramp is 2.1 optical
depths deep and lets half the light through a 300 m chord: the ghost.

Candidates checked and ruled out as the dominant cause:

* sigma_ms remap (`cloudMsSdfDepth` 128 m, shallow 0.25 / deep 0.05): `exp(-sigma_ms * D_sun)` spans
  0.92..1.00 on a lit face for any D_sun a 1.5 km deck can produce. Not a shading term at these values;
  neither the flatness nor a fix for it.
* `verticalLight`: strong but strictly vertical and present at both depths; not what changed in the A/B.
* micro-AO and powder: skin effects (128 m SDF band / density-gated), the same at both depths past the skin.
* `T_primary` = exp(-6.7 x D_sun): dies within ~150 m of density path. Rim only; this is the "sunlit rim"
  the user sees at 0.10 and this change does not touch it.

## The change

`nvdfLightingProfileDepthKm` (new; default 0 = follow `nvdfProfileDepthKm`; `padRetired8` CB slot, uint ->
float, layout unchanged). The CPU resolves 0 to `args.nvdfProfileDepthKm` after its clamp, the same float, so
`profileOut = saturate(-sdfLive / max(args.nvdfLightingProfileDepthKm, 1e-3))` is bit-identical to the old
`profileOut = profile` at the default. Only `profileOut` moved. Every density-side consumer stays on the
density ramp, each on its merits:

* the erosion frequency blends stay on `profileProv` (density depth): they pick WHICH octave carves at WHICH
  depth of the ramp `valueErosion` carves. Keyed on a deeper ramp, the 100 m carve band would only ever see
  profileProv <= 0.2, i.e. the pure low-frequency channels, and the "fine filaments deeper in" would sit past
  the surface where erosion has nothing left to carve.
* `edgeBand`, `valueErosion`, the sharpen: density, for the same reason (they decide where material is).
* The OD / bake overloads discard `profileOut`, so D_sun / D_ambient are untouched: silhouettes, shadows and
  coverage are identical at any setting.
* Both march call sites (layer 1 and the echo deck) receive the lighting profile through `nubisProfile`
  automatically; the evaluator itself is unchanged.

UI: `Lighting Depth` (0..3 km) directly under `Profile Depth` in the Shape tree.

Files: `src/dxvk/shaders/rtx/pass/atmosphere/cloud_nubis3_common.slangh` (step 7 + header doc),
`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh` (evaluator doc only),
`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h`, `src/dxvk/rtx_render/rtx_atmosphere.h`,
`src/dxvk/rtx_render/rtx_atmosphere.cpp`, `src/dxvk/rtx_render/rtx_atmosphere_ui.cpp`.

## What to set

`nvdfProfileDepthKm 0.10` (solidity, as now) plus `nvdfLightingProfileDepthKm 0.50` is the table's third row:
the 0.10 skin and silhouette with the 0.50 shading spread (wider, in fact, because the whole visible shell now
sits inside the lighting ramp). 0.30 is a middle ground; 1.00 heads toward an ambient-dominated soft interior
(M mean 0.17).

## Look risk

* Tonal balance moves toward the deep look, which is the point but is worth naming: on lit faces the body lobe
  drops (mean M 0.82 -> 0.33) and on every face the ambient comes back (mean 0.21 -> 0.77), so shaded sides get
  brighter and lit faces glow less. `cloudAmbientShadowStrength` (already 1) and `cloudMsLobeWeight` are the
  existing levers if the lit/shaded contrast needs rebalancing; this change touches neither.
* Micro-AO relief still stops at `cloudMsSdfDepth` (128 m, conf-only). The new 128..500 m band gets the
  profile gradient but no relief; raise `cloudMsSdfDepth` if relief is wanted deeper in.
* Bake keys: like every other lighting-only field (`cloudMicroAoStrength`, `cloudMsSdfDepth`, ...), the new CB
  field is not zeroed in `normalizeForVoxelGridKey`, so dragging the slider re-bakes the amortized D_sun /
  D_ambient grids. Pre-existing behaviour for its class; not changed here.
* Nothing changes at the default (0) for any existing conf.

# Three render bugs: silhouette outline, disc leak, banding (2026-09-08)

Baseline: the two fixes above, still uncommitted on `565c09ef3`. Two shader files, no new option, no CB
change, no default touched. Not built, not deployed, not committed. Reproduction state taken from the
parent's list (Density 6.70, Altitude 1300 m, Depth 1500 m, Shape Variety 2.00 -> 1.365 km effective
after the 0.65 x 2.1 km cap, Spacing 0.10 km, Max 96, Smoothing 0.85, Clamp 1.25, Render Scale 1.00) and
from the live conf (`Projects/Games/Fallout New Vegas/rtx.conf`, saved 06:05 today): everything not
listed there is at its default -- `nubis3AdaptiveStepKm` 0.025, `nvdfStepScale` 0.95, `sunIlluminance` 15,
`mieAnisotropy` 0.97, `cloudAerialHazePerKm` 0.05; the conf sets `cloudAerialFadePerKm` 0.05,
`cloudUndersideLightSigma` 1.5 and `sunElevation` 0.36 (degrees).

## 1. Bright outline along geometry silhouettes -- confirmed, fixed

**Root cause** (`composite.comp.slang`, `sampleCloudRenderRT`). The depth-aware upsample from `d5a1e4c02`
gated on `cloudRenderFullDimX != 0`, and `ensureCloudRenderRT` assigns `m_cloudRenderFullExtent`
unconditionally, so the four-tap path ran at native resolution too -- the "at scale 1 this never runs"
comment was wrong. At scale 1 the bilinear weights are {1, 0, 0, 0} so that alone would have been
harmless, except that the tap validation could never pass for sky: the cloud pass writes sky as
`kNoSurfaceKm = 1e9` (`cloud_march_common.slangh`), the composite calls sky `kCloudSkyDepthKm = 60000`,
and `|1e9 - 6e4| > 0.1 x 1e9`. Every sky pixel therefore fell through to the nearest-depth fallback,
and next to geometry the tap nearest in depth to 60000 is the GEOMETRY texel (`|3 - 60000| <
|1e9 - 60000|`) -- one of the +x / +y / +xy taps of the footprint -- whose march was clamped at that
surface and holds no cloud. Those sky pixels composited with opacity 0: raw cloudless sky, brighter than
a dark overcast deck, one texel wide on the sky side of every silhouette, i.e. the sky pixels directly
above a ridgeline. Width 1 texel at render resolution, 1-2 px after DLSS, more where the edge is
diagonal. The audit branch found exactly this independently (`4aaeb6ec6`, "OUTLINE"); its composite hunk
is what is ported here, with a divide guard kept on the zero-dims case.

**Fix.** Gate the four-tap path on an actual resolution mismatch (`rtDim != cloudRenderFullDim`) so
native resolution takes the exact `Load`, and clamp each tap's stored depth to `kCloudSkyDepthKm`
(hoisted to file scope) so sky-vs-sky validates below native too.

**Look.** Removes the rim. Nothing else changes at scale 1 -- the `Load` is what every composite before
`d5a1e4c02` did -- and the 3x3 neighbourhood statistics go from ~90 texture loads per pixel to ~20.

## 2. Sun / moon disc through dense, dark cloud -- one mechanism confirmed and fixed; a second is by design

There is no sun disc in this renderer: the "disc" is the Mie aureole the sky-view LUT bakes at
`mieAnisotropy` 0.97 (`hgPhase` peaks at 174x isotropic), a 2-3 degree blob of order 20-50 radiance
units at 5-20 degrees of elevation (sunIlluminance 15 x phase x slant Mie depth x transmittance), redder
and brighter lower down. The moon IS a disc (`evalMoonDisk`, replacing the sky where it covers). Both
reach the composite inside `radianceOutput` and are multiplied by `(1 - smoothedCloud.a)` there --
the premultiplied convention is honoured by every consumer; nothing is composited out of order.

**Mechanism A (fixed): the transmittance-limit exit.** Every march loop breaks at
`viewTransmittance < 0.01` and composites what it has, so up to 1% (0.5-1% in practice, the test runs
before the next step) of the background comes through a cloud the march already found opaque. For sky
(~1 unit) that is invisible. For the aureole it is 0.2-0.5 units on a 0.05 deck -- the disc "on top of"
the cloud; for the moon the same ratio against the night deck. Physically the deck passes exp(-10) at
sigma 6.7 over 1.5 km. Fix (`cloud_march_common.slangh`, `cloudShadeContextBeginRay` +
`CloudShadeContext::exitTransmittance`): the exit threshold is now ray-constant, 0.01 everywhere except
within ~8 degrees of the sun or moon, where it is 1e-4, with a log-space smoothstep to 0.01 by
~16 degrees (past 10 degrees the aureole is under 1 unit and 1% is invisible again). 1e-4 sits below the
2^-11 quantum the f16 opacity history has near 1, so the composite treats such pixels as fully opaque.
All three loops (lattice, adaptive, echo deck) read the same value; the secondary LUT bake inherits it.

**Mechanism B (not fixed, by design): the horizon fade.** The conf snapshot has the sun at 0.36 degrees.
A ray at that elevation meets the 1.3 km deck base at ~95 km and leaves the 2.8 km top at ~153 km
(planet curvature included), where `cloudAerialFadePerKm` 0.05 has scaled per-step extinction to
`exp(-0.05 x 95) = 0.9%` and the 96-step budget covers ~38 km of the 58 km span. T after the whole march
is >= 0.43 no matter how dense the deck: the far deck is translucent on purpose (the horizon-wall fade),
and the aureole shows through it. Below ~1.5 degrees of sun elevation this is what is on screen and fix A
does not touch it. The honest fix is to haze far cloud toward the sky colour instead of fading its
alpha -- an aerial-perspective rewrite and a horizon look change, out of scope here. Above ~3 degrees the
fade no longer dominates and A is the mechanism.

**Look.** No change outside 16 degrees of the sun / moon. Inside, the <= 1% background leak is gone --
on dark cloud the sun blob disappears, which is the bug; on bright cloud nothing was visible either way.

**Cost.** One smoothstep per ray; a handful of extra steps (~12 at ~0.4 optical depths per interior
step) only on the rays that look at the sun. Marching every opaque ray to 1e-4 instead would have been
~15-20% of the pass, which is why it is directional.

## 3. Banding -- a real march defect found and fixed; not visually confirmed as the banding seen

**What was verified NOT to be it.** The adaptive march IS active (`nubis3AdaptiveStepKm` 0.025 > 0,
spacing 0.10 > 0): 33 m steps at the deck base, 100 m at 12 km, 0.4 km cap. Jitter is per-RAY (one
128^2 x 64 blue-noise value per pixel per frame reused for every step of that ray), animated everywhere
(`nubis3JitterAnimateKm` default 1e9). Per-step optical depth at 6.7 is 0.2-0.7 in the interior, but a
jittered stratified lattice dithers that error and the EMA averages it; no undithered per-step
quantisation exists in the loop. The history RT is R16G16B16A16_SFLOAT (no EMA stall at 0.85), the
clamp box is built from the same signal it clips, and the composite upsample is an exact `Load` at
scale 1 (after fix 1). So neither the EMA nor the composite manufactures contours.

**What is wrong in the march (fixed).** The adaptive loop measures the SDF empty radius `r` at the
jittered sample `tSample = t + hash x step`, but applied the skip from the unjittered cursor `t + step`,
which sits `(1 - hash) x step` beyond the point `r` was measured from. The next sample could land
`step x (1 - hash) - 0.05 r` PAST the guaranteed-empty boundary -- for a km-scale approach skip,
hundreds of metres to a kilometre inside the body. Worked example, surface at 2.01 km, hash 0.5: the
second skip lands the sample 250 m inside with a 1.1 km Beer segment; hash 0.2: 240 m inside with a
1.5 km segment. At sigma 6.7 that one segment is 7-13 optical depths, so a single interior-shaded
sample decided the pixel and the lit skin was never sampled; the overshoot depth depends on the length
of the last empty stretch, so the error follows the sphere-trace's own cell structure -- contours, the
"onion-shell banding" `nvdfStepScale`'s tooltip already warns about; and with the jitter animated the
per-frame depth lottery became the speckle `565c09ef3` dithered ("the density-skip flipping against a
step much larger than the detail it samples" is this seen from the other side). The lattice path never
had it: its skip advances the sample INDEX, so it is sample-anchored by construction. The hybrid loop
(2026-07-16) decoupled cursor and sample and inherited the flaw. Fix: carry the far end of the proven
interval (`tEmptyUntilKm = tSample + r`) and skip `(tEmptyUntilKm - t) x nvdfStepScale` from the cursor.
The cursor then advances 95% of its remaining distance to the proven boundary each iteration -- the
textbook sphere trace, independent of the jitter -- still 2-3 iterations from km range to the adaptive
floor, so empty air costs the same; the difference is that the trace now stops AT the bounding surface
and the first in-cloud samples come from the 30-100 m adaptive stepping through the displacement band.

**Bisection, in-game, to tie the screen artefact to a mechanism.** `nvdfStepScale` 0 (uniform stepping,
slow): banding gone => it was the skip overshoot, this fix covers it. `cloudUndersideLightSigma` 1.5 ->
0.2: gone => it is the D_ambient grid (below). `cloudDensity` 6.7 -> 3: gone => a sigma-amplified
discretisation, one of the three here. `cloudHistoryWeight` 0: still there => not the EMA.

**Other sigma-amplified, undithered discretisations that can read as banding (unchanged).**
* D_ambient x `cloudUndersideLightSigma` 1.5 (the option's documented range is 0..0.5):
  `verticalLight = exp(-1.5 x 6.7 x D_ambient)` e-folds every 100 m of overlying water, which is one
  vertical voxel of the 256 x 32 x 256 grid over a 3.2 km deck (47 m over 1.5 km). Trilinear D through an
  exponential gives a slope break at every voxel plane of ~e per voxel: horizontal contour steps through
  the volume, on M and topAmbient. Only when `darkenStrength` > 0, i.e. sun above ~5 degrees
  (`smoothstep(0, 0.35, sun.y)`); off at the conf's 0.36 degrees.
* D_sun near section: 8 FIXED taps over the first 3 km (375 m spacing, deliberately de-jittered
  2026-07-30). `T_primary = exp(-6.7 x D_sun)` turns the lattice's aliasing against 300-500 m density
  structure (amplitude ~0.19 km.density) into 3.5x swings at that period, versus 2.1x at sigma 4 and 1.4x
  at 1.8 -- shells perpendicular to the sun on lit faces. The bake steps are amortised, so this is not
  a per-frame cost to fix, but it is a bake-step change and was left alone pending the bisection.
* The 96-sample cap: `69.3 x (sqrt(t_exit) - sqrt(t_entry))` steps for a fully dense span, so the budget
  runs out below ~9 degrees of elevation for a 1.5 km deck and ~25 degrees for the conf's 3.2 km one;
  partially cloudy rays are the ones that hit it, and their far cloud is simply dropped.

**Look risk of the fix.** Faces approached across long empty stretches -- undersides seen from below,
tower flanks across gaps -- get their skin lighting back instead of being shaded by a sample hundreds of
metres inside: brighter and more structured than the current look on exactly those faces. Nothing is
retuned; this is the artefact's removal, and the "painted / flat" read the lighting-profile split
above was chasing may recede with it. Cost: rays now sample the +-0.68 km displacement band at
30-100 m (gated to one or two texture taps per empty sample) rather than jumping past it, so a few more
samples per body approach; if far horizon cloud thins, that is the 96 cap and Max Cloud Samples is the
lever.

Files: `src/dxvk/shaders/rtx/pass/composite/composite.comp.slang` (bug 1),
`src/dxvk/shaders/rtx/pass/atmosphere/cloud_march_common.slangh` (bugs 2 and 3).

# Regression: the disc-leak fix drew its own disc (2026-09-08)

Baseline: the three-render-bugs batch above, still uncommitted on `565c09ef3`. One shader file
(`cloud_march_common.slangh`), no new option, no CB change, no default touched. Not built, not
deployed, not committed. Fixes 1 (silhouette outline) and 3 (sphere-trace overshoot) are untouched
and verified still present.

## The parent's diagnosis held, and here is the arithmetic that confirms it

The claim to check was that the angular gating was itself what the user saw: a large smooth
near-black disc centred on the sun with a small warm point at its centre, over a dawn sky.

The one thing that could have falsified it is the horizon fade. Section 2 above says a sun at the
conf's 0.36 degrees leaves `T >= 0.43` no matter how dense the deck, and a threshold change cannot
matter on a ray that never approaches any threshold. So: does the exit fire anywhere inside the
16-degree cone at all?

`viewTransmittance *= mix(1, stepTransmittance, aerialT_fade)`, so each step multiplies by at least
`1 - f` with `f = exp(-cloudAerialFadePerKm * t)`, whatever the density -- the fade is a floor on
per-step transmittance, not a scaling of sigma. Reaching 0.01 in the 96-step budget needs
`96 * f > ln(100)`, i.e. `f > 0.048`, i.e. **deck entry nearer than ~61 km**. Curved-earth entry to a
1.3 km base is `sqrt((R sin(elev))^2 + 2Rh) - R sin(elev)`: 95 km at 0.36 degrees (matching section
2's figure), 61 km at **~0.95 degrees**, 15 km at 5 degrees, 7.5 km at 10 degrees.

So inside a +-16 degree cone around a sun at 0.36 degrees, every ray above ~1 degree of elevation
DOES hit the exit and was retuned from 0.01 to 1e-4; every ray below it was untouched. That is a
disc reaching ~16 degrees up from the horizon, dark because the leak that lit it is gone and the
backlit deck under it is ~0.05, with the sun's own direction (0.36 degrees, in the fade band, never
gated) still glowing at its centre. Exactly the screenshot. Diagnosis confirmed, mechanism and
geometry both.

## Why plain zeroing was rejected too

The suggested replacement -- set `viewTransmittance = 0` when a loop breaks -- is safe for every
consumer (all of them were checked; see below) but is the *same class of artefact*, only rotated:

Parameterise a family of rays by column optical depth `tau`. Reported T today is `exp(-tau)` while
`tau < ~4.6`, then flat at ~0.005-0.01 (jitter-dithered) for every deeper ray -- continuous, because
a ray that crosses the threshold on its LAST step never reaches the `break` at all (the test is at
the top of the next iteration). Zeroing turns that plateau into a cliff: neighbours at `tau = 4.6`
and `tau = 5.0` report 0.01 and 0, a step of 1% of the background at the `T = 0.01` contour. Against
the sky that is 0.01 units and invisible; against the aureole it is 0.2-0.5 units on a 0.05 deck, a
5-10x edge -- the identical failure mode, on a contour that near a low sun is a near-horizontal line
at the elevation where the fade stops dominating. A hard cut through the glow instead of a circle
around it.

The general rule this batch has now learned twice: what the eye sees is `T x background`, so any
rule that removes the leak on one side of a boundary and keeps it on the other DRAWS that boundary.
The fix has to be continuous in T, not gated on anything.

## The change

Report transmittance on the scale whose zero is the march's own exit threshold. One place,
`marchCloudLayers`, after both slab crossings and the echo deck:

```
  T' = max(0, (T - e) / (1 - e)),   C' = C / (1 - e),   e = kCloudExitTransmittance = 0.01
```

* Every ray that broke on the transmittance limit was below `e` when it broke, by the test's own
  definition, so it reports exactly 0. The leak is gone completely, on every such ray, at every
  view angle, for one multiply-add per ray.
* T' is continuous and monotone in T and exactly 1 at T = 1, so the threshold draws no contour
  anywhere -- including at the `T = e` crossing, which is now the point where T' reaches 0 smoothly
  rather than a cliff.
* The radiance takes the same factor. That is not cosmetic: the energy the shift removes from the
  background is `(T - T') * Lbar` with `Lbar = C / (1 - T)` the mean cloud radiance the march
  already measured, and `T - T' = e(1-T)/(1-e)` collapses that to exactly `C * e/(1-e)`, i.e.
  `C' = C/(1-e)`. So the pair stays a valid premultiplied (radiance, transmittance) description of
  a cloud -- "the same cloud, 1% more opaque" -- and the composite is exact in the thin limit
  instead of running 1% of the background dark.

Also in this change: `exitTransmittance` is gone from `CloudShadeContext` entirely (no dead field);
`cloudShadeContextBeginRay` is byte-identical to its pre-regression form; the three loop tests and
the echo-deck gate in `marchCloudLayers` all read the new file-scope `kCloudExitTransmittance`
(same 0.01, no value change).

## Consumers checked (all fine with T' = 0, i.e. opacity exactly 1)

* `composite.comp.slang`: `loadCurrentCloudOpacity` returns `1 - a` -> 1.0; the premultiplied over
  becomes `radiance * 0 + rgb`, which is the correct "fully opaque". Nothing divides by alpha.
* `cloudDepthVisibility(.., cloudAlpha = 1)`: `tau = -log(max(0, 1e-6)) = 13.8` and
  `g = tau*1/max(1 - 0, 1e-6) = 13.8` -- both existing clamps carry it, no NaN, no divide by zero,
  and it is the continuous extension of g's 6.9 at alpha 0.999. Alpha-blend path only.
* History EMA: f16 opacity, 1.0 exact; the alpha gate is `> 0.001`; `clipCloudHistoryToAABB` is
  linear. Note f16 quantises opacity to 2^-11 near 1 anyway, so T' below ~2.4e-4 was already
  composited as fully opaque -- the shift only makes that exact.
* Secondary dome LUT (`atmosphere_sky.slangh`): same `1 - lut.a` convention, bilinear-filtered so
  hard zeros smooth; `pow(cloudViewT, starCloudExtinctionPower)` at T = 0 gives 0, which is the
  right answer (stars fully hidden by opaque cloud) and was already reachable via exp underflow.
* `sampleCloudRenderRT`'s four-tap upsample: weighted average of taps, no alpha normalisation.
* The echo deck is marched AFTER layer 1 into the same accumulators, and the rescale runs after
  both, so nothing composites a rescaled value against an unrescaled one. The `spanPiece` loop is
  likewise inside the rescale.

## Is the leak actually fixed, or only made uniform

Actually fixed, for the mechanism it addresses: any ray whose marched transmittance reaches 0.01
now contributes exactly zero background, and there is no view-angle, cloud-shape or screen-position
term in that statement. What is NOT fixed, and is out of scope by the parent's instruction, is
mechanism B: below ~1 degree of sun elevation the deck entry is past ~61 km, per-step extinction is
floored by `cloudAerialFadePerKm` and T never gets near 0.01, so the aureole still shows through far
horizon cloud. At the conf's 0.36 degrees that band is what fills the bottom of the frame, and it is
unchanged -- deliberately.

## Look risk

* Every cloud pixel is 1/0.99 = 1.0101x more opaque and 1.0101x brighter in its own radiance. That
  is ~1% and below both the JND and this pass's temporal noise floor; it is uniform, so it cannot
  draw an edge. Nothing was retuned to compensate and nothing should be.
* The glow the user liked near the sun WAS partly this leak on the opaque pixels, and it does not
  come back. What remains near the sun is the horizon-fade band (mechanism B) plus whatever thin
  cloud genuinely passes T > 0.01 -- both cloud-shaped, neither circular.
* Residual discontinuity: none from this change. The `T = e` contour is now the point where T'
  reaches zero with matching value, and the leak on either side of it differs by ~0.

## Note for whoever reads this next

This worktree was being edited concurrently while this fix was written (2026-09-08 07:09,
`rtx_atmosphere.cpp/.h`, `atmosphere_args.h`, `composite.comp.slang` -- a `cloudAerialInScatterStrength`
aerial-perspective change adding LUT in-scatter to `smoothedCloud.rgb` weighted by
`smoothedCloud.a`). That work does not touch `cloud_march_common.slangh` and does not interact with
this one beyond seeing a 1% larger alpha. Worth knowing that it will put warm air-light back in
front of opaque cloud near a low sun -- through the aerial-perspective volume, which is the
physically right mechanism, not through an early-out.

# Aerial perspective on the clouds: never wired, and the stand-in has the wrong sign (2026-09-08)

Baseline: the items above, still uncommitted on `565c09ef3`. Five files, one new option, one
CB pad consumed, no layout change, no existing default touched. Not built, not deployed, not
committed. `cloud_march_common.slangh` deliberately NOT edited (another agent holds it); the one
shader edit is a self-contained block inside `applyCloudComposite` in `composite.comp.slang`.

## Diagnosis: not wired at all, and the substitute fades toward black

**1. The real LUT never sees the clouds, by two independent mechanisms.**
`applyAerialPerspective` (`composite.comp.slang`) returns early on `primaryMiss` -- which is every
sky pixel, and sky pixels are exactly where the clouds are. Skipping the sky is correct *for the
sky* (the sky-view LUT already integrates the whole column), but the cloud is a separate signal
composited over that sky. And the call sits three lines BEFORE
`applyCloudComposite`, so even on a geometry hit the LUT is applied to the surface and the cloud is
then composited on top of the result, untouched. Nothing in the cloud march samples the volume
either (`grep -i aerial` over `pass/atmosphere/` hits only the LUT's own bake and the two per-km
scalars). The audit's future-work item is confirmed: this was never done.

**2. `cloudAerialHazePerKm` dims cloud radiance toward BLACK, not toward the sky.**
`cloud_march_common.slangh:1078` (layer 1) and `:1695` (echo deck):

    accumColor += stepRadiance * viewTransmittance * aerialT_haze;   // aerialT_haze = exp(-k*t)

A bare multiplicative extinction with no in-scatter term anywhere. Its own doc string claims it
"dims distant cloud samples toward atmospheric color" -- there is no atmospheric colour in the
expression. This is the wrong half of Beer-Lambert applied on its own, and it is the mechanism the
user is seeing.

Arithmetic at the live conf (`cloudAerialHazePerKm` at its 0.05 default -- the conf does not set it;
deck 1300..4500 m; flat-earth slant to the deck base ~ 1.3 km / sin(elevation)):

| view elevation | distance to deck base | `exp(-0.05 d)` = fraction of cloud radiance kept |
|---|---|---|
| 30 deg | 2.6 km | 88% |
| 10 deg | 7.5 km | 69% |
| 5 deg | 14.9 km | 47% |
| 2 deg | 37 km | 16% |
| 0.36 deg (the conf's sun elevation) | 95 km | 0.9% |

So the deck has lost half its light by 5 degrees above the horizon and 84% by 2 degrees, with
nothing added back. Aerial perspective does the opposite: distant cloud should LOSE CONTRAST against
the horizon sky, not lose luminance toward black. Net at 2 degrees for a dense deck (sigma 6.7,
1.5 km chord, 20 steps): alpha ~0.72 after the fade below, radiance ~0.11 x the cloud's own -- the
cloud region ends up roughly half as bright as the clear sky beside it. A dark band toward the
horizon where haze should be brightening it. That is "very broken".

**3. `cloudAerialFadePerKm` fades cloud ALPHA, confirming the parent's suspicion.**
Same two sites: `viewTransmittance *= mix(1, stepTransmittance, aerialT_fade)`, i.e. per-step
extinction scaled by `exp(-k*t)`. Distant cloud goes TRANSPARENT rather than hazy. At the conf's
0.05 that is under 1% of extinction by 95 km, which is the horizon-wall behaviour item 2 of the
render-bugs section above already documented as "by design, out of scope". It is left alone here for
the same reason: it is a coverage/silhouette control, changing it is a retune, and the honest
replacement for it is the LUT's own transmittance, which cannot be swapped in without rescaling a
config the user is happy with.

**4. The user has the whole feature switched off.** `rtx.atmosphere.aerialPerspective = False` in
`Projects/Games/Fallout New Vegas/rtx.conf` (default is True). With it false `getAtmosphereArgs`
sets `aerialPerspectiveLutSize = 0`, `dispatchAerialPerspectiveLut` never runs, and
`applyAerialPerspective` early-outs on every pixel. Aerial perspective currently affects NOTHING on
screen -- geometry included -- so "it does not affect the clouds" is at this moment true of
everything, and no shader change is visible until that option goes back on.

**5. Why it is probably off: the unit calibration, though not in the form suspected.** Both scales
resolve from the same `resolveUnitsPerMeter()` since 2026-09-06, so
`aerialPerspectiveWorldUnitsPerKm` and cloud `worldUnitsPerKm` are BOTH 10,000 units/km here and the
mismatch the parent asked about does not exist at this config (it appears only if
`cloudWorldCompression != 1`). But the conf sets neither `unitsPerMeter` nor `cloudWorldCompression`;
it sets `rtx.sceneScale = 0.1`, and `resolveUnitsPerMeter` falls back to `100 x sceneScale` =
10 units/m. Fallout: New Vegas is 70.4 units/m -- `cloudWorldUnitsPerKm`'s own comment says so, and
calls the 10,000 figure "the true 70,400 divided by a deliberate ~7x compression". So the
cloudscape's artistic 7x compression is silently leaking into the haze scale, and aerial perspective
believes every distance in the world is 7.04x larger than it is. Geometry haze is therefore ~7x too
strong -- a very plausible reason to have turned it off.

The config fix (NOT applied; rtx.conf is off-limits and this is a tuning decision):

    rtx.atmosphere.unitsPerMeter = 70.4
    rtx.atmosphere.cloudWorldCompression = 7.04

`cloudWorldUnitsPerKm` comes out at exactly 10000.0f either way (checked in float), so **the cloud
look is bit-identical**; only `aerialPerspectiveWorldUnitsPerKm` moves, 10,000 -> 70,400, which is
the correction. Two knock-ons: `aerialPerspectiveSceneShadowRangeMeters` (the conf raised it to 4600,
which at the wrong scale is 653 real metres) wants to come back to ~650, and the 32 km LUT range
stops clamping at cloud distance -- it would cover out to 225 cloud-km instead of 32.

**Refuted / verified leads.** `meanCloudDepthKm` IS genuinely written (`cloud_render.comp.slang:330`,
`cloudDepthOutput.y`) and IS read by the composite (`meanKm`), but only for the reprojection motion
vector and the alpha-blend depth gate -- nothing consumed it for haze. The premultiplied convention
is handled correctly everywhere (`rgb` premultiplied radiance, `a` = 1 - view transmittance,
composited `dst*(1-a) + rgb`); no ordering fault there. The `aerialPerspectiveWorldUnitsPerKm` doc
comment in `atmosphere_args.h` still says "cloud and sky units still use rtx.sceneScale", stale since
2026-09-06 -- left as-is, it is only a comment.

## The change

`cloudAerialInScatterStrength` (new; default 1.0; `padAerial8` slot, float -> float, layout
unchanged). In `applyCloudComposite`, after the history write and before both composites:

    smoothedCloud.rgb += max(lut.rgb, 0) * (nearFade * strength * smoothedCloud.a)

sampled from `AtmosphereAerialPerspectiveLut` at `meanCloudDepthKm` converted to world units
(`meanKm * args.worldUnitsPerKm`), through the same `aerialPerspectiveScreenUvToUvw` /
`aerialPerspectiveNearFade` / `sampleAerialPerspectiveLut` helpers the geometry path uses. Going via
world units rather than km is what keeps cloud and haze in agreement under a non-1
`cloudWorldCompression`.

Only the **in-scatter** half is taken. The LUT's transmittance is deliberately NOT applied:
`exp(-cloudAerialHazePerKm * t)` already occupies that role in the march, and applying both would dim
distant cloud twice. So the fix converts "fades to black" into "fades to the colour of the air in
front of it" -- exactly what the option's own documentation always claimed -- without rescaling
anything the user tuned.

Weighted by `smoothedCloud.a` because the rest of the pixel is sky, which already carries its own
whole column; adding unweighted air light would double it there. Placed after the history write for
the same reason the occlusion gate is: a per-frame view-distance term has no business inside a
temporal signal. Placed before both composites so the opaque and alpha-blend paths agree.

Bounded by construction, which is what makes it safe to default on: the volume integrates a strict
sub-segment of the sky's own column with a tamer forward lobe (`aerialPerspectiveMieAnisotropyMax`
0.8 vs the sky's `mieAnisotropy` 0.97), so what it can add is always less than the sky radiance the
cloud sits against. It cannot build a horizon wall brighter than the sky behind it, and it cannot
darken anything -- it is additive-only.

UI: `Cloud In-Scatter` (0..1) at the top of Atmosphere > Aerial Perspective, under Enable.

Files: `src/dxvk/shaders/rtx/pass/composite/composite.comp.slang` (the wiring -- a self-contained
block between the PSR gate and the premultiplied-over, no existing line changed; note the other
agent's outline fix lives in `sampleCloudRenderRT` in the same file, a different function),
`src/dxvk/shaders/rtx/pass/atmosphere/atmosphere_args.h` (padAerial8 -> the new field),
`src/dxvk/rtx_render/rtx_atmosphere.h`, `src/dxvk/rtx_render/rtx_atmosphere.cpp` (fill + zero it in
`normalizeForSkyLutCache`, as that block's header requires),
`src/dxvk/rtx_render/rtx_atmosphere_ui.cpp`.

## Look risk

* **Zero at the user's current config.** `aerialPerspective = False` makes `aerialPerspectiveLutSize`
  0 and the new block's first condition false, so nothing executes and the frame is bit-identical.
  To see the fix at all: `rtx.atmosphere.aerialPerspective = True`, and read finding 5 first --
  turning it on at the current 10 units/m gives geometry ~7x the haze it should have, which is very
  likely why it was turned off in the first place. Fix the calibration in the same sitting.
* **At stock defaults** (where `aerialPerspective` is already true) distant cloud gains air light it
  never had. That is the bug fix, and it is the one intentional look change here. `Cloud In-Scatter`
  0 restores the old behaviour exactly.
* The march's own haze and fade are untouched, so cloud silhouettes, coverage, alpha and the
  horizon-wall behaviour are all bit-identical. Only the colour of already-visible cloud moves, and
  only upward, and only toward the sky it is sitting against.
* Not verified in-game (no build/deploy this session). The arithmetic is checkable, the visual is not.
* Cost: one 3D texture sample per pixel that has cloud, inside a branch that is already false when
  the volume is off.

# Temporal ghosting: the history fetch loses bandwidth, not position (2026-09-08)

Baseline: everything above, still uncommitted on `565c09ef3`. One shader file changed for behaviour
(`composite.comp.slang`, one function), one comment-only edit (`debug_view.comp.slang`). No new
option, no CB change, no default touched, no retune. Not built, not deployed, not committed.

Symptom under investigation: turning the camera smears the sky and it clears over roughly a second;
a static camera is clean. Bisected in-game to `rtx.atmosphere.cloudHistoryWeight = 0`, so the cloud
EMA and its reprojection, not the march, not the upsample, not the aerial in-scatter.

## The failing link: the reconstruction filter, not the reprojection

**The reprojected coordinate is correct.** Read end to end and cleared:

* Jitter handling is consistent on BOTH sides. `cloudRotationMotionVectorPixels` projects the
  UNJITTERED pixel-centre direction (`cameraPixelCoordinateToDirection(..., jitter=false)`) through
  `cb.camera.prevWorldToProjection` -- which is `prevViewToProjection * prevWorldToView`, the
  unjittered previous matrix (`rtx_camera.cpp:931`; the jittered twin is a separate field, `:932`)
  -- and differences it against `cameraPixelCoordinateToNDC`, which is also unjittered. The
  suspected "unjittered direction vs jittered matrix" mismatch does not exist.
* The NDC -> pixel-index conversion is self-consistent (`d(pixel)/d(ndc) = (+res.x/2, -res.y/2)`,
  which is the `vec2(0.5, -0.5) * resolution` in the code), and the whole term is bit-identical to
  the engine's own `calcMotionVectorForRayMiss`, which drives the sky's DLSS motion vectors and is
  not ghosting.
* No off-by-one frame. `RtxAtmosphere::updateFrame` runs inside `updateRaytraceArgsConstantBuffer`,
  i.e. after `onInjectRtxFrameBegin`, and takes the basis from `ctx.getSceneManager().getCamera()`
  -- the same `RtCamera` whose shader constants the composite reads. `dispatchCloudScreenPass` runs
  later in the same frame (`rtx_context.cpp:696`). `onFrameAdvanceForCloudHistory` swaps the
  ping-pong once per frame in `updateFrame`, ahead of every consumer.
* The parallax term is identically zero here: `useCameraWorldOverride` is not in the conf, so the
  anchor is `camera.getPosition()`, which is permanently (0,0,0) on this engine ->
  `cloudAnchorDeltaYUpKm == 0`. Under pure rotation it would be zero anyway.
* Both new CB fields (`cloudHistoryClampGamma`, `cloudHistoryDepthTolerance`) are populated
  (`rtx_composite.cpp:519-520`), so the clip and the depth test really are running.
* Neither of today's other composite edits touches the temporal path: the outline fix is inside
  `sampleCloudRenderRT` (and at `cloudRenderResolutionScale = 1` it now takes the exact `Load`),
  and the aerial in-scatter block sits AFTER the history write.

**And the three rejectors are all blind to what is actually happening.**

* Age: every pixel writes `CompositeCloudHistoryCurr` every frame, so every in-bounds tap always
  carries the immediately-previous frame id. Under rotation the age test only ever rejects
  off-screen and never-written texels. It is not a disocclusion test.
* Depth: on sky both sides carry `kCloudSkyDepthKm`, so the test is an exact no-op precisely where
  the artefact is. (It is doing its real job on geometry, which is what it was added for.)
* The 3x3 clip: **structurally cannot see this artefact.** It bounds `|history - m1|`, and the
  defect leaves the history sitting AT `m1`. See below.

**The mechanism.** The EMA feeds on its own resampled output. With a static camera the motion vector
is exactly zero, `f == 0`, and `fetchCloudHistoryBilinear`'s four taps collapse to one texel: the
fetch is a copy, nothing is lost, the frame is clean -- which is exactly the reported boundary
condition. The moment the camera turns, every frame convolves the WHOLE accumulated history with
the bilinear kernel again, so `H = (1-w)C + w B H` gives a steady state

    H(xi) = C(xi) * (1 - w) / (1 - w * B(xi))

with `B` the kernel's frequency response. For linear interpolation at phase f,
`B = sqrt(1 - 2f(1-f)(1 - cos 2*pi*xi))`, i.e. `|cos(pi*xi)|` at the worst phase f = 0.5. At the
user's `w = 0.85`:

| spatial period (render texels) | B (bilinear, f=0.5) | steady-state contrast | B (Catmull-Rom) | steady state |
|---|---|---|---|---|
| 8 | 0.924 | 70% | 0.992 | 95% |
| 4 | 0.707 | 38% | 0.884 | 60% |
| 2 (Nyquist) | 0 | 15% | 0 | 15% |

and when motion stops `B -> 1` and the loss unwinds as `w^n`: 18 frames to 5%, which at this
scene's frame rate is the "about a second". A directional, motion-only loss of mid-frequency
contrast that decays at the EMA rate IS the reported smear -- and at `cloudHistoryWeight = 0` the
feedback term vanishes and so does the artefact, matching the bisection exactly.

The 2026-09-06 rectification's claim that "a heavy smear is structurally impossible at ANY history
weight" is the thing that fails here, and it fails for a specific reason worth recording: the clip
bounds DISPLACEMENT from the current frame's local mean, and this defect is a low-pass TOWARD that
mean. A resampling-blurred history is approximately `m1`, i.e. the exact centre of the clip box, so
`maxUnit < 1` and the clip returns it untouched. That fix bounded the wrong quantity, not the wrong
link -- which is why it was reported as principled but not a proof of root cause, and why the
bisection came back unchanged.

## The change

`fetchCloudHistoryBilinear` (`composite.comp.slang`) keeps its coordinate, its per-tap validation
and its fallback, and changes only its kernel: separable **Catmull-Rom** over the 4x4 footprint
around the same base texel. The cubic's response is flat to fourth order instead of second, which
is what breaks the compounding (the table above). Concretely:

1. `cloudHistoryCatmullRomWeights(f)` -- the four cubic-Hermite weights at offsets -1..+2. They sum
   to 1 at every phase, so no renormalisation, and both outer weights (and one inner one) vanish at
   f == 0 and f == 1.
2. `loadValidatedCloudHistoryTap(...)` -- the existing age + depth validation, lifted verbatim into
   a helper so the narrow and wide footprints validate on exactly the same rule.
3. One 4x4 sweep accumulates BOTH kernels. A tap is loaded only when its Catmull-Rom weight is
   non-zero, and a tap that is not loaded cannot veto -- so a static camera loads ONE texel (its
   old cost and its old answer, bit-identical), and a pure yaw (`f.y == 0`) loads one row of four.
4. If every consulted tap validated, the Catmull-Rom result is used, clamped to >= 0 (the negative
   outer lobes can ring premultiplied radiance / opacity below zero, where neither has a meaning;
   the overshoot on the other side is left to the 3x3 clip, which IS shaped to catch a large local
   excursion from the current mean). Otherwise the fetch returns the validated bilinear over its
   inner 2x2 with the existing renormalisation -- today's behaviour verbatim.

So the wide kernel runs only across a uniformly valid footprint (open sky, deck interior -- where
the smear lives) and silhouettes, disocclusions and screen borders keep the narrow one. Behaviour is
a strict superset of the old function: every case the old code handled resolves identically.

`cloudHistoryWeight` and `cloudHistoryClampGamma` are **untouched**. This recovers detail the filter
was throwing away rather than admitting less history, so the user's 0.85 / 1.25 tuning keeps its
full smoothing -- which is the point: `cloudHistoryWeight` 0 was never the fix, and lowering it or
tightening the clamp would only have traded the smear for noise.

Debug view 883 keeps its BILINEAR duplicate on purpose (comment added there): that view measures the
reprojected COORDINATE, and a wider reconstruction would blend some of the coordinate error away and
flatter the very thing being measured.

## Residual risk

* **Not built, not run.** The arithmetic is checkable; the visual is not. Balanced-delimiter and
  type review only. If Slang objects to anything it will be the `const float wx[4] = {...}` unpack
  or the `[unroll]`ed `int` loop bounds, both of which mirror idioms already in this file.
* **Cost.** Worst case 16 frame-id loads + 16 colour loads per pixel instead of 4 + 4, on frames
  where both axes have a fractional motion component. Static camera: 1 + 1. Pure yaw: 4 + 4. For
  scale, the 3x3 neighbourhood statistics in the same function already cost ~18 loads per pixel
  after the outline fix. If this shows up in a frame capture, the cheap retreat is to run the wide
  kernel only on the axis with the larger fractional phase (a 4x1 footprint, 4 + 4 loads), which
  recovers most of the benefit for yaw-dominated motion.
* **Ringing.** Catmull-Rom overshoots at high-contrast edges. Here the overshoot lands in a temporal
  history that is then clipped to the current frame's 3x3 box, so it is bounded by construction --
  but at `cloudHistoryClampGamma = 0` (the A/B setting, "never for use") that guard is off and
  ringing at deck silhouettes would be visible. Do not A/B the clamp to 0 with this in.
* **What is NOT fixed.** Nyquist-scale detail still collapses to 15% while panning; no interpolator
  can recover it, because a half-texel sample of an alternating signal genuinely is zero. If the
  smear is still objectionable after this, the next honest lever is a motion-aware history weight
  (drop `w` as the fractional phase approaches 0.5), which trades the residual blur for march noise
  while turning -- a real trade, and one that should be made with the artefact back on screen rather
  than in advance.
* The diagnosis rests on the reprojection being exact, which was established by reading rather than
  by measurement. Debug view 883's R channel (reprojected-history vs current-frame relative error)
  is the in-game confirmation: if it is near zero while panning, the coordinate is right and this
  change is aimed correctly; if it is large and directional, the coordinate is wrong after all and
  this change is treating a symptom.
