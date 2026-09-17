# What `rtx.sharc.gridScale` actually is - 2026-09-16

Investigation of the fork's "Grid density" option: what the parameter is, whether the name and
description are accurate, whether its stated independence from `rtx.sceneScale` is correct, and
whether the earlier "unit-free" conclusion in
[SHARC-open-world-diagnosis-2026-09-16.md](SHARC-open-world-diagnosis-2026-09-16.md) holds.

Nothing here was observed in game. Every expected in-game consequence below is marked unmeasured.

Abbreviations: `grid` = `src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `bindings` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_bindings.slangh`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `host` =
`src/dxvk/rtx_render/rtx_sharc.cpp`, `opt` = `src/dxvk/rtx_render/rtx_sharc.h`.

## Verdict in one paragraph

`gridScale` is SHARC's `sceneScale` under a different name, wired straight through with no
conversion. The direction claim in the option text is correct: larger values do give finer cells.
The independence claim is also correct, and not by luck - SHARC's hash grid is *scale-invariant by
construction*, so its scene scale is a dimensionless number and could not correctly depend on
`rtx.sceneScale` or on any other units-per-metre figure. The earlier document's "unit-free, about
0.8 degrees of arc per voxel" conclusion holds; I re-derived it independently and it is right.
NVIDIA's "controls voxel size in world space" is also right and does not contradict it. Nothing
about the grid is mis-wired. The defect is entirely in the wording: the old text stated the two
true facts without the one fact that makes them make sense, which is that a cell's size is a
fraction of camera distance rather than a length. Text fixed; no behaviour changed.

## 1. Is `gridScale` simply `sceneScale` renamed? Yes.

**Source.** `bindings:34` assigns `p.hashGridParameters.sceneScale = cb.sharcArgs.gridScale`, with
no scaling on either side. `host:94` clamps the option to `[1, 1000]` (falling back to 50 on a
non-finite value) and `host:147` copies it into `m_args.gridScale`; nothing else in the tree reads
or writes the field. The rest of `makeSharcParameters` is fixed: `logarithmBase =
SHARC_GRID_LOGARITHM_BASE` = `2.0f` (`bindings:33`, `SDK:57`, NVIDIA's documented default) and
`levelBias = 0.0f` (`bindings:35`, which `HashGridTypes.h:25` calls the recommended start).

So the fork invented a *name*, not a concept. `sharcSceneScale` in NVIDIA's constant-buffer list is
this field, and the recommended starting value of 50.0f in the same document is this default.

## 2. Direction: "larger values give finer cells" is correct

**Source.** `grid:163-168`:

```
float exponent = log2(gridParameters.logarithmBase) * (float(gridLevel) - gridParameters.levelBias);
return exp2(exponent) * rcp(gridParameters.sceneScale);
```

`rcp(sceneScale)` is a reciprocal, so voxel edge falls as the value rises. The old text was right.

**Cross-check against upstream.** The vendored `grid` was compared with NVIDIA's published
`HashGridCommon.h`: `GetLevel`, `GetVoxelSize` and `LogBase` are character-for-character the
upstream bodies, and `git log -- grid` shows they have not been touched since the file was
vendored in `061f13bb7`. The only fork edits to the file are the two commented `NV-DXVK` blocks -
level narrowed from 9 bits to 7 (`grid:68`) to free bits 61-62 for portal space
(`grid:83`, `grid:205`). That narrowing is sound: with 9 level bits the portal field would have run
into bit 63, which the SDK reserves for user data. Everything that consumes those fields does so
symbolically (`grid:160`, `194`, `232`; `SDK:791`, `806`, `817`), so nothing assumes the old width,
and the reachable level range is unaffected - see §5.

## 3. Unit-scale independence: correct, and necessarily so

This is the substantive question, so the derivation is spelled out.

**Source.** Level (`grid:153-161`): `gridLevel = 0.5 * log_base(|camera - p|^2) + levelBias`,
clamped to `[1, 127]` and truncated. Voxel (`grid:163-168`):
`voxel = base^(level - levelBias) / sceneScale`. Position is quantised at that voxel in absolute
world units (`grid:171-180`).

**Inference.** Substitute one into the other. With `base = 2` and `levelBias = 0`,
`level = floor(log2 d)` for a point at camera distance `d`, so `2^level` lies in `(d/2, d]` and

```
voxel = 2^floor(log2 d) / gridScale,   voxel / d in (1/(2*gridScale), 1/gridScale]
```

The ratio of voxel size to camera distance is fixed by `gridScale` alone. `voxel/d` is an angle in
radians, so **`gridScale` is 1/(angular cell size)** - a pure number with no length dimension. At
50 that is 0.0100 to 0.0200 rad, i.e. 0.57 to 1.15 degrees of arc, geometric mean 0.81 degrees.

The consequence for the two titles: scale every world coordinate by *k* (camera, hit and origin
alike, which is exactly what a different unit convention is) and `d` scales by *k*, `level`
increases by `log_base k`, `voxel` scales by *k*, and `gridPosition = floor(p / voxel)` is
**unchanged**. The grid is invariant under uniform scaling of world space. The same `gridScale`
therefore produces the same cell structure in Gamebryo units and in Source units, and the FNV and
Portal RTX measurements *are* directly comparable on this axis.

**Coupling it to `rtx.sceneScale` would break it, not fix it.** `rtx.sceneScale` is a
units-per-centimetre conversion constant (`rtx_options.h:346`, `getMeterToWorldUnitScale()` at
`:1514`); it is not a world transform, so changing it does not move a single position SHARC sees.
Multiplying `gridScale` by it would divide the cell's *angle* by the game's unit convention, which
is a quantity the grid has already cancelled out - the same physical scene would get finer or
coarser cells purely because a config file says 0.01 rather than 0.1. That the value is in practice
unreliable (most titles ship 0.01 regardless of their real units; FNV's own config says 0.1 at
`Fallout New Vegas/rtx.conf:112` against a Gamebryo world of roughly 70 units per metre, a
discrepancy already noted at `rtx_precipitation.cpp:750-751`) is a second reason on top of the
first, but the first reason stands on its own: even a perfectly accurate `rtx.sceneScale` should
not enter here.

**Where world units *do* enter.** Three absolute constants exist in the grid, and they are the
complete list:

| Constant | Where | Binds when | FNV / Portal |
| :-- | :-- | :-- | :-- |
| level floor of 1 | `grid:160` | `d < 2` world units; voxel stops shrinking at `2/gridScale` = 0.04 units | ~2.9 cm / ~3.8 cm from the eye |
| `HASH_GRID_POSITION_BIAS` 1e-4 | `grid:95` | never in practice; it is a tie-break at the origin | 0.25% of the smallest voxel |
| `max(distance2, 1e-10)` | `grid:157` | `d < 1e-5` world units | never |

(The centimetre figures come from the engines' own unit conventions - Gamebryo about 70 units
per metre, Source about 52.5 - not from anything in the fork or either rtx.conf.)

Only the first is worth a sentence: inside 2 world units of the camera the angular invariance
fails and cells become angularly coarse. In both titles that is a few centimetres, and the
too-close gate rejects almost everything there anyway (it would demand a segment longer than
`1.732 * 0.04` = 0.069 units). It would matter in a game authored at 1 unit = 1 metre, where the
floor would sit at 2 m. Neither of these games is like that. The upstream comment on
`HASH_GRID_POSITION_BIAS` ("may require adjustment for extreme scene scales") is the only place the
SDK itself acknowledges a unit dependence, and it is the least consequential of the three.

**No other SHARC option carries length units.** `minRoughness`, `minRoughnessSpecular`,
`maxEmissiveLuminance`, `radianceScale`, `minSampleCount`, `accumulationFrames`, `staleFrames`,
`updateTileSize`, `updateBounces` and `capacityLog2` are all dimensionless or counts
(`sharc_args.h:5-40`). So no part of the SHARC configuration needs per-title unit calibration, and
a tuning result carried from Portal RTX to FNV carries cleanly.

## 4. Reconciling the earlier document with NVIDIA's guidance

The two claims are both true and do not conflict.

- NVIDIA: "It controls voxel size in world space and is not tied to screen resolution." Correct.
  The voxel does have a world-space size, `d / gridScale`, and the knob does set it. The warning is
  against hard-coding the value and against deriving it from pixel counts. The fork complies:
  the value is exposed, defaults to NVIDIA's recommended 50, and has three debug views behind it
  (`rtx_debug_view.cpp:513-521`).
- Earlier document, §1: "unit-free, about 0.8 degrees of arc per voxel regardless of distance."
  Correct. I re-derived the bound `(d/(2*gridScale), d/gridScale]` and the 0.57-1.15 degree figure
  independently from `grid:153-168` and get the same answer.

The apparent conflict comes from reading "world space" as "absolute, independent of the camera".
It is not: the world-space size is *proportional to camera distance*, which is precisely what makes
the coefficient dimensionless. Both sentences describe the same equation.

I also re-derived the earlier document's key-aliasing argument and it holds, with a stronger bound
than it states. Two entries collide in the 17-bit position field (`grid:64`) only if their grid
coordinates differ by `2^17` on an axis, i.e. by `131072 * voxel` in world space. Two entries at the
same level both sat in the shell `[2^L, 2^(L+1))` from the camera that inserted them, so they are at
most `2^(L+2)` = `4 * gridScale` voxels apart. Aliasing therefore needs `gridScale >= 32768`, and
the clamp caps it at 1000 - a 32x margin at any legal setting, in any world, at any distance from
the origin. (Entries persist up to `staleFrames`, so in principle the camera could move between two
insertions; closing that would need a drift of ~2600x the hit distance within 32 frames, and a cut
that large clears the cache through `resetHistory` at `host:103`.) The earlier document's
"4 * 2^L = 200 voxels" is this same number for `gridScale = 50`.

## 5. Everything else about the grid

Checked and correct:

- **Level selection.** `0.5 * log_base(d^2)` is `log_base(d)`; the `0.5` is not a stray factor.
  `uint(clamp(x, 1, 127))` floors a value already >= 1, so truncation is floor. Upstream verbatim.
- **`logarithmBase` = 2.** Only sets how coarsely the continuous level is quantised; the mean
  angular cell size is `1/gridScale` for any base. NVIDIA's documented default.
- **`levelBias` = 0.** It shifts *where* level boundaries fall and moves the near-camera voxel
  floor (`2^(1-levelBias)/gridScale`); it does not change the mean cell angle. 0 is what
  `HashGridTypes.h:25` recommends and what upstream ships.
- **Clamping.** Lower clamp discussed in §3. Upper clamp of 127 binds at `d = 2^127`, i.e. never;
  the 9-to-7-bit narrowing did not cost anything reachable.
- **Portal space across a level change.** `SharcGetAdjacentLevelHashKey` (`SDK:781-827`) rebuilds a
  key by hand. The fork correctly re-ORs the portal bits (`SDK:823-825`) as well as the normal bits;
  without that, `SHARC_BLEND_ADJACENT_LEVELS` (on, `SDK:103`) would have blended portal-space cells
  into main space. This was already handled.
- **Constant-buffer layout.** `SharcArgs` places `vec3 cameraPositionPrev; float gridScale;` as one
  16-byte row (`sharc_args.h:8-9`), the struct is 80 bytes with a `static_assert` (`:45`), and both
  vec3 fields are followed by a 4-byte scalar, so HLSL packing and C++ packing agree. A misaligned
  `gridScale` was a plausible silent failure; it is not present.
- **Cache invalidation.** `host:104` clears on a `gridScale` change, which is required because
  existing keys were built at the old voxel size.
- **Interaction with the distance guard.** `hooks:455-459`:
  `farEnough = segmentHitDistance > 1.732051 * voxelSize`. Both sides are world-space lengths, so
  this is scale-invariant too. Substituting the voxel gives the useful form: **a segment must
  subtend more than `1.732 / gridScale` radians at the camera** - about 2.0 degrees at 50, i.e.
  1.7% to 3.5% of the hit's distance from the camera. Same for the footprint gate
  (`hooks:77-83`, `464-465`), which compares a lobe footprint against the same voxel.

One deviation from the guidance, harmless but worth recording: NVIDIA documents the range as
1.0-100.0; the fork's clamp and slider go to 1000 (`host:94`, `378`). That is a wider tuning range,
not a wrong one, and I left it alone - narrowing it would be a behaviour change that could
invalidate a config already using a higher value. The new description names NVIDIA's range.

One consequence of §5's last point that is *not* a bug but bears on FNV specifically: because the
guard threshold is a fixed fraction of camera distance, its absolute size grows with view distance.
A vertex 20,000 units out in open desert needs a segment of roughly 350-700 units to clear it, where
the same setting in a Portal room needs a few units. That is the intended behaviour of a distance-
proportional grid, but it means lowering `gridScale` to buy cell population (the earlier document's
recommendation 4, `gridScale = 25`) doubles that threshold, and the cost lands hardest outdoors.
Unmeasured; counters 5, 19 and 21 already split this and the "SHARC Query: Too Close Guard" debug view (`rtx_debug_view.cpp:500`)
shows it per pixel.

## 6. If `rtx.sceneScale` cannot be trusted, what should set voxel size per game?

**Nothing should, and nothing needs to.** The grid already normalises itself against world units
(§3), so there is no per-game quantity left for a derivation to supply. Concretely:

- **Deriving from scene or TLAS bounds would be a regression.** The logarithmic grid exists
  precisely so that world *extent* does not multiply cells - the visible cell count grows with
  `ln(d_far/d_near)`, not with area. Tying `gridScale` to bounds would reintroduce the dependence
  the design removes, and in a streaming open world the bounds change as cells load, so the value
  would drift and every change clears the cache (`host:104`). Strictly worse than a constant.
- **Deriving from resolution is what NVIDIA explicitly warns against**, and for a good reason: the
  cache carries a low-frequency term, so you want cells substantially coarser than a pixel
  (currently 16-32 px across at 2560 px / 90 degrees). Tracking resolution would make it noisier at
  high resolution for no benefit.
- **What does legitimately vary per title** is geometry density against cache capacity, and the
  balance between cell population and the too-close gate. Those are quality/cost trade-offs, not
  unit conversions, and a global knob tuned per title is the right shape for them.

Recommendation: keep `gridScale` as a global quality knob, document it as an angular density (done),
and do not build any derivation. If per-title variation turns out to be wanted after measurement,
the honest lever is the same knob set in that title's `rtx.conf`, not an automatic scale.

## 7. What changed

Text only. No shader, no host logic, no default, no clamp - the build is behaviourally identical
and nothing needs to be opt-in.

- `opt:45` - the `gridScale` description rewritten. It now says the parameter is the SDK's
  `sceneScale`, that a cell's edge is camera distance rounded down to a power of two divided by the
  value, that this makes it an angle rather than a length (0.57-1.15 degrees at 50) and *therefore*
  independent of `rtx.sceneScale`, that larger values give finer cells and quadratically more of
  them and loosen the too-close gate, NVIDIA's documented 1-100 range, and that changing it clears
  the cache. This string is also the in-game tooltip: `RemixGui` widgets bound to an `RtxOption`
  build their tooltip from the description (`rtx_imgui.cpp:151-161`).
- `host:378` - slider label "Grid density" -> "Grid density (SHARC scene scale)", so the panel
  itself says which SDK parameter this is. That name is where the impression that the fork invented
  a concept comes from.
- `RtxOptions.md:982` - the generated row regenerated by hand using the writer's own escaping table
  (`rtx_option_manager.cpp:421-444`), so `test_documentation` stays green without a rebuild.

Unchanged deliberately: the option key `rtx.sharc.gridScale` (renaming it would break existing
configs and the tuning already recorded against it) and the 1-1000 clamp.

## 8. What to check in game

All unmeasured. These verify the analysis rather than test a change.

1. **Confirm the angular law directly.** "SHARC Grid: Level / Voxel Size / Last Leg" puts voxel size
   in world units in G and the last resolve leg in B (`hooks:148`). Point at a surface, read G, and
   check `voxelSize * gridScale` is within a factor of two of the distance to that surface. Do it in
   both titles: if the numbers agree as ratios while the raw distances differ by the unit
   conventions, unit-independence is confirmed empirically, not just on paper.
2. **Confirm nothing moved.** The only edits are strings, so cache hit rate, eligibility and
   terminate rate should read identically to the current build at the same settings. Any change
   means something else moved.
3. **The tooltip.** Hover "Grid density (SHARC scene scale)" and confirm the new text reads sensibly
   in the panel.
4. **If tuning `gridScale` next**, the useful in-game reading is the too-close split (counters 5,
   19, 21 and the "Too Close" debug view) alongside "no cell" vs "under the sample floor"
   (counters 23, 24), because `gridScale` moves both in opposite directions: finer cells are easier
   to clear the gate with and harder to populate. Expect the trade to look different in the desert
   than indoors, since the gate's absolute size scales with view distance. Unmeasured.
