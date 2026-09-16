# SHARC in open scenery: why Fallout New Vegas misses where Portal RTX hits (2026-09-16)

Worktree `wsn3g`, branch `revised-9-10`, HEAD `6724da5c8`, tree clean. Source reading only:
nothing built, deployed, pushed or run; no source file modified. Statements are labelled
**source** (read from the tree, with file:line at HEAD) or **inference** (a conclusion drawn from
source facts plus stated assumptions). Every in-game impact below is **unmeasured**. The
measurement being explained is the user's: Portal RTX hit rate ~95%, eligible ~74%, terminates
~78%; FNV ~57% at best, ~20% over open scenery, same build, near-identical settings.

Abbreviations: `grid` = `src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`, `integrator` =
`src/dxvk/shaders/rtx/algorithm/integrator_indirect.slangh`, `pass` =
`src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slangh`, `host` =
`src/dxvk/rtx_render/rtx_sharc.cpp`.

## Verdict

The two halves of the miss split are the same disease at two severities, not two diseases.
A cell's readability is decided by how often an update path lands in it, and the resolve
arithmetic turns that interval directly into the bucket the panel will print:

| samples arrive every `k` frames | at FNV settings (`accumulationFrames` 4, `minSampleCount` 2, `staleFrames` 32) | panel bucket |
|---|---|---|
| `k` ≤ 3 | readable after 3–4 samples | hit |
| 4 ≤ `k` < 32 | exists, **never** readable | below sample floor |
| `k` ≥ 32, or never sampled | evicted / absent | no cell |
| bucket full on insert | absent, uncounted | no cell |

(Derivation in §3.) Outdoors the sample interval is long for most cells because a cell is fed
only by primary pixels whose bounce ray lands in it, and in open terrain that is a sliver of the
frame instead of the whole room; on top of that, most update paths launched from open ground
exit to the sky at the first bounce and insert nothing. So whichever way the split falls, the
remedy is sample density and retention (`accumulationFrames`, `updateTileSize`, `staleFrames`,
`gridScale`), with one exception: a "no cell" share that does not move under a capacity change
is density; one that does is capacity. Capacity is unlikely (§2) but is the one thing a single
config line settles.

Plainly: a constant-angle world-space hash is a worse fit for open terrain than for an enclosed
room, and no setting changes the ceiling on what it can *do* outdoors, because paths that exit
to the sky at the first bounce never consult it. The settings below can recover an order of
magnitude of per-cell sample density, which should move the hit rate a long way; whether that
buys frame time is bounded by the sky-exit share the panel already prints.

## 1. Does the grid hold screen footprint constant with distance? Yes.

**Source.** Level: `gridLevel = 0.5 * log_base(|camera − p|²) + levelBias`, clamped to
[1, 127] and truncated (`grid:153-160`). With base 2 and `levelBias` 0
(`sharc_bindings.slangh:33-35`) that is `L = floor(log2 d)`. Voxel: `2^(L − levelBias) / sceneScale`
(`grid:163-168`), and `sceneScale` is `cb.sharcArgs.gridScale` (`sharc_bindings.slangh:34`).
Position is quantised at that voxel in absolute world units (`grid:177`).

**Inference.** For a point at camera distance `d`, `2^L ≤ d < 2^(L+1)`, so the voxel edge is in
`(d / (2·gridScale), d / gridScale]`: at `gridScale` 50, between 0.57° and 1.15° of arc,
geometric mean about 0.8°, for a camera-facing surface. That is the intended constant-angle
behaviour, and it is unit-free: the level depends on world distance but the voxel divides the
same distance, so the FNV game unit and `rtx.sceneScale = 0.1` do not enter. The option text
("larger values give finer cells", "independent of rtx.sceneScale", `rtx_sharc.h:43`) is
correct. At 2560 px across a ~90° field of view (~28 px/deg) a voxel spans roughly 16–32 px,
i.e. 2–4 update tiles across, for a surface facing the camera.

Consequence: scene *extent* does not by itself multiply cells. The number of visible ground cells
grows with `ln(d_far / d_near)`, not with area (§2), so "`gridScale` 50 is wrong for a scene this
size" is not the diagnosis. `gridScale` remains the quadratic density lever (§7), for a different
reason.

Two corollaries worth knowing:

- Surfaces seen at a grazing angle (open ground from eye height) keep the 0.8° *width* but their
  on-screen *height* is 0.8° × sin(grazing angle): the ground at 100 m from a 1.7 m eye is ~1°
  off the view ray, so its cells are ~25 px wide and under a pixel tall. That matters less than
  it looks, because a cell is not fed by its own pixels (§4).
- Near the camera the level flips every time the distance to a point crosses a power of two, so
  for ground within a few metres of a running player the key changes several times a second.
  `SHARC_BLEND_ADJACENT_LEVELS` (`SDK:103`, on) seeds a fresh cell from its neighbour level in its
  first two frames only (`SDK:954`), and only if that neighbour is above the floor. Portal's
  camera moves at similar speed, so this is not the differentiator, but it is one more reason
  the near ground outdoors is perpetually young.

Key packing is sound at FNV coordinates. **Source.** 17 position bits per axis (`grid:64`) mask
the quantised coordinate. **Inference.** At world coordinates up to ~2×10^5 units and voxels
down to 0.04 units the coordinate wraps modulo 131072, but two cells at the same level lie in
the same distance band whose diameter is `4·2^L = 200` voxels, so no two live keys can alias.
Float precision at 2×10^5 units is ~0.016 units, ~1% of a 1 m-distance voxel: negligible.

## 2. Cell budget against 2^20

**Source.** Capacity `1 << clamp(capacityLog2, 18, 22)` (`host:91`); buckets are 16 slots of
linear probing (`HashGridTypes.h:14`, `grid:279-298`). An insertion that finds all 16 occupied
by other keys returns false (`grid:297`). The deferred update then returns without recording the
vertex (`update:42-45`); the non-deferred path ignores the return (`hooks:313`). **No counter
records either.** A failed vertex does not terminate the path; later vertices still credit the
earlier inserted ones (`update:16-23`, `57-67`), so the cache degrades by silently dropping
cells rather than thrashing.

**Inference, visible ground cells.** Flat ground in a wedge of angle φ from `d0` to `d1`, voxel
edge `≈ d/70`: cells ≈ 70² · φ · ln(d1/d0). For φ = 1.57 (90°), `d0` = 2 m, `d1` = 2 km:
≈ 53k. Terrain normals straddle the axis planes, so the three sign bits in the key
(`grid:197-200`) split each voxel into up to four cells (§6): ≤ 200k. Add rock and building
faces, distant relief and the hidden faces that bounce rays reach: order 10^5 cells *touched*
per frame in a wide vista. Cells live `staleFrames` = 32 frames after their last sample, so a
panning camera keeps 1–3× the per-frame set live: 10^5 to 3×10^5 against 1,048,576 slots, load
0.1–0.3. Sixteen-slot buckets essentially never fill at that load. **So capacity is unlikely to
be the cause**, but it is unmeasured and the panel cannot see it, which is why it is the first
"no cell" test in §7: one config line, no code.

## 3. Eviction and readability: the resolve arithmetic

**Source** (`SDK:830-985`). Per entry per frame, with `sampleNum` = new samples this frame and
`N` = resolved `accumulatedSampleNum`:

- `staleFrameNum` resets to 0 on a sample, else increments (`SDK:853`); at `staleFrames` the
  slot is cleared, key included (`SDK:859-868`). `staleFrames` is clamped to [8, 128] by the host
  (`host:151`), [8, 1024] by the SDK (`SDK:115`).
- With no new sample the resolve only increments the frame counters and returns (`SDK:870-877`);
  `N` and the radiance are untouched, so an idle cell stays readable until evicted.
- With a sample, if the cell's `accumulatedFrameNum` exceeds `accumulationFrames` (`A`), `N` is
  scaled by `A / accumulatedFrameNum` and the frame count is reset to `A` (`SDK:925-930`); then
  `N += sampleNum` (`SDK:947-949`).
- The query reads a cell only if `N > minSampleCount` (`SDK:741`, threshold from
  `sharc_sdk.slangh:110-111`).

**Inference, closed forms.** Let `m` = `minSampleCount`.

- Samples every frame, `s` per frame: `N → s·(A+1)`.
- One sample every `k` frames: at each sample the frame count is `A + k`, so
  `N ← N·A/(A+k) + 1`, fixed point `N∞ = 1 + A/k`. Readable in steady state iff `k < A/(m−1)`,
  i.e. **iff `k < A` at `m` = 2**.
- A cell returning after `k` idle frames with history `N`: `N·A/(A+k) + 1`. At `A` = 4 a cell
  that held 10 samples and was out of view for a second (60 frames) comes back as 1.6:
  unreadable. At `A` = 32 it comes back as 4.5.

At the FNV settings (`A` = 4, `m` = 2, stale 32): `k` ≤ 3 readable after 3 samples at `k` = 1
(frames 1–3), 3 samples at `k` = 2 (frame 5), 4 samples at `k` = 3 (frame 10); `k` = 4 converges
to exactly 2.0 and never passes the strict `>`; `4 ≤ k < 32` exists and is never readable;
`k ≥ 32` is evicted and re-created empty on the next path. This is the table in the verdict.

Why Portal does not notice: its cells receive samples every frame (§4), `N → s(A+1)` with `s`
well above 1, so `A` = 4 costs nothing there. It is a responsiveness setting that only bites
when samples are sparse.

`accumulationFrames`, `staleFrames` and `updateTileSize` take effect without clearing the cache
(`host:101-106` lists what clears); `gridScale` and `capacityLog2` clear it (`host:102`, `107`).

## 4. Sample density per cell: per screen area of the *illuminator*, not of the cell

**Source.** The update dispatch is tile-sized (`rtx_pathtracer_integrate_indirect.cpp:782-786`);
each tile picks one rotating source pixel (`pass:77-92`) and launches the path that pixel's
primary surface already sampled in the direct pass (`pass:171-174` loads
`RayOriginDirection`). Vertices are inserted at secondary hits only (`hooks:307-313`, called from
`integrator:576`); the primary is never inserted. Update paths run `min(updateBounces, 8)`
bounces with roulette off (`path_state.slangh:309-310`, `544-`), each hit adding one sample to
its cell (`update:64`, `sampleData` 1).

**Inference.** A cell `c` therefore receives, per frame, about
`(N_px / T²) · ⟨projected solid angle of c seen from the visible primaries⟩ / π` samples at the
first bounce, plus later-bounce arrivals. Its own on-screen size is irrelevant, except that a
flat surface never bounces onto itself. What sets `s(c)` is how much of the *frame* illuminates
`c`:

| | Portal RTX test chamber | FNV open desert |
|---|---|---|
| illuminators of a wall / ground cell | every other wall, i.e. most of the frame | the few nearby rocks or walls whose bounce rays reach it |
| first-bounce sky exits (no vertex, sample wasted) | none | most paths from ground and roofs (cosine-weighted hemisphere over a low horizon: ~70–90%) |
| vertices per update path (bounces 4, no roulette) | ~3–4 | ~0.5–1 |
| samples per frame at 2560×1440 native, `T` = 8 (58k paths) | ~150–230k | ~10–30k |
| cells fed per frame | ~10^5 (room walls at ~4 cm voxels) | ~10^5 (§2) |
| mean `s` per cell | ~1–3 | ~0.1–0.3, i.e. `k` ≈ 3–10 |
| `N∞` at `A` = 4 | ~5–15 | ~1.4–2.3 |

The FNV column is a mean over a wide spread: the ground ring around a nearby rock is fed by the
rock's ~10^3 update paths per frame across ~10^3 cells (`s` ≈ 0.5, marginal); distant relief is
fed by the ~0.5–1% of ground paths that leave within a few degrees of the horizon, spread over
~10^3 cells per 90° of azimuth (`s` ≈ 0.1, `k` ≈ 10, never readable at `A` = 4, readable at
`A` = 32 with `N∞` ≈ 4). Upscaler internal resolution scales both columns equally. These
figures are assumptions dressed as numbers; the split panel is the measurement. Their shape is
what matters: a 5–30× density gap, which is the size of the hit-rate gap.

## 5. The sky

**Source.** A miss adds sky radiance × path weight to the vertices already inserted
(`integrator:395-419`, `accumulateRadiance` → `update:16-23` or `SDK:545-579`); with none
inserted, nothing is written. A query path that misses at the first bounce never performs a
lookup and is counted as a sky exit (`integrator:549-558`, counter 10), outside the hit-rate
ratio. `skyGatherEligible` selects `skyIndirectRadianceScale` on the miss (`integrator:406-410`)
and is the `diffusePath` input to the gate (`hooks:277-278`); both stages set it the same way
(`integrator:638-641`, `1400`). Cloud shadow on the sun applies at every NEE in both stages
(`integrator:759-771`).

**Inference.** The sky changes nothing about *what* is inserted and nothing about eligibility.
It changes how many update paths insert anything (the wasted-path row above) and it caps the
cache's usefulness: terminations ≤ (1 − sky-exit share) × hit rate. If the panel's
"Path ends: sky" line reads 60–70% over open ground, a 95% hit rate would terminate ~30% of
paths, against Portal's 78%. In towns, canyons and interiors the sky share falls and FNV should
behave like Portal. One asymmetry, not a hit-rate matter: a specular arrival that terminates on a
cell reads back sky scaled for a diffuse gather, because the cell's sky term was scaled by the
*update* path's continuation lobe.

## 6. Other separators between an open world and a room

1. **Wasted update budget** (§4, §5): the single largest factor, and only `updateTileSize`
   restores it from settings.
2. **Terrain normal octants.** **Source.** The key's three normal bits are the signs of the
   geometry normal (`grid:197-200`), with a 10^-3 bias only for exactly axis-aligned normals.
   **Inference.** Portal's walls are axis-aligned, one class per wall. FNV terrain is near `+z`
   with `x`/`y` components of either sign, so adjacent triangles of one voxel land in two to four
   different cells and each cell sees a half or a quarter of the voxel's samples. Density ÷2–4
   on terrain specifically. A dominant-axis quantisation (six classes) would merge them without
   merging the two sides of a thin wall; see §8.
3. **Level flips near the camera** (§1): fresh cells several times a second on the near ground.
   Shared with Portal in mechanism, worse outdoors only because the near ground is the closest
   surface at eye height.
4. **Pan-and-return.** Cells off-screen get no samples, are evicted at 32 frames (0.53 s at
   60 fps) and, if they survive, have their history crushed on the first new sample at `A` = 4
   (§3). A room's cells rebuild in 3 frames; outdoor cells with `k` ≥ 4 never rebuild.
5. **Update depth is moot outdoors.** Update paths end at the sky; raising `updateBounces`
   buys vertices only indoors.
6. **Geometry LOD streaming** swaps distant meshes and shifts surfaces by up to a voxel;
   vegetation cutouts reject on opacity (`hooks:38-41`, counted, not a miss). Minor.

## 7. Ranked changes

Read the split after each step; the two buckets trade against each other (raising
`staleFrames` moves `k` ∈ [32, 128) cells from "no cell" into "below floor" unless `A` rises with
it). All impacts are unmeasured.

### If "below sample floor" dominates

1. **`rtx.sharc.accumulationFrames = 32`** (from 4). Readable iff `k < A`, so every cell that
   survives eviction becomes readable once it holds three samples, and returning cells keep
   their history (`N·32/(32+k)`). Expected: the below-floor bucket largely converts to hits over
   the first few seconds in a view. Cost: the indirect term reacts to lighting change with a
   ~32-frame time constant (sun motion in FNV is slow; a muzzle flash or explosion trails by
   ~0.5 s in the indirect channel only). Takes effect live. Try 64 if the bucket is still
   large; 64 is the host clamp (`host:150`).
2. **`rtx.sharc.updateTileSize = 4`** (from 8). Four times the samples per cell, `k` ÷ 4.
   Cost: the update pass traces 1/16 instead of 1/64 of the query's paths (with ~4 bounces and
   no roulette, that is roughly 16% of the query's segment count instead of 4%); the panel's
   "GPU ms: update" line reports it. Takes effect live.
3. **`rtx.sharc.staleFrames = 128`** (from 32), only together with 1. Alone it keeps cells whose
   history the next sample destroys.
4. **`rtx.sharc.gridScale = 25`** (from 50). Voxels 1.1–2.3° of arc: four times fewer cells,
   four times the samples per cell, and the too-close and footprint thresholds double with the
   voxel (`hooks:319`, `329`). Cost: coarser indirect lighting; in interiors an 8 cm voxel at
   3 m can average across an inside corner. Clears the cache. Try after 1–3, or first if the
   update-pass cost of 2 is unwelcome.
5. **`rtx.sharc.minSampleCount = 1`**, last. With `A` ≥ 32 any cell that has ever received two
   samples is readable; it brings back the two-sample noise the floor was added against.

### If "no cell" dominates

1. **`rtx.sharc.capacityLog2 = 22`** for one session (160 MiB; the resolve runs 4M threads a
   frame, "GPU ms: resolve" reports it). If the no-cell share does not move, capacity is
   exonerated: revert to 20, and the bucket is `k` ≥ 32 plus never-sampled cells, which is the
   density problem. Apply 2, 4 and then 3 + 1 from the list above; watch the bucket move into
   "below floor" as `staleFrames` rises and then into hits as `A` rises.
2. If it does move, keep 21 (80 MiB) or 22 and add the occupancy counter from §8 before
   tuning further, so the load is read rather than guessed.

### Either way

- Read "Path ends: sky" in the same window. It bounds the termination share (§5) and it is the
  honest figure for what the cache can do in that view. Do not chase the hit rate past the point
  where "Cache terminates" stops moving.
- Portal RTX needs none of this. Its cells are fed every frame; `A` = 4 is harmless there.
- The "Performance preset" button (`host:362-367`) sets tile 8, bounces 4, capacity 20: the FNV
  configuration. It was chosen for Portal's cost, where the cache was over-served.

## 8. Code changes considered and not made

None is needed to act on the split; the settings above give ≥ 16× density before any code is
touched, and the split has not been read yet. Listed so they can be picked up if the settings
plateau.

1. **Dominant-axis normal keying** (§6.2), default-off `rtx.sharc` option. Replace the three
   sign bits with the index of the largest normal component and its sign (six classes) in
   `HashGridComputeSpatialHashFromGridPosition` (`grid:188-208`), selected through a field on
   `HashGridParameters` (which already carries the fork's `portalSpace`,
   `HashGridTypes.h:26`) set in `makeSharcParameters`. Plumbing: `SharcArgs` sits mid-struct
   in `RaytraceArgs` (`raytrace_args.h:165`, `static_assert(sizeof == 80)`), so a new field
   would shift every later member and break the legacy-blob byte-identity the validator pins;
   `sharcArgs.enabled` is written (`host:155`) and read by nothing, so it can carry the flag
   without a layout change. Must join the cache-clear conditions (`host:101-106`). Expected
   gain: ×2–4 density on terrain only. Unmeasured.
2. **Insertion-failure and occupancy counters**, stats-only. Count `HashGridInsertEntry` false
   returns in `sharcInsertVertex` (`update:42-45`) and live keys in the resolve
   (`sharc_resolve.comp.slang`, one wave-reduced atomic per group). Needs the stats buffer bound
   to the update and resolve stages, which today only the query stats variants bind
   (`rtx_pathtracer_integrate_indirect.cpp:307-313`). Replaces the capacity-22 test with a
   reading.
3. **Sky-miss retry for update paths** (re-sample the first bounce when it misses, to stop
   spending 70–90% of the outdoor update budget on nothing). Unbiased per cell, since a cell's
   sample value does not depend on how the ray reached it. Not cheap: the first-bounce
   direction is sampled in the direct pass and stored (`pass:171-174`); re-sampling in the
   indirect pass means reconstructing the primary material there. Not worth it until the
   settings plateau.

## What only the game can settle

1. Which bucket dominates, per view type (open vista, town, interior). The panel at HEAD.
2. Whether "Path ends: sky" outdoors is 60–70% as assumed, which sets the ceiling on
   terminations.
3. The update-pass cost at tile 4 and the resolve cost at capacity 22, from "GPU ms".
4. Whether a 32–64 frame indirect lag is visible on transient lights in FNV.
5. Whether `gridScale` 25 flattens indirect lighting visibly indoors.
