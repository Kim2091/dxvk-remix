# Fireflies on reflective materials with `allowSpecularPaths` — 2026-09-16

Worktree `wsn3g`, branch `revised-9-10`, from HEAD `c66e2d29b`. Statements are labelled
**source** (read from the tree, `file:line` after this change unless marked HEAD),
**inference** (a conclusion from source facts plus stated assumptions) or **judgement**.
**Every in-game and frame-time impact below is unmeasured.** Neither game was run for this
work. The only measurements in play are the user's, and they are named where used.

Abbreviations: `host` = `src/dxvk/rtx_render/rtx_sharc.cpp`, `opt` =
`src/dxvk/rtx_render/rtx_sharc.h`, `args` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_args.h`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`, `sdkglue` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_sdk.slangh`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `grid` =
`src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `indirect.cpp` =
`src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp`, `variants` =
`src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slangh`.

The report this answers: `rtx.sharc.allowSpecularPaths` is **amazing at lower DLSS presets —
SHARC shows far more detail than the other samplers** — but produces **a lot of fireflies on
reflective materials**, and the only other setting that helps is **lowering `updateTileSize`,
which costs too much performance to be acceptable**.

---

## Verdict

**The update budget is resolution-dependent, and that is the whole story.** The user's two
observations — worse at a low DLSS preset, cured by a smaller tile — are not two facts. They
are one fact seen twice: both change the same number, the count of update paths launched per
frame, and that number divides every outlier deposited into a cell. §1 derives the divisor
from source; §2 puts numbers on it.

**Nothing about the specular half is broken.** `footprintGate` and `minSampleCount` are not
failing at their jobs; they were never tests of this quantity (§3). The cache's cells have
always held the variance they hold — `allowSpecularPaths` did not create it. What a specular
arrival does is *read a cell sharply*, so it displays a cell error that a diffuse arrival
smears. The variance is the defect; the specular read is the messenger.

**The SHARC SDK ships no outlier handling of any kind** — none disabled by this fork, none
available to enable (§4). The one deposit site, `SharcAddVoxelData` (`SDK:479-497`), scales
and atomically adds with no bound but the float16 range at pack time.

**Implemented: `rtx.sharc.maxDepositLuminance`, default `0` (off).** It bounds the luminance
of a single value an update path writes into a cell (`update:82-92`). It is the only remedy on
the list that costs **zero coverage** — it refuses no lookup, rejects no surface, and creates
no cell that would not otherwise exist — which is what the "do not regress the detail win"
constraint demands. Its price is bias, bounded by the threshold, and it is off until asked for.

**Not implemented, and argued down rather than skipped: a resolution-independent update
budget** (§5.3). It is feasible in one line and it is sound in principle. It is also, exactly,
the cost the user already rejected — and it takes that cost away from the users who chose a
low preset to get performance. The honest form of it is a `rtx.conf` table, given in §5.3.

---

## 1. The update budget is resolution-dependent — derivation

**Source, the dispatch.** The update pass is dispatched at tile granularity:

```cpp
const VkExtent3D& rayDims = nrcEnabled ? nrc.calcRaytracingResolution()
                                       : rtOutput.m_compositeOutputExtent;   // indirect.cpp:771-773
dispatchDims.width  = (rayDims.width  + tileSize - 1) / tileSize;            // indirect.cpp:790
dispatchDims.height = (rayDims.height + tileSize - 1) / tileSize;            // indirect.cpp:791
```

**Source, what `rayDims` is.** `m_raytracingOutput.m_compositeOutputExtent = m_downscaledExtent`
(`src/dxvk/rtx_render/rtx_resources.cpp:1161`) — the **pre-upscale render** resolution, the one
the DLSS preset sets. Not the output resolution.

**Source, the shader agrees.** The launch-side mapping uses the same quantity:
`tileCount = (cb.camera.resolution + tileSize - 1u) / tileSize`, and threads outside it
terminate (`variants:81-86`).

**Conclusion (source, not inference): the number of update paths per frame is
`ceil(w/N) * ceil(h/N)` over the render resolution.** Lower the DLSS preset and the update
budget falls with it, quadratically in the linear scale factor. The cache is a world-space
structure whose fill rate is set by a screen-space budget.

### 1.1 Why fewer paths means brighter fireflies — the divisor

**Source, the resolve blend** (`SDK:920-949`). For one cell, with `k` = deposits this frame
(`SDK:847`, the `data.w` counter incremented once per deposit at `SDK:495`) and `N` =
`accumulationFrames`:

```hlsl
if (accumulatedFrameNum > N) { sampleNumPrev *= N / accumulatedFrameNum; accumulatedFrameNum = N; }  // :925-930
sampleTotalInv  = rcp(sampleNumPrev + sampleNum);                                                     // :947
accumulatedRadiance = prev*(sampleNumPrev*inv) + thisFrame*(sampleNum*inv);                           // :948
```

**Inference, the steady state.** A cell fed at rate `k` for more than `N` frames has
`accumulatedFrameNum` pinned at `N`, so every frame scales the history by `N/(N+1)` and adds
`k`. The fixed point is `S = S*N/(N+1) + k`, i.e. **`S = (N+1)k`**. This frame's `k` samples
therefore carry `1/(N+1)` of the cell, and **one deposit of luminance `L` moves the cell's
stored radiance by `L / ((N+1)*k)`**.

At the shipped `accumulationFrames` 8 that is **`L / (9k)`**.

**Inference.** `k` is proportional to the total path count for a fixed scene and view. So a
firefly's amplitude in a cell is inversely proportional to the update path count — and that
count, by §1, is proportional to render pixels and to `1/N²` in the tile size. **Both halves
of the user's report move the same denominator.** That is why they behave identically.

### 1.2 Why a specular arrival shows it and a diffuse one does not

**Source.** A cache hit contributes `pathState.throughput * cachedRadiance` and ends the path
(`hooks:502-508`, `variants:110-125`). One vertex reads exactly one cell (`hooks:462-468`).

**Inference.** A near-mirror specular arrival maps a small screen neighbourhood onto a single
cell, so the whole of that cell's error lands coherently on a few pixels; a diffuse arrival at
the same surface integrates over a cosine lobe whose vertices scatter across many cells, and
the errors partially cancel. The same cell error is a dot in one case and a slight tint in the
other. NRD's specular history being shorter and narrower than its diffuse history makes this
worse downstream, but that half is inference about the denoiser, not read from this tree.

**Consequence, and it matters for the fix:** `allowSpecularPaths` did not add variance to the
cache. It added a reader that displays variance the cache already had. **Fixing the reader
throws away the feature; fixing the cell keeps it.**

---

## 2. Numbers, at the user's display

Path count is `ceil(w/8) * ceil(h/8)` at the shipped `updateTileSize` 8, over the DLSS render
resolution for a 2560x1440 output. **Arithmetic from §1, not measurement.**

| DLSS preset | render res | update paths/frame | vs DLAA | firefly amplitude vs DLAA | tile needed to match DLAA |
|---|---|---|---|---|---|
| DLAA (1.00) | 2560x1440 | 57,600 | 1.00x | 1.0x | 8 |
| Quality (0.667) | 1707x960 | 25,680 | 2.24x fewer | **2.2x** | 5 |
| Balanced (0.58) | 1485x835 | 19,530 | 2.95x fewer | **3.0x** | 5 |
| Performance (0.50) | 1280x720 | 14,400 | 4.00x fewer | **4.0x** | 4 |
| Ultra Perf (0.333) | 853x480 | 6,420 | 8.97x fewer | **9.0x** | 3 |

Two things fall out, and both match what the user reported:

- **At Ultra Performance a given outlier is worth about nine times more in a cell than at
  DLAA**, in the same scene with the same settings. The user sees fireflies worst at low
  presets because the cache genuinely is nine times noisier there.
- **Halving the tile quadruples the path count**, so `updateTileSize` 4 divides firefly
  amplitude by four — which is why it is the one other thing that helped. The cost is the same
  factor: the update pass traces 4x the paths (`indirect.cpp:790-791`). To fully compensate
  Ultra Performance you would need tile 3, i.e. about **7.1x** the update pass. That is the
  cost the user declined, and declining it was correct.

---

## 3. Why `footprintGate` and `minSampleCount` do not already cover this

The brief asks this explicitly, and the answer is the same for both: **neither is a test of
what a cell contains.**

### 3.1 `minSampleCount` bounds the sample *count*, not the variance of the mean

**Source.** The gate is `accumulatedSampleNum > SHARC_SAMPLE_NUM_THRESHOLD`, which
`sdkglue:41-43` expands to `cb.sharcArgs.minSampleCount` at the SDK's comparison sites.

**Inference.** By §1.1 a cell in steady state carries `accumulatedSampleNum = (N+1)k = 9k`.
Any cell being fed at all is far above 2 within a frame or two. The gate fires on *brand-new*
cells — which is exactly the job it was added for, and it did that job (the user's report:
`minSampleCount = 2` removed the camera-movement glow). It cannot fire on a well-fed cell
holding one hot sample, because such a cell has plenty of samples. **A count gate cannot see an
outlier; a mean of `n` samples containing one outlier is still hot for every `n` the gate would
accept.** Raising it does not help and costs freshly-revealed coverage.

### 3.2 `footprintGate`'s effective threshold is a roughness of about 0.17, not a variance test

**Source, the gate** (`hooks:77-84`): pass when
`segmentLength * sqrt(0.5*alpha^2/(1-alpha^2)) > voxelSize`, `alpha` being the launching
surface's GGX roughness.

**Source, the voxel** (`grid:153-168`): `gridLevel = floor(log2(d))` at `logarithmBase` 2, and
`voxelSize = 2^gridLevel / sceneScale`, `d` being the vertex's distance from the camera. So
`voxelSize` is within a factor of two of `d / gridScale`. The tuning guide states the same fact
as "a cell spans 0.57 to 1.15 degrees of arc anywhere in the scene".

**Inference.** For the reflections that matter here — a surface reflecting geometry roughly as
far from the camera as the reflector itself, so the segment length is comparable to the hit's
camera distance — `segmentLength / voxelSize` lands between `gridScale` and `2*gridScale`. The
distance cancels. The gate then reduces to

```
sqrt(0.5*alpha^2/(1-alpha^2)) > 1/gridScale   ->   alpha > 0.0283   ->   perceptual roughness > 0.168
```

at `gridScale` 50, and about 0.24 in the strict half of the voxel band. (This is the same
cancellation the adaptive doc found for the primary deposit, which gave `alpha > 0.028`.)

**So everything above a launching perceptual roughness of roughly 0.17-0.24 passes the gate** —
which is most reflective game materials short of polished metal. The gate is doing what it was
designed to do: keep an isotropic cache out of a lobe narrower than a cell. It says nothing
about whether the cell it lets through is trustworthy. **It is a directionality gate, not a
variance gate**, and a third gate of the same family would not be one either.

**Judgement.** This is why the fix is not another gate. Gates decide *whether* to read; the
defect is in *what is there to read*.

---

## 4. What the SDK offers — audited, and it offers nothing

Every tweakable in `SDK:69-135`, and why none is outlier handling:

| Toggle | Set to | Is it outlier handling? |
|---|---|---|
| `SHARC_ENABLE_RESPONSIVE_LIGHTING` | 0 | **No — it would make this worse.** It keeps a second cell on a *shorter* accumulation window (`responsiveFrameNum`, `SDK:924`), and a shorter window is a larger `1/(N+1)` weight for one frame's samples. |
| `SHARC_ENABLE_CACHE_RESAMPLING` | 0, deliberately (`sdkglue:30-32`) | **No — worse.** It substitutes a converged cell's own value into an update path's estimate (`SDK:614-637`), which would propagate a hot cell into its neighbours. The fork's comment — "keep the initial estimator finite and free of cache feedback" — already rules it out, and the brief excludes revisiting it. |
| `SHARC_ENABLE_FADE_ACCELERATION` | 0 | **No.** It snaps the history down after 32 consecutive *fading* frames (`SDK:931-944`). A firefly is one bright frame, the opposite case. |
| `SHARC_ENABLE_SH_ENCODING` | 0 | No — a storage format. |
| `SHARC_MATERIAL_DEMODULATION` | 0 | No — detail preservation. |
| `SHARC_SEPARATE_EMISSIVE` | 1 | No — it is why emissives are excluded at the hook. |
| `SHARC_BLEND_ADJACENT_LEVELS` | 1 | No — camera-motion reprojection (`SDK:951-975`). |
| `SAMPLE_NUM_THRESHOLD`, `STALE_FRAME_NUM_MIN`, `PROPAGATION_DEPTH`, `LINEAR_PROBE_WINDOW_SIZE`, `GRID_LEVEL_BIAS`, `USE_FP16`, `RESAMPLING_DEPTH_MIN`, `RESPONSIVE_ENTRY_PROBE_RANGE` | various | No. |

**Source.** The only `clamp` calls on radiance anywhere in the SDK are the float16 range clamps
at pack time, `+/-65504` (`SDK:425-427`), and the frame-count clamps (`SDK:854`, `:924`). The
deposit path is unbounded:

```hlsl
uint3 scaledRadiance = uint3(sampleValue * sampleWeight * sharcParameters.radianceScale);   // SDK:494
if (scaledRadiance.x != 0) InterlockedAdd(...);                                             // SDK:495-497
```

**Conclusion.** There is no disabled feature to switch on. Bounding a deposit is a gap in the
SDK, and filling it is fork work.

---

## 5. Ranked options, each with its coverage cost stated

Ranked on: does it reduce per-cell variance, what does it cost in coverage, what does it cost
in frame time. **All impacts unmeasured.**

### 5.1 Rank 1 — clamp what goes into a cell (**implemented**)

Bound the luminance of a single deposit at the one place deposits happen on the shipping
backend.

- **Coverage cost: none, provably.** It evaluates no eligibility term, refuses no lookup,
  rejects no surface, and creates or destroys no cell. Every cache hit that happened before
  still happens. This is the only option on the list of which that is true, and it is why it
  is the one that fits the "do not regress the detail win" constraint.
- **Frame-time cost: negligible.** One `dot`, one compare, one multiply
  (`fireflyFiltering`, `src/dxvk/shaders/rtx/utility/common.slangh:37-47`) per cached vertex
  per update path, on a pass dispatched at **1/64 of the render resolution** at tile 8
  (`indirect.cpp:790-791`). The threshold is uniform, so with the option off the branch is
  uniform and not taken. Unmeasured, but it is a handful of ALU on the smallest pass in the
  feature.
- **Correctness cost: bias.** A cell whose true radiance exceeds the threshold is stored dark,
  and energy is lost there. The loss is bounded by the threshold and local to that cell — the
  present behaviour is unbounded. Remix already accepts this trade globally at
  `rtx.fireflyFilteringLuminanceThreshold` 1000 (`rtx_options.h:378`, applied at
  `integrator_helpers.slangh:30`).
- **Why a tighter threshold than the global one is defensible (judgement).** The global filter
  guards a per-pixel estimate that a denoiser then averages over a wide spatial and temporal
  footprint. A cell is a mean of `k` samples — often a handful — read back sharply by a
  specular lobe with no such averaging in front of it. The cache's estimator has a far smaller
  `n` and a far smaller blur, so it warrants a tighter bound.

### 5.2 Rank 2 — raise `accumulationFrames`

- **Coverage cost: none; it slightly increases coverage.** Per the tuning guide's coupling, a
  cell fed every `k` frames is readable only when `k < accumulationFrames`, so raising it makes
  more sparse cells readable.
- **Frame-time cost: zero.** It is not in the clear condition (`host:103-108`), so it can be
  dragged live.
- **Effect:** by §1.1 the outlier weight is `1/((N+1)k)`, so 8 -> 16 nearly halves it and
  8 -> 24 divides it by about 2.8.
- **Why not the recommendation:** the cost is response lag in frames, and the user has
  repeatedly said responsiveness matters. **It is still the first thing to try, because it is
  free and needs no new build.**

### 5.3 Rank 3 — a resolution-independent update budget

**Feasible?** Yes, trivially. `updateTileSize` is one of only three options absent from the
clear condition (`host:103-108`, `:154`), and the dispatch is computed on the CPU from
`rayDims` (`indirect.cpp:788-793`), so deriving an effective tile size from
`sqrt(renderPixels / referencePixels)` is a few lines with no cache clear and no shader change.

**Sound?** In principle yes, and this is the strongest argument for it: a world-space cache's
fill requirement is set by the scene and the view frustum, not by how many pixels are being
shaded. Tying it to render resolution is an accident of the dispatch shape, not a design.

**Declined, on two grounds (judgement).**

1. **It is precisely the cost the user rejected.** Restoring DLAA's path count at Ultra
   Performance means tile 3, about 7.1x the update pass (§2). The user tried the manual form
   of this and called it unacceptable.
2. **It takes frame time away from the people who asked for frame time.** As the preset drops,
   the rest of the frame gets cheaper while this pass stays fixed, so the cache's share of the
   frame *rises* — worst exactly where the user chose a low preset to gain performance. A
   quality preset that gets relatively more expensive the lower you set it is a surprising
   thing to ship on by default.

**The honest form of it needs no code.** Set the tile per preset in `rtx.conf`, and spend as
much of §2's right-hand column as the frame budget allows:

```ini
# pick the row for the DLSS preset you actually play at
rtx.sharc.updateTileSize = 8   # DLAA
rtx.sharc.updateTileSize = 5   # Quality / Balanced
rtx.sharc.updateTileSize = 4   # Performance
rtx.sharc.updateTileSize = 3   # Ultra Performance   (about 7x the update pass - probably too much)
```

The clamp in §5.1 exists so that this table does not have to be paid in full.

### 5.4 Rank 4 — clamp what comes out (read-side)

- **Coverage cost: none** (nothing is refused, values are only dimmed).
- **Frame-time cost: higher than the insert clamp**, which is the part that settles it. The
  query pass is dispatched at the full render resolution (`indirect.cpp:799`, `:868-872`)
  against the update pass's 1/64 of it, and a query vertex reads a cell on every bounce. Same
  arithmetic per invocation, roughly sixty-four times as many invocations.
- **Effect: worse.** The cell stays hot, so every pixel that maps to it keeps reading it and
  the artifact becomes a *stable* capped bright patch rather than being removed — arguably more
  objectionable than a transient dot. And it dims legitimately bright cached radiance for
  diffuse readers, who do not have the problem.
- Worse on all three axes than clamping at insert. Not implemented.

### 5.5 Rank 5 — a specular-specific `minSampleCount` (**declined**)

The brief floats this as the targeted version of "more update paths". It is not, on two counts.

- **It does not address the defect.** By §3.1, a count floor bounds `n`, not the variance of a
  mean of `n`. More update paths reduce variance in every cell; a higher specular floor merely
  refuses reads from cells with few samples, which is a different and largely disjoint set from
  the cells holding an outlier.
- **Its coverage cost lands exactly on the thing being protected.** Every specular read from a
  cell below the new floor is refused, the path continues, and the multi-bounce detail the user
  values disappears — most at low presets, where cells are thinnest, which is where the user
  says the feature is at its best. **It buys quiet by deleting the feature.**
- The infrastructure for it exists (`pathState.skyGatherEligible` already separates the two
  arrival classes at `hooks:412-414`), so this is a decision not to build it, not an inability.

---

## 6. What was implemented

**`rtx.sharc.maxDepositLuminance`, float, default `0.0` (off).** Clamps the luminance of a
single value an update path deposits into a cell, hue-preserving.

| File | Change |
|---|---|
| `update:70-92` | The clamp, in `sharcFlushVertices`. |
| `args:40-48` | `float maxDepositLuminance` on `SharcArgs`; `static_assert(sizeof(SharcArgs) == 84)`. |
| `opt:49` | The `RTX_OPTION`. |
| `host:158-163` | Plumbing, with the deferred-backend condition. |
| `host:456-470` | The ImGui control and its tooltip. |
| `RtxOptions.md` | Regenerated by `meson test test_documentation`. |

**Where, and why there.** `sharcFlushVertices` is the single point at which a deposit enters a
cell on the deferred backend, which is the shipping one (`rtx.sharc.deferredUpdates` default
true). It is called at the end of every update path (`variants:1547-1549`) and once more before
a sky retry (`hooks:366-372`):

```slang
radiance = state.radiance[i] + state.weights[i] * radiance;
const vec3 deposit = fireflyFiltering(radiance, cb.sharcArgs.maxDepositLuminance);
SharcAddVoxelData(p, state.indices[i], deposit, vec3(1.0f), ...);     // update:88-90
```

Three deliberate choices:

- **The deposit is clamped; the running propagated value is not.** Each cell in the chain is
  bounded by the same rule independently, and path propagation is left exactly as it was. The
  alternative — clamping the carried value — would compound down the chain and change the
  estimator, for no gain.
- **It bounds the product, not an input.** `radiance` at this point is the outgoing reflected
  radiance for that cell, weights already applied, which is precisely the quantity the cell
  will store. Clamping a radiance *input* instead would leave a large path weight free to
  reconstitute the outlier.
- **The non-deferred backend is deliberately untouched**, and `host:162` forces the threshold
  to zero when `deferredUpdates` is off. On that backend the deposit happens inside the SDK
  (`SDK:545-576`, `:645-683`), where only the inputs are reachable from fork code, so any clamp
  there would be an approximation of this one. `deferredUpdates` exists to A/B the two backends
  against each other; making them differ approximately under a third option would spoil the
  comparison the toggle is for.

**It does not clear the cache**, unlike most quality options (`host:158-163`, absent from the
clear condition at `:103-108`). The threshold bounds values deposited from now on; cells
already holding an outlier wash it out within `accumulationFrames` frames on their own (§1.1),
so clearing 1M-4M cells to make the change land a tenth of a second sooner would cost far more
than it bought. **This makes it one of the few options that can be dragged live**, which is
exactly what a threshold you have to find by eye needs.

### 6.1 Considered and not built

- **A stats counter for how often the clamp fires.** It would make the threshold findable from
  the panel instead of by eye. The `sharcCount` infrastructure is query-pass only — the stats
  blobs are all `*sharc_query*stats*` (`indirect.cpp:70-75`, `:151-181`) — so an update-pass
  counter means a new family of shader permutations. Against a feature whose whole measured
  footprint is about 0.1 ms, that is not a proportionate build cost. Debug view **583 (Cached
  Radiance)** is the substitute; see §7.
- **A relative clamp against the cell's own accumulated mean.** Self-scaling, so no magic
  number — but it needs a `resolvedBuffer` read per deposit, a dependent random access into a
  160 MiB buffer at `capacityLog2` 22. Against the same 0.1 ms budget, an absolute threshold
  costing three ALU wins. It would also introduce feedback from a cell's history into what may
  enter it, slowing legitimate brightening — a responsiveness cost the user has said they care
  about.
- **A third eligibility gate.** Ruled out by §3: the defect is cell contents, not the decision
  to read.

---

## 7. What the user should test

Build is deployed to both installs (§8). **Every expectation below is unmeasured.**

**First, for free, before touching the new option.** `rtx.sharc.accumulationFrames = 16`. No
rebuild, no cache clear, drag it live. By §1.1 it should roughly halve firefly amplitude at the
cost of doubling the cache's response time to lighting changes. If the lag is acceptable, this
may be the whole fix and the new option can stay off.

**Then the new option, and start at a low DLSS preset — that is where §2 says the effect is
largest and where you see the problem.**

1. Set **Ultra Performance or Performance**, in a scene with the reflective materials that
   show it.
2. Open debug view **583 (Cached Radiance)** and look at the magnitudes of the cells you
   legitimately want bright. This gives the floor for the threshold. Note that 583 shows the
   cell *mean*, whereas the option bounds a single deposit, which is generally larger — so
   **set the threshold comfortably above the brightest legitimate cell, not at it.** A few
   times that value should catch only true outliers. (Inference, not measured.)
3. Set `rtx.sharc.maxDepositLuminance` to that number and drag it down. The panel control is
   **SHARC -> Max deposit luminance**. It takes effect within `accumulationFrames` frames
   without clearing the cache, so the response is visible in about a fifth of a second and you
   can sweep it live.
4. **Too far** looks like: bright cached areas going flat or dim — a lit doorway, a sunlit
   patch, a strong bounce — before the fireflies are gone. That is the bias in §5.1 becoming
   visible. Back off.
5. **The thing to check hardest, because it is the constraint this work was given:** that the
   detail advantage over the other samplers is still there. Compare against
   `rtx.integrateIndirectMode = 0` at the same preset. By construction the clamp refuses no
   lookup, so cache *coverage* cannot have changed; if detail is gone, the threshold is too low
   and is dimming real light, not the mechanism failing.

**Then confirm the resolution story directly**, which would turn §1 and §2 from arithmetic into
measurement and is the single most useful reading anyone could take here: with the new option
**off**, compare firefly severity at DLAA against Ultra Performance in the same spot. §2
predicts about 9x. Then set `updateTileSize = 3` at Ultra Performance and confirm the fireflies
go — and note what it costs on `GPU ms: update` with `rtx.sharc.measureGpuTime` on. That last
number has never been captured and would size every option in §5.

**What is not worth testing:** `minRoughnessSpecular`. With `footprintGate` on it is unused
(`hooks:59-60`, `opt:34`).

---

## 8. Build, validation, deployment

Release build, two synchronous `meson compile -C _Comp64Release` passes; the second relinked
nothing, confirming convergence. `d3d9.dll` **280,213,504 bytes**, 2026-09-16 21:28:18 (from
280,178,176). The ABI change is verified by construction: `static_assert(sizeof(SharcArgs) == 84)`
compiles on the C++ side and the shaders compile against `cb.sharcArgs.maxDepositLuminance`, so
both sides of the scalar-layout boundary agree or neither would have built.

Validators, all on the final artefacts:

- `validate_sharc_integration.py` — **61 PASS, 0 FAIL** (52 declared SHARC stages, reservoir
  guard, legacy TraceRay retention).
- `validate_sharc_resources.py` — **PASS**, 52 compiled stages, 14 foreign descriptors absent,
  DLL embedding verified.
- `validate_sharc_deferred.py` — 6 tests OK. `validate_sharc_estimator.py` — 5 tests OK.
- `meson test` — 16/17 OK. `test_documentation` passes after regenerating `RtxOptions.md`.
  **`test_graph_documentation` fails and is pre-existing and unrelated**: missing golden docs
  for the `GameValueReadBool` / `GameValueReadNumber` USD graph components. Not touched.

Deployed to **both** installs, neither game running, backup suffix
**`backup-pre-deposit-clamp-20260916-213514`**:

- `C:/Users/sparkles/Projects/Games/Fallout New Vegas/.trex` — `d3d9.dll` + `d3d9.pdb`, both
  `cmp`-identical to the build.
- `D:/SteamLibrary/steamapps/common/PortalRTX/bin/.trex` — `d3d9.dll`, `cmp`-identical. This
  also clears the outstanding copy noted at the end of `SHARC-implementation-status.md`, where
  Portal RTX was left behind because it was running.

**Nothing measured in game.**
