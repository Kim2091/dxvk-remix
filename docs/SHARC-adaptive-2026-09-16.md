# Can SHARC tune itself to the scene? (2026-09-16)

Worktree `wsn3g`, branch `revised-9-10`, starting from HEAD `c66e2d29b`, tree clean. Statements
are labelled **source** (read from the tree, `file:line` after this change unless marked HEAD),
**inference** (a conclusion from source facts plus stated assumptions) or **judgement** (a choice
the evidence narrows but does not determine). **Every in-game impact below is unmeasured** except
the four figures explicitly attributed to the user. Neither game was run for this work; Portal RTX
was in fact running throughout it, which is why it did not get a deployment (§10).

Abbreviations: `host` = `src/dxvk/rtx_render/rtx_sharc.cpp`, `opt` =
`src/dxvk/rtx_render/rtx_sharc.h`, `hooks` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_integrator_hooks.slangh`, `update` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_update.slangh`, `bindings` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_bindings.slangh`, `args` =
`src/dxvk/shaders/rtx/pass/sharc/sharc_args.h`, `SDK` =
`src/dxvk/shaders/rtx/external/sharc/SharcCommon.h`, `grid` =
`src/dxvk/shaders/rtx/external/sharc/HashGridCommon.h`, `indirect.cpp` =
`src/dxvk/rtx_render/rtx_pathtracer_integrate_indirect.cpp`, `variants` =
`src/dxvk/shaders/rtx/pass/integrate/integrate_indirect.slang`.

The measurements in play, all the user's:

| | value | where |
|---|---|---|
| Portal RTX, enclosed | hit 95%, eligible 74%, terminates 78% | interiors |
| FNV open desert | hit 88%, terminates 23.3%, sky exits 65.9% | exteriors |
| whole cache, net frame time | about 0.1 ms | Portal RTX |
| `rtx.sharc.updatePrimaryVertex` | about 0.1 ms | **scene not recorded — see §6.1** |

---

## Verdict

**Do not build an adaptive controller.** Three findings kill it independently, and any one of
them would be enough.

1. **Almost nothing can be adapted.** Exactly three of the twenty-two `rtx.sharc` options can change
   without clearing the cache: `accumulationFrames`, `staleFrames` and `updateTileSize` (§2).
   Every other option — including both the ones this work was commissioned about,
   `updatePrimaryVertex` and `updateSkyRetries` — forces a full clear of all three buffers
   (`host:103-137`). A controller that retunes them empties the cache on every decision, and a
   cleared cache is worth less than a badly tuned one for as long as it takes to refill.
2. **The signal costs more than the thing it would control, and arrives too late to use.** The
   counters live in a separate shader permutation gated behind two options that together measure
   about 1 ms (user), are read back through an 8-slot GPU-event rotation that silently drops
   frames, and are published only as a sum over 120 collected frames — mean age about one second,
   fully replaced about every two (§3). Paying 1 ms to steer a feature whose whole measured
   footprint is 0.1 ms is upside-down before any control theory is applied.
3. **The two knobs worth moving both change their own measurement**, with a settling time longer
   than the sampling period, which is the standard recipe for an oscillator and cannot be damped
   without making the controller slower than a human with the panel open (§4).

**The good news is that most of this is already handled, and handled better than a controller
would handle it.** `updateSkyRetries`, `footprintGate`, `minSampleCount`, `staleFrames` and the
eviction path are all *per-sample* adaptations that cost nothing where they are not needed, and
`gridScale` is scene-invariant by construction rather than by measurement (§1). That is the
correct pattern in this feature and the presets landed in `ed39787a9` cover the rest.

**On the sub-question — a cheap always-on guard for `updatePrimaryVertex` — the answer is also
no, for now, and the reasoning is worth keeping** because it rules the idea out on more than
taste: the natural derivation gives a threshold *below* the existing `minRoughness` floor, so by
the project's own gate criterion no extra guard is called for; the artefact it would prevent has
not been observed by anyone; and any threshold I could pick would be a judgement number defended
by nothing (§7.3–7.5). What *is* justified, and is implemented here, is one preset change forced
by the new 0.1 ms measurement (§8).

---

## 1. What is already self-limiting, and should stay a default

"Self-limiting" here means: the mechanism costs approximately nothing in scenes that do not need
it, without any detection step. These are the ones that already have that property, and they are
the model the rest of the feature should be judged against.

**`updateSkyRetries` — fully self-limiting. Source.** The retry hook returns immediately unless
the path's first bounce missed: `retriesLeft == 0 || pathState.continuePath || pathState.bounceIteration != 0`
(`hooks:352-354`), and only a first-bounce miss leaves `bounceIteration` at 0 with `continuePath`
false. Indoors the sky share is negligible (user, Portal: "sky negligible"), so the branch is not
taken and no extra ray is traced. **Inference:** this option needs no scene detection and never
will. Its default of 1 is correct everywhere and its cost is *literally* proportional to the
problem it solves.

**`footprintGate` — self-limiting, and the best example in the feature of adaptation done right.
Source.** `sharcFootprintValid` (`hooks:77-84`) tests the actual spread of the lobe that launched
the segment against the actual voxel size at the arrival point, per lookup. Diffuse arrivals skip
it entirely (`hooks:464`, `pathState.skyGatherEligible` short-circuits). **Inference:** in a
matte scene it never refuses anything; in a glossy scene it refuses exactly the lookups that would
glow. It replaced `minRoughnessSpecular`, a fixed threshold standing in for the same quantity, and
the user reported "an extreme improvement" from the swap
(`docs/SHARC-implementation-status.md`, 2026-09-16 entry). **That is the shape every future
"adaptive" idea in this feature should take: measure the quantity per sample in the shader, not
the scene per second on the host.**

**`minSampleCount` — self-limiting.** A floor that only ever binds on under-fed cells
(`SDK:741`). In a scene where cells are fed every frame it is never reached and costs one compare.

**`staleFrames` and the resolve's eviction — self-limiting.** An entry is only cleared once it has
gone `staleFrames` frames without a sample (`SDK:852-868`), and a slot holding no key returns after
one buffer load (`SDK:838-840`). Idle cells cost a load; live cells cost the resolve body.

**`gridScale` — scene-invariant by construction, which is better than adaptive.** A cell's edge is
the vertex's camera distance rounded down to a power of two, divided by the value (`grid:153-168`),
so the cell is an *angle* — 0.57° to 1.15° at 50 — in any game at any world scale. **Inference:**
there is nothing to calibrate per title, which is why one value has suited both games and why
`rtx.sceneScale` correctly does not enter. A controller that measured scene extent and retuned
`gridScale` would be re-deriving an invariance the mechanism already has.

**Not self-limiting, and worth naming as such:**

- **`capacityLog2`.** The resolve dispatches one thread per slot every frame (`host:183`,
  `sharc_resolve.comp.slang:8-9`) whether or not the slots hold anything. It is the one option
  that costs exactly as much in an empty scene as a full one. (The early-out at `SDK:838-840`
  makes an empty slot cheap, not free.)
- **`updateTileSize` and `updateBounces`.** Path count is fixed by resolution and tile; only
  segment length varies with content.
- **`updatePrimaryVertex`.** One deposit attempt per update path, unconditionally
  (`hooks:295-338`), with no test that could make it cheaper in a scene that does not need it.
  This is the option the rest of the document is about.

**Conclusion for question 1: the range of scenes is already covered by defaults, and by
mechanisms that adapt per sample rather than per scene.** The presets are the right place for the
remaining coarse choice, and after §8 they carry it correctly.

---

## 2. The hard constraint: which options can change without clearing the cache

This is the finding that decides the whole question, so it is stated exhaustively.

**Source, `host:103-108`.** The clear condition is

```
bool clear = resetHistory || m_resetRequested || compatibilityFlags != m_compatibilityFlags
  || !wasActive || m_lastFrame + 1 != frame
  || m_args.gridScale != scale || m_args.minRoughness != roughness
  || m_args.maxEmissiveLuminance != emissiveLimit
  || m_args.minRoughnessSpecular != roughnessSpecular
  || m_args.minSampleCount != sampleFloor
  || m_args.updateBounces != updateBounceLimit;
```

and `compatibilityFlags` (`host:79-86`) packs thirteen more: `wboitEnabled`, active ray portals,
`updatePrimaryVertex` (bit 8192), `updateSkyRetries` (bits 14+), opacity micromaps,
`allowRayPortals`, `deferredUpdates`, `queryRayGeneration`, `updateRayGeneration`,
`allowSpecularPaths`, `queryTraceRay`, `footprintGate`, and SER. A capacity change reallocates
the three buffers and sets `clear` too (`host:109-123`). A clear zeroes the hash, accumulation and
resolved buffers (`host:132-137`) and resets `m_cacheAge` to 0 (`host:138`).

**Source, what is written unconditionally and therefore takes effect live:** `host:152-154` set
`accumulationFrames`, `staleFrames` and `updateTileSize` into `m_args` with no corresponding
comparison in the clear condition and no bit in `compatibilityFlags`.

**So the complete list is:**

| Can change with no cache reset | Forces a full clear |
|---|---|
| `accumulationFrames` | `updatePrimaryVertex`, `updateSkyRetries`, `allowSpecularPaths`, `footprintGate`, `minRoughness`, `minRoughnessSpecular`, `maxEmissiveLuminance`, `minSampleCount`, `updateBounces`, `gridScale`, `capacityLog2`, `allowRayPortals`, and the four backend toggles |
| `staleFrames` | |
| `updateTileSize` | |

**Inference, and this is the answer the brief asked for plainly: the useful ones all force a
reset.** `updatePrimaryVertex` and `updateSkyRetries` — the two options this work exists to reason
about — are both in the right-hand column. An adaptive controller that toggled either would clear
1M–4M cells on every decision.

**What a clear actually costs.** The cache is unreadable until cells clear `minSampleCount`
(2 by default), which for a cell fed every frame takes 3 frames and for a cell fed every *k*
frames takes until `1 + accumulationFrames/k > 2`, i.e. never if `k ≥ accumulationFrames`
(`docs/SHARC-open-world-diagnosis-2026-09-16.md` §3, re-derived from `SDK:925-949` and `:741`).
**Inference:** indoors, a clear costs a handful of frames; outdoors, the sparse tail of the cache
takes tens of frames and some of it never comes back. And the panel cannot report what happened
for another ~120 frames (§3). So the minimum honest settle time for one control action is on the
order of **two seconds**, in a game where a door transition changes the regime instantly.

**Is any of those resets avoidable?** One is, on the source's own argument, and it is worth
recording even though it does not rescue the idea. `updateSkyRetries` changes *which* cells
receive samples, not what they hold — "what a cell stores does not depend on how the ray reached
it" (`hooks:341-343`, and the correctness argument in `docs/SHARC-sky-budget-2026-09-16.md` §4).
**Inference:** its bits in `compatibilityFlags` are conservative rather than required, and moving
it out would let it change live. `updatePrimaryVertex` is *not* in that position: it changes the
value deposited into a cell, so a mid-flight change leaves a cell holding a mixture of two
estimators for `accumulationFrames` frames. **Judgement:** do not make the `updateSkyRetries`
change. Its only use would be to feed a controller that §3 and §4 rule out anyway, and loosening a
cache-clear condition on an inference is how correctness bugs get in.

---

## 3. What signal would drive it

The counters exist and measure the right things: sky-exit share (slot 10, `integrator_indirect.slangh:554`),
eligibility (4) against resolved surfaces (2), hits (7) against misses (6), and the
no-cell/below-floor miss split (23/24, `hooks:475-483`). The problem is entirely in how they are
produced and delivered.

**They are a separate shader permutation. Source.** `SHARC_QUERY_STATS` is a variant define
(`variants:88-106`), and the host picks a different pipeline when stats are on
(`indirect.cpp:354-393`, `:611`). There are **60** `*sharc*stats*` blobs in the shader output
directory and **40** `_stats` includes in `indirect.cpp`. **Inference:** "always-on stats" means
either compiling the stats code into the production variants — paying the atomics and the extra
probe in every frame forever — or keeping the split and dispatching the stats variant always,
which is the same thing. There is no third option where the counters are free.

**What the counting costs. Source.** `sharcCount` does an `InterlockedAdd` on a 1-in-64 rotating
pixel sample (`bindings:17-23`), called up to eight times per resolved vertex in the worst path
(`hooks:418-423`, `:470-497`). The miss split additionally runs a **second** `HashGridFind` on
every miss (`hooks:479-481`) purely to tell "no cell" from "below floor". **Measured** (user, FNV):
the query pass read 8.82 ms with statistics off and 9.78 ms with them on — about 1 ms.

**The miss split's extra probe is avoidable, for the record.** `SharcGetCachedRadianceFromHash`
already performs the find and already knows both answers — it returns false at `SDK:737-738` when
the key is absent and at `SDK:765` when the sample count is below the floor — so an out-parameter
would give the split for nothing. That is an edit to the vendored SDK file, which this fork has so
far only parameterised by `#define` (`sharc_sdk.slangh:42-44`), and it would buy a cheaper
*diagnostic*, not a cheaper controller. Noted, not recommended.

**The delivery is slower than the counting. Source, `host:234-287`.**

- Stats are gated on `measureGpuTime() && collectQueryStats() && m_active` (`host:236`), so the
  ~1 ms timing cost is mandatory on top of the ~1 ms stats cost. `measureGpuTime` also inserts
  timestamp boundaries that break GPU overlap independently of what they measure.
- Readback rotates over 8 slots keyed on frame id (`host:252`). If the slot's event has not
  signalled, the function **returns without binding the stats buffer at all** (`host:254-257`), so
  that frame contributes no counts. Frames are silently dropped under load.
- A frame's counts are added to an accumulator and the published window is only replaced when 120
  *collected* frames have accumulated (`host:263-268`, `kStatsWindowFrames = 120` at `opt:74`).

**Inference, staleness.** The published figure is a sum over the last 120 collected frames. Its
mean sample age is ~60 frames and its oldest contribution is ≥128 frames including the event
rotation — so at 60 fps a controller reading it would be acting on a signal whose **mean age is
about one second** and which is **fully replaced only every two seconds**. Shortening the window
is possible but the counters are a 1-in-64 pixel sample and the comment at `opt:71-73` records why
the window exists: per-frame counts flicker unreadably and any frame without a lookup reads as 0%.
A controller on a 10-frame window would be steering on noise.

**Is a cheap always-on subset feasible?** The cheapest useful counter is the sky-exit share: one
atomic at one site (`integrator_indirect.slangh:553-555`), no extra probe. **Inference:** that
alone would cost far less than 1 ms — but it is a *query* statistic, it is the one number that
tells you what the cache **cannot** do rather than what to change, and it cannot distinguish the
two miss buckets, which is the only place the panel names a remedy. To drive anything you need the
split, and the split is the expensive counter. **So: no, there is no cheap always-on subset that
carries the information a controller needs.**

---

## 4. Oscillation

Take the controller at its most plausible: the three live options (§2), driven by the miss split.

The loop is direct and unavoidable, because each lever moves its own sensor:

- **`accumulationFrames` ↑** raises `N∞ = 1 + A/k` for every sparsely-fed cell (`SDK:925-949`),
  which converts "below sample floor" misses into hits. The controller sees success, and it has
  also just raised the indirect channel's time constant — at 32 frames, about half a second of lag
  at 60 fps. Nothing in the miss split can see lag, so the controller has no term that pushes back
  and will ratchet to the 64-frame clamp (`host:152`) in any scene with sparse cells. The failure
  is not oscillation; it is a one-way ratchet into the defect the user cares most about
  ("snappiness matters", `docs/SHARC-presets-2026-09-16.md` §5).
- **`updateTileSize` ↓** quadruples samples per cell and quadruples the update pass's traced work.
  It also changes *which* pixels are sampled, so it perturbs the statistic directly. Settling time
  is at least `accumulationFrames` frames for cells to re-equilibrate, plus 120 frames before the
  panel's window reflects it — against a sampling period of the same 120 frames. **A loop whose
  actuator settles no faster than its sensor samples is the textbook oscillator**, and the usual
  damping term (make the controller slower than the plant) here means acting less than once every
  few seconds, which is slower than a human dragging the slider.
- **`staleFrames` ↑** moves cells from "no cell" into "below sample floor" without making them
  readable unless `accumulationFrames` rises with it
  (`docs/SHARC-open-world-diagnosis-2026-09-16.md` §7). So the two buckets the controller reads
  **trade against each other under its own actions**. A naive controller watching "no cell" shrink
  would score that as progress while the image got no better.

**Could it be damped?** Yes, in the only way that ever works here: heavy hysteresis plus a long
minimum dwell. **Inference:** the dwell has to exceed the 120-frame window plus the settling time,
so a few seconds per action, three levers, and the scene may have changed twice in that time. What
you get is a device that reaches roughly the shipped defaults after ten seconds of standing still,
having spent about 1 ms a frame the whole time to find out. **That is strictly worse than shipping
the defaults.**

---

## 5. The cheaper alternatives, ranked

1. **Better defaults, which is where this already is.** The nine default moves in `ed39787a9`
   (`docs/SHARC-presets-2026-09-16.md` §9) put "set nothing in `rtx.conf`" at the user's tested
   configuration. That covers the modder who does not want to know anything, which is the stated
   goal. **This is the answer, and it is already shipped.**
2. **Per-sample mechanisms instead of per-scene ones.** `footprintGate` is the proof that this
   works: it removed a fixed threshold that had to be tuned per title and replaced it with a test
   of the actual quantity, per lookup, costing a few instructions. Any future "make it adapt"
   request should be first checked against "can this be a shader-side test of the real quantity?"
   before anything on the host is considered.
3. **Per-title config shipped with a mod.** Already how both installs work. Nothing to build.
4. **Make the panel name the remedy.** The panel prints the miss split (`host:348-349`) but names
   no fix; the tuning guide names the fix but the modder has to find the guide. One conditional
   line under the split — "below floor dominant: raise Accumulation frames" / "no cell dominant:
   try Capacity exponent 22 for one session" — would close that gap for the price of one
   `ImGui::Text`. **Recommended as the cheapest next step; deliberately not built here**, because
   it is always-on UI text in an already dense panel and the user gates panel changes on looking
   at them, which I cannot do.
5. **A one-off derivation at scene load.** Rejected on availability rather than principle: there
   is no cheap host-side "is this outdoors" signal. The sky probe is a GPU image
   (`rtx_context.h:278-279`), so reading it back has the same latency as the stats path, and scene
   extent does not separate a desert from a warehouse. There is nothing to derive *from*.
6. **A controller.** §2–§4.

---

## 6. `updatePrimaryVertex`: what its 0.1 ms is actually made of

The new measurement changes the shape of the sub-question, so it is worth establishing where the
time goes before asking how to cut it.

**Source, the per-path work** (`hooks:295-338`): three early-out tests; a decode of the primary's
material from `SharedMaterialData0/1` and `SharedSubsurfaceData` through `sharcPrimaryRejectReason`
(`hooks:263-280`); a read of `PrimaryWorldPositionWorldTriangleNormal`; two reads of the direct
lobe radiance textures (`hooks:316`); then one `HashGridInsertEntry` and one seed or
`SharcUpdateHit` (`hooks:327-337`). **Inference:** that is ~6 texture loads and one 64-bit atomic
insert per update path, on `ceil(resolution/tile)²` paths — about 57k at 2560×1440 and tile 8
before DLSS scaling. Constant per path, exactly as the brief says.

**But the per-path work is probably not all of the 0.1 ms, and the rest is *not* constant.**
**Inference, two second-order terms:**

- **Residency.** The deposit makes every camera-visible eligible surface a cell that is fed every
  frame, so cells that used to be evicted at `staleFrames` are now permanently live. A live slot
  runs the whole resolve body; a dead one returns after one load (`SDK:838-840`). The resolve is
  4M threads at `capacityLog2` 22, so converting a large set from "returns immediately" to "runs
  the body" is real time that shows on `GPU ms: resolve`, not on `GPU ms: update`.
- **Occupancy.** Higher load lengthens the linear probe for *every* insert and *every* query
  (`grid:279-297`, 16-slot buckets).

**Inference, and this matters for the adaptation question:** both second-order terms scale with
how many *new* cells the deposit makes resident. Indoors, most camera-visible surfaces are already
cached because bounce rays reach them, so the deposit mostly adds samples to existing cells and
adds few new ones. Outdoors it makes a large ground set resident that was previously being evicted
(`docs/SHARC-presets-2026-09-16.md` §3 estimates the outdoor live set at 1.5–3×10⁵). **So the cost
is probably larger where the benefit is larger, not flat against it** — which weakens the "flat
cost, scene-dependent benefit" profile that would have justified adapting it. Unmeasured, and the
panel can separate the two: `GPU ms: update` carries the per-path term, `GPU ms: resolve` carries
the residency term.

### 6.1 Is the benefit really scene-dependent?

**No — or at least, not only.** The option has two distinct mechanisms and only one of them is
about the sky.

1. **Recovering sky-bound paths.** Outdoor-only. 65.9% of FNV update paths insert nothing without
   it (user); indoors the sky share is negligible, so this mechanism contributes nothing in Portal.
2. **Feeding camera-visible surfaces densely, every frame.** Scene-independent. A cell 16–32 px
   across receives one sample per update tile that lands in it — 4–16 samples a frame at tile 8
   (`docs/SHARC-sky-budget-2026-09-16.md` §2) — against the 0.1–0.3 a secondary-fed cell gets.
   This is what makes freshly revealed geometry readable in its first frame, and it is why
   `minSampleCount = 2` stops costing coverage. **It works just as well in a room.**

**The evidence that mechanism 2 is doing real work indoors is the user's own report**: the quality
gain from `updatePrimaryVertex` was confirmed in game, and Portal RTX — where mechanism 1 cannot
fire at all — showed no artefacts. **Caveat, and it is the single most valuable missing fact in
this document:** the brief records "confirmed in game … and saw no artefacts in Portal RTX" without
saying which title the quality confirmation came from, and the 0.1 ms figure likewise has no scene
attached. If the quality confirmation was FNV-only, mechanism 2's indoor value is still an
inference. §9 asks for both attributions.

**Inference:** because mechanism 2 is scene-independent and is plausibly the larger half indoors,
there is **no clean scene signal that would tell a controller when to turn this option off.** It
is not "outdoors yes, indoors no". It is "on if you want the coverage, off if you want the
0.1 ms", which is a preset decision, not a measurement.

---

## 7. The cheap always-on guard: examined and declined

The brief's proposal: refuse the primary deposit on surfaces below some roughness, where the
camera-dependence actually matters. The sky-budget doc anticipated exactly this —
"the fix is a roughness floor for primary deposits specifically (a one-line change in
`sharcPrimaryRejectReason`)" (`docs/SHARC-sky-budget-2026-09-16.md` §6, step 4).

### 7.1 The risk is real, and here is its precise shape

**Source.** The deposit's value is `PrimaryDirectDiffuseLobeRadianceHitDistance + PrimaryDirectSpecularLobeRadianceHitDistance`
(`hooks:316`). The specular half is evaluated toward the camera.

**Inference.** At a *secondary* vertex the outgoing direction varies per sample, so the
view-dependent error `L_out(ω) − ⟨L_out⟩` averages out over the samples in a cell — the cell
converges to the hemispherical mean, which is what an isotropic cache means. At a *primary* vertex
every sample uses the camera's direction, so the error does **not** average out: it becomes a
systematic offset that tracks the camera. And because the primary deposit feeds a cell 4–16 times
a frame against a secondary rate of 0.1–0.3, **the primary samples dominate the mix**, so the
offset is not diluted either. That is the mechanism, stated more sharply than §3.3 of the
sky-budget doc stated it, and it is correct.

**Inference, where it concentrates.** The magnitude is the specular share of outgoing radiance
times the narrowness of the lobe. For a dielectric, F₀ ≈ 0.04 at normal incidence, so the
view-dependent part is a small fraction of a surface dominated by its diffuse albedo. For a
**metal**, the diffuse lobe is zero and the specular lobe is everything. So the risk class is
narrow and nameable: **smooth metals that pass `minRoughness`** — polished chrome, clean metal
panels, vehicle bodies. On those, a cache cell is a poor description of the surface *however* it
is fed; what the primary deposit adds is that the wrongness becomes camera-locked, and therefore
visible as motion rather than as a static error, which is perceptually worse.

### 7.2 What the guard would save in time

**Inference.** A refused deposit skips the insert and the two radiance reads but not the material
decode that decided to refuse. In Portal RTX, roughness rejects were 15.8% of resolved surfaces
with `minRoughness` already at its 0.05 floor (user) — so *most* surfaces are above the floor and a
guard set anywhere reasonable above it would refuse a minority of deposits. **There is no basis
for expecting a large fraction of the 0.1 ms back**, and I will not claim one.

### 7.3 The derivation says no guard is needed — by this project's own criterion

This is the substantive finding of the section.

**Source.** The project's existing test for "is this surface's lobe wide enough for a cell to stand
in for it" is `sharcFootprintValid` (`hooks:77-84`): a lobe of GGX alpha α travelling distance `d`
is admissible when `d·sqrt(0.5α²/(1−α²)) > voxelSize`. **Source,** the voxel edge at camera
distance `d` lies in `(d/(2·gridScale), d/gridScale]` (`grid:153-168`, re-derived in
`docs/SHARC-grid-scale-2026-09-16.md` §3).

**Inference.** Apply that criterion to the primary, where the segment is the camera ray, so
`segmentLength = d` and the voxel is the primary's own. The `d` cancels, leaving
`sqrt(0.5α²/(1−α²)) > 1/gridScale`. At `gridScale` 50 that is α > 0.0283 — **below** the
`minRoughness` floor of 0.05, which the primary deposit already applies (`hooks:276`, `:279`).

**So NVIDIA's own spatial-spread criterion, applied to the primary deposit, gives a threshold the
existing floor already exceeds, and would never fire.** The residual worry is *angular* — the
deposit's single view direction versus the hemisphere — which the footprint criterion does not
address at all. That is exactly why §3.3 of the sky-budget doc had to raise it separately, and it
means **any roughness floor for primary deposits would be a judgement number with no derivation
behind it.** This project's own history says what happens then: `minRoughnessSpecular` was a
judgement number standing in for a measurable quantity, it had to be cranked to 0.7 and still did
not fix the problem it was aimed at, and the fix was to measure the quantity instead.

### 7.4 The alternatives, and why each fails

- **A roughness floor for primaries, as a new option.** No derivation (§7.3); adds a twenty-third
  option to a feature whose stated goal is that a modder should not have to learn any of them; and
  a new `SharcArgs` field is not free — the struct is `static_assert(sizeof == 80)` inside
  `RaytraceArgs` (`args:44-46`, `raytrace_args.h:165`), so growing it shifts every later member
  and makes every shader blob in the tree differ, which is why `updateSkyRetries` was bit-packed
  into `sharcArgs.enabled` instead (`args:16-21`). The free bits 5–31 of `enabled` would take a
  quantised floor, so it is *possible*; it is just not *justified*.
- **Reusing `minRoughnessSpecular`, which is already inert while `footprintGate` is on.**
  Tempting — it is already in the struct, already clamped to ≥ `minRoughness` (`host:98-99`), and
  its semantics ("the floor for arrivals the cache cannot represent directionally") almost fit.
  **Rejected:** it would give one option two unrelated meanings depending on the state of a third,
  which is precisely the class of trap the tuning guide spends a section warning about.
- **Deposit only the diffuse lobe.** Removes the view dependence at the source and costs one
  texture read less. **Rejected:** it zeroes the deposit on metals, whose outgoing radiance is
  entirely specular, and metals are common (rusted FNV metal, Portal's panels). It would trade a
  subtle camera-tracking tint for cells that are systematically dark.
- **Deposit only when the first bounce escaped to sky**, making the option self-limiting the way
  the retries are. Implementable at the same site as the retry (`hooks:347-390` already detects
  the first-bounce miss exactly). **Rejected:** it keeps only mechanism 1 of §6.1 and discards
  mechanism 2, which is the half that works indoors and is plausibly the half the user's quality
  confirmation is about. It would make the option cheap by removing most of what it does.
- **Deposit only into cells that are currently starved.** **Rejected on sight:** the test is a
  `HashGridFind`, which is most of the insert's cost, and it is a closed feedback loop at the
  per-cell level — deposit, cell becomes well-fed, stop depositing, cell starves, deposit.

### 7.5 Recommendation on the sub-question

**Do not build the guard.** The artefact has not been observed by anyone; the derivation says the
existing floor already covers the spatial criterion; any angular threshold would be invented; and
the cost saving is unquantified and probably small.

**What to do instead, at zero cost: `rtx.sharc.minRoughness` already *is* the guard.** Raising it
from 0.05 refuses the primary deposit on exactly the surfaces at risk (`hooks:276`, `:279` feed the
same `sharcRejectReason` every secondary hit uses). It also refuses *reads* there — which is
arguably correct, since a near-mirror cell is a poor answer for any reader, not just a
camera-facing one. So the remedy exists, is documented, and needs no code.

**What would change this answer:** a sighting. §9 says what to look for and how to confirm it in
one A/B. If the user sees a camera-tracking tint on a cached glossy surface and raising
`minRoughness` fixes it at an unacceptable cost in coverage, *then* the primary-specific floor is
worth its option, and §7.4 says how to add it without breaking the struct layout.

---

## 8. The presets, and the one change this document makes

**The 0.1 ms measurement falsifies half of the reasoning that kept `updatePrimaryVertex` on in all
three presets.** `docs/SHARC-presets-2026-09-16.md` §1 justified it as "turning it off saves one
hash insert and costs most of the cache's coverage, so it is not a performance lever." The
coverage half stands. The cost half does not: 0.1 ms is now the **only** frame-time figure attached
to any preset value, and it is the same order as the whole cache's measured net benefit.

**Judgement, Balanced and Quality keep it.** The user confirmed the quality gain and wants it on;
0.1 ms is well under one percent of a ~16–18 ms frame
(`docs/GPU-stage-profiling-2026-09-13.md`); and §6.1 says the coverage it buys is not confined to
open worlds. Balanced is the tested anchor and nothing here overturns it.

**Judgement, Performance turns it off.** The Performance preset's entire purpose is to spend less,
and it already declines `updateSkyRetries` on the explicit argument that outdoors the cache's
product is a 1.01-segment tail worth near-zero frame time
(`docs/SHARC-presets-2026-09-16.md` §4). The identical argument applies with more force to a
measured cost. Every other value Performance moves — tile 12, bounces 3, capacity 20 — is an
*estimate* of "a fraction of a few tenths of a millisecond". If the preset cannot decline the one
item with a measurement behind it, it has almost nothing left to decline.

**The honest consequence, and it is in the tooltip.** Without the primary deposit, under open sky
most update paths exit to the sky and store nothing, so the Performance preset outdoors thins the
cache to near nothing while still paying for the update and resolve passes. For a user who plays
outdoors and wants the time back, `rtx.integrateIndirectMode = 0` is the honest setting rather
than that preset. Saying so in the tooltip is better than shipping a preset that quietly becomes a
cache-shaped no-op.

### 8.1 What was changed

`host:367-437`, C++ only — no shader source was touched, so every SHARC blob is byte-identical to
the `c66e2d29b` build (confirmed by mtime: the blobs date from the 19:23 sky-budget build and the
only recompiled object was `rtx_render_rtx_sharc.cpp.obj`).

- `updatePrimaryVertexObject().setDeferred(true)` moved out of `applyShared` (`host:379-393`) and
  into `applyUpdateBudget`, which now takes a fifth parameter (`host:395-401`).
- Quality `applyUpdateBudget(4, 8, 22, 2, true)`, Balanced `(8, 4, 22, 1, true)`, Performance
  `(12, 3, 20, 0, false)` (`host:404`, `:415`, `:426`).
- The Balanced and Performance tooltips now carry the 0.1 ms figure and, for Performance, the
  open-sky consequence above. The block comment at `host:367-377` now says five options trade
  along the axis, not four.

**Nothing changes for either existing install.** A preset only applies when clicked, and both
games' `rtx.conf` files set these keys explicitly
(`docs/SHARC-presets-2026-09-16.md` §9). No `RTX_OPTION` default or description changed, so
`RtxOptions.md` needs no regeneration.

---

## 9. What the user should test

All expectations below are **unmeasured**.

1. **Attribute the two figures that are missing a scene (§6.1).** Which title was
   `updatePrimaryVertex` measured at 0.1 ms in, and which title was the quality gain confirmed in?
   If the 0.1 ms was Portal and the quality gain was FNV-only, the case for turning the option off
   indoors becomes strong and §8's Balanced decision should be revisited. If the quality gain was
   also seen in Portal, mechanism 2 of §6.1 is confirmed and Balanced is right as it stands. **This
   costs nothing to answer and it is worth more than anything else on this list.**
2. **Split the 0.1 ms between the update pass and the resolve pass (§6).** Same view, stats on,
   read `GPU ms: update | resolve | query` with `updatePrimaryVertex` on and off, after the cache
   age passes ~120 frames each time. **Expected:** if most of it is on `update`, the cost is the
   per-path insert and is genuinely constant; if a meaningful part is on `resolve`, it is the
   residency term and it scales with scene openness. That decides whether "constant cost" is the
   right description of this option at all.
3. **Look for the camera-tracking tint (§7.1).** Portal RTX, a glossy-but-cached panel, camera
   orbiting it while the panel stays in view. Debug view **583 (Cached Radiance)** is the view that
   shows it directly — if the cached radiance on that panel brightens and dims with camera angle
   rather than staying put, that is the artefact. **Expected:** nothing, since the user has already
   looked and seen nothing. A sighting would reopen §7.
4. **Confirm the guard that already exists works, if step 3 finds anything.** Raise
   `rtx.sharc.minRoughness` 0.05 → 0.2 and see whether the tint goes and what the roughness reject
   share costs in coverage. That is the whole experiment; no code needed either way.
5. **The Performance preset, after this change.** In an interior and then outdoors: `GPU ms:
   update`, `GPU ms: resolve`, the hit rate, `Cache terminates` and the miss split, against
   Balanced. **Expected:** update and resolve both fall; the no-cell miss share rises sharply
   outdoors. If outdoors it does not fall by roughly the 0.1 ms, the measurement in step 1 was
   scene-specific and §8 should be revisited.
6. **Nothing here needs the panel open to work.** Steps 2 and 5 need it; steps 1, 3 and 4 do not.
   Turn `measureGpuTime` and `collectQueryStats` off again before any frame-time comparison —
   about 1 ms, measured (§3).

---

## 10. Build, validation and deployment

**Build.** Release, `python -m mesonbuild.mesonmain compile -C _Comp64Release -j 4`, synchronously,
twice. Pass A rebuilt exactly one object (`rtx_render_rtx_sharc.cpp.obj`) and relinked: `d3d9.dll`
280,178,176 bytes at 20:59:59, from 280,177,664 at 20:38:06. Pass B changed nothing (DLL mtime
identical). Both exit 0. The only lines matching "error" in either log are the pre-existing
`LNK4099` warnings on `mini_chromium.lib`. Logs `_Comp64Release/adaptive-build{A,B}.log`.

**Shaders untouched.** No shader source was edited; `integrate_indirect_sharc_update.h` and
`integrate_indirect_sharc_query_stats.h` still carry their 19:23 mtimes from the sky-budget build.

**Validator.** `validate_sharc_integration.py --shader-dir _Comp64Release/src/dxvk/rtx_shaders`
with the Vulkan SDK 1.4.357.0 `spirv-dis`/`spirv-val`: **61 PASS, exit 0**, including "52 declared
SHARC stages omit bindings 10/51 and the stealing flag" and "legacy TraceRay closest hit retains
the reservoir binding and sample stealing". Log `_Comp64Release/adaptive-validate.log`.

**Deployment.**

- **Fallout New Vegas: deployed.** Neither `FalloutNV.exe` nor `nvse_loader.exe` was running.
  `d3d9.dll` + `d3d9.pdb` copied to `Fallout New Vegas/.trex`, both `cmp`-verified against the
  build. Backups kept as **`backup-pre-perf-preset-primary-20260916-210436`**.
- **Portal RTX: NOT deployed, deliberately.** `NvRemixBridge.exe` (PID 27540, started 20:46:39,
  command line `… hl2.exe -game portal_rtx`) held `d3d9.dll` open for the whole session — Portal
  RTX was running. The rule is not to deploy while a game is running, so the copy was not forced
  and the backup taken for it was removed again so that no `backup-pre-…` file sits there
  implying a deployment that did not happen. **Portal RTX is still on the 20:38 build from
  `ed39787a9`**, which differs from this one only in the preset combo — so its behaviour is
  unchanged unless a preset is clicked. Re-run the deploy once Portal RTX is closed.

**Not pushed.** Committed locally.

---

## 11. What only the game can settle

1. The scene attribution of the 0.1 ms and of the quality confirmation (§9 step 1). Everything in
   §6.1 and §8 turns on it.
2. How the 0.1 ms splits between `GPU ms: update` and `GPU ms: resolve` (§9 step 2), which decides
   whether the option's cost is constant or scales with scene openness.
3. Whether the camera-tracking tint of §7.1 exists at all. One sighting reopens §7; continued
   absence closes it.
4. What the Performance preset actually returns now that it declines the primary deposit, and
   whether the outdoor thinning it causes is as severe as §8 expects.
5. Whether the panel hint of §5 item 4 is worth its line of screen space — a judgement only
   someone looking at the panel can make.
