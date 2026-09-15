# SHARC and ray portals (2026-09-15)

Worktree `wsn3g`, branch `revised-9-10`, HEAD `b810ad2fc`. Source reasoning only: nothing
was built, deployed, pushed or run, and no `.cpp`/`.h`/`.slang`/`.slangh`/build file was
modified. Fallon New Vegas has no ray portals, so none of this is reproducible on the game
this fork is tested against — every claim below is traced to source, and every place where
the source runs out and only a Portal-RTX-style scene could settle the question is marked.

Statements are labelled: **source** for something read directly out of the tree,
**inference** for a conclusion drawn from several source facts. No performance number and
no visual outcome is claimed anywhere; none was measured.

## Verdict

**Something in between, and not the same shape as `allowWboit` or `allowOpacityMicromap`.**

Three separate questions hide inside "is SHARC correct with portals", and they have three
different answers:

| Question | Answer |
|---|---|
| Does the *traversal* work? | Yes. The variant selection is complete and structurally identical to the non-SHARC path. Nothing about portal traversal is missing from the SHARC pipelines. |
| Is the *cache key* the wrong key — i.e. does a portal put geometry at fake positions that collide with real ones? | No. Remix portals are true ray teleportation into real world space. Position+normal stays a valid key for effectively the whole scene. |
| Is *cache insertion* sound? | **No, for a bounded subset.** Remix's world is not one scene: the TLAS instance mask a ray may see depends on `portalSpace`, and `portalSpace` is not in the key. Radiance is a function of `(P, N, portalSpace)` while the cache assumes `(P, N)`. |

So the gate is not stale conservatism in the WBOIT/OMM sense — there is a real invariant
violation underneath it — but it is also not the deep "portals break world-space caching"
problem the framing suggests. The violation is confined to `OBJECT_MASK_ALL_DYNAMIC`, which
in the indirect path means **the player model and its portal clone**, plus viewmodel-
originated paths. Every static surface, light and occluder in the scene is visible to every
ray regardless of portal space, and for those the cache key is exactly right.

The gate as written is also broader than the defect: it refuses SHARC whenever
`rtx.rayPortalModelTextureHashes` is merely *configured*, not when portals are actually
open. In Portal RTX that is the whole session.

The cheapest correct fix is described in [Fixes](#fixes). The short version: gate cache
**insertion** on `pathState.portalSpace == PORTAL_SPACE_NONE`, leave **queries**
unconditional. That is a two-line change at one call site, needs no new key bits, no extra
memory and no SDK patch — and it is *not* the option the brief guessed at, because the
brief's version ("exclude portal-reached vertices from insertion") and this one differ in
which side of the portal counts.

---

## 1. How Remix represents ray portals

### Portals teleport rays; they do not fake geometry

**Source.** `resolve.slangh:621` handles `surfaceMaterialTypeRayPortal`. On a hit inside an
active portal it sets `teleportMatrix` from
`rayPortalHitInfo.encodedPortalToOpposingPortalDirection.unpack()` and calls
`resolveVertexFinalContinue`, which at `resolve.slangh:292-297` does:

```slang
resolveVertexState.origin = (mul(teleportMatrix, vec4(surfaceInteraction.position, 1.0f))).xyz;
resolveVertexState.direction = normalize(mul(mat3(teleportMatrix), vec3(resolveVertexState.direction)));
```

The resolve loop then re-traces from that new origin. The next hit is on ordinary geometry
at its ordinary world position, reached by an ordinary ray.

This is the single most important fact for this investigation and it is favourable. There
is no "portal space" coordinate system in which post-portal hits live. `surfaceInteraction.position`
after two portal crossings is the same `float3` that a ray which walked there the long way
would produce.

**Source.** The same is true for visibility rays: `visibility.slangh:389-403` teleports the
shadow ray by the same matrix and continues.

**Source.** The unfolded, portal-*virtual* positions that do exist —
`virtualHitDistance`, `virtualMotionVector`, `accumulatedRotation`
(`geometry_resolver.slangh:266, 312`) — are computed for denoiser reprojection and depth
only. The G-buffer world position written at `geometry_resolver.slangh:376` is the true
`surfaceInteraction.position`, and that is what the indirect pass reads back
(`integrator_indirect.slangh:686, 1408`). SHARC is nowhere near the virtual positions.

### `numActiveRayPortals` and `rayPortalModelTextureHashes`

**Source.** `rtx_ray_portal_manager.cpp:339` sets `numActiveRayPortals` per frame, counting
in pairs (`activeRayPortalCount += 2`) only for portal pairs whose transforms resolved.
`maxRayPortalCount` is 2 and `maxRayPortalPairCount` is 1 (`ray_portal.h:32-35`), so in
practice this is 0 or 2.

`rayPortalModelTextureHashes` is a static config list of the textures that identify portal
geometry — for Portal RTX, `config.cpp:817` sets it to the orange and blue portal texture
hashes. It is non-empty for the entire session whether or not a portal is currently open.

**Source.** `rtx.rayPortalEnabled` (`rtx_camera_manager.h:90`) is declared and set to `True`
in the Portal config but is **read nowhere in the tree** — a repo-wide grep finds only the
declaration and the config default. It is dead. Anything reasoning about "are portals on"
has to use the hash list or the active count, which is what both `includePortals` and the
SHARC gate do.

### Portal space, and the ray mask that rides on it

**Source.** `ray_portal.slangh:33-37` defines four portal spaces:
`PORTAL_SPACE_NONE`, `PORTAL_SPACE_PORTAL_0`, `PORTAL_SPACE_PORTAL_1`,
`PORTAL_SPACE_PORTAL_COMBINED`. `updateStateOnPortalCrossing` (`resolve.slangh:54-122`)
advances it on every crossing and — this is the load-bearing part — **rewrites the ray
mask**:

```slang
case PORTAL_SPACE_NONE:
  newPortalSpace = PORTAL_SPACE_PORTAL_0 + rayPortalIndex;
  newDynamicMask = rayPortalIndex == cb.virtualInstancePortalIndex
    ? OBJECT_MASK_VIEWMODEL_VIRTUAL | OBJECT_MASK_PLAYER_MODEL
    : OBJECT_MASK_ALL_PLAYER_MODEL;
  if (rayMask & OBJECT_MASK_PLAYER_MODEL)
    newDynamicMask |= OBJECT_MASK_PLAYER_MODEL_VIRTUAL;
```

and ends with

```slang
rayMask = (rayMask & ~OBJECT_MASK_ALL_PLAYER_MODEL) | (newDynamicMask & OBJECT_MASK_ALL_PLAYER_MODEL);
portalSpace = newPortalSpace;
```

`PORTAL_SPACE_PORTAL_COMBINED` is a deliberate collapse — the comment says a ray could in
principle return to an immediate portal space but "we'd need more state and logic to resolve
it", so anything past two crossings with the same index is lumped together.

**Source.** `path_state.slangh:421-426` stores `portalSpace` in bits 0-1 of `PathState._data`,
so it is carried in the TraceRay payload and is live at the SHARC hook.

**Source.** `integrator_indirect.slangh:74, 82`, the indirect path inherits both from the
primary G-buffer hit:

```slang
pathState.portalSpace = geometryFlags.portalSpace;
...
pathState.rayMask = OBJECT_MASK_ALL | (geometryFlags.objectMask & OBJECT_MASK_ALL_DYNAMIC);
```

`OBJECT_MASK_ALL` (`instance_definitions.h:95`) is
`TRANSLUCENT | PORTAL | OPAQUE | ALPHA_BLEND` — all static world geometry, unconditionally.
The only portal-space-dependent bits are the four dynamic ones:
`VIEWMODEL`, `VIEWMODEL_VIRTUAL`, `PLAYER_MODEL`, `PLAYER_MODEL_VIRTUAL`
(`instance_definitions.h:77-86`).

### `useRayPortalVirtualInstanceMatching` is not what the name suggests

**Source.** `rtx_draw_call_tracker.cpp:286-352`. This is a CPU-side *replacement-asset
identity* heuristic: when a ViewModel draw call produces a brand-new `ReplacementInstance`
with no prims, `tryPortalMatch` looks for an existing instance near
`rayPortalManager.getVirtualPosition(key.worldPos, portalToOpposingPortalDirection)` and
reuses it, so a viewmodel whose transform jumped through a portal keeps its USD replacement
identity instead of re-instantiating. It creates no geometry, affects no position that
reaches a shader, and is irrelevant to the cache key. Mentioning it here only to close it
out.

### Virtual instances *are* real duplicate geometry

**Source.** `rtx_instance_manager.cpp:2008-2023` (viewmodel) and `:1843-1890` (player model).
A virtual instance is a full `RtInstance` copy, `teleportWithHistory(closestPortalInfo.portalToOpposingPortalDirection)`'d
to the portal-transformed world position, with its TLAS mask forced to the `_VIRTUAL` bit:

```cpp
virtualInstance->m_vkInstance.mask = OBJECT_MASK_VIEWMODEL_VIRTUAL;
virtualInstance->teleportWithHistory(closestPortalInfo.portalToOpposingPortalDirection);
```

For the player model, which copy is "real" flips depending on which side the game put the
player on (`:1864-1865`): `playerModelIsVirtual ? OBJECT_MASK_PLAYER_MODEL_VIRTUAL : OBJECT_MASK_PLAYER_MODEL`.

So: **distinct world positions, not aliased-onto-real-geometry positions** — but positions
that sit right at the exit portal's mouth, i.e. inside the far room, where they overlap the
volume that real far-side geometry occupies.

---

## 2. Where SHARC's keys come from, and what is in them

**Source.** `sharc_integrator_hooks.slangh:44-46`, the only place the cache is touched:

```slang
SharcHitData sharcHit = (SharcHitData)0;
sharcHit.positionWorld = surfaceInteraction.position;
sharcHit.normalWorld = surfaceInteraction.geometryNormal;
```

Insert (`sharc_update.slangh:42`) is `HashGridInsertEntry(..., hit.positionWorld, hit.normalWorld, ...)`.
Query (`sharc_integrator_hooks.slangh:56`) is
`HashGridComputeSpatialHashWithVoxelSize(sharcHit.positionWorld, sharcHit.normalWorld, ...)`.

**Source.** `HashGridCommon.h:61-75, 179-198` — with `HASH_GRID_COMPACT 0` the 64-bit key is:

| Bits | Contents |
|---|---|
| 0-50 | quantized grid X/Y/Z, 17 bits each |
| 51-59 | logarithmic level (9 bits) |
| 60-62 | three normal sign bits |
| 63 | reserved for user data — the SDK's "responsive lighting" flag |

**Source.** `sharc_sdk.slangh:16-17` sets `SHARC_ENABLE_RESPONSIVE_LIGHTING 0`, and
`SharcCommon.h:838` only reads bit 63 under that define. **Bit 63 is unused in this fork.**
That is exactly one spare bit, and it matters for the fix options below.

**Source.** The level comes from distance to the *real* camera
(`HashGridCommon.h:144-150`), and `makeSharcParameters()` (`sharc_bindings.slangh:29-31`)
feeds it `cb.sharcArgs.cameraPosition`, which `rtx_sharc.cpp:129-131` takes from
`ctx.getSceneManager().getCamera().getPosition()` — the main camera, not any virtual camera.

Nothing portal-related enters the key. There is no portal index, no portal space, no
instance mask.

---

## 3. The invariant SHARC needs, and exactly where Remix breaks it

SHARC's contract, restated precisely: **the radiance leaving a surface point must be a
function of that point's world position and normal alone.** Not "must be
direction-independent" — the SDK's own SH mode handles direction. Specifically: two
different paths arriving at the same `(P, N)` must be entitled to the same answer, because
the cache will give them the same cell.

### What does *not* break it

**Inference, from the source facts in §1.** The portal itself does not break it. A portal is
a wormhole: light that reaches `P` through a portal and light that reaches `P` the long way
are both incident radiance at the same physical point, and the reflected radiance the cache
stores is one physical quantity. Writing both into the same cell is not averaging two
lighting states; it is two Monte Carlo samples of one lighting state. That is what the cache
is for.

The accumulation model agrees. `sharc_update.slangh:57-68` chains
`radiance = state.radiance[i] + state.weights[i] * radiance` in reverse over the inserted
vertices. Throughput is accumulated through `accumulateThroughput` →
`sharcAccumulateThroughput` (`integrator_indirect.slangh:94-99`), and the portal's ring
attenuation lands in `radianceAttenuation` at `resolve.slangh:650`, which reaches the same
throughput chain. So radiance crossing a portal is correctly attenuated by the portal mask
and correctly propagated to the pre-portal vertex. The chain is geometry-agnostic; a
teleport between two inserted vertices is invisible to it and harmless.

### What does break it

**Source.** The ray mask. `updateStateOnPortalCrossing` rewrites it, and it feeds three
things that determine the radiance estimate at a vertex:

1. the continuation TraceRay's instance mask (`integrator_indirect.slangh:277`);
2. `evalNEESecondary`'s shadow-ray mask — `integrator.slangh:231`:
   `uint8_t rayMask = OBJECT_MASK_OPAQUE | (objectMask & OBJECT_MASK_ALL_DYNAMIC);`
   with `objectMask` = `pathState.rayMask` passed at `integrator_indirect.slangh:828`;
3. the unordered-resolve mask for particles and decals
   (`integrator_indirect.slangh:277`, `convertPrimaryRayMaskToUnordered`).

**Inference.** Therefore: at one world point `P` with normal `N`, an update path whose
`portalSpace` is `NONE` computes its lighting with the real player model as occluder and
bounce source; an update path whose `portalSpace` is `PORTAL_0` computes it with the virtual
clone instead, or with both. Both write to the same cell. The cell converges to a
sample-count-weighted average of two physically different lighting configurations, and a
query path in either portal space reads that average.

This is the genuine defect. It is real, it is not hypothetical, and it follows from source
alone.

### How big it is

**Source-bounded, magnitude is inference.** The divergence is confined to the four dynamic
mask bits. Static geometry, all analytic and emissive-triangle lights, all portal surfaces,
all alpha-blended and translucent geometry are in `OBJECT_MASK_ALL` and visible to every ray
regardless of portal space. Further, `resolve.slangh:110` clears viewmodel bits for indirect
rays entirely unless the ray originated on a viewmodel surface:

```slang
#else // Indirect Rays
  rayMask &= ~OBJECT_MASK_ALL_VIEWMODEL;
  if (hasRayOriginatedFromViewModel)
    rayMask = rayMask | (newDynamicMask & OBJECT_MASK_ALL_VIEWMODEL);
```

So for the overwhelming majority of SHARC's traffic the one divergent object is **the player
model and its portal clone**. Whether that produces a visible artifact — the player-model
clone's shadow or bounce colour bleeding into cells near a portal that a main-space path
also feeds — is exactly the thing that **cannot be settled from source**. It needs a Portal
RTX scene with the player near an open portal, `rtx.sharc.allowRayPortals` on, and an A/B
against `integrateIndirectMode` 0/1. I have not run it and make no claim about it.

### Two things I checked that turned out clean

**Source.** Volumetrics *are* portal-space-dependent — `volume_args.h:26-29` declares three
froxel volumes (`froxelVolumeMain`, `froxelVolumePortal0`, `froxelVolumePortal1`) and
`portalSpaceToVolumeHint(portalSpace)` picks between them. But every call site is inside
`#ifdef RESOLVE_OPACITY_LIGHTING_APPROXIMATION` or the G-buffer resolver, and that define
appears only in `rtx/pass/gbuffer/*.slang`. The comment at `resolve.slangh:156` says so
outright: "Currently this logic is only needed by the g-buffer pass which only deals with
primary rays". The indirect integrator never samples a froxel volume, so SHARC never caches
one. No defect here.

**Source.** Portal *transport sampling* for NEE is not implemented on the indirect path at
all — `integrator_indirect.slangh:826-828` passes `invalidRayPortalIndex` with the comment
"Todo: Ray Portal transport sampling in the future." So indirect NEE never samples a light
through a portal, with or without SHARC. The cache faithfully reproduces whatever the
integrator computes; this is a pre-existing integrator limitation, not a SHARC one, and it
is identical in the reference path SHARC would be compared against.

---

## 4. Should cached radiance be portal-transformed?

**No. Source, and it is unambiguous.**

`sharc_sdk.slangh:20-21` sets `SHARC_ENABLE_SH_ENCODING 0`, so a cell stores a single
scalar-triple of radiance (`SharcTypes.h:29-31`, `float16_t4 radianceData`) accumulated by
plain `InterlockedAdd` on three channels (`SharcCommon.h:495-498`). There is no direction to
rotate.

But the deeper answer holds even if SH encoding were switched on. The cached value belongs
to a **world-space point**, and after a teleport the query point *is* a genuine world-space
point in the ordinary world frame. The portal transform applies to the *ray*, in the segment
between two vertices — it never applies to a vertex's own frame. So there is no portal
transform that should be applied to a cached value, at any SH order.

Corroborating detail: `sharc_update.slangh:66` passes a constant
`vec3(0.0f, 0.0f, 1.0f)` direction with weight `0.0f` to `SharcAddVoxelData` precisely
because SH is off, and `SharcAddVoxelData` ignores both under `#if !SHARC_ENABLE_SH_ENCODING`.

Portals rotating and mirroring is a red herring for the cache. It is not a red herring for
the *denoiser*, which is why Remix carries `accumulatedRotation` and
`virtualMotionVector` — but those are a different subsystem.

---

## 5. What the pinned SDK says

SHARC 1.8.3, vendored at `src/dxvk/shaders/rtx/external/sharc/`, not a git submodule.

**Source.** The usage overview (`SharcCommon.h:17-45`) documents the three passes and the
buffer/barrier requirements. It describes the structure as a "world-space radiance cache"
and `HashGridCommon.h:12-22` says the grid "map[s] hit positions, and optionally coarse
normal direction, to stable cache entries".

**There is no documented assumption that portals violate.** The SDK never says "the scene
must be a single consistent scene" because for every renderer it was written for, that is
not a statement anyone would think to write down. The assumption is implicit in the key
layout: `(quantized position, level, normal octant)` and nothing else. The one extension
point the SDK leaves — bit 63, the responsive-lighting flag — is a *lighting-signal*
discriminator, not a scene-identity one, and it is unused here.

Two SDK details that a fix would collide with:

**Source.** `SHARC_BLEND_ADJACENT_LEVELS` defaults to 1 (`SharcCommon.h:102-103`) and is not
overridden in `sharc_sdk.slangh`, so it is on. When the camera moves,
`SharcResolveEntry` (`SharcCommon.h:947-972`) calls `SharcGetAdjacentLevelHashKey`, which at
`SharcCommon.h:781-824` **rebuilds the key from scratch** from position, level and normal
bits, carrying over only `HASH_GRID_NORMAL_BIT_MASK`. Any custom high bit is silently
dropped. So a portal-discriminating bit in the key would be lost during adjacent-level
blending, and portal-space-0 entries would blend into main-space entries on camera motion.
Fixing that means patching a vendored SDK file or turning the blend off.

**Source.** `SHARC_ENABLE_CACHE_RESAMPLING` is forced to 0 (`sharc_sdk.slangh:28-30`,
"Keep the initial estimator finite and free of cache feedback"), which removes a whole class
of cross-cell feedback that would otherwise compound any keying error. Helpful, and worth
knowing before anyone turns it on.

---

## 6. What the portal-aware SHARC variants actually differ by

**Source.** `integrate_indirect_closesthit.rchit.slang:210-348`. The portal and `_no_portals`
closest-hit variants differ by **exactly one define**:

| Variant family | `SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK` |
|---|---|
| `integrate_indirect_sharc_query_closesthit[_no_pom][_stats][_wboit]` | `SURFACE_MATERIAL_RESOLVE_TYPE_ALL` |
| `..._no_portals[_no_pom][_stats][_wboit]` | `SURFACE_MATERIAL_RESOLVE_TYPE_OPAQUE_TRANSLUCENT` |

`surface_material_hitgroup.h:29-32`: `ALL` = `OPAQUE | TRANSLUCENT | RAY_PORTAL`.
`OPAQUE_TRANSLUCENT` drops `RAY_PORTAL`. Miss shaders are the same story
(`integrate_indirect_miss.rmiss.slang:102-126`).

That define controls two things and only two:

1. **`resolve.slangh:621`** — whether the `surfaceMaterialTypeRayPortal` branch, and with it
   the entire teleport, is compiled in. This is the traversal half.
2. **`integrator_indirect.slangh:507, 532, 585, 1090`** — `#if (…& RAY_PORTAL) != 0` guards
   that wrap the NEE-cache insertion and the light-sampling / continuation-ray block in
   `if (materialType == surfaceMaterialTypeOpaque || materialType == surfaceMaterialTypeTranslucent)`,
   because in the portal build the same closest-hit shader also runs for portal-material hits
   and must not shade them. Comment at `:583`: "We don't need to sample lights or surface
   rays in the portal closest hit shader."

**Nothing in that define touches cache insertion or key computation.** `sharcOnResolvedVertex`
is called at `integrator_indirect.slangh:569`, *outside* both guard blocks and after
`if (pathState.continueResolving) return;` at `:537` — so a portal hit, which keeps
`continueResolving` true, never reaches the hook at all. That matches the design doc's own
rule (`docs/SHARC-design-revised-9-10.md:72`): "Never insert an intermediate
`continueResolving` event or a portal/render-target control surface." The implementation
honours it.

**So the answer to question 5 is: the portal variants handle traversal, and only traversal.**
They are necessary and they are correct. They say nothing about cache correctness, and the
existence of `_no_portals` variants is not evidence that anyone established cache
correctness — it is evidence that the shader-variant plumbing was done properly.

### The pipeline plumbing is complete

**Source.** `rtx_pathtracer_integrate_indirect.cpp:352-435`, `getSharcTracePipelineShaders`
selects raygen / miss / hit group on `includePortals`, computed at `:771` as
`rayPortalModelTextureHashes().size() > 0 || numActiveRayPortals > 0`. That is the identical
predicate and the identical single-hit-group structure the non-SHARC path uses at
`:1069-1200` (`material_rayportal` vs `material_opaque_translucent`). The gbuffer pass
computes it the same way at `rtx_pathtracer_gbuffer.cpp:736`.

The SHARC **update** passes need no portal variant at all: they are compute or raygen
(`integrate_indirect.slang:45-80`), they declare no
`SURFACE_MATERIAL_RESOLVE_TYPE_ACTIVE_MASK`, and
`polymorphic_surface_material.slangh:26-27` defaults it to `SURFACE_MATERIAL_RESOLVE_TYPE_ALL`.
So update paths always traverse portals, whether or not any exist. **Inference:** this is
correct but worth stating explicitly, because it means "turn on `allowRayPortals` and the
update pass won't know what to do with a portal" is *not* a failure mode. It knows.

---

## 7. Two secondary findings

### The `farEnough` heuristic weakens after a teleport

**Source.** `sharc_integrator_hooks.slangh:58`:

```slang
const bool farEnough = pathState.segmentHitDistance > 1.732051f * voxelSize;
```

`sqrt(3) * voxelSize` is the voxel diagonal, and the test exists to stop a vertex reading a
cell that the *previous* vertex also feeds — self-reference between two vertices in one
voxel. `resolve.slangh:404` sets `resolveVertexState.segmentHitDistance = rayInteraction.hitDistance;`
fresh on every resolve iteration, so after a teleport it measures the post-portal leg only.

**Inference.** That is the right distance for the voxel-size question (the ray really did
start at the exit portal) but the wrong pair of points for the self-reference question: the
previous *cache* vertex is on the near side of the portal. In Portal, the two mouths can be
metres apart on adjacent walls of one room, so the pre- and post-portal vertices can share a
voxel while the post-portal leg is long enough to pass the test. The guard silently
weakens exactly in the geometry portals are famous for.

This is a quality guard, not a correctness invariant, and it degrades only when two portal
mouths sit within one voxel of each other. Flagging it because a fix for the main defect
would not address it.

### Grid level uses true world distance, which is right and awkward

**Source.** `HashGridCommon.h:144-150` picks the level from `distance(cameraPosition, samplePosition)`
using the real camera. A surface seen through a portal can fill the screen while being far
away in world space.

**Inference.** The cache resolution there is therefore coarse relative to its screen
footprint. This is not incorrect — the voxel size is a world-space quantity and the world-space
distance is the honest input — but it means cached lighting through a portal is blockier per
screen pixel than cached lighting at the same apparent size in main space. Whether that is
visible is unmeasured and unmeasurable from here.

---

## 8. Fixes

Four options, cheapest first. All of them have the data they need already in hand:
`pathState.portalSpace` is live at the hook (`path_state.slangh:421-426`).

### A. Gate insertion on main space; leave queries unconditional

At `sharc_integrator_hooks.slangh`, add to the eligibility test, for the `SHARC_UPDATE`
branches only:

> insert only when `pathState.portalSpace == PORTAL_SPACE_NONE`

**What it buys.** Every cell is then written exclusively by paths that saw the main-space
dynamic-instance configuration. The `(P, N) → radiance` invariant is restored exactly,
because the only source of ambiguity was portal-space-dependent masking and no
portal-space path writes any more.

**Why queries stay unconditional.** A query in portal space reading a main-space-authored
cell is reading the right physical quantity for everything except the player-model
difference — which is the same error the cache already carries for the direct lighting at
that vertex, and strictly smaller than the error of not using the cache's many-frame
estimate at all. **Inference**, and the honest caveat: this trades a small, bounded bias for
a large variance reduction. It is the standard radiance-cache bargain and I believe it is
the right one, but a Portal scene would settle whether the bias is visible.

**Cost.** Two lines at one call site, no new key bits, no extra memory, no SDK patch, no new
shader variants. Cache coverage drops by whatever fraction of update paths originate on
through-portal pixels — **not measured, and unmeasurable without a portal scene**. Coverage
near a portal is partly restored anyway, because main-space update paths that cross *into*
portal space stop inserting there but the far-side geometry is also reachable by other
main-space paths through the portal's own mouth… which they are not, since crossing sets
`portalSpace`. So through-portal geometry would go uncached. That is the real cost of option
A and it should be stated plainly: **rooms only visible through a portal get no cache.**

Note this is the *opposite* polarity from the brief's phrasing. "Exclude portal-reached
vertices from insertion" and "insert only from main space" are the same rule; what matters
is that the *whole tail* of a path is excluded once it crosses, not just the first hit past
the portal — because `portalSpace` is sticky for the rest of the path.

### B. Put portal space in the key

Use bit 63 — the responsive-lighting bit, unused here — as a one-bit
`portalSpace != PORTAL_SPACE_NONE` discriminator.

**What it buys.** Full correctness for the main-space/portal-space split, and through-portal
geometry stays cached. `HashGrid_(Hash32)` XORs both 32-bit halves
(`HashGridCommon.h:126-132`), so bit 63 participates in slot selection properly, and the key
is otherwise only ever compared for equality.

**Costs, all concrete.**

1. Only one bit is free. `PortalSpace2BitsType` has four states. A one-bit key collapses
   `PORTAL_0`, `PORTAL_1` and `COMBINED` together, which is an *approximation*, not a fix —
   the two portal spaces genuinely differ in which player copy they see
   (`resolve.slangh:66-75`). Going to two bits means stealing a bit from position (17→16 per
   axis, halving addressable extent at each level) or from level (9→8).
2. `SharcGetAdjacentLevelHashKey` (`SharcCommon.h:781-824`) drops the bit. Either patch that
   vendored SDK function to carry it over, or set `SHARC_BLEND_ADJACENT_LEVELS 0` in
   `sharc_sdk.slangh` and lose the camera-motion blend.
3. Cells split, so the effective cache capacity for the same hash table falls near portals.
   By how much is scene-dependent and unmeasured.
4. `DebugOccupancy` (`HashGridCommon.h:335-337`) already colours entries by bit 63; that
   visualization would start meaning "portal space" instead of "responsive". Cosmetic.

### C. Insert from all spaces but strip the dynamic instances

Force the ray mask to exclude `OBJECT_MASK_ALL_DYNAMIC` on SHARC update paths, so the update
estimate is portal-space-independent by construction and the cache stores
static-scene-only radiance.

**What it buys.** Correct keying with no coverage loss and no key change. The cache becomes
an honest static-lighting cache.

**Cost.** The player model stops contributing bounce light and stops occluding in cached
lighting — everywhere, including scenes with no portals. That is a behaviour change the
whole fork pays for to fix a portal-only defect. **Inference:** almost certainly the wrong
trade unless the player-model contribution to indirect turns out to be negligible, which is
a measurement nobody here has.

### D. Leave the gate, narrow its predicate

Independent of the above, and worth doing regardless: the gate at `rtx_sharc.cpp:67-79`
fires on

```cpp
const bool rayPortals = args.numActiveRayPortals > 0
  || !RtxOptions::rayPortalModelTextureHashes().empty();
```

The hash-list half is the right predicate for *pipeline variant selection* — you must build
the portal-capable pipeline before a portal opens. It is the wrong predicate for *cache
activity*, where `numActiveRayPortals > 0` is the actual condition. Dropping the hash-list
term would let SHARC run in Portal RTX whenever no portal is currently open, with the
existing `compatibilityFlags` machinery at `:86-93` already clearing the cache the moment
one does.

**Caveat, stated because I could not rule it out from source:** this would mean SHARC
activates and deactivates as the player fires and closes portals, clearing an 80 MiB cache
each time (`capacityLog2` 21 → `capacity * 8` + `capacity * 16` × 2 bytes,
`rtx_sharc.cpp:107-117`). The three `ctx.clearBuffer` calls at `:127-131` are already on the
`clear` path and already happen on preset changes, so the mechanism exists — but whether
that thrash is acceptable in a game where portals open constantly is a judgement call, not a
source fact.

### Recommendation

**A + D**, in that order, if the goal is "make `allowRayPortals` mean something defensible".
A restores the invariant at essentially zero implementation cost and its one real
drawback — uncached through-portal rooms — is a coverage loss, not a correctness loss, and
degrades gracefully to the existing importance-sampled path. D makes the gate mean what it
says. B is the principled fix and should be reached for only if A's coverage loss turns out
to matter, because it costs a vendored-SDK patch and an approximation in the very bit budget
it is trying to make exact.

---

## 9. What only a test scene can settle

Listing these so nobody mistakes the reasoning above for a measurement.

1. **Is the player-model divergence visible at all?** The entire defect reduces to one
   object's occlusion and bounce. It could be imperceptible. Source cannot say.
2. **How much of a Portal RTX frame's update paths originate on through-portal pixels?**
   Determines option A's real coverage cost. Needs the existing
   `rtx.sharc.collectQueryStats` counters plus a portal-space split that does not exist yet.
3. **Does the weakened `farEnough` guard actually produce feedback brightening** in a room
   with two mouths close together, or does the normal-octant split in the key save it?
4. **Does the coarse grid level through a portal look blocky** at gameplay distances with
   `rtx.sharc.gridScale` at its 50.0 default.
5. **Whether A's query-side bias is visible** — main-space-authored cells read by
   portal-space paths.

Every one of those needs Portal RTX or an equivalent scene with an open portal pair. FNV
cannot produce a single data point on any of them.
