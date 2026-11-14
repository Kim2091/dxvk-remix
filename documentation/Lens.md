# Lenses

A lens is an analytic magnifier disc, such as the eyepiece of a telescopic sight. Camera rays that pass through the
disc are traced as magnified rays from the lens' objective, in the same frame and by the same passes as every other
primary ray. There is no second camera and no render-to-texture pass. This page describes how lenses work. The options
are listed in [RtxOptions.md](../RtxOptions.md) under `rtx.lens`.

## Model

A lens is an afocal telescope with magnification `M`. Its pose is the eyepiece center `E`, the optical axis `A`
(pointing out of the objective) and the reticle's up direction `V`, with `U = A x V`. Its optics are the eyepiece
radius, the objective distance and radius, the apparent field stop half-angle and the eye relief. The objective
center is `O = E + A * objectiveDistance`. The exit pupil sits `eyeRelief` behind `E` and has the radius
`objectiveRadius / M`.

## Lens slots

`LensManager` (`rtx_lens.h`) collects the frame's lenses from lens sources, sorts them nearest to the camera first and
keeps at most `rtx.lens.maxLenses`. Each lens has an id and keeps its slot for as long as it exists. `RaytraceArgs`
holds four slots for this frame and four for the previous frame, so slot `i` of both halves describes the same lens.
`lensMask` and `previousLensMask` mark the active slots.

- A lens without a previous slot (it just appeared, or the view history was reset) gets `lensFlagDiscardHistory`.
- A lens whose magnification changed by more than `rtx.lens.historyDiscardZoomStep` in one frame gets it too.
- A lens that disappeared keeps its previous slot for one frame, so the pixels it covered can discard their history.
- A new lens prefers a slot that was empty on the previous frame.

The lens sources are the debug camera lens (`rtx.lens.debugCameraLens`), fixed to the camera at the center of the
screen, lenses on mesh replacements in mod USD (see [Lenses in mod USD](#lenses-in-mod-usd)) and lenses a host
sends with its camera (see [Lenses from the Remix API](#lenses-from-the-remix-api)).

## Primary rays

`preprocessPrimaryRayState` tests the jittered camera ray `(c, d)` against the active lenses (`lensCreatePrimaryRay`
in `lens.slangh`):

1. The ray meets the eyepiece plane at `t = dot(E - c, A) / dot(d, A)`. Only `t > 0` counts.
2. The pixel is a lens pixel when `|c + t * d - E|` is below the eyepiece radius. The nearest lens wins.
3. With `dA = dot(d, A)` and `dT = d - dA * A`, the magnified direction is `d' = normalize(A + dT / (dA * M))`.
4. The traced ray starts at `O` along `d'`. Its cone spread angle is the screen spread divided by `M`, and its cone
   radius starts at 0.
5. The virtual hit distance starts at `t`, as it does at a ray portal. Depth and linear view Z are taken at the
   virtual distance along the unjittered camera ray, so the lens rim is a depth edge like a window frame.
6. The direction is marked altered, so the G-buffer stores the view direction and a miss reads the sky probe.
7. Lens rays skip the view model and the player model.

Every later pass finds the lens pixel again with `lensSlotAtPixel`, from the same jittered camera ray. Passes that
rebuild a position from the primary hit distance use the distance from the objective, which is the virtual distance
minus `t`.

## Motion

A lens pixel shows a point `X`, the virtual position along `d'` at the virtual distance minus `t`. Its previous pixel
is where the previous camera saw `X` through the previous frame's lens in the same slot
(`lensReprojectPosition`):

1. `X` includes the surface's own motion.
2. The previous objective direction is `normalize(X - O')`, split into `dA` and `dT` along the previous axis `A'`.
3. The previous eye-side direction is `e = normalize(A' + dT * M' / dA)`.
4. The previous virtual position is the previous camera position plus `e` times the previous virtual distance, which
   is the distance to the previous eyepiece plane plus `|X - O'|`.

The world-space virtual motion vector is the previous virtual position minus the current one, so NRD, RTXDI and
ReSTIR GI, which add it to the virtual position and project with the previous camera, land on the previous pixel.
The screen-space motion vector is that projection minus the pixel. A miss uses the same steps with its direction.

The history of a lens pixel is invalid when its lens has `lensFlagDiscardHistory`, when the slot had no lens on the
previous frame, or when the previous camera ray along `e` missed the previous eyepiece. The history of a pixel outside
every lens is invalid when its previous camera ray passed through a previous lens. DLSS-RR gets the discard value in
its disocclusion mask for these pixels. NRD and DLSS rely on their own depth tests.

RTXDI and ReSTIR GI compare surfaces by `RAB_GetSurfacePortalSpace`, which carries the lens slot above the portal
space, so temporal and spatial reuse never cross a lens rim. The previous frame's lens slot of a pixel is found from
the previous camera and the previous lenses.

### Clouds

The NUMOS screen clouds follow the lenses. `RtxNumos::beginFrame` passes the lens slots from `RaytraceArgs` to NUMOS as
`numos::FrameInputs::lenses`, without the pass-through lenses. The cloud pass marches each lens pixel along the magnified
ray, ends the march at the linear view Z's virtual distance minus the eyepiece distance, and reprojects its history
through the previous lens in the same slot, with the same history rules as above (NUMOS `docs/DESIGN.md`, "Lenses").
The composite gives a cloud-covered lens pixel the cloud's depth at the virtual distance and its motion through the lens
(`applyCloudMotionVectors`). The aerial perspective and the fog over the clouds still follow the camera ray.

## Overlay

The field stop, exit pupil falloff, vignette, glass tint and reticle are applied by `lens_overlay.comp.slang` to the
upscaled linear output, after denoising and upscaling and before post-processing. They never enter the G-buffer or
any history. Auto exposure meters the output just before the overlay, so lens pixels are metered at full
transmission. The overlay traces nothing. It tests the unjittered camera ray of each output pixel against the lenses.

- The apparent angle `theta` is measured between the camera ray and `A`, as tangents along `U` and `V`.
- The field stop passes `tan(theta)` below the field stop tangent, with a soft edge `rtx.lens.softEdgePixels` wide.
- The exit pupil transmission is the overlap of the eye pupil (radius `rtx.lens.eyePupilMm`) at the point where the
  camera ray crosses the exit pupil plane with the exit pupil, as a smoothstep of
  `(exitPupilRadius + eyePupilRadius - offset) / (2 * eyePupilRadius)`. An eye off to the side dims the whole image.
  An eye too close or too far dims one side first.
- The vignette is a cos^4 falloff that reaches 45 degrees at the field stop, scaled by `rtx.lens.vignetteStrength`.
- The field stop sits slightly out of focus and cuts into the eye's pupil, so over the outer tenth of the field the
  image fades out with a cos^2 before the soft edge, like the shadow at the edge of the eye box.
- Toward the field stop of a telescopic lens the image goes out of focus (field curvature): five bilinear taps
  across and along the edge, with a radius that grows from 0 at 85% of the field to `rtx.lens.edgeBlur` times three
  output pixels at the stop. Its colors split radially (lateral chromatic aberration): red is taken from further in
  and blue from further out, by up to `rtx.lens.chromaticFringe` times two output pixels at the stop, growing from 70%
  of the field. Both read the lens image from a copy of the output, which exists while a telescopic lens is active and
  either option is above 0. The direction toward the edge comes from the edge position one pixel to the right and
  below.
- The reticle is in the second focal plane: it is drawn at the apparent angle, so it keeps its size at every
  magnification. A reticle radiance of 0 draws it black. The styles are all drawn procedurally and anti-aliased, in
  reticle space where the field stop radius is 1:
  - crosshair, and duplex (posts from 0.35 out),
  - dot, and circle-dot (a dot inside a ring at 0.5),
  - mil-dot: a crosshair with dots every 0.1,
  - german: a fine cross with heavy posts from the left, the right and the bottom, from 0.3 out,
  - drop: posts from 0.5 out, a fine cross, ticks every 0.1 below the center and wider holdover bars at 0.2 and 0.4,
  - stadia: posts that taper from six line widths at the stop to one at 0.2, and range bars at 0.1 above and below,
  - frame: corner brackets of a rectangle with half size 0.5 by 0.35, ticks in from the middle of each side and a
    center dot,
  - grid: a fine grid every 0.2 inside 0.8, a center cross with range marks every 0.05 and a center dot,
  - dial: a ring at 0.6 with a tick every 30 degrees, longer every 90 degrees, and a center dot.

  The line width is the half-width of a line, and a dot's radius is two or three line widths. Mil-dot dots and center
  dots are the accent. With an accent radiance above 0 they are drawn in the accent color at that displayed
  radiance, otherwise like the rest of the reticle.
- While an active lens occludes outside (`lens:occludeOutside`, `REMIXAPI_CAMERA_LENS_FLAG_OCCLUDE_OUTSIDE`), every
  output pixel outside every lens is black, as behind a binocular or scope mask, in place of the peripheral blur. Each
  lens window then fades into the black the way a field stop does: a cos^2 over the outer tenth of the window and the
  soft edge (`rtx.lens.softEdgePixels`), measured in output pixels. Auto exposure meters before the overlay, so the
  black does not move it, and the HUD drawn after the frame stays visible.
- The markings have their own color when the lens gives one (`lens:markingColor`, the API's `markingColor`): they
  are drawn after the reticle and the accent, emitted at the marking radiance as a displayed value when it is above 0,
  otherwise lit at the accent's radiance or, with an unlit accent, as ink that passes the marking color's share of the
  image. Without a marking color they are drawn as above. The marking opacity (0 to 1) scales their coverage either
  way, so they can be see-through.
- Parallax (`lens:parallaxDistance`, the API's `parallaxDistance`, 0 for none): the reticle is focused at the parallax
  distance instead of infinity. With the eye's offset h from the axis, the entrance pupil views from M * h, a target at
  distance D moves against the reticle by M * h * (1 / D - 1 / P) in object space, and the eyepiece magnifies that M
  times, so the reticle and its labels shift by M^2 * h * (1 / D - 1 / P) in apparent tangent. D is the reticle
  center's distance, measured one frame late as for the rangefinder, and a miss counts as infinitely far. The shift is
  limited to 0.9 of the field stop radius. A pass-through reflex lens uses M = 1. The rangefinder readout does not
  shift.
- Glare and dirt (`lens:glareStrength`, `lens:dirtStrength`, 0 to 1, 0 for none) light the glass of lenses without a
  vision mode, inside the field stop, and stay subtle by construction. At full strength, veiling glare adds 7% of the
  lens' metered average brightness, and bright pixels in the lens image (displayed above white) scatter a soft halo
  from eight taps on a ring of 2% of the output height. The brightest distant light within about 38 degrees of the
  axis, growing smoothly toward it, adds up to 0.22 displayed and a warm ghost of 8% opposite its apparent position.
  That stray light needs a lit scene: it fades in as the lens' displayed average rises from 0.02 to 0.12, so daylight
  near the exposure target gets all of it and a dark room facing the light gets none. Dirt is a procedural pattern of soft smudges and specks fixed to the
  eyepiece: it absorbs up to 6% of the image and scatters the glare up to five times as strongly where it lies, so it
  shows mainly when glare lights it.
- A first focal plane reticle (`lens:firstFocalPlane`, `REMIXAPI_CAMERA_LENS_FLAG_FIRST_FOCAL_PLANE`) and its labels
  grow with the magnification: they are drawn at the magnification over the reference magnification times their
  size (`lens:referenceMagnification`, or the API's `referenceMagnification`, else `minMagnification`). Without a
  reference they keep their size. The rangefinder readout always keeps its size.
- A texture reticle (`texture`, `REMIXAPI_CAMERA_LENS_RETICLE_TEXTURE`) draws an image, `lens:reticleTexture` or the
  API's `reticleTexture`: a file path, or `0x` and the 16 hex digits of a texture made with `remixapi_CreateTexture`.
  The image covers the square from -1 to 1 of reticle space, or from -aperture aspect to aperture aspect across for a
  rectangle, and its alpha is the ink, drawn like the other reticles (multiplied in `reticleColor` when unlit,
  emitted when lit). Its color is not used. The runtime loads a file in the background on its first use, keeps it
  resident and caches it by path, and draws nothing until it is loaded. A trilinear sampler at the mip level of an
  output pixel keeps it smooth at every size. The accent, the rangefinder and the edge effects work as with the
  other reticles. Each lens slot binds its own texture.
- Markings are text in the reticle plane, drawn with a built-in font (digits, `+`, `-`, `.` and `m` as anti-aliased
  strokes on a seven-segment layout). With labels on, the grid gets range labels (+20 to +80 and -20 to -80) next to
  its horizontal lines and the frame gets scale numbers at its side ticks, in the reticle's ink. Other styles have no
  labels. The rangefinder shows the distance in meters along the axis to the surface at the reticle center, below the
  center in the accent color. The overlay thread at the reticle center's output pixel reads the primary hit distance
  there, removes the eye-to-eyepiece part, adds the objective distance, converts world units to meters
  (`rtx.sceneScale`) and stores the result in a small buffer, alternating between two halves per frame. The readout
  shows the previous frame's value, so it is one frame late. A jump of more than a tenth snaps, and smaller changes
  settle within a quarter of a second, so the digits do not flicker. A miss shows `---`.
- The edge effects grow toward the field stop of a telescopic lens or the window edge of a reflex lens: the edge
  tint is multiplied in from 0.3 of the way out, and the edge shade darkens from 0.5 of the way out. The edge glow is
  a thin rim reflection just inside the edge, about a pixel and a half wide, at a third of its color times its
  displayed strength.

## Kinds and apertures

A lens is telescopic or reflex.

- A telescopic lens remaps camera rays as described above, with the exit pupil, field stop and vignette.
- A reflex lens, such as a holographic or red-dot sight, has no exit pupil, field stop or vignette. Its reticle is drawn
  at infinity along the axis inside the window, so it depends only on the camera ray's direction and stays on target as
  the eye moves. Its glass shows a third of the tint. Its reticle radiance is a displayed value, independent of
  exposure, and a reticle without color and radiance is red and emissive. The field stop angle only sets the
  reticle's size.
- A reflex lens with an objective distance above 0 restarts camera rays at the objective, with its magnification
  (usually 1), so geometry inside the sight's housing, such as an opaque plate, is skipped. Motion vectors and history
  work as for a telescopic lens. With an objective distance of 0, camera rays pass unchanged (`lensFlagPassThrough`)
  and only the overlay sees the lens.

The window is a circle with the eyepiece radius, a rectangle with half height eyepiece radius and width aperture
aspect times its height, or a binocular pair: the union of two circles with the eyepiece radius whose centers sit the
aperture aspect times the radius to the left and right of the axis (below 1 they overlap). The shape is used by the
hit test, the field stop and its soft edge, the vignette, the edge fade, fringe and blur, the edge shade and tint, and
the window fade behind `occludeOutside`, for telescopic and reflex lenses alike. A binocular lens is one lens: the
magnification and the reticle stay centered on the axis, so both circles show one fused image. Each point measures
its edge position to the nearer circle's center, so the falloff follows the union's outline and the seam between the
circles is not darkened.

## Vision modes and highlighting

Vision modes and highlighting apply only to lens pixels, and run in the overlay after upscaling.

- Night: the lens image is amplified to its own exposure. The overlay meters one output pixel in 4x4 of each night
  vision lens into a small buffer, as the log2 luminance average, and adapts toward it at
  `rtx.lens.nightVision.adaptationSpeed`. The gain brings the average, times the tone mapper's exposure, to
  `rtx.lens.nightVision.targetLuminance`, from 1 up to `rtx.lens.nightVision.maxGainEv` stops, so auto exposure does
  not cap the lens. Bright points get a bloom from eight neighbors (`rtx.lens.nightVision.bloom`). From two stops above
  the target, the image overloads and washes out until the average falls. Blue-noise grain grows with the square root
  of the signal (`rtx.lens.nightVision.grain`). The result is in the lens' night tint, or
  `rtx.lens.nightVision.phosphorColor` when it gives none. The lens' night gain multiplies the metered gain, so a gain
  of 2.5 shows the image 2.5 times as bright as the metering alone would, also above the maximum gain.
- Thermal: the lens image is the heat, white hot. The G-buffer pass writes one byte of heat per lens pixel:
  instances with the HeatSource category are hot, emissive surfaces are warm, and the rest is at ambient temperature
  plus a small N dot L term from the brightest distant light. The overlay filters the heat bilinearly to the output.
- Highlighting: the G-buffer pass sets a bit for lens pixels whose primary surface has the HighlightHostile category,
  in lenses that highlight (`lens:highlightHostiles`, `REMIXAPI_CAMERA_LENS_FLAG_HIGHLIGHT_HOSTILES`, or every lens
  with `rtx.lens.highlightOnAllLenses`). The overlay draws the inner edge of the mask in a 3x3 neighborhood of render
  pixels in `rtx.lens.highlightColor`, and tints the inside by `rtx.lens.highlightTint`. Only directly visible
  surfaces are outlined.

The per-pixel features are a render resolution R16 image (8 bits of heat and the highlight bit), created while a
thermal or highlighting lens exists. The G-buffer pass writes it for every pixel only while one of them is active,
and the night vision copy of the output exists only while a night vision lens is active. With no lens active, none
of this runs.

`rtx.lens.peripheralBlur` blurs the image outside the lenses while a lens is active. Two passes blur the output
along x at half resolution and then along y, a 25-tap Gaussian each whose width grows with the option, and the
overlay blends the pixels outside every lens toward the result. At 0 the passes are skipped.

The categories come from the Remix API (`REMIXAPI_INSTANCE_CATEGORY_BIT_HIGHLIGHT_HOSTILE` and `_HEAT_SOURCE`), from
`rtx.highlightHostileTextures` and `rtx.heatSourceTextures` for D3D9 draws and API draws by texture, and from the
`remix_category:highlight_hostile` and `remix_category:heat_source` attributes in mod USD. They are surface flags, so
they can change every frame.

## Lenses in mod USD

A mod places a lens on a mesh replacement with the `RemixLensAPI` applied schema (the `RemixLens` USD plugin) on a
prim under the replacement, usually an `Xform`. The prim's transform is the lens pose:

- its origin is the eyepiece center,
- its local -Z axis is the optical axis, pointing out of the objective, as a USD camera looks,
- its local +Y axis is the reticle's up direction.

Lengths are in the prim's local units, so the lens follows the transform and scale of the prim and its replacement.

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `lens:eyepieceRadius` | float | 0 | The eyepiece disc radius. A lens without one is ignored. |
| `lens:objectiveDistance` | float | 0 | The distance from the eyepiece to the objective along the axis. Magnified rays start at the objective. |
| `lens:objectiveRadius` | float | 0 | The objective radius. The exit pupil radius is this divided by the magnification. |
| `lens:magnification` | float | 4 | The magnification, multiplied by `rtx.lens.magnificationScale`. |
| `lens:fieldStopDegrees` | float | 18 | The apparent field stop half-angle in degrees. |
| `lens:eyeRelief` | float | 0 | The ideal eye distance behind the eyepiece. |
| `lens:tint` | color3f | (1, 1, 1) | The glass transmission tint. |
| `lens:reticle` | token | `none` | `none`, `crosshair`, `duplex`, `dot`, `circleDot`, `milDot`, `german`, `drop`, `stadia`, `frame`, `grid`, `dial` or `texture`. |
| `lens:reticleColor` | color3f | (0, 0, 0) | The reticle color, used when the reticle radiance is above 0. |
| `lens:reticleRadiance` | float | 0 | The emitted reticle radiance. 0 draws a black reticle. |
| `lens:reticleLineWidth` | float | 0.004 | The half-width of a reticle line, as a fraction of the field stop radius. |
| `lens:visionMode` | token | `none` | `none`, `night` or `thermal`. See [Vision modes and highlighting](#vision-modes-and-highlighting). |
| `lens:kind` | token | `telescopic` | `telescopic` or `reflex`. See [Kinds and apertures](#kinds-and-apertures). |
| `lens:aperture` | token | `circle` | `circle`, `rectangle` or `binocular`. |
| `lens:apertureAspect` | float | 1 | The width over the height of a rectangular window, or the offset of each circle's center from the axis of a binocular window, in radii. |
| `lens:highlightHostiles` | bool | false | Outlines directly visible surfaces with the HighlightHostile category. |
| `lens:reticleAccentColor` | color3f | (0, 0, 0) | The color of the mil-dot dots and center dots. |
| `lens:reticleAccentRadiance` | float | 0 | The accent's displayed radiance. 0 draws the accent like the rest of the reticle. |
| `lens:edgeGlowColor` | color3f | (0, 0, 0) | The color of the glow ring at the field stop. |
| `lens:edgeGlowStrength` | float | 0 | The displayed strength of the glow ring. 0 draws none. |
| `lens:edgeShade` | float | 0 | The darkening toward the field stop, from 0 to 1. |
| `lens:edgeTint` | color3f | (1, 1, 1) | The color multiplied in toward the field stop. |
| `lens:nightTint` | color3f | (0, 0, 0) | The night vision phosphor color. Black uses `rtx.lens.nightVision.phosphorColor`. |
| `lens:nightGain` | float | 1 | A multiplier on the metered night vision gain. |
| `lens:markings` | token | `none` | `none`, `labels`, `rangefinder` or `labelsRangefinder`. See [Overlay](#overlay). |
| `lens:occludeOutside` | bool | false | Everything outside every lens is black while this lens is active. |
| `lens:noGrain` | bool | false | The vision mode draws no grain. |
| `lens:firstFocalPlane` | bool | false | The reticle grows with the magnification, relative to `lens:referenceMagnification`. |
| `lens:referenceMagnification` | float | 0 | The magnification at which a first focal plane reticle has its size. 0 keeps its size. |
| `lens:reticleTexture` | asset | none | The image of the `texture` reticle. Its alpha is the ink. |
| `lens:markingColor` | color3f | (0, 0, 0) | The color of the labels and the rangefinder readout. Black draws them in the accent. |
| `lens:markingRadiance` | float | 0 | The markings' displayed radiance. 0 draws them lit or as ink, as the accent is. |
| `lens:markingOpacity` | float | 1 | How opaque the markings are, from 0 to 1. |
| `lens:parallaxDistance` | float | 0 | The distance at which the reticle has no parallax, in the prim's units. 0 puts the reticle at infinity. |
| `lens:glareStrength` | float | 0 | Subtle glare from bright light near the axis, from 0 to 1. |
| `lens:dirtStrength` | float | 0 | Procedural dirt on the eyepiece, lit mainly by glare, from 0 to 1. |
| `lens:firstPersonOnly` | bool | false | Only first-person (view model) instances of the mesh get the lens. |

A minimal `mod.usda` entry. The eyepiece sits 12 units behind the replacement's origin, and the objective 30 units in
front of the eyepiece:

```usda
over "RootNode"
{
    over "meshes"
    {
        over "mesh_0123456789ABCDEF"
        {
            def Xform "ScopeLens" (
                prepend apiSchemas = ["RemixLensAPI"]
            )
            {
                double3 xformOp:translate = (0, 4, 12)
                uniform token[] xformOpOrder = ["xformOp:translate"]

                float lens:eyepieceRadius = 1.5
                float lens:objectiveDistance = 30
                float lens:objectiveRadius = 2
                float lens:magnification = 4
                float lens:eyeRelief = 8
                uniform token lens:reticle = "duplex"
                uniform bool lens:firstPersonOnly = 1
            }
        }
    }
}
```

The objective belongs just in front of the scope's front glass, since magnified rays start there and can hit the
scope's own geometry otherwise. Lens rays skip the view model, so a first-person scope never blocks its own image.

### Runtime magnification

`rtx.lens.magnificationScale` multiplies the magnification of every lens except the debug camera lens. A game
integration changes a sight's zoom at runtime by setting it through the Remix API, for example
`SetConfigVariable("rtx.lens.magnificationScale", "2.5")`. The value is written to the user layer and applies from
the next frame. A zoom step larger than `rtx.lens.historyDiscardZoomStep` discards the lens pixels' history for one
frame.

### Placement

The mod reader turns each lens prim into an `AssetReplacement` of type `eLens`, with the prim's transform to the
replacement's object space, the same space as its meshes. Inside a point instancer prototype, each instance gets its
own lens.

When `SceneManager::drawReplacements` builds a replacement instance with lenses, it registers them on an anchor: the
instance of the original draw, or of the first mesh prim that is not a point instancer prototype. The anchor stays
registered until that instance is freed, so the preserve path and retained replacements keep their lenses. After
the view model instances are made each frame, `SceneManager::addReplacementLenses` adds the lenses of every anchor
that is not hidden, from the anchor's current and previous transforms, so lens motion vectors follow the mesh. Each
lens' id comes from the anchor's instance id and the lens' index, so it keeps its slot for as long as the instance
lives.

- An anchor that is a view model reference is traced as its view model copy. Its lenses use the reference transform
  with the view model perspective correction applied (`InstanceManager::getViewModelTransforms`), which is where the
  copy is traced, even when the correction is projective and moves the copy's vertices instead of its transform.
  Lengths are measured on the moved lens, so they include the view model scale.
- A first-person-only lens is placed only on an anchor with a view model copy this frame, so other instances of the
  same mesh, such as one carried by another character, get none. Without `rtx.viewModel.enable` there are no view
  model copies, and first-person-only lenses never appear.
- Lenses are not placed on draws that the host GPU-instances, or on replacements that the retained scene expands
  for an API frontend's resolved draws.

## Lenses from the Remix API

A host that knows where its sight is, such as a game integration that tracks the player's aim, sends the lens with
its camera: each `remixapi_CameraInfoLensEXT` chained on the main camera's `remixapi_CameraInfo` is one lens, for
that frame only. A frame whose `SetupCamera` carries no lens has none. The fields match the `lens:*` attributes of
[Lenses in mod USD](#lenses-in-mod-usd), with the pose and lengths in world space and world units:

- `eyepieceCenter` is the eyepiece center, `axis` the optical axis from the eye toward the objective and `up` the
  reticle's up direction, which is made perpendicular to the axis.
- A zero `magnification`, `fieldStopDegrees`, `tint`, `reticleLineWidth` or `apertureAspect` takes its default: 1,
  18 degrees, white, 0.004 and 1. The other fields default to 0, as a zero-initialized struct has them.
- `reticle` is `REMIXAPI_CAMERA_LENS_RETICLE_NONE`, `_CROSSHAIR`, `_DUPLEX`, `_DOT`, `_CIRCLE_DOT`, `_MIL_DOT`,
  `_GERMAN`, `_DROP`, `_STADIA`, `_FRAME`, `_GRID` or `_DIAL`. Other values draw no reticle.
- `markings` takes `REMIXAPI_CAMERA_LENS_MARKING_LABELS` and `REMIXAPI_CAMERA_LENS_MARKING_RANGEFINDER`.
- `lensFlags` bit `REMIXAPI_CAMERA_LENS_FLAG_NO_GRAIN` turns off the vision mode's grain (`lens:noGrain`). Night
  vision is the only mode with grain, and thermal has none.
- `lensFlags` bit `REMIXAPI_CAMERA_LENS_FLAG_OCCLUDE_OUTSIDE` blacks out everything outside the lenses, and
  `REMIXAPI_CAMERA_LENS_FLAG_FIRST_FOCAL_PLANE` puts the reticle in the first focal plane.
- `referenceMagnification` is the first focal plane reticle's reference. 0 uses `minMagnification`.
- `parallaxDistance` (world units), `glareStrength` and `dirtStrength` add parallax and glass effects. 0 turns each off.
- `markingColor`, `markingRadiance` and `markingOpacity` color the markings. A zero color uses the accent, and a zero
  opacity means 1.
- `reticleTexture` is the path of a `REMIXAPI_CAMERA_LENS_RETICLE_TEXTURE` image. The runtime copies the string during
  `SetupCamera`.
- Two lenses can be up at once, for example both halves of a pair of binoculars, with `rtx.lens.maxLenses` 2 or more.
  The option is read every frame, so a host can raise it with `SetConfigVariable` while it needs the second lens.
  Each lens keeps its own slot, history, motion vectors, night vision meter and rangefinder.
- `reticleAccentColor`, `reticleAccentRadiance`, `edgeGlowColor`, `edgeGlowStrength` and `edgeShade` default to 0.
  A zero `edgeTint` means none, a zero `nightTint` uses `rtx.lens.nightVision.phosphorColor`, and a zero `nightGain`
  means 1.
- `kind` is `REMIXAPI_CAMERA_LENS_KIND_TELESCOPIC` or `_REFLEX`, `aperture` is `REMIXAPI_CAMERA_LENS_APERTURE_CIRCLE`,
  `_RECTANGLE` or `_BINOCULAR`, and `visionMode` is `REMIXAPI_CAMERA_LENS_VISION_NONE`, `_NIGHT` or `_THERMAL`. Other values take
  the first.
- `lensFlags` bit `REMIXAPI_CAMERA_LENS_FLAG_HIGHLIGHT_HOSTILES` turns on highlighting.
- `minMagnification` is for the host's zoom range. The runtime does not read it.
- `rtx.lens.magnificationScale` multiplies `magnification`, as it does for lenses in mod USD.

`id` must be nonzero and stay the same while the lens exists. The runtime keeps each id's pose of the previous frame
and passes it with the lens, so the lens' motion vectors and history follow it. A frame event
(`remixapi_CameraInfoFrameEventsEXT`) with `REMIXAPI_CAMERA_FRAME_EVENT_ORIGIN_SHIFT` moves that previous pose by
`originShift`. The id is hashed and tagged, so it never matches the id of a lens from mod USD
or of the debug camera lens.

A lens with id 0 or a non-finite value is skipped, and so is a lens on any camera but the main one. A lens sent
twice in one frame is added once. Lenses need `rtx.lens.enable`, and a runtime that predates the struct ignores it.
