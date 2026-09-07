/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// Cloud Render Compute Pass — Nubis Cubed 2023 (fork — 2026-05-12, C4).
//
// Per-pixel view-direction raymarch through the cloud slab using the Nubis
// Cubed 2023 lighting equations (paper pp. 137, 142). Writes premultiplied
// RGB + view-ray cloud transmittance alpha to AtmosphereCloudRender at the
// downscale render extent.
//
// The view-march replaced the old analytical sky-pass cloud rendering — the
// Wrenninge octave sum + isotropic deep-scatter floor (removed 2026-06-19) —
// with the page-137 two-HG-lobe direct term (sigma_ms remap on sun_dot + sdf
// depth) and the page-142 ambient pow(1 - dim_profile, 0.5) * exp(-D_ambient).
//
// Composited into primary sky-miss when cloudRenderRTEnable is on (gated by
// RTX_OPTION). A debug view (enum 876) shows the raw RT for inspection.
//
// Depth-aware march (fork — 2026-09-05, world-space cloud migration Stage 4a). This pass now runs
// AFTER RtxContext::dispatchPathTracing (RtxAtmosphere::dispatchCloudScreenPass, called from
// injectRTX), so PrimaryLinearViewZ holds a resolved primary-ray depth for every pixel this frame.
// The march below reads it (binding 15) and passes the converted distance as marchCloudLayers'
// tMaxClamp, so a cloud never marches past the geometry a primary ray already hit; a sky-miss
// pixel has no such geometry, so its tMax is 0 ("no clamp") and its result is UNCHANGED from Stage
// 3. The march also now writes a depth companion (binding 16, cloudDepthOutput): entry distance and
// a transmittance-weighted mean cloud depth, both km. Nothing composites it yet -- the sky-miss
// composite in evalSkyRadiance (atmosphere_sky.slangh) is untouched this stage; that is Stage 4b.
//
// Coordinate frame: ray-vs-slab intersection runs in camera-at-origin Y-up
// space (the established cloud-math convention — see `intersectCloudLayer`'s
// "Ray origin is at the camera (0,0,0) in Y-up space" comment), so the
// per-step `t` distances are measured from the camera. Per-sample positions
// passed to the density / lighting helpers, however, are WORLD-ANCHORED:
// `samplePos = args.cameraWorldPosYUpKm + viewDirYUp * t`. This anchors the
// noise / coverage / type fields and the D_sun / D_ambient voxel-grid
// lookups to the player's world position (km-scale Y-up coords pushed by
// setCloudShadowCameraPosition()) so the cloud volume translates with the
// world rather than rigidly with the camera — the parallax depth cue that
// distinguishes a 3D volume from cardboard cutouts.
//
// Per-pixel view directions are reconstructed CPU-side into the
// cloudRenderForwardYUp / RightYUp / UpYUp basis vectors in AtmosphereArgs;
// the shader builds viewDir per pixel from those basis vectors. Sun
// direction comes from args.sunDirection (already Y-up).

// Note: define ATMOSPHERE_AVAILABLE so the sampleDSun / sampleDAmbient
// helpers in atmosphere_common.slangh resolve to texture lookups (rather
// than the binding-free fallback stubs used by the LUT bakers). The
// global names the helpers reach for (AtmosphereCloudDSun, AtmosphereCloudDAmbient,
// AtmosphereCloudNoise3DSampler, BlueNoise)
// are declared at our local slots BELOW so they're in scope before the
// include's helper bodies are parsed.
#define ATMOSPHERE_AVAILABLE

// Bring in shader_types (for vec3/uint/etc.) and AtmosphereArgs ahead of the
// global texture declarations so AtmosphereArgs is defined before slot 0's CB.
#include "rtx/utility/shader_types.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// Linear/REPEAT sampler for the Nubis3 volume taps (matches the
// frac()-tile-wrap convention). Slot 1 held the legacy 256^3 noise volume —
// retired with the legacy model, 2026-07-16.
layout(binding = 2)
SamplerState AtmosphereCloudNoise3DSampler;

// Cloud voxel grids (D_sun and D_ambient), sampled via the same linear/REPEAT
// sampler. Names match the common_bindings.slangh globals so the helpers in
// atmosphere_common.slangh (sampleDSun / sampleDAmbient) resolve them.
layout(binding = 3)
Texture3D<float> AtmosphereCloudDSun;

layout(binding = 4)
Texture3D<float> AtmosphereCloudDAmbient;

layout(binding = 5)
Texture2DArray<float> BlueNoise;

layout(binding = 6, rgba16f)
RWTexture2D<float4> cloudRenderOutput;

// Sky-view LUT + cloud-sky-transmittance LUT for the page-142 ambient
// term. sampleSkyAmbientForVolume in atmosphere_common.slangh consumes
// these as explicit parameters (so the helper compiles in any TU).
// Linear/CLAMP sampler — sky-view LUT is keyed by (azimuth, elevation);
// REPEAT would alias the south pole onto the north.
layout(binding = 7)
Texture2D<float4> AtmosphereSkyViewLut;

layout(binding = 8)
Texture2D<float> AtmosphereCloudSkyTransmittanceLut;

layout(binding = 9)
SamplerState AtmosphereSkyViewSampler;

// Nubis3 NVDF/SDF model inputs (fork — Nubis3 conversion Phase B). The cloud
// body signed distance field (front buffer; raw km, negative inside, texture
// y = VERTICAL — see cloud_nvdf.h) and the 128^3 RGBA8 wispy/billowy detail
// volume. Consumed by sampleCloudDensityNubis3 via cloud_march_common's
// density-model branch; sampled with the linear/REPEAT cloud sampler (slot 2)
// — the SDF's vertical coordinate is half-texel clamped in the sampler code
// so REPEAT never wraps slab bottom into slab top.
layout(binding = 13)
Texture3D<float> AtmosphereCloudNvdfSdf;

layout(binding = 14)
Texture3D<float4> AtmosphereCloudDetailNoise3D;

// Depth-aware march inputs/outputs (fork — 2026-09-05, world-space cloud migration Stage 4a).
// Local slots 15/16 in THIS pass's own small descriptor set -- unrelated to the numbering of the
// shared common-ray-tracing bindings (BINDING_ATMOSPHERE_CLOUD_DEPTH_RT = 217 there; see
// common_binding_indices.h). PrimaryLinearViewZ is read-only here: RtxAtmosphere::dispatchCloudRender
// binds rtOutput.m_primaryLinearViewZ, the SAME resource RtxContext::dispatchPathTracing's
// G-buffer raytracing wrote for this frame just before this dispatch runs (see
// RtxAtmosphere::dispatchCloudScreenPass for why the ordering matters). cloudDepthOutput is this
// pass's own depth companion, written unconditionally every pixel just like cloudRenderOutput above.
layout(binding = 15)
Texture2D<float> PrimaryLinearViewZ;

// Widened to four channels (fork -- 2026-09-06, measured occlusion prefix): z = the distance of the
// full-res surface this texel stands for, w = the fraction of the ray's cloud opacity in front of
// it. z is what lets the composite pick a depth-consistent neighbour when the cloud pass runs below
// full resolution, instead of bilinear-blending a visibility across a silhouette.
layout(binding = 16, rgba32f)
RWTexture2D<float4> cloudDepthOutput;

// Include the helper library AFTER the texture globals are declared so the
// ATMOSPHERE_AVAILABLE-gated helper bodies resolve to our local declarations.
#include "rtx/pass/atmosphere/atmosphere_common.slangh"

// Shared slab-march + shade-context library (fork — 2026-06-10). Hosts
// marchCloudSlab / marchCloudLayers / buildCloudShadeContext /
// sampleCloudSunOpticalDepth_local — everything this pass shares with the
// secondary-ray cloud LUT bake (cloud_secondary_lut.comp.slang). Must
// follow atmosphere_common.slangh. The extraction is behavior-neutral:
// this pass calls marchCloudLayers with 0/0 distance clamps, which is
// bit-identical to the pre-extraction inline march.
//
// Lightning (fork — 2026-07-14): the screen pass is the ONLY consumer allowed
// to evaluate the transient in-cloud flash — the LUT bake persists across
// frames and must stay flash-free (see the gate comment in the header).
#define CLOUD_MARCH_LIGHTNING 1
#include "rtx/pass/atmosphere/cloud_march_common.slangh"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  uint2 pixelCoord = dispatchId.xy;
  uint2 outputDim;
  cloudRenderOutput.GetDimensions(outputDim.x, outputDim.y);
  if (any(pixelCoord >= outputDim)) return;

  AtmosphereArgs args = atmosphereArgs;

  // Trivially-clear sky: no clouds means full transmittance, zero radiance.
  // Effective coverage is layer 1's mean OR layer 2's mean (whichever is
  // enabled), so a "cirrus-only" preset (layer 1 coverage near zero, layer 2
  // enabled with its own coverage) still passes the early-out.
  float effectiveCoverageMean = args.cloudCoverageMean;
  if (args.cloudLayer2Enable != 0u) {
    effectiveCoverageMean = max(effectiveCoverageMean, args.cloudLayer2CoverageMean);
  }
  if (args.cloudEnabled < 0.5f || effectiveCoverageMean < 0.01f || args.cloudViewSamples == 0u) {
    cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    // Depth companion (fork — 2026-09-05, Stage 4a): (-1, 0) is the same "no entry / no cloud"
    // sentinel pair marchCloudLayers reports when it finds nothing -- written here too so every
    // texel of BOTH render targets is defined every frame, matching cloudRenderOutput's coverage.
    // Visibility 1, not 0: there is no cloud to occlude, and scaling nothing by one is still
    // nothing. Zero here would be a different kind of "no cloud" from the one loadCurrentCloudOpacity
    // already applies, for no benefit.
    cloudDepthOutput[pixelCoord] = vec4(-1.0f, 0.0f, kNoSurfaceKm, 1.0f);
    return;
  }

  // ===== Reconstruct per-pixel view ray in Y-up world space =====
  // NDC in [-1, 1] with top-left pixel mapping to (-1, +1) Vulkan convention.
  vec2 uv = (vec2(pixelCoord) + 0.5f) / vec2(outputDim);
  vec2 ndc = uv * 2.0f - 1.0f;
  ndc.y = -ndc.y;  // y-down pixels -> y-up NDC

  // Basis vectors are pre-multiplied by tan(halfFovX/Y) and aspect ratio
  // CPU-side, so the math is just a weighted sum.
  vec3 viewDirYUp = normalize(args.cloudRenderForwardYUp
                              + ndc.x * args.cloudRenderRightYUp
                              + ndc.y * args.cloudRenderUpYUp);

  // No below-horizon early-out (fork — 2026-09-05, world-space cloud migration Stage 1; ported
  // from archaeology commit 30d20a8f5).
  //
  // This used to read `if (viewDirYUp.y <= 0.001f) { write clear; return; }`, on the reasoning that
  // there are "no clouds above this pixel". That is true of the deck's TOP and irrelevant to its
  // UNDERSIDE, which from any eye below the deck fills every direction down to the true horizon.
  // Because the camera was welded to the surface (getEyeRadius(args) == args.planetRadius exactly —
  // still true this stage), the deck's far edge asymptoted to exactly y == 0, so cutting at y == 0
  // cut precisely along that edge and produced the hard horizon line the sky ended on — the single
  // most visible artifact of the old model.
  //
  // marchCloudSlab / marchEchoDeck now terminate these rays on the planet via cloudSlabSpan
  // (cloud_march_common.slangh) instead, which is a real occluder rather than a screen-space rule,
  // so the deck reaches the horizon and stops there on its own. Below-horizon sky radiance for the
  // page-142 ambient term (sampleSkyAmbientForVolume) is unaffected by this removal — that helper
  // has its own below-horizon clamp and is untouched this stage.

  // ===== The march is deliberately NOT clamped by geometry depth =====
  // (fork — 2026-09-05, world-space cloud migration Stage 4b fix.)
  //
  // Stage 4a clamped tMaxClamp to the slant range of the resolved surface, so a cloud never marched
  // past the geometry in front of it. That was exact, and it was wrong, for a reason that only
  // shows up in motion: it makes the contents of this render target a function of GEOMETRY. The
  // composite then runs a temporal EMA over that target and reprojects the history with a motion
  // vector built from the camera basis and the cloud's own depth -- which knows nothing about where
  // geometry moved. So whenever the depth buffer changed under a pixel, the EMA blended in a
  // history value computed against completely different occlusion, and the stale value took roughly
  // 1/(1-cloudHistoryWeight) frames to wash out. In practice: a heavy smear trailing the Pip-Boy as
  // it swings up, and cloud ghosting dragging behind world geometry while walking.
  //
  // A signal an EMA smooths must be a function of view direction only, so that rotation-only
  // reprojection is valid for it. This pass therefore marches the full reachable span and reports
  // entry / mean depth alongside it, and the OCCLUSION IS APPLIED PER-FRAME IN THE COMPOSITE, after
  // the EMA, where it carries no history (composite.comp.slang, applyCloudComposite). That is
  // already exactly how the alpha-blended path there worked.
  //
  // Cost: pixels with near geometry now march cloud that is later discarded. That is a real
  // regression in this pass's cost and belongs to Stage 6 -- a conservative clamp (a per-tile
  // maximum depth, say) could recover most of it without reintroducing a per-pixel geometry
  // dependence. Correctness first.
  //
  // PrimaryLinearViewZ (binding 15) stays bound but unread here, for that follow-up.

  float jitter = fastJitter(pixelCoord, args.cloudRenderFrameIdx).x;
  // Static per-pixel hash for far-field jitter (animated < 250 m, static beyond — no distant shimmer).
  float jitterStatic = fastJitter(pixelCoord, 0u).x;

  // ===== Static-per-frame shading context (cloud_march_common.slangh) =====
  // Sun/sky radiances, the full per-moon precompute, combined-moon shadow
  // direction, and the Wrenninge octave count — all pixel-independent, all
  // extracted verbatim into buildCloudShadeContext.
  CloudShadeContext ctx = buildCloudShadeContext(args);

  // Angular size of one output pixel (fork -- 2026-09-07), for the detail volume's
  // distance LOD. cloudRenderUpYUp is already pre-multiplied CPU-side by
  // tan(halfFovY), so it spans HALF the vertical view frustum at unit forward
  // distance: 2*|up| / outputDim.y radians per pixel row. This is the pass's OWN
  // output resolution, so it tracks cloudRenderResolutionScale automatically -- a
  // half-res cloud pass gets twice the footprint and one more mip level, which is
  // exactly right.
  ctx.pixelAngleRad = 2.0f * length(args.cloudRenderUpYUp) / float(max(outputDim.y, 1u));

  // ===== View-ray raymarch, both layers (cloud_march_common.slangh) =====
  // evalNubisCubedSampleCore inside marchCloudSlab uses sigma = args.cloudDensity
  // (no doubling — Nubis Cubed equations take raw cloudDensity as extinction).
  // marchCloudLayers sequences layer 1 (cumulus) then the optional layer-2
  // cirrus deck front-to-back, zeroing the moon coefficients for layer 2.
  // Both clamps are 0/no-op: the full reachable span is marched for every pixel, sky or geometry
  // alike, so this target stays a pure function of view direction and is therefore safe to smooth
  // temporally. See the no-clamp rationale above.
  // ===== Distance to the surface the primary ray hit =====
  // Read here, used ONLY to measure the occlusion prefix (fork -- 2026-09-06). This does not clamp
  // the march: the render target stays a full-span function of view direction, which is the
  // invariant the temporal EMA in the composite depends on and the reason the Stage 4a clamp was
  // removed. What geometry now affects is a single scalar the composite applies AFTER the EMA,
  // exactly where the occlusion ramp it replaces was already applied.
  float tSurfaceKm = kNoSurfaceKm;
  {
    uint2 depthDim;
    PrimaryLinearViewZ.GetDimensions(depthDim.x, depthDim.y);
    // Mapped through uv rather than assuming a shared grid, so this stays correct when the cloud
    // pass runs at a reduced resolution.
    const int2 depthCoord = clamp(int2(uv * vec2(depthDim)), int2(0), int2(depthDim) - int2(1));
    const float linearViewZ = PrimaryLinearViewZ[depthCoord];
    // Sky rays carry NRD's miss sentinel. It arrives through the CB rather than as a literal so it
    // cannot drift from the value NRD actually uses (this pass has no binding for NRD's own CB).
    if (linearViewZ < args.cloudMissLinearViewZ)
    {
      // Linear view Z is measured along the camera forward axis; the march needs slant range.
      const float cosToForward = max(dot(viewDirYUp, normalize(args.cloudRenderForwardYUp)), 1e-3f);
      tSurfaceKm = (abs(linearViewZ) / cosToForward) / max(args.worldUnitsPerKm, 1e-6f);
    }
  }

  vec3 accumColor = vec3(0.0f);
  float viewTransmittance = 1.0f;
  float entryDistanceKm;
  float meanCloudDepthKm;
  // Clamped at the primary surface (fork -- 2026-09-07). This restores the one line Stage 4a had
  // and e7298993f removed, and it is what makes cloud-vs-geometry occlusion EXACT rather than
  // estimated: the pass integrates [0, tSurface], so what it reports IS the cloud in front of the
  // surface. Every scalar-correction attempt before this failed because the in-scatter term
  // C_full * (alphaFront/alphaFull) assumes radiance per unit opacity is constant along the ray;
  // inside the deck it is not (near samples are shadowed interior, far ones lit edges), so the far
  // cloud's colour and pattern leaked onto the surface.
  //
  // Why this is safe now when it ghosted before: e7298993f blamed the clamp, but the actual defect
  // was that the temporal history had no way to notice the depth under a pixel had changed. The
  // design this migration follows always specified BOTH halves (docs/world-space-clouds-design.md
  // 4.4-B with 4.7) -- Stage 4a shipped only the clamp. The other half, per-tap depth validation in
  // fetchCloudHistoryBilinear, lands with this change. Rotation reprojection stays exact for a
  // clamped value, because under rotation with static geometry the surface depth along a world
  // direction is invariant; what changes it is moving objects and translation parallax, and the
  // depth test is precisely what rejects the former.
  marchCloudLayers(viewDirYUp, jitter, jitterStatic,
                   /*tMinClamp=*/0.0f,
                   /*tMaxClamp=*/tSurfaceKm,
                   args.cloudViewSamples, ctx, args,
                   accumColor, viewTransmittance,
                   entryDistanceKm, meanCloudDepthKm);

  // Output is premultiplied: rgb = accumulated cloud radiance,
  // a = view-ray cloud transmittance.
  cloudRenderOutput[pixelCoord] = vec4(accumColor, viewTransmittance);

  // Depth companion (fork — 2026-09-05, world-space cloud migration Stage 4a): r = entry distance,
  // g = transmittance-weighted mean cloud depth, both km. Written unconditionally so every texel is
  // defined every frame, matching cloudRenderOutput's coverage above. No consumer yet -- Stage 4b.
  // .z is the surface distance this texel was clamped at -- kept because a depth-aware upsample
  // needs it once the cloud pass runs below full resolution, and it is the natural thing to inspect
  // when the clamp misbehaves. .w retired with the measured-prefix experiment.
  cloudDepthOutput[pixelCoord] = vec4(entryDistanceKm, meanCloudDepthKm, tSurfaceKm, 0.0f);
}
