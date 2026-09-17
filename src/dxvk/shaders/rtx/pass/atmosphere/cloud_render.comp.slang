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

//!variant cloud_render.comp
//!variant cloud_render_no_moon_shadows.comp
//!> CLOUD_PROFILE_NO_MOON_SHADOWS=1
//!variant cloud_render_density_only.comp
//!> CLOUD_PROFILE_DENSITY_ONLY=1
//!variant cloud_render_wide.comp
//!> CLOUD_GROUP_X=16 CLOUD_GROUP_Y=4
//!variant cloud_render_small.comp
//!> CLOUD_GROUP_X=8 CLOUD_GROUP_Y=4
//!variant cloud_render_tight_bounds.comp
//!> CLOUD_TIGHT_DENSITY_BOUNDS=1
//!variant cloud_render_density_tight_bounds.comp
//!> CLOUD_TIGHT_DENSITY_BOUNDS=1 CLOUD_PROFILE_DENSITY_ONLY=1
//!end-variants

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
// NV-DXVK start: Use the active camera projection for Numos clouds.
// Cloud rays use the primary camera's inverse projection, without temporal jitter.
// Sun direction comes from args.sunDirection (already Y-up).
// NV-DXVK end

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
// NV-DXVK start: Use the active camera projection for Numos clouds.
#include "rtx/concept/camera/camera.slangh"
// NV-DXVK end

layout(binding = 0)
ConstantBuffer<AtmosphereArgs> atmosphereArgs;

// NV-DXVK start: Use the active camera projection for Numos clouds.
layout(binding = 17)
ConstantBuffer<Camera> cloudCamera;
// NV-DXVK end

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

// Previous frame's cloud RT + depth companion (fork -- 2026-09-16, temporal interleave). Read only
// on the pixels this frame does not march; see reprojectCloudPixel. RtxAtmosphere ping-pongs the
// two pairs, so these are the images the previous dispatch wrote, never the ones bound at 6 / 16.
layout(binding = 18)
Texture2D<float4> PrevCloudRender;

layout(binding = 19)
Texture2D<float4> PrevCloudDepth;

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

#ifndef CLOUD_GROUP_X
#define CLOUD_GROUP_X 8
#define CLOUD_GROUP_Y 8
#endif

struct CloudPixelResult {
  vec4 color;  // premultiplied cloud rgb, view-ray transmittance
  vec4 depth;  // entry km, transmittance-weighted mean km, surface km, unused
};

// Slant distance to the primary surface this pixel resolved, or kNoSurfaceKm on a sky miss.
float cloudSurfaceDistanceKm(vec2 uv, CameraDirections cloudRay, AtmosphereArgs args) {
  uint2 depthDim;
  PrimaryLinearViewZ.GetDimensions(depthDim.x, depthDim.y);
  // Mapped through uv rather than assuming a shared grid, so this stays correct when the cloud
  // pass runs at a reduced resolution.
  const int2 depthCoord = clamp(int2(uv * vec2(depthDim)), int2(0), int2(depthDim) - int2(1));
  const float linearViewZ = PrimaryLinearViewZ[depthCoord];
  // Sky rays carry NRD's miss sentinel. It arrives through the CB rather than as a literal so it
  // cannot drift from the value NRD actually uses (this pass has no binding for NRD's own CB).
  if (linearViewZ >= args.cloudMissLinearViewZ) {
    return kNoSurfaceKm;
  }
  // Linear view Z is measured along the camera forward axis; the march needs slant range.
  const float cosToForward = max(abs(cloudRay.viewDirection.z), 1e-3f);
  return (abs(linearViewZ) / cosToForward) / max(args.worldUnitsPerKm, 1e-6f);
}

CloudPixelResult marchCloudPixel(uint2 pixelCoord, vec3 viewDirYUp, float tSurfaceKm,
                                 bool reducedResolution, CloudShadeContext ctx, AtmosphereArgs args) {
  float jitterStatic = fastJitter(pixelCoord, 0u).x;
  // Reduced-resolution texels magnify animated march noise without cloud history accumulation.
  // Keep camera projection jitter in the caller: it must still match the primary geometry rays.
  float jitter = reducedResolution ? jitterStatic : fastJitter(pixelCoord, args.cloudRenderFrameIdx).x;

  // ===== View-ray raymarch, both layers (cloud_march_common.slangh) =====
  // evalNubisCubedSampleCore inside marchCloudSlab uses sigma = args.cloudDensity
  // (no doubling — Nubis Cubed equations take raw cloudDensity as extinction).
  // marchCloudLayers sequences layer 1 (cumulus) then the optional layer-2
  // cirrus deck front-to-back, zeroing the moon coefficients for layer 2.
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
  // 4.4-B with 4.7) -- Stage 4a shipped only the clamp. The other half, per-tap depth validation,
  // is what the composite's reduced-scale reconstruction and this pass's reprojection both apply.
  marchCloudLayers(viewDirYUp, jitter, jitterStatic,
                   /*tMinClamp=*/0.0f,
                   /*tMaxClamp=*/tSurfaceKm,
                   args.cloudViewSamples, ctx, args,
                   accumColor, viewTransmittance,
                   entryDistanceKm, meanCloudDepthKm);

  CloudPixelResult result;
  // Output is premultiplied: rgb = accumulated cloud radiance,
  // a = view-ray cloud transmittance.
  result.color = vec4(accumColor, viewTransmittance);
  // Depth companion (fork — 2026-09-05, world-space cloud migration Stage 4a): r = entry distance,
  // g = transmittance-weighted mean cloud depth, both km. .z is the surface distance this texel was
  // clamped at -- what lets the composite's reduced-scale upsample and this pass's own reprojection
  // pick depth-consistent neighbours. .w retired with the measured-prefix experiment.
  result.depth = vec4(entryDistanceKm, meanCloudDepthKm, tSurfaceKm, 0.0f);
  return result;
}

// Fills `result` from the previous frame's RT along this pixel's world direction (fork --
// 2026-09-16, temporal interleave). Rotation-exact for sky, because a direction reprojects without
// any depth; translation parallax on a deck kilometres away is far below a texel. A surface pixel
// additionally needs every tap to describe the same surface class at a matching distance, else the
// caller marches the pixel fresh -- silhouettes are always re-marched, never blended across.
bool reprojectCloudPixel(uint2 pixelCoord, uint2 outputDim, float tSurfaceKm, bool jitterRay,
                         AtmosphereArgs args, out CloudPixelResult result) {
  result.color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
  result.depth = vec4(-1.0f, 0.0f, tSurfaceKm, 0.0f);

  const vec2 uv = (vec2(pixelCoord) + 0.5f) / vec2(outputDim);
  // The same jitter choice as the fresh march, at both ends (fork -- 2026-09-17): a native-scale
  // pixel stands for its jittered ray, which is what the upscaler dejitters, so its previous value
  // is looked up along that ray through the previous jittered projection; the reduced-scale pass
  // marches unjittered rays, so it reprojects unjittered ones, and a camera at rest then maps every
  // texel onto exactly itself.
  const CameraDirections ray = cameraScreenUVToDirection(cloudCamera, uv, jitterRay);
  const vec4 prevClip = mul(jitterRay ? cloudCamera.prevWorldToProjectionJittered
                                      : cloudCamera.prevWorldToProjection,
                            vec4(ray.worldDirection, 0.0f));
  if (prevClip.w <= 1e-6f) {
    return false;
  }
  const vec2 prevUv = cameraNDCToScreenUV(cloudCamera, prevClip.xy / prevClip.w);
  if (any(prevUv < vec2(0.0f)) || any(prevUv >= vec2(1.0f))) {
    return false;
  }

  vec2 texelF = prevUv * vec2(outputDim) - 0.5f;
  // Float noise from the two matrix products lands a static camera a hair off its own texel;
  // snapping keeps that case an exact copy instead of a faint four-tap blur every other frame.
  const vec2 nearest = round(texelF);
  if (all(abs(texelF - nearest) < 0.01f)) {
    texelF = nearest;
  }
  const int2 baseTexel = int2(floor(texelF));
  const vec2 f = texelF - vec2(baseTexel);
  const int2 maxTexel = int2(outputDim) - int2(1);
  const float weights[4] = {
    (1.0f - f.x) * (1.0f - f.y), f.x * (1.0f - f.y),
    (1.0f - f.x) * f.y,          f.x * f.y };
  const int2 offsets[4] = { int2(0, 0), int2(1, 0), int2(0, 1), int2(1, 1) };
  const bool currentIsSky = tSurfaceKm >= kNoSurfaceKm;

  vec4 sum = vec4(0.0f);
  float weightSum = 0.0f;
  float cloudWeightSum = 0.0f;
  float meanSum = 0.0f;
  float entryKm = kNoSurfaceKm;
  [unroll]
  for (uint i = 0u; i < 4u; ++i) {
    const float weight = weights[i];
    if (weight <= 1e-4f) {
      continue;
    }
    const int2 texel = clamp(baseTexel + offsets[i], int2(0), maxTexel);
    const vec4 depth = PrevCloudDepth.Load(int3(texel, 0));
    const bool tapIsSky = depth.z >= kNoSurfaceKm;
    if (tapIsSky != currentIsSky) {
      return false;
    }
    if (!currentIsSky
        && abs(depth.z - tSurfaceKm) > args.cloudReprojectDepthTolerance * max(max(depth.z, tSurfaceKm), 1e-3f)) {
      return false;
    }
    const vec4 value = PrevCloudRender.Load(int3(texel, 0));
    sum += value * weight;
    weightSum += weight;
    // Depth must describe the same cloud taps as colour.
    const float cloudWeight = weight * saturate(1.0f - value.a);
    if (cloudWeight > 0.0f && depth.x >= 0.0f) {
      cloudWeightSum += cloudWeight;
      meanSum += depth.y * cloudWeight;
      entryKm = min(entryKm, depth.x);
    }
  }
  if (weightSum <= 1e-3f) {
    return false;
  }
  result.color = sum / weightSum;
  result.depth = cloudWeightSum > 1e-6f
    ? vec4(entryKm, meanSum / cloudWeightSum, tSurfaceKm, 0.0f)
    : vec4(-1.0f, 0.0f, tSurfaceKm, 0.0f);
  return true;
}

// Which pixel of a cell is marched this frame. Period 2 alternates the x parity per row and per
// frame (a checkerboard); period 4 walks the 2x2 diagonal-first, offset on alternate cells so the
// fresh pixels never line up in rows.
uint2 cloudFreshSubPixel(uint period, uint phase, uint2 cellCoord) {
  if (period == 2u) {
    return uint2((cellCoord.y + phase) & 1u, 0u);
  }
  const uint2 kQuadOrder[4] = { uint2(0u, 0u), uint2(1u, 1u), uint2(1u, 0u), uint2(0u, 1u) };
  return kQuadOrder[(phase + ((cellCoord.x + cellCoord.y) & 1u) * 2u) & 3u];
}

[shader("compute")]
[numthreads(CLOUD_GROUP_X, CLOUD_GROUP_Y, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
  uint2 outputDim;
  cloudRenderOutput.GetDimensions(outputDim.x, outputDim.y);

  AtmosphereArgs args = atmosphereArgs;

  // Temporal interleave (fork -- 2026-09-16, perf). Each thread owns a cell of `period` pixels and
  // marches exactly one of them; the others are reprojected from the previous frame, or marched
  // too when that fails. Keeping the split per THREAD rather than per pixel is what turns it into
  // time: this pass is warp-synchronous, so a per-lane skip would only idle lanes while the warp
  // still ran the longest march in it. RtxAtmosphere sets the period to 1 on any frame the cloud
  // inputs change, the camera cuts, or lightning is live, and clears the valid bit whenever the
  // other RT pair does not hold a complete previous frame at this extent.
  const uint rawPeriod = args.cloudScreenInterleave & 0xFFu;
  const uint period = rawPeriod == 4u ? 4u : (rawPeriod == 2u ? 2u : 1u);
  const uint phase = (args.cloudScreenInterleave >> 8u) & 0xFFu;
  const bool historyValid = (args.cloudScreenInterleave & 0x10000u) != 0u;
  const uint2 cellDim = uint2(period >= 2u ? 2u : 1u, period >= 4u ? 2u : 1u);
  const uint2 cellCoord = dispatchId.xy;
  const uint2 cellOrigin = cellCoord * cellDim;
  if (any(cellOrigin >= outputDim)) return;
  const uint2 freshSub = cloudFreshSubPixel(period, phase, cellCoord);

  // Trivially-clear sky: no clouds means full transmittance, zero radiance.
  // Effective coverage is layer 1's mean OR layer 2's mean (whichever is
  // enabled), so a "cirrus-only" preset (layer 1 coverage near zero, layer 2
  // enabled with its own coverage) still passes the early-out.
  float effectiveCoverageMean = args.cloudCoverageMean;
  if (args.cloudLayer2Enable != 0u) {
    effectiveCoverageMean = max(effectiveCoverageMean, args.cloudLayer2CoverageMean);
  }
  if (args.cloudEnabled < 0.5f || effectiveCoverageMean < 0.01f || args.cloudViewSamples == 0u) {
    for (uint sub = 0u; sub < period; ++sub) {
      const uint2 pixelCoord = cellOrigin + uint2(sub & 1u, sub >> 1u);
      if (any(pixelCoord >= outputDim)) continue;
      cloudRenderOutput[pixelCoord] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
      // Depth companion (fork — 2026-09-05, Stage 4a): (-1, 0) is the same "no entry / no cloud"
      // sentinel pair marchCloudLayers reports when it finds nothing -- written here too so every
      // texel of BOTH render targets is defined every frame, matching cloudRenderOutput's coverage.
      // Visibility 1, not 0: there is no cloud to occlude, and scaling nothing by one is still
      // nothing.
      cloudDepthOutput[pixelCoord] = vec4(-1.0f, 0.0f, kNoSurfaceKm, 1.0f);
    }
    return;
  }

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

  const bool reducedResolution = outputDim.x < args.cloudRenderFullDimX
                              || outputDim.y < args.cloudRenderFullDimY;
  // Reduced-scale sampling (fork -- 2026-09-17). Part of the smaller texel count buys finer steps:
  // the spacing and adaptive floor shrink by cloudScreenStepScale and the cap grows to match, on
  // this pass's copy of the args only (the dome and the bakes keep their own spacing).
  if (reducedResolution && args.cloudScreenStepScale > 0.0f && args.cloudScreenStepScale < 1.0f) {
    args.cloudViewStepKm      *= args.cloudScreenStepScale;
    args.nubis3AdaptiveStepKm *= args.cloudScreenStepScale;
    args.cloudViewSamplesMax  /= args.cloudScreenStepScale;
  }
  // The projection jitter is a sub-pixel term at native scale but a 1/(2*scale)-texel wobble on
  // every reduced-scale texel's ray, which flickers a still camera's blocks through the jitter
  // sequence; the reduced pass marches the unjittered ray instead. The surface clamp still reads
  // the jittered depth at the texel centre, and the composite reconstructs against that depth.
  const bool jitterRay = !reducedResolution;

  // ===== Static-per-frame shading context (cloud_march_common.slangh) =====
  // Sun/sky radiances, the full per-moon precompute, combined-moon shadow
  // direction, and the Wrenninge octave count — all pixel-independent, built once per thread.
  CloudShadeContext ctx = buildCloudShadeContext(args);

  for (uint sub = 0u; sub < period; ++sub) {
    const uint2 subOffset = uint2(sub & 1u, sub >> 1u);
    const uint2 pixelCoord = cellOrigin + subOffset;
    if (any(pixelCoord >= outputDim)) continue;

// NV-DXVK start: Use the active camera projection for Numos clouds.
    // Normalized UV supports a cloud target smaller than the primary camera resolution.
    const vec2 uv = (vec2(pixelCoord) + 0.5f) / vec2(outputDim);
    // At native scale this matches the primary depth ray, including its temporal jitter, at
    // geometry silhouettes; see jitterRay above for the reduced-scale choice.
    const CameraDirections cloudRay = cameraScreenUVToDirection(cloudCamera, uv, jitterRay);
// NV-DXVK end
    // Clamp to the primary surface; composite validates its taps against this depth.
    const float tSurfaceKm = cloudSurfaceDistanceKm(uv, cloudRay, args);

    CloudPixelResult result;
    const bool fresh = !historyValid || all(subOffset == freshSub);
    if (fresh || !reprojectCloudPixel(pixelCoord, outputDim, tSurfaceKm, jitterRay, args, result)) {
      const vec3 viewDirYUp = worldToAtmosphereYUp(
        cloudRay.worldDirection, args.isZUp != 0u, args.flipUpAxis != 0);
      result = marchCloudPixel(pixelCoord, viewDirYUp, tSurfaceKm, reducedResolution, ctx, args);
    }
    cloudRenderOutput[pixelCoord] = result.color;
    cloudDepthOutput[pixelCoord] = result.depth;
  }
}
