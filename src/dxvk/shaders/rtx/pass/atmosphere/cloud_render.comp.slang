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

layout(binding = 16, rg32f)
RWTexture2D<float2> cloudDepthOutput;

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
    cloudDepthOutput[pixelCoord] = vec2(-1.0f, 0.0f);
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

  // ===== Depth-aware march clamp (fork — 2026-09-05, world-space cloud migration Stage 4a) =====
  //
  // Resolution mismatch: this pass runs at cloudRenderResolutionScale x the downscale extent
  // (outputDim above), while PrimaryLinearViewZ lives at the FULL downscale extent --
  // args.cloudRenderFullDimX/Y, the very same published dims evalSkyRadiance uses to upsample THIS
  // pass's own output the other way (atmosphere_sky.slangh's cloudRenderRTEnable branch: uv =
  // (pixelCoord + 0.5) / fullDim, then bilinear-samples the smaller RT). uv here is that same
  // normalized coordinate already computed above for the view-ray reconstruction, so the mapping
  // into PrimaryLinearViewZ's resolution is just the reverse of that established convention:
  // primaryCoord = floor(uv * fullDim). NEAREST, not bilinear -- PrimaryLinearViewZ is a per-pixel
  // depth, and averaging two different surfaces' depths across a silhouette produces a value that
  // corresponds to neither, which is exactly the "cloud bleeding around silhouette edges" failure
  // mode a wrong mapping here would cause. Zero dims (first frames, extent not yet known) falls back
  // to "no clamp", mirroring evalSkyRadiance's own zero-dims fallback.
  float tMaxKm = 0.0f;  // 0 = no clamp (cloud_march_common.slangh's tMaxClamp convention)
  if (args.cloudRenderFullDimX != 0u && args.cloudRenderFullDimY != 0u) {
    ivec2 fullDim = ivec2(int(args.cloudRenderFullDimX), int(args.cloudRenderFullDimY));
    ivec2 primaryCoord = ivec2(uv * vec2(fullDim));
    primaryCoord = clamp(primaryCoord, ivec2(0), fullDim - ivec2(1));

    const float linearViewZ = PrimaryLinearViewZ[primaryCoord].x;

    // Miss sentinel (fork — 2026-09-05): exact equality against the SAME constant
    // NRDContext::getNrdArgs() (rtx_nrd_context.cpp) derives as
    // nrd::CommonSettings{}.denoisingRange + 1.0f, protected there by a static_assert pinning the
    // NRD library default -- mirrors debug_view.comp.slang enum 880/884's identical
    // `linearViewZ == cb.nrd.missLinearViewZ` test ("exact sentinel comparison, not a heuristic
    // threshold"). This pass has no binding for the RaytraceArgs CB that sentinel actually lives
    // in -- its own descriptor set carries only AtmosphereArgs + the cloud-specific textures above
    // -- so rather than plumb the whole CB through for one float, the compile-time-constant value
    // is reproduced directly. It is NOT runtime-configurable (the assert would fail to compile
    // before this could ever go stale silently).
    const float kPrimaryLinearViewZMissSentinel = 500001.0f;
    if (linearViewZ != kPrimaryLinearViewZMissSentinel) {
      // PrimaryLinearViewZ is a view-space DEPTH, not a distance along the ray -- divide by the
      // ray's projection onto the view (forward) axis to recover the slant range. debug_view.comp.slang
      // enum 884 does the identical conversion via cameraPixelCoordinateToDirection(cb.camera, ...);
      // this pass already carries the camera's Y-up forward basis vector (cloudRenderForwardYUp,
      // unit length -- see RtxAtmosphere::updateFrame's setCloudRenderCameraBasis push) and this
      // pixel's own Y-up view direction, so the same cosine-to-view-axis comes from those instead
      // of reconstructing a camera-space ray.
      const float viewAxisCosine = max(abs(dot(viewDirYUp, normalize(args.cloudRenderForwardYUp))), 1e-6f);
      const float cameraToSurfaceLength = linearViewZ / viewAxisCosine;
      const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKm, 1e-6f);
      tMaxKm = cameraToSurfaceLength * kmPerWorldUnit;
    }
    // else: sky-miss pixel, no surface to clamp against -- tMaxKm stays 0 ("no clamp"), so this
    // pixel's march is UNCHANGED from Stage 3. This is the acceptance-test invariant for this stage.
  }

  float jitter = fastJitter(pixelCoord, args.cloudRenderFrameIdx).x;
  // Static per-pixel hash for far-field jitter (animated < 250 m, static beyond — no distant shimmer).
  float jitterStatic = fastJitter(pixelCoord, 0u).x;

  // ===== Static-per-frame shading context (cloud_march_common.slangh) =====
  // Sun/sky radiances, the full per-moon precompute, combined-moon shadow
  // direction, and the Wrenninge octave count — all pixel-independent, all
  // extracted verbatim into buildCloudShadeContext.
  CloudShadeContext ctx = buildCloudShadeContext(args);

  // ===== View-ray raymarch, both layers (cloud_march_common.slangh) =====
  // evalNubisCubedSampleCore inside marchCloudSlab uses sigma = args.cloudDensity
  // (no doubling — Nubis Cubed equations take raw cloudDensity as extinction).
  // marchCloudLayers sequences layer 1 (cumulus) then the optional layer-2
  // cirrus deck front-to-back, zeroing the moon coefficients for layer 2.
  // tMinClamp stays 0/no-op (unchanged this stage — only the far bound is depth-aware); tMaxClamp
  // is now tMaxKm (fork — 2026-09-05, Stage 4a), 0 on a sky-miss pixel ("march the full slab",
  // bit-identical to the pre-Stage-4a march) and the converted slant range to the resolved surface
  // everywhere else.
  vec3 accumColor = vec3(0.0f);
  float viewTransmittance = 1.0f;
  float entryDistanceKm;
  float meanCloudDepthKm;

  marchCloudLayers(viewDirYUp, jitter, jitterStatic,
                   /*tMinClamp=*/0.0f,
                   /*tMaxClamp=*/tMaxKm,
                   args.cloudViewSamples, ctx, args,
                   accumColor, viewTransmittance,
                   entryDistanceKm, meanCloudDepthKm);

  // Output is premultiplied: rgb = accumulated cloud radiance,
  // a = view-ray cloud transmittance.
  cloudRenderOutput[pixelCoord] = vec4(accumColor, viewTransmittance);

  // Depth companion (fork — 2026-09-05, world-space cloud migration Stage 4a): r = entry distance,
  // g = transmittance-weighted mean cloud depth, both km. Written unconditionally so every texel is
  // defined every frame, matching cloudRenderOutput's coverage above. No consumer yet -- Stage 4b.
  cloudDepthOutput[pixelCoord] = vec2(entryDistanceKm, meanCloudDepthKm);
}
