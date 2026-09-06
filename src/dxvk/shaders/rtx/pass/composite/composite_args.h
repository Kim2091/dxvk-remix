/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "rtx/utility/shader_types.h"
#include "rtx/pass/volume_args.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"
#include "rtx/algorithm/accumulate.h"

#define DENOISER_MODE_OFF 0
#define DENOISER_MODE_RELAX 1
#define DENOISER_MODE_REBLUR 2

struct CompositeArgs {
  Camera camera;
  DomeLightArgs domeLightArgs;
  RayPortalHitInfo rayPortalHitInfos[maxRayPortalCount * 2];
  VolumeArgs volumeArgs;
  AccumulationArgs accumulationArgs;
  SparseRenderingArgs sparseRenderingArgs;
  // Needed for the aerial perspective volume's frustum basis and depth range.
  AtmosphereArgs atmosphereArgs;

  // -- Struct objects should go above this line to preserve alignment --

  // Fog
  vec3 fogColor;
  uint fogMode;

  float fogScale;
  float fogEnd;
  float fogDensity;
  float maxFogDistance;

  vec4 debugKnob;

  uint usePostFilter;
  uint demodulateRoughness;
  float roughnessDemodulationOffset;

  // One of DENOISER_MODE constants, affects signal conversion
  uint primaryDirectDenoiser;
  uint primaryIndirectDenoiser;
  uint secondaryCombinedDenoiser;  
  uint enableRtxdi;

  float primaryDirectMissLinearViewZ;
  uint enableReSTIRGI;
  float pixelHighlightReuseStrength;
  uint debugViewIdx;

  uint8_t compositePrimaryDirectDiffuse;
  uint8_t compositePrimaryDirectSpecular;
  uint8_t compositePrimaryIndirectDiffuse;
  uint8_t compositePrimaryIndirectSpecular;
  uint8_t compositeSecondaryCombinedDiffuse;
  uint8_t compositeSecondaryCombinedSpecular;
  // The number of active Ray Portals (Used for Ray Portal sampling). Always <= RAY_PORTAL_MAX_COUNT
  uint8_t numActiveRayPortals;
  uint8_t pad0;

  uint enableSeparatedDenoisers;
  uint frameIdx;

  uint outputSecondarySignalToParticleLayer;
  uint compositeVolumetricLight;
  uint outputParticleLayer;
  uint enableDemodulateAttenuation;

  uint enableStochasticAlphaBlend;
  uint stochasticAlphaBlendEnableFilter;
  uint stochasticAlphaBlendUseNeighborSearch;
  uint stochasticAlphaBlendSearchTheSameObject;

  uint stochasticAlphaBlendSearchIteration;
  float stochasticAlphaBlendInitialSearchRadius;
  float stochasticAlphaBlendRadiusExpandFactor;
  uint stochasticAlphaBlendShareNeighbors;

  float stochasticAlphaBlendNormalSimilarity;
  float stochasticAlphaBlendDepthDifference;
  float stochasticAlphaBlendPlanarDifference;
  uint stochasticAlphaBlendUseRadianceVolume;

  float stochasticAlphaBlendRadianceVolumeMultiplier;
  uint stochasticAlphaBlendDiscardBlackPixel;
  uint enhanceAlbedo;
  float skyBrightness;

  vec3 clearColorFinalColor;
  uint timeSinceStartMS;

  float alphaBlendSurfacePackMult; // for packing/unpacking hitT into Float16 in AlphaBlendSurface
  float postFilterThreshold;
  uint writeRayReconstructionHitDistance;
  float pad1;

  // Cloud composite parallax reprojection (fork — 2026-09-05, world-space cloud migration Stage 4b).
  // This frame's cloud-anchor world-space motion, Y-up km (RtxAtmosphere::getCloudAnchor().deltaKm
  // == posYUpKm - prevPosYUpKm; see rtx_atmosphere.h's CloudAnchor struct). NOT part of
  // AtmosphereArgs -- that struct cannot grow (see its own alignment-discipline comment) and this
  // value is composite-only, never feeding a bake or a LUT cache key. Needed because the reliable
  // reprojection matrix (cb.camera.prevWorldToProjection) only carries camera ROTATION on the
  // Gamebryo-family engine this migration targets: RtCamera::getPosition() reads a rotation-only D3D
  // view matrix (see RtxAtmosphere::updateFrame's anchor-resolution block), so a pure-direction
  // reprojection (as the existing rotation-only screen motion vector already does) is the only thing
  // that matrix can supply. The cloud RT now carries real depth (AtmosphereCloudDepth), so a
  // translating camera needs a real parallax correction on top of that rotation-only term -- this is
  // the one place a translation signal for that correction can come from. See
  // applyCloudComposite / cloudParallaxMotionVectorPixels in composite.comp.slang.
  vec3 cloudAnchorDeltaYUpKm;
  // Routes the cloud in-scatter composited onto GEOMETRY into the DLSS-RR transparency layer instead
  // of the denoised radiance (fork — 2026-09-05, ghosting follow-up; design register item 10). Rides
  // the former pad2 slot, so the CB layout is unchanged. Only consulted when outputParticleLayer is
  // set; see applyCloudComposite in composite.comp.slang. Mirrors
  // rtx.atmosphere.cloudCompositeRRTransparencyLayer.
  uint cloudCompositeRRTransparencyLayer;
};
