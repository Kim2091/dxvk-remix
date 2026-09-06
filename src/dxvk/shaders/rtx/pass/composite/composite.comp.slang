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
#include "rtx/pass/composite/composite_bindings.slangh"

#include "rtx/pass/composite/composite_args.h"
#include "rtx/pass/composite/composite.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/color.slangh"
#include "rtx/utility/froxel.slangh"
#include "rtx/utility/brdf.slangh"
#include "rtx/utility/demodulate_helpers.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/surface/alpha_blend_surface.slangh"
#include "rtx/utility/geometry_flags.slangh"

#include "rtx/algorithm/volume_composite_helpers.slangh"
#include "rtx/algorithm/accumulate.slangh"
#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/sparse_rendering/sparse_rendering.slangh"
#include "rtx/pass/sparse_rendering/compact_active_pixels.slangh"

float3 sampleDomeLightTexture(Sampler2D<float3> DomeLight, float3 worldDirection, float4x4 worldToDomeLightTransform)
{
  float3 domeSampleDirection = mul(worldToDomeLightTransform, float4(worldDirection, 0.0f)).xyz;
  float2 sampleUV = cartesianDirectionToLatLongSphere(domeSampleDirection);
  return DomeLight.SampleLevel(sampleUV, 0).xyz;
}

void preCompositeDebugView(uint2 pixelCoordinate)
{
  // Debug views that need to run before the union-active early-out so they cover every pixel.
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_SPARSE_RENDERING_DIRECT_PIXEL_SAMPLING_RATE:
    storeInDebugView(pixelCoordinate, SparseRenderingDirectPixelSamplingRate[pixelCoordinate]);
    break;
  case DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_PIXEL_SAMPLING_RATE:
    storeInDebugView(pixelCoordinate, SparseRenderingIndirectPixelSamplingRate[pixelCoordinate]);
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_RADIANCE:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_DIFFUSE_HIT_T:
    if (cb.enableSeparatedDenoisers)
    {
      storeInDebugView(pixelCoordinate, PrimaryIndirectDiffuseRadianceHitDistance[pixelCoordinate]);
    }
    break;
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_RADIANCE:
  case DEBUG_VIEW_DENOISED_PRIMARY_INDIRECT_SPECULAR_HIT_T:
    if (cb.enableSeparatedDenoisers)
    {
      storeInDebugView(pixelCoordinate, PrimaryIndirectSpecularRadianceHitDistance[pixelCoordinate]);
    }
    break;

  case DEBUG_VIEW_SPARSE_RENDERING_DIRECT_ACTIVE_PIXELS_PERCENTAGE:
    storeInDebugView(pixelCoordinate, SparseRenderingDirectPixelSamplingRate[pixelCoordinate]);
    break;

  case DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_ACTIVE_PIXELS_PERCENTAGE:
    storeInDebugView(pixelCoordinate, SparseRenderingIndirectPixelSamplingRate[pixelCoordinate]);
    break;

  case DEBUG_VIEW_SPARSE_RENDERING_DIRECT_ACTIVE_PIXELS_OUTPUT_SCALE:
  {
    const float rate = SparseRenderingDirectPixelSamplingRate[pixelCoordinate];
    storeInDebugView(pixelCoordinate, rate > 0.f ? 1.0 / rate : 0.f);
    break;
  }

  case DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_ACTIVE_PIXELS_OUTPUT_SCALE:
  {
    const float rate = SparseRenderingIndirectPixelSamplingRate[pixelCoordinate];
    storeInDebugView(pixelCoordinate, rate > 0.f ? 1.0 / rate : 0.f);
    break;
  }

  case DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_PIXELS_MASK:
  {
    if (isActivePixelFromMask(SparseRenderingActivePixelMask, pixelCoordinate))
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;
  }

  case DEBUG_VIEW_NAN:
  {
    bool isValid = true;
    const float directRate = SparseRenderingDirectPixelSamplingRate[pixelCoordinate];
    const float indirectRate = SparseRenderingIndirectPixelSamplingRate[pixelCoordinate];
    // DEBUG_VIEW_SPARSE_RENDERING_DIRECT_PIXEL_SAMPLING_RATE / _DIRECT_ACTIVE_PIXELS_PERCENTAGE
    // DEBUG_VIEW_SPARSE_RENDERING_DIRECT_ACTIVE_PIXELS_OUTPUT_SCALE
    isValid &= isValidValue(directRate);
    // DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_PIXEL_SAMPLING_RATE / _INDIRECT_ACTIVE_PIXELS_PERCENTAGE
    // DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_ACTIVE_PIXELS_OUTPUT_SCALE
    isValid &= isValidValue(indirectRate);
    // DEBUG_VIEW_SPARSE_RENDERING_ACTIVE_PIXELS_MASK
    isValid &= isValidValue(float(isActivePixelFromMask(SparseRenderingActivePixelMask, pixelCoordinate)));

    accumulateInDebugViewAnd(pixelCoordinate, isValid);
    break;
  }
  }
}

void updateAlphaBlendDebugView(
  uint2 pixelCoordinate,
  AlphaBlendSurface surface)
{
  // Output Debug Information
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_COLOR:
    storeInDebugView(pixelCoordinate, surface.isValid() ? surface.color.rgb : f16vec3(1,0,1));
    break;
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_NORMAL:
    storeInDebugView(pixelCoordinate, surface.isValid() ? surface.normal : f16vec3(0));
    break;
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_GEOMETRY_HASH:
    storeInDebugView(pixelCoordinate, r5g6b5ToColor(surface.geometryHash));
    break;    
  case DEBUG_VIEW_STOCHASTIC_ALPHA_BLEND_BACKGROUND_TRANSPARENCY:
    storeInDebugView(pixelCoordinate, surface.backgroundTransparency);
    break;
  }
}

void postCompositeDebugView(
  uint2 pixelCoordinate,
  GeometryFlags geometryFlags,
  vec4 particleLayer,
  bool pixelIsActive,
  bool isDirectActive,
  bool isIndirectActive)
{
  // Output Debug Information
  switch(cb.debugViewIdx)
  {
  case DEBUG_VIEW_GEOMETRY_FLAGS_FIRST_SAMPLED_LOBE_IS_SPECULAR:
    storeInDebugView(pixelCoordinate, geometryFlags.firstSampledLobeIsSpecular);
    break;
  case DEBUG_VIEW_RAY_RECONSTRUCTION_PARTICLE_LAYER:
    storeInDebugView(pixelCoordinate, particleLayer);
    break;
  case DEBUG_VIEW_RAY_RECONSTRUCTION_PARTICLE_LAYER_ALPHA:
    storeInDebugView(pixelCoordinate, 1.0 - particleLayer.w);
    break;
  case DEBUG_VIEW_RAY_RECONSTRUCTION_HIT_DISTANCE:
    if (cb.writeRayReconstructionHitDistance)
    {
      storeInDebugView(pixelCoordinate, RayReconstructionHitDistanceOutput[pixelCoordinate] / 1000.0);
    }
    break;
  case DEBUG_VIEW_SPARSE_RENDERING_UNION_ACTIVE_THREADS:
    if (pixelIsActive)
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;
  case DEBUG_VIEW_SPARSE_RENDERING_DIRECT_ACTIVE_THREADS:
    if (pixelIsActive && isDirectActive)
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;
  case DEBUG_VIEW_SPARSE_RENDERING_INDIRECT_ACTIVE_THREADS:
    if (pixelIsActive && isIndirectActive)
    {
      storeInDebugView(pixelCoordinate, 1.0);
    }
    break;

  case DEBUG_VIEW_NAN:
  {
    bool isValid = true;
    // DEBUG_VIEW_GEOMETRY_FLAGS_FIRST_SAMPLED_LOBE_IS_SPECULAR
    isValid &= isValidValue(float(geometryFlags.firstSampledLobeIsSpecular));
    // DEBUG_VIEW_RAY_RECONSTRUCTION_PARTICLE_LAYER / _ALPHA
    isValid &= isValidValue(particleLayer);
    // DEBUG_VIEW_RAY_RECONSTRUCTION_HIT_DISTANCE
    if (cb.writeRayReconstructionHitDistance)
    {
      isValid &= isValidValue(RayReconstructionHitDistanceOutput[pixelCoordinate]);
    }

    accumulateInDebugViewAnd(pixelCoordinate, isValid);
    break;
  }
  }
}

// Loads and remodulates the secondary (PSR) radiance contribution.
// Runs unconditionally on PSR pixels regardless of the active-pixel mask so the RR particle
// layer matches SR-off behavior even when the pixel is sparse-inactive.
vec3 loadSecondaryRadiance(
  uint2 pixelCoordinate,
  GeometryFlags geometryFlags,
  vec3 secondaryAlbedo,
  vec3 secondarySpecularAlbedo,
  vec3 secondaryAttenuation)
{
  if (!geometryFlags.secondarySurfaceMask)
  {
    return vec3(0);
  }

  vec3 secondaryCombinedDiffuseRadiance = SecondaryCombinedDiffuseRadianceHitDistance[pixelCoordinate].xyz;
  vec3 secondaryCombinedSpecularRadiance = SecondaryCombinedSpecularRadianceHitDistance[pixelCoordinate].xyz;

  unpackDenoiserRadiance(secondaryCombinedDiffuseRadiance, cb.secondaryCombinedDenoiser);
  unpackDenoiserRadiance(secondaryCombinedSpecularRadiance, cb.secondaryCombinedDenoiser);

  if (!cb.compositeSecondaryCombinedDiffuse)
  {
    secondaryCombinedDiffuseRadiance = vec3(0);
  }
  if (!cb.compositeSecondaryCombinedSpecular)
  {
    secondaryCombinedSpecularRadiance = vec3(0);
  }

  return (secondaryCombinedDiffuseRadiance * secondaryAlbedo +
          secondaryCombinedSpecularRadiance * secondarySpecularAlbedo) * secondaryAttenuation;
}

// Loads, combines, and remodulates the primary direct + indirect radiance contribution.
// Caller must gate this on pixelIsActive — direct/indirect inputs are only valid on active pixels.
vec3 loadPrimaryRadiance(
  uint2 pixelCoordinate,
  bool isDirectActive,
  bool isIndirectActive,
  vec3 primaryAlbedo,
  vec3 primarySpecularAlbedo,
  vec3 primaryAttenuation,
  float perceptualRoughness,
  bool primaryMiss)
{
  vec3 primaryDirectDiffuseRadiance = vec3(0);
  vec3 primaryDirectSpecularRadiance = vec3(0);

  // In combined-denoiser mode the indirect radiance is combined in the direct buffer 
  const bool readPrimaryDirect = isDirectActive || (!cb.enableSeparatedDenoisers && isIndirectActive);
  if (readPrimaryDirect)
  {
    primaryDirectDiffuseRadiance = PrimaryDirectDiffuseRadianceHitDistance[pixelCoordinate].xyz;
    primaryDirectSpecularRadiance = PrimaryDirectSpecularRadianceHitDistance[pixelCoordinate].xyz;

    unpackDenoiserRadiance(primaryDirectDiffuseRadiance, cb.primaryDirectDenoiser);
    unpackDenoiserRadiance(primaryDirectSpecularRadiance, cb.primaryDirectDenoiser);
  }

  vec3 primaryIndirectDiffuseRadiance = vec3(0);
  vec3 primaryIndirectSpecularRadiance = vec3(0);

  if (cb.enableSeparatedDenoisers && isIndirectActive)
  {
    if (cb.usePostFilter)
    {
      primaryIndirectDiffuseRadiance = applyPostFilter(PrimaryIndirectDiffuseRadianceHitDistance, pixelCoordinate, cb.primaryIndirectDenoiser);
      primaryIndirectSpecularRadiance = applyPostFilter(PrimaryIndirectSpecularRadianceHitDistance, pixelCoordinate, cb.primaryIndirectDenoiser);
    }
    else
    {
      primaryIndirectDiffuseRadiance = PrimaryIndirectDiffuseRadianceHitDistance[pixelCoordinate].xyz;
      primaryIndirectSpecularRadiance = PrimaryIndirectSpecularRadianceHitDistance[pixelCoordinate].xyz;

      unpackDenoiserRadiance(primaryIndirectDiffuseRadiance, cb.primaryIndirectDenoiser);
      unpackDenoiserRadiance(primaryIndirectSpecularRadiance, cb.primaryIndirectDenoiser);
    }
  }

  if (!cb.compositePrimaryDirectDiffuse)
  {
    primaryDirectDiffuseRadiance = vec3(0);
  }
  if (!cb.compositePrimaryDirectSpecular)
  {
    primaryDirectSpecularRadiance = vec3(0);
  }
  if (!cb.compositePrimaryIndirectDiffuse)
  {
    primaryIndirectDiffuseRadiance = vec3(0);
  }
  if (!cb.compositePrimaryIndirectSpecular)
  {
    primaryIndirectSpecularRadiance = vec3(0);
  }

  vec2 bsdfFactor = cb.enableRtxdi ? BSDFFactor[pixelCoordinate].xy : vec2(1);
  vec2 bsdfFactor2 = cb.enableReSTIRGI ? BSDFFactor2[pixelCoordinate].xy : vec2(1);

  if (cb.enhanceAlbedo && cb.enableReSTIRGI)
  {
    // Note: this is not a reachable path when Sparse Rendering is enabled (bsdfFactor2.x == 1.0f),
    // since only NRC path is supported. Otherwise, this would have to be done for inactive pixels as well.
    PrimaryAlbedo[pixelCoordinate].xyz = primaryAlbedo * sqrt(bsdfFactor2.x);
  }

  vec3 primaryCombinedDiffuseRadiance;
  vec3 primaryCombinedSpecularRadiance;

  if (cb.enableSeparatedDenoisers)
  {
    primaryCombinedDiffuseRadiance = primaryDirectDiffuseRadiance * bsdfFactor.x + primaryIndirectDiffuseRadiance * bsdfFactor2.x;
    primaryCombinedSpecularRadiance = primaryDirectSpecularRadiance * bsdfFactor.y + primaryIndirectSpecularRadiance * bsdfFactor2.y;
  }
  else
  {
    bsdfFactor.x = clamp(bsdfFactor.x, 0.7f, 1.3f);
    bsdfFactor = 0.5f * (bsdfFactor + bsdfFactor2);
    primaryCombinedDiffuseRadiance = primaryDirectDiffuseRadiance * bsdfFactor.x;
    primaryCombinedSpecularRadiance = primaryDirectSpecularRadiance * bsdfFactor.y;
  }

  float roughnessFactor = 1.0f;
  if (cb.demodulateRoughness)
  {
    roughnessFactor = getRoughnessDemodulationFactor(perceptualRoughness, cb.roughnessDemodulationOffset);
  }

  vec3 remodulatedTotalPrimaryRadiance =
    (primaryCombinedDiffuseRadiance * primaryAlbedo +
     primaryCombinedSpecularRadiance * primarySpecularAlbedo / roughnessFactor) * primaryAttenuation;

  if (cb.enableReSTIRGI)
  {
    float specularWeight = pow(perceptualRoughness, cb.pixelHighlightReuseStrength);
    LastFinalOutput[pixelCoordinate] = primaryMiss
      ? vec4(0)
      : vec4(primaryIndirectDiffuseRadiance, calcBt709Luminance(primaryIndirectSpecularRadiance) / roughnessFactor * specularWeight);
  }

  return primaryMiss ? cb.clearColorFinalColor : remodulatedTotalPrimaryRadiance;
}

void loadInputRadiance(
  uint2 pixelCoordinate,
  GeometryFlags geometryFlags,
  bool pixelIsActive,
  bool isDirectActive,
  bool isIndirectActive,
  vec3 primaryAlbedo,
  vec3 primarySpecularAlbedo,
  vec3 primaryAttenuation,
  vec3 secondaryAlbedo,
  vec3 secondarySpecularAlbedo,
  vec3 secondaryAttenuation,
  float perceptualRoughness,
  bool primaryMiss,
  out vec3 sharedRadiance,
  out vec3 remodulatedTotalPrimaryRadiance,
  out vec3 remodulatedTotalSecondaryRadiance)
{
  sharedRadiance = vec3(SharedRadianceRG[pixelCoordinate].xy, SharedRadianceB[pixelCoordinate]);

  // Secondary always runs on PSR pixels (independent of active-pixel mask) so RR particle
  // layer matches SR-off behavior.
  remodulatedTotalSecondaryRadiance = loadSecondaryRadiance(
    pixelCoordinate, geometryFlags,
    secondaryAlbedo, secondarySpecularAlbedo, secondaryAttenuation);

  remodulatedTotalPrimaryRadiance = vec3(0);
  if (pixelIsActive)
  {
    remodulatedTotalPrimaryRadiance = loadPrimaryRadiance(
      pixelCoordinate,
      isDirectActive, isIndirectActive,
      primaryAlbedo, primarySpecularAlbedo, primaryAttenuation,
      perceptualRoughness, primaryMiss);
  }
}

void applyVolumetricLighting(
  uint2 pixelCoordinate,
  inout RNG randomState,
  CameraDirections primaryRayDirections,
  bool pixelIsActive,
  bool primaryMiss,
  float viewDistance,
  float radianceScale,
  float volumetricRadianceScale,
  vec3 remodulatedTotalSecondaryRadiance,
  vec3 sharedRadiance,
  out vec3 volumetricPreintegratedRadiance,
  out vec3 volumeAttenuation,
  out vec3 alphaBlendOutput,
  out float backgroundAlpha,
  out vec3 radianceOutput,
  out vec4 particleLayerOutput,
  // Alpha-blend surface hit distance, world/game units, and whether one was actually found at this
  // pixel (fork — 2026-09-05, world-space cloud migration Stage 4b). AlphaBlendSurface is otherwise
  // fully local to this function; applyCloudComposite needs the surface's own distance to decide how
  // much of the cloud it should be fogged by (see cloudDepthVisibility) -- foliage rendered crisp
  // through a cloud the ground beneath it is already fogged by is the artefact this exists to avoid.
  out bool alphaBlendSurfaceValid,
  out float alphaBlendHitDistance)
{
  alphaBlendSurfaceValid = false;
  alphaBlendHitDistance = 0.0f;

  // Override the default volume sampling RNG seed offset
  setVolumeSamplingRngSeed(1);

  // Calculate volumetric radiance and attenuation if volumetric lighting is enabled

  volumetricPreintegratedRadiance = vec3(0);
  volumeAttenuation = vec3(1);
  alphaBlendOutput = vec3(0);
  backgroundAlpha = 1.0f;

  // Note: This is only evaluated when volumetrics are enabled, no external check is needed.
  // Todo: Using Primary linear view Z for now, not the best option as this does not work with PSR (among other effects like
  // Portals and particles), but fine for initial testing.
  // Skipped at union-inactive pixels: the integration is expensive and its result only contributes
  // through paths gated on remodulated radiance, which is zero at those pixels anyway.
  // When sparse volumetrics is disabled for this pixel's primary classification, volumetrics run at
  // every pixel (full rate) and are not scaled below.
  const bool fullRateVolumetrics = primaryMiss
    ? !bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryMiss)
    : !bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryHit);
  if (pixelIsActive || fullRateVolumetrics)
  {
    vec2 volumeUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
    volumeUV.x /= cb.volumeArgs.numFroxelVolumes;
    integrateVolumetricNEE(
      randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
      cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, primaryMiss, cameraGetWorldPosition(cb.camera), viewDistance, primaryRayDirections.worldDirection,
      volumetricPreintegratedRadiance, volumeAttenuation);
  }

  // Stochastic alpha blend

  if (cb.enableStochasticAlphaBlend)
  {    
    AlphaBlendSurface surface = AlphaBlendSurface.createFromPacked(AlphaBlendGBuffer[pixelCoordinate], cb.alphaBlendSurfacePackMult);
    vec3 cameraPosition = cameraGetWorldPosition(cb.camera);
    vec4 centerLight = AlphaBlendRadiance[pixelCoordinate];
    bool discardPixel = cb.stochasticAlphaBlendDiscardBlackPixel && all(centerLight.xyz == 0);

    if (!discardPixel && surface.isValid())
    {
      vec3 particleNormal = surface.normal;
      backgroundAlpha = surface.backgroundTransparency;
      alphaBlendSurfaceValid = true;
      alphaBlendHitDistance = surface.hitT;

      vec4 totalLight = vec4(centerLight.xyz, 1);

      if (cb.stochasticAlphaBlendEnableFilter)
      {
        float minHitT = surface.hitT;
        float maxHitT = surface.hitT;
        float4 multilayerLight = totalLight;
        for (int i = 0; i < 25; ++i)
        {
          int2 offset = int2(i % 5, i / 5) - 2;
          if (all(offset == 0))
          {
            continue;
          }
          int2 neighborPixel = clamp(int2(pixelCoordinate) + offset * 2, int2(0), cb.camera.resolution - 1);

          AlphaBlendSurface neighborSurface = AlphaBlendSurface.createFromPacked(AlphaBlendGBuffer[neighborPixel], cb.alphaBlendSurfacePackMult);
          if (neighborSurface.isValid())
          {
            const CameraDirections neighborRayDirections = cameraPixelCoordinateToDirection(cb.camera, neighborPixel.xy);
            vec3 neighborPosition = cameraPosition + neighborSurface.hitT * neighborRayDirections.worldDirection;

            const float transparencyBandwidth = 0.05;
            const float distanceWeight = 0.5;
            float weight = exp(
              -abs(neighborSurface.backgroundTransparency - surface.backgroundTransparency) / transparencyBandwidth +
              // Note: Convert offset to float to avoid need for integer dot product Vulkan feature to be enabled.
              -distanceWeight * dot(float2(offset), float2(offset))
            );

            minHitT = min(minHitT, neighborSurface.hitT);
            maxHitT = max(maxHitT, neighborSurface.hitT);

            vec4 neighborLight = AlphaBlendRadiance[neighborPixel];
            multilayerLight += vec4(neighborLight.xyz * weight, weight);
          }
        }
        multilayerLight.rgb /= multilayerLight.w;

        float depthRange = maxHitT - minHitT;
        float depthBandwidth = 0.05;
        totalLight.xyz = lerp(totalLight.xyz, multilayerLight.xyz, saturate(depthRange / surface.hitT / depthBandwidth));
      }
      alphaBlendOutput = totalLight.xyz;

      if (cb.volumeArgs.enable)
      {
        float alphaBlendSurfaceLinearViewZ = surface.hitT * primaryRayDirections.viewDirection.z;

        vec2 volumeUV = cameraPixelCoordinateToScreenUV(cb.camera, pixelCoordinate);
        volumeUV.x /= cb.volumeArgs.numFroxelVolumes;
        
        vec3 alphaVolumetricPreintegratedRadiance = vec3(0.0f, 0.0f, 0.0f);
        vec3 alphaVolumetricAttenuation = vec3(1.0f, 1.0f, 1.0f);
        integrateVolumetricNEE(
          randomState, VolumeFilteredRadianceAge, VolumeFilteredRadianceY, VolumeFilteredRadianceCoCg,
          cb.volumeArgs, cb.timeSinceStartMS, pixelCoordinate, volumeUV, false, cameraGetWorldPosition(cb.camera), surface.hitT, primaryRayDirections.worldDirection,
          alphaVolumetricPreintegratedRadiance, alphaVolumetricAttenuation);

        alphaBlendOutput *= alphaVolumetricAttenuation * (1 - backgroundAlpha);
        alphaBlendOutput += alphaVolumetricPreintegratedRadiance * (1 - backgroundAlpha);
      }
      else
      {
        const vec4 fogColor = calculateFog(surface.hitT);
        alphaBlendOutput = alphaBlendOutput * (1.0f - fogColor.a) + fogColor.rgb * fogColor.a;
        alphaBlendOutput *= (1 - backgroundAlpha);
      }

      // No sparse compensation scaling: alphaBlendOutput is already a dense per-pixel signal.
      // composite_alpha_blend writes AlphaBlendRadiance at every pixel, sampling the denoised
      // (dense) primary radiance buffers, so there are no sparse holes to compensate for.
      // Applying radianceScale here would over-count active pixels and amplify surfaces seen
      // through alpha-blended particles.

      if (surface.hasEmissive)
      {
        alphaBlendOutput += sharedRadiance;
      }
    }

    updateAlphaBlendDebugView(pixelCoordinate, surface);
  }

  particleLayerOutput = vec4(0);
  radianceOutput = vec3(0);
  if (cb.outputParticleLayer)
  {
    // Volumetric light composition is independent of secondary signal routing.
    if (cb.compositeVolumetricLight)
    {
      radianceOutput += volumetricPreintegratedRadiance * volumetricRadianceScale;
    }

    if (cb.outputSecondarySignalToParticleLayer)
    {
      // Note: secondary radiance is pre-scaled in the demodulate pass before NRD,
      // so the denoised output is already at the correct dense level.
      particleLayerOutput.xyz += remodulatedTotalSecondaryRadiance * volumeAttenuation;
    }
    else
    {
      radianceOutput += remodulatedTotalSecondaryRadiance * volumeAttenuation;
    }
  }
  else
  {
    radianceOutput += volumetricPreintegratedRadiance * volumetricRadianceScale;
    radianceOutput += remodulatedTotalSecondaryRadiance * volumeAttenuation;
  }
}

void applySkyContribution(
  uint2 pixelCoordinate,
  CameraDirections primaryRayDirections,
  bool primaryMiss,
  float radianceScale,
  vec3 volumeAttenuation,
  inout vec3 radianceOutput)
{
  if (!primaryMiss)
  {
    return;
  }

  if (!cb.domeLightArgs.active)
  {
    // Sky radiance (rasterized SkyProbe cubemap for skyMode==0, Hillaire
    // atmosphere for skyMode==1) is already accumulated into SharedRadiance
    // by the primary-miss handler in geometry_resolver.slangh. Adding SkyMatte
    // here as well double-counts the sky (NV-DXVK fork bug since sync 1448e4e77).
    return;
  }

  vec3 skyContribution = cb.domeLightArgs.radiance * sampleDomeLightTexture(SkyLight, primaryRayDirections.worldDirection, cb.domeLightArgs.worldToLightTransform) * volumeAttenuation;

  if (cb.sparseRenderingArgs.mode != SparseRenderingMode::Off &&
      cb.sparseRenderingArgs.enableSparsePrimaryRayMissComposition)
  {
    skyContribution *= radianceScale;
  }

  radianceOutput += skyContribution;
}

// Aerial perspective (Hillaire EGSR 2020, Section 5.4): the atmosphere's
// extinction and in-scatter between the camera and the shaded surface, which is what makes distant
// geometry hazy and desaturated.
//
// Skipped on primary misses: those directions carry the sky, whose radiance already integrates the
// whole atmospheric column. PSR pixels use their camera-to-mirror distance, so a reflection is
// correctly attenuated by the air in front of the mirror.
//
// The volume covers only the segment past the global volumetrics froxel grid, so its in-scatter is
// attenuated by that grid's transmittance. Strictly the near in-scatter already in radianceOutput
// should not then pick up the far transmittance in turn, but the froxel grid spans tens of metres
// against this volume's kilometres, so its transmittance is near one wherever the far term matters.
void applyAerialPerspective(
  uint2 pixelCoordinate,
  bool primaryMiss,
  float viewDistance,
  vec3 volumeAttenuation,
  inout vec3 radianceOutput)
{
  if (cb.atmosphereArgs.aerialPerspectiveLutSize == 0 ||
      cb.domeLightArgs.active ||
      primaryMiss ||
      viewDistance <= 0.0f)
  {
    return;
  }

  const vec2 screenUV = cameraPixelCoordinateToScreenUV(cb.camera, ivec2(pixelCoordinate));
  const vec3 uvw = aerialPerspectiveScreenUvToUvw(screenUV, viewDistance, cb.atmosphereArgs);

  // Surface at or inside the volume's near bound: the whole camera-to-surface segment belongs to the
  // global volumetrics froxel grid, which has already integrated it. The volume's first slice sits
  // at that bound, so without this the nearest slice would be applied on top - a second helping of
  // in-scatter over exactly the range that is already accounted for.
  if (uvw.z <= 0.0f)
  {
    return;
  }

  // Near-field exclusion, modelled on how fog holds off until fogStart. Driven by this pixel's own
  // view distance, so unlike the volume it gates it carries no neighbourhood term and cannot bleed
  // across a silhouette - see aerialPerspectiveNearFade.
  const float nearFade = aerialPerspectiveNearFade(viewDistance, cb.atmosphereArgs);

  if (nearFade <= 0.0f)
  {
    return;
  }

  const vec4 aerialPerspective = sampleAerialPerspectiveLut(
    AtmosphereAerialPerspectiveLut, AtmosphereAerialPerspectiveSampler, uvw, cb.atmosphereArgs);

  const vec3 inScatter = max(aerialPerspective.rgb, vec3(0.0f)) * volumeAttenuation * nearFade;

  // Fade toward the identity transform, not toward zero: at nearFade 0 this has to leave the surface
  // exactly as it found it, which means a transmittance of one rather than a black multiply.
  const float transmittance = lerp(1.0f, saturate(aerialPerspective.a), nearFade);

  radianceOutput = radianceOutput * transmittance + inScatter;
}

// ============================================================================
// Cloud composite (fork — 2026-09-05, world-space cloud migration Stage 4b)
// ============================================================================
//
// This whole section is the relocated other half of evalSkyRadiance's former primary-ray cloud
// composite (atmosphere_sky.slangh) -- see that function's header comment for the ordering
// argument (dispatchCloudScreenPass now runs after dispatchPathTracing, so the G-buffer pass that
// calls evalSkyRadiance can never see THIS frame's cloud march; composite runs after both, so it
// can). Composited into `radianceOutput` / `alphaBlendOutput` -- both composite-local, already
// DENOISED-and-combined locals (see compositeResult: remodulatedTotalPrimaryRadiance, the denoised
// primary signal, and sharedRadiance are folded into radianceOutput well before applyCloudComposite
// runs) -- never into SharedRadiance. Archaeology commit 30d20a8f5 added its cloud in-scatter to
// SharedRadiance from inside raygen, which composite.comp.slang adds AFTER the denoised primary
// radiance; that ordering sparkles on geometry and ghosts under ray reconstruction. Nothing here
// touches SharedRadianceRG/B.

// Bilinear upsample of the cloud render RT, verbatim port of evalSkyRadiance's former primary-ray
// read (atmosphere_sky.slangh, pre-Stage-4b) onto composite's own CompositeCloudRender /
// CompositeCloudRenderSampler binding. At cloudRenderResolutionScale = 1.0 the uv lands exactly on
// texel centers, so this returns bit-identical values to the removed code -- the acceptance
// invariant for this stage ("sky pixels unchanged"). CompositeCloudRenderSampler is a dedicated
// clamp-to-edge sampler (composite's own, not the shared REPEAT-U sky-view sampler evalSkyRadiance
// used) -- this content is screen-space, never spherical, so clamping is simpler and no less
// correct; the half-texel inset is kept anyway so the sampled uv range matches the removed code's
// exactly.
vec4 sampleCloudRenderRT(uint2 pixelCoordinate, AtmosphereArgs args)
{
  if (args.cloudRenderFullDimX != 0u && args.cloudRenderFullDimY != 0u)
  {
    vec2 uv = (vec2(pixelCoordinate) + 0.5f)
            / vec2(float(args.cloudRenderFullDimX), float(args.cloudRenderFullDimY));
    uint2 rtDim;
    CompositeCloudRender.GetDimensions(rtDim.x, rtDim.y);
    vec2 halfTexel = 0.5f / vec2(rtDim);
    uv = clamp(uv, halfTexel, vec2(1.0f) - halfTexel);
    return CompositeCloudRender.SampleLevel(CompositeCloudRenderSampler, uv, 0.0f);
  }
  return CompositeCloudRender.Load(int3(pixelCoordinate, 0));
}

// Point-sampled (never bilinear) read of the depth companion -- r = entry distance km, g =
// transmittance-weighted mean cloud depth km, both from RtxAtmosphere::dispatchCloudScreenPass /
// cloud_render.comp.slang. Mirrors sampleCloudRenderRT's dimension-ratio mapping so the same native
// texel a bilinear color fetch blends AROUND is the one this reads exactly, but with Load instead of
// SampleLevel: averaging two different surfaces' depths across a silhouette is meaningless (the same
// reasoning cloud_render.comp.slang's own PrimaryLinearViewZ read and debug views 880/884 already
// follow -- see their comments). (-1, 0) is the "no cloud entered" sentinel.
vec2 sampleCloudDepthKm(uint2 pixelCoordinate, AtmosphereArgs args)
{
  if (args.cloudRenderFullDimX != 0u && args.cloudRenderFullDimY != 0u)
  {
    vec2 uv = (vec2(pixelCoordinate) + 0.5f)
            / vec2(float(args.cloudRenderFullDimX), float(args.cloudRenderFullDimY));
    uint2 rtDim;
    CompositeCloudDepth.GetDimensions(rtDim.x, rtDim.y);
    ivec2 texel = clamp(ivec2(uv * vec2(rtDim)), ivec2(0), ivec2(rtDim) - ivec2(1));
    return CompositeCloudDepth.Load(int3(texel, 0));
  }
  return CompositeCloudDepth.Load(int3(pixelCoordinate, 0));
}

// Depth-ratio visibility rule (fork — 2026-09-05). Answers "how much of the marched cloud actually
// sits between the camera and a surface at surfaceKm" using only the depth companion's entry/mean.
// entryKm < 0 means the march never found any cloud along this ray at all (cloud_render.comp.slang's
// sentinel) -- 0 visibility, nothing to composite.
//
// Since e7298993f this is THE geometry occlusion for the cloud composite, not a safeguard (fork —
// 2026-09-05, ghosting follow-up; the earlier text here still described it as a no-op left over from
// Stage 4a's march-side clamp). The march no longer clamps to PrimaryLinearViewZ --
// cloud_render.comp.slang marches the full reachable span at every pixel so the RT, and the temporal
// history built on it, stays a pure function of view direction -- so the only thing standing between
// a marched cloud and the surface in front of it is this ramp, applied per frame AFTER the EMA. A
// surface at or beyond the transmittance-weighted mean depth gets the full composite, one nearer than
// the entry gets none, and one in between gets a proportional blend: a soft, depth-uncertain answer,
// which is the honest one for a volume whose only per-pixel depth summary is (entry, mean). Removing
// or short-circuiting it would fog every near surface, Pip-Boy included. The 50 m feather floor also
// bounds the ramp when the eye is inside the deck (entryKm == 0), where mean - entry can be
// arbitrarily small.
float cloudDepthVisibility(float entryKm, float meanKm, float surfaceKm)
{
  if (entryKm < 0.0f)
  {
    return 0.0f;
  }
  const float featherKm = max(meanKm - entryKm, 0.05f);  // 50 m floor guards the divide
  return saturate((surfaceKm - entryKm) / featherKm);
}

// Rotation-only screen motion vector for the cloud's own pixel, in pixels. Deliberate duplicate of
// geometry_resolver.slangh's calcMotionVectorForRayMiss rather than a shared call: that function
// also drives the general-purpose PrimaryScreenSpaceMotionVector / DLSSRR outputs (unrelated to
// clouds) and must not be touched by this migration. Exact same math, parameterized on an explicit
// world direction instead of re-deriving one from pixelCoordinate, since the caller already has
// primaryRayDirections.worldDirection on hand.
vec2 cloudRotationMotionVectorPixels(vec3 currentWorldDirection, uint2 pixelCoordinate)
{
  vec4 prevNDC = mul(cb.camera.prevWorldToProjection, vec4(currentWorldDirection, 0.0f));
  prevNDC /= prevNDC.w;
  const vec2 ndc = cameraPixelCoordinateToNDC(cb.camera, ivec2(pixelCoordinate));
  const vec2 motionVectorNDC = prevNDC.xy - ndc.xy;
  return motionVectorNDC * vec2(0.5, -0.5) * vec2(cb.camera.resolution);
}

// Translation parallax correction (fork — 2026-09-05). calcMotionVectorForRayMiss / the rotation-only
// term above treats the ray as a pure direction (w=0), exact for content at infinite distance but
// wrong for the cloud layer, which now carries real depth (AtmosphereCloudDepth). On the
// Gamebryo-family engine this migration targets, cb.camera.prevWorldToProjection's translation is
// permanently zero (RtCamera::getPosition() reads a rotation-only D3D view matrix -- see
// RtxAtmosphere::updateFrame's anchor-resolution block), so there is no way to recover camera
// translation from the projection matrices at all, at any depth. The separate cloud-anchor system
// (Stage 2) is the only source of a real translation signal on this target; this is its first
// shader-side consumer (cb.cloudAnchorDeltaYUpKm, populated CPU-side from
// RtxAtmosphere::getCloudAnchor().deltaKm -- see composite_args.h).
//
// Approximation, linearized around the current view ray: a world point P = cameraPos + viewDirYUp *
// depthKm stays fixed when the camera moves; only the component of this frame's camera motion
// PERPENDICULAR to the ray shifts P's apparent screen position (the parallel component only changes
// depth, not where it lands). The cloud-render camera basis vectors (cloudRenderRightYUp /
// UpYUp, pre-scaled by tan(halfFovX)*aspect / tan(halfFovY) -- see RtxAtmosphere::
// setCloudRenderCameraBasis) are exactly the vectors cloud_render.comp.slang itself uses to build a
// per-pixel viewDirYUp from NDC; inverting that construction to first order near the current ray
// converts the perpendicular world-space shift into an NDC delta. viewAxisCosine converts the
// EUCLIDEAN depthKm (the march's own "distance along the ray" convention -- see
// marchCloudSlab/marchCloudLayers) into the forward-axis (view-space) distance the NDC mapping is
// actually defined against, the same conversion cloud_render.comp.slang's own depth clamp already
// performs in the opposite direction.
//
// This is a per-pixel-local linearization (exact at screen center, mildly approximate toward wide-
// FOV edges where viewDirYUp diverges from the central forward axis) and ignores second-order
// rotation/translation coupling -- acceptable for a soft, depth-uncertain background layer, and
// materially better than the pure-rotation term alone for the case that actually regressed:
// strafing under the deck with the view direction roughly fixed.
vec2 cloudParallaxMotionVectorPixels(vec3 viewDirYUp, float depthKm, AtmosphereArgs args)
{
  const vec3 deltaKm = cb.cloudAnchorDeltaYUpKm;  // this frame's camera motion, Y-up km (cur - prev)

  // Strip the along-ray component -- it changes depth, not screen position.
  const vec3 deltaPerp = deltaKm - viewDirYUp * dot(deltaKm, viewDirYUp);

  const vec3 rightAxis = args.cloudRenderRightYUp;  // pre-scaled by tan(halfFovX) * aspect
  const vec3 upAxis    = args.cloudRenderUpYUp;     // pre-scaled by tan(halfFovY)
  const float rightLenSq = max(dot(rightAxis, rightAxis), 1e-9f);
  const float upLenSq    = max(dot(upAxis, upAxis), 1e-9f);

  const float viewAxisCosine = max(abs(dot(viewDirYUp, normalize(args.cloudRenderForwardYUp))), 1e-3f);
  const float zForwardKm = max(depthKm, 0.05f) * viewAxisCosine;

  vec2 ndcDelta;
  ndcDelta.x = dot(deltaPerp, rightAxis) / (zForwardKm * rightLenSq);
  ndcDelta.y = dot(deltaPerp, upAxis)    / (zForwardKm * upLenSq);

  // Same NDC -> pixel convention as the rotation-only term (y flip, half-resolution scale).
  return ndcDelta * vec2(0.5, -0.5) * vec2(cb.camera.resolution);
}

// Combined reprojection motion vector for the cloud history: rotation (reliable on this target) plus
// the translation parallax correction above (the signal the rotation-only term structurally cannot
// carry). `depthKm` should be the pixel's own mean cloud depth (already floored by the caller).
//
// UNJITTERED direction, deliberately (fork — 2026-09-05, ghosting follow-up). The cloud RT is marched
// from pixel-centre rays (cloud_render.comp.slang builds its own NDC from pixelCoord + 0.5, no
// jitter) and the history it feeds is indexed by pixel, so the quantity being reprojected is "the
// cloud along the UNJITTERED direction of pixel p". The rest of composite reconstructs
// primaryRayDirections WITH the DLSS jitter (cameraPixelCoordinateToDirection's two-argument overload
// defaults to jitter = true), which is right for everything that has to line up with the path
// tracer's jittered primary rays and wrong here: projecting a jittered direction through
// prevWorldToProjection and differencing against the unjittered NDC of cameraPixelCoordinateToNDC
// leaves the per-frame jitter (Halton 2/3, up to +-0.5 px at render resolution) baked into the motion
// vector, and floor(x + 0.5) then rounds the history fetch to the wrong neighbour on roughly half of
// all frames -- a permanent ~1 px wobble of the reprojected history under DLSS, i.e. a spatial blur of
// the smoothed deck, not a trail. calcMotionVectorForRayMiss, which this duplicates, passes
// jitter = false for exactly this reason; the Stage 4b port reused composite's jittered directions.
vec2 cloudReprojectionMotionVectorPixels(uint2 pixelCoordinate, float depthKm, AtmosphereArgs args)
{
  const CameraDirections cloudRayDirections =
    cameraPixelCoordinateToDirection(cb.camera, ivec2(pixelCoordinate), /*jitter=*/ false);

  const vec3 viewDirYUp = normalize(worldToAtmosphereYUp(
    cloudRayDirections.worldDirection, args.isZUp != 0u, args.flipUpAxis != 0));

  const vec2 rotationMV = cloudRotationMotionVectorPixels(cloudRayDirections.worldDirection, pixelCoordinate);
  const vec2 parallaxMV = cloudParallaxMotionVectorPixels(viewDirYUp, depthKm, args);
  return rotationMV + parallaxMV;
}

// Composites the Nubis Cubed cloud render RT (m_cloudRenderRT / m_cloudDepthRT, written this frame
// by RtxAtmosphere::dispatchCloudScreenPass, called right after dispatchPathTracing) over BOTH sky
// and geometry. Previously (pre-Stage-4a) this only ran on sky-miss pixels, because it lived inside
// evalSkyRadiance's primary-miss branch; it now also runs on opaque hits, gated per frame by
// cloudDepthVisibility against the resolved surface distance -- a hill whose surface pokes up INTO
// the deck genuinely has cloud in front of it along this ray, and the whole point of this stage is to
// fog it exactly as the sky is fogged, rather than leaving it (and any foliage on it) crisp.
//
// backgroundAlpha / particleLayerOutput (fork — 2026-09-05, ghosting follow-up): the alpha-blend
// surface's coverage (applyVolumetricLighting's backgroundAlpha) and the DLSS-RR transparency layer,
// needed by the alpha-blend in-scatter weighting and the ray-reconstruction routing respectively --
// see the two composite blocks at the end of this function.
void applyCloudComposite(
  uint2 pixelCoordinate,
  bool primaryMiss,
  float viewDistance,
  uint frameIdx,
  float backgroundAlpha,
  inout vec3 radianceOutput,
  inout vec3 alphaBlendOutput,
  inout vec4 particleLayerOutput,
  bool alphaBlendSurfaceValid,
  float alphaBlendHitDistance)
{
  AtmosphereArgs args = cb.atmosphereArgs;
  if (args.cloudRenderRTEnable == 0u)
  {
    return;
  }

  const vec4 cloudRT = sampleCloudRenderRT(pixelCoordinate, args);
  // Same alpha-convention flip evalSkyRadiance always applied: the RT stores TRANSMITTANCE, the
  // premultiplied-over composite below wants OPACITY.
  vec4 currentCloud = vec4(cloudRT.rgb, 1.0f - cloudRT.a);

  const vec2 cloudDepthKm = sampleCloudDepthKm(pixelCoordinate, args);
  const float entryKm = cloudDepthKm.x;
  const float meanKm  = cloudDepthKm.y;
  const float kmPerWorldUnit = 1.0f / max(args.worldUnitsPerKm, 1e-6f);

  // "The march found no span at all" is a property of the view ray, not of geometry, so it is safe
  // to apply before the EMA -- unlike the occlusion gate, which is deliberately deferred until
  // after it (see below).
  if (entryKm < 0.0f)
  {
    currentCloud = vec4(0.0f);
  }

  // NOTE: the geometry occlusion gate does NOT belong here (fork — 2026-09-05, Stage 4b fix).
  // It used to multiply currentCloud by cloudDepthVisibility() at this point, before the temporal
  // EMA below. That made the smoothed signal a function of geometry while the EMA's reprojection
  // only knows about view direction and cloud depth, so any change in the depth buffer under a
  // pixel blended in a history value computed against different occlusion entirely -- a smear
  // trailing the Pip-Boy as it swings up, and cloud ghosting dragging behind geometry while
  // walking. cloud_render.comp.slang's march-side tMaxClamp was removed for the same reason.
  // The gate now runs after the EMA, where it carries no history, exactly as the alpha-blended
  // path below always did.

  // ---- Temporal EMA (relocated from evalSkyRadiance; now parallax-aware -- see
  // cloudReprojectionMotionVectorPixels). Reprojects into the SAME cloud-history ping-pong pair
  // evalSkyRadiance used to own (RtxAtmosphere owns the actual swap; composite reads whichever pair
  // RtxAtmosphere::bindResources already resolved as Prev/Curr for this frame, during the G-buffer
  // pass's common-bindings setup -- composite must not swap again). ----
  vec4 smoothedCloud = currentCloud;
  if (all(cb.camera.resolution > uint2(0u, 0u)))
  {
    const vec2 motionVectorPixels = cloudReprojectionMotionVectorPixels(
      pixelCoordinate, max(meanKm, 0.05f), args);
    const vec2 prevPixelF = vec2(pixelCoordinate) + motionVectorPixels;
    const int2 prevPixel = int2(floor(prevPixelF + 0.5f));

    const bool reprojValid = all(prevPixel >= int2(0, 0))
                          && all(prevPixel < int2(cb.camera.resolution));

    bool historyValid = false;
    vec4 history = vec4(0.0f);
    if (reprojValid)
    {
      history = CompositeCloudHistoryPrev[prevPixel];

      // Age check: reject history whose stored frame-id doesn't match the immediately previous
      // frame (companion R16_UINT ping-pong; catches stale history at pixels that were
      // foreground-occluded last frame). Sentinel 0xFFFF collides harmlessly with startup and the
      // once-per-65536-frames wrap.
      const uint historyFrameId = CompositeCloudHistoryFrameIdPrev[prevPixel];
      const uint expectedPrevFrameId = (frameIdx - 1u) & 0xFFFFu;
      const bool ageValid = (historyFrameId == expectedPrevFrameId) && (historyFrameId != 0xFFFFu);

      historyValid = ageValid && ((history.a > 0.001f) || (currentCloud.a < 0.001f));
    }

    if (historyValid)
    {
      // Anchor-cut history rejection (fork — 2026-09-05): reuses the SAME signal Stage 2 built for
      // exactly this purpose rather than inventing a second one. RtxAtmosphere::getAtmosphereArgs
      // already zeroes args.cloudHistoryWeight for one frame when m_cloudAnchorCutThisFrame fires
      // (a teleport / cell transition / camera-world-override toggle) -- with historyWeight == 0 the
      // lerp below collapses to `currentCloud` unconditionally, which IS full history rejection.
      const float historyWeight = saturate(args.cloudHistoryWeight)
                                * (1.0f - 0.8f * saturate(args.lightningHistoryFade));
      smoothedCloud = lerp(currentCloud, history, historyWeight);
    }
    else
    {
      smoothedCloud = currentCloud;
    }

    CompositeCloudHistoryCurr[pixelCoordinate] = smoothedCloud;
    CompositeCloudHistoryFrameIdCurr[pixelCoordinate] = frameIdx & 0xFFFFu;
  }

  // ---- Opaque geometry: occlusion applied HERE, after the EMA (fork — 2026-09-05, Stage 4b fix).
  // smoothedCloud is the temporally stable, view-direction-only cloud along this ray; how much of
  // it sits in FRONT of the resolved surface is a per-frame property of this frame's depth buffer
  // and must never enter the history. Same shape as the alpha-blend path below, which was already
  // gating after the EMA (though see the (1 - backgroundAlpha) note there).
  float opaqueVisibility = 1.0f;
  if (!primaryMiss)
  {
    const float surfaceKm = viewDistance * kmPerWorldUnit;
    opaqueVisibility = cloudDepthVisibility(entryKm, meanKm, surfaceKm);
  }

  // Ray-reconstruction routing (fork — 2026-09-05, ghosting follow-up; design register item 10).
  // radianceOutput ends up in the composite output DLSS-RR denoises, and RR reprojects that signal
  // with the pixel's PRIMARY motion vector, which on a geometry hit is the surface's own. The cloud
  // in-scatter composited onto that surface is not attached to the surface -- under translation it
  // is screen-static, and under rotation it follows the rotation-only cloud reprojection above, not
  // the surface's parallax -- so RR drags it along with the surface every frame and its history
  // never agrees with the current frame: a bright cloud smear trailing any moving geometry (a
  // Pip-Boy swinging up, terrain sliding past while walking), the artefact the design register
  // anticipated for DLSS one stage after the EMA. The additive term is therefore routed into the RR
  // transparency layer (NGX pInTransparencyLayer -- the same premultiplied layer secondary/PSR
  // radiance and alpha-blended particles already use, composited by RR OVER its denoised output
  // without that output's temporal history; DxvkRayReconstruction::particleBufferMode), and only the
  // multiplicative transmittance stays in the denoised signal. That term is screen-static too, but a
  // darkening dragged with a surface reads as a soft shadow rather than a bright trail, and
  // demodulateAttenuation's 0.3 alpha floor makes a fully layer-based over impossible for dense cloud
  // without leaking 30% of the surface through it -- so the split is deliberate, not partial. Like
  // every other particle-layer term, the routed in-scatter is not touched by applyFog (see the
  // disabled particle-layer fog line there). Sky pixels keep the full over in radianceOutput: their
  // MV is rotation-only (calcMotionVectorForRayMiss), which IS the cloud's own motion, and this
  // preserves "sky pixels unchanged" against the pre-4b path. With cb.outputParticleLayer == 0 (no RR,
  // or particleBufferMode None) or rtx.atmosphere.cloudCompositeRRTransparencyLayer off, this is the
  // plain premultiplied over -- identical to the removed evalSkyRadiance operation, just relocated.
  const bool routeToTransparencyLayer =
    (cb.outputParticleLayer != 0u) && (cb.cloudCompositeRRTransparencyLayer != 0u);

  const float opaqueTransmittance = 1.0f - smoothedCloud.a * opaqueVisibility;
  const vec3  opaqueInScatter     = smoothedCloud.rgb * opaqueVisibility;
  if (routeToTransparencyLayer && !primaryMiss)
  {
    radianceOutput *= opaqueTransmittance;
    particleLayerOutput.xyz += opaqueInScatter;
  }
  else
  {
    // On a sky pixel opaqueVisibility is 1.0 and this is exactly the pre-4b operation.
    radianceOutput = radianceOutput * opaqueTransmittance + opaqueInScatter;
  }

  // ---- Alpha-blended (foliage / particle) surfaces ----
  // Depth-ratio rule (fork — 2026-09-05): the same cloudDepthVisibility gate as the opaque path,
  // evaluated against the alpha-blend surface's OWN hit distance (AlphaBlendGBuffer's hitT) rather
  // than the opaque PrimaryLinearViewZ behind it -- a surface at or beyond the cloud's mean depth is
  // "as deep into the cloud as the marched mass" and gets the full smoothedCloud composite; one
  // entirely in front of the cloud's entry gets none; a surface partway between entry and mean gets a
  // proportional blend. Foliage sitting on the same ground the opaque composite above just fogged
  // will read alphaSurfaceKm ~= surfaceKm and so end up fogged the same amount -- the specific "crisp
  // foliage in a foggy scene" artefact this item exists to avoid.
  if (alphaBlendSurfaceValid && entryKm >= 0.0f)
  {
    const float alphaSurfaceKm = alphaBlendHitDistance * kmPerWorldUnit;
    const float alphaVisibility = cloudDepthVisibility(entryKm, meanKm, alphaSurfaceKm);

    // alphaBlendOutput is PREMULTIPLIED by the surface's own coverage: applyVolumetricLighting scales
    // it by (1 - backgroundAlpha) before returning it, and finalOutput adds it to radianceOutput *
    // backgroundAlpha. Any in-scatter added over it must carry the same factor -- exactly as that
    // function's own fog and volumetric in-scatter terms do. Stage 4b added the cloud in-scatter at
    // full weight (fixed here, fork — 2026-09-05, ghosting follow-up): on a mostly transparent alpha
    // surface (backgroundAlpha -> 1) the pixel then received the cloud twice, once through
    // radianceOutput and once here -- a bright cloud fringe on every foliage edge inside the deck.
    // The transmittance needs no such factor; it multiplies a value that already carries it.
    const float alphaTransmittance = 1.0f - smoothedCloud.a * alphaVisibility;
    const vec3  alphaInScatter     = smoothedCloud.rgb * alphaVisibility * (1.0f - backgroundAlpha);
    if (routeToTransparencyLayer)
    {
      // Same RR reasoning as the opaque block: the alpha-blend radiance is part of the signal RR
      // denoises and reprojects with the primary motion vector, so its cloud in-scatter goes to the
      // transparency layer as well, transmittance stays multiplicative.
      alphaBlendOutput *= alphaTransmittance;
      particleLayerOutput.xyz += alphaInScatter;
    }
    else
    {
      alphaBlendOutput = alphaBlendOutput * alphaTransmittance + alphaInScatter;
    }
  }
}

void applyFog(
  float viewDistance,
  float radianceScale,
  inout vec3 radianceOutput,
  inout vec4 particleLayerOutput)
{
  if (cb.volumeArgs.enable)
  {
    return; // Volumetric fog is applied in volume integration
  }

  // Todo: Using view distance is a bit of a hack here, the primary emissive contribution will not be fully weighted correctly
  // as emissive hits will be over-attenuated.
  vec4 fogColor = calculateFog(viewDistance);

  if (cb.sparseRenderingArgs.mode != SparseRenderingMode::Off)
  {
    fogColor.xyz *= radianceScale;
  }

  // Blend fog over output
  radianceOutput = lerp(radianceOutput, fogColor.rgb, fogColor.a);
  
  // TODO: Disable fog for particle layer to avoid flickering, need to find a proper way to apply fog to particles
  // particleLayerOutput.rgb = lerp(particleLayerOutput.rgb, fogColor.rgb, fogColor.a);
}

void demodulateAttenuation(
  uint2 pixelCoordinate,
  vec3 primaryAttenuation,
  inout vec4 finalOutput,
  inout vec4 particleLayerOutput)
{
  // Filter attenuation to avoid abrupt change and improve stability.
  vec3 neighborAttenuation = 0;
  neighborAttenuation += r11g11b10ToColor(PrimaryAttenuation[clampPixelCoordinate(pixelCoordinate, int2(1, 0))]);
  neighborAttenuation += r11g11b10ToColor(PrimaryAttenuation[clampPixelCoordinate(pixelCoordinate, int2(-1, 0))]);
  neighborAttenuation += r11g11b10ToColor(PrimaryAttenuation[clampPixelCoordinate(pixelCoordinate, int2(0, 1))]);
  neighborAttenuation += r11g11b10ToColor(PrimaryAttenuation[clampPixelCoordinate(pixelCoordinate, int2(0, -1))]);

  const float neighborWeight = 0.5;
  const vec3 blurredAttenuation = (primaryAttenuation + neighborAttenuation * neighborWeight) / (1.0 + neighborWeight * 4.0);

  // Output particles to a separate layer for DLSS-RR.
  // Demodulate attenuation
  if (cb.enableDemodulateAttenuation)
  {
    float primaryAlpha = max(blurredAttenuation.r, max(blurredAttenuation.g, blurredAttenuation.b));
    float minAlpha = 0.3;
    primaryAlpha = max(primaryAlpha, minAlpha);
    finalOutput.xyz /= primaryAlpha;
    particleLayerOutput.w = 1 - primaryAlpha;
  }
}

vec4 compositeResult(
  uint2 pixelCoordinate,
  bool pixelIsActive,
  bool isDirectActive,
  bool isIndirectActive,
  GeometryFlags geometryFlags,
  float primaryLinearViewZ,
  bool primaryMiss,
  out vec4 particleLayerOutput)
{
  RNG randomState = createRNG(pixelCoordinate, cb.frameIdx, 0);

  // Load GBuffer Information.
  // PrimaryAttenuation and secondary GBuffer data are needed at every PSR pixel so the secondary
  // remodulation path in loadInputRadiance produces correct results regardless of sparse active
  // state — keeps the RR particle layer dense. Primary GBuffer reads stay gated since
  // loadPrimaryRadiance is skipped on sparse-inactive pixels.
  const vec3 primaryAttenuation = r11g11b10ToColor(PrimaryAttenuation[pixelCoordinate]);
  vec3 primaryAlbedo = vec3(0);
  vec3 primarySpecularAlbedo = vec3(0);
  float perceptualRoughness = 0.0f;
  vec3 secondaryAttenuation = vec3(0);
  vec3 secondaryAlbedo = vec3(0);
  vec3 secondarySpecularAlbedo = vec3(0);
  if (geometryFlags.secondarySurfaceMask)
  {
    secondaryAttenuation = r11g11b10ToColor(imageLoad(SecondaryAttenuation, pixelCoordinate));
    secondaryAlbedo = SecondaryAlbedo[pixelCoordinate].xyz;
    secondarySpecularAlbedo = SecondarySpecularAlbedo[pixelCoordinate].xyz;
  }
  if (pixelIsActive)
  {
    primaryAlbedo = PrimaryAlbedo[pixelCoordinate].xyz;
    primarySpecularAlbedo = PrimarySpecularAlbedo[pixelCoordinate].xyz;
    perceptualRoughness = PrimaryVirtualWorldNormalPerceptualRoughness[pixelCoordinate].w;
  }

  // Reconstruct the primary ray direction and view hit distance for the current pixel

  const CameraDirections primaryRayDirections = cameraPixelCoordinateToDirection(cb.camera, pixelCoordinate);
  const vec3 viewPosition = cameraReconstructViewPosition(cb.camera, primaryRayDirections.viewDirection, primaryLinearViewZ);
  // Note: Should be equal to the virtual hit distance just without needing to read it (since we need the linear view Z in world space for
  // other calculations here anyways and can just use it to reconstruct this).
  const float viewDistance = length(viewPosition);

  vec3 sharedRadiance;
  vec3 remodulatedTotalPrimaryRadiance;
  vec3 remodulatedTotalSecondaryRadiance;

  vec3 volumetricPreintegratedRadiance;
  vec3 volumeAttenuation;
  vec3 alphaBlendOutput;
  float backgroundAlpha;
  vec3 radianceOutput;
  vec3 sharedFinalOutput;
  bool alphaBlendSurfaceValid;
  float alphaBlendHitDistance;

  // Determine radiance scales for sparse rendering. Scaling only applies at union-active pixels;
  // at inactive pixels the contributions that would be scaled (alpha blend, fog, sky) come from
  // dense buffers and should not be amplified.
  //   Note: if we reach this point for a primary ray miss when sparse primary ray misses are disabled
  //   then do not scale up the result (i.e. keep radianceScale = 1).
  const bool sparseScalingActive = pixelIsActive &&
                                   cb.sparseRenderingArgs.mode != SparseRenderingMode::Off &&
                                   (!primaryMiss || cb.sparseRenderingArgs.enableSparsePrimaryRayMissComposition);

  float radianceScale = 1.f;
  // Volumetric in-scatter is integrated at any union-active pixel (independent of primaryMiss / sky
  // sparse-composition), so its compensation scale must be computed independently of sparseScalingActive.
  // When sparse volumetrics is disabled for this pixel's primary classification, volumetrics run at every pixel and need no compensation.
  float volumetricRadianceScale = 1.f;
  if (sparseScalingActive)
  {
    const float directRate = SparseRenderingDirectPixelSamplingRate[pixelCoordinate];
    const float indirectRate = SparseRenderingIndirectPixelSamplingRate[pixelCoordinate];

    // Sky / fog / volumetrics run for any union-active pixel and are not refilled by the
    // transparency-layer blur pass, so they need to compensate for union inactivity.
    // Direct/indirect activations are independent (decorrelated noise channels), so:
    //   P(union active) = 1 - (1 - directRate) * (1 - indirectRate)
    const float unionActiveProbability = 1.f - (1.f - directRate) * (1.f - indirectRate);
    radianceScale = calculateRadianceScaleFromSamplingRate(unionActiveProbability);
  }
  const bool sparseVolumetricsForThisPixel = primaryMiss
    ? bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryMiss)
    : bool(cb.sparseRenderingArgs.enableSparseVolumetricsPrimaryHit);
  if (pixelIsActive &&
      cb.sparseRenderingArgs.mode != SparseRenderingMode::Off &&
      sparseVolumetricsForThisPixel)
  {
    const float directRate = SparseRenderingDirectPixelSamplingRate[pixelCoordinate];
    const float indirectRate = SparseRenderingIndirectPixelSamplingRate[pixelCoordinate];
    const float unionActiveProbability = 1.f - (1.f - directRate) * (1.f - indirectRate);
    volumetricRadianceScale = calculateRadianceScaleFromSamplingRate(unionActiveProbability);
  }

  // Load Input Radiance
  loadInputRadiance(
    pixelCoordinate, geometryFlags, pixelIsActive,
    isDirectActive, isIndirectActive,
    primaryAlbedo, primarySpecularAlbedo, primaryAttenuation,
    secondaryAlbedo, secondarySpecularAlbedo, secondaryAttenuation,
    perceptualRoughness, primaryMiss,
    sharedRadiance,
    remodulatedTotalPrimaryRadiance, remodulatedTotalSecondaryRadiance);

  // Calculate volumetric lighting
  applyVolumetricLighting(
    pixelCoordinate, randomState, primaryRayDirections, pixelIsActive, primaryMiss, viewDistance,
    radianceScale, volumetricRadianceScale, remodulatedTotalSecondaryRadiance, sharedRadiance,
    volumetricPreintegratedRadiance, volumeAttenuation, alphaBlendOutput, backgroundAlpha,
    radianceOutput, particleLayerOutput,
    alphaBlendSurfaceValid, alphaBlendHitDistance);

  // Calculate final combined radiance output
  // Note: The primary hit accumulated radiance value is not affected by the throughput (rather takes attenuation into account
  // while accumulating already), so it can be added in here without issue.
  radianceOutput += remodulatedTotalPrimaryRadiance * volumeAttenuation;

  // Note: Shared radiance should always be pre-attenuated so no need to factor that in here - it contains multiple layers of signal so each contribution must be attenuated inline.
  radianceOutput += sharedRadiance;

  // Applied before the sky so it only affects geometry; the sky already carries its full column.
  applyAerialPerspective(pixelCoordinate, primaryMiss, viewDistance, volumeAttenuation, radianceOutput);

  // Cloud composite (fork — 2026-09-05, world-space cloud migration Stage 4b). Placed immediately
  // after aerial perspective and before the dome-light sky contribution / fog: the composite is the
  // same premultiplied-over evalSkyRadiance used to do, just relocated to where this frame's cloud RT
  // actually exists (see applyCloudComposite's own header comment). radianceOutput at this point is
  // exactly what evalSkyRadiance's primary branch used to see as `radiance` right before its own
  // (now-removed) composite: for a sky pixel that is remodulatedTotalPrimaryRadiance (clearColor,
  // zero contribution) + sharedRadiance (the cloudless sky/atmosphere/star/moon radiance
  // geometry_resolver.slangh wrote) + volumetrics; for a geometry hit it is the denoised primary
  // radiance with aerial perspective already folded in.
  // backgroundAlpha and particleLayerOutput (fork — 2026-09-05, ghosting follow-up): the alpha-blend
  // coverage the in-scatter must be weighted by, and the DLSS-RR transparency layer the geometry
  // in-scatter is routed into -- see applyCloudComposite's two composite blocks.
  applyCloudComposite(
    pixelCoordinate, primaryMiss, viewDistance, cb.frameIdx, backgroundAlpha,
    radianceOutput, alphaBlendOutput, particleLayerOutput, alphaBlendSurfaceValid, alphaBlendHitDistance);

  // Add sky radiance
  applySkyContribution(pixelCoordinate, primaryRayDirections, primaryMiss, radianceScale, volumeAttenuation, radianceOutput);

  // Apply fog 
  applyFog(viewDistance, radianceScale, radianceOutput, particleLayerOutput);

  // DLSS-RR needs to do alpha blending using the backgroundAlpha, so the input is untouched here.
  // Output final composite result for NRD denoiser.
  vec4 finalOutput = vec4(radianceOutput * backgroundAlpha + alphaBlendOutput, 1.f);

  if (cb.outputParticleLayer)
  {
    demodulateAttenuation(pixelCoordinate, primaryAttenuation, finalOutput, particleLayerOutput);

    particleLayerOutput.xyz += RayReconstructionParticleOutput[pixelCoordinate].xyz;
    RayReconstructionParticleOutput[pixelCoordinate] = sanitize(particleLayerOutput, 0.f);
  }

  return finalOutput;
}

[shader("compute")]
[numthreads(COMPOSITE_THREAD_GROUP_WIDTH, COMPOSITE_THREAD_GROUP_HEIGHT, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint2 localIndex : SV_GroupThreadID)
{
  if (any(threadIndex >= cb.camera.resolution))
  {
    return;
  }

  preCompositeDebugView(threadIndex);

  uvec2 pixelCoordinate;
  const bool pixelIsActive = loadPixelCoordinate(
    threadIndex, UnionActiveLocalPixelCoords,
    cb.camera.resolution, cb.sparseRenderingArgs, pixelCoordinate);

  // Load GBuffer Information.
  // Always loaded (even at union-inactive pixels) because compositeResult uses primaryLinearViewZ
  // to compute viewDistance for fog and the alpha-blend volumetric path, and consults geometryFlags
  // for secondarySurfaceMask / firstSampledLobeIsSpecular.
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(pixelCoordinate, SharedFlags);
  const float primaryLinearViewZ = PrimaryLinearViewZ[pixelCoordinate];
  // Note: Relying on the same sentinel value outputted to the linear view Z which NRD relies on, so this is safe (and lets us not have to read
  // in the cone radius texture which typically contains the flag we use to determine this).
  const bool primaryMiss = primaryLinearViewZ == cb.primaryDirectMissLinearViewZ
                           && !geometryFlags.performedAnyPSR();  // must not be PSR

  // Per-signal active flags: direct/indirect masks are independent. When sparse rendering is off,
  // both signals run at full rate so treat every pixel as active for both.
  // Computed before the inactive early-out so the RR hit distance write below sees the correct gates.
  bool isDirectActive = true;
  bool isIndirectActive = true;
  if (cb.sparseRenderingArgs.mode != SparseRenderingMode::Off)
  {
    isDirectActive = isActivePixelFromMask(SparseRenderingDirectActivePixelMask, pixelCoordinate);
    isIndirectActive = isActivePixelFromMask(SparseRenderingIndirectActivePixelMask, pixelCoordinate);
  }

  // Written from composite (rather than the prepare RR pass) so the value is gated by sparse-indirect activity
  if (cb.writeRayReconstructionHitDistance)
  {
    float hitDistance = 0;
    if (isIndirectActive && !primaryMiss &&
        (cb.enableReSTIRGI || geometryFlags.firstSampledLobeIsSpecular))
    {
      hitDistance = PrimaryIndirectSpecularRadianceHitDistance[pixelCoordinate].a;
    }
    RayReconstructionHitDistanceOutput[pixelCoordinate] = hitDistance;
  }

  vec4 particleLayerOutput;
  vec4 finalOutput = compositeResult(pixelCoordinate, pixelIsActive, isDirectActive, isIndirectActive, geometryFlags, primaryLinearViewZ, primaryMiss, particleLayerOutput);

  FinalOutput[pixelCoordinate] = accumulate(pixelCoordinate, finalOutput, cb.accumulationArgs, true, AccumulatedFinalOutput);

  // Note: we only debug output for active pixels
  postCompositeDebugView(pixelCoordinate, geometryFlags, particleLayerOutput, pixelIsActive, isDirectActive, isIndirectActive);
}
