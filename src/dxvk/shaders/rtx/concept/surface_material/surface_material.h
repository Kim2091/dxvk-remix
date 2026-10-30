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

#include "../../utility/shared_constants.h"

static const uint8_t opaqueLobeTypeDiffuseReflection = uint8_t(0u);
static const uint8_t opaqueLobeTypeSpecularReflection = uint8_t(1u);
static const uint8_t opaqueLobeTypeOpacityTransmission = uint8_t(2u);
static const uint8_t opaqueLobeTypeDiffuseTransmission = uint8_t(3u);

static const uint8_t translucentLobeTypeSpecularReflection = uint8_t(0u);
static const uint8_t translucentLobeTypeSpecularTransmission = uint8_t(1u);

struct MemoryPolymorphicSurfaceMaterial
{
  // Note: Currently aligned nicely to 64 bytes, avoid changing the size of this structure. Note however since this is smaller
  // than a L1 cacheline the actual size doesn't matter as much, so it is not heavily packed as the cache hitrate will be low
  // and the random access nature does not facilitate much memory coalescing. Since this structure can fit in 64 bytes however
  // it is best not to be too wasteful as this will align to L2's 32 byte cachelines better.

  // Note: Keeping these as uint4 ensures 16 byte memory alignment, which is important for aligned vector loading.
  uvec4 data[4];

  bool isOpaque()
  {
    // Note: First two bits of data are reserved for common polymorphic type
    return (data[0].x & surfaceMaterialTypeMask) == surfaceMaterialTypeOpaque;
  }

  bool hasValidDisplacement() {
    return isOpaque() && (data[0].x & OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT);
  }
};

struct OpaqueSurfaceMaterial
{
  // 0 - 3
  // bitmask of OPAQUE_SURFACE_MATERIAL_FLAG_* bits
  uint16_t flags;
  uint16_t samplerIndex;
  uint16_t albedoOpacityTextureIndex;
  uint16_t secondaryTextureIndex;

  // 4-7
  f16vec4 albedoOpacityConstant;

  // 8-11
  float16_t displaceIn;
  float16_t displaceOut;
  uint16_t heightTextureIndex;
  // note: thinFilmThicknessConstant should be between 0 and 1 
  float16_t thinFilmThicknessConstant;

  // For performance, we want to keep fields used in the visibility check in the first 32 bytes.
  // The fields below here are overridden to constant values in that code, so should be left at the end
  // If we add a new field that is used for visibility, it should go above this.
  // If it isn't used for visibility, it should go below and be overridden in opaqueSurfaceMaterialCreate().

  // 12-15
  uint16_t emissiveColorTextureIndex;
  uint16_t roughnessTextureIndex;
  uint16_t metallicTextureIndex;
  uint16_t normalTextureIndex;

  // 16-19
  f16vec3 emissiveColorConstant;
  float16_t emissiveIntensity;

  // 20-23
  float16_t roughnessConstant;
  float16_t metallicConstant;
  float16_t anisotropy;
  uint16_t tangentTextureIndex;

  // 24-25
  uint subsurfaceMaterialIndex;

  // 26
  uint16_t samplerFeedbackStamp;
  // Lower bound for roughness decoded from a smoothness map
  // (OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_SMOOTHNESS), as half-float bits: read it with
  // f16tof32(). The FNV flags word on an FNV-layout material (opaqueSurfaceMaterialFnvFlags()).
  uint16_t roughnessFloorBits;

  // 28-29
  uint dlssControlMask;

  // Todo: Fixed function blend state info here in the future (Actually this should go on a Legacy Material, or some sort of non-PBR Legacy Surface)

  // 30-31
  // On an FNV-layout material (see OPAQUE_SURFACE_MATERIAL_FLAG_FNV_LAYOUT) these two words are the
  // terrain-layers extension index instead; read it with opaqueSurfaceMaterialTerrainLayersIndex().
  // OPAQUE_SURFACE_MATERIAL_FLAG_METALLIC_FROM_SPECULAR parameters, 4 bits each
  // (value / 15): [0:3] spec low, [4:7] spec high, [8:11] albedo luminance max
  // (gamma; 15 = no albedo gate), [12:15] metal strength.
  uint16_t specularMetalParams;
  // Specular colour as RGB565 (gamma). 0 = keep the albedo on metal texels.
  uint16_t specularMetalTint;

  bool hasValidDisplacement() {
    return flags & OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT;
  }
};

struct TranslucentSurfaceMaterial
{
  // bitmask of TRANSLUCENT_SURFACE_MATERIAL_FLAG_* bits
  uint16_t flags;
  float16_t baseReflectivity;
  f16vec3 transmittanceColor;
  uint16_t samplerIndex;
  uint16_t transmittanceOrDiffuseTextureIndex;
  // encodes either the thin-walled thickness or the transmittance measurement distance
  // thin-walled thickness is represented as a negative number
  float16_t thicknessOrMeasurementDistance;
  uint16_t normalTextureIndex;
  uint16_t emissiveColorTextureIndex;
  float16_t emissiveIntensity;
  float16_t refractiveIndex;

  // 12-13: Source surface material index (21 bits).  Placed here (byte offset 24)
  // so the uint32_t is naturally 4-byte aligned — avoids Slang inserting padding
  // that would break reinterpret<> from the 64-byte MemoryPolymorphicSurfaceMaterial.
  uint32_t sourceSurfaceMaterialIndex;

  // 14-16
  f16vec3 emissiveColorConstant;

  // 17
  uint16_t samplerFeedbackStamp;

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  uint16_t data[14];
};

struct RayPortalSurfaceMaterial
{
  uint16_t flags;
  uint16_t rayPortalIndex;

  uint16_t maskTextureIndex;
  uint16_t maskTextureIndex2;

  float16_t rotationSpeed;
  float16_t emissiveIntensity;

  uint16_t samplerIndex;
  uint16_t samplerIndex2;

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  uint16_t data[24];

};

struct SubsurfaceMaterial
{
  uint16_t flags;

  uint16_t subsurfaceTransmittanceTextureIndex;
  uint16_t subsurfaceThicknessTextureIndex;
  uint16_t subsurfaceSingleScatteringAlbedoTextureIndex;

  float16_t volumetricAnisotropy;
  f16vec3 volumetricAttenuationCoefficient;
  float16_t measurementDistance;
  f16vec3 singleScatteringAlbedo;

  float16_t maxSampleRadius;

  // 13-15: valid when flags has SURFACE_MATERIAL_EXTENSION_FLAG_ALBEDO_TINT
  f16vec3 albedoTint;

  // padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  uint16_t data[16];
};

// Terrain-layers extension (remixapi_MaterialInfoTerrainLayersEXT). Mirrors RtTerrainLayersMaterial::writeGPUData
// in rtx_materials.h; lives in surfaceMaterialExtensions and is referenced by OpaqueSurfaceMaterial::terrainLayersMaterialIndex.
// Only 16-bit and 32-bit scalars so the reinterpret<> from the 64-byte MemoryPolymorphicSurfaceMaterial has no padding.
struct TerrainLayersMaterial
{
  // 0-1
  uint16_t flags; // TERRAIN_LAYERS_MATERIAL_FLAG_*
  uint16_t layerCount;
  // 2-9, 10-17
  uint16_t albedoTextureIndices[TERRAIN_LAYERS_MATERIAL_MAX_LAYERS];
  uint16_t normalTextureIndices[TERRAIN_LAYERS_MATERIAL_MAX_LAYERS];
  // 18-21
  uint16_t weightTextureIndex0;
  uint16_t weightTextureIndex1;
  uint16_t tintTextureIndex;
  uint16_t weightSamplerIndex;
  // 22-29
  float weightUvScaleX;
  float weightUvScaleY;
  float weightUvOffsetX;
  float weightUvOffsetY;
  // 30-31: terrain POM, surfaceMaterialExtensions index of the TerrainLayersHeightMaterial entry
  // (SURFACE_INDEX_INVALID = no layer has a height map)
  uint32_t heightMaterialIndex;
};

// Per-layer height maps of a terrain-layers entry. Mirrors RtTerrainLayersHeightMaterial::writeGPUData
// in rtx_materials.h; referenced by TerrainLayersMaterial::heightMaterialIndex.
struct TerrainLayersHeightMaterial
{
  // 0-1
  uint16_t layerMask; // bit i: heightTextureIndices[i] is valid
  uint16_t coneMask;  // SCCR terrain: bit i: heightTextureIndices[i] is a conservative height+cone texture
  // 2-9
  uint16_t heightTextureIndices[TERRAIN_LAYERS_MATERIAL_MAX_LAYERS];
  // 10-31: padding (to keep size matching with MemoryPolymorphicSurfaceMaterial)
  uint16_t data[22];
};

struct SubsurfaceMaterialInteraction
{
  uint32_t packedTransmittanceColor;
  float16_t measurementDistance;
  uint32_t packedSingleScatteringAlbedo;
  uint8_t volumetricAnisotropy;
  uint8_t maxSampleRadius;
};

struct OpaqueSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  float opacity;
  vec3 albedo;
  float normalDetail; // 1.0 - dot(shadingNormal, interpolatedNormal)
  vec3 baseReflectivity;
  // Note: These roughness values are non-perceptual roughness.
  float isotropicRoughness;
  vec2 anisotropicRoughness;
  // Note: fp16 may not be sufficient here for high radiance values, potentially change if clamping
  vec3 emissiveRadiance;
  SubsurfaceMaterialInteraction subsurfaceMaterialInteraction;
  // Note: A value of 0 in the thin film thickness indicates the thin film is disabled.
  float thinFilmThickness;
  uint32_t dlssControlMask;
  uint8_t flags;
};

struct DecalMaterialInteraction
{
  vec3 shadingNormal;
  vec3 albedo;
  vec3 baseReflectivity;
  vec3 emissiveRadiance;
  float opacity;
  float roughness;
  float anisotropy;
};

struct MemoryDecalMaterialInteraction
{
  uint4 packed;
  f16vec3 emissiveRadiance;
};

struct TranslucentSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  float normalDetail; // 1.0 - dot(shadingNormal, interpolatedNormal)

  float baseReflectivity;
  float refractiveIndex;
  vec3 transmittanceColor;
  vec3 emissiveRadiance;

  // diffuse layer parameters, only valid if TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_ALBEDO_LAYER is set in flags
  vec3 diffuseColor;
  float diffuseOpacity;

  // Note: Source values only used for serialization purposes.
  // Note: Used as much of a translucent material is constant and typically reading from the material even
  // if it requires an indirection should be better than reading/writing more data to per-pixel buffers. Additionally the
  // lack of the original values such as the transmittance measurement distance and color make it hard to send this compactly
  // without otherwise having to upload those to the Translucent Surface Material (could be done if needed though).
  uint sourceSurfaceMaterialIndex;
  // Note: Raw (gamma encoded) emissive color packed in R5G6B5 needed for more tight packing, not ideal as this carries live state
  // across other code but this is an easy way to get the required info.
  uint16_t sourcePackedGammaEmissiveColor;

  // encodes either the thin-walled thickness or the transmittance measurement distance
  // thin-walled thickness is represented as a negative number
  float thicknessOrMeasurementDistance;

  uint8_t flags;

  bool isAnimatedWater;
};

struct RayPortalSurfaceMaterialInteraction
{
  vec4 mask;

  uint8_t rayPortalIndex;
  uint8_t isInsidePortal;
};

// Note: GBuffer-specific serialization data, not as tightly packed as it could be but done in this manner to share
// data with other passes (NRD, RTXDI, etc) and reuse that data for deserialization to not duplicate information. Some
// of these values may not be populated depending on the material and will instead be set to the desired special output
// value to indicate non-presence to subsequent passes. Additionally some of this data is assumed to be packed later
// by the gbuffer helper functions just to avoid code duplication (not ideal but we probably need a better way to do this).
struct GBufferMemoryPolymorphicSurfaceMaterialInteraction
{
  f16vec3 worldShadingNormal;
  float16_t perceptualRoughness;
  f16vec3 albedo;
  f16vec3 baseReflectivity;

  uint data0;
  uint data1;
};

struct PolymorphicSurfaceMaterialInteraction
{
  vec3 shadingNormal;
  vec3 emissiveRadiance;
  vec3 vdata0;
  vec3 vdata1;

  float fdata0;
  float fdata1;
  float fdata2;
  float fdata3;
  float fdata4;
  float fdata5;

  uint idata0;
  uint16_t idata1;

  uint32_t i32data0;
  uint32_t i32data1;
  uint32_t dlssControlMask;

  uint8_t bdata0;
  uint8_t bdata1;
  uint8_t bdata2;

  uint8_t type;

  uint8_t isAnimatedWater;
};
