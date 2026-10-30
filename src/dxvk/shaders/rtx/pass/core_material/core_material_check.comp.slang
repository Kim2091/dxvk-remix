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

// rtx.coreMaterial.gpuCheck. One thread per surface compares the full surface material record, decoded with the
// OpaqueSurfaceMaterial, TranslucentSurfaceMaterial and RayPortalSurfaceMaterial layouts, the Surface alpha and flag
// properties and the extension's albedo tint against the core record decoded with CoreSurfaceMaterial. Every surface
// with a difference is counted, and the differing fields are OR-ed into a mask (core_material_check_args.h).

#include "rtx/pass/core_material/core_material_check_args.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/concept/surface/surface.h"
#include "rtx/concept/surface_material/surface_material.h"
#include "rtx/concept/surface_material/core_surface_material.slangh"
#include "rtx/concept/surface/legacy_surface.slangh"

layout(push_constant)
ConstantBuffer<CoreMaterialCheckArgs> cb;

layout(binding = CORE_MATERIAL_CHECK_BINDING_SURFACES)
StructuredBuffer<Surface> surfaces;

layout(binding = CORE_MATERIAL_CHECK_BINDING_OLD_MATERIALS)
StructuredBuffer<MemoryPolymorphicSurfaceMaterial> oldMaterials;

layout(binding = CORE_MATERIAL_CHECK_BINDING_EXTENSIONS)
StructuredBuffer<MemoryPolymorphicSurfaceMaterial> extensions;

layout(binding = CORE_MATERIAL_CHECK_BINDING_CORE_MATERIALS)
StructuredBuffer<uint4> coreMaterials;

layout(binding = CORE_MATERIAL_CHECK_BINDING_COUNTERS)
RWStructuredBuffer<uint> counters;

layout(binding = CORE_MATERIAL_CHECK_BINDING_CORE_EXTENSIONS)
StructuredBuffer<uint4> coreExtensions;

layout(binding = CORE_MATERIAL_CHECK_BINDING_LEGACY_SURFACES)
StructuredBuffer<uint4> legacySurfaces;

uint mismatchIf(bool differs, uint field)
{
  return differs ? field : 0u;
}

// The full record's 16-bit word at a byte offset. All full layouts are read as uint16_t data[32].
uint oldU16(MemoryPolymorphicSurfaceMaterial m, uint byteOffset)
{
  return (m.data[byteOffset >> 4][(byteOffset >> 2) & 3u] >> ((byteOffset & 2u) * 8u)) & 0xFFFFu;
}

[shader("compute")]
[numthreads(CORE_MATERIAL_CHECK_THREADS_PER_GROUP, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
  const uint surfaceIndex = threadId.x;
  if (surfaceIndex >= cb.surfaceCount)
  {
    return;
  }

  CoreSurfaceMaterial core;
  [unroll]
  for (uint i = 0; i < 4; ++i)
  {
    core.data[i] = coreMaterials[surfaceIndex * 4 + i];
  }

  const MemoryPolymorphicSurfaceMaterial old = oldMaterials[surfaceIndex];
  const Surface surface = surfaces[surfaceIndex];
  const uint oldType = old.data[0].x & surfaceMaterialTypeMask;

  uint fields = 0;

  // The common part holds the alpha state and surface flags (Surface.data2.w, packed by RtSurface::writeGPUData).
  const uint oldAlphaTestType = uint(surface.alphaTestType);
  const uint oldAlphaReference = (surface.data2.w >> 6) & 0xFFu;
  const uint oldBlendType = uint(surface.blendType);
  const uint expectedMode =
    surface.isDecal ? CORE_MATERIAL_ALPHA_MODE_DECAL :
    !surface.isBlendingDisabled ? CORE_MATERIAL_ALPHA_MODE_BLEND :
    oldAlphaTestType != uint(AlphaTestType::kAlways) ? CORE_MATERIAL_ALPHA_MODE_TEST : CORE_MATERIAL_ALPHA_MODE_OPAQUE;
  fields |= mismatchIf(
    core.alphaMode != expectedMode ||
    core.alphaTestType != oldAlphaTestType ||
    core.alphaTestReferenceValueRaw != oldAlphaReference ||
    core.blendType != oldBlendType ||
    core.isFullyOpaque != surface.isFullyOpaque ||
    core.isBlendingDisabled != surface.isBlendingDisabled ||
    core.isEmissiveBlend != surface.isEmissiveBlend ||
    core.isParticle != surface.isParticle ||
    core.isDecal != surface.isDecal ||
    core.invertedBlend != surface.invertedBlend,
    CORE_MATERIAL_CHECK_FIELD_ALPHA);
  fields |= mismatchIf(
    core.isEmissive != surface.isEmissive ||
    core.isMatte != surface.isMatte ||
    core.isAnimatedWater != surface.isAnimatedWater ||
    core.hasMaterialChanged != surface.hasMaterialChanged ||
    core.isMotionBlurMaskOut != surface.isMotionBlurMaskOut ||
    core.isPreservePath != surface.isPreservePath,
    CORE_MATERIAL_CHECK_FIELD_SURFACE_FLAGS);
  // The two fixed-function bits of the tail, and the spare bits above them.
  fields |= mismatchIf(
    core.hasLegacy != legacySurfaceStateHasOther(surface) ||
    core.hasTextureOps != legacySurfaceStateHasTextureOps(surface) ||
    (core.tailDword >> CORE_MATERIAL_TAIL_SPARE_SHIFT) != 0u,
    CORE_MATERIAL_CHECK_FIELD_LEGACY_BITS);

  if (oldType == surfaceMaterialTypeOpaque)
  {
    const OpaqueSurfaceMaterial o = reinterpret<OpaqueSurfaceMaterial>(old);

    fields |= mismatchIf(core.flags != uint(o.flags), CORE_MATERIAL_CHECK_FIELD_FLAGS);
    fields |= mismatchIf(core.samplerIndex != uint(o.samplerIndex), CORE_MATERIAL_CHECK_FIELD_SAMPLER);
    fields |= mismatchIf(
      core.albedoOpacityTextureIndex != uint(o.albedoOpacityTextureIndex) ||
      core.normalTextureIndex != uint(o.normalTextureIndex) ||
      core.roughnessTextureIndex != uint(o.roughnessTextureIndex) ||
      core.metallicTextureIndex != uint(o.metallicTextureIndex) ||
      core.emissiveColorTextureIndex != uint(o.emissiveColorTextureIndex),
      CORE_MATERIAL_CHECK_FIELD_TEXTURES);
    fields |= mismatchIf(core.samplerFeedbackStamp != uint(o.samplerFeedbackStamp), CORE_MATERIAL_CHECK_FIELD_FEEDBACK_STAMP);
    fields |= mismatchIf(any(core.albedoOpacityConstant != vec4(o.albedoOpacityConstant)), CORE_MATERIAL_CHECK_FIELD_ALBEDO_OPACITY);
    fields |= mismatchIf(
      any(core.emissiveColorConstant != vec3(o.emissiveColorConstant)) ||
      core.emissiveIntensity != float(o.emissiveIntensity),
      CORE_MATERIAL_CHECK_FIELD_EMISSIVE);
    // The old record's data[27] holds the FNV flags word and
    // data[30-31] the terrain-layers index, so roughnessFloor and specularMetal are only comparable on other materials.
    const bool fnvLayout =
      (uint(o.flags) & OPAQUE_SURFACE_MATERIAL_FLAG_FNV_LAYOUT_MASK) == OPAQUE_SURFACE_MATERIAL_FLAG_FNV_LAYOUT;
    fields |= mismatchIf(
      core.roughnessConstant != float(o.roughnessConstant) ||
      core.metallicConstant != float(o.metallicConstant) ||
      core.anisotropy != float(o.anisotropy) ||
      (!fnvLayout && core.roughnessFloor != f16tof32(uint(o.roughnessFloorBits))),
      CORE_MATERIAL_CHECK_FIELD_ROUGH_METAL);
    fields |= mismatchIf(core.dlssControlMask != o.dlssControlMask, CORE_MATERIAL_CHECK_FIELD_DLSS_MASK);
    fields |= mismatchIf(
      !fnvLayout && (core.specularMetalParams != uint(o.specularMetalParams) ||
                     core.specularMetalTint != uint(o.specularMetalTint)),
      CORE_MATERIAL_CHECK_FIELD_SPECULAR_METAL);
    fields |= mismatchIf(core.decalSortOrder != uint(surface.decalSortOrder), CORE_MATERIAL_CHECK_FIELD_DECAL_SORT_ORDER);

    // The albedo tint comes from the extension entry when its flag is set, and is 1 with the bit clear otherwise.
    bool hasTint = false;
    vec3 tint = vec3(1.0);
    bool hasSubsurface = false;
    const uint extensionIndex = o.subsurfaceMaterialIndex;
    if (extensionIndex != SURFACE_INDEX_INVALID && extensionIndex < cb.extensionCount)
    {
      const SubsurfaceMaterial ext = reinterpret<SubsurfaceMaterial>(extensions[extensionIndex]);
      hasTint = (uint(ext.flags) & SURFACE_MATERIAL_EXTENSION_FLAG_ALBEDO_TINT) != 0;
      hasSubsurface = (uint(ext.flags) & SURFACE_MATERIAL_EXTENSION_FLAG_NO_SUBSURFACE) == 0;
      if (hasTint)
      {
        tint = vec3(ext.albedoTint);
      }
    }
    fields |= mismatchIf(core.hasAlbedoTint != hasTint || any(core.albedoTint != tint), CORE_MATERIAL_CHECK_FIELD_ALBEDO_TINT);

    // The core side table has an entry exactly for a subsurface extension, POM, a thin film or a tangent texture.
    // Bytes 0-31 are the extension record's (or the no-extension placeholder), and 32-41 the full record's POM and
    // thin-film words.
    const bool needsEntry = hasSubsurface ||
      (uint(o.flags) & (OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT | OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER)) != 0 ||
      uint(o.tangentTextureIndex) != 0xFFFFu;
    const uint coreExtensionIndex = core.extensionIndex;
    if (needsEntry != (coreExtensionIndex != SURFACE_INDEX_INVALID) || (needsEntry && coreExtensionIndex >= cb.coreExtensionCount))
    {
      fields |= CORE_MATERIAL_CHECK_FIELD_EXTENSION_INDEX;
    }
    else if (needsEntry)
    {
      CoreSurfaceMaterialExtension entry;
      [unroll]
      for (uint i = 0; i < 4; ++i)
      {
        entry.data[i] = coreExtensions[coreExtensionIndex * 4 + i];
      }
      const bool extensionValid = extensionIndex != SURFACE_INDEX_INVALID && extensionIndex < cb.extensionCount;
      bool differs = false;
      for (uint b = 0; b < CORE_MATERIAL_EXT_SUBSURFACE_BYTES; b += 2)
      {
        const uint expected = extensionValid ? oldU16(extensions[extensionIndex], b) :
          (b == 0 ? SURFACE_MATERIAL_EXTENSION_FLAG_NO_SUBSURFACE : (b <= 6 ? CORE_MATERIAL_EXT_NO_EXTENSION_TEXTURE : 0u));
        differs = differs || entry.loadU16(b) != expected;
      }
      differs = differs ||
        entry.tangentTextureIndex != oldU16(old, 46) ||
        entry.heightTextureIndex != oldU16(old, 20) ||
        entry.loadU16(CORE_MATERIAL_EXT_OFFSET_DISPLACE_IN) != oldU16(old, 16) ||
        entry.loadU16(CORE_MATERIAL_EXT_OFFSET_DISPLACE_OUT) != oldU16(old, 18) ||
        entry.loadU16(CORE_MATERIAL_EXT_OFFSET_THIN_FILM_THICKNESS) != oldU16(old, 22);
      for (uint b = CORE_MATERIAL_EXT_USED_BYTES; b < CORE_MATERIAL_EXT_SIZE; b += 2)
      {
        differs = differs || entry.loadU16(b) != 0u;
      }
      fields |= mismatchIf(differs, CORE_MATERIAL_CHECK_FIELD_CORE_EXTENSION);
    }

    // The class is 3 for raytraced render targets, 2 (skin SSS) only with a subsurface extension entry, and 0 otherwise.
    const bool raytracedRenderTarget = (uint(o.flags) & OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET) != 0;
    const uint materialClass = core.materialClass;
    const bool classOk = raytracedRenderTarget ? materialClass == CORE_MATERIAL_CLASS_FALLBACK :
      (materialClass == CORE_MATERIAL_CLASS_OPAQUE || (materialClass == CORE_MATERIAL_CLASS_SKIN_SSS && hasSubsurface));
    fields |= mismatchIf(!classOk, CORE_MATERIAL_CHECK_FIELD_CLASS);
  }
  else if (oldType == surfaceMaterialTypeTranslucent)
  {
    bool differs = false;
    // Bytes 0-35 verbatim, 36-55 zero.
    for (uint b = 0; b < CORE_MATERIAL_CLASS_BYTES; b += 2)
    {
      const uint expected = b < CORE_MATERIAL_TRANSLUCENT_BYTES ? oldU16(old, b) : 0u;
      differs = differs || core.loadU16(b) != expected;
    }
    fields |= mismatchIf(differs, CORE_MATERIAL_CHECK_FIELD_TRANSLUCENT);
    fields |= mismatchIf(core.materialClass != CORE_MATERIAL_CLASS_TRANSLUCENT, CORE_MATERIAL_CHECK_FIELD_CLASS);
    fields |= mismatchIf(core.extensionIndex != SURFACE_INDEX_INVALID, CORE_MATERIAL_CHECK_FIELD_EXTENSION_INDEX);
    fields |= mismatchIf(core.hasAlbedoTint, CORE_MATERIAL_CHECK_FIELD_ALBEDO_TINT);
  }
  else
  {
    bool differs = false;
    for (uint b = 0; b < CORE_MATERIAL_CLASS_BYTES; b += 2)
    {
      differs = differs || core.loadU16(b) != oldU16(old, b);
    }
    fields |= mismatchIf(differs, CORE_MATERIAL_CHECK_FIELD_PORTAL);
    fields |= mismatchIf(core.materialClass != CORE_MATERIAL_CLASS_FALLBACK, CORE_MATERIAL_CHECK_FIELD_CLASS);
    fields |= mismatchIf(core.extensionIndex != SURFACE_INDEX_INVALID, CORE_MATERIAL_CHECK_FIELD_EXTENSION_INDEX);
  }

  // The legacy record, decoded with the LegacySurface module, against the Surface record's words.
  if (surfaceIndex < cb.legacySurfaceCount)
  {
    LegacySurface legacy;
    [unroll]
    for (uint i = 0; i < LEGACY_SURFACE_ROWS; ++i)
    {
      legacy.data[i] = legacySurfaces[surfaceIndex * LEGACY_SURFACE_ROWS + i];
    }
    const uint expectedFlags = (surface.data13.z & LEGACY_SURFACE_TEXTURE_FLAGS_SURFACE_MASK) |
      (surface.isTextureFactorBlend ? LEGACY_SURFACE_FLAG_TEXTURE_FACTOR_BLEND : 0u) |
      (surface.isVertexColorBakedLighting ? LEGACY_SURFACE_FLAG_VERTEX_COLOR_BAKED_LIGHTING : 0u) |
      (surface.isClipPlaneEnabled ? LEGACY_SURFACE_FLAG_CLIP_PLANE : 0u) |
      (surface.skipSurfaceInteractionSpritesheetAdjustment ? LEGACY_SURFACE_FLAG_SKIP_SPRITESHEET : 0u);
    const uint expectedSecondary = oldType == surfaceMaterialTypeOpaque ? oldU16(old, 6) : 0xFFFFu;
    const bool differs =
      any(legacy.data[0] != surface.data11) ||
      any(legacy.data[1] != surface.data12) ||
      any(legacy.data[2] != surface.data14) ||
      legacy.tFactor != surface.tFactor ||
      legacy.textureFlags != expectedFlags ||
      legacy.isEye != surface.isEye ||
      uint(legacy.texcoordGenerationMode) != uint(surface.texcoordGenerationMode) ||
      legacy.spriteSheetRows != surface.spriteSheetRows ||
      legacy.spriteSheetCols != surface.spriteSheetCols ||
      legacy.spriteSheetFPS != surface.spriteSheetFPS ||
      legacy.loadU8(LEGACY_SURFACE_OFFSET_SPRITESHEET + 3) != 0u ||
      legacy.secondaryTextureIndex != expectedSecondary ||
      legacy.loadU16(LEGACY_SURFACE_OFFSET_SECONDARY_TEXTURE + 2) != 0u ||
      any(legacy.data[4].xyz != surface.data15.xyz) ||
      legacy.data[4].w != 0u;
    // The TextureOps part, decoded from bytes 48-55 alone, against the Surface's texture-stage words.
    const LegacyTextureOps ops = legacyTextureOpsCreate(
      legacy.loadDword(LEGACY_SURFACE_OFFSET_TFACTOR), legacy.loadDword(LEGACY_SURFACE_OFFSET_TEXTURE_FLAGS));
    const bool opsDiffer =
      ops.tFactor != surface.tFactor ||
      uint(ops.textureColorArg1Source) != uint(surface.textureColorArg1Source) ||
      uint(ops.textureColorArg2Source) != uint(surface.textureColorArg2Source) ||
      uint(ops.textureColorOperation) != uint(surface.textureColorOperation) ||
      uint(ops.textureAlphaArg1Source) != uint(surface.textureAlphaArg1Source) ||
      uint(ops.textureAlphaArg2Source) != uint(surface.textureAlphaArg2Source) ||
      uint(ops.textureAlphaOperation) != uint(surface.textureAlphaOperation) ||
      ops.isTextureFactorBlend != surface.isTextureFactorBlend ||
      ops.isVertexColorBakedLighting != surface.isVertexColorBakedLighting;
    fields |= mismatchIf(differs || opsDiffer, CORE_MATERIAL_CHECK_FIELD_LEGACY);
  }

  uint unused;
  InterlockedAdd(counters[CORE_MATERIAL_CHECK_COUNTER_SURFACES], 1u, unused);
  if (fields != 0)
  {
    InterlockedAdd(counters[CORE_MATERIAL_CHECK_COUNTER_MISMATCHES], 1u, unused);
    InterlockedOr(counters[CORE_MATERIAL_CHECK_COUNTER_FIELDS], fields, unused);
    InterlockedMin(counters[CORE_MATERIAL_CHECK_COUNTER_FIRST], surfaceIndex, unused);
  }
}
