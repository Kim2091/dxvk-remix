#pragma once

// Scene-interface material derivation.
//
// The alpha state and the legacy (fixed-function) surface state of a surface, as pure functions of plain inputs, so
// the backend path (InstanceManager::updateInstance) and a frontend's MaterialTranslator
// (src/frontends/translator/material_translator.h) derive them with the same code. The backend never includes frontend
// code; frontends include this header.
//
// InstanceManager::calculateAlphaState forwards to deriveAlphaState, which reads RtxOptions forceCutoutAlpha and
// enableEmissiveBlendModeTranslation.

#include <cstdint>

#include "rtx_materials.h"
#include "rtx_types.h"

namespace dxvk {

  // The alpha state description of a surface (what RtSurface::alphaState holds).
  using AlphaStateDesc = RtSurface::AlphaState;

  // The fixed-function inputs of the alpha state derivation (D3D9 render / texture-stage state, or a frontend's own).
  struct FixedFunctionAlphaInput {
    VkCompareOp alphaTestCompareOp = VK_COMPARE_OP_ALWAYS;
    uint8_t alphaTestReferenceValue = 0;
    DxvkBlendMode blendMode {};
    // The alpha texture stage and the texture factor: the Portal player-eye exception reads them.
    DxvkRtTextureOperation textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
    RtTextureArgSource textureAlphaArg1Source = RtTextureArgSource::Texture;
    RtTextureArgSource textureAlphaArg2Source = RtTextureArgSource::None;
    uint32_t tFactor = 0xffffffff;

    static FixedFunctionAlphaInput fromLegacyMaterial(const LegacyMaterialData& legacy) {
      FixedFunctionAlphaInput in;
      in.alphaTestCompareOp = legacy.alphaTestCompareOp;
      in.alphaTestReferenceValue = legacy.alphaTestReferenceValue;
      in.blendMode = legacy.blendMode;
      in.textureAlphaOperation = legacy.textureAlphaOperation;
      in.textureAlphaArg1Source = legacy.textureAlphaArg1Source;
      in.textureAlphaArg2Source = legacy.textureAlphaArg2Source;
      in.tFactor = legacy.tFactor;
      return in;
    }
  };

  // Alpha test / blend state of a surface from its fixed-function inputs, its render material and its instance
  // categories (AlphaBlendToCutout, HairCards, Particle, decal categories).
  AlphaStateDesc deriveAlphaState(const FixedFunctionAlphaInput& input, const MaterialData& material, const CategoryFlags& categories);

  // The legacy (fixed-function) surface state InstanceManager::updateInstance copies into RtSurface: texture-stage
  // operations, texture factor, texgen, texture-factor blend, vertex-colour baked lighting, texture transform and clip
  // plane. Eye parameters are derived from the texture transform there (DrawCallState::isEye).
  struct LegacySurfaceDesc {
    RtTextureArgSource textureColorArg1Source = RtTextureArgSource::Texture;
    RtTextureArgSource textureColorArg2Source = RtTextureArgSource::None;
    DxvkRtTextureOperation textureColorOperation = DxvkRtTextureOperation::Modulate;
    RtTextureArgSource textureAlphaArg1Source = RtTextureArgSource::Texture;
    RtTextureArgSource textureAlphaArg2Source = RtTextureArgSource::None;
    DxvkRtTextureOperation textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
    uint32_t tFactor = 0xffffffff;
    TexGenMode texgenMode = TexGenMode::None;
    bool isTextureFactorBlend = false;
    bool isVertexColorBakedLighting = true;
    Matrix4 textureTransform = Matrix4();
    bool enableClipPlane = false;
    Vector4 clipPlane { 0.f };
  };

  // The legacy surface state of a draw (the fields updateInstance reads from the legacy material and transform data).
  LegacySurfaceDesc deriveLegacySurface(const LegacyMaterialData& legacy, const DrawCallTransforms& transforms);

  // First field that differs ("none" when equal), for parity checks. Floats compare bitwise.
  const char* firstAlphaStateDifference(const AlphaStateDesc& a, const AlphaStateDesc& b);
  const char* firstLegacySurfaceDifference(const LegacySurfaceDesc& a, const LegacySurfaceDesc& b);

} // namespace dxvk
