// Scene-interface material derivation. See rtx_fork_material_derivation.h.

#include "rtx_fork_material_derivation.h"

#include <atomic>
#include <cstring>

#include "rtx_options.h"

namespace dxvk {

  // The derivation InstanceManager::calculateAlphaState calls. categories.any(...) stands for drawCall.testCategoryFlags(...),
  // input.<field> for drawCall.getMaterialData().<field>, and material for materialData.
  AlphaStateDesc deriveAlphaState(const FixedFunctionAlphaInput& input, const MaterialData& material, const CategoryFlags& categories) {
    RtSurface::AlphaState out{};

    // Handle Alpha State for non-Opaque materials

    if (material.getType() == MaterialDataType::Translucent) {
      // Note: Explicitly ensure translucent materials are not considered fully opaque (even though this is the
      // default in the alpha state).
      out.isFullyOpaque = false;

      return out;
    } else if (material.getType() != MaterialDataType::Opaque) {
      return out;
    }

    // Determine if the Legacy Alpha State should be used based on the material data
    // Note: The Material Data may be either Legacy or Opaque here, both use the Opaque Surface Material.

    const auto& opaqueMaterialData = material.getOpaqueMaterialData();

    const bool useLegacyAlphaState = opaqueMaterialData.getUseLegacyAlphaState();

    // Handle Alpha Test State

    // Note: Even if the Alpha Test enable flag is set, we consider it disabled if the actual test type is set to always.
    const bool forceCutoutAlphaTest = categories.any(InstanceCategories::AlphaBlendToCutout);
    const bool forceHairCardAlphaTest = categories.any(InstanceCategories::HairCards);
    const bool forceAlphaTest = forceCutoutAlphaTest || forceHairCardAlphaTest;
    const bool legacyAlphaTestEnabled = (AlphaTestType)input.alphaTestCompareOp != AlphaTestType::kAlways;
    const bool materialAlphaTestEnabled = opaqueMaterialData.getAlphaTestType() != AlphaTestType::kAlways;

    // Note: Use the Opaque Material Data's alpha test state information directly if requested,
    // otherwise derive the alpha test state from the drawcall (via its legacy material data).
    if (forceCutoutAlphaTest) {
      out.alphaTestType = AlphaTestType::kGreater;
      out.alphaTestReferenceValue = static_cast<uint8_t>(RtxOptions::forceCutoutAlpha() * 255.0);
    } else if (forceHairCardAlphaTest) {
      if (!useLegacyAlphaState && materialAlphaTestEnabled) {
        out.alphaTestType = opaqueMaterialData.getAlphaTestType();
        out.alphaTestReferenceValue = opaqueMaterialData.getAlphaTestReferenceValue();
      } else if (legacyAlphaTestEnabled) {
        out.alphaTestType = (AlphaTestType)input.alphaTestCompareOp;
        out.alphaTestReferenceValue = input.alphaTestReferenceValue;
      } else {
        out.alphaTestType = AlphaTestType::kGreater;
        out.alphaTestReferenceValue = static_cast<uint8_t>(RtxOptions::forceCutoutAlpha() * 255.0);
      }
    } else if (!useLegacyAlphaState) {
      out.alphaTestType = opaqueMaterialData.getAlphaTestType();
      out.alphaTestReferenceValue = opaqueMaterialData.getAlphaTestReferenceValue();
    } else if (legacyAlphaTestEnabled) {
      out.alphaTestType = (AlphaTestType)input.alphaTestCompareOp;
      out.alphaTestReferenceValue = input.alphaTestReferenceValue;
    }

    // Handle Alpha Blend State

    bool blendEnabled = false;
    BlendType blendType = BlendType::kColor;
    bool invertedBlend = false;

    // Note: Use the Opaque Material Data's blend state information directly if requested,
    // otherwise derive the alpha blend state from the drawcall (via its legacy material data).
    if (forceAlphaTest) {
      blendEnabled = false;
    } else if (!useLegacyAlphaState) {
      blendEnabled = opaqueMaterialData.getBlendEnabled();
      blendType = opaqueMaterialData.getBlendType();
      invertedBlend = opaqueMaterialData.getInvertedBlend();
    } else if (input.blendMode.enableBlending) {
      const auto srcColorBlendFactor = input.blendMode.colorSrcFactor;
      const auto dstColorBlendFactor = input.blendMode.colorDstFactor;
      const auto colorBlendOp = input.blendMode.colorBlendOp;

      blendEnabled = true; // Note: Set to false later for cases which don't need it

      if (colorBlendOp == VkBlendOp::VK_BLEND_OP_ADD) {
        if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) {
          // Opaque Alias
          blendEnabled = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Standard Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Inverted Alpha Blending
          blendType = BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_ALPHA) {
          // Standard Reverse Emissive Alpha Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
          // Premultiplied Alpha vs Inverted Reverse Emissive Alpha Blending
          const auto srcAlphaBlendFactor = input.blendMode.alphaSrcFactor;
          const auto dstAlphaBlendFactor = input.blendMode.alphaDstFactor;
          const auto alphaBlendOp = input.blendMode.alphaBlendOp;
          const auto colorWriteMask = input.blendMode.writeMask;
          const bool alphaWritesDisabled = (colorWriteMask & VK_COLOR_COMPONENT_A_BIT) == 0;

          const bool looksPremultiplied =
            (alphaBlendOp == VkBlendOp::VK_BLEND_OP_ADD &&
             srcAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE &&
             dstAlphaBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) ||
            alphaWritesDisabled;

          if (looksPremultiplied) {
            // Premultiplied Alpha (ONE, ONE_MINUS_SRC_ALPHA)
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kAlphaEmissive : BlendType::kAlpha;
            invertedBlend = false;
          } else {
            // Inverted Reverse Emissive Alpha Blending
            blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseAlphaEmissive : BlendType::kReverseAlpha;
            invertedBlend = true;
          }
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Standard Color Blending
          blendType = BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Inverted Color Blending
          blendType = BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Standard Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Inverted Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = true;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Standard Reverse Emissive Color Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kReverseColorEmissive : BlendType::kReverseColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR) {
          // Emissive Color Blending: Src + Dst*(1-SrcColor) — bright source is emissive, dark is transparent
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kColorEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ONE) {
          // Emissive Blending
          blendType = RtxOptions::enableEmissiveBlendModeTranslation() ? BlendType::kEmissive : BlendType::kColor;
          invertedBlend = false;
        } else if (
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO) ||
          (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_ZERO && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR)
          ) {
          // Standard Multiplicative Blending
          blendType = BlendType::kMultiplicative;
          invertedBlend = false;
        } else if (srcColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_DST_COLOR && dstColorBlendFactor == VkBlendFactor::VK_BLEND_FACTOR_SRC_COLOR) {
          // Double Multiplicative Blending
          blendType = BlendType::kDoubleMultiplicative;
          invertedBlend = false;
        } else {
          blendEnabled = false;
        }
      } else {
        blendEnabled = false;
      }
    }

    // Special case for the player model eyes in Portal:
    // They are rendered with blending enabled but 1.0 is added to alpha from the texture.
    // Detect this case here and turn such geometry into non-alpha-blended, otherwise
    // the eyes end up in the unordered TLAS and are not rendered correctly.
    const auto& drawMaterialData = input;
    if (blendEnabled && blendType == BlendType::kAlpha && !invertedBlend &&
        drawMaterialData.textureAlphaOperation == DxvkRtTextureOperation::Add &&
        drawMaterialData.textureAlphaArg1Source == RtTextureArgSource::Texture &&
        drawMaterialData.textureAlphaArg2Source == RtTextureArgSource::TFactor &&
        (drawMaterialData.tFactor >> 24) == 0xff) {
      blendEnabled = false;
    }

    if (blendEnabled) {
      out.blendType = blendType;
      out.invertedBlend = invertedBlend;
      // Note: Emissive blend flag must match which blend types are expected to use emissive override in the shader to appear emissive.
      out.emissiveBlend = isBlendTypeEmissive(blendType);

      // Handle Particle/Decal Flags
      // Note: Particles/Decals currently require blending be enabled, be it through the game's original draw call (if legacy alpha state is used),
      // or through the manually specified alpha state.

      // Note: Particles are differentiated from typical objects with opacity by labeling their source material textures as being particle textures.
      out.isParticle = categories.any(InstanceCategories::Particle);
      out.isDecal = categories.any(DECAL_CATEGORY_FLAGS);
    } else {
      out.invertedBlend = false;
      out.emissiveBlend = false;

      // A decal-category surface that alpha-tests without blending is a decal. As alpha-tested geometry in the primary
      // TLAS it would hole the coplanar surface beneath it: a cut-out texel continues the ray from an origin pushed
      // through the decal plane, past that surface. As a decal it is blended onto the surface the primary ray resolves,
      // and with blending disabled its opacity is the alpha test result, 0 or 1. Fully opaque decal-category surfaces
      // have no cut-out texel and stay as they are.
      if (out.alphaTestType != AlphaTestType::kAlways && categories.any(DECAL_CATEGORY_FLAGS)) {
        out.isDecal = true;
        static std::atomic<bool> s_loggedTestOnlyDecal { false };
        if (!s_loggedTestOnlyDecal.exchange(true, std::memory_order_relaxed)) {
          Logger::info(str::format("[DecalRoute] decal-category surface without blending (alpha test ",
                                   static_cast<int>(out.alphaTestType), " ref ", static_cast<uint32_t>(out.alphaTestReferenceValue),
                                   ", material 0x", std::hex, material.getHash(), std::dec,
                                   ") routed as a decal."
                                   " Logged once."));
        }
      }
    }

    // Set the fully opaque flag
    // Note: Fully opaque surfaces can only be signaled when no blending or alpha testing is done as well as no translucency material wise is used.
    // This is important for signaling when to not use the opacity channel in materials when it is not being used for anything.

    out.isFullyOpaque = !blendEnabled && out.alphaTestType == AlphaTestType::kAlways; // use the blend/test type from the output, rather than legacy for this so replacements can override
    out.isBlendingDisabled = !blendEnabled;

    return out;
  }

  LegacySurfaceDesc deriveLegacySurface(const LegacyMaterialData& legacy, const DrawCallTransforms& transforms) {
    LegacySurfaceDesc desc;
    desc.textureColorArg1Source = legacy.textureColorArg1Source;
    desc.textureColorArg2Source = legacy.textureColorArg2Source;
    desc.textureColorOperation = legacy.textureColorOperation;
    desc.textureAlphaArg1Source = legacy.textureAlphaArg1Source;
    desc.textureAlphaArg2Source = legacy.textureAlphaArg2Source;
    desc.textureAlphaOperation = legacy.textureAlphaOperation;
    desc.tFactor = legacy.tFactor;
    desc.texgenMode = transforms.texgenMode;
    desc.isTextureFactorBlend = legacy.isTextureFactorBlend;
    desc.isVertexColorBakedLighting = legacy.isVertexColorBakedLighting;
    desc.textureTransform = transforms.textureTransform;
    desc.enableClipPlane = transforms.enableClipPlane;
    desc.clipPlane = transforms.clipPlane;
    return desc;
  }

  const char* firstAlphaStateDifference(const AlphaStateDesc& a, const AlphaStateDesc& b) {
    if (a.isBlendingDisabled != b.isBlendingDisabled) return "isBlendingDisabled";
    if (a.isFullyOpaque != b.isFullyOpaque) return "isFullyOpaque";
    if (a.alphaTestType != b.alphaTestType) return "alphaTestType";
    if (a.alphaTestReferenceValue != b.alphaTestReferenceValue) return "alphaTestReferenceValue";
    if (a.blendType != b.blendType) return "blendType";
    if (a.invertedBlend != b.invertedBlend) return "invertedBlend";
    if (a.emissiveBlend != b.emissiveBlend) return "emissiveBlend";
    if (a.isParticle != b.isParticle) return "isParticle";
    if (a.isDecal != b.isDecal) return "isDecal";
    return "none";
  }

  const char* firstLegacySurfaceDifference(const LegacySurfaceDesc& a, const LegacySurfaceDesc& b) {
    if (a.textureColorArg1Source != b.textureColorArg1Source || a.textureColorArg2Source != b.textureColorArg2Source ||
        a.textureColorOperation != b.textureColorOperation) return "textureColorStage";
    if (a.textureAlphaArg1Source != b.textureAlphaArg1Source || a.textureAlphaArg2Source != b.textureAlphaArg2Source ||
        a.textureAlphaOperation != b.textureAlphaOperation) return "textureAlphaStage";
    if (a.tFactor != b.tFactor) return "tFactor";
    if (a.texgenMode != b.texgenMode) return "texgenMode";
    if (a.isTextureFactorBlend != b.isTextureFactorBlend) return "isTextureFactorBlend";
    if (a.isVertexColorBakedLighting != b.isVertexColorBakedLighting) return "isVertexColorBakedLighting";
    if (std::memcmp(&a.textureTransform, &b.textureTransform, sizeof(Matrix4)) != 0) return "textureTransform";
    if (a.enableClipPlane != b.enableClipPlane) return "enableClipPlane";
    if (std::memcmp(&a.clipPlane, &b.clipPlane, sizeof(Vector4)) != 0) return "clipPlane";
    return "none";
  }

} // namespace dxvk
