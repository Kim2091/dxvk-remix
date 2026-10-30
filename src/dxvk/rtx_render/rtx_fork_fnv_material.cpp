// Applies the legacy material protocol. See rtx_fork_fnv_material.h.

#include "rtx_fork_fnv_material.h"

#include <cmath>

#include "rtx_materials.h"
#include "../dxvk_image.h"
#include "../../util/util_bit.h"

namespace dxvk {

  XXH64_hash_t FnvLegacyProtocol::computeHash() const {
    if (modifier == 0) {
      return kEmptyHash;
    }

    struct {
      uint32_t modifier;
      uint32_t glossBits;
      uint32_t emittance;
      uint32_t normalAlpha; // normalHasAlpha | normalAlphaMode << 1, because the mode changes the material.
      XXH64_hash_t textures[4];
    } data {};

    data.modifier = modifier;
    data.glossBits = glossBits;
    data.emittance = emittance;
    data.normalAlpha = (normalHasAlpha ? 1u : 0u) | (uint32_t(FnvMaterialProtocol::normalAlphaMode()) << 1);
    data.textures[0] = diffuseTexture.getImageHash();
    data.textures[1] = normalTexture.getImageHash();
    data.textures[2] = glowTexture.getImageHash();
    data.textures[3] = heightTexture.getImageHash();
    return XXH3_64bits(&data, sizeof(data));
  }

  bool FnvMaterialProtocol::d3d9FormatHasGlossAlpha(uint32_t d3dFormat) {
    // These are D3DFORMAT values from d3d9types.h, written as numbers so this file needs no D3D9 headers.
    constexpr uint32_t kA8R8G8B8 = 21, kA4R4G4B4 = 26, kA8R3G3B2 = 29, kA2B10G10R10 = 31, kA8B8G8R8 = 32,
                       kA2R10G10B10 = 35, kA16B16G16R16 = 36, kA8L8 = 51, kA4L4 = 52, kA8 = 28,
                       kA16B16G16R16F = 113, kA32B32G32R32F = 116;
    constexpr uint32_t kDXT2 = 0x32545844u, kDXT3 = 0x33545844u, kDXT4 = 0x34545844u, kDXT5 = 0x35545844u;
    switch (d3dFormat) {
    case kA8R8G8B8: case kA4R4G4B4: case kA8R3G3B2: case kA8B8G8R8: case kA16B16G16R16:
    case kA8L8: case kA4L4: case kA8: case kA16B16G16R16F: case kA32B32G32R32F:
    case kDXT2: case kDXT3: case kDXT4: case kDXT5:
      return true;
    case kA2B10G10R10: case kA2R10G10B10: // 2-bit alpha is too coarse for a gloss map.
    default:
      return false;
    }
  }

  bool FnvMaterialProtocol::textureHasGlossAlpha(const TextureRef& texture) {
    const DxvkImageView* view = texture.getImageView();
    if (view == nullptr) {
      return false;
    }
    const VkComponentSwizzle a = view->info().swizzle.a;
    if (a == VK_COMPONENT_SWIZZLE_ONE || a == VK_COMPONENT_SWIZZLE_ZERO) {
      return false;
    }
    switch (view->info().format) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16: case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK:
      // A swizzle that routes a color channel into alpha (e.g. L8 -> RRRR) is not a gloss map either.
      return a == VK_COMPONENT_SWIZZLE_IDENTITY || a == VK_COMPONENT_SWIZZLE_A;
    default:
      // BC1 (1-bit alpha at most), BC4/BC5 and RGB-only formats have no gloss alpha.
      return false;
    }
  }

  void FnvMaterialProtocol::applyToOpaque(const FnvLegacyProtocol& protocol, OpaqueMaterialData& opaqueMat) {
    if (!protocol.isActive()) {
      return;
    }

    uint32_t shaderFlags = 0;

    if (LegacyMaterialDefaults::useAlbedoTextureIfPresent()) {
      opaqueMat.setAlbedoOpacityTexture(protocol.diffuseTexture);
    }

    opaqueMat.setMetallicConstant(0.0f);

    // Blinn-Phong glossiness -> GGX: alpha = sqrt(2 / (g + 2)), perceptual roughness = sqrt(alpha).
    if (protocol.glossBits != 0) {
      const float gloss = bit::cast<float>(protocol.glossBits);
      if (std::isfinite(gloss) && gloss >= 0.0f) {
        const float alphaGGX = std::sqrt(2.0f / (gloss + 2.0f));
        opaqueMat.setRoughnessConstant(std::sqrt(alphaGGX));
      }
    }

    if (protocol.normalTexture.isValid()) {
      opaqueMat.setNormalTexture(protocol.normalTexture);
      if (protocol.modifier & FnvLegacyProtocol::kNormalRgbTangentSpace) {
        shaderFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_NORMAL_RGB_TANGENT_SPACE;
      }
      if (protocol.modifier & FnvLegacyProtocol::kNormalAlphaIsSpecMask) {
        if (!normalAlphaIsGloss()) {
          // In legacy mode alpha is a specular mask, and roughness = lerp(1, r, normal.a).
          shaderFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_NORMAL_ALPHA;
        } else if (protocol.normalHasAlpha) {
          // In gloss mode alpha is gloss, and roughness = lerp(min, max, 1 - normal.a).
          shaderFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_NORMAL_ALPHA |
                         OPAQUE_SURFACE_MATERIAL_FLAG_NORMAL_ALPHA_IS_GLOSS;
        }
        // A normal map without an alpha channel (DXT1, X8R8G8B8, ...) keeps the gloss-exponent roughness constant.
      }
    }

    if (protocol.glowTexture.isValid() && glowEmissiveIntensity() > 0.0f) {
      Vector3 tint(1.0f, 1.0f, 1.0f);
      if (protocol.emittance != 0) {
        // A D3DCOLOR is laid out as A8R8G8B8.
        tint = Vector3(float((protocol.emittance >> 16) & 0xFF),
                       float((protocol.emittance >> 8) & 0xFF),
                       float(protocol.emittance & 0xFF)) / 255.0f;
      }
      opaqueMat.setEmissiveColorTexture(protocol.glowTexture);
      opaqueMat.setEmissiveColorConstant(tint);
      opaqueMat.setEmissiveIntensity(glowEmissiveIntensity());
      opaqueMat.setEnableEmission(true);
      shaderFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TEXTURE_TINTED;
    }

    if (protocol.heightTexture.isValid() && heightDisplaceIn() > 0.0f) {
      opaqueMat.setHeightTexture(protocol.heightTexture);
      opaqueMat.setDisplaceIn(heightDisplaceIn());
      opaqueMat.setDisplaceOut(0.0f);
    }

    opaqueMat.setForkFnvFlags(shaderFlags);
  }

} // namespace dxvk
