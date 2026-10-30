#pragma once

// The legacy material protocol lets a host wrapper describe each D3D9 draw's material through otherwise unused
// render states:
//   RS  42  CategoryFlags  Raw InstanceCategories bitmask, ORed into the draw's categories (d3d9_rtx.cpp).
//                          Bit 31 (kPerDrawPath) keeps the draw on the D3D9 frontend's per-draw path, so the
//                          frontend neither hands it to the scene nor takes its camera.
//   RS 149  Modifier       Bits 0-3 hold the diffuse slot, 4-7 the normal slot, 8-11 the glow slot and 12-15 the
//                          height slot, with 0xF meaning absent. Bit 16 marks an RGB tangent-space normal map.
//                          Bit 17 makes normal alpha drive roughness, as set by rtx.fnvProtocol.normalAlphaMode.
//   RS 169  Gloss          IEEE float bits of the Blinn-Phong glossiness exponent.
//   RS 177  Emittance      D3DCOLOR tint of the glow map.
// Both 0xFEFEFEFE (the wrapper's "clear" value) and 0 (the D3D9 device default) mean absent for every render state.
// Draws without RS 149 go through the standard legacy material conversion.
//
// The D3D9 side decodes the render states once per draw (captureFnvLegacyProtocol in d3d9_rtx_utils.cpp) and only
// reads device state. The opaque material shader does the texture math (RGB normals, roughness from normal alpha,
// glow tint), driven by OPAQUE_SURFACE_MATERIAL_FLAG_* bits (shared_constants.h).

#include <cstdint>

#include "rtx_option.h"
#include "rtx_texture.h"

namespace dxvk {

  struct OpaqueMaterialData;

  struct FnvLegacyProtocol {
    static constexpr uint32_t kRsCategoryFlags = 42;
    static constexpr uint32_t kRsModifier = 149;
    static constexpr uint32_t kRsGloss = 169;
    static constexpr uint32_t kRsEmittance = 177;

    static constexpr uint32_t kSentinel = 0xFEFEFEFEu;
    static constexpr uint32_t kSlotMask = 0xFu;
    static constexpr uint32_t kSlotAbsent = 0xFu;
    static constexpr uint32_t kDiffuseSlotShift = 0;
    static constexpr uint32_t kNormalSlotShift = 4;
    static constexpr uint32_t kGlowSlotShift = 8;
    static constexpr uint32_t kHeightSlotShift = 12;
    static constexpr uint32_t kNormalRgbTangentSpace = 1u << 16;
    static constexpr uint32_t kNormalAlphaIsSpecMask = 1u << 17;
    // This RS 42 bit lies above every InstanceCategories bit.
    static constexpr uint32_t kPerDrawPath = 1u << 31;

    // Returns the payload of a raw render-state value, with both "absent" encodings mapped to 0.
    static uint32_t readSlot(uint32_t renderStateValue) {
      return renderStateValue == kSentinel ? 0u : renderStateValue;
    }

    uint32_t modifier = 0;   // RS 149. 0 means the draw does not use the protocol.
    uint32_t glossBits = 0;  // RS 169. 0 means absent.
    uint32_t emittance = 0;  // RS 177. 0 means absent, which tints white.
    // True when the captured normal texture's D3D9 format has a usable alpha channel (see d3d9FormatHasGlossAlpha).
    // DXT1 (1-bit alpha), X8R8G8B8, R5G6B5 and similar formats read alpha as 1.0, which as gloss would make a mirror.
    bool normalHasAlpha = false;
    TextureRef diffuseTexture;
    TextureRef normalTexture;
    TextureRef glowTexture;
    TextureRef heightTexture;

    bool isActive() const {
      return modifier != 0 && diffuseTexture.isValid();
    }

    // Folded into LegacyMaterialData::computeIdentityHash so the preserve path rebuilds the material when any
    // protocol input changes.
    // Returns kEmptyHash when the draw does not use the protocol.
    XXH64_hash_t computeHash() const;
  };

  class FnvMaterialProtocol {
  public:
    RTX_OPTION("rtx.fnvProtocol", bool, enable, true,
               "Decodes the legacy material protocol that a game wrapper writes into D3D9 render states 42, 149, 169 and 177:\n"
               "per-draw instance categories, diffuse/normal/glow/height texture slots, glossiness and glow tint.\n"
               "Draws that do not set render state 149 are unaffected. When off, every draw uses the standard legacy material conversion.");
    RTX_OPTION("rtx.fnvProtocol", bool, apiGlowTint, false,
               "Tints the emissive texture of a Remix API opaque material by its emissiveColorConstant when that color is not white,\n"
               "the way the legacy material protocol tints a glow map by its emittance color (render state 177).\n"
               "When off, an API material's emissive texture is used alone. Turn it on when the game integration sends tinted glow maps.");
    RTX_OPTION("rtx.fnvProtocol", float, glowEmissiveIntensity, 1.0f,
               "Emissive intensity of glow maps routed through the legacy material protocol (render state 149, bits 8-11).\n"
               "The emissive color is the glow texture times the emittance tint (render state 177). rtx.emissiveIntensity still applies.");
    RTX_OPTION("rtx.fnvProtocol", int, normalAlphaMode, 1,
               "How the alpha channel of protocol normal maps (render state 149, bit 17) and of terrain-layer normal maps is read.\n"
               "1: Gloss, the inverted roughness used by many texture packs. Roughness = lerp(glossRoughnessMin, glossRoughnessMax, 1 - normal.a).\n"
               "    This only applies when the normal texture has an alpha channel (DXT3/DXT5/A8R8G8B8, ...). Normal maps without alpha\n"
               "    (DXT1, X8R8G8B8, ...) keep the constant roughness derived from the glossiness exponent (render state 169).\n"
               "0: Legacy specular mask. Roughness = lerp(1, r, normal.a), where r is the gloss-exponent roughness. Terrain layers ignore normal alpha.");
    RTX_OPTION("rtx.fnvProtocol", float, glossRoughnessMin, 0.0f,
               "Roughness at normal-map gloss 1.0 (alpha 255) when rtx.fnvProtocol.normalAlphaMode = 1. Raise it to keep full-gloss texels from turning into mirrors.");
    RTX_OPTION("rtx.fnvProtocol", float, glossRoughnessMax, 1.0f,
               "Roughness at normal-map gloss 0.0 (alpha 0) when rtx.fnvProtocol.normalAlphaMode = 1.");
    RTX_OPTION("rtx.fnvProtocol", float, heightDisplaceIn, 0.05f,
               "Parallax occlusion displacement depth (displace_in) for height maps routed through the legacy material protocol\n"
               "(render state 149, bits 12-15). 0 disables POM for those materials.");
    RTX_OPTION("rtx.fnvProtocol", bool, logMaterials, true,
               "Logs one [MaterialDump] line per opaque surface material built, whether from D3D9 legacy state, the Remix API\n"
               "or a replacement. The line lists the route, every texture's hash / format / size / sRGB view / gloss alpha, the\n"
               "terrain-layer gloss mask, the roughness and metallic constants before and after rtx.opaqueMaterial.* scale and\n"
               "bias, the legacy specular level and Fresnel grazing, emissive, displacement and alpha handling. Use it to compare\n"
               "an API material with the same material drawn through D3D9. At most 4000 lines are logged per session.");
    RTX_OPTION("rtx.fnvProtocol", fast_unordered_set, logMaterialTextures, {},
               "Hashes whose materials [MaterialDump] logs every time they are built, past the 4000-line cap of\n"
               "rtx.fnvProtocol.logMaterials. Use the hashes the Remix picker shows: a D3D9 material's albedo image hash, or a material hash.");

    static bool normalAlphaIsGloss() {
      return normalAlphaMode() == 1;
    }

    // D3D9 formats whose alpha channel can carry a gloss map. 1-bit alpha (DXT1, A1R5G5B5) counts as no alpha.
    static bool d3d9FormatHasGlossAlpha(uint32_t d3dFormat);

    // Makes the same decision for a resolved texture, such as a Remix API upload, from the image view's Vulkan format
    // and alpha swizzle. BC1 (any variant), BC4/BC5, alpha-less formats and a null view count as no alpha.
    static bool textureHasGlossAlpha(const TextureRef& texture);

    // Applies an active protocol capture on top of the legacy opaque material (LegacyMaterialData::as<>).
    static void applyToOpaque(const FnvLegacyProtocol& protocol, OpaqueMaterialData& opaqueMat);
  };

} // namespace dxvk
