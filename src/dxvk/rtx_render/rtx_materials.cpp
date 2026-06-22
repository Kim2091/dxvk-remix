/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_materials.h"

#include "rtx_fork_hooks.h"
#include "rtx_options.h"

namespace dxvk {

bool getEnableDiffuseLayerOverrideHack() {
  return TranslucentMaterialOptions::enableDiffuseLayerOverride();
}

float getEmissiveIntensity() {
  return RtxOptions::emissiveIntensity();
}

float getDisplacementFactor() {
  return RtxOptions::Displacement::displacementFactor();
}

float getDisplacementInFactor() {
  return RtxOptions::Displacement::displacementFactor() * RtxOptions::Displacement::displacementInFactor();
}

float getDisplacementOutFactor() {
  return RtxOptions::Displacement::displacementFactor() * RtxOptions::Displacement::displacementOutFactor();
}

dxvk::OpaqueMaterialData LegacyMaterialData::createDefault() {
  OpaqueMaterialData opaqueMat;
  opaqueMat.setAnisotropyConstant(LegacyMaterialDefaults::anisotropy());
  opaqueMat.setEmissiveIntensity(LegacyMaterialDefaults::emissiveIntensity());
  opaqueMat.setAlbedoConstant(LegacyMaterialDefaults::albedoConstant());
  opaqueMat.setOpacityConstant(LegacyMaterialDefaults::opacityConstant());
  opaqueMat.setRoughnessConstant(LegacyMaterialDefaults::roughnessConstant());
  opaqueMat.setMetallicConstant(LegacyMaterialDefaults::metallicConstant());
  opaqueMat.setEmissiveColorConstant(LegacyMaterialDefaults::emissiveColorConstant());
  opaqueMat.setEnableEmission(LegacyMaterialDefaults::enableEmissive());
  opaqueMat.setEnableThinFilm(LegacyMaterialDefaults::enableThinFilm());
  opaqueMat.setAlphaIsThinFilmThickness(LegacyMaterialDefaults::alphaIsThinFilmThickness());
  opaqueMat.setThinFilmThicknessConstant(LegacyMaterialDefaults::thinFilmThicknessConstant());
  return opaqueMat;
}

template<> OpaqueMaterialData LegacyMaterialData::as() const {
  // Legacy materials have parameters that can directly carry over onto the opaque material.
  const OpaqueMaterialData defaultLegacyOpaqueMaterial = createDefault();
  // Copy off the defaults, and make dynamic adjustments for the remaining params from this legacy material
  OpaqueMaterialData opaqueMat(defaultLegacyOpaqueMaterial);
  if (LegacyMaterialDefaults::useAlbedoTextureIfPresent()) {
    // Fork: when the wrapper-side PS classifier has tagged a sampler slot as
    // the actual diffuse texture (via the RS-149 protocol decoded in
    // setLegacyMaterialState), prefer it over colorTextures[0]. Slot 0 in the
    // FFP binding may carry a non-diffuse texture (e.g. NormalMap-only PS
    // shaders), and getColorTexture()'s binning loop can't always recover
    // the right one. Falls back to upstream behaviour when the protocol
    // wasn't written.
    if (protocolDiffuseTexture.isValid()) {
      opaqueMat.setAlbedoOpacityTexture(protocolDiffuseTexture);
    } else {
      opaqueMat.setAlbedoOpacityTexture(getColorTexture());
    }
  }
  // Fork: route the protocol-captured normal-map slot into the opaque
  // material's normal channel. Empty -> upstream secondary-texture path.
  if (normalTexture.isValid()) {
    opaqueMat.setNormalTexture(normalTexture);
    // Fork: FNV's normal maps are RGB tangent-space DXT-compressed (DirectX
    // convention, green-down), not Remix's native octahedral encoding. The
    // per-material flag drives a sample-time decode branch in
    // opaque_surface_material_interaction.slangh. Toggle the RtxOption off to
    // leave protocol-captured normals on the octahedral path (useful when the
    // user has pre-baked octahedral assets via LightspeedOctahedralConverter).
    if (LegacyMaterialDefaults::autoNormalTangentSpace()) {
      opaqueMat.setIsTangentSpaceNormalOverride(true);
    }
    // Fork: Bethesda DXT5n convention -- the same NormalMap's alpha channel encodes
    // specular intensity. The shader derives `roughness = 1.0 - normalSample.a`.
    // Gated separately from the tangent-space flag so a user can keep the normal
    // decode but disable the spec-as-roughness inversion if a USD replacement
    // provides its own roughness map.
    if (LegacyMaterialDefaults::autoRoughnessFromSpecular()) {
      opaqueMat.setIsRoughnessFromNormalAlphaOverride(true);
    }
  } else if (getColorTexture2().isValid()) {
    opaqueMat.setSecondaryTexture(getColorTexture2());
  }

  // Fork: route the protocol-captured glow slot into the opaque material's
  // emissive-color channel and flip enableEmission. Intensity follows the
  // existing rtx.legacyMaterial.emissiveIntensity legacy default and the
  // RtxOptions::emissiveIntensity master multiplier applied downstream in
  // rtx_scene_manager.cpp. Empty -> no auto-emissive (upstream behaviour).
  if (emissiveTexture.isValid() && LegacyMaterialDefaults::autoEmissive()) {
    opaqueMat.setEmissiveColorTexture(emissiveTexture);
    opaqueMat.setEnableEmission(true);
  }

  // Fork: route the protocol-captured height slot into the opaque material's
  // height channel for parallax-occlusion mapping. The OpaqueMaterialData
  // defaults set displaceIn=0.05/displaceOut=0.0 (from the constants table in
  // rtx_material_data.h), so binding a height texture is sufficient to activate
  // Remix's POM path -- RtOpaqueSurfaceMaterial::hasValidDisplacement() gates
  // on (displaceIn > 0 || displaceOut > 0) AND a non-null height texture index.
  // Users can tune depth with the existing rtx.displacement.* knobs.
  if (heightTexture.isValid() && LegacyMaterialDefaults::autoHeightMap()) {
    opaqueMat.setHeightTexture(heightTexture);
  }

  // Fork: route FNV multi-layer terrain captures (kRemixMultiLayerTerrainBit in
  // remixModifierFromD3D) from LegacyMaterialData's terrain{Albedo,Normal}Textures
  // arrays into OpaqueMaterialData's matching slots. No-op when the protocol bit
  // is unset. The scene manager (rtx_scene_manager.cpp) resolves the TextureRefs
  // to indices and registers an RtMultiLayerTerrainMaterial extension cache entry.
  fork_hooks::applyLegacyProtocolMultiLayerTerrain(*this, opaqueMat);

  // Indicate that we have an exact sampler to use on this material, directly from game
  if (getSampler().ptr()) {
    opaqueMat.setSamplerOverride(getSampler());
  }
  // Ignore colormap alpha of legacy texture if tagged as 'ignoreAlphaOnTextures'
  bool ignoreAlphaChannel = LegacyMaterialDefaults::ignoreAlphaChannel();
  if (!ignoreAlphaChannel) {
    ignoreAlphaChannel = lookupHash(RtxOptions::ignoreAlphaOnTextures(), getHash());
  }
  opaqueMat.setIgnoreAlphaChannel(ignoreAlphaChannel);
  return opaqueMat;
}

template<> TranslucentMaterialData LegacyMaterialData::as() const {
  TranslucentMaterialData transluscentMat;
  if (getSampler().ptr()) {
    transluscentMat.setSamplerOverride(getSampler());
  }
  return transluscentMat;
}

template<> RayPortalMaterialData LegacyMaterialData::as() const {
  RayPortalMaterialData portalMat;
  portalMat.getMaskTexture() = getColorTexture();
  portalMat.getMaskTexture2() = getColorTexture2();
  portalMat.setEnableEmission(true);
  portalMat.setEmissiveIntensity(1.f);
  portalMat.setSpriteSheetCols(1);
  portalMat.setSpriteSheetRows(1);
  if (getSampler().ptr()) {
    portalMat.setSamplerOverride(getSampler());
  }
  return portalMat;
}


} // namespace dxvk
