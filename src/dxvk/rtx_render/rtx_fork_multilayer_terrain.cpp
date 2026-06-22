// rtx_fork_multilayer_terrain.cpp -- routes multi-layer terrain captures from
// LegacyMaterialData (populated by setLegacyMaterialState when the wrapper
// sets kRemixMultiLayerTerrainBit) into the corresponding OpaqueMaterialData
// staging fields. The scene manager later resolves TextureRefs to texture
// indices and registers an RtMultiLayerTerrainMaterial in the extension cache,
// storing the resulting aux index on RtOpaqueSurfaceMaterial.
//
// See docs/superpowers/plans/2026-05-22-fnv-multilayer-terrain.md.

#include "rtx_fork_hooks.h"
#include "rtx_materials.h"

namespace dxvk::fork_hooks {

  void applyLegacyProtocolMultiLayerTerrain(const LegacyMaterialData& materialData, OpaqueMaterialData& opaqueOut) {
    if ((materialData.remixModifierFromD3D & kRemixMultiLayerTerrainBit) == 0u) {
      return;
    }
    if (materialData.terrainLayerCount == 0) {
      return;
    }
    opaqueOut.terrainLayerCount = materialData.terrainLayerCount;
    for (uint32_t i = 0; i < materialData.terrainLayerCount; i++) {
      opaqueOut.terrainAlbedoTextures[i] = materialData.terrainAlbedoTextures[i];
      opaqueOut.terrainNormalTextures[i] = materialData.terrainNormalTextures[i];
    }
  }

} // namespace dxvk::fork_hooks
