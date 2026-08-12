// src/dxvk/rtx_render/rtx_fork_submit.cpp
//
// Fork-owned file. Contains the implementations of fork_hooks:: functions
// for the SceneManager::submitExternalDraw path, lifted from
// rtx_scene_manager.cpp during the 2026-04-18 fork touchpoint-pattern refactor.
//
// See docs/fork-touchpoints.md for the full fork-hooks catalogue.
//
// NOTE: externalDrawObjectPicking accesses SceneManager::m_drawCallMeta, which
// is a private member. This file requires that SceneManager declare
// fork_hooks::externalDrawObjectPicking as a friend, OR that DrawCallMetaInfo
// and m_drawCallMeta be made accessible via a public accessor. Flagged for
// Phase 4 build-validation fixup.

#include "rtx_fork_hooks.h"

#include "rtx_asset_replacer.h"   // AssetReplacer, AssetReplacement
#include "rtx_options.h"          // RtxOptions::*, fast_unordered_set, InstanceCategories
#include "rtx_scene_manager.h"    // SceneManager, DrawCallMetaInfo

#include "dxvk_device.h"          // DxvkDevice::getCommon()->getResources()

#include "../../util/util_string.h"  // str::format for the category-key diagnostic
#include "../../util/util_once.h"    // ONCE

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>

namespace dxvk {
namespace fork_hooks {

namespace {
  // Names in InstanceCategories declaration order (rtx_types.h).
  static const char* const kCategoryNames[] = {
    "WorldUI", "WorldMatte", "Sky", "Ignore", "IgnoreLights", "IgnoreAntiCulling",
    "IgnoreMotionBlur", "IgnoreOpacityMicromap", "IgnoreAlphaChannel", "Hidden",
    "Particle", "Beam", "DecalStatic", "DecalDynamic", "DecalSingleOffset",
    "DecalNoOffset", "AlphaBlendToCutout", "Terrain", "AnimatedWater",
    "ThirdPersonPlayerModel", "ThirdPersonPlayerBody", "IgnoreBakedLighting",
    "IgnoreTransparencyLayer", "ParticleEmitter", "SmoothNormals", "HairCards",
    "MakeEmissive",
  };
  static_assert(sizeof(kCategoryNames) / sizeof(kCategoryNames[0]) ==
                  static_cast<size_t>(InstanceCategories::Count),
                "InstanceCategories changed - update kCategoryNames");

  std::atomic<bool> g_categoryKeyLogEnabled { false };

  // Diagnostic for API games: prints the identity each external draw is
  // categorized under, once per distinct key. This is the value to type into
  // rtx.skyBoxTextures and friends, and it is the only way to read the key for
  // an untextured draw, which has no grid thumbnail to click.
  //
  // Capped so a client that re-hashes its meshes every frame (CPU-transformed
  // geometry does) cannot turn the log into a leak. Toggling the option clears
  // the seen-set, so a second look re-dumps.
  void logCategoryKeyOnce(bool enabled, XXH64_hash_t key, bool keyIsMeshHash, XXH64_hash_t meshHash,
                          const CategoryFlags& categories) {
    constexpr size_t kMaxLogged = 512;

    // Common path when the diagnostic is off: one relaxed load, no lock.
    const bool wasEnabled = g_categoryKeyLogEnabled.exchange(enabled, std::memory_order_relaxed);
    const bool toggled = (wasEnabled != enabled);
    if (!enabled && !toggled) {
      return;
    }

    static std::mutex s_mutex;
    static std::unordered_set<XXH64_hash_t> s_seen;

    std::lock_guard lock { s_mutex };
    if (toggled) {
      s_seen.clear();
    }
    if (!enabled) {
      return;
    }
    if (s_seen.size() >= kMaxLogged) {
      ONCE(Logger::info(str::format(
        "[RTX-API-Cat] key log capped at ", kMaxLogged,
        " distinct keys; toggle rtx.logApiDrawCategoryKeys off and on to restart")));
      return;
    }
    if (!s_seen.insert(key).second) {
      return;
    }

    std::string cats;
    for (uint32_t i = 0; i < static_cast<uint32_t>(InstanceCategories::Count); i++) {
      if (categories.test(static_cast<InstanceCategories>(i))) {
        cats += cats.empty() ? "" : "|";
        cats += kCategoryNames[i];
      }
    }

    Logger::info(str::format(
      "[RTX-API-Cat] key=0x", std::hex, key, std::dec,
      keyIsMeshHash ? " (mesh hash, draw has no albedo texture)" : " (albedo texture hash)",
      " mesh=0x", std::hex, meshHash, std::dec,
      " categories=", cats.empty() ? "<none>" : cats.c_str()));
  }
} // namespace

  // ---------------------------------------------------------------------------
  // externalDrawMeshReplacement
  //
  // Checks for a USD mesh/light replacement keyed on the API mesh handle hash,
  // same as the D3D9 draw-call path. Returns the replacement vector pointer if
  // one is found (caller must call determineMaterialData + drawReplacements and
  // then return), or null if no replacement exists.
  // ---------------------------------------------------------------------------
  std::vector<AssetReplacement>* externalDrawMeshReplacement(
      AssetReplacer& replacer, XXH64_hash_t meshHash) {
    // Check for mesh/light replacements keyed on the API mesh handle, same as
    // the D3D9 draw-call path. This lets .usd replacements target API-submitted
    // meshes (e.g. hash the remix API mesh handle and author a replacement).
    return replacer.getReplacementsForMesh(meshHash);
  }

  // ---------------------------------------------------------------------------
  // externalDrawMaterialReplacement
  //
  // Checks for a USD material replacement via getReplacementMaterial() and
  // updates the caller's material pointer in-place if one is found.
  // ---------------------------------------------------------------------------
  void externalDrawMaterialReplacement(
      AssetReplacer& replacer, const MaterialData*& material) {
    // Check for material replacement (matches the D3D9 draw path behavior).
    MaterialData* pReplacementMaterial = replacer.getReplacementMaterial(material->getHash());
    if (pReplacementMaterial != nullptr) {
      material = pReplacementMaterial;
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawTextureCategories
  //
  // Auto-applies all texture-based instance categories for API-submitted draws,
  // and routes the two categories that have no consumer on this path.
  //
  // Identity: the albedo texture hash when the material has one — the same value
  // the dev-menu grid is keyed on, since fork_hooks::createTexture stamps the
  // client's CreateTexture hash onto the image and registers it under that key.
  // When there is no albedo texture the draw falls back to meshHash, which is
  // the client's own remixapi_MeshInfo::hash (the runtime reinterprets it as the
  // mesh handle verbatim, rtx_remix_api.cpp:1025). Without that fallback an
  // untextured API draw resolves to hash 0 and is skipped here *and* by the
  // object-picking hook below — i.e. uncategorizable and unclickable by
  // construction, which is what "that big mesh cannot be selected" was.
  //
  // The resolved identity is written to outTextureHash so subsequent hooks —
  // object picking — key on the same value.
  // ---------------------------------------------------------------------------
  void externalDrawTextureCategories(
      const MaterialData* material,
      DrawCallState& drawCall,
      XXH64_hash_t meshHash,
      const CategoryFlags& baseCategories,
      CameraType::Enum baseCameraType,
      XXH64_hash_t& outTextureHash) {
    // Reset to what the client submitted on this instance. submitExternalDraw
    // reuses one DrawCallState across every submesh and setCategory is add-only,
    // so without this a tag on submesh 0 leaks onto submesh 1 — and, worse, a
    // category *un*-ticked in the dev menu would never clear within a frame.
    drawCall.categories = baseCategories;
    drawCall.cameraType = baseCameraType;

    outTextureHash = 0;
    if (material != nullptr && material->getType() == MaterialDataType::Opaque) {
      const auto& opaqueMat = material->getOpaqueMaterialData();
      if (opaqueMat.getAlbedoOpacityTexture().isValid()) {
        outTextureHash = opaqueMat.getAlbedoOpacityTexture().getImageHash();
      }
    }

    const bool keyIsMeshHash = (outTextureHash == 0 || outTextureHash == kEmptyHash);
    if (keyIsMeshHash) {
      outTextureHash = meshHash;
    }

    if (outTextureHash == 0 || outTextureHash == kEmptyHash) {
      return;
    }

    const XXH64_hash_t key = outTextureHash;
    auto applyCategory = [&](const fast_unordered_set& hashSet, InstanceCategories cat) {
      if (hashSet.find(key) != hashSet.end()) {
        drawCall.setCategory(cat, true);
      }
    };

    // Parity with DrawCallState::setupCategoriesForTexture() (rtx_types.cpp) plus
    // the SmoothNormals line the D3D9 layer applies separately (d3d9_rtx.cpp).
    // Every row the dev-menu grid offers that maps to an InstanceCategory is
    // here; the ones that don't (uiTextures, lightmapTextures, lightConverter,
    // raytracedRenderTargetTextures) have no category to set.
    applyCategory(RtxOptions::skyBoxTextures(), InstanceCategories::Sky);
    applyCategory(RtxOptions::ignoreTextures(), InstanceCategories::Ignore);
    applyCategory(RtxOptions::worldSpaceUiTextures(), InstanceCategories::WorldUI);
    applyCategory(RtxOptions::worldSpaceUiBackgroundTextures(), InstanceCategories::WorldMatte);
    applyCategory(RtxOptions::particleTextures(), InstanceCategories::Particle);
    applyCategory(RtxOptions::beamTextures(), InstanceCategories::Beam);
    applyCategory(RtxOptions::decalTextures(), InstanceCategories::DecalStatic);
    applyCategory(RtxOptions::dynamicDecalTextures(), InstanceCategories::DecalDynamic);
    applyCategory(RtxOptions::singleOffsetDecalTextures(), InstanceCategories::DecalSingleOffset);
    applyCategory(RtxOptions::nonOffsetDecalTextures(), InstanceCategories::DecalNoOffset);
    applyCategory(RtxOptions::terrainTextures(), InstanceCategories::Terrain);
    applyCategory(RtxOptions::animatedWaterTextures(), InstanceCategories::AnimatedWater);
    applyCategory(RtxOptions::ignoreLights(), InstanceCategories::IgnoreLights);
    applyCategory(RtxOptions::antiCullingTextures(), InstanceCategories::IgnoreAntiCulling);
    applyCategory(RtxOptions::motionBlurMaskOutTextures(), InstanceCategories::IgnoreMotionBlur);
    applyCategory(RtxOptions::hideInstanceTextures(), InstanceCategories::Hidden);
    applyCategory(RtxOptions::opacityMicromapIgnoreTextures(), InstanceCategories::IgnoreOpacityMicromap);
    applyCategory(RtxOptions::ignoreAlphaOnTextures(), InstanceCategories::IgnoreAlphaChannel);
    applyCategory(RtxOptions::ignoreBakedLightingTextures(), InstanceCategories::IgnoreBakedLighting);
    applyCategory(RtxOptions::ignoreTransparencyLayerTextures(), InstanceCategories::IgnoreTransparencyLayer);
    applyCategory(RtxOptions::playerModelTextures(), InstanceCategories::ThirdPersonPlayerModel);
    applyCategory(RtxOptions::playerModelBodyTextures(), InstanceCategories::ThirdPersonPlayerBody);
    applyCategory(RtxOptions::particleEmitterTextures(), InstanceCategories::ParticleEmitter);
    applyCategory(RtxOptions::hairCardTextures(), InstanceCategories::HairCards);
    applyCategory(RtxOptions::smoothNormalsTextures(), InstanceCategories::SmoothNormals);
    applyCategory(RtxOptions::emissiveTextures(), InstanceCategories::MakeEmissive);

    // Mesh-hash sky, mirroring DrawCallState::setupCategoriesForGeometry(). The
    // API mesh handle IS the client's mesh hash, so rtx.skyBoxGeometries works
    // here even when the draw has a perfectly good albedo texture it shares with
    // non-sky geometry.
    const fast_unordered_set& skyGeometries = RtxOptions::skyBoxGeometries();
    if (skyGeometries.find(meshHash) != skyGeometries.end()) {
      drawCall.setCategory(InstanceCategories::Sky, true);
    }

    // Ignore must win over Sky. The dev menu stores categories in independent
    // hash sets, so a texture can legitimately remain in skyBoxTextures after
    // it is retagged Ignore. Promoting that draw to the sky camera first lets
    // the raster-sky path render it when skyMode is not Numos, bypassing the
    // Hidden bit below. Treat Ignore as an explicit request to remove the draw
    // from every sky path before deciding its camera type.
    const bool ignored = drawCall.testCategoryFlags(InstanceCategories::Ignore);
    if (ignored) {
      drawCall.removeCategory(InstanceCategories::Sky);
      drawCall.cameraType = CameraType::Main;
    }

    // Sky, routed. The category alone does nothing on this path: the only code
    // that turns sky into "does not occlude" tests drawCall.cameraType
    // (rtx_instance_manager.cpp:1024), and for API draws cameraType was frozen
    // from the instance's own API categoryFlags long before this hook runs.
    // D3D9 closes the same loop in CameraManager::processCameraData, which
    // external draws never reach. Promote it here instead.
    //
    // Deliberately AFTER submitExternalDraw baked the transforms: those come
    // from getCamera(state.cameraType) and there is no registered sky camera on
    // the API path, so re-baking against CameraType::Sky would hand the draw an
    // uninitialized RtCamera. The instance ends up hidden either way, so the
    // main-camera transforms it already has are the safe ones to keep.
    if (drawCall.testCategoryFlags(InstanceCategories::Sky)) {
      drawCall.cameraType = CameraType::Sky;
    }

    // Ignore, routed. Its only consumer anywhere is d3d9_rtx.cpp, inside the
    // D3D9 interception layer, so the category is inert for external draws —
    // including when supplied through REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE.
    // Fold it onto Hidden, which is consumed in shared code
    // (rtx_instance_manager.cpp:1018 -> mask 0 -> skipped in the TLAS build).
    // Dropping the submesh outright the way D3D9 does is the wrong shape here:
    // replacementInstance->prims[i] would keep pointing at an RtInstance that
    // stops being updated, and we would be leaning on instance GC to reap it.
    if (ignored) {
      drawCall.setCategory(InstanceCategories::Hidden, true);
    }

    logCategoryKeyOnce(RtxOptions::logApiDrawCategoryKeys(), key, keyIsMeshHash, meshHash,
                       drawCall.getCategoryFlags());
  }

  // ---------------------------------------------------------------------------
  // patchOpaqueMaterialFromCategories
  //
  // Category-driven material patches, applied where updateInstance already
  // patches materials for WorldUI. Both categories here are set from the
  // dev-menu texture lists on BOTH draw paths (setupCategoriesForTexture for
  // D3D9, externalDrawTextureCategories above for API draws), so this is the
  // one consumer each needs.
  //
  // Toggling a tag applies on the next frame: the patched MaterialData flows
  // through preserveInstance's onInstanceUpdated event into surface-material
  // creation, and those caches rebuild per frame.
  //
  // Known edge, accepted: a material shared between a MESH-hash-tagged
  // untextured draw and an untagged draw patches per instance, so the tag
  // stays per-draw correct — but the shared surface-material dedup may fold
  // them while their MaterialData hash matches. Texture-hash tags (the common
  // case) cannot hit this: same material implies same albedo implies same tag.
  // ---------------------------------------------------------------------------
  void patchOpaqueMaterialFromCategories(
      const RtInstance& instance,
      const MaterialData*& materialData,
      MaterialData& tmpStorage) {
    if (materialData->getType() != MaterialDataType::Opaque) {
      return;
    }

    const bool makeEmissive = instance.testCategoryFlags(InstanceCategories::MakeEmissive);
    const bool ignoreAlpha = instance.testCategoryFlags(InstanceCategories::IgnoreAlphaChannel) &&
                             !materialData->getOpaqueMaterialData().getIgnoreAlphaChannel();
    if (!makeEmissive && !ignoreAlpha) {
      return;
    }

    // Deep copy once, then patch in place. Guarded so a second patch in one
    // update cannot self-assign tmpStorage over itself.
    if (materialData != &tmpStorage) {
      tmpStorage = *materialData;
      materialData = &tmpStorage;
    }
    auto& opaque = tmpStorage.getOpaqueMaterialData();

    if (makeEmissive) {
      opaque.setEnableEmission(true);
      opaque.setEmissiveIntensity(RtxOptions::emissiveTexturesIntensity());
      if (opaque.getAlbedoOpacityTexture().isValid()) {
        opaque.setEmissiveColorTexture(opaque.getAlbedoOpacityTexture());
      } else {
        // Untextured draw (tagged by mesh hash): glow with its albedo colour.
        opaque.setEmissiveColorConstant(opaque.getAlbedoConstant());
      }
    }

    if (ignoreAlpha) {
      opaque.setIgnoreAlphaChannel(true);
    }
  }

  // ---------------------------------------------------------------------------
  // externalDrawObjectPicking
  //
  // Stores per-draw texture hash metadata in SceneManager::m_drawCallMeta when
  // object picking is active, mirroring the D3D9 draw path which populates
  // m_drawCallMeta in processDrawCallState. API draws supply their own
  // drawCallID via remixapi_InstanceInfoObjectPickingEXT, so we store it here.
  //
  // ACCESS NOTE: this function uses SceneManager::m_drawCallMeta (private) and
  // SceneManager::DrawCallMetaInfo (private nested type). A friend declaration
  // for this function is required in SceneManager, or the member / type must be
  // made accessible via a public helper. See file-level comment above.
  // ---------------------------------------------------------------------------
  void externalDrawObjectPicking(
      DxvkDevice& device,
      DrawCallState& drawCall,
      XXH64_hash_t textureHash,
      SceneManager& scene) {
    // Store texture hash metadata for object picking (mirrors the D3D9 draw
    // path which populates m_drawCallMeta in processDrawCallState). API draws
    // supply their own drawCallID via remixapi_InstanceInfoObjectPickingEXT,
    // so we hash it in directly here.
    //
    // textureHash is whatever externalDrawTextureCategories resolved as this
    // draw's tagging identity — the albedo hash, or the mesh hash for untextured
    // geometry. Recording the same value both ways is what makes a world click
    // land on the category the popup then edits.
    //
    // The generic writer, SceneManager::trackObjectPickingMeta, also runs for
    // external draws (rtx_scene_manager.cpp:1402) but reads the *legacy*
    // material's color texture, which API draws never populate — it would store
    // an empty hash. It uses emplace and runs after us, so our entry wins; the
    // "multiple draw calls with the same objectPickingValue" warning it then
    // logs once is expected on this path, not a client bug.
    const bool objectPickingActive = device.getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (objectPickingActive && drawCall.drawCallID != 0 &&
        textureHash != 0 && textureHash != kEmptyHash) {
      auto meta = SceneManager::DrawCallMetaInfo {};
      meta.legacyTextureHash = textureHash;

      std::lock_guard lock { scene.m_drawCallMeta.mutex };
      auto [iter, isNew] = scene.m_drawCallMeta.infos[scene.m_drawCallMeta.ticker].emplace(drawCall.drawCallID, meta);
      ONCE_IF_FALSE(isNew, Logger::warn(
        "Found multiple API draw calls with the same \'objectPickingValue\'. "
        "Some objects might not be available through object picking"));
    }
  }

} // namespace fork_hooks
} // namespace dxvk
