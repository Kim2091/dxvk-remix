/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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

#define RTX_REMIX_PNEXT_CHECK_STRUCTS

#include "rtx_asset_data_manager.h"
#include "rtx_asset_replacer.h"
#include "rtx_light_manager.h"
#include "rtx_lights.h"
#include "rtx_lens.h"
#include "rtx_objectpicking.h"
#include "rtx_option.h"
#include "rtx_globals.h"
#include "rtx_options.h"
#include "rtx_debug_view.h"
#include "rtx_fork_game_state.h"

#include "../dxvk_device.h"
#include "../dxvk_objects.h"
#include "../imgui/dxvk_imgui.h"
#include "rtx_context.h"
#include "rtx_fork_camera_origin.h"
#include "../dxvk_perf_counters.h"
#include "rtx_texture_manager.h"
#include "rtx_vram_budget.h"
#include "rtx_geometry_utils.h"

#include <remix/remix_c.h>
#include "rtx_remix_pnext.h"
#include "rtx_remix_api_device.h"
#include "rtx_host_settings.h"
#include "../../d3d11/d3d11_device.h"
#include "../../d3d11/d3d11_texture.h"

#include "../dxvk_image.h"
#include "../dxvk_util.h"

#include "../../util/util_math.h"
#include "../../util/util_vector.h"
#include "../../util/util_string.h"
#include "../../util/util_sentry.h"

#include "../../d3d9/d3d9_swapchain.h"
#include "../../d3d9/d3d9_texture.h"
#include "../../dxgi/dxgi_interfaces.h"

#include "rtx_option_layer.h"
#include "../../util/util_hash_set_layer.h"
#include "../../util/xxHash/xxhash.h"

#include "rtx_fork_hooks.h"
#include "rtx_fork_fnv_material.h"
#include "rtx_frontend_interface.h"
#include "rtx_fork_api_recorder.h"
#include "rtx_retained_scene.h"
#include "rtx_fork_d3d9_identity_trace.h"  // remixapi_fork_ReplayIdentityTrace
#include "rtx_option_manager.h"
#include "rtx/pass/screen_overlay/screen_overlay.h"  // SCREEN_OVERLAY_FLAG_*

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace dxvk {
  HRESULT CreateD3D9(
          bool           Extended,
          IDirect3D9Ex** ppDirect3D9Ex,
          bool           WithExternalSwapchain,
          bool           WithDrawCallConversion,
          bool           WithRemixAPI);

  extern bool g_allowSrgbConversionForOutput;
  extern bool g_forceKeepObjectPickingImage;
  extern bool g_remixApiEditorModeEnabled;   // defined in rtx_scene_manager.cpp

  extern std::array<uint8_t, 3> g_customHighlightColor;
}

namespace {
  struct PreparedD3D11Texture {
    PreparedD3D11Texture() = default;
    ~PreparedD3D11Texture() {
      dxvk::vram_budget::unregisterImportToken(this);
    }
    PreparedD3D11Texture(const PreparedD3D11Texture&) = delete;
    PreparedD3D11Texture& operator=(const PreparedD3D11Texture&) = delete;

    dxvk::Rc<dxvk::DxvkDevice> device;
    dxvk::Rc<dxvk::DxvkImageView> view;
    // D3D11_USAGE_IMMUTABLE: the game can never write the image again, so
    // Import may sample it in place instead of copying it.
    bool immutable = false;
    // PrepareD3D11TextureEx: longest side, in texels, of the largest mip to
    // import. 0 = no host cap.
    uint32_t maxDimension = 0;
    // The source texture, so Import can mark an in-place DEFAULT import for
    // copy-on-write. Held only as long as the token.
    dxvk::Com<ID3D11Texture2D> texture;
    // Prepare left the game's pending initial uploads unsubmitted
    // (rtx.hostImport.deferPrepareFlush); Import's CS command submits them
    // before it registers or copies the image.
    bool initFlushDeferred = false;
  };

  // Where D3D11 imports went, reported every 256 imports so a log shows how
  // much texture memory sharing and the mip cap are saving. "Not duplicated"
  // is the game image's size: what a copy would have cost.
  struct D3D11ImportStats {
    std::atomic<uint64_t> imports { 0 };
    std::atomic<uint64_t> shared { 0 };
    std::atomic<uint64_t> sharedBytes { 0 };
    std::atomic<uint64_t> copied { 0 };
    std::atomic<uint64_t> copiedBytes { 0 };
    std::atomic<uint64_t> capped { 0 };
    std::atomic<uint64_t> cappedSkippedBytes { 0 };
    std::atomic<uint64_t> copiedOnWrite { 0 };
    std::atomic<uint64_t> copiedOnWriteBytes { 0 };
    // remixapi_ReplaceD3D11TextureEx (and an import under a live hash): the
    // entry was re-pointed at a new image in place / the handle was gone by
    // the time the replace reached the render thread.
    std::atomic<uint64_t> replaced { 0 };
    std::atomic<uint64_t> replaceDropped { 0 };

    void note(bool isShared, uint64_t bytes, uint64_t skippedBytes) {
      (isShared ? shared : copied) += 1;
      (isShared ? sharedBytes : copiedBytes) += bytes;
      if (skippedBytes > 0) {
        capped += 1;
        cappedSkippedBytes += skippedBytes;
      }
      if ((++imports % 256) == 0) {
        dxvk::Logger::info(dxvk::str::format(
          "[HostImport] ", imports.load(), " imports: shared ", shared.load(), " (",
          sharedBytes.load() >> 20, " MiB not duplicated), copied ", copied.load(), " (",
          copiedBytes.load() >> 20, " MiB), mip cap applied to ", capped.load(), " (",
          cappedSkippedBytes.load() >> 20, " MiB of top mips skipped), copied on write ",
          copiedOnWrite.load(), " (", copiedOnWriteBytes.load() >> 20, " MiB), replaced in place ",
          replaced.load(), " (", replaceDropped.load(), " dropped: handle gone)"));
      }
    }
  };
  D3D11ImportStats s_d3d11ImportStats;

  // DEFAULT-usage images Remix samples in place (rtx.hostImport.shareDefaultTextures),
  // with the texture hashes registered on each. CS thread only: filled by the
  // import's CS command, drained by remixHostImportCopyOnWrite.
  std::unordered_map<const dxvk::DxvkImage*, std::vector<uint64_t>> s_sharedDefaultImports;
  // Every image registered by an import, with its texture hashes. CS thread only, filled by
  // registerImportedView and drained when the game destroys the texture (remixHostImportDetachReleased).
  std::unordered_map<const dxvk::DxvkImage*, std::vector<uint64_t>> s_importedImageHashes;
  uint64_t s_apiVersion{ 0 };
  IDirect3D9Ex* s_dxvkD3D9 { nullptr };
  std::unique_ptr<dxvk::RemixApiDevice> s_dxvkDevice;
  // Lock order invariant:
  //   s_mutex is taken BEFORE dxvk::RtxOptionImpl::getUpdateMutex().
  // Any code path that needs both must respect this order. RtxOption mutators
  // (setValue / addHash / clearHash / removeHash and friends) internally
  // acquire the RtxOption update mutex, so callers that already hold s_mutex
  // (e.g. mutateTextureHashOption via remixapi_AddTextureHash /
  // remixapi_RemoveTextureHash) must not invert this order. Conversely, any
  // lambda enqueued via EmitCs that runs under the RtxOption update mutex
  // must not re-enter s_mutex.
  dxvk::mutex s_mutex {};

  // The frame-boundary callback state lives in rtx_fork_api_entry.cpp. Access it through fork_hooks:
  //   notifyBeginScene()        — DrawInstance / DrawLightInstance call sites
  //   registerCallbacks()       — remixapi_RegisterCallbacks body
  //   shutdownCallbacks()       — remixapi_Shutdown
  //   presentCallbackDispatch() — remixapi_Present after native Present call

  // Global pending queues; applied to the device at safe points (frame/present)
  struct PendingLightUpdate { remixapi_LightHandle handle; std::optional<dxvk::RtLight> rtLight; };
  struct PendingDomeUpdate {
    remixapi_LightHandle handle;
    dxvk::Matrix4        transform;
    std::filesystem::path texturePath;
    dxvk::Vector3        radiance;
  };
  struct PendingLightCreate {
    remixapi_LightHandle handle;
    std::optional<dxvk::RtLight> rtLight;
    bool isDome;
    dxvk::Vector3 radiance;
    dxvk::Matrix4 transform;
    std::filesystem::path texturePath;
  };
  // Owned copy of a mesh surface buffered by remixapi_CreateMeshBatched
  // so the source pointers can safely go out of scope before the render
  // thread consumes them.
  struct OwnedSurface {
    std::vector<remixapi_HardcodedVertex> vertices;
    std::vector<uint32_t> indices;
    bool hasSkinning;
    uint32_t bonesPerVertex;
    std::vector<float> blendWeights;
    std::vector<uint32_t> blendIndices;
    remixapi_MaterialHandle material;
  };
  struct PendingMeshCreate {
    remixapi_MeshHandle handle;
    uint64_t hash;
    std::vector<OwnedSurface> surfaces;
  };
  std::vector<PendingLightCreate> s_pendingLightCreates;
  std::vector<PendingLightUpdate> s_pendingLightUpdates;
  std::vector<PendingDomeUpdate>  s_pendingDomeUpdates;
  std::vector<remixapi_LightHandle> s_pendingLightDestroys;
  std::vector<PendingMeshCreate>    s_pendingMeshCreates;
  // True while s_pendingMeshCreates may be non-empty. Written under s_mutex, read without it
  // by DrawInstance so an idle frame does not take s_mutex twice per draw.
  std::atomic<bool> s_hasPendingMeshCreates { false };
  // Mesh handles created through the API and not destroyed yet, guarded by s_mutex. Mirrors
  // AssetReplacer::m_extMeshes (the CS applies creates and destroys in the order they were
  // queued), so a repeated create of a live handle, which the CS would drop, can be dropped
  // before the host thread pays for buffers and copies.
  std::unordered_set<remixapi_MeshHandle> s_liveMeshHandles;
  // Track handles that were updated or created this frame to prevent re-adding after deletion in the same frame
  std::unordered_set<remixapi_LightHandle> s_handlesDeletedThisFrame;
  // Sticky: set once any external (C-API) light is registered. Used to gate the
  // per-frame persistent-light re-instancing on the native-D3D9 present path so
  // native-only consumers (no C-API lights) skip it entirely. See
  // remixapi_AutoInstancePersistentLights.
  std::atomic<bool> s_externalLightApiUsed { false };


  dxvk::RemixApiDevice* tryAsDxvk() {
    return s_dxvkDevice.get();
  }

  // from rtx_mod_usd.cpp
  XXH64_hash_t hack_getNextGeomHash() {
    // Deliberately lock-free. This runs on the CS thread via
    // buildExternalMeshSurfacesFromOwned <- applyPendingMeshCreatesOnCs, and a CS lambda
    // must never take s_mutex: the thread that holds s_mutex may be waiting on the device
    // lock, which is held by a thread waiting on CS progress. Taking it here closes that
    // cycle. An atomic gives the same descending sequence without the edge.
    static std::atomic<uint64_t> s_id { UINT64_MAX };
    const uint64_t id = s_id.fetch_sub(1, std::memory_order_relaxed) - 1;
    return XXH64(&id, sizeof(id), 0);
  }


  template<typename T>
  size_t sizeInBytes(const T* values, size_t count) {
    return sizeof(T) * count;
  }


  bool isHResultAliasedWithRemixErrorCode(HRESULT hr) {
    switch (hr) {
    case REMIXAPI_ERROR_CODE_HRESULT_NO_REQUIRED_GPU_FEATURES:
    case REMIXAPI_ERROR_CODE_HRESULT_DRIVER_VERSION_BELOW_MINIMUM:
    case REMIXAPI_ERROR_CODE_HRESULT_DXVK_INSTANCE_EXTENSION_FAIL:
    case REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_INSTANCE_FAIL:
    case REMIXAPI_ERROR_CODE_HRESULT_VK_CREATE_DEVICE_FAIL: return true;
    default: return false;
    }
  }


  namespace convert {
    using namespace dxvk;

    std::string tostr(const remixapi_MaterialHandle& h) {
      static_assert(sizeof h == sizeof uint64_t);
      return std::to_string(reinterpret_cast<uint64_t>(h));
    }

    Matrix4 tomat4(const remixapi_Transform& transform) {
      const auto& m = transform.matrix;
      return Matrix4 {
        m[0][0], m[1][0], m[2][0], 0.f,
        m[0][1], m[1][1], m[2][1], 0.f,
        m[0][2], m[1][2], m[2][2], 0.f,
        m[0][3], m[1][3], m[2][3], 1.f
      };
    }

    Vector2 tovec2(const remixapi_Float2D& v) {
      return Vector2{ v.x, v.y };
    }

    Vector3 tovec3(const remixapi_Float3D& v) {
      return Vector3 { v.x, v.y, v.z };
    }

    Vector3d tovec3d(const remixapi_Float3D& v) {
      return Vector3d{ v.x, v.y, v.z };
    }

    std::vector<float> toAnimatedFloat1D(const remixapi_AnimatedFloat1D& animated) {
      std::vector<float> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(animated.pData[i]);
        }
      }
      return result;
    }

    std::vector<vec2> toAnimatedFloat2D(const remixapi_AnimatedFloat2D& animated) {
      std::vector<vec2> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec2(animated.pData[i].x, animated.pData[i].y));
        }
      }
      return result;
    }

    std::vector<vec3> toAnimatedFloat3D(const remixapi_AnimatedFloat3D& animated) {
      std::vector<vec3> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec3(animated.pData[i].x, animated.pData[i].y, animated.pData[i].z));
        }
      }
      return result;
    }

    std::vector<vec4> toAnimatedFloat4D(const remixapi_AnimatedFloat4D& animated) {
      std::vector<vec4> result;
      if (animated.pData && animated.numberElements > 0) {
        result.reserve(animated.numberElements);
        for (uint32_t i = 0; i < animated.numberElements; ++i) {
          result.push_back(vec4(animated.pData[i].x, animated.pData[i].y, animated.pData[i].z, animated.pData[i].w));
        }
      }
      return result;
    }

    constexpr bool tobool(remixapi_Bool b) {
      return !!b;
    }

    std::filesystem::path topath(remixapi_Path p) {
      if (!p) {
        return {};
      }
      return p;
    }

    // --

    struct PreloadSource {
      std::filesystem::path albedoTexture;
      std::filesystem::path normalTexture;
      std::filesystem::path tangentTexture;
      std::filesystem::path emissiveTexture;
      std::filesystem::path transmittanceTexture;
      std::filesystem::path roughnessTexture;
      std::filesystem::path metallicTexture;
      std::filesystem::path heightTexture;
      std::filesystem::path subsurfaceTransmittanceTexture;
      std::filesystem::path subsurfaceThicknessTexture;
      std::filesystem::path subsurfaceSingleScatteringAlbedoTexture;
      std::filesystem::path subsurfaceRadiusTexture;

      // remixapi_MaterialInfoTerrainLayersEXT, copied out of the caller's structs (textures resolved later).
      struct TerrainLayers {
        uint32_t layerCount = 0;
        std::filesystem::path albedoTextures[REMIXAPI_TERRAIN_MAX_LAYERS];
        std::filesystem::path normalTextures[REMIXAPI_TERRAIN_MAX_LAYERS];
        std::filesystem::path heightTextures[REMIXAPI_TERRAIN_MAX_LAYERS]; // Terrain POM
        std::filesystem::path weightTextures[2];
        std::filesystem::path tintTexture;
        float weightUvScale[2] = { 1.f, 1.f };
        float weightUvOffset[2] = { 0.f, 0.f };
        bool normalsRgbTangentSpace = false;
      };
      std::shared_ptr<const TerrainLayers> terrainLayers;
    };

    // Copies an optional remixapi_MaterialInfoTerrainLayersEXT out of an opaque material's pNext chain.
    std::shared_ptr<const PreloadSource::TerrainLayers> makeTerrainLayersSource(const remixapi_MaterialInfo& info) {
      const auto* ext = pnext::find<remixapi_MaterialInfoTerrainLayersEXT>(&info);
      if (!ext || ext->layerCount == 0) {
        return nullptr;
      }
      auto src = std::make_shared<PreloadSource::TerrainLayers>();
      src->layerCount = std::min<uint32_t>(ext->layerCount, REMIXAPI_TERRAIN_MAX_LAYERS);
      for (uint32_t i = 0; i < src->layerCount; ++i) {
        src->albedoTextures[i] = topath(ext->albedoTextures[i]);
        src->normalTextures[i] = topath(ext->normalTextures[i]);
        src->heightTextures[i] = topath(ext->heightTextures[i]);
      }
      src->weightTextures[0] = topath(ext->weightTextures[0]);
      src->weightTextures[1] = topath(ext->weightTextures[1]);
      src->tintTexture = topath(ext->tintTexture);
      src->weightUvScale[0] = ext->weightUvScale.x;
      src->weightUvScale[1] = ext->weightUvScale.y;
      src->weightUvOffset[0] = ext->weightUvOffset.x;
      src->weightUvOffset[1] = ext->weightUvOffset.y;
      src->normalsRgbTangentSpace = tobool(ext->normalsRgbTangentSpace);
      return src;
    }

    PreloadSource makePreloadSource(const remixapi_MaterialInfo& info) {
      // TODO: C++20 designated initializers
      if (auto extOpaque = pnext::find<remixapi_MaterialInfoOpaqueEXT>(&info)) {
        auto extSubsurface = pnext::find<remixapi_MaterialInfoOpaqueSubsurfaceEXT>(&info);
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          {},                           // transmittanceTexture;
          topath(extOpaque->roughnessTexture),  // roughnessTexture;
          topath(extOpaque->metallicTexture),   // metallicTexture;
          topath(extOpaque->heightTexture),     // heightTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceTransmittanceTexture          : nullptr), // subsurfaceTransmittanceTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceThicknessTexture              : nullptr), // subsurfaceThicknessTexture;
          topath(extSubsurface ? extSubsurface->subsurfaceSingleScatteringAlbedoTexture : nullptr), // subsurfaceSingleScatteringAlbedoTexture;
          topath(s_apiVersion >= REMIXAPI_VERSION_MAKE(0, 5, 1) && extSubsurface ? extSubsurface->subsurfaceRadiusTexture : nullptr), // subsurfaceRadiusTexture;
          makeTerrainLayersSource(info), // terrainLayers
        };
      }
      if (auto extTranslucent = pnext::find<remixapi_MaterialInfoTranslucentEXT>(&info)) {
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          topath(extTranslucent->transmittanceTexture), // transmittanceTexture;
          {}, // roughnessTexture;
          {}, // metallicTexture;
          {}, // heightTexture;
          {}, // subsurfaceTransmittanceTexture;
          {}, // subsurfaceThicknessTexture;
          {}, // subsurfaceSingleScatteringAlbedoTexture;
          {}, // subsurfaceRadiusTexture;
        };
      }
      if (auto extPortal = pnext::find<remixapi_MaterialInfoPortalEXT>(&info)) {
        return PreloadSource {
          topath(info.albedoTexture),   // albedoTexture;
          topath(info.normalTexture),   // normalTexture;
          topath(info.tangentTexture),  // tangentTexture;
          topath(info.emissiveTexture), // emissiveTexture;
          {}, // transmittanceTexture;
          {}, // roughnessTexture;
          {}, // metallicTexture;
          {}, // heightTexture;
          {}, // subsurfaceTransmittanceTexture;
          {}, // subsurfaceThicknessTexture;
          {}, // subsurfaceSingleScatteringAlbedoTexture;
          {}, // subsurfaceRadiusTexture;
        };
      }
      return {};
    }

    MaterialData toRtMaterialFinalized(dxvk::DxvkContext& ctx, const MaterialData& materialWithoutPreload, const PreloadSource& preload) {
      auto preloadTexture = [&ctx](const std::filesystem::path& path)->TextureRef {
        if (path.empty()) {
          return {};
        }

        // "0x<hex>" pseudo-paths refer to API-uploaded textures (remixapi_CreateTexture) by image hash.
        TextureRef hashRef;
        if (dxvk::fork_hooks::textureHashPathLookup(ctx, path, hashRef)) {
          return hashRef;
        }

        auto assetData = AssetDataManager::get().findAsset(path.string());
        if (assetData == nullptr) {
          return {};
        }
        auto uploadedTexture = ctx.getCommonObjects()->getTextureManager()
          .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, false);
        return TextureRef { uploadedTexture };
      };

      switch (materialWithoutPreload.getType()) {
      case MaterialDataType::Opaque:
      {
        const auto& src = materialWithoutPreload.getOpaqueMaterialData();
        MaterialData result { OpaqueMaterialData{
          preloadTexture(preload.albedoTexture),
          preloadTexture(preload.normalTexture),
          preloadTexture(preload.tangentTexture),
          preloadTexture(preload.heightTexture),
          preloadTexture(preload.roughnessTexture),
          preloadTexture(preload.metallicTexture),
          preloadTexture(preload.emissiveTexture),
          preloadTexture(preload.subsurfaceTransmittanceTexture),
          preloadTexture(preload.subsurfaceThicknessTexture),
          preloadTexture(preload.subsurfaceSingleScatteringAlbedoTexture),
          preloadTexture(preload.subsurfaceRadiusTexture),
          TextureRef(),
          src.getAnisotropyConstant(),
          src.getEmissiveIntensity(),
          src.getAlbedoConstant(),
          src.getOpacityConstant(),
          src.getRoughnessConstant(),
          src.getMetallicConstant(),
          src.getEmissiveColorConstant(),
          src.getEnableEmission(),
          src.getSkyLitParticle(), // Sky-lit particle flag rides through preload
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getEnableThinFilm(),
          src.getAlphaIsThinFilmThickness(),
          src.getThinFilmThicknessConstant(),
          src.getUseLegacyAlphaState(),
          src.getBlendEnabled(),
          src.getBlendType(),
          src.getInvertedBlend(),
          src.getAlphaTestType(),
          src.getAlphaTestReferenceValue(),
          src.getDisplaceIn(),
          src.getDisplaceOut(),
          src.getSubsurfaceTransmittanceColor(),
          src.getSubsurfaceMeasurementDistance(),
          src.getSubsurfaceSingleScatteringAlbedo(),
          src.getSubsurfaceVolumetricAnisotropy(),
          src.getSubsurfaceDiffusionProfile(),
          src.getSubsurfaceRadius(),
          src.getSubsurfaceRadiusScale(),
          src.getSubsurfaceMaxSampleRadius(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV(),
          src.getEnableDLSSControlMask(),
          src.getDLSSControlMaskIntensity(),
          src.getDLSSControlMaskToneStrength(),
          src.getDLSSControlMaskStructuralStrength(),
          src.getRoughnessEncoding(),
          src.getSpecularMetalStrength(),
          src.getSpecularMetalParams(),
          src.getSpecularMetalTint(),
          src.getAlbedoTint()
        } };
        // The shader decode flags the API conversion set (tinted emissive).
        result.getOpaqueMaterialData().setForkFnvFlags(src.getForkFnvFlags());
        // Terrain-layers extension; every layer/weight/tint texture is resolved here, once, on the CS thread.
        if (preload.terrainLayers) {
          const auto& t = *preload.terrainLayers;
          auto layers = std::make_shared<TerrainLayersMaterialData>();
          layers->layerCount = std::min<uint32_t>(t.layerCount, TerrainLayersMaterialData::kMaxLayers);
          for (uint32_t i = 0; i < layers->layerCount; ++i) {
            layers->albedoTextures[i] = preloadTexture(t.albedoTextures[i]);
            layers->normalTextures[i] = preloadTexture(t.normalTextures[i]);
            layers->heightTextures[i] = preloadTexture(t.heightTextures[i]);
            // The texture's own format says whether its alpha can be gloss. The FNV wrapper uploads normal
            // layers without meaningful alpha (DXT1 sources, constant-255 alpha) as BC1, so no extra field is needed.
            if (dxvk::FnvMaterialProtocol::textureHasGlossAlpha(layers->normalTextures[i])) {
              layers->normalHasAlphaMask |= 1u << i;
            }
          }
          layers->weightTextures[0] = preloadTexture(t.weightTextures[0]);
          layers->weightTextures[1] = preloadTexture(t.weightTextures[1]);
          layers->tintTexture = preloadTexture(t.tintTexture);
          layers->weightUvScale[0] = t.weightUvScale[0];
          layers->weightUvScale[1] = t.weightUvScale[1];
          layers->weightUvOffset[0] = t.weightUvOffset[0];
          layers->weightUvOffset[1] = t.weightUvOffset[1];
          layers->normalsRgbTangentSpace = t.normalsRgbTangentSpace;
          result.getOpaqueMaterialData().setTerrainLayers(std::move(layers));
        }
        return result;
      }
      case MaterialDataType::Translucent: 
      {
        const auto& src = materialWithoutPreload.getTranslucentMaterialData();
        return MaterialData { TranslucentMaterialData {
          preloadTexture(preload.normalTexture),
          preloadTexture(preload.transmittanceTexture),
          preloadTexture(preload.emissiveTexture),
          src.getRefractiveIndex(),
          src.getTransmittanceColor(),
          src.getTransmittanceMeasurementDistance(),
          src.getEnableEmission(),
          src.getEmissiveIntensity(),
          src.getEmissiveColorConstant(),
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getEnableThinWalled(),
          src.getThinWallThickness(),
          src.getEnableDiffuseLayer(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV()
        } };
      }
      case MaterialDataType::RayPortal:
      {
        const auto& src = materialWithoutPreload.getRayPortalMaterialData();
        return MaterialData { RayPortalMaterialData {
          preloadTexture(preload.emissiveTexture),
          {}, // unused
          src.getRayPortalIndex(),
          src.getSpriteSheetRows(),
          src.getSpriteSheetCols(),
          src.getSpriteSheetFPS(),
          src.getRotationSpeed(),
          src.getEnableEmission(),
          src.getEmissiveIntensity(),
          src.getFilterMode(),
          src.getWrapModeU(),
          src.getWrapModeV()
        } };
      }
      default: assert(0); return materialWithoutPreload;
      }
    }

    MaterialData toRtMaterialWithoutTexturePreload(const remixapi_MaterialInfo& info) {
      if (auto extOpaque = pnext::find<remixapi_MaterialInfoOpaqueEXT>(&info)) {
        auto extSubsurface = pnext::find<remixapi_MaterialInfoOpaqueSubsurfaceEXT>(&info);
        auto extSpecGloss = pnext::find<remixapi_MaterialInfoOpaqueSpecGlossEXT>(&info);
        auto extAlbedoTint = pnext::find<remixapi_MaterialInfoOpaqueAlbedoTintEXT>(&info);
        // Upstream introduced these fields at 0.6.5; on the Remix Plus ABI line that is 0.1000.1.
        const bool hasDlssControlMask = s_apiVersion >= REMIXAPI_VERSION_MAKE(0, 1000, 1);
        MaterialData result { OpaqueMaterialData {
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          {},
          extOpaque->anisotropy,
          info.emissiveIntensity,
          tovec3(extOpaque->albedoConstant),
          extOpaque->opacityConstant,
          extOpaque->roughnessConstant,
          extOpaque->metallicConstant,
          tovec3(info.emissiveColorConstant),
          info.emissiveIntensity > 0.f,
          false, // OpaqueMaterial::SkyLitParticle - not exposed via remixapi
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          tobool(extOpaque->thinFilmThickness_hasvalue),
          tobool(extOpaque->alphaIsThinFilmThickness),
          extOpaque->thinFilmThickness_hasvalue ? extOpaque->thinFilmThickness_value : 200.f, // default OpaqueMaterial::ThinFilmThicknessConstant
          tobool(extOpaque->useDrawCallAlphaState), // OpaqueMaterial::UseLegacyAlphaState
          tobool(extOpaque->blendType_hasvalue),
          extOpaque->blendType_hasvalue ? static_cast<BlendType>(extOpaque->blendType_value) : BlendType::kAlpha,  // default OpaqueMaterial::BlendType
          tobool(extOpaque->invertedBlend),
          static_cast<AlphaTestType>(extOpaque->alphaTestType),
          extOpaque->alphaReferenceValue,
          extOpaque->displaceIn,
          (s_apiVersion >= REMIXAPI_VERSION_MAKE(0, 4, 2)) ? extOpaque->displaceOut : 0.f,
          extSubsurface ? tovec3(extSubsurface->subsurfaceTransmittanceColor) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceMeasurementDistance : 0.f,
          extSubsurface ? tovec3(extSubsurface->subsurfaceSingleScatteringAlbedo) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceVolumetricAnisotropy : 0.f,
          extSubsurface ? static_cast<bool>(extSubsurface->subsurfaceDiffusionProfile) : false,
          extSubsurface ? tovec3(extSubsurface->subsurfaceRadius) : Vector3{ 0.5f, 0.5f, 0.5f },
          extSubsurface ? extSubsurface->subsurfaceRadiusScale : 0.f,
          extSubsurface ? extSubsurface->subsurfaceMaxSampleRadius : 0.f,
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
          hasDlssControlMask ? tobool(extOpaque->enableDlssControlMask) : true,
          hasDlssControlMask ? extOpaque->dlssControlMaskIntensity : 1.f,
          hasDlssControlMask ? extOpaque->dlssControlMaskToneStrength : 1.f,
          hasDlssControlMask ? extOpaque->dlssControlMaskStructuralStrength : 1.f,
          // Specular/gloss source maps. Only the smoothness encodings
          // are meaningful for a roughness texture; anything else is ignored.
          uint8_t(extSpecGloss &&
                  (extSpecGloss->roughnessEncoding == REMIXAPI_TEXTURE_ENCODING_SMOOTHNESS_IN_RED ||
                   extSpecGloss->roughnessEncoding == REMIXAPI_TEXTURE_ENCODING_SMOOTHNESS_IN_GREEN)
                  ? extSpecGloss->roughnessEncoding : REMIXAPI_TEXTURE_ENCODING_NONE),
          extSpecGloss ? extSpecGloss->metalStrength : 0.f,
          extSpecGloss ? Vector3{ extSpecGloss->specularMetalLow, extSpecGloss->specularMetalHigh,
                                  extSpecGloss->metalAlbedoLuminanceMax }
                       : Vector3{ 0.35f, 0.65f, 1.f },
          extSpecGloss ? tovec3(extSpecGloss->specularTint) : Vector3{ 0.f, 0.f, 0.f },
          // Per-material albedo texture tint (1,1,1 = none).
          extAlbedoTint ? tovec3(extAlbedoTint->albedoTint) : Vector3{ 1.f, 1.f, 1.f },
        } };
        // An emissive texture tinted by a non-white emissiveColorConstant, as the
        // FNV protocol tints a glow map by its emittance colour (OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TEXTURE_TINTED;
        // no shader change). Upstream ignores the constant whenever an emissive texture is bound, so this is opt-in.
        const bool hasEmissiveTexture = info.emissiveTexture && info.emissiveTexture[0] != L'\0';
        const bool tintedConstant = info.emissiveColorConstant.x != 1.f || info.emissiveColorConstant.y != 1.f ||
                                    info.emissiveColorConstant.z != 1.f;
        if (dxvk::FnvMaterialProtocol::apiGlowTint() && hasEmissiveTexture && info.emissiveIntensity > 0.f && tintedConstant) {
          result.getOpaqueMaterialData().setForkFnvFlags(OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TEXTURE_TINTED);
        }
        return result;
      }
      if (auto extTranslucent = pnext::find<remixapi_MaterialInfoTranslucentEXT>(&info)) {
        return MaterialData { TranslucentMaterialData {
          {},
          {},
          {},
          extTranslucent->refractiveIndex,
          tovec3(extTranslucent->transmittanceColor),
          extTranslucent->transmittanceMeasurementDistance,
          info.emissiveIntensity > 0.f,
          info.emissiveIntensity,
          tovec3(info.emissiveColorConstant),
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          tobool(extTranslucent->thinWallThickness_hasvalue),
          extTranslucent->thinWallThickness_hasvalue ? extTranslucent->thinWallThickness_value : 0.001f, // default TranslucentMaterial::ThinWallThickness
          tobool(extTranslucent->useDiffuseLayer),
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
        } };
      }
      if (auto extPortal = pnext::find<remixapi_MaterialInfoPortalEXT>(&info)) {
        return MaterialData { RayPortalMaterialData {
          {},
          {}, // unused
          extPortal->rayPortalIndex,
          info.spriteSheetRow,
          info.spriteSheetCol,
          info.spriteSheetFps,
          extPortal->rotationSpeed,
          info.emissiveIntensity > 0.f,
          info.emissiveIntensity,
          info.filterMode,
          info.wrapModeU,
          info.wrapModeV,
        } };
      }

      assert(0);
      return MaterialData { OpaqueMaterialData {} };
    }

    // --
    frontend::CameraView toCameraView(remixapi_CameraType from) {
      switch (from) {
      case REMIXAPI_CAMERA_TYPE_WORLD: return frontend::CameraView::Main;
      case REMIXAPI_CAMERA_TYPE_VIEW_MODEL: return frontend::CameraView::ViewModel;
      case REMIXAPI_CAMERA_TYPE_SKY: return frontend::CameraView::Sky;
      default: assert(0); return frontend::CameraView::Main;
      }
    }

    // The camera's view and matrices. Parameterized cameras also carry worldToView in double precision.
    frontend::CameraSetup toCameraSetup(const remixapi_CameraInfo& info) {
      frontend::CameraSetup result;
      result.view = toCameraView(info.type);
      if (auto params = pnext::find<remixapi_CameraInfoParameterizedEXT>(&info)) {
        {
          const auto newViewToWorld = Matrix4d {
           Vector4d{ normalize(tovec3d(params->right)), 0.0 },
           Vector4d{ normalize(tovec3d(params->up)), 0.0 },
           Vector4d{ normalize(tovec3d(params->forward)), 0.0 },
           Vector4d{ tovec3d(params->position), 1.0 },
          };
          result.preciseWorldToView = inverse(newViewToWorld);
          result.worldToView = Matrix4 { *result.preciseWorldToView };
        }
        {
          constexpr bool isLhs = true;
          auto proj = float4x4 {};
          proj.SetupByHalfFovy(
            DegToRad(params->fovYInDegrees) / 2,
            params->aspect,
            params->nearPlane,
            params->farPlane,
            isLhs ? PROJ_LEFT_HANDED : 0);
          static_assert(sizeof result.viewToProjection == sizeof proj);
          memcpy(&result.viewToProjection, &proj, sizeof float4x4);
        }
        return result;
      }
      result.worldToView = Matrix4 { info.view };
      result.viewToProjection = Matrix4 { info.projection };
      return result;
    }

    // --

    std::optional<RtLightShaping> toRtLightShaping(const remixapi_LightInfoLightShaping* info) {
      if (info) {
        return RtLightShaping::tryCreate(
          true,
          tovec3(info->direction),
          std::cos(DegToRad(info->coneAngleDegrees)),
          info->coneSoftness,
          info->focusExponent
        );
      }

      // Note: Default constructed Light Shaping returned when no info is provided to have a valid but disabled
      // Light Shaping object (different from returning an empty optional here, which means creation of a Light
      // Shaping failed).
      return RtLightShaping{};
    }

    std::optional<RtLight> toRtLight(const remixapi_LightInfo& info) {
      if (auto src = pnext::find<remixapi_LightInfoUSDEXT>(&info)) {
        if (auto lightData = LightData::tryCreate(*src)) {
          return lightData->toRtLight();
        }
        return {};
      }
      if (auto src = pnext::find<remixapi_LightInfoSphereEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtSphereLight::tryCreate(
          tovec3(src->position),
          tovec3(info.radiance),
          src->radius,
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoRectEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtRectLight::tryCreate(
          tovec3(src->position),
          Vector2{src->xSize, src->ySize},
          tovec3(src->xAxis),
          tovec3(src->yAxis),
          tovec3(src->direction),
          tovec3(info.radiance),
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoDiskEXT>(&info)) {
        const auto shaping = toRtLightShaping(src->shaping_hasvalue ? &src->shaping_value : nullptr);

        if (!shaping.has_value()) {
          return {};
        }

        return RtDiskLight::tryCreate(
          tovec3(src->position),
          Vector2{src->xRadius, src->yRadius},
          tovec3(src->xAxis),
          tovec3(src->yAxis),
          tovec3(src->direction),
          tovec3(info.radiance),
          *shaping,
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoCylinderEXT>(&info)) {
        return RtCylinderLight::tryCreate(
          tovec3(src->position),
          src->radius,
          tovec3(src->axis),
          src->axisLength,
          tovec3(info.radiance),
          src->volumetricRadianceScale
        );
      }
      if (auto src = pnext::find<remixapi_LightInfoDistantEXT>(&info)) {
        return RtDistantLight::tryCreate(
          tovec3(src->direction),
          DegToRad(src->angularDiameterDegrees * 0.5f),
          tovec3(info.radiance),
          src->volumetricRadianceScale
        );
      }

      // Note: Return an empty optional if the LightInfo struct does not contain a supported
      // LightInfo extension struct.
      return {};
    }

    // --

    frontend::CameraView categoryToCameraView(remixapi_InstanceCategoryFlags flags) {
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_VIEW_MODEL) {
        return frontend::CameraView::ViewModel;
      }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SKY) {
        return frontend::CameraView::Sky;
      }
      return frontend::CameraView::Main;
    }

    CategoryFlags toRtCategories(remixapi_InstanceCategoryFlags flags) {
      CategoryFlags result { 0 };
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_UI                 ){ result.set(InstanceCategories::WorldUI               ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_WORLD_MATTE              ){ result.set(InstanceCategories::WorldMatte            ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SKY                      ){ result.set(InstanceCategories::Sky                   ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE                   ){ result.set(InstanceCategories::Ignore                ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_LIGHTS            ){ result.set(InstanceCategories::IgnoreLights          ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ANTI_CULLING      ){ result.set(InstanceCategories::IgnoreAntiCulling     ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_MOTION_BLUR       ){ result.set(InstanceCategories::IgnoreMotionBlur      ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_OPACITY_MICROMAP  ){ result.set(InstanceCategories::IgnoreOpacityMicromap ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_ALPHA_CHANNEL     ){ result.set(InstanceCategories::IgnoreAlphaChannel    ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HIDDEN                   ){ result.set(InstanceCategories::Hidden                ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE                 ){ result.set(InstanceCategories::Particle              ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_BEAM                     ){ result.set(InstanceCategories::Beam                  ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_STATIC             ){ result.set(InstanceCategories::DecalStatic           ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_DYNAMIC            ){ result.set(InstanceCategories::DecalDynamic          ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_SINGLE_OFFSET      ){ result.set(InstanceCategories::DecalSingleOffset     ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_DECAL_NO_OFFSET          ){ result.set(InstanceCategories::DecalNoOffset         ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_ALPHA_BLEND_TO_CUTOUT    ){ result.set(InstanceCategories::AlphaBlendToCutout    ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_TERRAIN                  ){ result.set(InstanceCategories::Terrain               ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_ANIMATED_WATER           ){ result.set(InstanceCategories::AnimatedWater         ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_MODEL){ result.set(InstanceCategories::ThirdPersonPlayerModel); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_THIRD_PERSON_PLAYER_BODY ){ result.set(InstanceCategories::ThirdPersonPlayerBody ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_IGNORE_BAKED_LIGHTING    ){ result.set(InstanceCategories::IgnoreBakedLighting   ); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE_EMITTER)         { result.set(InstanceCategories::ParticleEmitter); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_SMOOTH_NORMALS)            { result.set(InstanceCategories::SmoothNormals); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HAIR_CARDS)                { result.set(InstanceCategories::HairCards); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HIGHLIGHT_HOSTILE)         { result.set(InstanceCategories::HighlightHostile); }
      if (flags & REMIXAPI_INSTANCE_CATEGORY_BIT_HEAT_SOURCE)               { result.set(InstanceCategories::HeatSource); }
      
      static_assert((int)InstanceCategories::Count == 27, "Instance categories changed, please update Remix SDK");
      return result;
    }

    RtxParticleSystemDesc toRtParticleDesc(const remixapi_InstanceInfoParticleSystemLegacyEXT& info) {
      RtxParticleSystemDesc desc {};

      // Lifetimes
      desc.minTimeToLive = info.minTimeToLive;
      desc.maxTimeToLive = info.maxTimeToLive;

      // Initial 
      desc.spawnRatePerSecond = info.spawnRatePerSecond;
      desc.initialVelocityFromMotion = info.initialVelocityFromMotion;
      desc.initialVelocityFromNormal = info.initialVelocityFromNormal;
      desc.initialVelocityConeAngleDegrees = info.initialVelocityConeAngleDegrees;
      desc.gravityForce = info.gravityForce;
      desc.motionTrailMultiplier = info.motionTrailMultiplier;

      // Turbulence
      desc.turbulenceFrequency = info.turbulenceFrequency;
      desc.turbulenceForce = info.turbulenceForce;

      // Convert spawn/target pairs to 2-element animated vectors
      // Color: spawn at index 0, target at index 1
      desc.minColor = { vec4(info.minSpawnColor.x, info.minSpawnColor.y, info.minSpawnColor.z, info.minSpawnColor.w), 
                        vec4(info.minTargetColor.x, info.minTargetColor.y, info.minTargetColor.z, info.minTargetColor.w) };
      desc.maxColor = { vec4(info.maxSpawnColor.x, info.maxSpawnColor.y, info.maxSpawnColor.z, info.maxSpawnColor.w), 
                        vec4(info.maxTargetColor.x, info.maxTargetColor.y, info.maxTargetColor.z, info.maxTargetColor.w) };

      // Size: spawn at index 0, target at index 1
      desc.minSize = { vec2(info.minSpawnSize, info.minSpawnSize), vec2(info.minTargetSize, info.minTargetSize) };
      desc.maxSize = { vec2(info.maxSpawnSize, info.maxSpawnSize), vec2(info.maxTargetSize, info.maxTargetSize) };

      // Rotation speed: spawn at index 0, target at index 1
      desc.minRotationSpeed = { info.minSpawnRotationSpeed, info.minTargetRotationSpeed };
      desc.maxRotationSpeed = { info.maxSpawnRotationSpeed, info.maxTargetRotationSpeed };

      // Velocity: spawn at index 0, target at index 1
      desc.maxVelocity = { vec3(info.maxSpeed, info.maxSpeed, info.maxSpeed), vec3(info.maxSpeed, info.maxSpeed, info.maxSpeed) };

      // Collision
      desc.collisionThickness = info.collisionThickness;
      desc.collisionRestitution = info.collisionRestitution;
      desc.collisionMode = ParticleCollisionMode::Bounce;

      // Counts/flags
      desc.maxNumParticles = info.maxNumParticles;
      desc.useTurbulence = static_cast<uint8_t>(info.useTurbulence);
      desc.alignParticlesToVelocity = static_cast<uint8_t>(info.alignParticlesToVelocity);
      desc.useSpawnTexcoords = static_cast<uint8_t>(info.useSpawnTexcoords);
      desc.enableCollisionDetection = static_cast<uint8_t>(info.enableCollisionDetection);
      desc.enableMotionTrail = static_cast<uint8_t>(info.enableMotionTrail);
      desc.hideEmitter = static_cast<uint8_t>(info.hideEmitter);

      // Types/modes
      desc.billboardType = static_cast<ParticleBillboardType>(info.billboardType);
      desc.spriteSheetMode = ParticleSpriteSheetMode::UseMaterialSpriteSheet;
      desc.randomFlipAxis = ParticleRandomFlipAxis::None;

      return desc;
    }

    RtxParticleSystemDesc toRtParticleDesc(const remixapi_InstanceInfoParticleSystemEXT& info) {
      RtxParticleSystemDesc desc;
      desc.maxNumParticles = info.maxNumParticles;

      // Animated float conversions
      desc.minColor = toAnimatedFloat4D(info.minColor);
      desc.maxColor = toAnimatedFloat4D(info.maxColor);
      desc.minRotationSpeed = toAnimatedFloat1D(info.minRotationSpeed);
      desc.maxRotationSpeed = toAnimatedFloat1D(info.maxRotationSpeed);
      desc.minSize = toAnimatedFloat2D(info.minSize);
      desc.maxSize = toAnimatedFloat2D(info.maxSize);
      desc.maxVelocity = toAnimatedFloat3D(info.maxVelocity);

      desc.minTimeToLive = info.minTimeToLive;
      desc.maxTimeToLive = info.maxTimeToLive;
      desc.initialVelocityFromNormal = info.initialVelocityFromNormal;
      desc.initialVelocityConeAngleDegrees = info.initialVelocityConeAngleDegrees;
      desc.initialVelocityFromMotion = info.initialVelocityFromMotion;
      desc.turbulenceFrequency = info.turbulenceFrequency;
      desc.turbulenceForce = info.turbulenceForce;
      desc.motionTrailMultiplier = info.motionTrailMultiplier;
      desc.spawnRatePerSecond = info.spawnRatePerSecond;
      desc.collisionThickness = info.collisionThickness;
      desc.collisionRestitution = info.collisionRestitution;
      desc.gravityForce = info.gravityForce;
      desc.billboardType = static_cast<ParticleBillboardType>(info.billboardType);
      desc.hideEmitter = static_cast<uint8_t>(info.hideEmitter);
      desc.enableMotionTrail = static_cast<uint8_t>(info.enableMotionTrail);
      desc.useTurbulence = static_cast<uint8_t>(info.useTurbulence);
      desc.alignParticlesToVelocity = static_cast<uint8_t>(info.alignParticlesToVelocity);
      desc.useSpawnTexcoords = static_cast<uint8_t>(info.useSpawnTexcoords);
      desc.enableCollisionDetection = static_cast<uint8_t>(info.enableCollisionDetection);
      desc.dragCoefficient = info.dragCoefficient;
      desc.initialRotationDeviationDegrees = info.initialRotationDeviationDegrees;
      desc.spawnBurstDuration = info.spawnBurstDuration;
      desc.attractorRadius = info.attractorRadius;
      desc.attractorPosition = tovec3(info.attractorPosition);
      desc.attractorForce = info.attractorForce;
      desc.spriteSheetMode = static_cast<ParticleSpriteSheetMode>(info.spriteSheetMode);
      desc.collisionMode = static_cast<ParticleCollisionMode>(info.collisionMode);
      desc.randomFlipAxis = static_cast<ParticleRandomFlipAxis>(info.randomFlipAxis);
      desc.restrictVelocityX = static_cast<uint8_t>(info.restrictVelocityX);
      desc.restrictVelocityY = static_cast<uint8_t>(info.restrictVelocityY);
      desc.restrictVelocityZ = static_cast<uint8_t>(info.restrictVelocityZ);

      return desc;
    }

    frontend::LegacyBlendDesc toLegacyBlendDesc(const remixapi_InstanceInfoBlendEXT& ext, bool vertexColorIsBakedLighting) {
      frontend::LegacyBlendDesc blend;
      blend.alphaTestEnabled = ext.alphaTestEnabled;
      blend.alphaTestReferenceValue = ext.alphaTestReferenceValue;
      blend.alphaTestCompareOp = ext.alphaTestCompareOp;
      blend.alphaBlendEnabled = ext.alphaBlendEnabled;
      blend.srcColorBlendFactor = ext.srcColorBlendFactor;
      blend.dstColorBlendFactor = ext.dstColorBlendFactor;
      blend.colorBlendOp = ext.colorBlendOp;
      blend.srcAlphaBlendFactor = ext.srcAlphaBlendFactor;
      blend.dstAlphaBlendFactor = ext.dstAlphaBlendFactor;
      blend.alphaBlendOp = ext.alphaBlendOp;
      blend.writeMask = ext.writeMask;
      blend.textureColorArg1Source = ext.textureColorArg1Source;
      blend.textureColorArg2Source = ext.textureColorArg2Source;
      blend.textureColorOperation = ext.textureColorOperation;
      blend.textureAlphaArg1Source = ext.textureAlphaArg1Source;
      blend.textureAlphaArg2Source = ext.textureAlphaArg2Source;
      blend.textureAlphaOperation = ext.textureAlphaOperation;
      blend.tFactor = ext.tFactor;
      blend.isTextureFactorBlend = ext.isTextureFactorBlend;
      blend.isVertexColorBakedLighting = vertexColorIsBakedLighting;
      return blend;
    }

    // What the host stated about an instance, in the scene interface's terms. The extension structs are copied, so the
    // description outlives the host's pointers.
    frontend::SceneInstanceDesc toSceneInstanceDesc(const remixapi_InstanceInfo& info) {
      frontend::SceneInstanceDesc desc;
      desc.mesh = reinterpret_cast<frontend::MeshHandle>(info.mesh);
      desc.objectToWorld = tomat4(info.transform);
      desc.cameraView = categoryToCameraView(info.categoryFlags);
      desc.categories = toRtCategories(info.categoryFlags).raw();
      desc.doubleSided = tobool(info.doubleSided);
      if (auto objectPicking = pnext::find<remixapi_InstanceInfoObjectPickingEXT>(&info)) {
        desc.objectPickingValue = objectPicking->objectPickingValue;
      }
      if (auto extBones = pnext::find<remixapi_InstanceInfoBoneTransformsEXT>(&info)) {
        const uint32_t boneCount =
          extBones->boneTransforms_count < REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT ?
          extBones->boneTransforms_count : REMIXAPI_INSTANCE_INFO_MAX_BONES_COUNT;
        std::vector<Matrix4>& bones = desc.boneTransforms.emplace();
        bones.reserve(boneCount);
        for (uint32_t boneIdx = 0; boneIdx < boneCount; boneIdx++) {
          bones.push_back(tomat4(extBones->boneTransforms_values[boneIdx]));
        }
      }
      if (auto extBlend = pnext::find<remixapi_InstanceInfoBlendEXT>(&info)) {
        const bool vertexColorIsBakedLighting = (s_apiVersion >= REMIXAPI_VERSION_MAKE(0, 5, 2))
          ? extBlend->isVertexColorBakedLighting != 0 : RtxOptions::vertexColorIsBakedLighting();
        desc.blend = toLegacyBlendDesc(*extBlend, vertexColorIsBakedLighting);
      }
      if (auto extParticles = pnext::find<remixapi_InstanceInfoParticleSystemEXT>(&info)) {
        desc.particles.emplace(toRtParticleDesc(*extParticles));
      }
      if (auto extParticles = pnext::find<remixapi_InstanceInfoParticleSystemLegacyEXT>(&info)) {
        desc.particles.emplace(toRtParticleDesc(*extParticles));
      }
      if (auto extInstancing = pnext::find<remixapi_InstanceInfoGpuInstancingEXT>(&info)) {
        if (extInstancing->instanceTransforms_count > 0 && extInstancing->instanceTransforms_values) {
          desc.gpuInstancing.reserve(extInstancing->instanceTransforms_count);
          for (uint32_t i = 0; i < extInstancing->instanceTransforms_count; ++i) {
            desc.gpuInstancing.push_back(tomat4(extInstancing->instanceTransforms_values[i]));
          }
        }
      }
      if (auto extMaterialOverride = pnext::find<remixapi_InstanceInfoMaterialOverrideEXT>(&info)) {
        desc.materialOverride = reinterpret_cast<frontend::MaterialHandle>(extMaterialOverride->material);
      }
      return desc;
    }

    frontend::PreparedInstance toPreparedInstance(const remixapi_InstanceInfo& info) {
      return frontend::prepareInstance(toSceneInstanceDesc(info));
    }
  }
}

namespace {
  // remixapi_AssetKeyEXT (rtx.scene.assetKeys). Found by walking the chain: it can extend a mesh or a material, which the
  // typed pnext lookup (one root type per struct) does not allow.
  const remixapi_AssetKeyEXT* findAssetKey(const void* pNext) {
    struct Header { remixapi_StructType sType; void* pNext; };
    for (const Header* h = static_cast<const Header*>(pNext); h != nullptr; h = static_cast<const Header*>(h->pNext)) {
      if (h->sType == REMIXAPI_STRUCT_TYPE_ASSET_KEY_EXT) {
        return reinterpret_cast<const remixapi_AssetKeyEXT*>(h);
      }
      if (h->sType == REMIXAPI_STRUCT_TYPE_NONE) {
        break;
      }
    }
    return nullptr;
  }

  dxvk::AssetResolver::AssetKey toAssetKey(const remixapi_AssetKeyEXT& ext) {
    dxvk::AssetResolver::AssetKey key;
    key.key = ext.key;
    if (ext.aliases_values != nullptr && ext.aliases_count > 0) {
      key.aliases.assign(ext.aliases_values, ext.aliases_values + ext.aliases_count);
    }
    if (ext.assetName != nullptr) {
      key.name = ext.assetName;
    }
    return key;
  }

  // Queued after the asset's own create command, so a material's content hash exists when it runs.
  void emitAssetKey(dxvk::RemixApiDevice* remixDevice, const void* pNext, uint64_t handle, bool isMesh) {
    const remixapi_AssetKeyEXT* ext = findAssetKey(pNext);
    if (ext == nullptr) {
      return;
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs([cKey = toAssetKey(*ext), handle, isMesh](dxvk::DxvkContext* ctx) mutable {
      auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
      if (isMesh) {
        assets->assetResolver().setMeshKey(handle, std::move(cKey));
      } else if (const dxvk::MaterialData* material = assets->accessExternalMaterial(reinterpret_cast<remixapi_MaterialHandle>(handle))) {
        assets->assetResolver().setMaterialKey(material->getHash(), std::move(cKey));
      }
      assets->noteChangedExternalAsset(handle);
    });
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMaterial(
    const remixapi_MaterialInfo* info,
    remixapi_MaterialHandle* out_handle) {
    REMIX_FORK_RECORD(CreateMaterial, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateMaterial, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_MATERIAL_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_MaterialHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_MaterialHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // async load
    {
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs([cHandle = handle,
                         cMaterialData = convert::toRtMaterialWithoutTexturePreload(*info),
                         cPreloadSrc = convert::makePreloadSource(*info)](dxvk::DxvkContext* ctx) {
      auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
      assets->makeMaterialWithTexturePreload(
        *ctx,
        cHandle,
        convert::toRtMaterialFinalized(*ctx, cMaterialData, cPreloadSrc));
    });
    }
    emitAssetKey(remixDevice, info->pNext, info->hash, false);

    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyMaterial(
    remixapi_MaterialHandle handle) {
    REMIX_FORK_RECORD(DestroyMaterial, handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DestroyMaterial, 1 }; // rtx.profile.detailedLogging
    if (auto remixDevice = tryAsDxvk()) {
      std::lock_guard lock { s_mutex };
      remixDevice->EmitCs([cHandle = handle](dxvk::DxvkContext* ctx) {
        static_cast<dxvk::RtxContext*>(ctx)->drainSceneCommands();  // rtx.scene.commandQueue
        auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
        assets->destroyExternalMaterial(cHandle);
      });
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
  }

  // Skinned meshes are re-read by GPU skinning every frame, so their buffers
  // live in device memory, filled once from the host copy on the command
  // stream (the policy UsdMod applies to its dynamic meshes). Static meshes
  // stay host-visible: Remix copies them into its own device-local geometry
  // buffers when it builds them and does not read the source again.
  struct StagedMeshBuffer {
    dxvk::Rc<dxvk::DxvkBuffer> dst;
    dxvk::Rc<dxvk::DxvkBuffer> src;
  };

  dxvk::Rc<dxvk::DxvkBuffer> stageToDeviceLocal(dxvk::DxvkDevice* device,
      const dxvk::Rc<dxvk::DxvkBuffer>& host, std::vector<StagedMeshBuffer>& staged,
      VkBufferUsageFlags extraUsage = 0) {
    if (host == nullptr) {
      return host;
    }
    dxvk::DxvkBufferCreateInfo info = host->info();
    info.usage |= extraUsage;
    auto local = device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      dxvk::DxvkMemoryStats::Category::RTXBuffer,
      extraUsage != 0 ? "Remix API static mesh buffer" : "Remix API skinned mesh buffer");
    staged.push_back({ local, host });
    return local;
  }

  void copyStagedMeshBuffers(dxvk::DxvkContext* ctx, const std::vector<StagedMeshBuffer>& staged) {
    for (const auto& s : staged) {
      ctx->copyBuffer(s.dst, 0, s.src, 0, s.src->info().size);
    }
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMesh(
    const remixapi_MeshInfo* info,
    remixapi_MeshHandle* out_handle) {
    REMIX_FORK_RECORD(CreateMesh, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateMesh, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_MESH_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_MeshHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_MeshHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    emitAssetKey(remixDevice, info->pNext, info->hash, true);  // rtx.scene.assetKeys

    // Opt-in in-place geometry refresh: the caller is re-submitting a mesh that
    // is already registered under this hash, with identical topology but moved
    // vertex positions (FaceGen morphs). Keeping the hash -- and therefore the
    // handle, the external-draw identity hash, and the spatial-map hash --
    // stable is what preserves the instance's previous-frame correspondence.
    const bool refreshGeometry = pnext::find<remixapi_MeshInfoRefreshGeometryEXT>(info) != nullptr;

    // A re-create of a live handle without the refresh extension is dropped by
    // AssetReplacer::registerExternalMesh; drop it here, before the buffers and copies.
    if (!refreshGeometry && dxvk::RtxOptions::HostImport::skipRepeatedMeshCreate()) {
      std::lock_guard lock { s_mutex };
      if (s_liveMeshHandles.count(handle) != 0) {
        *out_handle = handle;
        return REMIXAPI_ERROR_CODE_SUCCESS;
      }
    }

    auto allocatedSurfaces = std::vector<dxvk::RasterGeometry> {};
    auto stagedBuffers = std::vector<StagedMeshBuffer> {};

    for (size_t i = 0; i < info->surfaces_count; i++) {
      const remixapi_MeshInfoSurfaceTriangles& src = info->surfaces_values[i];

      const size_t vertexDataSize = sizeInBytes(src.vertices_values, src.vertices_count);
      const size_t indexDataSize = sizeInBytes(src.indices_values, src.indices_count);

      auto allocBuffer = [](dxvk::RemixApiDevice* device, size_t sizeInBytes) -> dxvk::Rc<dxvk::DxvkBuffer> {
        if (sizeInBytes == 0) {
          return {};
        }
        auto bufferInfo = dxvk::DxvkBufferCreateInfo {};
        {
          bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
          bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
          bufferInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
          bufferInfo.size = dxvk::align(sizeInBytes, dxvk::CACHE_LINE_SIZE);
        }
        return device->GetDXVKDevice()->createBuffer(
            bufferInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            dxvk::DxvkMemoryStats::Category::RTXBuffer,
           "Remix API mesh buffer");
      };

      dxvk::Rc<dxvk::DxvkBuffer> vertexBuffer = allocBuffer(remixDevice, vertexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> indexBuffer = allocBuffer(remixDevice, indexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> skinningBuffer = nullptr;

      auto vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
      memcpy(vertexSlice.mapPtr(0), src.vertices_values, vertexDataSize);

      auto indexSlice = dxvk::DxvkBufferSlice {};
      if (indexDataSize > 0) {
        indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
        memcpy(indexSlice.mapPtr(0), src.indices_values, indexDataSize);
      }

      auto blendWeightsSlice = dxvk::DxvkBufferSlice {};
      auto blendIndicesSlice = dxvk::DxvkBufferSlice {};
      if (src.skinning_hasvalue) {
        size_t wordsPerCompressedTuple = dxvk::divCeil(src.skinning_value.bonesPerVertex, 4u);
        size_t sizeInBytes_weights = sizeInBytes(src.skinning_value.blendWeights_values, src.skinning_value.blendWeights_count);
        size_t sizeInBytes_indices = src.vertices_count * wordsPerCompressedTuple * sizeof(uint32_t);

        skinningBuffer = allocBuffer(remixDevice, sizeInBytes_weights + sizeInBytes_indices);

        // Encode bone indices into compressed byte form
        auto compressedBlendIndices = std::vector<uint32_t> {};
        compressedBlendIndices.resize(src.vertices_count * wordsPerCompressedTuple);
        for (size_t vert = 0; vert < src.vertices_count; vert++) {
          const uint32_t* dstCompressed = &compressedBlendIndices[vert * wordsPerCompressedTuple];
          const uint32_t* blendIndicesStorage = &src.skinning_value.blendIndices_values[vert * src.skinning_value.bonesPerVertex];

          for (int j = 0; j < src.skinning_value.bonesPerVertex; j += 4) {
            uint32_t vertIndices = 0;
            for (int k = 0; k < 4 && j + k < src.skinning_value.bonesPerVertex; ++k) {
              vertIndices |= blendIndicesStorage[j + k] << 8 * k;
            }
            *(uint32_t*) &dstCompressed[j / 4] = vertIndices;
          }
        }

        assert(sizeInBytes_indices == compressedBlendIndices.size() * sizeof(compressedBlendIndices[0]));

        blendWeightsSlice = dxvk::DxvkBufferSlice { skinningBuffer, 0, sizeInBytes_weights };
        blendIndicesSlice = dxvk::DxvkBufferSlice { skinningBuffer, sizeInBytes_weights, sizeInBytes_indices };

        memcpy(blendWeightsSlice.mapPtr(0), src.skinning_value.blendWeights_values, sizeInBytes_weights);
        memcpy(blendIndicesSlice.mapPtr(0), compressedBlendIndices.data(), sizeInBytes_indices);

        dxvk::DxvkDevice* dxvkDevice = remixDevice->GetDXVKDevice().ptr();
        vertexBuffer = stageToDeviceLocal(dxvkDevice, vertexBuffer, stagedBuffers);
        indexBuffer = stageToDeviceLocal(dxvkDevice, indexBuffer, stagedBuffers);
        skinningBuffer = stageToDeviceLocal(dxvkDevice, skinningBuffer, stagedBuffers);
        vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
        if (indexDataSize > 0) {
          indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
        }
        blendWeightsSlice = dxvk::DxvkBufferSlice { skinningBuffer, 0, sizeInBytes_weights };
        blendIndicesSlice = dxvk::DxvkBufferSlice { skinningBuffer, sizeInBytes_weights, sizeInBytes_indices };
      } else if (dxvk::RtxOptions::HostImport::deviceLocalStaticMeshes() && indexDataSize > 0) {
        // Static mesh: upload device-local once, marked so its BlasEntry uses these buffers in
        // place; the host-visible copy is freed after the upload.
        dxvk::DxvkDevice* dxvkDevice = remixDevice->GetDXVKDevice().ptr();
        vertexBuffer = stageToDeviceLocal(dxvkDevice, vertexBuffer, stagedBuffers, dxvk::kAliasableGeometryUsage);
        indexBuffer = stageToDeviceLocal(dxvkDevice, indexBuffer, stagedBuffers, dxvk::kAliasableGeometryUsage);
        vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
        indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
      }

      auto dst = dxvk::RasterGeometry {};
      {
        dst.externalMaterial = src.material;
        dst.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        dst.cullMode = VK_CULL_MODE_NONE; // this will be overwritten by the instance info at draw time
        dst.frontFace = VK_FRONT_FACE_CLOCKWISE;
        dst.vertexCount = src.vertices_count; assert(src.vertices_count < std::numeric_limits<uint32_t>::max());
        dst.positionBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, position), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.normalBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, normal), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
        dst.texcoordBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, texcoord), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32_SFLOAT };
        dst.color0Buffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, color), sizeof(remixapi_HardcodedVertex), VK_FORMAT_B8G8R8A8_UNORM };
        if (src.skinning_hasvalue) {
          dst.numBonesPerVertex = src.skinning_value.bonesPerVertex;
          dst.blendWeightBuffer = dxvk::RasterBuffer { blendWeightsSlice, 0, uint32_t(sizeof(float) * src.skinning_value.bonesPerVertex), VK_FORMAT_R32_SFLOAT };
          dst.blendIndicesBuffer = dxvk::RasterBuffer { blendIndicesSlice, 0, uint32_t(sizeof(uint32_t) * dxvk::divCeil(src.skinning_value.bonesPerVertex, 4u)), VK_FORMAT_R8G8B8A8_USCALED };
        }

        dst.indexCount = src.indices_count;
        static_assert(sizeof(src.indices_values[0]) == 4);
        dst.indexBuffer = dxvk::RasterBuffer { indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32 };
        // look comments in UsdMod::Impl::processMesh, rtx_mod_usd.cpp
        dst.hashes[dxvk::HashComponents::Indices] = dst.hashes[dxvk::HashComponents::VertexPosition] = hack_getNextGeomHash();
        dst.hashes[dxvk::HashComponents::VertexTexcoord] = hack_getNextGeomHash();
        dst.hashes[dxvk::HashComponents::GeometryDescriptor] = hack_getNextGeomHash();
        dst.hashes[dxvk::HashComponents::VertexLayout] = hack_getNextGeomHash();
        dst.hashes.precombine();
      }
      allocatedSurfaces.push_back(std::move(dst));
    }

    std::lock_guard lock { s_mutex };
    s_liveMeshHandles.insert(handle);

    remixDevice->EmitCs([cHandle = handle, cSurfaces = std::move(allocatedSurfaces),
                         cStaged = std::move(stagedBuffers),
                         cRefresh = refreshGeometry](dxvk::DxvkContext* ctx) mutable {
      copyStagedMeshBuffers(ctx, cStaged);
      auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
      assets->registerExternalMesh(cHandle, std::move(cSurfaces), cRefresh);
    });

    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Shared builder used by both the synchronous remixapi_CreateMesh path and
  // the batched path (remixapi_CreateMeshBatched) when the render thread
  // materializes deferred pending mesh creates. Allocates host-visible DXVK
  // buffers, memcpys owned vertex/index/skinning data, and fills a vector of
  // dxvk::RasterGeometry ready for registerExternalMesh().
  std::vector<dxvk::RasterGeometry> buildExternalMeshSurfacesFromOwned(
      dxvk::DxvkContext* ctx, const std::vector<OwnedSurface>& surfaces) {
    auto allocatedSurfaces = std::vector<dxvk::RasterGeometry> {};
    allocatedSurfaces.reserve(surfaces.size());
    auto stagedBuffers = std::vector<StagedMeshBuffer> {};

    auto allocBuffer = [](dxvk::RemixApiDevice* device, size_t sizeInBytes) -> dxvk::Rc<dxvk::DxvkBuffer> {
      if (sizeInBytes == 0) {
        return {};
      }
      auto bufferInfo = dxvk::DxvkBufferCreateInfo {};
      bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      bufferInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      bufferInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
      bufferInfo.size = dxvk::align(sizeInBytes, dxvk::CACHE_LINE_SIZE);
      return device->GetDXVKDevice()->createBuffer(
          bufferInfo,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          dxvk::DxvkMemoryStats::Category::RTXBuffer,
          "Remix API mesh buffer");
    };

    for (const OwnedSurface& src : surfaces) {
      const size_t vertexDataSize = sizeInBytes(src.vertices.data(), src.vertices.size());
      const size_t indexDataSize = sizeInBytes(src.indices.data(), src.indices.size());

      dxvk::Rc<dxvk::DxvkBuffer> vertexBuffer = allocBuffer(s_dxvkDevice.get(), vertexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> indexBuffer = allocBuffer(s_dxvkDevice.get(), indexDataSize);
      dxvk::Rc<dxvk::DxvkBuffer> skinningBuffer = nullptr;

      auto vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
      if (vertexDataSize > 0) {
        memcpy(vertexSlice.mapPtr(0), src.vertices.data(), vertexDataSize);
      }

      auto indexSlice = dxvk::DxvkBufferSlice {};
      if (indexDataSize > 0) {
        indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
        memcpy(indexSlice.mapPtr(0), src.indices.data(), indexDataSize);
      }

      auto blendWeightsSlice = dxvk::DxvkBufferSlice {};
      auto blendIndicesSlice = dxvk::DxvkBufferSlice {};
      if (src.hasSkinning) {
        size_t wordsPerCompressedTuple = dxvk::divCeil(src.bonesPerVertex, 4u);
        size_t sizeInBytes_weights = sizeInBytes(src.blendWeights.data(), src.blendWeights.size());
        size_t sizeInBytes_indices = src.vertices.size() * wordsPerCompressedTuple * sizeof(uint32_t);

        skinningBuffer = allocBuffer(s_dxvkDevice.get(), sizeInBytes_weights + sizeInBytes_indices);

        auto compressedBlendIndices = std::vector<uint32_t> {};
        compressedBlendIndices.resize(src.vertices.size() * wordsPerCompressedTuple);
        for (size_t vert = 0; vert < src.vertices.size(); vert++) {
          uint32_t* dstCompressed = &compressedBlendIndices[vert * wordsPerCompressedTuple];
          const uint32_t* blendIndicesStorage = &src.blendIndices[vert * src.bonesPerVertex];
          for (uint32_t j = 0; j < src.bonesPerVertex; j += 4) {
            uint32_t vertIndices = 0;
            for (uint32_t k = 0; k < 4 && j + k < src.bonesPerVertex; ++k) {
              vertIndices |= blendIndicesStorage[j + k] << 8 * k;
            }
            dstCompressed[j / 4] = vertIndices;
          }
        }
        assert(sizeInBytes_indices == compressedBlendIndices.size() * sizeof(compressedBlendIndices[0]));

        blendWeightsSlice = dxvk::DxvkBufferSlice { skinningBuffer, 0, sizeInBytes_weights };
        blendIndicesSlice = dxvk::DxvkBufferSlice { skinningBuffer, sizeInBytes_weights, sizeInBytes_indices };
        memcpy(blendWeightsSlice.mapPtr(0), src.blendWeights.data(), sizeInBytes_weights);
        memcpy(blendIndicesSlice.mapPtr(0), compressedBlendIndices.data(), sizeInBytes_indices);

        dxvk::DxvkDevice* dxvkDevice = ctx->getDevice().ptr();
        vertexBuffer = stageToDeviceLocal(dxvkDevice, vertexBuffer, stagedBuffers);
        indexBuffer = stageToDeviceLocal(dxvkDevice, indexBuffer, stagedBuffers);
        skinningBuffer = stageToDeviceLocal(dxvkDevice, skinningBuffer, stagedBuffers);
        vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
        if (indexDataSize > 0) {
          indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
        }
        blendWeightsSlice = dxvk::DxvkBufferSlice { skinningBuffer, 0, sizeInBytes_weights };
        blendIndicesSlice = dxvk::DxvkBufferSlice { skinningBuffer, sizeInBytes_weights, sizeInBytes_indices };
      } else if (dxvk::RtxOptions::HostImport::deviceLocalStaticMeshes() && indexDataSize > 0) {
        // Same as remixapi_CreateMesh: device-local, aliasable by the BlasEntry.
        dxvk::DxvkDevice* dxvkDevice = ctx->getDevice().ptr();
        vertexBuffer = stageToDeviceLocal(dxvkDevice, vertexBuffer, stagedBuffers, dxvk::kAliasableGeometryUsage);
        indexBuffer = stageToDeviceLocal(dxvkDevice, indexBuffer, stagedBuffers, dxvk::kAliasableGeometryUsage);
        vertexSlice = dxvk::DxvkBufferSlice { vertexBuffer };
        indexSlice = dxvk::DxvkBufferSlice { indexBuffer };
      }

      auto dst = dxvk::RasterGeometry {};
      dst.externalMaterial = src.material;
      dst.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      dst.cullMode = VK_CULL_MODE_NONE; // this will be overwritten by the instance info at draw time
      dst.frontFace = VK_FRONT_FACE_CLOCKWISE;
      dst.vertexCount = static_cast<uint32_t>(src.vertices.size());
      assert(src.vertices.size() < std::numeric_limits<uint32_t>::max());
      dst.positionBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, position), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
      dst.normalBuffer   = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, normal),   sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32B32_SFLOAT };
      dst.texcoordBuffer = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, texcoord), sizeof(remixapi_HardcodedVertex), VK_FORMAT_R32G32_SFLOAT };
      dst.color0Buffer   = dxvk::RasterBuffer { vertexSlice, offsetof(remixapi_HardcodedVertex, color),    sizeof(remixapi_HardcodedVertex), VK_FORMAT_B8G8R8A8_UNORM };
      if (src.hasSkinning) {
        dst.numBonesPerVertex = src.bonesPerVertex;
        dst.blendWeightBuffer  = dxvk::RasterBuffer { blendWeightsSlice, 0, uint32_t(sizeof(float) * src.bonesPerVertex), VK_FORMAT_R32_SFLOAT };
        dst.blendIndicesBuffer = dxvk::RasterBuffer { blendIndicesSlice, 0, uint32_t(sizeof(uint32_t) * dxvk::divCeil(src.bonesPerVertex, 4u)), VK_FORMAT_R8G8B8A8_USCALED };
      }
      dst.indexCount = static_cast<uint32_t>(src.indices.size());
      dst.indexBuffer = dxvk::RasterBuffer { indexSlice, 0, sizeof(uint32_t), VK_INDEX_TYPE_UINT32 };
      // look comments in UsdMod::Impl::processMesh, rtx_mod_usd.cpp
      dst.hashes[dxvk::HashComponents::Indices] = dst.hashes[dxvk::HashComponents::VertexPosition] = hack_getNextGeomHash();
      dst.hashes[dxvk::HashComponents::VertexTexcoord] = hack_getNextGeomHash();
      dst.hashes[dxvk::HashComponents::GeometryDescriptor] = hack_getNextGeomHash();
      dst.hashes[dxvk::HashComponents::VertexLayout] = hack_getNextGeomHash();
      dst.hashes.precombine();

      allocatedSurfaces.push_back(std::move(dst));
    }
    copyStagedMeshBuffers(ctx, stagedBuffers);
    return allocatedSurfaces;
  }

  // Drain s_pendingMeshCreates and register each pending mesh with the asset
  // replacer. Must be called on the render (CS) thread; accepts a DxvkContext.
  void applyPendingMeshCreatesOnCs(dxvk::DxvkContext* ctx, std::vector<PendingMeshCreate>& meshCreates) {
    if (meshCreates.empty()) {
      return;
    }
    auto& assets = ctx->getCommonObjects()->getSceneManager().getAssetReplacer();
    for (auto& mesh : meshCreates) {
      auto allocatedSurfaces = buildExternalMeshSurfacesFromOwned(ctx, mesh.surfaces);
      assets->registerExternalMesh(mesh.handle, std::move(allocatedSurfaces));
    }
  }

  // Flush pending mesh creates immediately on the calling thread via EmitCs.
  // Used from the DrawInstance path to make batched meshes available before
  // the first draw that references them.
  void flushPendingMeshes(dxvk::RemixApiDevice* remixDevice) {
    // Idle check without s_mutex: DrawInstance takes it again right after, so the locked
    // check doubled the lock traffic of every draw when nothing is queued.
    if (dxvk::RtxOptions::HostImport::lockFreeIdleMeshFlush() && !s_hasPendingMeshCreates.load()) {
      return;
    }
    std::vector<PendingMeshCreate> meshCreates;
    {
      std::lock_guard lock { s_mutex };
      if (s_pendingMeshCreates.empty()) {
        return;
      }
      meshCreates.swap(s_pendingMeshCreates);
      s_hasPendingMeshCreates.store(false);
    }

    auto devLock = remixDevice->LockDevice();
    remixDevice->EmitCs([meshCreates = std::move(meshCreates)](dxvk::DxvkContext* ctx) mutable {
      applyPendingMeshCreatesOnCs(ctx, meshCreates);
    });
  }

  // Batched mesh creation. Unlike remixapi_CreateMesh, this does not allocate
  // DXVK buffers or emit a CS command right away; it instead deep-copies the
  // caller-owned vertex/index/skinning data into s_pendingMeshCreates and
  // lets a later flush (DrawInstance / Present / AutoInstancePersistentLights)
  // materialize the mesh on the render thread. This lets API consumers submit
  // meshes outside a frame boundary without taking the device lock per mesh.
  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMeshBatched(
    const remixapi_MeshInfo* info,
    remixapi_MeshHandle* out_handle) {
    REMIX_FORK_RECORD(CreateMeshBatched, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateMeshBatched, 1 }; // rtx.profile.detailedLogging
    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_MESH_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_MeshHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_MeshHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (dxvk::RemixApiDevice* keyDevice = tryAsDxvk()) {
      emitAssetKey(keyDevice, info->pNext, info->hash, true);  // rtx.scene.assetKeys
    }

    if (dxvk::RtxOptions::HostImport::skipRepeatedMeshCreate()) {
      std::lock_guard lock { s_mutex };
      if (s_liveMeshHandles.count(handle) != 0) {
        // The CS would ignore this registration; skip the deep copy.
        *out_handle = handle;
        return REMIXAPI_ERROR_CODE_SUCCESS;
      }
    }

    PendingMeshCreate pending;
    pending.handle = handle;
    pending.hash = info->hash;
    pending.surfaces.reserve(info->surfaces_count);

    for (uint32_t i = 0; i < info->surfaces_count; i++) {
      const auto& src = info->surfaces_values[i];
      OwnedSurface dst;
      dst.material = src.material;

      dst.vertices.resize(src.vertices_count);
      if (src.vertices_count > 0) {
        memcpy(dst.vertices.data(), src.vertices_values, src.vertices_count * sizeof(remixapi_HardcodedVertex));
      }

      dst.indices.resize(src.indices_count);
      if (src.indices_count > 0) {
        memcpy(dst.indices.data(), src.indices_values, src.indices_count * sizeof(uint32_t));
      }

      dst.hasSkinning = src.skinning_hasvalue;
      if (src.skinning_hasvalue) {
        dst.bonesPerVertex = src.skinning_value.bonesPerVertex;
        dst.blendWeights.resize(src.skinning_value.blendWeights_count);
        if (src.skinning_value.blendWeights_count > 0) {
          memcpy(dst.blendWeights.data(), src.skinning_value.blendWeights_values, src.skinning_value.blendWeights_count * sizeof(float));
        }
        dst.blendIndices.resize(src.skinning_value.blendIndices_count);
        if (src.skinning_value.blendIndices_count > 0) {
          memcpy(dst.blendIndices.data(), src.skinning_value.blendIndices_values, src.skinning_value.blendIndices_count * sizeof(uint32_t));
        }
      } else {
        dst.bonesPerVertex = 0;
      }

      pending.surfaces.push_back(std::move(dst));
    }

    {
      std::lock_guard lock { s_mutex };
      s_pendingMeshCreates.push_back(std::move(pending));
      s_hasPendingMeshCreates.store(true);
      s_liveMeshHandles.insert(handle);
    }

    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyMesh(
    remixapi_MeshHandle handle) {
    REMIX_FORK_RECORD(DestroyMesh, handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DestroyMesh, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    s_liveMeshHandles.erase(handle);
    remixDevice->EmitCs([cHandle = handle](dxvk::DxvkContext* ctx) {
      static_cast<dxvk::RtxContext*>(ctx)->drainSceneCommands();  // rtx.scene.commandQueue
      ctx->getCommonObjects()->getSceneManager()
        .destroyExternalMesh(dxvk::Rc<dxvk::DxvkContext>(ctx), cHandle);
      ctx->getCommonObjects()->getSceneManager().getAssetReplacer()->assetResolver().eraseMeshKey(
        reinterpret_cast<uint64_t>(cHandle));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // A lens chained on the main camera (remixapi_CameraInfoLensEXT), with the id the lens manager knows it by.
  struct ApiLens {
    uint64_t id;
    dxvk::LensDesc desc;
  };

  // The host's id is hashed and tagged with bit 62, so it differs from the ids of mod USD lenses, which have bit 63
  // set, and from the debug camera lens' id.
  uint64_t toLensId(uint64_t hostId) {
    return (XXH3_64bits(&hostId, sizeof(hostId)) & ~(3ull << 62)) | (1ull << 62);
  }

  bool isFinite(const remixapi_CameraInfoLensEXT& ext) {
    const float values[] = {
      ext.eyepieceCenter.x, ext.eyepieceCenter.y, ext.eyepieceCenter.z,
      ext.axis.x, ext.axis.y, ext.axis.z,
      ext.up.x, ext.up.y, ext.up.z,
      ext.eyepieceRadius, ext.objectiveDistance, ext.objectiveRadius, ext.magnification, ext.fieldStopDegrees,
      ext.eyeRelief,
      ext.tint.x, ext.tint.y, ext.tint.z,
      ext.reticleColor.x, ext.reticleColor.y, ext.reticleColor.z,
      ext.reticleRadiance, ext.reticleLineWidth,
      ext.apertureAspect, ext.minMagnification, ext.referenceMagnification,
      ext.markingColor.x, ext.markingColor.y, ext.markingColor.z, ext.markingRadiance, ext.markingOpacity,
      ext.parallaxDistance, ext.glareStrength, ext.dirtStrength,
      ext.reticleAccentColor.x, ext.reticleAccentColor.y, ext.reticleAccentColor.z, ext.reticleAccentRadiance,
      ext.edgeGlowColor.x, ext.edgeGlowColor.y, ext.edgeGlowColor.z, ext.edgeGlowStrength, ext.edgeShade,
      ext.edgeTint.x, ext.edgeTint.y, ext.edgeTint.z,
      ext.nightTint.x, ext.nightTint.y, ext.nightTint.z, ext.nightGain,
    };
    return std::all_of(std::begin(values), std::end(values), [](float v) { return std::isfinite(v); });
  }

  // Zero fields take the LensDesc defaults, so a host can leave out what it does not know.
  dxvk::LensDesc toLensDesc(const remixapi_CameraInfoLensEXT& ext) {
    using namespace dxvk;

    LensDesc desc;
    desc.pose.eyepieceCenter = convert::tovec3(ext.eyepieceCenter);
    desc.pose.axis = convert::tovec3(ext.axis);
    desc.pose.up = convert::tovec3(ext.up);
    desc.eyepieceRadius = ext.eyepieceRadius;
    desc.objectiveDistance = ext.objectiveDistance;
    desc.objectiveRadius = ext.objectiveRadius;
    desc.eyeRelief = ext.eyeRelief;
    desc.reticleColor = convert::tovec3(ext.reticleColor);
    desc.reticleRadiance = ext.reticleRadiance;
    if (ext.magnification != 0.f) {
      desc.magnification = ext.magnification;
    }
    if (ext.fieldStopDegrees != 0.f) {
      desc.fieldStopDegrees = ext.fieldStopDegrees;
    }
    if (ext.tint.x != 0.f || ext.tint.y != 0.f || ext.tint.z != 0.f) {
      desc.tint = convert::tovec3(ext.tint);
    }
    if (ext.reticleLineWidth != 0.f) {
      desc.reticleLineWidth = ext.reticleLineWidth;
    }
    if (ext.apertureAspect > 0.f) {
      desc.apertureAspect = ext.apertureAspect;
    }
    switch (ext.reticle) {
    case REMIXAPI_CAMERA_LENS_RETICLE_CROSSHAIR: desc.reticle = LensReticle::Crosshair; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_DUPLEX: desc.reticle = LensReticle::Duplex; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_DOT: desc.reticle = LensReticle::Dot; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_CIRCLE_DOT: desc.reticle = LensReticle::CircleDot; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_MIL_DOT: desc.reticle = LensReticle::MilDot; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_GERMAN: desc.reticle = LensReticle::German; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_DROP: desc.reticle = LensReticle::Drop; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_STADIA: desc.reticle = LensReticle::Stadia; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_FRAME: desc.reticle = LensReticle::Frame; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_GRID: desc.reticle = LensReticle::Grid; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_DIAL: desc.reticle = LensReticle::Dial; break;
    case REMIXAPI_CAMERA_LENS_RETICLE_TEXTURE: desc.reticle = LensReticle::Texture; break;
    default: desc.reticle = LensReticle::None; break;
    }
    desc.kind = ext.kind == REMIXAPI_CAMERA_LENS_KIND_REFLEX ? LensKind::Reflex : LensKind::Telescopic;
    switch (ext.aperture) {
    case REMIXAPI_CAMERA_LENS_APERTURE_RECTANGLE: desc.aperture = LensAperture::Rectangle; break;
    case REMIXAPI_CAMERA_LENS_APERTURE_BINOCULAR: desc.aperture = LensAperture::Binocular; break;
    default: desc.aperture = LensAperture::Circle; break;
    }
    switch (ext.visionMode) {
    case REMIXAPI_CAMERA_LENS_VISION_NIGHT: desc.visionMode = LensVisionMode::Night; break;
    case REMIXAPI_CAMERA_LENS_VISION_THERMAL: desc.visionMode = LensVisionMode::Thermal; break;
    default: desc.visionMode = LensVisionMode::None; break;
    }
    desc.highlightHostiles = (ext.lensFlags & REMIXAPI_CAMERA_LENS_FLAG_HIGHLIGHT_HOSTILES) != 0;
    desc.occludeOutside = (ext.lensFlags & REMIXAPI_CAMERA_LENS_FLAG_OCCLUDE_OUTSIDE) != 0;
    desc.noGrain = (ext.lensFlags & REMIXAPI_CAMERA_LENS_FLAG_NO_GRAIN) != 0;
    desc.firstFocalPlane = (ext.lensFlags & REMIXAPI_CAMERA_LENS_FLAG_FIRST_FOCAL_PLANE) != 0;
    desc.markingColor = convert::tovec3(ext.markingColor);
    desc.markingRadiance = ext.markingRadiance;
    desc.parallaxDistance = std::max(ext.parallaxDistance, 0.f);
    desc.glareStrength = std::clamp(ext.glareStrength, 0.f, 1.f);
    desc.dirtStrength = std::clamp(ext.dirtStrength, 0.f, 1.f);
    if (ext.markingOpacity > 0.f) {
      desc.markingOpacity = ext.markingOpacity;
    }
    desc.referenceMagnification = ext.referenceMagnification > 0.f ? ext.referenceMagnification
      : ext.minMagnification > 0.f ? ext.minMagnification
      : 0.f;
    // The path is copied here, on the host's thread, so the host's string may go away after SetupCamera returns.
    if (ext.reticleTexture != nullptr && ext.reticleTexture[0] != L'\0') {
      try {
        desc.reticleTexture = std::filesystem::path(ext.reticleTexture).string();
      } catch (const std::exception&) {
        // A path the narrow encoding cannot hold draws no texture.
        desc.reticleTexture.clear();
      }
    }
    desc.reticleAccentColor = convert::tovec3(ext.reticleAccentColor);
    desc.reticleAccentRadiance = ext.reticleAccentRadiance;
    desc.edgeGlowColor = convert::tovec3(ext.edgeGlowColor);
    desc.edgeGlowStrength = ext.edgeGlowStrength;
    desc.edgeShade = ext.edgeShade;
    if (ext.edgeTint.x != 0.f || ext.edgeTint.y != 0.f || ext.edgeTint.z != 0.f) {
      desc.edgeTint = convert::tovec3(ext.edgeTint);
    }
    desc.nightTint = convert::tovec3(ext.nightTint);
    if (ext.nightGain > 0.f) {
      desc.nightGain = ext.nightGain;
    }
    desc.reticleLabels = (ext.markings & REMIXAPI_CAMERA_LENS_MARKING_LABELS) != 0;
    desc.rangefinder = (ext.markings & REMIXAPI_CAMERA_LENS_MARKING_RANGEFINDER) != 0;
    return desc;
  }

  // Returns the lenses chained on the camera.
  // Lenses with id 0 or a non-finite value are left out.
  std::vector<ApiLens> findCameraLenses(const remixapi_CameraInfo& info) {
    std::vector<ApiLens> lenses;
    struct Header { remixapi_StructType sType; void* pNext; };
    for (const Header* h = static_cast<const Header*>(info.pNext); h != nullptr; h = static_cast<const Header*>(h->pNext)) {
      if (h->sType == REMIXAPI_STRUCT_TYPE_NONE) {
        break;
      }
      if (h->sType != REMIXAPI_STRUCT_TYPE_CAMERA_INFO_LENS_EXT) {
        continue;
      }
      const auto& ext = *reinterpret_cast<const remixapi_CameraInfoLensEXT*>(h);
      if (ext.id == 0 || !isFinite(ext)) {
        static bool s_warned = false;
        if (!s_warned) {
          s_warned = true;
          dxvk::Logger::warn("[remixapi_SetupCamera] Skipped a remixapi_CameraInfoLensEXT with id 0 or a non-finite value.");
        }
        continue;
      }
      lenses.push_back(ApiLens { toLensId(ext.id), toLensDesc(ext) });
    }
    return lenses;
  }

  // The poses of the lenses hosts chain on the main camera, kept for a frame for motion vectors.
  // Only the CS thread uses it.
  class ApiLensHistory {
  public:
    // Adds the frame's lenses to the lens manager, each with its pose on the previous frame when it had one.
    // originShift is the translation of the frame's origin shift event, if there is one.
    void addLenses(dxvk::LensManager& lensManager, const std::vector<ApiLens>& lenses, uint32_t frameId,
                   const dxvk::Vector3* originShift) {
      if (frameId != m_frameId) {
        m_frameId = frameId;
        for (auto it = m_poses.begin(); it != m_poses.end();) {
          it = it->second.frameId + 1 == frameId ? std::next(it) : m_poses.erase(it);
        }
      }

      for (const ApiLens& lens : lenses) {
        auto it = m_poses.find(lens.id);
        // A lens sent twice in a frame is added once.
        if (it != m_poses.end() && it->second.frameId == frameId) {
          continue;
        }

        dxvk::LensPose previousPose;
        const dxvk::LensPose* pPreviousPose = nullptr;
        if (it != m_poses.end()) {
          previousPose = it->second.pose;
          if (originShift != nullptr) {
            previousPose.eyepieceCenter += *originShift;
          }
          pPreviousPose = &previousPose;
        }

        lensManager.addLens(lens.id, lens.desc, pPreviousPose);
        m_poses[lens.id] = PoseEntry { lens.desc.pose, frameId };
      }
    }

  private:
    struct PoseEntry {
      dxvk::LensPose pose;
      uint32_t frameId;
    };

    std::unordered_map<uint64_t, PoseEntry> m_poses;
    uint32_t m_frameId = UINT32_MAX;
  };

  ApiLensHistory s_apiLensHistory;

  remixapi_ErrorCode REMIXAPI_CALL remixapi_SetupCamera(
    const remixapi_CameraInfo* info) {
    REMIX_FORK_RECORD(SetupCamera, info);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::SetupCamera, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_CAMERA_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    // ensure that near plane is not modified, to keep the projection matrix
    // exactly as the client provided, so depth buffers would have expected results,
    // for a client to be able to reproject to world space using the projection matrices
    if (dxvk::RtxOptions::enableNearPlaneOverride()) {
      assert(0);
      const_cast<bool&>(dxvk::RtxOptions::enableNearPlaneOverride()) = false;
    }
    dxvk::frontend::CameraSetup camera = convert::toCameraSetup(*info);
    // REMIXAPI_CAMERA_FRAME_EVENT_FALLBACK: the camera is only held. SceneManager::applyFallbackMainCamera makes it the
    // Main camera at the frame's ray tracing if no draw set one. It takes no origin and no other events (remix_c.h).
    if (auto events = pnext::find<remixapi_CameraInfoFrameEventsEXT>(info);
        events && (events->flags & REMIXAPI_CAMERA_FRAME_EVENT_FALLBACK)) {
      if (info->type != REMIXAPI_CAMERA_TYPE_WORLD) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      // Lenses sent with a fallback camera are world space, so they are added whichever camera becomes the main one.
      std::vector<ApiLens> lenses = findCameraLenses(*info);
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([cWorldToView = camera.worldToView, cViewToProjection = camera.viewToProjection,
                           cLenses = std::move(lenses)](dxvk::DxvkContext* ctx) {
        ctx->getCommonObjects()->getSceneManager().getCameraManager().holdFallbackCamera(cWorldToView, cViewToProjection);
        if (!cLenses.empty() && dxvk::LensManager::enable()) {
          s_apiLensHistory.addLenses(ctx->getCommonObjects()->getSceneManager().getLensManager(), cLenses,
                                     ctx->getDevice()->getCurrentFrameId(), nullptr);
        }
      });
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    // rtx.worldOrigin.enable (rtx_fork_camera_origin.h): the host's render origin is read here, on the host's thread,
    // so it travels with this camera through the command stream in order with the host's draws.
    dxvk::Vector3 worldOrigin { 0.f, 0.f, 0.f };
    if (dxvk::WorldOrigin::enable() && dxvk::fork_camera_origin::tryReadWorldOriginOffsetFromGameState(worldOrigin)) {
      camera.worldOrigin = worldOrigin;
    }
    // Frame events the host states (remixapi_CameraInfoFrameEventsEXT). Their bits are the interface's bits.
    static_assert(REMIXAPI_CAMERA_FRAME_EVENT_CAMERA_CUT == dxvk::frontend::kFrameEventCameraCut);
    static_assert(REMIXAPI_CAMERA_FRAME_EVENT_HISTORY_INVALID == dxvk::frontend::kFrameEventHistoryInvalid);
    static_assert(REMIXAPI_CAMERA_FRAME_EVENT_ORIGIN_SHIFT == dxvk::frontend::kFrameEventOriginShift);
    if (auto events = pnext::find<remixapi_CameraInfoFrameEventsEXT>(info)) {
      camera.events = events->flags;
      if (camera.events & REMIXAPI_CAMERA_FRAME_EVENT_ORIGIN_SHIFT) {
        camera.originShift = convert::tovec3(events->originShift);
      }
    }
    // Lenses (remixapi_CameraInfoLensEXT) belong to the main camera. They reach the lens manager on the CS thread,
    // after the camera, so they land in the frame the camera is for.
    std::vector<ApiLens> lenses;
    if (info->type == REMIXAPI_CAMERA_TYPE_WORLD) {
      lenses = findCameraLenses(*info);
    }
    auto devLock = remixDevice->LockDevice();
    remixDevice->EmitCs([cCamera = std::move(camera), cLenses = std::move(lenses)](dxvk::DxvkContext* ctx) {
      dxvk::frontend::SceneInterface { *static_cast<dxvk::RtxContext*>(ctx) }.setupCamera(cCamera);
      if (!cLenses.empty() && dxvk::LensManager::enable()) {
        const bool originShifted = (cCamera.events & dxvk::frontend::kFrameEventOriginShift) != 0;
        s_apiLensHistory.addLenses(ctx->getCommonObjects()->getSceneManager().getLensManager(), cLenses,
                                   ctx->getDevice()->getCurrentFrameId(), originShifted ? &cCamera.originShift : nullptr);
      }
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_SetCameraMediumMaterial(
    const remixapi_CameraMediumInfo* info) {
    REMIX_FORK_RECORD(SetCameraMediumMaterial, info);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_CAMERA_MEDIUM_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    const remixapi_MaterialHandle handle = info->medium;

    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs([handle](dxvk::DxvkContext* ctx) {
      auto& sceneManager = ctx->getCommonObjects()->getSceneManager();
      if (handle == nullptr) {
        sceneManager.clearExternalStartInMediumMaterial();
        return;
      }

      const dxvk::MaterialData* material = sceneManager.getAssetReplacer()->accessExternalMaterial(handle);
      if (material == nullptr) {
        dxvk::Logger::warn("SetCameraMediumMaterial: material handle not found");
        return;
      }
      if (material->getType() != dxvk::MaterialDataType::Translucent) {
        dxvk::Logger::warn("SetCameraMediumMaterial: material must be translucent");
        return;
      }

      sceneManager.setExternalStartInMediumMaterial(*material);
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawInstance(
    const remixapi_InstanceInfo* info) {
    REMIX_FORK_RECORD(DrawInstance, info);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DrawInstance, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    ScopedCpuProfileZone();

    // Fires beginScene on the first draw of a frame.
    dxvk::fork_hooks::notifyBeginScene();

    // Ensure any meshes queued via remixapi_CreateMeshBatched are registered
    // with the asset replacer before this draw references them. Takes s_mutex
    // itself, so it must run before the lock_guard below.
    flushPendingMeshes(remixDevice);

    // Hoist conversion outside of mutex
    dxvk::frontend::PreparedInstance instance = convert::toPreparedInstance(*info);

    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cInstance = std::move(instance)](dxvk::DxvkContext* dxvkCtx) mutable {
      dxvk::frontend::SceneInterface { *static_cast<dxvk::RtxContext*>(dxvkCtx) }.drawInstance(std::move(cInstance));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateLight(
    const remixapi_LightInfo* info,
    remixapi_LightHandle* out_handle) {
    REMIX_FORK_RECORD(CreateLight, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateLight, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_LIGHT_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_LightHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_LightHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // async load
    std::lock_guard lock { s_mutex };
    if (auto src = pnext::find<remixapi_LightInfoDomeEXT>(info)) {
      // Special case for dome lights
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([cHandle = handle,
                          cRadiance = convert::tovec3(info->radiance),
                          cTransform = convert::tomat4(src->transform),
                          cTexturePath = convert::topath(src->colorTexture)]
                          (dxvk::DxvkContext* ctx) {
        auto preloadTexture = [&ctx](const std::filesystem::path& path)->dxvk::TextureRef {
          if (path.empty()) {
            return {};
          }
          auto assetData = dxvk::AssetDataManager::get().findAsset(path.string().c_str());
          if (assetData == nullptr) {
            return {};
          }
          auto uploadedTexture = ctx->getCommonObjects()->getTextureManager()
            .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true);
          return dxvk::TextureRef { uploadedTexture };
        };

        dxvk::DomeLight domeLight;
        domeLight.radiance = cRadiance;
        domeLight.worldToLight = inverse(cTransform);
        domeLight.texture = preloadTexture(cTexturePath);

        // Ensures a texture stays in VidMem
        uint32_t unused;
        ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);

        auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
        lightMgr.addExternalDomeLight(cHandle, domeLight);
      });
    } else {
      // Regular analytical light handling
      auto rtLight = convert::toRtLight(*info);

      // Note: If the toRtLight conversion process returns an empty optional, the specified LightInfo did
      // not contain the proper arguments to create a light with.
      if (!rtLight.has_value()) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }

      // Set the isDynamic flag from the LightInfo
      rtLight->isDynamic = info->isDynamic;

      // Set the ignoreViewModel flag from the LightInfo
      rtLight->ignoreViewModel = info->ignoreViewModel;

      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([cHandle = handle, cRtLight = *rtLight](dxvk::DxvkContext* ctx) {
        auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
        lightMgr.addExternalLight(cHandle, cRtLight);
      });
    }

    *out_handle = handle;
    // Auto-register for persistent instancing on device
    {
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([cHandle = handle](dxvk::DxvkContext* ctx) {
        auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
        lightMgr.registerPersistentExternalLight(cHandle);
      });
    }
    s_externalLightApiUsed.store(true, std::memory_order_relaxed);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateLightBatched(
    const remixapi_LightInfo* info,
    remixapi_LightHandle* out_handle) {
    REMIX_FORK_RECORD(CreateLightBatched, info, out_handle);
    // Validate inputs
    if (!out_handle || !info || info->sType != REMIXAPI_STRUCT_TYPE_LIGHT_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    static_assert(sizeof(remixapi_LightHandle) == sizeof(info->hash));
    auto handle = reinterpret_cast<remixapi_LightHandle>(info->hash);
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // Queue the light creation for batched processing at Present time
    std::lock_guard lock { s_mutex };

    PendingLightCreate pending;
    pending.handle = handle;

    if (auto src = pnext::find<remixapi_LightInfoDomeEXT>(info)) {
      // Dome light
      pending.isDome = true;
      pending.radiance = convert::tovec3(info->radiance);
      pending.transform = convert::tomat4(src->transform);
      pending.texturePath = convert::topath(src->colorTexture);
    } else {
      // Analytical light
      pending.isDome = false;
      auto rtLight = convert::toRtLight(*info);

      if (!rtLight.has_value()) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }

      // Set the isDynamic flag from the LightInfo
      rtLight->isDynamic = info->isDynamic;

      // Set the ignoreViewModel flag from the LightInfo
      rtLight->ignoreViewModel = info->ignoreViewModel;

      pending.rtLight = std::move(rtLight);
    }

    s_pendingLightCreates.push_back(std::move(pending));
    *out_handle = handle;

    s_externalLightApiUsed.store(true, std::memory_order_relaxed);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyLight(
    remixapi_LightHandle handle) {
    REMIX_FORK_RECORD(DestroyLight, handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DestroyLight, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    {
      std::lock_guard lock { s_mutex };
      s_pendingLightDestroys.push_back(handle);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }


  remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawLightInstance(
    remixapi_LightHandle lightHandle) {
    REMIX_FORK_RECORD(DrawLightInstance, lightHandle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DrawLightInstance, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!lightHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // beginScene on first API submission per frame (lights-only frames)
    dxvk::fork_hooks::notifyBeginScene();

    // async load
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs([lightHandle](dxvk::DxvkContext* ctx) {
      auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
      lightMgr.addExternalLightInstance(lightHandle);
    });

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_SetConfigVariable(
    const char* key,
    const char* value) {
    REMIX_FORK_RECORD(SetConfigVariable, key, value);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::SetConfigVariable, 1 }; // rtx.profile.detailedLogging
    std::lock_guard lock { s_mutex };

    if (!key || key[0] == '\0' || !value) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::string strKey = std::string{ key };

    dxvk::RtxOptionImpl* option = dxvk::RtxOptionImpl::getOptionByFullName(strKey);
    if (!option) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    dxvk::Config newSetting;
    newSetting.setOptionMove(std::move(strKey), std::string{ value });

    // readOption writes the layer value and the dirty-option map, both of
    // which the render thread walks under the update mutex in
    // RtxOptionManager::applyPendingValues. Unguarded, a game pushing an option
    // every frame (FNV's sun/moon/camera drive) races that walk and eventually
    // corrupts the map mid-iteration.
    std::lock_guard optionLock { dxvk::RtxOptionImpl::getUpdateMutex() };
    option->readOption(newSetting, dxvk::RtxOptionLayer::getUserLayer());

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_RequestVramCompaction() {
    REMIX_FORK_RECORD0(RequestVramCompaction);
    return dxvk::fork_hooks::requestVramCompaction(tryAsDxvk());
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_RequestTextureVramFree() {
    REMIX_FORK_RECORD0(RequestTextureVramFree);
    return dxvk::fork_hooks::requestTextureVramFree(tryAsDxvk());
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_GetVramStats(
      remixapi_VramStats* out_stats) {
    REMIX_FORK_RECORD(GetVramStats, out_stats);
    return dxvk::fork_hooks::getVramStats(tryAsDxvk(), out_stats);
  }

  // The helper body lives in rtx_fork_api_entry.cpp as
  // fork_hooks::mutateTextureHashOption (add flag replaces the local
  // TextureHashMutation enum). Both call sites hold s_mutex across the call
  // per the lock-ordering rule documented alongside s_mutex.

  remixapi_ErrorCode REMIXAPI_CALL remixapi_AddTextureHash(
    const char* textureCategory,
    const char* textureHash) {
    REMIX_FORK_RECORD(AddTextureHash, textureCategory, textureHash);
    std::lock_guard lock { s_mutex };
    return dxvk::fork_hooks::mutateTextureHashOption(textureCategory, textureHash, /*add*/ true);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_RemoveTextureHash(
    const char* textureCategory,
    const char* textureHash) {
    REMIX_FORK_RECORD(RemoveTextureHash, textureCategory, textureHash);
    std::lock_guard lock { s_mutex };
    return dxvk::fork_hooks::mutateTextureHashOption(textureCategory, textureHash, /*add*/ false);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_GetTextureHash(
    IDirect3DTexture9* texture,
    uint64_t* out_hash) {
    REMIX_FORK_RECORD(dxvk_GetTextureHash, texture, out_hash);
    return dxvk::fork_hooks::dxvkGetTextureHash(texture, out_hash);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateTexture(
    const remixapi_TextureInfo* info,
    remixapi_TextureHandle* out_handle) {
    REMIX_FORK_RECORD(CreateTexture, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateTexture, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    std::lock_guard lock { s_mutex };
    return dxvk::fork_hooks::createTexture(remixDevice, info, out_handle);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyTexture(
    remixapi_TextureHandle handle) {
    REMIX_FORK_RECORD(DestroyTexture, handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DestroyTexture, 1 }; // rtx.profile.detailedLogging
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    std::lock_guard lock { s_mutex };
    return dxvk::fork_hooks::destroyTexture(remixDevice, handle);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_pick_RequestObjectPicking(
    const remixapi_Rect2D* pixelRegion,
    PFN_remixapi_pick_RequestObjectPickingUserCallback callback,
    void* callbackUserData) {
    REMIX_FORK_RECORD(pick_RequestObjectPicking, pixelRegion, callback, callbackUserData);
    if (!pixelRegion || !callback) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };

    dxvk::ObjectPicking& picking = remixDevice->GetDXVKDevice()->getCommon()
      ->metaDebugView().ObjectPicking;

    picking.request(
      dxvk::Vector2i{ pixelRegion->left, pixelRegion->top },
      dxvk::Vector2i{ pixelRegion->right, pixelRegion->bottom },
      // invoke user's callback on result
      [callback, callbackUserData](std::vector<dxvk::ObjectPickingValue>&& objectPickingValues, std::optional<XXH64_hash_t>) {
        callback(objectPickingValues.data(), uint32_t(objectPickingValues.size()), callbackUserData);
      }
    );
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_pick_HighlightObjects(
    const uint32_t* objectPickingValues_values,
    uint32_t objectPickingValues_count,
    uint8_t colorR,
    uint8_t colorG,
    uint8_t colorB) {
    REMIX_FORK_RECORD(pick_HighlightObjects, objectPickingValues_values, objectPickingValues_count, colorR, colorG, colorB);
    if (!objectPickingValues_values) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };

    if (objectPickingValues_count > 0) {
      dxvk::g_customHighlightColor = { colorR, colorG, colorB };

      const auto frameId = remixDevice->GetDXVKDevice()->getCurrentFrameId();
      const auto values = std::vector< dxvk::ObjectPickingValue > {
        objectPickingValues_values,
        objectPickingValues_values + objectPickingValues_count,
      };
      remixDevice->GetDXVKDevice()->getCommon()->metaDebugView().Highlighting
        .requestHighlighting(
          &values,
          dxvk::HighlightColor::FromVariable,
          frameId);   // thread-safe
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  bool g_combineGuiInFinalColor = true;

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_CreateD3D9(
    const remixapi_StartupInfo& info,
    IDirect3D9Ex** out_pD3D9) {
    IDirect3D9Ex* d3d9ex = nullptr;

    auto hr = dxvk::CreateD3D9(true, &d3d9ex, info.forceNoVkSwapchain, false, true);
    if (FAILED(hr) || !d3d9ex) {
      if (isHResultAliasedWithRemixErrorCode(hr)) {
        return static_cast<remixapi_ErrorCode>(hr);
      }
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    g_combineGuiInFinalColor = info.combineGuiInFinalColor;
    dxvk::g_allowSrgbConversionForOutput = !info.disableSrgbConversionForOutput;
    dxvk::g_allowMappingLegacyHashToObjectPickingValue = !info.editorModeEnabled;
    dxvk::g_remixApiEditorModeEnabled = info.editorModeEnabled;

    // slightly different initial settings for HdRemix
    if (info.editorModeEnabled) {
      const_cast<dxvk::LightManager::FallbackLightMode&>(dxvk::LightManager::fallbackLightMode()) = dxvk::LightManager::FallbackLightMode::Never;
      const_cast<bool&>(dxvk::DxvkPostFx::desaturateOthersOnHighlight()) = false;
      const_cast<bool&>(dxvk::RtxOptions::showUICursor()) = false;
    }

    *out_pD3D9 = d3d9ex;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // HdRemix has editorModeEnabled=true
  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_CreateD3D9_legacy(
    remixapi_Bool editorModeEnabled,
    IDirect3D9Ex** out_pD3D9) {
    if (dxvk::ApiRecorder::shouldRecord()) { return dxvk::ApiRecorder::dxvk_CreateD3D9(&remixapi_dxvk_CreateD3D9_legacy, editorModeEnabled, out_pD3D9); }
    auto i = remixapi_StartupInfo {};
    {
      i.sType = REMIXAPI_STRUCT_TYPE_STARTUP_INFO;
      i.disableSrgbConversionForOutput = editorModeEnabled;
      i.forceNoVkSwapchain = editorModeEnabled;
      i.editorModeEnabled = editorModeEnabled;
      i.combineGuiInFinalColor = !editorModeEnabled;
      static_assert(sizeof(remixapi_StartupInfo) == 40, "If changing, also set defaults here");
    }
    return remixapi_dxvk_CreateD3D9(i, out_pD3D9);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_RegisterD3D9Device(
    IDirect3DDevice9Ex* d3d9Device) {
    if (!d3d9Device) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto dxvkDevice = dynamic_cast<dxvk::D3D9DeviceEx*>(d3d9Device);
    if (!dxvkDevice) {
      return REMIXAPI_ERROR_CODE_REGISTERING_NON_REMIX_D3D9_DEVICE;
    }
    IDirect3D9* dxvkD3d9 = nullptr;
    HRESULT hr = dxvkDevice->GetDirect3D(&dxvkD3d9);
    if (FAILED(hr) || !dxvkD3d9) {
      assert(0);
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    auto dxvkD3d9Ex = dynamic_cast<IDirect3D9Ex*>(dxvkD3d9);
    if (!dxvkD3d9Ex) {
      assert(0);
      return REMIXAPI_ERROR_CODE_NOT_INITIALIZED;
    }
    // if D3D9 already exists, check that user-provided D3D9 corresponds to our s_dxvkD3D9
    if (s_dxvkD3D9) {
      assert(s_dxvkD3D9 == dxvkD3d9Ex);
    }
    s_dxvkD3D9 = dxvkD3d9Ex;
    {
      // A new device starts with no registered meshes. Forgetting live handles is always
      // safe: the worst case is a repeated create the CS then drops.
      std::lock_guard lock { s_mutex };
      s_liveMeshHandles.clear();
    }
    s_dxvkDevice = std::make_unique<dxvk::RemixApiDevice>(dxvkDevice);
    dxvk::g_dxvkDeviceNative = dxvkDevice->GetDXVKDevice().ptr();
    dxvk::ApiRecorder::onDeviceRegistered(false, d3d9Device);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_GetExternalSwapchain(
    uint64_t* out_vkImage,
    uint64_t* out_vkSemaphoreRenderingDone,
    uint64_t* out_vkSemaphoreResumeSemaphore) {
    REMIX_FORK_RECORD(dxvk_GetExternalSwapchain, out_vkImage, out_vkSemaphoreRenderingDone, out_vkSemaphoreResumeSemaphore);
    auto* host = tryAsDxvk();
    auto* remixDevice = host ? host->d3d9() : nullptr;
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!out_vkImage || !out_vkSemaphoreRenderingDone || !out_vkSemaphoreResumeSemaphore) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (auto pres = remixDevice->GetExternalPresenter()) {
      *out_vkImage = reinterpret_cast<uint64_t>(pres->GetVkImage(0));
      *out_vkSemaphoreRenderingDone = reinterpret_cast<uint64_t>(pres->GetFrameCompleteVkSemaphore());
      *out_vkSemaphoreResumeSemaphore = reinterpret_cast<uint64_t>(pres->GetFrameResumeVkSemaphore());
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_GetSharedD3D11TextureHandle(
    void** out_sharedHandle,
    uint32_t* out_width,
    uint32_t* out_height) {
    REMIX_FORK_RECORD(dxvk_GetSharedD3D11TextureHandle, out_sharedHandle, out_width, out_height);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    return dxvk::fork_hooks::getSharedD3D11TextureHandle(remixDevice, out_sharedHandle, out_width, out_height);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_GetVkImage(
    IDirect3DSurface9* source,
    uint64_t* out_vkImage) {
    REMIX_FORK_RECORD(dxvk_GetVkImage, source, out_vkImage);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!source || !out_vkImage) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    dxvk::D3D9Surface* surface = static_cast<dxvk::D3D9Surface*>(source);
    dxvk::D3D9CommonTexture* texInfo = surface ? surface->GetCommonTexture() : nullptr;
    if (texInfo) {
      *out_vkImage = reinterpret_cast<uint64_t>(texInfo->GetImage()->handle());
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_CopyRenderingOutput(
    IDirect3DSurface9* destination,
    remixapi_dxvk_CopyRenderingOutputType type) {
    REMIX_FORK_RECORD(dxvk_CopyRenderingOutput, destination, type);
    auto* host = tryAsDxvk();
    auto* remixDevice = host ? host->d3d9() : nullptr;
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!destination) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    dxvk::D3D9Surface* destSurface = static_cast<dxvk::D3D9Surface*>(destination);
    dxvk::D3D9CommonTexture* destTexInfo = destSurface ? destSurface->GetCommonTexture() : nullptr;
    dxvk::Rc<dxvk::DxvkImage> destImage = destTexInfo->GetImage();
    dxvk::Rc<dxvk::DxvkImageView> destImageView = destTexInfo->GetSampleView(false);

    if (!destTexInfo) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    if (type == REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_OBJECT_PICKING) {
      // suppress resource clean up
      if (!dxvk::g_forceKeepObjectPickingImage) {
        dxvk::g_forceKeepObjectPickingImage = true;
        return REMIXAPI_ERROR_CODE_SUCCESS;
      }
    }

#pragma warning(push)
#pragma warning(error : 4061) // all switch cases must be handled explicitly

    switch (type) {
    case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR:
    case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_DEPTH:
    case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_NORMALS:
    case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_OBJECT_PICKING:
      break;
    case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_GUI:
      if (!dxvk::ImGUI::enableExternalPresenter()) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
      }
      break;
    default:
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::lock_guard lock { s_mutex };

    // Pass in the backbuffer incase we need it
    IDirect3DSurface9* pSurface = nullptr;
    remixDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &pSurface);
    dxvk::D3D9Surface* backBufferSurface = static_cast<dxvk::D3D9Surface*>(pSurface);
    dxvk::Rc<dxvk::DxvkImage> backbuffer0 = backBufferSurface->GetCommonTexture()->GetImage();

    remixDevice->EmitCs([cDest = std::move(destImage), 
                         cDestView = std::move(destImageView),
                         cBackbuffer = std::move(backbuffer0),
                         type = type] (dxvk::DxvkContext* dxvkCtx) {
      auto* ctx = static_cast<dxvk::RtxContext*>(dxvkCtx);

      dxvk::Resources& resourceManager = ctx->getCommonObjects()->getResources();
      const dxvk::Resources::RaytracingOutput& rtOutput = resourceManager.getRaytracingOutput();

      dxvk::Rc<dxvk::DxvkImage> srcImage = nullptr;
      switch (type) {
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR:
        srcImage = cBackbuffer;
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_DEPTH:
        srcImage = rtOutput.m_primaryDepth.image;
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_NORMALS:
        srcImage = rtOutput.m_primaryWorldShadingNormal.image;
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_OBJECT_PICKING:
        srcImage = rtOutput.m_primaryObjectPicking.image;
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_GUI: {
        dxvk::DxvkRenderTargets renderTargets;
        renderTargets.color[0].view = cDestView;
        renderTargets.color[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ctx->bindRenderTargets(renderTargets);
        ctx->getCommonObjects()->getImgui().render(ctx, { cDestView->imageInfo().extent.width, cDestView->imageInfo().extent.height });

        break;
      }
      default:
        assert(!"unexpected remixapi_dxvk_CopyRenderingOutputType value");
        return;
      }

      if (srcImage.ptr()) {
        dxvk::RtxContext::blitImageHelper(ctx, srcImage, cDest, VkFilter::VK_FILTER_NEAREST);
      }
    });

#pragma warning(pop)

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }


  remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_SetDefaultOutput(
    remixapi_dxvk_CopyRenderingOutputType type, const remixapi_Float4D* color) {
    REMIX_FORK_RECORD(dxvk_SetDefaultOutput, type, color);
    if (!color) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }

    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs([type, cColor = *color](dxvk::DxvkContext* ctx) {
      dxvk::RtxGlobals& globals = ctx->getCommonObjects()->getSceneManager().getGlobals();
      switch (type) {
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_FINAL_COLOR:
        globals.clearColorFinalColor = vec3(cColor.x, cColor.y, cColor.z);
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_DEPTH:
        globals.clearColorDepth = cColor.x;
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_NORMALS:
        globals.clearColorNormal = vec3(cColor.x, cColor.y, cColor.z);
        break;
      case REMIXAPI_DXVK_COPY_RENDERING_OUTPUT_TYPE_OBJECT_PICKING:
        // converting binary value of color.x into uint to avoid losing precision.
        globals.clearColorPicking = reinterpret_cast<const uint&>(cColor.x);
        break;
      default:
        break;
      }
    });
    
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_Startup(const remixapi_StartupInfo* info) {
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_STARTUP_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    assert(!!(s_dxvkD3D9) == !!(s_dxvkDevice));
    if (s_dxvkD3D9 || s_dxvkDevice) {
      return REMIXAPI_ERROR_CODE_ALREADY_EXISTS;
    }

    IDirect3D9Ex* d3d9 = nullptr;
    {
      remixapi_ErrorCode status = remixapi_dxvk_CreateD3D9(*info, &d3d9);
      if (status != REMIXAPI_ERROR_CODE_SUCCESS) {
        return status;
      }
    }

    HWND hwnd = nullptr;
    uint32_t width = 0, height = 0;

    if (info->hwnd) {
      hwnd = info->hwnd;
      auto hwndRect = RECT {};
      GetClientRect(info->hwnd, &hwndRect);
      width = static_cast<uint32_t>(std::max(0l, hwndRect.right - hwndRect.left));
      height = static_cast<uint32_t>(std::max(0l, hwndRect.bottom - hwndRect.top));
    }

    IDirect3DDevice9Ex* d3d9Device = nullptr;
    {
      auto presInfo = D3DPRESENT_PARAMETERS {};
      {
        presInfo.BackBufferWidth = width;
        presInfo.BackBufferHeight = height;
        presInfo.BackBufferFormat = D3DFMT_UNKNOWN;
        presInfo.BackBufferCount = 0;
        presInfo.MultiSampleType = D3DMULTISAMPLE_NONE;
        presInfo.MultiSampleQuality = 0;
        presInfo.SwapEffect = D3DSWAPEFFECT_DISCARD;
        presInfo.hDeviceWindow = hwnd;
        presInfo.Windowed = true;
        presInfo.EnableAutoDepthStencil = false;
        presInfo.AutoDepthStencilFormat = D3DFMT_UNKNOWN;
        presInfo.Flags = 0;
        presInfo.FullScreen_RefreshRateInHz = 0;
        presInfo.PresentationInterval = 0;
      }

      HRESULT hr = d3d9->CreateDeviceEx(
        D3DADAPTER_DEFAULT,
        D3DDEVTYPE_HAL,
        hwnd,
        D3DCREATE_HARDWARE_VERTEXPROCESSING,
        &presInfo,
        nullptr,
        &d3d9Device);
      if (FAILED(hr) || !d3d9Device) {
        d3d9->Release();
        if (isHResultAliasedWithRemixErrorCode(hr)) {
          // return special aliased HRESULT
          return static_cast<remixapi_ErrorCode>(hr);
        }
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
      }
    }
    {
      remixapi_ErrorCode status = remixapi_dxvk_RegisterD3D9Device(d3d9Device);
      if (status != REMIXAPI_ERROR_CODE_SUCCESS) {
        d3d9->Release();
        return status;
      }
      assert(s_dxvkD3D9 && s_dxvkDevice);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_Shutdown(void) {
    dxvk::ApiRecorder::onShutdown();
    // Clears the frame-boundary callback state.
    dxvk::fork_hooks::shutdownCallbacks();
    if (s_dxvkDevice) {
      dxvk::g_dxvkDeviceNative = nullptr;
      auto host = std::move(s_dxvkDevice);
      if (auto* context = host->d3d11()) {
        context->Flush();
        context->SynchronizeCsThread(dxvk::DxvkCsThread::SynchronizeAll);
        context->DisableRemixHost();
        context->Release();
      } else {
        auto* device = host->d3d9();
        while (device->Release() != 0) { }
      }
    }
    if (s_dxvkD3D9) {
      while (true) {
        ULONG left = s_dxvkD3D9->Release();
        if (left == 0) {
          break;
        }
      }
      s_dxvkD3D9 = nullptr;
    }

    // Make sure Sentry doesn't keep the process alive when it should be shutting down.
    dxvk::sentry::shutdown();

    dxvk::ApiRecorder::onShutdownDone();
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_Present(const remixapi_PresentInfo* info) {
    REMIX_FORK_RECORD(Present, info);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::Present, 1 }; // rtx.profile.detailedLogging
    auto* host = tryAsDxvk();
    auto* remixDevice = host ? host->d3d9() : nullptr;
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    // Apply pending creates, updates and auto-instance persistent lights once per frame
    std::vector<PendingLightCreate> creates;
    std::vector<PendingLightUpdate> updates;
    std::vector<PendingDomeUpdate> domeUpdates;
    std::vector<remixapi_LightHandle> destroys;
    std::vector<PendingMeshCreate> meshCreates;
    {
      std::lock_guard lock { s_mutex };
      creates.swap(s_pendingLightCreates);
      updates.swap(s_pendingLightUpdates);
      domeUpdates.swap(s_pendingDomeUpdates);
      destroys.swap(s_pendingLightDestroys);
      meshCreates.swap(s_pendingMeshCreates);
      s_hasPendingMeshCreates.store(false);
    }
    // Build tombstone set for this frame to avoid re-adding deleted lights
    std::unordered_set<remixapi_LightHandle> tombstones;
    tombstones.insert(destroys.begin(), destroys.end());

    // The device lock is scoped so it is released before remixDevice->Present below.
    // Holding it across Present inverts the process-wide order: the swapchain calls
    // remixapi_AutoInstancePersistentLights, which takes s_mutex, while any other API
    // thread holds s_mutex and spins on this same device lock. Every other entry point
    // takes s_mutex first and the device lock second; this must match.
    {
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([creates = std::move(creates), updates = std::move(updates), domeUpdates = std::move(domeUpdates), destroys = std::move(destroys), tombstones = std::move(tombstones), meshCreates = std::move(meshCreates)](dxvk::DxvkContext* ctx) mutable {
        auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
        // Apply destroys first
        for (auto h : destroys) {
          if (h) {
            lightMgr.unregisterPersistentExternalLight(h);
            lightMgr.removeExternalLight(h);
          }
        }
        // Apply creates (skip if in tombstone set - deleted in same frame)
        for (auto& create : creates) {
          if (tombstones.find(create.handle) != tombstones.end()) {
            continue; // Skip if deleted in same frame
          }

          if (create.isDome) {
            // Build dome light on the render thread
            auto preloadTexture = [ctx](const std::filesystem::path& path) {
              if (path.empty()) {
                return dxvk::TextureRef{};
              }
              auto assetData = dxvk::AssetDataManager::get().findAsset(path.string().c_str());
              if (assetData == nullptr) {
                return dxvk::TextureRef{};
              }
              auto uploadedTexture = ctx->getCommonObjects()->getTextureManager()
                .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true);
              return dxvk::TextureRef{ uploadedTexture };
            };

            dxvk::DomeLight domeLight;
            domeLight.radiance = create.radiance;
            domeLight.worldToLight = inverse(create.transform);
            domeLight.texture = preloadTexture(create.texturePath);

            uint32_t unused;
            ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);

            lightMgr.addExternalDomeLight(create.handle, domeLight);
          } else if (create.rtLight.has_value()) {
            // Analytical light
            lightMgr.addExternalLight(create.handle, *create.rtLight);
          }
          // Register all created lights as persistent
          lightMgr.registerPersistentExternalLight(create.handle);
          lightMgr.addExternalLightInstance(create.handle);
        }
        // Apply updates (skip if in tombstone set)
        for (auto& upd : updates) {
          if (upd.rtLight.has_value() && tombstones.find(upd.handle) == tombstones.end()) {
            lightMgr.registerPersistentExternalLight(upd.handle);
            lightMgr.addExternalLight(upd.handle, *upd.rtLight);
            lightMgr.addExternalLightInstance(upd.handle);
          }
        }
        for (auto& du : domeUpdates) {
          if (tombstones.find(du.handle) != tombstones.end()) {
            continue;
          }
          // Build dome light on the render thread to safely access managers
          auto preloadTexture = [ctx](const std::filesystem::path& path) {
            if (path.empty()) {
              return dxvk::TextureRef{};
            }
            auto assetData = dxvk::AssetDataManager::get().findAsset(path.string().c_str());
            if (assetData == nullptr) {
              return dxvk::TextureRef{};
            }
            auto uploadedTexture = ctx->getCommonObjects()->getTextureManager()
              .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true);
            return dxvk::TextureRef{ uploadedTexture };
          };

          dxvk::DomeLight domeLight;
          domeLight.radiance = du.radiance;
          domeLight.worldToLight = inverse(du.transform);
          domeLight.texture = preloadTexture(du.texturePath);

          uint32_t unused;
          ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);

          lightMgr.registerPersistentExternalLight(du.handle);
          lightMgr.addExternalDomeLight(du.handle, domeLight);
          lightMgr.addExternalLightInstance(du.handle);
        }

        // Apply any mesh creates not already flushed by remixapi_DrawInstance
        applyPendingMeshCreatesOnCs(ctx, meshCreates);

        lightMgr.queueAutoInstancePersistent();
      });

      // Forward any pending screen overlay to the render thread for this frame.
      dxvk::fork_hooks::presentScreenOverlayFlush(host);
    }


    // The endScene callback fires before the native present.
    dxvk::fork_hooks::presentEndSceneDispatch();

    HRESULT hr = remixDevice->Present(NULL, NULL, info ? info->hwndOverride : NULL, NULL);
    if (FAILED(hr)) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    // present callback + s_inFrame reset after successful native present
    dxvk::fork_hooks::presentCallbackDispatch();

    UINT windowWidth = 0, windowHeight = 0;
    {
      HWND hwnd = info && info->hwndOverride ? info->hwndOverride : remixDevice->GetWindow();
      if (hwnd) {
        RECT hwndRect = {};
        GetClientRect(hwnd, &hwndRect);
        windowWidth = static_cast<UINT>(std::max(0l, hwndRect.right - hwndRect.left));
        windowHeight = static_cast<UINT>(std::max(0l, hwndRect.bottom - hwndRect.top));
      }
    }

    if (windowWidth > 0 && windowHeight > 0) {
      IDirect3DSwapChain9* swapchain = nullptr;
      hr = remixDevice->GetSwapChain(0, &swapchain);
      if (FAILED(hr) || !swapchain) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
      }
      D3DPRESENT_PARAMETERS presentParams{};
      hr = swapchain->GetPresentParameters(&presentParams);
      if (FAILED(hr)) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
      }

      // reset swapchain if window has changed
      if (presentParams.BackBufferWidth != windowWidth && //
          presentParams.BackBufferHeight != windowHeight) {
        presentParams.BackBufferWidth = windowWidth;
        presentParams.BackBufferHeight = windowHeight;
        remixDevice->ResetEx(&presentParams, nullptr);
      }
    }

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  bool isVersionCompatible(uint64_t userVersion) {
    constexpr uint64_t compiledVersion = REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR, REMIXAPI_VERSION_MINOR, REMIXAPI_VERSION_PATCH);

    bool isDevelopment = 
      REMIXAPI_VERSION_GET_MAJOR(userVersion) == 0 &&
      REMIXAPI_VERSION_GET_MAJOR(compiledVersion) == 0;

    if (isDevelopment) {
      // each minor change is breaking
      return REMIXAPI_VERSION_GET_MINOR(userVersion) == REMIXAPI_VERSION_GET_MINOR(compiledVersion);
    }

    if (REMIXAPI_VERSION_GET_MAJOR(userVersion) == REMIXAPI_VERSION_GET_MAJOR(compiledVersion)) {
      // user version must be before the currently compiled version:
      // features that are requested by a user must be available in the current binary
      if (REMIXAPI_VERSION_GET_MINOR(userVersion) <= REMIXAPI_VERSION_GET_MINOR(compiledVersion)) {
        return true;
      }
    }

    return false;
  }
}

namespace dxvk {
  namespace {
    // Copies the mips of an imported view that fit in maxDimension pixels (0 = all) into a
    // Remix-owned image. Returns null when the allocation fails.
    Rc<DxvkImageView> copyImportView(DxvkContext* ctx, const Rc<DxvkImage>& image,
                                     const DxvkImageViewCreateInfo& viewInfo, uint32_t maxDimension,
                                     const char* name, uint32_t* skippedMips) {
      uint32_t skipMips = 0;
      if (maxDimension > 0) {
        while (skipMips + 1 < viewInfo.numLevels) {
          const VkExtent3D extent = image->mipLevelExtent(viewInfo.minLevel + skipMips);
          if (std::max(extent.width, extent.height) <= maxDimension) {
            break;
          }
          ++skipMips;
        }
      }
      *skippedMips = skipMips;

      DxvkImageCreateInfo copyInfo = {};
      copyInfo.type        = VK_IMAGE_TYPE_2D;
      copyInfo.format      = viewInfo.format;
      copyInfo.flags       = 0;
      copyInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      copyInfo.extent      = image->mipLevelExtent(viewInfo.minLevel + skipMips);
      copyInfo.numLayers   = 1;
      copyInfo.mipLevels   = viewInfo.numLevels - skipMips;
      copyInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      copyInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                           | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      copyInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      copyInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
      copyInfo.layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      auto copyImage = ctx->getDevice()->createImage(copyInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        DxvkMemoryStats::Category::RTXMaterialTexture, name);
      if (copyImage == nullptr) {
        return nullptr;
      }
      DxvkImageViewCreateInfo copyViewInfo = viewInfo;
      copyViewInfo.minLevel = 0;
      copyViewInfo.minLayer = 0;
      copyViewInfo.numLevels = copyInfo.mipLevels;
      auto copyView = ctx->getDevice()->createImageView(copyImage, copyViewInfo);

      for (uint32_t mip = 0; mip < copyInfo.mipLevels; ++mip) {
        const VkImageSubresourceLayers dstLayers = { VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1 };
        const VkImageSubresourceLayers srcLayers = {
          VK_IMAGE_ASPECT_COLOR_BIT, viewInfo.minLevel + skipMips + mip, viewInfo.minLayer, 1 };
        ctx->copyImage(copyImage, dstLayers, VkOffset3D { 0, 0, 0 },
                       image, srcLayers, VkOffset3D { 0, 0, 0 }, copyImage->mipLevelExtent(mip));
      }
      return copyView;
    }

    // Same key, same table index: the surface materials that recorded the index sample the
    // copy from the next bindless update on, and host materials are re-pointed at frame end.
    void replaceImportView(DxvkContext* ctx, uint64_t hash, const Rc<DxvkImageView>& view, HostTextureEncoding encoding) {
      const TextureRef ref(view, hash, hash, encoding);
      ctx->getCommonObjects()->getTextureManager().replaceTexture(ref);
      ctx->getCommonObjects()->getSceneManager().getAssetReplacer()->noteTextureReplaced(ref);
      ctx->getCommonObjects()->getImgui().ReleaseTexture(hash);
      ctx->getCommonObjects()->getImgui().AddTexture(hash, view, 1);
    }
  }

  // CS thread, from vram_budget::processReleasedImports: the game destroyed the D3D11 texture
  // behind an import. Each hash that still names its image moves to a copy of the mips up to
  // rtx.vramBudget.releasedImportMaxSize, so the game's image is freed.
  uint32_t remixHostImportDetachReleased(DxvkContext* ctx, DxvkImage* image) {
    const int maxSize = VramBudget::releasedImportMaxSize();
    auto found = s_importedImageHashes.find(image);
    if (maxSize <= 0 || found == s_importedImageHashes.end()) {
      return 0;
    }
    const std::vector<uint64_t> hashes = std::move(found->second);
    s_importedImageHashes.erase(found);
    s_sharedDefaultImports.erase(image);

    auto& textureManager = ctx->getCommonObjects()->getTextureManager();
    uint32_t detached = 0;
    for (const uint64_t hash : hashes) {
      TextureRef current;
      if (!textureManager.findTextureByUniqueKey(size_t(hash), current)) {
        continue;
      }
      const DxvkImageView* currentView = current.getImageView();
      if (currentView == nullptr || currentView->image().ptr() != image) {
        continue;
      }
      uint32_t skipped = 0;
      Rc<DxvkImageView> copyView = copyImportView(ctx, currentView->image(), currentView->info(),
        uint32_t(maxSize), "Remix imported D3D11 texture (game released it)", &skipped);
      if (copyView == nullptr) {
        continue;
      }
      replaceImportView(ctx, hash, copyView, current.getHostEncoding());
      s_importedImageHashes[copyView->image().ptr()].push_back(hash);
      ++detached;
    }
    return detached;
  }

  // Declared in d3d11_texture.h. CS thread, queued by a D3D11 context just
  // before it writes an image marked by an in-place DEFAULT import.
  void remixHostImportCopyOnWrite(DxvkContext* ctx, const Rc<DxvkImage>& image) {
    auto found = s_sharedDefaultImports.find(image.ptr());
    if (found == s_sharedDefaultImports.end()) {
      return;
    }
    const std::vector<uint64_t> hashes = std::move(found->second);
    s_sharedDefaultImports.erase(found);

    auto& textureManager = ctx->getCommonObjects()->getTextureManager();
    for (const uint64_t hash : hashes) {
      TextureRef current;
      if (!textureManager.findTextureByUniqueKey(size_t(hash), current)) {
        continue;  // destroyed since
      }
      const DxvkImageView* currentView = current.getImageView();
      if (currentView == nullptr || currentView->image().ptr() != image.ptr()) {
        continue;  // already replaced, or the hash now names another image
      }
      // rtx.vramBudget: under device-local pressure the copy keeps fewer top mips.
      uint32_t skipMips = 0;
      Rc<DxvkImageView> copyView = copyImportView(ctx, image, currentView->info(), vram_budget::importDimensionCap(),
        "Remix imported D3D11 texture (copy on write)", &skipMips);
      if (copyView == nullptr) {
        Logger::err(str::format("[HostImport] copy-on-write allocation failed for 0x", std::hex, hash,
                                "; the texture keeps sampling the rewritten image"));
        continue;
      }
      const VkDeviceSize copyBytes = copyView->image()->memSize();
      if (skipMips > 0) {
        vram_budget::noteImportCapped(copyBytes * ((uint64_t(1) << (2 * std::min(skipMips, 8u))) - 1));
      }
      replaceImportView(ctx, hash, copyView, current.getHostEncoding());
      s_importedImageHashes[copyView->image().ptr()].push_back(hash);

      s_d3d11ImportStats.copiedOnWrite += 1;
      s_d3d11ImportStats.copiedOnWriteBytes += copyBytes;
    }
  }

  // CS thread. A destroyed texture no longer needs copying when its image is written.
  void remixHostImportForgetShared(uint64_t hash, const DxvkImage* image) {
    auto imported = s_importedImageHashes.find(image);
    if (imported != s_importedImageHashes.end()) {
      auto& hashes = imported->second;
      hashes.erase(std::remove(hashes.begin(), hashes.end(), hash), hashes.end());
      if (hashes.empty()) {
        s_importedImageHashes.erase(imported);
      }
    }
    auto found = s_sharedDefaultImports.find(image);
    if (found == s_sharedDefaultImports.end()) {
      return;
    }
    auto& hashes = found->second;
    hashes.erase(std::remove(hashes.begin(), hashes.end(), hash), hashes.end());
    if (hashes.empty()) {
      s_sharedDefaultImports.erase(found);
    }
  }
}

namespace {
  // ---- Retained instances, second implementation: API front end (rtx.retained.useV2) -------------
  // Same contract as remix_c.h describes. Conversion and validation run on the calling thread outside
  // s_mutex; each call then queues exactly one command-stream lambda under s_mutex, however many
  // instances it carries. The lambdas go through SceneManager::retainedV2*, which pass each command to
  // whichever implementation owns the registered instances when it runs, so a switch between queueing
  // and execution loses nothing.

  // Handles are counted from a base the first implementation's counter (from 1) never reaches, so every
  // handle stays unique, and keeps working, across a switch.
  std::atomic<uint64_t> s_retainedV2LastHandle { uint64_t(1) << 48 };
  // Front end the retained exports use: -1 = not decided yet, else rtx.retained.useV2 as last read by
  // DrawRetainedInstances (read outside s_mutex: s_mutex is taken before the option mutex).
  std::atomic<int> s_retainedV2Front { -1 };

  void retainedV2ReadFront() {
    s_retainedV2Front.store(dxvk::RtxOptions::Retained::useV2() ? 1 : 0, std::memory_order_relaxed);
  }

  bool retainedV2Front() {
    if (s_retainedV2Front.load(std::memory_order_relaxed) < 0) {
      retainedV2ReadFront();
    }
    return s_retainedV2Front.load(std::memory_order_relaxed) > 0;
  }

  bool retainedV2InfoValid(const remixapi_InstanceInfo& info) {
    return info.sType == REMIXAPI_STRUCT_TYPE_INSTANCE_INFO && info.mesh != nullptr;
  }

  // ---- Scene interface: lock-free transport of retained commands (rtx.scene.commandQueue) -----------------------
  // The front end the retained exports use, as last read by DrawRetainedInstances (like s_retainedV2Front). Needs
  // rtx.scene.directRetained and the first implementation. Commands go into the scene's MPSC queue without s_mutex or
  // a command-stream lambda; DrawRetainedInstances queues the frame marker and one lambda that drains the queue.
  std::atomic<int> s_sceneQueueFront { 0 };

  void sceneQueueReadFront() {
    const bool on = dxvk::scene::Options::commandQueue() && dxvk::scene::Options::directRetained() &&
      !dxvk::RtxOptions::Retained::useV2();
    s_sceneQueueFront.store(on ? 1 : 0, std::memory_order_relaxed);
  }

  bool sceneQueueFront() {
    return s_sceneQueueFront.load(std::memory_order_relaxed) != 0;
  }

  void pushSceneCommands(dxvk::RemixApiDevice* remixDevice, std::vector<dxvk::scene::SceneCommand>&& commands) {
    auto batch = std::make_unique<dxvk::scene::SceneCommandBatch>();
    batch->commands = std::move(commands);
    remixDevice->GetDXVKDevice()->getCommon()->getSceneManager().retainedScene().queue().push(std::move(batch));
  }

  dxvk::scene::SceneCommand sceneDescribe(dxvk::scene::SceneCommand::Type type, uint64_t handle,
                                          dxvk::frontend::PreparedInstance&& instance) {
    dxvk::scene::SceneCommand command;
    command.type = type;
    command.handle = handle;
    command.instance = std::move(instance);
    return command;
  }

  // CreateRetainedInstance (count 1) and CreateRetainedInstances: all converted first, then one lambda.
  // Nothing is created unless every description is valid.
  remixapi_ErrorCode retainedV2Create(const remixapi_InstanceInfo* infos, uint32_t count, uint64_t* outHandles) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!infos || !outHandles) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    for (uint32_t i = 0; i < count; ++i) {
      if (!retainedV2InfoValid(infos[i])) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    // A batched CreateMesh must be registered before an instance names it.
    flushPendingMeshes(remixDevice);
    std::vector<dxvk::frontend::PreparedInstance> states(count);
    for (uint32_t i = 0; i < count; ++i) {
      states[i] = convert::toPreparedInstance(infos[i]);
      if (!states[i]) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    const uint64_t firstHandle = s_retainedV2LastHandle.fetch_add(count, std::memory_order_relaxed) + 1;
    if (sceneQueueFront()) {  // rtx.scene.commandQueue
      std::vector<dxvk::scene::SceneCommand> commands;
      commands.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        commands.push_back(sceneDescribe(dxvk::scene::SceneCommand::Type::Create, firstHandle + i, std::move(states[i])));
      }
      pushSceneCommands(remixDevice, std::move(commands));
    } else {
      std::lock_guard lock { s_mutex };
      remixDevice->EmitCs<false>([cFirst = firstHandle, cStates = std::move(states)](dxvk::DxvkContext* dxvkCtx) mutable {
        dxvk::frontend::SceneInterface scene { *static_cast<dxvk::RtxContext*>(dxvkCtx) };
        for (size_t i = 0; i < cStates.size(); ++i) {
          scene.describeInstance(cFirst + i, std::move(cStates[i]));
        }
      });
    }
    for (uint32_t i = 0; i < count; ++i) {
      outHandles[i] = firstHandle + i;
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // UpdateRetainedInstance: replaces the whole description; the instance keeps its visibility.
  remixapi_ErrorCode retainedV2Update(uint64_t handle, const remixapi_InstanceInfo* info) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle || !info || !retainedV2InfoValid(*info)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    flushPendingMeshes(remixDevice);
    dxvk::frontend::PreparedInstance instance = convert::toPreparedInstance(*info);
    if (!instance) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cHandle = handle, cInstance = std::move(instance)](dxvk::DxvkContext* dxvkCtx) mutable {
      dxvk::frontend::SceneInterface { *static_cast<dxvk::RtxContext*>(dxvkCtx) }.describeInstance(cHandle, std::move(cInstance));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // UpdateRetainedInstances: transform and/or visibility changes, one lambda for all of them.
  remixapi_ErrorCode retainedV2UpdateMany(const remixapi_RetainedInstanceUpdate* updates, uint32_t count) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!updates) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    struct Change {
      uint64_t handle;
      dxvk::Matrix4 absolute;
      uint32_t flags;
      bool visible;
    };
    std::vector<Change> changes(count);
    for (uint32_t i = 0; i < count; ++i) {
      const remixapi_RetainedInstanceUpdate& update = updates[i];
      if (!update.handle) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      Change& change = changes[i];
      change.handle = update.handle;
      change.flags = update.flags & (REMIXAPI_RETAINED_UPDATE_TRANSFORM | REMIXAPI_RETAINED_UPDATE_VISIBILITY);
      change.visible = update.visible != 0;
      if (change.flags & REMIXAPI_RETAINED_UPDATE_TRANSFORM) {
        change.absolute = convert::tomat4(update.transform);
      }
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cChanges = std::move(changes)](dxvk::DxvkContext* dxvkCtx) {
      dxvk::frontend::SceneInterface scene { *static_cast<dxvk::RtxContext*>(dxvkCtx) };
      for (const Change& change : cChanges) {
        if (change.flags & REMIXAPI_RETAINED_UPDATE_TRANSFORM) {
          scene.setTransform(change.handle, change.absolute);
        }
        if (change.flags & REMIXAPI_RETAINED_UPDATE_VISIBILITY) {
          scene.setVisible(change.handle, change.visible);
        }
      }
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode retainedV2Destroy(const uint64_t* handles, uint32_t count) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!handles) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::vector<uint64_t> gone(handles, handles + count);
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cGone = std::move(gone)](dxvk::DxvkContext* dxvkCtx) {
      dxvk::frontend::SceneInterface scene { *static_cast<dxvk::RtxContext*>(dxvkCtx) };
      for (const uint64_t handle : cGone) {
        scene.destroyInstance(handle);
      }
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode retainedV2Draw(const float* sceneOrigin) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    dxvk::Vector3 origin { 0.f, 0.f, 0.f };
    if (sceneOrigin) {
      if (!std::isfinite(sceneOrigin[0]) || !std::isfinite(sceneOrigin[1]) || !std::isfinite(sceneOrigin[2])) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      origin = dxvk::Vector3 { sceneOrigin[0], sceneOrigin[1], sceneOrigin[2] };
    }
    dxvk::fork_hooks::notifyBeginScene();
    std::lock_guard lock { s_mutex };
    // The replay itself picks the implementation (RtxContext::commitRetainedExternalGeometry).
    remixDevice->EmitCs<false>([cOrigin = origin](dxvk::DxvkContext* dxvkCtx) {
      dxvk::frontend::SceneInterface { *static_cast<dxvk::RtxContext*>(dxvkCtx) }.drawRetainedInstances(dxvk::frontend::FrameInfo { cOrigin });
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }
}

extern "C"
{
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_AutoInstancePersistentLights(void) {
    REMIX_FORK_RECORD0(AutoInstancePersistentLights);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    // Drain pending work and apply on render thread at a safe point
    std::vector<PendingLightCreate> creates;
    std::vector<PendingLightUpdate> updates;
    std::vector<PendingDomeUpdate> domeUpdates;
    std::vector<remixapi_LightHandle> destroys;
    std::vector<PendingMeshCreate> meshCreates;
    {
      std::lock_guard lock { s_mutex };
      s_handlesDeletedThisFrame.clear();
      creates.swap(s_pendingLightCreates);
      updates.swap(s_pendingLightUpdates);
      domeUpdates.swap(s_pendingDomeUpdates);
      destroys.swap(s_pendingLightDestroys);
      meshCreates.swap(s_pendingMeshCreates);
      s_hasPendingMeshCreates.store(false);
    }
    // Native-present fast path. With no C-API scene work queued this frame and no external light ever
    // registered, there is nothing to apply or re-instance. An empty CS chunk (and an extra LockDevice)
    // on every native D3D9 Present disturbs the light pipeline, so this skips to the self-gating overlay
    // flush. C-API light users set s_externalLightApiUsed and keep the per-frame path.
    const bool hasPendingApiWork =
        !creates.empty() || !updates.empty() || !domeUpdates.empty() ||
        !destroys.empty() || !meshCreates.empty();
    if (!hasPendingApiWork && !s_externalLightApiUsed.load(std::memory_order_relaxed)) {
      dxvk::fork_hooks::presentScreenOverlayFlush(remixDevice);
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    auto devLock = remixDevice->LockDevice();
    remixDevice->EmitCs([creates = std::move(creates), updates = std::move(updates), domeUpdates = std::move(domeUpdates), destroys = std::move(destroys), meshCreates = std::move(meshCreates)](dxvk::DxvkContext* ctx) mutable {
      auto& lightMgr = ctx->getCommonObjects()->getSceneManager().getLightManager();
      // Apply destroys first
      for (auto h : destroys) {
        if (h) {
          lightMgr.unregisterPersistentExternalLight(h);
          lightMgr.removeExternalLight(h);
        }
      }
      // Apply creates
      for (auto& create : creates) {
        if (create.isDome) {
          // Build dome light on the render thread
          auto preloadTexture = [ctx](const std::filesystem::path& path) {
            if (path.empty()) {
              return dxvk::TextureRef{};
            }
            auto assetData = dxvk::AssetDataManager::get().findAsset(path.string().c_str());
            if (assetData == nullptr) {
              return dxvk::TextureRef{};
            }
            auto uploadedTexture = ctx->getCommonObjects()->getTextureManager()
              .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true);
            return dxvk::TextureRef{ uploadedTexture };
          };

          dxvk::DomeLight domeLight;
          domeLight.radiance = create.radiance;
          domeLight.worldToLight = inverse(create.transform);
          domeLight.texture = preloadTexture(create.texturePath);

          uint32_t unused;
          ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);

          lightMgr.addExternalDomeLight(create.handle, domeLight);
        } else if (create.rtLight.has_value()) {
          // Analytical light
          lightMgr.addExternalLight(create.handle, *create.rtLight);
        }
        // Register all created lights as persistent
        lightMgr.registerPersistentExternalLight(create.handle);
        lightMgr.addExternalLightInstance(create.handle);
      }
      // Apply analytical updates
      for (auto& upd : updates) {
        if (upd.rtLight.has_value()) {
          lightMgr.registerPersistentExternalLight(upd.handle);
          lightMgr.addExternalLight(upd.handle, *upd.rtLight);
          lightMgr.addExternalLightInstance(upd.handle);
        }
      }
      // Apply dome updates
      for (auto& du : domeUpdates) {
        auto preloadTexture = [ctx](const std::filesystem::path& path) {
          if (path.empty()) {
            return dxvk::TextureRef{};
          }
          auto assetData = dxvk::AssetDataManager::get().findAsset(path.string().c_str());
          if (assetData == nullptr) {
            return dxvk::TextureRef{};
          }
          auto uploadedTexture = ctx->getCommonObjects()->getTextureManager()
            .preloadTextureAsset(assetData, dxvk::ColorSpace::AUTO, true);
          return dxvk::TextureRef{ uploadedTexture };
        };

        dxvk::DomeLight domeLight;
        domeLight.radiance = du.radiance;
        domeLight.worldToLight = inverse(du.transform);
        domeLight.texture = preloadTexture(du.texturePath);

        uint32_t unused;
        ctx->getCommonObjects()->getSceneManager().trackTexture(domeLight.texture, unused, true, true);

        lightMgr.registerPersistentExternalLight(du.handle);
        lightMgr.addExternalDomeLight(du.handle, domeLight);
        lightMgr.addExternalLightInstance(du.handle);
      }

      // Apply any mesh creates not already flushed by remixapi_DrawInstance
      applyPendingMeshCreatesOnCs(ctx, meshCreates);

      // Ensure persistent auto-instancing happens every frame
      lightMgr.queueAutoInstancePersistent();
    });

    // Forward any pending screen overlay to the render thread for this frame.
    dxvk::fork_hooks::presentScreenOverlayFlush(remixDevice);

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_UpdateLightDefinition(
    remixapi_LightHandle handle,
    const remixapi_LightInfo* info) {
    REMIX_FORK_RECORD(UpdateLightDefinition, handle, info);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!handle || !info) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    // An update can create-or-replace an external light, so treat it as
    // external-light usage for the persistent re-instancing gate.
    s_externalLightApiUsed.store(true, std::memory_order_relaxed);
    // Handle dome light update if present in pNext chain
    if (auto extDome = pnext::find<remixapi_LightInfoDomeEXT>(info)) {
      auto cTransform = convert::tomat4(extDome->transform);
      auto cTexturePath = convert::topath(extDome->colorTexture);
      auto cRadiance = convert::tovec3(info->radiance);
      {
        std::lock_guard lock { s_mutex };
        s_pendingDomeUpdates.push_back(PendingDomeUpdate{ handle, cTransform, cTexturePath, cRadiance });
      }
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }

    // For analytical lights require base LightInfo; convert immediately
    if (info->sType != REMIXAPI_STRUCT_TYPE_LIGHT_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto rt = convert::toRtLight(*info);
    if (!rt.has_value()) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    // Set the isDynamic flag from the LightInfo
    rt->isDynamic = info->isDynamic;

    // Set the ignoreViewModel flag from the LightInfo
    rt->ignoreViewModel = info->ignoreViewModel;

    {
      std::lock_guard lock { s_mutex };
      s_pendingLightUpdates.push_back(PendingLightUpdate{ handle, std::move(rt) });
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RegisterCallbacks(
    PFN_remixapi_BridgeCallback beginSceneCallback,
    PFN_remixapi_BridgeCallback endSceneCallback,
    PFN_remixapi_BridgeCallback presentCallback) {
    REMIX_FORK_RECORD(RegisterCallbacks, beginSceneCallback, endSceneCallback, presentCallback);
    dxvk::fork_hooks::registerCallbacks(beginSceneCallback, endSceneCallback, presentCallback);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_UIState REMIXAPI_CALL remixapi_GetUIState(void) {
    return dxvk::fork_hooks::getUiState(tryAsDxvk());
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_SetUIState(remixapi_UIState state) {
    REMIX_FORK_RECORD(SetUIState, state);
    return dxvk::fork_hooks::setUiState(tryAsDxvk(), state);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawScreenOverlay(
    const void*     pPixelData,
    uint32_t        width,
    uint32_t        height,
    remixapi_Format format,
    float           opacity) {
    REMIX_FORK_RECORD(DrawScreenOverlay, pPixelData, width, height, format, opacity);
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    return dxvk::fork_hooks::drawScreenOverlay(remixDevice, pPixelData, width, height, format, opacity);
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_SetGameValue(
    const char* key,
    const char* value) {
    REMIX_FORK_RECORD(SetGameValue, key, value);
    if (!key || key[0] == '\0' || !value) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    dxvk::fork_game_state::GameStateStore::get().set(
      std::string{ key }, std::string{ value });

    // rtx.worldOrigin.enable, D3D9 camera path: a host whose camera arrives as D3D9 transforms has no
    // SetupCamera to carry its origin, so the origin travels from here instead -- read on the host's thread
    // and sent through the command stream, in order with the D3D9 draws around it. The render thread applies
    // it ahead of the frame's Main camera update (SceneManager::applyPendingHostWorldOrigin); a host that uses
    // SetupCamera gets it applied there and this is dropped. Every component write re-sends the triple, so a
    // host that writes x, y, z before its first draw leaves the complete one last.
    if (std::strncmp(key, "__mcrtx.world_origin.", 21) == 0 && dxvk::WorldOrigin::enable()) {
      dxvk::Vector3 worldOrigin { 0.f, 0.f, 0.f };
      dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
      if (remixDevice && dxvk::fork_camera_origin::tryReadWorldOriginOffsetFromGameState(worldOrigin)) {
        std::lock_guard lock { s_mutex };
        auto devLock = remixDevice->LockDevice();
        remixDevice->EmitCs([worldOrigin](dxvk::DxvkContext* ctx) {
          ctx->getCommonObjects()->getSceneManager().setPendingHostWorldOrigin(worldOrigin);
        });
      }
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  remixapi_ErrorCode REMIXAPI_CALL remixapi_GetGameValue(
    const char* key,
    char*       out_buffer,
    uint32_t    in_buffer_size,
    uint32_t*   out_actual_size) {
    REMIX_FORK_RECORD(GetGameValue, key, out_buffer, in_buffer_size, out_actual_size);
    if (key == nullptr || key[0] == '\0') {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (out_actual_size == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (in_buffer_size > 0 && out_buffer == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    std::string value;
    if (!dxvk::fork_game_state::GameStateStore::get().tryGet(std::string{ key }, value)) {
      *out_actual_size = 0;
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }

    const uint32_t needed = static_cast<uint32_t>(value.size()) + 1u;
    *out_actual_size = needed;
    if (in_buffer_size >= needed) {
      std::memcpy(out_buffer, value.data(), value.size());
      out_buffer[value.size()] = '\0';
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---- Retained instances (after doodlum/dxvk-remix 5e39a218a) -----------
  // See remix_c.h. Conversion runs outside s_mutex -- it is the expensive
  // half and touches nothing shared. Unlike per-frame DrawInstance, updates
  // and destroys are batched: one lock and one command-stream lambda for all
  // of a frame's changes, however many objects moved.

  static std::atomic<uint64_t> s_nextRetainedInstance { 0 };

  static remixapi_ErrorCode setRetainedInstance(uint64_t handle, const remixapi_InstanceInfo* info) {
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh || !handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    // A batched CreateMesh must be registered before a draw names it.
    flushPendingMeshes(remixDevice);
    dxvk::frontend::PreparedInstance instance = convert::toPreparedInstance(*info);
    if (!instance) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (sceneQueueFront()) {  // rtx.scene.commandQueue
      std::vector<dxvk::scene::SceneCommand> commands;
      commands.push_back(sceneDescribe(dxvk::scene::SceneCommand::Type::Describe, handle, std::move(instance)));
      pushSceneCommands(remixDevice, std::move(commands));
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cHandle = handle, cInstance = std::move(instance)](dxvk::DxvkContext* dxvkCtx) mutable {
      dxvk::frontend::SceneInterface { *static_cast<dxvk::RtxContext*>(dxvkCtx) }.describeInstance(cHandle, std::move(cInstance));
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateRetainedInstance(
      const remixapi_InstanceInfo* info, uint64_t* out_handle) {
    REMIX_FORK_RECORD(CreateRetainedInstance, info, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateRetained, 1 }; // rtx.profile.detailedLogging
    if (retainedV2Front()) {  // rtx.retained.useV2
      return retainedV2Create(info, 1u, out_handle);
    }
    if (!out_handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const uint64_t handle = ++s_nextRetainedInstance;
    const auto result = setRetainedInstance(handle, info);
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      *out_handle = handle;
    }
    return result;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_UpdateRetainedInstance(
      uint64_t handle, const remixapi_InstanceInfo* info) {
    REMIX_FORK_RECORD(UpdateRetainedInstance, handle, info);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::UpdateRetained, 1 }; // rtx.profile.detailedLogging
    if (retainedV2Front()) {  // rtx.retained.useV2
      return retainedV2Update(handle, info);
    }
    return setRetainedInstance(handle, info);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_UpdateRetainedInstances(
      const remixapi_RetainedInstanceUpdate* updates, uint32_t count) {
    REMIX_FORK_RECORD(UpdateRetainedInstances, updates, count);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::UpdateRetainedBatch, count }; // rtx.profile.detailedLogging
    if (retainedV2Front()) {  // rtx.retained.useV2
      return retainedV2UpdateMany(updates, count);
    }
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!updates) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    struct Converted {
      uint64_t handle;
      dxvk::Matrix4 transform;
      uint32_t flags;
      bool visible;
    };
    std::vector<Converted> converted;
    converted.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& u = updates[i];
      if (!u.handle) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      converted.push_back(Converted {
        u.handle,
        (u.flags & REMIXAPI_RETAINED_UPDATE_TRANSFORM) ? convert::tomat4(u.transform) : dxvk::Matrix4 {},
        u.flags,
        u.visible != 0 });
    }
    if (sceneQueueFront()) {  // rtx.scene.commandQueue
      std::vector<dxvk::scene::SceneCommand> commands(converted.size());
      for (size_t i = 0; i < converted.size(); ++i) {
        commands[i].type = dxvk::scene::SceneCommand::Type::Update;
        commands[i].handle = converted[i].handle;
        commands[i].update.handle = converted[i].handle;
        commands[i].update.flags = converted[i].flags;
        commands[i].update.objectToWorld = converted[i].transform;
        commands[i].update.visible = converted[i].visible;
      }
      pushSceneCommands(remixDevice, std::move(commands));
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cUpdates = std::move(converted)](dxvk::DxvkContext* dxvkCtx) {
      dxvk::frontend::SceneInterface scene { *static_cast<dxvk::RtxContext*>(dxvkCtx) };
      for (const auto& u : cUpdates) {
        if (u.flags & REMIXAPI_RETAINED_UPDATE_TRANSFORM) {
          scene.setTransform(u.handle, u.transform);
        }
        if (u.flags & REMIXAPI_RETAINED_UPDATE_VISIBILITY) {
          scene.setVisible(u.handle, u.visible);
        }
      }
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyRetainedInstances(
      const uint64_t* handles, uint32_t count) {
    REMIX_FORK_RECORD(DestroyRetainedInstances, handles, count);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DestroyRetained, count }; // rtx.profile.detailedLogging
    if (retainedV2Front()) {  // rtx.retained.useV2
      return retainedV2Destroy(handles, count);
    }
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!handles) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (sceneQueueFront()) {  // rtx.scene.commandQueue
      std::vector<dxvk::scene::SceneCommand> commands(count);
      for (uint32_t i = 0; i < count; ++i) {
        commands[i].type = dxvk::scene::SceneCommand::Type::Destroy;
        commands[i].handle = handles[i];
      }
      pushSceneCommands(remixDevice, std::move(commands));
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    std::vector<uint64_t> doomed(handles, handles + count);
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cHandles = std::move(doomed)](dxvk::DxvkContext* dxvkCtx) {
      dxvk::frontend::SceneInterface scene { *static_cast<dxvk::RtxContext*>(dxvkCtx) };
      for (uint64_t h : cHandles) {
        scene.destroyInstance(h);
      }
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // Dry-run replay of a D3D9 identity trace.
  // For tools/remix_replay --identity-trace: the runtime is started but presents nothing, so the replay runs on the
  // calling thread with a private draw-call tracker. Options set before the call (rtx.conf, SetConfigVariable) are
  // resolved here, since no frame end does it.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_fork_ReplayIdentityTrace(
      const wchar_t* path,
      remixapi_fork_IdentityReplayResult* result) {
    if (!path || !result) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    std::lock_guard lock { s_mutex };
    dxvk::DxvkDevice* device = remixDevice->GetDXVKDevice().ptr();
    dxvk::RtxOptionManager::applyPendingValues(device, false);
    return dxvk::D3d9IdentityTrace::replay(device, path, *result) ? REMIXAPI_ERROR_CODE_SUCCESS
                                                                   : REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  // Batched CreateRetainedInstance (remix_c.h): one lock and one command-stream lambda for all of them.
  // Nothing is created unless every description is valid. Works with either implementation: the queued
  // commands reach whichever one owns the registered instances.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateRetainedInstances(
      const remixapi_InstanceInfo* infos, uint32_t count, uint64_t* out_handles) {
    REMIX_FORK_RECORD(CreateRetainedInstances, infos, count, out_handles);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::CreateRetained, count }; // rtx.profile.detailedLogging
    return retainedV2Create(infos, count, out_handles);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawRetainedInstances(const float* sceneOrigin) {
    REMIX_FORK_RECORD(DrawRetainedInstances, sceneOrigin);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::DrawRetained, 1 }; // rtx.profile.detailedLogging
    retainedV2ReadFront();  // rtx.retained.useV2: the front end follows the option from here on
    if (retainedV2Front()) {
      return retainedV2Draw(sceneOrigin);
    }
    const bool queueBefore = sceneQueueFront();
    sceneQueueReadFront();  // rtx.scene.commandQueue: likewise
    dxvk::RemixApiDevice* remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    dxvk::Vector3 origin { 0.f, 0.f, 0.f };
    if (sceneOrigin) {
      for (uint32_t i = 0; i < 3; ++i) {
        if (!std::isfinite(sceneOrigin[i])) {
          return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
        }
      }
      origin = dxvk::Vector3 { sceneOrigin[0], sceneOrigin[1], sceneOrigin[2] };
    }
    dxvk::fork_hooks::notifyBeginScene();
    if (sceneQueueFront()) {
      std::vector<dxvk::scene::SceneCommand> commands(1);
      commands[0].type = dxvk::scene::SceneCommand::Type::Frame;
      commands[0].origin = origin;
      pushSceneCommands(remixDevice, std::move(commands));
      std::lock_guard lock { s_mutex };
      remixDevice->EmitCs<false>([](dxvk::DxvkContext* dxvkCtx) {
        static_cast<dxvk::RtxContext*>(dxvkCtx)->drainSceneCommands();
      });
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    std::lock_guard lock { s_mutex };
    remixDevice->EmitCs<false>([cOrigin = origin, cDrain = queueBefore](dxvk::DxvkContext* dxvkCtx) {
      auto* ctx = static_cast<dxvk::RtxContext*>(dxvkCtx);
      if (cDrain) {
        // The queue was in use until now: what it holds comes first.
        ctx->drainSceneCommands();
      }
      dxvk::frontend::SceneInterface { *ctx }.drawRetainedInstances(dxvk::frontend::FrameInfo { cOrigin });
    });
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RegisterD3D11Device(void* device) {
    auto* native = device ? dynamic_cast<dxvk::D3D11Device*>(static_cast<ID3D11Device*>(device)) : nullptr;
    if (!native) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard lock { s_mutex };
    if (s_dxvkDevice || s_dxvkD3D9) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    ID3D11DeviceContext* context = nullptr;
    native->GetImmediateContext(&context);
    auto* immediate = static_cast<dxvk::D3D11ImmediateContext*>(context);
    immediate->EnableRemixHost();
    s_liveMeshHandles.clear();
    s_dxvkDevice = std::make_unique<dxvk::RemixApiDevice>(immediate, native->GetDXVKDevice());
    dxvk::g_dxvkDeviceNative = native->GetDXVKDevice().ptr();
    dxvk::Logger::info("[Remix.D3D11] Registered native D3D11 API host; game command stream and swapchain");
    dxvk::ApiRecorder::onDeviceRegistered(true);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RenderD3D11(void* destination, remixapi_Bool render) {
    REMIX_FORK_RECORD(RenderD3D11, destination, render);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::RenderD3D11, 1 }; // rtx.profile.detailedLogging
    auto* host = tryAsDxvk();
    if (!host || !host->d3d11()) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    auto* texture = destination ? dynamic_cast<dxvk::D3D11Texture2D*>(
      static_cast<ID3D11Texture2D*>(destination)) : nullptr;
    if (!texture) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    dxvk::Com<ID3D11Device> owner;
    texture->GetDevice(&owner);
    dxvk::Com<ID3D11Device> registered;
    host->d3d11()->GetDevice(&registered);
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    if (owner != registered || desc.SampleDesc.Count != 1 || desc.ArraySize != 1 ||
        !(desc.BindFlags & D3D11_BIND_RENDER_TARGET)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    auto status = remixapi_AutoInstancePersistentLights();
    if (status != REMIXAPI_ERROR_CODE_SUCCESS) {
      return status;
    }
    dxvk::fork_hooks::presentEndSceneDispatch();
    {
      std::lock_guard lock { s_mutex };
      host->d3d11()->RenderRemix(texture->GetCommonTexture()->GetImage(), render != 0);
    }
    dxvk::fork_hooks::presentCallbackDispatch();
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  static remixapi_ErrorCode drawScreenOverlayD3D11(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count, const float* alphaCutoffs) {
    auto* host = tryAsDxvk();
    if (!host || !host->d3d11()) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (count > REMIXAPI_SCREEN_OVERLAY_MAX_LAYERS || (count != 0 && !layers)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    // Layers stay composited until replaced, so a refusal must also drop the
    // current set or a host falling back to the CPU path would see both.
    static bool s_gpuOverlayLive = false;
    auto submit = [host](std::vector<dxvk::RtxContext::GpuScreenOverlaySource>&& sources) {
      std::lock_guard lock { s_mutex };
      auto devLock = host->LockDevice();
      s_gpuOverlayLive = !sources.empty();
      host->EmitCs([cSources = std::move(sources)](dxvk::DxvkContext* ctx) mutable {
        static_cast<dxvk::RtxContext*>(ctx)->setGpuScreenOverlayLayers(std::move(cSources));
      });
    };
    if (!dxvk::RtxOptions::enableGpuScreenOverlay()) {
      if (s_gpuOverlayLive) {
        submit({});
      }
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }

    dxvk::Com<ID3D11Device> registered;
    host->d3d11()->GetDevice(&registered);
    std::vector<dxvk::RtxContext::GpuScreenOverlaySource> sources;
    sources.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const auto& layer = layers[i];
      auto* texture = layer.texture2D ? dynamic_cast<dxvk::D3D11Texture2D*>(
        static_cast<ID3D11Texture2D*>(layer.texture2D)) : nullptr;
      if (!texture || !std::isfinite(layer.opacity)) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::Com<ID3D11Device> owner;
      texture->GetDevice(&owner);
      D3D11_TEXTURE2D_DESC desc;
      texture->GetDesc(&desc);
      const auto image = texture->GetCommonTexture()->GetImage();
      if (owner != registered || image == nullptr || desc.SampleDesc.Count != 1 ||
          desc.ArraySize != 1 || desc.Usage == D3D11_USAGE_STAGING) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::RtxContext::GpuScreenOverlaySource source;
      source.image = image;
      source.flags = (layer.flags & REMIXAPI_SCREEN_OVERLAY_LAYER_PREMULTIPLIED)
        ? SCREEN_OVERLAY_FLAG_PREMULTIPLIED : 0u;
      source.opacity = std::clamp(layer.opacity, 0.0f, 1.0f);
      bool fullRect = !(layer.rect[2] > 0.0f) || !(layer.rect[3] > 0.0f);
      for (float v : layer.rect) {
        fullRect = fullRect || !std::isfinite(v);
      }
      if (!fullRect) {
        std::copy(std::begin(layer.rect), std::end(layer.rect), std::begin(source.rect));
      }
      if (alphaCutoffs) {
        const float cutoff = alphaCutoffs[i];
        source.alphaCutoff = std::isfinite(cutoff) ? std::clamp(cutoff, 0.0f, 1.0f) : 0.0f;
      }
      sources.push_back(std::move(source));
    }
    submit(std::move(sources));
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawScreenOverlayD3D11(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count) {
    REMIX_FORK_RECORD(DrawScreenOverlayD3D11, layers, count);
    return drawScreenOverlayD3D11(layers, count, nullptr);
  }

  // Per-layer alpha cutoff. An optional named export, whose presence is the
  // capability check (see remix_c.h).
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawScreenOverlayD3D11Ex(
      const remixapi_ScreenOverlayLayerD3D11* layers, uint32_t count, const float* alphaCutoffs) {
    REMIX_FORK_RECORD(DrawScreenOverlayD3D11Ex, layers, count, alphaCutoffs);
    return drawScreenOverlayD3D11(layers, count, alphaCutoffs);
  }

  static remixapi_ErrorCode prepareD3D11Texture(
      void* texture2D, uint32_t dxgiFormat, uint32_t maxDimension, remixapi_D3D11TextureToken* out_token) {
    if (!out_token) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    *out_token = nullptr;
    auto remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (!texture2D) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    try {
      dxvk::Com<IDXGIRemixTexture> source;
      if (FAILED(static_cast<IUnknown*>(texture2D)->QueryInterface(
          __uuidof(IDXGIRemixTexture), reinterpret_cast<void**>(&source)))) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      auto token = std::make_unique<PreparedD3D11Texture>();
      token->device = remixDevice->GetDXVKDevice();
      void* hostContext = remixDevice->d3d11()
        ? static_cast<ID3D11DeviceContext*>(remixDevice->d3d11()) : nullptr;
      BOOL deferInitFlush = dxvk::RtxOptions::HostImport::deferPrepareFlush() && hostContext != nullptr;
      if (FAILED(source->PrepareRemixView(token->device->handle(),
          static_cast<DXGI_FORMAT>(dxgiFormat), hostContext, &token->view, &deferInitFlush)) || token->view == nullptr) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      token->initFlushDeferred = deferInitFlush != FALSE;
      dxvk::Com<ID3D11Texture2D> texture;
      if (SUCCEEDED(static_cast<IUnknown*>(texture2D)->QueryInterface(
          __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&texture)))) {
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        token->immutable = desc.Usage == D3D11_USAGE_IMMUTABLE;
        token->texture = std::move(texture);
      }
      token->maxDimension = maxDimension;
      dxvk::vram_budget::registerImportToken(token.get(), token->texture.ptr(), token->view->image()->memSize());
      *out_token = token.release();
      return REMIXAPI_ERROR_CODE_SUCCESS;
    } catch (const std::exception& e) {
      dxvk::Logger::err(dxvk::str::format("PrepareD3D11Texture: ", e.what()));
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_PrepareD3D11Texture(
      void* texture2D, uint32_t dxgiFormat, remixapi_D3D11TextureToken* out_token) {
    REMIX_FORK_RECORD(PrepareD3D11Texture, texture2D, dxgiFormat, out_token);
    return prepareD3D11Texture(texture2D, dxgiFormat, 0, out_token);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_PrepareD3D11TextureEx(
      void* texture2D, uint32_t dxgiFormat, uint32_t maxDimension, remixapi_D3D11TextureToken* out_token) {
    REMIX_FORK_RECORD(PrepareD3D11TextureEx, texture2D, dxgiFormat, maxDimension, out_token);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::PrepareD3D11Texture, 1 }; // rtx.profile.detailedLogging
    return prepareD3D11Texture(texture2D, dxgiFormat, maxDimension, out_token);
  }

  // CS thread. Registers an imported view under hash, or -- when hash is
  // already live and in-place replacement is on (always for
  // remixapi_ReplaceD3D11TextureEx) -- re-points that entry at the new view:
  // same unique key, same texture-table index, so every surface material that
  // recorded the index samples the new image from the next bindless update on,
  // with no material, mesh or BLAS rebuilt. The old view leaves the table
  // here; the bindless table still holds it until its slot is rewritten, and
  // either per-frame tracking or rtx.trackBindlessResourcesOnChange hands it
  // to a command list, so the old image is freed only after every submission
  // that could read it has retired. API materials hold their own TextureRef;
  // AssetReplacer::applyTextureRefreshes re-points it at the new view at the
  // end of the frame, so materials do not keep the old image alive.
  static void registerImportedView(dxvk::DxvkContext* ctx, const dxvk::Rc<dxvk::DxvkImageView>& view,
                                   uint64_t hash, dxvk::HostTextureEncoding hostEncoding,
                                   bool shareDefault, bool replaceOnly) {
    auto& textureManager = ctx->getCommonObjects()->getTextureManager();
    auto& imgui = ctx->getCommonObjects()->getImgui();
    const bool inPlace = replaceOnly || dxvk::RtxOptions::HostImport::enableInPlaceTextureReplace();
    dxvk::TextureRef current;
    const bool live = inPlace && textureManager.findTextureByUniqueKey(size_t(hash), current) &&
                      current.getImageHash() == hash;
    if (live) {
      // Before the new registration below: old and new may be one image.
      if (const dxvk::DxvkImageView* currentView = current.getImageView()) {
        dxvk::remixHostImportForgetShared(hash, currentView->image().ptr());
      }
    } else if (replaceOnly) {
      // Destroyed before the replace reached this thread: do not resurrect it.
      s_d3d11ImportStats.replaceDropped += 1;
      return;
    }
    if (shareDefault) {
      s_sharedDefaultImports[view->image().ptr()].push_back(hash);
    }
    s_importedImageHashes[view->image().ptr()].push_back(hash);
    dxvk::vram_budget::setReleasedImportHandler(&dxvk::remixHostImportDetachReleased);
    auto ref = dxvk::TextureRef(view, hash, hash, hostEncoding);
    if (live) {
      textureManager.replaceTexture(ref);
      ctx->getCommonObjects()->getSceneManager().getAssetReplacer()->noteTextureReplaced(ref);
      imgui.ReleaseTexture(hash);
      s_d3d11ImportStats.replaced += 1;
    } else {
      uint32_t index;
      textureManager.addTexture(ref, 0, false, index);
    }
    imgui.AddTexture(hash, view, 1);
  }

  static remixapi_ErrorCode importD3D11Texture(
      remixapi_D3D11TextureToken token, uint64_t hash, uint32_t encoding, bool replaceOnly,
      remixapi_TextureHandle* out_handle) {
    if (!out_handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    *out_handle = nullptr;
    if (!token || !hash || encoding > REMIXAPI_TEXTURE_ENCODING_SMOOTHNESS_IN_GREEN) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const auto hostEncoding = static_cast<dxvk::HostTextureEncoding>(encoding);
    auto remixDevice = tryAsDxvk();
    if (!remixDevice) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    auto prepared = static_cast<const PreparedD3D11Texture*>(token);
    if (prepared->device != remixDevice->GetDXVKDevice()) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    try {
      const auto& srcView = prepared->view;
      const auto srcImage = srcView->image();

      // Mips larger than the host's cap (PrepareD3D11TextureEx) or
      // rtx.hostImport.maxCopiedTextureDimension, whichever is smaller, are
      // left out. The game keeps its own full chain either way.
      uint32_t maxDimension = prepared->maxDimension;
      const int optionCap = dxvk::RtxOptions::HostImport::maxCopiedTextureDimension();
      if (optionCap > 0 && (maxDimension == 0 || uint32_t(optionCap) < maxDimension)) {
        maxDimension = uint32_t(optionCap);
      }
      uint32_t firstMip = 0;
      if (maxDimension > 0) {
        while (firstMip + 1 < srcView->info().numLevels) {
          const VkExtent3D extent = srcImage->mipLevelExtent(srcView->info().minLevel + firstMip);
          if (std::max(extent.width, extent.height) <= maxDimension) {
            break;
          }
          ++firstMip;
        }
      }

      const bool recordCopy = dxvk::ApiRecorder::forcesImportCopy();
      const bool shareImmutable = !recordCopy && prepared->immutable && dxvk::RtxOptions::HostImport::shareImmutableTextures();
      const bool shareDefault = !recordCopy && !prepared->immutable && prepared->texture != nullptr &&
        dxvk::RtxOptions::HostImport::shareDefaultTextures();
      if (shareImmutable || shareDefault) {
        // An IMMUTABLE image can never be written after its initial upload, so
        // the recycling hazard that forces the copy below cannot occur: a new
        // texture object always gets a new image. Sample the game's image in
        // place. The TextureRef (and the ImGui catalog) hold the view, and the
        // view holds the DxvkImage, so the game releasing its texture only
        // drops its own reference; the memory lives until DestroyTexture.
        // A capped import samples through a view that starts at the first mip
        // that fits; that saves no memory (the game owns the image) but keeps
        // what Remix samples the same as a capped copy would be.
        dxvk::Rc<dxvk::DxvkImageView> sharedView = srcView;
        if (firstMip > 0) {
          dxvk::DxvkImageViewCreateInfo viewInfo = srcView->info();
          viewInfo.minLevel += firstMip;
          viewInfo.numLevels -= firstMip;
          sharedView = remixDevice->GetDXVKDevice()->createImageView(srcImage, viewInfo);
        }
        std::lock_guard lock { s_mutex };
        auto devLock = remixDevice->LockDevice();
        //
        // A DEFAULT image can be rewritten -- FO4 recycles texture objects for
        // new content -- so it is shared copy-on-write: the texture is marked,
        // and the first write through any context has Remix copy the image
        // (remixHostImportCopyOnWrite) before the write lands. Materials keep
        // the contents they were imported with, as with an up-front copy, and
        // an image the game never rewrites is never duplicated.
        s_d3d11ImportStats.note(true, srcImage->memSize(), 0);
        remixDevice->EmitCs([view = std::move(sharedView), hash, hostEncoding, shareDefault, replaceOnly,
                             flushHost = prepared->initFlushDeferred ? remixDevice->d3d11() : nullptr](dxvk::DxvkContext* ctx) {
          if (flushHost) {
            // Prepare deferred this: the game's initial upload must be queued
            // before any submission that can sample the image.
            flushHost->FlushRemixInitUploads();
          }
          // Prepare queued the producer's upload ahead of this. It was recorded
          // with the image's own stages, which do not name compute or ray
          // tracing; the next frame opens with one memory barrier covering
          // every image imported since the last one.
          view->image()->addShaderReadStages(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
          static_cast<dxvk::RtxContext*>(ctx)->requestImportedImageBarrier();
          registerImportedView(ctx, view, hash, hostEncoding, shareDefault, replaceOnly);
        });
        if (shareDefault) {
          // After the registration above is queued, so any write that sees
          // the mark is ordered after it on the CS thread.
          dxvk::GetCommonTexture(prepared->texture.ptr())->MarkRemixShared();
        }
        if (prepared->texture != nullptr) {
          dxvk::GetCommonTexture(prepared->texture.ptr())->MarkRemixImported();
        }
        *out_handle = reinterpret_cast<remixapi_TextureHandle>(hash);
        return REMIXAPI_ERROR_CODE_SUCCESS;
      }

      // Otherwise Remix takes its own GPU copy rather than sampling the game's
      // image. A host may recycle a texture object for different content
      // (FO4's streamer reuses DEFAULT textures), and every material still
      // naming the old import would then show the new pixels -- normal maps
      // turning up as albedo. The copy is GPU to GPU on the shared queue: no
      // readback, no decode, and the game image is released as soon as the
      // token is.
      //
      // rtx.vramBudget: under device-local pressure a new copy keeps fewer top
      // mips. Shared imports above cost Remix nothing and are left alone.
      uint32_t pressureSkippedMips = 0;
      if (const uint32_t pressureCap = dxvk::vram_budget::importDimensionCap()) {
        while (firstMip + 1 < srcView->info().numLevels) {
          const VkExtent3D extent = srcImage->mipLevelExtent(srcView->info().minLevel + firstMip);
          if (std::max(extent.width, extent.height) <= pressureCap) {
            break;
          }
          ++firstMip;
          ++pressureSkippedMips;
        }
      }
      dxvk::DxvkImageCreateInfo copyInfo = {};
      copyInfo.type        = VK_IMAGE_TYPE_2D;
      copyInfo.format      = srcView->info().format;
      copyInfo.flags       = 0;
      copyInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      copyInfo.extent      = srcImage->mipLevelExtent(srcView->info().minLevel + firstMip);
      copyInfo.numLayers   = 1;
      copyInfo.mipLevels   = srcView->info().numLevels - firstMip;
      copyInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      copyInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                           | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      copyInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      copyInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
      copyInfo.layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      auto dxvkDevice = remixDevice->GetDXVKDevice();
      auto copyImage = dxvkDevice->createImage(copyInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        dxvk::DxvkMemoryStats::Category::RTXMaterialTexture, "Remix imported D3D11 texture");
      if (copyImage == nullptr) {
        return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
      }
      dxvk::DxvkImageViewCreateInfo copyViewInfo = srcView->info();
      copyViewInfo.minLevel = 0;
      copyViewInfo.minLayer = 0;
      copyViewInfo.numLevels = copyInfo.mipLevels;
      auto copyView = dxvkDevice->createImageView(copyImage, copyViewInfo);
      s_d3d11ImportStats.note(false, copyImage->memSize(),
        firstMip > 0 && srcImage->memSize() > copyImage->memSize() ? srcImage->memSize() - copyImage->memSize() : 0);
      if (pressureSkippedMips > 0) {
        // Each skipped top mip would have made the chain ~4x larger.
        dxvk::vram_budget::noteImportCapped(copyImage->memSize() * ((uint64_t(1) << (2 * std::min(pressureSkippedMips, 8u))) - 1));
      }

      auto readback = recordCopy ? dxvk::ApiRecorder::reserveTextureReadback() : dxvk::ApiRecorder::TextureReadback {};
      std::lock_guard lock { s_mutex };
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([srcView, copyView, hash, hostEncoding, firstMip, replaceOnly, readback = std::move(readback),
                           flushHost = prepared->initFlushDeferred ? remixDevice->d3d11() : nullptr](dxvk::DxvkContext* ctx) {
        if (flushHost) {
          // Prepare deferred this: queue the game's initial upload ahead of the copy.
          flushHost->FlushRemixInitUploads();
        }
        const auto& src = srcView->image();
        const auto& dst = copyView->image();
        for (uint32_t mip = 0; mip < dst->info().mipLevels; ++mip) {
          const VkImageSubresourceLayers dstLayers = { VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1 };
          const VkImageSubresourceLayers srcLayers = {
            VK_IMAGE_ASPECT_COLOR_BIT, srcView->info().minLevel + firstMip + mip, srcView->info().minLayer, 1 };
          ctx->copyImage(dst, dstLayers, VkOffset3D { 0, 0, 0 },
                         src, srcLayers, VkOffset3D { 0, 0, 0 }, dst->mipLevelExtent(mip));
        }
        // copyImage recorded the transfer and its barriers on this context;
        // DXVK orders the copy before any later read of dst.
        registerImportedView(ctx, copyView, hash, hostEncoding, false, replaceOnly);
        dxvk::ApiRecorder::readbackTexture(ctx, dst, readback);
      });
      *out_handle = reinterpret_cast<remixapi_TextureHandle>(hash);
      return REMIXAPI_ERROR_CODE_SUCCESS;
    } catch (const std::exception& e) {
      dxvk::Logger::err(dxvk::str::format("ImportD3D11Texture: ", e.what()));
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ImportD3D11TextureEx(
      remixapi_D3D11TextureToken token, uint64_t hash, uint32_t encoding, remixapi_TextureHandle* out_handle) {
    REMIX_FORK_RECORD(ImportD3D11TextureEx, token, hash, encoding, out_handle);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::ImportD3D11Texture, 1 }; // rtx.profile.detailedLogging
    return importD3D11Texture(token, hash, encoding, false, out_handle);
  }

  // Same rules as ImportD3D11TextureEx (share / copy-on-write / copy, mip
  // cap, barriers); the result is applied to the existing entry in place.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ReplaceD3D11TextureEx(
      remixapi_D3D11TextureToken token, remixapi_TextureHandle handle, uint32_t encoding) {
    REMIX_FORK_RECORD(ReplaceD3D11TextureEx, token, handle, encoding);
    const dxvk::perf::ApiScope perfApiScope { dxvk::perf::Api::ReplaceD3D11Texture, 1 }; // rtx.profile.detailedLogging
    if (!dxvk::RtxOptions::HostImport::enableInPlaceTextureReplace()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (!handle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    remixapi_TextureHandle out = nullptr;
    return importD3D11Texture(token, reinterpret_cast<uint64_t>(handle), encoding, true, &out);
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ImportD3D11Texture(
      remixapi_D3D11TextureToken token, uint64_t hash, remixapi_TextureHandle* out_handle) {
    REMIX_FORK_RECORD(ImportD3D11Texture, token, hash, out_handle);
    return remixapi_ImportD3D11TextureEx(token, hash, REMIXAPI_TEXTURE_ENCODING_NONE, out_handle);
  }

  REMIXAPI void REMIXAPI_CALL remixapi_ReleaseD3D11Texture(remixapi_D3D11TextureToken token) {
    delete static_cast<PreparedD3D11Texture*>(token);
  }

  // D3D9 import (see remix_c.h). The D3D9 counterpart of a shared D3D11 import:
  // the TextureRef holds a view of the game's own DxvkImage, registered through
  // registerImportedView like a D3D11 import, plus the hash -> texture map
  // createTexture fills, so "0x<hash>" material paths resolve it the same way
  // after RtxTextureManager::clear(). There is no copy-on-write on this path:
  // the host imports only content it will not rewrite (FNV: MANAGED textures
  // loaded from DDS files). The registered device is the one the host draws
  // with (through the bridge: NvRemixBridge.exe's device), so the game's
  // upload and the registration go down the same command stream in order.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_ImportD3D9Texture(
      IDirect3DBaseTexture9* texture, uint64_t hash, uint32_t flags, uint32_t encoding,
      remixapi_TextureHandle* out_handle) {
    const bool probe = texture == nullptr && out_handle == nullptr;
    if (out_handle) {
      *out_handle = nullptr;
    }
    if (!dxvk::RtxOptions::HostImport::enableD3D9TextureImport()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    auto remixDevice = tryAsDxvk();
    if (!remixDevice || !remixDevice->d3d9()) {
      return REMIXAPI_ERROR_CODE_REMIX_DEVICE_WAS_NOT_REGISTERED;
    }
    if (probe) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    constexpr uint32_t kKnownFlags = REMIXAPI_D3D9_TEXTURE_IMPORT_SRGB | REMIXAPI_D3D9_TEXTURE_IMPORT_OPAQUE_ALPHA;
    if (!texture || !out_handle || !hash || (flags & ~kKnownFlags) != 0 ||
        encoding > REMIXAPI_TEXTURE_ENCODING_SMOOTHNESS_IN_GREEN) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const auto hostEncoding = static_cast<dxvk::HostTextureEncoding>(encoding);
    try {
      if (texture->GetType() != D3DRTYPE_TEXTURE) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      auto* texture2D = dynamic_cast<dxvk::D3D9Texture2D*>(static_cast<IDirect3DTexture9*>(texture));
      if (!texture2D) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::D3D9CommonTexture* common = texture2D->GetCommonTexture();
      const auto* desc = common->Desc();
      // Same device (a second device's image lives on another command stream),
      // an image to sample (SYSTEMMEM / SCRATCH have none), content the host
      // fills from the CPU or with UpdateTexture: never written by the GPU as a
      // render target, never CPU-mapped every frame, no mips generated at draw.
      if (common->Device() != remixDevice->d3d9() ||
          !(common->IsManaged() || desc->Pool == D3DPOOL_DEFAULT) ||
          common->IsRenderTarget() || common->IsDepthStencil() || common->IsDynamic() ||
          common->IsAutomaticMip() || desc->ArraySize != 1 || desc->MultiSample != D3DMULTISAMPLE_NONE) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      const bool srgb = (flags & REMIXAPI_D3D9_TEXTURE_IMPORT_SRGB) != 0;
      const dxvk::Rc<dxvk::DxvkImage> image = common->GetImage();
      if (image == nullptr || (srgb && !common->IsSrgbCompatible()) ||
          image->info().layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        // A LINEAR-tiled or storage-capable (format-converted) image stays in
        // GENERAL layout; Remix's bindless table samples SHADER_READ_ONLY.
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      dxvk::Rc<dxvk::DxvkImageView> view = common->GetSampleView(srgb);
      if (view == nullptr) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }

      // D3D9 uploads a MANAGED texture's CPU copy lazily, at the first draw
      // that binds it. A texture only ever drawn through the API may never be
      // bound, so submit its pending uploads now; they are queued on the device
      // command stream ahead of the registration below. Takes the device lock.
      if (common->IsManaged()) {
        common->PreLoadAll();
      }

      // Mips larger than rtx.hostImport.maxCopiedTextureDimension are left out
      // of the view, exactly as for a shared D3D11 import.
      uint32_t firstMip = 0;
      if (const int optionCap = dxvk::RtxOptions::HostImport::maxCopiedTextureDimension(); optionCap > 0) {
        while (firstMip + 1 < view->info().numLevels) {
          const VkExtent3D extent = image->mipLevelExtent(view->info().minLevel + firstMip);
          if (std::max(extent.width, extent.height) <= uint32_t(optionCap)) {
            break;
          }
          ++firstMip;
        }
      }
      const bool opaqueAlpha = (flags & REMIXAPI_D3D9_TEXTURE_IMPORT_OPAQUE_ALPHA) != 0;
      if (firstMip > 0 || opaqueAlpha) {
        dxvk::DxvkImageViewCreateInfo viewInfo = view->info();
        viewInfo.minLevel += firstMip;
        viewInfo.numLevels -= firstMip;
        if (opaqueAlpha) {
          viewInfo.swizzle.a = VK_COMPONENT_SWIZZLE_ONE;
        }
        view = remixDevice->GetDXVKDevice()->createImageView(image, viewInfo);
        if (view == nullptr) {
          return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
        }
      }

      static std::atomic<uint64_t> s_imports { 0 };
      static std::atomic<uint64_t> s_importedBytes { 0 };
      s_importedBytes += image->memSize();
      if ((++s_imports % 256) == 0) {
        dxvk::Logger::info(dxvk::str::format("[HostImport] ", s_imports.load(), " D3D9 imports shared in place (",
                                             s_importedBytes.load() >> 20, " MiB not duplicated)"));
      }

      std::lock_guard lock { s_mutex };
      auto devLock = remixDevice->LockDevice();
      remixDevice->EmitCs([view = std::move(view), hash, hostEncoding](dxvk::DxvkContext* ctx) {
        // The image's own stages do not name ray tracing; the next frame opens
        // with one memory barrier covering every image imported since the last.
        view->image()->addShaderReadStages(
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR);
        static_cast<dxvk::RtxContext*>(ctx)->requestImportedImageBarrier();
        registerImportedView(ctx, view, hash, hostEncoding, false, false);
        dxvk::fork_hooks::registerApiTexture(hash, dxvk::TextureRef(view, hash, hash, hostEncoding));
      });
      *out_handle = reinterpret_cast<remixapi_TextureHandle>(hash);
      return REMIXAPI_ERROR_CODE_SUCCESS;
    } catch (const dxvk::DxvkError& e) {
      dxvk::Logger::err(dxvk::str::format("ImportD3D9Texture: ", e.message()));
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    } catch (const std::exception& e) {
      dxvk::Logger::err(dxvk::str::format("ImportD3D9Texture: ", e.what()));
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_RegisterHostSetting(
      const remixapi_HostSettingInfo* info, remixapi_HostSettingHandle* out_handle) {
    REMIX_FORK_RECORD(RegisterHostSetting, info, out_handle);
    if (out_handle) {
      *out_handle = 0;
    }
    if (!info || !out_handle || info->structSize < sizeof(remixapi_HostSettingInfo)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const uint32_t handle = dxvk::host_settings::registerSetting(*info);
    if (handle == 0) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI uint64_t REMIXAPI_CALL remixapi_GetHostSettingsGeneration(void) {
    return dxvk::host_settings::generation();
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_GetHostSettingValue(
      remixapi_HostSettingHandle handle, double* out_value) {
    if (!out_value) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return dxvk::host_settings::getValue(handle, *out_value)
      ? REMIXAPI_ERROR_CODE_SUCCESS : REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_SetHostSettingValue(
      remixapi_HostSettingHandle handle, double value) {
    REMIX_FORK_RECORD(SetHostSettingValue, handle, value);
    return dxvk::host_settings::setValue(handle, value)
      ? REMIXAPI_ERROR_CODE_SUCCESS : REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  REMIXAPI void REMIXAPI_CALL remixapi_ClearHostSettings(void) {
    REMIX_FORK_RECORD0(ClearHostSettings);
    dxvk::host_settings::clear();
  }

  REMIXAPI uint32_t REMIXAPI_CALL remixapi_GetSpecGlossMaterialVersion(void) {
    return 1;
  }

  REMIXAPI uint32_t REMIXAPI_CALL remixapi_GetMaterialAlbedoTintVersion(void) {
    return 1;
  }

  // remixapi_InstanceInfoMaterialOverrideEXT revision. The override itself is always
  // honoured; rtx.hostImport.shareGeometryAcrossMaterials only controls geometry sharing.
  REMIXAPI uint32_t REMIXAPI_CALL remixapi_GetInstanceMaterialOverrideVersion(void) {
    return 1;
  }

  // rtx.vramBudget state for the host. Lock-free; any thread.
  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_GetMemoryPressure(remixapi_MemoryPressureInfo* out_info) {
    REMIX_FORK_RECORD(GetMemoryPressure, out_info);
    if (!out_info || out_info->structSize < sizeof(remixapi_MemoryPressureInfo)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const dxvk::vram_budget::HostInfo info = dxvk::vram_budget::hostInfo();
    out_info->structSize         = sizeof(remixapi_MemoryPressureInfo);
    out_info->level              = info.level;
    out_info->requestHostRelease = info.requestHostRelease;
    out_info->usageBytes         = info.usageBytes;
    out_info->budgetBytes        = info.budgetBytes;
    out_info->bytesToRelease     = info.bytesToRelease;
    out_info->frame              = info.frame;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_InitializeLibrary(const remixapi_InitializeLibraryInfo* info,
                                                                       remixapi_Interface* out_result) {
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!out_result) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!isVersionCompatible(info->version)) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    s_apiVersion = info->version;
    dxvk::ApiRecorder::onInitializeLibrary(info->version);

    auto interf = remixapi_Interface {};
    {
      // Upstream vtable slots
      interf.Startup = remixapi_Startup;
      interf.Shutdown = remixapi_Shutdown;
      interf.Present = remixapi_Present;
      interf.CreateMaterial = remixapi_CreateMaterial;
      interf.DestroyMaterial = remixapi_DestroyMaterial;
      interf.CreateMesh = remixapi_CreateMesh;
      interf.DestroyMesh = remixapi_DestroyMesh;
      interf.SetupCamera = remixapi_SetupCamera;
      interf.SetCameraMediumMaterial = remixapi_SetCameraMediumMaterial;
      interf.DrawInstance = remixapi_DrawInstance;
      interf.CreateLight = remixapi_CreateLight;
      interf.DestroyLight = remixapi_DestroyLight;
      interf.DrawLightInstance = remixapi_DrawLightInstance;
      interf.SetConfigVariable = remixapi_SetConfigVariable;
      interf.dxvk_CreateD3D9 = remixapi_dxvk_CreateD3D9_legacy;
      interf.dxvk_RegisterD3D9Device = remixapi_dxvk_RegisterD3D9Device;
      interf.dxvk_GetExternalSwapchain = remixapi_dxvk_GetExternalSwapchain;
      interf.dxvk_GetVkImage = remixapi_dxvk_GetVkImage;
      interf.dxvk_CopyRenderingOutput = remixapi_dxvk_CopyRenderingOutput;
      interf.dxvk_SetDefaultOutput = remixapi_dxvk_SetDefaultOutput;
      interf.pick_RequestObjectPicking = remixapi_pick_RequestObjectPicking;
      interf.pick_HighlightObjects = remixapi_pick_HighlightObjects;
      // Vtable slots whose functions are in the anonymous namespace, so they are assigned here where they are visible.
      interf.CreateMeshBatched = remixapi_CreateMeshBatched;
      interf.AddTextureHash = remixapi_AddTextureHash;
      interf.RemoveTextureHash = remixapi_RemoveTextureHash;
      interf.CreateTexture = remixapi_CreateTexture;
      interf.DestroyTexture = remixapi_DestroyTexture;
      interf.dxvk_GetTextureHash = remixapi_dxvk_GetTextureHash;
      interf.dxvk_GetSharedD3D11TextureHandle = remixapi_dxvk_GetSharedD3D11TextureHandle;
      interf.GetUIState = remixapi_GetUIState;
      interf.SetUIState = remixapi_SetUIState;
      interf.DrawScreenOverlay = remixapi_DrawScreenOverlay;
      interf.CreateLightBatched = remixapi_CreateLightBatched;
      interf.SetGameValue = remixapi_SetGameValue;
      interf.RequestVramCompaction = remixapi_RequestVramCompaction;
      interf.GetVramStats = remixapi_GetVramStats;
      interf.RequestTextureVramFree = remixapi_RequestTextureVramFree;
      interf.GetGameValue = remixapi_GetGameValue;
      // The extern "C" vtable slots.
      dxvk::fork_hooks::remixApiVtableInit(interf);
    }
    static_assert(sizeof(interf) == 328, "Add/remove function registration");

    *out_result = interf;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

}
