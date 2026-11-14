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
#include <limits>
#include <mutex>
#include <vector>
#include <ppl.h>
#include <intrin.h>
#include <atomic>

#include "rtx_host_calibration.h"
#include "rtx_asset_replacer.h"
#include "rtx_fork_hooks.h"
#include "rtx_fork_material_census.h"
#include "rtx_fork_core_material.h"
#include "rtx_fork_d3d9_identity.h"
#include "rtx_fork_d3d9_identity_trace.h"
#include "rtx_scene_manager.h"
#include "rtx_retained_v2.h"
#include "rtx_retained_scene.h"
#include "rtx_scene_transient.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_vram_budget.h"
#include "dxvk_device.h"
#include "dxvk_context.h"
#include "dxvk_buffer.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_preserved_object_picking.h"
#include "rtx_terrain_baker.h"
#include "rtx_texture_manager.h"
#include "rtx_texture.h"
#include "rtx_xess.h"

#include <assert.h>

#include "../d3d9/d3d9_state.h"
#include "vulkan/vulkan_core.h"

#include "rtx_game_capturer.h"
#include "rtx_matrix_helpers.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_lights_data.h"
#include "rtx_light_utils.h"
#include "rtx_fork_camera_origin.h" // rtx.worldOrigin.enable
#include "rtx_fork_sccr_shell.h" // SCCR shell tracing

#include "../util/util_global_time.h"
#include "../util/util_struct_hash.h"

#include "rtx/pass/particles/particle_system_common.h"

namespace {
  // helper function to ensure generating spatialMapHash for external draws is done the same way in multiple places.
  XXH64_hash_t spatialMapHashForExternalDrawMesh(remixapi_MeshHandle mesh) {
    const uintptr_t meshId = reinterpret_cast<uintptr_t>(mesh);
    return XXH3_64bits(&meshId, sizeof(meshId));
  }
  // rtx.scene.directRetained: retained draws on the legacy lane keep their replacement instances in their own spatial
  // map, so the tracker's nearest-neighbour match never trades them with per-frame draws of the same mesh.
  constexpr XXH64_hash_t kRetainedSpatialMapSalt = 0x52455441494E4544ull;
  XXH64_hash_t retainedSpatialMapHashForExternalDrawMesh(remixapi_MeshHandle mesh) {
    return spatialMapHashForExternalDrawMesh(mesh) ^ kRetainedSpatialMapSalt;
  }
} // namespace

namespace dxvk {

  // Compute a hash that can be used to check if an external draw is identical to the previous frame's draw.
  XXH64_hash_t ExternalDrawState::computeExternalDrawIdentityHash() const {
    struct ExternalDrawIdentityHashData {
      uintptr_t meshId;
      XXH64_hash_t materialHash;
      XXH64_hash_t boneHash;
      CameraType::Enum cameraType;
      uint32_t categoriesRaw;
      XXH64_hash_t particleDescHash;
      XXH64_hash_t gpuInstancingHash;
      TexGenMode texgenMode;
      uint8_t usesVertexShader;
      uint8_t usesPixelShader;
      uint8_t zWriteEnable;
      uint8_t zEnable;
      uint8_t skyAutoDetected;
      uint8_t _pad0;
      uint8_t _pad1;
      Matrix4 objectToWorld;
      Matrix4 textureTransform;
    };

    ExternalDrawIdentityHashData data{};

    const DrawCallTransforms& transforms = drawCall.getTransformData();
    data.meshId = reinterpret_cast<uintptr_t>(mesh);
    data.materialHash = drawCall.getMaterialData().getHash();
    if (materialOverride != nullptr) {
      // A per-instance material override is part of the draw's identity, so a
      // per-frame DrawInstance that swaps it is a different draw.
      const uintptr_t overrideId = reinterpret_cast<uintptr_t>(materialOverride);
      data.materialHash ^= XXH3_64bits(&overrideId, sizeof(overrideId));
    }
    data.boneHash = drawCall.getSkinningState().boneHash;
    data.cameraType = cameraType;
    data.categoriesRaw = categories.raw();

    if (optionalParticleDesc.has_value()) {
      data.particleDescHash = optionalParticleDesc->calcHash();
    }

    if (!gpuInstancingTransforms.empty()) {
      data.gpuInstancingHash = XXH3_64bits(
          gpuInstancingTransforms.data(),
          gpuInstancingTransforms.size() * sizeof(Matrix4));
    }

    data.texgenMode = transforms.texgenMode;
    data.usesVertexShader = drawCall.usesVertexShader ? 1u : 0u;
    data.usesPixelShader = drawCall.usesPixelShader ? 1u : 0u;
    data.zWriteEnable = drawCall.zWriteEnable ? 1u : 0u;
    data.zEnable = drawCall.zEnable ? 1u : 0u;
    data.skyAutoDetected = drawCall.skyAutoDetected ? 1u : 0u;
    data.objectToWorld = transforms.objectToWorld;
    data.textureTransform = transforms.textureTransform;

    if (retainedHandle != 0) {
      // Stable identity for the draw's whole registered lifetime; placement
      // changes arrive as transform drift on the same instance.
      data.gpuInstancingHash = XXH3_64bits(&retainedHandle, sizeof(retainedHandle));
      data.objectToWorld = Matrix4 {};
    }

    return hashStructByMemory<ExternalDrawIdentityHashData,
        &ExternalDrawIdentityHashData::meshId,
        &ExternalDrawIdentityHashData::materialHash,
        &ExternalDrawIdentityHashData::boneHash,
        &ExternalDrawIdentityHashData::cameraType,
        &ExternalDrawIdentityHashData::categoriesRaw,
        &ExternalDrawIdentityHashData::particleDescHash,
        &ExternalDrawIdentityHashData::gpuInstancingHash,
        &ExternalDrawIdentityHashData::texgenMode,
        &ExternalDrawIdentityHashData::usesVertexShader,
        &ExternalDrawIdentityHashData::usesPixelShader,
        &ExternalDrawIdentityHashData::zWriteEnable,
        &ExternalDrawIdentityHashData::zEnable,
        &ExternalDrawIdentityHashData::skyAutoDetected,
        &ExternalDrawIdentityHashData::_pad0,
        &ExternalDrawIdentityHashData::_pad1,
        &ExternalDrawIdentityHashData::objectToWorld,
        &ExternalDrawIdentityHashData::textureTransform>(data);
  }

  SceneManager::SceneManager(DxvkDevice* device)
    : CommonDeviceObject(device)
    , m_instanceManager(device, this)
    , m_accelManager(device)
    , m_lightManager(device)
    , m_graphManager()
    , m_rayPortalManager(device, this)
    , m_lensManager(device)
    , m_drawCallCache(device)
    , m_drawCallTracker(device)
    , m_bindlessResourceManager(device)
    , m_pReplacer(new AssetReplacer())
    , m_terrainBaker(new TerrainBaker())
    , m_cameraManager(device)
    , m_uniqueObjectSearchDistance(RtxOptions::uniqueObjectDistance()) {
    InstanceEventHandler instanceEvents(this);
    instanceEvents.onInstanceAddedCallback = [this](RtInstance& instance) { onInstanceAdded(instance); };
    instanceEvents.onInstanceUpdatedCallback = [this](RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, bool hasTransformChanged, bool hasVerticesChanged, bool isFirstUpdateThisFrame) { onInstanceUpdated(instance, drawCall, material, hasTransformChanged, hasVerticesChanged, isFirstUpdateThisFrame); };
    instanceEvents.onInstanceDestroyedCallback = [this](RtInstance& instance) { onInstanceDestroyed(instance); };
    m_instanceManager.addEventHandler(instanceEvents);

    // rtx.retained.useV2: the second retained-instance implementation hears synchronously about every
    // instance and replacement instance that dies, so the pointers it holds never dangle.
    m_retainedV2 = std::make_unique<RetainedInstanceStore>(*this);
    m_retainedV2Active = RtxOptions::Retained::useV2();
    // rtx.scene.directRetained: the scene's own retained instances, told the same way.
    m_retainedScene = std::make_unique<RetainedScene>(*this);
    m_instanceManager.setInstanceFreedHook([this](RtInstance* instance) {
      m_accelManager.onInstanceFreed(instance);
      m_retainedV2->onInstanceFreed(*instance);
      m_retainedScene->onInstanceFreed(*instance);
      std::lock_guard lock { m_lensAnchorMutex };
      m_lensAnchors.erase(instance);
    });
    m_drawCallTracker.setRetainedV2Listener([](void* owner, uint64_t ticket) {
      static_cast<RetainedInstanceStore*>(owner)->onReplacementInstanceGone(ticket);
    }, m_retainedV2.get());
    
    if (env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME") != "") {
      m_beginUsdExportFrameNum = stoul(env::getEnvVar("DXVK_RTX_CAPTURE_ENABLE_ON_FRAME"));
    }
  }

  SceneManager::~SceneManager() {
    // Members are destroyed after this body; the accel manager may go before the instance manager.
    m_instanceManager.setInstanceFreedHook(nullptr);
    m_drawCallTracker.setRetainedV2Listener(nullptr, nullptr);
  }

  uint32_t SceneManager::getSoftwareOmmTableIndex() const {
    return (m_opacityMicromapManager && m_opacityMicromapManager->isSoftware())
      ? m_opacityMicromapManager->getSoftwareTableIndex() : UINT32_MAX;
  }

  uint32_t SceneManager::getCoreMaterialTableIndex() const {
    return m_coreMaterials ? m_coreMaterials->getTableIndex() : UINT32_MAX;
  }

  uint32_t SceneManager::getCoreMaterialExtTableIndex() const {
    return m_coreMaterials ? m_coreMaterials->getExtensionTableIndex() : UINT32_MAX;
  }

  bool SceneManager::getCoreMaterialClassCounts(uint32_t (&counts)[4]) const {
    return m_coreMaterials ? m_coreMaterials->getClassCounts(counts) : false;
  }

  uint32_t SceneManager::getLegacySurfaceTableIndex() const {
    return m_coreMaterials ? m_coreMaterials->getLegacyTableIndex() : UINT32_MAX;
  }

  bool SceneManager::getCoreMaterialLegacyBits(bool& anyLegacy, bool& anyTextureOps, bool& anyEye) const {
    return m_coreMaterials ? m_coreMaterials->getLegacyBits(anyLegacy, anyTextureOps, anyEye) : false;
  }

  bool SceneManager::hasAnyMods() const {
    return m_pReplacer->hasAnyMods();
  }

  std::vector<Mod::State> SceneManager::getReplacementStates() const {
    return m_pReplacer->getReplacementStates();
  }

  void SceneManager::initialize(Rc<DxvkContext> ctx) {
    ScopedCpuProfileZone();
    m_pReplacer->initialize(ctx);
  }

  void SceneManager::logStatistics() {
    if (m_opacityMicromapManager.get()) {
      m_opacityMicromapManager->logStatistics();
    }
  }

  Vector3 SceneManager::getSceneUp() {
    return RtxOptions::zUp() ? Vector3(0.f, 0.f, 1.f) : Vector3(0.f, 1.f, 0.f);
  }

  Vector3 SceneManager::getSceneForward() {
    return RtxOptions::zUp() ? Vector3(0.f, 1.f, 0.f) : Vector3(0.f, 0.f, 1.f);
  }

  Vector3 SceneManager::calculateSceneRight() {
    const Vector3 up = SceneManager::getSceneUp();
    const Vector3 forward = SceneManager::getSceneForward();
    return RtxOptions::leftHandedCoordinateSystem() ? cross(up, forward) : cross(forward, up);
  }

  Vector3 SceneManager::worldToSceneOrientedVector(const Vector3& worldVector) {
    return RtxOptions::zUp() ? worldVector : Vector3(worldVector.x, worldVector.z, worldVector.y);
  }

  Vector3 SceneManager::sceneToWorldOrientedVector(const Vector3& sceneVector) {
    // Same transform applies to and from
    return worldToSceneOrientedVector(sceneVector);
  }

  float SceneManager::getTotalMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
  
    const bool temporalUpscaling = RtxOptions::isDLSSOrRayReconstructionEnabled() || RtxOptions::isXeSSEnabled() || RtxOptions::isFSREnabled() || RtxOptions::isTAAEnabled();
    
    float totalUpscaleMipBias = 0.0f;

    // log2(render width / output width), the bias the DLSS, FSR and XeSS guides give, plus the user's offset.
    // The ratio is below 1, so the bias is negative and selects sharper mips.
    if (temporalUpscaling) {
      totalUpscaleMipBias = log2(resourceManager.getUpscaleRatio()) + RtxOptions::upscalingMipBias();
    }

    return totalUpscaleMipBias + RtxOptions::nativeMipBias();
  }

  float SceneManager::getCalculatedUpscalingMipBias() {
    auto& resourceManager = m_device->getCommon()->getResources();
    
    const bool temporalUpscaling = RtxOptions::isXeSSEnabled() || RtxOptions::isFSREnabled();
    if (!temporalUpscaling) {
      return 0.0f;
    }
    
    float calculatedUpscalingBias = -log2(resourceManager.getUpscaleRatio());
    return calculatedUpscalingBias;
  }

  void SceneManager::clear(Rc<DxvkContext> ctx, bool needWfi) {
    ScopedCpuProfileZone();

    ++m_clearEpoch;
    // rtx.scene.directRetained: releases its geometry's buffer slots before the check below.
    m_retainedScene->onSceneClear();
    // rtx.coreMaterial.build releases its table slot too.
    if (m_coreMaterials) {
      m_coreMaterials->releaseAll(*this);
    }
    // rtx.alphaBake.softwareOmm stores and table too.
    if (m_opacityMicromapManager && m_opacityMicromapManager->isSoftware()) {
      m_opacityMicromapManager->releaseSoftwareSlots(*this);
    }

    auto& textureManager = m_device->getCommon()->getTextureManager();

    // Only clear once after the scene disappears, to avoid adding a WFI on every frame through clear().
    if (needWfi) {
      if (ctx.ptr())
        ctx->flushCommandList();
      m_device->waitForIdle();
    }

    // The SCCR shell cache holds buffer slots of its own. Release them before the check below.
    m_device->getCommon()->metaSccrShell().onSceneClear(*this);
    m_device->getCommon()->metaSccrTess().onSceneClear(*this);

    // unregister all geometry buffers and verify that nothing has leaked.
    // (this function could be debug only, but this isn't a hot path.) 
    verifyAndReleaseBufferCache();

    // We still need to clear caches even if the scene wasn't rendered
    m_bufferCache.clear();
    m_preCreationSurfaceMaterialMap.clear();
    m_volumeMaterialCache.clear();

    // Clear ReplacementInstances first: their destructors call clear() which
    // accesses prims[] to mark entities for GC and clear back-pointers.
    // Entities must still be alive at this point.
    D3d9IdentityTrace::recordClear(m_device->getCurrentFrameId(), D3d9Identity::pendingClearReason());
    D3d9Identity::onSceneClear(m_device->getCurrentFrameId(), m_drawCallTracker.getReplacementInstances().size());
    m_drawCallTracker.clear();
    // Retained pins point at the replacement instances just destroyed (the tracker epoch bump
    // above would drop them at the next replay too).
    unpinAllRetained();
    // rtx.retained.useV2: same for the second implementation's held and resident entries.
    m_retainedV2->onSceneClear();

    // Called before instance manager's clear, so that it resets all tracked instances in Opacity Micromap manager at once
    if (m_opacityMicromapManager.get())
      m_opacityMicromapManager->clear();

    // Invalidate AccelManager's bucket cache before InstanceManager::clear() deletes
    // every RtInstance. The cache holds raw RtInstance* in m_cachedBuckets[].instances /
    // .surfaces and m_instanceBucketIndex, and the next frame's mergeInstancesIntoBlas
    // dirty check would dereference those (now-freed) pointers. The per-instance
    // onInstanceDestroyed -> removeInstanceFromBucketCache hook only patches the index
    // map, not the vectors, so a bulk reset must drop the cache wholesale.
    m_accelManager.clear();

    // Instance destruction fires onInstanceDestroyed -> releaseSurfaceMaterial for every
    // game-submitted instance, decrementing texture ref counts and feature counts.
    // The surface material cache must still be valid here so those releases are not
    // silently skipped by the bounds check in releaseSurfaceMaterial.
    m_instanceManager.clear();

    // After all instances are destroyed their retain/release pairs must be balanced.
    // If any count is non-zero here there is a retain/release mismatch.
    if (m_activePOMCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: POM count is ", m_activePOMCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: POM count non-zero after instance destruction");
      m_activePOMCount = 0;
    }
    if (m_sssCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: SSS count is ", m_sssCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: SSS count non-zero after instance destruction");
      m_sssCount = 0;
    }
    if (m_thinOpaqueCount != 0) {
      Logger::err(str::format("[RTX] SceneManager::clear: thin-opaque count is ", m_thinOpaqueCount, " after instance destruction (retain/release mismatch)"));
      assert(false && "SceneManager::clear: thin-opaque count non-zero after instance destruction");
      m_thinOpaqueCount = 0;
    }
    // The live material counts read by the frame feature mask.
    {
      struct { uint32_t& count; const char* name; } liveCounts[] = {
        { m_thinFilmCount, "thin-film" },
        { m_rayPortalCount, "ray-portal" },
        { m_translucentCount, "translucent" },
        { m_raytracedRenderTargetCount, "raytraced-render-target" },
      };
      for (auto& entry : liveCounts) {
        if (entry.count != 0) {
          Logger::err(str::format("[RTX] SceneManager::clear: ", entry.name, " count is ", entry.count, " after instance destruction (retain/release mismatch)"));
          entry.count = 0;
        }
      }
    }
    textureManager.assertAllRefCountsZero();

    // Now safe to clear material caches; all ref counts have been released.
    m_surfaceMaterialCache.clear();
    m_surfaceMaterialExtensionCache.clear();
    m_lightManager.clear();
    m_graphManager.clear();
    m_rayPortalManager.clear();
    m_drawCallCache.clear();
    textureManager.clear();
    m_textureCacheGenerationValidForPreserve = textureManager.getTextureCacheGeneration();

    m_previousFrameSceneAvailable = false;
    m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    m_fogStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_externalStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
    m_lastResolvedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
    m_lastUploadedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
  }

  void SceneManager::invalidateChangedReplacements(const AssetChanges& changes) {
    ScopedCpuProfileZone();
    if (changes.empty()) {
      return;
    }

    for (const auto& pReplacementInstance : m_drawCallTracker.getReplacementInstances()) {
      ReplacementInstance* pInstance = pReplacementInstance.get();
      if (pInstance == nullptr) {
        continue;
      }
      const auto& bucket = pInstance->activeReplacements;
      if (bucket && bucket->stale.load(std::memory_order_acquire)) {
        pInstance->clear();
        continue;
      }
      if (!changes.dirtyMatHashes.empty() &&
          changes.dirtyMatHashes.count(pInstance->materialHash) != 0) {
        pInstance->clear();
      }
    }
  }

  void SceneManager::garbageCollection() {
    ScopedCpuProfileZone();
    // rtx.parallelScenePrepScans: each collection below first finds what it acts on with a scan of
    // every entry on worker threads, then acts on those alone, in its original order.
    const bool parallelScans = RtxOptions::parallelScenePrepScans();
    auto gcStart = std::chrono::steady_clock::now();
    const auto gcPhase = [&](uint32_t nestedIndex) {
      const auto now = std::chrono::steady_clock::now();
      addPrepNestedMs(nestedIndex, std::chrono::duration<double, std::milli>(now - gcStart).count());
      gcStart = now;
    };

    // BlasEntry GC: remove entries not touched recently.
    // Only GC entries with no linked instances -- instances still reference the BlasEntry
    // for TLAS build, and destroying it would cause a one-frame visibility gap.
    if (m_device->getCurrentFrameId() > RtxOptions::numFramesToKeepGeometryData()) {
      const size_t oldestFrame = m_device->getCurrentFrameId() - RtxOptions::numFramesToKeepGeometryData();
      auto& entries = m_drawCallCache.getEntries();
      if (parallelScans && entries.size() >= 4096) {
        // The multimap is scanned bucket by bucket on worker threads (read only); the few entries to
        // collect are then erased here. Erasure order follows the buckets rather than the
        // container's iteration order, which only changes which freed buffer-table slots are
        // reused first.
        using Candidate = std::pair<XXH64_hash_t, const BlasEntry*>;
        const size_t bucketCount = entries.bucket_count();
        constexpr size_t kBucketsPerChunk = 2048;
        const uint32_t chunkCount = uint32_t((bucketCount + kBucketsPerChunk - 1) / kBucketsPerChunk);
        std::vector<std::vector<Candidate>> candidates(chunkCount);
        concurrency::parallel_for<uint32_t>(0, chunkCount, [&](uint32_t chunk) {
          const size_t end = std::min(bucketCount, size_t(chunk + 1) * kBucketsPerChunk);
          for (size_t bucket = size_t(chunk) * kBucketsPerChunk; bucket < end; ++bucket) {
            for (auto it = entries.cbegin(bucket); it != entries.cend(bucket); ++it) {
              if (it->second.frameLastTouched < oldestFrame && it->second.getLinkedInstances().empty()) {
                candidates[chunk].emplace_back(it->first, &it->second);
              }
            }
          }
        });
        for (const auto& chunkCandidates : candidates) {
          for (const Candidate& candidate : chunkCandidates) {
            auto range = entries.equal_range(candidate.first);
            for (auto it = range.first; it != range.second; ++it) {
              if (&it->second == candidate.second) {
                unregisterGeometryBuffers(it->second.modifiedGeometryData);
                entries.erase(it);
                break;
              }
            }
          }
        }
      } else {
        for (auto iter = entries.begin(); iter != entries.end(); ) {
          if (iter->second.frameLastTouched < oldestFrame &&
              iter->second.getLinkedInstances().empty()) {
            unregisterGeometryBuffers(iter->second.modifiedGeometryData);
            iter = entries.erase(iter);
          } else {
            ++iter;
          }
        }
      }
    }
    gcPhase(kPrepNestedGcGeometry);

    // ReplacementInstance GC: marks owned instances/lights for GC
    // and clears their back-pointers while they are still alive.
    D3d9IdentityTrace::recordCollect(m_device->getCurrentFrameId(), getCamera().isCameraCut(), m_isAntiCullingSupported);
    m_drawCallTracker.garbageCollectReplacementInstances(
        getCamera(), m_isAntiCullingSupported, parallelScans, m_lastKeptCutFrame);
    // rtx.scene.directTransient: the same lifetime rule for the scene's transient (DrawInstance) entries.
    m_retainedScene->transient().collect(getCamera(), m_isAntiCullingSupported);
    gcPhase(kPrepNestedGcReplacementInstances);

    // Instance/light GC: removes entities marked for GC by ReplacementInstance::clear()
    // or marked on creation (ephemeral copies). Back-pointers are already null.
    m_instanceManager.garbageCollection(parallelScans);
    m_accelManager.garbageCollection();
    m_lightManager.garbageCollection(getCamera());
    m_rayPortalManager.garbageCollection();
    gcPhase(kPrepNestedGcInstancesAndRest);
  }

  void SceneManager::onDestroy() {
    // First, because the mods' rebuild workers read through RtxInitializer and the
    // texture manager, and DxvkObjects destroys both before it destroys us. This is
    // the only point on the teardown path where everything they touch is still alive.
    m_pReplacer->onDestroy();

    m_accelManager.onDestroy();
    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onDestroy();
    }
  }

  template<bool isNew>
  SceneManager::ObjectCacheState SceneManager::processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    ScopedCpuProfileZone();
    ObjectCacheState result = ObjectCacheState::KBuildBVH;
    const RasterGeometry& input = drawCallState.getGeometryData();
    RaytraceGeometry& inOutGeometry = pBlas->modifiedGeometryData;

    // Need to call updateBufferCache() on *every* exit path.  This needs to run after changes
    // have been made, but before we actually exit the function.
    struct BufferCacheGuard {
      SceneManager& sceneManager;
      BlasEntry* pBlas;
      ~BufferCacheGuard() { sceneManager.updateBufferCache(pBlas); }
    } bufferCacheGuard { *this, pBlas };

    const bool texcoordsChanged = !isNew &&
      input.hashes[HashComponents::VertexTexcoord] != inOutGeometry.hashes[HashComponents::VertexTexcoord];

    // Determine the optimal object state for this geometry
    if (!isNew) {
      // This is a geometry we've seen before, that requires updating
      //  'inOutGeometry' has valid historical data
      if (input.hashes[HashComponents::Indices] == inOutGeometry.hashes[HashComponents::Indices]) {
        // Position changes require a BVH refit; UV-only changes may reuse the BVH.
        if (input.hashes[HashComponents::VertexPosition] == inOutGeometry.hashes[HashComponents::VertexPosition]
         && input.hashes[HashComponents::VertexShader] == inOutGeometry.hashes[HashComponents::VertexShader]
         && drawCallState.getSkinningState().boneHash == inOutGeometry.lastBoneHash) {
          result = ObjectCacheState::kUpdateInstance;
        } else {
          result = ObjectCacheState::kUpdateBVH;
        }
      }
    }

    RaytraceGeometry& output = inOutGeometry;

    output.lastBoneHash = drawCallState.getSkinningState().boneHash;

    // Update draw parameters
    output.cullMode = drawCallState.getCullMode();
    output.frontFace = input.frontFace;

    // Copy the hashes over
    output.hashes = input.hashes;

    if (!input.positionBuffer.defined()) {
      ONCE(Logger::err("processGeometryInfo: no position data on input detected"));
      return ObjectCacheState::kInvalid;
    }

    if (input.vertexCount == 0) {
      ONCE(Logger::err("processGeometryInfo: input data is violating some assumptions"));
      return ObjectCacheState::kInvalid;
    }

    // Set to 1 if inspection of the GeometryData structures contents on CPU is desired
    #define DEBUG_GEOMETRY_MEMORY 0
    constexpr VkMemoryPropertyFlags memoryProperty = DEBUG_GEOMETRY_MEMORY ? (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // Assume we won't need this, and update the value if required
    output.previousPositionBuffer = RaytraceBuffer();

    // When the SmoothNormals category is set and the input has no normals, force the interleaved
    // vertex layout to include space for normals. The smooth normals compute pass will fill them in later.
    const bool needsSmoothNormals = drawCallState.getCategoryFlags().test(InstanceCategories::SmoothNormals);
    const bool forceNormals = needsSmoothNormals && !input.normalBuffer.defined();

    // When smooth normals state changes (added or removed), promote to kUpdateBVH so the vertex
    // data is re-interleaved and the smooth normals dispatch runs (or original normals are restored).
    if (needsSmoothNormals != output.smoothNormalsApplied && result == ObjectCacheState::kUpdateInstance) {
      result = ObjectCacheState::kUpdateBVH;
    }
    if (!needsSmoothNormals) {
      output.smoothNormalsApplied = false;
    }

    // If forceNormals is true, we can't use the fast "already interleaved" path since
    // we need to change the layout to include normal space.
    const size_t vertexStride = (input.isVertexDataInterleaved() && input.areFormatsGpuFriendly() && !forceNormals)
      ? input.positionBuffer.stride()
      : RtxGeometryUtils::computeOptimalVertexStride(input, forceNormals);

    if (result == ObjectCacheState::kUpdateInstance && texcoordsChanged) {
      const auto sameLayout = [](const RasterBuffer& source, const RaytraceBuffer& cached) {
        return source.defined() == cached.defined() && (!source.defined() ||
          (source.offsetFromSlice() == cached.offsetFromSlice() &&
           source.stride() == cached.stride() && source.vertexFormat() == cached.vertexFormat()));
      };
      // In-place copies must preserve BLAS position addresses and any post-processed vertex data.
      const bool canRefreshInPlace = optimizeAnimatedTexcoords() &&
        !drawCallState.usesVertexShader && drawCallState.getSkinningState().numBones == 0 &&
        input.numBonesPerVertex == 0 && !needsSmoothNormals &&
        (!m_opacityMicromapManager || !m_opacityMicromapManager->isActive()) &&
        input.isVertexDataInterleaved() && input.areFormatsGpuFriendly() &&
        input.vertexCount == output.vertexCount &&
        output.historyBuffer[0]->info().size == align(vertexStride * input.vertexCount, CACHE_LINE_SIZE) &&
        !isAliasableGeometryBuffer(*output.historyBuffer[0]) &&
        sameLayout(input.positionBuffer, output.positionBuffer) &&
        sameLayout(input.normalBuffer, output.normalBuffer) &&
        sameLayout(input.texcoordBuffer, output.texcoordBuffer) &&
        sameLayout(input.color0Buffer, output.color0Buffer);

      if (canRefreshInPlace) {
        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output);
        m_instanceManager.notifySceneChanged();
      } else {
        result = ObjectCacheState::kUpdateBVH;
      }
    }

    if (result == ObjectCacheState::KBuildBVH) {
      RtxGeometryUtils::logTexcoordPrecision(input, drawCallState.getMaterialData().getColorTexture().getImageHash(),
                                             drawCallState.getTransformData().objectToWorld[3].xyz());
    }

    switch (result) {
      case ObjectCacheState::KBuildBVH: {
        // Set up the ideal vertex params, if input vertices are interleaved, it's safe to assume the positionBuffer stride is the vertex stride
        output.vertexCount = input.vertexCount;

        const size_t vertexBufferSize = output.vertexCount * vertexStride;

        // Set up the ideal index params
        output.indexCount = input.isTopologyRaytraceReady() ? input.indexCount : RtxGeometryUtils::getOptimalTriangleListSize(input);
        const VkIndexType indexBufferType = input.isTopologyRaytraceReady() ? input.indexBuffer.indexType() : RtxGeometryUtils::getOptimalIndexFormat(output.vertexCount);
        const size_t indexStride = (indexBufferType == VK_INDEX_TYPE_UINT16) ? 2 : 4;

        // Make sure we're not stomping something else...
        assert(output.indexCacheBuffer == nullptr && output.historyBuffer[0] == nullptr);

        // rtx.hostImport.deviceLocalStaticMeshes: a static API mesh already sits in device-local
        // buffers in exactly the layout built here (interleaved, GPU-friendly, uint32 triangle
        // list, 64-byte aligned size), marked with kAliasableGeometryUsage. Use those buffers as
        // this entry's geometry instead of allocating and copying a second one. Anything that
        // would write the entry's vertex buffer (skinning, smooth normals) is excluded here, and
        // the update paths below replace an aliased buffer before writing.
        if (RtxOptions::HostImport::deviceLocalStaticMeshes()) {
          const Rc<DxvkBuffer>& vb = input.positionBuffer.buffer();
          const Rc<DxvkBuffer>& ib = input.indexBuffer.defined() ? input.indexBuffer.buffer() : Rc<DxvkBuffer>();
          const bool aliasVertices = vb != nullptr && isAliasableGeometryBuffer(*vb) &&
            input.isVertexDataInterleaved() && input.areFormatsGpuFriendly() && !forceNormals && !needsSmoothNormals &&
            input.numBonesPerVertex == 0 && drawCallState.getSkinningState().numBones == 0 &&
            input.positionBuffer.offset() == 0 && vertexStride == input.positionBuffer.stride() &&
            vb->info().size == align(vertexBufferSize, CACHE_LINE_SIZE);
          const bool aliasIndices = aliasVertices && ib != nullptr && isAliasableGeometryBuffer(*ib) &&
            input.isTopologyRaytraceReady() && indexBufferType == VK_INDEX_TYPE_UINT32 &&
            input.indexBuffer.offset() + input.indexBuffer.offsetFromSlice() == 0 &&
            ib->info().size == align(output.indexCount * indexStride, CACHE_LINE_SIZE);
          if (aliasVertices && aliasIndices) {
            output.indexCacheBuffer = ib;
            output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);
            output.historyBuffer[0] = vb;
            RtxGeometryUtils::processGeometryBuffers(input, output);
            break;
          }
        }

        // Create a index buffer and vertex buffer we can use for raytracing.
        DxvkBufferCreateInfo info;
        info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

        info.size = align(output.indexCount * indexStride, CACHE_LINE_SIZE);
        output.indexCacheBuffer = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXSceneGeometry, "Index Cache Buffer");

        if (!RtxGeometryUtils::cacheIndexDataOnGPU(ctx, input, output)) {
          ONCE(Logger::err("processGeometryInfo: failed to cache index data on GPU"));
          return ObjectCacheState::kInvalid;
        }

        output.indexBuffer = RaytraceBuffer(DxvkBufferSlice(output.indexCacheBuffer), 0, indexStride, indexBufferType);

        info.size = align(vertexBufferSize, CACHE_LINE_SIZE);
        output.historyBuffer[0] = m_device->createBuffer(info, memoryProperty, DxvkMemoryStats::Category::RTXSceneGeometry, "Geometry Buffer");

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        break;
      }
      case ObjectCacheState::kUpdateBVH: {
        bool invalidateHistory = false;

        // New vertices mean a new SCCR shell cache key. The second time makes the geometry dynamic (plain SCCR).
        SccrShellManager::onGeometryVerticesChanged(output);

        // Stride changed, so we must recreate the previous buffer and use identical data
        if (output.historyBuffer[0]->info().size != align(vertexStride * input.vertexCount, CACHE_LINE_SIZE)) {
          auto desc = output.historyBuffer[0]->info();
          desc.usage &= ~kAliasableGeometryUsage;
          desc.size = align(vertexStride * input.vertexCount, CACHE_LINE_SIZE);
          output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXSceneGeometry, "Geometry Buffer");

          // Invalidate the current buffer
          output.historyBuffer[1] = nullptr;

          // Mark this object for realignment
          invalidateHistory = true;
        }

        // Use the previous updates vertex data for previous position lookup
        std::swap(output.historyBuffer[0], output.historyBuffer[1]);

        // A buffer aliased from the API mesh (rtx.hostImport.deviceLocalStaticMeshes) is never
        // written: it stays as history only, and a buffer of our own takes the write.
        if (output.historyBuffer[0].ptr() == nullptr || isAliasableGeometryBuffer(*output.historyBuffer[0])) {
          // First frame this object has been dynamic need to allocate a 2nd frame of data to preserve history.
          DxvkBufferCreateInfo desc = output.historyBuffer[1]->info();
          desc.usage &= ~kAliasableGeometryUsage;
          output.historyBuffer[0] = m_device->createBuffer(desc, memoryProperty, DxvkMemoryStats::Category::RTXSceneGeometry, "Geometry Buffer");
        }

        RtxGeometryUtils::cacheVertexDataOnGPU(ctx, input, output, forceNormals);

        // Sometimes, we need to invalidate history, do that here by copying the current buffer to the previous..
        if (invalidateHistory) {
          ctx->copyBuffer(output.historyBuffer[1], 0, output.historyBuffer[0], 0, output.historyBuffer[1]->info().size);
        }

        // Assign the previous buffer using the last slice (copy most params from the position, just change buffer)
        output.previousPositionBuffer = RaytraceBuffer(DxvkBufferSlice(output.historyBuffer[1], 0, output.positionBuffer.length()), output.positionBuffer.offsetFromSlice(), output.positionBuffer.stride(), output.positionBuffer.vertexFormat());
        break;
      }
      default:
        break;
    }

    // Update color buffer in BVH with DrawCallState
    // The user can disable/enable color buffer for specific materials, so we manually sync the DrawCallState and BVH here to keep the color buffer in BVH updated.
    // Note, we don't setup kUpdateBVH because it's too waste to update all buffers if only the color buffer needs to be updated.
    if (output.color0Buffer.defined() && !drawCallState.getGeometryData().color0Buffer.defined()) {
      // Remove the color buffer in BVH if the color buffer from drawcall is removed by ignoreBakedLighting
      output.color0Buffer = RaytraceBuffer();
    } else if (!output.color0Buffer.defined() && drawCallState.getGeometryData().color0Buffer.defined()) {
      // Write the color buffer back to BVH if the color buffer is enabled again
      const DxvkBufferSlice slice = DxvkBufferSlice(output.historyBuffer[0]);
      const auto& colorBuffer = drawCallState.getGeometryData().color0Buffer;
      output.color0Buffer = RaytraceBuffer(slice, colorBuffer.offsetFromSlice(), colorBuffer.stride(), colorBuffer.vertexFormat());
    }

    return result;
  }


  void SceneManager::applyWorldOrigin(const Vector3& origin) {
    if (m_worldOriginValid && !(origin == m_worldOrigin)) {
      // Old render position + old origin = new render position + new origin.
      const Vector3 d = m_worldOrigin - origin;
      uint32_t shifted = 0;
      for (RtInstance* instance : m_instanceManager.getInstanceTable()) {
        if (instance == nullptr) {
          continue;
        }
        Matrix4 current = instance->surface.objectToWorld;
        Matrix4 previous = instance->surface.prevObjectToWorld;
        current[3].x += d.x;
        current[3].y += d.y;
        current[3].z += d.z;
        previous[3].x += d.x;
        previous[3].y += d.y;
        previous[3].z += d.z;
        instance->teleport(current, previous);
        ++shifted;
      }
      if (shifted != 0) {
        m_instanceManager.notifySceneChanged();
      }
      // Legacy lights arrive re-expressed around the new origin too, as new lights (identity is position); the
      // old copies age out like any light the game stopped sending (LightManager::garbageCollectionInternal).
      static uint32_t s_logs = 0;
      if (s_logs++ < 32) {
        Logger::info(str::format("[WorldOrigin] origin moved by (", -d.x, ", ", -d.y, ", ", -d.z,
                                 ") render units; ", shifted, " instances re-expressed around the new origin"));
      }
    }
    m_worldOrigin = origin;
    m_worldOriginValid = true;
  }

  void SceneManager::applyPendingHostWorldOrigin() {
    if (!m_pendingHostWorldOriginValid) {
      return;
    }
    CameraManager& cameraManager = getCameraManager();
    if (!WorldOrigin::enable() || cameraManager.apiWorldOriginSeen()) {
      // Off, or the host carries its origin on a Main-camera SetupCamera (which already applied it): drop it.
      m_pendingHostWorldOriginValid = false;
      return;
    }
    // Only ahead of the frame's Main camera update; otherwise keep it for the next frame's first draw.
    if (!cameraManager.applyHostWorldOrigin(CameraType::Main, m_pendingHostWorldOrigin)) {
      return;
    }
    const bool firstOrigin = !m_worldOriginValid;
    applyWorldOrigin(m_pendingHostWorldOrigin);
    m_pendingHostWorldOriginValid = false;
    if (firstOrigin) {
      Logger::info(str::format("[WorldOrigin] D3D9 camera: host world origin (", m_worldOrigin.x, ", ",
                               m_worldOrigin.y, ", ", m_worldOrigin.z, ") render units"));
    }
  }

  void SceneManager::applyFallbackMainCamera() {
    CameraManager& cameraManager = getCameraManager();
    Matrix4 worldToView;
    Matrix4 viewToProjection;
    if (!cameraManager.takeFallbackCamera(worldToView, viewToProjection) ||
        cameraManager.isCameraValid(CameraType::Main)) {
      return;
    }
    applyPendingHostWorldOrigin();
    cameraManager.applyFrameCamera(CameraType::Main, worldToView, viewToProjection);
    static uint32_t s_logs = 0;
    if (s_logs++ < 8) {
      Logger::info(str::format("[RTX] Frame ", m_device->getCurrentFrameId(),
                               ": no draw set the Main camera; using the host's fallback camera (SetupCamera)"));
    }
  }

  void SceneManager::onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame) {
    ScopedCpuProfileZone();

    // Commit this frame's texture registrations for preserve next frame. Must run before
    // manageTextureVram(), which may clear the cache and bump the generation so the
    // following frame takes the dynamic path for every draw call.
    m_textureCacheGenerationValidForPreserve =
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration();

    manageTextureVram();

    // Update graphs before applying any hot-reload: invalidateChangedReplacements runs
    // below and marks GraphInstances for GC in the same frame-end, but garbageCollection
    // only runs at the start of the next frame (in prepareSceneData). Running update here
    // ensures it always operates on fully valid instances with live backing data.
    // RtxOptions will still be pending, so any changes to them will apply next frame.
    if (raytracedThisFrame) {
      m_graphManager.update(ctx);
    }

    AssetChanges changes;
    const bool changedSync = m_pReplacer->checkForChanges(ctx);
    const bool appliedRebuild = m_pReplacer->applyPendingRebuilds(ctx, changes);
    // rtx.scene.selectiveModChanges: an API host keeps its scene (textures, geometry, BLASes, OMMs) on a mod reload;
    // replacement instances are rebuilt and the scene's direct instances follow the asset epoch.
    const bool keepSceneOnModChange = changedSync && !m_enqueueDelayedClear &&
      scene::Options::selectiveModChanges() && m_pReplacer->hasExternalMeshes();
    if (keepSceneOnModChange) {
      for (const auto& replacementInstance : m_drawCallTracker.getReplacementInstances()) {
        if (replacementInstance != nullptr) {
          replacementInstance->clear();
        }
      }
      m_retainedScene->noteModsChanged(nullptr);
      Logger::info("[SceneRetained] mods changed: scene kept, replacement instances rebuilt (rtx.scene.selectiveModChanges)");
    } else if (m_enqueueDelayedClear || changedSync) {
      D3d9Identity::setPendingClearReason(m_enqueueDelayedClear ? D3d9Identity::ClearReason::CameraCut
                                                                : D3d9Identity::ClearReason::ModsChanged);
      clear(ctx, true);
      m_enqueueDelayedClear = false;
    } else if (appliedRebuild) {
      invalidateChangedReplacements(changes);
      if (scene::Options::selectiveModChanges()) {
        m_retainedScene->noteModsChanged(&changes);
      }
    }

    vram_budget::processReleasedImports(ctx.ptr());
    vram_budget::noteMaterialTexturesRefreshed(m_pReplacer->applyTextureRefreshes());

    logAccelMemoryBreakdown();
    D3d9IdentityTrace::onFrameEnd(m_device->getCurrentFrameId());
    D3d9Identity::onFrameEnd(m_device->getCurrentFrameId());

    // rtx.vramBudget: pressure level and its actions, after this frame's builds.
    vram_budget::update(m_device, m_accelManager, m_opacityMicromapManager.get(),
                        uint32_t(m_instanceManager.getInstanceTable().size()));

    m_cameraManager.onFrameEnd();
    m_instanceManager.onFrameEnd();
    m_previousFrameSceneAvailable = raytracedThisFrame && RtxOptions::enablePreviousTLAS();

    if (raytracedThisFrame) {
      std::lock_guard lock { m_drawCallMeta.mutex };
      const uint8_t curTick = m_drawCallMeta.ticker;
      const uint8_t nextTick = (m_drawCallMeta.ticker + 1) % m_drawCallMeta.MaxTicks;

      m_drawCallMeta.ready[curTick] = true;

      m_drawCallMeta.infos[nextTick].clear();
      m_drawCallMeta.ready[nextTick] = false;
      m_drawCallMeta.ticker = nextTick;
    }

    m_terrainBaker->onFrameEnd(ctx);

    if (m_opacityMicromapManager) {
      m_opacityMicromapManager->onFrameEnd();
    }

    m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    m_fogStartInMediumMaterialIndex_inCache = UINT32_MAX;
    m_startInMediumMaterialIndex_inCache = UINT32_MAX;

    if (m_uniqueObjectSearchDistance != RtxOptions::uniqueObjectDistance()) {
      m_uniqueObjectSearchDistance = RtxOptions::uniqueObjectDistance();
      m_drawCallTracker.rebuildSpatialMaps(m_uniqueObjectSearchDistance * 2.f);
    }

    // Not currently safe to cache these across frames (due to texture indices and rtx options potentially changing)
    m_preCreationSurfaceMaterialMap.clear();

    // Clear replacement material hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameReplacementMaterialHashes();
    
    // Clear mesh hashes before the next frame.  These are used by components, so must clear after graphManager updates.
    clearFrameMeshHashes();

    // Hash usage is recorded next frame only if a hash checker component read it recently (the graph update ran
    // before this point).
    m_framesSinceHashUsageConsumer = m_hashUsageConsumerSeen.exchange(false, std::memory_order_relaxed)
      ? 0 : std::min<uint32_t>(m_framesSinceHashUsageConsumer + 1, 1000u);
    s_hashUsageTrackingWanted.store(m_framesSinceHashUsageConsumer <= 2,
                                    std::memory_order_relaxed);

    // Reset the fog state to get it re-discovered on the next frame
    ImGUI::SetFogStates(m_fogStates, m_fog.getHash());
    m_fog = FogState();
    m_fogStates.clear();
    
    // Any drawcall translation invalidation has been consumed by this point. Clear the flag before new dirty options are processed.
    RtxOptionManager::clearDrawcallTranslationInvalid();
  }

  std::unordered_set<XXH64_hash_t> uniqueHashes;


  void SceneManager::discoverFogState(const DrawCallState& input) {
    discoverFog(input.getFogState(), &input);
  }

  void SceneManager::discoverFog(const FogState& fog, const DrawCallState* pInput) {
    if (fog.mode != D3DFOG_NONE) {
      XXH64_hash_t fogHash = fog.getHash();
      if (m_fogStates.find(fogHash) == m_fogStates.end()) {
        std::shared_ptr<MaterialData> pFogReplacement = m_pReplacer->getReplacementMaterial(fogHash);
        if (pFogReplacement && pInput == nullptr) {
          return;  // No draw of this fog to create the medium material from; left undiscovered
        }
        // Only do anything if we haven't seen this fog before.
        m_fogStates[fogHash] = fog;

        if (pFogReplacement) {
          const DrawCallState& input = *pInput;
          // Track this replacement material hash for hash checking
          trackReplacementMaterialHash(fogHash);
          // Fog has been replaced by a translucent material to start the camera in,
          // meaning that it was being used to indicate 'underwater' or something similar.
          if (pFogReplacement->getType() != MaterialDataType::Translucent) {
            Logger::warn(str::format("Fog replacement materials must be translucent.  Ignoring material for ", std::hex, m_fog.getHash()));
          } else {
            uint32_t id = UINT32_MAX;
            createSurfaceMaterial(*pFogReplacement, input, &id);
            assert(id != UINT32_MAX);
            m_fogStartInMediumMaterialIndex_inCache = id;
          }
        } else if (m_fog.mode == D3DFOG_NONE) {
          // render the first unreplaced fog.
          m_fog = fog;
        }
      }
    }
  }

  bool SceneManager::hasMeshReplacements(const DrawCallState& input) const {
    if (m_pReplacer->getReplacementsForMesh(input.getHash(RtxOptions::geometryAssetHashRule())) != nullptr) {
      return true;
    }
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0 &&
        m_pReplacer->getReplacementsForMesh(input.getHashLegacy(rules::LegacyAssetHash0)) != nullptr) {
      return true;
    }
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1 &&
        m_pReplacer->getReplacementsForMesh(input.getHashLegacy(rules::LegacyAssetHash1)) != nullptr) {
      return true;
    }
    return false;
  }

  std::shared_ptr<const ReplacementBucket> SceneManager::findMeshReplacements(const DrawCallState& input, uint32_t* matchedRule) const {
    if (auto bucket = m_pReplacer->getReplacementsForMesh(input.getHash(RtxOptions::geometryAssetHashRule()))) {
      *matchedRule = 0;
      return bucket;
    }
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0) {
      if (auto bucket = m_pReplacer->getReplacementsForMesh(input.getHashLegacy(rules::LegacyAssetHash0))) {
        *matchedRule = 1;
        return bucket;
      }
    }
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1) {
      if (auto bucket = m_pReplacer->getReplacementsForMesh(input.getHashLegacy(rules::LegacyAssetHash1))) {
        *matchedRule = 2;
        return bucket;
      }
    }
    return nullptr;
  }

  void SceneManager::submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData) {
    ScopedCpuProfileZone();
    if (m_bufferCache.getTotalCount() >= kBufferCacheLimit && m_bufferCache.getActiveCount() >= kBufferCacheLimit) {
      ONCE(Logger::info("[RTX-Compatibility-Info] This application is pushing more unique buffers than is currently supported - some objects may not raytrace."));
      return;
    }

    if (!m_frontendFogDiscovery) {  // The frontend discovers fog in its frame order
      discoverFogState(input);
    }


    const XXH64_hash_t activeReplacementHash = input.getHash(RtxOptions::geometryAssetHashRule());
    
    // Track this mesh hash for mesh hash checking
    trackMeshHash(activeReplacementHash);

    std::shared_ptr<const ReplacementBucket> pReplacements = m_pReplacer->getReplacementsForMesh(activeReplacementHash);

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash0);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash0: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1) {
      if (!pReplacements) {
        const XXH64_hash_t legacyHash = input.getHashLegacy(rules::LegacyAssetHash1);
        trackMeshHash(legacyHash);
        pReplacements = m_pReplacer->getReplacementsForMesh(legacyHash);
        if (RtxOptions::logLegacyHashReplacementMatches() && pReplacements && uniqueHashes.find(legacyHash) == uniqueHashes.end()) {
          uniqueHashes.insert(legacyHash);
          Logger::info(str::format("[Legacy-Hash-Replacement] Found a mesh referenced from legacyHash1: ", std::hex, legacyHash, ", new hash: ", std::hex, activeReplacementHash));
        }
      }
    }

    ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(
        input, m_rayPortalManager, overrideMaterialData, pReplacements != nullptr);
    // False for every draw unless the frontend is submitting a handed-over object's draw.
    replacementInstance->handedToScene = m_submittingHandedOver;

    const uint32_t currentFrameId = m_device->getCurrentFrameId();
    const bool secondSubmissionThisFrame = (replacementInstance->frameLastSeen == currentFrameId);

    // Preserve path: L1 means this draw still matches the same ReplacementInstance by identity; replacer reload is
    // expected to clear the draw-call tracker (see SceneManager::clear), so we do not re-verify mesh prims or
    // activeReplacements pointers every frame. If a path exists where replacements bind without a clear, use dynamic
    // (drawReplacements) for that transition -- drawReplacements already reconciles activeReplacements and prims.
    //
    // Static path reuses BlasEntry::modifiedGeometryData as-is, valid only if no other draw this
    // frame rebuilt it (kUpdateBVH); frameLastUpdated tracks that, while frameLastTouched also
    // fires for exact-match sibling draws that share a BlasEntry without rebuilding it.
    auto blasAlreadyTouchedByOtherDraw = [replacementInstance, currentFrameId]() -> bool {
      for (const auto& prim : replacementInstance->prims) {
        RtInstance* inst = prim.getInstance();
        if (inst == nullptr) {
          continue;
        }
        BlasEntry* pBlas = inst->getBlas();
        if (pBlas == nullptr) {
          continue;
        }
        if (pBlas->frameLastUpdated == currentFrameId) {
          return true;
        }
      }
      return false;
    };

    // Per-frame override materials (terrain bake, etc.) can introduce particle systems
    // without a prior dynamic update on this RI.
    const bool overrideMaterialHasParticles = overrideMaterialData != nullptr
        && overrideMaterialData->getParticleSystemDesc() != nullptr;

    // The RI's prims must already be wired up for this exact replacements bucket.
    // drawReplacements re-initializes prims when activeReplacements changes (e.g. async
    // replacement load completes after the RI was created without replacements, hot-reload
    // swaps the bucket, or a variant toggle changes the lookup key). The preserve path has
    // no equivalent reinitialization, so fall back to dynamic for that transition.
    const bool activeReplacementsMatch =
        replacementInstance->activeReplacements.get() == pReplacements.get();

    // Terrain draws share a per-frame override OpaqueMaterialData built from the
    // TerrainBaker cascade set. Cascade images don't carry a stable identity hash,
    // so MaterialData::getHash() doesn't change when the cascade set grows or shrinks
    // (e.g. when the normal cascade is first added on top of an albedo-only set on
    // an earlier frame). The preserve path would then reuse the surfaceMaterialIndex
    // computed before the new cascade textures existed, leaving the secondary
    // texture slots invalid. Force the dynamic path on any frame in which a cascade
    // image was created or resized; subsequent stable frames take the preserve path
    // with the freshly built RtSurfaceMaterial.
    const bool terrainCascadesJustChanged =
        input.getCategoryFlags().test(InstanceCategories::Terrain) &&
        m_terrainBaker->cascadeCompositionChangedThisFrame();

    const bool cachedTexturesValidForPreserve =
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration() ==
        m_textureCacheGenerationValidForPreserve;

    const XXH64_hash_t legacyMaterialIdentityHash = input.getMaterialData().computeIdentityHash();
    const bool legacyMaterialIdentityHashMatch =
        replacementInstance->legacyMaterialIdentityHash == legacyMaterialIdentityHash;

    const bool usePreservePath =
        RtxOptions::enablePreservePath() &&
        !replacementInstance->prims.empty() &&
        replacementInstance->dirtyFlags.isClear() &&
        !RtxOptionManager::isDrawcallTranslationInvalid() &&
        !secondSubmissionThisFrame &&
        !input.getCategoryFlags().test(InstanceCategories::ParticleEmitter) &&
        !RtxOptions::shouldConvertToLight(input.getMaterialData().getHash()) &&
        !blasAlreadyTouchedByOtherDraw() &&
        !overrideMaterialHasParticles&&
        activeReplacementsMatch &&
        legacyMaterialIdentityHashMatch &&
        !terrainCascadesJustChanged &&
        cachedTexturesValidForPreserve;

    // [PerfD3D9] preserved / dynamic and the first reason the preserve path was not taken
    // (same tests, same order as usePreservePath; they have no side effects).
    if (perf::g_perfDetailed.load(std::memory_order_relaxed)) {
      using C = perf::D3D9Ctr;
      C reason = C::Preserved;
      if (!usePreservePath) {
        if (!RtxOptions::enablePreservePath()) reason = C::RejectPreserveOff;
        else if (replacementInstance->prims.empty()) reason = C::RejectNoPrims;
        else if (!replacementInstance->dirtyFlags.isClear()) reason = C::RejectDirtyFlags;
        else if (RtxOptionManager::isDrawcallTranslationInvalid()) reason = C::RejectTranslationInvalid;
        else if (secondSubmissionThisFrame) reason = C::RejectSecondSubmission;
        else if (input.getCategoryFlags().test(InstanceCategories::ParticleEmitter)) reason = C::RejectParticleEmitter;
        else if (RtxOptions::shouldConvertToLight(input.getMaterialData().getHash())) reason = C::RejectConvertToLight;
        else if (blasAlreadyTouchedByOtherDraw()) reason = C::RejectBlasTouched;
        else if (overrideMaterialHasParticles) reason = C::RejectOverrideParticles;
        else if (!activeReplacementsMatch) reason = C::RejectReplacementsChanged;
        else if (!legacyMaterialIdentityHashMatch) reason = C::RejectMaterialIdentity;
        else if (terrainCascadesJustChanged) reason = C::RejectTerrainCascades;
        else reason = C::RejectTextureCache;
        perf::countD3D9(C::Dynamic);
      }
      perf::countD3D9(reason);
    }
    const bool timeSubmitPath = m_timeD3D9Submit;
    const auto submitPathStart = timeSubmitPath ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    // The draw state to read from after the path ran (preserveReplacementInstance returns `input`).
    const DrawCallState* pLiveInput = &input;
    if (usePreservePath) {
      pLiveInput = &preserveReplacementInstance(ctx, input, pReplacements, replacementInstance);
    } else {
      // Any RtxOption read inside the dynamic update should force a full update when changed.
      RTX_OPTION_INVALIDATION_SCOPE(RtxOptionFlags::InvalidatesDrawcallTranslation);
      MaterialData renderMaterialData = determineMaterialData(overrideMaterialData, input);
      if (!activeReplacementsMatch) {
        replacementInstance->clear();
      }

      // Recompute dynamic-feature bits on each dynamic update.
      replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);

      // Create / process: full geometry cache and instance update.
      if (pReplacements != nullptr) {
        drawReplacements(ctx, &input, pReplacements, renderMaterialData, replacementInstance);
      } else {
        RtInstance* existingInstance = (replacementInstance->prims.size() > 0)
            ? replacementInstance->prims[0].getInstance() : nullptr;

        RtInstance* instance = processDrawCallState(ctx, input, renderMaterialData,
            *replacementInstance, existingInstance, nullptr);
        if (instance != nullptr) {
          if (replacementInstance->root.getUntyped() == nullptr) {
            replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), 1, nullptr);
          }
          if (replacementInstance->prims[0].getUntyped() != instance) {
            instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, 0, instance,
                PrimInstance::Type::Instance);
          }
        }
      }
      replacementInstance->legacyMaterialIdentityHash = legacyMaterialIdentityHash;
    }
    if (timeSubmitPath) {
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitPathStart).count();
      (usePreservePath ? m_d3d9PreserveMs : m_d3d9DynamicMs) += ms;
    }

    const DrawCallState& liveInput = *pLiveInput;
    replacementInstance->frameLastSeen = currentFrameId;
    replacementInstance->categoryFlags = liveInput.getCategoryFlags().raw();
    replacementInstance->isSkinned = liveInput.getSkinningState().numBones > 0;

    // Cache this submission's texture-coordinate projection so that next frame's
    // computeDirtyFlags can detect drift. Writing here (after either path has run)
    // mirrors how objectToWorld is updated downstream of the dirty-flag check.
    replacementInstance->textureTransform = liveInput.getTransformData().textureTransform;
    replacementInstance->texgenMode = liveInput.getTransformData().texgenMode;

    // For standalone draw calls, store the object-space bounding box for anti-culling.
    // For replacement draw calls, the aggregate AABB is computed inside drawReplacements.
    if (pReplacements == nullptr) {
      const auto& geoBBox = liveInput.getGeometryData().boundingBox;
      if (geoBBox.isValid()) {
        replacementInstance->geometryBoundingBox = geoBBox;
        replacementInstance->objectToWorld = liveInput.getTransformData().objectToWorld;
      }
    }
  }

  MaterialData SceneManager::determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input) {
    ScopedCpuProfileZone();
    // First see if we have an explicit override
    if (overrideMaterialData != nullptr) {
      return *overrideMaterialData;
    } 

    // test if any direct material replacements exist
    const XXH64_hash_t legacyMaterialHash = input.getMaterialData().getHash();
    std::shared_ptr<MaterialData> pReplacementMaterial = m_pReplacer->getReplacementMaterial(legacyMaterialHash);
    if (pReplacementMaterial != nullptr) {
      // Make a copy - dont modify the replacement data.
      MaterialData renderMaterialData = *pReplacementMaterial;
      // merge in the input material from game
      renderMaterialData.mergeLegacyMaterial(input.getMaterialData());
      return renderMaterialData;
    }

    // Check if a Ray Portal override is needed
    size_t rayPortalTextureIndex;
    if (RtxOptions::getRayPortalTextureIndex(input.getMaterialData().getHash(), rayPortalTextureIndex)) {
      assert(rayPortalTextureIndex < maxRayPortalCount);
      assert(rayPortalTextureIndex < (std::numeric_limits<uint8_t>::max)());

      MaterialData renderMaterialData = input.getMaterialData().as<RayPortalMaterialData>();
      renderMaterialData.getRayPortalMaterialData().setRayPortalIndex(rayPortalTextureIndex);
      return renderMaterialData;
    }

    // Standard legacy material conversion
    return MaterialData::fromLegacy(input.getMaterialData());
  }

  void SceneManager::createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance) {
    const float effectLightIntensity = RtxOptions::effectLightIntensity();
    if (effectLightIntensity <= 0.f)
      return;

    const RasterGeometry& geometryData = input.getGeometryData();

    const GeometryBufferData bufferData(geometryData);
    
    if (!bufferData.indexData && geometryData.indexCount > 0 || !bufferData.positionData)
      return;

    // Find centroid of point cloud.
    Vector3 centroid = Vector3();
    uint32_t counter = 0;
    if (geometryData.indexCount > 0) {
      for (uint32_t i = 0; i < geometryData.indexCount; i++) {
        const uint16_t index = bufferData.getIndex(i);
        centroid += bufferData.getPosition(index);
        ++counter;
      }
    } else {
      for (uint32_t i = 0; i < geometryData.vertexCount; i++) {
        centroid += bufferData.getPosition(i);
        ++counter;
      }
    }
    centroid /= (float) counter;
    
    const Vector4 renderingPos = input.getTransformData().objectToView * Vector4(centroid.x, centroid.y, centroid.z, 1.0f);
    // Note: False used in getViewToWorld since the renderingPos of the object is defined with respect to the game's object to view
    // matrix, not our freecam's, and as such we want to convert it back to world space using the matching matrix.
    const Vector4 worldPos{ getCamera().getViewToWorld(false) * Vector4d{ renderingPos } };

    RtLightShaping shaping{};

    float lightRadius = std::max(RtxOptions::effectLightRadius(), 1e-3f);
    const Vector3 lightPosition { worldPos.x, worldPos.y, worldPos.z };
    Vector3 lightRadiance;
    if (RtxOptions::effectLightPlasmaBall()) {
      // Todo: Make these options more configurable via config options.
      const double timeMilliseconds = static_cast<double>(GlobalTime::get().absoluteTimeMs());
      const double animationPhase = sin(timeMilliseconds * 0.006) * 0.5 + 0.5;
      lightRadiance = lerp(Vector3(1.f, 0.921f, 0.738f), Vector3(1.f, 0.521f, 0.238f), animationPhase);
    } else {
      const D3DCOLORVALUE originalColor = input.getMaterialData().getLegacyMaterial().Diffuse;
      lightRadiance = Vector3(originalColor.r, originalColor.g, originalColor.b) * RtxOptions::effectLightColor();
    }
    const float surfaceArea = 4.f * kPi * lightRadius * lightRadius;
    const float radianceFactor = 1e5f * effectLightIntensity / surfaceArea;
    lightRadiance *= radianceFactor;

    RtLight rtLight(RtSphereLight(lightPosition, lightRadiance, lightRadius, shaping));
    rtLight.isDynamic = true;

    const XXH64_hash_t materialHash = input.getMaterialData().getHash();
    if (m_effectLightMaterialsLogged.insert(materialHash).second) {
      Logger::info(str::format("[EffectLight] first effect light from rtx.lightConverter material 0x", std::hex, materialHash, std::dec,
                               " at (", lightPosition.x, ", ", lightPosition.y, ", ", lightPosition.z, "), ",
                               length(lightPosition - getCamera().getPosition()), " units from the camera"));
    }

    m_lightManager.addLight(rtLight, input, RtLightAntiCullingType::MeshReplacement);
  }

  namespace {
    // Cache for SceneManager::composeHostInstancesForReplacement, keyed by (owner, index).
    struct HostInstanceCompositionEntry {
      // The host list the composition was made from (kept alive for the content comparison).
      std::shared_ptr<const std::vector<Matrix4>> source;
      std::shared_ptr<const std::vector<Matrix4>> composed;
      const std::vector<Matrix4>* prototypes = nullptr;  // AssetReplacement::instancesToObject
      Matrix4 replacementToObject;
      uint32_t lastUsedFrame = 0;
    };
    struct HostInstanceCompositionKey {
      const ReplacementInstance* owner;
      size_t index;
      bool operator==(const HostInstanceCompositionKey& o) const { return owner == o.owner && index == o.index; }
    };
    struct HostInstanceCompositionKeyHash {
      size_t operator()(const HostInstanceCompositionKey& k) const {
        return std::hash<const void*>()(k.owner) ^ (k.index * 0x9E3779B97F4A7C15ull);
      }
    };
    dxvk::mutex s_hostInstanceCompositionMutex;
    std::unordered_map<HostInstanceCompositionKey, HostInstanceCompositionEntry, HostInstanceCompositionKeyHash> s_hostInstanceCompositions;
    uint32_t s_hostInstanceCompositionSweepFrame = 0;
    // Entries unused this many frames are dropped (their ReplacementInstance is gone or no longer
    // host-instanced). A reused owner address is harmless: the entry is revalidated on content.
    constexpr uint32_t kHostInstanceCompositionMaxAge = 120;
  }

  std::shared_ptr<const std::vector<Matrix4>> SceneManager::composeHostInstancesForReplacement(
      const std::shared_ptr<const std::vector<Matrix4>>& hostInstances,
      const AssetReplacement& replacement,
      const ReplacementInstance* owner,
      size_t index,
      uint32_t frameId) {
    const std::vector<Matrix4>* prototypes =
      (replacement.instancesToObject && !replacement.instancesToObject->empty()) ? replacement.instancesToObject.get() : nullptr;
    if (prototypes == nullptr && isIdentityExact(replacement.replacementToObject)) {
      // Nothing to compose: the replacement sits exactly where the original mesh did.
      return hostInstances;
    }

    std::lock_guard lock { s_hostInstanceCompositionMutex };

    // Sweep stale entries at most once per frame.
    if (frameId != s_hostInstanceCompositionSweepFrame) {
      s_hostInstanceCompositionSweepFrame = frameId;
      for (auto it = s_hostInstanceCompositions.begin(); it != s_hostInstanceCompositions.end();) {
        if (frameId - it->second.lastUsedFrame > kHostInstanceCompositionMaxAge) {
          it = s_hostInstanceCompositions.erase(it);
        } else {
          ++it;
        }
      }
    }

    HostInstanceCompositionEntry& entry = s_hostInstanceCompositions[HostInstanceCompositionKey { owner, index }];
    entry.lastUsedFrame = frameId;

    const std::vector<Matrix4>& host = *hostInstances;
    const bool sameReplacement = entry.composed != nullptr &&
      entry.prototypes == prototypes &&
      entry.replacementToObject == replacement.replacementToObject;
    if (sameReplacement) {
      if (entry.source == hostInstances) {
        return entry.composed;
      }
      if (entry.source != nullptr && entry.source->size() == host.size() &&
          std::memcmp(entry.source->data(), host.data(), host.size() * sizeof(Matrix4)) == 0) {
        entry.source = hostInstances;
        return entry.composed;
      }
    }

    auto composed = std::make_shared<std::vector<Matrix4>>();
    if (prototypes != nullptr) {
      // A USD PointInstancer replacement already folded replacementToObject into its prototypes.
      composed->reserve(host.size() * prototypes->size());
      for (const Matrix4& h : host) {
        for (const Matrix4& p : *prototypes) {
          composed->push_back(h * p);
        }
      }
    } else {
      composed->reserve(host.size());
      for (const Matrix4& h : host) {
        composed->push_back(h * replacement.replacementToObject);
      }
    }

    entry.source = hostInstances;
    entry.composed = std::move(composed);
    entry.prototypes = prototypes;
    entry.replacementToObject = replacement.replacementToObject;
    return entry.composed;
  }

  std::optional<DrawCallState> SceneManager::buildReplacementMeshDrawCallState(
      const DrawCallState& input,
      const AssetReplacement& replacement,
      const ReplacementInstance* owner,
      size_t index,
      uint32_t frameId) {
    if (replacement.includeOriginal) {
      auto newDrawCallState = std::make_optional<DrawCallState>(input);
      newDrawCallState->modifyCategoryFlags() = replacement.categories.applyCategoryFlags(newDrawCallState->getCategoryFlags());
      return newDrawCallState;
    }

    if (replacement.type == AssetReplacement::eMesh) {
      auto newDrawCallState = std::make_optional<DrawCallState>(input);

      DrawCallTransforms& transforms = newDrawCallState->modifyTransformData();
      const bool hostInstanced = input.getTransformData().instancesFromHost &&
        input.getTransformData().instancesToObject != nullptr &&
        !input.getTransformData().instancesToObject->empty();
      if (hostInstanced) {
        // The replaced draw is GPU-instanced by the host (Remix API instancing). Keep its
        // base transform and instance the replacement at every host placement; dropping the list
        // here drew the replacement once, at the (often identity) base transform.
        transforms.instancesToObject = composeHostInstancesForReplacement(
          input.getTransformData().instancesToObject, replacement, owner, index, frameId);
        transforms.instancesFromHost = true;
      } else {
        transforms.objectToWorld = transforms.objectToWorld * replacement.replacementToObject;
        transforms.objectToView = transforms.objectToView * replacement.replacementToObject;
        if (replacement.instancesToObject && !replacement.instancesToObject->empty()) {
          transforms.instancesToObject = replacement.instancesToObject;
        } else {
          transforms.instancesToObject = nullptr;
        }
        transforms.instancesFromHost = false;
      }
      // Mesh replacements don't support these.
      transforms.textureTransform = Matrix4();
      transforms.texgenMode = TexGenMode::None;

      // RasterGeometry is a value member of MeshReplacement, not separately allocated.
      // The aliasing ctor makes a shared_ptr<RasterGeometry> that shares the ref-count
      // of the MeshReplacement, keeping it alive for the BlasEntry's lifetime.
      newDrawCallState->overrideGeometryData(
        std::shared_ptr<const RasterGeometry>(replacement.geometry, &replacement.geometry->data));
      newDrawCallState->modifyCategoryFlags() = replacement.categories.applyCategoryFlags(newDrawCallState->getCategoryFlags());
      return newDrawCallState;
    }

    return std::nullopt;
  }

  void SceneManager::drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::shared_ptr<const ReplacementBucket>& pReplacements, MaterialData& renderMaterialData, ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    if (pReplacements == nullptr) {
      assert(false && "pReplacements should never be nullptr here");
      ONCE(Logger::err("pReplacements should never be nullptr in SceneManager::drawReplacements"));
      return;
    }

    // Prim slots are sized for one bucket, and an instance without a slot is never collected.
    // Prims of a plain draw or of another bucket therefore go first.
    if (replacementInstance->activeReplacements != pReplacements) {
      replacementInstance->clear();
    }

    // If the index contains an RtInstance, get a pointer to it.
    auto getExistingInstance = [replacementInstance](size_t idx) -> RtInstance* {
      if (replacementInstance->prims.size() <= idx) {
        return nullptr;
      }
      return replacementInstance->prims[idx].getInstance();
    };

    const std::vector<AssetReplacement>& replacements = pReplacements->replacements;
    for (size_t i = 0; i < replacements.size(); i++) {
      auto& replacement = replacements[i];
      RtInstance* instance = nullptr;

      std::optional<DrawCallState> newDrawCallState = SceneManager::buildReplacementMeshDrawCallState(
          *input, replacement, replacementInstance, i, m_device->getCurrentFrameId());
      if (newDrawCallState.has_value()) {
        // Note: Material Data replaced if a replacement is specified in the Mesh Replacement.
        // Only meaningful when geometry is replaced (eMesh); the includeOriginal branch keeps the
        // game's original material data.
        if (!replacement.includeOriginal && replacement.type == AssetReplacement::eMesh && replacement.materialData != nullptr) {
          renderMaterialData = *replacement.materialData;
        }

        const RtxParticleSystemDesc* pParticleSystemDesc = replacement.particleSystem.has_value() ? &replacement.particleSystem.value() : nullptr;
        instance = processDrawCallState(ctx, *newDrawCallState, renderMaterialData, *replacementInstance, getExistingInstance(i), pParticleSystemDesc);
      }

      if (instance != nullptr) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          // This is the first time this replacementInstance is used, and the first mesh drawn
          //  as part of this replacementInstance, so invoke setup and set the root.
          replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), replacements.size(), pReplacements);
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        } else if (replacementInstance->prims.size() > i &&
                   replacementInstance->prims[i].getUntyped() != instance) {
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance, PrimInstance::Type::Instance);
        }
      }
    }

    processReplacementLights(input, pReplacements.get(), replacementInstance);
    processReplacementGraphs(ctx, input, pReplacements.get(), replacementInstance);
    updateLensAnchor(*input, pReplacements, *replacementInstance);

    replacementInstance->recalculateBoundingBox(
        input->getTransformData().objectToWorld,
        &input->getGeometryData().boundingBox);
  }
  
  namespace {
    // Returns the lens moved by a transform, which may be projective.
    // Lengths are measured on the moved lens, so they follow the transform's scale.
    LensDesc transformLens(const LensDesc& lens, const Matrix4d& lensToWorld) {
      auto transformPoint = [&lensToWorld](const Vector3d& point) {
        const Vector4d result = lensToWorld * Vector4d(point.x, point.y, point.z, 1.0);
        const double w = std::abs(result.w) > 1e-12 ? result.w : 1.0;
        return Vector3d(result.x / w, result.y / w, result.z / w);
      };

      const Vector3d center { lens.pose.eyepieceCenter };
      const Vector3d authoredUp { lens.pose.up };
      const Vector3d axis = safeNormalize(Vector3d { lens.pose.axis }, Vector3d(0.0, 0.0, -1.0));
      const Vector3d up = safeNormalize(authoredUp - axis * dot(authoredUp, axis), Vector3d(0.0, 1.0, 0.0));

      // The directions are sampled one eyepiece radius from the center, where a projective transform is close to linear.
      const double step = std::max(static_cast<double>(lens.eyepieceRadius), 1e-4);
      const Vector3d worldCenter = transformPoint(center);
      double axisLength;
      const Vector3d worldAxis = safeNormalizeGetLength(transformPoint(center + axis * step) - worldCenter, Vector3d(0.0, 0.0, -1.0), axisLength);
      double upLength;
      const Vector3d worldUp = safeNormalizeGetLength(transformPoint(center + up * step) - worldCenter, Vector3d(0.0, 1.0, 0.0), upLength);
      const float axisScale = static_cast<float>(axisLength / step);
      const float radiusScale = static_cast<float>(upLength / step);

      LensDesc result = lens;
      result.pose.eyepieceCenter = Vector3 { worldCenter };
      result.pose.axis = Vector3 { worldAxis };
      result.pose.up = Vector3 { worldUp };
      result.eyepieceRadius = lens.eyepieceRadius * radiusScale;
      result.objectiveRadius = lens.objectiveRadius * radiusScale;
      result.objectiveDistance = lens.objectiveDistance * axisScale;
      result.eyeRelief = lens.eyeRelief * axisScale;
      result.parallaxDistance = lens.parallaxDistance * axisScale;
      return result;
    }
  }

  void SceneManager::updateLensAnchor(const DrawCallState& input, const std::shared_ptr<const ReplacementBucket>& pReplacements,
                                      const ReplacementInstance& replacementInstance) {
    const std::vector<AssetReplacement>& replacements = pReplacements->replacements;
    const bool hasLens = std::any_of(replacements.begin(), replacements.end(), [](const AssetReplacement& replacement) {
      return replacement.type == AssetReplacement::eLens && replacement.lensData.has_value();
    });
    if (!hasLens) {
      return;
    }

    const DrawCallTransforms& inputTransforms = input.getTransformData();
    if (inputTransforms.instancesFromHost && inputTransforms.instancesToObject != nullptr &&
        !inputTransforms.instancesToObject->empty()) {
      ONCE(Logger::warn("[Lens] A mesh replacement with lenses replaces a draw that the host GPU-instances. Its lenses are not placed."));
      return;
    }

    // The anchor is the first mesh instance with a single transform: the original draw, or a mesh prim that is not
    // a point instancer prototype.
    const RtInstance* anchor = nullptr;
    Matrix4d objectToAnchor;
    for (size_t i = 0; i < replacements.size() && i < replacementInstance.prims.size(); ++i) {
      const AssetReplacement& replacement = replacements[i];
      const RtInstance* instance = replacementInstance.prims[i].getInstance();
      if (instance == nullptr) {
        continue;
      }
      if (replacement.includeOriginal) {
        anchor = instance;
        break;
      }
      if (replacement.type == AssetReplacement::eMesh &&
          (replacement.instancesToObject == nullptr || replacement.instancesToObject->empty())) {
        anchor = instance;
        objectToAnchor = inverse(Matrix4d { replacement.replacementToObject });
        break;
      }
    }

    if (anchor == nullptr) {
      ONCE(Logger::warn("[Lens] A mesh replacement with lenses has no mesh to place them on. Its lenses are not placed."));
      return;
    }

    LensAnchor lensAnchor;
    lensAnchor.bucket = pReplacements;
    for (size_t i = 0; i < replacements.size(); ++i) {
      const AssetReplacement& replacement = replacements[i];
      if (replacement.type == AssetReplacement::eLens && replacement.lensData.has_value()) {
        lensAnchor.lensToAnchor.emplace_back(static_cast<uint32_t>(i), objectToAnchor * Matrix4d { replacement.replacementToObject });
      }
    }

    std::lock_guard lock { m_lensAnchorMutex };
    m_lensAnchors[anchor] = std::move(lensAnchor);
  }

  void SceneManager::addReplacementLenses() {
    ScopedCpuProfileZone();

    const uint32_t frameId = m_device->getCurrentFrameId();
    if (!LensManager::enable() || m_lensesAddedFrameId == frameId) {
      return;
    }
    m_lensesAddedFrameId = frameId;

    std::lock_guard lock { m_lensAnchorMutex };
    for (const auto& [anchor, lensAnchor] : m_lensAnchors) {
      if (anchor->isHidden() || anchor->isMarkedForGC() || lensAnchor.bucket->stale) {
        continue;
      }

      // A view model reference is traced as its view model copy, so its lenses go where the copy is.
      Matrix4d anchorToWorld;
      Matrix4d prevAnchorToWorld;
      const bool hasViewModelCopy = m_instanceManager.getViewModelTransforms(*anchor, anchorToWorld, prevAnchorToWorld);
      // An instance drawn with the view model camera is first person even when view model copies are off.
      const bool firstPerson = hasViewModelCopy || anchor->isCameraRegistered(CameraType::ViewModel);
      if (!hasViewModelCopy) {
        // No ray hits an instance with a zero mask, such as a view model reference without a copy.
        if (anchor->getVkInstance().mask == 0) {
          continue;
        }
        anchorToWorld = Matrix4d { anchor->getTransform() };
        prevAnchorToWorld = Matrix4d { anchor->getPrevTransform() };
      }

      for (const auto& [index, lensToAnchor] : lensAnchor.lensToAnchor) {
        const LensReplacement& lens = *lensAnchor.bucket->replacements[index].lensData;
        if (lens.firstPersonOnly && !firstPerson) {
          continue;
        }

        const LensDesc current = transformLens(lens.desc, anchorToWorld * lensToAnchor);
        const LensDesc previous = transformLens(lens.desc, prevAnchorToWorld * lensToAnchor);

        // The top bit keeps these ids apart from the ids of the lens manager's own lenses.
        const struct { uint64_t instanceId; uint64_t lensIndex; } key { anchor->getId(), index };
        const uint64_t id = XXH3_64bits(&key, sizeof(key)) | (1ull << 63);
        m_lensManager.addLens(id, current, &previous.pose);
      }
    }
  }

  void SceneManager::processReplacementLights(
      const DrawCallState* input, const ReplacementBucket* pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    for (size_t i = 0; i < pReplacements->replacements.size(); i++) {
      auto&& replacement = pReplacements->replacements[i];
      if (replacement.type == AssetReplacement::eLight) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          Logger::err(str::format(
              "Light prims anchored to a mesh replacement must also include actual meshes.  mesh hash: ",
              std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
          ));
          break;
        }
        if (replacement.lightData.has_value()) {
          RtLight objectSpaceLight = replacement.lightData->toRtLight();

          // Transform to world space for the actual light creation
          RtLight localLight = objectSpaceLight;
          const DrawCallTransforms& inputTransforms = input->getTransformData();
          if (inputTransforms.instancesFromHost && inputTransforms.instancesToObject != nullptr &&
              !inputTransforms.instancesToObject->empty()) {
            // Lights are not instanced. A host-instanced draw gets its replacement light
            // once, at the first placement (the base transform alone is often identity).
            ONCE(Logger::warn(
              "[HostInstancing] A mesh replacement with lights replaces a draw that the host GPU-instances:"
              " its lights are placed once, at the first instance, not per instance."));
            localLight.applyTransform(inputTransforms.objectToWorld * (*inputTransforms.instancesToObject)[0]);
          } else {
            localLight.applyTransform(inputTransforms.objectToWorld);
          }

          RtLight* existingLight = (replacementInstance->prims.size() > i)
              ? replacementInstance->prims[i].getLight() : nullptr;
          if (existingLight != nullptr) {
            if (existingLight->getPrimInstanceOwner().getReplacementInstance() != replacementInstance) {
              ONCE(assert(false && "light in a replacementInstance believes it is owned by a different replacementInstance."));
            }
            m_lightManager.updateExternallyTrackedLight(existingLight, localLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(localLight);
            newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
          }
        }
      }
    }
  }

  void SceneManager::processReplacementGraphs(
      Rc<DxvkContext> ctx, const DrawCallState* input, const ReplacementBucket* pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    for (size_t i = 0; i < pReplacements->replacements.size(); i++) {
      auto&& replacement = pReplacements->replacements[i];
      if (replacement.type == AssetReplacement::eGraph) {
        bool hasGraph = (replacementInstance->prims.size() > i) &&
                        (replacementInstance->prims[i].getGraph() != nullptr);
        if (!hasGraph) {
          if (!replacement.graphState) {
            Logger::err(str::format(
                "Graph prims missing graph state in mesh replacement.  mesh hash: ",
                std::hex, input->getHash(RtxOptions::geometryAssetHashRule())
            ));
            break;
          }
          GraphInstance* graphInstance = m_graphManager.addInstance(ctx, replacement.graphState);
          if (graphInstance) {
            graphInstance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, graphInstance, PrimInstance::Type::Graph);
          }
        }
      }
    }
  }

  void SceneManager::updateBufferCache(BlasEntry* pBlas) {
    ScopedCpuProfileZone();

    // Acquire or re-acquire a stable buffer-cache slot. If the buffer matches the
    // currently registered slot, this is a no-op (fast path for static geometry).
    // If any slot changed, propagate updated indices to all currently-linked instances
    // immediately so they are never stale.
    bool changed = false;
    auto updateSlot = [&](const RaytraceBuffer& buf, uint32_t& idx) {
      if (!buf.defined()) {
        if (idx != kSurfaceInvalidBufferIndex) {
          m_bufferCache.release(idx);
          idx = kSurfaceInvalidBufferIndex;
          changed = true;
        }
        return;
      }
      if (m_bufferCache.isRegistered(idx, buf)) {
        return;
      }
      if (idx != kSurfaceInvalidBufferIndex) {
        m_bufferCache.release(idx);
      }
      idx = m_bufferCache.acquire(buf);
      changed = true;
    };

    RaytraceGeometry& geo = pBlas->modifiedGeometryData;
    updateSlot(geo.indexBuffer,            geo.indexBufferIndex);
    updateSlot(geo.normalBuffer,           geo.normalBufferIndex);
    updateSlot(geo.color0Buffer,           geo.color0BufferIndex);
    updateSlot(geo.texcoordBuffer,         geo.texcoordBufferIndex);
    updateSlot(geo.positionBuffer,         geo.positionBufferIndex);
    updateSlot(geo.previousPositionBuffer, geo.previousPositionBufferIndex);

    if (changed) {
      syncLinkedInstances(pBlas);
    }
  }

  void SceneManager::syncLinkedInstances(BlasEntry* pBlas) {
    for (RtInstance* inst : pBlas->getLinkedInstances()) {
      inst->syncBufferIndicesFromBlas();
    }
  }

  void SceneManager::unregisterGeometryBuffers(RaytraceGeometry& geo) {
    auto releaseSlot = [&](uint32_t& idx) {
      if (idx != kSurfaceInvalidBufferIndex) {
        m_bufferCache.release(idx);
        idx = kSurfaceInvalidBufferIndex;
      }
    };
    releaseSlot(geo.indexBufferIndex);
    releaseSlot(geo.normalBufferIndex);
    releaseSlot(geo.color0BufferIndex);
    releaseSlot(geo.texcoordBufferIndex);
    releaseSlot(geo.positionBufferIndex);
    releaseSlot(geo.previousPositionBufferIndex);
  }

  void SceneManager::verifyAndReleaseBufferCache() {
    for (auto& [hash, entry] : m_drawCallCache.getEntries()) {
      unregisterGeometryBuffers(entry.modifiedGeometryData);
    }
    if (m_bufferCache.getActiveCount() != 0) {
      ONCE(Logger::err(str::format("RetainedBufferTable: ", m_bufferCache.getActiveCount(),
                                   " buffer slot(s) still registered after scene clear — registration/release imbalance")));
      ONCE(assert(false && "RetainedBufferTable: detected possible buffer leak."));
    }
  }

  void SceneManager::preserveInstance(
      RtInstance& instance,
      const DrawCallState* pInput) {
    ScopedCpuProfileZone();
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas == nullptr) {
      return;
    }

    instance.setFrameLastUpdated(m_device->getCurrentFrameId());

    // One-shot cleanup for the first preserve frame after a dynamic update; isPreservePath is
    // false on the dynamic path, so flipping it here doubles as the "first frame" signal.
    if (!instance.surface.isPreservePath) {
      instance.surface.isPreservePath = true;

      // The last dynamic update may have left hasMaterialChanged == true.
      instance.surface.hasMaterialChanged = false;

      // The last dynamic update may have left prevObjectToWorld != objectToWorld and
      // isStatic == false (e.g. after a transform-changing move()). Re-sync once.
      if (!instance.surface.isStatic) {
        instance.surface.prevObjectToWorld = instance.surface.objectToWorld;
        instance.surface.isStatic = true;
      }
    }

    // On the first preserve encounter per frame, release the previousPositionBuffer slot
    // so instances don't carry a stale previous-position index forward. Subsequent preserve
    // draws on the same BLAS this frame skip this (frameLastTouched already == current).
    if (pBlas->frameLastTouched != m_device->getCurrentFrameId()
        && pBlas->modifiedGeometryData.previousPositionBuffer.defined()) {
      pBlas->modifiedGeometryData.previousPositionBuffer = RaytraceBuffer();
      updateBufferCache(pBlas);
    }

    pBlas->frameLastTouched = m_device->getCurrentFrameId();

    const uint32_t surfaceMatIdx = instance.surface.surfaceMaterialIndex;

    // Ray Portal refresh on the preserve path. RayPortalManager::clear() wipes m_rayPortalInfos
    // every frame in endFrame, so processRayPortalData must repopulate the slot for any portal
    // instance still present. The cached RtSurfaceMaterial supplies the RayPortalIndex and the
    // resource-cache key processRayPortalData needs.
    if (instance.getMaterialType() == MaterialDataType::RayPortal &&
        surfaceMatIdx < m_surfaceMaterialCache.getTotalCount()) {
      const RtSurfaceMaterial& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[surfaceMatIdx];
      if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
        m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
      }
    }

    // m_billboards is cleared every frame in InstanceManager::onFrameEnd, so re-run the
    // billboard step to repopulate it and refresh m_firstBillboard / m_billboardCount for
    // this frame's unordered TLAS. pInput is null on the anti-culling preserve path;
    // those instances are off-screen and don't need portal-space billboards.
    if (pInput != nullptr && m_cameraManager.isCameraValid(CameraType::Main)) {
      m_instanceManager.refreshBillboardsForCurrentFrame(
          instance,
          pInput->cameraType,
          m_cameraManager.getMainCamera().getDirection(false));
    }
  }

  void SceneManager::syncPreservedReplacementMeshesState(
      const DrawCallState& input,
      const ReplacementBucket* pReplacements,
      ReplacementInstance* replacementInstance) {
    if (pReplacements == nullptr) {
      return;
    }
    // Push this frame's per-replacement DrawCallState into each prim's BlasEntry::input.
    // Preserve path: RtInstance state is unchanged from the last dynamic update; only BLAS
    // input tracks remix state. The construction matches drawReplacements() (both share
    // buildReplacementMeshDrawCallState) so dynamic and preserve paths feed identical
    // DrawCallStates into the BlasEntry.
    for (size_t i = 0; i < pReplacements->replacements.size(); i++) {
      if (replacementInstance->prims.size() <= i) {
        break;
      }
      RtInstance* instance = replacementInstance->prims[i].getInstance();
      if (instance == nullptr) {
        continue;
      }
      BlasEntry* pBlas = instance->getBlas();
      if (pBlas == nullptr) {
        continue;
      }
      std::optional<DrawCallState> newDrawCallState =
          SceneManager::buildReplacementMeshDrawCallState(input, pReplacements->replacements[i],
            replacementInstance, i, m_device->getCurrentFrameId());
      if (newDrawCallState.has_value()) {
        pBlas->input = *newDrawCallState;
      }
    }
  }

  const DrawCallState& SceneManager::preserveReplacementInstance(
      Rc<DxvkContext> ctx,
      const DrawCallState& input,
      const std::shared_ptr<const ReplacementBucket>& pReplacements,
      ReplacementInstance* replacementInstance) {
    ScopedCpuProfileZone();
    // Refresh BlasEntry::input with this frame's draw state BEFORE dispatching preserveInstance.
    // refreshBillboardsForCurrentFrame -> createBeams / createBillboards consult
    // pBlas->input.getGeometryData() and call mapPtr() on its RasterBuffer slices to read
    // CPU-side positions/texcoords. If pBlas->input is left over from a previous frame, those
    // slices may reference vertex memory that has been recycled (notably with useSharedHeap=off
    // in the bridge), giving zero-fill positions and tripping the normalize() assert in
    // createBeams. Refreshing here keeps the dynamic and preserve paths feeding the same
    // frame's geometry into billboard / beam creation.
    if (pReplacements != nullptr) {
      syncPreservedReplacementMeshesState(input, pReplacements.get(), replacementInstance);
    } else if (replacementInstance->prims.size() > 0) {
      RtInstance* inst = replacementInstance->prims[0].getInstance();
      if (inst != nullptr && inst->getBlas() != nullptr) {
        inst->getBlas()->input = input;
      }
    }

    // Re-register per-frame buffer cache indices (same order as dynamic path: buffers resolved first).
    // No MaterialData is threaded through: SceneManager::preserveInstance reads the cached
    // RtSurfaceMaterial via surfaceMaterialIndex (Ray Portals included), and InstanceManager
    // event handlers contract for a null material on the preserve path.
    preserveInstancesWithObjectPicking(
      replacementInstance->prims,
      input.drawCallID,
      [&](RtInstance& instance) {
        preserveInstance(instance, &input);
        m_instanceManager.preserveInstance(instance, input, nullptr);
      },
      [&] {
        trackObjectPickingMeta(input, input.drawCallID);
      });

    replacementInstance->recalculateBoundingBox(
        input.getTransformData().objectToWorld, &input.getGeometryData().boundingBox);
    return input;
  }

  SceneManager::ObjectCacheState SceneManager::onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas) {
    // This is a new object.
    ObjectCacheState result = processGeometryInfo<true>(ctx, drawCallState, pBlas);
    
    assert(result == ObjectCacheState::KBuildBVH);

    pBlas->frameLastUpdated = m_device->getCurrentFrameId();
    m_instanceManager.notifySceneChanged();

    return result;
  }
  
  SceneManager::ObjectCacheState SceneManager::onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas,
                                                                    bool rebuildTopologyInPlace) {
    if (pBlas->frameLastTouched == m_device->getCurrentFrameId()) {
      pBlas->cacheMaterial(drawCallState.getMaterialData());
      return SceneManager::ObjectCacheState::kUpdateInstance;
    }

    // An entry its owner may hand a draw
    // with other index / vertex counts, topology, index type or index data. Rebuild it in place from this draw, exactly
    // as a new entry would be built (processGeometryInfo<true>: fresh index / vertex buffers, no previous positions), and
    // force a full BLAS build for it (frameTopologyChanged).
    bool rebuiltInPlace = false;
    ObjectCacheState result = ObjectCacheState::kInvalid;
    if (rebuildTopologyInPlace) {
      const RasterGeometry& now = drawCallState.getGeometryData();
      const RasterGeometry& last = pBlas->input.getGeometryData();
      const bool topologyChanged =
        now.vertexCount != last.vertexCount || now.indexCount != last.indexCount || now.topology != last.topology ||
        now.indexBuffer.defined() != last.indexBuffer.defined() ||
        (now.indexBuffer.defined() && now.indexBuffer.indexType() != last.indexBuffer.indexType()) ||
        now.hashes[HashComponents::Indices] != pBlas->modifiedGeometryData.hashes[HashComponents::Indices] ||
        now.hashes[HashComponents::GeometryDescriptor] != pBlas->modifiedGeometryData.hashes[HashComponents::GeometryDescriptor];
      if (topologyChanged) {
        // Start from a default RaytraceGeometry, as a new entry does, keeping only the buffer-cache slots so
        // updateBufferCache (run by processGeometryInfo) releases or re-registers them.
        RaytraceGeometry& geometry = pBlas->modifiedGeometryData;
        RaytraceGeometry fresh;
        fresh.positionBufferIndex = geometry.positionBufferIndex;
        fresh.previousPositionBufferIndex = geometry.previousPositionBufferIndex;
        fresh.normalBufferIndex = geometry.normalBufferIndex;
        fresh.texcoordBufferIndex = geometry.texcoordBufferIndex;
        fresh.color0BufferIndex = geometry.color0BufferIndex;
        fresh.indexBufferIndex = geometry.indexBufferIndex;
        geometry = std::move(fresh);
        result = processGeometryInfo<true>(ctx, drawCallState, pBlas);
        rebuiltInPlace = true;
        if (m_opacityMicromapManager != nullptr) {
          for (const RtInstance* linked : pBlas->getLinkedInstances()) {
            m_opacityMicromapManager->onInstanceTopologyChanged(*linked);
          }
        }
        if (result == ObjectCacheState::KBuildBVH) {
          pBlas->frameTopologyChanged = m_device->getCurrentFrameId();
          pBlas->frameLastUpdated = m_device->getCurrentFrameId();
          m_instanceManager.notifySceneChanged();
        }
      }
    }

    // TODO: If mesh is static, no need to do any of the below, just use the existing modifiedGeometryData and set result to kInstanceUpdate.
    if (!rebuiltInPlace) {
      result = processGeometryInfo<false>(ctx, drawCallState, pBlas);
    }

    // We dont expect to hit the rebuild path here - since this would indicate an index buffer or other topological change, and that *should* trigger a new scene object (since the hash would change).
    // Except for an in-place rebuild (rebuildTopologyInPlace) above.
    assert(result != ObjectCacheState::KBuildBVH || rebuiltInPlace);

    if (result == ObjectCacheState::kUpdateBVH) {
      pBlas->frameLastUpdated = m_device->getCurrentFrameId();
      m_instanceManager.notifySceneChanged();
    }
    
    pBlas->clearMaterialCache();
    pBlas->input = drawCallState; // cache the draw state for the next time.
    return result;
  }
  
  void SceneManager::onInstanceAdded(RtInstance& instance) {
    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->linkInstance(&instance);
    }
  }

  void SceneManager::onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, const bool hasTransformChanged, const bool hasVerticesChanged, const bool isFirstUpdateThisFrame) {
    // The preserve path passes a null material and reuses RtSurfaceMaterial via
    // RtInstance::surface.surfaceMaterialIndex; the dynamic path always provides a material.
    if (instance.surface.isPreservePath) {
      assert(material == nullptr);
      return;
    }
    assert(material != nullptr);
    auto capturer = m_device->getCommon()->capturer();
    if (hasTransformChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
    }

    if (hasVerticesChanged) {
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::PositionsUpdate);
      capturer->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::NormalsUpdate);
    }
    
    // Create and bind the RT material
    uint32_t newMatIdx = kInvalidMaterialCacheIndex;
    const RtSurfaceMaterial& surfaceMaterial = createSurfaceMaterial(*material, drawCall, &newMatIdx);

    if(isFirstUpdateThisFrame) {
      const uint32_t oldMatIdx = instance.surface.surfaceMaterialIndex;
      m_instanceManager.bindMaterial(instance, surfaceMaterial);
      if (newMatIdx != oldMatIdx) {
        retainSurfaceMaterial(newMatIdx);
        releaseSurfaceMaterial(oldMatIdx);
      }
    }

    // Update portal
    if (surfaceMaterial.getType() == RtSurfaceMaterialType::RayPortal) {
      m_rayPortalManager.processRayPortalData(instance, surfaceMaterial);
    }
  }

  void SceneManager::retainSurfaceMaterial(uint32_t matIdx) {
    // Retain is always called with newMatIdx from createSurfaceMaterial, which must return
    // a valid in-bounds index. An out-of-bounds index here indicates a logic error upstream.
    if (matIdx >= m_surfaceMaterialCache.getTotalCount()) {
      Logger::err(str::format("[RTX] retainSurfaceMaterial: matIdx ", matIdx,
        " out of bounds (cache size ", m_surfaceMaterialCache.getTotalCount(),
        "; createSurfaceMaterial returned invalid index"));
      assert(false && "retainSurfaceMaterial: matIdx out of bounds");
      return;
    }
    auto& textureManager = m_device->getCommon()->getTextureManager();
    const RtSurfaceMaterial& mat = m_surfaceMaterialCache.getObjectTable()[matIdx];
    mat.forEachTextureIndex([&](uint32_t texIdx) {
      textureManager.retainTexture(texIdx);
    });
    if (mat.getType() == RtSurfaceMaterialType::Opaque) {
      const RtOpaqueSurfaceMaterial& opaque = mat.getOpaqueSurfaceMaterial();
      if (opaque.hasValidDisplacement()) {
        ++m_activePOMCount;
      }
      // Textures of the terrain-layers extension entry.
      const uint32_t terrainLayersIdx = opaque.getTerrainLayersMaterialIndex();
      if (terrainLayersIdx != SURFACE_INDEX_INVALID &&
          terrainLayersIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        m_surfaceMaterialExtensionCache.getObjectTable()[terrainLayersIdx].forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.retainTexture(texIdx);
        });
      }
      // Live counts for the frame feature mask (rtx_fork_frame_features.h).
      if (opaque.getEnableThinFilm()) {
        ++m_thinFilmCount;
      }
      if (opaque.getIsRaytracedRenderTarget()) {
        ++m_raytracedRenderTargetCount;
      }
      const uint32_t subsurfaceIdx = opaque.getSubsurfaceMaterialIndex();
      if (subsurfaceIdx != SURFACE_INDEX_INVALID &&
          subsurfaceIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        const RtSurfaceMaterial& extMat = m_surfaceMaterialExtensionCache.getObjectTable()[subsurfaceIdx];
        extMat.forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.retainTexture(texIdx);
        });
        if (extMat.getType() == RtSurfaceMaterialType::Subsurface) {
          const float radiusScale = extMat.getSubsurfaceMaterial().getSubsurfaceRadiusScale();
          if (radiusScale > 0.0f)      ++m_sssCount;
          else if (radiusScale < 0.0f) ++m_thinOpaqueCount;
        }
      }
    } else if (mat.getType() == RtSurfaceMaterialType::Translucent) {
      ++m_translucentCount;
    } else if (mat.getType() == RtSurfaceMaterialType::RayPortal) {
      ++m_rayPortalCount;
    }
  }

  void SceneManager::releaseSurfaceMaterial(uint32_t matIdx) {
    // Out-of-bounds covers instances that were never bound (surfaceMaterialIndex ==
    // kSurfaceInvalidSurfaceMaterialIndex). Renderer-created instances skip
    // onInstanceDestroyedCallback entirely so they never reach here.
    if (matIdx >= m_surfaceMaterialCache.getTotalCount()) {
      return;
    }
    auto& textureManager = m_device->getCommon()->getTextureManager();
    const RtSurfaceMaterial& mat = m_surfaceMaterialCache.getObjectTable()[matIdx];
    mat.forEachTextureIndex([&](uint32_t texIdx) {
      textureManager.releaseTexture(texIdx);
    });
    // Mirror of the live counts taken in retainSurfaceMaterial.
    auto releaseLiveCount = [](uint32_t& count, const char* name) {
      if (count == 0) {
        Logger::err(str::format("[RTX] releaseSurfaceMaterial: ", name, " count underflow (mismatched retain/release)"));
      } else {
        --count;
      }
    };
    if (mat.getType() == RtSurfaceMaterialType::Opaque) {
      const RtOpaqueSurfaceMaterial& opaque = mat.getOpaqueSurfaceMaterial();
      if (opaque.hasValidDisplacement()) {
        if (m_activePOMCount == 0) {
          Logger::err("[RTX] releaseSurfaceMaterial: POM count underflow (mismatched retain/release)");
          assert(false && "releaseSurfaceMaterial: POM count underflow");
        } else {
          --m_activePOMCount;
        }
      }
      // Textures of the terrain-layers extension entry.
      const uint32_t terrainLayersIdx = opaque.getTerrainLayersMaterialIndex();
      if (terrainLayersIdx != SURFACE_INDEX_INVALID &&
          terrainLayersIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        m_surfaceMaterialExtensionCache.getObjectTable()[terrainLayersIdx].forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.releaseTexture(texIdx);
        });
      }
      if (opaque.getEnableThinFilm()) {
        releaseLiveCount(m_thinFilmCount, "thin-film");
      }
      if (opaque.getIsRaytracedRenderTarget()) {
        releaseLiveCount(m_raytracedRenderTargetCount, "raytraced-render-target");
      }
      const uint32_t subsurfaceIdx = opaque.getSubsurfaceMaterialIndex();
      if (subsurfaceIdx != SURFACE_INDEX_INVALID &&
          subsurfaceIdx < m_surfaceMaterialExtensionCache.getTotalCount()) {
        const RtSurfaceMaterial& extMat = m_surfaceMaterialExtensionCache.getObjectTable()[subsurfaceIdx];
        extMat.forEachTextureIndex([&](uint32_t texIdx) {
          textureManager.releaseTexture(texIdx);
        });
        if (extMat.getType() == RtSurfaceMaterialType::Subsurface) {
          const float radiusScale = extMat.getSubsurfaceMaterial().getSubsurfaceRadiusScale();
          if (radiusScale > 0.0f) {
            if (m_sssCount == 0) {
              Logger::err("[RTX] releaseSurfaceMaterial: SSS count underflow (mismatched retain/release)");
              assert(false && "releaseSurfaceMaterial: SSS count underflow");
            } else {
              --m_sssCount;
            }
          } else if (radiusScale < 0.0f) {
            if (m_thinOpaqueCount == 0) {
              Logger::err("[RTX] releaseSurfaceMaterial: thin-opaque count underflow (mismatched retain/release)");
              assert(false && "releaseSurfaceMaterial: thin-opaque count underflow");
            } else {
              --m_thinOpaqueCount;
            }
          }
        }
      }
    } else if (mat.getType() == RtSurfaceMaterialType::Translucent) {
      releaseLiveCount(m_translucentCount, "translucent");
    } else if (mat.getType() == RtSurfaceMaterialType::RayPortal) {
      releaseLiveCount(m_rayPortalCount, "ray-portal");
    }
  }

  void SceneManager::onInstanceDestroyed(RtInstance& instance) {
    releaseSurfaceMaterial(instance.surface.surfaceMaterialIndex);

    // Evict from the AccelManager bucket cache to prevent stale pointer ABA issues.
    m_accelManager.removeInstanceFromBucketCache(&instance);

    BlasEntry* pBlas = instance.getBlas();
    if (pBlas != nullptr) {
      pBlas->unlinkInstance(&instance);
    }
  }

  // Helper to populate the texture cache with this resource (and patch sampler if required for texture)
  // If 'inout_samplerFeedbackStamp' is non-null and still INVALID, it is seeded from the
  // first present texture; that stamp is then reused so subsequent textures in the same
  // material are grouped together for sampler-feedback streaming.
  void SceneManager::trackTexture(const TextureRef& inputTexture,
                                  uint32_t& textureIndex,
                                  bool hasTexcoords,
                                  bool async,
                                  uint16_t* inout_samplerFeedbackStamp) {
    if (!hasTexcoords) {
      ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Trying to bind a texture to a mesh without UVs.  Was this intended?")));
      return;
    }

    if (inout_samplerFeedbackStamp != nullptr && *inout_samplerFeedbackStamp == SAMPLER_FEEDBACK_INVALID && inputTexture.getManagedTexture() != nullptr) {
      *inout_samplerFeedbackStamp = inputTexture.getManagedTexture()->m_samplerFeedbackStamp;
    }

    const uint16_t stamp = inout_samplerFeedbackStamp != nullptr ? *inout_samplerFeedbackStamp : SAMPLER_FEEDBACK_INVALID;
    auto& textureManager = m_device->getCommon()->getTextureManager();
    textureManager.addTexture(inputTexture, stamp, async, textureIndex);
  }

  RtInstance* SceneManager::processDrawCallState(const Rc<DxvkContext>& ctx, const DrawCallState& drawCallState, const MaterialData& renderMaterialData, ReplacementInstance& replacementInstance, RtInstance* existingInstance, const RtxParticleSystemDesc* pParticleSystemDesc) {
    ScopedCpuProfileZone();

    if (renderMaterialData.getIgnored()) {
      return nullptr;
    }

    // Clear the preserve-path flag at the start of a dynamic update.
    if (existingInstance != nullptr) {
      existingInstance->surface.isPreservePath = false;
    }

    // [PerfCpu] apiDraws split (per-frame DrawInstance only, while [PerfCpu] logging is on).
    const bool timeApiDraw = m_timeApiDraws;
    const auto apiDrawStart = timeApiDraw ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    ObjectCacheState result = ObjectCacheState::kInvalid;
    BlasEntry* pBlas = nullptr;
    const bool cacheExisted = m_drawCallCache.get(drawCallState, &pBlas) == DrawCallCache::CacheState::kExisted;
    if (D3d9Identity::statsActive()) {
      D3d9Identity::noteCacheLookup(cacheExisted);
    }
    if (cacheExisted) {
      result = onSceneObjectUpdated(ctx, drawCallState, pBlas);
    } else {
      result = onSceneObjectAdded(ctx, drawCallState, pBlas);
    }
    
    assert(pBlas != nullptr);
    assert(result != ObjectCacheState::kInvalid);

    // Update the input state, so we always have a reference to the original draw call state
    pBlas->frameLastTouched = m_device->getCurrentFrameId();

    // Generate smooth normals for geometry that is flagged via the SmoothNormals texture category.
    // This is useful for older D3D9 games where geometry may lack smooth normals, especially
    // when using the VertexShader Capture mechanism. The smooth normals are computed on the GPU
    // from the triangle mesh (area-weighted) and written into the normal buffer.
    // Only dispatch on BVH build/update - for static geometry, positions don't change so
    // the normals computed on the first pass remain valid for subsequent frames.
    if (drawCallState.getCategoryFlags().test(InstanceCategories::SmoothNormals) &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSmoothNormals(ctx, drawCallState.getGeometryData(), pBlas->modifiedGeometryData);
      pBlas->modifiedGeometryData.smoothNormalsApplied = true;
      // dispatchSmoothNormals() changes the vertex buffers, which need to be re-synced.
      syncLinkedInstances(pBlas);
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
      m_instanceManager.notifySceneChanged();
    }

    if (drawCallState.getSkinningState().numBones > 0 &&
        drawCallState.getGeometryData().numBonesPerVertex > 0 &&
        (result == ObjectCacheState::KBuildBVH || result == ObjectCacheState::kUpdateBVH)) {
      m_device->getCommon()->metaGeometryUtils().dispatchSkinning(drawCallState, pBlas->modifiedGeometryData);
      pBlas->frameLastUpdated = pBlas->frameLastTouched;
      m_instanceManager.notifySceneChanged();
    }

    // rtx.retainedPinSteadyEntries: retained entries pinned on this geometry skipped the check that
    // sends an entry whose geometry another draw updated this frame to a full replay.
    if (pBlas->frameLastUpdated == m_device->getCurrentFrameId()) {
      wakeRetainedPinsOf(*pBlas);
      // rtx.retained.useV2: entries holding this geometry would now fail their preserve check.
      m_retainedV2->onGeometryUpdated(*pBlas);
    }

    const auto apiDrawGeometryEnd = timeApiDraw ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (timeApiDraw) {
      m_apiDrawGeometryMs += std::chrono::duration<double, std::milli>(apiDrawGeometryEnd - apiDrawStart).count();
    }

    // Note: The material data can be modified in instance manager
    RtInstance* instance = m_instanceManager.processSceneObject(m_cameraManager, m_rayPortalManager, *pBlas, drawCallState, renderMaterialData, existingInstance);
    if (timeApiDraw) {
      m_apiDrawInstanceMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - apiDrawGeometryEnd).count();
    }

    // Check if a light should be created for this Material
    if (instance && RtxOptions::shouldConvertToLight(drawCallState.getMaterialData().getHash())) {
      createEffectLight(ctx, drawCallState, instance);
    }

    if (instance) {
      trackObjectPickingMeta(drawCallState, instance->surface.objectPickingValue);
    }

    // Priority ordering for particle system descriptors is: Mesh, Material, Texture.  This matches the implementation in toolkit.
    // By this point, pParticleSystemDesc will contain the information from a mesh replacement (if one exists), so we just handle
    // materials replacements, and texture taggin categories below.
    RtxParticleSystemDesc globalParticleDesc; // Storage for global desc if needed
    if (!pParticleSystemDesc) {
      pParticleSystemDesc = renderMaterialData.getParticleSystemDesc();
    }
    if (!pParticleSystemDesc && drawCallState.getCategoryFlags().test(InstanceCategories::ParticleEmitter)) {
      globalParticleDesc = RtxParticleSystemManager::createGlobalParticleSystemDesc();
      pParticleSystemDesc = &globalParticleDesc;
    }
    if (instance && pParticleSystemDesc) {
      RtxParticleSystemManager& particleSystem = device()->getCommon()->metaParticleSystem();
      particleSystem.spawnParticles(ctx.ptr(), *pParticleSystemDesc, instance->getVectorIdx(), instance->getId(), drawCallState, renderMaterialData);

      if (pParticleSystemDesc->hideEmitter) {
        instance->setHidden(true);
      }

      replacementInstance.dirtyFlags.set(ReplacementInstance::DirtyFlag::ParticleSystem);
    }

    return instance; 
  }

  void SceneManager::trackObjectPickingMeta(
      const DrawCallState& drawCallState,
      ObjectPickingValue objectPickingValue) {
    const bool objectPickingActive = m_device->getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (!objectPickingActive || !g_allowMappingLegacyHashToObjectPickingValue) {
      return;
    }
    XXH64_hash_t h = drawCallState.getMaterialData().getColorTexture().getImageHash();
    if (h == kEmptyHash) {
      // A Remix API draw: no legacy textures, its albedo's hash rides on the draw.
      h = drawCallState.externalPickingTextureHash;
    }
    trackObjectPickingMeta(objectPickingValue, h, drawCallState.getMaterialData().getColorTexture2().getImageHash());
  }

  void SceneManager::trackObjectPickingMeta(
      ObjectPickingValue objectPickingValue,
      XXH64_hash_t textureHash,
      XXH64_hash_t textureHash2) {
    const bool objectPickingActive = m_device->getCommon()->getResources().getRaytracingOutput()
      .m_primaryObjectPicking.isValid();
    if (!objectPickingActive || !g_allowMappingLegacyHashToObjectPickingValue) {
      return;
    }

    auto meta = DrawCallMetaInfo {};
    if (textureHash != kEmptyHash) {
      meta.legacyTextureHash = textureHash;
    }
    if (textureHash2 != kEmptyHash) {
      meta.legacyTextureHash2 = textureHash2;
    }

    std::lock_guard lock { m_drawCallMeta.mutex };
    auto [iter, isNew] = m_drawCallMeta.infos[m_drawCallMeta.ticker].emplace(objectPickingValue, meta);
    ONCE_IF_FALSE(isNew, Logger::warn(
      "Found multiple draw calls with the same \'objectPickingValue\'. "
      "Ignoring further MetaInfo-s, some objects might be not be available through object picking"));
  }

  ObjectPickingValue SceneManager::externalObjectPickingValue(const DrawCallState& drawCall,
                                                              ReplacementInstance& replacementInstance) {
    if (drawCall.drawCallID != 0) {
      return drawCall.drawCallID;
    }
    if (replacementInstance.externalPickingValue == 0) {
      // Upper half of the range: D3D9 draw call IDs count up from 0 each frame. Never
      // 0xFFFFFFFF, the picking image's clear value.
      constexpr ObjectPickingValue kExternalPickingBase = 0x80000000u;
      replacementInstance.externalPickingValue = kExternalPickingBase | (m_nextExternalPickingValue++ % 0x7FFFFFFFu);
    }
    return replacementInstance.externalPickingValue;
  }

  // Texture formats by name for [MaterialDump] (the ones FNV content and the wrapper's uploads use).
  static std::string forkVkFormatName(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_SRGB: return "R8G8B8A8_SRGB";
    case VK_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB: return "B8G8R8A8_SRGB";
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return "A8B8G8R8_UNORM";
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return "A8B8G8R8_SRGB";
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return "BC1_RGB_UNORM";
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return "BC1_RGB_SRGB";
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return "BC1_RGBA_UNORM";
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return "BC1_RGBA_SRGB";
    case VK_FORMAT_BC2_UNORM_BLOCK: return "BC2_UNORM";
    case VK_FORMAT_BC2_SRGB_BLOCK: return "BC2_SRGB";
    case VK_FORMAT_BC3_UNORM_BLOCK: return "BC3_UNORM";
    case VK_FORMAT_BC3_SRGB_BLOCK: return "BC3_SRGB";
    case VK_FORMAT_BC4_UNORM_BLOCK: return "BC4_UNORM";
    case VK_FORMAT_BC5_UNORM_BLOCK: return "BC5_UNORM";
    case VK_FORMAT_BC7_UNORM_BLOCK: return "BC7_UNORM";
    case VK_FORMAT_BC7_SRGB_BLOCK: return "BC7_SRGB";
    case VK_FORMAT_R16G16B16A16_SFLOAT: return "R16G16B16A16_SFLOAT";
    case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return "B4G4R4A4_UNORM";
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return "R5G6B5_UNORM";
    case VK_FORMAT_B5G6R5_UNORM_PACK16: return "B5G6R5_UNORM";
    default: return str::format("vk", static_cast<int>(format));
    }
  }

  const RtSurfaceMaterial& SceneManager::createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                               const DrawCallState& drawCallState,
                                                               uint32_t* out_indexInCache) {
    ScopedCpuProfileZone();
    const bool hasTexcoords = drawCallState.hasTextureCoordinates();
    const auto renderMaterialDataType = renderMaterialData.getType();

    // We're going to use this to create a modified sampler for replacement textures.
    // Legacy and replacement materials should follow same filtering but due to lack of override capability per texture
    // legacy textures use original sampler to stay true to the original intent while replacements use more advanced filtering
    // for better quality by default.
    const Rc<DxvkSampler>& samplerOverride = renderMaterialData.getSamplerOverride();
    Rc<DxvkSampler> sampler = samplerOverride;
    // If the original sampler if valid and there isnt an override sampler
    // go ahead with patching and maybe merging the sampler states
    if (samplerOverride == nullptr && drawCallState.getMaterialData().getSampler().ptr() != nullptr) {
      DxvkSamplerCreateInfo samplerInfo = drawCallState.getMaterialData().getSampler()->info(); // Use sampler create info struct as convenience
      renderMaterialData.populateSamplerInfo(samplerInfo);

      sampler = patchSampler(samplerInfo.magFilter,
                             samplerInfo.addressModeU, samplerInfo.addressModeV, samplerInfo.addressModeW,
                             samplerInfo.borderColor);
    }
    if (drawCallState.isEye()) {
      // force eye whites and iris to not repeat
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        {}
      );
    }
    uint32_t samplerIndex = trackSampler(sampler);
    uint32_t samplerIndex2 = UINT32_MAX;
    if (renderMaterialDataType == MaterialDataType::RayPortal) {
      samplerIndex2 = trackSampler(drawCallState.getMaterialData().getSampler2());
    }

    XXH64_hash_t preCreationHash = renderMaterialData.getHash();
    preCreationHash = XXH64(&samplerIndex, sizeof(samplerIndex), preCreationHash);
    preCreationHash = XXH64(&samplerIndex2, sizeof(samplerIndex2), preCreationHash);
    preCreationHash = XXH64(&hasTexcoords, sizeof(hasTexcoords), preCreationHash);
    preCreationHash = XXH64(&drawCallState.isUsingRaytracedRenderTarget, sizeof(drawCallState.isUsingRaytracedRenderTarget), preCreationHash);
    const bool isHairCard = drawCallState.testCategoryFlags(InstanceCategories::HairCards);
    preCreationHash = XXH64(&isHairCard, sizeof(isHairCard), preCreationHash);

    // For Opaque materials, fold in a bitmask of which texture slots are populated. MaterialData::getHash()
    // sums TextureRef::getImageHash() across slots, but render-target-backed TextureRefs (notably
    // TerrainBaker cascades) have a zero image hash, so two materials that differ only in cascade
    // composition (e.g. "albedo only" vs "albedo + normal + roughness") collide on getHash() and the
    // per-frame cache would serve the earlier "albedo only" entry to later draws that have valid
    // secondary cascades.
    if (renderMaterialDataType == MaterialDataType::Opaque) {
      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();
      uint32_t texturePresenceMask = 0;
      texturePresenceMask |= opaqueMaterialData.getAlbedoOpacityTexture().isImageEmpty()             ? 0u : (1u << 0);
      texturePresenceMask |= opaqueMaterialData.getSecondaryTexture().isImageEmpty()                 ? 0u : (1u << 1);
      texturePresenceMask |= opaqueMaterialData.getNormalTexture().isImageEmpty()                    ? 0u : (1u << 2);
      texturePresenceMask |= opaqueMaterialData.getTangentTexture().isImageEmpty()                   ? 0u : (1u << 3);
      texturePresenceMask |= opaqueMaterialData.getHeightTexture().isImageEmpty()                    ? 0u : (1u << 4);
      texturePresenceMask |= opaqueMaterialData.getRoughnessTexture().isImageEmpty()                 ? 0u : (1u << 5);
      texturePresenceMask |= opaqueMaterialData.getMetallicTexture().isImageEmpty()                  ? 0u : (1u << 6);
      texturePresenceMask |= opaqueMaterialData.getEmissiveColorTexture().isImageEmpty()             ? 0u : (1u << 7);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceTransmittanceTexture().isImageEmpty()   ? 0u : (1u << 8);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceThicknessTexture().isImageEmpty()       ? 0u : (1u << 9);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture().isImageEmpty() ? 0u : (1u << 10);
      texturePresenceMask |= opaqueMaterialData.getSubsurfaceRadiusTexture().isImageEmpty()          ? 0u : (1u << 11);
      // The terrain layers' normal maps too (bits 12-19): their gloss mask is decided from the resolved
      // views below, so a layer normal that resolves later rebuilds the material.
      if (const auto& terrainLayers = opaqueMaterialData.getTerrainLayers(); terrainLayers) {
        const uint32_t layerCount = std::min(terrainLayers->layerCount, TerrainLayersMaterialData::kMaxLayers);
        for (uint32_t i = 0; i < layerCount; ++i) {
          texturePresenceMask |= terrainLayers->normalTextures[i].isImageEmpty() ? 0u : (1u << (12 + i));
        }
      }
      preCreationHash = XXH64(&texturePresenceMask, sizeof(texturePresenceMask), preCreationHash);

      // Folding in the sRGB-linearization toggle makes a runtime change of rtx.linearizeSrgbTextures rebuild the
      // cached opaque materials with the new albedo / emissive sRGB flags. Without it the preCreationHash cache
      // below would keep serving materials built under the old setting.
      const uint32_t srgbLinearizeToggle = RtxOptions::linearizeSrgbTextures() ? 1u : 0u;
      preCreationHash = XXH64(&srgbLinearizeToggle, sizeof(srgbLinearizeToggle), preCreationHash);
      // The same for the spec/gloss metalness toggle.
      const uint32_t specularMetalToggle = RtxOptions::enableSpecularMetal() ? 1u : 0u;
      preCreationHash = XXH64(&specularMetalToggle, sizeof(specularMetalToggle), preCreationHash);
      // And the albedo tint toggle (rtx.hostImport.enableMaterialAlbedoTint).
      const uint32_t albedoTintToggle = RtxOptions::HostImport::enableMaterialAlbedoTint() ? 1u : 0u;
      preCreationHash = XXH64(&albedoTintToggle, sizeof(albedoTintToggle), preCreationHash);
      // And for rtx.subsurface.transmittanceFollowsAlbedo.
      const uint32_t transmittanceFollowsAlbedoToggle = RtxOptions::SubsurfaceScattering::transmittanceFollowsAlbedo() ? 1u : 0u;
      preCreationHash = XXH64(&transmittanceFollowsAlbedoToggle, sizeof(transmittanceFollowsAlbedoToggle), preCreationHash);
      // And the emissive scale (rtx.emissiveIntensity x rtx.hostCalibration.*), so calibrating is live.
      const float emissiveScale = RtxOptions::emissiveIntensity() * HostCalibration::emissiveScale();
      preCreationHash = XXH64(&emissiveScale, sizeof(emissiveScale), preCreationHash);
    }

    auto iter = m_preCreationSurfaceMaterialMap.find(preCreationHash);
    if (iter != m_preCreationSurfaceMaterialMap.end()) {
      if (out_indexInCache) {
        *out_indexInCache = iter->second;
      }
      return m_surfaceMaterialCache.at(iter->second);
    }

    std::optional<RtSurfaceMaterial> surfaceMaterial;

    if (renderMaterialDataType == MaterialDataType::Opaque || drawCallState.isUsingRaytracedRenderTarget) {
      uint32_t albedoOpacityTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t secondaryTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t tangentTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t heightTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t roughnessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t metallicTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceMaterialIndex = SURFACE_INDEX_INVALID;
      uint32_t subsurfaceTransmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceThicknessTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      uint32_t subsurfaceSingleScatteringAlbedoTextureIndex = kSurfaceMaterialInvalidTextureIndex;

      float anisotropy;
      float emissiveIntensity;
      Vector4 albedoOpacityConstant;
      float roughnessConstant;
      float metallicConstant;
      Vector3 emissiveColorConstant;
      bool enableEmissive;
      bool thinFilmEnable = false;
      bool alphaIsThinFilmThickness = false;
      float thinFilmThicknessConstant = 0.0f;
      float displaceIn = 0.0f;
      float displaceOut = 0.0f;
      bool isUsingRaytracedRenderTarget = drawCallState.isUsingRaytracedRenderTarget;
      uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

      Vector3 subsurfaceTransmittanceColor(0.0f, 0.0f, 0.0f);
      float subsurfaceMeasurementDistance = 0.0f;
      Vector3 subsurfaceSingleScatteringAlbedo(0.0f, 0.0f, 0.0f);
      float subsurfaceVolumetricAnisotropy = 0.0f;

      float subsurfaceRadiusScale = 0.0f;
      float subsurfaceMaxSampleRadius = 0.0f;

      bool ignoreAlphaChannel = false;

      constexpr Vector4 kWhiteModeAlbedo = Vector4(0.7f, 0.7f, 0.7f, 1.0f);

      const auto& opaqueMaterialData = renderMaterialData.getOpaqueMaterialData();

      if (RtxOptions::useWhiteMaterialMode()) {
        albedoOpacityConstant = kWhiteModeAlbedo;
        metallicConstant = 0.f;
        roughnessConstant = 1.f;
      } else {
        trackTexture(opaqueMaterialData.getAlbedoOpacityTexture(), albedoOpacityTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getRoughnessTexture(), roughnessTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getMetallicTexture(), metallicTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(opaqueMaterialData.getSecondaryTexture(), secondaryTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

        albedoOpacityConstant.xyz() = opaqueMaterialData.getAlbedoConstant();
        albedoOpacityConstant.w = opaqueMaterialData.getOpacityConstant();
        metallicConstant = opaqueMaterialData.getMetallicConstant();
        roughnessConstant = opaqueMaterialData.getRoughnessConstant();
      }

      trackTexture(opaqueMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getTangentTexture(), tangentTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      // In SCCR mode the height map is replaced by its derived height+cone texture, where it has one.
      bool pomConeAvailable = false;
      const bool displaced = opaqueMaterialData.getDisplaceIn() > 0.0f || opaqueMaterialData.getDisplaceOut() > 0.0f;
      trackTexture(m_device->getCommon()->metaPomConeMaps().resolveHeight(opaqueMaterialData.getHeightTexture(), displaced, pomConeAvailable),
                   heightTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
      trackTexture(opaqueMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

      // x the host calibration (rtx_host_calibration.h): the game's emissive units to physical.
      emissiveIntensity = opaqueMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity() * HostCalibration::emissiveScale();
      emissiveColorConstant = opaqueMaterialData.getEmissiveColorConstant();
      enableEmissive = opaqueMaterialData.getEnableEmission();
      anisotropy = opaqueMaterialData.getAnisotropyConstant();
        
      thinFilmEnable = opaqueMaterialData.getEnableThinFilm();
      // See hasCreatedThinFilmMaterial().
      if (thinFilmEnable) {
        m_thinFilmMaterialCreated.store(true, std::memory_order_relaxed);
      }
      alphaIsThinFilmThickness = opaqueMaterialData.getAlphaIsThinFilmThickness();
      thinFilmThicknessConstant = opaqueMaterialData.getThinFilmThicknessConstant();
      displaceIn = opaqueMaterialData.getDisplaceIn();
      displaceOut = opaqueMaterialData.getDisplaceOut();

      ignoreAlphaChannel = opaqueMaterialData.getIgnoreAlphaChannel();

      // Per-material albedo tint (remixapi_MaterialInfoOpaqueAlbedoTintEXT), carried by the surface
      // material extension entry. Only meaningful with an albedo texture; white mode drops it with the texture.
      const Vector3 albedoTint = opaqueMaterialData.getAlbedoTint();
      const bool hasAlbedoTint = RtxOptions::HostImport::enableMaterialAlbedoTint()
        && albedoOpacityTextureIndex != kSurfaceMaterialInvalidTextureIndex
        && (albedoTint.x != 1.f || albedoTint.y != 1.f || albedoTint.z != 1.f);

      subsurfaceMeasurementDistance = opaqueMaterialData.getSubsurfaceMeasurementDistance() * RtxOptions::SubsurfaceScattering::surfaceThicknessScale();

      const bool isSubsurfaceScatteringDiffusionProfile = opaqueMaterialData.getSubsurfaceDiffusionProfile();

      // Conservative (also set for diffusion-profile materials that carry a distance).
      if (RtxOptions::SubsurfaceScattering::enableThinOpaque() && subsurfaceMeasurementDistance > 0.0f) {
        m_thinOpaqueMaterialCreated.store(true, std::memory_order_relaxed);
      }

      if ((RtxOptions::SubsurfaceScattering::enableThinOpaque()       && subsurfaceMeasurementDistance > 0.0f) ||
          (RtxOptions::SubsurfaceScattering::enableDiffusionProfile() && isSubsurfaceScatteringDiffusionProfile)) {

        subsurfaceTransmittanceColor = opaqueMaterialData.getSubsurfaceTransmittanceColor();
        subsurfaceVolumetricAnisotropy = opaqueMaterialData.getSubsurfaceVolumetricAnisotropy();

        if (isSubsurfaceScatteringDiffusionProfile) {
          // NOTE: reuse of the variable!
          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceRadius(); 
          subsurfaceMaxSampleRadius = std::max(0.F, opaqueMaterialData.getSubsurfaceMaxSampleRadius());
          subsurfaceRadiusScale = std::max(opaqueMaterialData.getSubsurfaceRadiusScale(), 1e-5f);
          assert(subsurfaceRadiusScale > 0);
        } else /* if thin opaque */ {
          assert(subsurfaceMeasurementDistance > 0);

          subsurfaceSingleScatteringAlbedo = opaqueMaterialData.getSubsurfaceSingleScatteringAlbedo();
          subsurfaceMaxSampleRadius = 0;
          subsurfaceRadiusScale = -1;
          assert(subsurfaceRadiusScale < 0);  // if < 0, then shaders assume that
                                              // this material is not SubsurfaceScatter, but just SingleScatter
                                              // same here, but <0.F
        }

        if (RtxOptions::SubsurfaceScattering::enableTextureMaps()) {
          trackTexture(opaqueMaterialData.getSubsurfaceTransmittanceTexture(), subsurfaceTransmittanceTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

          if (isSubsurfaceScatteringDiffusionProfile) {
            // NOTE: reuse of 'subsurfaceSingleScatteringAlbedoTextureIndex' variable!
            trackTexture(opaqueMaterialData.getSubsurfaceRadiusTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
          } else {
            trackTexture(opaqueMaterialData.getSubsurfaceSingleScatteringAlbedoTexture(), subsurfaceSingleScatteringAlbedoTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
            trackTexture(opaqueMaterialData.getSubsurfaceThicknessTexture(), subsurfaceThicknessTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
          }
        }

        // rtx.subsurface.transmittanceFollowsAlbedo. Under a diffusion profile the transmittance
        // texture replaces the diffuse lobe, so when it IS the albedo texture it must be read like the albedo: no second
        // gamma decode of an sRGB view, and the albedo tint applied (see subsurfaceMaterialInteractionCreate).
        uint8_t transmittanceFlags = 0;
        if (isSubsurfaceScatteringDiffusionProfile &&
            RtxOptions::SubsurfaceScattering::transmittanceFollowsAlbedo() &&
            subsurfaceTransmittanceTextureIndex != kSurfaceMaterialInvalidTextureIndex &&
            subsurfaceTransmittanceTextureIndex == albedoOpacityTextureIndex) {
          transmittanceFlags |= SURFACE_MATERIAL_EXTENSION_FLAG_TRANSMITTANCE_TINT;
          const DxvkImageView* transmittanceView = opaqueMaterialData.getSubsurfaceTransmittanceTexture().getImageView();
          if (RtxOptions::linearizeSrgbTextures() && transmittanceView != nullptr &&
              TextureUtils::isSRGB(transmittanceView->info().format)) {
            transmittanceFlags |= SURFACE_MATERIAL_EXTENSION_FLAG_TRANSMITTANCE_SRGB;
          }
        }

        const auto subsurfaceMaterial = RtSubsurfaceMaterial{
          subsurfaceTransmittanceTextureIndex,
          subsurfaceThicknessTextureIndex,
          subsurfaceSingleScatteringAlbedoTextureIndex,
          subsurfaceTransmittanceColor,
          subsurfaceMeasurementDistance,
          subsurfaceSingleScatteringAlbedo,
          subsurfaceVolumetricAnisotropy,
          subsurfaceRadiusScale,
          subsurfaceMaxSampleRadius,
          true, // hasSubsurface
          hasAlbedoTint,
          albedoTint,
          transmittanceFlags,
        };
        subsurfaceMaterialIndex = m_surfaceMaterialExtensionCache.track(subsurfaceMaterial);
      } else if (hasAlbedoTint) {
        // An extension entry that carries only the albedo tint.
        const auto tintOnlyMaterial = RtSubsurfaceMaterial{
          kSurfaceMaterialInvalidTextureIndex,
          kSurfaceMaterialInvalidTextureIndex,
          kSurfaceMaterialInvalidTextureIndex,
          Vector3(0.f), 0.f, Vector3(0.f), 0.f,
          0.f, // radius scale 0: counted as neither SSS nor thin opaque
          0.f,
          false, // hasSubsurface
          true,
          albedoTint,
        };
        subsurfaceMaterialIndex = m_surfaceMaterialExtensionCache.track(tintOnlyMaterial);
      }

      // Detect whether the albedo/emissive source textures use an sRGB VkFormat. If so, the sampler hardware
      // linearizes them on read, so the shader must skip its own gammaToLinear() to avoid double linearization.
      // With linearizeSrgbTextures() off the flags stay clear and the shader always applies the software
      // conversion. The resolved image-view format is available whenever the texture is loaded. An unloaded
      // texture reports isImageEmpty(), which feeds the material cache key, so the material is rebuilt with
      // the right flag once the texture resolves.
      const bool srgbLinearizeEnabled = RtxOptions::linearizeSrgbTextures();
      auto textureUsesSrgbFormat = [srgbLinearizeEnabled](const TextureRef& tex) -> bool {
        if (!srgbLinearizeEnabled) {
          return false;
        }
        const DxvkImageView* view = tex.getImageView();
        return view != nullptr && TextureUtils::isSRGB(view->info().format);
      };
      const bool albedoTextureIsSrgb = textureUsesSrgbFormat(opaqueMaterialData.getAlbedoOpacityTexture());
      const bool emissiveTextureIsSrgb = textureUsesSrgbFormat(opaqueMaterialData.getEmissiveColorTexture());

      // Textures imported from a host carry the host's encoding; the shader
      // decodes it instead of the host converting pixels on the CPU.
      uint16_t hostEncodingFlags = 0;
      if (opaqueMaterialData.getNormalTexture().getHostEncoding() == HostTextureEncoding::NormalTangentXY) {
        hostEncodingFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_NORMAL_TANGENT_XY;
      }
      // A texture uploaded without an encoding (host CPU path) can still be
      // declared a smoothness map by the material (spec/gloss extension).
      HostTextureEncoding roughnessEncoding = opaqueMaterialData.getRoughnessTexture().getHostEncoding();
      if (roughnessEncoding == HostTextureEncoding::None) {
        const uint8_t materialEncoding = opaqueMaterialData.getRoughnessEncoding();
        if (materialEncoding == uint8_t(HostTextureEncoding::SmoothnessInRed) ||
            materialEncoding == uint8_t(HostTextureEncoding::SmoothnessInGreen)) {
          roughnessEncoding = static_cast<HostTextureEncoding>(materialEncoding);
        }
      }
      switch (roughnessEncoding) {
      case HostTextureEncoding::SmoothnessInGreen:
        hostEncodingFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_SMOOTHNESS_IN_GREEN;
        [[fallthrough]];
      case HostTextureEncoding::SmoothnessInRed:
        hostEncodingFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_SMOOTHNESS;
        break;
      default:
        break;
      }
      // For a smoothness map the API's roughnessConstant is the floor. The
      // constant itself still stands in for the texture where it is skipped
      // (simplified NEE shading), so keep it at a mid value there.
      float roughnessFloor = 0.0f;
      if ((hostEncodingFlags & OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_SMOOTHNESS)
          && roughnessTextureIndex != kSurfaceMaterialInvalidTextureIndex) {
        roughnessFloor = std::clamp(roughnessConstant, 0.0f, 1.0f);
        roughnessConstant = std::max(roughnessFloor, 0.5f);
      }

      // Spec/gloss metalness: the red channel of a SMOOTHNESS_IN_GREEN map is
      // the specular level. Parameters are packed into the material's last
      // 32 bits (see surface_material.h); the flag uses the last free bit.
      uint32_t specularMetalPacked = 0;
      const float specularMetalStrength = opaqueMaterialData.getSpecularMetalStrength();
      if (RtxOptions::enableSpecularMetal() && specularMetalStrength > 0.0f
          && (hostEncodingFlags & OPAQUE_SURFACE_MATERIAL_FLAG_SMOOTHNESS_IN_GREEN)
          && roughnessTextureIndex != kSurfaceMaterialInvalidTextureIndex) {
        auto q = [](float v, float maxCode) {
          return uint32_t(std::clamp(v, 0.0f, 1.0f) * maxCode + 0.5f);
        };
        const Vector3& params = opaqueMaterialData.getSpecularMetalParams();
        const Vector3& tint = opaqueMaterialData.getSpecularMetalTint();
        const uint32_t strengthCode = q(specularMetalStrength, 15.0f);
        if (strengthCode != 0) {
          const uint32_t paramBits =
            q(params.x, 15.0f) | (q(params.y, 15.0f) << 4) | (q(params.z, 15.0f) << 8) | (strengthCode << 12);
          const uint32_t tintBits = q(tint.x, 31.0f) | (q(tint.y, 63.0f) << 5) | (q(tint.z, 31.0f) << 11);
          specularMetalPacked = paramBits | (tintBits << 16);
          hostEncodingFlags |= OPAQUE_SURFACE_MATERIAL_FLAG_METALLIC_FROM_SPECULAR;
        }
      }

      // FNV material protocol v2 shader flags and the terrain-layers extension entry.
      const uint32_t forkFnvFlags = opaqueMaterialData.getForkFnvFlags();
      uint32_t terrainLayersMaterialIndex = SURFACE_INDEX_INVALID;
      // Which layer normal maps carry a gloss alpha, decided here from the views the material is built
      // with (and ORed with what the API-side preload decided, which saw views that may not have been resolved).
      uint32_t layerGlossMask = 0;
      if (const auto& terrainLayers = opaqueMaterialData.getTerrainLayers();
          terrainLayers && terrainLayers->layerCount > 0 && !RtxOptions::useWhiteMaterialMode()) {
        std::array<uint32_t, RtTerrainLayersMaterial::kMaxLayers> albedoIndices;
        std::array<uint32_t, RtTerrainLayersMaterial::kMaxLayers> normalIndices;
        albedoIndices.fill(kSurfaceMaterialInvalidTextureIndex);
        normalIndices.fill(kSurfaceMaterialInvalidTextureIndex);
        static_assert(RtTerrainLayersMaterial::kMaxLayers == TerrainLayersMaterialData::kMaxLayers);
        const uint32_t layerCount = std::min(terrainLayers->layerCount, RtTerrainLayersMaterial::kMaxLayers);
        layerGlossMask = terrainLayers->normalHasAlphaMask;
        for (uint32_t i = 0; i < layerCount; ++i) {
          trackTexture(terrainLayers->albedoTextures[i], albedoIndices[i], hasTexcoords, true, &samplerFeedbackStamp);
          trackTexture(terrainLayers->normalTextures[i], normalIndices[i], hasTexcoords, true, &samplerFeedbackStamp);
          if (FnvMaterialProtocol::textureHasGlossAlpha(terrainLayers->normalTextures[i])) {
            layerGlossMask |= 1u << i;
          }
        }
        uint32_t weightIndex0 = kSurfaceMaterialInvalidTextureIndex;
        uint32_t weightIndex1 = kSurfaceMaterialInvalidTextureIndex;
        uint32_t tintIndex = kSurfaceMaterialInvalidTextureIndex;
        trackTexture(terrainLayers->weightTextures[0], weightIndex0, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(terrainLayers->weightTextures[1], weightIndex1, hasTexcoords, true, &samplerFeedbackStamp);
        trackTexture(terrainLayers->tintTexture, tintIndex, hasTexcoords, true, &samplerFeedbackStamp);
        // Weight and tint maps span the whole terrain block: bilinear, clamped (no wrap bleeding at the edges).
        const uint32_t weightSamplerIndex = trackSampler(patchSampler(
          VK_FILTER_LINEAR,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          {}));
        // Per-layer height maps go into their own extension entry. Only when the material asks for
        // displacement (OpaqueEXT displaceIn/displaceOut): the opaque material then borrows the first layer height map
        // as its heightTextureIndex (unless it has its own), which is what sets OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT
        // and routes the hit through every existing POM path; the shader (pomHeightSourceCreate) sees
        // heightMaterialIndex and marches the blended layer heights instead of that one map.
        std::array<uint32_t, RtTerrainLayersMaterial::kMaxLayers> heightIndices;
        heightIndices.fill(kSurfaceMaterialInvalidTextureIndex);
        uint32_t heightMaterialIndex = SURFACE_INDEX_INVALID;
        if (displaceIn > 0.0f || displaceOut > 0.0f) {
          uint32_t firstHeightIndex = kSurfaceMaterialInvalidTextureIndex;
          // In SCCR mode each layer height gets a CONSERVATIVE derived height+cone texture (its
          // own, even when the same map is also some material's single height): the blended march bounds the blend by
          // the layer cones, which holds for conservative cones only (pom_sccr_terrain.slangh). coneMask records which
          // layers got one; other modes keep the source textures and coneMask 0.
          uint32_t coneMask = 0;
          bool firstHeightCone = false;
          for (uint32_t i = 0; i < layerCount; ++i) {
            bool layerCone = false;
            trackTexture(m_device->getCommon()->metaPomConeMaps().resolveHeight(
                           terrainLayers->heightTextures[i], true, layerCone, PomConeMapManager::ConeVariant::Conservative),
                         heightIndices[i], hasTexcoords, true, &samplerFeedbackStamp);
            if (layerCone && heightIndices[i] != kSurfaceMaterialInvalidTextureIndex) {
              coneMask |= 1u << i;
            }
            if (firstHeightIndex == kSurfaceMaterialInvalidTextureIndex) {
              firstHeightIndex = heightIndices[i];
              firstHeightCone = layerCone;
            }
          }
          if (firstHeightIndex != kSurfaceMaterialInvalidTextureIndex) {
            heightMaterialIndex = m_surfaceMaterialExtensionCache.track(RtTerrainLayersHeightMaterial { heightIndices, coneMask });
            if (heightTextureIndex == kSurfaceMaterialInvalidTextureIndex) {
              heightTextureIndex = firstHeightIndex;
              // The borrowed index is a derived texture, so the cone flag follows it (only the
              // cone debug views read it for a terrain material; conservative cones are valid relaxed ones too).
              pomConeAvailable = firstHeightCone;
            }
          }
        }
        const RtTerrainLayersMaterial terrainLayersMaterial {
          layerCount, albedoIndices, normalIndices,
          weightIndex0, weightIndex1, tintIndex, weightSamplerIndex,
          Vector2(terrainLayers->weightUvScale[0], terrainLayers->weightUvScale[1]),
          Vector2(terrainLayers->weightUvOffset[0], terrainLayers->weightUvOffset[1]),
          terrainLayers->normalsRgbTangentSpace,
          // Per-layer gloss from normal alpha, only for layers whose normal texture has an alpha channel.
          FnvMaterialProtocol::normalAlphaIsGloss() ? layerGlossMask : 0u,
          &heightIndices, heightMaterialIndex
        };
        terrainLayersMaterialIndex = m_surfaceMaterialExtensionCache.track(terrainLayersMaterial);
      }

      RtOpaqueSurfaceMaterial opaqueSurfaceMaterial{
        albedoOpacityTextureIndex, normalTextureIndex,
        tangentTextureIndex, heightTextureIndex, roughnessTextureIndex,
        metallicTextureIndex, emissiveColorTextureIndex,
        anisotropy, emissiveIntensity,
        albedoOpacityConstant,
        roughnessConstant, metallicConstant,
        emissiveColorConstant, enableEmissive,
        ignoreAlphaChannel, thinFilmEnable, alphaIsThinFilmThickness,
        thinFilmThicknessConstant, samplerIndex, displaceIn, displaceOut,
        opaqueMaterialData.getEnableDLSSControlMask(),
        opaqueMaterialData.getDLSSControlMaskIntensity(),
        opaqueMaterialData.getDLSSControlMaskToneStrength(),
        opaqueMaterialData.getDLSSControlMaskStructuralStrength(),
        subsurfaceMaterialIndex, isUsingRaytracedRenderTarget, isHairCard,
        samplerFeedbackStamp,
        secondaryTextureIndex,
        albedoTextureIsSrgb, emissiveTextureIsSrgb,
        opaqueMaterialData.getSkyLitParticle(),
        renderMaterialData.usesLegacyDefaults() || LegacyMaterialDefaults::applyToApiMaterials(),
        hostEncodingFlags, roughnessFloor,
        specularMetalPacked,
        (forkFnvFlags & OPAQUE_SURFACE_MATERIAL_FLAG_NORMAL_RGB_TANGENT_SPACE) != 0,
        (forkFnvFlags & OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_NORMAL_ALPHA) != 0,
        (forkFnvFlags & OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TEXTURE_TINTED) != 0,
        terrainLayersMaterialIndex,
        (forkFnvFlags & OPAQUE_SURFACE_MATERIAL_FLAG_NORMAL_ALPHA_IS_GLOSS) != 0
      };

      opaqueSurfaceMaterial.setPomConeAvailable(pomConeAvailable);
      surfaceMaterial.emplace(opaqueSurfaceMaterial);

      // [MaterialDump], once per material built (rtx.fnvProtocol.logMaterials): every input the shader
      // shades with, so one API material can be read against its D3D9 twin. The constants are shown before and
      // after rtx.opaqueMaterial.* scale and bias (opaque_surface_material_interaction.slangh applies those to
      // every opaque material, whichever route built it).
      if (FnvMaterialProtocol::logMaterials()) {
        static uint32_t s_materialDumps = 0;
        // Picked hashes (rtx.fnvProtocol.logMaterialTextures) are logged past the cap, on every build.
        const fast_unordered_set& watchedHashes = FnvMaterialProtocol::logMaterialTextures();
        const bool watched = !watchedHashes.empty() &&
          (lookupHash(watchedHashes, preCreationHash) || lookupHash(watchedHashes, renderMaterialData.getHash()) ||
           lookupHash(watchedHashes, drawCallState.getMaterialData().getHash()) ||
           // API draws: the picker shows the imported texture's D3D9 image hash, not the API texture's own.
           lookupHash(watchedHashes, drawCallState.externalPickingTextureHash) ||
           lookupHash(watchedHashes, opaqueMaterialData.getAlbedoOpacityTexture().getImageHash()));
        if (watched || s_materialDumps < 4000) {
          if (!watched) {
            ++s_materialDumps;
          }
          auto describeTexture = [](const TextureRef& texture) -> std::string {
            const DxvkImageView* view = texture.getImageView();
            if (view == nullptr) {
              return texture.isValid() ? "unresolved" : "none";
            }
            const VkFormat format = view->info().format;
            return str::format("0x", std::hex, texture.getImageHash(), std::dec, " ", forkVkFormatName(format), " ",
                               view->imageInfo().extent.width, "x", view->imageInfo().extent.height,
                               " mips=", view->imageInfo().mipLevels, " srgbView=", TextureUtils::isSRGB(format) ? 1 : 0,
                               " glossAlpha=", FnvMaterialProtocol::textureHasGlossAlpha(texture) ? 1 : 0);
          };
          const bool legacy = renderMaterialData.usesLegacyDefaults();
          const bool ownMaterial = drawCallState.getMaterialData().getHash() == renderMaterialData.getHash();
          const float roughnessShaded = std::clamp(roughnessConstant * OpaqueMaterialOptions::roughnessScale() + OpaqueMaterialOptions::roughnessBias(), 0.0f, 1.0f);
          const float metallicShaded = std::clamp(metallicConstant * OpaqueMaterialOptions::metallicScale() + OpaqueMaterialOptions::metallicBias(), 0.0f, 1.0f);
          // The sampler the material samples with: a CLAMP recorded on a wall whose UVs leave [0, 1] stretches
          // its edge texels; the anisotropy decides the blur at grazing angles.
          std::string samplerDesc = "none";
          if (sampler != nullptr) {
            const DxvkSamplerCreateInfo& si = sampler->info();
            samplerDesc = str::format("addressU=", static_cast<int>(si.addressModeU), " addressV=", static_cast<int>(si.addressModeV),
                                      " mag=", static_cast<int>(si.magFilter), " aniso=", si.useAnisotropy ? si.maxAnisotropy : 0.0f,
                                      " lodBias=", si.mipmapLodBias);
          }
          // VkCompareOp and AlphaTestType share these values.
          auto compareName = [](uint32_t op) -> const char* {
            constexpr const char* kNames[] = { "NEVER", "LESS", "EQUAL", "LESSEQUAL", "GREATER", "NOTEQUAL", "GREATEREQUAL", "ALWAYS" };
            return op < std::size(kNames) ? kNames[op] : "?";
          };
          // The material's own alpha state, which deriveAlphaState uses unless the material takes the draw's.
          // An API cutout lives here: its draw part below shows the instance's blend state, not the material's test.
          const std::string materialAlphaDesc = str::format(
            "drawAlphaState=", opaqueMaterialData.getUseLegacyAlphaState() ? 1 : 0,
            " test=", compareName(static_cast<uint32_t>(opaqueMaterialData.getAlphaTestType())),
            " ref=", static_cast<uint32_t>(opaqueMaterialData.getAlphaTestReferenceValue()),
            " blend=", opaqueMaterialData.getBlendEnabled() ? 1 : 0);
          // The draw's own alpha inputs: the D3D9 render state as the legacy material captured it, or an API instance's
          // InstanceInfoBlendEXT. A cutout the game draws through D3D9 needs alphaTest=1 here, or the surface is traced opaque.
          const LegacyMaterialData& drawMaterial = drawCallState.getMaterialData();
          const std::string drawAlphaDesc = str::format(
            "alphaTest=", drawMaterial.alphaTestEnabled ? 1 : 0, " func=", compareName(static_cast<uint32_t>(drawMaterial.alphaTestCompareOp)),
            " ref=", static_cast<uint32_t>(drawMaterial.alphaTestReferenceValue),
            " blend=", drawMaterial.blendMode.enableBlending ? 1 : 0,
            " src=", static_cast<int>(drawMaterial.blendMode.colorSrcFactor), " dst=", static_cast<int>(drawMaterial.blendMode.colorDstFactor),
            " alphaOp=", static_cast<int>(drawMaterial.textureAlphaOperation),
            " categories=0x", std::hex, drawCallState.getCategoryFlags().raw(), std::dec);
          std::string layers;
          if (const auto& terrainLayers = opaqueMaterialData.getTerrainLayers(); terrainLayers && terrainLayers->layerCount > 0) {
            layers = str::format(" | layers=", terrainLayers->layerCount, " glossMaskAtCreate=0x", std::hex, terrainLayers->normalHasAlphaMask,
                                 " glossMaskNow=0x", layerGlossMask, std::dec, " rgbNormals=", terrainLayers->normalsRgbTangentSpace ? 1 : 0,
                                 " layer0 albedo{", describeTexture(terrainLayers->albedoTextures[0]), "} layer0 normal{",
                                 describeTexture(terrainLayers->normalTextures[0]), "}");
          }
          Logger::info(str::format("[MaterialDump] 0x", std::hex, preCreationHash, std::dec, watched ? " WATCHED" : "",
                                   " game=0x", std::hex, drawCallState.externalPickingTextureHash, std::dec,
                                   " route=", legacy ? "d3d9-legacy" : ownMaterial ? "api" : "replacement",
                                   " legacyDefaults=", (legacy || LegacyMaterialDefaults::applyToApiMaterials()) ? 1 : 0,
                                   " fnvFlags=0x", std::hex, forkFnvFlags, " hostEnc=0x", hostEncodingFlags, std::dec,
                                   " fnvLayout=", opaqueSurfaceMaterial.usesFnvLayout() ? 1 : 0,
                                   " | albedo{", describeTexture(opaqueMaterialData.getAlbedoOpacityTexture()),
                                   "} normal{", describeTexture(opaqueMaterialData.getNormalTexture()),
                                   "} glow{", describeTexture(opaqueMaterialData.getEmissiveColorTexture()),
                                   "} height{", describeTexture(opaqueMaterialData.getHeightTexture()), "}", layers,
                                   " | roughnessConstant=", roughnessConstant, " -> shaded ", roughnessShaded,
                                   " (scale ", OpaqueMaterialOptions::roughnessScale(), " bias ", OpaqueMaterialOptions::roughnessBias(), ")",
                                   " metallicConstant=", metallicConstant, " -> shaded ", metallicShaded,
                                   " (scale ", OpaqueMaterialOptions::metallicScale(), " bias ", OpaqueMaterialOptions::metallicBias(), ")",
                                   " albedoConst=(", albedoOpacityConstant.x, ", ", albedoOpacityConstant.y, ", ", albedoOpacityConstant.z,
                                   ") opacity=", albedoOpacityConstant.w, " albedoBias=", OpaqueMaterialOptions::albedoBias(),
                                   " specularLevel=", LegacyMaterialDefaults::specularLevel(), " fresnelGrazing=", LegacyMaterialDefaults::fresnelGrazing(),
                                   " emissive=", enableEmissive ? 1 : 0, " emissiveIntensity=", emissiveIntensity,
                                   " displaceIn=", displaceIn, " ignoreAlpha=", ignoreAlphaChannel ? 1 : 0,
                                   " albedoSrgbFlag=", albedoTextureIsSrgb ? 1 : 0, " sampler=", samplerIndex, " [", samplerDesc, "]",
                                   " | material alpha: ", materialAlphaDesc,
                                   " | draw: material=0x", std::hex, drawCallState.getMaterialData().getHash(), std::dec,
                                   " texcoords=", hasTexcoords ? 1 : 0, " ", drawAlphaDesc, " frame=", m_device->getCurrentFrameId()));
        }
      }
    } else if (renderMaterialDataType == MaterialDataType::Translucent) {
      surfaceMaterial.emplace(createTranslucentSurfaceMaterial(renderMaterialData.getTranslucentMaterialData(), samplerIndex, hasTexcoords));
    } else if (renderMaterialDataType == MaterialDataType::RayPortal) {
      m_rayPortalMaterialCreated.store(true, std::memory_order_relaxed); // See hasCreatedRayPortalMaterial().
      const auto& rayPortalMaterialData = renderMaterialData.getRayPortalMaterialData();

      uint32_t maskTextureIndex = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture(), maskTextureIndex, hasTexcoords, false);
      uint32_t maskTextureIndex2 = kSurfaceMaterialInvalidTextureIndex;
      trackTexture(rayPortalMaterialData.getMaskTexture2(), maskTextureIndex2, hasTexcoords, false);

      uint8_t rayPortalIndex = rayPortalMaterialData.getRayPortalIndex();
      float rotationSpeed = rayPortalMaterialData.getRotationSpeed();
      bool enableEmissive = rayPortalMaterialData.getEnableEmission();
      float emissiveIntensity = rayPortalMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity() * HostCalibration::emissiveScale();

      const RtRayPortalSurfaceMaterial rayPortalSurfaceMaterial{
        maskTextureIndex, maskTextureIndex2, rayPortalIndex,
        rotationSpeed, enableEmissive, emissiveIntensity, samplerIndex, samplerIndex2
      };

      surfaceMaterial.emplace(rayPortalSurfaceMaterial);
    }

    assert(surfaceMaterial.has_value());
    assert(surfaceMaterial->validate());

    // Cache this
    const uint32_t index = m_surfaceMaterialCache.track(*surfaceMaterial);
    m_preCreationSurfaceMaterialMap[preCreationHash] = index;
    if (out_indexInCache) {
      *out_indexInCache = index;
    }
    return m_surfaceMaterialCache.at(index);
  }


  RtTranslucentSurfaceMaterial SceneManager::createTranslucentSurfaceMaterial(const TranslucentMaterialData& translucentMaterialData,
                                                                              uint32_t samplerIndex,
                                                                              bool hasTexcoords) {
    uint32_t normalTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t transmittanceTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint32_t emissiveColorTextureIndex = kSurfaceMaterialInvalidTextureIndex;
    uint16_t samplerFeedbackStamp = SAMPLER_FEEDBACK_INVALID;

    trackTexture(translucentMaterialData.getNormalTexture(), normalTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
    trackTexture(translucentMaterialData.getTransmittanceTexture(), transmittanceTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);
    trackTexture(translucentMaterialData.getEmissiveColorTexture(), emissiveColorTextureIndex, hasTexcoords, true, &samplerFeedbackStamp);

    return RtTranslucentSurfaceMaterial{
      normalTextureIndex,
      transmittanceTextureIndex,
      emissiveColorTextureIndex,
      translucentMaterialData.getRefractiveIndex() * std::clamp(TranslucentMaterialOptions::refractiveIndexScale(), 0.0f, 3.0f),
      translucentMaterialData.getTransmittanceMeasurementDistance(),
      translucentMaterialData.getTransmittanceColor(),
      translucentMaterialData.getEnableEmission(),
      translucentMaterialData.getEmissiveIntensity() * RtxOptions::emissiveIntensity() * HostCalibration::emissiveScale(),
      translucentMaterialData.getEmissiveColorConstant(),
      translucentMaterialData.getEnableThinWalled(),
      translucentMaterialData.getThinWallThickness(),
      translucentMaterialData.getEnableDiffuseLayer(),
      samplerIndex,
      samplerFeedbackStamp
    };
  }

  Rc<DxvkSampler> SceneManager::getOrCreateExternalSampler() {
    if (m_externalSampler == nullptr) {
      auto s = DxvkSamplerCreateInfo {};
      {
        s.magFilter = VK_FILTER_LINEAR;
        s.minFilter = VK_FILTER_LINEAR;
        s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        s.mipmapLodBias = 0.f;
        s.mipmapLodMin = 0.f;
        s.mipmapLodMax = 0.f;
        s.useAnisotropy = VK_FALSE;
        s.maxAnisotropy = 1.f;
        s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        s.compareToDepth = VK_FALSE;
        s.compareOp = VK_COMPARE_OP_NEVER;
        s.borderColor = VkClearColorValue {};
        s.usePixelCoord = VK_FALSE;
      }
      m_externalSampler = m_device->createSampler(s);
    }

    return m_externalSampler;
  }

  void SceneManager::setExternalStartInMediumMaterial(const MaterialData& translucentMaterial) {
    assert(translucentMaterial.getType() == MaterialDataType::Translucent);

    const auto samplerIndex = trackSampler(getOrCreateExternalSampler());
    const auto surfaceMaterial = RtSurfaceMaterial(
      createTranslucentSurfaceMaterial(translucentMaterial.getTranslucentMaterialData(), samplerIndex, true));

    m_externalStartInMediumMaterialIndex_inCache = m_surfaceMaterialCache.track(surfaceMaterial);
  }

  void SceneManager::clearExternalStartInMediumMaterial() {
    m_externalStartInMediumMaterialIndex_inCache = UINT32_MAX;
  }

  void SceneManager::setStartInMediumMaterial(const MaterialData& translucentMaterial) {
    assert(translucentMaterial.getType() == MaterialDataType::Translucent);
    std::lock_guard lock { m_startInMediumMaterialMutex };
    m_pendingClearStartInMediumMaterial = false;
    m_pendingStartInMediumMaterial = translucentMaterial;
  }

  void SceneManager::clearStartInMediumMaterial() {
    std::lock_guard lock { m_startInMediumMaterialMutex };
    m_pendingStartInMediumMaterial.reset();
    m_pendingClearStartInMediumMaterial = true;
  }

  std::optional<XXH64_hash_t> SceneManager::findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue) {
    std::lock_guard lock { m_drawCallMeta.mutex };

    auto tryFindIn = [](const std::unordered_map<ObjectPickingValue, DrawCallMetaInfo>& table, ObjectPickingValue toFind)
      -> std::optional<XXH64_hash_t> {
      auto found = table.find(toFind);
      if (found != table.end()) {
        const DrawCallMetaInfo& meta = found->second;
        if (meta.legacyTextureHash != kEmptyHash) {
          return meta.legacyTextureHash;
        }
      }
      return std::nullopt;
    };

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        if (auto h = tryFindIn(m_drawCallMeta.infos[tick], objectPickingValue)) {
          return h;
        }
      }
    }
    return std::nullopt;
  }

  std::vector<ObjectPickingValue> SceneManager::gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash) {
    std::lock_guard lock { m_drawCallMeta.mutex };
    assert(texHash != kEmptyHash);

    const int ticksToCheck[] = {
      m_drawCallMeta.ticker, // current tick
      (m_drawCallMeta.ticker + m_drawCallMeta.MaxTicks - 1) % m_drawCallMeta.MaxTicks, // prev tick
    };

    auto correspondingValues = std::vector<ObjectPickingValue> {};
    for (int tick : ticksToCheck) {
      if (m_drawCallMeta.ready[tick]) {
        for (const auto& [pickingValue, meta] : m_drawCallMeta.infos[tick]) {
          if (texHash == meta.legacyTextureHash) {
            correspondingValues.push_back(pickingValue);
          } else if (texHash == meta.legacyTextureHash2) {
            correspondingValues.push_back(pickingValue);
          }
        }
        break;
      }
    }
    return correspondingValues;
  }

  SceneManager::SamplerIndex SceneManager::trackSampler(Rc<DxvkSampler> sampler) {
    if (sampler == nullptr) {
      ONCE(Logger::warn("Found a null sampler. Fallback to linear-repeat"));
      sampler = patchSampler(
        VK_FILTER_LINEAR,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VkClearColorValue {});
    }
    return m_samplerCache.track(sampler);
  }

  Rc<DxvkSampler> SceneManager::patchSampler( const VkFilter filterMode,
                                              const VkSamplerAddressMode addressModeU,
                                              const VkSamplerAddressMode addressModeV,
                                              const VkSamplerAddressMode addressModeW,
                                              const VkClearColorValue borderColor) {
    auto& resourceManager = m_device->getCommon()->getResources();
    // Create a sampler to account for DLSS lod bias and any custom filtering overrides the user has set
    return resourceManager.getSampler(
      filterMode,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      addressModeU,
      addressModeV,
      addressModeW,
      borderColor,
      getTotalMipBias(),
      RtxOptions::useAnisotropicFiltering());
  }

  void SceneManager::addLight(const D3DLIGHT9& light) {
    ScopedCpuProfileZone();
    // Attempt to convert the D3D9 light to RT

    std::optional<LightData> lightData = LightData::tryCreate(light);

    // Note: Skip adding this light if it is somehow malformed such that it could not be created.
    if (!lightData.has_value()) {
      return;
    }

    RtLight rtLight = lightData->toRtLight();
    const XXH64_hash_t assetKey = LightUtils::legacyAssetKey(light);
    if (assetKey != kEmptyHash) {
      rtLight.setAssetKey(assetKey);
    }
    const std::shared_ptr<const ReplacementBucket> pReplacements = m_pReplacer->getReplacementsForLight(rtLight.getInitialHash());

    // Build identity hash from the light's stable hash + position. Used by
    // both the replacement and the toggle-off cleanup paths below; must stay
    // in sync between them so they target the same RI.
    // A light with an asset key is identified by the key alone, so it keeps its replacement while it moves.
    const XXH64_hash_t lightAssetHash = rtLight.getInitialHash();
    const Vector3 lightPos = rtLight.getPosition();
    const XXH64_hash_t lightIdHash = assetKey != kEmptyHash ? assetKey : XXH64(&lightPos, sizeof(Vector3), lightAssetHash);

    if (pReplacements) {
      const Matrix4 lightTransform = LightUtils::getLightTransform(light);

      const ReplacementInstance::LookupKey lightKey { lightIdHash, lightAssetHash, kEmptyHash, kEmptyHash, lightPos, lightTransform };
      ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(lightKey);

      // Reinitialize if the bucket changed. Unchanged buckets keep the same address
      // (in-place updates), so pointer inequality is sufficient. clear() re-dirties
      // the bounding box that lightBoundingBox below depends on.
      if (replacementInstance->root.getUntyped() != nullptr &&
          replacementInstance->activeReplacements.get() != pReplacements.get()) {
        replacementInstance->clear();
      }

      // All lights in a light replacement are externally tracked, with their
      // lifecycle managed by the ReplacementInstance. This unifies root and sub-light
      // handling: create on first frame, update on subsequent frames.
      // TODO(TREX-1091) to implement meshes as light replacements, replace the below loop with a call to drawReplacements.
      const bool needsBBoxUpdate = replacementInstance->boundingBoxDirty;
      AxisAlignedBoundingBox litBBox;
      for (size_t i = 0; i < pReplacements->replacements.size(); i++) {
        const auto& replacement = pReplacements->replacements[i];
        if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          LightData replacementLight = replacement.lightData.value();

          // Merge the d3d9 light into replacements based on overrides.
          // Must happen before AABB extraction: some entries (e.g. the translated
          // original game light) have Unknown lightType and zero position/radius
          // until merged with the d3d9 light.
          replacementLight.merge(light);

          // Convert to runtime light
          RtLight rtReplacementLight = replacementLight.toRtLight();

          if (needsBBoxUpdate) {
            const Vector3 pos = rtReplacementLight.getPosition();
            float lightRadius = 0.f;
            if (rtReplacementLight.getType() == RtLightType::Sphere) {
              lightRadius = rtReplacementLight.getSphereLight().getRadius();
            }
            for (uint32_t j = 0; j < 3; j++) {
              litBBox.minPos[j] = std::min(litBBox.minPos[j], pos[j] - lightRadius);
              litBBox.maxPos[j] = std::max(litBBox.maxPos[j], pos[j] + lightRadius);
            }
          }

          // Transform the replacement light by the legacy light
          if (replacementLight.relativeTransform()) {
            rtReplacementLight.applyTransform(lightTransform);
          }

          RtLight* existingLight = (replacementInstance->prims.size() > i)
              ? replacementInstance->prims[i].getLight() : nullptr;
          if (existingLight != nullptr) {
            m_lightManager.updateExternallyTrackedLight(existingLight, rtReplacementLight);
          } else {
            RtLight* newLight = m_lightManager.createExternallyTrackedLight(rtReplacementLight);
            if (newLight != nullptr) {
              if (replacementInstance->prims.empty()) {
                replacementInstance->setup(PrimInstance(newLight, PrimInstance::Type::Light), pReplacements->replacements.size(), pReplacements);
              }
              newLight->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, newLight, PrimInstance::Type::Light);
              if (replacementInstance->root.getUntyped() == nullptr) {
                replacementInstance->root = PrimInstance(newLight, PrimInstance::Type::Light);
              }
            }
          }
        } else {
          assert(false); // We don't support meshes as children of lights yet.
        }
      }

      replacementInstance->frameLastSeen = m_device->getCurrentFrameId();
      replacementInstance->objectToWorld = lightTransform;
      if (needsBBoxUpdate) {
        if (litBBox.isValid()) {
          replacementInstance->lightBoundingBox = litBBox;
        }
        replacementInstance->boundingBoxDirty = false;
      }
    } else {
      // If this light previously had a replacement (e.g. enableReplacementLights
      // was just toggled off), the externally-tracked replacement lights are
      // still alive in LightManager -- they have no frame-age GC; their
      // lifecycle is owned by the RI. Tear down the RI so its prims get marked
      // for GC; otherwise the user sees the replacement lights and the original
      // game light rendering simultaneously until DrawCallTracker collects the
      // RI ~numFramesToKeepInstances frames later.
      if (ReplacementInstance* stale = m_drawCallTracker.findReplacementInstanceByIdentity(lightIdHash)) {
        if (stale->activeReplacements != nullptr) {
          stale->clear();
        }
      }

      // This is a light coming from the game directly, so use the appropriate API for filter rules
      m_lightManager.addGameLight(light.Type, rtLight, LightUtils::legacyIdentity(light));
    }
  }

  // rtx.incrementalSurfaceMaterialUpload: the per-surface material table, kept across frames, filled
  // on worker threads, compared with what the buffer holds, and uploaded in changed runs. Point-
  // instancer duplicate slots are written by the GPU culling pass from their template, so they are
  // never written or compared here (their shadow bytes are poisoned so a CPU record landing there
  // later is always uploaded). The camera-medium record, if any, goes last.
  void SceneManager::uploadSurfaceMaterialsIncremental(const Rc<RtxContext>& ctx, size_t surfaceMaterialsGPUSize) {
    const std::vector<RtInstance*>& surfaces = m_accelManager.getOrderedInstances();
    const uint32_t surfaceCount = uint32_t(surfaces.size());
    const uint32_t recordCount = surfaceCount + (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex ? 1u : 0u);
    assert(size_t(recordCount) * kSurfaceMaterialGPUSize == surfaceMaterialsGPUSize);

    std::vector<unsigned char>& data = m_surfaceMaterialsGPUData;
    data.resize(surfaceMaterialsGPUSize);
    m_surfaceMaterialSlotFlags.resize(recordCount);

    const bool canDiff = m_uploadedSurfaceMaterialBuffer == m_surfaceMaterialBuffer &&
      m_uploadedSurfaceMaterialData.size() >= kSurfaceMaterialGPUSize;
    const size_t comparable = canDiff ? std::min(m_uploadedSurfaceMaterialData.size(), data.size()) / kSurfaceMaterialGPUSize : 0;
    const auto& materialTable = m_surfaceMaterialCache.getObjectTable();
    unsigned char* out = data.data();
    const unsigned char* uploaded = m_uploadedSurfaceMaterialData.data();
    uint8_t* flags = m_surfaceMaterialSlotFlags.data();
    constexpr uint8_t kDirty = 1, kGpuOwned = 2;

    // A surface bound to a tessellated mesh gets its material without displacement.
    const SccrTessManager& sccrTess = m_device->getCommon()->metaSccrTess();
    const auto writeRecord = [&](uint32_t slot, const RtSurfaceMaterial& material, const RtInstance* instance) {
      size_t offset = size_t(slot) * kSurfaceMaterialGPUSize;
      material.writeGPUData(out, offset, slot);
      if (instance != nullptr) {
        sccrTess.patchMaterialRecord(out + size_t(slot) * kSurfaceMaterialGPUSize, *instance);
      }
      const bool dirty = !canDiff || slot >= comparable ||
        std::memcmp(out + size_t(slot) * kSurfaceMaterialGPUSize, uploaded + size_t(slot) * kSurfaceMaterialGPUSize, kSurfaceMaterialGPUSize) != 0;
      flags[slot] = dirty ? kDirty : 0;
    };

    constexpr uint32_t kChunk = 1024;
    const uint32_t chunkCount = (surfaceCount + kChunk - 1) / kChunk;
    const auto processChunk = [&](uint32_t chunk) {
      const uint32_t begin = chunk * kChunk;
      const uint32_t end = std::min(surfaceCount, begin + kChunk);
      for (uint32_t i = begin; i < end; ++i) {
        const RtSurface& surf = surfaces[i]->surface;
        // For PointInstancer duplicates (entries beyond the template), skip
        // writeGPUData - the GPU culling shader copies the template material.
        if (surf.instancesToObject != nullptr &&
            surf.surfaceIndexOfFirstInstance != SIZE_MAX &&
            i > surf.surfaceIndexOfFirstInstance) {
          flags[i] = kGpuOwned;
          continue;
        }
        assert(surf.surfaceMaterialIndex < materialTable.size());
        writeRecord(i, materialTable[surf.surfaceMaterialIndex], surfaces[i]);
      }
    };
    if (chunkCount <= 2) {
      for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        processChunk(chunk);
      }
    } else {
      concurrency::parallel_for<uint32_t>(0, chunkCount, processChunk);
    }

    if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex) {
      writeRecord(surfaceCount, materialTable[m_startInMediumMaterialIndex_inCache], nullptr);
      m_startInMediumMaterialIndex = surfaceCount;
    } else {
      m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
    }
    m_lastUploadedStartInMediumMaterialIndexInCache = m_startInMediumMaterialIndex_inCache;

    if (!canDiff) {
      // The sequential path uploaded a zero-filled table; keep the GPU-owned slots identical.
      for (uint32_t i = 0; i < recordCount; ++i) {
        if (flags[i] == kGpuOwned) {
          std::memset(out + size_t(i) * kSurfaceMaterialGPUSize, 0, kSurfaceMaterialGPUSize);
        }
      }
      ctx->writeToBuffer(m_surfaceMaterialBuffer, 0, data.size(), data.data());
    } else {
      constexpr uint32_t kMergeGap = 8;
      uint32_t runStart = UINT32_MAX, runEnd = 0;
      const auto flushRun = [&]() {
        if (runStart != UINT32_MAX) {
          const size_t offset = size_t(runStart) * kSurfaceMaterialGPUSize;
          ctx->writeToBuffer(m_surfaceMaterialBuffer, offset, size_t(runEnd - runStart) * kSurfaceMaterialGPUSize, out + offset);
          runStart = UINT32_MAX;
        }
      };
      for (uint32_t i = 0; i < recordCount; ++i) {
        if (flags[i] != kDirty) {
          continue;
        }
        if (runStart != UINT32_MAX && i > runEnd + kMergeGap) {
          flushRun();
        }
        if (runStart == UINT32_MAX) {
          runStart = i;
        }
        runEnd = i + 1;
      }
      flushRun();
    }

    m_uploadedSurfaceMaterialBuffer = m_surfaceMaterialBuffer;
    m_uploadedSurfaceMaterialData.swap(data);
    for (uint32_t i = 0; i < recordCount; ++i) {
      if (flags[i] == kGpuOwned) {
        std::memset(m_uploadedSurfaceMaterialData.data() + size_t(i) * kSurfaceMaterialGPUSize, 0xA5, kSurfaceMaterialGPUSize);
      }
    }
  }

  void SceneManager::prepareSceneData(Rc<RtxContext> ctx, DxvkBarrierSet& execBarriers) {
    ScopedGpuProfileZone(ctx, "Build Scene");
    // [FrameTiming] CPU breakdown of this function (see takeScenePrepPhases).
    auto prepPhaseStart = std::chrono::steady_clock::now();
    const auto prepPhase = [&](uint32_t phase) {
      const auto now = std::chrono::steady_clock::now();
      const double phaseMs = std::chrono::duration<double, std::milli>(now - prepPhaseStart).count();
      m_prepPhaseMs[phase] += phaseMs;
      m_prepPhaseLastMs[phase] += phaseMs;
      prepPhaseStart = now;
    };
    m_prepPhaseFrames++;
    std::fill(std::begin(m_prepPhaseLastMs), std::end(m_prepPhaseLastMs), 0.0);

  #ifdef REMIX_DEVELOPMENT
    if (m_device->getCurrentFrameId() == RtxOptions::dumpAllInstancesOnFrame()) {
      // Print all RtInstances for debugging
      printAllRtInstances();
    }
  #endif

    // Needs to happen before garbageCollection to avoid destroying dynamic lights
    {
      const auto lightMatchStart = std::chrono::steady_clock::now();
      m_lightManager.dynamicLightMatching();
      addPrepNestedMs(kPrepNestedLightMatch,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lightMatchStart).count());
    }

    garbageCollection();
    prepPhase(0);

    // Re-register buffers, textures, and materials for anti-culled instances.
    // These instances survived GC but the game didn't submit draw calls for them
    // this frame, so their per-frame table indices (buffer cache, material cache)
    // are stale. Without this, they would render with wrong geometry or textures.
    {
      const auto antiCullStart = std::chrono::steady_clock::now();
      const uint32_t currentFrameId = m_device->getCurrentFrameId();
      const auto& replacementInstances = m_drawCallTracker.getReplacementInstances();
      const auto preserveUnseen = [&](ReplacementInstance* ri) {
        for (auto& prim : ri->prims) {
          RtInstance* instance = prim.getInstance();
          if (instance != nullptr) {
            preserveInstance(*instance);
          }
        }
      };
      if (RtxOptions::parallelScenePrepScans() && replacementInstances.size() >= 4096) {
        // rtx.parallelScenePrepScans: find the unseen ones on worker threads (a cache miss per
        // replacement instance, nearly all of them seen), then preserve those in order.
        constexpr uint32_t kChunk = 1024;
        const uint32_t count = uint32_t(replacementInstances.size());
        const uint32_t chunkCount = (count + kChunk - 1) / kChunk;
        std::vector<std::vector<uint32_t>> unseen(chunkCount);
        concurrency::parallel_for<uint32_t>(0, chunkCount, [&](uint32_t chunk) {
          const uint32_t end = std::min(count, (chunk + 1) * kChunk);
          for (uint32_t i = chunk * kChunk; i < end; ++i) {
            if (i + 8 < end) {
              _mm_prefetch(reinterpret_cast<const char*>(&replacementInstances[i + 8]->frameLastSeen), _MM_HINT_T0);
            }
            if (replacementInstances[i]->frameLastSeen != currentFrameId) {
              unseen[chunk].push_back(i);
            }
          }
        });
        for (const auto& chunkUnseen : unseen) {
          for (const uint32_t i : chunkUnseen) {
            preserveUnseen(replacementInstances[i].get());
          }
        }
      } else {
        for (auto& ri : replacementInstances) {
          if (ri->frameLastSeen == currentFrameId) {
            continue;
          }
          preserveUnseen(ri.get());
        }
      }
      addPrepNestedMs(kPrepNestedAntiCull,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - antiCullStart).count());
    }

    m_graphManager.applySceneOverrides(ctx);

    m_terrainBaker->prepareSceneData(ctx);

    // Binds every instance's mesh to its cached SCCR shell buffer and builds the missing ones. This runs after garbage
    // collection and the anti-cull pass, so the instance table is final, and before the bindless buffer table is
    // uploaded, so a buffer built now is in this frame's table and the surface upload writes its slot this frame.
    m_device->getCommon()->metaSccrShell().prepareFrame(*ctx);
    // Binds flagged instances to their tessellated meshes and builds missing ones from the
    // shell buffers just bound; its buffer slots join this frame's table too.
    m_device->getCommon()->metaSccrTess().prepareFrame(*ctx);

    auto& textureManager = m_device->getCommon()->getTextureManager();
    {
      const auto bindlessStart = std::chrono::steady_clock::now();
      m_bindlessResourceManager.prepareSceneData(ctx, textureManager.getTextureTable(), getBufferTable(), getSamplerTable());
      addPrepNestedMs(kPrepNestedBindless,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - bindlessStart).count());
    }
    prepPhase(1);

    // If there are no instances, we should do nothing!
    if (m_instanceManager.getActiveCount() == 0) {
      // Clear the ray portal data before the next frame
      m_rayPortalManager.clear();
      return;
    }

    m_rayPortalManager.prepareSceneData(ctx);
    // Note: only main camera needs to be teleportation corrected as only that one is used for ray tracing & denoising
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::Main));
    m_rayPortalManager.fixCameraInBetweenPortals(m_cameraManager.getCamera(CameraType::ViewModel));
    m_rayPortalManager.createVirtualCameras(m_cameraManager);
    const bool didTeleport = m_rayPortalManager.detectTeleportationAndCorrectCameraHistory(
      m_cameraManager.getCamera(CameraType::Main),
      m_cameraManager.isCameraValid(CameraType::ViewModel) ? &m_cameraManager.getCamera(CameraType::ViewModel) : nullptr);

    {
      const uint32_t previousStartInMediumMaterialIndexInCache = m_lastResolvedStartInMediumMaterialIndexInCache;
      std::optional<MaterialData> pendingStartInMediumMaterial;
      uint32_t persistentStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
      bool clearStartInMediumMaterial = false;
      {
        std::lock_guard lock { m_startInMediumMaterialMutex };
        clearStartInMediumMaterial = m_pendingClearStartInMediumMaterial;
        m_pendingClearStartInMediumMaterial = false;
        if (m_pendingStartInMediumMaterial.has_value()) {
          pendingStartInMediumMaterial = std::move(m_pendingStartInMediumMaterial);
          m_pendingStartInMediumMaterial.reset();
        }
      }

      if (clearStartInMediumMaterial) {
        m_persistentStartInMediumMaterial.reset();
      }

      if (pendingStartInMediumMaterial.has_value()) {
        m_persistentStartInMediumMaterial = std::move(pendingStartInMediumMaterial);
      }

      if (m_persistentStartInMediumMaterial.has_value()) {
        assert(m_persistentStartInMediumMaterial->getType() == MaterialDataType::Translucent);
        const auto samplerIndex = trackSampler(getOrCreateExternalSampler());
        const auto surfaceMaterial = RtSurfaceMaterial(
          createTranslucentSurfaceMaterial(m_persistentStartInMediumMaterial->getTranslucentMaterialData(), samplerIndex, true));
        persistentStartInMediumMaterialIndexInCache = m_surfaceMaterialCache.track(surfaceMaterial);
      }

      m_startInMediumMaterialIndex_inCache = m_externalStartInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex
        ? m_externalStartInMediumMaterialIndex_inCache
        : persistentStartInMediumMaterialIndexInCache != kInvalidMaterialCacheIndex
          ? persistentStartInMediumMaterialIndexInCache
          : m_fogStartInMediumMaterialIndex_inCache;

      if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex &&
          m_startInMediumMaterialIndex_inCache >= m_surfaceMaterialCache.getObjectTable().size()) {
        Logger::debug(str::format(
          "[RTX] Ignoring stale camera medium material cache index ", m_startInMediumMaterialIndex_inCache,
          " on frame ", m_device->getCurrentFrameId(),
          "; surfaceMaterialCacheSize=", m_surfaceMaterialCache.getObjectTable().size()));
        m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
      }

      if (m_startInMediumMaterialIndex_inCache != previousStartInMediumMaterialIndexInCache) {
        Logger::debug(str::format(
          "[RTX] View history invalidated due to camera medium change on frame ", m_device->getCurrentFrameId(),
          ": previousStartInMediumInCache=", previousStartInMediumMaterialIndexInCache,
          ", currentStartInMediumInCache=", m_startInMediumMaterialIndex_inCache,
          ", cleared=", clearStartInMediumMaterial ? "true" : "false",
          ", pendingSet=", pendingStartInMediumMaterial.has_value() ? "true" : "false"));
        m_cameraManager.getMainCamera().invalidateViewHistory(m_device->getCurrentFrameId());
      }
      m_lastResolvedStartInMediumMaterialIndexInCache = m_startInMediumMaterialIndex_inCache;
    }

    if (m_cameraManager.isCameraCutThisFrame()) {
      if (!didTeleport) {
        D3d9IdentityTrace::recordCut(m_device->getCurrentFrameId());
      }
      // Ignore camera cut events on teleportation so we don't flush the caches
      if (!didTeleport && !implicitClearsAllowed()) {
        // rtx.scene.noImplicitClears / d3d9CutAsEvent: history only (the camera's own cut detection resets it).
        // Objects not drawn since before this frame age out without anti-culling (m_lastKeptCutFrame).
        m_lastKeptCutFrame = m_device->getCurrentFrameId();
        D3d9Identity::noteCutKept();
        Logger::info(str::format("Camera cut detected on frame ", m_device->getCurrentFrameId(),
                                 m_pReplacer->hasExternalMeshes() ? " (scene kept: rtx.scene.noImplicitClears)"
                                                                  : " (scene kept: rtx.scene.d3d9CutAsEvent)"));
      } else if (!didTeleport) {
        Logger::info(str::format("Camera cut detected on frame ", m_device->getCurrentFrameId()));
        m_enqueueDelayedClear = true;
      }
    }

    // Initialize/remove opacity micromap manager
    // rtx.alphaBake.softwareOmm runs the manager in software mode (no micromap API calls);
    // a mode change replaces the manager.
    const bool softwareOmm = RtxOptions::getEnableSoftwareOmm();
    if (RtxOptions::getEnableOpacityMicromap() || softwareOmm) {
      if (!m_opacityMicromapManager.get() ||
          // Reset the manager on camera cuts
          m_enqueueDelayedClear ||
          m_opacityMicromapManager->isSoftware() != softwareOmm) {
        if (m_opacityMicromapManager.get()) {
          if (m_opacityMicromapManager->isSoftware()) {
            m_opacityMicromapManager->releaseSoftwareSlots(*this);
          }
          m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());
          // Every micromap of the old manager goes away with it: BLASes built with them hold them
          // (rtx.ommHoldBoundMicromaps) until rebuilt; rebuild them all against the new manager.
          m_accelManager.invalidateOpacityMicromapBindings();
          m_instanceManager.notifySceneChanged();
        }

        m_opacityMicromapManager = std::make_unique<OpacityMicromapManager>(m_device, softwareOmm);
        m_instanceManager.addEventHandler(m_opacityMicromapManager->getInstanceEventHandler());
        // Seed candidates with instances that were added before the event handler was registered
        m_opacityMicromapManager->seedCandidates(m_instanceManager.getInstanceTable());
        Logger::info("[RTX] Opacity Micromap: enabled");
      }
    } else if (m_opacityMicromapManager.get()) {
      if (m_opacityMicromapManager->isSoftware()) {
        m_opacityMicromapManager->releaseSoftwareSlots(*this); // rtx.alphaBake.softwareOmm
      }
      m_accelManager.invalidateOpacityMicromapBindings();
      m_instanceManager.notifySceneChanged();
      m_instanceManager.removeEventHandler(m_opacityMicromapManager.get());
      m_opacityMicromapManager = nullptr;
      Logger::info("[RTX] Opacity Micromap: disabled");
    }

    // rtx.coreMaterial.build. Nothing is allocated while it is off.
    if (RtxOptions::CoreMaterial::build()) {
      if (!m_coreMaterials) {
        m_coreMaterials = std::make_unique<CoreMaterialBuilder>(m_device);
        Logger::info("[CoreMaterial] build: enabled");
      }
    } else if (m_coreMaterials) {
      m_coreMaterials->releaseAll(*this);
      m_coreMaterials = nullptr;
      Logger::info("[CoreMaterial] build: disabled");
    }

    RtxParticleSystemManager& particles = m_device->getCommon()->metaParticleSystem();
    particles.simulate(ctx.ptr());

    m_instanceManager.findPortalForVirtualInstances(m_cameraManager, m_rayPortalManager);
    m_instanceManager.createViewModelInstances(ctx, m_cameraManager, m_rayPortalManager);
    // Lenses are placed after the view model instances exist, with the transforms the TLAS instances get.
    addReplacementLenses();
    m_instanceManager.createPlayerModelVirtualInstances(ctx, m_cameraManager, m_rayPortalManager);

    prepPhase(2);
    ctx->recordGpuStageTiming("SceneSetup");
    m_accelManager.mergeInstancesIntoBlas(ctx, execBarriers, textureManager.getTextureTable(), m_cameraManager, m_instanceManager, m_opacityMicromapManager.get());
    // The OmmBakeAndBuild marker inside mergeInstancesIntoBlas splits OMM work from the BLAS builds.
    ctx->recordGpuStageTiming("BlasBuild");
    // rtx.alphaBake.softwareOmm. Surface slots are assigned here; the per-surface store
    // table follows them.
    if (m_opacityMicromapManager && m_opacityMicromapManager->isSoftware()) {
      m_opacityMicromapManager->updateSoftwareTable(ctx, *this);
    }
    // rtx.alphaBake.softwareOmmSelfTest (nothing while the option stays off).
    OpacityMicromapManager::softwareOmmSelfTestFrame(ctx);
    prepPhase(3);
    {
      static_assert(kPrepTopPhaseCount + AccelManager::kMergeSubPhaseCount + kPrepNestedCount == kPrepPhaseCount, "prep.merge.* names");
      const double* mergeSubMs = m_accelManager.getMergeSubPhaseMs();
      for (uint32_t i = 0; i < AccelManager::kMergeSubPhaseCount; ++i) {
        m_prepPhaseMs[kPrepTopPhaseCount + i] += mergeSubMs[i];
        m_prepPhaseLastMs[kPrepTopPhaseCount + i] += mergeSubMs[i];
      }
    }

    // Call on the other managers to prepare their GPU data for the current scene
    m_accelManager.prepareSceneData(ctx, execBarriers, m_instanceManager);
    // [PerfGpu] sub-stages of the scene data and TLAS work.
    ctx->recordGpuStageTiming("SceneInstanceUpload");
    prepPhase(4);
    m_lightManager.prepareSceneData(ctx, m_cameraManager);
    ctx->recordGpuStageTiming("SceneLightUpload");
    prepPhase(5);

    // Upload surface material buffer BEFORE the GPU culling dispatch so the
    // compute shader can copy template material entries to per-instance slots.
    // For PointInstancer duplicate entries we skip writeGPUData and advance past
    // the gap - the GPU shader will fill those slots.
    //
    // When the scene is unchanged (fast-skip path in mergeInstancesIntoBlas),
    // the surface order and material data are normally identical to last frame.
    // Baked terrain materials are updated independently of acceleration-structure
    // scene generation, so keep their surface-material upload live.
    const bool startInMediumStateChanged = m_startInMediumMaterialIndex_inCache != m_lastUploadedStartInMediumMaterialIndexInCache;
    bool materialSlotFlagsThisFrame = false; // [PointInstancerCheck]
    const bool updateSurfaceMaterials =
      !m_accelManager.wasSceneUnchangedThisFrame() ||
      TerrainBaker::needsTerrainBaking() ||
      startInMediumStateChanged;
    if (updateSurfaceMaterials) {
      DxvkBufferCreateInfo matInfo;
      matInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      matInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      matInfo.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      // A frame can have cached materials but no surface in the acceleration structure yet (for example the first
      // frames of a load); there is nothing to upload then, and a zero-sized buffer cannot be created.
      const bool hasSurfaceMaterialEntries = m_accelManager.getSurfaceCount() > 0 ||
        m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex;
      if (m_surfaceMaterialCache.getTotalCount() > 0 && hasSurfaceMaterialEntries) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterials");
        // Note: We duplicate the materials in the buffer so we don't have to do pointer chasing on the GPU
        size_t surfaceMaterialsGPUSize = m_accelManager.getSurfaceCount() * kSurfaceMaterialGPUSize;
        const uint32_t expectedSurfaceMaterialEntries = m_accelManager.getSurfaceCount()
          + (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex ? 1u : 0u);
        if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex) {
          surfaceMaterialsGPUSize += kSurfaceMaterialGPUSize;
        }

        matInfo.size = align(surfaceMaterialsGPUSize, kBufferAlignment);
        if (m_surfaceMaterialBuffer == nullptr || matInfo.size > m_surfaceMaterialBuffer->info().size) {
          m_surfaceMaterialBuffer = m_device->createBuffer(matInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Buffer");
        }

        // Material records read a few option values; capture them once (see beginMaterialGpuWriteBatch).
        beginMaterialGpuWriteBatch();
        if (RtxOptions::incrementalSurfaceMaterialUpload()) {
          uploadSurfaceMaterialsIncremental(ctx, surfaceMaterialsGPUSize);
          materialSlotFlagsThisFrame = true;
        } else {
        m_uploadedSurfaceMaterialBuffer = nullptr;
        m_uploadedSurfaceMaterialData.clear();
        std::size_t dataOffset = 0;
        uint32_t surfaceIndex = 0;
        std::vector<unsigned char> surfaceMaterialsGPUData(surfaceMaterialsGPUSize);
        for (auto&& pInstance : m_accelManager.getOrderedInstances()) {
          // For PointInstancer duplicates (entries beyond the template), skip
          // writeGPUData - the GPU culling shader copies the template material.
          const auto& surf = pInstance->surface;
          if (surf.instancesToObject != nullptr &&
              surf.surfaceIndexOfFirstInstance != SIZE_MAX &&
              surfaceIndex > surf.surfaceIndexOfFirstInstance) {
            dataOffset += kSurfaceMaterialGPUSize;
          } else {
            assert(surf.surfaceMaterialIndex < m_surfaceMaterialCache.getObjectTable().size());
            auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[surf.surfaceMaterialIndex];
            surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
            // As in uploadSurfaceMaterialsIncremental.
            m_device->getCommon()->metaSccrTess().patchMaterialRecord(
              surfaceMaterialsGPUData.data() + dataOffset - kSurfaceMaterialGPUSize, *pInstance);
          }
          surfaceIndex++;
        }

        if (m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex) {
          auto&& surfaceMaterial = m_surfaceMaterialCache.getObjectTable()[m_startInMediumMaterialIndex_inCache];
          surfaceMaterial.writeGPUData(surfaceMaterialsGPUData.data(), dataOffset, surfaceIndex);
          m_startInMediumMaterialIndex = surfaceIndex;
          surfaceIndex++;
        } else {
          m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
        }

        assert(surfaceIndex == expectedSurfaceMaterialEntries);
        assert(dataOffset == surfaceMaterialsGPUSize);
        assert(surfaceMaterialsGPUData.size() == surfaceMaterialsGPUSize);
        m_lastUploadedStartInMediumMaterialIndexInCache = m_startInMediumMaterialIndex_inCache;

        ctx->writeToBuffer(m_surfaceMaterialBuffer, 0, surfaceMaterialsGPUData.size(), surfaceMaterialsGPUData.data());
        } // !rtx.incrementalSurfaceMaterialUpload
        // rtx.coreMaterial.build writes its records in the same option-capture batch.
        if (m_coreMaterials) {
          m_coreMaterials->writeAndUpload(ctx, *this);
        }
        endMaterialGpuWriteBatch();
      }
    } else {
      m_startInMediumMaterialIndex = m_startInMediumMaterialIndex_inCache != kInvalidMaterialCacheIndex
        ? m_accelManager.getSurfaceCount()
        : SURFACE_INDEX_INVALID;
    }

    // GPU-driven PointInstancer culling: overwrites visible instance placeholders
    // in m_vkInstanceBuffer with proper transforms and masks, copies per-instance
    // surface and material data from templates. Must run after prepareSceneData
    // (which uploads placeholders) and before buildTlas.
    // Keep the core table current when the old upload was skipped; host self-check.
    if (m_coreMaterials) {
      m_coreMaterials->finishFrame(ctx, *this);
    }
    // rtx.coreMaterial.census, logging only, every 300 frames over the surface list just uploaded.
    if (RtxOptions::CoreMaterial::census()) {
      runMaterialCensus(m_accelManager.getOrderedInstances(), m_surfaceMaterialCache.getObjectTable(),
                        m_surfaceMaterialExtensionCache.getObjectTable(), m_device->getCurrentFrameId());
    }
    ctx->recordGpuStageTiming("SceneSurfaceMaterialUpload");
    prepPhase(6);
    {
      const uint32_t checkFrames = RtxOptions::accelIncrementalSelfCheckFrames();
      if (checkFrames != 0 && m_device->getCurrentFrameId() % checkFrames == 0) {
        // kGpuOwned in uploadSurfaceMaterialsIncremental.
        m_accelManager.selfCheckPointInstancers(materialSlotFlagsThisFrame ? &m_surfaceMaterialSlotFlags : nullptr, 2);
      }
    }
    m_accelManager.dispatchPointInstancerCulling(ctx, m_cameraManager, m_surfaceMaterialBuffer);
    ctx->recordGpuStageTiming("ScenePointInstancerCulling");

    // Build the TLAS
    m_accelManager.buildTlas(ctx);
    prepPhase(7);
    // The rest of the scene data and TLAS work: the TLAS build, and -- unless
    // rtx.profile.serializeGpuStageTimestamps is on -- whatever earlier work (BLAS builds/refits,
    // OMM bakes) the TLAS barrier had to wait for.
    ctx->recordGpuStageTiming("TlasBuild");

    // Todo: These updates require a lot of temporary buffer allocations and memcopies, ideally we should memcpy directly into a mapped pointer provided by Vulkan,
    // but we have to create a buffer to pass to DXVK's updateBuffer for now.
    // Skip when scene is unchanged - buffers from last frame are still valid.
    if (!m_accelManager.wasSceneUnchangedThisFrame()) {
      // Allocate the instance buffer and copy its contents from host to device memory
      DxvkBufferCreateInfo info;
      info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;

      // Surface Material Extension Buffer
      if (m_surfaceMaterialExtensionCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateSurfaceMaterialExtensions");
        const auto surfaceMaterialExtensionsGPUSize = m_surfaceMaterialExtensionCache.getTotalCount() * kSurfaceMaterialGPUSize;

        info.size = align(surfaceMaterialExtensionsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_surfaceMaterialExtensionBuffer == nullptr || info.size > m_surfaceMaterialExtensionBuffer->info().size) {
          m_surfaceMaterialExtensionBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Surface Material Extension Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> surfaceMaterialExtensionsGPUData(surfaceMaterialExtensionsGPUSize);

        uint32_t surfaceIndex = 0;
        for (auto&& surfaceMaterialExtension : m_surfaceMaterialExtensionCache.getObjectTable()) {
          surfaceMaterialExtension.writeGPUData(surfaceMaterialExtensionsGPUData.data(), dataOffset, surfaceIndex);
          surfaceIndex++;
        }

        assert(dataOffset == surfaceMaterialExtensionsGPUSize);
        assert(surfaceMaterialExtensionsGPUData.size() == surfaceMaterialExtensionsGPUSize);

        ctx->writeToBuffer(m_surfaceMaterialExtensionBuffer, 0, surfaceMaterialExtensionsGPUData.size(), surfaceMaterialExtensionsGPUData.data());
      }

      // Volume Material buffer
      if (m_volumeMaterialCache.getTotalCount() > 0) {
        ScopedGpuProfileZone(ctx, "updateVolumeMaterials");
        const auto volumeMaterialsGPUSize = m_volumeMaterialCache.getTotalCount() * kVolumeMaterialGPUSize;

        info.size = align(volumeMaterialsGPUSize, kBufferAlignment);
        info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (m_volumeMaterialBuffer == nullptr || info.size > m_volumeMaterialBuffer->info().size) {
          m_volumeMaterialBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Volume Material Buffer");
        }

        std::size_t dataOffset = 0;
        std::vector<unsigned char> volumeMaterialsGPUData(volumeMaterialsGPUSize);

        for (auto&& volumeMaterial : m_volumeMaterialCache.getObjectTable()) {
          volumeMaterial.writeGPUData(volumeMaterialsGPUData.data(), dataOffset);
        }

        assert(dataOffset == volumeMaterialsGPUSize);
        assert(volumeMaterialsGPUData.size() == volumeMaterialsGPUSize);

        ctx->writeToBuffer(m_volumeMaterialBuffer, 0, volumeMaterialsGPUData.size(), volumeMaterialsGPUData.data());
      }
    }

    ctx->emitMemoryBarrier(0,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_ACCESS_SHADER_READ_BIT);

    // rtx.coreMaterial.gpuCheck, after the point-instancer copies and the extension upload.
    if (m_coreMaterials) {
      m_coreMaterials->gpuCheck(ctx, *this);
    }

    // Update stats
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBlasCount, AccelManager::getBlasCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxBufferCount, m_bufferCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxTextureCount, textureManager.getTextureTable().size());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxReplacementTextureCount, textureManager.getActiveReplacementTextures());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxInstanceCount, m_instanceManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialCount, m_surfaceMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSurfaceMaterialExtensionCount, m_surfaceMaterialExtensionCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxVolumeMaterialCount, m_volumeMaterialCache.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxLightCount, m_lightManager.getActiveCount());
    m_device->statCounters().setCtr(DxvkStatCounter::RtxSamplers, m_samplerCache.getActiveCount());

    auto capturer = m_device->getCommon()->capturer();
    if (m_device->getCurrentFrameId() == m_beginUsdExportFrameNum) {
      capturer->triggerNewCapture();
    }
    capturer->step(ctx, ctx->getCommonObjects()->getLastKnownWindowHandle());

    // Clear the ray portal data before the next frame
    m_rayPortalManager.clear();

    // Check Anti-Culling Support:
    // When the game doesn't set up the View Matrix, we must disable Anti-Culling to prevent visual corruption.
    m_isAntiCullingSupported = (getCamera().getViewToWorld() != Matrix4d());
    prepPhase(8);
  }

  static_assert(std::is_same_v< decltype(RtSurface::objectPickingValue), ObjectPickingValue>);

  // Counts external draws by which path they took, and reports once a second.
  //
  // This exists because the failure mode of a preserve path is silence: the capture path's
  // equivalent sat unreachable from external draws indefinitely, and an unreachable fast
  // path looks exactly like one that is working but not helping. A ratio here says which.
  void SceneManager::reportExternalPreserveOutcome(bool preserved) {
    if (preserved) {
      ++m_externalDrawsPreserved;
    } else {
      ++m_externalDrawsResubmitted;
    }

    // Frame check first: it is cheap, and the option getter takes the global option mutex
    // (this runs once per preserved draw). Same outcome in either order.
    const uint32_t frameId = m_device->getCurrentFrameId();

    // Frame ids advance once per present, so this is roughly a second at any frame rate
    // worth measuring.
    if (frameId - m_externalPreserveReportFrame < 60) {
      return;
    }

    if (!RtxOptions::enableExternalPreservePath()) {
      return;
    }

    const uint32_t frames = std::max(frameId - m_externalPreserveReportFrame, 1u);
    m_externalPreserveReportFrame = frameId;

    const uint64_t preservedCount = m_externalDrawsPreserved;
    const uint64_t resubmittedCount = m_externalDrawsResubmitted;
    m_externalDrawsPreserved = 0;
    m_externalDrawsResubmitted = 0;

    const uint64_t total = preservedCount + resubmittedCount;

    if (total == 0) {
      return;
    }

    Logger::info(str::format(
      "[ExternalPreserve] per frame: preserved ", preservedCount / frames,
      " / resubmitted ", resubmittedCount / frames,
      " (", (100 * preservedCount) / total, "% preserved)"));
  }

  // ---- rtx.retained.useV2 ------------------------------------------------------------------------

  // Hands the registered instances, and what their last replay produced, to the other
  // implementation, which can then keep preserving them instead of replaying them all. The first
  // implementation's side of this is glue that goes away with it.
  void SceneManager::selectRetainedImplementation(bool useV2) {
    if (useV2 == m_retainedV2Active) {
      return;
    }
    std::vector<RetainedInstanceStore::Handover> handover;
    if (useV2) {
      handover.reserve(m_retainedExternalDraws.size());
      for (RetainedExternalDraw& entry : m_retainedExternalDraws) {
        RetainedInstanceStore::Handover item;
        item.handle = entry.handle;
        item.desc = std::move(entry.state);
        item.absolute = entry.absoluteObjectToWorld;
        item.visible = entry.visible;
        item.hiddenFromTlas = entry.hiddenFromTlas;
        item.forceFull = entry.described;
        if (entry.hasFullReplay) {
          item.record.valid = true;
          item.record.identity = entry.identityHash;
          item.record.applied = entry.replayedObjectToWorld;
          item.record.submeshes = entry.replayedSubmeshes;
          item.record.materialHash = entry.replayedMaterialHash;
        }
        handover.push_back(std::move(item));
      }
      m_retainedExternalDraws.clear();
      m_retainedExternalIndex.clear();
      unpinAllRetained();
      m_retainedPins.clear();
      m_retainedSteadyByRI.clear();
      m_retainedSteadyByAsset.clear();
      const size_t count = handover.size();
      m_retainedV2Active = true;
      m_retainedV2->importEntries(handover);
      Logger::info(str::format("[RetainedV2] rtx.retained.useV2 on: ", count, " retained instances handed to the second implementation"));
    } else {
      m_retainedV2->exportEntries(handover);
      m_retainedV2Active = false;
      m_retainedExternalDraws.reserve(m_retainedExternalDraws.size() + handover.size());
      for (RetainedInstanceStore::Handover& item : handover) {
        if (m_retainedExternalIndex.find(item.handle) != m_retainedExternalIndex.end()) {
          continue;
        }
        RetainedExternalDraw entry;
        entry.handle = item.handle;
        entry.state = std::move(item.desc);
        entry.absoluteObjectToWorld = item.absolute;
        entry.visible = item.visible;
        entry.described = item.forceFull;
        entry.hiddenFromTlas = item.hiddenFromTlas;
        entry.hasFullReplay = item.record.valid;
        entry.identityHash = item.record.identity;
        entry.replayedObjectToWorld = item.record.applied;
        entry.replayedSubmeshes = item.record.submeshes;
        entry.replayedMaterialHash = item.record.materialHash;
        m_retainedExternalIndex.emplace(entry.handle, m_retainedExternalDraws.size());
        m_retainedExternalDraws.push_back(std::move(entry));
      }
      unpinAllRetained();
      m_retainedPins.assign(m_retainedExternalDraws.size(), RetainedPin {});
      Logger::info(str::format("[RetainedV2] rtx.retained.useV2 off: ", handover.size(), " retained instances handed to the first implementation"));
    }
  }

  void SceneManager::setRetainedExternalDraw(uint64_t handle, ExternalDrawState&& state) {
    if (m_retainedV2Active) {  // rtx.retained.useV2 owns the instances (queued before a switch)
      m_retainedV2->describe(handle, std::move(state));
      return;
    }
    if (m_retainedScene->active()) {  // rtx.scene.directRetained
      m_retainedScene->describe(handle, std::move(state));
      return;
    }
    setRetainedExternalDrawLegacy(handle, std::move(state));
  }

  void SceneManager::setRetainedExternalDrawLegacy(uint64_t handle, ExternalDrawState&& state) {
    const Matrix4 absolute = state.drawCall.getTransformData().objectToWorld;
    const auto found = m_retainedExternalIndex.find(handle);
    if (found != m_retainedExternalIndex.end()) {
      auto& entry = m_retainedExternalDraws[found->second];
      entry.absoluteObjectToWorld = absolute;
      entry.state = std::move(state);
      entry.described = true;
      unpinRetained(found->second);
      return;
    }
    m_retainedExternalIndex.emplace(handle, m_retainedExternalDraws.size());
    m_retainedExternalDraws.push_back(RetainedExternalDraw { handle, std::move(state), absolute, true, true });
    if (m_retainedPins.size() + 1 == m_retainedExternalDraws.size()) {
      m_retainedPins.push_back(RetainedPin {});
    }
  }

  void SceneManager::setRetainedExternalDrawTransform(uint64_t handle, const Matrix4& objectToWorld) {
    if (m_retainedV2Active) {  // rtx.retained.useV2 owns the instances (queued before a switch)
      m_retainedV2->setTransform(handle, objectToWorld);
      return;
    }
    if (m_retainedScene->active()) {  // rtx.scene.directRetained
      m_retainedScene->setTransform(handle, objectToWorld);
      return;
    }
    setRetainedExternalDrawTransformLegacy(handle, objectToWorld);
  }

  void SceneManager::setRetainedExternalDrawTransformLegacy(uint64_t handle, const Matrix4& objectToWorld) {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found != m_retainedExternalIndex.end()) {
      Matrix4& current = m_retainedExternalDraws[found->second].absoluteObjectToWorld;
      if (!(current == objectToWorld)) {
        unpinRetained(found->second);
      }
      current = objectToWorld;
    }
  }

  void SceneManager::setRetainedExternalDrawVisible(uint64_t handle, bool visible) {
    if (m_retainedV2Active) {  // rtx.retained.useV2 owns the instances (queued before a switch)
      m_retainedV2->setVisible(handle, visible);
      return;
    }
    if (m_retainedScene->active()) {  // rtx.scene.directRetained
      m_retainedScene->setVisible(handle, visible);
      return;
    }
    setRetainedExternalDrawVisibleLegacy(handle, visible);
  }

  void SceneManager::setRetainedExternalDrawVisibleLegacy(uint64_t handle, bool visible) {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found != m_retainedExternalIndex.end()) {
      if (m_retainedExternalDraws[found->second].visible != visible) {
        unpinRetained(found->second);
      }
      m_retainedExternalDraws[found->second].visible = visible;
    }
  }

  void SceneManager::removeRetainedExternalDraw(uint64_t handle) {
    if (m_retainedV2Active) {  // rtx.retained.useV2 owns the instances (queued before a switch)
      m_retainedV2->destroy(handle);
      return;
    }
    if (m_retainedScene->active()) {  // rtx.scene.directRetained
      m_retainedScene->destroy(handle);
      return;
    }
    removeRetainedExternalDrawLegacy(handle);
  }

  void SceneManager::removeRetainedExternalDrawLegacy(uint64_t handle) {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found == m_retainedExternalIndex.end()) {
      return;
    }
    // The RtInstance itself ages out through the ordinary unseen-instance GC
    // once it stops being replayed.
    const size_t index = found->second;
    const size_t last = m_retainedExternalDraws.size() - 1;
    const bool pinsInStep = m_retainedPins.size() == m_retainedExternalDraws.size();
    unpinRetained(index);
    if (index != last) {
      m_retainedExternalDraws[index] = std::move(m_retainedExternalDraws[last]);
      m_retainedExternalIndex[m_retainedExternalDraws[index].handle] = index;
      if (pinsInStep) {
        m_retainedPins[index] = m_retainedPins[last];
      }
    }
    m_retainedExternalDraws.pop_back();
    if (pinsInStep) {
      m_retainedPins.pop_back();
    }
    m_retainedExternalIndex.erase(found);
  }

  // ---- rtx.scene.directRetained: the first implementation's side of the scene's legacy lane --------------------

  bool SceneManager::implicitClearsAllowed() const {
    return m_pReplacer->hasExternalMeshes() ? !scene::Options::noImplicitClears() : !scene::Options::d3d9CutAsEvent();
  }

  void SceneManager::selectSceneImplementation(bool direct) {
    if (direct == m_retainedScene->active()) {
      return;
    }
    if (direct) {
      m_retainedScene->activate();
    } else {
      m_retainedScene->deactivate();
    }
  }

  void SceneManager::dropRetainedExternalDrawNow(uint64_t handle) {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found == m_retainedExternalIndex.end()) {
      return;
    }
    const RetainedExternalDraw& entry = m_retainedExternalDraws[found->second];
    if (entry.hasFullReplay) {
      if (ReplacementInstance* replacementInstance = m_drawCallTracker.findReplacementInstanceByIdentity(entry.identityHash)) {
        replacementInstance->clear();
      }
    }
    removeRetainedExternalDrawLegacy(handle);
  }

  void SceneManager::listRetainedExternalDrawsLegacy(std::vector<std::pair<uint64_t, bool>>& out) const {
    out.reserve(out.size() + m_retainedExternalDraws.size());
    for (const RetainedExternalDraw& entry : m_retainedExternalDraws) {
      out.emplace_back(entry.handle, entry.visible);
    }
  }

  bool SceneManager::copyRetainedExternalDrawLegacy(uint64_t handle, ExternalDrawState& out) const {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found == m_retainedExternalIndex.end()) {
      return false;
    }
    const RetainedExternalDraw& entry = m_retainedExternalDraws[found->second];
    out = entry.state;
    out.drawCall.modifyTransformData().objectToWorld = entry.absoluteObjectToWorld;
    return true;
  }

  // rtx.retainedPreciseInvalidation: drop the cached steady result (and the pin) of the entry with
  // this retained handle; with ri set, only if that is still the replacement instance it cached.
  void SceneManager::dropRetainedSteady(uint64_t handle, const ReplacementInstance* ri) {
    const auto found = m_retainedExternalIndex.find(handle);
    if (found == m_retainedExternalIndex.end()) {
      return;
    }
    RetainedExternalDraw& entry = m_retainedExternalDraws[found->second];
    if (ri != nullptr && entry.steadyRI != ri) {
      return;
    }
    entry.steadyRI = nullptr;
    unpinRetained(found->second);
  }

  void SceneManager::consumeRetainedSteadyInvalidations() {
    if (!m_retainedPreciseOn) {
      return;
    }
    const auto& invalidations = m_drawCallTracker.getRetainedSteadyInvalidations();
    for (size_t i = m_retainedInvalidationsConsumed; i < invalidations.size(); ++i) {
      const auto found = m_retainedSteadyByRI.find(invalidations[i]);
      if (found != m_retainedSteadyByRI.end()) {
        dropRetainedSteady(found->second, invalidations[i]);
        m_retainedSteadyByRI.erase(found);
      }
    }
    m_retainedInvalidationsConsumed = invalidations.size();
  }

  void SceneManager::wakeRetainedPinsOf(BlasEntry& blas) {
    if (blas.retainedPinGeneration != m_retainedPinGeneration || m_retainedPinCount == 0) {
      return;
    }
    for (RtInstance* instance : blas.getLinkedInstances()) {
      if (instance->getRetainedPinGeneration() != m_retainedPinGeneration) {
        continue;
      }
      const auto found = m_retainedExternalIndex.find(instance->getRetainedPinHandle());
      if (found != m_retainedExternalIndex.end()) {
        unpinRetained(found->second);
      }
      instance->setRetainedPin(0, 0);
    }
  }

  void SceneManager::setRetainedExternalDrawHiddenFromTlas(RetainedExternalDraw& entry, bool hidden) {
    entry.hiddenFromTlas = hidden;
    if (!entry.hasFullReplay) {
      return;
    }
    ReplacementInstance* replacementInstance =
      m_drawCallTracker.findReplacementInstanceByIdentity(entry.identityHash);
    if (replacementInstance == nullptr) {
      return;
    }
    for (const PrimInstance& prim : replacementInstance->prims) {
      if (RtInstance* instance = prim.getInstance()) {
        m_instanceManager.setInstanceHiddenExternally(*instance, hidden);
      }
    }
  }

  // The retained counterpart of the external preserve path. An entry whose description,
  // transform, mesh geometry and material are all what its last full replay used can
  // refresh the instance that replay produced, instead of copying its ExternalDrawState
  // and re-running submitExternalDraw (sampler setup, identity hashing, replacement
  // lookup, draw-call cache, geometry processing, instance and material update).
  //
  // Unlike rtx.enableExternalPreservePath it does not rely on the draw-call tracker's
  // dirty flags to see a moved transform: a retained draw's identity ignores its
  // transform, so an L1 hit reports "unchanged" for a moved instance. The entry's own
  // record of what it last replayed is checked instead.
  bool SceneManager::tryPreserveRetainedExternalDraw(RetainedExternalDraw& entry, const Matrix4& objectToWorld,
                                                     bool steadyOn, bool verifySteady, bool* tookSteady) {
    // rtx.retainedSteadyFastPath: the cached result of the entry-level lookups below still holds
    // while neither epoch moved (see the option). Checked against the live tracker epoch, since
    // full replays earlier in this loop can destroy or re-key a cached replacement instance.
    consumeRetainedSteadyInvalidations();
    const bool steady = steadyOn && entry.steadyRI != nullptr &&
      entry.steadyAssetEpoch == m_retainedSteadyAssetEpoch &&
      entry.steadyTrackerEpoch == retainedSteadyTrackerEpoch();
    const bool useSteady = steady && !verifySteady;
    // Any rejection drops the cached result; the next success re-caches it.
    const auto reject = [&entry]() { entry.steadyRI = nullptr; return false; };
    const auto steadyMismatch = [&](const char* what) {
      ++m_retainedSteadyMismatches;
      Logger::warn(str::format("[RetainedSteadyCheck] frame ", m_device->getCurrentFrameId(), " handle ", entry.handle,
        ": the steady fast path would have preserved this entry but the full check ", what,
        ". Mismatches so far: ", m_retainedSteadyMismatches));
    };

    // Read once per replay in submitRetainedExternalDraws (option getters lock a mutex).
    if (!m_retainedPreserveOptionsOn) {
      return reject();
    }
    if (entry.described || !entry.hasFullReplay || entry.hiddenFromTlas ||
        !(entry.replayedObjectToWorld == objectToWorld)) {
      return reject();
    }
    const ExternalDrawState& state = entry.state;
    if (state.optionalParticleDesc.has_value() ||
        state.drawCall.getSkinningState().numBones > 0) {
      return reject();
    }
    if (RtxOptionManager::isDrawcallTranslationInvalid() ||
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration() !=
          m_textureCacheGenerationValidForPreserve) {
      return reject();
    }

    ReplacementInstance* replacementInstance = useSteady
      ? entry.steadyRI
      : m_drawCallTracker.findReplacementInstanceByIdentity(entry.identityHash);
    if (steady && verifySteady && replacementInstance != entry.steadyRI) {
      steadyMismatch("found a different replacement instance");
    }
    // A prim of a USD replacement takes the full replay, which rebuilds it when the replacement has gone.
    if (replacementInstance == nullptr ||
        replacementInstance->prims.size() != 1 ||
        replacementInstance->activeReplacements != nullptr ||
        !replacementInstance->dirtyFlags.isClear()) {
      return reject();
    }
    RtInstance* instance = replacementInstance->prims[0].getInstance();
    if (instance == nullptr || instance->isMarkedForGC() ||
        instance->testCategoryFlags(InstanceCategories::ParticleEmitter)) {
      return reject();
    }
    BlasEntry* pBlas = instance->getBlas();
    const uint32_t currentFrameId = m_device->getCurrentFrameId();
    // Another draw sharing this BlasEntry rebuilt its geometry this frame.
    if (pBlas == nullptr || pBlas->frameLastUpdated == currentFrameId) {
      return reject();
    }

    std::shared_ptr<const std::vector<RasterGeometry>> submeshes;
    const AxisAlignedBoundingBox* submeshBoundingBox = &entry.steadyBbox;
    if (!useSteady) {
      // The mesh geometry and material the instance was built from are still current.
      // A geometry refresh publishes a new submesh vector, so identity of the vector is
      // enough; the weak reference cannot be fooled by address reuse.
      submeshes = m_pReplacer->accessExternalMesh(state.mesh);
      if (submeshes == nullptr || submeshes->size() != 1 ||
          submeshes.get() != entry.replayedSubmeshes.lock().get()) {
        if (steady) {
          steadyMismatch("found the mesh changed");
        }
        return reject();
      }
      const MaterialData* material = resolveExternalDrawMaterial(state, (*submeshes)[0]);
      const XXH64_hash_t materialHash = material != nullptr ? material->getHash() : kEmptyHash;
      if (materialHash != entry.replayedMaterialHash ||
          (material != nullptr && material->getParticleSystemDesc() != nullptr) ||
          RtxOptions::shouldConvertToLight(materialHash)) {
        if (steady) {
          steadyMismatch("found the material changed or converted to a light");
        }
        return reject();
      }
      if (fork_hooks::externalDrawMeshReplacement(*m_pReplacer, reinterpret_cast<XXH64_hash_t>(state.mesh)) != nullptr) {
        if (steady) {
          steadyMismatch("found a USD mesh replacement");
        }
        return reject();
      }
      submeshBoundingBox = &(*submeshes)[0].boundingBox;
      if (steady && (entry.steadyBbox.minPos != submeshBoundingBox->minPos ||
                     entry.steadyBbox.maxPos != submeshBoundingBox->maxPos)) {
        steadyMismatch("found different submesh bounds");
      }
      if (steadyOn) {
        replacementInstance->retainedSteady = true;
        entry.steadyRI = replacementInstance;
        entry.steadyTrackerEpoch = retainedSteadyTrackerEpoch();
        entry.steadyAssetEpoch = m_retainedSteadyAssetEpoch;
        entry.steadyBbox = *submeshBoundingBox;
        if (m_retainedPreciseOn) {
          // rtx.retainedPreciseInvalidation: what can invalidate this entry's cached result on its
          // own -- its replacement instance being destroyed or re-keyed, and its mesh, submesh
          // material or override material being registered, refreshed or destroyed.
          m_retainedSteadyByRI.insert_or_assign(replacementInstance, entry.handle);
          const auto index = [&](uint64_t asset, uint64_t& indexed) {
            if (asset != 0 && asset != indexed) {
              m_retainedSteadyByAsset.emplace(asset, entry.handle);
              indexed = asset;
            }
          };
          index(reinterpret_cast<uint64_t>(state.mesh), entry.indexedMesh);
          index(reinterpret_cast<uint64_t>((*submeshes)[0].externalMaterial), entry.indexedMaterial);
          index(reinterpret_cast<uint64_t>(state.materialOverride), entry.indexedOverride);
        }
      } else {
        entry.steadyRI = nullptr;
      }
    } else if (tookSteady) {
      *tookSteady = true;
    }

    // BlasEntry::input is the last full draw state for this geometry. It may come from
    // another instance sharing the BlasEntry, which is why the bounding box below uses
    // this entry's own transform; the preserve helpers only read its camera type and
    // geometry, which are the same for every draw of this mesh.
    const DrawCallState& input = pBlas->input;
    const ObjectPickingValue pickingValue = externalObjectPickingValue(state.drawCall, *replacementInstance);
    preserveInstancesWithObjectPicking(
      replacementInstance->prims,
      pickingValue,
      [&](RtInstance& preserved) {
        preserveInstance(preserved, &input);
        m_instanceManager.preserveInstance(preserved, input, nullptr);
      },
      [&] {
        // Not from `input`: another instance sharing the geometry may have drawn it last.
        trackObjectPickingMeta(pickingValue, replacementInstance->externalPickingTextureHash);
      });

    replacementInstance->recalculateBoundingBox(objectToWorld, submeshBoundingBox);
    replacementInstance->frameLastSeen = currentFrameId;
    reportExternalPreserveOutcome(true);
    return true;
  }

  void SceneManager::noteRetainedFullReplay(uint64_t handle, remixapi_MeshHandle mesh, XXH64_hash_t identity,
                                            bool redescribed, bool visible) {
    RetainedDiag& diag = m_retainedDiag;
    ++diag.fullReplays;
    const auto note = [&](RetainedDiag::Reason reason) {
      uint32_t& count = diag.counts[reason];
      if (count < 4) {
        diag.examples[reason][count] = handle;
      }
      ++count;
    };
    const auto submeshes = m_pReplacer->accessExternalMesh(mesh);
    if (submeshes == nullptr || submeshes->empty()) {
      note(RetainedDiag::NoMesh);
      return;
    }
    if (identity == 0) {
      note(RetainedDiag::NoIdentity);
      return;
    }
    ReplacementInstance* replacementInstance = m_drawCallTracker.findReplacementInstanceByIdentity(identity);
    if (replacementInstance == nullptr) {
      note(RetainedDiag::NoReplacement);
      return;
    }
    // Not re-described, yet its replacement instance is new this frame: the one its previous replay
    // produced was destroyed in between (garbage-collected as unseen, or taken by another draw).
    if (!redescribed && replacementInstance->frameCreated == m_device->getCurrentFrameId()) {
      note(RetainedDiag::Recreated);
    }
    RtInstance* instance = replacementInstance->prims.empty() ? nullptr : replacementInstance->prims[0].getInstance();
    if (instance == nullptr) {
      note(RetainedDiag::NoInstance);
      return;
    }
    if (visible && instance->isHidden()) {
      note(RetainedDiag::Hidden);
    }
    if (instance->isMarkedForGC()) {
      note(RetainedDiag::MarkedForGC);
    }
  }

  void SceneManager::logRetainedDiagnostics(uint32_t replays, uint32_t lateReplays, uint32_t deferredReplays) {
    static constexpr const char* kNames[RetainedDiag::Count] = {
      "noMesh", "noIdentity", "noReplacement", "recreated", "noInstance", "hidden", "markedForGC" };
    RetainedDiag& diag = m_retainedDiag;
    std::string reasons;
    for (uint32_t r = 0; r < RetainedDiag::Count; ++r) {
      reasons += str::format(" ", kNames[r], "=", diag.counts[r]);
      if (diag.counts[r] != 0) {
        reasons += " (";
        for (uint32_t i = 0; i < std::min(diag.counts[r], 4u); ++i) {
          reasons += str::format(i ? "," : "", diag.examples[r][i]);
        }
        reasons += ")";
      }
    }
    Logger::info(str::format("[RetainedDiag] frame ", m_device->getCurrentFrameId(), " impl=",
      m_retainedV2Active ? "v2" : "v1", " entries=", m_retainedV2Active ? 0 : m_retainedExternalDraws.size(),
      " | replays=", replays, " afterInjection=", lateReplays, " deferredToInjection=", deferredReplays,
      " | fullReplays=", diag.fullReplays, reasons));
    diag = RetainedDiag {};
  }

  void SceneManager::submitRetainedExternalDraws(const Rc<DxvkContext>& ctx, RetainedReplayStats* stats) {
    ScopedCpuProfileZone();
    const uint32_t hiddenPolicy = RtxOptions::retainedHiddenPolicy();
    const bool diagnostics = RtxOptions::Retained::diagnosticFrames() != 0;
    const uint32_t currentFrameId = m_device->getCurrentFrameId();

    m_retainedPreserveOptionsOn = RtxOptions::enableRetainedPreservePath() && RtxOptions::enablePreservePath();

    // rtx.retainedSteadyFastPath: note whether any input of the skipped checks changed.
    const bool steadyOn = RtxOptions::retainedSteadyFastPath();
    const uint32_t steadyCheckFrames = RtxOptions::retainedSteadySelfCheckFrames();
    const bool verifySteady = steadyOn && steadyCheckFrames != 0 && currentFrameId % steadyCheckFrames == 0;
    // rtx.retainedPreciseInvalidation: API mesh / material changes and destroyed or re-keyed
    // replacement instances invalidate only the entries that use them, instead of every entry.
    {
      const bool precise = RtxOptions::retainedPreciseInvalidation();
      if (precise != m_retainedPreciseOn) {
        m_retainedPreciseOn = precise;
        m_retainedSteadyByRI.clear();
        m_retainedSteadyByAsset.clear();
        ++m_retainedSteadyAssetEpoch; // every cached result was made under the other rule
      }
      m_retainedInvalidationsConsumed = 0;
      std::vector<uint64_t>& changedAssets = m_retainedChangedAssetsScratch;
      m_pReplacer->takeChangedExternalAssets(changedAssets);
      if (m_retainedPreciseOn) {
        for (const uint64_t asset : changedAssets) {
          auto range = m_retainedSteadyByAsset.equal_range(asset);
          for (auto it = range.first; it != range.second; ++it) {
            dropRetainedSteady(it->second, nullptr);
          }
          m_retainedSteadyByAsset.erase(range.first, range.second);
        }
      }
      changedAssets.clear();
    }
    if (steadyOn) {
      RetainedSteadyInputs inputs;
      inputs.assetEpoch = m_retainedPreciseOn ? m_pReplacer->getExternalAssetGlobalEpoch() : m_pReplacer->getExternalAssetEpoch();
      const auto& lightConverter = RtxOptions::lightConverter();
      inputs.lightConverterCount = lightConverter.size();
      uint64_t sum = 0, mix = 0;
      for (const XXH64_hash_t h : lightConverter) {
        sum += h;
        mix ^= h * 0x9E3779B97F4A7C15ull;
      }
      inputs.lightConverterSig = sum ^ (mix << 1);
      inputs.replacementMeshes = RtxOptions::getEnableReplacementMeshes();
      if (!(inputs == m_retainedSteadyInputs)) {
        m_retainedSteadyInputs = inputs;
        ++m_retainedSteadyAssetEpoch;
      }
    }

    // rtx.retainedPinSteadyEntries: drop every pin if anything the skipped checks read changed.
    // (Per-entry changes -- host updates, geometry updates by other draws -- drop single pins where
    // they happen; see unpinRetained / wakeRetainedPinsOf.)
    const bool objectPickingActive =
      m_device->getCommon()->getResources().getRaytracingOutput().m_primaryObjectPicking.isValid();
    const bool pinOn = steadyOn && m_retainedPreserveOptionsOn && !objectPickingActive &&
      RtxOptions::retainedPinSteadyEntries();
    {
      RetainedPinInputs inputs;
      consumeRetainedSteadyInvalidations();
      inputs.trackerEpoch = retainedSteadyTrackerEpoch();
      inputs.assetEpoch = m_retainedSteadyAssetEpoch;
      const uint64_t textureGeneration = m_device->getCommon()->getTextureManager().getTextureCacheGeneration();
      inputs.textureCacheGeneration = textureGeneration;
      inputs.sceneOrigin = m_retainedSceneOrigin;
      inputs.hiddenPolicy = hiddenPolicy;
      const bool changed =
        inputs.trackerEpoch != m_retainedPinInputs.trackerEpoch ||
        inputs.assetEpoch != m_retainedPinInputs.assetEpoch ||
        inputs.textureCacheGeneration != m_retainedPinInputs.textureCacheGeneration ||
        !(inputs.sceneOrigin == m_retainedPinInputs.sceneOrigin) ||
        inputs.hiddenPolicy != m_retainedPinInputs.hiddenPolicy;
      // The steady path itself rejects every entry while these hold, so no entry may stay pinned.
      const bool preserveBlocked = RtxOptionManager::isDrawcallTranslationInvalid() ||
        textureGeneration != m_textureCacheGenerationValidForPreserve;
      if (!pinOn || changed || preserveBlocked || m_retainedPins.size() != m_retainedExternalDraws.size()) {
        unpinAllRetained();
        m_retainedPins.resize(m_retainedExternalDraws.size());
      }
      m_retainedPinInputs = inputs;
    }
    const bool canPin = pinOn && !verifySteady;
    uint32_t pinnedThisFrame = 0;

    // The per-entry path for an entry that is not (or no longer) pinned.
    const auto processEntry = [&](size_t entryIndex, bool wasPinned) {
      RetainedExternalDraw& entry = m_retainedExternalDraws[entryIndex];
      // Visibility is the host's. What a hidden entry does is rtx.retainedHiddenPolicy:
      // 0 stops replaying it (it then ages out and is rebuilt on return), 1 keeps
      // replaying it as if visible, 2 keeps it resident but out of the TLAS.
      const bool excludeFromTlas = !entry.visible && hiddenPolicy == 2;

      if (entry.hiddenFromTlas && entry.visible) {
        // Shown again: un-hide, and take the full path once so the instance's hidden
        // state, transform and material are all re-derived for this frame.
        setRetainedExternalDrawHiddenFromTlas(entry, false);
        entry.described = true;
      }

      if (excludeFromTlas) {
        if (!entry.hiddenFromTlas) {
          setRetainedExternalDrawHiddenFromTlas(entry, true);
        }
        // Keep the instance, its BLAS and geometry alive without replaying it. Its
        // transform may go stale (scene origin moves) while hidden; the full replay on
        // return corrects it before it re-enters the TLAS.
        if (entry.hasFullReplay) {
          if (ReplacementInstance* replacementInstance =
                m_drawCallTracker.findReplacementInstanceByIdentity(entry.identityHash)) {
            replacementInstance->frameLastSeen = currentFrameId;
          }
        }
        if (stats) {
          stats->skipped++;
        }
        return;
      }

      if (!entry.visible && hiddenPolicy == 0) {
        if (stats) {
          stats->skipped++;
        }
        return;
      }

      Matrix4 objectToWorld = entry.absoluteObjectToWorld;
      objectToWorld[3].x -= m_retainedSceneOrigin.x;
      objectToWorld[3].y -= m_retainedSceneOrigin.y;
      objectToWorld[3].z -= m_retainedSceneOrigin.z;

      bool tookSteady = false;
      if (tryPreserveRetainedExternalDraw(entry, objectToWorld, steadyOn, verifySteady, &tookSteady)) {
        if (stats) {
          stats->preserved++;
          stats->steady += tookSteady ? 1 : 0;
        }
        if (wasPinned) {
          ONCE(Logger::info("[RetainedPinCheck] pinned retained entries pass the full preserve check. Further passes are not logged."));
        }
        // rtx.retainedPinSteadyEntries: an entry that took the steady path is pinned if nothing the
        // pinned path leaves out can apply to it.
        if (canPin && tookSteady && entry.steadyRI != nullptr && entry.visible && !entry.hiddenFromTlas &&
            entry.steadyRI->prims.size() == 1) {
          RtInstance* instance = entry.steadyRI->prims[0].getInstance();
          BlasEntry* blas = instance != nullptr ? instance->getBlas() : nullptr;
          if (blas != nullptr &&
              !instance->usesUnorderedApproximations() &&
              instance->getBillboardCount() == 0 &&
              !instance->isPlayerModelInstance() &&
              instance->getMaterialType() != MaterialDataType::RayPortal &&
              blas->input.cameraType == CameraType::Main) {
            m_retainedPins[entryIndex] = RetainedPin { instance, blas, entry.steadyRI };
            ++m_retainedPinCount;
            instance->setRetainedPin(entry.handle, m_retainedPinGeneration);
            blas->retainedPinGeneration = m_retainedPinGeneration;
          }
        }
        return;
      }
      if (wasPinned) {
        // rtx.retainedSteadySelfCheckFrames: the full checks would not have preserved an entry that
        // was pinned, so a pin invalidation source was missed. It takes the full path now.
        ++m_retainedPinMismatches;
        Logger::warn(str::format("[RetainedPinCheck] frame ", currentFrameId, " handle ", entry.handle,
          ": a pinned entry failed the full preserve check (a missed pin invalidation). Mismatches so far: ",
          m_retainedPinMismatches));
      }
      const auto fullStart = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

      // submitExternalDraw consumes its state (moves the instancing transforms
      // out, rewrites geometry per submesh), so replay a copy and keep the
      // registration for the next frame.
      auto replay = std::make_unique<ExternalDrawState>(entry.state);
      replay->retainedHandle = entry.handle;
      replay->forceFullUpdate = entry.described;
      const bool redescribed = entry.described;
      entry.described = false;
      replay->drawCall.modifyTransformData().objectToWorld = objectToWorld;
      const remixapi_MeshHandle mesh = entry.state.mesh;
      // A host move (not a re-description): the transform differs from the last full replay's.
      const bool moved = entry.hasFullReplay && !redescribed && !(entry.replayedObjectToWorld == objectToWorld);
      submitExternalDraw(ctx, std::move(replay));

      // Record what this full replay used, for the preserve check next frame.
      entry.identityHash = m_lastExternalDrawIdentityHash;
      entry.hasFullReplay = entry.identityHash != 0;
      entry.replayedObjectToWorld = objectToWorld;
      const auto submeshes = m_pReplacer->accessExternalMesh(mesh);
      entry.replayedSubmeshes = submeshes;
      const MaterialData* material = (submeshes != nullptr && !submeshes->empty())
        ? resolveExternalDrawMaterial(entry.state, (*submeshes)[0]) : nullptr;
      entry.replayedMaterialHash = material != nullptr ? material->getHash() : kEmptyHash;
      if (diagnostics) {
        noteRetainedFullReplay(entry.handle, mesh, entry.identityHash, redescribed, entry.visible);
      }
      if (diagnostics && moved) {
        // [RetainedMover] (rtx.retained.diagnosticFrames): a moving entry's replay, to follow a mover that vanishes.
        // pos is the host's absolute transform (match it to the host's own trace by position), instPos the rebased
        // translation the RT instance carries after this replay. Logged for an entry's first 20 moves, every 60th
        // after, and every move whose replacement or RT instance is missing, new this frame, hidden, marked for GC or
        // not at the replayed transform.
        static std::unordered_map<uint64_t, uint32_t> s_moverMoves;
        const uint32_t n = ++s_moverMoves[entry.handle];
        ReplacementInstance* ri = entry.hasFullReplay
          ? m_drawCallTracker.findReplacementInstanceByIdentity(entry.identityHash) : nullptr;
        RtInstance* inst = (ri != nullptr && !ri->prims.empty()) ? ri->prims[0].getInstance() : nullptr;
        const Vector3 want = objectToWorld[3].xyz();
        const Vector3 have = inst != nullptr ? inst->surface.objectToWorld[3].xyz() : Vector3(0.f);
        const Vector3 delta = have - want;
        const bool off = inst != nullptr && dot(delta, delta) > 1.0e-2f;
        const bool odd = ri == nullptr || inst == nullptr || ri->frameCreated == currentFrameId ||
          inst->isHidden() || inst->isMarkedForGC() || off;
        if (n <= 20 || n % 60 == 0 || odd) {
          const Vector3 hostPos = entry.absoluteObjectToWorld[3].xyz();
          Logger::info(str::format("[RetainedMover] frame ", currentFrameId, " handle ", entry.handle, " move #", n,
            " pos=(", hostPos.x, ", ", hostPos.y, ", ", hostPos.z, ") visible=", entry.visible ? 1 : 0,
            " | ri=", ri != nullptr ? int64_t(ri->id) : int64_t(-1),
            ri != nullptr && ri->frameCreated == currentFrameId ? " (new this frame)" : "",
            " inst=", inst != nullptr ? int64_t(inst->getId()) : int64_t(-1),
            " hidden=", inst != nullptr && inst->isHidden() ? 1 : 0,
            " gc=", inst != nullptr && inst->isMarkedForGC() ? 1 : 0,
            " instPos=(", have.x, ", ", have.y, ", ", have.z, ")", off ? " NOT AT THE REPLAYED TRANSFORM" : ""));
        }
      }
      if (stats) {
        stats->full++;
        stats->fullMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fullStart).count();
      }
    };

    m_retainedDeferredPins.clear();
    for (size_t entryIndex = 0; entryIndex < m_retainedExternalDraws.size(); ++entryIndex) {
      // Pinned: keep the instance, its geometry and its replacement instance alive for this frame
      // (the steady path's per-frame side effects for such an entry), without touching the entry.
      bool wasPinned = false;
      if (m_retainedPinCount != 0 && m_retainedPins[entryIndex].instance != nullptr) {
        consumeRetainedSteadyInvalidations();
        if (retainedSteadyTrackerEpoch() != m_retainedPinInputs.trackerEpoch) {
          // A full replay earlier in this loop destroyed or re-keyed a replacement instance.
          unpinAllRetained();
          m_retainedPinInputs.trackerEpoch = retainedSteadyTrackerEpoch();
        } else if (m_retainedPins[entryIndex].instance == nullptr) {
          // Unpinned just now by a precise invalidation: the per-entry path below.
        } else if (verifySteady) {
          // rtx.retainedSteadySelfCheckFrames: the pinned entry takes the full checks below instead.
          unpinRetained(entryIndex);
          wasPinned = true;
        } else {
          // Stamped after the loop, on worker threads (rtx.retainedParallelPinnedStamps).
          m_retainedDeferredPins.push_back(uint32_t(entryIndex));
          continue;
        }
      }
      processEntry(entryIndex, wasPinned);
    }

    // Pinned entries, after every other entry (so a draw of this frame that updated pinned geometry
    // has already woken its pins). A replay above may have destroyed or re-keyed a replacement
    // instance, in which case no pin pointer may be used; entries woken above take the per-entry
    // path, which can wake others, so repeat until nothing changes.
    {
      std::vector<uint32_t>& deferred = m_retainedDeferredPins;
      bool changed = true;
      while (changed && !deferred.empty()) {
        changed = false;
        consumeRetainedSteadyInvalidations();
        if (retainedSteadyTrackerEpoch() != m_retainedPinInputs.trackerEpoch) {
          unpinAllRetained();
          m_retainedPinInputs.trackerEpoch = retainedSteadyTrackerEpoch();
        }
        size_t kept = 0;
        for (size_t k = 0; k < deferred.size(); ++k) {
          const uint32_t entryIndex = deferred[k];
          if (m_retainedPins[entryIndex].instance == nullptr) {
            processEntry(entryIndex, false);
            changed = true;
          } else {
            deferred[kept++] = entryIndex;
          }
        }
        deferred.resize(kept);
      }

      // What the steady path's preserveInstancesWithObjectPicking call did for each entry, minus
      // what cannot have changed while it is pinned: the picking value (the entry's draw call id),
      // the picking meta (picking is off), the billboard refresh (a pinned instance is not unordered
      // and has no billboards) and the bounding-box refresh (same inputs).
      const auto stampSerial = [&](uint32_t entryIndex) {
        const RetainedPin& pin = m_retainedPins[entryIndex];
        preserveInstance(*pin.instance, nullptr);
        m_instanceManager.preserveInstance(*pin.instance, pin.blas->input, nullptr);
        pin.replacementInstance->frameLastSeen = currentFrameId;
      };

      // rtx.retainedParallelPinnedStamps: the stamps are per instance / per replacement instance;
      // the only shared field is the geometry's frameLastTouched, written with the same value by
      // every thread. Anything else those two preserve calls can do -- the first-frame preserve-path
      // reset, releasing the geometry's previous-position buffer, view-model registration, or an
      // instance event that has work to do (the OMM manager's texel calculation) -- sends the entry
      // to the serial path below. The instance event handlers are only the scene manager's (a no-op
      // on the preserve path) and the OMM manager's; with any other registered, everything is serial.
      const bool ommActive = m_opacityMicromapManager != nullptr;
      const bool parallelStamps = RtxOptions::retainedParallelPinnedStamps() && deferred.size() >= 1024 &&
        m_instanceManager.getEventHandlerCount() == (ommActive ? 2u : 1u);
      if (!parallelStamps) {
        for (const uint32_t entryIndex : deferred) {
          stampSerial(entryIndex);
        }
      } else {
        std::vector<uint8_t>& serialFlags = m_retainedStampSerialScratch;
        serialFlags.assign(deferred.size(), 0);
        constexpr uint32_t kStampChunk = 512;
        const uint32_t stampCount = uint32_t(deferred.size());
        const uint32_t stampChunks = (stampCount + kStampChunk - 1) / kStampChunk;
        concurrency::parallel_for<uint32_t>(0, stampChunks, [&](uint32_t chunk) {
          const uint32_t end = std::min(stampCount, (chunk + 1) * kStampChunk);
          for (uint32_t k = chunk * kStampChunk; k < end; ++k) {
            if (k + 4 < end) {
              const RetainedPin& ahead = m_retainedPins[deferred[k + 4]];
              _mm_prefetch(reinterpret_cast<const char*>(ahead.instance), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(&ahead.blas->frameLastTouched), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(ahead.replacementInstance), _MM_HINT_T0);
            }
            const RetainedPin& pin = m_retainedPins[deferred[k]];
            RtInstance& instance = *pin.instance;
            BlasEntry& blas = *pin.blas;
            // Shared between threads (several pinned entries draw one geometry): accessed through
            // volatile, every writer storing the same value.
            volatile uint32_t& blasTouched = blas.frameLastTouched;
            const CameraType::Enum cameraType = blas.input.cameraType;
            const bool needsSerial =
              !instance.surface.isPreservePath ||
              (blas.modifiedGeometryData.previousPositionBuffer.defined() &&
               blasTouched != currentFrameId) ||
              cameraType == CameraType::ViewModel ||
              (ommActive && OpacityMicromapManager::needsPreserveUpdate(instance));
            if (needsSerial) {
              serialFlags[k] = 1;
              continue;
            }
            // SceneManager::preserveInstance, as it runs for a pinned instance.
            instance.setFrameLastUpdated(currentFrameId);
            blasTouched = currentFrameId;
            // InstanceManager::preserveInstance: camera registration; the events are no-ops here.
            instance.registerCamera(cameraType, currentFrameId);
            pin.replacementInstance->frameLastSeen = currentFrameId;
          }
        });
        for (uint32_t k = 0; k < stampCount; ++k) {
          if (serialFlags[k] != 0) {
            stampSerial(deferred[k]);
          }
        }
      }
      pinnedThisFrame = uint32_t(deferred.size());
      if (stats) {
        stats->preserved += pinnedThisFrame;
        stats->steady += pinnedThisFrame;
        stats->pinned += pinnedThisFrame;
      }
    }

    consumeRetainedSteadyInvalidations();
    m_drawCallTracker.clearRetainedSteadyInvalidations();
    m_retainedInvalidationsConsumed = 0;

    // Pinned entries count as preserved in the [ExternalPreserve] report, as they did on the steady path.
    if (pinnedThisFrame != 0) {
      m_externalDrawsPreserved += pinnedThisFrame - 1;
      reportExternalPreserveOutcome(true);
    }
  }

  // Material for one submesh of an external draw. A live per-instance override
  // (remixapi_InstanceInfoMaterialOverrideEXT) wins over the material the mesh was created with.
  const MaterialData* SceneManager::resolveExternalDrawMaterial(const ExternalDrawState& state,
                                                               const RasterGeometry& submesh) const {
    // Frontend descriptions carry their render material.
    if (state.resolvedMaterial != nullptr) {
      return state.resolvedMaterial.get();
    }
    if (state.materialOverride != nullptr) {
      if (const MaterialData* overrideMaterial = m_pReplacer->accessExternalMaterial(state.materialOverride)) {
        return overrideMaterial;
      }
    }
    return m_pReplacer->accessExternalMaterial(submesh.externalMaterial);
  }

  // BlasEntry identity for material-override draws (DrawCallState::geometryShareKey).
  // Everything that changes the BLAS itself or its opacity handling stays in the key:
  // categories (texture categories can turn a draw into a decal / cutout), and, for any
  // material that alpha-tests or blends, the material hash. Opaque materials all map to the
  // same key, so their instances share one geometry copy and one BLAS. OMMs are only built
  // for alpha-tested / blended surfaces and depend on the opacity texture, so they never end
  // up on a BLAS whose instances disagree on it (the dynamic-BLAS path additionally refuses
  // OMMs for BLASes with several instances).
  static XXH64_hash_t computeGeometryShareKey(const DrawCallState& drawCall, const MaterialData& material) {
    const LegacyMaterialData& legacy = drawCall.getMaterialData();
    bool opacityVaries =
      legacy.alphaTestEnabled ||
      (AlphaTestType) legacy.alphaTestCompareOp != AlphaTestType::kAlways ||
      legacy.blendMode.enableBlending;
    if (material.getType() != MaterialDataType::Opaque) {
      opacityVaries = true;
    } else {
      const OpaqueMaterialData& opaque = material.getOpaqueMaterialData();
      if (!opaque.getUseLegacyAlphaState() &&
          (opaque.getAlphaTestType() != AlphaTestType::kAlways || opaque.getBlendEnabled())) {
        opacityVaries = true;
      }
    }

    struct {
      uint32_t categories;
      uint32_t opacityVaries;
      XXH64_hash_t materialHash;
    } key {};
    key.categories = drawCall.getCategoryFlags().raw();
    key.opacityVaries = opacityVaries ? 1u : 0u;
    key.materialHash = opacityVaries ? material.getHash() : kEmptyHash;
    // Never kEmptyHash, so a zeroed key cannot be confused with "no key".
    return XXH3_64bits(&key, sizeof(key)) | 1ull;
  }

  std::shared_ptr<const std::vector<RasterGeometry>> SceneManager::externalSubmeshes(const ExternalDrawState& state) const {
    if (state.inlineSubmeshes != nullptr) {
      return state.inlineSubmeshes;
    }
    return m_pReplacer->accessExternalMesh(state.mesh);
  }

  void SceneManager::prepareExternalDrawCall(ExternalDrawState& state) {
    // A frontend-resolved draw keeps its own samplers (D3D9 sampler state).
    if (!state.frontendResolved) {
      Rc<DxvkSampler> externalSampler = getOrCreateExternalSampler();
      state.drawCall.modifyMaterialData().samplers[0] = externalSampler;
      state.drawCall.modifyMaterialData().samplers[1] = externalSampler;
    }
    {
      const RtCamera& rtCamera = m_cameraManager.getCamera(state.cameraType);
      state.drawCall.modifyTransformData().worldToView = rtCamera.getWorldToViewf();
      state.drawCall.modifyTransformData().viewToProjection = rtCamera.getViewToProjectionf();
      state.drawCall.modifyTransformData().objectToView = state.drawCall.getTransformData().worldToView * state.drawCall.getTransformData().objectToWorld;
    }

    if (!state.gpuInstancingTransforms.empty()) {
      state.drawCall.modifyTransformData().instancesToObject =
        std::make_shared<const std::vector<Matrix4>>(std::move(state.gpuInstancingTransforms));
      state.drawCall.modifyTransformData().instancesFromHost = true;
    }
  }

  const MaterialData& SceneManager::prepareExternalSubmeshDrawCall(ExternalDrawState& state,
                                                                   const std::shared_ptr<const std::vector<RasterGeometry>>& submeshesRef,
                                                                   size_t index,
                                                                   std::shared_ptr<MaterialData>& keepAlive) {
    const std::vector<RasterGeometry>& submeshes = *submeshesRef;
    // submeshes[index] is an element of the vector owned by submeshesRef, not separately
    // allocated. The aliasing ctor shares submeshesRef's ref-count while pointing at
    // the element, keeping the vector alive if destroyExternalMesh erases the entry
    // while a BlasEntry still holds this pointer.
    state.drawCall.overrideGeometryData(
      std::shared_ptr<const RasterGeometry>(submeshesRef, &submeshes[index]));

    if (state.frontendResolved) {
      // The frontend's own draw: its cull mode, material hash and categories stay; picking meta is tracked by the
      // build (trackObjectPickingMeta) as for D3D9 draws.
      keepAlive = nullptr;
      state.drawCall.externalPickingTextureHash = kEmptyHash;
      static MaterialData defaultResolvedMaterial(LegacyMaterialData::createDefault());
      state.drawCall.materialIndependentGeometry = false;
      state.drawCall.geometryShareKey = kEmptyHash;
      return state.resolvedMaterial != nullptr ? *state.resolvedMaterial : defaultResolvedMaterial;
    }
    state.drawCall.overrideCullMode(state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);

    XXH64_hash_t textureHash = 0;

    const MaterialData* material = resolveExternalDrawMaterial(state, submeshes[index]);
    keepAlive = nullptr;
    if (material != nullptr) {
      // Texture categories and the picking hash key off the draw's own albedo, as a D3D9 draw's
      // key off its legacy texture, not off a USD replacement material's.
      fork_hooks::externalDrawTextureCategories(material, state.drawCall, textureHash);

      keepAlive = fork_hooks::externalDrawMaterialReplacement(*m_pReplacer, material);

      state.drawCall.modifyMaterialData().setHashOverride(material->getHash());
    }

    // Recorded for object picking by the build (trackObjectPickingMeta).
    state.drawCall.externalPickingTextureHash = textureHash != 0 ? textureHash : kEmptyHash;

    static MaterialData defaultMaterialData(LegacyMaterialData::createDefault());
    const MaterialData& materialData = material != nullptr ? *material : defaultMaterialData;

    // Per-instance material override. Geometry identity in the draw-call cache
    // stops depending on the material (see DrawCallState::materialIndependentGeometry).
    const bool shareAcrossMaterials = state.materialOverride != nullptr &&
      RtxOptions::HostImport::shareGeometryAcrossMaterials();
    state.drawCall.materialIndependentGeometry = shareAcrossMaterials;
    state.drawCall.geometryShareKey = shareAcrossMaterials
      ? computeGeometryShareKey(state.drawCall, materialData)
      : kEmptyHash;
    return materialData;
  }

  void SceneManager::submitExternalDraw(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> pstate) {
    ScopedCpuProfileZone();
    // rtx.worldOrigin.enable: as on the D3D9 draw path, shift history before this frame's instances land.
    applyPendingHostWorldOrigin();
    m_lastExternalDrawIdentityHash = 0;

    auto& state = *pstate;

    prepareExternalDrawCall(state);

    const XXH64_hash_t meshHash = reinterpret_cast<XXH64_hash_t>(state.mesh);

    // Fetch submeshes once: they drive both the replacement path (needs submeshes[0]
    // as geometry template) and the default iteration path.
    const auto submeshesRef = m_pReplacer->accessExternalMesh(state.mesh);
    if (submeshesRef == nullptr || submeshesRef->empty()) {
      Logger::err(str::format("[RTX-Mesh] External mesh has no submeshes: 0x", std::hex, meshHash, std::dec));
      return;
    }
    const auto& submeshes = *submeshesRef;

    // Persistence-tracking setup happens before the replacement-lookup early-out
    // so the same ReplacementInstance can be threaded through both paths:
    // drawReplacements() requires a non-null instance and uses it to drive
    // RtInstance reuse across frames for the replacement primitives.
    const XXH64_hash_t identityHash = state.computeExternalDrawIdentityHash();
    m_lastExternalDrawIdentityHash = identityHash;
    const bool retainedOwnMap = state.retainedHandle != 0 && m_retainedScene->active();
    const XXH64_hash_t spatialMapHash = retainedOwnMap
      ? retainedSpatialMapHashForExternalDrawMesh(state.mesh)
      : spatialMapHashForExternalDrawMesh(state.mesh);
    const Matrix4& xform = state.drawCall.getTransformData().objectToWorld;
    const XXH64_hash_t matHash = state.drawCall.getMaterialData().getHash();
    const Vector3 worldPos = xform[3].xyz();

    // With the preserve path on, the vertex-position hash is part of the lookup, so a host rewriting a mesh's
    // vertices in place (RefreshGeometryEXT, skinning, foliage wind) drifts the key and sets
    // DirtyFlag::VertexPosHash, which the preserve path below relies on. Submesh 0 stands for the draw, since
    // that path is single-submesh only.
    // Without the option the hash stays kEmptyHash: the L2 lookup also matches candidates on it
    // (DrawCallTracker::findOrCreateReplacementInstance), and a real hash would send vertex-rewriting hosts
    // into the nearest-neighbour search and could create a new instance every frame.
    const XXH64_hash_t vertexPositionHash = RtxOptions::enableExternalPreservePath()
      ? submeshes[0].hashes[HashComponents::VertexPosition]
      : kEmptyHash;

    const ReplacementInstance::LookupKey externalKey { identityHash, spatialMapHash, matHash, vertexPositionHash, worldPos, xform };
    ReplacementInstance* replacementInstance = m_drawCallTracker.findOrCreateReplacementInstance(externalKey);

    // The draw's instances are picked by this value (the instance manager copies drawCallID into
    // RtSurface::objectPickingValue). Without one from the host every API instance would share 0.
    state.drawCall.drawCallID = externalObjectPickingValue(state.drawCall, *replacementInstance);

    auto pReplacements = fork_hooks::externalDrawMeshReplacement(*m_pReplacer, meshHash);

    // A plain draw owns one prim per submesh, and an instance without a slot is never collected.
    // Prims of a replacement or of another submesh count therefore go first.
    if (pReplacements == nullptr &&
        (replacementInstance->activeReplacements != nullptr ||
         (!replacementInstance->prims.empty() && replacementInstance->prims.size() != submeshes.size()))) {
      replacementInstance->clear();
    }

    // An external draw that has not drifted since its last full submission reuses the instances it produced
    // instead of re-deriving their draw state, as submitDrawState does for D3D9 draws.
    // The conditions mirror submitDrawState's, minus those that cannot arise here (no legacy material to hash,
    // no terrain cascades, no ray-portal conversion). It is single-submesh only: preserveReplacementInstance
    // refreshes one BlasEntry::input from one DrawCallState, which is wrong for several prims.
    RtInstance* firstPrim = (replacementInstance->prims.size() == 1)
      ? replacementInstance->prims[0].getInstance() : nullptr;

    const bool externalPreserveEligible =
        RtxOptions::enableExternalPreservePath() &&
        RtxOptions::enablePreservePath() &&
        pReplacements == nullptr &&
        !state.forceFullUpdate &&
        submeshes.size() == 1 &&
        firstPrim != nullptr &&
        firstPrim->getBlas() != nullptr &&
        // A retained draw's identity ignores its transform, so the L1 lookup above
        // clears the drift flags even when the instance moved. Check the transform.
        (state.retainedHandle == 0 || firstPrim->surface.objectToWorld == xform) &&
        replacementInstance->dirtyFlags.isClear() &&
        !RtxOptionManager::isDrawcallTranslationInvalid() &&
        !state.optionalParticleDesc.has_value() &&
        m_device->getCommon()->getTextureManager().getTextureCacheGeneration() ==
            m_textureCacheGenerationValidForPreserve;

    // Dynamic-feature bits are cleared only when entering the dynamic path; a preserved
    // draw has not run processDrawCallState and must keep whatever the last full
    // submission left set.
    if (!externalPreserveEligible) {
      replacementInstance->dirtyFlags.clr(ReplacementInstance::kDynamicFeatureMask);
    }

    if (externalPreserveEligible) {
      // Same geometry and cull-mode overrides the dynamic path applies, so the refreshed
      // BlasEntry::input describes this frame's draw rather than the one that built it.
      state.drawCall.overrideGeometryData(
        std::shared_ptr<const RasterGeometry>(submeshesRef, &submeshes[0]));
      state.drawCall.overrideCullMode(state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);

      // preserveReplacementInstance overwrites BlasEntry::input with this draw call.
      // Keep the entry's material-independent identity (the share key it was matched on),
      // or the next full draw of an override instance would no longer find it.
      if (state.materialOverride != nullptr) {
        const DrawCallState& previousInput = firstPrim->getBlas()->input;
        state.drawCall.materialIndependentGeometry = previousInput.materialIndependentGeometry;
        state.drawCall.geometryShareKey = previousInput.geometryShareKey;
      }
      // The albedo hash the last full submission resolved, for object picking.
      state.drawCall.externalPickingTextureHash = replacementInstance->externalPickingTextureHash;

      preserveReplacementInstance(ctx, state.drawCall, nullptr, replacementInstance);

      replacementInstance->frameLastSeen = m_device->getCurrentFrameId();

      reportExternalPreserveOutcome(true);
      return;
    }

    reportExternalPreserveOutcome(false);

    if (pReplacements) {
      // Copy the DrawCallState so we don't mutate the caller's state. Point geometryData
      // at submeshes[0] as the replacement geometry template, clear externalMaterial so
      // the USD replacement material takes precedence, and use a neutral default material
      // since the replacement will provide its own.
      DrawCallState replacementDrawCall = state.drawCall;
      RasterGeometry& replacementGeometry = replacementDrawCall.modifyGeometryData();
      replacementGeometry = submeshes[0];
      replacementGeometry.cullMode = state.doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
      replacementGeometry.externalMaterial = nullptr;

      // A replacement that brings no material of its own (a mesh prim without a binding,
      // or an includeOriginal prim) keeps the draw's own material, with any USD material
      // replacement applied, as the D3D9 path does.
      MaterialData renderMaterialData(LegacyMaterialData::createDefault());
      if (const MaterialData* ownMaterial = resolveExternalDrawMaterial(state, submeshes[0])) {
        const std::shared_ptr<MaterialData> replacedMaterialKeepAlive =
          fork_hooks::externalDrawMaterialReplacement(*m_pReplacer, ownMaterial);
        renderMaterialData = *ownMaterial;
      }
      drawReplacements(ctx, &replacementDrawCall, pReplacements, renderMaterialData, replacementInstance);
      // drawReplacements does not mark the replacement instance seen. Unmarked, a second replaced draw of the same
      // mesh this frame could take it over through the tracker's nearest-neighbour match, and the end of the frame
      // treats it as gone, so the original and the replacement prims vanish. It is marked here, as on the D3D9 path,
      // except for a per-frame draw the host GPU-instances: marking one of those piles up point-instancer batches
      // (see docs/HANDOFF-phase2-fo4.md).
      const DrawCallTransforms& drawTransforms = state.drawCall.getTransformData();
      const bool hostInstanced = drawTransforms.instancesFromHost && drawTransforms.instancesToObject != nullptr &&
        !drawTransforms.instancesToObject->empty();
      if (retainedOwnMap || !hostInstanced) {
        replacementInstance->frameLastSeen = m_device->getCurrentFrameId();
      }
      return;
    }

    AxisAlignedBoundingBox geometryBBox;

    for (size_t i = 0; i < submeshes.size(); i++) {
      // Keeps a USD replacement material alive for the rest of this iteration when one is found.
      std::shared_ptr<MaterialData> replacementMaterialKeepAlive;
      const MaterialData& materialData = prepareExternalSubmeshDrawCall(state, submeshesRef, i, replacementMaterialKeepAlive);
      if (i == 0) {
        replacementInstance->externalPickingTextureHash = state.drawCall.externalPickingTextureHash;
      }

      const RtxParticleSystemDesc* pParticles = nullptr;
      if (state.optionalParticleDesc.has_value()) {
        pParticles = &state.optionalParticleDesc.value();
      }

      RtInstance* existingInstance = (replacementInstance->prims.size() > i)
          ? replacementInstance->prims[i].getInstance() : nullptr;

      RtInstance* instance = processDrawCallState(ctx, state.drawCall, materialData, *replacementInstance, existingInstance, pParticles);

      if (instance != nullptr) {
        if (replacementInstance->root.getUntyped() == nullptr) {
          replacementInstance->setup(PrimInstance(instance, PrimInstance::Type::Instance), submeshes.size(), nullptr);
        }
        if (replacementInstance->prims.size() > i &&
            replacementInstance->prims[i].getUntyped() != instance) {
          instance->getPrimInstanceOwner().setReplacementInstance(replacementInstance, i, instance,
              PrimInstance::Type::Instance);
        }
      }

      geometryBBox.unionWith(submeshes[i].boundingBox);
    }

    replacementInstance->frameLastSeen = m_device->getCurrentFrameId();

    if (geometryBBox.isValid()) {
      replacementInstance->geometryBoundingBox = geometryBBox;
      replacementInstance->objectToWorld = xform;
    }
  }

  // Set from remixapi_dxvk_CreateD3D9() when the client is an editor/DCC host (HdRemix).
  bool g_remixApiEditorModeEnabled = false;

  void SceneManager::destroyExternalMesh(const Rc<DxvkContext>& ctx, remixapi_MeshHandle handle) {
    if (!handle) {
      return;
    }

    m_drawCallTracker.removeReplacementInstancesWithSpatialMapHash(
        spatialMapHashForExternalDrawMesh(handle));
    // rtx.scene.directRetained: the retained draws' own spatial map (empty unless the scene interface was on).
    m_drawCallTracker.removeReplacementInstancesWithSpatialMapHash(
        retainedSpatialMapHashForExternalDrawMesh(handle));
    m_pReplacer->destroyExternalMesh(handle);

    // An API-driven client (HdRemix) stops presenting the moment its scene becomes empty, so
    // injectRTX() - and with it garbageCollection(), the sceneKeepAliveFrames-based clear() and
    // onFrameEnd()/manageTextureVram() - never runs again, and the DXVK frame id stops advancing.
    // The whole scene would stay committed and the next scene would be allocated on top of it
    // (DXVK memory is a high water mark), leaking a full scene worth of VRAM per open/close
    // cycle. The last DestroyMesh is the authoritative teardown signal and it already runs on the
    // dxvk-cs thread with a live context, so do the teardown here.
    if (ctx.ptr() != nullptr &&
        g_remixApiEditorModeEnabled &&
        RtxOptions::clearSceneOnLastExternalMeshDestroyed() &&
        !m_pReplacer->hasExternalMeshes()) {
      clear(ctx, /* needWfi = */ true);
      // DXVK doesnt free chunks for us by default (its high water mark), so hand the geometry,
      // BLAS, OMM and buffer-cache memory freed above back to the system.
      m_device->getCommon()->memoryManager().freeUnusedChunks();
    }
  }

  namespace {
    bool ifTrue_andThenSetFalse(std::atomic_bool& atomicBool) {
      bool expected = true;
      if (atomicBool.compare_exchange_strong(expected, false)) {
        return true;
      }
      return false;
    }
  } // unnamed

  void SceneManager::requestTextureVramFree() {
    m_forceFreeTextureMemory.store(true);
  }

  void SceneManager::requestVramCompaction() {
    m_forceFreeUnusedDxvkAllocatorChunks.store(true);
  }

  void SceneManager::manageTextureVram() {
    bool freeUnused = false;
    bool freeTextures = false;
    {
      if (ifTrue_andThenSetFalse(m_forceFreeTextureMemory)) {
        freeTextures = true;
        freeUnused = true;
      }
      if (ifTrue_andThenSetFalse(m_forceFreeUnusedDxvkAllocatorChunks)) {
        freeUnused = true;
      }
    }

    if (freeTextures) {
      m_device->getCommon()->getTextureManager().clear();

      if (m_opacityMicromapManager) {
        m_opacityMicromapManager->clear();
      }
    }

    if (freeUnused) {
      // DXVK doesnt free chunks for us by default (its high water mark) so force release some memory back to the system here.
      m_device->getCommon()->memoryManager().freeUnusedChunks();
    }
  }

  void SceneManager::printAllRtInstances() {
  #ifdef REMIX_DEVELOPMENT
    
    const auto& instances = m_instanceManager.getInstanceTable();
    Logger::info(str::format("=== Printing all RtInstances (", instances.size(), " total) ==="));
    
    for (size_t i = 0; i < instances.size(); ++i) {
      const RtInstance* instance = instances[i];
      if (instance != nullptr) {
        Logger::info(str::format("Instance ", i, ":"));
        instance->printDebugInfo();
      } else {
        Logger::warn(str::format("Instance ", i, ": nullptr"));
      }
    }
    
    Logger::info("=== End RtInstances Print ===");
  #endif
  }

  void SceneManager::trackReplacementMaterialHash(XXH64_hash_t materialHash) {
    if (!s_hashUsageTrackingWanted.load(std::memory_order_relaxed)) {
      perf::countD3D9(perf::D3D9Ctr::HashUsageSkipped);
      return;
    }
    if (materialHash != kEmptyHash) {
      m_currentFrameReplacementMaterialHashes[materialHash]++;
    }
  }

  bool SceneManager::isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const {
    return m_currentFrameReplacementMaterialHashes.find(materialHash) != m_currentFrameReplacementMaterialHashes.end();
  }

  uint32_t SceneManager::getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const {
    auto it = m_currentFrameReplacementMaterialHashes.find(materialHash);
    return (it != m_currentFrameReplacementMaterialHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameReplacementMaterialHashes() {
    m_currentFrameReplacementMaterialHashes.clear();
  }

  void SceneManager::trackMeshHash(XXH64_hash_t meshHash) {
    if (!s_hashUsageTrackingWanted.load(std::memory_order_relaxed)) {
      perf::countD3D9(perf::D3D9Ctr::HashUsageSkipped);
      return;
    }
    if (meshHash != kEmptyHash) {
      m_currentFrameMeshHashes[meshHash]++;
    }
  }

  bool SceneManager::isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const {
    return m_currentFrameMeshHashes.find(meshHash) != m_currentFrameMeshHashes.end();
  }

  uint32_t SceneManager::getMeshHashUsageCount(XXH64_hash_t meshHash) const {
    auto it = m_currentFrameMeshHashes.find(meshHash);
    return (it != m_currentFrameMeshHashes.end()) ? it->second : 0;
  }

  void SceneManager::clearFrameMeshHashes() {
    m_currentFrameMeshHashes.clear();
  }

  void SceneManager::logAccelMemoryBreakdown() {
    const uint32_t interval = RtxOptions::accelMemoryLogIntervalFrames();
    const uint32_t frameId = m_device->getCurrentFrameId();
    if (interval == 0 || frameId % interval != 0) {
      return;
    }
    auto mib = [](VkDeviceSize bytes) { return str::format(bytes >> 20, " MiB"); };
    auto poolSummary = [](const DxvkMemoryAllocator::ChunkUsage& usage) {
      std::string out;
      for (uint32_t p = 0; p < uint32_t(DxvkMemoryAllocator::ChunkPool::Count); p++) {
        const DxvkMemoryAllocator::PoolUsage& pool = usage.pools[p];
        out += str::format(" ", DxvkMemoryAllocator::chunkPoolName(DxvkMemoryAllocator::ChunkPool(p)), " ",
                           pool.chunkCount, "/", pool.drainingCount, "/", pool.chunkBytes >> 20, "/", pool.freeBytes >> 20);
      }
      return out;
    };

    // Geometry copies: every BlasEntry holds a device-local index copy and one vertex
    // buffer, two for geometry that has ever been refit (skinned, morphs) to keep history.
    const size_t drawCallCacheEntries = m_drawCallCache.getEntries().size();
    uint32_t entries = 0, doubleBuffered = 0, shared = 0, unlinked = 0;
    size_t linkedInstances = 0;
    VkDeviceSize indexBytes = 0, vertexBytes = 0, historyBytes = 0;
    for (auto& [hash, entry] : m_drawCallCache.getEntries()) {
      entries++;
      const RaytraceGeometry& geo = entry.modifiedGeometryData;
      indexBytes += geo.indexCacheBuffer != nullptr ? geo.indexCacheBuffer->info().size : 0;
      vertexBytes += geo.historyBuffer[0] != nullptr ? geo.historyBuffer[0]->info().size : 0;
      if (geo.historyBuffer[1] != nullptr) {
        doubleBuffered++;
        historyBytes += geo.historyBuffer[1]->info().size;
      }
      const size_t linked = entry.getLinkedInstances().size();
      linkedInstances += linked;
      shared += linked > 1 ? 1 : 0;
      unlinked += linked == 0 ? 1 : 0;
    }
    // rtx.scene.directRetained: the scene's own geometry, counted the same way.
    m_retainedScene->forEachGeometry([&](const BlasEntry& entry) {
      entries++;
      const RaytraceGeometry& geo = entry.modifiedGeometryData;
      indexBytes += geo.indexCacheBuffer != nullptr ? geo.indexCacheBuffer->info().size : 0;
      vertexBytes += geo.historyBuffer[0] != nullptr ? geo.historyBuffer[0]->info().size : 0;
      if (geo.historyBuffer[1] != nullptr) {
        doubleBuffered++;
        historyBytes += geo.historyBuffer[1]->info().size;
      }
      const size_t linked = entry.getLinkedInstances().size();
      linkedInstances += linked;
      shared += linked > 1 ? 1 : 0;
      unlinked += linked == 0 ? 1 : 0;
    });

    AccelManager::MemoryBreakdown as;
    m_accelManager.getMemoryBreakdown(as);
    const VkDeviceSize ommScratch = m_opacityMicromapManager ? m_opacityMicromapManager->getScratchBufferBytes() : 0;

    const VkDeviceSize accounted = indexBytes + vertexBytes + historyBytes + as.pooledBytes +
      as.dynamicBytes + as.scratchBytes + as.tlasBytes + as.frameBufferBytes + ommScratch;

    uint32_t retainedHidden = 0;
    for (const auto& entry : m_retainedExternalDraws) {
      retainedHidden += entry.visible ? 0 : 1;
    }

    Logger::info(str::format(
      "[AccelMem] frame ", frameId,
      " | geometry: ", entries, " BlasEntries (", shared, " shared, ", unlinked, " unlinked, ", linkedInstances, " instances), index ",
      mib(indexBytes), ", vertex ", mib(vertexBytes), ", history ", mib(historyBytes), " (", doubleBuffered, " double-buffered)",
      " | merged: ", as.bucketCount, " buckets, ", as.bucketInstances, " instances, ", as.bucketPrims, " prims; pool ", as.pooledCount,
      " BLAS ", mib(as.pooledBytes), " (idle ", mib(as.pooledIdleBytes), ")",
      " | per-geometry BLAS: ", as.dynamicCount, " ", mib(as.dynamicBytes),
      " | scratch held ", mib(as.scratchBytes), " (needed last frame ", mib(as.scratchRequiredLastFrame), "), OMM scratch ", mib(ommScratch),
      " | TLAS ", mib(as.tlasBytes), ", frame buffers ", mib(as.frameBufferBytes),
      " | sum ", mib(accounted),
      " | retained ", m_retainedExternalDraws.size(), " (", retainedHidden, " hidden)"));

    // The mesh-side memory that a long session can grow, by owner: the Remix API meshes' own buffers (replaced on every
    // geometry refresh), the BlasEntries of the draw call cache and of the retained scene (live and retired), and the
    // allocator's bytes per category over every heap. A category that climbs across a session points at its owner.
    {
      size_t apiMeshes = 0, apiSubmeshes = 0, apiSkinned = 0;
      VkDeviceSize apiDeviceLocalBytes = 0, apiHostBytes = 0;
      std::unordered_set<const DxvkBuffer*> apiBuffers;
      auto countBuffer = [&](const DxvkBufferSlice& slice) {
        if (!slice.defined() || !apiBuffers.insert(slice.buffer().ptr()).second) {
          return;
        }
        const bool deviceLocal = (slice.buffer()->memFlags() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        (deviceLocal ? apiDeviceLocalBytes : apiHostBytes) += slice.buffer()->info().size;
      };
      m_pReplacer->forEachExternalMesh([&](remixapi_MeshHandle, const std::vector<RasterGeometry>& submeshes) {
        apiMeshes++;
        for (const RasterGeometry& geometry : submeshes) {
          apiSubmeshes++;
          apiSkinned += geometry.blendWeightBuffer.defined() ? 1 : 0;
          countBuffer(geometry.positionBuffer);
          countBuffer(geometry.indexBuffer);
          countBuffer(geometry.blendWeightBuffer);
          countBuffer(geometry.blendIndicesBuffer);
        }
      });

      VkDeviceSize rtxBuffer = 0, sceneGeometry = 0, accel = 0, omm = 0;
      const VkPhysicalDeviceMemoryProperties memoryProperties = m_device->adapter()->memoryProperties();
      for (uint32_t heap = 0; heap < memoryProperties.memoryHeapCount; heap++) {
        const DxvkMemoryStats stats = m_device->getMemoryStats(heap);
        rtxBuffer += stats.usedByCategory(DxvkMemoryStats::Category::RTXBuffer);
        sceneGeometry += stats.usedByCategory(DxvkMemoryStats::Category::RTXSceneGeometry);
        accel += stats.usedByCategory(DxvkMemoryStats::Category::RTXAccelerationStructure);
        omm += stats.usedByCategory(DxvkMemoryStats::Category::RTXOpacityMicromap);
      }

      Logger::info(str::format(
        "[MeshMem] frame ", frameId,
        " | api meshes ", apiMeshes, " (", apiSubmeshes, " submeshes, ", apiSkinned, " skinned), buffers ", apiBuffers.size(),
        " device-local ", mib(apiDeviceLocalBytes), " host ", mib(apiHostBytes),
        " | blas entries: draw call cache ", drawCallCacheEntries, ", retained ", m_retainedScene->geometryCount(),
        " (", m_retainedScene->retiredGeometryCount(), " retired), retained entries ", m_retainedScene->entryCount(),
        " | allocator used: RTXBuffer ", mib(rtxBuffer), ", scene geometry ", mib(sceneGeometry), ", accel ", mib(accel),
        ", OMM ", mib(omm)));
    }

    // BLAS/TLAS build activity since the last line. "dirty" counts cached merged buckets
    // invalidated, by the first cause found in each; "rebuilt/refit/skipped" counts the merged
    // BLAS builds that followed (new instances and dirty buckets' survivors are re-bucketed
    // together, so the two do not match one to one).
    const AccelManager::BuildStats bs = m_accelManager.takeBuildStats();
    const uint32_t f = std::max(bs.frames, 1u);
    Logger::info(str::format(
      "[AccelMem] builds over ", bs.frames, " frames (", bs.unchangedFrames, " unchanged)",
      " | merged dirty: move ", bs.dirtyByMove, ", hide/show ", bs.dirtyByHide, ", destroy ", bs.dirtyByDestroy,
      ", geometry ", bs.dirtyByGeometry, ", OMM ", bs.dirtyByOmm, ", other ", bs.dirtyByOther, ", all-dirty frames ", bs.dirtyAll,
      ", OMM rebind batches ", bs.ommRebindBatches, " (", bs.ommRebindHashes, " new OMMs)",
      "; added instances ", bs.addedInstances,
      " | merged BLAS: rebuilt ", bs.bucketsBuilt, ", refit ", bs.bucketsRefit, ", unchanged-skip ", bs.bucketsBuildSkipped,
      " (", bs.bucketPrimsBuilt / f, " prims/frame), cap splits ", bs.bucketsSplitByCap, ", cluster pulls ", bs.clusterPulls, ", young folds ", bs.clusterFolds, ", OMM-BLAS refits avoided ", bs.ommBlasRefitsAvoided,
      " | per-geometry BLAS: built ", bs.dynamicBuilt, ", refit ", bs.dynamicRefit, ", refit-limit rebuilds ", bs.refitLimitRebuilds,
      "; promoted to own BLAS: moved ", bs.promotedMoved, ", hidden/shown ", bs.promotedToggled, ", chronically dirty ", bs.promotedChronic,
      ", mover backoffs ", bs.moverBackoffs,
      " | TLAS: built ", bs.tlasBuilt, ", skipped ", bs.tlasSkipped,
      " | compaction: queried ", bs.compactionQueries, ", compacted ", bs.compacted, " (", mib(bs.compactedBytesBefore),
      " -> ", mib(bs.compactedBytesAfter), "), not worth ", bs.compactionNotWorth, ", discarded ", bs.compactionDiscarded,
      " | surface upload ", mib(bs.surfaceBytesUploaded), " of ", mib(bs.surfaceBytesTotal),
      " | instance upload ", mib(bs.instanceBytesUploaded), " of ", mib(bs.instanceBytesTotal)));

    if (RtxOptions::accelDirtyLogTopN() != 0) {
      m_accelManager.logDirtyAttribution(RtxOptions::accelDirtyLogTopN(), bs.frames, m_instanceManager.getInstanceTable());
    }

    // Device-local heaps: allocated vs used, and how much of the gap sits in partially
    // used sub-allocation chunks (fragmentation) versus whole empty chunks.
    DxvkMemoryAllocator& allocator = m_device->getCommon()->memoryManager();
    const VkPhysicalDeviceMemoryProperties& props = allocator.getMemoryProperties();
    for (uint32_t heapId = 0; heapId < props.memoryHeapCount; heapId++) {
      if (!(props.memoryHeaps[heapId].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
        continue;
      }
      const DxvkMemoryStats& stats = allocator.getMemoryHeaps()[heapId].stats;
      const DxvkMemoryAllocator::ChunkUsage chunks = allocator.getChunkUsage(heapId);
      const VkDeviceSize allocated = stats.totalAllocated();
      const VkDeviceSize dedicated = allocated > chunks.chunkBytes ? allocated - chunks.chunkBytes : 0;
      Logger::info(str::format(
        "[AccelMem] heap ", heapId, ": allocated ", mib(allocated), ", used ", mib(stats.totalUsed()),
        " | chunks ", chunks.chunkCount, " (", chunks.emptyChunkCount, " empty) ", mib(chunks.chunkBytes),
        ", free inside chunks ", mib(chunks.freeBytes), ", largest free slice ", mib(chunks.largestFreeSlice),
        " | dedicated ", mib(dedicated),
        " | used by category: app buffers ", mib(stats.usedByCategory(DxvkMemoryStats::Category::AppBuffer)),
        ", app textures ", mib(stats.usedByCategory(DxvkMemoryStats::Category::AppTexture)),
        ", replacement geometry ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXReplacementGeometry)),
        ", accel ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXAccelerationStructure)),
        ", scene geometry ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXSceneGeometry)),
        ", OMM ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXOpacityMicromap)),
        ", RTX buffers ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXBuffer)),
        ", material textures ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXMaterialTexture)),
        ", render targets ", mib(stats.usedByCategory(DxvkMemoryStats::Category::RTXRenderTarget)),
        " | chunk pools (chunks, draining, MiB, free MiB):", poolSummary(chunks)));
    }
  }

}  // namespace dxvk
