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
#include "rtx_fork_sccr_shell.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <vector>

#include "rtx_context.h"
#include "rtx_fork_pom_cone_map.h"
#include "rtx_fork_sccr_tess.h"
#include "rtx_instance_manager.h"
#include "rtx_materials.h"
#include "rtx_options.h"
#include "rtx_scene_manager.h"
#include "rtx_shader_manager.h"
#include "rtx_types.h"
#include "../dxvk_buffer.h"
#include "../dxvk_device.h"
#include "../dxvk_scoped_annotation.h"
#include "../dxvk_util.h"
#include "../../util/log/log.h"
#include "../../util/util_once.h"
#include "../../util/util_string.h"

#include "rtx/pass/sccr_shell/sccr_shell_args.h"

#include <rtx_shaders/sccr_shell_adjacency.h>

namespace dxvk {
  namespace {
    class SccrShellAdjacencyShader : public ManagedShader {
      SHADER_SOURCE(SccrShellAdjacencyShader, VK_SHADER_STAGE_COMPUTE_BIT, sccr_shell_adjacency)

      PUSH_CONSTANTS(SccrShellBuildArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(SCCR_SHELL_BINDING_POSITION_RO)
        STRUCTURED_BUFFER(SCCR_SHELL_BINDING_INDEX_RO)
        RW_STRUCTURED_BUFFER(SCCR_SHELL_BINDING_VERTEX_TABLE)
        RW_STRUCTURED_BUFFER(SCCR_SHELL_BINDING_EDGE_TABLE)
        RW_STRUCTURED_BUFFER(SCCR_SHELL_BINDING_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SccrShellAdjacencyShader);

    // Scratch slices are bound as storage buffers: keep their offsets aligned for any device.
    constexpr VkDeviceSize kScratchAlignment = 256;
    // How often the cache is searched for buffers nothing was bound to for kCacheKeepTime.
    constexpr std::chrono::milliseconds kEvictionInterval { 1000 };
    // A cached buffer no instance was bound to for this long is dropped (a mesh out of view for a while, then back
    // within it, needs no rebuild), unless the memory cap needs it sooner.
    constexpr std::chrono::seconds kCacheKeepTime { 60 };
    // A queued mesh whose geometry has not been in the instance table for this many frames is dropped (queued again if
    // it comes back).
    constexpr uint64_t kPendingKeepFrames = 300;
    // A mesh that did not fit under the memory cap is tried again after this long.
    constexpr std::chrono::milliseconds kOverBudgetRetry { 1000 };
    // A changed status is logged at most this often (and once after a run of builds).
    constexpr std::chrono::seconds kReportInterval { 10 };

    // The march budgets of each SccrShellManager::Quality level, in its order. High is the code default.
    struct QualityBudgets {
      uint32_t shellSteps;
      uint32_t shellPrisms;
      uint32_t shellShadowPrisms;
      uint32_t coneSteps;
      uint32_t binarySteps;
      uint32_t coneShadowSteps;
      uint32_t indirectShadowSteps;
      // The hand-off distance in units of rtx.displacement.referenceWorldUnitsPerUv, the world size the relief
      // depth is given in, so it scales with the scene's units (2000 world units at the default 256).
      float handOffReferenceUvs;
      float stepLodScale;
    };
    constexpr QualityBudgets kQualityBudgets[] = {
      { 128, 32, 4, 16, 5, 8, 2, 7.8125f, 1.0f },
      { 96, 24, 3, 12, 4, 6, 1, 5.859375f, 0.75f },
      { 64, 16, 2, 8, 3, 4, 1, 3.90625f, 0.5f },
    };

    float handOffDistance(const QualityBudgets& budgets) {
      return budgets.handOffReferenceUvs * RtxOptions::Displacement::referenceWorldUnitsPerUv();
    }

    // The smallest power of two >= value, at least 256 and at most 2^31.
    uint32_t tableSize(uint64_t value) {
      uint64_t size = 256;
      while (size < value && size < (uint64_t(1) << 31)) {
        size <<= 1;
      }
      return static_cast<uint32_t>(size);
    }
  }

  // One built shell buffer and the bindless slot it holds from its build until the cache drops it.
  struct SccrShellCacheEntry {
    Rc<DxvkBuffer> buffer;
    size_t bytes = 0;
    uint32_t slot = kSurfaceInvalidBufferIndex;
    // The last frame an instance was bound to it (or it was built for one), and when: eviction keeps buffers bound this
    // frame, and drops the least recently bound first.
    uint64_t lastBoundFrame = 0;
    std::chrono::steady_clock::time_point lastUsed {};
    // The scanInstances pass that last counted it (Stats::boundMeshes counts each entry once per pass).
    uint64_t lastCountedScan = 0;
  };

  SccrShellManager::SccrShellManager(DxvkDevice* device)
    : CommonDeviceObject(device) {
    m_scratch = std::make_unique<RtxStagingDataAlloc>(
      device,
      "RtxStagingDataAlloc: SCCR shell scratch",
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      (VkBufferUsageFlags) (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT),
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
  }

  SccrShellManager::~SccrShellManager() = default;

  void SccrShellManager::onDestroy() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    // The scene manager (and its buffer table) is torn down with the device: drop the entries without releasing slots.
    m_pending.clear();
    m_pendingOrder.clear();
    m_unsupported.clear();
    m_cache.clear();
    m_bytes = 0;
    if (++m_generation == 0) {
      m_generation = 1;
    }
    m_frameActive = false;
    m_wasActive = false;
    m_frameStats = Stats();
    m_scratch = nullptr;
  }

  bool SccrShellManager::active() {
    return enable() && RtxOptions::Displacement::mode() == DisplacementMode::SCCR;
  }

  bool SccrShellManager::isSolidOpaque(const RtInstance& instance) {
    const uint32_t flags = instance.getVkInstance().flags;
    // Opaque in the opaque TLAS (the "all other fully opaques" branch of the instance manager's flag update): no opacity
    // micromap, alpha test, blending or clip plane applies, so replacing its triangles with real relief changes nothing
    // else. Not a view or player model.
    if (!instance.surface.alphaState.isFullyOpaque || instance.surface.isClipPlaneEnabled) {
      return false;
    }
    if (instance.getGeometryFlags() != VK_GEOMETRY_OPAQUE_BIT_KHR ||
        (flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) != 0 || instance.usesUnorderedApproximations()) {
      return false;
    }
    return !instance.isViewModel() && !instance.isPlayerModelInstance();
  }

  size_t SccrShellManager::DataKeyHash::operator()(const DataKey& key) const {
    uint64_t h = key.positionHash;
    h ^= key.indexHash + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= ((uint64_t(key.vertexCount) << 32) | key.primitiveCount) + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return size_t(h);
  }

  SccrShellManager::DataKey SccrShellManager::keyOf(const RaytraceGeometry& geometry) {
    DataKey key;
    key.positionHash = geometry.hashes[HashComponents::VertexPosition];
    key.indexHash = geometry.hashes[HashComponents::Indices];
    key.vertexCount = geometry.vertexCount;
    key.primitiveCount = geometry.calculatePrimitiveCount();
    return key;
  }

  SccrShellManager::Eligibility SccrShellManager::classifyMaterial(const SceneManager& sceneManager,
                                                                   const RtSurfaceMaterial* material) {
    if (material == nullptr) {
      return Eligibility::NoMaterial;
    }
    if (material->getType() != RtSurfaceMaterialType::Opaque) {
      return Eligibility::NotOpaque;
    }
    const RtOpaqueSurfaceMaterial& opaque = material->getOpaqueSurfaceMaterial();
    if (!opaque.hasValidDisplacement()) {
      return Eligibility::NotDisplaced;
    }
    // No shell data for blended terrain: large, gently varying triangles
    // (little flat-plane error) and the most expensive samples. The shader then finds no shell buffer and keeps the
    // terrain march. Blended means the march blends layer heights: the terrain-layers entry has a height material
    // (pomHeightSourceCreate). A terrain-layers entry alone is not terrain: the FNV wrapper gives every API static with
    // a normal map a one-layer entry (weight 1, no layer heights) for the RGB tangent-space decode and per-layer gloss,
    // and its own height map is marched like any mesh's, so such an entry stays eligible.
    const uint32_t terrainLayersIndex = opaque.getTerrainLayersMaterialIndex();
    if (terrainLayersIndex != SURFACE_INDEX_INVALID) {
      const RtSurfaceMaterial* layers = sceneManager.getSurfaceMaterialExtensionOrNull(terrainLayersIndex);
      // An entry that cannot be read is treated as terrain (no data rather than a march on the wrong heights).
      if (layers == nullptr || layers->getType() != RtSurfaceMaterialType::TerrainLayers ||
          layers->getTerrainLayersMaterial().getHeightMaterialIndex() != SURFACE_INDEX_INVALID) {
        return Eligibility::Terrain;
      }
    }
    return Eligibility::Eligible;
  }

  SccrShellManager::Eligibility SccrShellManager::classifyGeometry(const BlasEntry& blas) {
    // Skinned meshes are rebuilt every frame; a geometry whose vertices changed twice is animated.
    if (blas.input.getSkinningState().numBones > 0 && blas.input.getGeometryData().numBonesPerVertex > 0) {
      return Eligibility::Skinned;
    }
    if (blas.modifiedGeometryData.shellMemo.vertexUpdates >= 2) {
      return Eligibility::Animated;
    }
    return Eligibility::Eligible;
  }

  SccrShellManager::Eligibility SccrShellManager::classify(const SceneManager& sceneManager, const RtInstance& instance) {
    const BlasEntry* blas = instance.getBlas();
    if (blas == nullptr) {
      return Eligibility::NoGeometry;
    }
    const Eligibility material =
      classifyMaterial(sceneManager, sceneManager.getSurfaceMaterialOrNull(instance.surface.surfaceMaterialIndex));
    if (material != Eligibility::Eligible) {
      return material;
    }
    return classifyGeometry(*blas);
  }

  void SccrShellManager::onGeometryVerticesChanged(RaytraceGeometry& geometry) {
    SccrShellGeometryMemo& memo = geometry.shellMemo;
    // Counted only once the scan has seen the geometry with an eligible material.
    if (memo.generation != 0 && memo.vertexUpdates < 255) {
      ++memo.vertexUpdates;
    }
    // The new hashes no longer match the memo's key either; this only makes the miss explicit.
    memo.entry = nullptr;
    memo.slot = kSurfaceInvalidBufferIndex;
  }

  uint32_t SccrShellManager::slotFor(const RtInstance& instance) const {
    if (!m_frameActive) {
      return kSurfaceInvalidBufferIndex;
    }
    if (classify(m_device->getCommon()->getSceneManager(), instance) != Eligibility::Eligible) {
      return kSurfaceInvalidBufferIndex;
    }
    const RaytraceGeometry& geometry = instance.getBlas()->modifiedGeometryData;
    const SccrShellGeometryMemo& memo = geometry.shellMemo;
    const DataKey key = keyOf(geometry);
    // The scan memoised nearly every geometry this frame: a few compares. Otherwise (an instance outside the instance
    // table, a key changed since) one lookup.
    if (memo.entry != nullptr && memo.generation == m_generation && memo.positionHash == key.positionHash &&
        memo.indexHash == key.indexHash && memo.vertexCount == key.vertexCount && memo.primitiveCount == key.primitiveCount) {
      return memo.slot;
    }
    const auto found = m_cache.find(key);
    return found != m_cache.end() ? found->second->slot : kSurfaceInvalidBufferIndex;
  }

  void SccrShellManager::patchSurfaceRecord(unsigned char* record, const RtInstance& instance) const {
    // data15.w is the record's last dword (RtSurface::writeGPUData); the high 16 bits are reserved (0).
    const uint32_t slot = slotFor(instance);
    const uint32_t value = slot < kSurfaceInvalidBufferIndex ? slot : kSurfaceInvalidBufferIndex;
    std::memcpy(record + kSurfaceGPUSize - sizeof(uint32_t), &value, sizeof(value));
    // SCCR real geometry: a bound instance's record points at its tessellated mesh instead (and has no shell slot).
    m_device->getCommon()->metaSccrTess().patchSurfaceRecord(record, instance);
  }

  bool SccrShellManager::isRealGeometryCandidate(const SceneManager& sceneManager, const RtInstance& instance) {
    if (classify(sceneManager, instance) != Eligibility::Eligible) {
      return false;
    }
    if (!isSolidOpaque(instance)) {
      return false;
    }
    const RtSurface& surface = instance.surface;
    // The tessellator reads the buffer's texture coordinates as they are and keeps the record's texture transform, so
    // only an identity transform keeps heights and shading on the same coordinates.
    return surface.instancesToObject == nullptr && instance.getBlas()->buildGeometries.size() == 1 &&
           surface.texgenMode == TexGenMode::None && surface.spriteSheetFPS == 0 && surface.textureTransform == Matrix4();
  }

  Rc<DxvkBuffer> SccrShellManager::shellBufferFor(const RtInstance& instance) const {
    if (!m_frameActive || classify(m_device->getCommon()->getSceneManager(), instance) != Eligibility::Eligible) {
      return nullptr;
    }
    const RaytraceGeometry& geometry = instance.getBlas()->modifiedGeometryData;
    const SccrShellGeometryMemo& memo = geometry.shellMemo;
    const DataKey key = keyOf(geometry);
    if (memo.entry != nullptr && memo.generation == m_generation && memo.positionHash == key.positionHash &&
        memo.indexHash == key.indexHash && memo.vertexCount == key.vertexCount && memo.primitiveCount == key.primitiveCount) {
      return memo.entry->buffer;
    }
    const auto found = m_cache.find(key);
    return found != m_cache.end() ? found->second->buffer : nullptr;
  }

  void SccrShellManager::scanInstances(SceneManager& sceneManager) {
    ++m_scanSerial;
    // The frame's counts, written over m_frameStats' walk fields at the end (the rest is prepareFrame's).
    Stats counts;
    // SCCR real geometry: every instance's flag is rewritten here, before mergeInstancesIntoBlas keys its bucket.
    // Candidates get their own TLAS entry (RtInstance::isRealGeometry, read by the BLAS routing): once their mesh is
    // built the bumps are real, and until then plain shell tracing shows them.
    const bool flagRealGeometry = SccrTessManager::active();
    bool flagsChanged = false;
    for (RtInstance* instance : sceneManager.getInstanceTable()) {
      ++counts.instances;
      const Eligibility eligibility =
        instance != nullptr ? classify(sceneManager, *instance) : Eligibility::NoGeometry;
      if (instance != nullptr) {
        const bool realGeometry = flagRealGeometry && eligibility == Eligibility::Eligible &&
          isRealGeometryCandidate(sceneManager, *instance) &&
          m_device->getCommon()->metaSccrTess().acceptsMesh(*instance->getBlas());
        if (instance->isRealGeometry != realGeometry) {
          instance->isRealGeometry = realGeometry;
          flagsChanged = true;
        }
      }
      switch (eligibility) {
      case Eligibility::Eligible: break;
      case Eligibility::NoGeometry: ++counts.noGeometry; break;
      case Eligibility::NoMaterial: ++counts.noMaterial; break;
      case Eligibility::NotOpaque: ++counts.notOpaque; break;
      case Eligibility::NotDisplaced: ++counts.notDisplaced; break;
      case Eligibility::Terrain: ++counts.terrain; break;
      case Eligibility::Skinned: ++counts.skinnedSurfaces; break;
      case Eligibility::Animated: ++counts.animatedSurfaces; break;
      }
      if (eligibility != Eligibility::Eligible) {
        continue;
      }
      ++counts.eligibleSurfaces;
      BlasEntry& blas = *instance->getBlas();

      RaytraceGeometry& geometry = blas.modifiedGeometryData;
      SccrShellGeometryMemo& memo = geometry.shellMemo;
      const DataKey key = keyOf(geometry);
      const bool memoHit = memo.entry != nullptr && memo.generation == m_generation &&
        memo.positionHash == key.positionHash && memo.indexHash == key.indexHash &&
        memo.vertexCount == key.vertexCount && memo.primitiveCount == key.primitiveCount;
      if (!memoHit) {
        // A miss is looked up again every frame (a buffer may have been built since); a hit stays valid until an entry
        // is erased (m_generation) or the geometry's key changes.
        memo.positionHash = key.positionHash;
        memo.indexHash = key.indexHash;
        memo.vertexCount = key.vertexCount;
        memo.primitiveCount = key.primitiveCount;
        memo.generation = m_generation;
        const auto found = m_cache.find(key);
        memo.entry = found != m_cache.end() ? found->second.get() : nullptr;
        memo.slot = memo.entry != nullptr ? memo.entry->slot : kSurfaceInvalidBufferIndex;
      }

      if (memo.entry != nullptr) {
        ++counts.boundSurfaces;
        memo.entry->lastBoundFrame = m_frame;
        memo.entry->lastUsed = m_now;
        if (memo.entry->lastCountedScan != m_scanSerial) {
          memo.entry->lastCountedScan = m_scanSerial;
          ++counts.boundMeshes;
        }
        continue;
      }

      // No data: queue the mesh once per key, through the BlasEntry seen this frame (the one the build will read).
      if (m_unsupported.find(key) != m_unsupported.end()) {
        ++counts.unsupportedSurfaces;
        continue;
      }
      const auto [pending, inserted] = m_pending.try_emplace(key);
      if (inserted) {
        m_pendingOrder.push_back(key);
      }
      pending->second.blas = &blas;
      pending->second.lastSeenFrame = m_frame;
    }
    m_frameStats.eligibleSurfaces = counts.eligibleSurfaces;
    m_frameStats.boundSurfaces = counts.boundSurfaces;
    m_frameStats.boundMeshes = counts.boundMeshes;
    m_frameStats.dynamicSurfaces = counts.skinnedSurfaces + counts.animatedSurfaces;
    m_frameStats.noGeometry = counts.noGeometry;
    m_frameStats.noMaterial = counts.noMaterial;
    m_frameStats.notOpaque = counts.notOpaque;
    m_frameStats.notDisplaced = counts.notDisplaced;
    m_frameStats.terrain = counts.terrain;
    m_frameStats.skinnedSurfaces = counts.skinnedSurfaces;
    m_frameStats.animatedSurfaces = counts.animatedSurfaces;
    m_frameStats.unsupportedSurfaces = counts.unsupportedSurfaces;
    m_frameStats.instances = counts.instances;
    m_flagsSet = m_flagsSet || flagRealGeometry;
    // The flag is part of the merged-BLAS bucket key, but nothing else marks the scene changed when only it moves (an
    // option toggled, an instance turning dynamic): without this the unchanged-scene path would keep the old buckets
    // and AccelManager's dirty scan would never see the new key.
    if (flagsChanged) {
      sceneManager.notifyAccelSceneChanged();
    }
  }

  SccrShellManager::Cache::iterator SccrShellManager::eraseEntry(SceneManager& sceneManager, Cache::iterator it) {
    SccrShellCacheEntry& entry = *it->second;
    if (entry.slot != kSurfaceInvalidBufferIndex) {
      sceneManager.releaseBufferSlot(entry.slot);
    }
    m_bytes -= std::min(m_bytes, entry.bytes);
    // Memos pointing at it are stale from now on; this frame's surface records are written after this (by
    // patchSurfaceRecord, which only follows a memo of the current generation), so none of them gets the slot.
    if (++m_generation == 0) {
      m_generation = 1;
    }
    return m_cache.erase(it);
  }

  void SccrShellManager::evictCache(SceneManager& sceneManager, size_t bytesNeeded) {
    std::vector<Cache::iterator> candidates;
    for (auto it = m_cache.begin(); it != m_cache.end();) {
      const SccrShellCacheEntry& entry = *it->second;
      const bool boundThisFrame = entry.lastBoundFrame == m_frame;
      if (!boundThisFrame && m_now - entry.lastUsed >= kCacheKeepTime) {
        it = eraseEntry(sceneManager, it);
        ++m_sessionEvicted;
        continue;
      }
      if (bytesNeeded > 0 && !boundThisFrame) {
        candidates.push_back(it);
      }
      ++it;
    }
    if (bytesNeeded == 0) {
      return;
    }
    const size_t capBytes = size_t(maxMB()) << 20;
    // Erasing one entry leaves the iterators to the others valid.
    std::sort(candidates.begin(), candidates.end(), [](const Cache::iterator& a, const Cache::iterator& b) {
      return a->second->lastUsed < b->second->lastUsed;
    });
    for (const Cache::iterator& it : candidates) {
      if (m_bytes + bytesNeeded <= capBytes) {
        break;
      }
      eraseEntry(sceneManager, it);
      ++m_sessionEvicted;
    }
  }

  void SccrShellManager::releaseAll(SceneManager& sceneManager) {
    for (auto& [key, entry] : m_cache) {
      if (entry->slot != kSurfaceInvalidBufferIndex) {
        sceneManager.releaseBufferSlot(entry->slot);
      }
    }
    m_cache.clear();
    m_bytes = 0;
    m_pending.clear();
    m_pendingOrder.clear();
    m_unsupported.clear();
    if (++m_generation == 0) {
      m_generation = 1;
    }
  }

  void SccrShellManager::onSceneClear(SceneManager& sceneManager) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    releaseAll(sceneManager);
    m_frameActive = false;
  }

  void SccrShellManager::prepareFrame(RtxContext& ctx) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    m_frameActive = false;
    // Destroyed (onDestroy): no scratch to build with.
    if (m_scratch == nullptr) {
      return;
    }
    SceneManager& sceneManager = m_device->getCommon()->getSceneManager();
    if (!active() && m_flagsSet) {
      // Shell tracing off: the walk below does not run, so drop the real geometry flags here, once.
      for (RtInstance* instance : sceneManager.getInstanceTable()) {
        if (instance != nullptr) {
          instance->isRealGeometry = false;
        }
      }
      m_flagsSet = false;
      sceneManager.notifyAccelSceneChanged();
    }
    if (!active()) {
      if (m_wasActive) {
        const size_t released = m_cache.size();
        releaseAll(sceneManager);
        // Always logged, so the log shows when the data was wanted at all.
        Logger::info(str::format("[SCCR shell] Shell tracing off (", enable() ? "not SCCR mode" : "disabled", "): ",
                                 released, " cached shell buffers released."));
      }
      m_wasActive = false;
      m_frameStats = Stats();
      return;
    }

    // A new weld tolerance changes the adjacency: drop everything; this frame's scan queues it again.
    const float currentWeldEpsilon = std::max(weldEpsilon(), 0.0f);
    if (m_wasActive && currentWeldEpsilon != m_builtWeldEpsilon) {
      releaseAll(sceneManager);
    }
    m_builtWeldEpsilon = currentWeldEpsilon;
    // The first frame of a run of shell tracing always reports (see below).
    const bool justActivated = !m_wasActive;
    m_wasActive = true;
    m_frameActive = true;
    ++m_frame;
    m_now = std::chrono::steady_clock::now();

    if (m_now - m_lastEviction >= kEvictionInterval) {
      evictCache(sceneManager, 0);
      m_lastEviction = m_now;
    }

    scanInstances(sceneManager);
    const uint32_t generationBeforeBuilds = m_generation;
    const bool builtAny = buildPending(ctx, sceneManager);
    if (builtAny) {
      // The build dispatches write through storage bindings; the surfaces read the buffers bindlessly in this frame's
      // ray tracing, which DXVK cannot track.
      ctx.emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_SHADER_READ_BIT);
    }
    if (builtAny || m_generation != generationBeforeBuilds) {
      // Memoise (and count) the meshes built just now, and forget buffers a build evicted. patchSurfaceRecord would get
      // both right anyway (it looks up whatever is not memoised in the current generation); this keeps it to compares.
      scanInstances(sceneManager);
    }

    uint32_t overBudget = 0;
    for (const auto& [key, pending] : m_pending) {
      if (pending.overBudget) {
        ++overBudget;
      }
    }
    m_frameStats.pendingBuilds = static_cast<uint32_t>(m_pending.size());
    m_frameStats.overBudgetBuilds = overBudget;
    m_frameStats.cachedBuffers = static_cast<uint32_t>(m_cache.size());
    m_frameStats.unsupportedMeshes = static_cast<uint32_t>(m_unsupported.size());
    m_frameStats.bytes = m_bytes;

    // A line once a run of builds is over, and otherwise when something changed, at most every kReportInterval.
    const Stats& s = m_frameStats;
    uint64_t signature = 0xCBF29CE484222325ull;
    for (const uint64_t value : { uint64_t(s.boundSurfaces), uint64_t(s.eligibleSurfaces), uint64_t(s.pendingBuilds),
                                  uint64_t(s.cachedBuffers), uint64_t(s.dynamicSurfaces), uint64_t(m_sessionBuilds),
                                  uint64_t(m_sessionEvicted), uint64_t(m_sessionCancelled), uint64_t(s.noGeometry),
                                  uint64_t(s.noMaterial), uint64_t(s.notOpaque), uint64_t(s.notDisplaced),
                                  uint64_t(s.terrain), uint64_t(s.unsupportedSurfaces) }) {
      signature = (signature ^ value) * 0x100000001B3ull;
    }
    const bool buildRunOver = m_builtSinceReport > 0 && !builtAny;
    if (justActivated || buildRunOver ||
        (signature != m_reportedSignature && m_now - m_lastReport >= kReportInterval)) {
      logReport();
      m_reportedSignature = signature;
    }
  }

  bool SccrShellManager::buildPending(RtxContext& ctx, SceneManager& sceneManager) {
    if (m_pendingOrder.empty()) {
      return false;
    }

    ScopedGpuProfileZone(&ctx, "SccrShell");

    const uint64_t triangleBudget = std::max<uint64_t>(buildTrianglesPerFrame(), 1);
    uint64_t trianglesBuilt = 0;
    bool builtAny = false;
    // Each queued key at most once this frame.
    size_t remaining = m_pendingOrder.size();
    while (remaining-- > 0) {
      const DataKey key = m_pendingOrder.front();
      m_pendingOrder.pop_front();
      const auto found = m_pending.find(key);
      if (found == m_pending.end()) {
        continue;
      }
      PendingBuild& pending = found->second;
      if (pending.lastSeenFrame != m_frame) {
        // No instance of it this frame: its BlasEntry may be gone, so it is not touched. Wait a while, then forget it.
        if (m_frame - pending.lastSeenFrame > kPendingKeepFrames) {
          m_pending.erase(found);
          ++m_sessionCancelled;
        } else {
          m_pendingOrder.push_back(key);
        }
        continue;
      }
      if (pending.overBudget && m_now < pending.retryAfter) {
        m_pendingOrder.push_back(key);
        continue;
      }
      const uint64_t triangles = key.primitiveCount;
      if (builtAny && trianglesBuilt + triangles > triangleBudget) {
        m_pendingOrder.push_front(key);
        break;
      }
      switch (build(ctx, sceneManager, key, *pending.blas)) {
      case BuildResult::Built:
        m_pending.erase(found);
        builtAny = true;
        trianglesBuilt += triangles;
        ++m_builtSinceReport;
        break;
      case BuildResult::Unsupported:
        m_unsupported.insert(key);
        m_pending.erase(found);
        break;
      case BuildResult::OverBudget:
        pending.overBudget = true;
        pending.retryAfter = m_now + kOverBudgetRetry;
        m_pendingOrder.push_back(key);
        break;
      case BuildResult::Retry:
        m_pendingOrder.push_back(key);
        break;
      }
    }
    return builtAny;
  }

  void SccrShellManager::logReport() {
    const Stats& s = m_frameStats;
    Logger::info(str::format("[SCCR shell] ", s.boundSurfaces, " of ", s.eligibleSurfaces,
                             " displaced static surfaces have shell data this frame (", s.boundMeshes, " meshes); ",
                             s.pendingBuilds, " meshes waiting for a build (", s.overBudgetBuilds, " over the ", maxMB(),
                             " MB cap), ", s.dynamicSurfaces, " surfaces skinned or animated (plain SCCR), ",
                             s.unsupportedMeshes, " meshes unsupported; ", s.cachedBuffers, " buffers cached, ",
                             (s.bytes + (size_t(1) << 19)) >> 20, " MB. Built since the last line: ", m_builtSinceReport,
                             ". Session: ", m_sessionBuilds, " built, ", m_sessionEvicted, " evicted, ", m_sessionCancelled,
                             " queued meshes dropped (geometry gone before its build). Instances with a displaced material: ",
                             m_device->getCommon()->getSceneManager().getActivePOMCount(),
                             " (0 turns displacement off in the shaders). Of ", s.instances,
                             " instances walked, without shell data because: no geometry ", s.noGeometry,
                             ", no material ", s.noMaterial, ", not opaque ", s.notOpaque, ", not displaced ",
                             s.notDisplaced, ", blended terrain ", s.terrain, ", skinned ", s.skinnedSurfaces,
                             ", vertices animated ", s.animatedSurfaces, ", mesh unsupported ", s.unsupportedSurfaces,
                             "."));
    m_builtSinceReport = 0;
    m_lastReport = m_now;
  }

  SccrShellManager::BuildResult SccrShellManager::build(RtxContext& ctx, SceneManager& sceneManager, const DataKey& key,
                                                        BlasEntry& blas) {
    RaytraceGeometry& geometry = blas.modifiedGeometryData;
    // The scan queued it from this geometry this frame; if the key moved since, let the next scan sort it out.
    if (!(keyOf(geometry) == key)) {
      return BuildResult::Retry;
    }
    const uint32_t numTriangles = geometry.calculatePrimitiveCount();
    const uint32_t numVertices = geometry.vertexCount;
    const VkFormat positionFormat = geometry.positionBuffer.vertexFormat();
    const bool indexed = geometry.usesIndices();
    const bool supportedIndices = !indexed ||
      geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16 || geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT32;
    if (!geometry.positionBuffer.defined() || numTriangles == 0 || numVertices == 0 ||
        numTriangles >= SCCR_SHELL_MAX_TRIANGLES || !supportedIndices ||
        (positionFormat != VK_FORMAT_R32G32B32_SFLOAT && positionFormat != VK_FORMAT_R32G32B32A32_SFLOAT) ||
        (geometry.positionBuffer.stride() % 4) != 0) {
      return BuildResult::Unsupported;
    }

    const size_t dwords = size_t(SCCR_SHELL_HEADER_DWORDS) + 3 * size_t(numTriangles) + 3 * size_t(numVertices);
    const size_t bytes = dwords * sizeof(uint32_t);
    const size_t capBytes = size_t(maxMB()) << 20;
    if (m_bytes + bytes > capBytes) {
      // Cached buffers no instance is bound to this frame go first, least recently bound first.
      evictCache(sceneManager, bytes);
    }
    if (m_bytes + bytes > capBytes) {
      return BuildResult::OverBudget;
    }

    DxvkBufferCreateInfo info = {};
    info.size = align(bytes, 16);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    info.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    const Rc<DxvkBuffer> shellBuffer = m_device->createBuffer(
      info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SCCR shell data");
    if (shellBuffer == nullptr) {
      return BuildResult::OverBudget;
    }

    // The buffer's own bindless slot, held until the cache drops it. Surfaces read 16 bits of it (data15.w).
    const RaytraceBuffer shellView(DxvkBufferSlice(shellBuffer), 0u, uint32_t(sizeof(uint32_t)), VK_FORMAT_R32_UINT);
    const uint32_t slot = sceneManager.acquireBufferSlot(shellView);
    if (slot >= kSurfaceInvalidBufferIndex) {
      sceneManager.releaseBufferSlot(slot);
      ONCE(Logger::warn("[SCCR shell] The bindless buffer table is full: no slot for shell data."));
      return BuildResult::OverBudget;
    }

    // Load factor at most 0.5 in both tables (about 1.5 edges per triangle).
    const uint32_t vertexTableSize = tableSize(2 * uint64_t(numVertices));
    const uint32_t edgeTableSize = tableSize(3 * uint64_t(numTriangles));
    const VkDeviceSize vertexTableBytes = VkDeviceSize(vertexTableSize) * SCCR_SHELL_VERTEX_SLOT_DWORDS * sizeof(uint32_t);
    const VkDeviceSize edgeTableBytes = VkDeviceSize(edgeTableSize) * SCCR_SHELL_EDGE_SLOT_DWORDS * sizeof(uint32_t);
    const DxvkBufferSlice vertexTable = m_scratch->alloc(kScratchAlignment, vertexTableBytes);
    const DxvkBufferSlice edgeTable = m_scratch->alloc(kScratchAlignment, edgeTableBytes);

    // Weld grid (rtx.displacement.shell.weldEpsilon): a fraction of the mesh's object-space bounding diagonal. Without a
    // valid box (or with the option at 0) positions weld by their exact bits.
    float weldCell = 0.0f;
    const AxisAlignedBoundingBox& boundingBox = blas.input.getGeometryData().boundingBox;
    if (m_builtWeldEpsilon > 0.0f && boundingBox.isValid()) {
      const Vector3 extent = boundingBox.maxPos - boundingBox.minPos;
      const float diagonal = std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
      if (std::isfinite(diagonal) && diagonal > 0.0f) {
        weldCell = m_builtWeldEpsilon * diagonal;
      }
    }

    // Header: counts, version and weld mode now; the passes OR their flags into it.
    float depthLimitScale = 1.0f;
    uint32_t header[SCCR_SHELL_HEADER_DWORDS];
    header[SCCR_SHELL_HEADER_TRIANGLES] = numTriangles;
    header[SCCR_SHELL_HEADER_VERTICES] = numVertices;
    header[SCCR_SHELL_HEADER_FLAGS] = SCCR_SHELL_VERSION |
      ((weldCell > 0.0f ? SCCR_SHELL_WELD_GRID : SCCR_SHELL_WELD_EXACT) << SCCR_SHELL_FLAG_WELD_MODE_SHIFT);
    std::memcpy(&header[SCCR_SHELL_HEADER_DEPTH_LIMIT_SCALE], &depthLimitScale, sizeof(depthLimitScale));
    ctx.writeToBuffer(shellBuffer, 0, sizeof(header), header);
    ctx.clearBuffer(vertexTable.buffer(), vertexTable.offset(), vertexTableBytes, 0);
    ctx.clearBuffer(edgeTable.buffer(), edgeTable.offset(), edgeTableBytes, 0);

    SccrShellBuildArgs args {};
    // Offsets from the start of the bound buffers, as the surface shaders address them through their slices.
    args.positionOffset = static_cast<uint32_t>(geometry.positionBuffer.offset() + geometry.positionBuffer.offsetFromSlice());
    args.positionStride = geometry.positionBuffer.stride();
    args.indexOffset = indexed ? static_cast<uint32_t>(geometry.indexBuffer.offset() + geometry.indexBuffer.offsetFromSlice()) : 0u;
    args.indexStride = !indexed ? 0u : geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16 ? 2u : 4u;
    args.numTriangles = numTriangles;
    args.numVertices = numVertices;
    args.vertexTableSize = vertexTableSize;
    args.edgeTableSize = edgeTableSize;
    args.weldCell = weldCell;

    // The geometry's buffers are read now, by dispatches recorded into this frame's command list, which keeps them alive
    // until the GPU is done even if the scene manager collects the BlasEntry right after this frame.
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SccrShellAdjacencyShader::getShader());
    ctx.bindResourceBuffer(SCCR_SHELL_BINDING_POSITION_RO, DxvkBufferSlice(geometry.positionBuffer.buffer()));
    // Without indices nothing reads the index binding, but it must be bound: the vertex buffer stands in.
    ctx.bindResourceBuffer(SCCR_SHELL_BINDING_INDEX_RO,
                           DxvkBufferSlice(indexed ? geometry.indexBuffer.buffer() : geometry.positionBuffer.buffer()));
    ctx.bindResourceBuffer(SCCR_SHELL_BINDING_VERTEX_TABLE, vertexTable);
    ctx.bindResourceBuffer(SCCR_SHELL_BINDING_EDGE_TABLE, edgeTable);
    ctx.bindResourceBuffer(SCCR_SHELL_BINDING_OUTPUT, DxvkBufferSlice(shellBuffer));

    const auto dispatchPhase = [&](uint32_t phase, uint32_t threads) {
      const uint32_t groups = (threads + SCCR_SHELL_GROUP_SIZE - 1) / SCCR_SHELL_GROUP_SIZE;
      const uint32_t groupsPerRow = std::min<uint32_t>(groups, SCCR_SHELL_MAX_GROUPS_PER_ROW);
      const uint32_t rows = (groups + groupsPerRow - 1) / groupsPerRow;
      args.phase = phase;
      args.threadCount = threads;
      args.rowThreads = groupsPerRow * SCCR_SHELL_GROUP_SIZE;
      ctx.pushConstants(0, sizeof(args), &args);
      ctx.dispatch(groupsPerRow, rows, 1);
    };
    // Each phase reads what the previous one wrote; the context orders the dispatches on the bound storage buffers. The
    // fold guard needs the welded directions (phase 1) and must be done before the vertex words (phase 4).
    dispatchPhase(SCCR_SHELL_PHASE_WELD, numTriangles);
    dispatchPhase(SCCR_SHELL_PHASE_FOLD, numTriangles);
    dispatchPhase(SCCR_SHELL_PHASE_EDGES, 3 * numTriangles);
    dispatchPhase(SCCR_SHELL_PHASE_RESOLVE, 3 * numTriangles);
    dispatchPhase(SCCR_SHELL_PHASE_VERTICES, numVertices);

    // Into the cache. A surface asked for it this frame (the scan queued it), so it counts as bound now: a later build
    // this frame never evicts it.
    const auto existing = m_cache.find(key);
    if (existing != m_cache.end()) {
      eraseEntry(sceneManager, existing);
    }
    std::unique_ptr<SccrShellCacheEntry> entry = std::make_unique<SccrShellCacheEntry>();
    entry->buffer = shellBuffer;
    entry->bytes = bytes;
    entry->slot = slot;
    entry->lastBoundFrame = m_frame;
    entry->lastUsed = m_now;
    m_cache.emplace(key, std::move(entry));
    m_bytes += bytes;
    ++m_sessionBuilds;
    return BuildResult::Built;
  }

  void SccrShellManager::setQualityLevel(Quality quality) {
    if (quality == Quality::Custom) {
      return;
    }
    const QualityBudgets& budgets = kQualityBudgets[static_cast<uint32_t>(quality)];
    maxStepsObject().setDeferred(budgets.shellSteps);
    maxPrismsObject().setDeferred(budgets.shellPrisms);
    maxShadowPrismsObject().setDeferred(budgets.shellShadowPrisms);
    maxDistanceObject().setDeferred(handOffDistance(budgets));
    RtxOptions::Displacement::coneStepsObject().setDeferred(budgets.coneSteps);
    RtxOptions::Displacement::binaryStepsObject().setDeferred(budgets.binarySteps);
    RtxOptions::Displacement::coneShadowStepsObject().setDeferred(budgets.coneShadowSteps);
    RtxOptions::Displacement::indirectShadowStepsObject().setDeferred(budgets.indirectShadowSteps);
    RtxOptions::Displacement::stepLodScaleObject().setDeferred(budgets.stepLodScale);
  }

  SccrShellManager::Quality SccrShellManager::qualityLevel() {
    for (uint32_t level = 0; level < std::size(kQualityBudgets); ++level) {
      const QualityBudgets& budgets = kQualityBudgets[level];
      if (maxSteps() == budgets.shellSteps && maxPrisms() == budgets.shellPrisms &&
          maxShadowPrisms() == budgets.shellShadowPrisms && maxDistance() == handOffDistance(budgets) &&
          RtxOptions::Displacement::coneSteps() == budgets.coneSteps &&
          RtxOptions::Displacement::binarySteps() == budgets.binarySteps &&
          RtxOptions::Displacement::coneShadowSteps() == budgets.coneShadowSteps &&
          RtxOptions::Displacement::indirectShadowSteps() == budgets.indirectShadowSteps &&
          RtxOptions::Displacement::stepLodScale() == budgets.stepLodScale) {
        return static_cast<Quality>(level);
      }
    }
    return Quality::Custom;
  }

  uint32_t SccrShellManager::marchFlags(bool conservativeCones) {
    uint32_t flags = 0;
    if (foldGuard()) {
      flags |= SCCR_SHELL_MARCH_FLAG_FOLD_GUARD;
    }
    if (shadows()) {
      flags |= SCCR_SHELL_MARCH_FLAG_SHADOWS;
    }
    // Every cone the fields hold is conservative (coarse cones, or generated without a ray walk): the outward shadow
    // march may cone-step on them. Not while any map holds relaxed cones, also right after a switch to
    // CoarseOnly, until the coarse pass has overwritten them.
    if (conservativeCones) {
      flags |= SCCR_SHELL_MARCH_FLAG_CONSERVATIVE_CONES;
    }
    flags |= std::min(maxShadowPrisms(), uint32_t(SCCR_SHELL_MARCH_SHADOW_PRISMS_MASK)) << SCCR_SHELL_MARCH_SHADOW_PRISMS_SHIFT;
    return flags;
  }

  SccrShellManager::Stats SccrShellManager::stats() const {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    return m_frameStats;
  }
}
