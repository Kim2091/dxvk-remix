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
#include "rtx_fork_sccr_tess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "rtx_context.h"
#include "rtx_fork_sccr_shell.h"
#include "rtx_instance_manager.h"
#include "rtx_materials.h"
#include "rtx_options.h"
#include "rtx_scene_manager.h"
#include "rtx_shader_manager.h"
#include "rtx_texture.h"
#include "rtx_texture_manager.h"
#include "rtx_types.h"
#include "../dxvk_buffer.h"
#include "../dxvk_device.h"
#include "../dxvk_scoped_annotation.h"
#include "../dxvk_util.h"
#include "../../util/log/log.h"
#include "../../util/util_once.h"
#include "../../util/util_string.h"

#include "rtx/pass/sccr_shell/sccr_shell_args.h"
#include "rtx/pass/sccr_tess/sccr_tess_args.h"
#include "rtx/utility/shared_constants.h"

#include <rtx_shaders/sccr_tess.h>

namespace dxvk {
  namespace {
    class SccrTessShader : public ManagedShader {
      SHADER_SOURCE(SccrTessShader, VK_SHADER_STAGE_COMPUTE_BIT, sccr_tess)

      PUSH_CONSTANTS(SccrTessArgs)

      BEGIN_PARAMETER()
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_POSITION_RO)
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_NORMAL_RO)
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_TEXCOORD_RO)
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_COLOR_RO)
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_INDEX_RO)
        STRUCTURED_BUFFER(SCCR_TESS_BINDING_SHELL_RO)
        SAMPLER2D(SCCR_TESS_BINDING_HEIGHT)
        RW_STRUCTURED_BUFFER(SCCR_TESS_BINDING_VERTEX_OUTPUT)
        RW_STRUCTURED_BUFFER(SCCR_TESS_BINDING_INDEX_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(SccrTessShader);

    constexpr std::chrono::milliseconds kEvictionInterval { 1000 };
    // A mesh no instance was bound to for this long is dropped, like the shell cache's kCacheKeepTime.
    constexpr std::chrono::seconds kCacheKeepTime { 60 };
    constexpr uint64_t kPendingKeepFrames = 300;
    constexpr std::chrono::milliseconds kOverBudgetRetry { 1000 };
    // A queued mesh waiting for its shell data or its full height texture is looked at again after this long.
    constexpr std::chrono::milliseconds kWaitRetry { 250 };
    constexpr std::chrono::seconds kReportInterval { 10 };
    // A dropped mesh's buffers and BLAS live this many frames longer, because last frame's TLAS (ReSTIR's previous TLAS)
    // and surface records may reference them, and frames in flight read them.
    constexpr uint64_t kGraveFrames = 6;
    constexpr uint32_t kVertexBytes = SCCR_TESS_VERTEX_DWORDS * sizeof(uint32_t);

    uint64_t mix(uint64_t h, uint64_t value) {
      return (h ^ value) * 0x100000001B3ull;
    }

    uint64_t floatBits(float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      return bits;
    }

    float vectorLength(const Vector3& v) {
      return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    Vector3 columnOf(const Matrix4& m, int column) {
      return Vector3(m[column].x, m[column].y, m[column].z);
    }

    bool isFloatPositionFormat(VkFormat format) {
      return format == VK_FORMAT_R32G32B32_SFLOAT || format == VK_FORMAT_R32G32B32A32_SFLOAT;
    }

    // Returns a 16- or 32-bit vertex index of a CPU-visible geometry (GeometryBufferData).
    uint32_t cpuIndex(const GeometryBufferData& data, bool indexed, bool indices16, uint32_t i) {
      if (!indexed) {
        return i;
      }
      if (indices16) {
        return data.indexData[i * data.indexStride];
      }
      uint32_t value;
      std::memcpy(&value, data.indexData + i * data.indexStride, sizeof(value));
      return value;
    }

    Vector3 cpuNormal(const GeometryBufferData& data, VkFormat format, uint32_t vertex) {
      const float* base = data.normalData + size_t(vertex) * data.normalStride;
      if (format == VK_FORMAT_R32_UINT) {
        uint32_t word;
        std::memcpy(&word, base, sizeof(word));
        // The word holds an unsigned octahedral direction as two 16-bit unorms (unsignedOctahedralToSphereDirection).
        float x = float(word & 0xFFFFu) / 65535.0f * 2.0f - 1.0f;
        float y = float(word >> 16u) / 65535.0f * 2.0f - 1.0f;
        const float z = 1.0f - std::fabs(x) - std::fabs(y);
        if (z < 0.0f) {
          const float ox = (1.0f - std::fabs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
          const float oy = (1.0f - std::fabs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
          x = ox;
          y = oy;
        }
        return Vector3(x, y, z);
      }
      return Vector3(base[0], base[1], base[2]);
    }

    // A mesh has hard edges when copies of one position have normals more than 30 degrees apart, as at box corners and
    // on brush geometry. The class is read from the CPU-visible input geometry.
    // It is Unreadable when the positions or normals are not CPU-visible.
    enum class MeshClass : uint8_t { Smooth, HardEdges, Unreadable };
    MeshClass classifyHardEdges(const RasterGeometry& input) {
      constexpr float kHardEdgeCosine = 0.866f;
      const GeometryBufferData cpu(input);
      const VkFormat normalFormat = input.normalBuffer.defined() ? input.normalBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
      if (cpu.positionData == nullptr || cpu.normalData == nullptr || !isFloatPositionFormat(input.positionBuffer.vertexFormat()) ||
          (normalFormat != VK_FORMAT_R32_UINT && normalFormat != VK_FORMAT_R32G32B32_SFLOAT &&
           normalFormat != VK_FORMAT_R32G32B32A32_SFLOAT)) {
        return MeshClass::Unreadable;
      }
      std::unordered_map<uint64_t, Vector3> firstNormal;
      firstNormal.reserve(input.vertexCount);
      for (uint32_t vertex = 0; vertex < input.vertexCount; ++vertex) {
        const Vector3 position = cpu.getPosition(vertex);
        uint64_t h = 0xCBF29CE484222325ull;
        h = mix(h, floatBits(position.x == 0.0f ? 0.0f : position.x));
        h = mix(h, floatBits(position.y == 0.0f ? 0.0f : position.y));
        h = mix(h, floatBits(position.z == 0.0f ? 0.0f : position.z));
        Vector3 normal = cpuNormal(cpu, normalFormat, vertex);
        const float length = vectorLength(normal);
        if (!(length > 0.0f) || !std::isfinite(length)) {
          continue;
        }
        normal = normal * (1.0f / length);
        const auto [it, inserted] = firstNormal.try_emplace(h, normal);
        if (!inserted && dot(it->second, normal) < kHardEdgeCosine) {
          return MeshClass::HardEdges;
        }
      }
      return MeshClass::Smooth;
    }
  }

  // One tessellated mesh holds its vertices and indices, each with its own bindless slot held until the entry is
  // dropped, and its BLAS.
  struct SccrTessCacheEntry {
    Rc<DxvkBuffer> vertices;
    Rc<DxvkBuffer> indices;
    Rc<DxvkAccelStructure> blas;
    uint64_t blasAddress = 0;
    uint32_t vertexSlot = kSurfaceInvalidBufferIndex;
    uint32_t indexSlot = kSurfaceInvalidBufferIndex;
    uint32_t triangles = 0;
    uint32_t vertexCount = 0;
    bool hasColor = false;
    size_t bytes = 0;
    uint64_t lastBoundFrame = 0;
    std::chrono::steady_clock::time_point lastUsed {};
    uint64_t lastCountedFrame = 0;
  };


  bool SccrTessManager::Key::operator==(const Key& other) const {
    return positionHash == other.positionHash && indexHash == other.indexHash && texcoordHash == other.texcoordHash &&
           materialHash == other.materialHash && vertexCount == other.vertexCount &&
           primitiveCount == other.primitiveCount && scale[0] == other.scale[0] && scale[1] == other.scale[1] &&
           scale[2] == other.scale[2];
  }

  size_t SccrTessManager::KeyHash::operator()(const Key& key) const {
    uint64_t h = 0xCBF29CE484222325ull;
    h = mix(h, key.positionHash);
    h = mix(h, key.indexHash);
    h = mix(h, key.texcoordHash);
    h = mix(h, key.materialHash);
    h = mix(h, (uint64_t(key.vertexCount) << 32) | key.primitiveCount);
    h = mix(h, (uint64_t(uint16_t(key.scale[0])) << 32) | (uint64_t(uint16_t(key.scale[1])) << 16) | uint16_t(key.scale[2]));
    return size_t(h);
  }

  SccrTessManager::SccrTessManager(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  SccrTessManager::~SccrTessManager() = default;

  void SccrTessManager::onDestroy() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    // The scene manager and its buffer table go away with the device, so everything is dropped without releasing slots.
    m_pending.clear();
    m_pendingOrder.clear();
    m_unsupported.clear();
    m_bound.clear();
    m_cache.clear();
    m_graveyard.clear();
    m_bytes = 0;
    m_frameActive = false;
    m_wasActive = false;
    m_frameStats = Stats();
    m_scratch = nullptr;
  }

  bool SccrTessManager::active() {
    return enable() && SccrShellManager::active() && RtxOptions::Displacement::referenceWorldUnitsPerUv() > 0.0f;
  }

  uint64_t SccrTessManager::settingsSignature() const {
    uint64_t h = 0xCBF29CE484222325ull;
    h = mix(h, floatBits(targetEdge()));
    h = mix(h, maxSegments());
    h = mix(h, maxTrianglesPerMesh());
    h = mix(h, floatBits(phongShape()));
    h = mix(h, floatBits(RtxOptions::Displacement::displacementFactor()));
    h = mix(h, floatBits(RtxOptions::Displacement::displacementInFactor()));
    h = mix(h, floatBits(RtxOptions::Displacement::displacementOutFactor()));
    h = mix(h, floatBits(RtxOptions::Displacement::referenceWorldUnitsPerUv()));
    h = mix(h, SccrShellManager::foldGuard() ? 1u : 0u);
    // The shell data's welded ids and directions, which the tessellator uses at the corners, depend on the weld.
    h = mix(h, floatBits(SccrShellManager::weldEpsilon()));
    return h;
  }

  const SccrTessCacheEntry* SccrTessManager::boundEntry(const RtInstance& instance) const {
    if (!m_frameActive || !instance.isRealGeometry) {
      return nullptr;
    }
    const auto found = m_bound.find(&instance);
    return found != m_bound.end() ? found->second : nullptr;
  }

  uint64_t SccrTessManager::blasReference(const RtInstance& instance, uint64_t fallback) const {
    const SccrTessCacheEntry* entry = boundEntry(instance);
    return entry != nullptr ? entry->blasAddress : fallback;
  }

  uint32_t SccrTessManager::primitiveCount(const RtInstance& instance, uint32_t fallback) const {
    const SccrTessCacheEntry* entry = boundEntry(instance);
    return entry != nullptr ? entry->triangles : fallback;
  }

  void SccrTessManager::patchSurfaceRecord(unsigned char* record, const RtInstance& instance) const {
    const SccrTessCacheEntry* entry = boundEntry(instance);
    if (entry == nullptr) {
      return;
    }
    // Writes the record again from the instance's surface, with its geometry fields on the tessellated buffers: one
    // interleaved vertex buffer (sccr_tess_args.h) and 32-bit indices from 0. Everything else (material, transforms,
    // flags, hashes, picking) stays the instance's. writeGPUData leaves data15.w without a shell slot, because the shell
    // data indexes the base mesh's triangles. Candidates are never point instancers and never split (firstIndexOffset 0).
    RtSurface surface = instance.surface;
    surface.positionBufferIndex = entry->vertexSlot;
    surface.positionOffset = SCCR_TESS_VERTEX_POSITION_OFFSET;
    surface.positionStride = kVertexBytes;
    surface.previousPositionBufferIndex = kSurfaceInvalidBufferIndex;
    surface.normalBufferIndex = entry->vertexSlot;
    surface.normalOffset = SCCR_TESS_VERTEX_NORMAL_OFFSET;
    surface.normalStride = kVertexBytes;
    surface.normalFormat = VK_FORMAT_R32G32B32_SFLOAT;
    surface.texcoordBufferIndex = entry->vertexSlot;
    surface.texcoordOffset = SCCR_TESS_VERTEX_TEXCOORD_OFFSET;
    surface.texcoordStride = kVertexBytes;
    if (entry->hasColor) {
      surface.color0BufferIndex = entry->vertexSlot;
      surface.color0Offset = SCCR_TESS_VERTEX_COLOR_OFFSET;
      surface.color0Stride = kVertexBytes;
    } else {
      surface.color0BufferIndex = kSurfaceInvalidBufferIndex;
      surface.color0Offset = 0;
      surface.color0Stride = 0;
    }
    surface.indexBufferIndex = entry->indexSlot;
    surface.indexStride = sizeof(uint32_t);
    surface.firstIndex = 0;
    size_t offset = 0;
    surface.writeGPUData(record, offset);
  }

  void SccrTessManager::patchMaterialRecord(unsigned char* record, const RtInstance& instance) const {
    if (boundEntry(instance) == nullptr) {
      return;
    }
    // The opaque material record starts with its 16-bit flags (RtOpaqueSurfaceMaterial::writeGPUData), and candidates
    // are always opaque (SccrShellManager::classifyMaterial). Without the displacement flag no pass marches, offsets
    // depth or self-shadows the height map on this surface.
    uint16_t flags;
    std::memcpy(&flags, record, sizeof(flags));
    flags &= uint16_t(~uint16_t(OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT));
    std::memcpy(record, &flags, sizeof(flags));
  }

  bool SccrTessManager::acceptsMesh(const BlasEntry& blas) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    const RaytraceGeometry& geometry = blas.modifiedGeometryData;
    uint64_t key = 0xCBF29CE484222325ull;
    key = mix(key, geometry.hashes[HashComponents::VertexPosition]);
    key = mix(key, geometry.hashes[HashComponents::Indices]);
    key = mix(key, (uint64_t(geometry.vertexCount) << 32) | geometry.calculatePrimitiveCount());
    auto found = m_meshClass.find(key);
    if (found == m_meshClass.end()) {
      const MeshClass meshClass = classifyHardEdges(blas.input.getGeometryData());
      found = m_meshClass.emplace(key, uint8_t(meshClass)).first;
      if (meshClass == MeshClass::HardEdges) {
        ++m_hardEdgedMeshes;
      }
    }
    const MeshClass meshClass = MeshClass(found->second);
    return meshClass == MeshClass::Smooth;
  }

  SccrTessManager::Key SccrTessManager::keyOf(const SceneManager& sceneManager, const RtInstance& instance,
                                              bool& valid) const {
    Key key;
    valid = false;
    const BlasEntry* blas = instance.getBlas();
    const RtSurfaceMaterial* material = sceneManager.getSurfaceMaterialOrNull(instance.surface.surfaceMaterialIndex);
    if (blas == nullptr || material == nullptr || material->getType() != RtSurfaceMaterialType::Opaque) {
      return key;
    }
    const RaytraceGeometry& geometry = blas->modifiedGeometryData;
    key.positionHash = geometry.hashes[HashComponents::VertexPosition];
    key.indexHash = geometry.hashes[HashComponents::Indices];
    key.texcoordHash = geometry.hashes[HashComponents::VertexTexcoord];
    key.materialHash = material->getOpaqueSurfaceMaterial().getHash();
    key.vertexCount = geometry.vertexCount;
    key.primitiveCount = geometry.calculatePrimitiveCount();
    for (int axis = 0; axis < 3; ++axis) {
      const float length = vectorLength(columnOf(instance.surface.objectToWorld, axis));
      if (!(length > 0.0f) || !std::isfinite(length)) {
        return key;
      }
      key.scale[axis] = int16_t(std::clamp(std::lround(std::log2(length) * 32.0f), -32000l, 32000l));
    }
    valid = true;
    return key;
  }

  void SccrTessManager::scanInstances(SceneManager& sceneManager) {
    m_bound.clear();
    Stats counts;
    const Vector3 camera = sceneManager.getCamera().getPosition();
    const float distanceLimit = buildDistance();
    for (const RtInstance* instance : sceneManager.getInstanceTable()) {
      if (instance == nullptr || !instance->isRealGeometry) {
        continue;
      }
      ++counts.candidateInstances;
      bool valid;
      const Key key = keyOf(sceneManager, *instance, valid);
      if (!valid) {
        continue;
      }
      const auto found = m_cache.find(key);
      if (found != m_cache.end()) {
        SccrTessCacheEntry& entry = *found->second;
        m_bound.emplace(instance, &entry);
        ++counts.boundInstances;
        entry.lastBoundFrame = m_frame;
        entry.lastUsed = m_now;
        if (entry.lastCountedFrame != m_frame) {
          entry.lastCountedFrame = m_frame;
          ++counts.boundMeshes;
        }
        continue;
      }
      if (m_unsupported.find(key) != m_unsupported.end()) {
        continue;
      }
      // A mesh is built once an instance's bounds come within buildDistance of the camera.
      if (distanceLimit > 0.0f) {
        const AxisAlignedBoundingBox& box = instance->getBlas()->input.getGeometryData().boundingBox;
        float distance = vectorLength(camera - columnOf(instance->surface.objectToWorld, 3));
        if (box.isValid()) {
          const Vector3 center = (box.minPos + box.maxPos) * 0.5f;
          const Vector3 halfExtent = (box.maxPos - box.minPos) * 0.5f;
          const Vector4 worldCenter = instance->surface.objectToWorld * Vector4(center.x, center.y, center.z, 1.0f);
          float maxScale = 0.0f;
          for (int axis = 0; axis < 3; ++axis) {
            maxScale = std::max(maxScale, vectorLength(columnOf(instance->surface.objectToWorld, axis)));
          }
          distance = vectorLength(camera - Vector3(worldCenter.x, worldCenter.y, worldCenter.z)) -
                     vectorLength(halfExtent) * maxScale;
        }
        if (distance > distanceLimit) {
          continue;
        }
      }
      const auto [pending, inserted] = m_pending.try_emplace(key);
      if (inserted) {
        m_pendingOrder.push_back(key);
      }
      pending->second.instance = instance;
      pending->second.lastSeenFrame = m_frame;
    }
    m_frameStats.candidateInstances = counts.candidateInstances;
    m_frameStats.boundInstances = counts.boundInstances;
    m_frameStats.boundMeshes = counts.boundMeshes;
  }

  SccrTessManager::Cache::iterator SccrTessManager::eraseEntry(Cache::iterator it) {
    m_bytes -= std::min(m_bytes, it->second->bytes);
    m_graveyard.push_back(Grave { std::move(it->second), m_frame });
    return m_cache.erase(it);
  }

  void SccrTessManager::collectGraveyard(SceneManager& sceneManager, bool all) {
    for (auto it = m_graveyard.begin(); it != m_graveyard.end();) {
      if (!all && m_frame - it->frame < kGraveFrames) {
        ++it;
        continue;
      }
      SccrTessCacheEntry& entry = *it->entry;
      if (entry.vertexSlot != kSurfaceInvalidBufferIndex) {
        sceneManager.releaseBufferSlot(entry.vertexSlot);
      }
      if (entry.indexSlot != kSurfaceInvalidBufferIndex) {
        sceneManager.releaseBufferSlot(entry.indexSlot);
      }
      it = m_graveyard.erase(it);
    }
  }

  void SccrTessManager::evictCache(size_t bytesNeeded) {
    std::vector<Cache::iterator> candidates;
    for (auto it = m_cache.begin(); it != m_cache.end();) {
      const SccrTessCacheEntry& entry = *it->second;
      const bool boundThisFrame = entry.lastBoundFrame == m_frame;
      if (!boundThisFrame && m_now - entry.lastUsed >= kCacheKeepTime) {
        it = eraseEntry(it);
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
    std::sort(candidates.begin(), candidates.end(), [](const Cache::iterator& a, const Cache::iterator& b) {
      return a->second->lastUsed < b->second->lastUsed;
    });
    for (const Cache::iterator& it : candidates) {
      if (m_bytes + bytesNeeded <= capBytes) {
        break;
      }
      eraseEntry(it);
      ++m_sessionEvicted;
    }
  }

  void SccrTessManager::releaseAll(SceneManager& sceneManager) {
    for (auto it = m_cache.begin(); it != m_cache.end();) {
      it = eraseEntry(it);
    }
    m_bytes = 0;
    m_pending.clear();
    m_pendingOrder.clear();
    m_unsupported.clear();
    m_bound.clear();
  }

  void SccrTessManager::onSceneClear(SceneManager& sceneManager) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    releaseAll(sceneManager);
    m_meshClass.clear();
    m_hardEdgedMeshes = 0;
    // The device has gone idle for the clear, so nothing in flight references the dropped meshes.
    collectGraveyard(sceneManager, true);
    m_frameActive = false;
  }

  void SccrTessManager::prepareFrame(RtxContext& ctx) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    m_frameActive = false;
    SceneManager& sceneManager = m_device->getCommon()->getSceneManager();
    ++m_frame;
    m_now = std::chrono::steady_clock::now();
    collectGraveyard(sceneManager, false);

    if (!active()) {
      if (m_wasActive) {
        const size_t released = m_cache.size();
        releaseAll(sceneManager);
        // The TLAS and the surface records go back to the base meshes this frame.
        sceneManager.notifyAccelSceneChanged();
        Logger::info(str::format("[SCCR tess] Real geometry off: ", released, " tessellated meshes released."));
      }
      m_wasActive = false;
      m_frameStats = Stats();
      return;
    }

    const uint64_t settings = settingsSignature();
    bool changed = false;
    if (m_wasActive && settings != m_settings) {
      releaseAll(sceneManager);
      changed = true;
    }
    m_settings = settings;
    const bool justActivated = !m_wasActive;
    m_wasActive = true;
    m_frameActive = true;

    const size_t cachedBefore = m_cache.size();
    if (m_now - m_lastEviction >= kEvictionInterval) {
      evictCache(0);
      m_lastEviction = m_now;
    }

    scanInstances(sceneManager);
    const bool builtAny = buildPending(ctx, sceneManager);
    if (builtAny || m_cache.size() != cachedBefore) {
      // Binds the meshes built just now and forgets the ones dropped.
      scanInstances(sceneManager);
      changed = true;
    }
    if (changed || justActivated) {
      // A binding changed, so the TLAS references, surface records and material records must be written again.
      // The unchanged-scene path would keep last frame's.
      sceneManager.notifyAccelSceneChanged();
    }

    uint32_t waiting = 0;
    for (const auto& [key, pending] : m_pending) {
      if (pending.waiting) {
        ++waiting;
      }
    }
    uint64_t triangles = 0;
    for (const auto& [key, entry] : m_cache) {
      triangles += entry->triangles;
    }
    m_frameStats.pendingBuilds = uint32_t(m_pending.size());
    m_frameStats.waitingForData = waiting;
    m_frameStats.cachedMeshes = uint32_t(m_cache.size());
    m_frameStats.unsupportedMeshes = uint32_t(m_unsupported.size());
    m_frameStats.cachedTriangles = triangles;
    m_frameStats.hardEdgedMeshes = m_hardEdgedMeshes;
    m_frameStats.bytes = m_bytes;

    const Stats& s = m_frameStats;
    uint64_t signature = 0xCBF29CE484222325ull;
    for (const uint64_t value : { uint64_t(s.candidateInstances), uint64_t(s.boundInstances), uint64_t(s.pendingBuilds),
                                  uint64_t(s.cachedMeshes), uint64_t(s.unsupportedMeshes), uint64_t(m_sessionBuilds),
                                  uint64_t(m_sessionEvicted) }) {
      signature = mix(signature, value);
    }
    const bool buildRunOver = m_builtSinceReport > 0 && !builtAny;
    if (justActivated || buildRunOver || (signature != m_reportedSignature && m_now - m_lastReport >= kReportInterval)) {
      logReport();
      m_reportedSignature = signature;
    }
  }

  bool SccrTessManager::buildPending(RtxContext& ctx, SceneManager& sceneManager) {
    if (m_pendingOrder.empty()) {
      return false;
    }
    ScopedGpuProfileZone(&ctx, "SccrTess");

    const uint64_t triangleBudget = std::max<uint64_t>(buildTrianglesPerFrame(), 1);
    uint64_t trianglesBuilt = 0;
    bool builtAny = false;
    size_t remaining = m_pendingOrder.size();
    while (remaining-- > 0) {
      const Key key = m_pendingOrder.front();
      m_pendingOrder.pop_front();
      const auto found = m_pending.find(key);
      if (found == m_pending.end()) {
        continue;
      }
      PendingBuild& pending = found->second;
      if (pending.lastSeenFrame != m_frame) {
        if (m_frame - pending.lastSeenFrame > kPendingKeepFrames) {
          m_pending.erase(found);
        } else {
          m_pendingOrder.push_back(key);
        }
        continue;
      }
      if ((pending.overBudget || pending.waiting) && m_now < pending.retryAfter) {
        m_pendingOrder.push_back(key);
        continue;
      }
      if (builtAny && trianglesBuilt >= triangleBudget) {
        m_pendingOrder.push_front(key);
        break;
      }
      switch (build(ctx, sceneManager, key, *pending.instance)) {
      case BuildResult::Built:
        builtAny = true;
        trianglesBuilt += m_cache.find(key) != m_cache.end() ? m_cache.find(key)->second->triangles : 0;
        ++m_builtSinceReport;
        m_pending.erase(found);
        break;
      case BuildResult::Unsupported:
        m_unsupported.insert(key);
        m_pending.erase(found);
        break;
      case BuildResult::OverBudget:
        pending.overBudget = true;
        pending.waiting = false;
        pending.retryAfter = m_now + kOverBudgetRetry;
        m_pendingOrder.push_back(key);
        break;
      case BuildResult::Wait:
        pending.waiting = true;
        pending.retryAfter = m_now + kWaitRetry;
        m_pendingOrder.push_back(key);
        break;
      case BuildResult::Retry:
        m_pendingOrder.push_back(key);
        break;
      }
    }
    if (builtAny) {
      // The BLAS builds wrote acceleration structures that the TLAS build and the ray tracing passes read.
      // The vertex and index buffers are read bindlessly, which DXVK cannot track.
      ctx.emitMemoryBarrier(0,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
          VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
    }
    return builtAny;
  }

  SccrTessManager::BuildResult SccrTessManager::build(RtxContext& ctx, SceneManager& sceneManager, const Key& key,
                                                      const RtInstance& instance) {
    bool valid;
    if (!(keyOf(sceneManager, instance, valid) == key) || !valid) {
      return BuildResult::Retry;
    }
    const BlasEntry& blas = *instance.getBlas();
    const RaytraceGeometry& geometry = blas.modifiedGeometryData;
    const RtOpaqueSurfaceMaterial& material =
      sceneManager.getSurfaceMaterialOrNull(instance.surface.surfaceMaterialIndex)->getOpaqueSurfaceMaterial();

    // The surfaces read this geometry, and so does the shell build.
    const uint32_t numTriangles = geometry.calculatePrimitiveCount();
    const uint32_t numVertices = geometry.vertexCount;
    const bool indexed = geometry.usesIndices();
    const bool indices16 = indexed && geometry.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16;
    const VkFormat normalFormat = geometry.normalBuffer.defined() ? geometry.normalBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
    const bool supportedNormals = normalFormat == VK_FORMAT_R32_UINT || normalFormat == VK_FORMAT_R32G32B32_SFLOAT ||
                                  normalFormat == VK_FORMAT_R32G32B32A32_SFLOAT;
    const bool hasColor = geometry.color0Buffer.defined() && geometry.color0Buffer.stride() % 4 == 0;
    if (!geometry.positionBuffer.defined() || numTriangles == 0 || numVertices == 0 ||
        !isFloatPositionFormat(geometry.positionBuffer.vertexFormat()) || geometry.positionBuffer.stride() % 4 != 0 ||
        (indexed && !indices16 && geometry.indexBuffer.indexType() != VK_INDEX_TYPE_UINT32) || !supportedNormals ||
        geometry.normalBuffer.stride() % 4 != 0 || !geometry.texcoordBuffer.defined() ||
        geometry.texcoordBuffer.vertexFormat() != VK_FORMAT_R32G32_SFLOAT || geometry.texcoordBuffer.stride() % 4 != 0) {
      return BuildResult::Unsupported;
    }

    // The shell data, built within its own budget, may not be there yet. Neither may the streamed height texture,
    // whose loaded mips are checked below once the mip this mesh samples is known.
    const Rc<DxvkBuffer> shellBuffer = m_device->getCommon()->metaSccrShell().shellBufferFor(instance);
    if (shellBuffer == nullptr) {
      return BuildResult::Wait;
    }
    const RtxTextureManager& textureManager = m_device->getCommon()->getTextureManager();
    const uint32_t heightTextureIndex = material.getHeightTextureIndex();
    if (heightTextureIndex >= textureManager.getTextureTable().size()) {
      return BuildResult::Unsupported;
    }
    const TextureRef& heightTexture = textureManager.getTextureTable()[heightTextureIndex];
    DxvkImageView* heightView = heightTexture.getImageView();
    if (heightView == nullptr) {
      return BuildResult::Wait;
    }

    // Measures the input geometry, which has the same triangles, on the CPU: the mean edge, the mean UV edge and the
    // side the welded directions point to. The welded directions follow cross(p1 - p0, p2 - p0), and the vertex
    // normals point out.
    const RasterGeometry& input = blas.input.getGeometryData();
    const GeometryBufferData cpu(input);
    const bool inputIndexed = input.indexBuffer.defined();
    const bool inputIndices16 = inputIndexed && input.indexBuffer.indexType() == VK_INDEX_TYPE_UINT16;
    const uint32_t inputTriangles = input.calculatePrimitiveCount();
    const VkFormat inputNormalFormat = input.normalBuffer.defined() ? input.normalBuffer.vertexFormat() : VK_FORMAT_UNDEFINED;
    if (cpu.positionData == nullptr || (inputIndexed && cpu.indexData == nullptr) || cpu.normalData == nullptr ||
        !isFloatPositionFormat(input.positionBuffer.vertexFormat()) || inputTriangles != numTriangles ||
        (inputNormalFormat != VK_FORMAT_R32_UINT && inputNormalFormat != VK_FORMAT_R32G32B32_SFLOAT &&
         inputNormalFormat != VK_FORMAT_R32G32B32A32_SFLOAT)) {
      ONCE(Logger::info("[SCCR tess] A candidate mesh's vertices are not readable on the CPU (or not float positions with "
                        "normals): it keeps SCCR. Further such meshes are not logged."));
      return BuildResult::Unsupported;
    }
    const Matrix4& objectToWorld = instance.surface.objectToWorld;
    const Vector3 scale(vectorLength(columnOf(objectToWorld, 0)), vectorLength(columnOf(objectToWorld, 1)),
                        vectorLength(columnOf(objectToWorld, 2)));
    double edgeSum = 0.0;
    double uvEdgeSum = 0.0;
    double area = 0.0;
    double orientationSum = 0.0;
    uint32_t uvEdges = 0;
    const float target = std::max(targetEdge(), 1e-3f);
    const uint32_t segmentLimit = std::clamp<uint32_t>(maxSegments(), 1u, SCCR_TESS_MAX_SEGMENTS);
    double adaptiveTriangles = 0.0;
    for (uint32_t t = 0; t < inputTriangles; ++t) {
      uint32_t v[3];
      Vector3 p[3];
      for (uint32_t k = 0; k < 3; ++k) {
        v[k] = cpuIndex(cpu, inputIndexed, inputIndices16, t * 3 + k);
        if (v[k] >= input.vertexCount) {
          return BuildResult::Unsupported;
        }
        const Vector3 objectPosition = cpu.getPosition(v[k]);
        p[k] = Vector3(objectPosition.x * scale.x, objectPosition.y * scale.y, objectPosition.z * scale.z);
      }
      float longest = 0.0f;
      for (uint32_t k = 0; k < 3; ++k) {
        const float edge = vectorLength(p[(k + 1) % 3] - p[k]);
        edgeSum += edge;
        longest = std::max(longest, edge);
        if (cpu.texcoordData != nullptr) {
          const Vector2 d = cpu.getTexCoord(v[(k + 1) % 3]) - cpu.getTexCoord(v[k]);
          uvEdgeSum += std::sqrt(double(d.x) * d.x + double(d.y) * d.y);
          ++uvEdges;
        }
      }
      const Vector3 objectFace = cross(cpu.getPosition(v[1]) - cpu.getPosition(v[0]), cpu.getPosition(v[2]) - cpu.getPosition(v[0]));
      area += 0.5 * vectorLength(cross(p[1] - p[0], p[2] - p[0]));
      for (uint32_t k = 0; k < 3; ++k) {
        orientationSum += dot(objectFace, cpuNormal(cpu, inputNormalFormat, v[k]));
      }
      const double segments = std::clamp(std::ceil(double(longest) / target), 1.0, double(segmentLimit));
      adaptiveTriangles += segments * segments;
    }
    const double meanEdge = edgeSum / (3.0 * inputTriangles);
    const float orientation = orientationSum >= 0.0 ? 1.0f : -1.0f;

    // N is the mean edge over the target, within maxSegments, lowered until the mesh fits maxTrianglesPerMesh.
    uint32_t segments = uint32_t(std::clamp(std::lround(meanEdge / target), 1l, long(segmentLimit)));
    const uint64_t triangleCap = std::max<uint64_t>(maxTrianglesPerMesh(), numTriangles);
    while (segments > 1 && uint64_t(numTriangles) * segments * segments > triangleCap) {
      --segments;
    }
    const uint64_t microTriangles = uint64_t(numTriangles) * segments * segments;
    const uint64_t microVertices = uint64_t(numTriangles) * (segments + 1) * (segments + 2) / 2;
    if (microVertices >= (uint64_t(1) << 31) || microTriangles >= (uint64_t(1) << 30)) {
      return BuildResult::Unsupported;
    }

    // The height mip matches one micro-edge in UV at the texture's full size, and is the same for the whole mesh.
    // A streamed texture only needs that mip loaded, since the view starts at the asset's mip m_currentMip_begin.
    // The mesh keeps the relief it was built with.
    uint32_t loadedFirstMip = 0;
    if (const Rc<ManagedTexture>& managed = heightTexture.getManagedTexture(); managed != nullptr) {
      if (managed->m_state == ManagedTexture::State::kFailed) {
        return BuildResult::Unsupported;
      }
      loadedFirstMip = managed->m_currentMip_begin;
    }
    const VkExtent3D topExtent = heightView->mipLevelExtent(0);
    const float fullTextureSize = float(std::max(std::max(topExtent.width, topExtent.height), 1u)) * float(1u << std::min(loadedFirstMip, 15u));
    double uvEdgeMean = uvEdges != 0 ? uvEdgeSum / uvEdges : 0.0;
    const float heightLodFull = uvEdgeMean > 0.0 ?
      std::max(0.0f, float(std::log2(uvEdgeMean / segments * fullTextureSize))) : 0.0f;
    if (float(loadedFirstMip) > std::floor(heightLodFull)) {
      return BuildResult::Wait;
    }
    const float heightLod = heightLodFull - float(loadedFirstMip);

    // The BLAS size is queried first, so the memory cap covers all of it.
    VkAccelerationStructureGeometryKHR asGeometry {};
    asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    VkAccelerationStructureGeometryTrianglesDataKHR& triangles = asGeometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexStride = kVertexBytes;
    triangles.maxVertex = uint32_t(microVertices - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &asGeometry;
    const uint32_t primitiveCount = uint32_t(microTriangles);
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    m_device->vkd()->vkGetAccelerationStructureBuildSizesKHR(
      m_device->handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitiveCount, &sizeInfo);

    const VkDeviceSize vertexBytes = microVertices * kVertexBytes;
    const VkDeviceSize indexBytes = microTriangles * 3 * sizeof(uint32_t);
    const size_t bytes = size_t(vertexBytes + indexBytes + sizeInfo.accelerationStructureSize);
    const size_t capBytes = size_t(maxMB()) << 20;
    if (m_bytes + bytes > capBytes) {
      evictCache(bytes);
    }
    if (m_bytes + bytes > capBytes) {
      return BuildResult::OverBudget;
    }

    DxvkBufferCreateInfo info = {};
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    info.size = align(vertexBytes, 16);
    const Rc<DxvkBuffer> vertexBuffer = m_device->createBuffer(
      info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SCCR tessellated vertices");
    info.size = align(indexBytes, 16);
    const Rc<DxvkBuffer> indexBuffer = m_device->createBuffer(
      info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SCCR tessellated indices");
    DxvkBufferCreateInfo asInfo = {};
    asInfo.size = sizeInfo.accelerationStructureSize;
    asInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    asInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    asInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const Rc<DxvkAccelStructure> accel = m_device->createAccelStructure(
      asInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, "BLAS SCCR tessellated");
    if (vertexBuffer == nullptr || indexBuffer == nullptr || accel == nullptr) {
      return BuildResult::OverBudget;
    }

    const uint32_t vertexSlot = sceneManager.acquireBufferSlot(
      RaytraceBuffer(DxvkBufferSlice(vertexBuffer), 0u, kVertexBytes, VK_FORMAT_R32G32B32_SFLOAT));
    const uint32_t indexSlot = sceneManager.acquireBufferSlot(
      RaytraceBuffer(DxvkBufferSlice(indexBuffer), 0u, uint32_t(sizeof(uint32_t)), VK_FORMAT_R32_UINT));
    if (vertexSlot >= kSurfaceInvalidBufferIndex || indexSlot >= kSurfaceInvalidBufferIndex) {
      sceneManager.releaseBufferSlot(vertexSlot);
      sceneManager.releaseBufferSlot(indexSlot);
      ONCE(Logger::warn("[SCCR tess] The bindless buffer table is full: no slot for a tessellated mesh."));
      return BuildResult::OverBudget;
    }

    const float worldDepth =
      (std::fabs(material.getDisplaceIn() * getDisplacementInFactor()) + material.getDisplaceOut() * getDisplacementOutFactor()) *
      std::max(RtxOptions::Displacement::referenceWorldUnitsPerUv(), 0.0f);

    SccrTessArgs args {};
    args.positionOffset = uint32_t(geometry.positionBuffer.offset() + geometry.positionBuffer.offsetFromSlice());
    args.positionStride = geometry.positionBuffer.stride();
    args.normalOffset = uint32_t(geometry.normalBuffer.offset() + geometry.normalBuffer.offsetFromSlice());
    args.normalStride = geometry.normalBuffer.stride();
    args.texcoordOffset = uint32_t(geometry.texcoordBuffer.offset() + geometry.texcoordBuffer.offsetFromSlice());
    args.texcoordStride = geometry.texcoordBuffer.stride();
    args.colorOffset = hasColor ? uint32_t(geometry.color0Buffer.offset() + geometry.color0Buffer.offsetFromSlice()) : 0u;
    args.colorStride = hasColor ? geometry.color0Buffer.stride() : 0u;
    args.indexOffset = indexed ? uint32_t(geometry.indexBuffer.offset() + geometry.indexBuffer.offsetFromSlice()) : 0u;
    args.numTriangles = numTriangles;
    args.numVertices = numVertices;
    args.segments = segments;
    args.flags = (normalFormat == VK_FORMAT_R32_UINT ? SCCR_TESS_NORMAL_ENCODED : 0u) |
                 (indices16 ? SCCR_TESS_INDICES_16 : 0u) | (!indexed ? SCCR_TESS_NO_INDICES : 0u) |
                 (hasColor ? SCCR_TESS_HAS_COLOR : 0u) | (SccrShellManager::foldGuard() ? SCCR_TESS_FOLD_GUARD : 0u);
    args.objectScale = vec3(scale.x, scale.y, scale.z);
    args.worldDepth = worldDepth;
    args.orientation = orientation;
    args.heightLod = heightLod;
    args.phongShape = std::clamp(phongShape(), 0.0f, 1.0f);

    const Rc<DxvkSampler> sampler = m_device->getCommon()->getResources().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    // The geometry's buffers are read by dispatches recorded into this frame's command list, which keeps them alive.
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SccrTessShader::getShader());
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_POSITION_RO, DxvkBufferSlice(geometry.positionBuffer.buffer()));
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_NORMAL_RO, DxvkBufferSlice(geometry.normalBuffer.buffer()));
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_TEXCOORD_RO, DxvkBufferSlice(geometry.texcoordBuffer.buffer()));
    // Unused bindings must still be bound, so the position buffer stands in.
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_COLOR_RO,
                           DxvkBufferSlice(hasColor ? geometry.color0Buffer.buffer() : geometry.positionBuffer.buffer()));
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_INDEX_RO,
                           DxvkBufferSlice(indexed ? geometry.indexBuffer.buffer() : geometry.positionBuffer.buffer()));
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_SHELL_RO, DxvkBufferSlice(shellBuffer));
    ctx.bindResourceView(SCCR_TESS_BINDING_HEIGHT, heightTexture.getImageView(), nullptr);
    ctx.bindResourceSampler(SCCR_TESS_BINDING_HEIGHT, sampler);
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_VERTEX_OUTPUT, DxvkBufferSlice(vertexBuffer));
    ctx.bindResourceBuffer(SCCR_TESS_BINDING_INDEX_OUTPUT, DxvkBufferSlice(indexBuffer));

    const auto dispatchPhase = [&](uint32_t phase, uint64_t threads) {
      const uint32_t groups = uint32_t((threads + SCCR_TESS_GROUP_SIZE - 1) / SCCR_TESS_GROUP_SIZE);
      const uint32_t groupsPerRow = std::min<uint32_t>(groups, SCCR_TESS_MAX_GROUPS_PER_ROW);
      const uint32_t rows = (groups + groupsPerRow - 1) / groupsPerRow;
      args.phase = phase;
      args.threadCount = uint32_t(threads);
      args.rowThreads = groupsPerRow * SCCR_TESS_GROUP_SIZE;
      ctx.pushConstants(0, sizeof(args), &args);
      ctx.dispatch(groupsPerRow, rows, 1);
    };
    dispatchPhase(SCCR_TESS_PHASE_VERTICES, microVertices);
    dispatchPhase(SCCR_TESS_PHASE_INDICES, microTriangles);

    // Builds the BLAS from the buffers just written. The build is recorded directly, because DXVK does not track
    // acceleration structure builds.
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_SHADER_READ_BIT);
    const VkDeviceSize scratchAlignment =
      std::max<VkDeviceSize>(m_device->properties().khrDeviceAccelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment, 1);
    const VkDeviceSize scratchNeeded = sizeInfo.buildScratchSize + scratchAlignment;
    if (m_scratch == nullptr || m_scratch->info().size < scratchNeeded) {
      DxvkBufferCreateInfo scratchInfo {};
      scratchInfo.size = align(scratchNeeded, 256);
      scratchInfo.access = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      scratchInfo.stages = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      scratchInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      m_scratch = m_device->createBuffer(scratchInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         DxvkMemoryStats::Category::RTXAccelerationStructure, "SCCR tessellation BVH scratch");
    }
    triangles.vertexData.deviceAddress = vertexBuffer->getDeviceAddress() + SCCR_TESS_VERTEX_POSITION_OFFSET;
    triangles.indexData.deviceAddress = indexBuffer->getDeviceAddress();
    buildInfo.dstAccelerationStructure = accel->getAccelStructure();
    buildInfo.scratchData.deviceAddress = align(m_scratch->getDeviceAddress(), scratchAlignment);
    VkAccelerationStructureBuildRangeInfoKHR buildRange {};
    buildRange.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pBuildRange = &buildRange;
    ctx.getCommandList()->vkCmdBuildAccelerationStructuresKHR(1, &buildInfo, &pBuildRange);
    ctx.getCommandList()->trackResource<DxvkAccess::Read>(vertexBuffer);
    ctx.getCommandList()->trackResource<DxvkAccess::Read>(indexBuffer);
    ctx.getCommandList()->trackResource<DxvkAccess::Write>(m_scratch);
    ctx.getCommandList()->trackResource<DxvkAccess::Write>(accel);
    // The scratch buffer is reused by the next build this frame.
    ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    std::unique_ptr<SccrTessCacheEntry> entry = std::make_unique<SccrTessCacheEntry>();
    entry->vertices = vertexBuffer;
    entry->indices = indexBuffer;
    entry->blas = accel;
    entry->blasAddress = accel->getAccelDeviceAddress();
    entry->vertexSlot = vertexSlot;
    entry->indexSlot = indexSlot;
    entry->triangles = primitiveCount;
    entry->vertexCount = uint32_t(microVertices);
    entry->hasColor = hasColor;
    entry->bytes = bytes;
    entry->lastBoundFrame = m_frame;
    entry->lastUsed = m_now;
    m_cache.emplace(key, std::move(entry));
    m_bytes += bytes;
    ++m_sessionBuilds;

    // Logs what this mesh costs, and how many triangles per-triangle segment counts would give at the same target.
    Logger::info(str::format("[SCCR tess] Built mesh 0x", std::hex, key.positionHash, std::dec, ": ", numTriangles,
                             " base triangles, ", numVertices, " vertices; surface ", uint64_t(area), " sq units, mean edge ",
                             float(meanEdge), " units (scale ", scale.x, "); ", segments, " segments per edge -> ",
                             microTriangles, " triangles (", (bytes + (size_t(1) << 19)) >> 20, " MB, BLAS ",
                             (sizeInfo.accelerationStructureSize + (VkDeviceSize(1) << 19)) >> 20,
                             " MB); per-triangle rates at the same target would give ", uint64_t(adaptiveTriangles),
                             "; depth ", worldDepth, " units, height mip ", heightLodFull, " (", loadedFirstMip, " loaded first), ",
                             orientation > 0.0f ? "outward" : "inward", " welded directions."));
    return BuildResult::Built;
  }

  void SccrTessManager::logReport() {
    const Stats& s = m_frameStats;
    Logger::info(str::format("[SCCR tess] ", s.boundInstances, " of ", s.candidateInstances,
                             " candidate instances use a tessellated mesh this frame (", s.boundMeshes, " meshes); ",
                             s.pendingBuilds, " meshes waiting for a build (", s.waitingForData,
                             " for their shell data or full height texture), ", s.unsupportedMeshes, " unsupported; ",
                             s.hardEdgedMeshes, " hard-edged meshes seen (kept on shell tracing); ",
                             s.cachedMeshes, " meshes cached, ", s.cachedTriangles, " triangles, ",
                             (s.bytes + (size_t(1) << 19)) >> 20, " of ", maxMB(), " MB. Built since the last line: ",
                             m_builtSinceReport, ". Session: ", m_sessionBuilds, " built, ", m_sessionEvicted, " evicted."));
    m_builtSinceReport = 0;
    m_lastReport = m_now;
  }

  SccrTessManager::Stats SccrTessManager::stats() const {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    return m_frameStats;
  }
}
