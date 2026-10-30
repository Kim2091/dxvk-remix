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
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "../../util/thread.h"
#include "../../util/rc/util_rc_ptr.h"

namespace dxvk {
  class RtxContext;
  class RtInstance;
  class SceneManager;
  class DxvkBuffer;
  class DxvkAccelStructure;
  struct BlasEntry;

  // One tessellated mesh in SccrTessManager's cache, defined in rtx_fork_sccr_tess.cpp.
  struct SccrTessCacheEntry;

  // SCCR real geometry turns the relief of static displaced meshes into real triangles.
  // Every ray, including shadows, bounce light, reflections and the ground at a
  // rock's foot, then meets the bumps without the shell march or carving.
  //
  // Each mesh is built once. It is cut into a regular grid of N segments per base edge, where N comes from the mesh's
  // mean edge and rtx.displacement.realGeometry.targetEdge, capped by maxTrianglesPerMesh. Every point is pushed in
  // along the shell data's welded directions by the material's height map, exactly as the shell's prisms place the
  // relief, and the result gets its own ordinary BLAS (sccr_tess.comp.slang, sccr_tess_args.h). Nothing is rebuilt per
  // frame. A mesh is built when an instance of it is near the camera, shared by every instance of it at the same scale,
  // and kept while used.
  //
  // Each frame, SccrShellManager's walk flags the candidates (RtInstance::isRealGeometry). The flag depends only on
  // draw state and material, and a flagged instance gets its own TLAS entry and no carving. prepareFrame then binds
  // each flagged instance to its built mesh. A bound instance's TLAS entry references the tessellated BLAS
  // (blasReference), and its Surface record points at the tessellated buffers (patchSurfaceRecord). Its material record
  // has displacement switched off (patchMaterialRecord), because the depth is in the geometry, so there is no POM,
  // no SCCR depth offset and no POM self-shadowing. An instance whose mesh is not built yet keeps the base mesh and
  // SCCR, so nothing waits for a build or renders black.
  class SccrTessManager : public CommonDeviceObject {
  public:
    explicit SccrTessManager(DxvkDevice* device);
    ~SccrTessManager();

    void onDestroy() override;

    // Returns true when real geometry is enabled, shell tracing is active (its data is the tessellator's input) and
    // the reference density is above 0.
    static bool active();

    // Binds flagged instances to built meshes and builds missing ones within buildTrianglesPerFrame.
    // SceneManager::prepareSceneData calls it once per frame, right after SccrShellManager::prepareFrame (whose shell
    // buffers it reads) and before the bindless table is uploaded.
    void prepareFrame(RtxContext& ctx);

    // Returns whether the mesh of a candidate instance (SccrShellManager::isRealGeometryCandidate) may be tessellated.
    // SccrShellManager's walk calls it. A mesh with hard edges keeps shell tracing. A hard edge is a position whose
    // copies have normals more than 30 degrees apart, as at box corners and on brush geometry. A mesh whose vertices are
    // not readable on the CPU keeps shell tracing too. The answer is decided once per mesh from its geometry, so the
    // candidate flag, which moves the instance between BLAS buckets, is stable.
    bool acceptsMesh(const BlasEntry& blas);

    // Releases every slot when SceneManager::clear runs, as SccrShellManager::onSceneClear does.
    void onSceneClear(SceneManager& sceneManager);

    // The lookups below serve the scene upload. They are read-only and safe on any thread, because nothing changes
    // the bindings between one prepareFrame and the next.

    // Returns the tessellated BLAS for a TLAS entry of a bound instance.
    // Returns fallback when the instance is not bound.
    uint64_t blasReference(const RtInstance& instance, uint64_t fallback) const;
    // Returns the tessellated triangle count of a bound instance's surface, for the NEE cache's primitive prefix sum.
    // Returns fallback when the instance is not bound.
    uint32_t primitiveCount(const RtInstance& instance, uint32_t fallback) const;
    // Rewrites a bound instance's Surface record with the tessellated buffers and no shell slot.
    // SccrShellManager::patchSurfaceRecord calls it after writing the shell slot.
    void patchSurfaceRecord(unsigned char* record, const RtInstance& instance) const;
    // Clears the displacement flag in a bound instance's material record, after that record was written.
    void patchMaterialRecord(unsigned char* record, const RtInstance& instance) const;

    struct Stats {
      uint32_t candidateInstances = 0; // Instances flagged by RtInstance::isRealGeometry.
      uint32_t boundInstances = 0;     // Instances using a tessellated mesh this frame.
      uint32_t boundMeshes = 0;
      uint32_t pendingBuilds = 0;
      uint32_t waitingForData = 0;     // Queued meshes whose shell data or full height texture is not loaded yet.
      uint32_t cachedMeshes = 0;
      uint32_t unsupportedMeshes = 0;
      uint32_t hardEdgedMeshes = 0;    // Meshes classified as hard-edged since the last scene clear.
      uint64_t cachedTriangles = 0;
      size_t bytes = 0;
    };
    Stats stats() const;

    RTX_OPTION("rtx.displacement.realGeometry", bool, enable, false,
               "SCCR real geometry: cuts smooth displaced meshes (rocks, terrain-like props) into small triangles and pushes "
               "them in by their height map once, so shadows, bounce light and reflections see the real bumps without ray "
               "marching. Meshes with hard edges (walls, boxes, brush geometry), moving meshes and meshes not built yet keep "
               "shell tracing. Needs SCCR mode and shell tracing on.");
    RTX_OPTION("rtx.displacement.realGeometry", float, targetEdge, 6.0f,
               "SCCR real geometry: the intended length of a small triangle's edge, in world units. Each mesh is cut into "
               "the same number of segments on every edge, chosen from its mean edge length. Smaller values are finer, and "
               "memory grows with the square of the segment count. Changing it rebuilds every mesh.");
    RTX_OPTION("rtx.displacement.realGeometry", uint32_t, maxSegments, 16,
               "SCCR real geometry: the most segments one base edge is cut into, up to 64. Changing it rebuilds every mesh.");
    RTX_OPTION("rtx.displacement.realGeometry", uint32_t, maxTrianglesPerMesh, 131072,
               "SCCR real geometry: the most small triangles one mesh may have. A mesh that would have more is cut "
               "coarser. Changing it rebuilds every mesh.");
    RTX_OPTION("rtx.displacement.realGeometry", float, phongShape, 0.5f,
               "SCCR real geometry: how much the base triangles are rounded towards the mesh's smooth shape (Phong "
               "tessellation of the welded directions) before the relief is applied, from 0 to 1. 0 keeps flat triangles, "
               "and 0.75 is the value commonly used for Phong tessellation. Changing it rebuilds every mesh.");
    RTX_OPTION("rtx.displacement.realGeometry", float, buildDistance, 4000.0f,
               "SCCR real geometry: a mesh is built once an instance of it comes within this distance of the camera, in "
               "world units to its bounds. Built meshes are used at any distance and are dropped after a minute without a "
               "bound instance. 0 builds every candidate mesh.");
    RTX_OPTION("rtx.displacement.realGeometry", uint32_t, maxMB, 1024,
               "SCCR real geometry: the most video memory, in MB, that the tessellated meshes may take for vertices, indices "
               "and BLAS. A mesh that would go over it keeps SCCR and is tried again once memory frees up.");
    RTX_OPTION("rtx.displacement.realGeometry", uint32_t, buildTrianglesPerFrame, 1000000,
               "SCCR real geometry: how many small triangles are built per frame, with at least one mesh per frame, so "
               "loading many meshes at once does not stall a frame. A mesh keeps SCCR until it is built.");

  private:
    // Identifies everything a build reads: the mesh (positions, indices, texture coordinates, counts), the material
    // (its hash covers the height texture and the depths) and the instance scale the depth is converted with.
    struct Key {
      uint64_t positionHash = 0;
      uint64_t indexHash = 0;
      uint64_t texcoordHash = 0;
      uint64_t materialHash = 0;
      uint32_t vertexCount = 0;
      uint32_t primitiveCount = 0;
      // The log2 of each object axis' world length, in 1/32 steps (about 2 %).
      int16_t scale[3] = { 0, 0, 0 };
      bool operator==(const Key& other) const;
    };
    struct KeyHash {
      size_t operator()(const Key& key) const;
    };
    using Cache = std::unordered_map<Key, std::unique_ptr<SccrTessCacheEntry>, KeyHash>;

    struct PendingBuild {
      const RtInstance* instance = nullptr; // Only dereferenced when lastSeenFrame is the current frame.
      uint64_t lastSeenFrame = 0;
      std::chrono::steady_clock::time_point retryAfter {};
      bool overBudget = false;
      bool waiting = false;
    };

    enum class BuildResult { Built, Unsupported, OverBudget, Wait, Retry };

    Key keyOf(const SceneManager& sceneManager, const RtInstance& instance, bool& valid) const;
    void scanInstances(SceneManager& sceneManager);
    bool buildPending(RtxContext& ctx, SceneManager& sceneManager);
    BuildResult build(RtxContext& ctx, SceneManager& sceneManager, const Key& key, const RtInstance& instance);
    // Drops an entry and moves its slots and buffers to the graveyard for a few frames, because last frame's TLAS,
    // which ReSTIR traces, and last frame's surface records may still reference them.
    Cache::iterator eraseEntry(Cache::iterator it);
    void evictCache(size_t bytesNeeded);
    void releaseAll(SceneManager& sceneManager);
    void collectGraveyard(SceneManager& sceneManager, bool all);
    void logReport();
    const SccrTessCacheEntry* boundEntry(const RtInstance& instance) const;
    // Hashes the settings every build depends on. A change drops all meshes.
    uint64_t settingsSignature() const;

    mutable dxvk::mutex m_mutex;
    Cache m_cache;
    std::unordered_map<Key, PendingBuild, KeyHash> m_pending;
    std::deque<Key> m_pendingOrder;
    std::unordered_set<Key, KeyHash> m_unsupported;
    // The verdicts of acceptsMesh by mesh (MeshClass in the .cpp), and how many of them were hard-edged.
    std::unordered_map<uint64_t, uint8_t> m_meshClass;
    uint32_t m_hardEdgedMeshes = 0;
    // The bindings prepareFrame made this frame, read by the upload's lookups.
    std::unordered_map<const RtInstance*, const SccrTessCacheEntry*> m_bound;
    struct Grave {
      std::unique_ptr<SccrTessCacheEntry> entry;
      uint64_t frame = 0;
    };
    std::vector<Grave> m_graveyard;
    size_t m_bytes = 0;
    uint64_t m_frame = 0;
    std::chrono::steady_clock::time_point m_now {};
    std::chrono::steady_clock::time_point m_lastEviction {};
    std::chrono::steady_clock::time_point m_lastReport {};
    uint64_t m_settings = 0;
    bool m_wasActive = false;
    bool m_frameActive = false;
    Stats m_frameStats;
    Rc<DxvkBuffer> m_scratch;
    uint32_t m_sessionBuilds = 0;
    uint32_t m_sessionEvicted = 0;
    uint32_t m_builtSinceReport = 0;
    uint64_t m_reportedSignature = 0;
  };
}
