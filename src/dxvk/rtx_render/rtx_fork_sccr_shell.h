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

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_staging.h"
#include "../../util/thread.h"

namespace dxvk {
  class RtxContext;
  class RtInstance;
  class SceneManager;
  struct RtSurfaceMaterial;
  struct BlasEntry;
  struct RaytraceGeometry;

  // One built shell buffer in SccrShellManager's cache, with its own bindless buffer slot
  // (defined in rtx_fork_sccr_shell.cpp). Declared here so RaytraceGeometry's memo (SccrShellGeometryMemo, rtx_types.h)
  // can point at the entry it found.
  struct SccrShellCacheEntry;

  // Per static displaced mesh, a buffer of position-welded
  // triangle adjacency and welded inner-shell directions (layout: sccr_shell_args.h), built on the GPU
  // (sccr_shell_adjacency.comp.slang, modelled on RtxGeometryUtils::dispatchSmoothNormals) and bound through the
  // bindless buffer table; Surface.shellBufferIndex (data15.w) points at it.
  //
  // The buffers live in a cache keyed by what a build reads (the geometry's position and index hashes, vertex and
  // triangle counts). Each is registered in the bindless table once, when it is built, and released only when the cache
  // drops it (memory cap, kCacheKeepTime unbound, shell tracing off, a new weld tolerance, scene clear). Nothing ties a
  // buffer to a BlasEntry: the scene manager collects and recreates those all the time (numFramesToKeepBLAS 1), and a
  // recreated geometry with the same vertices finds the same buffer.
  //
  // Each frame prepareFrame (SceneManager::prepareSceneData, before the bindless table is uploaded) walks the instance
  // table, looks every eligible instance's mesh up in the cache (memoised on the geometry), queues the missing ones and
  // builds within buildTrianglesPerFrame. Then AccelManager's surface upload writes data15.w of every Surface record
  // through patchSurfaceRecord, the only writer of that field. A buffer built or dropped in a frame is therefore in (or
  // out of) that frame's bindless table and surface records together: no record can point at a released slot, and no
  // surface waits for an instance update to get its data.
  class SccrShellManager : public CommonDeviceObject {
  public:
    explicit SccrShellManager(DxvkDevice* device);
    ~SccrShellManager();

    // Releases the scratch allocator, whose reference to the device would otherwise keep the device alive
    // (CommonDeviceObject::onDestroy; called from DxvkObjects::onDestroy), and forgets the cache without releasing its
    // slots (the scene manager's table goes with the device). prepareFrame does nothing afterwards.
    void onDestroy() override;

    // Whether the shell data is wanted at all: enable on and SCCR mode.
    static bool active();
    // Once per frame, from SceneManager::prepareSceneData after garbage collection and the anti-cull preserve pass,
    // before the bindless resources are uploaded: drops the data when shell tracing is off (or the weld tolerance
    // changed), binds every eligible instance's mesh to its cached buffer, queues the missing ones and builds within
    // buildTrianglesPerFrame. A build reads the geometry's buffers at that moment, through a BlasEntry seen in this
    // frame's instance table; a queued mesh whose geometry is gone waits, and is dropped after kPendingKeepFrames.
    void prepareFrame(RtxContext& ctx);

    // AccelManager's surface upload, right after RtSurface::writeGPUData wrote the Surface record that starts at record:
    // overwrites its data15.w with the slot of instance's shell buffer (0xFFFF without one). Read-only, so it may run on
    // the upload's worker threads; nothing changes the cache while the upload runs (both run on the thread that records
    // the frame, prepareFrame first). Then SccrTessManager::patchSurfaceRecord (real geometry) gets its turn.
    void patchSurfaceRecord(unsigned char* record, const RtInstance& instance) const;

    // SCCR real geometry (SccrTessManager): whether instance may use a tessellated mesh, from draw state and material
    // only (never whether the shell data exists yet, so the flag is stable): shell-eligible, fully opaque and opaque in
    // the TLAS, no clip plane, not a view or player model, not a point instancer, one build geometry, texture coordinates from the buffer as they are (no transform,
    // generation or sprite sheet). Two-sided draws qualify.
    static bool isRealGeometryCandidate(const SceneManager& sceneManager, const RtInstance& instance);
    // The shell buffer bound to instance's mesh this frame (null without one), for the tessellator.
    Rc<DxvkBuffer> shellBufferFor(const RtInstance& instance) const;

    // The SCCR march budgets as one quality level. Custom means the budgets match none of the levels.
    enum class Quality { High, Medium, Low, Custom };
    // Writes the shell and relief march budgets of a level (deferred). Called by the graphics presets and the settings UI.
    static void setQualityLevel(Quality quality);
    // The level the current budgets match.
    static Quality qualityLevel();

    // SceneManager::clear, before it verifies and clears the bindless buffer table: releases every cached buffer's slot
    // and forgets the queue (its BlasEntry pointers are about to dangle).
    void onSceneClear(SceneManager& sceneManager);

    // processGeometryInfo, kUpdateBVH: the vertices changed, so the geometry's position hash, and with it its cache key,
    // changes (the old buffer stays cached until evicted). The second time on a geometry the scan has seen makes it
    // dynamic (plain SCCR). Static: it only touches the geometry's memo.
    static void onGeometryVerticesChanged(RaytraceGeometry& geometry);

    // For the settings UI and the log: the last frame's counts.
    struct Stats {
      // Instances with an eligible material (opaque, displaced, not terrain) on static geometry, and those of them whose
      // mesh had shell data bound this frame.
      uint32_t eligibleSurfaces = 0;
      uint32_t boundSurfaces = 0;
      // Distinct meshes bound this frame; meshes waiting for a build (of those, the ones that did not fit under maxMB
      // at their last try); cached buffers; meshes that cannot be built.
      uint32_t boundMeshes = 0;
      uint32_t pendingBuilds = 0;
      uint32_t overBudgetBuilds = 0;
      uint32_t cachedBuffers = 0;
      uint32_t unsupportedMeshes = 0;
      // An eligible material on skinned or vertex-animated geometry (plain SCCR): skinnedSurfaces + animatedSurfaces.
      uint32_t dynamicSurfaces = 0;
      size_t bytes = 0;
      // Why the other instances of the table got no shell data, one count per instance (diagnostics, so a zero above
      // explains itself in the log): no geometry linked; a material index outside the material cache; a material that
      // is not opaque; an opaque one without displacement; blended terrain (marches layer heights); skinned; vertices
      // changed twice; eligible, but its mesh cannot be built (counted in eligibleSurfaces too).
      uint32_t noGeometry = 0;
      uint32_t noMaterial = 0;
      uint32_t notOpaque = 0;
      uint32_t notDisplaced = 0;
      uint32_t terrain = 0;
      uint32_t skinnedSurfaces = 0;
      uint32_t animatedSurfaces = 0;
      uint32_t unsupportedSurfaces = 0;
      // Instances walked.
      uint32_t instances = 0;
    };
    Stats stats() const;

    // RaytraceArgs::pomSccrShellFlags: SCCR_SHELL_MARCH_FLAG_* from the options (sccr_shell_args.h). conservativeCones:
    // PomConeMapManager::conesConservative.
    static uint32_t marchFlags(bool conservativeCones);

    RTX_OPTION("rtx.displacement.shell", bool, enable, false,
               "SCCR shell tracing: marches displaced static meshes through prisms between the mesh and an inner shell "
               "offset along the vertex normals, so deep relief stays attached to the mesh across triangle edges "
               "instead of sliding. Builds per-mesh data (see the SCCR Shell Edges debug view) and marches primary and "
               "reflected (PSR) hits closer than rtx.displacement.shell.maxDistance. The outline stays the mesh's.\n"
               "While on, SCCR ignores the zero plane (the mesh is the top of the relief) and always uses world-unit "
               "depth, so rtx.displacement.referenceWorldUnitsPerUv must be above 0. SCCR mode only. Skinned and "
               "vertex-animated meshes and blended terrain layers use plain SCCR. Turning it off, or leaving SCCR mode, "
               "frees the data.");
    RTX_OPTION("rtx.displacement.shell", float, maxDistance, 2000.0f,
               "SCCR shell tracing: the hit distance, in world units, beyond which pixels use plain SCCR instead of the "
               "shell march. Nearer pixels also use plain SCCR where the whole relief depth is less than a pixel. 0 "
               "turns the march off, but the data is still built. The SCCR quality levels and graphics presets set it "
               "in proportion to rtx.displacement.referenceWorldUnitsPerUv (2000 at 256 for High).");
    RTX_OPTION("rtx.displacement.shell", uint32_t, maxPrisms, 32,
               "SCCR shell tracing: the most neighbouring prisms one march crosses into. Grazing views near outlines at "
               "large depths cross the most, about 20 on a smooth boulder. A march that runs out while rising above the "
               "relief counts as an outline. Otherwise the pixel uses plain SCCR (magenta in the SCCR Shell Result "
               "debug view).");
    RTX_OPTION("rtx.displacement.shell", uint32_t, maxSteps, 128,
               "SCCR shell tracing: the most steps one march takes, counting every solve, crossing and refinement step. "
               "A march that runs out is handled like one that runs out of prisms (rtx.displacement.shell.maxPrisms).");
    RTX_OPTION("rtx.displacement.shell", bool, foldGuard, true,
               "SCCR shell tracing: limits the shell depth at vertices where the inner shell would fold over itself (at "
               "sharp convex corners, thin slabs and small features), so the relief gets shallower there instead of "
               "breaking up. When off, the full depth is used everywhere, and marches that reach a folded prism stop "
               "there (cyan in the SCCR Shell Result debug view).");
    RTX_OPTION("rtx.displacement.shell", bool, shadows, true,
               "SCCR shell tracing: direct-light self-shadows of shell pixels march towards the light through the same "
               "prisms as the view ray, so crevice shadows stay on the carved relief. Indirect light keeps the cheaper "
               "texture-space self-shadow. When off, direct light uses the texture-space self-shadow too.");
    RTX_OPTION("rtx.displacement.shell", uint32_t, maxShadowPrisms, 4,
               "SCCR shell tracing: the most neighbouring prisms one self-shadow march crosses into on its way to the "
               "light (at most 255). A shadow ray that needs more counts as lit from there.");
    RTX_OPTION("rtx.displacement.shell", float, weldEpsilon, 0.0f,
               "SCCR shell tracing: how close two vertex positions must be to count as one when triangles are joined to "
               "their neighbours, as a fraction of the mesh's bounding box diagonal. 0 joins only identical positions, "
               "which suits meshes whose split vertices (hard edges and UV seams) copy the position exactly. Raise it "
               "(1e-6 to 1e-4) for meshes that show red edges or magenta corners in the SCCR Shell Edges debug view "
               "where the surface is closed. Too much merges real details. Changing it rebuilds all shell data.");
    RTX_OPTION("rtx.displacement.shell", uint32_t, maxMB, 128,
               "SCCR shell tracing: the most memory, in MB, the per-mesh shell data may use (12 bytes per triangle and "
               "12 per vertex). A mesh that would exceed it uses plain SCCR, and is tried again once memory is freed.");
    RTX_OPTION("rtx.displacement.shell", uint32_t, buildTrianglesPerFrame, 500000,
               "SCCR shell tracing: the number of triangles of shell data built per frame (at least one mesh), so "
               "loading many meshes at once never stalls a frame. A mesh uses plain SCCR until its data is built.");

  private:
    // What a build reads (vertex positions and indices, by the geometry's hashes, and their counts): the cache key.
    struct DataKey {
      uint64_t positionHash = 0;
      uint64_t indexHash = 0;
      uint32_t vertexCount = 0;
      uint32_t primitiveCount = 0;
      bool operator==(const DataKey& other) const {
        return positionHash == other.positionHash && indexHash == other.indexHash && vertexCount == other.vertexCount &&
               primitiveCount == other.primitiveCount;
      }
    };
    struct DataKeyHash {
      size_t operator()(const DataKey& key) const;
    };
    // Entries by pointer: the type is only complete in the .cpp, and memos keep pointers to them.
    using Cache = std::unordered_map<DataKey, std::unique_ptr<SccrShellCacheEntry>, DataKeyHash>;

    // A mesh waiting for its build. blas is only followed when lastSeenFrame is the current frame (set by this frame's
    // scan, after garbage collection), so a geometry collected since is never touched.
    struct PendingBuild {
      BlasEntry* blas = nullptr;
      uint64_t lastSeenFrame = 0;
      // Over the memory cap at the last try: not again before this time.
      std::chrono::steady_clock::time_point retryAfter {};
      bool overBudget = false;
    };

    // Retry: the geometry's key moved since the scan queued it (left queued for the next scan to sort out).
    enum class BuildResult { Built, Unsupported, OverBudget, Retry };

    // Why an instance gets no shell data (Eligible: it does, once its mesh is built). The walk counts each reason
    // (Stats); patchSurfaceRecord asks the same question, so the two always agree.
    enum class Eligibility { Eligible, NoGeometry, NoMaterial, NotOpaque, NotDisplaced, Terrain, Skinned, Animated };
    // Opaque, displaced, and not blended terrain. material may be null (an index outside the material cache).
    static Eligibility classifyMaterial(const SceneManager& sceneManager, const RtSurfaceMaterial* material);
    // Skinned, or vertices changed twice: plain SCCR.
    static Eligibility classifyGeometry(const BlasEntry& blas);
    // Both, for one instance of the table.
    static Eligibility classify(const SceneManager& sceneManager, const RtInstance& instance);
    // For an Eligible instance: fully opaque and opaque in the opaque TLAS (no alpha test or blend, no clip plane, no
    // forced non-opacity), not a view or player model. Draw state and material only.
    static bool isSolidOpaque(const RtInstance& instance);
    static DataKey keyOf(const RaytraceGeometry& geometry);
    // The slot instance's record gets: valid only for an eligible material on static geometry whose mesh is cached.
    uint32_t slotFor(const RtInstance& instance) const;

    // One walk of the instance table (m_mutex held): memoises each eligible mesh's cache entry on its geometry, queues
    // the missing ones (one entry per key) and counts the frame's statistics.
    void scanInstances(SceneManager& sceneManager);
    // Builds pending meshes within buildTrianglesPerFrame (at least one per frame). True if anything was built.
    bool buildPending(RtxContext& ctx, SceneManager& sceneManager);
    BuildResult build(RtxContext& ctx, SceneManager& sceneManager, const DataKey& key, BlasEntry& blas);
    // Drops one cached buffer: releases its slot and invalidates every memo (m_generation). m_mutex held.
    Cache::iterator eraseEntry(SceneManager& sceneManager, Cache::iterator it);
    // Drops cached buffers not bound for kCacheKeepTime, and (least recently bound first, never one bound this frame) as
    // many more as it takes to fit bytesNeeded under the cap (bytesNeeded 0: only the old ones). m_mutex held.
    void evictCache(SceneManager& sceneManager, size_t bytesNeeded);
    // Everything: the cache (slots released), the queue, the unsupported list. m_mutex held.
    void releaseAll(SceneManager& sceneManager);
    void logReport();

    std::unique_ptr<RtxStagingDataAlloc> m_scratch;
    mutable dxvk::mutex m_mutex;
    Cache m_cache;
    std::unordered_map<DataKey, PendingBuild, DataKeyHash> m_pending;
    // Build order of m_pending (a key may appear after its entry is gone: skipped).
    std::deque<DataKey> m_pendingOrder;
    std::unordered_set<DataKey, DataKeyHash> m_unsupported;
    size_t m_bytes = 0;
    // Bumped whenever an entry is erased from the cache: a memo's entry pointer is followed only in the generation it
    // was taken in (0 is never current). Misses are not memoised (looked up each frame), so additions need no bump.
    uint32_t m_generation = 1;
    uint64_t m_frame = 0;
    // Counts scanInstances passes (up to two per frame), for Stats::boundMeshes.
    uint64_t m_scanSerial = 0;
    std::chrono::steady_clock::time_point m_now {};
    // prepareFrame ran with shell tracing active this frame: patchSurfaceRecord binds data.
    bool m_frameActive = false;
    // Some instance may still carry RtInstance::isRealGeometry from a frame with real geometry on (cleared once shell
    // tracing is off).
    bool m_flagsSet = false;
    bool m_wasActive = false;
    // The weldEpsilon the current data was built with (a change rebuilds everything).
    float m_builtWeldEpsilon = 0.0f;
    std::chrono::steady_clock::time_point m_lastEviction {};
    // This frame's counts (scanInstances).
    Stats m_frameStats;
    // Session counters for the log: buffers built, cached buffers evicted (memory cap or unused), queued meshes dropped
    // because their geometry did not come back for kPendingKeepFrames.
    uint32_t m_sessionBuilds = 0;
    uint32_t m_sessionEvicted = 0;
    uint32_t m_sessionCancelled = 0;
    uint32_t m_builtSinceReport = 0;
    std::chrono::steady_clock::time_point m_lastReport {};
    // A digest of what the last report said, to skip identical lines.
    uint64_t m_reportedSignature = 0;
  };
}
