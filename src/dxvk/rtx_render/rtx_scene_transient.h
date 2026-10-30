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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "rtx_scene_manager.h"

namespace dxvk {

  class RetainedScene;

  // The transient lane: per-frame DrawInstance calls become scene-owned
  // instances matched frame to frame, without a replacement instance or the draw-call cache.
  //
  // A draw is matched by key (mesh, material override, categories, camera type, submesh count) to the nearest
  // entry of that key not yet drawn this frame. Then: same description and pose, same transform -> kept (the preserve
  // pair, which re-registers view- and player-model lists); transform only (main camera, plain instance) -> moved in
  // place; new pose -> geometry updated in place (skinning) and the instance updated; anything else -> rebuilt.
  // Entries not drawn in a frame age out by translator::LifetimePolicy (the replacement-instance rule, anti-culling
  // included); kept unseen entries get the same per-frame stamp the anti-cull pass gives replacement instances.
  //
  // Particles (description, material or emitter category), USD mesh replacements and light conversion are not
  // taken: those draws keep the external-draw path. CS thread only.
  class TransientLane {
  public:
    static constexpr uint64_t kIdBit = uint64_t(1) << 62;  // RtInstance scene-owned handles of transient entries

    struct Stats {
      uint32_t kept = 0, moved = 0, posed = 0, built = 0, unseenKept = 0, collected = 0, declined = 0;
      // built, by reason: new entry, moved but not a plain main-camera instance (full update), description changed
      // (material / picking / instancing list / texture transform), assets changed (mesh refreshed or material),
      // blocked frame (texture cache, translation, mods).
      uint32_t builtNew = 0, builtMovedFull = 0, builtDescription = 0, builtAssets = 0, builtBlocked = 0;
    };

    TransientLane(SceneManager& scene, RetainedScene& retained);
    ~TransientLane();

    // rtx.scene.directTransient as last applied (read once per frame, at the end of injectRTX).
    bool enabled() const { return m_enabled; }
    void applyOption(bool enabled);

    // False: not taken; the caller uses submitExternalDraw. state is left as it was in that case.
    bool draw(const Rc<DxvkContext>& ctx, ExternalDrawState& state);

    // SceneManager::garbageCollection, after the replacement-instance collection: unseen entries age out or are kept.
    void collect(RtCamera& camera, bool antiCullingSupported);

    void onInstanceFreed(const RtInstance& instance);
    void onSceneClear();

    Stats takeStats() { Stats s = m_stats; m_stats = Stats {}; return s; }
    size_t size() const { return m_entries.size(); }

  private:
    struct Submesh {
      RtInstance* instance = nullptr;
      void* geometry = nullptr;   // RetainedScene::Geometry
    };

    struct Entry {
      uint64_t id = 0;
      XXH64_hash_t key = 0;
      XXH64_hash_t descHash = 0;
      XXH64_hash_t boneHash = 0;
      std::weak_ptr<const std::vector<RasterGeometry>> submeshes;
      std::vector<XXH64_hash_t> materialHashes;   // resolved per submesh at the last build
      std::vector<Submesh> subs;
      Matrix4 objectToWorld;
      AxisAlignedBoundingBox bounds {};           // object space, all submeshes
      uint32_t categories = 0;
      uint32_t frameCreated = 0;
      uint32_t frameLastSeen = 0;
      bool skinned = false;
      bool wasMoving = false;
      bool plainMain = false;                     // plain instances (no billboards, portals): moves can be applied in place
      CameraType::Enum cameraType = CameraType::Main;
    };

    SceneManager& m_scene;
    RetainedScene& m_retained;
    bool m_enabled = false;
    uint64_t m_nextId = 1;
    uint64_t m_assetGlobalEpoch = ~0ull;
    std::unordered_map<uint64_t, Entry> m_entries;
    std::unordered_map<XXH64_hash_t, std::vector<uint64_t>> m_byKey;
    Stats m_stats;

    static XXH64_hash_t matchKey(const ExternalDrawState& state, size_t submeshCount);
    static XXH64_hash_t descriptionHash(const ExternalDrawState& state);
    bool accepts(const ExternalDrawState& state, const std::vector<RasterGeometry>& submeshes) const;
    Entry* match(XXH64_hash_t key, const Matrix4& objectToWorld, uint32_t frame);
    bool build(const Rc<DxvkContext>& ctx, Entry& entry, ExternalDrawState& state,
               const std::shared_ptr<const std::vector<RasterGeometry>>& submeshesRef);
    void keep(Entry& entry);
    void release(Entry& entry);
    void erase(uint64_t id);
    bool assetsCurrent(const Entry& entry, const ExternalDrawState& state,
                       const std::shared_ptr<const std::vector<RasterGeometry>>& submeshesRef) const;
  };

} // namespace dxvk
