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

  // Remix API retained instances, second implementation (rtx.retained.useV2).
  //
  // Every registered instance is an Entry in a slot it keeps for life (free-list reuse). Entries
  // are in one of four places:
  //   awake    -- in m_awake; the next replay evaluates it (changed, new, or never holdable).
  //   held     -- in m_held; it passed the full preserve checks and nothing it depends on has
  //               changed since, so the replay only stamps it alive (one parallel pass).
  //   resident -- in m_resident; hidden under rtx.retainedHiddenPolicy 2: out of the TLAS, its
  //               replacement instance stamped alive.
  //   idle     -- nowhere; hidden with nothing to keep alive (policy 0, or never replayed).
  // Host commands and runtime events move entries between them. A replay walks m_awake only.
  //
  // Pointers in m_held / m_resident stay valid because every way they can die is reported
  // synchronously: the draw-call tracker (replacement instance destroyed or re-keyed), the
  // instance manager's freed hook, and SceneManager::clear. Releasing an entry never
  // dereferences them: the tags left on instances and replacement instances carry the entry's
  // serial, which the release bumps.
  //
  // CS thread only.
  class RetainedInstanceStore {
  public:
    // What an entry's last full replay produced. Lets a replay check the entry against the
    // runtime's current state instead of replaying it.
    struct ReplayRecord {
      bool valid = false;
      XXH64_hash_t identity = 0;                                  // external-draw identity hash
      Matrix4 applied;                                            // rebased transform the RT instance got
      std::weak_ptr<const std::vector<RasterGeometry>> submeshes; // the mesh's submesh vector it used
      XXH64_hash_t materialHash = 0;                              // resolved material hash it used
    };

    // One registered instance, as handed between the two implementations when
    // rtx.retained.useV2 changes (see SceneManager::selectRetainedImplementation).
    struct Handover {
      uint64_t handle = 0;
      ExternalDrawState desc;
      Matrix4 absolute;
      bool visible = true;
      bool hiddenFromTlas = false;  // its instances were hidden externally (policy 2)
      bool forceFull = false;       // description replaced since its last replay
      ReplayRecord record;
    };

    struct FrameStats {
      bool timeFullReplays = false; // in: time the full replays ([PerfCpu] retained.full)
      double fullMs = 0.0;
      uint32_t full = 0;            // full external-draw path (submitExternalDraw)
      uint32_t validated = 0;       // passed the full checks this frame and were preserved per entry
      uint32_t held = 0;            // stamped by the keep-alive pass
      uint32_t moved = 0;           // transform applied to the RT instance directly
      uint32_t hidden = 0;          // hidden entries evaluated this frame (not replayed)
      uint32_t awake = 0;           // entries evaluated this frame
      uint32_t released = 0;        // holds released since the last replay
      uint32_t checked = 0;         // compared by the self-check
    };

    explicit RetainedInstanceStore(SceneManager& scene);
    ~RetainedInstanceStore();

    RetainedInstanceStore(const RetainedInstanceStore&) = delete;
    RetainedInstanceStore& operator=(const RetainedInstanceStore&) = delete;

    // Host commands, in command-stream order.
    // Registers the handle, or replaces its whole description (keeps its visibility).
    void describe(uint64_t handle, ExternalDrawState&& desc);
    void setTransform(uint64_t handle, const Matrix4& absolute);
    void setVisible(uint64_t handle, bool visible);
    void destroy(uint64_t handle);
    // DrawRetainedInstances: bring the RT scene up to date with every registered instance.
    void replay(const Rc<DxvkContext>& ctx, const Vector3& origin, FrameStats& stats);

    // Runtime events. Each may arrive at any time on the CS thread, including from inside replay().
    void onReplacementInstanceGone(uint64_t ticket);  // DrawCallTracker, before the RI dies or is re-keyed
    void onInstanceFreed(const RtInstance& instance); // InstanceManager, before the memory is freed
    void onGeometryUpdated(BlasEntry& blas);          // a draw rebuilt/refit this geometry this frame
    void onSceneClear();                              // SceneManager::clear, after every RI died
    uint64_t holdEpoch() const { return m_holdEpoch; }

    // Implementation switch. import takes the entries (it moves out of them); export empties the store.
    void importEntries(std::vector<Handover>& entries);
    void exportEntries(std::vector<Handover>& out);

    size_t size() const { return m_slotOf.size(); }

  private:
    static constexpr uint32_t kNone = UINT32_MAX;

    enum : uint16_t {
      kLive          = 1u << 0,  // slot in use
      kVisible       = 1u << 1,  // host visibility
      kInAwakeList   = 1u << 2,  // present in m_awake
      kForceFull     = 1u << 3,  // next evaluation takes the full path (description replaced)
      kHiddenFromTlas = 1u << 4, // we hid its instances (policy 2)
      kMovePending   = 1u << 5,  // held entry got a new transform
      kIdle          = 1u << 6,  // hidden, nothing to do per frame
    };

    struct Entry {
      uint64_t handle = 0;
      uint32_t serial = 0;        // bumped on every hold / resident start and end, and on slot reuse
      uint32_t heldIndex = kNone;
      uint32_t residentIndex = kNone;
      uint32_t nextFree = kNone;
      uint32_t evaluatedReplay = 0;         // last replay (m_replayCount) that evaluated it
      uint32_t listMark = 0;                // awake-list compaction dedupe (m_replayCount)
      uint16_t flags = 0;
      Matrix4 absolute;                     // host transform (before the scene origin)
      ExternalDrawState desc;
      ReplayRecord record;
      AxisAlignedBoundingBox bounds {};     // submesh bounds at the last successful check
      uint64_t indexedAssets[3] = {};       // mesh, submesh material, override material (while held)
    };

    // Everything the keep-alive stamp touches, 32 bytes per held entry.
    struct Held {
      RtInstance* instance;
      BlasEntry* blas;
      ReplacementInstance* ri;
      uint32_t slot;
      uint32_t stampedFrame;  // frame whose stamps this entry already has
    };

    struct Resident {
      ReplacementInstance* ri;
      uint32_t slot;
    };

    // Result of the full preserve checks.
    struct Check {
      ReplacementInstance* ri = nullptr;
      RtInstance* instance = nullptr;
      BlasEntry* blas = nullptr;
      std::shared_ptr<const std::vector<RasterGeometry>> submeshes;
      const MaterialData* material = nullptr;
      XXH64_hash_t materialHash = 0;
      const char* failure = nullptr;
    };

    enum class Outcome : uint8_t { Awake, Held, Resident, Idle };

    // Per-replay values, read once.
    struct FrameInputs {
      uint32_t frame = 0;
      uint32_t policy = 0;
      bool preserveOn = false;   // rtx.enableRetainedPreservePath && rtx.enablePreservePath
      bool blocked = false;      // draw-call translation invalid, or texture cache not valid for preserve
      bool picking = false;      // object picking output active
      bool holdOn = false;       // entries may be held
      bool directOn = false;     // transforms / visibility may be applied to instances directly
      float uiOffset = 0.f;      // rtx.worldSpaceUiBackgroundOffset
    } m_now;

    // What the current holds were made under; a change releases them all.
    struct HoldInputs {
      uint64_t trackerClear = ~0ull;
      uint64_t assetGlobal = ~0ull;
      uint64_t lightConversion = ~0ull;
      bool replacementMeshes = false;
      bool holdOn = false;
      uint32_t policy = ~0u;
    } m_holdInputs;

    SceneManager& m_scene;
    std::vector<Entry> m_entries;
    uint32_t m_freeHead = kNone;
    std::unordered_map<uint64_t, uint32_t> m_slotOf;
    std::vector<uint32_t> m_awake;
    std::vector<uint8_t> m_awakeKeep;
    std::vector<Held> m_held;
    std::vector<Resident> m_resident;
    std::unordered_multimap<uint64_t, uint64_t> m_assetIndex;  // API mesh / material handle -> entry tag
    std::vector<uint64_t> m_changedAssets;
    std::vector<uint32_t> m_fullStamps;
    std::vector<uint8_t> m_stampDeferred;
    std::vector<uint8_t> m_originStatus;
    std::vector<uint32_t> m_refused;
    uint64_t m_holdEpoch = 1;
    uint32_t m_serialCounter = 0;   // every serial comes from here, so tags never repeat (until 2^32 bumps)
    uint32_t m_replayCount = 0;
    uint32_t m_requeueBudget = 0;
    Vector3 m_origin { 0.f, 0.f, 0.f };
    bool m_inReplay = false;
    uint32_t m_released = 0;
    uint32_t m_checkCursor = 0;
    uint32_t m_residentCheckCursor = 0;
    uint32_t m_checkMismatches = 0;
    uint64_t m_preservedForReport = 0;

    static uint64_t makeTag(uint32_t slot, uint32_t serial) { return (uint64_t(serial) << 32) | uint64_t(slot + 1u); }
    bool decodeTag(uint64_t tag, uint32_t& slot) const;
    Matrix4 rebase(const Matrix4& absolute) const;

    uint32_t allocateSlot();
    void freeSlot(uint32_t slot);
    void wake(uint32_t slot);
    void release(uint32_t slot);          // held -> not held (no dereference)
    void releaseResident(uint32_t slot);  // resident -> not resident (no dereference)
    void releaseAllHolds();
    void requeueAll();
    void addResident(uint32_t slot, ReplacementInstance* ri);
    bool tryHold(uint32_t slot, const Check& check);
    void indexAssets(Entry& entry, uint64_t tag, const Check& check);
    void compactAssetIndex();

    void readFrameInputs();
    void applyGlobalInvalidations();
    void consumeChangedAssets();
    void applyOriginChange(const Vector3& origin, FrameStats& stats);
    void selfCheck(FrameStats& stats);
    const char* checkHeld(const Entry& entry, const Held& held) const;
    Outcome evaluate(const Rc<DxvkContext>& ctx, uint32_t slot, FrameStats& stats);
    void stampHeld(FrameStats& stats);
    void compactAwake();

    bool fullCheck(const Entry& entry, Check& check) const;
    bool isPlainInstance(const RtInstance& instance, const BlasEntry& blas) const;
    bool relocate(Entry& entry, RtInstance& instance, BlasEntry& blas, ReplacementInstance& ri,
                  const Matrix4& target, const AxisAlignedBoundingBox& bounds, bool notifyScene, bool* outMoved);
    void noteMoved(const RtInstance& instance);
    void preserveNow(Entry& entry, const Check& check, const Matrix4& target);
    void fullReplay(const Rc<DxvkContext>& ctx, Entry& entry, const Matrix4& target, FrameStats& stats);
    bool setPrimsHidden(ReplacementInstance& ri, bool hidden);
  };

}  // namespace dxvk
