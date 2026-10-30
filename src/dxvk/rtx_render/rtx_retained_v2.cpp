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
#include "rtx_retained_v2.h"

#include <chrono>
#include <cstring>
#include <intrin.h>
#include <ppl.h>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "../util/util_once.h"
#include "rtx_asset_replacer.h"
#include "rtx_context.h"
#include "rtx_fork_hooks.h"
#include "rtx_game_capturer.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_options.h"
#include "rtx_preserved_object_picking.h"
#include "rtx_texture_manager.h"

namespace dxvk {

  namespace {
    bool sameBits(const Matrix4& a, const Matrix4& b) {
      return std::memcmp(&a, &b, sizeof(Matrix4)) == 0;
    }

    bool sameBits(const Vector3& a, const Vector3& b) {
      return std::memcmp(&a, &b, sizeof(Vector3)) == 0;
    }

    // The light-conversion list is a set; a change to it releases every hold (a held entry's material
    // may have started or stopped converting to a light). Order-independent fold of its contents.
    uint64_t lightConversionSignature() {
      const auto& hashes = RtxOptions::lightConverter();
      uint64_t sum = uint64_t(hashes.size()) * 0x9E3779B97F4A7C15ull;
      uint64_t folded = 0x6A09E667F3BCC909ull;
      for (const XXH64_hash_t hash : hashes) {
        sum += hash * 0xD6E8FEB86659FD93ull;
        folded ^= (hash << 29) | (hash >> 35);
      }
      return sum ^ (folded * 0xBF58476D1CE4E5B9ull);
    }

    // The instance hidden flag the full path would derive, for instances whose hidden state the
    // external hide/show must not override (categories, sky, alpha tested / blended toggles).
    bool fullPathHides(const RtInstance& instance) {
      const BlasEntry* blas = instance.getBlas();
      return instance.testCategoryFlags(InstanceCategories::Hidden) ||
        (blas != nullptr && blas->input.cameraType == CameraType::Sky) ||
        instance.hiddenByAlphaToggles();
    }

    constexpr size_t kParallelStampThreshold = 2048;
    constexpr uint32_t kStampChunk = 512;
  }

  RetainedInstanceStore::RetainedInstanceStore(SceneManager& scene)
    : m_scene(scene) {
  }

  RetainedInstanceStore::~RetainedInstanceStore() = default;

  // ---- slots, tags, lists ---------------------------------------------------------------------

  bool RetainedInstanceStore::decodeTag(uint64_t tag, uint32_t& slot) const {
    if (tag == 0) {
      return false;
    }
    slot = uint32_t(tag & 0xFFFFFFFFull) - 1u;
    if (slot >= m_entries.size()) {
      return false;
    }
    const Entry& entry = m_entries[slot];
    return (entry.flags & kLive) != 0 && entry.serial == uint32_t(tag >> 32);
  }

  Matrix4 RetainedInstanceStore::rebase(const Matrix4& absolute) const {
    Matrix4 rebased = absolute;
    rebased[3].x -= m_origin.x;
    rebased[3].y -= m_origin.y;
    rebased[3].z -= m_origin.z;
    return rebased;
  }

  uint32_t RetainedInstanceStore::allocateSlot() {
    uint32_t slot;
    if (m_freeHead != kNone) {
      slot = m_freeHead;
      m_freeHead = m_entries[slot].nextFree;
    } else {
      slot = uint32_t(m_entries.size());
      m_entries.emplace_back();
    }
    Entry& entry = m_entries[slot];
    entry.nextFree = kNone;
    // A slot freed since the last replay may still sit in the awake list; it stays there once.
    entry.flags &= kInAwakeList;
    entry.heldIndex = kNone;
    entry.residentIndex = kNone;
    entry.serial = ++m_serialCounter;
    return slot;
  }

  void RetainedInstanceStore::freeSlot(uint32_t slot) {
    Entry& entry = m_entries[slot];
    entry.handle = 0;
    entry.flags &= kInAwakeList;
    entry.desc = ExternalDrawState {};
    entry.record = ReplayRecord {};
    entry.bounds = AxisAlignedBoundingBox {};
    std::fill(std::begin(entry.indexedAssets), std::end(entry.indexedAssets), 0ull);
    entry.serial = ++m_serialCounter;
    entry.nextFree = m_freeHead;
    m_freeHead = slot;
  }

  void RetainedInstanceStore::wake(uint32_t slot) {
    Entry& entry = m_entries[slot];
    entry.flags &= ~kIdle;
    if ((entry.flags & kInAwakeList) == 0) {
      entry.flags |= kInAwakeList;
      m_awake.push_back(slot);
    } else if (m_inReplay && entry.evaluatedReplay == m_replayCount && m_requeueBudget != 0) {
      // Already evaluated by this replay (held a moment ago, then released by something later in the
      // same replay, e.g. another entry's full replay updating shared geometry): once more before the
      // replay ends, so it does not go unstamped this frame.
      --m_requeueBudget;
      m_awake.push_back(slot);
    }
  }

  void RetainedInstanceStore::release(uint32_t slot) {
    Entry& entry = m_entries[slot];
    const uint32_t index = entry.heldIndex;
    if (index == kNone) {
      return;
    }
    const uint32_t last = uint32_t(m_held.size() - 1);
    if (index != last) {
      m_held[index] = m_held[last];
      m_entries[m_held[index].slot].heldIndex = index;
    }
    m_held.pop_back();
    entry.heldIndex = kNone;
    entry.serial = ++m_serialCounter;
    ++m_released;
  }

  void RetainedInstanceStore::releaseResident(uint32_t slot) {
    Entry& entry = m_entries[slot];
    const uint32_t index = entry.residentIndex;
    if (index == kNone) {
      return;
    }
    const uint32_t last = uint32_t(m_resident.size() - 1);
    if (index != last) {
      m_resident[index] = m_resident[last];
      m_entries[m_resident[index].slot].residentIndex = index;
    }
    m_resident.pop_back();
    entry.residentIndex = kNone;
    entry.serial = ++m_serialCounter;
  }

  void RetainedInstanceStore::releaseAllHolds() {
    for (const Held& held : m_held) {
      Entry& entry = m_entries[held.slot];
      entry.heldIndex = kNone;
      entry.serial = ++m_serialCounter;
      wake(held.slot);
    }
    m_released += uint32_t(m_held.size());
    m_held.clear();
    m_assetIndex.clear();
    ++m_holdEpoch;  // leaves every BlasEntry mark stale
  }

  void RetainedInstanceStore::requeueAll() {
    releaseAllHolds();
    for (const Resident& resident : m_resident) {
      Entry& entry = m_entries[resident.slot];
      entry.residentIndex = kNone;
      entry.serial = ++m_serialCounter;
    }
    m_resident.clear();
    for (uint32_t slot = 0; slot < m_entries.size(); ++slot) {
      if (m_entries[slot].flags & kLive) {
        wake(slot);
      }
    }
  }

  void RetainedInstanceStore::addResident(uint32_t slot, ReplacementInstance* ri) {
    Entry& entry = m_entries[slot];
    entry.residentIndex = uint32_t(m_resident.size());
    entry.serial = ++m_serialCounter;
    m_resident.push_back(Resident { ri, slot });
    ri->retainedV2Ticket = makeTag(slot, entry.serial);
  }

  bool RetainedInstanceStore::isPlainInstance(const RtInstance& instance, const BlasEntry& blas) const {
    // What the keep-alive stamp and the direct move leave out must not apply to the instance:
    // billboards and unordered resolve are rebuilt per frame from the transform, player-model and
    // view-model instances are re-registered per frame, portals refresh their portal data.
    return !instance.usesUnorderedApproximations() &&
      instance.getBillboardCount() == 0 &&
      !instance.isPlayerModelInstance() &&
      instance.getMaterialType() != MaterialDataType::RayPortal &&
      blas.input.cameraType == CameraType::Main;
  }

  bool RetainedInstanceStore::tryHold(uint32_t slot, const Check& check) {
    if (!m_now.holdOn || !isPlainInstance(*check.instance, *check.blas)) {
      return false;
    }
    Entry& entry = m_entries[slot];
    entry.heldIndex = uint32_t(m_held.size());
    entry.serial = ++m_serialCounter;
    const uint64_t tag = makeTag(slot, entry.serial);
    // The caller preserved or moved the entry this frame, so it already has this frame's stamps.
    m_held.push_back(Held { check.instance, check.blas, check.ri, slot, m_now.frame });
    check.instance->setRetainedV2Tag(tag);
    check.ri->retainedV2Ticket = tag;
    check.blas->retainedV2HoldEpoch = m_holdEpoch;
    entry.bounds = (*check.submeshes)[0].boundingBox;
    indexAssets(entry, tag, check);
    return true;
  }

  void RetainedInstanceStore::indexAssets(Entry& entry, uint64_t tag, const Check& check) {
    entry.indexedAssets[0] = reinterpret_cast<uint64_t>(entry.desc.mesh);
    entry.indexedAssets[1] = reinterpret_cast<uint64_t>((*check.submeshes)[0].externalMaterial);
    entry.indexedAssets[2] = reinterpret_cast<uint64_t>(entry.desc.materialOverride);
    for (const uint64_t asset : entry.indexedAssets) {
      if (asset != 0) {
        m_assetIndex.emplace(asset, tag);
      }
    }
    // Tags of released holds are dropped lazily; rebuild once they dominate.
    if (m_assetIndex.size() > 4 * m_held.size() + 8192) {
      compactAssetIndex();
    }
  }

  void RetainedInstanceStore::compactAssetIndex() {
    m_assetIndex.clear();
    for (const Held& held : m_held) {
      const Entry& entry = m_entries[held.slot];
      const uint64_t tag = makeTag(held.slot, entry.serial);
      for (const uint64_t asset : entry.indexedAssets) {
        if (asset != 0) {
          m_assetIndex.emplace(asset, tag);
        }
      }
    }
  }

  // ---- host commands --------------------------------------------------------------------------

  void RetainedInstanceStore::describe(uint64_t handle, ExternalDrawState&& desc) {
    uint32_t slot;
    const auto found = m_slotOf.find(handle);
    if (found == m_slotOf.end()) {
      slot = allocateSlot();
      m_slotOf.emplace(handle, slot);
      Entry& entry = m_entries[slot];
      entry.handle = handle;
      entry.flags |= kLive | kVisible;
      entry.record = ReplayRecord {};
    } else {
      slot = found->second;
      // A hidden-resident entry stays resident (and hidden); it takes the full path when shown.
      release(slot);
    }
    Entry& entry = m_entries[slot];
    entry.absolute = desc.drawCall.getTransformData().objectToWorld;
    entry.desc = std::move(desc);
    entry.flags = uint16_t((entry.flags | kForceFull) & ~kMovePending);
    wake(slot);
  }

  void RetainedInstanceStore::setTransform(uint64_t handle, const Matrix4& absolute) {
    const auto found = m_slotOf.find(handle);
    if (found == m_slotOf.end()) {
      return;
    }
    const uint32_t slot = found->second;
    Entry& entry = m_entries[slot];
    if (sameBits(entry.absolute, absolute)) {
      return;
    }
    entry.absolute = absolute;
    if (entry.heldIndex != kNone) {
      // Stays held: the replay moves its instance directly through the held pointers.
      entry.flags |= kMovePending;
      wake(slot);
    } else if (entry.residentIndex == kNone && (entry.flags & kIdle) == 0) {
      wake(slot);
    }
    // Hidden entries (resident or idle) pick the transform up when they are shown.
  }

  void RetainedInstanceStore::setVisible(uint64_t handle, bool visible) {
    const auto found = m_slotOf.find(handle);
    if (found == m_slotOf.end()) {
      return;
    }
    const uint32_t slot = found->second;
    Entry& entry = m_entries[slot];
    if (((entry.flags & kVisible) != 0) == visible) {
      return;
    }
    entry.flags = uint16_t(visible ? (entry.flags | kVisible) : (entry.flags & ~kVisible));
    // Under a policy that ray traces hidden entries as if visible, a held entry has nothing to do.
    const uint32_t policy = m_holdInputs.policy;
    if (entry.heldIndex != kNone && policy != 0 && policy != 2) {
      return;
    }
    wake(slot);
  }

  void RetainedInstanceStore::destroy(uint64_t handle) {
    const auto found = m_slotOf.find(handle);
    if (found == m_slotOf.end()) {
      return;
    }
    const uint32_t slot = found->second;
    m_slotOf.erase(found);
    // Its instance stops being kept alive and ages out through the unseen-instance GC.
    release(slot);
    releaseResident(slot);
    freeSlot(slot);
  }

  // ---- runtime events ---------------------------------------------------------------------------

  void RetainedInstanceStore::onReplacementInstanceGone(uint64_t ticket) {
    uint32_t slot;
    if (!decodeTag(ticket, slot)) {
      return;
    }
    Entry& entry = m_entries[slot];
    if (entry.heldIndex != kNone) {
      release(slot);
      wake(slot);
    } else if (entry.residentIndex != kNone) {
      releaseResident(slot);
      wake(slot);
    }
  }

  void RetainedInstanceStore::onInstanceFreed(const RtInstance& instance) {
    uint32_t slot;
    if (!decodeTag(instance.getRetainedV2Tag(), slot)) {
      return;
    }
    if (m_entries[slot].heldIndex != kNone) {
      release(slot);
      wake(slot);
    }
  }

  void RetainedInstanceStore::onGeometryUpdated(BlasEntry& blas) {
    if (blas.retainedV2HoldEpoch != m_holdEpoch || m_held.empty()) {
      return;
    }
    for (RtInstance* instance : blas.getLinkedInstances()) {
      uint32_t slot;
      if (decodeTag(instance->getRetainedV2Tag(), slot) && m_entries[slot].heldIndex != kNone) {
        // Its preserve would now fail (geometry updated this frame): the full evaluation instead.
        release(slot);
        wake(slot);
      }
    }
  }

  void RetainedInstanceStore::onSceneClear() {
    // Every replacement instance, instance and geometry is gone: nothing held or resident survives,
    // and nothing is hidden any more.
    for (const Held& held : m_held) {
      Entry& entry = m_entries[held.slot];
      entry.heldIndex = kNone;
      entry.serial = ++m_serialCounter;
    }
    m_held.clear();
    for (const Resident& resident : m_resident) {
      Entry& entry = m_entries[resident.slot];
      entry.residentIndex = kNone;
      entry.serial = ++m_serialCounter;
    }
    m_resident.clear();
    m_assetIndex.clear();
    ++m_holdEpoch;
    for (uint32_t slot = 0; slot < m_entries.size(); ++slot) {
      Entry& entry = m_entries[slot];
      if (entry.flags & kLive) {
        entry.flags &= ~kHiddenFromTlas;
        wake(slot);
      }
    }
  }

  // ---- implementation switch --------------------------------------------------------------------

  void RetainedInstanceStore::importEntries(std::vector<Handover>& entries) {
    for (Handover& handover : entries) {
      if (handover.handle == 0) {
        continue;
      }
      uint32_t slot;
      const auto found = m_slotOf.find(handover.handle);
      if (found != m_slotOf.end()) {
        slot = found->second;
        release(slot);
        releaseResident(slot);
      } else {
        slot = allocateSlot();
        m_slotOf.emplace(handover.handle, slot);
      }
      Entry& entry = m_entries[slot];
      entry.handle = handover.handle;
      entry.flags = uint16_t((entry.flags & kInAwakeList) | kLive |
        (handover.visible ? kVisible : 0) |
        (handover.hiddenFromTlas ? kHiddenFromTlas : 0) |
        (handover.forceFull ? kForceFull : 0));
      entry.absolute = handover.absolute;
      entry.desc = std::move(handover.desc);
      entry.record = std::move(handover.record);
      wake(slot);
    }
    entries.clear();
  }

  void RetainedInstanceStore::exportEntries(std::vector<Handover>& out) {
    out.reserve(out.size() + m_slotOf.size());
    for (Entry& entry : m_entries) {
      if ((entry.flags & kLive) == 0) {
        continue;
      }
      Handover handover;
      handover.handle = entry.handle;
      handover.desc = std::move(entry.desc);
      handover.absolute = entry.absolute;
      handover.visible = (entry.flags & kVisible) != 0;
      handover.hiddenFromTlas = (entry.flags & kHiddenFromTlas) != 0;
      handover.forceFull = (entry.flags & kForceFull) != 0;
      handover.record = std::move(entry.record);
      // A pending move is carried by the absolute transform; the receiver compares it with record.applied.
      out.push_back(std::move(handover));
    }
    m_entries.clear();
    m_freeHead = kNone;
    m_slotOf.clear();
    m_awake.clear();
    m_held.clear();
    m_resident.clear();
    m_assetIndex.clear();
    ++m_holdEpoch;
    m_holdInputs = HoldInputs {};
  }

  // ---- replay -----------------------------------------------------------------------------------

  void RetainedInstanceStore::readFrameInputs() {
    DxvkDevice* device = m_scene.device();
    m_now.frame = device->getCurrentFrameId();
    m_now.policy = RtxOptions::retainedHiddenPolicy();
    m_now.preserveOn = RtxOptions::enableRetainedPreservePath() && RtxOptions::enablePreservePath();
    m_now.blocked = RtxOptionManager::isDrawcallTranslationInvalid() ||
      device->getCommon()->getTextureManager().getTextureCacheGeneration() != m_scene.m_textureCacheGenerationValidForPreserve;
    m_now.picking = device->getCommon()->getResources().getRaytracingOutput().m_primaryObjectPicking.isValid();
    m_now.directOn = m_now.preserveOn && !m_now.blocked && !m_now.picking;
    m_now.holdOn = m_now.directOn && RtxOptions::Retained::v2HoldSteadyEntries();
    m_now.uiOffset = RtxOptions::worldSpaceUiBackgroundOffset();
  }

  void RetainedInstanceStore::applyGlobalInvalidations() {
    HoldInputs now;
    now.trackerClear = m_scene.m_drawCallTracker.getClearGeneration();
    now.assetGlobal = m_scene.m_pReplacer->getExternalAssetGlobalEpoch();
    now.lightConversion = lightConversionSignature();
    now.replacementMeshes = RtxOptions::getEnableReplacementMeshes();
    now.holdOn = m_now.holdOn;
    now.policy = m_now.policy;

    if (m_holdInputs.trackerClear != ~0ull && now.trackerClear != m_holdInputs.trackerClear) {
      // SceneManager::clear reports itself; this catches a tracker cleared any other way.
      onSceneClear();
    }
    if (now.policy != m_holdInputs.policy) {
      // Every entry's place depends on the policy.
      requeueAll();
    } else if (now.assetGlobal != m_holdInputs.assetGlobal ||
               now.lightConversion != m_holdInputs.lightConversion ||
               now.replacementMeshes != m_holdInputs.replacementMeshes ||
               now.holdOn != m_holdInputs.holdOn) {
      // Mods, reloads, variants, light conversion or the preserve options changed what a held
      // entry's checks would find, for any entry.
      releaseAllHolds();
    }
    if (m_now.blocked && !m_held.empty()) {
      // Texture cache cleared or draw-call translation invalid: this frame every preserve path is
      // closed, so every visible entry takes the full path (as with the first implementation).
      releaseAllHolds();
    }
    m_holdInputs = now;
  }

  void RetainedInstanceStore::consumeChangedAssets() {
    m_changedAssets.clear();
    m_scene.m_pReplacer->takeChangedExternalAssets(m_changedAssets);
    for (const uint64_t asset : m_changedAssets) {
      const auto range = m_assetIndex.equal_range(asset);
      for (auto it = range.first; it != range.second; ++it) {
        uint32_t slot;
        if (decodeTag(it->second, slot) && m_entries[slot].heldIndex != kNone) {
          release(slot);
          wake(slot);
        }
      }
      m_assetIndex.erase(range.first, range.second);
    }
    m_changedAssets.clear();
  }

  bool RetainedInstanceStore::relocate(Entry& entry, RtInstance& instance, BlasEntry& blas, ReplacementInstance& ri,
                                       const Matrix4& target, const AxisAlignedBoundingBox& bounds,
                                       bool notifyScene, bool* outMoved) {
    if (!m_scene.m_instanceManager.relocateExternally(instance, target, CameraType::Main, m_now.uiOffset, notifyScene, outMoved)) {
      return false;
    }
    // What the full path also leaves behind for a moved draw: the geometry counts as touched this
    // frame, and the replacement instance carries the new placement (anti-culling, bounds).
    // frameLastTouched may be shared with other entries relocated on other threads; every writer
    // stores the same value.
    reinterpret_cast<volatile uint32_t&>(blas.frameLastTouched) = m_now.frame;
    ri.recalculateBoundingBox(target, &bounds);
    ri.frameLastSeen = m_now.frame;
    entry.record.applied = target;
    return true;
  }

  void RetainedInstanceStore::noteMoved(const RtInstance& instance) {
    m_scene.device()->getCommon()->capturer()->setInstanceUpdateFlag(instance, GameCapturer::InstFlag::XformUpdate);
  }

  void RetainedInstanceStore::applyOriginChange(const Vector3& origin, FrameStats& stats) {
    if (sameBits(origin, m_origin)) {
      return;
    }
    m_origin = origin;
    // Every registered transform is rebased by the new origin. Awake entries pick it up when they
    // are evaluated; hidden ones when they are shown. Held entries get it here, applied to their
    // instances directly -- no full replay. Merged BLASes bake world-space transforms, so the
    // buckets holding them still rebuild; that is the one cost an origin move keeps.
    const uint32_t count = uint32_t(m_held.size());
    if (count == 0) {
      return;
    }
    // 0 = unchanged, 1 = moved, 2 = relocated but bit-identical, 3 = refused (full evaluation)
    std::vector<uint8_t>& status = m_originStatus;
    status.assign(count, 0);
    const auto rebaseOne = [&](uint32_t k) {
      Held& held = m_held[k];
      Entry& entry = m_entries[held.slot];
      entry.flags &= ~kMovePending;  // the absolute transform (with any pending move) is applied now
      const Matrix4 target = rebase(entry.absolute);
      if (sameBits(target, entry.record.applied)) {
        return;
      }
      bool moved = false;
      if (relocate(entry, *held.instance, *held.blas, *held.ri, target, entry.bounds, false, &moved)) {
        held.stampedFrame = m_now.frame;
        status[k] = moved ? 1 : 2;
      } else {
        status[k] = 3;
      }
    };
    if (count >= kParallelStampThreshold && RtxOptions::Retained::v2ParallelKeepAlive()) {
      // Per entry: its own instance, replacement instance and record; the shared geometry fields are
      // written with the same value by every writer; the scene-generation bump is done once below.
      const uint32_t chunks = (count + kStampChunk - 1) / kStampChunk;
      concurrency::parallel_for<uint32_t>(0, chunks, [&](uint32_t chunk) {
        const uint32_t end = std::min(count, (chunk + 1) * kStampChunk);
        for (uint32_t k = chunk * kStampChunk; k < end; ++k) {
          rebaseOne(k);
        }
      });
    } else {
      for (uint32_t k = 0; k < count; ++k) {
        rebaseOne(k);
      }
    }
    bool anyMoved = false;
    m_refused.clear();
    for (uint32_t k = 0; k < count; ++k) {
      if (status[k] == 1 || status[k] == 2) {
        ++stats.moved;
      }
      if (status[k] == 1) {
        anyMoved = true;
        noteMoved(*m_held[k].instance);
      } else if (status[k] == 3) {
        m_refused.push_back(m_held[k].slot);
      }
    }
    if (anyMoved) {
      m_scene.m_instanceManager.notifySceneChanged();
    }
    for (const uint32_t slot : m_refused) {
      release(slot);
      wake(slot);
    }
  }

  bool RetainedInstanceStore::fullCheck(const Entry& entry, Check& check) const {
    const auto fail = [&check](const char* why) {
      check.failure = why;
      return false;
    };
    if (!entry.record.valid) {
      return fail("has no replay record");
    }
    const ExternalDrawState& desc = entry.desc;
    if (desc.optionalParticleDesc.has_value() || desc.drawCall.getSkinningState().numBones > 0) {
      return fail("is a particle system or skinned");
    }
    check.ri = m_scene.m_drawCallTracker.findReplacementInstanceByIdentity(entry.record.identity);
    if (check.ri == nullptr) {
      return fail("has no replacement instance for its identity");
    }
    if (check.ri->prims.size() != 1) {
      return fail("has a replacement instance with other than one prim");
    }
    if (!check.ri->dirtyFlags.isClear()) {
      return fail("has a replacement instance with dirty flags");
    }
    check.instance = check.ri->prims[0].getInstance();
    if (check.instance == nullptr || check.instance->isMarkedForGC() ||
        check.instance->testCategoryFlags(InstanceCategories::ParticleEmitter)) {
      return fail("has no live instance");
    }
    check.blas = check.instance->getBlas();
    if (check.blas == nullptr) {
      return fail("has no geometry");
    }
    if (check.blas->frameLastUpdated == m_now.frame) {
      return fail("has geometry another draw updated this frame");
    }
    // A geometry refresh publishes a new submesh vector; the weak reference cannot be fooled by address reuse.
    check.submeshes = m_scene.m_pReplacer->accessExternalMesh(desc.mesh);
    if (check.submeshes == nullptr || check.submeshes->size() != 1 ||
        check.submeshes.get() != entry.record.submeshes.lock().get()) {
      return fail("found its mesh changed");
    }
    check.material = m_scene.resolveExternalDrawMaterial(desc, (*check.submeshes)[0]);
    check.materialHash = check.material != nullptr ? check.material->getHash() : kEmptyHash;
    if (check.materialHash != entry.record.materialHash ||
        (check.material != nullptr && check.material->getParticleSystemDesc() != nullptr) ||
        RtxOptions::shouldConvertToLight(check.materialHash)) {
      return fail("found its material changed or converted to a light");
    }
    if (fork_hooks::externalDrawMeshReplacement(*m_scene.m_pReplacer, reinterpret_cast<XXH64_hash_t>(desc.mesh)) != nullptr) {
      return fail("found a USD mesh replacement");
    }
    return true;
  }

  void RetainedInstanceStore::preserveNow(Entry& entry, const Check& check, const Matrix4& target) {
    // The preserve path for one entry: its instance is refreshed from the geometry's last full draw
    // state (camera type and geometry are the same for every draw of the mesh), not re-derived.
    const DrawCallState& input = check.blas->input;
    const ObjectPickingValue pickingValue = m_scene.externalObjectPickingValue(entry.desc.drawCall, *check.ri);
    preserveInstancesWithObjectPicking(
      check.ri->prims,
      pickingValue,
      [&](RtInstance& instance) {
        m_scene.preserveInstance(instance, &input);
        m_scene.m_instanceManager.preserveInstance(instance, input, nullptr);
      },
      [&] {
        // Not from `input`: another instance sharing the geometry may have drawn it last.
        m_scene.trackObjectPickingMeta(pickingValue, check.ri->externalPickingTextureHash);
      });
    check.ri->recalculateBoundingBox(target, &(*check.submeshes)[0].boundingBox);
    check.ri->frameLastSeen = m_now.frame;
    ++m_preservedForReport;
  }

  void RetainedInstanceStore::fullReplay(const Rc<DxvkContext>& ctx, Entry& entry, const Matrix4& target, FrameStats& stats) {
    const auto start = stats.timeFullReplays ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    // submitExternalDraw consumes its state (instancing list, per-submesh geometry), so it gets a copy.
    auto state = std::make_unique<ExternalDrawState>(entry.desc);
    state->retainedHandle = entry.handle;
    state->forceFullUpdate = (entry.flags & kForceFull) != 0;
    state->drawCall.modifyTransformData().objectToWorld = target;
    const bool redescribed = (entry.flags & kForceFull) != 0;
    entry.flags &= ~kForceFull;
    m_scene.submitExternalDraw(ctx, std::move(state));

    ReplayRecord& record = entry.record;
    record.identity = m_scene.m_lastExternalDrawIdentityHash;
    record.valid = record.identity != 0;
    record.applied = target;
    const auto submeshes = m_scene.m_pReplacer->accessExternalMesh(entry.desc.mesh);
    record.submeshes = submeshes;
    const MaterialData* material = (submeshes != nullptr && !submeshes->empty())
      ? m_scene.resolveExternalDrawMaterial(entry.desc, (*submeshes)[0]) : nullptr;
    record.materialHash = material != nullptr ? material->getHash() : kEmptyHash;
    if (RtxOptions::Retained::diagnosticFrames() != 0) {
      m_scene.noteRetainedFullReplay(entry.handle, entry.desc.mesh, record.identity, redescribed,
                                     (entry.flags & kVisible) != 0);
    }

    ++stats.full;
    if (stats.timeFullReplays) {
      stats.fullMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
  }

  bool RetainedInstanceStore::setPrimsHidden(ReplacementInstance& ri, bool hidden) {
    // Returns whether every prim may be shown by the external flag alone; otherwise (category-hidden,
    // sky, or not a plain instance) the caller also sends the entry down the full path, which
    // re-derives the hidden state.
    bool plain = true;
    for (const PrimInstance& prim : ri.prims) {
      RtInstance* instance = prim.getInstance();
      if (instance == nullptr) {
        continue;
      }
      m_scene.m_instanceManager.setInstanceHiddenExternally(*instance, hidden);
      if (!hidden && (fullPathHides(*instance) || instance->getBlas() == nullptr ||
                      !isPlainInstance(*instance, *instance->getBlas()))) {
        plain = false;
      }
    }
    return plain;
  }

  RetainedInstanceStore::Outcome RetainedInstanceStore::evaluate(const Rc<DxvkContext>& ctx, uint32_t slot, FrameStats& stats) {
    Entry& entry = m_entries[slot];
    entry.evaluatedReplay = m_replayCount;
    ++stats.awake;
    // rtx.retainedHiddenPolicy: 0 = hidden entries are not replayed (they age out), 2 = hidden entries
    // stay resident out of the TLAS, anything else = hidden entries are ray traced as if visible.
    const uint32_t policy = m_now.policy;
    const bool shown = (entry.flags & kVisible) != 0 || (policy != 0 && policy != 2);

    // Held: only a hide or a pending move can bring it here.
    if (entry.heldIndex != kNone) {
      const Held held = m_held[entry.heldIndex];
      if (!shown) {
        release(slot);
        ++stats.hidden;
        if (policy == 2) {
          m_scene.m_instanceManager.setInstanceHiddenExternally(*held.instance, true);
          entry.flags |= kHiddenFromTlas;
          addResident(slot, held.ri);
          return Outcome::Resident;
        }
        entry.flags |= kIdle;
        return Outcome::Idle;
      }
      if ((entry.flags & kMovePending) == 0) {
        return Outcome::Held;
      }
      entry.flags &= ~kMovePending;
      const Matrix4 target = rebase(entry.absolute);
      if (sameBits(target, entry.record.applied)) {
        return Outcome::Held;
      }
      bool moved = false;
      if (m_now.directOn && relocate(entry, *held.instance, *held.blas, *held.ri, target, entry.bounds, true, &moved)) {
        m_held[entry.heldIndex].stampedFrame = m_now.frame;
        if (moved) {
          noteMoved(*held.instance);
        }
        ++stats.moved;
        return Outcome::Held;
      }
      release(slot);
      // Not movable directly (mirroring change, world-space UI offset, ...): full evaluation below.
    }

    // Resident (hidden, out of the TLAS).
    if (entry.residentIndex != kNone) {
      if (!shown && policy == 2) {
        return Outcome::Resident;
      }
      ReplacementInstance* ri = m_resident[entry.residentIndex].ri;
      releaseResident(slot);
      if (!shown) {
        // The policy no longer keeps hidden entries resident: it ages out, still hidden.
        ++stats.hidden;
        entry.flags |= kIdle;
        return Outcome::Idle;
      }
      // Shown again: back into the TLAS directly; then the normal evaluation (its transform or
      // assets may have changed while it was hidden).
      if (!setPrimsHidden(*ri, false)) {
        entry.flags |= kForceFull;
      }
      entry.flags &= ~kHiddenFromTlas;
    }

    // Hidden, not resident.
    if (!shown) {
      ++stats.hidden;
      if (policy == 2 && entry.record.valid) {
        if (ReplacementInstance* ri = m_scene.m_drawCallTracker.findReplacementInstanceByIdentity(entry.record.identity)) {
          if ((entry.flags & kHiddenFromTlas) == 0) {
            setPrimsHidden(*ri, true);
            entry.flags |= kHiddenFromTlas;
          }
          addResident(slot, ri);
          return Outcome::Resident;
        }
      }
      entry.flags |= kIdle;
      return Outcome::Idle;
    }

    // Visible.
    if (entry.flags & kHiddenFromTlas) {
      // Hidden by an earlier evaluation and not resident (policy changed, replacement instance lost):
      // show whatever is left and re-derive everything once.
      if (entry.record.valid) {
        if (ReplacementInstance* ri = m_scene.m_drawCallTracker.findReplacementInstanceByIdentity(entry.record.identity)) {
          setPrimsHidden(*ri, false);
        }
      }
      entry.flags = uint16_t((entry.flags & ~kHiddenFromTlas) | kForceFull);
    }
    entry.flags &= ~kMovePending;
    const Matrix4 target = rebase(entry.absolute);
    if ((entry.flags & kForceFull) == 0 && m_now.preserveOn && !m_now.blocked) {
      Check check;
      if (fullCheck(entry, check)) {
        if (sameBits(target, entry.record.applied)) {
          preserveNow(entry, check, target);
          ++stats.validated;
          return tryHold(slot, check) ? Outcome::Held : Outcome::Awake;
        }
        bool moved = false;
        if (m_now.directOn && isPlainInstance(*check.instance, *check.blas) &&
            relocate(entry, *check.instance, *check.blas, *check.ri, target, (*check.submeshes)[0].boundingBox, true, &moved)) {
          if (moved) {
            noteMoved(*check.instance);
          }
          ++stats.moved;
          return tryHold(slot, check) ? Outcome::Held : Outcome::Awake;
        }
      }
    }
    fullReplay(ctx, entry, target, stats);
    return Outcome::Awake;
  }

  void RetainedInstanceStore::stampHeld(FrameStats& stats) {
    const uint32_t frame = m_now.frame;
    OpacityMicromapManager* omm = m_scene.m_opacityMicromapManager.get();
    // The quick stamp is what SceneManager::preserveInstance(instance, nullptr) and
    // InstanceManager::preserveInstance do for a held instance, provided that: it is already on
    // the preserve path (no first-frame history re-sync), its geometry has no previous-position
    // buffer left to release, and the instance events have nothing to do -- the only handlers are
    // the scene manager's (a no-op on the preserve path) and the OMM manager's (work only when
    // needsPreserveUpdate). Anything else gets the two calls themselves, on this thread.
    const bool quickStampOk = m_scene.m_instanceManager.getEventHandlerCount() == (omm != nullptr ? 2u : 1u);
    const auto needsFullStamp = [&](const Held& held) {
      return !quickStampOk ||
        !held.instance->surface.isPreservePath ||
        (held.blas->modifiedGeometryData.previousPositionBuffer.defined() && held.blas->frameLastTouched != frame) ||
        (omm != nullptr && OpacityMicromapManager::needsPreserveUpdate(*held.instance));
    };
    const auto quickStamp = [frame](Held& held) {
      held.instance->setFrameLastUpdated(frame);
      reinterpret_cast<volatile uint32_t&>(held.blas->frameLastTouched) = frame;
      held.instance->registerCamera(CameraType::Main, frame);
      held.ri->frameLastSeen = frame;
      held.stampedFrame = frame;
    };

    const uint32_t count = uint32_t(m_held.size());
    m_fullStamps.clear();
    if (count >= kParallelStampThreshold && quickStampOk && RtxOptions::Retained::v2ParallelKeepAlive()) {
      std::vector<uint8_t>& deferred = m_stampDeferred;
      deferred.assign(count, 0);
      const uint32_t chunks = (count + kStampChunk - 1) / kStampChunk;
      concurrency::parallel_for<uint32_t>(0, chunks, [&](uint32_t chunk) {
        const uint32_t end = std::min(count, (chunk + 1) * kStampChunk);
        for (uint32_t k = chunk * kStampChunk; k < end; ++k) {
          if (k + 4 < end) {
            const Held& ahead = m_held[k + 4];
            _mm_prefetch(reinterpret_cast<const char*>(ahead.instance), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&ahead.blas->frameLastTouched), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&ahead.ri->frameLastSeen), _MM_HINT_T0);
          }
          Held& held = m_held[k];
          if (held.stampedFrame == frame) {
            continue;
          }
          if (needsFullStamp(held)) {
            deferred[k] = 1;
            continue;
          }
          quickStamp(held);
        }
      });
      for (uint32_t k = 0; k < count; ++k) {
        if (deferred[k] != 0) {
          m_fullStamps.push_back(k);
        }
      }
    } else {
      for (uint32_t k = 0; k < count; ++k) {
        if (k + 4 < count) {
          const Held& ahead = m_held[k + 4];
          _mm_prefetch(reinterpret_cast<const char*>(ahead.instance), _MM_HINT_T0);
          _mm_prefetch(reinterpret_cast<const char*>(&ahead.blas->frameLastTouched), _MM_HINT_T0);
        }
        Held& held = m_held[k];
        if (held.stampedFrame == frame) {
          continue;
        }
        if (needsFullStamp(held)) {
          m_fullStamps.push_back(k);
        } else {
          quickStamp(held);
        }
      }
    }
    for (const uint32_t k : m_fullStamps) {
      if (k >= m_held.size() || m_held[k].stampedFrame == frame) {
        continue;
      }
      RtInstance& instance = *m_held[k].instance;
      BlasEntry& blas = *m_held[k].blas;
      m_scene.preserveInstance(instance, nullptr);
      m_scene.m_instanceManager.preserveInstance(instance, blas.input, nullptr);
      if (k < m_held.size() && m_held[k].instance == &instance) {
        m_held[k].ri->frameLastSeen = frame;
        m_held[k].stampedFrame = frame;
      }
    }
    stats.held = count;
    m_preservedForReport += count;

    // Hidden-resident entries: only their replacement instance is kept alive (their instances stay
    // out of the TLAS and are not stamped).
    for (const Resident& resident : m_resident) {
      resident.ri->frameLastSeen = frame;
    }
  }

  void RetainedInstanceStore::compactAwake() {
    size_t kept = 0;
    for (size_t i = 0; i < m_awake.size(); ++i) {
      const uint32_t slot = m_awake[i];
      Entry& entry = m_entries[slot];
      if (entry.listMark == m_replayCount) {
        continue;  // second position of an entry evaluated twice
      }
      entry.listMark = m_replayCount;
      const bool stays = (entry.flags & kLive) != 0 &&
        entry.heldIndex == kNone && entry.residentIndex == kNone && (entry.flags & kIdle) == 0;
      if (stays) {
        m_awake[kept++] = slot;
      } else {
        entry.flags &= ~kInAwakeList;
      }
    }
    m_awake.resize(kept);
  }

  const char* RetainedInstanceStore::checkHeld(const Entry& entry, const Held& held) const {
    Check check;
    if (!fullCheck(entry, check)) {
      return check.failure;
    }
    if (check.ri != held.ri || check.instance != held.instance || check.blas != held.blas) {
      return "found a different replacement instance, instance or geometry than the one it holds";
    }
    if (!isPlainInstance(*check.instance, *check.blas)) {
      return "is no longer an instance that may be held";
    }
    // The transform the full path would give the instance: host transform minus the scene origin
    // (plus the world-space UI background offset where it applies).
    Matrix4 expected = rebase(entry.absolute);
    if (!sameBits(expected, entry.record.applied)) {
      return "has a host transform its instance was never given";
    }
    if (m_now.uiOffset != 0.f && check.instance->testCategoryFlags(InstanceCategories::WorldMatte)) {
      expected[3] += expected[2] * m_now.uiOffset;
    }
    if (!sameBits(expected, check.instance->surface.objectToWorld)) {
      return "has an instance transform that differs from the host transform minus the scene origin";
    }
    if (check.instance->isHidden() && !fullPathHides(*check.instance)) {
      return "has a hidden instance although the entry is shown";
    }
    if (check.material != nullptr) {
      const MaterialData* material = check.material;
      const std::shared_ptr<MaterialData> keepAlive = fork_hooks::externalDrawMaterialReplacement(*m_scene.m_pReplacer, material);
      if (material != nullptr && check.instance->getMaterialHash() != material->getHash()) {
        return "has an instance material that differs from the entry's material";
      }
    }
    return nullptr;
  }

  void RetainedInstanceStore::selfCheck(FrameStats& stats) {
    const uint32_t every = RtxOptions::Retained::v2SelfCheckFrames();
    if (every == 0 || m_now.frame % every != 0) {
      return;
    }
    const uint32_t sample = std::max(1u, RtxOptions::Retained::v2SelfCheckSample());
    std::vector<uint32_t> suspects;
    const auto report = [&](const Entry& entry, const char* why) {
      ++m_checkMismatches;
      Logger::warn(str::format("[RetainedV2Check] frame ", m_now.frame, " handle ", entry.handle, ": ", why,
        " (the full path would differ; it takes the full path now). Mismatches so far: ", m_checkMismatches));
    };

    const uint32_t heldCount = uint32_t(m_held.size());
    const uint32_t heldSample = std::min(sample, heldCount);
    for (uint32_t n = 0; n < heldSample; ++n) {
      const Held held = m_held[(m_checkCursor + n) % heldCount];
      const Entry& entry = m_entries[held.slot];
      if (entry.flags & kMovePending) {
        continue;  // moved by the host; this replay applies it
      }
      ++stats.checked;
      if (const char* why = checkHeld(entry, held)) {
        report(entry, why);
        suspects.push_back(held.slot);
      }
    }
    m_checkCursor = heldCount != 0 ? (m_checkCursor + heldSample) % heldCount : 0;

    const uint32_t residentCount = uint32_t(m_resident.size());
    const uint32_t residentSample = std::min(std::max(1u, sample / 4), residentCount);
    for (uint32_t n = 0; n < residentSample; ++n) {
      const Resident resident = m_resident[(m_residentCheckCursor + n) % residentCount];
      const Entry& entry = m_entries[resident.slot];
      ++stats.checked;
      const char* why = nullptr;
      if (!entry.record.valid ||
          m_scene.m_drawCallTracker.findReplacementInstanceByIdentity(entry.record.identity) != resident.ri) {
        why = "is resident with a replacement instance its identity no longer finds";
      } else {
        for (const PrimInstance& prim : resident.ri->prims) {
          if (const RtInstance* instance = prim.getInstance()) {
            if (!instance->isHidden()) {
              why = "is hidden but its instance is in the TLAS";
              break;
            }
          }
        }
      }
      if (why != nullptr) {
        report(entry, why);
        suspects.push_back(resident.slot);
      }
    }
    m_residentCheckCursor = residentCount != 0 ? (m_residentCheckCursor + residentSample) % residentCount : 0;

    for (const uint32_t slot : suspects) {
      Entry& entry = m_entries[slot];
      if (entry.heldIndex != kNone) {
        release(slot);
        entry.flags |= kForceFull;
      } else {
        releaseResident(slot);
      }
      wake(slot);
    }
    if (suspects.empty() && stats.checked != 0) {
      ONCE(Logger::info("[RetainedV2Check] sampled retained-v2 entries match the full path. Further passes are not logged."));
    }
  }

  void RetainedInstanceStore::replay(const Rc<DxvkContext>& ctx, const Vector3& origin, FrameStats& stats) {
    ScopedCpuProfileZone();
    ++m_replayCount;
    readFrameInputs();
    applyGlobalInvalidations();
    consumeChangedAssets();
    applyOriginChange(origin, stats);
    selfCheck(stats);

    // The changed entries (and those that can never be held). Entries appended while this runs --
    // released by events of this replay -- are evaluated in this replay too.
    m_inReplay = true;
    m_requeueBudget = uint32_t(m_awake.size()) + 4096u;
    for (size_t i = 0; i < m_awake.size(); ++i) {
      const uint32_t slot = m_awake[i];
      Entry& entry = m_entries[slot];
      if ((entry.flags & kLive) == 0) {
        continue;
      }
      if (entry.evaluatedReplay == m_replayCount && entry.heldIndex != kNone && (entry.flags & kMovePending) == 0) {
        continue;  // duplicate position of an entry held since
      }
      evaluate(ctx, slot, stats);
    }
    m_inReplay = false;

    stampHeld(stats);
    compactAwake();

    stats.released = m_released;
    m_released = 0;

    // [ExternalPreserve] accounting (held and preserved entries count as preserved draws).
    if (m_preservedForReport != 0) {
      m_scene.m_externalDrawsPreserved += m_preservedForReport - 1;
      m_scene.reportExternalPreserveOutcome(true);
      m_preservedForReport = 0;
    }
  }

}  // namespace dxvk
