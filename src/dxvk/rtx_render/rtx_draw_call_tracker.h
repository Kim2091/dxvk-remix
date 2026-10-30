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

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtx_types.h"
#include "rtx_fork_d3d9_identity.h"
#include "../util/util_spatial_map.h"

namespace dxvk {

class DxvkDevice;
class RtCamera;
class RayPortalManager;
namespace translator {
  struct LifetimeFrame;
}

// Tracks draw calls across frames by mapping each draw call to a ReplacementInstance.
// Uses a two-level hash lookup (identity hash for L1, tracking hash + spatial proximity
// for L2) to match the current frame's draw calls against previous frames.
class DrawCallTracker {
public:
  DrawCallTracker(DxvkDevice* device);
  ~DrawCallTracker() = default;

  DrawCallTracker(const DrawCallTracker&) = delete;
  DrawCallTracker& operator=(const DrawCallTracker&) = delete;

  // Removes replacement instances whose spatialMapHash matches (same bucket key as
  // m_assetSpatialMaps). Used when a draw source tied to that bucket is invalidated.
  void removeReplacementInstancesWithSpatialMapHash(XXH64_hash_t spatialMapHash);

  // Find or create a ReplacementInstance from a DrawCallState.
  // L1 identity includes legacy material hash from DrawCallState plus the hash of
  // overrideMaterialData when non-null (terrain bake, particles, etc.); full merged
  // render MaterialData is not required for tracking.
  // Portal-aware matching for ViewModel draw calls is performed using the RayPortalManager.
  ReplacementInstance* findOrCreateReplacementInstance(
    const ReplacementInstance::LookupKey& key);

  // Wrapper for the above function that constructs the LookupKey from the DrawCallState.
  // hasReplacements: the draw has USD mesh replacements (rtx.d3d9Identity.traceFrames records it).
  ReplacementInstance* findOrCreateReplacementInstance(
      const DrawCallState& drawCallState,
      const RayPortalManager& rayPortalManager,
      const MaterialData* overrideMaterialData,
      bool hasReplacements = false);

  // The lookup once its key is formed: L1 / L2 / L3. Shared by the draw path above and the identity-trace replay
  // (rtx_fork_d3d9_identity_trace.h). identityLookup: the classifier inputs ([D3d9Identity]), or nullptr.
  ReplacementInstance* findOrCreateKeyed(
      const ReplacementInstance::LookupKey& key,
      const D3d9Identity::Lookup* identityLookup);

  // Identity-trace replay: frame ids come from *frame instead of the device (nullptr = the device's).
  void setFrameOverride(const uint32_t* frame) { m_frameOverride = frame; }

  // Look up an existing ReplacementInstance by L1 identity hash only. Returns
  // nullptr if no RI exists for this key. Unlike findOrCreateReplacementInstance,
  // this never allocates and never runs the L2 spatial search -- intended
  // for "is there stale state to clean up?" probes that must not pollute the
  // tracker with empty RIs (e.g. game lights when their replacement was just
  // toggled off).
  // outMatchKind is always set to how the instance was resolved.
  ReplacementInstance* findReplacementInstanceByIdentity(XXH64_hash_t identityHash);

  // Garbage collect expired ReplacementInstances. Object anti-culling keeps mesh RIs alive
  // if their bounding box intersects the camera frustum. Light content uses the camera's
  // light anti-culling frustum (wider FOV) and extended lifetime.
  void garbageCollectReplacementInstances(
      RtCamera& camera,
      bool isAntiCullingSupported,
      bool parallelScan = false,   // rtx.parallelScenePrepScans
      uint32_t keptCutFrame = ~0u); // SceneManager::m_lastKeptCutFrame (rtx.scene.d3d9CutAsEvent / noImplicitClears)
  // The same once its inputs are read (lifetime.currentFrame and forceCollect are filled in here); the identity-trace
  // replay calls it with a LifetimeFrame from the trace.
  void collectReplacementInstances(const translator::LifetimeFrame& lifetimeIn, bool parallelScan = false);

  size_t getReplacementInstanceCount() const { return m_replacementInstances.size(); }

  void clear();

  // Rebuild all spatial maps with a new cell size.
  // Call when uniqueObjectDistance changes.
  void rebuildSpatialMaps(float cellSize);

  const std::vector<std::unique_ptr<ReplacementInstance>>& getReplacementInstances() const {
    return m_replacementInstances;
  }

  // Compute a hash that uniquely identifies a draw call for L1 matching.
  // When overrideMaterialData is non-null, its hash is mixed in (e.g. terrain bake, particle material).
  static XXH64_hash_t computeIdentityHash(
      const DrawCallState& drawCallState,
      const MaterialData* overrideMaterialData = nullptr);

  // rtx.retainedSteadyFastPath: changes whenever a ReplacementInstance cached by a retained
  // entry (ReplacementInstance::retainedSteady) is destroyed or re-keyed, or on clear().
  uint64_t getRetainedSteadyEpoch() const { return m_retainedSteadyEpoch; }
  // rtx.retainedPreciseInvalidation: the same events, split into clear() (invalidates everything;
  // this epoch) and the individual replacement instances destroyed or re-keyed since the last
  // take (addresses only, never dereferenced: they may be freed).
  uint64_t getRetainedSteadyClearEpoch() const { return m_retainedSteadyClearEpoch; }
  size_t getRetainedSteadyInvalidationCount() const { return m_retainedSteadyInvalidations.size(); }
  const std::vector<const ReplacementInstance*>& getRetainedSteadyInvalidations() const { return m_retainedSteadyInvalidations; }
  void clearRetainedSteadyInvalidations() { m_retainedSteadyInvalidations.clear(); }

  // rtx.retained.useV2: RetainedInstanceStore is told synchronously, while the replacement instance
  // is still alive, when one carrying a nonzero retainedV2Ticket is destroyed or re-keyed (the
  // ticket is cleared first). clear() destroys every replacement instance without telling anyone;
  // it bumps getClearGeneration() instead.
  using RetainedV2Listener = void (*)(void* owner, uint64_t ticket);
  void setRetainedV2Listener(RetainedV2Listener listener, void* owner) {
    m_retainedV2Listener = listener;
    m_retainedV2ListenerOwner = owner;
  }
  uint64_t getClearGeneration() const { return m_clearGeneration; }

private:
  RetainedV2Listener m_retainedV2Listener = nullptr;
  void* m_retainedV2ListenerOwner = nullptr;
  uint64_t m_clearGeneration = 0;
  void reportToRetainedV2(ReplacementInstance* ri) {
    if (ri != nullptr && ri->retainedV2Ticket != 0) {
      const uint64_t ticket = ri->retainedV2Ticket;
      ri->retainedV2Ticket = 0;
      if (m_retainedV2Listener != nullptr) {
        m_retainedV2Listener(m_retainedV2ListenerOwner, ticket);
      }
    }
  }

  uint64_t m_retainedSteadyEpoch = 0;
  uint64_t m_retainedSteadyClearEpoch = 0;
  std::vector<const ReplacementInstance*> m_retainedSteadyInvalidations;
  std::vector<uint8_t> m_gcStaleScratch; // rtx.parallelScenePrepScans
  void noteRekeyOrDestroy(const ReplacementInstance* ri) {
    if (ri != nullptr && ri->retainedSteady) {
      ++m_retainedSteadyEpoch;
      if (m_retainedSteadyInvalidations.size() < 65536) {
        m_retainedSteadyInvalidations.push_back(ri);
      } else {
        // Nobody taking them: treat as a clear.
        m_retainedSteadyInvalidations.clear();
        ++m_retainedSteadyClearEpoch;
      }
    }
  }
  // Cleans up a ReplacementInstance's map/index entries and prims. Does NOT remove
  // the entry from m_replacementInstances -- callers must handle that separately.
  // reason: rtx.d3d9Identity.log only.
  void destroyReplacementInstance(ReplacementInstance* replacementInstance,
                                  D3d9Identity::DestroyReason reason = D3d9Identity::DestroyReason::Gc);

  DxvkDevice* m_device;
  const uint32_t* m_frameOverride = nullptr;  // setFrameOverride
  uint32_t frameId() const;

  std::unordered_map<XXH64_hash_t, ReplacementInstance*> m_identityHashMap;

  using ReplacementSpatialMap = SpatialMap<ReplacementInstance>;
  std::unordered_map<XXH64_hash_t, ReplacementSpatialMap> m_assetSpatialMaps;

  // Attempts to match a newly created ReplacementInstance through ray portals.
  // If a match is found, destroys the new RI and returns the reassociated match.
  // Returns nullptr if no portal match is found.
  ReplacementInstance* tryPortalMatch(
      ReplacementInstance* newInstance,
      const ReplacementInstance::LookupKey& key,
      const RayPortalManager& rayPortalManager);

  // Erases an entry from a spatial map bucket and removes the bucket if empty.
  void eraseFromSpatialMap(
      std::unordered_map<XXH64_hash_t, ReplacementSpatialMap>& mapOfMaps,
      XXH64_hash_t bucketKey,
      XXH64_hash_t transformHash,
      const ReplacementInstance* data);

  // Reassociates an existing ReplacementInstance with a new draw call identity.
  // Updates position fields and optionally moves within spatial maps.
  // moveInAssetMap: if non-null, the RI is moved within this asset spatial map.
  ReplacementInstance* reassociateMatch(
      ReplacementInstance* match,
      const ReplacementInstance::LookupKey& key,
      ReplacementSpatialMap* moveInAssetMap);

  std::vector<std::unique_ptr<ReplacementInstance>> m_replacementInstances;
  uint32_t m_nextReplacementInstanceId = 0;

  // rtx.d3d9Identity.log: the classifier inputs of the D3D9 draw being looked up (nullptr while the log is
  // off and for lookups without a DrawCallState).
  const D3d9Identity::Lookup* m_identityLookup = nullptr;

};

}  // namespace dxvk
