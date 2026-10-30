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
#include <ppl.h>
#include <intrin.h>
#include <cmath>
#include <string>
#include "rtx_draw_call_tracker.h"
#include "translator/rtx_translator_lifetime.h"
#include "rtx_options.h"
#include "rtx_ray_portal_manager.h"
#include "rtx_intersection_test.h"
#include "rtx_fork_d3d9_alpha.h"
#include "rtx_fork_d3d9_identity.h"
#include "rtx_fork_d3d9_identity_trace.h"
#include "rtx_fork_fnv_material.h"
#include "dxvk_device.h"
#include "dxvk_perf_counters.h"
#include "../util/util_struct_hash.h"


namespace dxvk {

  // Compute which lookup-key fields differ from the RI's cached values and write
  // the lookup-drift bits to ri->dirtyFlags (dynamic-feature bits are preserved).
  // Caller is expected to invoke this BEFORE overwriting the RI's cached fields.
  static void computeDirtyFlags(
      ReplacementInstance* ri, const ReplacementInstance::LookupKey& key) {
    ri->dirtyFlags.clr(ReplacementInstance::kLookupDriftMask);
    // This function is only called when the full identity hash doesn't match, so something must have changed.
    if (std::memcmp(&ri->objectToWorld, &key.transform, sizeof(Matrix4)) != 0) {
      ri->dirtyFlags.set(ReplacementInstance::DirtyFlag::Transform);
    }
    if (ri->vertexPositionHash != key.vertexPositionHash) {
      ri->dirtyFlags.set(ReplacementInstance::DirtyFlag::VertexPosHash);
    }
    if (ri->materialHash != key.materialHash) {
      ri->dirtyFlags.set(ReplacementInstance::DirtyFlag::MaterialHash);
    }
    // textureTransform/texgenMode aren't fed into a dedicated dirty bit yet —
    // surface state on the preserve path is reused as-is, so any drift here (terrain
    // baker rewriting view→cascade space, free-camera view-space texgen) needs to
    // force the dynamic path until the preserve path can refresh those fields. Bundle
    // into the catch-all Other bit so usePreservePath's isClear() check fails.
    if (std::memcmp(&ri->textureTransform, &key.textureTransform, sizeof(Matrix4)) != 0 ||
        ri->texgenMode != key.texgenMode) {
      ri->dirtyFlags.set(ReplacementInstance::DirtyFlag::Other);
    }
    if ((ri->dirtyFlags & ReplacementInstance::kLookupDriftMask).isClear()) {
      ri->dirtyFlags.set(ReplacementInstance::DirtyFlag::Other);
    }
  }

  // A lookup whose material (a D3D9 draw's albedo image hash) is listed in
  // rtx.fnvProtocol.logMaterialTextures is logged whenever its exact identity match fails, with how it compares to the
  // instance it was matched to (or, for a new instance, to the nearest instance of the same topology, claimed this
  // frame or not). Instance churn on a picked surface -- a new instance every few frames, or one instance handed from
  // placement to placement -- then reads straight from the log: which of transform / vertex positions / material
  // changed, and how far the match moved. At most 20000 lines per session.
  static bool trackerWatched(const ReplacementInstance::LookupKey& key) {
    const fast_unordered_set& watched = FnvMaterialProtocol::logMaterialTextures();
    return !watched.empty() && lookupHash(watched, key.materialHash);
  }

  static void logTrackerWatch(const char* outcome, const ReplacementInstance::LookupKey& key, const ReplacementInstance* other,
                              float distSqr, uint32_t frameId) {
    static uint32_t s_lines = 0;
    if (s_lines >= 20000) {
      return;
    }
    ++s_lines;
    std::string otherDesc = " other=none";
    if (other != nullptr) {
      const Vector4& t = other->objectToWorld[3];
      otherDesc = str::format(" other: ri=", other->id, " created=", other->frameCreated, " lastSeen=", other->frameLastSeen,
                              " dist=", distSqr >= 0.f ? std::sqrt(distSqr) : -1.f,
                              " sameXform=", std::memcmp(&other->objectToWorld, &key.transform, sizeof(Matrix4)) == 0 ? 1 : 0,
                              " sameVtx=", other->vertexPositionHash == key.vertexPositionHash ? 1 : 0,
                              " sameMat=", other->materialHash == key.materialHash ? 1 : 0,
                              " sameIdentity=", other->identityHash == key.identityHash ? 1 : 0,
                              " sameTexXform=", (std::memcmp(&other->textureTransform, &key.textureTransform, sizeof(Matrix4)) == 0 &&
                                                 other->texgenMode == key.texgenMode) ? 1 : 0,
                              " pos=(", t.x, ", ", t.y, ", ", t.z, ")",
                              " vtx=0x", std::hex, other->vertexPositionHash, std::dec);
    }
    const Vector4& p = key.transform[3];
    Logger::info(str::format("[TrackerWatch] frame=", frameId, " ", outcome,
                             " material=0x", std::hex, key.materialHash, " topology=0x", key.spatialMapHash,
                             " identity=0x", key.identityHash, " vtx=0x", key.vertexPositionHash, std::dec,
                             " pos=(", p.x, ", ", p.y, ", ", p.z, ") centroid=(", key.worldPos.x, ", ", key.worldPos.y, ", ",
                             key.worldPos.z, ")", otherDesc));
  }

  DrawCallTracker::DrawCallTracker(DxvkDevice* device)
    : m_device(device) {
  }

  uint32_t DrawCallTracker::frameId() const {
    return m_frameOverride != nullptr ? *m_frameOverride : m_device->getCurrentFrameId();
  }

  XXH64_hash_t DrawCallTracker::computeIdentityHash(
      const DrawCallState& drawCallState,
      const MaterialData* overrideMaterialData) {
    struct IdentityHashData{
      XXH64_hash_t geoHash;
      XXH64_hash_t matHash;
      XXH64_hash_t boneHash;
      XXH64_hash_t overrideMaterialHash;
      Matrix4 xform;
      Matrix4 textureTransform;
      uint32_t categories;
      // Packed: cameraType (bits 0-7), texgenMode (bits 8-15), isUsingRaytracedRenderTarget (bit 16).
      uint32_t miscFlags;
    };

    // {} value-initializes the entire object.
    IdentityHashData data{};

    data.geoHash = drawCallState.getGeometryData().getHashForRule(rules::FullGeometryHash);
    data.matHash = drawCallState.getMaterialData().getHash();
    data.overrideMaterialHash = overrideMaterialData != nullptr
        ? overrideMaterialData->getHash()
        : kEmptyHash;
    data.xform = drawCallState.getTransformData().objectToWorld;
    data.categories = drawCallState.getCategoryFlags().raw();
    data.boneHash = drawCallState.getSkinningState().boneHash;
    data.textureTransform = drawCallState.getTransformData().textureTransform;

    // Note: these could be packed tighter, if we have more small values that need to be included.
    static_assert(CameraType::Count <= (1u << 8), "CameraType no longer fits in the 8 bits allocated to it in IdentityHashData::miscFlags");
    static_assert(static_cast<uint32_t>(TexGenMode::Count) <= (1u << 8), "TexGenMode no longer fits in the 8 bits allocated to it in IdentityHashData::miscFlags");
    data.miscFlags = static_cast<uint32_t>(drawCallState.cameraType)
        | (static_cast<uint32_t>(drawCallState.getTransformData().texgenMode) << 8)
        | (drawCallState.isUsingRaytracedRenderTarget ? (1u << 16) : 0u);

    return hashStructByMemory<IdentityHashData,
        &IdentityHashData::geoHash,
        &IdentityHashData::matHash,
        &IdentityHashData::boneHash,
        &IdentityHashData::overrideMaterialHash,
        &IdentityHashData::xform,
        &IdentityHashData::textureTransform,
        &IdentityHashData::categories,
        &IdentityHashData::miscFlags>(data);
  }

  void DrawCallTracker::eraseFromSpatialMap(
      std::unordered_map<XXH64_hash_t, ReplacementSpatialMap>& mapOfMaps,
      XXH64_hash_t bucketKey,
      XXH64_hash_t transformHash,
      const ReplacementInstance* data) {
    auto iter = mapOfMaps.find(bucketKey);
    if (iter != mapOfMaps.end()) {
      iter->second.erase(transformHash, data);
      if (iter->second.size() == 0) {
        mapOfMaps.erase(iter);
      }
    }
  }

  ReplacementInstance* DrawCallTracker::reassociateMatch(
      ReplacementInstance* match,
      const ReplacementInstance::LookupKey& key,
      ReplacementSpatialMap* moveInAssetMap) {
    noteRekeyOrDestroy(match);
    reportToRetainedV2(match);
    m_identityHashMap.erase(match->identityHash);
    match->identityHash = key.identityHash;
    match->vertexPositionHash = key.vertexPositionHash;
    match->materialHash = key.materialHash;
    match->centroid = key.worldPos;

    if (moveInAssetMap) {
      match->spatialCacheTransformHash = moveInAssetMap->move(
          match->spatialCacheTransformHash, key.worldPos, key.transform, match);
    }

    m_identityHashMap[key.identityHash] = match;
    return match;
  }

  void DrawCallTracker::removeReplacementInstancesWithSpatialMapHash(XXH64_hash_t spatialMapKey) {
    for (size_t i = 0; i < m_replacementInstances.size();) {
      ReplacementInstance* replacementInstance = m_replacementInstances[i].get();
      if (replacementInstance->spatialMapHash == spatialMapKey) {
        destroyReplacementInstance(replacementInstance, D3d9Identity::DestroyReason::Replacements);
        std::swap(m_replacementInstances[i], m_replacementInstances.back());
        m_replacementInstances.pop_back();
        continue;
      }
      ++i;
    }
  }

  ReplacementInstance* DrawCallTracker::findOrCreateReplacementInstance(
      const ReplacementInstance::LookupKey& key) {
    ScopedCpuProfileZone();
    const uint32_t currentFrameId = frameId();

    // Level 1: exact identity match.
    // Draw calls with the same identity hash (same geometry, material, transform) return
    // the same ReplacementInstance even if already seen this frame. This handles two-pass rendering where
    // the game draws the same mesh twice (e.g., base pass + overlay). The second pass
    // merges into the same instance via mergeInstanceHeuristics in updateInstance.
    auto exactMatchIter = m_identityHashMap.find(key.identityHash);
    if (exactMatchIter != m_identityHashMap.end()) {
      // identityHash includes transform/material/vertex hashes, so all key fields
      // match this RI's cached values by construction. Nothing changed since last
      // submission of this identity.
      //
      // Only clear lookup-drift bits on the first lookup of a new frame. A second lookup
      // within the same frame (two-pass rendering) must not clobber flags the L2
      // path set when first matching this RI for the current frame. Dynamic-feature
      // bits persist until the dynamic path runs.
      ReplacementInstance* match = exactMatchIter->second;
      if (match->frameLastSeen != currentFrameId) {
        match->dirtyFlags.clr(ReplacementInstance::kLookupDriftMask);
      }
      perf::countD3D9(perf::D3D9Ctr::TrackerExact);
      if (m_identityLookup != nullptr) {
        D3d9Identity::noteExact(*match, *m_identityLookup, currentFrameId);
      }
      return match;
    }

    // Level 2: tracking hash (topological, stable for animated geometry) + spatial proximity
    const float uniqueObjectDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();
    const float spatialMapCellSize = RtxOptions::uniqueObjectDistance() * 2.f;

    auto l2Filter = [&](const ReplacementInstance* candidate) {
      return candidate->frameLastSeen != currentFrameId &&
             candidate->materialHash == key.materialHash;
    };

    auto spatialMapIter = m_assetSpatialMaps.find(key.spatialMapHash);
    if (spatialMapIter != m_assetSpatialMaps.end()) {
      // Try exact transform + vertex position hash match first
      ReplacementInstance* exactTransformMatch = nullptr;
      spatialMapIter->second.forEachAtTransform(key.transform, [&](const ReplacementInstance* candidate) {
        if (candidate->vertexPositionHash == key.vertexPositionHash && l2Filter(candidate)) {
          exactTransformMatch = const_cast<ReplacementInstance*>(candidate);
          return true;
        }
        return false;
      });
      if (exactTransformMatch != nullptr) {
        // Compute diff before reassociation. Transform/vertex/spatialMap match
        // by construction here; only material can diverge (l2Filter requires
        // material match, so usually nothing differs at this point).
        if (trackerWatched(key)) {
          logTrackerWatch("l2-same-transform", key, exactTransformMatch, 0.f, currentFrameId);
        }
        computeDirtyFlags(exactTransformMatch, key);
        if (m_identityLookup != nullptr) {
          D3d9Identity::noteL2Match(*exactTransformMatch, *m_identityLookup, currentFrameId, false, -1.f, uniqueObjectDistanceSqr);
        }
        noteRekeyOrDestroy(exactTransformMatch);
        reportToRetainedV2(exactTransformMatch);
        m_identityHashMap.erase(exactTransformMatch->identityHash);
        exactTransformMatch->identityHash = key.identityHash;
        m_identityHashMap[key.identityHash] = exactTransformMatch;
        perf::countD3D9(perf::D3D9Ctr::TrackerTransform);
        return exactTransformMatch;
      }

      // Spatial nearest-neighbor search
      float nearestDistSqr = FLT_MAX;
      const ReplacementInstance* nearestMatch = spatialMapIter->second.getNearestData(
        key.worldPos, uniqueObjectDistanceSqr, nearestDistSqr, l2Filter);

      if (nearestMatch != nullptr) {
        // Compute diff before reassociation overwrites the cached fields. The
        // transform differs (otherwise we would have hit the exact-transform
        // branch above); other fields may also have changed.
        ReplacementInstance* match = const_cast<ReplacementInstance*>(nearestMatch);
        if (trackerWatched(key)) {
          logTrackerWatch("l2-moved", key, match, nearestDistSqr, currentFrameId);
        }
        computeDirtyFlags(match, key);
        if (m_identityLookup != nullptr) {
          D3d9Identity::noteL2Match(*match, *m_identityLookup, currentFrameId, true, nearestDistSqr, uniqueObjectDistanceSqr);
        }
        perf::countD3D9(perf::D3D9Ctr::TrackerSpatial);
        return reassociateMatch(match, key, &spatialMapIter->second);
      }

    }

    // [D3d9Alpha]: why no previous instance was matched (detailed logging only).
    if (D3d9Alpha::statsActive()) {
      D3d9Alpha::TrackerCreateReason reason = D3d9Alpha::TrackerCreateReason::NewTopology;
      if (spatialMapIter != m_assetSpatialMaps.end()) {
        float distSqr = FLT_MAX;
        const auto unmatched = [&](const ReplacementInstance* candidate) {
          return candidate->frameLastSeen != currentFrameId;
        };
        reason = spatialMapIter->second.getNearestData(key.worldPos, uniqueObjectDistanceSqr, distSqr, unmatched) != nullptr
          ? D3d9Alpha::TrackerCreateReason::MaterialMismatch
          : D3d9Alpha::TrackerCreateReason::NoneInRange;
      }
      D3d9Alpha::noteTrackerCreation(reason, key.spatialMapHash, key.materialHash);
    }

    // rtx.d3d9Identity.log: the nearest instance of this topology not matched yet this frame, any material.
    // The spatial map searches the 2x2x2 cells around the position (cell = 2x uniqueObjectDistance), so distances
    // beyond 1x are only partly covered.
    bool identityTopologyKnown = false;
    bool identityMaterialBlocked = false;
    float identityNearestDistSqr = -1.f;
    if (m_identityLookup != nullptr && spatialMapIter != m_assetSpatialMaps.end()) {
      identityTopologyKnown = true;
      float distSqr = FLT_MAX;
      const auto unmatched = [&](const ReplacementInstance* candidate) {
        return candidate->frameLastSeen != currentFrameId;
      };
      if (spatialMapIter->second.getNearestData(key.worldPos, 16.f * uniqueObjectDistanceSqr, distSqr, unmatched) != nullptr) {
        identityNearestDistSqr = distSqr;
        identityMaterialBlocked = distSqr <= uniqueObjectDistanceSqr;
      }
    }

    // A new instance for a watched material, against the nearest instance of its topology
    // whether or not a draw already claimed it this frame (lastSeen = this frame means it did).
    if (trackerWatched(key)) {
      const ReplacementInstance* nearestAny = nullptr;
      float nearestAnyDistSqr = -1.f;
      if (spatialMapIter != m_assetSpatialMaps.end()) {
        float distSqr = FLT_MAX;
        nearestAny = spatialMapIter->second.getNearestData(key.worldPos, 16.f * uniqueObjectDistanceSqr, distSqr,
                                                           [](const ReplacementInstance*) { return true; });
        if (nearestAny != nullptr) {
          nearestAnyDistSqr = distSqr;
        }
      }
      logTrackerWatch(spatialMapIter == m_assetSpatialMaps.end() ? "new (topology unseen)" : "new", key, nearestAny,
                      nearestAnyDistSqr, currentFrameId);
    }

    // Level 3: no match — create a new ReplacementInstance.
    auto newReplacementInstance = std::make_unique<ReplacementInstance>(
        key, m_nextReplacementInstanceId++, currentFrameId);
    ReplacementInstance* replacementInstance = newReplacementInstance.get();
    if (m_identityLookup != nullptr) {
      D3d9Identity::noteCreated(*replacementInstance, *m_identityLookup, currentFrameId, identityTopologyKnown,
                                identityMaterialBlocked, identityNearestDistSqr, uniqueObjectDistanceSqr);
    }

    m_identityHashMap[key.identityHash] = replacementInstance;

    auto [mapIter, inserted] = m_assetSpatialMaps.try_emplace(key.spatialMapHash, spatialMapCellSize);
    replacementInstance->spatialCacheTransformHash =
        mapIter->second.insert(key.worldPos, key.transform, replacementInstance);

    m_replacementInstances.push_back(std::move(newReplacementInstance));

    perf::countD3D9(perf::D3D9Ctr::TrackerNew);
    return replacementInstance;
  }

  ReplacementInstance* DrawCallTracker::findOrCreateReplacementInstance(
      const DrawCallState& drawCallState,
      const RayPortalManager& rayPortalManager,
      const MaterialData* overrideMaterialData,
      bool hasReplacements) {
    // Tracking hash uses topology only (indices + geometry descriptor), matching how the
    // baseline grouped instances by BlasEntry (topological hash). The material hash is used
    // as a filter in the spatial search, not as part of the key — this allows matching
    // even when the material changes between frames (LOD, texture animation).
    const auto& hashes = drawCallState.getGeometryData().hashes;
    const Matrix4& objectToWorld = drawCallState.getTransformData().objectToWorld;
    const XXH64_hash_t topologyHash = hashes.getHashForRule<rules::TopologicalHash>();

    const ReplacementInstance::LookupKey key {
      computeIdentityHash(drawCallState, overrideMaterialData),
      topologyHash,
      drawCallState.getMaterialData().getHash(),
      hashes[HashComponents::VertexPosition],
      drawCallState.getGeometryData().boundingBox.getTransformedCentroid(objectToWorld),
      objectToWorld,
      drawCallState.getTransformData().textureTransform,
      drawCallState.getTransformData().texgenMode
    };

    // rtx.d3d9Identity.log / traceFrames: identity inputs for the classifier hooks in the lookup below, and the
    // identity trace's record of this lookup.
    const bool stats = D3d9Identity::statsActive();
    const bool tracing = D3d9IdentityTrace::recording();
    D3d9Identity::Lookup identityLookup;
    if (stats || tracing) {
      D3d9Identity::buildLookup(drawCallState, overrideMaterialData != nullptr ? overrideMaterialData->getHash() : kEmptyHash,
                                topologyHash, key.worldPos, identityLookup);
    }
    if (tracing) {
      D3d9IdentityTrace::recordDraw(frameId(), identityLookup, key, drawCallState.getGeometryData().boundingBox, hasReplacements);
    }
    ReplacementInstance* result = findOrCreateKeyed(key, stats ? &identityLookup : nullptr);

    // Portal-aware matching for ViewModel draw calls: if the normal lookup created a
    // brand-new ReplacementInstance (no existing prims), try matching through portals.
    if (result->prims.empty() &&
        drawCallState.cameraType == CameraType::ViewModel &&
        RtxOptions::useRayPortalVirtualInstanceMatching()) {
      ReplacementInstance* portalResult = tryPortalMatch(result, key, rayPortalManager);
      if (portalResult != nullptr) {
        perf::countD3D9(perf::D3D9Ctr::TrackerPortal);
        return portalResult;
      }
    }

    return result;
  }

  ReplacementInstance* DrawCallTracker::findOrCreateKeyed(
      const ReplacementInstance::LookupKey& key,
      const D3d9Identity::Lookup* identityLookup) {
    m_identityLookup = identityLookup;
    ReplacementInstance* result = findOrCreateReplacementInstance(key);
    m_identityLookup = nullptr;
    return result;
  }

  ReplacementInstance* DrawCallTracker::findReplacementInstanceByIdentity(XXH64_hash_t identityHash) {
    auto it = m_identityHashMap.find(identityHash);
    return (it != m_identityHashMap.end()) ? it->second : nullptr;
  }

  ReplacementInstance* DrawCallTracker::tryPortalMatch(
      ReplacementInstance* newInstance,
      const ReplacementInstance::LookupKey& key,
      const RayPortalManager& rayPortalManager) {
    const uint32_t currentFrameId = frameId();
    const float uniqueObjectDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();

    auto spatialMapIter = m_assetSpatialMaps.find(key.spatialMapHash);
    if (spatialMapIter == m_assetSpatialMaps.end()) {
      return nullptr;
    }

    auto portalFilter = [&](const ReplacementInstance* candidate) {
      return candidate != newInstance &&
             candidate->frameLastSeen != currentFrameId &&
             candidate->materialHash == key.materialHash;
    };

    for (auto& rayPortalPair : rayPortalManager.getRayPortalPairInfos()) {
      if (!rayPortalPair.has_value()) {
        continue;
      }
      for (uint32_t portalIdx = 0; portalIdx < 2; portalIdx++) {
        const auto& rayPortal = rayPortalPair->pairInfos[portalIdx];
        const Vector3 virtualPos = rayPortalManager.getVirtualPosition(
            key.worldPos, rayPortal.portalToOpposingPortalDirection);

        float nearestDistSqr = FLT_MAX;
        const ReplacementInstance* portalMatch = spatialMapIter->second.getNearestData(
            virtualPos, uniqueObjectDistanceSqr, nearestDistSqr, portalFilter);

        if (portalMatch != nullptr) {
          destroyReplacementInstance(newInstance, D3d9Identity::DestroyReason::Portal);
          // When this code was written, newInstance would always be the last element
          // of m_replacementInstances. If this assert fires, that assumption is no
          // longer true. Either restore that assumption, or change this code to search
          // for the correct ReplacementInstance to remove.
          assert(m_replacementInstances.back().get() == newInstance &&
                 "tryPortalMatch: newInstance must be the last element");
          m_replacementInstances.pop_back();

          ReplacementInstance* match = const_cast<ReplacementInstance*>(portalMatch);
          computeDirtyFlags(match, key);
          return reassociateMatch( match, key, &spatialMapIter->second);
        }
      }
    }

    return nullptr;
  }

  void DrawCallTracker::destroyReplacementInstance(ReplacementInstance* replacementInstance, D3d9Identity::DestroyReason reason) {
    if (!replacementInstance) {
      return;
    }

    if (D3d9Identity::statsActive()) {
      D3d9Identity::noteDestroyed(*replacementInstance, reason, frameId());
    }

    noteRekeyOrDestroy(replacementInstance);
    reportToRetainedV2(replacementInstance);
    m_identityHashMap.erase(replacementInstance->identityHash);

    eraseFromSpatialMap(m_assetSpatialMaps, replacementInstance->spatialMapHash,
        replacementInstance->spatialCacheTransformHash, replacementInstance);

    replacementInstance->clear();
  }

  void DrawCallTracker::garbageCollectReplacementInstances(
      RtCamera& camera,
      bool isAntiCullingSupported,
      bool parallelScan,
      uint32_t keptCutFrame) {
    translator::LifetimeFrame lifetime = translator::LifetimeFrame::read(camera, isAntiCullingSupported, false);
    lifetime.keptCutFrame = keptCutFrame;
    collectReplacementInstances(lifetime, parallelScan);
  }

  void DrawCallTracker::collectReplacementInstances(const translator::LifetimeFrame& lifetimeIn, bool parallelScan) {
    const uint32_t currentFrame = frameId();
    const bool forceGC = (m_replacementInstances.size() >=
        RtxOptions::AntiCulling::Object::numObjectsToKeep());
    translator::LifetimeFrame lifetime = lifetimeIn;
    lifetime.currentFrame = currentFrame;
    lifetime.forceCollect = forceGC;
    const uint32_t numFramesToKeepObjects = lifetime.framesToKeepObjects;

    // rtx.parallelScenePrepScans: the stale replacement instances are found on worker threads, and the
    // loop below then only reads their flags. The flags move with the instances when the loop swap-removes one.
    std::vector<uint8_t>& stale = m_gcStaleScratch;
    const bool useStaleFlags = parallelScan && m_replacementInstances.size() >= 4096;
    if (useStaleFlags) {
      const uint32_t count = uint32_t(m_replacementInstances.size());
      stale.resize(count);
      constexpr uint32_t kChunk = 1024;
      concurrency::parallel_for<uint32_t>(0, (count + kChunk - 1) / kChunk, [&](uint32_t chunk) {
        const uint32_t end = std::min(count, (chunk + 1) * kChunk);
        for (uint32_t i = chunk * kChunk; i < end; ++i) {
          if (i + 8 < end) {
            _mm_prefetch(reinterpret_cast<const char*>(&m_replacementInstances[i + 8]->frameLastSeen), _MM_HINT_T0);
          }
          const ReplacementInstance& ri = *m_replacementInstances[i];
          stale[i] = (ri.frameLastSeen + numFramesToKeepObjects <= currentFrame ||
                      (ri.handedToScene && ri.frameLastSeen < currentFrame)) ? 1 : 0;
        }
      });
    }

    for (size_t i = 0; i < m_replacementInstances.size();) {
      if (useStaleFlags && stale[i] == 0) {
        ++i;
        continue;
      }
      ReplacementInstance* replacementInstance = m_replacementInstances[i].get();

      // An instance whose object the scene took over goes in the first frame without a draw.
      const bool handedToScene = replacementInstance->handedToScene && replacementInstance->frameLastSeen < currentFrame;
      if (replacementInstance->frameLastSeen + numFramesToKeepObjects <= currentFrame || handedToScene) {
        // translator::LifetimePolicy (shared with the scene's transient lane).
        translator::UnseenObject unseen;
        unseen.frameLastSeen = replacementInstance->frameLastSeen;
        unseen.frameCreated = replacementInstance->frameCreated;
        unseen.categories = replacementInstance->categoryFlags;
        unseen.skinned = replacementInstance->isSkinned;
        unseen.wasMovingWhenLastSeen = replacementInstance->dirtyFlags.test(ReplacementInstance::DirtyFlag::Transform);
        unseen.geometryBounds = &replacementInstance->geometryBoundingBox;
        unseen.lightBounds = &replacementInstance->lightBoundingBox;
        unseen.objectToWorld = &replacementInstance->objectToWorld;
        const bool neverKeepUnseen = handedToScene;
        const bool keepAlive = !neverKeepUnseen && translator::LifetimePolicy::keepUnseen(unseen, lifetime);
        const bool beforeKeptCut = lifetime.seenBeforeKeptCut(replacementInstance->frameLastSeen);

        if (!keepAlive) {
          if (beforeKeptCut) {
            D3d9Identity::noteCollectedAfterKeptCut(); // rtx.d3d9Identity.log
          }
          destroyReplacementInstance(replacementInstance, D3d9Identity::DestroyReason::Gc);
          std::swap(m_replacementInstances[i], m_replacementInstances.back());
          m_replacementInstances.pop_back();
          if (useStaleFlags) {
            stale[i] = stale.back();
            stale.pop_back();
          }
          continue;
        }
      }
      ++i;
    }
  }

  void DrawCallTracker::clear() {
    ++m_retainedSteadyEpoch;
    ++m_retainedSteadyClearEpoch;
    m_retainedSteadyInvalidations.clear();
    ++m_clearGeneration;
    m_identityHashMap.clear();
    m_assetSpatialMaps.clear();
    m_replacementInstances.clear();
  }

  void DrawCallTracker::rebuildSpatialMaps(float cellSize) {
    for (auto& [hash, spatialMap] : m_assetSpatialMaps) {
      spatialMap.rebuild(cellSize);
    }
  }

}  // namespace dxvk
