// The [D3d9Identity] classifier. See rtx_fork_d3d9_identity.h.

#include "rtx_fork_d3d9_identity.h"
#include "rtx_fork_d3d9_identity_trace.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtx_imgui.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_options.h"
#include "rtx_types.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  std::atomic<bool> D3d9Identity::s_logActive { false };
  D3d9Identity::ClearReason D3d9Identity::s_pendingClearReason = D3d9Identity::ClearReason::Other;

  namespace {
    constexpr uint32_t kLogIntervalFrames = 300;
    constexpr uint32_t kGraveyardFrames = 8;
    constexpr uint32_t kStaticFrames = 8;
    constexpr uint32_t kAfterClearFrames = 2;
    constexpr size_t kMaxGraveyard = 1 << 16;
    constexpr size_t kMaxTopKeys = 4096;

    // L3 reasons, in classification order.
    enum class NewReason : uint8_t {
      AfterClear,       // within kAfterClearFrames frames of a scene clear
      ResurrectedGap1,  // an instance with the same layout, VS, material and cell died 1 frame ago, same topology
      ResurrectedGap2to8,
      Recount,          // ... same layout, VS, material and cell, but other counts (topology)
      RecountDynamic,   // ... and the draw's vertices come from a dynamic / UP buffer
      MaterialMismatch, // an unmatched instance of the topology in range, other material
      NoneInRange,      // instances of the topology exist, none unmatched in range
      NeverSeen,        // none of the above
      Count
    };
    const char* const kNewReasonNames[] = {
      "afterClear", "resurrected1", "resurrected2to8", "recount", "recountDynamic", "materialMismatch", "noneInRange", "neverSeen"
    };
    static_assert(sizeof(kNewReasonNames) / sizeof(kNewReasonNames[0]) == size_t(NewReason::Count));

    // l1miss.* components.
    enum class Component : uint8_t {
      VertexShader, Geometry, Transform, Material, OverrideMaterial, Bones, Categories, CameraType, TexXform, RtTarget, Count
    };
    const char* const kComponentNames[] = {
      "vs", "geometry", "transform", "material", "overrideMaterial", "bones", "categories", "cameraType", "texXform", "rtTarget"
    };
    static_assert(sizeof(kComponentNames) / sizeof(kComponentNames[0]) == size_t(Component::Count));

    // D3d9Identity::CategoryCtr names and the categories they test.
    const char* const kCategoryNames[] = { "particle", "beam", "emitter", "hairCards", "sky", "worldUI", "none" };
    static_assert(sizeof(kCategoryNames) / sizeof(kCategoryNames[0]) == size_t(D3d9Identity::CategoryCtr::Count));
    const InstanceCategories kBreakdownCategories[] = {
      InstanceCategories::Particle, InstanceCategories::Beam, InstanceCategories::ParticleEmitter,
      InstanceCategories::HairCards, InstanceCategories::Sky, InstanceCategories::WorldUI
    };
    static_assert(sizeof(kBreakdownCategories) / sizeof(kBreakdownCategories[0]) + 1 == size_t(D3d9Identity::CategoryCtr::Count));

    // Calls count(index) for each breakdown category in `categories`, or for `none`.
    template<typename F>
    void forEachBreakdownCategory(const CategoryFlags& categories, F&& count) {
      bool any = false;
      for (uint32_t i = 0; i < sizeof(kBreakdownCategories) / sizeof(kBreakdownCategories[0]); ++i) {
        if (categories.test(kBreakdownCategories[i])) {
          count(i);
          any = true;
        }
      }
      if (!any) {
        count(uint32_t(D3d9Identity::CategoryCtr::None));
      }
    }

    struct GraveEntry {
      XXH64_hash_t topology;
      uint32_t frame;
    };

    struct TopKey {
      uint32_t created[uint32_t(NewReason::Count)] = {};
      uint8_t flags = 0;
      Vector3 samplePos = Vector3(0.f);
      uint32_t total() const {
        uint32_t sum = 0;
        for (uint32_t c : created) {
          sum += c;
        }
        return sum;
      }
    };

    // Written and read on the CS thread only.
    struct Stats {
      uint32_t frames = 0;
      uint64_t created[uint32_t(NewReason::Count)] = {};
      uint64_t createdTotal = 0;
      uint64_t newFlags[5] = {};                   // by Flag bit
      uint64_t newCategory[uint32_t(D3d9Identity::CategoryCtr::Count)] = {};
      uint64_t newSkyDrawCallId = 0;               // new instances that are Sky by rtx.skyDrawcallIdThreshold
      uint64_t noneInRangeDist[3] = {};            // 1-2x, 2-4x, none found (search covers up to ~2-4x)
      uint64_t l1Miss = 0;
      uint64_t l1MissComponent[uint32_t(Component::Count)] = {};
      uint64_t l1MissVsConstOnly = 0;
      uint64_t l1MissMulti = 0;
      uint64_t l1MissUnknown = 0;                  // no component differs (instance traced before the log was on)
      uint64_t l2Exact = 0;
      uint64_t l2Spatial = 0;
      uint64_t spatialDist[3] = {};                // <0.25x, 0.25-0.5x, 0.5-1x of uniqueObjectDistance
      uint64_t relinks = 0;
      uint64_t staticSwap = 0;
      uint64_t cacheNew = 0;
      uint64_t cacheExisted = 0;
      uint64_t destroyed[uint32_t(D3d9Identity::DestroyReason::Count)] = {};
      uint64_t clears[uint32_t(D3d9Identity::ClearReason::Count)] = {};
      uint64_t lostToClears = 0;
      uint64_t cutsKept = 0;                       // camera cuts that kept the scene (rtx.scene.d3d9CutAsEvent)
      uint64_t collectedAfterKeptCut = 0;          // instances collected because last drawn before a kept cut
      std::unordered_map<XXH64_hash_t, TopKey> topKeys;
    };
    Stats s_stats;

    // Destroyed instances of the last kGraveyardFrames frames, by graveyardKey().
    std::unordered_map<XXH64_hash_t, GraveEntry> s_graveyard;
    // This frame's moves of instances that had been static for kStaticFrames frames: XXH(topology, old transform).
    std::unordered_set<XXH64_hash_t> s_staticMoves;
    uint32_t s_lastClearFrame = 0;
    bool s_hadClear = false;
    uint32_t s_lastPruneFrame = 0;

    XXH64_hash_t hashMatrix(const Matrix4& m) {
      return XXH3_64bits(&m, sizeof(m));
    }

    XXH64_hash_t graveyardKey(XXH64_hash_t layout, XXH64_hash_t vsBytecode, XXH64_hash_t material, const Vector3& pos) {
      const float cell = std::max(RtxOptions::uniqueObjectDistance(), 1e-3f);
      struct Key {
        XXH64_hash_t layout, vsBytecode, material;
        int32_t cx, cy, cz, pad;
      } key { layout, vsBytecode, material,
              int32_t(std::floor(pos.x / cell)), int32_t(std::floor(pos.y / cell)), int32_t(std::floor(pos.z / cell)), 0 };
      return XXH3_64bits(&key, sizeof(key));
    }

    XXH64_hash_t staticMoveKey(XXH64_hash_t topology, XXH64_hash_t transform) {
      const XXH64_hash_t data[2] = { topology, transform };
      return XXH3_64bits(data, sizeof(data));
    }

    void countFlags(uint8_t flags) {
      for (uint32_t bit = 0; bit < 5; ++bit) {
        if (flags & (1u << bit)) {
          s_stats.newFlags[bit]++;
        }
      }
    }

    void checkStaticSwap(const D3d9Identity::Lookup& lookup) {
      if (!s_staticMoves.empty() && s_staticMoves.erase(staticMoveKey(lookup.topology, lookup.trace.transform)) != 0) {
        s_stats.staticSwap++;
      }
    }

    void prune(uint32_t frameId) {
      for (auto it = s_graveyard.begin(); it != s_graveyard.end();) {
        it = (it->second.frame + kGraveyardFrames < frameId) ? s_graveyard.erase(it) : std::next(it);
      }
    }
  }

  void D3d9Identity::showImguiSettings() {
    RemixGui::Checkbox("D3D9 Identity: [D3d9Identity] Log (counters only)", &logObject());
    RemixGui::DragInt("D3D9 Identity: Log Top Topologies", &logTopNObject(), 1.f, 0, 64);
    RemixGui::DragInt("D3D9 Identity: Record Identity Trace (frames, 0 = off)", &D3d9IdentityTrace::traceFramesObject(), 10.f, 0, 3600);
  }

  void D3d9Identity::buildLookup(const DrawCallState& drawCallState, XXH64_hash_t overrideMaterialHash,
                                 XXH64_hash_t topologyHash, const Vector3& worldPos, Lookup& out) {
    const GeometryHashes& hashes = drawCallState.getGeometryData().hashes;
    const DrawCallTransforms& transforms = drawCallState.getTransformData();
    ReplacementInstance::IdentityTrace& t = out.trace;

    HashRule noVs(rules::FullGeometryHash);
    noVs.clr(HashComponents::VertexShader);
    t.geometryNoVs = hashes.getHashForRule(noVs);
    t.vertexShader = hashes[HashComponents::VertexShader];
    t.fullVertexShader = hashes[HashComponents::VertexShader];
    t.material = drawCallState.getMaterialData().getHash();
    t.overrideMaterial = overrideMaterialHash;
    t.bones = drawCallState.getSkinningState().boneHash;
    t.transform = hashMatrix(transforms.objectToWorld);
    {
      struct TexXform {
        Matrix4 m;
        uint32_t texgen;
      } tx { transforms.textureTransform, uint32_t(transforms.texgenMode) };
      t.textureTransform = XXH3_64bits(&tx, sizeof(tx));
    }
    t.layout = hashes[HashComponents::VertexLayout];
    t.vsBytecode = hashes.vertexShaderBytecode;
    t.categories = drawCallState.getCategoryFlags().raw();
    t.cameraType = uint32_t(drawCallState.cameraType);
    t.rtTarget = drawCallState.isUsingRaytracedRenderTarget;
    t.valid = true;

    uint8_t flags = 0;
    if (drawCallState.vertexCaptured) {
      flags |= FlagVertexCaptured;
    }
    static const Matrix4 kIdentity;
    if (std::memcmp(&transforms.objectToWorld, &kIdentity, sizeof(Matrix4)) == 0) {
      flags |= FlagIdentityXform;
    }
    if (drawCallState.dynamicSource) {
      flags |= FlagDynamicSource;
    }
    if (drawCallState.getSkinningState().numBones > 0) {
      flags |= FlagSkinned;
    }
    if (drawCallState.getCategoryFlags().any(InstanceCategories::Particle, InstanceCategories::ParticleEmitter, InstanceCategories::Beam)) {
      flags |= FlagParticle;
    }
    if (drawCallState.skyFromDrawCallId) {
      flags |= FlagSkyDrawCallId;
    }
    t.flags = flags;

    out.worldPos = worldPos;
    out.topology = topologyHash;
  }

  void D3d9Identity::noteExact(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId) {
    const uint32_t lastMove = ri.identityTrace.valid ? ri.identityTrace.lastMoveFrame : ri.frameCreated;
    ri.identityTrace = lookup.trace;
    ri.identityTrace.lastMoveFrame = lastMove;
  }

  void D3d9Identity::noteL2Match(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId, bool moved,
                                 float distanceSqr, float thresholdSqr) {
    checkStaticSwap(lookup);

    const ReplacementInstance::IdentityTrace& a = ri.identityTrace;
    const ReplacementInstance::IdentityTrace& b = lookup.trace;
    s_stats.l1Miss++;
    if (!a.valid) {
      s_stats.l1MissUnknown++;
    } else {
      bool changed[uint32_t(Component::Count)] = {};
      changed[uint32_t(Component::VertexShader)] = a.vertexShader != b.vertexShader;
      changed[uint32_t(Component::Geometry)] = a.geometryNoVs != b.geometryNoVs;
      changed[uint32_t(Component::Transform)] = a.transform != b.transform;
      changed[uint32_t(Component::Material)] = a.material != b.material;
      changed[uint32_t(Component::OverrideMaterial)] = a.overrideMaterial != b.overrideMaterial;
      changed[uint32_t(Component::Bones)] = a.bones != b.bones;
      changed[uint32_t(Component::Categories)] = a.categories != b.categories;
      changed[uint32_t(Component::CameraType)] = a.cameraType != b.cameraType;
      changed[uint32_t(Component::TexXform)] = a.textureTransform != b.textureTransform;
      changed[uint32_t(Component::RtTarget)] = a.rtTarget != b.rtTarget;
      uint32_t count = 0;
      for (uint32_t i = 0; i < uint32_t(Component::Count); ++i) {
        if (changed[i]) {
          s_stats.l1MissComponent[i]++;
          ++count;
        }
      }
      if (count == 0) {
        s_stats.l1MissUnknown++;
      } else if (count == 1 && changed[uint32_t(Component::VertexShader)]) {
        s_stats.l1MissVsConstOnly++;
      } else if (count > 1) {
        s_stats.l1MissMulti++;
      }
    }

    if (distanceSqr < 0.f) {
      s_stats.l2Exact++;
    } else {
      s_stats.l2Spatial++;
      const float ratioSqr = thresholdSqr > 0.f ? distanceSqr / thresholdSqr : 0.f;
      s_stats.spatialDist[ratioSqr < 0.0625f ? 0 : ratioSqr < 0.25f ? 1 : 2]++;
    }

    uint32_t lastMove = a.valid ? a.lastMoveFrame : ri.frameCreated;
    if (moved) {
      // A long-static instance taken away by a draw elsewhere: if a later draw of this frame sits exactly at the old
      // transform, the two traded instances (draw-order dependence, C6).
      if (a.valid && lastMove + kStaticFrames <= frameId && ri.frameCreated + kStaticFrames <= frameId) {
        s_staticMoves.insert(staticMoveKey(lookup.topology, a.transform));
      }
      lastMove = frameId;
    }
    ri.identityTrace = b;
    ri.identityTrace.lastMoveFrame = lastMove;
  }

  void D3d9Identity::noteCreated(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId, bool topologyKnown,
                                 bool materialBlocked, float nearestUnmatchedDistSqr, float thresholdSqr) {
    checkStaticSwap(lookup);

    NewReason reason = NewReason::NeverSeen;
    if (s_hadClear && frameId <= s_lastClearFrame + kAfterClearFrames) {
      reason = NewReason::AfterClear;
    } else {
      auto grave = s_graveyard.find(graveyardKey(lookup.trace.layout, lookup.trace.vsBytecode, lookup.trace.material, lookup.worldPos));
      if (grave != s_graveyard.end() && grave->second.frame + kGraveyardFrames >= frameId) {
        if (grave->second.topology == lookup.topology) {
          reason = (frameId - grave->second.frame <= 1) ? NewReason::ResurrectedGap1 : NewReason::ResurrectedGap2to8;
        } else {
          reason = (lookup.trace.flags & FlagDynamicSource) ? NewReason::RecountDynamic : NewReason::Recount;
        }
        s_graveyard.erase(grave);
      } else if (topologyKnown) {
        if (materialBlocked) {
          reason = NewReason::MaterialMismatch;
        } else {
          reason = NewReason::NoneInRange;
          const float ratioSqr = (nearestUnmatchedDistSqr >= 0.f && thresholdSqr > 0.f) ? nearestUnmatchedDistSqr / thresholdSqr : -1.f;
          s_stats.noneInRangeDist[ratioSqr < 0.f ? 2 : ratioSqr < 4.f ? 0 : 1]++;
        }
      }
    }

    s_stats.created[uint32_t(reason)]++;
    s_stats.createdTotal++;
    countFlags(lookup.trace.flags);
    forEachBreakdownCategory(CategoryFlags(lookup.trace.categories), [](uint32_t i) { s_stats.newCategory[i]++; });
    if (lookup.trace.flags & FlagSkyDrawCallId) {
      s_stats.newSkyDrawCallId++;
    }

    auto it = s_stats.topKeys.find(lookup.topology);
    if (it == s_stats.topKeys.end() && s_stats.topKeys.size() < kMaxTopKeys) {
      it = s_stats.topKeys.emplace(lookup.topology, TopKey{}).first;
    }
    if (it != s_stats.topKeys.end()) {
      it->second.created[uint32_t(reason)]++;
      it->second.flags |= lookup.trace.flags;
      it->second.samplePos = lookup.worldPos;
    }

    ri.identityTrace = lookup.trace;
    ri.identityTrace.lastMoveFrame = frameId;
  }

  void D3d9Identity::noteDestroyed(const ReplacementInstance& ri, DestroyReason reason, uint32_t frameId) {
    s_stats.destroyed[uint32_t(reason)]++;
    if (reason != DestroyReason::Gc || !ri.identityTrace.valid) {
      return;
    }
    if (s_graveyard.size() >= kMaxGraveyard) {
      s_graveyard.clear();
    }
    s_graveyard[graveyardKey(ri.identityTrace.layout, ri.identityTrace.vsBytecode, ri.identityTrace.material, ri.centroid)] =
      GraveEntry { ri.spatialMapHash, frameId };
  }

  void D3d9Identity::noteCacheLookup(bool existed) {
    (existed ? s_stats.cacheExisted : s_stats.cacheNew)++;
  }

  void D3d9Identity::noteRelink() {
    s_stats.relinks++;
  }

  void D3d9Identity::noteCutKept() {
    if (statsActive()) {
      s_stats.cutsKept++;
    }
  }

  void D3d9Identity::noteCollectedAfterKeptCut() {
    if (statsActive()) {
      s_stats.collectedAfterKeptCut++;
    }
  }

  void D3d9Identity::onSceneClear(uint32_t frameId, size_t instanceCount) {
    const ClearReason reason = s_pendingClearReason;
    s_pendingClearReason = ClearReason::Other;
    if (!statsActive()) {
      return;
    }
    s_stats.clears[uint32_t(reason)]++;
    s_stats.lostToClears += instanceCount;
    s_lastClearFrame = frameId;
    s_hadClear = true;
    s_graveyard.clear();
    s_staticMoves.clear();
  }

  void D3d9Identity::onFrameEnd(uint32_t frameId) {
    const bool active = log();
    const bool wasActive = s_logActive.exchange(active, std::memory_order_relaxed);
    s_staticMoves.clear();
    const auto resetCounters = []() {
      s_stats = Stats{};
    };
    if (!active) {
      if (wasActive) {
        resetCounters();
        s_graveyard.clear();
      }
      return;
    }
    if (!wasActive) {
      // Starts counting at the next frame.
      resetCounters();
      return;
    }
    s_stats.frames++;
    if (frameId - s_lastPruneFrame >= kGraveyardFrames) {
      prune(frameId);
      s_lastPruneFrame = frameId;
    }
    if (s_stats.frames < kLogIntervalFrames) {
      return;
    }

    const double n = std::max(1.0, double(s_stats.frames));
    const auto pf = [n](uint64_t v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", double(v) / n);
      return std::string(buf);
    };
    const Stats& s = s_stats;

    std::string line = str::format("[D3d9Identity] frames=", s.frames, " (per frame) | new=", pf(s.createdTotal));
    for (uint32_t i = 0; i < uint32_t(NewReason::Count); ++i) {
      line += str::format(" ", kNewReasonNames[i], "=", pf(s.created[i]));
    }
    line += str::format(" noneInRangeNearest(1-2x/2-4x/none)=", pf(s.noneInRangeDist[0]), "/", pf(s.noneInRangeDist[1]), "/", pf(s.noneInRangeDist[2]),
      " | newFlags vc=", pf(s.newFlags[0]), " identityXform=", pf(s.newFlags[1]), " dynamicSource=", pf(s.newFlags[2]),
      " skinned=", pf(s.newFlags[3]), " particle=", pf(s.newFlags[4]), " | newCategory");
    for (uint32_t i = 0; i < uint32_t(CategoryCtr::Count); ++i) {
      line += str::format(" ", kCategoryNames[i], "=", pf(s.newCategory[i]));
    }
    line += str::format(" skyByDrawCallId=", pf(s.newSkyDrawCallId), " (skyDrawcallIdThreshold=", RtxOptions::skyDrawcallIdThreshold(), ")",
      " | l1miss=", pf(s.l1Miss), " vsConstOnly=", pf(s.l1MissVsConstOnly), " multi=", pf(s.l1MissMulti), " unknown=", pf(s.l1MissUnknown));
    for (uint32_t i = 0; i < uint32_t(Component::Count); ++i) {
      line += str::format(" l1miss.", kComponentNames[i], "=", pf(s.l1MissComponent[i]));
    }
    line += str::format(" | l2 exactTransform=", pf(s.l2Exact), " spatial=", pf(s.l2Spatial),
      " spatialDist(<0.25x/<0.5x/<1x)=", pf(s.spatialDist[0]), "/", pf(s.spatialDist[1]), "/", pf(s.spatialDist[2]),
      " | relinks=", pf(s.relinks), " staticSwap=", pf(s.staticSwap),
      " | cache new=", pf(s.cacheNew), " existed=", pf(s.cacheExisted),
      " | destroyed gc=", pf(s.destroyed[uint32_t(DestroyReason::Gc)]),
      " replacements=", pf(s.destroyed[uint32_t(DestroyReason::Replacements)]),
      " portal=", pf(s.destroyed[uint32_t(DestroyReason::Portal)]),
      " | clears(total) cut=", s.clears[uint32_t(ClearReason::CameraCut)],
      " invalidCamera=", s.clears[uint32_t(ClearReason::InvalidCamera)],
      " mods=", s.clears[uint32_t(ClearReason::ModsChanged)],
      " other=", s.clears[uint32_t(ClearReason::Other)],
      " instancesLost=", s.lostToClears,
      " cutsKept=", s.cutsKept, " collectedAfterKeptCut=", s.collectedAfterKeptCut);
    Logger::info(line);

    const uint32_t topN = logTopN();
    if (topN != 0 && !s.topKeys.empty()) {
      std::vector<std::pair<XXH64_hash_t, const TopKey*>> sorted;
      sorted.reserve(s.topKeys.size());
      for (const auto& [hash, entry] : s.topKeys) {
        sorted.emplace_back(hash, &entry);
      }
      const size_t count = std::min<size_t>(topN, sorted.size());
      std::partial_sort(sorted.begin(), sorted.begin() + count, sorted.end(), [](const auto& a, const auto& b) {
        return a.second->total() > b.second->total();
      });
      std::string top = "[D3d9Identity] top new (topology: total afterClear/res1/res2to8/recount/recountDyn/matMismatch/noneInRange/neverSeen, flags vc|idXform|dyn|skin|particle, sample pos):";
      for (size_t i = 0; i < count; ++i) {
        const TopKey& e = *sorted[i].second;
        top += str::format(" 0x", std::hex, sorted[i].first, std::dec, ": ", e.total(), " ");
        for (uint32_t r = 0; r < uint32_t(NewReason::Count); ++r) {
          top += str::format(r ? "/" : "", e.created[r]);
        }
        char pos[96];
        std::snprintf(pos, sizeof(pos), " (%.1f %.1f %.1f)", e.samplePos.x, e.samplePos.y, e.samplePos.z);
        top += str::format(" f", (e.flags & FlagVertexCaptured) ? "v" : "-", (e.flags & FlagIdentityXform) ? "i" : "-",
                           (e.flags & FlagDynamicSource) ? "d" : "-", (e.flags & FlagSkinned) ? "s" : "-",
                           (e.flags & FlagParticle) ? "p" : "-", pos, ";");
      }
      Logger::info(top);
    }

    s_stats = Stats{};
  }

} // namespace dxvk
