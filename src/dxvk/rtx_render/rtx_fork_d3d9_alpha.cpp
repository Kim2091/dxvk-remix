// Options, shader constants, dev-menu controls and [D3d9Alpha] counters for the D3D9 alpha / replacement GPU work.
// See rtx_fork_d3d9_alpha.h.

#include "rtx_fork_d3d9_alpha.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtx_imgui.h"
#include "rtx_opacity_micromap_manager.h"
#include "../dxvk_perf_counters.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    struct TrackerTopologyStats {
      uint32_t created[uint32_t(D3d9Alpha::TrackerCreateReason::Count)] = {};
      XXH64_hash_t lastMaterial = 0;
      uint32_t total() const {
        uint32_t sum = 0;
        for (uint32_t c : created) {
          sum += c;
        }
        return sum;
      }
    };

    // Written on the CS thread only (scene preparation and draw submission), read by the perf log on the same thread.
    struct Stats {
      uint32_t frames = 0;
      uint64_t sharedOmm[6] = {};        // by SharedOmmOutcome
      uint64_t sharedOmmInstances = 0;   // instances of the BLASes above
      uint64_t trackerCreated[uint32_t(D3d9Alpha::TrackerCreateReason::Count)] = {};
      std::unordered_map<XXH64_hash_t, TrackerTopologyStats> trackerByTopology;
    };
    Stats s_stats;

    // Bounds the per-topology map between two log lines.
    constexpr size_t kMaxTrackedTopologies = 4096;
  }

  bool D3d9Alpha::statsActive() {
    return perf::g_perfDetailed.load(std::memory_order_relaxed);
  }

  void D3d9Alpha::noteSharedBlasOmm(SharedOmmOutcome outcome, uint32_t instanceCount) {
    s_stats.sharedOmm[uint32_t(outcome)]++;
    s_stats.sharedOmmInstances += instanceCount;
  }

  void D3d9Alpha::noteFrame() {
    s_stats.frames++;
  }

  void D3d9Alpha::noteTrackerCreation(TrackerCreateReason reason, XXH64_hash_t topologyHash, XXH64_hash_t materialHash) {
    s_stats.trackerCreated[uint32_t(reason)]++;
    auto it = s_stats.trackerByTopology.find(topologyHash);
    if (it == s_stats.trackerByTopology.end()) {
      if (s_stats.trackerByTopology.size() >= kMaxTrackedTopologies) {
        return;
      }
      it = s_stats.trackerByTopology.emplace(topologyHash, TrackerTopologyStats{}).first;
    }
    it->second.created[uint32_t(reason)]++;
    it->second.lastMaterial = materialHash;
  }

  void D3d9Alpha::logStats(const OpacityMicromapManager* omm) {
    const double n = std::max(1.0, double(s_stats.frames));
    const auto perFrame = [n](uint64_t v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", double(v) / n);
      return std::string(buf);
    };
    const auto& so = s_stats.sharedOmm;
    const auto& tc = s_stats.trackerCreated;

    // sharedOmm*: per frame, multi-instance per-geometry BLASes whose instances use OMMs, by outcome.
    // wouldBind = compatible while rtx.opacityMicromap.bindSharedBlas is off. builtNeverBound: micromaps in
    // the cache that no BLAS ever bound (now).
    Logger::info(str::format("[D3d9Alpha] frames=", s_stats.frames,
      " | sharedOmmBlas bound=", perFrame(so[uint32_t(SharedOmmOutcome::Bound)]),
      " notReady=", perFrame(so[uint32_t(SharedOmmOutcome::NotReady)]),
      " wouldBind=", perFrame(so[uint32_t(SharedOmmOutcome::OptionOff)]),
      " mixedUsers=", perFrame(so[uint32_t(SharedOmmOutcome::MixedUsers)]),
      " hashMismatch=", perFrame(so[uint32_t(SharedOmmOutcome::HashMismatch)]),
      " twoState=", perFrame(so[uint32_t(SharedOmmOutcome::TwoState)]),
      " instances=", perFrame(s_stats.sharedOmmInstances),
      " builtNeverBound=", omm ? omm->countBuiltNeverBound() : 0,
      " | trackerCreated newTopology=", perFrame(tc[uint32_t(TrackerCreateReason::NewTopology)]),
      " materialMismatch=", perFrame(tc[uint32_t(TrackerCreateReason::MaterialMismatch)]),
      " noneInRange=", perFrame(tc[uint32_t(TrackerCreateReason::NoneInRange)]),
      " | bindSharedBlas=", bindSharedBlas() ? 1 : 0));

    const uint32_t topN = trackerCreationLogTopN();
    if (topN != 0 && !s_stats.trackerByTopology.empty()) {
      std::vector<std::pair<XXH64_hash_t, const TrackerTopologyStats*>> sorted;
      sorted.reserve(s_stats.trackerByTopology.size());
      for (const auto& [hash, entry] : s_stats.trackerByTopology) {
        sorted.emplace_back(hash, &entry);
      }
      const size_t count = std::min<size_t>(topN, sorted.size());
      std::partial_sort(sorted.begin(), sorted.begin() + count, sorted.end(), [](const auto& a, const auto& b) {
        return a.second->total() > b.second->total();
      });
      std::string line = "[D3d9Alpha] trackerCreated top (topology hash: newTopology/materialMismatch/noneInRange, last material):";
      for (size_t i = 0; i < count; ++i) {
        const TrackerTopologyStats& e = *sorted[i].second;
        line += str::format(" 0x", std::hex, sorted[i].first, std::dec, ": ",
          e.created[uint32_t(TrackerCreateReason::NewTopology)], "/",
          e.created[uint32_t(TrackerCreateReason::MaterialMismatch)], "/",
          e.created[uint32_t(TrackerCreateReason::NoneInRange)],
          " mat 0x", std::hex, e.lastMaterial, std::dec, ";");
      }
      Logger::info(line);
    }

    s_stats = Stats{};
  }

} // namespace dxvk
