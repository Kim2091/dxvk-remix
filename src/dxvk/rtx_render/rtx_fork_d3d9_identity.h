#pragma once

// D3D9 instance identity: the [D3d9Identity] classifier.
//
// Off by default; with rtx.d3d9Identity.log off nothing is collected and no code path changes. The classifier only
// counts: why the draw-call tracker created a new replacement instance (L3), which identity input changed when an
// exact (L1) match failed, relinks, static swaps, clears and the draw-call cache outcome. The line is written every
// 300 frames.

#include <atomic>
#include <cstdint>
#include <memory>

#include "rtx_option.h"
#include "rtx_types.h"
#include "../../util/xxHash/xxhash.h"

namespace dxvk {

  class D3d9Identity {
  public:
    RTX_OPTION("rtx.d3d9Identity", bool, log, false,
               "Diagnostic: every 300 frames, logs a [D3d9Identity] line that explains why D3D9 draws get new "
               "replacement instances: which identity input changed when the exact match failed, recently destroyed "
               "instances, relinks, static swaps, scene clears, draw-call cache results and the top topologies. "
               "Counters only, and nothing is collected while off.");
    RTX_OPTION("rtx.d3d9Identity", uint32_t, logTopN, 8,
               "[D3d9Identity] log: how many topology hashes with the most new replacement instances to list. 0 = none.");


    // Flags of a draw / replacement instance (IdentityTrace::flags, top-key flags).
    enum Flag : uint8_t {
      FlagVertexCaptured = 1 << 0,
      FlagIdentityXform  = 1 << 1,
      FlagDynamicSource  = 1 << 2,
      FlagSkinned        = 1 << 3,
      FlagParticle       = 1 << 4,
      FlagSkyDrawCallId  = 1 << 5, // Sky category assigned by rtx.skyDrawcallIdThreshold (untextured, early draw ID)
    };

    // Category breakdown of new instances, in log order. A draw with several of these categories counts in each;
    // `none` = none of them.
    enum class CategoryCtr : uint8_t { Particle, Beam, ParticleEmitter, HairCards, Sky, WorldUI, None, Count };

    // Why a replacement instance was destroyed.
    enum class DestroyReason : uint8_t { Gc, Replacements, Portal, Count };

    // Why the scene was cleared (SceneManager::clear).
    enum class ClearReason : uint8_t { Other, CameraCut, InvalidCamera, ModsChanged, Count };

    // Per-frame cached option state (refreshed at frame end; readable from any thread).
    static bool statsActive() { return s_logActive.load(std::memory_order_relaxed); }

    static void showImguiSettings();

    // SceneManager frame end (CS thread): refreshes the cached options, prunes the graveyard, writes the line.
    static void onFrameEnd(uint32_t frameId);

    // --- Draw-call tracker hooks (CS thread, only while statsActive()). ---

    // The identity inputs of the draw being looked up. Built by beginLookup from the DrawCallState.
    struct Lookup {
      ReplacementInstance::IdentityTrace trace;
      Vector3 worldPos;
      XXH64_hash_t topology = 0;
    };
    static void buildLookup(const DrawCallState& drawCallState, XXH64_hash_t overrideMaterialHash,
                            XXH64_hash_t topologyHash, const Vector3& worldPos, Lookup& out);

    // L1 hit.
    static void noteExact(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId);
    // L2 match (exact transform, spatial or material fallback). distanceSqr < 0 = exact transform.
    static void noteL2Match(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId, bool moved,
                            float distanceSqr, float thresholdSqr);
    // L3. nearestUnmatchedDistSqr: nearest instance of the topology not yet seen this frame (any material),
    // < 0 = none within the search radius; materialBlocked = one within the threshold had another material.
    static void noteCreated(ReplacementInstance& ri, const Lookup& lookup, uint32_t frameId, bool topologyKnown,
                            bool materialBlocked, float nearestUnmatchedDistSqr, float thresholdSqr);
    static void noteDestroyed(const ReplacementInstance& ri, DestroyReason reason, uint32_t frameId);

    // --- Other hooks (CS thread, only while statsActive()). ---
    static void noteCacheLookup(bool existed);
    static void noteRelink();
    // Set right before SceneManager::clear by the call site that knows the reason.
    static void setPendingClearReason(ClearReason reason) { s_pendingClearReason = reason; }
    static ClearReason pendingClearReason() { return s_pendingClearReason; }
    // SceneManager::clear, before the tracker drops its instanceCount replacement instances.
    static void onSceneClear(uint32_t frameId, size_t instanceCount);
    // rtx.scene.d3d9CutAsEvent / noImplicitClears. CS thread, counted only while statsActive().
    static void noteCutKept();
    static void noteCollectedAfterKeptCut();

  private:
    static std::atomic<bool> s_logActive;
    static ClearReason s_pendingClearReason;
  };

} // namespace dxvk
