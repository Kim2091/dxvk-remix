#pragma once

// GPU/BLAS work for alpha-tested/blended content and replacement assets on the standard D3D9 path.
//
// The controls are in the dev menu (Rendering, "D3D9 Alpha / Replacements", next to "Opacity Micromap"):
// - opacityMicromap.bindSharedBlas: per-geometry BLASes shared by several instances bind the opacity micromap their
//   instances all request (OpacityMicromapManager::checkSharedBlasOmm).
//
// [D3d9Alpha] counters are written with the detailed performance log (rtx.profile.detailedLogging) and are not
// collected while it is off.

#include <cstdint>

#include "rtx_option.h"
#include "../../util/xxHash/xxhash.h"

namespace dxvk {

  class OpacityMicromapManager;

  class D3d9Alpha {
  public:
    RTX_OPTION("rtx.opacityMicromap", bool, bindSharedBlas, true,
               "Lets a per-geometry BLAS shared by several instances (meshes above rtx.minPrimsInDynamicBLAS that are "
               "drawn more than once, such as repeated foliage and fences) use an opacity micromap, when every instance "
               "asks for the same one (same source hash, texture factor and colour flags). Unknown micro-triangles keep "
               "the exact alpha test, so the result is the same as for single-instance geometry.");
    RTX_OPTION("rtx", uint32_t, trackerCreationLogTopN, 8,
               "[D3d9Alpha] log: how many topology hashes with the most instance creations to list. 0 = none.");

    // Counters, collected only while detailed performance logging is on.
    static bool statsActive();

    // AccelManager: one call per multi-instance per-geometry BLAS with OMM users (result of checkSharedBlasOmm).
    enum class SharedOmmOutcome : uint8_t { Bound, NotReady, MixedUsers, HashMismatch, TwoState, OptionOff };
    static void noteSharedBlasOmm(SharedOmmOutcome outcome, uint32_t instanceCount);
    static void noteFrame();

    // DrawCallTracker: why a new replacement instance was created (Level 3).
    enum class TrackerCreateReason : uint8_t {
      NewTopology,      // no previous instance of this topology at all
      MaterialMismatch, // an unmatched previous instance in range, but with another material
      NoneInRange,      // none unmatched in range: moved further than uniqueObjectDistance, or more draws than before
      Count
    };
    static void noteTrackerCreation(TrackerCreateReason reason, XXH64_hash_t topologyHash, XXH64_hash_t materialHash);

    // Writes and resets the [D3d9Alpha] line (rtx perf log, after [OmmChurn]).
    static void logStats(const OpacityMicromapManager* omm);
  };

} // namespace dxvk
