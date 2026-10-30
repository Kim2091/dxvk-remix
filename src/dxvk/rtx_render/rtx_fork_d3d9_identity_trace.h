// D3D9 identity trace and its dry-run replay.
//
// Recording: every draw-call tracker lookup (the classifier inputs, the L1 identity hash, the spatial key,
// transform and bounds), every tracker GC point, camera cut, invalid-camera frame and other scene clear, with no
// geometry or texture data. Replay: the same records through a private DrawCallTracker on a virtual frame clock with
// the runtime's current options, so identity options can be compared without the game and without rendering.

#pragma once

#include <cstdint>
#include <string>

#include "rtx_fork_d3d9_identity.h"
#include "rtx_fork_d3d9_identity_trace_format.h"
#include "rtx_option.h"
#include "rtx_types.h"

namespace dxvk {

  class DxvkDevice;

  class D3d9IdentityTrace {
  public:
    RTX_OPTION("rtx.d3d9Identity", uint32_t, traceFrames, 0,
               "Records the next N frames of D3D9 draw identity to rtx-remix/identity-trace-<date>-<time>.d9it in the "
               "game folder, then stops. Each draw lookup stores its hashes, transform and bounds (about 240 bytes, no "
               "geometry or textures), plus garbage collections, camera cuts and scene clears. remix_replay "
               "--identity-trace replays it without rendering, to compare identity options. Set it to 0 and back to "
               "record again. 0 = off.");

    // CS thread: true while frames are being recorded.
    static bool recording() { return s_recording; }

    // Hooks (CS thread; each returns at once while not recording).
    // A tracker lookup from a DrawCallState.
    static void recordDraw(uint32_t frameId, const D3d9Identity::Lookup& lookup, const ReplacementInstance::LookupKey& key,
                           const AxisAlignedBoundingBox& bounds, bool hasReplacements);
    // SceneManager::garbageCollection, before the tracker GC.
    static void recordCollect(uint32_t frameId, bool cameraCut, bool antiCullingSupported);
    // A detected camera cut (not a teleport), whether or not it cleared the scene.
    static void recordCut(uint32_t frameId);
    // A frame without a valid camera past rtx.sceneKeepAliveFrames, whether or not it cleared the scene.
    static void recordInvalidCamera(uint32_t frameId);
    // SceneManager::clear for any other reason (cuts and invalid cameras are the two events above).
    static void recordClear(uint32_t frameId, D3d9Identity::ClearReason reason);

    // SceneManager frame end (CS thread): starts and stops recording from the option and writes the frame.
    static void onFrameEnd(uint32_t frameId);

    // remixapi_fork_ReplayIdentityTrace. The calling thread runs the replay; no frame may be in flight (remix_replay
    // presents nothing in this mode). False on a file error (result.message says why).
    static bool replay(DxvkDevice* device, const wchar_t* path, remixapi_fork_IdentityReplayResult& result);

  private:
    static bool s_recording;
  };

} // namespace dxvk
