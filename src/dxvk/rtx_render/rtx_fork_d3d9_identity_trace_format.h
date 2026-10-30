#pragma once

// D3D9 identity trace replay: the result of remixapi_fork_ReplayIdentityTrace. Shared by the runtime
// (rtx_remix_api.cpp) and tools/remix_replay, so it depends on remix_c.h and the standard library only.

#include <cstdint>

#include <remix/remix_c.h>

extern "C" {

  struct remixapi_fork_IdentityReplayResult {
    uint64_t frames;          // FrameEnd records replayed
    uint64_t draws;           // tracker lookups
    uint64_t exact;           // L1 identity hits
    uint64_t matched;         // L2 matches (exact transform, spatial, material fallback)
    uint64_t created;         // L3: new replacement instances
    uint64_t destroyed;       // collected by the GC or dropped by clears
    uint64_t clears;          // scene clears replayed (cuts and invalid cameras only while rtx.scene.d3d9CutAsEvent is off)
    uint64_t cutsKept;        // cuts that kept the scene (rtx.scene.d3d9CutAsEvent on)
    uint64_t peakInstances;   // most replacement instances alive at a frame end
    uint64_t endInstances;    // alive after the last frame
    char message[1024];       // what was recorded (options, frames), or the error
  };

  // Replays an identity trace (rtx.d3d9Identity.traceFrames) through a private draw-call tracker with the runtime's
  // current options. Needs a started runtime (remixapi_Startup); presents nothing. The runtime log gets the usual
  // [D3d9Identity] lines when rtx.d3d9Identity.log is on.
  typedef remixapi_ErrorCode (REMIXAPI_CALL* PFN_remixapi_fork_ReplayIdentityTrace)(
    const wchar_t* path,
    remixapi_fork_IdentityReplayResult* result);

}
