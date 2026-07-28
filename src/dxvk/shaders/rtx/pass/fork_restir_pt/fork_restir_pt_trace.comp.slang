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
/*
* Portions of this file are a port of the ReSTIR PT reference implementation:
*   Copyright (c) 2022, Daqi Lin.  All rights reserved.
*   ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/TracePass.cs.slang
* Licensed under BSD-3-Clause (see ThirdPartyLicenses.txt, "ReSTIR PT" entry).
*/

// ReSTIR PT (fork) -- phase 1 trace + replay-parity pass.
//
// ports ReSTIRPTPass/TracePass.cs.slang:45-74 (main). Runs one of two modes,
// selected by the per-dispatch push constant restirPtArgs.mode, and is
// dispatched twice per frame by DxvkForkReSTIRPTRayQuery:
//
//   (a) TRACE. Trace an indirect path from the primary G-buffer surface, write
//       float4(L, packed terminal descriptor) to the parity buffer, and paint L
//       into DEBUG_VIEW_RESTIR_PT_TRACE.
//
//   (b) REPLAY-VERIFY. Read the parity buffer back, re-trace the same path
//       through restirPtTraceReplayPath from its (srcPixel, srcFrame) identity,
//       and paint max|L - L_replayed| * 1e6 into DEBUG_VIEW_RESTIR_PT_REPLAY_DELTA.
//
// Running (b) as a *second dispatch of the same shader* rather than inline is
// deliberate: any per-invocation state a called helper keeps (notably the
// resolver's static offset-555 RNG, resolve.slangh:753) restarts identically in
// a fresh invocation, so the two runs are genuinely comparable. It also means
// the replay path exercises byte-for-byte the code the phase 3+ resampling
// passes will call.
//
// Pass/fail: the replay-delta view must be pure BLACK. The x1e6 scale means a
// single float ULP of divergence lights a pixel up visibly.

#define RAB_HAS_CURRENT_GBUFFER 1
#define ATMOSPHERE_AVAILABLE

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_bindings.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/noise.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/texture.slangh"
#include "rtx/utility/sampling.slangh"
#include "rtx/utility/packing.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/concept/ray/ray.slangh"
#include "rtx/concept/surface/surface.slangh"
#include "rtx/concept/surface_material/surface_material.slangh"
#include "rtx/concept/light/light.slangh"
#include "rtx/algorithm/resolve.slangh"
#include "rtx/algorithm/visibility.slangh"
#include "rtx/algorithm/lighting.slangh"

#include "rtx/algorithm/fork_restir_pt/restir_pt_trace_core.slangh"

// Note: named restirPtArgs rather than the usual `cb` because the common
// bindings already bind RaytraceArgs under that name.
layout(push_constant)
ConstantBuffer<ForkReSTIRPTArgs> restirPtArgs;

// Packs the descriptor a replay needs to reproduce a base path's termination
// into the parity entry's .w lane. Bit-exact through float32 because the payload
// is two small integers reinterpreted, never arithmetic.
float restirPtPackTerminalDescriptor(uint pathLength, uint terminalKind)
{
  return asfloat((pathLength & 0xFFFFu) | (terminalKind << 16));
}

void restirPtUnpackTerminalDescriptor(float packed, out uint pathLength, out uint terminalKind)
{
  const uint bits = asuint(packed);
  pathLength = bits & 0xFFFFu;
  terminalKind = bits >> 16;
}

// Row-major index into the parity buffer. The host allocates a 16x16-block
// padded pixel count (the ReSTIR GI reservoir sizing pattern,
// rtx_restir_gi_rayquery.cpp:323-331), which is always >= resolution.x *
// resolution.y, so a plain row-major stride is in range.
uint restirPtParityIndex(uint2 pixel)
{
  return pixel.y * uint(cb.camera.resolution.x) + pixel.x;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const Camera camera = cb.camera;

  if (threadId.x >= uint(camera.resolution.x) || threadId.y >= uint(camera.resolution.y))
  {
    return;
  }

  const uint parityIndex = restirPtParityIndex(threadId);
  const bool isReplayMode = (restirPtArgs.mode == FORK_RESTIR_PT_MODE_REPLAY_VERIFY);

  // ports ReSTIRPTPass/TracePass.cs.slang:52-53 -- skip pixels with no valid
  // primary hit. There the test is against the V-buffer; here it is the
  // resolved G-buffer surface the ReSTIR GI reuse passes anchor on.
  const RAB_Surface primarySurface = RAB_GetGBufferSurface(int2(threadId), false);

  if (!RAB_IsSurfaceValid(primarySurface))
  {
    if (!isReplayMode)
    {
      RestirPtParityBuffer[parityIndex] = float4(0.0f, 0.0f, 0.0f, restirPtPackTerminalDescriptor(0u, kRestirPtTerminalNone));
    }

    // Note: leave the debug view untouched on background pixels so the parity
    // view reads as black there rather than as an unwritten garbage texel.
    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TRACE || cb.debugView == DEBUG_VIEW_RESTIR_PT_REPLAY_DELTA)
    {
      storeInDebugView(threadId, vec3(0.0f));
    }

    return;
  }

  const uint maxBounces = max(cb.restirPtMaxBounces, 1u);

  if (!isReplayMode)
  {
    uint pathLength = 0u;
    uint terminalKind = kRestirPtTerminalNone;

    const vec3 L = restirPtTracePath(
      u16vec2(threadId), cb.frameIdx, primarySurface, maxBounces, pathLength, terminalKind);

    RestirPtParityBuffer[parityIndex] = float4(L, restirPtPackTerminalDescriptor(pathLength, terminalKind));

    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TRACE)
    {
      storeInDebugView(threadId, L);
    }

    return;
  }

  // Replay-verify.
  const float4 parityEntry = RestirPtParityBuffer[parityIndex];

  uint basePathLength = 0u;
  uint baseTerminalKind = kRestirPtTerminalNone;
  restirPtUnpackTerminalDescriptor(parityEntry.w, basePathLength, baseTerminalKind);

  uint replayPathLength = 0u;
  uint replayTerminalKind = kRestirPtTerminalNone;

  const vec3 replayedL = restirPtTraceReplayPath(
    /*dstPixel=*/ u16vec2(threadId),
    /*srcPixel=*/ u16vec2(threadId),
    /*srcFrameIdx=*/ cb.frameIdx,
    primarySurface, maxBounces,
    int(basePathLength), baseTerminalKind,
    replayPathLength, replayTerminalKind);

  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_REPLAY_DELTA)
  {
    // Radiance divergence, amplified so one ULP is visible.
    const vec3 delta = abs(parityEntry.xyz - replayedL);
    float error = max(delta.x, max(delta.y, delta.z)) * 1.0e6f;

    // A NaN on either side is a hard failure, not a small one -- paint it
    // saturated so it cannot be mistaken for noise.
    if (!isValidValue(parityEntry.xyz) || !isValidValue(replayedL))
    {
      error = 1.0e6f;
    }

    // Control-flow divergence is just as fatal as value divergence and is
    // otherwise invisible when both radiances happen to be zero, so fold the
    // terminal-descriptor mismatch in at full brightness too.
    if (replayPathLength != basePathLength || replayTerminalKind != baseTerminalKind)
    {
      error = 1.0e6f;
    }

    storeInDebugView(threadId, vec3(error));
  }
}
