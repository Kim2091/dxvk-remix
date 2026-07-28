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

// ReSTIR PT (fork) -- trace pass.
//
// ports ReSTIRPTPass/TracePass.cs.slang:45-74 (main). Two entirely separate
// jobs, which the host keeps mutually exclusive:
//
// (1) REAL MODE (RESTIR_PT_FLAG_MODE_ACTIVE, i.e. IntegrateIndirectMode::ReSTIRPT).
//     Runs in integrate_indirect's dispatch slot and produces what the rest of
//     the frame expects from that slot:
//       - primary-selected pixels: one initially-resampled path per pixel in a
//         RestirPtReservoir, plus vec4(0, 0, 0, firstSegmentLength) into
//         IndirectRadianceHitDistance. The radiance lanes are deliberately ZERO:
//         integrate_nee routes that texel straight into the primary indirect
//         channel it owns (integrate_nee.comp.slang:110-131), and the fork final
//         shading pass adds F * weight into the same channel afterwards. Writing
//         radiance here as well would double-count -- PT would read about 2x GI.
//         The alpha lane still carries the NRD hit-T, which is why it is written
//         at all.
//       - secondary-selected (PSR) pixels: no reservoir, full radiance into
//         IndirectRadianceHitDistance, which demodulate folds into
//         SecondaryCombined* (demodulate.comp.slang:776-798). Same structure GI
//         mode has for those pixels.
//
// (2) DEBUG MODE (rtx.restirPT.enableDebugTrace, any non-PT indirect mode).
//     The phase 1 replay-parity harness, unchanged: dispatched twice, once
//     tracing into a scratch buffer and once re-tracing every path from its
//     (srcPixel, srcFrame) identity and painting max|L - L_replayed| * 1e6 into
//     debug view 881. That view must be pure BLACK; the x1e6 scale means a single
//     float ULP of divergence lights a pixel visibly. It writes nothing else.
//
// Running the replay as a *second dispatch of the same shader* rather than
// inline is deliberate: any per-invocation state a called helper keeps (notably
// the resolver's static offset-555 RNG, resolve.slangh:753) restarts identically
// in a fresh invocation, so the two runs are genuinely comparable.

#define RAB_HAS_CURRENT_GBUFFER 1
#define ATMOSPHERE_AVAILABLE
// Closes phase 1's deviation D7. The live indirect integrator defines this
// (integrate_indirect.slang:217), which gates calcVolumetricAttenuation per
// resolve segment and on miss (resolve.slangh:340-344, 408-414); without it the
// kernel applied fog to its NEE shadow rays (visibility.slangh:512-515 does it
// unconditionally) but not to its path segments. Pure math from cb.volumeArgs --
// no textures, no RNG, so replay-safe.
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1

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
#include "rtx/algorithm/integrator_helpers.slangh"
// Note: nee_cache.h, not integrator.slangh. Phase 1's deviation D2 kept the
// integrator out to avoid dragging in the NEE-cache binding set; phase 2 needs
// exactly that set (and nothing else from the integrator), so the leaf header
// comes in directly and the three fork-local leaf twins stay.
#include "rtx/algorithm/nee_cache.h"

#include "rtx/algorithm/fork_restir_pt/restir_pt_trace_core.slangh"

// Note: named restirPtArgs rather than the usual `cb` because the common
// bindings already bind RaytraceArgs under that name.
layout(push_constant)
ConstantBuffer<ForkReSTIRPTArgs> restirPtArgs;

// Fork-local twin of IndirectSampleData::unpack (integrator.slangh:48-55),
// reduced to the single field the continuation entry point needs. Duplicated
// rather than included for the same reason as the other leaf twins -- pulling in
// integrator.slangh would pull in the whole indirect payload. Keep in sync.
float restirPtUnpackFirstSampledSolidAnglePdf(uint encodedIndirectSampleData)
{
  return abs(unpackFloat2x16(encodedIndirectSampleData)).y;
}

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

// ---------------------------------------------------------------------------
// (1) Real mode
// ---------------------------------------------------------------------------

void restirPtRealModeMain(uint2 threadId)
{
  const uint reservoirIndex = restirPtReservoirIndex(threadId, uint2(cb.camera.resolution));

  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(threadId, SharedFlags);

  // ports integrate_indirect.slangh:116-141 -- which surface's miss matters
  // depends on which surface was selected for integration.
  const bool primaryGBufferMiss = isGBufferMiss(threadId, PrimaryConeRadius);
  const bool secondaryGBufferMiss = isGBufferMiss(threadId, SecondaryConeRadius);
  const bool gbufferMissOfASelectedIntegrationSurface =
    geometryFlags.primarySelectedIntegrationSurface ? primaryGBufferMiss : secondaryGBufferMiss;

  // Every lane writes a reservoir, always. An unwritten lane is stale memory
  // from an earlier frame, and final shading would happily shade it.
  RestirPtReservoir reservoir;
  reservoir.init();

  vec3 L = vec3(0.0f);
  float firstSegmentLength = 0.0f;

  if (gbufferMissOfASelectedIntegrationSurface)
  {
    IndirectRadianceHitDistance[threadId] = vec4(vec3(0.0f), kEmptyPixelHitDistance);
    RestirPtReservoirBuffer[reservoirIndex] = reservoir;

    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TRACE ||
        cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_WEIGHT ||
        cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_VALID ||
        cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_F)
    {
      storeInDebugView(threadId, vec3(0.0f));
    }

    return;
  }

  // With secondary bounces disabled the indirect integrator produces nothing
  // (integrator_indirect.slangh:1425). Matching that exactly -- a zero bounce cap
  // makes the trace loop a no-op and leaves the reservoir empty -- is cheaper and
  // more honest than forcing the flag on for this pass alone.
  const uint maxBounces = cb.enableSecondaryBounces ? max(cb.restirPtMaxBounces, 1u) : 0u;

  if (geometryFlags.primarySelectedIntegrationSurface)
  {
    const RAB_Surface primarySurface = RAB_GetGBufferSurface(int2(threadId), false);

    if (RAB_IsSurfaceValid(primarySurface))
    {
      uint pathLength = 0u;
      uint terminalKind = kRestirPtTerminalNone;

      L = restirPtTracePath(
        u16vec2(threadId), cb.frameIdx, primarySurface, maxBounces,
        reservoir, firstSegmentLength, pathLength, terminalKind);
    }

    // Radiance lanes ZERO -- see the write contract in the file header.
    IndirectRadianceHitDistance[threadId] = vec4(vec3(0.0f), firstSegmentLength);
  }
  else
  {
    // PSR / secondary-selected: continue the ray integrate_direct already
    // sampled and stashed (integrator_direct.slangh:207-213). No reservoir --
    // ReSTIR GI contributes nothing to these pixels either.
    const uint mediumMaterialIndex = SharedMediumMaterialIndex[threadId];

    const vec4 rayOriginDirection = RayOriginDirection[threadId];
    const vec4 throughputConeRadius = ThroughputConeRadius[threadId];

    const vec3 rayOrigin = rayOriginDirection.xyz;
    const vec3 rayDirection = signedOctahedralToSphereDirection(
      snorm2x16ToFloat2x32(floatBitsToUint(rayOriginDirection.w)));
    const vec3 throughput = throughputConeRadius.xyz;
    const float coneRadius = throughputConeRadius.w;

    const float firstSampledSolidAnglePdf =
      restirPtUnpackFirstSampledSolidAnglePdf(FirstSampledLobeData[threadId]);

    // ports integrator_indirect.slangh:1349-1356 -- a zero pdf or a dead
    // throughput means integrate_direct produced no valid indirect ray.
    if (firstSampledSolidAnglePdf > 0.0f && any(throughput > vec3(0.0f)))
    {
      L = restirPtTraceContinuationPath(
        u16vec2(threadId), cb.frameIdx,
        rayOrigin, rayDirection, throughput, coneRadius,
        firstSampledSolidAnglePdf, mediumMaterialIndex, geometryFlags.insideMedium,
        geometryFlags.portalSpace, geometryFlags.objectMask,
        geometryFlags.firstSampledLobeIsSpecular,
        maxBounces, firstSegmentLength);
    }

    // Note: not sanitized here. The indirect integrator sanitizes only the
    // primary-selected signal (integrator_indirect.slangh:1478-1483); the
    // secondary one is sanitized in demodulate, after direct lighting is folded
    // in. Same division of labour.
    IndirectRadianceHitDistance[threadId] = vec4(L, firstSegmentLength);
  }

  RestirPtReservoirBuffer[reservoirIndex] = reservoir;

  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TRACE)
  {
    storeInDebugView(threadId, L);
  }
  else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_WEIGHT)
  {
    storeInDebugView(threadId, vec3(reservoir.weight));
  }
  else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_VALID)
  {
    // M is pinned to 1 by PathBuilder::finalize whenever any candidate was
    // streamed in, and stays 0 when none was -- so this reads as a
    // "did this pixel find any light at all" mask.
    storeInDebugView(threadId, vec3(reservoir.M, reservoir.weight > 0.0f ? 1.0f : 0.0f, 0.0f));
  }
  else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_F)
  {
    storeInDebugView(threadId, reservoir.F);
  }
}

// ---------------------------------------------------------------------------
// (2) Debug mode -- the phase 1 replay-parity harness
// ---------------------------------------------------------------------------

void restirPtDebugModeMain(uint2 threadId, bool isReplayMode)
{
  // Note: the parity buffer keeps the phase 1 row-major-padded scheme, which is
  // the same one restirPtReservoirIndex implements.
  const uint parityIndex = restirPtReservoirIndex(threadId, uint2(cb.camera.resolution));

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
    float firstSegmentLength = 0.0f;
    RestirPtReservoir reservoir;

    const vec3 L = restirPtTracePath(
      u16vec2(threadId), cb.frameIdx, primarySurface, maxBounces,
      reservoir, firstSegmentLength, pathLength, terminalKind);

    RestirPtParityBuffer[parityIndex] = float4(L, restirPtPackTerminalDescriptor(pathLength, terminalKind));

    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TRACE)
    {
      storeInDebugView(threadId, L);
    }
    else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_WEIGHT)
    {
      storeInDebugView(threadId, vec3(reservoir.weight));
    }
    else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_RESERVOIR_F)
    {
      storeInDebugView(threadId, reservoir.F);
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

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const Camera camera = cb.camera;

  if (threadId.x >= uint(camera.resolution.x) || threadId.y >= uint(camera.resolution.y))
  {
    return;
  }

  // The two modes are mutually exclusive and the host enforces it (mode wins);
  // the resources the other mode needs are null-bound, so neither branch may
  // touch the other's buffers.
  if ((cb.restirPtFlags & RESTIR_PT_FLAG_MODE_ACTIVE) != 0u)
  {
    restirPtRealModeMain(threadId);
    return;
  }

  restirPtDebugModeMain(threadId, restirPtArgs.mode == FORK_RESTIR_PT_MODE_REPLAY_VERIFY);
}
