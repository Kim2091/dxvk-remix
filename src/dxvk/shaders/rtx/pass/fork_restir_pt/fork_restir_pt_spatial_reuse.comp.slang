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
*   ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/SpatialReuse.cs.slang
* Licensed under BSD-3-Clause (see ThirdPartyLicenses.txt, "ReSTIR PT" entry).
*/

// ReSTIR PT (fork) -- spatial reuse, PAIRWISE resampling MIS, reconnection shift.
//
// ports ReSTIRPTPass/SpatialReuse.cs.slang:64-78 (rejection), :151-183 (setup),
// :340-411 (the pairwise loop) and :534-538 (the output guards and store). NOT
// ported: the Talbot / Constant MIS variants, Bekaert path reuse, the small-window
// and N-rooks patterns, multi-spp, and the whole NRD block at :552-634 -- phase 2's
// final shading pass owns denoiser routing and runs after this.
//
// One dispatch per round. Round r reads reservoir page r % 2 and writes page
// (r + 1) % 2; the host binds the two as buffer slices of one allocation, so this
// shader keeps addressing through restirPtReservoirIndex and never learns about
// paging. The round index arrives in the push constant's `mode` field.
//
// EVERY LANE WRITES THE OUTPUT PAGE, ALWAYS. A lane that skips reuse (G-buffer
// miss, PSR / secondary-selected, invalid surface) copies its input reservoir
// through verbatim. Final shading reads whichever page the last round wrote, so a
// lane that wrote nothing would be shaded from stale memory -- the same hazard the
// trace pass's "every lane writes a reservoir, always" comment names.

#define RAB_HAS_CURRENT_GBUFFER 1
// Matches the trace pass: visibility rays and the resolver fold volumetric
// attenuation, which is what makes the shift's connection segment comparable with
// a natively traced one (phase 2 deviation D7).
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_spatial_bindings.slangh"

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
#include "rtx/algorithm/resolve.slangh"
#include "rtx/algorithm/visibility.slangh"

#include "rtx/algorithm/fork_restir_pt/restir_pt_shift.slangh"

// Note: named restirPtArgs rather than the usual `cb` because the common bindings
// already bind RaytraceArgs under that name.
layout(push_constant)
ConstantBuffer<ForkReSTIRPTArgs> restirPtArgs;

// ---------------------------------------------------------------------------
// THE ENHANCED SECTION 3 SEAM -- neighbour selection and acceptance
// ---------------------------------------------------------------------------
//
// These two functions are the ONLY place the reuse loop decides which pixels it
// talks to. ReSTIR PT Enhanced's paired / reciprocal spatial reuse replaces
// exactly this pair with a pairing texture computed in a prepass; keeping them
// factored (rather than inlined into the loop, which is how the reference writes
// them) is what makes that a drop-in later. Do not fold them into the loop.

// Replaces getNextNeighborPixel (SpatialReuse.cs.slang:131-139), whose Default
// pattern reads a precomputed 8192-entry R2 low-discrepancy disk texture from a
// random start index (ReSTIRPTPass.cpp:804-826). A uniform disk sample from the
// murmur3 stream is used instead: the texture is a variance nicety rather than a
// correctness feature, it costs a binding plus a host-side texture, and Enhanced 3
// deletes the whole component anyway.
//
// Edge handling mirrors ReSTIR GI's (ResamplingFunctions.slangh:1053-1055),
// reimplemented here rather than called because that helper also folds in GI's
// own frame-index dither and 4-pixel quantisation.
int2 restirPtSelectSpatialNeighbor(uint2 pixel, float radiusPixels, inout RAB_RandomSamplerState rng)
{
  const float u1 = RAB_GetNextRandom(rng);
  const float u2 = RAB_GetNextRandom(rng);

  // Uniform over the disk: radius * sqrt(u1), angle 2*pi*u2.
  const float radius = radiusPixels * sqrt(saturate(u1));
  const float angle = twoPi * u2;

  int2 neighborPixel = int2(pixel) + int2(round(radius * cos(angle)), round(radius * sin(angle)));

  const int2 boundary = int2(cb.camera.resolution) - 1;

  neighborPixel.x = neighborPixel.x < 0 ? -neighborPixel.x
    : (neighborPixel.x > boundary.x ? 2 * boundary.x - neighborPixel.x : neighborPixel.x);
  neighborPixel.y = neighborPixel.y < 0 ? -neighborPixel.y
    : (neighborPixel.y > boundary.y ? 2 * boundary.y - neighborPixel.y : neighborPixel.y);

  return neighborPixel;
}

// ports isValidGeometry (SpatialReuse.cs.slang:71-78), which the reference enables
// by default (ReSTIRPTPass.h:181, gFeatureBasedRejection). The thresholds are the
// reference's and are deliberately NOT knobs: it hardcodes them, and Enhanced 4
// rethinks connectibility wholesale rather than tuning these.
//
// Two additions with dxvk-remix precedent, both from the ReSTIR GI spatial pass
// (ResamplingFunctions.slangh:1128-1131): the surface-validity test, and the
// portal-space equality test -- reusing across a ray portal boundary would
// connect two different spaces.
static const float kRestirPtSpatialNormalThreshold = 0.5f;
static const float kRestirPtSpatialDepthThreshold = 0.1f;

bool restirPtIsAcceptableNeighbor(RAB_Surface centralSurface, RAB_Surface neighborSurface)
{
  if (!RAB_IsSurfaceValid(neighborSurface))
  {
    return false;
  }

  if (RAB_GetSurfacePortalSpace(centralSurface) != RAB_GetSurfacePortalSpace(neighborSurface))
  {
    return false;
  }

  if (dot(RAB_GetSurfaceNormal(centralSurface), RAB_GetSurfaceNormal(neighborSurface)) < kRestirPtSpatialNormalThreshold)
  {
    return false;
  }

  const float centralDepth = RAB_GetSurfaceLinearDepth(centralSurface);
  const float neighborDepth = RAB_GetSurfaceLinearDepth(neighborSurface);

  return abs(centralDepth - neighborDepth) < kRestirPtSpatialDepthThreshold * centralDepth;
}

// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (threadId.x >= uint(cb.camera.resolution.x) || threadId.y >= uint(cb.camera.resolution.y))
  {
    return;
  }

  const uint2 resolution = uint2(cb.camera.resolution);
  const uint reservoirIndex = restirPtReservoirIndex(threadId, resolution);

  // ports :157-158 -- the central reservoir is loaded once and kept, because the
  // pairwise loop needs BOTH the running destination reservoir and an untouched
  // copy of the canonical sample.
  const RestirPtReservoir centralReservoir = RestirPtReservoirInput[reservoirIndex];

  const bool gBufferMiss = isGBufferMiss(threadId, PrimaryConeRadius);
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(threadId, SharedFlags);
  const RAB_Surface centralSurface = RAB_GetGBufferSurface(int2(threadId), false);

  // ports :166. Copy-through cases, per the write discipline in the file header.
  // PSR / secondary-selected pixels bypass reservoirs entirely (the trace pass
  // leaves theirs empty and their light travels through
  // IndirectRadianceHitDistance), so reuse has nothing to do for them.
  const bool canReuse =
    ((cb.restirPtFlags & RESTIR_PT_FLAG_SPATIAL_REUSE) != 0u) &&
    !gBufferMiss &&
    geometryFlags.primarySelectedIntegrationSurface &&
    RAB_IsSurfaceValid(centralSurface);

  if (!canReuse)
  {
    RestirPtReservoirOutput[reservoirIndex] = centralReservoir;

    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_SPATIAL_REUSE ||
        cb.debugView == DEBUG_VIEW_RESTIR_PT_SHIFT_PARITY)
    {
      storeInDebugView(threadId, vec3(0.0f));
    }

    return;
  }

  const uint roundIndex = restirPtArgs.mode;

  // Resampling randomness comes from the murmur3-backed RAB_RandomSamplerState,
  // NOT the fork's dimension-addressed blue noise stream: resampling decisions
  // never need replaying, and the codebase warns that a table-based blue noise RNG
  // lacks the entropy resampling needs (RtxdiApplicationBridge.slangh:344). Pass
  // ids 24+ are clear of RTXDI's 0-6, integrate_nee's 7 and the path builder's 23.
  RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId, cb.frameIdx, 24u + roundIndex);

  // --- The self-shift parity harness (debug view 887) -----------------------
  // Shift the pixel's own reservoir onto its OWN surface. A correct reconnection
  // shift must then reproduce F exactly with a Jacobian of exactly 1, before any
  // neighbour is involved -- so this view is black if and only if the shift math
  // and the pdf/eval conventions are right. It short-circuits the reuse loop
  // (and copies the reservoir through) so the reading is never contaminated by
  // neighbour selection or MIS.
  //
  // FORK DEVIATION from plan-phase3 step 4, which specifies evalVisibility =
  // false on the grounds that a self-connection ray is degenerate. It is not: with
  // dst == src the connection segment is the path's ORIGINAL first segment, from
  // the primary vertex to the reconnection vertex, and tracing it is what supplies
  // the segment attenuation that the trace kernel folded into prefixThp and that
  // the shift otherwise omits. With visibility off this view would read non-black
  // everywhere fog or alpha-tested geometry attenuates a first segment -- i.e.
  // most of FNV outdoors -- for a reason that has nothing to do with the shift
  // math it exists to test. An occluded self-connection (which should never
  // happen) makes the shift return zero, and the shiftable test below then reads
  // the pixel as black rather than as a false failure.
  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_SHIFT_PARITY)
  {
    vec3 error = vec3(0.0f);

    if (centralReservoir.weight > 0.0f)
    {
      float selfJacobian = 0.0f;
      const vec3 selfIntegrand = restirPtComputeShiftedIntegrandReconnection(
        selfJacobian, centralSurface, centralSurface, centralReservoir, /*evalVisibility=*/ true);

      // Only reservoirs the shift can actually express are judged: a stored path
      // whose reconnection data is absent (no rc hit and not a sky reconnection)
      // legitimately shifts to zero and would otherwise read as a false failure.
      const bool shiftable = any(selfIntegrand > vec3(0.0f)) || selfJacobian > 0.0f;

      if (shiftable)
      {
        const vec3 integrandDelta = abs(selfIntegrand - centralReservoir.F);
        const float relativeScale = max(1e-4f, RestirPtReservoir::toScalar(centralReservoir.F));

        error = vec3(
          max(integrandDelta.x, max(integrandDelta.y, integrandDelta.z)) / relativeScale * 100.0f,
          abs(selfJacobian - 1.0f) * 100.0f,
          0.0f);
      }
    }

    RestirPtReservoirOutput[reservoirIndex] = centralReservoir;
    storeInDebugView(threadId, error);

    return;
  }

  // --- Pairwise resampling MIS ----------------------------------------------
  // ports :168-172 -- pairwise starts from an EMPTY reservoir, not from the loaded
  // one. Accumulating into the loaded reservoir would count the canonical sample
  // twice (once implicitly, once through the explicit canonical merge below).
  RestirPtReservoir dstReservoir;
  dstReservoir.init();

  const int neighborCount = int(cb.restirPtSpatialNeighborCount);
  const float neighborRadius = cb.restirPtSpatialRadius;

  int validNeighborCount = 0;
  float canonicalWeight = 1.0f;

  // ports :345-401.
  for (int i = 0; i < neighborCount; ++i)
  {
    const int2 neighborPixel = restirPtSelectSpatialNeighbor(threadId, neighborRadius, rng);

    const RAB_Surface neighborSurface = RAB_GetGBufferSurface(neighborPixel, false);

    if (!restirPtIsAcceptableNeighbor(centralSurface, neighborSurface))
    {
      continue;
    }

    const RestirPtReservoir neighborReservoir =
      RestirPtReservoirInput[restirPtReservoirIndex(uint2(neighborPixel), resolution)];

    ++validNeighborCount;

    // (a) ports :372-379. Shift the CANONICAL sample onto the NEIGHBOUR, to find
    // how well the neighbour's own domain could have produced it. That is what
    // decrements the canonical sample's MIS weight.
    //
    // Note the divisor is the CONFIGURED neighbour count, not validNeighborCount --
    // transcribed from the reference as-is; the deferred /(validNeighborCount + 1)
    // after the loop is what normalises the partition.
    float prefixJacobian = 0.0f;
    const vec3 prefixIntegrand = restirPtComputeShiftedIntegrandReconnection(
      prefixJacobian, neighborSurface, centralSurface, centralReservoir, /*evalVisibility=*/ true);

    const float prefixApproxPdf = RestirPtReservoir::computeWeight(prefixIntegrand, false) * prefixJacobian;

    canonicalWeight += 1.0f;

    if (prefixApproxPdf > 0.0f)
    {
      canonicalWeight -= prefixApproxPdf * neighborReservoir.M /
        (prefixApproxPdf * neighborReservoir.M +
         centralReservoir.M * RestirPtReservoir::computeWeight(centralReservoir.F, false) / float(neighborCount));
    }

    // (b) ports :381-400. Shift the NEIGHBOUR onto the central pixel. forceMerge
    // means "hypothetically selected": tempDstReservoir comes back holding the
    // shifted integrand together with the neighbour's M and weight, which is
    // exactly what the pairwise weight below is computed from.
    RestirPtReservoir tempDstReservoir = dstReservoir;

    float dstJacobian = 0.0f;
    const bool possibleToBeSelected = restirPtShiftAndMergeReservoir(
      dstJacobian, centralSurface, tempDstReservoir, neighborSurface, neighborReservoir,
      /*evalVisibility=*/ true, rng, /*misWeight=*/ 1.0f, /*forceMerge=*/ true);

    float neighborWeight = 0.0f;

    if (possibleToBeSelected)
    {
      const float neighborPHat =
        RestirPtReservoir::computeWeight(neighborReservoir.F, false) / dstJacobian;

      neighborWeight = neighborPHat * neighborReservoir.M /
        (neighborPHat * neighborReservoir.M +
         RestirPtReservoir::computeWeight(tempDstReservoir.F, false) * centralReservoir.M / float(neighborCount));

      // ports :397. A degenerate Jacobian can still make this NaN or infinite even
      // though the shift itself succeeded; a single one of those poisons the
      // reservoir and then the denoiser history.
      if (isnan(neighborWeight) || isinf(neighborWeight))
      {
        neighborWeight = 0.0f;
      }
    }

    restirPtMergeReservoirWithResamplingMIS(
      tempDstReservoir.F, dstJacobian, dstReservoir, tempDstReservoir, rng, neighborWeight);
  }

  // ports :403 -- the canonical sample, shifted onto itself, hence Jacobian 1.
  restirPtMergeReservoirWithResamplingMIS(
    centralReservoir.F, 1.0f, dstReservoir, centralReservoir, rng, canonicalWeight);

  // ports :405-409. The division is the DEFERRED pairwise normalisation: the
  // pairwise weights above form a partition of unity scaled by (k + 1), and
  // dropping this brightens the whole image by roughly that factor.
  if (dstReservoir.weight > 0.0f)
  {
    dstReservoir.finalizeGRIS();
    dstReservoir.weight /= float(validNeighborCount + 1);
  }

  // ports :534-535.
  if (dstReservoir.weight < 0.0f)
  {
    dstReservoir.weight = 0.0f;
  }

  if (isnan(dstReservoir.weight) || isinf(dstReservoir.weight))
  {
    dstReservoir.weight = 0.0f;
  }

  // ports :538.
  RestirPtReservoirOutput[reservoirIndex] = dstReservoir;

  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_SPATIAL_REUSE)
  {
    vec3 color = dstReservoir.F * dstReservoir.weight;

    if (any(isnan(color)) || any(isinf(color)) || any(color < vec3(0.0f)))
    {
      color = vec3(0.0f);
    }

    storeInDebugView(threadId, color);
  }
  else if (cb.debugView == DEBUG_VIEW_NAN)
  {
    accumulateInDebugViewAnd(threadId, isValidValue(dstReservoir.F * dstReservoir.weight));
  }
}
