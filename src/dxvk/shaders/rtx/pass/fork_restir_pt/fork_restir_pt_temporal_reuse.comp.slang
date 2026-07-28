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
*   ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/TemporalReuse.cs.slang
* Licensed under BSD-3-Clause (see ThirdPartyLicenses.txt, "ReSTIR PT" entry).
*/

// ReSTIR PT (fork) -- temporal reuse, TALBOT resampling MIS, reconnection shift.
//
// ports ReSTIRPTPass/TemporalReuse.cs.slang:85-143 (setup, reprojection, M-cap),
// :146-226 (the Talbot loop -- with one deliberate deviation at :192, THE
// CONSISTENT-TARGET RULE below) and :298-299 (the output guards and store). NOT ported:
// the Constant / ConstantBiased / ConstantBinary MIS branches (:230-296),
// mergeReservoirNoResampling, gNoResamplingForTemporalReuse, the hybrid shift's
// reconnectionDataBuffer reads (:168-172, :200-204 -- under the pure reconnection
// shift those are dummyRcData), and the NRD/output block, which phase 2's final
// shading pass owns and which runs after this.
//
// ONE dispatch, IN PLACE on the current reservoir page, reading the history page at
// the reprojected pixel. In-place is safe because every lane touches only its own
// index in the current page; the reference does the same (:91, :299).
//
// A LANE THAT DECLINES TO REUSE WRITES NOTHING. That is the opposite of the spatial
// pass's "every lane writes, always" discipline, and it is correct for the opposite
// reason: this pass reads and writes the SAME page, so declining to write leaves the
// reservoir the trace pass produced exactly where the spatial rounds expect it.
//
// ---------------------------------------------------------------------------
// THE TEMPORAL SOURCE SURFACE -- the one real deviation in this file
// ---------------------------------------------------------------------------
//
// A shift needs the SOURCE pixel's full primary surface: its position (for the
// geometry Jacobian) and its material plus view direction (for srcPDF1). For
// temporal reuse the source is the previous frame's surface at the reprojected
// pixel, and the reference simply has one -- it keeps a temporal V-buffer and
// re-runs loadShadingData on it (TemporalReuse.cs.slang:67-78).
//
// dxvk-remix keeps no such thing. RAB_GetGBufferSurface(pixel, true)
// (RtxdiApplicationBridge.slangh:272-284) returns THREE fields -- hitDistance, an
// octahedral shading normal and portalSpace -- out of the rg32f GBufferLast texture.
// No position, no albedo, no roughness, no view direction. A BSDF cannot be
// evaluated on it. What the frame actually carries from last frame is the primary
// world position and triangle normal (PreviousWorldPosition_WorldTriangleNormal) and
// the previous camera matrices. Materials are simply not kept.
//
// So restirPtBuildTemporalSurface takes the CURRENT pixel's surface and replaces
// its position and view direction with their previous-frame values, keeping the
// material. That is sound because of what the acceptance test below actually
// asserts: temporal reuse runs only when the previous frame's surface at prevPixel
// and this frame's surface at pixel pass a depth AND triangle-normal test, which is
// the statement "these are the same surface point". Under a passing test the two
// materials ARE the same material at the same UV, one frame apart. The residual is
// texture-LOD drift from one frame of camera motion, parallax UV drift, and animated
// material parameters -- all small, all continuous in camera motion (so none can
// produce a discontinuity), and all partially cancelling because srcPDF1 enters the
// Jacobian as a ratio against a dstPDF1 built from the same material.
//
// Position and view direction are NOT approximated. They are the two inputs the
// geometry Jacobian and the lobe-selection probabilities are most sensitive to, and
// both come from history exactly.
//
// The failure mode is the standard temporal false-accept: a reprojection that passes
// depth and normal while landing on a DIFFERENT surface with coincidentally matching
// depth and normal. Every temporal pass in this renderer has it; the gradient
// validation below is the existing mitigation.
//
// GBufferLast is deliberately NOT used despite carrying an exact previous shading
// normal: it is written by the RTXDI and ReSTIR GI SPATIAL passes
// (rtxdi_spatial_reuse.comp.slang:91,107 and restir_gi_spatial_reuse.comp.slang:54),
// neither of which is guaranteed to run in ReSTIR PT mode, so reading it would
// silently mix in a stale frame's normals whenever RTXDI direct is off.
//
// ---------------------------------------------------------------------------
// THE CONSISTENT-TARGET RULE -- the phase 4 energy-runaway fix
// ---------------------------------------------------------------------------
//
// Talbot MIS is unbiased iff, for every path, the candidate weights sum to
// exactly 1. Each weight is a ratio built from the same two target functions --
// the central pixel's and the temporal pixel's -- so the partition holds if and
// only if every appearance of a domain's target in the arithmetic is computed
// by ONE function of the path.
//
// The temporal target appears in two places: the canonical candidate's MIS
// denominator (the central -> temporal shift) and the temporal candidate's own
// numerator. The reference feeds that numerator from the STORED history
// integrand (TemporalReuse.cs.slang:192). That is sound there because its
// temporal surface is the true previous-frame surface, so storage and
// re-evaluation are the same function. Here the temporal surface is
// RECONSTRUCTED (see above), and the two can disagree: the denominator's
// re-evaluation traces its visibility ray from a rebuilt origin and evaluates a
// borrowed material, and can report zero for a path the real previous surface
// demonstrably held. When it does, the canonical weight becomes 1 while the
// temporal weight keeps its stored-value ~M_T/(M_T+M_C): the weights sum to
// nearly 2, the surplus is written into the history page, and next frame it is
// multiplied again. The fixed point of that loop is ~(1 + temporalHistoryLength)
// times the true value -- the in-game white-out this rule exists to prevent.
// (A disagreement in the opposite slot -- the temporal -> central shift failing
// -- merely zeroes the temporal candidate's weight, which loses reuse, never
// gains energy. Only the numerator/denominator split above can inflate.)
//
// The rule: THE STORED HISTORY INTEGRAND IS NEVER USED AS A TARGET VALUE. It
// remains the temporal candidate's sample payload and rides its unbiased
// contribution weight, but every target evaluation of the temporal domain --
// numerator and denominator alike -- goes through
// restirPtComputeShiftedIntegrandReconnection against the reconstructed
// temporal surface. The numerator's evaluation is the self-shift of the history
// path (THE THIRD SHIFT, in the Talbot body below). Both usages are then one
// function by construction, so the weights partition unity for ANY
// reconstruction error -- known or unknown -- and resampling stays unbiased.
//
// What reconstruction imperfection costs under this rule: reuse efficiency
// (a path the rebuilt origin cannot see gets temporal weight 0 and the pixel
// falls back to its canonical sample at weight 1, which is unbiased), plus a
// small declared DEFICIT -- never a surplus -- on paths the re-evaluation
// claims for the temporal domain that the true history could not actually
// hold (it then discounts the canonical weight with no temporal candidate to
// compensate). That set is empty for a static camera -- position, TLAS and
// material all coincide -- and is a one-pixel parallax sliver under motion,
// where the acceptance test and its jitter already churn the history. Bounded
// and convergent, unlike the surplus, which compounded without bound.

#define RAB_HAS_CURRENT_GBUFFER 1
// Matches the trace and spatial passes: visibility rays and the resolver fold
// volumetric attenuation, which is what makes the shift's connection segment
// comparable with a natively traced one (phase 2 deviation D7).
#define RESOLVER_USE_VOLUMETRIC_ATTENUATION 1

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_temporal_bindings.slangh"

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

// ---------------------------------------------------------------------------
// Reprojection
// ---------------------------------------------------------------------------
//
// Ported from ReSTIR GI's temporal pass (restir_gi_temporal_reuse.comp.slang:194-206)
// rather than from the reference, which reads a screen-space motion vector texture
// (TemporalReuse.cs.slang:119-127). This renderer's VIRTUAL motion vector already
// accounts for ray portals and PSR and is what every other temporal pass here
// consumes, so using anything else would disagree with the rest of the frame.
//
// NOT ported: ReSTIR GI's reflection reprojection (:221-277). It exists to reproject
// virtual sample positions for glossy surfaces, and this port has no virtual samples
// -- its reservoirs store a real reconnection vertex.

static const float kRestirPtTemporalDepthTolerance = 0.01f;

// Acceptance verdicts, so debug view 889 can tell them apart. Ordered by how the
// view paints them; see the debug view registration.
static const uint kRestirPtTemporalRejectNone = 0u;      // accepted
static const uint kRestirPtTemporalRejectNoHistory = 1u; // off-screen / no previous frame / not reusable
static const uint kRestirPtTemporalRejectGeometry = 2u;  // depth / normal / portal mismatch
static const uint kRestirPtTemporalRejectStale = 3u;     // gradient validation discarded it

// Builds the previous-frame source surface. See THE TEMPORAL SOURCE SURFACE block.
RAB_Surface restirPtBuildTemporalSurface(
  RAB_Surface centralSurface, MinimalSurfaceInteraction previousInteraction, float previousHitDistance)
{
  const vec3 previousCameraPosition = cameraGetPreviousWorldPosition(cb.camera);

  RAB_Surface temporalSurface = centralSurface;

  // The previous world position texture carries no position error (it is not stored
  // alongside), and the shift traces a visibility ray FROM this surface in the
  // central -> temporal direction, so a zero error would under-offset that ray and
  // invite self-intersection. The central pixel's error is the same surface's, which
  // is the whole premise of this reconstruction.
  previousInteraction.positionError = centralSurface.minimalSurfaceInteraction.positionError;

  temporalSurface.minimalSurfaceInteraction = previousInteraction;
  temporalSurface.minimalRayInteraction.viewDirection = safeNormalize(
    previousCameraPosition - previousInteraction.position,
    centralSurface.minimalRayInteraction.viewDirection);
  temporalSurface.hitDistance = previousHitDistance;
  temporalSurface.virtualWorldPosition = previousInteraction.position;

  return temporalSurface;
}

// Returns the acceptance verdict and, when accepted, the reprojected pixel and the
// reconstructed temporal source surface.
uint restirPtReproject(
  uint2 pixel,
  RAB_Surface centralSurface,
  inout RAB_RandomSamplerState rng,
  out int2 previousPixel,
  inout RAB_Surface temporalSurface)
{
  previousPixel = int2(pixel);

  const Camera camera = cb.camera;

  // ports getTemporalSearchRadius's portal guard (restir_gi_temporal_reuse.comp.slang:44-45):
  // a teleport this frame, or a pixel seen through both portal spaces at once, has
  // no meaningful correspondence with last frame.
  if (cb.teleportationPortalIndex != 0 || centralSurface.portalSpace == PORTAL_SPACE_PORTAL_COMBINED)
  {
    return kRestirPtTemporalRejectNoHistory;
  }

  const vec3 virtualMotionVector = PrimaryVirtualMotionVector[pixel].xyz;
  const vec3 previousVirtualWorldPosition = centralSurface.virtualWorldPosition + virtualMotionVector;

  vec4 previousNdc = mul(camera.prevWorldToProjection, vec4(previousVirtualWorldPosition, 1.0f));

  // Behind the previous camera: the perspective divide would fold it back onto the
  // screen at a mirrored position.
  if (previousNdc.w <= 0.0f)
  {
    return kRestirPtTemporalRejectNoHistory;
  }

  previousNdc.xyz /= previousNdc.w;

  const vec2 previousPixelCenter = cameraNDCToScreenUV(camera, previousNdc.xy) * vec2(camera.resolution);

  // FORK DEVIATION from TemporalReuse.cs.slang:126, which adds a raw [0,1) sample to
  // the reprojected coordinate -- that biases the chosen pixel half a pixel toward
  // +x/+y. Centring the jitter makes it a stochastic round to one of the pixels the
  // exact position touches, which is what the jitter is for: decorrelating a slow
  // camera pan from repeatedly resampling the same history lane.
  const vec2 jitter = ((cb.restirPtFlags & RESTIR_PT_FLAG_TEMPORAL_JITTER) != 0u)
    ? vec2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5f
    : vec2(0.0f);

  previousPixel = int2(floor(previousPixelCenter + jitter));

  if (any(previousPixel < int2(0)) || any(previousPixel >= camera.resolution))
  {
    return kRestirPtTemporalRejectNoHistory;
  }

  const MinimalSurfaceInteraction previousInteraction =
    minimalSurfaceInteractionReadFromGBuffer(previousPixel, PreviousWorldPosition_WorldTriangleNormal);

  if (!previousInteraction.isValid)
  {
    return kRestirPtTemporalRejectNoHistory;
  }

  const vec3 previousCameraPosition = cameraGetPreviousWorldPosition(camera);
  const float previousHitDistance = length(previousInteraction.position - previousCameraPosition);
  const float expectedPreviousHitDistance =
    calculateExpectedPreviousHitDistance(camera, previousVirtualWorldPosition);

  // ports restir_gi_temporal_reuse.comp.slang:204-205 -- the tolerance is widened at
  // grazing incidence, where one pixel of screen space spans a lot of depth.
  const float viewDotTriangleNormal = abs(dot(
    centralSurface.minimalRayInteraction.viewDirection,
    centralSurface.minimalSurfaceInteraction.geometryNormal));
  const float depthThreshold = kRestirPtTemporalDepthTolerance / max(viewDotTriangleNormal, 0.01f);

  if (abs(previousHitDistance - expectedPreviousHitDistance) >
      depthThreshold * max(previousHitDistance, expectedPreviousHitDistance))
  {
    return kRestirPtTemporalRejectGeometry;
  }

  // ports :206. TRIANGLE normals on both sides, not shading normals: the triangle
  // normal is what the previous-frame texture carries and what the shift's geometry
  // Jacobian is measured against.
  const float normalThreshold =
    lerp(0.995f, 0.5f, float(centralSurface.opaqueSurfaceMaterialInteraction.isotropicRoughness));

  if (dot(previousInteraction.geometryNormal, centralSurface.minimalSurfaceInteraction.geometryNormal) < normalThreshold)
  {
    return kRestirPtTemporalRejectGeometry;
  }

  temporalSurface = restirPtBuildTemporalSurface(centralSurface, previousInteraction, previousHitDistance);

  return kRestirPtTemporalRejectNone;
}

// ---------------------------------------------------------------------------
// History validity: surface remap and gradient validation
// ---------------------------------------------------------------------------

// The reservoir's rcVertexHit.surfaceIndex indexes surfaces[], WHICH IS REBUILT
// EVERY FRAME. A history reservoir's index names LAST frame's surface list, so using
// it unremapped reconstructs a reconnection vertex on whatever object happens to
// occupy that slot this frame -- a silent, arbitrarily wrong contribution, and the
// most likely single cause of "temporal reuse produces garbage after a scene change".
//
// The renderer already owns the remap: surfaceMapping[lastFrameSurfaceID], bound at
// BINDING_SURFACE_MAPPING_BUFFER by COMMON_RAYTRACING_BINDINGS (so this pass needs no
// slot for it), returning SURFACE_INDEX_INVALID when the surface is gone.
// visibility.slangh:45-60 uses it in precisely this role under usePreviousTLAS, and
// nee_cache_light.slangh:43-47 for stored hits.
//
// Returns false when the stored reconnection vertex cannot be brought forward, in
// which case the pixel simply does not reuse this frame.
bool restirPtRemapHistoryReservoir(inout RestirPtReservoir historyReservoir)
{
  if (!historyReservoir.rcVertexHit.isValid())
  {
    // A sky-escape reconnection carries no surface at all -- nothing to remap, and
    // the escape direction is frame-independent.
    return true;
  }

  const int remapped = surfaceMapping[historyReservoir.rcVertexHit.surfaceIndex];

  if (remapped < 0 || uint(remapped) == SURFACE_INDEX_INVALID)
  {
    return false;
  }

  historyReservoir.rcVertexHit.surfaceIndex = uint(remapped);

  return true;
}

// ports validateSample (restir_gi_temporal_reuse.comp.slang:66-96).
//
// The problem it solves: nothing in the resampling loop ever re-evaluates a stored
// path's suffix, so a reservoir keeps re-contributing light from an emitter that has
// since been switched off or moved -- for temporalHistoryLength frames. RTXDI's
// gradient pass already measures "the lighting at this screen tile changed", so the
// fix is to project the reconnection vertex to screen space and ask.
//
// Returns true when the history is still trustworthy. An off-screen reconnection
// vertex gets no verdict and is kept, which is the same limitation ReSTIR GI
// documents -- gradients only exist for what is on screen.
//
// DISCARDING rather than ReSTIR GI's "restore the input reservoir" is the same
// outcome in this port's control flow: with no temporal candidate offered, the
// Talbot loop degenerates to the canonical sample, which IS this frame's reservoir.
// It also keeps the invalidation on one side of the MIS -- a candidate that is not
// offered is not in anyone's denominator either, so nothing is biased.
bool restirPtValidateHistory(RestirPtReservoir historyReservoir, RAB_Surface centralSurface)
{
  if ((cb.restirPtFlags & RESTIR_PT_FLAG_LIGHTING_VALIDATION) == 0u)
  {
    return true;
  }

  if (!historyReservoir.rcVertexHit.isValid())
  {
    // A sky reconnection has no screen-space position to validate, and the sky does
    // not appear in the direct-lighting gradients anyway.
    return true;
  }

  // Position only -- this needs no material, and the material half is the half that
  // does texture reads (restirPtReconstructRcVertexGeometry).
  SurfaceInteraction rcSurfaceInteraction;
  RayInteraction rcRayInteraction;

  const float rcSpreadAngle = restirPtShiftSpreadAngleFromSolidAnglePdf(
    cb.screenSpacePixelSpreadHalfAngle, historyReservoir.cachedJacobian.x);

  if (!restirPtReconstructRcVertexGeometry(
        historyReservoir.rcVertexHit,
        centralSurface.minimalSurfaceInteraction.position,
        centralSurface.minimalRayInteraction.coneRadius,
        rcSpreadAngle,
        rcSurfaceInteraction, rcRayInteraction))
  {
    // The surface survived the remap but is no longer opaque / no longer loadable.
    // Treat that as a stale history rather than reusing something unreconstructable.
    return false;
  }

  const vec3 rcPosition = rcSurfaceInteraction.position;

  // ports :70-72.
  vec4 ndc = mul(cb.camera.worldToProjectionJittered, vec4(rcPosition, 1.0f));

  if (ndc.w <= 0.0f)
  {
    return true;
  }

  ndc.xyz /= ndc.w;

  const bool isInsideScreen =
    all(ndc.xyz <= vec3(1.0f, 1.0f, 1.0f)) && all(ndc.xyz >= vec3(-1.0f, -1.0f, 0.0f));

  if (!isInsideScreen)
  {
    return true;
  }

  // ports :77-78.
  const ivec2 confidencePixel = ivec2(cameraNDCToScreenUV(cb.camera, ndc.xy) * vec2(cb.camera.resolution));
  const vec2 gradient = RtxdiGradients[uint3(uint2(confidencePixel) / RTXDI_GRAD_FACTOR, 0)];

  if (gradient.x <= cb.restirPtLightingValidationThreshold)
  {
    return true;
  }

  // ports :81-88 -- the gradient has to be ABOUT this point, not about something in
  // front of it, or every pixel behind a changing light would be invalidated too.
  const float sampleSurfaceDepth = gradient.y;
  const float reservoirDepth = length(rcPosition - cameraGetWorldPosition(cb.camera));
  const float depthTolerance = 0.05f;

  if (abs(sampleSurfaceDepth - reservoirDepth) < max(sampleSurfaceDepth, reservoirDepth) * depthTolerance)
  {
    return false;
  }

  return true;
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

  const bool gBufferMiss = isGBufferMiss(threadId, PrimaryConeRadius);
  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(threadId, SharedFlags);
  const RAB_Surface centralSurface = RAB_GetGBufferSurface(int2(threadId), false);

  // The same copy-through set the spatial pass uses, plus cb.enablePreviousTLAS.
  // That flag is RtxOptions::enablePreviousTLAS() && isPreviousFrameSceneAvailable()
  // (rtx_context.cpp:1216), i.e. exactly "the surface mapping buffer exists and the
  // previous frame was raytraced" -- without it a history reservoir's stored surface
  // index cannot be brought forward at all, so there is nothing safe to reuse.
  const bool canReuse =
    ((cb.restirPtFlags & RESTIR_PT_FLAG_TEMPORAL_REUSE) != 0u) &&
    (cb.enablePreviousTLAS != 0u) &&
    !gBufferMiss &&
    geometryFlags.primarySelectedIntegrationSurface &&
    RAB_IsSurfaceValid(centralSurface);

  uint verdict = kRestirPtTemporalRejectNoHistory;

  // The stored history target vs its re-evaluation on the reconstructed temporal
  // surface, for debug view 889's asymmetry overlay -- the one-look instrument
  // for the energy-runaway mechanism (see THE CONSISTENT-TARGET RULE). Written
  // only on accepted reprojections.
  float debugHistoryTargetStored = 0.0f;
  float debugHistoryTargetReevaluated = 0.0f;
  bool debugHistoryTargetsValid = false;

  // Note: NOT written back on any early-out. This pass reads and writes the same
  // page, so declining to write is what leaves the trace pass's reservoir intact.
  const RestirPtReservoir centralReservoir = RestirPtReservoirCurrent[reservoirIndex];
  RestirPtReservoir dstReservoir = centralReservoir;

  if (canReuse)
  {
    // Pass id 27: clear of RTXDI's 0-6, integrate_nee's 7, the path builder's 23 and
    // the spatial rounds' 24-26. Resampling randomness comes from the murmur3-backed
    // RAB_RandomSamplerState, NOT the fork's dimension-addressed blue noise -- see
    // the contract comment in restir_pt_rng.slangh.
    RAB_RandomSamplerState rng = RAB_InitRandomSampler(threadId, cb.frameIdx, 27u);

    int2 previousPixel = int2(threadId);
    RAB_Surface temporalSurface = centralSurface;

    verdict = restirPtReproject(threadId, centralSurface, rng, previousPixel, temporalSurface);

    if (verdict == kRestirPtTemporalRejectNone)
    {
      RestirPtReservoir temporalReservoir =
        RestirPtReservoirHistory[restirPtReservoirIndex(uint2(previousPixel), resolution)];

      if (!restirPtRemapHistoryReservoir(temporalReservoir))
      {
        verdict = kRestirPtTemporalRejectStale;
      }
      else if (!restirPtValidateHistory(temporalReservoir, centralSurface))
      {
        verdict = kRestirPtTemporalRejectStale;
      }
      else
      {
        // ports :93. Read BEFORE dstReservoir is re-initialised -- the M-cap and both
        // Talbot denominators are stated in terms of the CURRENT reservoir's
        // confidence weight.
        const float currentM = centralReservoir.M;

        // ports :139 -- the M-cap, the one knob that trades temporal stability
        // against lag. Without it M grows without bound and the current frame's
        // sample is eventually never selected, which is what "the image freezes" and
        // "a light change takes seconds" look like.
        temporalReservoir.M = min(cb.restirPtTemporalHistoryLength * currentM, temporalReservoir.M);

        // ports :98 -- Talbot starts from an EMPTY reservoir; the canonical sample is
        // merged explicitly below as candidate i = -1.
        dstReservoir.init();

        // ports :151-220, unrolled over the reference's two-candidate loop
        // (curSampleId = -1, prevSampleId = 0) because with exactly two candidates
        // the inner j loop has one term per branch and unrolling makes each MIS
        // denominator readable.

        // --- i = curSampleId: the canonical (current-frame) sample --------------
        {
          // ports :162-164.
          const RestirPtReservoir tempDstReservoir = centralReservoir;
          const bool possibleToBeSelected = tempDstReservoir.weight > 0.0f;

          float pSum = 0.0f;
          float pSelf = 0.0f;

          if (possibleToBeSelected)
          {
            // ports :184-186, j = curSampleId.
            pSelf = RestirPtReservoir::computeWeight(tempDstReservoir.F, false) * currentM;
            pSum = pSelf;

            // ports :206-209, j = prevSampleId. THE SECOND SHIFT: the current
            // sample projected into the TEMPORAL domain, to find how well the
            // previous frame's pixel could have produced it. Used only as a scalar
            // in this denominator -- nothing is merged from it.
            float temporalNeighborJacobian = 0.0f;
            const vec3 temporalNeighborIntegrand = restirPtComputeShiftedIntegrandReconnection(
              temporalNeighborJacobian, temporalSurface, centralSurface, tempDstReservoir,
              /*evalVisibility=*/ true);

            pSum += RestirPtReservoir::computeWeight(temporalNeighborIntegrand, false) *
              temporalNeighborJacobian * temporalReservoir.M;
          }

          // ports :214.
          const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

          // ports :219. Jacobian 1: the canonical sample shifted onto its own surface.
          restirPtMergeReservoirWithResamplingMIS(
            tempDstReservoir.F, 1.0f, dstReservoir, tempDstReservoir, rng, misWeight);
        }

        // --- i = prevSampleId: the temporal sample -----------------------------
        {
          // ports :156, :174-175. THE FIRST SHIFT: temporal -> central. forceMerge
          // means "hypothetically selected", so tempDstReservoir comes back holding
          // the shifted integrand together with the temporal reservoir's M and
          // weight -- the same idiom phase 3's pairwise loop uses.
          RestirPtReservoir tempDstReservoir = dstReservoir;

          float dstJacobian = 0.0f;
          const bool possibleToBeSelected = restirPtShiftAndMergeReservoir(
            dstJacobian, centralSurface, tempDstReservoir, temporalSurface, temporalReservoir,
            /*evalVisibility=*/ true, rng, /*misWeight=*/ 1.0f, /*forceMerge=*/ true);

          // FORK DEVIATION from :192, and the phase 4 energy-runaway fix. THE THIRD
          // SHIFT: the temporal candidate's own target, RE-EVALUATED by shifting the
          // history path onto the reconstructed temporal surface itself -- the same
          // evaluator, the same visibility origin and the same borrowed material the
          // canonical candidate's denominator term above uses. The reference reads
          // toScalar(temporalReservoir.F) here instead; see THE CONSISTENT-TARGET
          // RULE in the header for why that is a compounding energy inflation on a
          // reconstructed surface and why this restores the partition of unity.
          //
          // The self-shift's Jacobian is exactly 1 whenever the integrand is
          // non-zero (identical surfaces on both sides -- the debug view 887
          // identity); it is folded in anyway so this term keeps precisely the
          // q * J form of the denominator term it must match.
          float selfJacobian = 0.0f;
          const vec3 selfIntegrand = restirPtComputeShiftedIntegrandReconnection(
            selfJacobian, temporalSurface, temporalSurface, temporalReservoir,
            /*evalVisibility=*/ true);

          const float reevaluatedHistoryTarget =
            RestirPtReservoir::computeWeight(selfIntegrand, false) * selfJacobian;

          // Debug view 889's asymmetry overlay reads these; no other consumer.
          debugHistoryTargetStored = RestirPtReservoir::computeWeight(temporalReservoir.F, false);
          debugHistoryTargetReevaluated = reevaluatedHistoryTarget;
          debugHistoryTargetsValid = true;

          float pSum = 0.0f;
          float pSelf = 0.0f;

          if (possibleToBeSelected)
          {
            // ports :184-185, j = curSampleId.
            pSum = RestirPtReservoir::computeWeight(tempDstReservoir.F, false) * currentM;

            // ports :192-194, j == i, with the reference's stored target replaced by
            // the re-evaluated one (see above). possibleToBeSelected implies the
            // merge's weight was non-zero, which implies dstJacobian is non-zero --
            // so this division is safe by construction, not by luck.
            pSelf = reevaluatedHistoryTarget / dstJacobian * temporalReservoir.M;
            pSum += pSelf;
          }

          // ports :214.
          const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

          // ports :219. Merged unconditionally, exactly as the reference does: even a
          // failed shift contributes its confidence weight M, which is what makes the
          // running M the honest count of paths this pixel's estimate is drawn from.
          restirPtMergeReservoirWithResamplingMIS(
            tempDstReservoir.F, dstJacobian, dstReservoir, tempDstReservoir, rng, misWeight);
        }

        // ports :222-225. No division by a candidate count here, unlike the spatial
        // pass: Talbot's weights already form a partition of unity over the
        // candidates, which is the whole difference between it and pairwise MIS.
        if (dstReservoir.weight > 0.0f)
        {
          dstReservoir.finalizeGRIS();
        }

        // ports :298.
        if (dstReservoir.weight < 0.0f || isnan(dstReservoir.weight) || isinf(dstReservoir.weight))
        {
          dstReservoir.weight = 0.0f;
        }

        // ports :299.
        RestirPtReservoirCurrent[reservoirIndex] = dstReservoir;
      }
    }
  }

  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TEMPORAL_REUSE)
  {
    vec3 color = dstReservoir.F * dstReservoir.weight;

    if (any(isnan(color)) || any(isinf(color)) || any(color < vec3(0.0f)))
    {
      color = vec3(0.0f);
    }

    storeInDebugView(threadId, color);
  }
  else if (cb.debugView == DEBUG_VIEW_RESTIR_PT_TEMPORAL_REPROJECTION)
  {
    // Black is the default: no history at all (off screen, behind the previous
    // camera, no previous frame, or this pixel does not reuse).
    vec3 color = vec3(0.0f);

    if (verdict == kRestirPtTemporalRejectNone)
    {
      color = vec3(0.0f, 1.0f, 0.0f);  // accepted

      // The asymmetry overlay: YELLOW where the stored history target and its
      // re-evaluation on the reconstructed temporal surface disagree about the
      // path even being possible. Before the consistent-target fix, these were
      // exactly the pixels whose canonical MIS weight was inflated to 1 with no
      // compensating decrease -- i.e. the energy-runaway regions. After the fix
      // they are the pixels where the temporal candidate declines (unbiased,
      // just less reuse); a yellow-heavy view means the temporal surface
      // reconstruction is starving reuse, not that energy is wrong.
      if (debugHistoryTargetsValid &&
          (debugHistoryTargetStored > 0.0f) != (debugHistoryTargetReevaluated > 0.0f))
      {
        color = vec3(1.0f, 1.0f, 0.0f);
      }
    }
    else if (verdict == kRestirPtTemporalRejectGeometry)
    {
      color = vec3(1.0f, 0.0f, 0.0f);  // depth / normal / portal mismatch
    }
    else if (verdict == kRestirPtTemporalRejectStale)
    {
      color = vec3(0.0f, 0.0f, 1.0f);  // surface remap failed, or gradient validation
    }

    storeInDebugView(threadId, color);
  }
  else if (cb.debugView == DEBUG_VIEW_NAN)
  {
    accumulateInDebugViewAnd(threadId, isValidValue(dstReservoir.F * dstReservoir.weight));
  }
}
