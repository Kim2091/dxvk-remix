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

// ReSTIR PT (fork) -- final shading.
//
// ports ReSTIRPTPass/SpatialReuse.cs.slang:534-618, minus everything that is not
// shading: no reuse (phase 3+), and no Falcor-side demodulation -- dxvk-remix
// demodulates in its own pass afterwards (demodulate.comp.slang), which is why
// this writes plain radiance into the primary indirect channels rather than
// radiance/reflectance.
//
// It occupies the exact structural slot ReSTIR GI final shading occupies
// (rtx_restir_gi_rayquery.cpp:427-459, dispatched from rtx_context.cpp:679-681):
// after integrate_nee, after the NRC resolve and RTXDI confidence, before
// demodulate. So demodulate / NRD / DLSS-RR / composite see a shape they already
// handle, unchanged.
//
// WHAT IT DOES NOT DO, deliberately (see the plan's resolved question 4):
//   - no base-reflectivity adjustment. GI final shading performs it
//     (restir_gi_final_shading.comp.slang:247-251) *because* cb.enableReSTIRGI
//     suppresses it in integrate_direct. In PT mode enableReSTIRGI is false, so
//     integrate_direct already did it (integrate_direct.slangh:188-196) and
//     doing it again would double-adjust.
//   - no BSDFFactor2 write. Composite reads that texture only under
//     cb.enableReSTIRGI (composite.comp.slang:312), false here.
//   - nothing for secondary-selected (PSR) pixels. Their indirect light travelled
//     as raw radiance through IndirectRadianceHitDistance and demodulate folds it
//     into SecondaryCombined* (demodulate.comp.slang:776-798).

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/common_bindings.slangh"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_final_shading_bindings.slangh"

#include "rtx/utility/common.slangh"
#include "rtx/utility/math.slangh"
#include "rtx/utility/debug_view_helpers.slangh"
#include "rtx/utility/gbuffer_helpers.slangh"
#include "rtx/algorithm/integrator_helpers.slangh"

// ---------------------------------------------------------------------------
// OUTLIER SUPPRESSION
// ---------------------------------------------------------------------------
//
// ReSTIR PT shipped phases 1-4 with NONE of this, which is a gap rather than a
// deliberate omission: ReSTIR GI beside it has three separate mechanisms
// (rtx.restirGI.useBoilingFilter, fireflyThreshold, and a firefly clamp on the
// initial sample before it ever enters a reservoir).
//
// WHY A RESAMPLING LOOP NEEDS THIS AND A PLAIN PATH TRACER DOES NOT. A path
// tracer's firefly is one sample in one pixel in one frame, and the denoiser
// averages it away. A reservoir does the opposite of averaging: a sample with a
// huge contribution weight is very likely to be SELECTED, so spatial reuse copies
// it to neighbours and temporal reuse re-selects it for ~temporalHistoryLength
// frames. The two together retain and spread it. That is visible as a bright spot
// that grows over seconds while the image mean stays perfectly stable -- unbiased
// and unusable at the same time, which is why no amount of MIS auditing finds it.
//
// Both mechanisms WRITE BACK to the reservoir buffer. That is the load-bearing
// part: final shading reads the page that becomes next frame's history page
// (finalPage() in rtx_fork_restir_pt_rayquery.cpp), so clearing an outlier here is
// what stops it being inherited. Suppressing only the shaded output would leave the
// firefly sitting in the reservoir, still spreading, still immortal.
//
// This is deliberately BIASED -- it removes energy. It is a variance/bias trade
// every ReSTIR implementation makes, and it must never be used to paper over an
// unbiasedness bug: if the image mean is wrong, fix the mean. These knobs exist for
// what remains after the mean is right.
//
// The boiling filter is borrowed from the RTXDI SDK by way of ReSTIR GI's
// (restir_gi_final_shading.comp.slang:62-120): take the local average of nonzero
// reservoir radiance across the workgroup and clear whatever towers over it.
#define RESTIR_PT_BOILING_GROUP_WIDTH  16
#define RESTIR_PT_BOILING_GROUP_HEIGHT 8
#define RESTIR_PT_BOILING_GROUP_THREADS (RESTIR_PT_BOILING_GROUP_WIDTH * RESTIR_PT_BOILING_GROUP_HEIGHT)

// One entry per wavefront in the group. Sized for the smallest plausible wave
// width (8) so it is correct on any hardware; the surplus is a few hundred bytes.
groupshared float s_boilingWeights[RESTIR_PT_BOILING_GROUP_THREADS / 8];
groupshared uint s_boilingCounts[RESTIR_PT_BOILING_GROUP_THREADS / 8];

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID, uint2 localIndex : SV_GroupThreadID, uint2 groupIndex : SV_GroupID)
{
  // --- The boiling filter, BEFORE any early-out ----------------------------
  //
  // Placement is a correctness requirement, not a preference. The filter needs
  // GroupMemoryBarrierWithGroupSync, which is undefined unless every thread in the
  // group reaches it, and the three early returns below are all per-pixel. So it
  // runs first, under a GROUP-UNIFORM gate: only for groups lying entirely inside
  // the image. That is the same guard ReSTIR GI uses
  // (restir_gi_final_shading.comp.slang:317) and it also keeps partial groups from
  // skewing the average with pixels that do not exist.
  const uint2 groupCorner = groupIndex * uint2(RESTIR_PT_BOILING_GROUP_WIDTH, RESTIR_PT_BOILING_GROUP_HEIGHT);
  const bool groupFullyInside =
    all(groupCorner + uint2(RESTIR_PT_BOILING_GROUP_WIDTH, RESTIR_PT_BOILING_GROUP_HEIGHT) - 1 < uint2(cb.camera.resolution));

  if ((cb.restirPtFlags & RESTIR_PT_FLAG_BOILING_FILTER) != 0u && groupFullyInside)
  {
    const uint filterReservoirIndex = restirPtReservoirIndex(threadIndex, uint2(cb.camera.resolution));
    RestirPtReservoir filterReservoir = RestirPtReservoirBuffer[filterReservoirIndex];

    const vec3 filterRadiance = filterReservoir.F * filterReservoir.weight;
    const bool filterValid =
      filterReservoir.weight > 0.0f && isValidValue(filterRadiance) && all(filterRadiance >= vec3(0.0f));
    const float filterLuminance = filterValid ? calcBt709Luminance(filterRadiance) : 0.0f;

    // Average of the NONZERO reservoirs only -- counting the empty ones would drag
    // the average down and make the filter progressively more aggressive the
    // darker the neighbourhood is.
    float waveWeight = WaveActiveSum(filterLuminance);
    uint waveCount = WaveActiveCountBits(filterValid && filterLuminance > 0.0f);

    const uint linearThreadIndex = localIndex.x + localIndex.y * RESTIR_PT_BOILING_GROUP_WIDTH;
    const uint waveIndex = linearThreadIndex / WaveGetLaneCount();

    if (WaveIsFirstLane())
    {
      s_boilingWeights[waveIndex] = waveWeight;
      s_boilingCounts[waveIndex] = waveCount;
    }

    GroupMemoryBarrierWithGroupSync();

    // Reduce the per-wavefront partials into one group average, in one wavefront.
    const uint waveTotal = (RESTIR_PT_BOILING_GROUP_THREADS + WaveGetLaneCount() - 1) / WaveGetLaneCount();

    if (linearThreadIndex < waveTotal)
    {
      waveWeight = WaveActiveSum(s_boilingWeights[linearThreadIndex]);
      waveCount = WaveActiveSum(s_boilingCounts[linearThreadIndex]);

      if (linearThreadIndex == 0)
      {
        s_boilingWeights[0] = (waveCount > 0u) ? (waveWeight / float(waveCount)) : 0.0f;
      }
    }

    GroupMemoryBarrierWithGroupSync();

    const float averageLuminance = s_boilingWeights[0];

    // Clear the reservoir outright rather than rescaling its radiance. Rescaling
    // would leave the offending PATH in the reservoir with a merely smaller weight,
    // and reuse would keep selecting and re-spreading it; the point is to break the
    // loop, not to dim one frame of it.
    if (averageLuminance > 0.0f &&
        filterLuminance > averageLuminance * cb.restirPtBoilingFilterThreshold)
    {
      filterReservoir.init();
      RestirPtReservoirBuffer[filterReservoirIndex] = filterReservoir;
    }
  }

  if (any(threadIndex >= cb.camera.resolution))
  {
    return;
  }

  const bool gBufferMiss = isGBufferMiss(threadIndex, PrimaryConeRadius);

  if (gBufferMiss)
  {
    // ports :324-331's twin in the dxvk world -- the same empty write ReSTIR GI
    // final shading and integrate_nee both make on miss, so denoisers never see
    // uninitialised data in miss regions.
    PrimaryIndirectDiffuseLobeRadianceHitDistance[threadIndex] = vec4(vec3(0.0f), kEmptyPixelHitDistance);
    PrimaryIndirectSpecularLobeRadianceHitDistance[threadIndex] = vec4(vec3(0.0f), kEmptyPixelHitDistance);

    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_FINAL_SHADING)
    {
      storeInDebugView(threadIndex, vec3(0.0f));
    }

    return;
  }

  const GeometryFlags geometryFlags = geometryFlagsReadFromGBuffer(threadIndex, SharedFlags);

  if (!geometryFlags.primarySelectedIntegrationSurface)
  {
    // PSR pixels: nothing to add, and nothing to clear either -- integrate_nee
    // already wrote the (zero) primary indirect channels for them.
    if (cb.debugView == DEBUG_VIEW_RESTIR_PT_FINAL_SHADING)
    {
      storeInDebugView(threadIndex, vec3(0.0f));
    }

    return;
  }

  const uint reservoirIndex = restirPtReservoirIndex(threadIndex, uint2(cb.camera.resolution));
  RestirPtReservoir reservoir = RestirPtReservoirBuffer[reservoirIndex];

  // ports :534-535 and :540. A negative, NaN or infinite contribution weight is
  // a bug upstream, but it must never reach the denoiser: NRD propagates a
  // single NaN across the whole history.
  if (reservoir.weight < 0.0f || isnan(reservoir.weight) || isinf(reservoir.weight))
  {
    reservoir.weight = 0.0f;
  }

  vec3 radiance = reservoir.F * reservoir.weight;

  if (any(isnan(radiance)) || any(isinf(radiance)) || any(radiance < vec3(0.0f)))
  {
    radiance = vec3(0.0f);
  }

  // The absolute firefly clamp, the per-pixel companion to the neighbourhood-
  // relative boiling filter above. Twin of the clamp ReSTIR GI applies to its
  // initial sample (restir_gi_temporal_reuse.comp.slang:138-140), including the
  // reasoning in its comment: a resampling loop tolerates a much higher threshold
  // than a plain integrator does, because it produces a stable result rather than
  // a one-frame spike -- so setting this too low DARKENS the image instead of
  // cleaning it. Uses the same fireflyFiltering helper (common.slangh:37), which
  // scales the colour by the clamped/original luminance ratio and so preserves hue.
  if (cb.restirPtFireflyThreshold > 0.0f)
  {
    const vec3 clampedRadiance = fireflyFiltering(radiance, cb.restirPtFireflyThreshold);

    // WRITE THE CLAMP BACK INTO THE RESERVOIR, not just into the shaded output.
    // This page becomes next frame's history; clamping only the output would leave
    // the unclamped path in the reservoir, still selectable, still spreading
    // through spatial reuse.
    //
    // AND `rcVertexIrradiance` IS THE FIELD THAT MATTERS -- scaling F alone made
    // this whole block a no-op for the reuse loop, which is how it first shipped.
    // A shift NEVER READS the stored F: it rebuilds the candidate's magnitude from
    // the frozen suffix, `dstF1 / dstPDF1 * (dstF2 / dstPDF2 * rcVertexIrradiance)`
    // (restir_pt_shift.slangh:829, and :696 for the escape branch). F is the RIS
    // target and the directly-shaded radiance; rcVertexIrradiance is what every
    // future frame re-derives from. Clamping only F therefore re-clamps the output
    // every frame -- so it measures as working -- while the full-amplitude path
    // sails on through the history page untouched. Both are scaled here, by the
    // same factor, so the target, the radiance and the suffix stay consistent.
    //
    // (The boiling filter above does not have this hazard: init() clears the whole
    // reservoir including the suffix, which is why it works and this did not.)
    if (any(clampedRadiance != radiance) && any(radiance > vec3(0.0f)))
    {
      const float clampScale = calcBt709Luminance(clampedRadiance) / max(calcBt709Luminance(radiance), 1e-9f);

      reservoir.F *= clampScale;
      reservoir.rcVertexIrradiance *= clampScale;
      RestirPtReservoirBuffer[reservoirIndex] = reservoir;

      radiance = clampedRadiance;
    }
  }

  // ports :572-573. The lobe class of the bounce BEFORE the reconnection vertex
  // is the primary scatter's class (the trace kernel records it at path length
  // 1), which is exactly the diffuse/specular split the denoiser channels want.
  // The delta refinement is the reference's: a delta reflection is not a
  // "specular primary hit" for denoising purposes, it is a mirror, and its
  // radiance belongs in the diffuse channel's residual rather than in a specular
  // lobe NRD will try to reproject.
  const bool isSpecularBounce = reservoir.pathFlags.decodeIsSpecularBounce(true);
  const bool isDeltaEvent = reservoir.pathFlags.decodeIsDeltaEvent(true);
  const bool isSpecularPrimaryHit = isSpecularBounce && !isDeltaEvent;

  // ADD IN PLACE. integrate_nee owns these texels: it routed the trace pass's
  // hit-T out of IndirectRadianceHitDistance into the right channel's .a and
  // added its own first-bounce NEE-cache light to .rgb
  // (integrate_nee.comp.slang:110-131, :250-251). Both must survive, so the
  // alpha lane is preserved untouched -- the same thing GI final shading does at
  // restir_gi_final_shading.comp.slang:381-382.
  if (isSpecularPrimaryHit)
  {
    vec4 output = PrimaryIndirectSpecularLobeRadianceHitDistance[threadIndex];
    output.xyz += radiance;
    PrimaryIndirectSpecularLobeRadianceHitDistance[threadIndex] = output;
  }
  else
  {
    vec4 output = PrimaryIndirectDiffuseLobeRadianceHitDistance[threadIndex];
    output.xyz += radiance;
    PrimaryIndirectDiffuseLobeRadianceHitDistance[threadIndex] = output;
  }

  if (cb.debugView == DEBUG_VIEW_RESTIR_PT_FINAL_SHADING)
  {
    storeInDebugView(threadIndex, radiance);
  }
  else if (cb.debugView == DEBUG_VIEW_NAN)
  {
    accumulateInDebugViewAnd(threadIndex, isValidValue(radiance));
  }
}
