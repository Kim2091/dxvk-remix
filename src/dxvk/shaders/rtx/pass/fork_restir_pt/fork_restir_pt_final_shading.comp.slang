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

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint2 threadIndex : SV_DispatchThreadID)
{
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
