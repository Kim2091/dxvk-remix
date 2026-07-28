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
#pragma once

#include "../dxvk_format.h"
#include "../dxvk_include.h"

#include "../spirv/spirv_code_buffer.h"
#include "rtx_resources.h"
#include "rtx_options.h"

namespace dxvk {

  class RtxContext;
  class DxvkPipelineManager;

  // ReSTIR PT (Lin et al. 2022) -- fork-owned host pass.
  //
  // Runs in one of two mutually exclusive roles, decided per frame:
  //
  //  (a) INDIRECT ILLUMINATION MODE (IntegrateIndirectMode::ReSTIRPT). The trace
  //      kernel dispatches in integrate_indirect's slot -- that pass early-outs --
  //      and writes a RestirPtReservoir per pixel plus the
  //      IndirectRadianceHitDistance texel the rest of the frame expects from
  //      that slot. A second pass, final shading, then adds F * weight into the
  //      primary indirect channels at the same point in the frame ReSTIR GI's
  //      final shading occupies, so demodulate / NRD / DLSS-RR / composite are
  //      untouched. RTXDI direct lighting composes exactly as it always did.
  //
  //  (b) DEBUG HARNESS (rtx.restirPT.enableDebugTrace, in any other indirect
  //      mode). The phase 1 replay-parity test: two dispatches, trace then
  //      replay-and-compare, writing only a scratch buffer and a debug view. The
  //      rendered image is untouched.
  //
  // With neither on, the pass is not dispatched and RtxPass frees its buffers, so
  // the default path stays bit-identical.
  //
  // Reference: ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/ReSTIRPTPass.cpp
  // (BSD-3-Clause, see ThirdPartyLicenses.txt).
  class DxvkForkReSTIRPTRayQuery : public RtxPass {

  public:

    DxvkForkReSTIRPTRayQuery(DxvkDevice* device);
    ~DxvkForkReSTIRPTRayQuery() = default;

    // Dispatched from RtxContext::dispatchIntegrate, in integrate_indirect's slot
    // and BEFORE dispatchNEE (integrate_nee reads IndirectRadianceHitDistance).
    void dispatchTrace(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    // Dispatched from RtxContext beside ReSTIR GI's own dispatch: after the NRC
    // resolve and RTXDI confidence, before demodulate. No-op unless the ReSTIR PT
    // indirect mode is selected.
    void dispatchFinalShading(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void showImguiSettings();

    // Fills the ReSTIR PT block of RaytraceArgs. Called once per frame from
    // RtxContext::updateRaytraceArgsConstantBuffer; the per-dispatch trace/replay
    // mode selector is a push constant instead (see ForkReSTIRPTArgs).
    void setRaytraceArgs(RaytraceArgs& constants) const;

    // Emissive accounting policy. The A/B for the one phase 2 change that moves
    // energy: the removal of the reference's `suppressAsDirect`. Values match
    // RESTIR_PT_EMISSIVE_MIS_* in raytrace_args.h.
    enum class EmissiveMisMode : int {
      Suppress = 0,   // Reference behaviour: drop length-1 emissive as "direct".
      None = 1,       // Weight 1 everywhere: double-counts against integrate_nee.
      NeeCache = 2,   // Default: BSDF-side MIS against the NEE cache at length 1.

      Count
    };

  protected:
    virtual void onFrameBegin(Rc<DxvkContext>& ctx, const FrameBeginContext& frameBeginCtx) override;

  private:
    virtual bool isEnabled() const override;
    virtual void createDownscaledResource(Rc<DxvkContext>& ctx, const VkExtent3D& downscaledExtent) override;
    virtual void releaseDownscaledResource() override;

    // One RestirPtReservoir per padded pixel. Allocated only in the ReSTIR PT
    // indirect mode.
    Rc<DxvkBuffer> m_reservoirBuffer;

    // Debug-only scratch: float4 per padded pixel, {radiance.xyz, packed terminal
    // descriptor}. Written by the trace dispatch, read back by the replay-verify
    // dispatch. Allocated only while the debug harness is on.
    Rc<DxvkBuffer> m_parityBuffer;

    // Which of the two resource sets the current allocation is for. RtxPass only
    // (re)allocates across an activation transition, and this pass can change
    // which set it needs while staying active -- see onFrameBegin.
    bool m_allocatedForIndirectMode = false;

    RTX_OPTION("rtx.restirPT", bool, enableDebugTrace, false,
               "Dispatches the fork-owned ReSTIR PT trace kernel as a debug-only pass, in addition to the normal frame. "
               "Produces no image change on its own -- inspect it via the 'ReSTIR PT Trace' and 'ReSTIR PT Replay Parity Delta' debug views. "
               "Ignored when Indirect Illumination Mode is already ReSTIR PT, which runs the kernel for real.");
    RTX_OPTION("rtx.restirPT", bool, replayParityTest, true,
               "Runs the replay-verify dispatch after the debug trace dispatch, re-tracing every path from its (source pixel, source frame) identity. "
               "The 'ReSTIR PT Replay Parity Delta' debug view must be pure black; any lit pixel means the random-replay discipline is broken. "
               "Costs a second full trace, so it can be turned off once parity is established. Debug harness only.");
    RTX_OPTION("rtx.restirPT", int, maxBounces, 4,
               "Maximum number of indirect path vertices the ReSTIR PT trace kernel visits. "
               "Capped at 15 to match the 4-bit path length field of the reference's reservoir flags (ReSTIRPTPass/PathReservoir.slang).");
    RTX_OPTION("rtx.restirPT", bool, enableRussianRoulette, true,
               "Enables throughput-based Russian roulette inside the ReSTIR PT trace kernel. "
               "Ported from the reference's terminatePathByRussianRoulette; the draw is dimension-addressed so it never desynchronises replay.");
    RTX_OPTION("rtx.restirPT", float, specularRoughnessThreshold, 0.2f,
               "Perceptual roughness at or below which a non-diffuse lobe is classified as a specular bounce. "
               "Twin of the reference's specularRoughnessThreshold and of rtx.restirGI.virtualSampleRoughnessThreshold.");
    RTX_OPTION("rtx.restirPT", float, deltaRoughnessThreshold, 0.001f,
               "Perceptual roughness below which an opaque specular lobe is classified as a delta event. "
               "Delta vertices can never host a reconnection, so this is the knob that decides which near-mirror surfaces are replay-only.");
    RTX_OPTION("rtx.restirPT", EmissiveMisMode, emissiveMisMode, EmissiveMisMode::NeeCache,
               "How emissive-mesh light hit by BSDF sampling is accounted for. This is the A/B for the one phase 2 change that moves energy.\n"
               "Suppress (0): the reference's behaviour - light seen one scatter off the primary surface is dropped as 'direct'. "
               "Correct in the reference because ScreenSpaceReSTIR owns it there; too DIM here, because RTXDI only covers analytic lights.\n"
               "None (1): accumulate with weight 1 everywhere. Too BRIGHT, because integrate_nee's first-bounce NEE cache counts the same light.\n"
               "NEE Cache (2, default): BSDF-side MIS against the NEE cache selection pdf at the first indirect vertex, weight 1 beyond. "
               "Should sit between the other two on emissive-mesh-lit surfaces; if it does not, the selection pdf is wrong.");
    RTX_OPTION("rtx.restirPT", bool, neeCacheTaskFeedback, true,
               "Inserts NEE cache tasks at emissive hits inside the ReSTIR PT kernel, as the indirect integrator does. "
               "This is how the cache DISCOVERS emissive triangles; integrate_nee's own feedback only reinforces existing candidates. "
               "Turning it off must not change the kernel's estimate at all - only, over minutes, the quality of integrate_nee's first-bounce sampling.");
  };

}
