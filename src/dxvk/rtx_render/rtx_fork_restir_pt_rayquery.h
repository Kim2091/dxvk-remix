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
  // PHASE 1 SCOPE. This is not yet an indirect-illumination mode; it is the
  // replay-discipline harness that the rest of the port is gated on. When
  // enabled it dispatches the fork trace kernel twice per frame -- once to trace
  // and once to replay-and-compare -- writing only its own parity buffer and,
  // when the matching debug view is selected, the debug-view texture. It runs
  // *in addition to* the normal frame and changes nothing about the rendered
  // image, so the default path stays bit-identical with the toggle off (the pass
  // is not dispatched at all, and RtxPass frees its buffer).
  //
  // Reference: ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/ReSTIRPTPass.cpp
  // (BSD-3-Clause, see ThirdPartyLicenses.txt).
  class DxvkForkReSTIRPTRayQuery : public RtxPass {

  public:

    DxvkForkReSTIRPTRayQuery(DxvkDevice* device);
    ~DxvkForkReSTIRPTRayQuery() = default;

    void dispatch(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void showImguiSettings();

    // Fills the ReSTIR PT block of RaytraceArgs. Called once per frame from
    // RtxContext::updateRaytraceArgsConstantBuffer; the per-dispatch replay-mode
    // bit is patched in by dispatch() itself.
    void setRaytraceArgs(RaytraceArgs& constants) const;

  private:
    virtual bool isEnabled() const override;
    virtual void createDownscaledResource(Rc<DxvkContext>& ctx, const VkExtent3D& downscaledExtent) override;
    virtual void releaseDownscaledResource() override;

    // Phase-1-only scratch: float4 per padded pixel, {radiance.xyz, packed
    // terminal descriptor}. Written by the trace dispatch, read back by the
    // replay-verify dispatch. Retired in phase 2 when the reservoir carries the
    // path identity instead.
    Rc<DxvkBuffer> m_parityBuffer;

    RTX_OPTION("rtx.restirPT", bool, enableDebugTrace, false,
               "Dispatches the fork-owned ReSTIR PT trace kernel as a debug-only pass, in addition to the normal frame. "
               "Produces no image change on its own -- inspect it via the 'ReSTIR PT Trace' and 'ReSTIR PT Replay Parity Delta' debug views. "
               "This is the phase 1 harness for the ReSTIR PT port; it is not an indirect illumination mode yet.");
    RTX_OPTION("rtx.restirPT", bool, replayParityTest, true,
               "Runs the replay-verify dispatch after the trace dispatch, re-tracing every path from its (source pixel, source frame) identity. "
               "The 'ReSTIR PT Replay Parity Delta' debug view must be pure black; any lit pixel means the random-replay discipline is broken. "
               "Costs a second full trace, so it can be turned off once parity is established.");
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
  };

}
