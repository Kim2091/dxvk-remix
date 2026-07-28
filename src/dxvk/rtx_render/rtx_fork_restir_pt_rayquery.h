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
    //
    // Runs the temporal pass and then the spatial reuse rounds first, inside the
    // same mode gate, so phases 3 and 4 need no new rtx_context.cpp touchpoint.
    void dispatchFinalShading(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void showImguiSettings();

    // Fills the ReSTIR PT block of RaytraceArgs. Called once per frame from
    // RtxContext::updateRaytraceArgsConstantBuffer; the per-dispatch trace/replay
    // mode selector is a push constant instead (see ForkReSTIRPTArgs).
    //
    // Takes the context because one of its flags -- gradient-based history
    // validation -- is only meaningful when the RTXDI gradient pass actually ran
    // this frame, which is a cross-pass question (see usesDenoiserGradient).
    void setRaytraceArgs(RtxContext& ctx, RaytraceArgs& constants) const;

    // True when this pass needs RTXDI's lighting-change gradients this frame.
    // Read by DxvkRtxdiRayQuery::getEnableDenoiserGradient, which is what decides
    // whether the gradient passes run at all; without this the gradient texture is
    // never produced in ReSTIR PT mode under DLSS-RR and temporal validation would
    // silently do nothing. Exists as a predicate rather than as three public option
    // accessors so the enablement rule lives in one place.
    bool usesDenoiserGradient() const;

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

    // One page of the reservoir allocation, as a bindable slice. `page` is NOT
    // taken modulo anything -- with temporal reuse the page roles rotate across
    // frames and a silent wrap would alias the history page onto a working one.
    // Callers pass an index the schedule below produced.
    DxvkBufferSlice reservoirPageSlice(uint32_t page) const;

    // Reservoir pages this configuration needs: 3 with temporal reuse (two working
    // pages plus a history page that must survive the frame boundary), 2 without.
    uint32_t requiredPageCount() const;

    // Spatial reuse rounds actually dispatched this frame (0 when reuse is off).
    uint32_t activeSpatialRounds() const;

    // True when the temporal pass will be dispatched this frame.
    bool temporalReuseActiveThisFrame() const;

    // THE PAGE SCHEDULE. With H = m_historyPage, the two pages that are NOT the
    // history page are this frame's working pair, in index order. The trace pass
    // writes the first of them; the spatial rounds ping-pong between them; final
    // shading reads whichever was written last, and that page becomes the next
    // frame's history. Returns the working pair.
    void workingPages(uint32_t& firstPage, uint32_t& secondPage) const;

    // The page final shading reads this frame, i.e. the last one written.
    uint32_t finalPage() const;

    // One temporal reuse dispatch, in place on the trace pass's page, reading the
    // history page. Called from dispatchFinalShading before the spatial rounds.
    void dispatchTemporalReuse(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    // 1..N spatial reuse rounds, ping-ponging the two working reservoir pages.
    // Called from dispatchFinalShading, before the final shading dispatch and
    // inside the same mode gate.
    void dispatchSpatialReuse(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    // TWO or THREE pages of one RestirPtReservoir per padded pixel. Allocated only
    // in the ReSTIR PT indirect mode.
    //
    // Spatial reuse cannot run in place -- pixel A reads neighbour B's reservoir
    // while B's own thread overwrites it -- so round r reads one working page and
    // writes the other. The pages are bound as buffer SLICES of this one
    // allocation, which is why no shader needs page arithmetic: the reference
    // swaps whole buffers per round (ReSTIRPTPass.cpp:1665-1675) and ReSTIR GI
    // uses pages with shader-side indexing (rtx_restir_gi_rayquery.cpp:321-331);
    // slices are the cheap middle.
    //
    // Phase 4 adds the third page, and with it the ROTATION: last frame's final
    // page is this frame's history page and is untouched until the temporal pass
    // has read it, which is what lets this port skip the reference's per-frame
    // copyResource (ReSTIRPTPass.cpp:770-771). One page is ~191 MiB at 1080p
    // render resolution, so the third one is not free and the option's description
    // says so.
    Rc<DxvkBuffer> m_reservoirBuffer;

    // Size in bytes of ONE reservoir page, i.e. the slice stride.
    VkDeviceSize m_reservoirPageSize = 0;

    // Pages actually allocated. Tracked because the needed count changes with an
    // option, not only with an activation transition -- see onFrameBegin.
    uint32_t m_allocatedPageCount = 0;

    // The page holding last frame's final reservoirs. Advanced at the end of every
    // frame the pass runs in indirect mode.
    uint32_t m_historyPage = 0;

    // False for the first frame after any (re)allocation: there is no history to
    // reuse yet, and the pages hold undefined memory. The reference's
    // `skipTemporalReuse = mReservoirFrameCount == 0` (ReSTIRPTPass.cpp:681).
    bool m_historyValid = false;

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
    RTX_OPTION("rtx.restirPT", bool, enableSpatialReuse, true,
               "Enables the ReSTIR PT spatial reuse pass: each pixel resamples a few screen-space neighbours' paths onto its own surface "
               "through the reconnection shift, with pairwise resampling MIS. Reduces indirect noise at the cost of one extra pass "
               "(2 x neighbourCount visibility rays and surface reconstructions per pixel per round) and one extra reservoir page of VRAM.\n"
               "Turning it OFF must reproduce the phase 2 image exactly - the trace kernel is unaffected by this toggle.\n"
               "The expected artifact when it is ON is radius-scale correlated splotches re-rolled every frame; judge it THROUGH the denoiser, "
               "not on the raw debug views.");
    RTX_OPTION("rtx.restirPT", int, spatialNeighborCount, 1,
               "Number of screen-space neighbours each pixel considers per spatial reuse round. The reference's default is 3; this ships at 1 "
               "because each neighbour costs two reconnection shifts (two visibility rays and two surface reconstructions) and the PT mode's "
               "performance headroom over ReSTIR GI is only about 10%. Raise it and watch a Tracy capture of the 'ReSTIR PT Spatial Reuse' zone.");
    RTX_OPTION("rtx.restirPT", float, spatialRadius, 20.0f,
               "Screen-space radius, in pixels, of the disk each spatial neighbour is drawn from. The reference's default. "
               "A larger radius spreads the correlation artifact thinner but pulls in less similar surfaces (which the feature-based rejection "
               "then discards, wasting the sample).");
    RTX_OPTION("rtx.restirPT", int, spatialRounds, 1,
               "Number of spatial reuse rounds per frame, each a separate dispatch that ping-pongs the reservoir pages. "
               "More rounds widen the effective reuse footprint at linear cost.");
    RTX_OPTION("rtx.restirPT", float, jacobianRejectionThreshold, 0.0f,
               "Discards a shifted sample when max(J, 1/J) exceeds 1 + this, where J is the shift Jacobian. Zero or negative disables the test, "
               "which is how the reference ships it; the reference's value when enabled is 10. Discarding this way stays unbiased because the "
               "sample is removed from its own MIS weight as well as from the estimate.");
    RTX_OPTION("rtx.restirPT", float, shiftParityThreshold, 0.02f,
               "Error threshold for the 'ReSTIR PT Shift Parity (Self)' debug view (887). Below this reads BLACK; above it, the remaining range is painted "
               "linearly across the full channel, so BRIGHTNESS IS MAGNITUDE.\n"
               "The error is a SYMMETRIC relative difference against the larger of the two magnitudes, per channel: 0.33 means the shift and the stored "
               "integrand differ by 1.5x, 0.5 by 2x, 0.8 by 5x, and 1.0 means one of them is zero. It is bounded in [0,1] and uses no luma anywhere, so - "
               "unlike the metric this replaced - it can neither manufacture a huge reading out of a dim reservoir nor inflate blue-dominant sky-lit paths by "
               "up to 8.8x.\n"
               "Blue pixels are reservoirs too dim on both sides to be worth judging: excluded, not passing. Set the threshold to 0 to see the raw field when "
               "hunting a residual rather than gating on one.");
    RTX_OPTION("rtx.restirPT", bool, spatialSkyReconnection, true,
               "Lets a path whose FIRST scatter left the scene be reused, by treating its escape direction as the reconnection vertex. "
               "The reference only does this under its hybrid shift, so turning this off reproduces strict reference behaviour - at the cost of "
               "nearly all outdoor reuse, since a sky escape one bounce off the primary surface is the dominant path class outdoors. "
               "No brightness change either way; if there is one, the escape MIS convention is wrong.");
    RTX_OPTION("rtx.restirPT", bool, enableTemporalReuse, true,
               "Enables the ReSTIR PT temporal reuse pass: each pixel resamples the previous frame's reservoir at the reprojected pixel "
               "through the same reconnection shift the spatial pass uses, with Talbot resampling MIS over the two candidates.\n"
               "This is what makes indirect light STABLE rather than merely less noisy - it is also the first part of this port that can "
               "boil, because it is the first with cross-frame feedback. If low-frequency luminance pulsing appears on large flat surfaces, "
               "lower Temporal History Length first.\n"
               "Costs two reconnection shifts and two visibility rays per pixel (about one spatial neighbour's worth) AND A THIRD RESERVOIR "
               "PAGE OF VRAM - roughly +191 MiB at 1080p render resolution, since last frame's reservoirs have to survive the frame boundary. "
               "Turning it off gives that page back.");
    RTX_OPTION("rtx.restirPT", float, temporalHistoryLength, 20.0f,
               "M-cap: the previous frame's confidence weight is clamped to this multiple of the current reservoir's before resampling. "
               "The reference's default (ReSTIRPTPass.h). This is THE stability-versus-lag knob - raise it for a quieter image that responds "
               "more slowly to change, lower it for the opposite.\n"
               "Raising it must not change the average brightness of anything. If it does, the M-cap or a Talbot MIS denominator is wrong.");
    RTX_OPTION("rtx.restirPT", bool, temporalReprojectionJitter, true,
               "Stochastically rounds the reprojected pixel to one of the pixels the exact reprojected position touches, instead of always "
               "taking the containing one. Decorrelates a slowly panning camera from repeatedly resampling the same history lane, which is "
               "one of the cheaper defences against temporal correlation artifacts.");
    RTX_OPTION("rtx.restirPT", bool, validateLightingChange, true,
               "Discards a temporal sample when RTXDI's lighting-change gradients say the light reaching its reconnection vertex has changed. "
               "Twin of rtx.restirGI.validateLightingChange. With this off, switching an emitter off or moving it makes the indirect response "
               "GHOST for roughly Temporal History Length frames - that A/B is how you check the validation is actually firing (watch the blue "
               "channel of the 'ReSTIR PT Temporal Reprojection' debug view).\n"
               "Only effective where the reconnection vertex is on screen; gradients do not exist for anything else.");
    RTX_OPTION("rtx.restirPT", float, lightingValidationThreshold, 0.05f,
               "Gradient magnitude above which a temporal sample is treated as stale. Lower discards more history (more responsive, noisier); "
               "higher discards less (quieter, more ghosting).");
    RTX_OPTION("rtx.restirPT", bool, neeCacheTaskFeedback, true,
               "Inserts NEE cache tasks at emissive hits inside the ReSTIR PT kernel, as the indirect integrator does. "
               "This is how the cache DISCOVERS emissive triangles; integrate_nee's own feedback only reinforces existing candidates. "
               "Turning it off must not change the kernel's estimate at all - only, over minutes, the quality of integrate_nee's first-bounce sampling.");
  };

}
