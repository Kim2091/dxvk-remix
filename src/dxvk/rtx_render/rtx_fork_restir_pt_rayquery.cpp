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
*   ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/ReSTIRPTPass.cpp
* Licensed under BSD-3-Clause (see ThirdPartyLicenses.txt, "ReSTIR PT" entry).
*/
#include "rtx_fork_restir_pt_rayquery.h"
#include "dxvk_device.h"
#include "rtx_shader_manager.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/concept/light/light_types.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_imgui.h"

#include <rtx_shaders/fork_restir_pt_trace.h>
#include <rtx_shaders/fork_restir_pt_spatial_reuse.h>
#include <rtx_shaders/fork_restir_pt_final_shading.h>

namespace dxvk {

  // THE MIS-WEIGHT-1 TRIPWIRE.
  //
  // The ReSTIR PT kernel hands its NEE samples a multiple-importance-sampling
  // weight of exactly 1, on the grounds that its NEE pool (sampleLightBasicRIS,
  // lighting.slangh:90-193) draws only from the ANALYTIC light array -- sphere,
  // rect, disk, cylinder, distant -- none of which has scene geometry a BSDF ray
  // could hit. NEE and BSDF sampling are therefore disjoint domains and no MIS is
  // needed. The full argument is in restir_pt_trace_core.slangh, under
  // "THE MIS-WEIGHT-1 ASSUMPTION".
  //
  // If a sixth light type is ever added, that argument has to be re-checked
  // BEFORE it ships: a geometry-backed emitter in this pool turns those weights
  // into a silent energy double-count with no failing test and no visible tell
  // beyond "PT mode looks brighter than GI mode on mesh-lit surfaces".
  //
  // This is the cheapest available runtime-free guard. It cannot detect a
  // mesh-light converter that adds to the pool without adding a type -- read the
  // shader-side block for that case.
  static_assert(lightTypeCount == 5,
                "ReSTIR PT: the light type set changed. Re-read THE MIS-WEIGHT-1 ASSUMPTION in "
                "rtx/algorithm/fork_restir_pt/restir_pt_trace_core.slangh before shipping this - "
                "the kernel's NEE MIS weights are 1 only because the analytic light pool contains "
                "no geometry a BSDF ray can hit.");

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    // Path length cap. Matches the reference's kMaximumPathLength
    // (ReSTIRPTPass/StaticParams.slang) and the 4-bit path-length field of the
    // reservoir flags (ReSTIRPTPass/PathReservoir.slang:21-140) -- exceeding it
    // here would silently truncate on store.
    constexpr int kRestirPtMaxBouncesLimit = 15;

    // Reservoir pages in the single reservoir allocation. Two, because spatial
    // reuse round r reads page r % 2 and writes page (r + 1) % 2; reusing in place
    // is a data race by construction. Phase 4 decides whether temporal history
    // wants a third page or swap bookkeeping.
    constexpr uint32_t kRestirPtReservoirPageCount = 2u;

    RemixGui::ComboWithKey<DxvkForkReSTIRPTRayQuery::EmissiveMisMode> emissiveMisModeCombo {
      "Emissive MIS Mode",
      RemixGui::ComboWithKey<DxvkForkReSTIRPTRayQuery::EmissiveMisMode>::ComboEntries { {
          {DxvkForkReSTIRPTRayQuery::EmissiveMisMode::Suppress, "Suppress (reference)",
            "The reference implementation's behaviour: emissive light seen one scatter off the primary surface is dropped as 'direct'.\n"
            "Correct there because ScreenSpaceReSTIR DI owns it; too DIM here, because RTXDI covers analytic lights only.\n"
            "This is the 'old behaviour' leg of the A/B."},
          {DxvkForkReSTIRPTRayQuery::EmissiveMisMode::None, "No MIS",
            "Accumulate emissive hits with weight 1 everywhere. Too BRIGHT on emissive-mesh-lit surfaces, because integrate_nee's\n"
            "first-bounce NEE cache sampling is counting the same light. This is the double-count leg of the A/B."},
          {DxvkForkReSTIRPTRayQuery::EmissiveMisMode::NeeCache, "NEE Cache MIS",
            "Default. BSDF-side MIS against the NEE cache selection pdf at the first indirect vertex, weight 1 beyond it -\n"
            "the same pair the live indirect integrator applies. Should land between the other two modes; if it does not,\n"
            "the searchCandidate pdf is wrong."}
      } }
    };

    class ForkReSTIRPTTraceShader : public ManagedShader
    {
      SHADER_SOURCE(ForkReSTIRPTTraceShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_restir_pt_trace)

      BINDLESS_ENABLED()

      PUSH_CONSTANTS(ForkReSTIRPTArgs)

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        // Inputs
        TEXTURE2D(FORK_RESTIR_PT_BINDING_WORLD_SHADING_NORMAL_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_HIT_DISTANCE_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_ALBEDO_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_BASE_REFLECTIVITY_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_WORLD_POSITION_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_VIEW_DIRECTION_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_CONE_RADIUS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_POSITION_ERROR_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SUBSURFACE_DATA_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT)
        SAMPLER(FORK_RESTIR_PT_BINDING_LINEAR_WRAP_SAMPLER)
        SAMPLERCUBE(FORK_RESTIR_PT_BINDING_SKYPROBE)

        // NEE cache
        STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_NEE_CACHE)
        STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_NEE_CACHE_SAMPLE)
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_NEE_CACHE_TASK)
        RW_TEXTURE2D(FORK_RESTIR_PT_BINDING_NEE_CACHE_THREAD_TASK)
        STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_PRIMITIVE_ID_PREFIX_SUM)

        // integrate_direct -> integrate_indirect handoff (PSR continuation)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_RAY_ORIGIN_DIRECTION_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_THROUGHPUT_CONE_RADIUS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_FIRST_SAMPLED_LOBE_DATA_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_BINDING_SECONDARY_CONE_RADIUS_INPUT)

        // Inputs / Outputs
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT)

        // Outputs
        RW_TEXTURE2D(FORK_RESTIR_PT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_RESERVOIR_OUTPUT)

      END_PARAMETER()
    };

    // Spatial reuse cap. Each round is a full dispatch with 2 * neighbourCount
    // reconnection shifts per pixel, so both of these are deliberately small.
    constexpr int kRestirPtMaxSpatialNeighbors = 8;
    constexpr int kRestirPtMaxSpatialRounds = 3;

    class ForkReSTIRPTSpatialReuseShader : public ManagedShader
    {
      SHADER_SOURCE(ForkReSTIRPTSpatialReuseShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_restir_pt_spatial_reuse)

      BINDLESS_ENABLED()

      // The round index rides in ForkReSTIRPTArgs::mode -- see the DUAL USE note
      // where the struct is declared.
      PUSH_CONSTANTS(ForkReSTIRPTArgs)

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        // Inputs -- primary G-buffer
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_WORLD_SHADING_NORMAL_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_HIT_DISTANCE_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_ALBEDO_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_BASE_REFLECTIVITY_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_WORLD_POSITION_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_VIEW_DIRECTION_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_CONE_RADIUS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_POSITION_ERROR_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DATA_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT)

        // Reservoir ping-pong (two slices of one allocation)
        STRUCTURED_BUFFER(FORK_RESTIR_PT_SR_BINDING_RESERVOIR_INPUT)
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_SR_BINDING_RESERVOIR_OUTPUT)

      END_PARAMETER()
    };

    class ForkReSTIRPTFinalShadingShader : public ManagedShader
    {
      SHADER_SOURCE(ForkReSTIRPTFinalShadingShader, VK_SHADER_STAGE_COMPUTE_BIT, fork_restir_pt_final_shading)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        // Inputs
        TEXTURE2D(FORK_RESTIR_PT_FS_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(FORK_RESTIR_PT_FS_BINDING_PRIMARY_CONE_RADIUS_INPUT)
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_FS_BINDING_RESERVOIR_INPUT)

        // Inputs / Outputs
        RW_TEXTURE2D(FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_DIFFUSE_INPUT_OUTPUT)
        RW_TEXTURE2D(FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_SPECULAR_INPUT_OUTPUT)

      END_PARAMETER()
    };
  }

  DxvkForkReSTIRPTRayQuery::DxvkForkReSTIRPTRayQuery(DxvkDevice* device) : RtxPass(device) {
  }

  void DxvkForkReSTIRPTRayQuery::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    if (!isEnabled()) {
      return;
    }

    ForkReSTIRPTTraceShader::getShader();

    if (RtxOptions::useReSTIRPT()) {
      ForkReSTIRPTFinalShadingShader::getShader();
      ForkReSTIRPTSpatialReuseShader::getShader();
    }
  }

  bool DxvkForkReSTIRPTRayQuery::isEnabled() const {
    return enableDebugTrace() || RtxOptions::useReSTIRPT();
  }

  void DxvkForkReSTIRPTRayQuery::showImguiSettings() {
    const bool indirectModeActive = RtxOptions::useReSTIRPT();

    if (indirectModeActive) {
      ImGui::TextWrapped("Active as the Indirect Illumination mode. Debug views: 'ReSTIR PT Trace' (raw traced radiance), "
                         "'ReSTIR PT Final Shading' (F * weight - compare the two), plus the reservoir weight/valid/integrand views.");
    } else {
      ImGui::TextWrapped("Not the current Indirect Illumination mode. The debug trace below runs the kernel alongside the "
                         "normal frame for the replay-parity test only, and changes no pixels.");
      RemixGui::Checkbox("Enable ReSTIR PT Debug Trace", &enableDebugTraceObject());
      RemixGui::Checkbox("Replay Parity Test", &replayParityTestObject());
    }

    RemixGui::DragInt("Max Bounces", &maxBouncesObject(), 1.f, 1, kRestirPtMaxBouncesLimit, "%d", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::Checkbox("Russian Roulette", &enableRussianRouletteObject());
    RemixGui::DragFloat("Specular Roughness Threshold", &specularRoughnessThresholdObject(), 0.01f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Delta Roughness Threshold", &deltaRoughnessThresholdObject(), 0.0001f, 0.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
    emissiveMisModeCombo.getKey(&emissiveMisModeObject());
    RemixGui::Checkbox("NEE Cache Task Feedback", &neeCacheTaskFeedbackObject());

    // Phase 3. Every one of these has to be flippable live: the phase's whole
    // verification story is stare-and-toggle A/Bs (reuse on/off must be
    // energy-neutral, sky reconnection on/off must be energy-neutral, and the
    // count/radius/rounds sweep is how the correlation artifact is tuned down).
    ImGui::Separator();
    RemixGui::Checkbox("Spatial Reuse", &enableSpatialReuseObject());
    RemixGui::DragInt("Spatial Neighbor Count", &spatialNeighborCountObject(), 0.1f, 1, kRestirPtMaxSpatialNeighbors, "%d", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Spatial Radius (px)", &spatialRadiusObject(), 0.5f, 1.0f, 100.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragInt("Spatial Rounds", &spatialRoundsObject(), 0.05f, 1, kRestirPtMaxSpatialRounds, "%d", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Jacobian Rejection Threshold", &jacobianRejectionThresholdObject(), 0.1f, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::Checkbox("Sky Reconnection", &spatialSkyReconnectionObject());
    RemixGui::DragFloat("Shift Parity Threshold (view 887)", &shiftParityThresholdObject(), 0.005f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
  }

  void DxvkForkReSTIRPTRayQuery::setRaytraceArgs(RaytraceArgs& constants) const {
    constants.restirPtMaxBounces = static_cast<uint32_t>(std::clamp(maxBounces(), 1, kRestirPtMaxBouncesLimit));
    constants.restirPtSpecularRoughnessThreshold = specularRoughnessThreshold();
    constants.restirPtDeltaRoughnessThreshold = deltaRoughnessThreshold();

    uint32_t flags = 0u;

    if (enableRussianRoulette()) {
      flags |= RESTIR_PT_FLAG_ENABLE_RUSSIAN_ROULETTE;
    }

    // Note: keyed off the option rather than isActive(), because this runs during
    // constant buffer fill and has to agree with what dispatchTrace will do.
    if (RtxOptions::useReSTIRPT()) {
      flags |= RESTIR_PT_FLAG_MODE_ACTIVE;
    }

    if (neeCacheTaskFeedback()) {
      flags |= RESTIR_PT_FLAG_NEE_CACHE_TASK_FEEDBACK;
    }

    const uint32_t misMode = std::min(static_cast<uint32_t>(emissiveMisMode()), RESTIR_PT_EMISSIVE_MIS_MODE_MASK);
    flags |= (misMode & RESTIR_PT_EMISSIVE_MIS_MODE_MASK) << RESTIR_PT_EMISSIVE_MIS_MODE_SHIFT;

    // Note: the spatial reuse flag gates the reuse SHADER only. The trace kernel
    // records reconnection data unconditionally, which is what makes "reuse off"
    // reproduce the phase 2 image rather than a differently-built reservoir.
    if (enableSpatialReuse()) {
      flags |= RESTIR_PT_FLAG_SPATIAL_REUSE;
    }

    if (spatialSkyReconnection()) {
      flags |= RESTIR_PT_FLAG_SPATIAL_SKY_RECONNECTION;
    }

    constants.restirPtFlags = flags;

    constants.restirPtSpatialNeighborCount =
      static_cast<uint32_t>(std::clamp(spatialNeighborCount(), 1, kRestirPtMaxSpatialNeighbors));
    constants.restirPtSpatialRounds =
      static_cast<uint32_t>(std::clamp(spatialRounds(), 1, kRestirPtMaxSpatialRounds));
    constants.restirPtSpatialRadius = std::max(1.0f, spatialRadius());
    constants.restirPtJacobianRejectionThreshold = jacobianRejectionThreshold();
  }

  void DxvkForkReSTIRPTRayQuery::onFrameBegin(Rc<DxvkContext>& ctx, const FrameBeginContext& frameBeginCtx) {
    RtxPass::onFrameBegin(ctx, frameBeginCtx);

    if (!isActive()) {
      return;
    }

    // RtxPass only (re)allocates across an activation transition. This pass stays
    // active across a switch between the debug harness and the real indirect mode
    // -- and those two need different buffers -- so the mode change has to force
    // its own re-allocation, or the newly needed buffer would simply be null.
    const bool indirectModeActive = RtxOptions::useReSTIRPT();
    const bool haveNeededBuffer = indirectModeActive ? (m_reservoirBuffer != nullptr) : (m_parityBuffer != nullptr);

    if (indirectModeActive != m_allocatedForIndirectMode || !haveNeededBuffer) {
      releaseDownscaledResource();
      createDownscaledResource(ctx, frameBeginCtx.downscaledExtent);
    }
  }

  void DxvkForkReSTIRPTRayQuery::createDownscaledResource(
    Rc<DxvkContext>& ctx,
    const VkExtent3D& downscaledExtent) {

    // Note: 16x16-block padded pixel count, matching the ReSTIR GI reservoir
    // sizing pattern (rtx_restir_gi_rayquery.cpp:323-331). The shaders index
    // row-major with the unpadded width (restirPtReservoirIndex), so the padding
    // is pure slack.
    constexpr uint32_t kBlockSize = 16;
    const uint32_t widthBlocks = (downscaledExtent.width + kBlockSize - 1) / kBlockSize;
    const uint32_t heightBlocks = (downscaledExtent.height + kBlockSize - 1) / kBlockSize;
    const uint32_t paddedPixels = widthBlocks * heightBlocks * kBlockSize * kBlockSize;

    DxvkBufferCreateInfo bufferInfo;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    bufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    m_allocatedForIndirectMode = RtxOptions::useReSTIRPT();

    if (m_allocatedForIndirectMode) {
      // TWO pages, ping-ponged by the spatial reuse rounds. Page stride is
      // paddedPixels * 96 B; paddedPixels is a multiple of 256 (16x16 blocks), so
      // the stride is a multiple of 24576 B and clears any plausible
      // minStorageBufferOffsetAlignment by orders of magnitude.
      m_reservoirPageSize = static_cast<VkDeviceSize>(paddedPixels) * RESTIR_PT_RESERVOIR_SIZE_BYTES;
      bufferInfo.size = m_reservoirPageSize * kRestirPtReservoirPageCount;

      m_reservoirBuffer = ctx->getDevice()->createBuffer(
        bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR PT Reservoir Buffer");
    } else {
      // float4 per pixel: radiance.xyz + packed terminal descriptor.
      bufferInfo.size = static_cast<VkDeviceSize>(paddedPixels) * 4 * sizeof(float);

      m_parityBuffer = ctx->getDevice()->createBuffer(
        bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR PT Parity Buffer");
    }
  }

  void DxvkForkReSTIRPTRayQuery::releaseDownscaledResource() {
    m_reservoirBuffer = nullptr;
    m_parityBuffer = nullptr;
    m_reservoirPageSize = 0;
  }

  DxvkBufferSlice DxvkForkReSTIRPTRayQuery::reservoirPageSlice(uint32_t page) const {
    assert(m_reservoirBuffer != nullptr && m_reservoirPageSize != 0);

    return DxvkBufferSlice(
      m_reservoirBuffer,
      static_cast<VkDeviceSize>(page % kRestirPtReservoirPageCount) * m_reservoirPageSize,
      m_reservoirPageSize);
  }

  void DxvkForkReSTIRPTRayQuery::dispatchTrace(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {

    if (!isActive()) {
      return;
    }

    // Mode-active and the debug harness are mutually exclusive, and the mode
    // wins: re-running the parity check is done by dropping back to a non-PT
    // indirect mode. Each role allocates only the buffer it needs, so an
    // unallocated buffer here means the frame is not set up for that role yet.
    const bool indirectModeActive = RtxOptions::useReSTIRPT();

    if (indirectModeActive ? (m_reservoirBuffer == nullptr) : (m_parityBuffer == nullptr)) {
      return;
    }

    ScopedGpuProfileZone(ctx, indirectModeActive ? "ReSTIR PT" : "ReSTIR PT (Debug)");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    const VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_WORLD_SHADING_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_PERCEPTUAL_ROUGHNESS_INPUT, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_HIT_DISTANCE_INPUT, rtOutput.m_primaryHitDistance.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_ALBEDO_INPUT, rtOutput.m_primaryAlbedo.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_BASE_REFLECTIVITY_INPUT, rtOutput.m_primaryBaseReflectivity.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_VIEW_DIRECTION_INPUT, rtOutput.m_primaryViewDirection.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_POSITION_ERROR_INPUT, rtOutput.m_primaryPositionError.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SHARED_SURFACE_INDEX_INPUT, rtOutput.m_sharedSurfaceIndex.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SUBSURFACE_DATA_INPUT, rtOutput.m_sharedSubsurfaceData.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT, rtOutput.m_sharedSubsurfaceDiffusionProfileData.view, nullptr);

    // Note: Clamp to edge on the probe to avoid interpolation to black on the
    // edges of the view; wrap on the dome-light sampler. Same pair the indirect
    // integrator uses (rtx_pathtracer_integrate_indirect.cpp:408-417).
    Rc<DxvkSampler> linearClampSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    Rc<DxvkSampler> linearWrapSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    ctx->bindResourceSampler(FORK_RESTIR_PT_BINDING_LINEAR_WRAP_SAMPLER, linearWrapSampler);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SKYPROBE, ctx->getResourceManager().getSkyProbe(ctx).view, nullptr);
    ctx->bindResourceSampler(FORK_RESTIR_PT_BINDING_SKYPROBE, linearClampSampler);

    // NEE cache. Bound in both roles: nee_cache.h references these globals
    // unconditionally, and the debug harness compiles the same kernel.
    Rc<DxvkBuffer> primitiveIDPrefixSumBuffer = ctx->getSceneManager().getCurrentFramePrimitiveIDPrefixSumBuffer();

    ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_NEE_CACHE, DxvkBufferSlice(rtOutput.m_neeCache, 0, rtOutput.m_neeCache->info().size));
    ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_NEE_CACHE_SAMPLE, DxvkBufferSlice(rtOutput.m_neeCacheSample, 0, rtOutput.m_neeCacheSample->info().size));
    ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_NEE_CACHE_TASK, DxvkBufferSlice(rtOutput.m_neeCacheTask, 0, rtOutput.m_neeCacheTask->info().size));
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_NEE_CACHE_THREAD_TASK, rtOutput.m_neeCacheThreadTask.view, nullptr);
    ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_PRIMITIVE_ID_PREFIX_SUM, DxvkBufferSlice(primitiveIDPrefixSumBuffer, 0, primitiveIDPrefixSumBuffer->info().size));

    // The integrate_direct handoff. Note m_indirectThroughputConeRadius ALIASES
    // m_primaryIndirectDiffuseRadiance (rtx_resources.cpp:1152) and is read
    // before anything writes the latter -- the same order integrate_indirect
    // uses, which is why its Read binding sits here and the Write binding below
    // comes last (rtx_pathtracer_integrate_indirect.cpp:472-483).
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_RAY_ORIGIN_DIRECTION_INPUT, rtOutput.m_indirectRayOriginDirection.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_FIRST_SAMPLED_LOBE_DATA_INPUT, rtOutput.m_indirectFirstSampledLobeData.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT, rtOutput.m_sharedMediumMaterialIndex.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_SECONDARY_CONE_RADIUS_INPUT, rtOutput.m_secondaryConeRadius.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_BINDING_THROUGHPUT_CONE_RADIUS_INPUT, rtOutput.m_indirectThroughputConeRadius.view(Resources::AccessType::Read), nullptr);

    if (indirectModeActive) {
      ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT, DxvkBufferSlice(nullptr, 0, 0));
      // Page 0 is always the trace pass's output; spatial round r then reads
      // page r % 2 and writes page (r + 1) % 2.
      ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_RESERVOIR_OUTPUT, reservoirPageSlice(0u));
      ctx->bindResourceView(FORK_RESTIR_PT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT, rtOutput.m_indirectRadianceHitDistance.view(Resources::AccessType::Write), nullptr);
    } else {
      // Debug harness: this pass must not touch the frame's real outputs, so the
      // two real-mode resources are null-bound (the NRC-mode precedent,
      // rtx_pathtracer_integrate_indirect.cpp:571) and the shader's mode check
      // keeps every lane away from them.
      ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT, DxvkBufferSlice(m_parityBuffer, 0, m_parityBuffer->info().size));
      ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_RESERVOIR_OUTPUT, DxvkBufferSlice(nullptr, 0, 0));
      ctx->bindResourceView(FORK_RESTIR_PT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT, nullptr, nullptr);
    }

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ForkReSTIRPTTraceShader::getShader());

    {
      ScopedGpuProfileZone(ctx, "ReSTIR PT Trace");
      ctx->setFramePassStage(RtxFramePassStage::ReSTIR_PT_Trace);

      ForkReSTIRPTArgs pushArgs = {};
      pushArgs.mode = FORK_RESTIR_PT_MODE_TRACE;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    if (indirectModeActive || !replayParityTest()) {
      return;
    }

    {
      ScopedGpuProfileZone(ctx, "ReSTIR PT Replay Verify");
      ctx->setFramePassStage(RtxFramePassStage::ReSTIR_PT_ReplayVerify);

      // Note: a second dispatch of the SAME shader, not an inline second pass.
      // Any per-invocation state a shared helper keeps (notably the resolver's
      // static offset-555 RNG, resolve.slangh:753) restarts identically in a
      // fresh invocation, which is what makes the two runs comparable at all.
      ForkReSTIRPTArgs pushArgs = {};
      pushArgs.mode = FORK_RESTIR_PT_MODE_REPLAY_VERIFY;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
  }

  uint32_t DxvkForkReSTIRPTRayQuery::activeSpatialRounds() const {
    if (!enableSpatialReuse()) {
      return 0u;
    }

    return static_cast<uint32_t>(std::clamp(spatialRounds(), 1, kRestirPtMaxSpatialRounds));
  }

  void DxvkForkReSTIRPTRayQuery::dispatchSpatialReuse(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {

    const uint32_t rounds = activeSpatialRounds();

    if (rounds == 0u) {
      return;
    }

    ScopedGpuProfileZone(ctx, "ReSTIR PT Spatial Reuse");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    const VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    // The same 13-entry primary G-buffer set the trace pass binds, in the spatial
    // slots -- RAB_GetGBufferSurface reads these by name.
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_WORLD_SHADING_NORMAL_INPUT, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_PERCEPTUAL_ROUGHNESS_INPUT, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_HIT_DISTANCE_INPUT, rtOutput.m_primaryHitDistance.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_ALBEDO_INPUT, rtOutput.m_primaryAlbedo.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_BASE_REFLECTIVITY_INPUT, rtOutput.m_primaryBaseReflectivity.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_VIEW_DIRECTION_INPUT, rtOutput.m_primaryViewDirection.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_POSITION_ERROR_INPUT, rtOutput.m_primaryPositionError.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_SHARED_SURFACE_INDEX_INPUT, rtOutput.m_sharedSurfaceIndex.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DATA_INPUT, rtOutput.m_sharedSubsurfaceData.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_SR_BINDING_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT, rtOutput.m_sharedSubsurfaceDiffusionProfileData.view, nullptr);

    for (uint32_t round = 0u; round < rounds; ++round) {
      ctx->setFramePassStage(RtxFramePassStage::ReSTIR_PT_SpatialReuse);

      // Ping-pong. Round 0 reads what the trace pass wrote (page 0). The read and
      // write slices are always different pages, so the automatic compute barriers
      // DxvkContext::dispatch emits are what serialises one round against the next.
      ctx->bindResourceBuffer(FORK_RESTIR_PT_SR_BINDING_RESERVOIR_INPUT, reservoirPageSlice(round));
      ctx->bindResourceBuffer(FORK_RESTIR_PT_SR_BINDING_RESERVOIR_OUTPUT, reservoirPageSlice(round + 1u));

      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ForkReSTIRPTSpatialReuseShader::getShader());

      ForkReSTIRPTArgs pushArgs = {};
      pushArgs.mode = round;  // dual use: the round index, see ForkReSTIRPTArgs.
      pushArgs.debugParam = std::clamp(shiftParityThreshold(), 0.0f, 1.0f);
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
  }

  void DxvkForkReSTIRPTRayQuery::dispatchFinalShading(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {

    if (!isActive() || !RtxOptions::useReSTIRPT() || m_reservoirBuffer == nullptr) {
      return;
    }

    // Spatial reuse runs here rather than from rtx_context.cpp: it sits between
    // the trace pass and final shading, both of which this class already owns, so
    // phase 3 adds no new dispatch site upstream.
    dispatchSpatialReuse(ctx, rtOutput);

    ScopedGpuProfileZone(ctx, "ReSTIR PT Final Shading");
    ctx->setFramePassStage(RtxFramePassStage::ReSTIR_PT_FinalShading);

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    const VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 8, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    ctx->bindResourceView(FORK_RESTIR_PT_FS_BINDING_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_FS_BINDING_PRIMARY_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    // Whichever page the last spatial round wrote -- page 0 when reuse is off.
    ctx->bindResourceBuffer(FORK_RESTIR_PT_FS_BINDING_RESERVOIR_INPUT, reservoirPageSlice(activeSpatialRounds()));

    // ReadWrite: integrate_nee already wrote these and its contribution has to
    // survive. Same access pattern ReSTIR GI final shading uses
    // (rtx_restir_gi_rayquery.cpp:451-452).
    ctx->bindResourceView(FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_DIFFUSE_INPUT_OUTPUT, rtOutput.m_primaryIndirectDiffuseRadiance.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindResourceView(FORK_RESTIR_PT_FS_BINDING_PRIMARY_INDIRECT_SPECULAR_INPUT_OUTPUT, rtOutput.m_primaryIndirectSpecularRadiance.view(Resources::AccessType::ReadWrite), nullptr);

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ForkReSTIRPTFinalShadingShader::getShader());
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
}
