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
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_imgui.h"

#include <rtx_shaders/fork_restir_pt_trace.h>

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    // Path length cap. Matches the reference's kMaximumPathLength
    // (ReSTIRPTPass/StaticParams.slang) and the 4-bit path-length field of the
    // reservoir flags (ReSTIRPTPass/PathReservoir.slang:21-140) that phase 2
    // introduces -- exceeding it here would silently truncate later.
    constexpr int kRestirPtMaxBouncesLimit = 15;

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

        // Inputs / Outputs
        RW_STRUCTURED_BUFFER(FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT)

      END_PARAMETER()
    };
  }

  DxvkForkReSTIRPTRayQuery::DxvkForkReSTIRPTRayQuery(DxvkDevice* device) : RtxPass(device) {
  }

  void DxvkForkReSTIRPTRayQuery::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    if (!enableDebugTrace()) {
      return;
    }

    ForkReSTIRPTTraceShader::getShader();
  }

  bool DxvkForkReSTIRPTRayQuery::isEnabled() const {
    return enableDebugTrace();
  }

  void DxvkForkReSTIRPTRayQuery::showImguiSettings() {
    RemixGui::Checkbox("Enable ReSTIR PT Debug Trace", &enableDebugTraceObject());
    RemixGui::Checkbox("Replay Parity Test", &replayParityTestObject());
    RemixGui::DragInt("Max Bounces", &maxBouncesObject(), 1.f, 1, kRestirPtMaxBouncesLimit, "%d", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::Checkbox("Russian Roulette", &enableRussianRouletteObject());
    RemixGui::DragFloat("Specular Roughness Threshold", &specularRoughnessThresholdObject(), 0.01f, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    RemixGui::DragFloat("Delta Roughness Threshold", &deltaRoughnessThresholdObject(), 0.0001f, 0.0f, 1.0f, "%.4f", ImGuiSliderFlags_AlwaysClamp);
  }

  void DxvkForkReSTIRPTRayQuery::setRaytraceArgs(RaytraceArgs& constants) const {
    constants.restirPtMaxBounces = static_cast<uint32_t>(std::clamp(maxBounces(), 1, kRestirPtMaxBouncesLimit));
    constants.restirPtFlags = enableRussianRoulette() ? RESTIR_PT_FLAG_ENABLE_RUSSIAN_ROULETTE : 0u;
    constants.restirPtSpecularRoughnessThreshold = specularRoughnessThreshold();
    constants.restirPtDeltaRoughnessThreshold = deltaRoughnessThreshold();
  }

  void DxvkForkReSTIRPTRayQuery::createDownscaledResource(
    Rc<DxvkContext>& ctx,
    const VkExtent3D& downscaledExtent) {

    // Note: 16x16-block padded pixel count, matching the ReSTIR GI reservoir
    // sizing pattern (rtx_restir_gi_rayquery.cpp:323-331). The shader indexes
    // row-major with the unpadded width, so the padding is pure slack.
    constexpr uint32_t kBlockSize = 16;
    const uint32_t widthBlocks = (downscaledExtent.width + kBlockSize - 1) / kBlockSize;
    const uint32_t heightBlocks = (downscaledExtent.height + kBlockSize - 1) / kBlockSize;
    const uint32_t paddedPixels = widthBlocks * heightBlocks * kBlockSize * kBlockSize;

    DxvkBufferCreateInfo bufferInfo;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    bufferInfo.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    // float4 per pixel: radiance.xyz + packed terminal descriptor.
    bufferInfo.size = static_cast<VkDeviceSize>(paddedPixels) * 4 * sizeof(float);

    m_parityBuffer = ctx->getDevice()->createBuffer(
      bufferInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "ReSTIR PT Parity Buffer");
  }

  void DxvkForkReSTIRPTRayQuery::releaseDownscaledResource() {
    m_parityBuffer = nullptr;
  }

  void DxvkForkReSTIRPTRayQuery::dispatch(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {

    if (!isActive() || m_parityBuffer == nullptr) {
      return;
    }

    ScopedGpuProfileZone(ctx, "ReSTIR PT (Debug)");

    const auto& numRaysExtent = rtOutput.m_compositeOutputExtent;
    const VkExtent3D workgroups = util::computeBlockCount(numRaysExtent, VkExtent3D { 16, 16, 1 });

    ctx->bindCommonRayTracingResources(rtOutput);

    // Bind once -- both dispatches read the identical resource set and differ
    // only by the push-constant mode.
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

    ctx->bindResourceBuffer(FORK_RESTIR_PT_BINDING_PARITY_INPUT_OUTPUT, DxvkBufferSlice(m_parityBuffer, 0, m_parityBuffer->info().size));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ForkReSTIRPTTraceShader::getShader());

    {
      ScopedGpuProfileZone(ctx, "ReSTIR PT Trace");
      ctx->setFramePassStage(RtxFramePassStage::ReSTIR_PT_Trace);

      ForkReSTIRPTArgs pushArgs = {};
      pushArgs.mode = FORK_RESTIR_PT_MODE_TRACE;
      ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }

    if (!replayParityTest()) {
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
}
