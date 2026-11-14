/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_tone_mapping.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx_host_calibration.h"

#include <rtx_shaders/auto_exposure_halo_meter.h>
#include <rtx_shaders/auto_exposure_halo_resolve.h>
#include "rtx_imgui.h"
#include "rtx_context.h"
#include "rtx_pre_exposure.h"
#include "rtx_photometer.h"
#include "rtx_numos.h"
#include "numos/renderer.h"

#include <algorithm>
#include <cmath>
#include "rtx_render/rtx_debug_view.h"

#include "rtx/utility/debug_view_indices.h"

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    // Halo Meter (halocedlaart Halo CE addon's Meter / MeterResolve, see the .comp.slang
    // headers for provenance).
    class AutoExposureHaloMeterShader : public ManagedShader {
      SHADER_SOURCE(AutoExposureHaloMeterShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_halo_meter)

      PUSH_CONSTANTS(ToneMappingHaloMeterArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(AUTO_EXPOSURE_HALO_COLOR_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_HALO_ALBEDO_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_HALO_LINEAR_VIEW_Z_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_HALO_SHARED_FLAGS_INPUT)
        TEXTURE2D(AUTO_EXPOSURE_HALO_PRIMARY_SURFACE_FLAGS_INPUT)
        RW_TEXTURE2D(AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(AutoExposureHaloMeterShader);

    class AutoExposureHaloResolveShader : public ManagedShader {
      SHADER_SOURCE(AutoExposureHaloResolveShader, VK_SHADER_STAGE_COMPUTE_BIT, auto_exposure_halo_resolve)

      PUSH_CONSTANTS(ToneMappingHaloMeterArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D(AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_HALO_STATE_INPUT_OUTPUT)
        RW_TEXTURE1D(AUTO_EXPOSURE_HALO_EXPOSURE_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(AutoExposureHaloResolveShader);
  }

  DxvkAutoExposure::DxvkAutoExposure(DxvkDevice* device)
  : CommonDeviceObject(device), m_vkd(device->vkd())  {
  }

  DxvkAutoExposure::~DxvkAutoExposure()  {  }

  void DxvkAutoExposure::showImguiSettings() {
    RemixGui::Checkbox("Auto Exposure", &enabledObject());
    if (enabled()) {
      ImGui::Indent();
      if (ImGui::Button("Preset: Addon-Faithful")) {
        addonFaithfulObject().setDeferred(true);
        maxEvObject().setDeferred(8.0f);
      }
      // Ranges are the addon's own slider ranges (addon.cpp).
      RemixGui::DragFloat("Exposure Target",  &targetObject(),       0.001f, 0.02f,  0.6f, "%.3f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Adaptation Speed", &speedObject(),        0.01f,  0.1f,  16.0f, "%.2f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Centre Weighting", &centreWeightObject(), 0.01f,  0.0f,   1.0f, "%.2f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Exposure Floor",   &minEvObject(),        0.1f, -24.0f,   0.0f, "%.1f EV", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Exposure Ceiling", &maxEvObject(),        0.1f,   0.0f,  24.0f, "%.1f EV", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::Checkbox("Meter Lighting (divide albedo out)", &demodulateAlbedoObject());
      RemixGui::Checkbox("Exclude View Model", &excludeViewModelObject());
      ImGui::BeginDisabled(!excludeViewModel());
      RemixGui::DragFloat("View Model Hold Threshold", &excludeViewModelHoldThresholdObject(), 0.01f, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::SetTooltipToLastWidgetOnHover("The exposure holds while the view model covers more than this share of the metered frame.");
      RemixGui::DragFloat("View Model Spike Margin", &excludeViewModelSpikeMarginObject(), 0.01f, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::SetTooltipToLastWidgetOnHover("The exposure holds for up to half a second while the view model's share of the metered frame "
                                              "jumps this far above its average over the last second, as when a sight kicks back.");
      ImGui::EndDisabled();
      RemixGui::Checkbox("Addon-Faithful Sky (no floor / hold)", &addonFaithfulObject());
      ImGui::BeginDisabled(addonFaithful());
      RemixGui::DragFloat("Sky Weight Max",       &skyWeightMaxObject(), 0.01f,  0.0f,   1.0f, "%.2f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Sky Luminance Low",    &skyLumLowObject(),    0.001f, 0.0f,  10.0f, "%.4f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Sky Luminance High",   &skyLumHighObject(),   0.001f, 0.0f,  10.0f, "%.4f",    ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Darkest Scene (Mean Floor)", &meanFloorEvObject(), 0.1f, -24.0f, 4.0f, "%.1f EV", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Min Scene Coverage",   &minCoverageObject(),  0.005f, 0.0f,   0.5f, "%.3f",    ImGuiSliderFlags_AlwaysClamp);
      ImGui::EndDisabled();
      RemixGui::DragFloat("Night Compensation",   &compensationNightStopsObject(), 0.05f, -8.0f, 8.0f, "%.2f stops", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::SetTooltipToLastWidgetOnHover("Exposure compensation curve: bias at night (scene EV100 at or below Night EV), "
                                              "fading to 0 at Day EV. About -2 to -4 keeps a night darker than mid grey.");
      RemixGui::DragFloat("Night EV",             &compensationNightEvObject(),    0.1f, -12.0f, 16.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Day EV",               &compensationDayEvObject(),      0.1f, -12.0f, 20.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
      ImGui::TextDisabled("Global Exposure (tonemapper) stays an EV offset on top of the meter.");
      RemixGui::Separator();
      ImGui::Unindent();
    }
    PreExposure::showImguiSettings();
  }

  void DxvkAutoExposure::createResources(Rc<DxvkContext> ctx) {
    if (m_exposure.image != nullptr) {
      return;
    }

    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_1D;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.stages = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_1D;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;
    viewInfo.format = desc.format;

    desc.extent = VkExtent3D{ 1, 1, 1 };

    viewInfo.format = desc.format = VK_FORMAT_R32_SFLOAT;
    // Transfer source: PreExposure reads the exposure back (rtx_pre_exposure.cpp).
    // The transfer stage and access make DXVK's barriers after clears and compute writes cover that copy.
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    desc.stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    desc.access |= VK_ACCESS_TRANSFER_READ_BIT;
    m_exposure.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "autoexposure");
    m_exposure.view = device()->createImageView(m_exposure.image, viewInfo);
    ctx->changeImageLayout(m_exposure.image, VK_IMAGE_LAYOUT_GENERAL);
    desc.stages &= ~VK_PIPELINE_STAGE_TRANSFER_BIT;
    desc.access &= ~VK_ACCESS_TRANSFER_READ_BIT;

    // Meter state: (adapted exposure, has-metered flag). The per-tile partials are sized lazily.
    // The state also holds the running average of the excluded view model share and the current spike's held time.
    viewInfo.format = desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo.usage = desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    m_haloState.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "autoexposure halo meter state");
    m_haloState.view = device()->createImageView(m_haloState.image, viewInfo);
    ctx->changeImageLayout(m_haloState.image, VK_IMAGE_LAYOUT_GENERAL);
    m_haloResetState = true;
  }

  void DxvkAutoExposure::createHaloPartials(Rc<DxvkContext> ctx, uint32_t groupsX, uint32_t groupsY) {
    if (m_haloPartials.image != nullptr && m_haloPartialsX == groupsX && m_haloPartialsY == groupsY) {
      return;
    }

    DxvkImageCreateInfo desc;
    desc.type = VK_IMAGE_TYPE_2D;
    desc.flags = 0;
    desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    desc.numLayers = 1;
    desc.mipLevels = 1;
    desc.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    desc.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    desc.tiling = VK_IMAGE_TILING_OPTIMAL;
    desc.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Two texels per tile: the meter without the excluded view model pixels, then the metered weight with them.
    desc.extent = VkExtent3D { groupsX * 2, groupsY, 1 };
    // (sum w*log, sum w, surface centre weight, total centre weight) per tile.
    desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    desc.usage = VK_IMAGE_USAGE_STORAGE_BIT;

    DxvkImageViewCreateInfo viewInfo;
    viewInfo.type = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel = 0;
    viewInfo.numLevels = 1;
    viewInfo.minLayer = 0;
    viewInfo.numLayers = 1;
    viewInfo.format = desc.format;
    viewInfo.usage = desc.usage;

    m_haloPartials.image = device()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXRenderTarget, "autoexposure halo meter partials");
    m_haloPartials.view = device()->createImageView(m_haloPartials.image, viewInfo);
    ctx->changeImageLayout(m_haloPartials.image, VK_IMAGE_LAYOUT_GENERAL);
    m_haloPartialsX = groupsX;
    m_haloPartialsY = groupsY;
  }

  void DxvkAutoExposure::dispatchHaloMeter(
    Rc<DxvkContext> ctx,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds) {
    RtxContext* rtxCtx = static_cast<RtxContext*>(ctx.ptr());

    // The addon snaps (instead of adapting) on its first frame and whenever the camera
    // teleports (> 20 units in a frame); Remix's camera-cut detection is the analogue.
    if (rtxCtx->getSceneManager().getCamera().isCameraCut()) {
      m_haloResetState = true;
    }

    VkImageSubresourceRange subRange = {};
    subRange.layerCount = 1;
    subRange.levelCount = 1;
    subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    if (m_haloResetState) {
      // Zero clears the has-metered flag too, so the resolve snaps this frame.
      VkClearColorValue clearColor = {};
      ctx->clearColorImage(m_haloState.image, clearColor, subRange);
      m_haloResetState = false;
    }

    const VkExtent3D extent = rtOutput.m_finalOutputExtent;
    const VkExtent3D groups = util::computeBlockCount(extent, VkExtent3D { AUTO_EXPOSURE_HALO_TILE, AUTO_EXPOSURE_HALO_TILE, 1 });
    createHaloPartials(ctx, groups.width, groups.height);

    NrdArgs primaryDirectNrdArgs;
    NrdArgs primaryIndirectNrdArgs;
    NrdArgs secondaryNrdArgs;
    rtxCtx->getDenoiseArgs(primaryDirectNrdArgs, primaryIndirectNrdArgs, secondaryNrdArgs);

    ToneMappingHaloMeterArgs args = {};
    args.groupsX = groups.width;
    args.groupsY = groups.height;
    args.centreWeight = centreWeight();
    args.target = target();
    args.minEv = minEv();
    args.maxEv = maxEv();
    args.primaryMissLinearViewZ = primaryDirectNrdArgs.missLinearViewZ;
    args.invPreExposure = PreExposure::inverse();
    {
      // Scene EV100 of the metered grey card: its luminance in cd/m^2 from the sky provider's calibration
      // (NUMOS's lux per unit is also its cd/m^2 per radiance unit), 1 without one.
      const numos::FrameOutputs* numosOutputs = device()->getCommon()->metaNumos().outputs();
      const float candelaPerUnit = numosOutputs && numosOutputs->luxPerUnit > 0.0f ? numosOutputs->luxPerUnit : 1.0f;
      args.evOffset = std::log2(candelaPerUnit * 100.0f / 12.5f);
      args.compensationNightStops = compensationNightStops();
      args.compensationNightEv = compensationNightEv();
      args.compensationDayEv = compensationDayEv();
    }
    args.demodulateAlbedo = demodulateAlbedo() ? 1u : 0u;
    args.excludeViewModel = excludeViewModel() ? 1u : 0u;
    args.viewModelHoldThreshold = std::clamp(excludeViewModelHoldThreshold(), 0.0f, 1.0f);
    args.viewModelSpikeMargin = std::clamp(excludeViewModelSpikeMargin(), 0.0f, 1.0f);
    if (addonFaithful()) {
      // The addon: sky at full weight whatever its brightness, no mean floor, no hold.
      args.skyWeightMax = 1.0f;
      args.skyLumLow = 0.0f;
      args.skyLumHigh = 0.0f;   // high <= low => full weight
      args.meanFloor = 0.0f;
      args.minCoverage = 0.0f;
    } else {
      args.skyWeightMax = std::clamp(skyWeightMax(), 0.0f, 1.0f);
      args.skyLumLow = std::max(skyLumLow(), 0.0f);
      args.skyLumHigh = std::max(skyLumHigh(), 0.0f);
      args.meanFloor = std::exp2(std::clamp(meanFloorEv(), -24.0f, 8.0f));
      args.minCoverage = std::clamp(minCoverage(), 0.0f, 1.0f);
    }
    {
      // The addon's stops-per-second speed as a per-frame fraction, 1 - exp(-dt * speed). dt is wall-clock
      // time (1/60 s before the first measured frame), clamped to [0, 0.25] s so a hitch does not land the
      // whole adaptation in one frame.
      float dt = frameTimeMilliseconds > 0.0f ? frameTimeMilliseconds * 0.001f : (1.0f / 60.0f);
      dt = std::clamp(dt, 0.0f, 0.25f);
      const float speedValue = speed() > 0.0f ? speed() : 0.0f;
      args.adaptation = 1.0f - std::exp(-dt * speedValue);
      args.frameSeconds = dt;
    }

    ctx->pushConstants(0, sizeof(args), &args);

    {
      ScopedGpuProfileZone(ctx, "Halo Meter");
      rtxCtx->setFramePassStage(RtxFramePassStage::AutoExposure_Histogram);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_COLOR_INPUT, rtOutput.m_finalOutput.view(Resources::AccessType::Read), nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_ALBEDO_INPUT, rtOutput.m_primaryAlbedo.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_LINEAR_VIEW_Z_INPUT, rtOutput.m_primaryLinearViewZ.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_PRIMARY_SURFACE_FLAGS_INPUT, rtOutput.m_primarySurfaceFlags.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT, m_haloPartials.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposureHaloMeterShader::getShader());
      ctx->dispatch(groups.width, groups.height, 1);
    }

    {
      ScopedGpuProfileZone(ctx, "Halo Meter Resolve");
      rtxCtx->setFramePassStage(RtxFramePassStage::AutoExposure_Exposure);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT, m_haloPartials.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_STATE_INPUT_OUTPUT, m_haloState.view, nullptr);
      ctx->bindResourceView(AUTO_EXPOSURE_HALO_EXPOSURE_OUTPUT, m_exposure.view, nullptr);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, AutoExposureHaloResolveShader::getShader());
      ctx->dispatch(1, 1, 1);
    }
  }

  void DxvkAutoExposure::dispatch(
    Rc<DxvkContext> ctx,
    Rc<DxvkSampler> linearSampler,
    const Resources::RaytracingOutput& rtOutput,
    const float frameTimeMilliseconds,
    bool resetHistory) {

    ScopedGpuProfileZone(ctx, "Auto Exposure");

    m_resetState |= resetHistory;

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    if (m_exposure.image.ptr() == nullptr) {
      createResources(ctx);
      m_resetState = true;
    }

    if (m_resetState || !enabled()) {
      // Exposure scale 1.0, so a disabled or reset frame is unmodified.
      VkClearColorValue clearColor;
      clearColor.float32[0] = clearColor.float32[1] = clearColor.float32[2] = clearColor.float32[3] = 1.0f;

      VkImageSubresourceRange subRange = {};
      subRange.layerCount = 1;
      subRange.levelCount = 1;
      subRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

      ctx->clearColorImage(m_exposure.image, clearColor, subRange);

      // The meter snaps on the next frame it runs, as the addon does after its state buffer is cleared.
      m_haloResetState = true;
    }

    if (enabled()) {
      dispatchHaloMeter(ctx, rtOutput, frameTimeMilliseconds);
      PreExposure::queueExposureReadback(ctx.ptr(), m_exposure);
    }
    // The final output still holds the pre-tonemap image here.
    Photometer::queueReadback(ctx.ptr(), rtOutput);

    m_resetState = false;
  }
}
