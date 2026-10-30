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
#include <algorithm>
#include <vector>

#include "../../util/log/log.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_context.h"
#include "rtx_dlss_neural_rendering.h"
#include "rtx_imgui.h"
#include "rtx_ngx_wrapper.h"

namespace dxvk {

  DlssNeuralRendering::DlssNeuralRendering(DxvkDevice* device)
    : DxvkDLSS(device) {
  }

  bool DlssNeuralRendering::supportsDlssNeuralRendering() const {
    return m_device->getCommon()->metaNGXContext().supportsDlssNeuralRendering();
  }

  void DlssNeuralRendering::release() {
    m_dlssNeuralRenderingContexts.clear();
    m_createdPreset = -1;
    m_createdPassCount = 0;
    mRecreate = true;
  }

  void DlssNeuralRendering::onDestroy() {
    release();
  }

  bool DlssNeuralRendering::useDlssNeuralRendering() const {
    return isActive();
  }

  bool DlssNeuralRendering::isEnabled() const {
    const NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();
    return DlssNeuralRendering::enable() &&
      (!ngxContext.isDlssNeuralRenderingSupportChecked() || ngxContext.supportsDlssNeuralRendering());
  }

  bool DlssNeuralRendering::onActivation(Rc<DxvkContext>& ctx) {
    const bool initialized = initializeDlssNeuralRendering(ctx);
    mRecreate = !initialized;
    if (!initialized) {
      disableAfterFailure("DLSS-NR activation failed; disabling DLSS-NR.");
    }
    return initialized;
  }

  void DlssNeuralRendering::onDeactivation() {
    if (isFeatureInitialized()) {
      m_device->waitForIdle();
    }
    release();
  }

  int DlssNeuralRendering::getClampedPassCount() {
    return std::clamp(DlssNeuralRendering::passCount(), 1, kNeuralUpliftMaxPassCount);
  }

  bool DlssNeuralRendering::isFeatureInitialized() const {
    if (m_dlssNeuralRenderingContexts.empty()) {
      return false;
    }

    for (const std::unique_ptr<NGXNeuralRenderingContext>& context : m_dlssNeuralRenderingContexts) {
      if (!context || !context->isNeuralRenderingInitialized()) {
        return false;
      }
    }

    return true;
  }

  bool DlssNeuralRendering::hasVramAllocation() const {
    if (!isFeatureInitialized()) {
      return false;
    }

    for (const std::unique_ptr<NGXNeuralRenderingContext>& context : m_dlssNeuralRenderingContexts) {
      if (!context->hasVramAllocation()) {
        return false;
      }
    }

    return true;
  }

  uint64_t DlssNeuralRendering::getVramAllocationBytes() const {
    uint64_t total = 0;
    for (const std::unique_ptr<NGXNeuralRenderingContext>& context : m_dlssNeuralRenderingContexts) {
      if (context) {
        total += context->getVramAllocationBytes();
      }
    }
    return total;
  }

  bool DlssNeuralRendering::dispatch(
      Rc<RtxContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      bool resetHistory,
      bool useRayReconstructionGuides) {
    ScopedGpuProfileZone(ctx, "DlssNeuralRendering");
    ctx->setFramePassStage(RtxFramePassStage::DLSSNR);

    if (!useDlssNeuralRendering()) {
      return false;
    }

    const int passes = getClampedPassCount();
    mRecreate |= m_createdPreset != DlssNeuralRendering::preset() || m_createdPassCount != passes;

    if (mRecreate) {
      const bool initialized = initializeDlssNeuralRendering(ctx);
      mRecreate = !initialized;
      if (!initialized) {
        disableAfterFailure("DLSS-NR feature recreation failed; disabling DLSS-NR.");
        return false;
      }
    }

    if (!isFeatureInitialized() || static_cast<int>(m_dlssNeuralRenderingContexts.size()) != passes) {
      disableAfterFailure("DLSS-NR context is not initialized; disabling DLSS-NR.");
      return false;
    }

    const Resources::Resource* pMotionVectorInput = useRayReconstructionGuides
      ? &rtOutput.m_primaryScreenSpaceMotionVectorDLSSRR
      : &rtOutput.m_primaryScreenSpaceMotionVector;
    const Resources::Resource* pDepthInput = useRayReconstructionGuides
      ? &rtOutput.m_primaryDepthDLSSRR.resource(Resources::AccessType::Read)
      : &rtOutput.m_primaryDepth;

    // Later passes copy the previous output back into the input, so the input is written as well.
    const Resources::Resource& colorInput = rtOutput.m_neuralRenderingInput.resource(
      passes > 1 ? Resources::AccessType::ReadWrite : Resources::AccessType::Read);
    const Resources::Resource& colorOutput = rtOutput.m_neuralRenderingOutput.resource(Resources::AccessType::Write);

    float jitterOffset[2];
    device()->getCommon()->getSceneManager().getCamera().getJittering(jitterOffset);

    // Screen-space motion vectors are expressed in render-pixel units.
    float motionVectorScale[2] = { 1.f, 1.f };

    const std::vector<Rc<DxvkImageView>> pReadOnlyInputs = {
      pMotionVectorInput->view,
      pDepthInput->view,
      rtOutput.m_controlMask.view,
    };

    for (auto input : pReadOnlyInputs) {
      if (input == nullptr) {
        continue;
      }

      barriers.accessImage(
        input->image(),
        input->imageSubresources(),
        input->imageInfo().layout,
        input->imageInfo().stages,
        input->imageInfo().access,
        input->imageInfo().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

#ifdef REMIX_DEVELOPMENT
      ctx->cacheResourceAliasingImageView(input);
#endif
    }

    const Rc<DxvkImageView>& inputView = colorInput.view;
    const Rc<DxvkImageView>& outputView = colorOutput.view;

#ifdef REMIX_DEVELOPMENT
    ctx->cacheResourceAliasingImageView(inputView);
    ctx->cacheResourceAliasingImageView(outputView);
#endif

    // Note: Add texture inputs added here to the pReadOnlyInputs array above to properly access the images.
    NGXNeuralRenderingContext::NGXNeuralRenderingBuffers buffers;
    buffers.pInColor = &colorInput;
    buffers.pOutColor = &colorOutput;
    buffers.pMotionVectors = pMotionVectorInput;
    buffers.pDepth = pDepthInput;
    buffers.pControlMask = &rtOutput.m_controlMask;

    NGXNeuralRenderingContext::NGXNeuralRenderingSettings settings = {};
    settings.jitterOffset[0] = jitterOffset[0];
    settings.jitterOffset[1] = jitterOffset[1];
    settings.motionVectorScale[0] = motionVectorScale[0];
    settings.motionVectorScale[1] = motionVectorScale[1];
    settings.resetAccumulation = resetHistory || m_resetHistoryOnNextEvaluation;
    settings.intensity             = DlssNeuralRendering::intensity();
    settings.toneStrength          = DlssNeuralRendering::toneStrength();
    settings.structuralStrength    = DlssNeuralRendering::structuralStrength();
    settings.model                 = static_cast<uint32_t>(static_cast<int>(DlssNeuralRendering::model()));
    settings.useAutoMask           = DlssNeuralRendering::useAutoMask();
    settings.skinStructureStrength = settings.useAutoMask
      ? DlssNeuralRendering::skinStructureStrength()
      : 0.0f;

    const VkImageSubresourceLayers copyLayers = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    const VkExtent3D outputExtent = outputView->image()->info().extent;
    bool evaluateSucceeded = false;

    for (int pass = 0; pass < passes; ++pass) {
      if (pass == 0) {
        barriers.accessImage(
          inputView->image(),
          inputView->imageSubresources(),
          inputView->imageInfo().layout,
          inputView->imageInfo().stages,
          inputView->imageInfo().access,
          inputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

        barriers.accessImage(
          outputView->image(),
          outputView->imageSubresources(),
          outputView->imageInfo().layout,
          outputView->imageInfo().stages,
          outputView->imageInfo().access,
          outputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT);
      } else {
        // Feed the previous pass's output back in. copyImage transitions layouts itself, so these
        // barriers only order the NGX accesses against the transfer.
        barriers.accessImage(
          outputView->image(),
          outputView->imageSubresources(),
          outputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT,
          outputView->imageInfo().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT);

        barriers.accessImage(
          inputView->image(),
          inputView->imageSubresources(),
          inputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT,
          inputView->imageInfo().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT);

        barriers.recordCommands(ctx->getCommandList());

        ctx->copyImage(
          inputView->image(), copyLayers, { 0, 0, 0 },
          outputView->image(), copyLayers, { 0, 0, 0 },
          outputExtent);

        barriers.accessImage(
          inputView->image(),
          inputView->imageSubresources(),
          inputView->imageInfo().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_WRITE_BIT,
          inputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_READ_BIT);

        barriers.accessImage(
          outputView->image(),
          outputView->imageSubresources(),
          outputView->imageInfo().layout,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT,
          outputView->imageInfo().layout,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_SHADER_WRITE_BIT);
      }

      barriers.recordCommands(ctx->getCommandList());

      evaluateSucceeded = m_dlssNeuralRenderingContexts[pass]->evaluateNeuralRendering(ctx, buffers, settings);
      if (!evaluateSucceeded) {
        break;
      }
    }

    barriers.accessImage(
      outputView->image(),
      outputView->imageSubresources(),
      outputView->imageInfo().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT,
      outputView->imageInfo().layout,
      outputView->imageInfo().stages,
      outputView->imageInfo().access);

    barriers.accessImage(
      inputView->image(),
      inputView->imageSubresources(),
      inputView->imageInfo().layout,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      inputView->imageInfo().layout,
      inputView->imageInfo().stages,
      inputView->imageInfo().access);

    barriers.recordCommands(ctx->getCommandList());

    // NGX captured the raw image views, so the views must stay alive with the command list.
    ctx->getCommandList()->trackResource<DxvkAccess::None>(inputView);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(inputView->image());
    ctx->getCommandList()->trackResource<DxvkAccess::None>(outputView);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(outputView->image());

    if (!evaluateSucceeded) {
      disableAfterFailure("DLSS-NR evaluation failed; disabling DLSS-NR.");
      return false;
    }

    m_resetHistoryOnNextEvaluation = false;
    return true;
  }

  void DlssNeuralRendering::showDlssNeuralRenderingImguiSettings(bool showDeveloperSettings) {
    static auto modelCombo = RemixGui::ComboWithKey<Model>(
      "Model",
      { {
        { Model::Model0, "Model A" },
        { Model::Model1, "Model B" },
        { Model::Model2, "Model C" },
      } });

    // A combo keeps a drag from recreating the feature at every pass count it crosses.
    static auto passCountCombo = RemixGui::ComboWithKey<int>(
      "Passes",
      { {
        { 1, "1" }, { 2, "2" }, { 3, "3" }, { 4, "4" }, { 5, "5" },
        { 6, "6" }, { 7, "7" }, { 8, "8" }, { 9, "9" }, { 10, "10" },
      } });
    static_assert(kNeuralUpliftMaxPassCount == 10, "The pass count combo lists 1 to kNeuralUpliftMaxPassCount.");

    if (showDeveloperSettings && DlssNeuralRendering::showVramUsage()) {
      if (hasVramAllocation()) {
        constexpr double kBytesPerMiB = 1024.0 * 1024.0;
        const double vramAllocationMiB = static_cast<double>(getVramAllocationBytes()) / kBytesPerMiB;
        ImGui::Text("DLSS-NR SDK VRAM Allocation: %.2f MiB", vramAllocationMiB);
      } else {
        ImGui::TextDisabled("DLSS-NR SDK VRAM Allocation: Not Available");
      }
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("SDK-reported aggregate allocation across all passes, queried at feature creation. Excludes Remix-managed input/output textures.");
      }
    }

    RemixGui::Checkbox("DLSS 3D-Guided Neural Generation", &DlssNeuralRendering::enableObject());

    constexpr ImGuiSliderFlags flags = ImGuiSliderFlags_AlwaysClamp;
    ImGui::BeginDisabled(!DlssNeuralRendering::enable());
    passCountCombo.getKey(&DlssNeuralRendering::passCountObject());
    modelCombo.getKey(&DlssNeuralRendering::modelObject());
    RemixGui::SliderFloat("Overall Intensity", &DlssNeuralRendering::intensityObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Structure Intensity", &DlssNeuralRendering::structuralStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::SliderFloat("Tone Intensity", &DlssNeuralRendering::toneStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    RemixGui::Checkbox("Auto Character Mask", &DlssNeuralRendering::useAutoMaskObject());
    ImGui::BeginDisabled(!DlssNeuralRendering::useAutoMask());
    RemixGui::SliderFloat("Character Intensity", &DlssNeuralRendering::skinStructureStrengthObject(), 0.0f, 1.0f, "%.2f", flags);
    ImGui::EndDisabled();
    ImGui::EndDisabled();

    if (!showDeveloperSettings) {
      return;
    }

    ImGui::Separator();
    if (RemixGui::CollapsingHeader("Advanced Settings")) {
      ImGui::Indent();
      RemixGui::Checkbox("Highlight Recovery", &DlssNeuralRendering::enableHighlightRecoveryObject());
      // Thresholds: (exposed HDR max-channel min/max, LDR max-channel min/max)
      RemixGui::DragFloat4("Highlight Recovery Thresholds", &DlssNeuralRendering::highlightRecoveryThresholdsObject(), 0.01f, 0.0f, 32.0f, "%.3f", flags);
      RemixGui::Checkbox("Volumetric Improvement", &DlssNeuralRendering::enableVolumetricControlMaskObject());
      RemixGui::DragInt("Network Preset", &DlssNeuralRendering::presetObject(), 1.0f, 0, kNeuralUpliftMaxPreset, "%d", flags);
      RemixGui::Checkbox("Show SDK VRAM Usage", &DlssNeuralRendering::showVramUsageObject());
      ImGui::Unindent();
    }
  }

  void DlssNeuralRendering::setDlssNeuralRenderingSettings(const uint32_t displaySize[2]) {
    ScopedCpuProfileZone();

    m_displaySize[0] = displaySize[0];
    m_displaySize[1] = displaySize[1];

    mRecreate = true;
  }

  bool DlssNeuralRendering::initializeDlssNeuralRendering(Rc<DxvkContext> renderContext) {
    const int passes = getClampedPassCount();
    const int preset = std::clamp(DlssNeuralRendering::preset(), 0, kNeuralUpliftMaxPreset);

    // Recorded before the attempt so a failure is not retried as a pending settings change.
    m_createdPreset = DlssNeuralRendering::preset();
    m_createdPassCount = passes;

    if (isFeatureInitialized()) {
      m_device->waitForIdle();
    }

    m_dlssNeuralRenderingContexts.resize(passes);

    NGXContext& ngxContext = m_device->getCommon()->metaNGXContext();
    for (std::unique_ptr<NGXNeuralRenderingContext>& context : m_dlssNeuralRenderingContexts) {
      if (!context) {
        context = ngxContext.createDlssNeuralRenderingContext();
      }

      if (!context) {
        m_dlssNeuralRenderingContexts.clear();
        return false;
      }

      context->initialize(renderContext, m_displaySize, static_cast<uint32_t>(preset), DlssNeuralRendering::showVramUsage());
      if (!context->isNeuralRenderingInitialized()) {
        m_dlssNeuralRenderingContexts.clear();
        return false;
      }
    }

    m_resetHistoryOnNextEvaluation = true;
    return true;
  }

  void DlssNeuralRendering::disableAfterFailure(const char* pReason) {
    Logger::err(pReason);
    mRecreate = false;
    DlssNeuralRendering::enable.setDeferred(false);
  }
} // namespace dxvk
