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
#include "rtx_neural_uplift.h"

#include <algorithm>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_ngx_wrapper.h"
#include "rtx_options.h"

namespace dxvk {

  DxvkNeuralUplift::DxvkNeuralUplift(DxvkDevice* device)
    : CommonDeviceObject(device)
    , RtxPass(device) {
  }

  DxvkNeuralUplift::~DxvkNeuralUplift() {
    release();
  }

  bool DxvkNeuralUplift::isSupported() const {
    return m_device->getCommon()->metaNGXContext().supportsNeuralUplift();
  }

  bool DxvkNeuralUplift::isEnabled() const {
    return enable() && isSupported();
  }

  void DxvkNeuralUplift::onDestroy() {
    if (m_context) {
      m_context->releaseNGXFeature();
    }
    m_context = nullptr;
  }

  void DxvkNeuralUplift::release() {
    m_recreate = true;
    m_evaluatedLastFrame = false;
    m_intermediateColor.reset();

    // Drop the context entirely rather than just the feature: it owns the loaded snippet and its
    // NGX init, and bypassCallerCheck is applied at load time.
    m_context.reset();
    m_contextCreationAttempted = false;
    m_createdExtent = { 0, 0, 0 };
  }

  void DxvkNeuralUplift::createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) {
    // Matches m_finalOutput, which is what this pass reads and writes. dispatch() re-checks against
    // the resource it is actually handed, so a format change there cannot go unnoticed.
    m_intermediateColor = Resources::createImageResource(
      ctx, "neural uplift color input", targetExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
  }

  void DxvkNeuralUplift::releaseTargetResource() {
    m_intermediateColor.reset();
  }

  void DxvkNeuralUplift::onDeactivation() {
    if (m_context) {
      // The feature owns GPU resources DXVK knows nothing about and so cannot keep alive, which
      // is why NGXNeuralUpliftContext::initialize waits before releasing one to rebuild it. The
      // same applies to releasing one for good.
      m_device->waitForIdle();
      m_context->releaseNGXFeature();
    }
    m_recreate = true;
    m_evaluatedLastFrame = false;
    m_createdExtent = { 0, 0, 0 };
  }

  const Resources::Resource* DxvkNeuralUplift::selectDepth(const Resources::RaytracingOutput& rtOutput) const {
    const Resources::Resource* depth =
      useLinearDepth() ? &rtOutput.m_primaryLinearViewZ : &rtOutput.m_primaryDepth;

    return depth->image != nullptr ? depth : nullptr;
  }

  void DxvkNeuralUplift::initializeFeature(Rc<DxvkContext> ctx, const VkExtent3D& outputExtent, int passCount) {
    // Recorded BEFORE the attempt can fail. These fields are what dispatch() diffs against to
    // decide whether to rebuild, so recording them only on success would leave a failed attempt
    // looking like a pending settings change and retry it every frame - each retry a waitForIdle.
    m_createdPreset = preset();
    m_createdFeatureId = featureId();
    m_createdDepthInverted = effectiveDepthInverted();
    m_createdExtent = outputExtent;
    m_createdBypassCallerCheck = bypassCallerCheck();
    m_createdPassCount = passCount;

    if (!m_context && !m_contextCreationAttempted) {
      m_contextCreationAttempted = true;
      m_context = m_device->getCommon()->metaNGXContext().createNeuralUpliftContext(bypassCallerCheck());
    }

    if (!m_context) {
      m_statusReason = "snippet unavailable";
      return;
    }

    if (!m_context->isLibraryLoaded()) {
      m_statusReason = m_context->notLoadedReason().empty() ? "snippet failed to load" : "snippet load failed";
      return;
    }

    m_device->waitForIdle();

    // The preset option is a raw number because NVIDIA documents no mapping; clamp it to the range
    // the snippet ships rather than handing NGX an arbitrary value.
    const uint32_t clampedPreset = static_cast<uint32_t>(
      std::clamp(preset(), 0, static_cast<int>(NVSDK_NGX_DLSSNR_Hint_Render_Preset_7)));

    uint32_t outputSize[2] = { outputExtent.width, outputExtent.height };

    m_context->initialize(
      ctx,
      outputSize,
      m_createdDepthInverted,
      static_cast<NVSDK_NGX_DLSSNR_Hint_Render_Preset>(clampedPreset),
      static_cast<uint32_t>(featureId()),
      static_cast<uint32_t>(passCount),
      // DLAA: the pass is resolution-preserving, so there is no quality tier to pick.
      NVSDK_NGX_PerfQuality_Value_DLAA);

    if (m_context->isNeuralUpliftInitialized()) {
      m_initCount++;
      m_statusReason = "active";
      // A feature this new has no history behind it, whatever the frame thinks about continuity.
      // True for every pass equally now that each owns its own handle, which is exactly why this
      // single flag is still enough - see the reset comment in the pass loop below.
      m_forceHistoryReset = true;
    } else {
      m_statusReason = "feature creation failed";
    }
  }

  void DxvkNeuralUplift::dispatch(RtxContext* ctx,
                                  DxvkBarrierSet& barriers,
                                  const Resources::RaytracingOutput& rtOutput,
                                  bool displayEncoded,
                                  bool resetHistory) {
    if (!isActive()) {
      return;
    }

    // The model is trained on display-encoded frames and this is the only point in the chain
    // where one exists, so a frame the sRGB pass skipped has no valid input to offer it - running
    // anyway would hand the network a linear image, which is the gamma domain it was not trained
    // on. Rare (screenshot captures) or permanent (a host that set disableSrgbConversionForOutput),
    // and in both cases declining is the correct answer rather than a fallback.
    if (!displayEncoded) {
      m_statusReason = "skipped: frame is not display-encoded";
      m_evaluatedLastFrame = false;
      // Whatever the next frame is, it is not continuous with the last one the model saw.
      m_forceHistoryReset = true;
      return;
    }

    const Resources::Resource& inOutColor = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);

    m_evaluatedLastFrame = false;

    if (inOutColor.image == nullptr || inOutColor.view == nullptr) {
      m_statusReason = "no color input";
      return;
    }

    ScopedGpuProfileZone(ctx, "Neural Uplift");
    ctx->setFramePassStage(RtxFramePassStage::NeuralUplift);

    const Resources::Resource* depth = selectDepth(rtOutput);
    const Resources::Resource* motionVectors = rtOutput.m_primaryScreenSpaceMotionVector.image != nullptr
      ? &rtOutput.m_primaryScreenSpaceMotionVector
      : nullptr;

    m_lastHadDepth = depth != nullptr;
    m_lastHadMotionVectors = motionVectors != nullptr;

    const VkExtent3D outputExtent = inOutColor.image->info().extent;

    // Each pass after the first re-reads what the previous one wrote, so the whole copy/evaluate
    // sequence repeats; only the frame-level read-in and write-out happen once. Computed before the
    // recreate check below because the feature set now has one NGX handle per pass (see
    // NGXNeuralUpliftContext::initialize), so a passCount change has to rebuild it exactly like a
    // resolution or preset change does - there is no way to grow or shrink that set in place.
    const int passes = std::clamp(passCount(), 1, kNeuralUpliftMaxPassCount);

    // bypassCallerCheck is applied when the snippet is loaded, not when the feature is created, so
    // changing it has to drop the whole context and load again.
    if (m_createdBypassCallerCheck != bypassCallerCheck() && m_contextCreationAttempted) {
      m_context.reset();
      m_contextCreationAttempted = false;
      m_recreate = true;
    }

    m_recreate |= (m_createdPreset != preset())
      || (m_createdFeatureId != featureId())
      || (m_createdDepthInverted != effectiveDepthInverted())
      || (m_createdExtent.width != outputExtent.width)
      || (m_createdExtent.height != outputExtent.height)
      || (m_createdPassCount != passes);

    if (m_recreate) {
      initializeFeature(ctx, outputExtent, passes);
      m_recreate = false;
    }

    if (!m_context || !m_context->isNeuralUpliftInitialized()) {
      return;
    }

    // RtxPass sizes the staging copy from the target extent on activation and on resize; this
    // covers the case where the colour it is actually handed disagrees, as a format change would.
    if (m_intermediateColor.image == nullptr ||
        m_intermediateColor.image->info().extent.width != outputExtent.width ||
        m_intermediateColor.image->info().extent.height != outputExtent.height ||
        m_intermediateColor.image->info().format != inOutColor.image->info().format) {
      Rc<DxvkContext> dxvkCtx = ctx;
      m_intermediateColor = Resources::createImageResource(
        dxvkCtx,
        "neural uplift color input",
        outputExtent,
        inOutColor.image->info().format);
    }

    NGXNeuralUpliftContext::NGXBuffers buffers;
    buffers.pInColor = &m_intermediateColor;
    buffers.pInOutput = &inOutColor;
    buffers.pInDepth = depth;
    buffers.pInMotionVectors = motionVectors;

    NGXNeuralUpliftContext::NGXSettings settings;
    // Clamped here rather than left to the snippet, which silently pins anything above 2 to 2 - so
    // an out-of-range value would otherwise read in the UI as a style that is not applied.
    settings.style = static_cast<uint32_t>(std::clamp(style(), 0, kNeuralUpliftMaxStyle));
    settings.intensity = intensity();
    settings.styleStrength = std::clamp(styleStrength(), 0.0f, 1.0f);
    settings.localStructureStrength = localStructureStrength();
    // Passed through unclamped: -1 is a meaningful sentinel, not an out-of-range value.
    settings.skinStructureStrength = skinStructureStrength();
    settings.autoMask = autoMask();
    settings.depthInverted = m_createdDepthInverted;
    // A zero scale is not "no motion vectors", it is a motion field that says nothing moved, which
    // is worse than either alternative: the snippet still reprojects, and does it through a history
    // that never lines up with the frame. The option can arrive at 0 from a config file or an
    // environment variable as well as the UI, so it is caught here rather than only being made
    // unreachable in the panel.
    const float sanitizedMotionVectorScale = motionVectorScale() != 0.0f ? motionVectorScale() : 1.0f;
    settings.motionVectorScale[0] = sanitizedMotionVectorScale;
    settings.motionVectorScale[1] = sanitizedMotionVectorScale;

    // Hoisted out of the loop: these only ever move from their steady state into SHADER_READ, so
    // re-issuing the transition on a later pass would describe a source state they are no longer
    // in. They stay readable for every pass.
    const Resources::Resource* readOnlyInputs[] = { motionVectors, depth };

    for (const Resources::Resource* input : readOnlyInputs) {
      if (input == nullptr || input->view == nullptr) {
        continue;
      }

      barriers.accessImage(
        input->image,
        input->view->imageSubresources(),
        input->image->info().layout,
        input->image->info().stages,
        input->image->info().access,
        input->image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    }

    bool evaluated = false;

    for (int pass = 0; pass < passes; ++pass) {
      // Forcing a reset on every pass past the first was an earlier, unvalidated fix for a real
      // problem: with a single shared NGX handle, pass 2's history holds this frame's pass 1 - an
      // image that has not moved - while the motion vectors describe a whole frame of movement, so
      // reprojecting through them again pulls history off the geometry. It did not work. Reading
      // both RenoDX (renodx-dlss.addon64, which keeps its retained-feature cache keyed only on
      // width/height/performance/preset - never a pass index, so it is not a source of a
      // per-pass-handle precedent either way) and the snippet itself (nvngx_dlssnr.dll's cg2r.cpp,
      // where CG2R_CreatePrevOutput/CG2R_ResetTemporalHistoryOnControlChange hang off a per-feature
      // "this", not a module global) confirmed the history lives on the NGX handle, not the
      // evaluate call - Reset only changes how that one call blends against it, it does not stop
      // the call's output from becoming the new history for whoever reads that handle next. So a
      // single shared handle across N passes means the *last* pass of frame X always seeds frame
      // X+1's first read, at whatever enhancement depth N passes produced - the stack compounds
      // frame over frame without bound, which reads as passes "repeatedly accumulating" and, once
      // it saturates, flicker. Passes now each get an independent handle (see
      // NGXNeuralUpliftContext::initialize), so pass k's history is always "this same pass, last
      // frame" - a stable depth - and the original single-pass rule generalises to every pass
      // unchanged: reset only on a real discontinuity, never because of where the pass sits in the
      // chain.
      settings.resetAccumulation = resetHistory || m_forceHistoryReset;

      barriers.accessImage(
        inOutColor.image,
        inOutColor.view->imageSubresources(),
        inOutColor.image->info().layout,
        inOutColor.image->info().stages,
        inOutColor.image->info().access,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);

      // On a later pass the staging copy is coming back from the previous evaluation reading it,
      // not from its steady state. The source scope has to say so, or the copy below is free to
      // overwrite it while the snippet is still sampling the pass before.
      const VkPipelineStageFlags stagingSrcStages = (pass == 0)
        ? m_intermediateColor.image->info().stages
        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      const VkAccessFlags stagingSrcAccess = (pass == 0)
        ? m_intermediateColor.image->info().access
        : VK_ACCESS_SHADER_READ_BIT;

      barriers.accessImage(
        m_intermediateColor.image,
        m_intermediateColor.view->imageSubresources(),
        m_intermediateColor.image->info().layout,
        stagingSrcStages,
        stagingSrcAccess,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT);

      barriers.recordCommands(ctx->getCommandList());

      const VkImageSubresourceLayers copyLayers = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      ctx->copyImage(
        m_intermediateColor.image, copyLayers, { 0, 0, 0 },
        inOutColor.image, copyLayers, { 0, 0, 0 },
        outputExtent);

      // NGX reads and writes through its own descriptors, so the barriers below only have to put
      // each image in the layout and access scope the snippet compute work expects. The staging
      // image is handled separately from the read-only inputs because the copy above just left it in
      // TRANSFER_DST_OPTIMAL, not in its steady-state layout.
      barriers.accessImage(
        m_intermediateColor.image,
        m_intermediateColor.view->imageSubresources(),
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        m_intermediateColor.image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT);

      barriers.accessImage(
        inOutColor.image,
        inOutColor.view->imageSubresources(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        inOutColor.image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT);

      barriers.recordCommands(ctx->getCommandList());

      // pass selects which of the `passes` handles created above this call lands on - see the
      // comment on the loop above for why each pass has its own.
      const NVSDK_NGX_Result evaluateResult = m_context->evaluate(ctx, buffers, settings, static_cast<uint32_t>(pass));
      evaluated = NVSDK_NGX_SUCCEED(evaluateResult);

      barriers.accessImage(
        inOutColor.image,
        inOutColor.view->imageSubresources(),
        inOutColor.image->info().layout,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        inOutColor.image->info().layout,
        inOutColor.image->info().stages,
        inOutColor.image->info().access);

      barriers.recordCommands(ctx->getCommandList());

      // A failed pass leaves the output holding whatever the last good pass wrote, so there is no
      // point spending the remaining passes on it.
      if (!evaluated) {
        break;
      }
    }

    // A rejected evaluation left the snippet history where the previous one put it, so the next
    // frame would reproject across the gap. Held until an evaluation actually succeeds.
    //
    // One flag covers every pass's handle, not just the one that failed: on a mid-chain failure the
    // passes before it did advance their own history validly, so this resets a few handles that did
    // not strictly need it, but distinguishing "this pass's handle is fine, that one's is not" needs
    // a per-pass flag for a failure mode expected to be rare, and getting that wrong the other way
    // (calling a handle continuous when it is not) is the actual bug this pass exists to avoid.
    m_forceHistoryReset = !evaluated;

    // The staging copy is the only resource this pass owns, so it is the only one that can be
    // destroyed while the snippet's work still references it - releaseTargetResource() on
    // deactivation drops it, and so does the recreate above. NGX captured the raw VkImageView, so
    // the view has to be tracked as well as the image: a DxvkImageView holds a reference to its
    // image, not the other way around, and tracking only the image leaves the view free to go.
    ctx->getCommandList()->trackResource<DxvkAccess::None>(m_intermediateColor.view);
    ctx->getCommandList()->trackResource<DxvkAccess::Read>(m_intermediateColor.image);
    ctx->getCommandList()->trackResource<DxvkAccess::None>(inOutColor.view);
    ctx->getCommandList()->trackResource<DxvkAccess::Write>(inOutColor.image);

    m_evaluatedLastFrame = evaluated;
    m_statusReason = evaluated ? "active" : "evaluate failed";
  }

  void DxvkNeuralUplift::showImguiStatusLine() {
    if (!isSupported()) {
      const std::string& reason = m_device->getCommon()->metaNGXContext().getNeuralUpliftNotSupportedReason();
      ImGui::Text("Neural Uplift: unavailable - %s",
                  reason.empty() ? "nvngx_dlssnr.dll not found" : reason.c_str());
      return;
    }

    ImGui::Text("Neural Uplift: %s (feature id %d, preset %d, inits %u)",
                m_statusReason, featureId(), preset(), m_initCount);

    // With neither depth nor motion vectors the snippet has nothing to reproject through and runs
    // as a purely spatial filter, which is a different effect from the one it is meant to produce.
    if (enable() && !(m_lastHadDepth && m_lastHadMotionVectors)) {
      ImGui::TextWrapped("Inputs: %s, %s - running spatially only. Presets that rely on temporal "
                         "reprojection cannot be judged in this state.",
                         m_lastHadDepth ? "depth" : "NO depth",
                         m_lastHadMotionVectors ? "motion vectors" : "NO motion vectors");
    }
  }

  void DxvkNeuralUplift::showImguiSettings() {
    RemixGui::Checkbox("Enable Neural Uplift (DLSS-NR)", &enableObject());

    showImguiStatusLine();

    if (!enable()) {
      return;
    }

    ImGui::Indent();

    // The labels below follow RenoDX rather than the names the snippet uses internally, because
    // RenoDX is where most people have met these controls and its vocabulary is what they will
    // search for. Where the two disagree the snippet's own name is kept in the tooltip, since it
    // is the accurate one: NVIDIA's debug output calls these localTone, localStructure and
    // intensity, and "Model" here is DLSSNR.Style, not the network chosen by Network Preset
    // further down.
    RemixGui::Combo("Model", &styleObject(), "Model A\0Model B\0Model C\0");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.Style. Model A is neutral; B and C apply different baked structure/tone\n"
                        "biases. These are conditioning inputs to one network, not three networks -\n"
                        "the network itself is chosen by Network Preset under Inputs / Bring-up.\n"
                        "Values above 2 are clamped to 2 by the snippet.");
    }

    RemixGui::DragFloat("Local Tone Intensity", &styleStrengthObject(), 0.01f, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.LocalToneStrength. Despite the name this is not a tone control: it is\n"
                        "how far the selected Model is blended in from neutral, and 0 makes any Model\n"
                        "a no-op. The snippet clamps it to 0-1.");
    }

    RemixGui::DragFloat("Overall Intensity", &intensityObject(), 0.01f, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.Intensity. Blend of the enhanced image against the original. Below 1 the\n"
                        "snippet keeps an extra copy of the input to blend against, so it also costs\n"
                        "slightly more.");
    }

    RemixGui::Checkbox("Character Mask", &autoMaskObject());
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.UseAutoMask. Lets the snippet decide per pixel where to apply the effect,\n"
                        "which is what separates characters from the rest of the scene. The two structure\n"
                        "strengths below are applied through this mask, so turning it off disables them.");
    }

    ImGui::BeginDisabled(!autoMask());
    RemixGui::DragFloat("Structure Intensity", &localStructureStrengthObject(), 0.01f, 0.0f, 2.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.LocalStructureStrength. Local detail enhancement.");
    }
    RemixGui::DragFloat("Skin Structure Strength", &skinStructureStrengthObject(), 0.01f, -1.0f, 2.0f, "%.2f");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("DLSSNR.SkinStructureStrength. -1 means: use Structure Intensity for skin too.\n"
                        "That is the default, and is not the same as 0, which explicitly disables\n"
                        "structure enhancement on skin.");
    }
    ImGui::EndDisabled();

    RemixGui::DragInt("Pass Count", &passCountObject(), 1.0f, 1, kNeuralUpliftMaxPassCount, "%d");
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Runs the pass this many times over, each one enhancing what the last produced.\n"
                        "Costs a full evaluation per pass. Passes after the first run without temporal\n"
                        "history, so they sharpen spatially rather than accumulating across frames.\n"
                        "Changing this resets the history.");
    }

    if (ImGui::CollapsingHeader("Inputs / Bring-up")) {
      ImGui::Indent();
      RemixGui::DragInt("Network Preset (inert in 310.8)", &presetObject(), 1.0f, 0, 7, "%d");
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("DLSSNR.Hint.Render.Preset - the network selector, distinct from Model above.\n"
                          "Selects nothing in the current snippet: it ships one set of weights and falls\n"
                          "back to them for every value, logging the fallback. Changing it still recreates\n"
                          "the feature and resets the temporal history, which is the only difference you\n"
                          "will see.");
      }
      RemixGui::Checkbox("Use Linear View Z Depth", &useLinearDepthObject());
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Off feeds the same primary depth DLSS consumes; on feeds primary linear view Z.");
      }
      if (!useLinearDepth()) {
        RemixGui::Checkbox("Depth Inverted", &depthInvertedObject());
      }
      // Minimum is 0.01 rather than 0 so the control cannot express a value the pass would silently
      // replace with 1: dragging to the end of the slider and reading back "0.00" while the snippet
      // is handed 1.0 would be a worse bring-up experience than not offering it.
      RemixGui::DragFloat("Motion Vector Scale", &motionVectorScaleObject(), 0.01f, 0.01f, 10.0f, "%.2f");
      RemixGui::DragInt("NGX Feature ID", &featureIdObject(), 1.0f, 0, 63, "%d");
      RemixGui::Checkbox("Bypass Snippet Caller Check", &bypassCallerCheckObject());
      ImGui::Unindent();
    }

    ImGui::Unindent();
  }

} // namespace dxvk
