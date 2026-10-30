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

#include <memory>
#include <vector>

#include "rtx_dlss.h"

namespace dxvk {

  class NGXNeuralRenderingContext;

  // Highest NVSDK_NGX_DLSSNR_Hint_Render_Preset the SDK header defines (Preset_G).
  constexpr int kNeuralUpliftMaxPreset = 7;

  // Upper bound on rtx.neuralUplift.passCount. Each pass costs a full evaluation plus its own
  // NGX feature handle, history and scratch memory.
  constexpr int kNeuralUpliftMaxPassCount = 10;

  class DlssNeuralRendering : public DxvkDLSS {
  public:
    explicit DlssNeuralRendering(DxvkDevice* device);

    bool supportsDlssNeuralRendering() const;

    // Draws the player settings, plus the advanced and diagnostic ones when showDeveloperSettings is set.
    void showDlssNeuralRenderingImguiSettings(bool showDeveloperSettings);

    // Returns true when NGX produced a valid output image.
    bool dispatch(
      Rc<RtxContext> ctx,
      DxvkBarrierSet& barriers,
      const Resources::RaytracingOutput& rtOutput,
      bool resetHistory,
      bool useRayReconstructionGuides);

    void release();

    bool useDlssNeuralRendering() const;

    void setDlssNeuralRenderingSettings(const uint32_t displaySize[2]);

    virtual void onDestroy();

    enum class Model : int {
      Model0 = 0,
      Model1 = 1,
      Model2 = 2,
    };

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enable, false,
                    "Enables DLSS 3D-Guided Neural Generation (DLSS-NR), a neural enhancement of the finished frame that "
                    "adjusts detail, lighting and colour. It costs significant GPU time for every pass, see "
                    "rtx.neuralUplift.passCount. It needs an NVIDIA GPU and driver that support DLSS-NR, and it turns "
                    "itself off without that support.",
                    args.environment = "RTX_NEURAL_UPLIFT_ENABLE",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, intensity, 1.0f,
                    "How strongly the enhanced image replaces the original, 0 to 1. 1 is the full effect, and lower "
                    "values blend back toward the original image.",
                    args.environment = "RTX_NEURAL_UPLIFT_INTENSITY",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, structuralStrength, 0.7f,
                    "How much DLSS-NR adjusts details, shadows and materials, 0 to 1. Lower values keep the original "
                    "structure, and higher values change it more.",
                    args.environment = "RTX_NEURAL_UPLIFT_STRUCTURAL_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, toneStrength, 0.3f,
                    "How much DLSS-NR adjusts lightness, darkness and colour, 0 to 1. Lower values keep the original "
                    "tone and colour, and higher values change them more.",
                    args.environment = "RTX_NEURAL_UPLIFT_TONE_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", Model, model, Model::Model0,
                    "Selects one of three DLSS-NR looks: 0: Model A, 1: Model B, 2: Model C. The result depends on the "
                    "scene, so compare them where you play. They are variations of one network, which "
                    "rtx.neuralUplift.preset chooses.",
                    args.environment = "RTX_NEURAL_UPLIFT_MODEL",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, useAutoMask, true,
                    "Lets DLSS-NR find characters on its own and give them their own structure strength, "
                    "rtx.neuralUplift.skinStructureStrength. The automatic mask is combined with the per-material "
                    "control mask, keeping the lower of the two at each pixel. Turn it off to use the per-material "
                    "mask alone.",
                    args.environment = "RTX_NEURAL_UPLIFT_USE_AUTO_MASK",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", float, skinStructureStrength, 0.5f,
                    "How much DLSS-NR adjusts the structure of characters, 0 to 1, while rtx.neuralUplift.useAutoMask "
                    "is on. Keeping it below rtx.neuralUplift.structuralStrength stops faces from being over-sharpened. "
                    "It has no effect while the automatic mask is off.",
                    args.environment = "RTX_NEURAL_UPLIFT_SKIN_STRUCTURE_STRENGTH",
                    args.minValue = 0.0f, args.maxValue = 1.0f,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", int, preset, 2,
                    "Network selection hint (DLSSNR.Hint.Render.Preset), 0 to 7. 0 lets the SDK choose, and 1 to 7 "
                    "select presets A to G. 2 (preset B) is the lightweight network. Which presets exist depends on the "
                    "installed DLSS-NR version. Unlike rtx.neuralUplift.model, which picks a look within the network, "
                    "this picks the network. Changing it recreates the feature and resets its history.",
                    args.environment = "RTX_NEURAL_UPLIFT_PRESET",
                    args.minValue = 0, args.maxValue = kNeuralUpliftMaxPreset,
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_ARGS("rtx.neuralUplift", int, passCount, 1,
                    "Runs DLSS-NR this many times in a row on each frame, 1 to 10. Each pass works on the previous "
                    "pass's output, so the effect stacks. Every pass costs a full DLSS-NR evaluation, which is "
                    "significant GPU time, plus its own video memory. Changing this recreates the feature and resets "
                    "its history.",
                    args.environment = "RTX_NEURAL_UPLIFT_PASS_COUNT",
                    args.minValue = 1, args.maxValue = kNeuralUpliftMaxPassCount,
                    args.flags = RtxOptionFlags::UserSetting);

    // Highlight recovery substitutes the original pre-NR HDR colour for pixels that were very bright
    // and would land in the active tone mapper's shoulder.
    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enableHighlightRecovery, true,
                    "Restores highlights that DLSS-NR's low dynamic range processing compresses, when the image is "
                    "mapped back to HDR.",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION("rtx.neuralUplift", Vector4, highlightRecoveryThresholds, Vector4(1.0f, 8.0f, 0.75f, 0.99f),
               "The edges of the highlight recovery blend: the exposed HDR maximum-channel range (x to y) and the "
               "LDR maximum-channel range (z to w).");

    RTX_OPTION_ARGS("rtx.neuralUplift", bool, enableVolumetricControlMask, true,
                    "Weakens the DLSS-NR control mask through fog, volumetrics and alpha-blended surfaces, in "
                    "proportion to how much they hide.",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION("rtx.neuralUplift", bool, showVramUsage, false,
               "Shows the VRAM DLSS-NR allocates in its settings panel, summed over all passes. It is queried when the "
               "feature is created, so turn this on before enabling DLSS-NR. Remix's own input and output images are "
               "not included.");

  protected:
    bool isEnabled() const override;
    bool onActivation(Rc<DxvkContext>& ctx) override;
    void onDeactivation() override;

  private:
    bool initializeDlssNeuralRendering(Rc<DxvkContext> pRenderContext);
    void disableAfterFailure(const char* pReason);
    bool isFeatureInitialized() const;
    bool hasVramAllocation() const;
    uint64_t getVramAllocationBytes() const;
    static int getClampedPassCount();

    uint32_t m_displaySize[2] = {};

    // One context per pass. Each owns one NGX feature handle, and the temporal history lives on the
    // handle: a shared handle would let the last pass of one frame seed the first pass of the next,
    // compounding the enhancement frame over frame.
    std::vector<std::unique_ptr<NGXNeuralRenderingContext>> m_dlssNeuralRenderingContexts;

    int m_createdPreset = -1;
    int m_createdPassCount = 0;

    // A newly created feature has no history to reproject.
    bool m_resetHistoryOnNextEvaluation = true;
  };
} // namespace dxvk
