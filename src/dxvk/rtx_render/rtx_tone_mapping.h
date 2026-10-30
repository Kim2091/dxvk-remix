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
#pragma once

#include "dxvk_format.h"
#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_resources.h"

#include "../spirv/spirv_code_buffer.h"
#include "../util/util_matrix.h"
#include "rtx_options.h"

namespace dxvk {

  class DxvkDevice;
  class DxvkPipelineManager;

  // Global tonemapper: exposure, the tone curve (shaders/rtx/pass/tonemap/tonemap_operator.slangh),
  // sRGB encode and dither, and the HDR10 output settings.
  class DxvkToneMapping: public CommonDeviceObject {
  public:
    explicit DxvkToneMapping(DxvkDevice* device);
    ~DxvkToneMapping();

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      bool performSRGBConversion = true,
      bool autoExposureEnabled = true);

    void dispatchFastToneMapping(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      bool autoExposureEnabled);

    void dispatchInverseToneMapping(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::RaytracingOutput& rtOutput,
      bool autoExposureEnabled);

    bool isEnabled() const { return tonemappingEnabled(); }

    // HDR10 output. The D3D9 swap chain sets this when it created an ST.2084 swap chain and a
    // float back buffer for rtx.tonemap.hdrEnabled. While it is set, the image stays display
    // referred relative to paper white (1 = rtx.tonemap.hdrPaperWhiteNits, up to the peak), and
    // the present blit encodes it to BT.2020 PQ.
    static bool hdrOutputActive();
    static void setHdrOutputActive(bool active);
    // The tone curve's peak relative to paper white: hdrPeakNits / hdrPaperWhiteNits while HDR
    // output is active, 1 otherwise.
    static float tonemapPeak();
    // HDR output stores the back buffer relative to the UI brightness (1.0 = UI white =
    // hdrUiNits), so the game's SDR UI lands at hdrUiNits. The scene is scaled by
    // hdrPaperWhiteNits / hdrUiNits before its gamma encode to stay at paper white, and the
    // present blit encodes PQ with 1.0 = hdrUiNits.
    static float hdrUiNitsClamped();
    static float hdrSceneToBackBufferScale();

    // Async shader prewarming (upstream initializer path). Operator-only pipeline,
    // so only the apply-tonemapping shader needs prewarming.
    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void showEffectSettings();

  private:
    // Screenshots convert HDR captures to SDR with the tone curve settings.
    friend class RtxScreenshot;

    // The tone curve chart in the developer menu, with handles that edit the settings that shape the curve.
    static void showToneCurve();

    void dispatchApplyToneMapping(
      Rc<RtxContext> ctx,
      Rc<DxvkImageView> exposureView,
      const Resources::Resource& inputBuffer,
      const Resources::Resource& colorBuffer,
      bool performSRGBConversion,
      bool autoExposureEnabled);

    Rc<vk::DeviceFn> m_vkd;

    enum class DitherMode : uint32_t {
      None = 0,
      Spatial,
      SpatialTemporal,
    };

    // Mirrors the tonemapPrimaries* constants in shaders/rtx/pass/tonemap/tonemapping.h.
    enum class TonemapPrimaries : uint32_t {
      Custom1 = 0,
      Custom2,
      Bt709,
    };

    RTX_OPTION("rtx.tonemap", float, exposureBias, 0.f, "The exposure value to use for the global tonemapper when auto exposure is disabled, or a bias multiplier on top of the auto exposure's calculated exposure value.");
    RTX_OPTION("rtx.tonemap", bool, tonemappingEnabled, true, "A flag to enable or disable the global tonemapper.");
    RTX_OPTION("rtx.tonemap", TonemapPrimaries, primaries, TonemapPrimaries::Custom1,
               "The colour basis the tone curve runs in. 0 = Custom Primaries 1, 1 = Custom Primaries 2, 2 = BT.709 (the "
               "curve per channel, with no basis change, so bright saturated colours skew towards the primaries).");

    // HDR10 output settings.
    RTX_OPTION("rtx.tonemap", bool, hdrEnabled, true,
               "Allows HDR10 output (BT.2020 primaries, ST.2084 PQ, 10-bit swap chain). The output follows Windows by "
               "itself: when the game creates its swap chain, HDR10 is used if Windows has HDR on for that display, "
               "and plain SDR otherwise. In HDR the image is tone mapped to rtx.tonemap.hdrPeakNits, the scene's "
               "mid-tones sit at rtx.tonemap.hdrPaperWhiteNits and the game's UI at rtx.tonemap.hdrUiNits. Set False "
               "to always output SDR.");
    RTX_OPTION("rtx.tonemap", float, hdrPeakNits, 1000.0f,
               "HDR output: the display's peak brightness, in nits. The tone curve rolls highlights off towards it. "
               "Set it to the peak the display reports (for example in the Windows HDR Calibration app). Range 80 to "
               "10000.");
    RTX_OPTION("rtx.tonemap", float, hdrPaperWhiteNits, 203.0f,
               "HDR output: paper white, in nits. The brightness of the scene's SDR white: mid-tones are scaled to it. "
               "203 is the BT.2408 reference. Raise it for a brighter scene. The UI has its own rtx.tonemap.hdrUiNits. "
               "Range 80 to 1000, and at most rtx.tonemap.hdrPeakNits.");
    RTX_OPTION("rtx.tonemap", float, hdrUiNits, 203.0f,
               "HDR output: the brightness of the game's UI and HUD (and the Remix menu), in nits, as the nits of UI "
               "white. Independent of rtx.tonemap.hdrPaperWhiteNits, so the UI can be dimmer than the scene. Range 80 "
               "to 1000.");

    // Dithering settings.
    RTX_OPTION("rtx.tonemap", DitherMode, ditherMode, DitherMode::SpatialTemporal,
               "Tonemap dither mode selection, dithering allows for reduction of banding artifacts in the final rendered output from quantization using a small amount of monochromatic noise. Impact typically most visible in darker regions with smooth lighting gradients.\n"
               "Enabling dithering will make the rendered image slightly noisier, though usually dither noise is fairly imperceptible in most cases without looking closely. Generally dithered results will also look better than the alternative of banding artifacts due to increasing perceptual precision of the signal.\n"
               "Note that temporal dithering may increase perceptual precision further but may also introduce more noticeable noise in the final output in some cases due to the noise pattern changing every frame unlike a purely spatial approach.\n"
               "Supported enum values are 0 = None (Disabled), 1 = Spatial (Enabled, Spatial dithering only), 2 = SpatialTemporal (Enabled, Spatial and temporal dithering).\n"
               "Generally enabling dithering is recommended, but disabling it may be useful in some niche cases for improving compression ratios in images or videos at the cost of quality (as noise while it may not be very visible may be more difficult to compress), or for capturing \"raw\" post-tonemapped data from the renderer.");
  };

}
