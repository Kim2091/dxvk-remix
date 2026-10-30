/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "dxvk_include.h"
#include "dxvk_context.h"
#include "rtx_resources.h"

namespace dxvk {

  class DxvkDevice;

  class DxvkBloom: public RtxPass {
    
  public:
    explicit DxvkBloom(DxvkDevice* device);
    ~DxvkBloom();

    DxvkBloom(const DxvkBloom&) = delete;
    DxvkBloom(DxvkBloom&&) noexcept = delete;
    DxvkBloom& operator=(const DxvkBloom&) = delete;
    DxvkBloom& operator=(DxvkBloom&&) noexcept = delete;

    void dispatch(
      Rc<RtxContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const Resources::Resource& inOutColorBuffer);

    void showEffectSettings();

  private:
    void dispatchDownsampleStep(
      Rc<DxvkContext> ctx,
      const Rc<DxvkSampler>& borderSampler,
      const Resources::Resource& inputBuffer,
      const Resources::Resource& outputBuffer);

    void dispatchUpsampleStep(
      Rc<DxvkContext> ctx,
      const Rc<DxvkSampler>& linearSampler,
      const Resources::Resource& inputBuffer,
      const Resources::Resource& outputBuffer);

    void dispatchComposite(
      Rc<DxvkContext> ctx,
      const Rc<DxvkSampler> &linearSampler,
      const Resources::Resource& inOutColorBuffer,
      const Resources::Resource& bloomBuffer);

    virtual void createTargetResource(Rc<DxvkContext>& ctx, const VkExtent3D& targetExtent) override;
    virtual void releaseTargetResource() override;

    virtual bool isEnabled() const override;

    Rc<vk::DeviceFn> m_vkd;

    constexpr static int MaxBloomSteps = 8;
    // Each image is 1/2 resolution of the previous.
    Resources::Resource m_bloomBuffer[MaxBloomSteps] = {};

    // Energy-conserving bloom after Froyok (Lena Piquet): no threshold, levels lerped together, and the scene
    // lerped towards the bloom rather than having it added. See shaders/rtx/pass/bloom/bloom.h.
    RTX_OPTION_ENV("rtx.bloom", bool, enable, true, "RTX_BLOOM_ENABLE", "Enable bloom - glowing halos around intense, bright areas.");
    RTX_OPTION("rtx.bloom", float, intensity, 0.1f,
               "How far the image is blended towards the bloom, 0 to 1. The bloom is a lerp, not an addition, so it "
               "spreads light from bright areas into their surroundings without adding energy and needs no threshold. "
               "0.1 is subtle, 0.3 strong, and above 0.5 the image turns blurry.");
    RTX_OPTION("rtx.bloom", float, radius, 0.8f,
               "The bloom's spread, 0 to 1: how much each wider level counts against the narrower one below it. Higher "
               "values give a wider, softer glow with a long tail, lower values a tight glow. 0.7 to 0.85 is close to "
               "a Gaussian.");
    RTX_OPTION_ARGS("rtx.bloom", int, steps, 8,
                    "Number of downsampling steps, each half the size of the previous, [2..8]. More steps reach further "
                    "from bright areas, which matters at higher resolutions.",
                    args.minValue = 2,
                    args.maxValue = MaxBloomSteps);
  };
  
}
