/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

  // Auto exposure: the incident-light meter of the halocedlaart Halo CE path tracer addon (composite.hlsl's
  // Meter / MeterResolve). Each pixel's lighting (albedo divided out) is metered as its channel average, the
  // center-weighted log mean of that drives the exposure, limited in EV and adapted at the same speed at any
  // frame rate. Remix adds sky weighting, a mean floor, a coverage hold and an EV compensation curve.
  class DxvkAutoExposure: public CommonDeviceObject {
  public:
    explicit DxvkAutoExposure(DxvkDevice* device);
    ~DxvkAutoExposure();

    void dispatch(
      Rc<DxvkContext> ctx,
      Rc<DxvkSampler> linearSampler,
      const Resources::RaytracingOutput& rtOutput,
      const float frameTimeMilliseconds,
      bool resetHistory = false);

    void showImguiSettings();

    void createResources(Rc<DxvkContext> ctx);
    const Resources::Resource& getExposureTexture() const { return m_exposure; }

  private:

    void dispatchHaloMeter(
      Rc<DxvkContext> ctx,
      const Resources::RaytracingOutput& rtOutput,
      const float frameTimeMilliseconds);

    void createHaloPartials(Rc<DxvkContext> ctx, uint32_t groupsX, uint32_t groupsY);

    Rc<vk::DeviceFn> m_vkd;

    // The adapted exposure the tonemapper multiplies in (1x1 R32F).
    Resources::Resource m_exposure;

    // One float4 (sum w*log, sum w, surface centre weight, total centre weight) per 16x16 tile, and the
    // adapted exposure + "has metered" flag carried between frames (the addon's meterState).
    Resources::Resource m_haloPartials;
    Resources::Resource m_haloState;
    uint32_t m_haloPartialsX = 0;
    uint32_t m_haloPartialsY = 0;

    bool m_resetState = true;
    bool m_haloResetState = true;

    RTX_OPTION("rtx.autoExposure", bool, enabled, true,
               "Adjusts exposure automatically, so the image is neither too bright nor too dark. The meter is the "
               "incident-light meter of the Halo CE path tracer addon by halocedlaart (rtx.autoExposure.halo.*): a "
               "centre-weighted mean of each pixel's lighting channel average, with albedo divided out and the view "
               "model excluded, limited in EV and adapted at the same speed at any frame rate. "
               "rtx.tonemap.exposureBias adds an EV offset on top.");

    // Meter parameters. Names, defaults and ranges are the addon's
    // (addon.cpp: AutoExposureTarget / Speed / Centre / MinEv / MaxEv).
    RTX_OPTION("rtx.autoExposure.halo", float, target, 0.2f,
               "Halo Meter: the brightness the metered frame is driven to, as how bright an 18% grey card in this light "
               "comes out. 0.18 is a photographic grey card. Lower values give a darker, more contrasted image. The "
               "addon allows 0.02 to 0.6.");
    RTX_OPTION("rtx.autoExposure.halo", float, speed, 2.0f,
               "Halo Meter: adaptation speed. Each frame moves exposure 1 - exp(-dt * speed) of the way to its target, "
               "with dt the frame time in seconds (at most 0.25), so it adapts the same at any frame rate. The addon "
               "allows 0.1 to 16.");
    RTX_OPTION("rtx.autoExposure.halo", float, centreWeight, 0.75f,
               "Halo Meter: how much more the middle of the frame counts than the edges. 0 meters the whole frame "
               "evenly, 1 ignores the edges entirely. The falloff is a circle in units of half the frame height, "
               "centred 10% of the frame height below the middle so the sky at the top counts less.");
    RTX_OPTION("rtx.autoExposure.halo", float, minEv, -16.0f,
               "Halo Meter: the lowest exposure, in stops. The addon allows -24 to 0.");
    RTX_OPTION("rtx.autoExposure.halo", float, maxEv, 4.0f,
               "Halo Meter: the highest exposure, in stops. 4 (16x) suits a game in its own light units and keeps dark "
               "interiors and nights from being lifted into noise. The addon uses 8 (256x). A game calibrated to "
               "physical units through a sky provider (NUMOS, where scene EV100 is about meter EV + 16) needs about 16, "
               "and keeps nights and interiors dark with the compensation curve (rtx.autoExposure.halo.compensation*). "
               "Range 0 to 24.");
    RTX_OPTION("rtx.autoExposure.halo", bool, demodulateAlbedo, true,
               "Halo Meter: meters the lighting rather than the image. Each pixel is divided by its albedo (at least "
               "0.05) and multiplied by 0.18, so dark and pale textures under the same light meter the same. Sky pixels "
               "count as albedo 1. This is the addon's behaviour. Turn it off only to compare.");
    RTX_OPTION("rtx.autoExposure.halo", bool, excludeViewModel, true,
               "Halo Meter: ignores view model pixels (the first-person weapon and hands), like the addon's "
               "first-person meter mask. Pixels drawn with the view model camera count as view model also when no "
               "view model copy is traced. Pixels seen through a lens are metered. While the view model's share of the "
               "metered frame spikes (rtx.autoExposure.halo.excludeViewModelSpikeMargin) or exceeds "
               "rtx.autoExposure.halo.excludeViewModelHoldThreshold, the exposure holds for up to half a second.");
    RTX_OPTION_ARGS("rtx.autoExposure.halo", float, excludeViewModelHoldThreshold, 0.85f,
                    "Halo Meter: the share of the metered frame, from 0 to 1, above which excluded view model pixels make "
                    "the exposure hold for up to half a second instead of adapting. 1 never holds this way.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.autoExposure.halo", float, excludeViewModelSpikeMargin, 0.15f,
                    "Halo Meter: how far, from 0 to 1, the view model's share of the metered frame must rise above its "
                    "average over about the last second to hold the exposure for up to half a second, as when a sight "
                    "kicks back into the frame. The average resets on camera cuts. 1 never holds this way.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION("rtx.autoExposure.halo", float, skyWeightMax, 1.0f,
               "Halo Meter: the metering weight of a bright sky pixel relative to scene surfaces. The weight fades with "
               "the sky's own brightness between rtx.autoExposure.halo.skyLumLow and skyLumHigh, so a day sky counts, "
               "dusk less and a night sky barely at all. Otherwise a night sky meters as near-black and pushes exposure "
               "to its maximum. 0 = never meter the sky.");
    RTX_OPTION("rtx.autoExposure.halo", float, skyLumLow, 0.01f,
               "Halo Meter: sky radiance (the linear channel average, before exposure) at or below which a sky pixel "
               "has no metering weight.");
    RTX_OPTION("rtx.autoExposure.halo", float, skyLumHigh, 0.1f,
               "Halo Meter: sky radiance at or above which a sky pixel has the full rtx.autoExposure.halo.skyWeightMax "
               "weight. Set it at or below skyLumLow for full weight at any brightness.");
    RTX_OPTION("rtx.autoExposure.halo", float, meanFloorEv, -24.0f,
               "Halo Meter: the darkest scene the meter believes, in stops (log2 of the metered grey-card level). "
               "Exposure can then never exceed target / 2^meanFloorEv, however dark the frame, which acts as a second "
               "exposure limit in scene terms. -24 = off, so rtx.autoExposure.halo.maxEv is the only limit. With a "
               "physically calibrated sky (NUMOS), values above about -20 hold every night at a fixed exposure. In a "
               "game's own units it only matters if set above log2(target) - maxEv.");
    RTX_OPTION("rtx.autoExposure.halo", float, minCoverage, 0.05f,
               "Halo Meter: holds the current exposure while less than this fraction of the centre-weighted frame is "
               "scene surface (not sky or view model), for example when looking straight up at a night sky. 0 = never "
               "hold. Range 0 to 0.5.");
    RTX_OPTION("rtx.autoExposure.halo", float, compensationNightStops, 0.0f,
               "Halo Meter compensation curve: the exposure bias, in stops, for scenes at or below "
               "rtx.autoExposure.halo.compensationNightEv (the EV100 of the metered grey card). It fades linearly to 0 "
               "at rtx.autoExposure.halo.compensationDayEv. Negative values keep nights darker than mid grey, as they "
               "look to the eye (about -2 to -4). 0 = no curve.");
    RTX_OPTION("rtx.autoExposure.halo", float, compensationNightEv, 0.0f,
               "Halo Meter compensation curve: the scene EV100 at and below which the full "
               "rtx.autoExposure.halo.compensationNightStops applies. A moonlit landscape is about -2 to -3, deep "
               "twilight or a street under lamps about 0 to 4. EV100 needs a calibrated sky provider (NUMOS). Without "
               "one it is measured in Remix units.");
    RTX_OPTION("rtx.autoExposure.halo", float, compensationDayEv, 12.0f,
               "Halo Meter compensation curve: the scene EV100 at and above which no compensation applies. An interior "
               "is about 5 to 7, an overcast day 12 and sunlight 15. With the night at EV 0 and the day at 12, a dim "
               "interior gets about half the night bias, so it stays darker than mid grey.");
    RTX_OPTION("rtx.autoExposure.halo", bool, addonFaithful, false,
               "Halo Meter: uses the addon's exact sky handling: the sky metered at albedo 1 with full weight at any "
               "brightness, no mean floor and no hold. Overrides rtx.autoExposure.halo.skyWeightMax, skyLumLow, "
               "skyLumHigh, meanFloorEv and minCoverage. Use it with rtx.autoExposure.halo.maxEv = 8 for the addon's "
               "limit.");
  };

}
