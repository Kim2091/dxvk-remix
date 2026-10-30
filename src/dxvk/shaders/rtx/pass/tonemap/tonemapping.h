/*
* Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TONEMAPPING_H
#define TONEMAPPING_H

#include "rtx/utility/shader_types.h"

// Apply-tonemapping pass bindings.
#define TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT         0
#define TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT          1
#define TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT       2
#define TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT         3

// Dither modes.
static const uint32_t ditherModeNone             = 0;
static const uint32_t ditherModeSpatialOnly      = 1;
static const uint32_t ditherModeSpatialTemporal  = 2;

// Tone curve basis (rtx.tonemap.primaries). Mirrors DxvkToneMapping::TonemapPrimaries.
static const uint32_t tonemapPrimariesCustom1    = 0;
static const uint32_t tonemapPrimariesCustom2    = 1;
static const uint32_t tonemapPrimariesBt709      = 2;  // The curve per channel, no basis change.

// The color grade applied before the tone curve (rtx_color_grade_math.h, color_grade::Gpu): the grade's matrix,
// balance and saturation folded into one matrix, its contrast around mid gray (0.18), and the tone curve's anchor
// and shoulder.
struct ToneMappingGradeArgs {
  vec4 row0;                 // xyz = the first row of the matrix, w = contrast (1 = none)
  vec4 row1;                 // xyz = the second row, w = the tone curve's anchor (0.18 by default)
  vec4 row2;                 // xyz = the third row, w = the tone curve's shoulder (0.01 by default)
};

// Halo-meter pass bindings (auto_exposure_halo_meter / _resolve).
#define AUTO_EXPOSURE_HALO_COLOR_INPUT                    0
#define AUTO_EXPOSURE_HALO_ALBEDO_INPUT                   1
#define AUTO_EXPOSURE_HALO_LINEAR_VIEW_Z_INPUT            2
#define AUTO_EXPOSURE_HALO_SHARED_FLAGS_INPUT             3
#define AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT          4
#define AUTO_EXPOSURE_HALO_STATE_INPUT_OUTPUT             5
#define AUTO_EXPOSURE_HALO_EXPOSURE_OUTPUT                6

// Tile edge of the Halo meter's first pass (16x16 threads, one partial per tile).
// Kept in step with the addon's kMeterTile.
#define AUTO_EXPOSURE_HALO_TILE                           16

// Inputs for the Halo-meter auto exposure. Ported from composite.hlsl's Meter /
// MeterResolve in the halocedlaart addon; field meanings follow its CompositeConstants.
struct ToneMappingHaloMeterArgs {
  uint  groupsX;            // Tiles across the metered image.
  uint  groupsY;            // Tiles down the metered image.
  float centreWeight;       // 0 = meter the frame evenly, 1 = ignore the edges entirely.
  float target;             // Mid grey the metered (grey-card) level is driven to.

  float adaptation;         // Blend this frame: 1 - exp(-dt * speed), dt clamped to [0, 0.25] s.
  float minEv;              // Exposure floor, stops.
  float maxEv;              // Exposure ceiling, stops.
  float primaryMissLinearViewZ; // NRD miss sentinel in PrimaryLinearViewZ (sky pixels).

  uint  demodulateAlbedo;   // 1 = meter lighting (divide albedo out), the addon's behaviour.
  uint  excludeViewModel;   // 1 = zero weight on view-model pixels (addon's first-person mask).
  float skyWeightMax;       // Weight of a bright sky (primary-miss) pixel. Addon: 1 at any brightness.
  float minCoverage;        // Hold the exposure when the non-sky, non-view-model share of the
                            // centre-weighted frame is below this. 0 = the addon (never hold).

  float skyLumLow;          // Sky weight = skyWeightMax * smoothstep(skyLumLow, skyLumHigh, sky radiance
  float skyLumHigh;         // channel average, pre-exposure). high <= low => full weight (addon).
  float meanFloor;          // Floor on the metered geometric mean (linear, 2^meanFloorEv); 0 = none (addon).
  float invPreExposure;     // 1 / the renderer's pre-exposure (rtx_pre_exposure.h): the meter works in physical units.

  // Exposure compensation curve (rtx.autoExposure.halo.compensation*): bias in stops on the wanted exposure,
  // compensationNightStops at a scene EV100 at or below compensationNightEv, 0 at or above compensationDayEv,
  // linear in EV between. Scene EV100 = log2(metered grey-card level) + evOffset.
  float evOffset;           // log2(cd/m^2 per Remix radiance unit x 100 / 12.5)
  float compensationNightStops;
  float compensationNightEv;
  float compensationDayEv;
};

// Inputs for the apply-tonemapping pass (exposure, the tone curve in tonemap_operator.slangh,
// sRGB encode and dither).
struct ToneMappingApplyToneMappingArgs {
  // Global flags (16 bytes).
  uint toneMappingEnabled;
  uint performSRGBConversion;
  uint enableAutoExposure;
  uint ditherMode;

  // Tonemap state (16 bytes).
  uint frameIndex;
  float exposureFactor;      // 2^exposureBias, optionally premultiplied by auto-exposure.
  float tonemapPeak;         // The curve's peak relative to paper white: 1 for SDR, peak / paper white for HDR.
  uint hdrOutput;            // 1 = HDR10 output: keep values above 1 (up to tonemapPeak) for the present encode.

  // Tone curve selection (16 bytes).
  uint tonemapPrimaries;     // tonemapPrimaries*
  uint pad0;
  uint pad1;
  uint pad2;

  // Colour grade (48 bytes).
  ToneMappingGradeArgs grade;
};

struct FastToneMappingArgs {
  uint enableAutoExposure;
  float exposureFactor;
  uint enableHighlightRecovery;
  uint pad0;

  // Highlight recovery (inverse tonemap): substitute the original HDR back in for pixels that were
  // very bright in HDR and land in the active curve's shoulder. Thresholds are smoothstep edges:
  //   .x/.y = exposed HDR max-channel preserve range
  //   .z/.w = exposed LDR max-channel shoulder range
  vec4 highlightRecoveryThresholds;

  // Tone curve selection and colour grade, as in ToneMappingApplyToneMappingArgs.
  uint tonemapPrimaries;
  uint pad1;
  uint pad2;
  uint pad3;

  ToneMappingGradeArgs grade;
};

#ifdef __cplusplus
// 16B flags + 16B state + 16B tone curve selection + 48B grade.
static_assert(sizeof(ToneMappingApplyToneMappingArgs) == 96,
              "ToneMappingApplyToneMappingArgs layout drift.");
#endif

#endif  // TONEMAPPING_H
