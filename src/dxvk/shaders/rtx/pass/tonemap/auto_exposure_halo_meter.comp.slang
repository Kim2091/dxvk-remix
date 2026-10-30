/*
* Halo Meter auto exposure, pass 1 of 2: per-tile weighted log sums.
*
* Provenance: a Slang port of the `Meter` compute entry point in shaders/composite.hlsl
* of the "halocedlaart" Halo CE (MCC) path-tracer ReShade addon (archive
* halocedlaart5.rar, RenoDX-based) by TheGreatHmmmmm, who licensed it to this project under
* the MIT License (Copyright (c) 2026 TheGreatHmmmmm; the full notice is in ThirdPartyLicenses.txt).
* The addon builds on Prism, whose MIT notice is reproduced below.
*
*   MIT License
*
*   Copyright (c) 2026 Musa Haji
*   Copyright (c) 2026 OopyDoopy / KickFister / Jon
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in all
*   copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*   SOFTWARE.
*
* The meter takes the channel average rather than luminance, so a scene lit by one hue meters
* like a neutral one. It takes the geometric mean, so a small bright object moves exposure only
* a little. It is centre weighted with a circular falloff in units of half the frame height,
* around a point 10% of the frame height below the middle, so the sky at the top of the frame
* counts less.
* It meters lighting like an incident meter: each pixel is divided by its albedo and multiplied
* by an 18% card.
*
* Inputs: the post-upscale linear HDR output before exposure, PrimaryAlbedo at render
* resolution (sky counts as albedo 1), and the view-model bit of SharedFlags as the mask.
*/
#include "rtx/pass/tonemap/tonemapping.h"

[[vk::binding(AUTO_EXPOSURE_HALO_COLOR_INPUT)]]
RWTexture2D<float4> InColorBuffer;

[[vk::binding(AUTO_EXPOSURE_HALO_ALBEDO_INPUT)]]
Texture2D<float4> PrimaryAlbedo;

[[vk::binding(AUTO_EXPOSURE_HALO_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

layout(r16ui, binding = AUTO_EXPOSURE_HALO_SHARED_FLAGS_INPUT)
Texture2D<uint16_t> SharedFlags;

// Per tile: x = sum(w * log), y = sum(w), z = sum of centre weight over in-scene surface
// (non-sky, non-view-model) pixels, w = sum of centre weight over all pixels. z / w is the
// coverage the resolve uses to decide whether there is anything worth metering.
layout(rgba32f, binding = AUTO_EXPOSURE_HALO_PARTIALS_INPUT_OUTPUT)
RWTexture2D<float4> OutPartials;

layout(push_constant)
ConstantBuffer<ToneMappingHaloMeterArgs> cb;

// Below this a pixel counts as black rather than as an enormous negative log.
static const float kMeterFloor = 1e-4f;
// Floor under the albedo divisor. It caps the gain on specular and emissive light at 20x.
static const float kMeterAlbedoFloor = 0.05f;
// The grey card the demodulated lighting is put back onto (incident-meter convention).
static const float kMeterReferenceAlbedo = 0.18f;

static const uint kMeterThreads = AUTO_EXPOSURE_HALO_TILE * AUTO_EXPOSURE_HALO_TILE;

// GeometryFlags::isViewModel is bit 2. It is read directly to avoid the headers
// rtx/utility/geometry_flags.slangh pulls in.
static const uint kGeometryFlagIsViewModel = 1u << 2;

groupshared float g_meterLogSum[kMeterThreads];
groupshared float g_meterWeightSum[kMeterThreads];
groupshared float g_meterSurfaceSum[kMeterThreads];
groupshared float g_meterCentreSum[kMeterThreads];

float meterWeight(uint2 pixel, uint2 size)
{
  const float2 uv = (float2(pixel) + 0.5f) / float2(size);
  // The weighting centre sits 10% of the frame height below the middle, so the sky that fills
  // the top of the frame does not dominate the meter.
  float2 offset = (uv - float2(0.5f, 0.6f)) * 2.0f;
  // Units of half the frame height: a circle on screen, not an ellipse.
  offset.x *= float(size.x) / max(float(size.y), 1.0f);
  return 1.0f - saturate(cb.centreWeight) * smoothstep(0.0f, 1.0f, length(offset));
}

void meterReduce(uint index)
{
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = kMeterThreads / 2u; stride > 0u; stride >>= 1u)
  {
    if (index < stride)
    {
      g_meterLogSum[index] += g_meterLogSum[index + stride];
      g_meterWeightSum[index] += g_meterWeightSum[index + stride];
      g_meterSurfaceSum[index] += g_meterSurfaceSum[index + stride];
      g_meterCentreSum[index] += g_meterCentreSum[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
}

[shader("compute")]
[numthreads(AUTO_EXPOSURE_HALO_TILE, AUTO_EXPOSURE_HALO_TILE, 1)]
void main(uint2 id : SV_DispatchThreadID, uint2 group : SV_GroupID, uint index : SV_GroupIndex)
{
  uint2 size;
  InColorBuffer.GetDimensions(size.x, size.y);

  float weight = 0.0f;
  float weightedLog = 0.0f;
  float surfaceWeight = 0.0f;
  float centreWeight = 0.0f;

  if (all(id < size))
  {
    uint2 gbufferSize;
    PrimaryAlbedo.GetDimensions(gbufferSize.x, gbufferSize.y);

    // The G-buffer is at render resolution, so it is read with a nearest lookup.
    // Filtering across a material edge would be worse.
    const float2 gbufferUv = (float2(id) + 0.5f) / float2(size);
    const uint2 gbufferPixel = min(uint2(gbufferUv * float2(gbufferSize)), gbufferSize - 1u);

    // Physical units, so the sky thresholds, mean floor and EV limits keep their meaning.
    const float3 rawRadiance = InColorBuffer[id].rgb * cb.invPreExposure;
    // A non-finite pixel would pin the exposure at its floor for good, so it is left out of the meter.
    // It still counts as scene coverage.
    const bool finitePixel = all(isfinite(rawRadiance));
    const float3 radiance = finitePixel ? max(rawRadiance, float3(0.0f)) : float3(0.0f);

    const bool primaryMiss = PrimaryLinearViewZ[gbufferPixel] == cb.primaryMissLinearViewZ;

    float3 lighting = radiance;
    if (cb.demodulateAlbedo != 0)
    {
      // Sky (a primary miss) divides by one and meters as the light it is.
      const float3 albedo = primaryMiss
        ? float3(1.0f)
        : max(PrimaryAlbedo[gbufferPixel].rgb, float3(kMeterAlbedoFloor));
      lighting = radiance * (kMeterReferenceAlbedo / albedo);
    }

    // The view model gets zero weight, so it adds to neither the log sum nor the divisor.
    float inScene = 1.0f;
    if (cb.excludeViewModel != 0 && (uint(SharedFlags[gbufferPixel]) & kGeometryFlagIsViewModel) != 0)
    {
      inScene = 0.0f;
    }

    const float average = (lighting.r + lighting.g + lighting.b) * (1.0f / 3.0f);
    centreWeight = meterWeight(id, size);
    // A sky pixel's weight follows its own radiance, so a dark night sky cannot drag the mean
    // to the floor. skyLumHigh <= skyLumLow gives full weight.
    float skyFactor = 1.0f;
    if (primaryMiss)
    {
      const float skyLum = (radiance.r + radiance.g + radiance.b) * (1.0f / 3.0f);
      const float ramp = cb.skyLumHigh > cb.skyLumLow
        ? smoothstep(cb.skyLumLow, cb.skyLumHigh, skyLum)
        : 1.0f;
      skyFactor = saturate(cb.skyWeightMax) * ramp;
    }
    weight = finitePixel ? centreWeight * inScene * skyFactor : 0.0f;
    weightedLog = weight * log(max(average, kMeterFloor));
    surfaceWeight = primaryMiss ? 0.0f : centreWeight * inScene;
  }

  g_meterLogSum[index] = weightedLog;
  g_meterWeightSum[index] = weight;
  g_meterSurfaceSum[index] = surfaceWeight;
  g_meterCentreSum[index] = centreWeight;
  meterReduce(index);

  if (index == 0u)
  {
    OutPartials[group] = float4(g_meterLogSum[0], g_meterWeightSum[0], g_meterSurfaceSum[0], g_meterCentreSum[0]);
  }
}
