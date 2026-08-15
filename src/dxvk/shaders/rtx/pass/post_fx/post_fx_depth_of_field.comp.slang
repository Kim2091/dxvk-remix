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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/post_fx/post_fx.h"

[[vk::binding(POST_FX_DOF_INPUT)]]
Texture2D<float4> InColorTexture;

[[vk::binding(POST_FX_DOF_PRIMARY_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

[[vk::binding(POST_FX_DOF_OUTPUT)]]
RWTexture2D<float4> OutColorTexture;

[[vk::binding(POST_FX_DOF_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

[[vk::binding(POST_FX_DOF_FOCUS_STATE_INPUT)]]
RWTexture1D<float> FocusState;

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

// Projects a blur disc size in millimeters onto the output image and
// normalizes it to the maximum blur radius.
float lensCoCToNormalized(const float cocMm)
{
  const float maxRadiusPixels = max(cb.maxBlurRadius * cb.resolutionScale, 1e-4f);
  return min(cocMm * cb.sensorToPixels, maxRadiusPixels) / maxRadiusPixels;
}

float calculateCircleOfConfusion(const float viewDepth, const float focusDistance)
{
  // Thin-lens blur disc size based on [Lee2008], as popularized by
  // CinematicDOF (Frans Bouma):
  //   CoC[mm] = ((f^2 / N) / (Zf - f)) * |Z - Zf| / Z
  // with focal length f and aperture f-number N as artistic lens parameters.
  // The hyperbolic 1/Z falloff grows continuously from the focus plane and
  // asymptotes in the far field, so there are no focus-band edges at all.
  // cb.apertureTerm precomputes f^2 / N, cb.sensorToPixels projects the CoC
  // through the 24 mm full-frame sensor height into pixels.
  const float focusDistanceMm = max(focusDistance * cb.worldUnitToMm, 1e-3f);
  const float viewDepthMm = max(viewDepth * cb.worldUnitToMm, 1e-3f);
  const float cocMm = cb.apertureTerm / max(focusDistanceMm - cb.focalLength, 1.0f)
    * abs(viewDepthMm - focusDistanceMm) / viewDepthMm;
  const float normalizedCoC = lensCoCToNormalized(cocMm);
  return viewDepth < focusDistance ? -normalizedCoC : normalizedCoC;
}

bool isMissDepth(const float rawViewDepth)
{
  return rawViewDepth == cb.missLinearViewZ || abs(rawViewDepth) == cb.missLinearViewZ;
}

float getDepthOfFieldCoCFromViewDepth(const float rawViewDepth, const float focusDistance)
{
  const float viewDepth = abs(rawViewDepth);

  // The denoiser's miss value is a far-field sentinel. Sky sits at infinity,
  // where the lens curve asymptotes to apertureTerm / (Zf - f): focusing far
  // away leaves the sky nearly sharp instead of always fully blurred.
  if (isMissDepth(rawViewDepth))
  {
    const float focusDistanceMm = max(focusDistance * cb.worldUnitToMm, 1e-3f);
    return lensCoCToNormalized(cb.apertureTerm / max(focusDistanceMm - cb.focalLength, 1.0f));
  }

  return calculateCircleOfConfusion(viewDepth, focusDistance);
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(
  in uint2 threadId : SV_GroupThreadID,
  in uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
  {
    return;
  }

  const float2 pixelCenter = float2(pixelPos) + 0.5f;
  const float2 uv = pixelCenter * cb.invImageSize;
  const float focusDistance = max(cb.autoFocusEnabled != 0
    ? FocusState[0] + cb.autoFocusOffset
    : cb.focusDistance, 1e-4f);
  const float4 centerColor = InColorTexture[pixelPos];
  const float rawCenterViewDepth = PrimaryLinearViewZ.SampleLevel(LinearSampler, uv, 0.0f);
  const float centerViewDepth = abs(rawCenterViewDepth);
  const bool centerIsMiss = isMissDepth(rawCenterViewDepth);
  const float centerCoC = getDepthOfFieldCoCFromViewDepth(rawCenterViewDepth, focusDistance);
  const float maxBlurRadius = cb.maxBlurRadius * cb.resolutionScale;
  const float centerRadius = abs(centerCoC) * maxBlurRadius;

  if (maxBlurRadius <= 0.0f || cb.sampleCount == 0)
  {
    OutColorTexture[pixelPos] = centerColor;
    return;
  }

  // A per-pixel rotated spiral is a compact approximation of a Poisson disk.
  // The depth test below turns the disk into a scatter-as-gather kernel.
  const float noise = frac(sin(dot(float2(pixelPos), float2(12.9898f, 78.233f))
    + float(cb.frameIdx) * 0.6180339f) * 43758.5453f);
  const float phase = noise * 6.2831853f;
  const float goldenAngle = 2.3999632f;
  const uint sampleCount = min(cb.sampleCount, 64u);

  // Gather the destination surface over its own CoC. Foreground samples that
  // belong to a nearer layer are excluded here and composited from their own
  // source footprints below.
  float3 resolvedColor = centerColor.rgb;
  if (centerRadius > 0.0f)
  {
    float3 accumulatedColor = centerColor.rgb;
    float accumulatedWeight = 1.0f;

    [loop]
    for (uint i = 0; i < sampleCount; ++i)
    {
      const float sampleFraction = (float(i) + 0.5f) / float(sampleCount);
      const float sampleRadius = sqrt(sampleFraction) * centerRadius;
      const float angle = phase + float(i) * goldenAngle;
      const float2 sampleOffset = float2(cos(angle), sin(angle)) * sampleRadius;
      const float2 sampleUv = (pixelCenter + sampleOffset) * cb.invImageSize;
      const float rawSampleViewDepth = PrimaryLinearViewZ.SampleLevel(LinearSampler, sampleUv, 0.0f);
      const float sampleViewDepth = abs(rawSampleViewDepth);
      const bool sampleIsMiss = isMissDepth(rawSampleViewDepth);
      const float sampleCoC = getDepthOfFieldCoCFromViewDepth(rawSampleViewDepth, focusDistance);
      const float sampleCoCRadius = abs(sampleCoC) * maxBlurRadius;
      const float depthTolerance = max(centerViewDepth * 1e-3f, 1e-3f);
      const bool sampleIsInFront = !sampleIsMiss
        && (centerIsMiss || sampleViewDepth < centerViewDepth - depthTolerance);
      const bool isForegroundSource = sampleCoC < 0.0f && sampleIsInFront;

      if (sampleCoCRadius >= sampleRadius && !isForegroundSource)
      {
        // Attenuating inner taps while keeping outer taps bright gives
        // defocused highlights a controllable disk and halo shape. The tap
        // index, rather than the randomized angle, defines the radial ramp.
        const float tapIndexFraction = sampleCount > 1u
          ? float(i) / float(sampleCount - 1u)
          : 1.0f;
        const float tapIntensity = lerp(saturate(cb.bokehMinIntensity), 1.0f, tapIndexFraction);
        accumulatedColor += InColorTexture.SampleLevel(LinearSampler, sampleUv, 0.0f).rgb * tapIntensity;
        accumulatedWeight += tapIntensity;
      }
    }

    resolvedColor = accumulatedColor / accumulatedWeight;
  }

  // Evaluate foreground scattering in gather form. Samples are distributed
  // over the maximum possible support, but each sample is accepted using the
  // source pixel's CoC, not the destination's. This allows a near-field CoC
  // disc to cross a geometry silhouette and cover a focused destination.
  float3 foregroundColor = 0.0f;
  float foregroundWeight = 0.0f;
  const float maxBlurRadiusSq = maxBlurRadius * maxBlurRadius;
  const float minBokehIntensity = saturate(cb.bokehMinIntensity);
  const float meanBokehIntensity = minBokehIntensity
    + (1.0f - minBokehIntensity) * (2.0f / 3.0f);

  [loop]
  for (uint i = 0; i < sampleCount; ++i)
  {
    // sqrt distributes equal-weight samples uniformly over the maximum CoC
    // disc, which makes this a Monte Carlo estimate of source splatting.
    const float sampleFraction = (float(i) + 0.5f) / float(sampleCount);
    const float sampleRadius = sqrt(sampleFraction) * maxBlurRadius;
    const float angle = phase + (float(i) + 0.5f) * goldenAngle;
    const float2 sampleOffset = float2(cos(angle), sin(angle)) * sampleRadius;
    const float2 sampleUv = (pixelCenter + sampleOffset) * cb.invImageSize;
    const float rawSampleViewDepth = PrimaryLinearViewZ.SampleLevel(LinearSampler, sampleUv, 0.0f);
    const float sampleViewDepth = abs(rawSampleViewDepth);
    const bool sampleIsMiss = isMissDepth(rawSampleViewDepth);
    const float sampleCoC = getDepthOfFieldCoCFromViewDepth(rawSampleViewDepth, focusDistance);
    const float sampleCoCRadius = -sampleCoC * maxBlurRadius;
    const float depthTolerance = max(centerViewDepth * 1e-3f, 1e-3f);
    const bool sampleIsInFront = !sampleIsMiss
      && (centerIsMiss || sampleViewDepth < centerViewDepth - depthTolerance);

    if (sampleCoC < 0.0f && sampleIsInFront && sampleCoCRadius > 0.0f)
    {
      const float coverage = smoothstep(-0.75f, 0.75f, sampleCoCRadius - sampleRadius);
      if (coverage > 0.0f)
      {
        const float radialFraction = saturate(sampleRadius / sampleCoCRadius);
        const float bokehWeight = lerp(minBokehIntensity, 1.0f, radialFraction)
          / max(meanBokehIntensity, 1e-4f);

        // Uniform sampling uses the maximum disc. Correct for the source
        // disc's smaller area so a full opaque foreground still integrates to
        // unit coverage regardless of its CoC radius.
        const float footprintWeight = coverage * bokehWeight * maxBlurRadiusSq
          / max(sampleCoCRadius * sampleCoCRadius, 1.0f) / float(sampleCount);
        foregroundColor += InColorTexture.SampleLevel(LinearSampler, sampleUv, 0.0f).rgb
          * footprintWeight;
        foregroundWeight += footprintWeight;
      }
    }
  }

  if (foregroundWeight > 0.0f)
  {
    resolvedColor = lerp(resolvedColor, foregroundColor / foregroundWeight,
      saturate(foregroundWeight));
  }

  OutColorTexture[pixelPos] = float4(resolvedColor, centerColor.a);
}
