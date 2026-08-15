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

float getDepthOfFieldCoC(const float2 uv, const float focusDistance)
{
  const float rawViewDepth = PrimaryLinearViewZ.SampleLevel(LinearSampler, uv, 0.0f);
  const float viewDepth = abs(rawViewDepth);

  // The denoiser's miss value is a far-field sentinel. Sky sits at infinity,
  // where the lens curve asymptotes to apertureTerm / (Zf - f): focusing far
  // away leaves the sky nearly sharp instead of always fully blurred.
  if (rawViewDepth == cb.missLinearViewZ || viewDepth == cb.missLinearViewZ)
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

  const float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  const float focusDistance = max(cb.autoFocusEnabled != 0
    ? FocusState[0] + cb.autoFocusOffset
    : cb.focusDistance, 1e-4f);
  const float4 centerColor = InColorTexture[pixelPos];
  const float centerCoC = getDepthOfFieldCoC(uv, focusDistance);
  const float centerRadius = abs(centerCoC) * cb.maxBlurRadius * cb.resolutionScale;

  if (centerRadius <= 0.0f || cb.sampleCount == 0)
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

  float3 accumulatedColor = centerColor.rgb;
  float accumulatedWeight = 1.0f;

  [loop]
  for (uint i = 0; i < sampleCount; ++i)
  {
    const float sampleFraction = (float(i) + 0.5f) / float(sampleCount);
    const float sampleRadius = sqrt(sampleFraction) * centerRadius;
    const float angle = phase + float(i) * goldenAngle;
    const float2 sampleOffset = float2(cos(angle), sin(angle)) * sampleRadius;
    const float2 sampleUv = (float2(pixelPos) + 0.5f + sampleOffset) * cb.invImageSize;
    const float sampleCoC = getDepthOfFieldCoC(sampleUv, focusDistance);
    const float sampleCoCRadius = abs(sampleCoC) * cb.maxBlurRadius * cb.resolutionScale;
    const float sampleDistance = length(sampleOffset);

    // Foreground taps are intentionally not depth-rejected, allowing a
    // near-field tap to bleed over a sharper background pixel.
    if (sampleCoCRadius >= sampleDistance)
    {
      // Attenuating inner taps while keeping outer taps bright gives
      // defocused highlights a controllable disk and halo shape. The tap
      // index, rather than the randomized angle, defines the radial ramp.
      const float tapIndexFraction = sampleCount > 1u
        ? float(i) / float(sampleCount - 1u)
        : 1.0f;
      const float tapIntensity = lerp(saturate(cb.bokehMinIntensity), 1.0f, tapIndexFraction);
      const float weight = (sampleCoC < 0.0f ? 1.25f : 1.0f) * tapIntensity;
      accumulatedColor += InColorTexture.SampleLevel(LinearSampler, sampleUv, 0.0f).rgb * weight;
      accumulatedWeight += weight;
    }
  }

  OutColorTexture[pixelPos] = float4(accumulatedColor / accumulatedWeight, centerColor.a);
}
