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

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

float calculateCircleOfConfusion(const float viewDepth)
{
  const float signedDistance = viewDepth - cb.focusDistance;

  if (signedDistance < -cb.focusRange)
  {
    return -saturate((-signedDistance - cb.focusRange) / max(cb.nearTransition, 1e-4f));
  }

  if (signedDistance > cb.focusRange)
  {
    return saturate((signedDistance - cb.focusRange) / max(cb.farTransition, 1e-4f));
  }

  return 0.0f;
}

float getDepthOfFieldCoC(const float2 uv)
{
  const float rawViewDepth = PrimaryLinearViewZ.SampleLevel(LinearSampler, uv, 0.0f);
  const float viewDepth = abs(rawViewDepth);

  // The denoiser's miss value is a far-field sentinel. Treat it as far depth
  // while retaining the explicit comparison for exact sky pixels.
  if (rawViewDepth == cb.missLinearViewZ || viewDepth == cb.missLinearViewZ)
  {
    return 1.0f;
  }

  return calculateCircleOfConfusion(viewDepth);
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
  const float4 centerColor = InColorTexture[pixelPos];
  const float centerCoC = getDepthOfFieldCoC(uv);
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
    const float sampleCoC = getDepthOfFieldCoC(sampleUv);
    const float sampleCoCRadius = abs(sampleCoC) * cb.maxBlurRadius * cb.resolutionScale;
    const float sampleDistance = length(sampleOffset);

    // Foreground taps are intentionally not depth-rejected, allowing a
    // near-field tap to bleed over a sharper background pixel.
    if (sampleCoCRadius >= sampleDistance)
    {
      const float weight = sampleCoC < 0.0f ? 1.25f : 1.0f;
      accumulatedColor += InColorTexture.SampleLevel(LinearSampler, sampleUv, 0.0f).rgb * weight;
      accumulatedWeight += weight;
    }
  }

  OutColorTexture[pixelPos] = float4(accumulatedColor / accumulatedWeight, centerColor.a);
}
