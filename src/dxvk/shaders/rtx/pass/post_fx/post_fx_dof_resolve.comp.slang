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

[[vk::binding(POST_FX_DOF_RESOLVE_INPUT)]]
Texture2D<float4> InColorTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_PRIMARY_LINEAR_VIEW_Z_INPUT)]]
Texture2D<float> PrimaryLinearViewZ;

[[vk::binding(POST_FX_DOF_RESOLVE_OUTPUT)]]
RWTexture2D<float4> OutColorTexture;

[[vk::binding(POST_FX_DOF_RESOLVE_LINEAR_SAMPLER)]]
SamplerState LinearSampler;

[[vk::binding(POST_FX_DOF_RESOLVE_FOCUS_STATE_INPUT)]]
RWTexture1D<float> FocusState;

layout(push_constant)
ConstantBuffer<PostFxDepthOfFieldArgs> cb;

float lensCoCToNormalized(const float cocMm)
{
  const float maxRadiusPixels = max(cb.maxBlurRadius * cb.resolutionScale, 1e-4f);
  return min(cocMm * cb.sensorToPixels, maxRadiusPixels) / maxRadiusPixels;
}

float calculateCircleOfConfusion(const float viewDepth, const float focusDistance)
{
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

  if (rawViewDepth == cb.missLinearViewZ || viewDepth == cb.missLinearViewZ)
  {
    const float focusDistanceMm = max(focusDistance * cb.worldUnitToMm, 1e-3f);
    return lensCoCToNormalized(cb.apertureTerm / max(focusDistanceMm - cb.focalLength, 1.0f));
  }

  return calculateCircleOfConfusion(viewDepth, focusDistance);
}

float3 sampleKawase(const float2 uv, const float radiusPixels)
{
  const float2 offset = radiusPixels * cb.invImageSize;
  float3 filteredColor = InColorTexture.SampleLevel(LinearSampler, uv, 0.0f).rgb * 0.5f;
  filteredColor += InColorTexture.SampleLevel(LinearSampler, uv + float2(-offset.x, -offset.y), 0.0f).rgb * 0.125f;
  filteredColor += InColorTexture.SampleLevel(LinearSampler, uv + float2( offset.x, -offset.y), 0.0f).rgb * 0.125f;
  filteredColor += InColorTexture.SampleLevel(LinearSampler, uv + float2(-offset.x,  offset.y), 0.0f).rgb * 0.125f;
  filteredColor += InColorTexture.SampleLevel(LinearSampler, uv + float2( offset.x,  offset.y), 0.0f).rgb * 0.125f;
  return filteredColor;
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

  // Give sparse bokeh samples a small finite footprint. This reconstruction
  // is intentionally local and CoC-weighted; it does not infer silhouettes or
  // alter depth visibility.
  const float kawaseRadius = min(0.5f + centerRadius * 0.125f, 2.0f);
  const float kawaseBlend = saturate(cb.bokehFilterStrength) * saturate(centerRadius * 0.25f);
  float3 resolvedColor = centerColor.rgb;
  if (kawaseBlend > 0.0f)
  {
    resolvedColor = lerp(resolvedColor, sampleKawase(uv, kawaseRadius), kawaseBlend);
  }

  OutColorTexture[pixelPos] = float4(resolvedColor, centerColor.a);
}
