/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx/pass/bloom/bloom.h"

// Sampled with clamp-to-border (black outside the image): a bright area leaving the screen fades out of the bloom
// instead of acting as an infinite light at the edge.
layout(binding = BLOOM_DOWNSAMPLE_INPUT)
Sampler2D<float4> InColorBuffer;

layout(binding = BLOOM_DOWNSAMPLE_OUTPUT)
RWTexture2D<float4> OutColorBuffer;

layout(push_constant)
ConstantBuffer<BloomDownsampleArgs> cb;

float2 calcUV(const int2 downsampledPix)
{
  return (float2(downsampledPix) + float2(0.5)) * cb.downsampledOutputSizeInverse;
}

// Froyok's downsample: the 13-tap pattern of Call of Duty: Advanced Warfare (SIGGRAPH 2014, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare"), weighted as half the inner 4 taps and half the outer 3x3 grid.
float3 downsample13tap(const int2 downsampledPix)
{
  const float2 centerUv = calcUV(downsampledPix);

  // Inner 4 taps at +-1 texel: (1 / 4) * 0.5 each.
  float3 inner = InColorBuffer.SampleLevel(centerUv + float2(-1, 1) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 1, 1) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2(-1,-1) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 1,-1) * cb.inputSizeInverse, 0).rgb;

  // Outer 3x3 grid at 0 / +-2 texels: (1 / 9) * 0.5 each.
  float3 outer = InColorBuffer.SampleLevel(centerUv + float2(-2, 2) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 0, 2) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 2, 2) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2(-2, 0) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 0, 0) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 2, 0) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2(-2,-2) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 0,-2) * cb.inputSizeInverse, 0).rgb
               + InColorBuffer.SampleLevel(centerUv + float2( 2,-2) * cb.inputSizeInverse, 0).rgb;

  return inner * 0.125f + outer * (0.5f / 9.0f);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 ipos : SV_DispatchThreadID)
{
  if (any(ipos >= cb.downsampledOutputSize))
  {
    return;
  }

  float3 result = downsample13tap(int2(ipos));

  // A NaN or Inf in the scene would spread over the whole bloom chain.
  if (any(isnan(result)) || any(isinf(result)))
  {
    result = float3(0.0f);
  }

  OutColorBuffer[ipos] = float4(result, 0);
}
