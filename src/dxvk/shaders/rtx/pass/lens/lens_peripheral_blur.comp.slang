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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// One direction of the separable Gaussian that blurs the image outside the lenses (rtx.lens.peripheralBlur).
// The first pass reads the output and writes half resolution, averaging the two input rows or columns under each tap.

#include "rtx/concept/lens/lens.h"

layout(binding = LENS_BLUR_BINDING_INPUT)
RWTexture2D<float4> BlurInput;

layout(binding = LENS_BLUR_BINDING_OUTPUT)
RWTexture2D<float4> BlurOutput;

layout(push_constant)
ConstantBuffer<LensBlurArgs> push;

vec3 lensBlurLoad(vec2 position)
{
  const ivec2 pixel = clamp(ivec2(floor(position)), ivec2(0), ivec2(push.inputResolution) - 1);

  return BlurInput[pixel].rgb;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= push.outputResolution))
  {
    return;
  }

  // The input position under the center of this output pixel, and the input offset that covers the other row or
  // column of a halving pass.
  const vec2 center = (vec2(threadId) + 0.5f) * float(push.downsample);
  const vec2 across = push.downsample > 1u ? (abs(push.tapStep.x) > 0.0f ? vec2(0.0f, 0.5f) : vec2(0.5f, 0.0f)) : vec2(0.0f);
  const float sigma = max(push.sigma, 1e-3f);

  vec3 sum = vec3(0.0f);
  float weightSum = 0.0f;

  for (int tap = -kLensBlurRadius; tap <= kLensBlurRadius; ++tap)
  {
    const float weight = exp(-0.5f * float(tap * tap) / (sigma * sigma));
    const vec2 position = center + push.tapStep * float(tap);
    const vec3 value = push.downsample > 1u
      ? 0.5f * (lensBlurLoad(position - across) + lensBlurLoad(position + across))
      : lensBlurLoad(position);

    sum += value * weight;
    weightSum += weight;
  }

  BlurOutput[threadId] = vec4(sum / weightSum, 1.0f);
}
