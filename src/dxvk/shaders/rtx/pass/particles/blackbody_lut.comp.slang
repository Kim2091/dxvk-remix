/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

// Fire colormap generation compute shader.
// Generates a 256x1 RGBA texture mapping normalized temperature [0,1] to a
// fire gradient (black -> dark red -> orange -> yellow -> white) with an
// alpha ramp that rises from 0 to 1 over the low end of the range.
// This replaces the old blackbody LUT with an artist-friendly fire palette
// matching NvFlow's colormap approach.

#include "rtx/utility/shader_types.h"

layout(binding = 0)
RWTexture2D<float4> ColormapOutput;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint threadID : SV_DispatchThreadID)
{
  float u = float(threadID) / 255.0;

  // Fire gradient: black -> dark red -> orange -> yellow -> white
  float3 color;
  if (u < 0.1)
  {
    color = lerp(float3(0.0, 0.0, 0.0), float3(0.3, 0.0, 0.0), u / 0.1);
  }
  else if (u < 0.3)
  {
    color = lerp(float3(0.3, 0.0, 0.0), float3(1.0, 0.3, 0.0), (u - 0.1) / 0.2);
  }
  else if (u < 0.5)
  {
    color = lerp(float3(1.0, 0.3, 0.0), float3(1.0, 0.7, 0.1), (u - 0.3) / 0.2);
  }
  else if (u < 0.8)
  {
    color = lerp(float3(1.0, 0.7, 0.1), float3(1.0, 0.95, 0.7), (u - 0.5) / 0.3);
  }
  else
  {
    color = lerp(float3(1.0, 0.95, 0.7), float3(1.0, 1.0, 1.0), (u - 0.8) / 0.2);
  }

  // Alpha: ramp from 0 to 1 over [0, 0.15], then 1.0
  float alpha = saturate(u / 0.15);

  ColormapOutput[uint2(threadID, 0)] = float4(color, alpha);
}
