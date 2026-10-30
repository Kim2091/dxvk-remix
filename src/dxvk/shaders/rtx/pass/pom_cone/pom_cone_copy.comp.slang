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

// The copy passes of the cone maps (POM_CONE_* modes in pom_cone_args.h). Copies read through the source
// view with Load(), so its format, sRGB decode and swizzle give the same value a POM march samples. A 2-channel output
// written to a 1-channel image keeps only its first channel (the height pyramid). Cone values are k / 255 or k / 65535
// already (rounded down by the passes that made them), so writing them to the UNORM G channel keeps them exactly.
// With level set, every copied height is raised by 1 - top (POM_CONE_LEVEL) and clamped to 1. A uniform raise keeps every
// cone valid, and the clamp only lowers the field, so cones generated from the raw height stay conservative.

#include "rtx/pass/pom_cone/pom_cone_args.h"

layout(binding = POM_CONE_BINDING_SOURCE)
Texture2D<float4> Source;

layout(binding = POM_CONE_BINDING_CONE)
Texture2D<float4> Cone;

layout(binding = POM_CONE_BINDING_OUTPUT)
RWTexture2D<float2> Output;

layout(binding = POM_CONE_BINDING_LEVEL)
Texture2D<float4> Level;

layout(push_constant)
ConstantBuffer<PomConeCopyArgs> cb;

groupshared float levelTops[POM_CONE_LEVEL_GRID * POM_CONE_LEVEL_GRID];

// The top of the height field (POM_CONE_LEVEL): the brightest texel of a small mip, so a few bright texels of mip 0
// average out and the broad top of the field sets the level. Thread threadId takes one cell of a 16x16 grid.
float levelCellMean(uint2 threadId)
{
  uint width, height, mipLevels;
  Source.GetDimensions(0, width, height, mipLevels);
  uint mip = 0;
  while (mip + 1 < mipLevels && max(width >> mip, height >> mip) > POM_CONE_LEVEL_GRID)
  {
    ++mip;
  }
  const uint2 extent = max(uint2(width >> mip, height >> mip), uint2(1, 1));
  const uint2 cell = (extent + POM_CONE_LEVEL_GRID - 1) / POM_CONE_LEVEL_GRID;
  const uint2 origin = threadId * cell;
  if (any(origin >= extent))
  {
    return 0.0f;
  }
  // A map without small mips: the mean of up to 8x8 texels spread over the cell.
  const uint2 step = max(cell / 8, uint2(1, 1));
  float sum = 0.0f;
  float count = 0.0f;
  for (uint y = origin.y; y < min(origin.y + cell.y, extent.y); y += step.y)
  {
    for (uint x = origin.x; x < min(origin.x + cell.x, extent.x); x += step.x)
    {
      sum += Source.Load(int3(x, y, int(mip))).r;
      count += 1.0f;
    }
  }
  return sum / count;
}

float levelled(float height)
{
  return cb.level != 0 ? min(height + 1.0f - Level.Load(int3(0, 0, 0)).r, 1.0f) : height;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
  if (cb.mode == POM_CONE_LEVEL)
  {
    levelTops[groupIndex] = levelCellMean(threadId);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = POM_CONE_LEVEL_GRID * POM_CONE_LEVEL_GRID / 2; stride > 0; stride /= 2)
    {
      if (groupIndex < stride)
      {
        levelTops[groupIndex] = max(levelTops[groupIndex], levelTops[groupIndex + stride]);
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0)
    {
      Output[uint2(0, 0)] = float2(levelTops[0], 0.0f);
    }
    return;
  }

  const uint2 position = cb.origin + threadId;
  if (any(position >= cb.extent))
  {
    return;
  }

  if (cb.mode == POM_CONE_REDUCE_MAX)
  {
    const int2 last = int2(cb.sourceExtent) - int2(1, 1);
    const int2 base = int2(position) * 2;
    float height = 0.0f;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
      const int2 texel = min(base + int2(i & 1, i >> 1), last);
      height = max(height, Source.Load(int3(texel, 0)).r);
    }
    Output[position] = float2(height, 0.0f);
    return;
  }

  const int2 texel = int2(position);
  const float height = levelled(Source.Load(int3(texel, int(cb.sourceMip))).r);
  float cone = 0.0f;
  if (cb.mode == POM_CONE_COPY_HEIGHT_CONE)
  {
    cone = Cone.Load(int3(texel, 0)).r;
  }
  else if (cb.mode == POM_CONE_COPY_HEIGHT_COARSE)
  {
    const int shift = int(cb.coarseShift);
    cone = Cone.Load(int3(texel.x >> shift, texel.y >> shift, 0)).r;
  }
  Output[position] = float2(height, cone);
}
