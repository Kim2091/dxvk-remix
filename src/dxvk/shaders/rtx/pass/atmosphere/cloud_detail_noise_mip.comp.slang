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

/*
 * Nubis3 detail-noise MIP CHAIN (fork -- 2026-09-07).
 *
 * One level of a 2x2x2 BOX filter over the 128^3 detail volume, run once per
 * level right after cloud_detail_noise_baker fills level 0. Nubis Cubed p.101
 * mips this same volume and picks the level from view distance; without a
 * chain, the erosion field's 31-167 m content is sampled by a march whose step
 * is 25-100 m, which aliases -- and the temporal EMA then converges the
 * aliasing into the fine grain the whole detail round has been chasing.
 *
 * BOX, specifically, and not a Gaussian or a bilinear blit: every channel of
 * this volume is normalized to a mean of EXACTLY 0.5 at bake time, and the
 * sampler's erosion, relief and lobe terms are all built as (channel - 0.5)
 * or a mean-matched lerp. A box filter is the only cheap kernel that preserves
 * the mean exactly at every level, so coverage, shading bias and the
 * silhouette's zero-mean displacement stay identical no matter which LOD a
 * sample lands on. A Gaussian with clamped edges would not (this volume is
 * periodic, so its "edges" wrap).
 *
 * Both levels are bound as storage images rather than one sampled + one
 * storage: the read is a plain 8-texel gather at exact integer coordinates,
 * so there is nothing for a sampler to do, and the rgba8 format qualifier
 * keeps the read legal without StorageImageReadWithoutFormat.
 */

// Level N-1 (read) and level N (write) of the SAME image. The caller inserts a
// compute->compute write->read barrier between levels.
layout(binding = 0, rgba8)
RWTexture3D<float4> cloudDetailNoiseSrcLevel;

layout(binding = 1, rgba8)
RWTexture3D<float4> cloudDetailNoiseDstLevel;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 threadId : SV_DispatchThreadID) {
  uint3 dstDim;
  cloudDetailNoiseDstLevel.GetDimensions(dstDim.x, dstDim.y, dstDim.z);
  if (any(threadId >= dstDim)) {
    return;
  }

  const uint3 s = threadId * 2u;
  float4 sum = cloudDetailNoiseSrcLevel[s + uint3(0u, 0u, 0u)]
             + cloudDetailNoiseSrcLevel[s + uint3(1u, 0u, 0u)]
             + cloudDetailNoiseSrcLevel[s + uint3(0u, 1u, 0u)]
             + cloudDetailNoiseSrcLevel[s + uint3(1u, 1u, 0u)]
             + cloudDetailNoiseSrcLevel[s + uint3(0u, 0u, 1u)]
             + cloudDetailNoiseSrcLevel[s + uint3(1u, 0u, 1u)]
             + cloudDetailNoiseSrcLevel[s + uint3(0u, 1u, 1u)]
             + cloudDetailNoiseSrcLevel[s + uint3(1u, 1u, 1u)];

  cloudDetailNoiseDstLevel[threadId] = sum * 0.125f;
}
