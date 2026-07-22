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

// NGX passthrough mode, pre-post-process injection: combines the DLSS color output with
// the game's original alpha channel before the result is written back over the scene
// color. UE3's D3D9 path stores the scene depth in the scene color alpha (see UE3
// Common.usf CalcSceneDepth: "depth stored in alpha" when depth textures are
// unsupported); shadow projection, fog, and depth of field read it back, and DLSS output
// alpha is undefined, so the write-back must carry the game's own alpha through.
//
// Note: only used when render extent == display extent (the pre-post injection is
// DLAA-only), so cb.resolution matches the output size and both inputs are 1:1.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

layout(binding = 0)
ConstantBuffer<NgxPassthroughArgs> cb;

layout(binding = 1)
Texture2D<float4> DlssColor;

layout(binding = 2)
Texture2D<float4> OriginalColor;

layout(binding = 3, rgba16f)
RWTexture2D<float4> MergedOutput;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 pixelCoordinate : SV_DispatchThreadID) {
  if (pixelCoordinate.x >= uint(cb.resolution.x) || pixelCoordinate.y >= uint(cb.resolution.y)) {
    return;
  }

  MergedOutput[pixelCoordinate] = float4(
    DlssColor[pixelCoordinate].rgb,
    OriginalColor[pixelCoordinate].a);
}
