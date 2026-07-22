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

// NGX passthrough object velocity pass, vertex stage: rasterizes a dynamic object (whose
// LocalToWorld changed since the previous frame) at its current transform and carries the
// per-vertex NDC motion (previous - current, matching the camera reprojection convention)
// to the fragment stage. Equivalent to UE3's own VelocityShader.usf vertex math, minus the
// backface stretch.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

[[vk::push_constant]]
ConstantBuffer<NgxVelocityPushConstants> pc;

struct VertexOutput {
  float4 position : SV_Position;
  // Raw clip-space positions, divided per pixel in the fragment stage: perspective-correct
  // interpolation of the full float4 reconstructs the exact projective mapping at every
  // fragment, where interpolating pre-divided NDC would bend the velocity field inside
  // large triangles spanning a depth range (UE3's VelocityShader carries clip positions
  // for the same reason)
  float4 currClip : TEXCOORD0;
  float4 prevClip : TEXCOORD1;
};

[shader("vertex")]
VertexOutput main([[vk::location(0)]] float4 localPosition: POSITION) {
  // Vulkan vertex fetch fills missing components with (0, 0, 0, 1) for FLOAT3 attributes
  const float4 clipPosition = mul(pc.clipFromLocal, localPosition);

  VertexOutput output;
  output.position = clipPosition;
  output.currClip = clipPosition;
  output.prevClip = mul(pc.prevClipFromLocal, localPosition);

  return output;
}
