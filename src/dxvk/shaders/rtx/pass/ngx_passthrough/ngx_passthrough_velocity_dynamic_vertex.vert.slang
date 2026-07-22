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

// NGX passthrough object velocity pass, dynamic-mesh vertex stage: for CPU-modified
// meshes (UE3 CPU-skins morph/cloth-augmented skeletal meshes into dedicated dynamic
// buffers - e.g. the first person arms) the motion lives in the vertex positions
// themselves. The current positions come from the game's live vertex buffer, the
// previous frame's from the capture's snapshot bound as a second stream; both transform
// through their frame's clip matrices like the rigid variant.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

[[vk::push_constant]]
ConstantBuffer<NgxVelocityPushConstants> pc;

struct VertexOutput {
  float4 position : SV_Position;
  float4 currClip : TEXCOORD0;
  float4 prevClip : TEXCOORD1;
};

[shader("vertex")]
VertexOutput main([[vk::location(0)]] float3 currPosition: POSITION,
                  [[vk::location(1)]] float3 prevPosition: TEXCOORD2) {
  const float4 clipPosition = mul(pc.clipFromLocal, float4(currPosition, 1.0f));

  VertexOutput output;
  output.position = clipPosition;
  output.currClip = clipPosition;
  output.prevClip = mul(pc.prevClipFromLocal, float4(prevPosition, 1.0f));

  return output;
}
