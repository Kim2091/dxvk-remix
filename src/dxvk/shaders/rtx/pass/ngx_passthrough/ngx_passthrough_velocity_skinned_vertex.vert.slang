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

// NGX passthrough object velocity pass, skinned vertex stage: replays UE3's GPU skinning
// (verified against the game's shader bytecode: indices scaled by the 3-registers-per-
// bone stride, palettes as 3x4 rows with the translation in .w, position at w = 1) with
// the current and the previous frame's bone palettes to produce true per-vertex motion
// for skeletal meshes. Skinning outputs mesh-local space; the clip transforms compose
// LocalToWorld exactly like the rigid variant.

#include "rtx/pass/ngx_passthrough/ngx_passthrough_args.h"

[[vk::push_constant]]
ConstantBuffer<NgxVelocityPushConstants> pc;

layout(binding = 1)
ConstantBuffer<NgxVelocityRasterArgs> cb;

// All palettes for the frame, one block per skinned draw:
//   [header][previous palette][current palette]
// with cb.bonePaletteRegisterCount registers per palette. The header carries the draw's
// bone index scale (x) and consumed influence count (y). The draw's block is selected by
// an instance-rate attribute fetched from an identity-ramp buffer at a per-draw offset -
// deliberately not SV_InstanceID, which slang emulates with D3D semantics (base instance
// subtracted), defeating firstInstance-based selection.
layout(binding = 2)
StructuredBuffer<float4> BonePalettes;

struct VertexOutput {
  float4 position : SV_Position;
  float4 currClip : TEXCOORD0;
  float4 prevClip : TEXCOORD1;
};

float3 skinPosition(float4 localPosition, uint4 blendIndices, float4 blendWeights,
                    uint paletteBase, uint boneIndexScale) {
  float3 result = float3(0.0f, 0.0f, 0.0f);

  [unroll]
  for (uint influence = 0; influence < 4; influence++) {
    const uint boneBase = paletteBase + blendIndices[influence] * boneIndexScale;
    result += blendWeights[influence] * float3(
      dot(BonePalettes[boneBase + 0], localPosition),
      dot(BonePalettes[boneBase + 1], localPosition),
      dot(BonePalettes[boneBase + 2], localPosition));
  }

  return result;
}

[shader("vertex")]
VertexOutput main([[vk::location(0)]] float3 localPosition: POSITION,
                  [[vk::location(1)]] uint4 blendIndices: BLENDINDICES,
                  [[vk::location(2)]] float4 blendWeights: BLENDWEIGHT,
                  [[vk::location(3)]] uint paletteSlot: TEXCOORD3) {
  const float4 position = float4(localPosition, 1.0f);

  const uint blockBase = paletteSlot * (1 + 2 * cb.bonePaletteRegisterCount);
  const float4 header = BonePalettes[blockBase];
  const uint boneIndexScale = uint(header.x);
  const uint influenceCount = uint(header.y);
  const uint prevBase = blockBase + 1;
  const uint currBase = prevBase + cb.bonePaletteRegisterCount;

  // Only the components the game's shader variant consumes are meaningful (UE3 builds
  // 1/2/4-influence variants; a rigid-skin vertex may carry garbage weights). Mask to
  // the consumed count and renormalize; a degenerate sum falls back to the first bone
  // (the rigid-skin implicit weight 1).
  float4 weights = blendWeights;
  [unroll]
  for (uint component = 0; component < 4; component++) {
    if (component >= influenceCount) {
      weights[component] = 0.0f;
    }
  }

  const float weightSum = weights.x + weights.y + weights.z + weights.w;
  if (weightSum > 1e-4f) {
    weights /= weightSum;
  } else {
    weights = float4(1.0f, 0.0f, 0.0f, 0.0f);
  }

  const float3 currLocal = skinPosition(position, blendIndices, weights, currBase, boneIndexScale);
  const float3 prevLocal = skinPosition(position, blendIndices, weights, prevBase, boneIndexScale);

  const float4 clipPosition = mul(pc.clipFromLocal, float4(currLocal, 1.0f));

  VertexOutput output;
  output.position = clipPosition;
  output.currClip = clipPosition;
  output.prevClip = mul(pc.prevClipFromLocal, float4(prevLocal, 1.0f));

  return output;
}
