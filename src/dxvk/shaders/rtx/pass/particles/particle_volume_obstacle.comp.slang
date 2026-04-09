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

// Obstacle rasterization compute shader for particle volumes.
// For each cell in the volume grid, reprojects the cell center into the previous
// frame's screen space and samples the GBuffer world position to determine whether
// the cell is occupied by a solid surface.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"

layout(binding = PARTICLE_VOLUME_BINDING_PREV_WORLD_POSITION_INPUT)
Texture2D<vec4> PrevWorldPosition;

layout(binding = PARTICLE_VOLUME_BINDING_OBSTACLE_OUTPUT)
RWTexture3D<float> ObstacleOutput;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Compute world-space position of this cell's center
  const vec3 cellCenter = particleVolumeGridToWorld(threadID, cb);

  // Compute the cell radius from the cell size diagonal
  const vec3 cellSize = (cb.aabbMax - cb.aabbMin) / vec3(cb.gridDimension);
  const float cellRadius = length(cellSize) * 0.5f;

  // Project cell center into previous frame clip space
  const vec4 clipPos = mul(cb.prevWorldToProjection, vec4(cellCenter, 1.0f));

  // Cell center is behind the camera — treat as open
  if (clipPos.w <= 0.0f)
  {
    ObstacleOutput[threadID] = 0.0f;
    return;
  }

  // Convert to NDC then to UV; flip Y for Vulkan (NDC +Y is up, UV +Y is down)
  const vec2 ndc = clipPos.xy / clipPos.w;
  const vec2 uv  = ndc * vec2(0.5f, -0.5f) + vec2(0.5f, 0.5f);

  // Out-of-screen — treat as open
  if (any(lessThan(uv, vec2(0.0f, 0.0f))) || any(greaterThan(uv, vec2(1.0f, 1.0f))))
  {
    ObstacleOutput[threadID] = 0.0f;
    return;
  }

  // Map UV to integer pixel coordinates
  const ivec2 pixel = ivec2(
    int(uv.x * float(cb.renderingWidth)),
    int(uv.y * float(cb.renderingHeight))
  );

  // Sample the previous frame GBuffer world position
  const vec3 surfaceWorldPos = PrevWorldPosition[pixel].xyz;

  // Mark cell solid if the surface is within one cell radius of the cell center
  const float dist = length(surfaceWorldPos - cellCenter);
  ObstacleOutput[threadID] = (dist <= cellRadius) ? 1.0f : 0.0f;
}
