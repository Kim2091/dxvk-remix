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

// Screen-space volume composite pass.
// For each pixel, reconstructs the primary ray from the GBuffer world position,
// tests the volume AABB for ray intersection, and if hit, ray-marches through
// the density/temperature fields to accumulate fire emission and smoke extinction.
// The result is blended into the composited color buffer.

#include "rtx/pass/particles/particle_volume_binding_indices.h"
#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_march.slangh"

layout(binding = PARTICLE_VOLUME_BINDING_CONSTANTS)
ConstantBuffer<ParticleVolumeConstants> cb;

layout(binding = PARTICLE_VOLUME_BINDING_DENSITY_INPUT)
Texture3D<float> t_density;

layout(binding = PARTICLE_VOLUME_BINDING_TEMPERATURE_INPUT)
Texture3D<float> t_temperature;

layout(binding = PARTICLE_VOLUME_BINDING_BLACKBODY_LUT_INPUT)
Texture2D<float4> t_blackbodyLUT;

layout(binding = PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)
SamplerState s_linear;

layout(binding = PARTICLE_VOLUME_BINDING_COMPOSITE_WORLD_POS_INPUT)
Texture2D<float4> t_worldPosition;

layout(binding = PARTICLE_VOLUME_BINDING_COMPOSITE_COLOR_INOUT)
RWTexture2D<float4> t_color;

// Standard slab-based ray-AABB intersection test.
bool particleVolumeRayAABBIntersect(
  float3 rayOrigin, float3 rayDir,
  float3 aabbMin, float3 aabbMax,
  inout float tMin, inout float tMax)
{
  float3 invDir = 1.0f / rayDir;
  float3 t0 = (aabbMin - rayOrigin) * invDir;
  float3 t1 = (aabbMax - rayOrigin) * invDir;
  float3 tNear = min(t0, t1);
  float3 tFar  = max(t0, t1);
  tMin = max(max(tNear.x, tNear.y), tNear.z);
  tMax = min(min(tFar.x, tFar.y), tFar.z);
  tMin = max(tMin, 0.0f);
  return tMax > tMin;
}

[shader("compute")]
[numthreads(16, 8, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
  if (threadID.x >= cb.renderingWidth || threadID.y >= cb.renderingHeight)
  {
    return;
  }

  const uint2 pixel = threadID.xy;

  // Read the primary hit world position from the GBuffer.
  // The .xyz contains world position; .w encodes the triangle normal.
  float4 worldPosData = t_worldPosition[pixel];
  float3 worldPos = worldPosData.xyz;

  // Reconstruct the primary ray from camera to the hit point.
  float3 rayOrigin = cb.cameraPosition;
  float3 toSurface = worldPos - rayOrigin;
  float hitDistance = length(toSurface);

  // For sky pixels (no surface hit), the world position may be zero or degenerate.
  // Skip volume rendering for sky pixels -- proper handling would require camera
  // inverse projection matrices which are not currently passed to this shader.
  // Volumes against geometry (the common case) are handled correctly.
  if (hitDistance < 0.01f)
  {
    return;
  }

  float3 rayDir = toSurface / hitDistance;

  // Test ray against the volume AABB.
  float tMin = 0.0f;
  float tMax = 0.0f;
  if (!particleVolumeRayAABBIntersect(rayOrigin, rayDir, cb.aabbMin, cb.aabbMax, tMin, tMax))
  {
    return;
  }

  // Clamp the far distance to the surface hit distance so that
  // volumes behind geometry are not rendered.
  tMax = min(tMax, hitDistance);
  if (tMax <= tMin)
  {
    return;
  }

  // Ray-march through the volume.
  ParticleVolumeMarchResult marchResult = particleVolumeMarch(
    rayOrigin, rayDir, tMin, tMax,
    t_density, t_temperature, t_blackbodyLUT, s_linear, cb);

  // Skip compositing if the volume is fully transparent at this pixel.
  if (marchResult.transmittance >= 0.999f)
  {
    return;
  }

  // Read the existing composited color and blend the volume contribution.
  float4 existingColor = t_color[pixel];
  float3 blended = existingColor.rgb * marchResult.transmittance + marchResult.radiance;
  t_color[pixel] = float4(blended, existingColor.a);
}
