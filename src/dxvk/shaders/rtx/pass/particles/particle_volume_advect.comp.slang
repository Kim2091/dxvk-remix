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

// Semi-Lagrangian advection compute shader for particle volumes.
// Dispatched twice per frame:
//   Pass 5 (advectVelocity == 0): advects density and temperature fields.
//   Pass 6 (advectVelocity == 1): advects the velocity field itself.
// One thread per volume cell.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"

Texture3D<vec4>  VelocityInput    : register(t0, space1);
Texture3D<float> DensityInput     : register(t1, space1);
Texture3D<float> TemperatureInput : register(t2, space1);
Texture3D<float> ObstacleInput    : register(t3, space1);
SamplerState     LinearSampler    : register(s0, space1);

RWTexture3D<float> DensityOutput     : register(u0, space2);
RWTexture3D<float> TemperatureOutput : register(u1, space2);
RWTexture3D<vec4>  VelocityOutput    : register(u2, space2);

struct AdvectPushConstants
{
  uint advectVelocity; // 0 = advect density/temp, 1 = advect velocity
  uint useMacCormack;  // 0 = basic semi-Lagrangian, 1 = MacCormack (reserved for future)
};

[[vk::push_constant]] AdvectPushConstants pushConst;

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Obstacle cells receive zero output
  if (ObstacleInput[threadID] > 0.5f)
  {
    if (pushConst.advectVelocity == 0u)
    {
      DensityOutput[threadID]     = 0.0f;
      TemperatureOutput[threadID] = 0.0f;
    }
    else
    {
      VelocityOutput[threadID] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    return;
  }

  const float dt = cb.deltaTimeSecs;

  // World-space position of this cell's center
  const vec3 cellWorldPos = particleVolumeGridToWorld(threadID, cb);

  // Sample velocity at this cell and backtrace
  const vec3 velocity  = particleVolumeSampleVelocity(VelocityInput, LinearSampler, cellWorldPos, cb);
  vec3       sourcePos = cellWorldPos - velocity * dt;

  // Clamp source position to the volume AABB
  sourcePos = clamp(sourcePos, cb.aabbMin, cb.aabbMax);

  if (pushConst.advectVelocity == 0u)
  {
    // Advect density and temperature
    float density     = particleVolumeSampleDensity(DensityInput, LinearSampler, sourcePos, cb);
    float temperature = particleVolumeSampleTemperature(TemperatureInput, LinearSampler, sourcePos, cb);

    // Apply dissipation and cooling
    density     *= max(1.0f - cb.smokeDissipationRate * dt, 0.0f);
    temperature *= max(1.0f - cb.coolingRate          * dt, 0.0f);

    DensityOutput[threadID]     = density;
    TemperatureOutput[threadID] = temperature;
  }
  else
  {
    // Advect velocity
    const vec3 advectedVel = particleVolumeSampleVelocity(VelocityInput, LinearSampler, sourcePos, cb);
    VelocityOutput[threadID] = vec4(advectedVel, 0.0f);
  }
}
