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

// Pressure projection compute shader for particle volumes.
// Subtracts the pressure gradient from the velocity field to enforce
// incompressibility. Runs once after all Jacobi pressure iterations.
// One thread per volume cell.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"

Texture3D<float>  PressureInput   : register(t0, space1);
Texture3D<float>  ObstacleInput   : register(t1, space1);
RWTexture3D<vec4> VelocityOutput  : register(u0, space2);

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Obstacle cells: zero velocity
  if (ObstacleInput[threadID] > 0.5f)
  {
    VelocityOutput[threadID] = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  const ivec3 cell   = ivec3(threadID);
  const ivec3 dimMax = ivec3(cb.gridDimension) - ivec3(1, 1, 1);

  // Clamp neighbor coordinates to grid bounds
  const ivec3 cxp = clamp(cell + ivec3(1, 0, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cxn = clamp(cell - ivec3(1, 0, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cyp = clamp(cell + ivec3(0, 1, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cyn = clamp(cell - ivec3(0, 1, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 czp = clamp(cell + ivec3(0, 0, 1), ivec3(0, 0, 0), dimMax);
  const ivec3 czn = clamp(cell - ivec3(0, 0, 1), ivec3(0, 0, 0), dimMax);

  // Pressure gradient via central differences
  const vec3 gradient = vec3(
    PressureInput[cxp] - PressureInput[cxn],
    PressureInput[cyp] - PressureInput[cyn],
    PressureInput[czp] - PressureInput[czn]
  ) * 0.5f;

  // Load current velocity and subtract pressure gradient
  const vec4 velSample = VelocityOutput[threadID];
  vec3 vel = velSample.xyz - gradient;

  // Enforce Neumann boundary conditions: zero out velocity components
  // pointing into obstacle neighbors
  if (ObstacleInput[cxp] > 0.5f && vel.x > 0.0f) { vel.x = 0.0f; }
  if (ObstacleInput[cxn] > 0.5f && vel.x < 0.0f) { vel.x = 0.0f; }
  if (ObstacleInput[cyp] > 0.5f && vel.y > 0.0f) { vel.y = 0.0f; }
  if (ObstacleInput[cyn] > 0.5f && vel.y < 0.0f) { vel.y = 0.0f; }
  if (ObstacleInput[czp] > 0.5f && vel.z > 0.0f) { vel.z = 0.0f; }
  if (ObstacleInput[czn] > 0.5f && vel.z < 0.0f) { vel.z = 0.0f; }

  // Write updated velocity, preserving the w component
  VelocityOutput[threadID] = vec4(vel, velSample.w);
}
