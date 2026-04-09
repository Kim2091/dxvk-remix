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

// Pressure Jacobi iteration compute shader for particle volumes.
// Solves the pressure Poisson equation via a single Jacobi iteration step.
// Ping-pong between two pressure buffers for N iterations per frame.
// One thread per volume cell.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"

Texture3D<vec4>  VelocityInput  : register(t0, space1);
Texture3D<float> PressureInput  : register(t1, space1);
Texture3D<float> ObstacleInput  : register(t2, space1);
RWTexture3D<float> PressureOutput : register(u0, space2);

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Obstacle cells: copy pressure unchanged (no Jacobi update)
  if (ObstacleInput[threadID] > 0.5f)
  {
    PressureOutput[threadID] = PressureInput[threadID];
    return;
  }

  const ivec3 cell    = ivec3(threadID);
  const ivec3 dimMax  = ivec3(cb.gridDimension) - ivec3(1, 1, 1);

  // Clamp neighbor coordinates to grid bounds
  const ivec3 cxp = clamp(cell + ivec3(1, 0, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cxn = clamp(cell - ivec3(1, 0, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cyp = clamp(cell + ivec3(0, 1, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 cyn = clamp(cell - ivec3(0, 1, 0), ivec3(0, 0, 0), dimMax);
  const ivec3 czp = clamp(cell + ivec3(0, 0, 1), ivec3(0, 0, 0), dimMax);
  const ivec3 czn = clamp(cell - ivec3(0, 0, 1), ivec3(0, 0, 0), dimMax);

  // Sample velocity at the 6 face neighbors
  // Zero out the relevant velocity component at obstacle boundaries
  vec3 vxp = VelocityInput[cxp].xyz;
  vec3 vxn = VelocityInput[cxn].xyz;
  vec3 vyp = VelocityInput[cyp].xyz;
  vec3 vyn = VelocityInput[cyn].xyz;
  vec3 vzp = VelocityInput[czp].xyz;
  vec3 vzn = VelocityInput[czn].xyz;

  if (ObstacleInput[cxp] > 0.5f) { vxp.x = 0.0f; }
  if (ObstacleInput[cxn] > 0.5f) { vxn.x = 0.0f; }
  if (ObstacleInput[cyp] > 0.5f) { vyp.y = 0.0f; }
  if (ObstacleInput[cyn] > 0.5f) { vyn.y = 0.0f; }
  if (ObstacleInput[czp] > 0.5f) { vzp.z = 0.0f; }
  if (ObstacleInput[czn] > 0.5f) { vzn.z = 0.0f; }

  // Divergence via central differences
  const float divergence = ((vxp.x - vxn.x) + (vyp.y - vyn.y) + (vzp.z - vzn.z)) * 0.5f;

  // Sample pressure at the 6 face neighbors
  const float pxp = PressureInput[cxp];
  const float pxn = PressureInput[cxn];
  const float pyp = PressureInput[cyp];
  const float pyn = PressureInput[cyn];
  const float pzp = PressureInput[czp];
  const float pzn = PressureInput[czn];

  // Jacobi iteration: new pressure = (sum of neighbor pressures - divergence) / 6
  const float pressureNew = (pxp + pxn + pyp + pyn + pzp + pzn - divergence) / 6.0f;

  PressureOutput[threadID] = pressureNew;
}
