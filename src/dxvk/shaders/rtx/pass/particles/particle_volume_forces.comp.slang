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

// Force application compute shader for particle volumes.
// Applies buoyancy, wind, and vorticity confinement forces to the fluid
// velocity field. One thread per volume cell.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"

Texture3D<float> TemperatureInput : register(t0, space1);
Texture3D<float> ObstacleInput    : register(t1, space1);
RWTexture3D<vec4> VelocityOutput  : register(u0, space2);

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  if (any(greaterThanEqual(threadID, cb.gridDimension)))
  {
    return;
  }

  // Skip obstacle cells
  if (ObstacleInput[threadID] > 0.5f)
  {
    return;
  }

  const ivec3 cell = ivec3(threadID);
  const float dt   = cb.deltaTimeSecs;

  // Load current velocity
  vec4 velSample = VelocityOutput[threadID];
  vec3 vel       = velSample.xyz;

  // Buoyancy: hot fluid rises along the scene up direction
  const float temperature = TemperatureInput[threadID];
  vel += cb.upDirection * (temperature * cb.buoyancyCoefficient * cb.sceneScale * dt);

  // Wind: uniform directional force
  vel.xyz += cb.windDirection * dt;

  // Vorticity confinement
  if (cb.vorticityConfinement > 0.0f)
  {
    const vec3 curl = particleVolumeCalcCurl(VelocityOutput, cell, cb);

    // Compute curl magnitude at the six face-neighbours for gradient estimation
    const float cxp = length(particleVolumeCalcCurl(VelocityOutput, cell + ivec3(1, 0, 0), cb));
    const float cxn = length(particleVolumeCalcCurl(VelocityOutput, cell - ivec3(1, 0, 0), cb));
    const float cyp = length(particleVolumeCalcCurl(VelocityOutput, cell + ivec3(0, 1, 0), cb));
    const float cyn = length(particleVolumeCalcCurl(VelocityOutput, cell - ivec3(0, 1, 0), cb));
    const float czp = length(particleVolumeCalcCurl(VelocityOutput, cell + ivec3(0, 0, 1), cb));
    const float czn = length(particleVolumeCalcCurl(VelocityOutput, cell - ivec3(0, 0, 1), cb));

    // Central-difference gradient of curl magnitude
    const vec3 grad = vec3(cxp - cxn, cyp - cyn, czp - czn) * 0.5f;

    // Normalise to obtain confinement direction; guard against zero-length gradient
    const float gradLen = length(grad);
    if (gradLen > 1e-6f)
    {
      const vec3 N = grad / gradLen;
      vel.xyz += cross(N, curl) * cb.vorticityConfinement * dt;
    }
  }

  // Write updated velocity, preserving the w component
  VelocityOutput[threadID] = vec4(vel, velSample.w);
}
