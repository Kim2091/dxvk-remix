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

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"
#include "rtx/pass/particles/particle_system_binding_indices.h"

// Particle buffer input (one entry per particle)
[[vk::binding(PARTICLE_VOLUME_BINDING_PARTICLES_INPUT)]]
StructuredBuffer<GpuParticle> Particles;

// Volume output grids (read-write)
[[vk::binding(PARTICLE_VOLUME_BINDING_DENSITY_OUTPUT)]]
RWTexture3D<float> DensityOutput;

[[vk::binding(PARTICLE_VOLUME_BINDING_TEMPERATURE_OUTPUT)]]
RWTexture3D<float> TemperatureOutput;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint threadID : SV_DispatchThreadID)
{
  // Bounds-check against the total particle count for this dispatch
  if (threadID >= cb.particleCount)
  {
    return;
  }

  const GpuParticle particle = Particles[threadID];

  // Skip particles that are not actively contributing
  if (particle.isDead() || particle.isSleeping())
  {
    return;
  }

  // Skip particles whose world-space position lies outside the volume AABB
  if (any(particle.position < cb.aabbMin) || any(particle.position > cb.aabbMax))
  {
    return;
  }

  // Compute the base grid cell and the trilinear fractional offsets
  ivec3 baseCell;
  vec3 frac;
  particleVolumeCalcSplatWeights(particle.position, cb, baseCell, frac);

  // Pre-compute per-axis weights: (1-t) for lower neighbor, t for upper neighbor
  const float wx0 = 1.0f - frac.x;
  const float wx1 = frac.x;
  const float wy0 = 1.0f - frac.y;
  const float wy1 = frac.y;
  const float wz0 = 1.0f - frac.z;
  const float wz1 = frac.z;

  // Values to splat
  const float densityContrib     = cb.smokeDensity;
  const float temperatureContrib = cb.fuelAmount * cb.burnTemperature;

  // Iterate over all 8 neighboring cells in the trilinear kernel
  for (uint iz = 0u; iz < 2u; iz++)
  {
    const float wz = (iz == 0u) ? wz0 : wz1;

    for (uint iy = 0u; iy < 2u; iy++)
    {
      const float wy = (iy == 0u) ? wy0 : wy1;

      for (uint ix = 0u; ix < 2u; ix++)
      {
        const float wx = (ix == 0u) ? wx0 : wx1;

        const ivec3 cell = baseCell + ivec3(ix, iy, iz);

        // Skip cells that fall outside the volume
        if (!particleVolumeIsInBounds(cell, cb))
        {
          continue;
        }

        const float weight = wx * wy * wz;
        const uint3 coord  = uint3(cell);

        // Note: may need atomic implementation for correctness under concurrent
        // writes from multiple threads splatting into the same cell.
        // InterlockedAdd on a uint reinterpretation of the float texel is the
        // portable Vulkan/Slang approach; direct += is used here for simplicity
        // and will be refined during integration testing.
        DensityOutput[coord]     += densityContrib     * weight;
        TemperatureOutput[coord] += temperatureContrib * weight;
      }
    }
  }
}
