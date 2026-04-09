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
#pragma once

#include "rtx/pass/particles/particle_volume_binding_indices.h"

// Per-system per-frame constants uploaded to the GPU for all volume passes
struct ParticleVolumeConstants
{
  mat4 worldToVolume;
  mat4 volumeToWorld;
  mat4 prevWorldToProjection;
  vec3 aabbMin;
  float deltaTimeSecs;
  vec3 aabbMax;
  float absoluteTimeSecs;
  uvec3 gridDimension;
  uint frameIdx;
  float smokeDensity;
  float smokeAbsorptionCrossSection;
  float smokeDissipationRate;
  float fuelAmount;
  float burnTemperature;
  float coolingRate;
  float buoyancyCoefficient;
  float vorticityConfinement;
  vec3 windDirection;
  float emissionIntensityScale;
  float sceneScale;
  uint pressureIterations;
  uint renderingWidth;
  uint renderingHeight;
  float fluidCouplingStrength;
  vec3 upDirection;
  uint particleCount;
};

#ifndef __cplusplus

// Convert a 3D grid cell coordinate to the world-space center of that cell
vec3 particleVolumeGridToWorld(uvec3 gridCoord, ParticleVolumeConstants cb)
{
  const vec3 uvw = (vec3(gridCoord) + 0.5f) / vec3(cb.gridDimension);
  return cb.aabbMin + uvw * (cb.aabbMax - cb.aabbMin);
}

// Convert a world-space position to a fractional grid coordinate
vec3 particleVolumeWorldToGrid(vec3 worldPos, ParticleVolumeConstants cb)
{
  const vec3 uvw = (worldPos - cb.aabbMin) / (cb.aabbMax - cb.aabbMin);
  return uvw * vec3(cb.gridDimension) - 0.5f;
}

// Returns true if the given grid coordinate is within the volume bounds
bool particleVolumeIsInBounds(ivec3 gridCoord, ParticleVolumeConstants cb)
{
  return all(greaterThanEqual(gridCoord, ivec3(0, 0, 0))) &&
         all(lessThan(gridCoord, ivec3(cb.gridDimension)));
}

#endif // !__cplusplus
