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
#include "rtx/utility/shader_types.h"

// Per-system per-frame constants uploaded to the GPU for all volume passes
struct ParticleVolumeConstants
{
  mat4 worldToVolume;
  mat4 volumeToWorld;
  mat4 prevWorldToProjection;

  vec3 aabbMin;               float deltaTimeSecs;    // keep existing name to avoid renaming in all shaders
  vec3 aabbMax;               float absoluteTimeSecs; // keep existing name
  uvec3 gridDimension;        uint frameIdx;

  // Combustion (Nguyen/Fedkiw/NvFlow)
  float ignitionTemp;         // [0,1] threshold for combustion (default 0.05)
  float burnPerTemp;          // burn rate per excess temp (default 4.0)
  float fuelPerBurn;          // fuel consumed per burn (default 0.25) — clamps burn: burn = min(burn, fuelPerBurn * fuel)
  float tempPerBurn;          // temp generated per burn (default 5.0)
  float smokePerBurn;         // smoke generated per burn (default 3.0)
  float coolingRate;          // exponential cooling (default 1.5)
  float divergencePerBurn;    // pressure expansion (default 0.0)
  float emitterCoupleRate;    // emission blend rate /sec (default 3.0)

  // Damping (CPU-computed per frame)
  vec4 dampingRate;           // 1 - pow(1 - damping, dt), per channel (temp/fuel/burn/smoke)
  vec4 fadeRate;              // absolute fade /sec, per channel
  vec4 velDampingRate;        // velocity damping (xyz used, w unused)
  vec4 velFadeRate;           // velocity fade (xyz used, w unused)

  // Forces
  float buoyancyPerTemp;      // default 2.0
  float buoyancyPerSmoke;     // default 0.0
  float buoyancyMaxSmoke;     // default 1.0
  float gravityMagnitude;     // default 100.0
  vec3 upDirection;           float vorticityConfinement;
  vec3 windDirection;         float fluidCouplingStrength;

  // Rendering
  float absorptionCrossSection; // smoke extinction coefficient
  float colorScale;           // emission brightness multiplier
  float alphaScale;           // smoke opacity multiplier
  float shadowFactor;         // burn brightness modulation

  // Pipeline
  uint pressureIterations;
  uint particleCount;
  uint renderingWidth;
  uint renderingHeight;
  vec3 cameraPosition;        float maxTimeToLive;
  float sceneScale;
  float fuelAmount;           // normalized fuel injection target [0,1]
  uint debugMode;             // 0=off, 1=temperature, 2=fuel, 3=burn, 4=smoke, 5=velocity mag, 6=density4 all
  float pad1;
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
