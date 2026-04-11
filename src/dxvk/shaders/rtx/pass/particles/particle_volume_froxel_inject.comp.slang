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

// Froxel grid injection compute shader for particle volumes.
// Downsamples packed density4 (smoke/temperature) and colormap emission into
// the global froxel grid so the rest of the volumetric pipeline (god rays,
// dust lighting, atmospheric scattering) accounts for smoke and fire.
// One thread per froxel cell.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"

[[vk::binding(PARTICLE_VOLUME_BINDING_DENSITY4_INPUT)]]
Texture3D<vec4>   Density4Input;

[[vk::binding(PARTICLE_VOLUME_BINDING_COLORMAP_INPUT)]]
Texture2D<vec4>   ColormapInput;

[[vk::binding(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)]]
SamplerState      LinearSampler;

[[vk::binding(PARTICLE_VOLUME_BINDING_FROXEL_DENSITY_OUTPUT)]]
RWTexture3D<vec4> FroxelOutput;

// Per-dispatch froxel grid parameters
struct FroxelInjectConstants
{
  uvec3 froxelGridDimensions;
  float froxelMaxDistance;
  float froxelDepthExponent;
  float nearPlane;
  float pad0;
  float pad1;
  mat4  froxelToWorld; // Transforms froxel UVW (mapped to [-1,1]^3) to world space
};

layout(binding = 1)
ConstantBuffer<FroxelInjectConstants> fc;

// Negligibility thresholds: froxels below both values are skipped early
static const float kSmokeThreshold       = 1e-5f;
static const float kTemperatureThreshold = 0.01f; // normalized [0,1]

// Convert a froxel integer coordinate to a world-space position.
// Z uses a power-law distribution so near slabs are thinner than far slabs.
vec3 froxelToWorldPosition(uvec3 froxelCoord, FroxelInjectConstants froxelCb)
{
  vec3 uvw = (vec3(froxelCoord) + 0.5f) / vec3(froxelCb.froxelGridDimensions);

  // Apply depth exponent for non-linear (logarithmic-style) depth distribution
  uvw.z = pow(uvw.z, froxelCb.froxelDepthExponent);

  const vec4 worldPos = mul(froxelCb.froxelToWorld, vec4(uvw * 2.0f - 1.0f, 1.0f));
  return worldPos.xyz / worldPos.w;
}

// Returns true when worldPos lies strictly inside the particle volume AABB
bool isInsideVolumeAABB(vec3 worldPos)
{
  return all(greaterThan(worldPos, cb.aabbMin)) &&
         all(lessThan(worldPos, cb.aabbMax));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uvec3 threadID : SV_DispatchThreadID)
{
  // Bounds-check against the froxel grid dimensions
  if (any(greaterThanEqual(threadID, fc.froxelGridDimensions)))
  {
    return;
  }

  // Reconstruct world-space position for the centre of this froxel cell
  const vec3 worldPos = froxelToWorldPosition(threadID, fc);

  // Reject froxels that do not overlap the particle volume AABB
  if (!isInsideVolumeAABB(worldPos))
  {
    return;
  }

  // Sample packed density4 from the particle volume via trilinear interpolation
  const vec4 d = particleVolumeSampleDensity4(Density4Input, LinearSampler, worldPos, cb);
  const float smoke = d.w;
  const float temp  = d.x;

  // Skip froxels with negligible contribution to avoid polluting the froxel grid
  if (smoke < kSmokeThreshold && temp < kTemperatureThreshold)
  {
    return;
  }

  // Convert smoke to extinction coefficient
  const float extinction = smoke * cb.absorptionCrossSection;

  // Convert temperature to emission colour via the precomputed colormap
  const vec4 colormapColor = ColormapInput.SampleLevel(LinearSampler, vec2(temp, 0.5f), 0);
  const vec3 emission = colormapColor.rgb * colormapColor.a * cb.colorScale;

  // Additive contribution: existing froxel radiance is preserved; particle
  // emission and extinction are blended on top.
  // Layout: xyz = emission (rgb), w = extinction (scalar)
  const vec4 existing = FroxelOutput[threadID];
  FroxelOutput[threadID] = vec4(existing.xyz + emission, existing.w + extinction);
}
