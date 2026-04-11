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

// Fire light extraction compute shader for ReSTIR integration.
// Scans the packed density4 field (temperature channel) of a particle volume
// and extracts a light cluster that can be registered as a sphere light in the
// ReSTIR light pool.  Uses the colormap for emission color lookup.
// One workgroup processes the entire volume via parallel reduction; the result
// is a single VolumeLightCluster written atomically to the output buffer.

#include "rtx/pass/particles/particle_volume_common.h"
#include "rtx/pass/particles/particle_volume_bindings.slangh"
#include "rtx/pass/particles/particle_volume_helpers.slangh"

// ---------------------------------------------------------------------------
// Resource bindings
// ---------------------------------------------------------------------------

[[vk::binding(PARTICLE_VOLUME_BINDING_DENSITY4_INPUT)]]
Texture3D<vec4>     Density4Input;

[[vk::binding(PARTICLE_VOLUME_BINDING_COLORMAP_INPUT)]]
Texture2D<vec4>     ColormapInput;

[[vk::binding(PARTICLE_VOLUME_BINDING_LINEAR_SAMPLER)]]
SamplerState        LinearSampler;

// Output cluster buffer (max 4 clusters)
struct VolumeLightCluster
{
  vec3  position;
  float radius;
  vec3  emission;
  float pad0;
};

[[vk::binding(PARTICLE_VOLUME_BINDING_LIGHT_OUTPUT)]]
RWStructuredBuffer<VolumeLightCluster> LightOutput;

[[vk::binding(PARTICLE_VOLUME_BINDING_LIGHT_OUTPUT + 1)]]
RWStructuredBuffer<uint>               LightCountOutput;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static const float kMinLightTemperature = 0.1f; // normalized [0,1] threshold for light contribution
static const uint  kMaxClusters         = 4u;
static const uint  kGroupSize           = 128u;

// ---------------------------------------------------------------------------
// Shared memory for parallel reduction
// ---------------------------------------------------------------------------

groupshared vec3  s_positionSum[128];
groupshared vec3  s_emissionSum[128];
groupshared float s_weightSum[128];
groupshared uint  s_cellCount[128];

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint threadID : SV_GroupIndex)
{
  // Total number of cells in the volume
  const uint totalCells = cb.gridDimension.x * cb.gridDimension.y * cb.gridDimension.z;

  // Each thread processes a strided range of cells
  const uint cellsPerThread = (totalCells + kGroupSize - 1u) / kGroupSize;
  const uint cellStart      = threadID * cellsPerThread;
  const uint cellEnd        = min(cellStart + cellsPerThread, totalCells);

  // Per-thread accumulators
  vec3  posSum    = vec3(0.0f, 0.0f, 0.0f);
  vec3  emitSum   = vec3(0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  uint  cellCount = 0u;

  for (uint flatIdx = cellStart; flatIdx < cellEnd; flatIdx++)
  {
    // Reconstruct 3D grid coordinate from flat index
    const uint x = flatIdx % cb.gridDimension.x;
    const uint y = (flatIdx / cb.gridDimension.x) % cb.gridDimension.y;
    const uint z = flatIdx / (cb.gridDimension.x * cb.gridDimension.y);
    const uvec3 gridCoord = uvec3(x, y, z);

    const vec4 d = Density4Input[gridCoord];
    const float temp = d.x;

    if (temp > kMinLightTemperature)
    {
      const vec3  worldPos = particleVolumeGridToWorld(gridCoord, cb);
      const vec4  colormapColor = ColormapInput.SampleLevel(LinearSampler, vec2(temp, 0.5f), 0);
      const vec3  emission = colormapColor.rgb * colormapColor.a * cb.colorScale;
      const float weight   = temp;

      posSum    += worldPos * weight;
      emitSum   += emission;
      weightSum += weight;
      cellCount += 1u;
    }
  }

  // Store per-thread results into shared memory
  s_positionSum[threadID] = posSum;
  s_emissionSum[threadID] = emitSum;
  s_weightSum[threadID]   = weightSum;
  s_cellCount[threadID]   = cellCount;

  GroupMemoryBarrierWithGroupSync();

  // ---------------------------------------------------------------------------
  // Parallel reduction: stride-halving from 64 down to 1
  // ---------------------------------------------------------------------------

  if (threadID < 64u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 64u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 64u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 64u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 64u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 32u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 32u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 32u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 32u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 32u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 16u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 16u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 16u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 16u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 16u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 8u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 8u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 8u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 8u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 8u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 4u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 4u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 4u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 4u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 4u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 2u)
  {
    s_positionSum[threadID] += s_positionSum[threadID + 2u];
    s_emissionSum[threadID] += s_emissionSum[threadID + 2u];
    s_weightSum[threadID]   += s_weightSum[threadID   + 2u];
    s_cellCount[threadID]   += s_cellCount[threadID   + 2u];
  }
  GroupMemoryBarrierWithGroupSync();

  if (threadID < 1u)
  {
    s_positionSum[0] += s_positionSum[1];
    s_emissionSum[0] += s_emissionSum[1];
    s_weightSum[0]   += s_weightSum[1];
    s_cellCount[0]   += s_cellCount[1];
  }
  GroupMemoryBarrierWithGroupSync();

  // ---------------------------------------------------------------------------
  // Thread 0 writes the cluster result
  // ---------------------------------------------------------------------------

  if (threadID == 0u && s_weightSum[0] > 0.0f)
  {
    // Claim an output slot atomically
    uint slotIdx = 0u;
    InterlockedAdd(LightCountOutput[0], 1u, slotIdx);

    // Discard if we have exceeded the maximum cluster count
    if (slotIdx < kMaxClusters)
    {
      VolumeLightCluster cluster;
      cluster.position = s_positionSum[0] / s_weightSum[0];
      cluster.emission = s_emissionSum[0];
      cluster.radius   = length(cb.aabbMax - cb.aabbMin) * 0.5f;
      cluster.pad0     = 0.0f;

      LightOutput[slotIdx] = cluster;
    }
  }
}
