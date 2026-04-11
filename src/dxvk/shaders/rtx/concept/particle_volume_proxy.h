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
#ifndef PARTICLE_VOLUME_PROXY_H
#define PARTICLE_VOLUME_PROXY_H

// GPU-side descriptor for a particle volume proxy registered in the unordered TLAS.
// The resolve shader reads this buffer when a ray hits an OBJECT_MASK_UNORDERED_VOLUME_PROXY
// intersection primitive, then performs an inline ray-march through the volume.
//
// Layout: 20 floats = 80 bytes (5 x vec4), naturally aligned for StructuredBuffer access.
struct MemoryParticleVolume
{
  vec3 aabbMin;
  float absorptionCrossSection;   // Smoke extinction coefficient

  vec3 aabbMax;
  float colorScale;               // Emission brightness multiplier

  uvec3 gridDimension;
  uint volumeIndex;               // Index into the per-frame volume texture arrays (0..MAX_PARTICLE_VOLUMES-1)

  vec3 upDirection;               // Scene up direction used as shadow march direction for self-shadowing
  float alphaScale;               // Smoke opacity multiplier

  float shadowFactor;             // Burn brightness modulation
  uint debugMode;                 // 0=off, 1=temp, 2=fuel, 3=burn, 4=smoke, 5=vel mag, 6=all
  float pad1;
  float pad2;
};

// Flag bit set in instanceCustomIndex for intersection primitives that are
// particle volume proxies rather than billboards. The lower bits contain the
// index into the particleVolumeProxies StructuredBuffer.
#define PARTICLE_VOLUME_PROXY_CUSTOM_INDEX_FLAG (1u << 23)
#define PARTICLE_VOLUME_PROXY_INDEX_MASK        0x007FFFFFu

#endif
