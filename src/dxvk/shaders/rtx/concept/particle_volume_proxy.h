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
// Layout: 12 floats = 48 bytes, naturally aligned for StructuredBuffer access.
struct MemoryParticleVolume
{
  vec3 aabbMin;
  float smokeAbsorptionCrossSection;

  vec3 aabbMax;
  float emissionIntensityScale;

  uvec3 gridDimension;
  uint volumeIndex;   // Index into the per-frame volume texture arrays (0..MAX_PARTICLE_VOLUMES-1)
};

// Flag bit set in instanceCustomIndex for intersection primitives that are
// particle volume proxies rather than billboards. The lower bits contain the
// index into the particleVolumeProxies StructuredBuffer.
#define PARTICLE_VOLUME_PROXY_CUSTOM_INDEX_FLAG (1u << 23)
#define PARTICLE_VOLUME_PROXY_INDEX_MASK        0x007FFFFFu

#endif
