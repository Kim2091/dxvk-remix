/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx/pass/atmosphere/atmosphere_common.slangh"
#include "rtx/pass/atmosphere/cloud_nubis3_common.slangh"

layout(binding = 0) ConstantBuffer<AtmosphereArgs> atmosphereArgs;
layout(binding = 1, rgba16f) RWTexture3D<float4> densityOutput;
layout(binding = 2) SamplerState cloudSampler;
layout(binding = 3) Texture3D<float> cloudNvdfSdf;
layout(binding = 4) Texture3D<float4> cloudDetailNoise;

[shader("compute")]
[numthreads(8, 4, 8)]
void main(uint3 id : SV_DispatchThreadID) {
  uint3 dimensions;
  densityOutput.GetDimensions(dimensions.x, dimensions.y, dimensions.z);
  if (any(id >= dimensions)) {
    return;
  }
  AtmosphereArgs args = atmosphereArgs;
  if (args.cloudEnabled < 0.5f || args.cloudCoverageMean < 0.01f || args.cloudViewSamples == 0u) {
    densityOutput[id] = 0.0f;
    return;
  }
  const vec3 uvw = (vec3(id) + 0.5f) / vec3(dimensions);
  const vec3 worldPos = cloudVoxelUVWToWorld(uvw, args);
  const vec2 controls = sampleCloudBakeControls(worldPos.xz, args);
  const float heightFraction = computeCloudHeightFractionC(worldPos, getCloudPlanetCenterWorldKm(args),
    args.planetRadius, args.cloudAltitude, args.cloudThickness);
  float profile, detail, sdfKm, emptyRadiusKm;
  const float density = sampleCloudDensityNubis3(worldPos, heightFraction, controls.x, controls.y,
    1.0f, length(worldPos - args.cameraWorldPosYUpKm), args,
    cloudNvdfSdf, cloudDetailNoise, cloudSampler, false, profile, detail, sdfKm, emptyRadiusKm);
  densityOutput[id] = vec4(density, profile, detail, sdfKm);
}
