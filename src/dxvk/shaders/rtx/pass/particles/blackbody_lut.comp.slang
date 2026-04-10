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

// Blackbody LUT generation compute shader.
// Maps temperature (500K - 6500K) to RGB emission color using the Tanner Helland
// analytical approximation (temperature → linear sRGB).
// Output: 256x1 RWTexture2D<float4> where RGB = normalized color, A = relative intensity.

layout(binding = 0)
RWTexture2D<float4> BlackbodyLUT;

// Direct temperature (Kelvin) to linear sRGB.
// Tanner Helland approximation, valid 1000K-40000K.  Values below 1000K are
// clamped — they produce deep red, which is correct for barely-glowing embers.
float3 blackbodyColor(float tempK)
{
  float t = max(tempK, 1000.0f) / 100.0f;
  float3 rgb;

  // Red
  if (t <= 66.0f)
    rgb.x = 1.0f;
  else
    rgb.x = saturate(1.292936f * pow(t - 60.0f, -0.1332047f));

  // Green
  if (t <= 66.0f)
    rgb.y = saturate(0.3900816f * log(t) - 0.6318414f);
  else
    rgb.y = saturate(1.129891f * pow(t - 60.0f, -0.0755148f));

  // Blue
  if (t >= 66.0f)
    rgb.z = 1.0f;
  else if (t <= 19.0f)
    rgb.z = 0.0f;
  else
    rgb.z = saturate(0.5432068f * log(t - 10.0f) - 1.1962540f);

  return rgb;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
  if (threadID.x >= 256u)
  {
    return;
  }

  // Map texel index [0, 255] to temperature range [500K, 6500K].
  static const float kMinTemp = 500.0f;
  static const float kMaxTemp = 6500.0f;
  float temperature = kMinTemp + (float(threadID.x) / 255.0f) * (kMaxTemp - kMinTemp);

  float3 rgb = blackbodyColor(temperature);

  // Normalize intensity relative to a reference temperature (6500K).
  // Physical blackbody follows T^4 (Stefan-Boltzmann), but that makes low
  // temperatures (campfire ~1000K) produce <0.1% of peak intensity — invisible
  // in practice.  A linear curve lets emissionIntensityScale control brightness
  // artistically while keeping the full temperature range usable.
  float relativeIntensity = temperature / kMaxTemp;

  // Normalize color so the brightest channel equals 1.
  float maxComp = max(rgb.x, max(rgb.y, rgb.z));
  if (maxComp > 0.0f)
  {
    rgb /= maxComp;
  }

  // Write result: normalized RGB color + perceptual relative intensity (alpha).
  BlackbodyLUT[uint2(threadID.x, 0u)] = float4(rgb, relativeIntensity);
}
