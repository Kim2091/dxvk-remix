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
// Maps temperature (500K - 6500K) to RGB emission color using Planck's blackbody
// radiation law integrated against CIE 1931 XYZ color matching functions.
// Output: 256x1 RWTexture2D<float4> where RGB = normalized color, A = relative intensity.

layout(binding = 0)
RWTexture2D<float4> BlackbodyLUT;

// CIE 1931 2-degree observer color matching functions (approximation).
// Wyman, Sloan, Shirley (2013) analytic Gaussian approximations.
float3 cieXyzFromWavelength(float lambda)
{
  float t1 = (lambda - 442.0f) * ((lambda < 442.0f) ? 0.0624f : 0.0374f);
  float t2 = (lambda - 599.8f) * ((lambda < 599.8f) ? 0.0264f : 0.0323f);
  float t3 = (lambda - 474.0f) * ((lambda < 474.0f) ? 0.0264f : 0.0323f);
  float t4 = (lambda - 556.3f) * ((lambda < 556.3f) ? 0.0264f : 0.0323f);

  float x = 0.362f * exp(-0.5f * t1 * t1) + 1.056f * exp(-0.5f * t2 * t2) - 0.065f * exp(-0.5f * 0.0f);
  float y = 0.821f * exp(-0.5f * t3 * t3) + 0.286f * exp(-0.5f * t4 * t4);

  float t5 = (lambda - 437.0f) * ((lambda < 437.0f) ? 0.0845f : 0.0278f);
  float t6 = (lambda - 459.0f) * ((lambda < 459.0f) ? 0.0385f : 0.0725f);
  float z = 1.217f * exp(-0.5f * t5 * t5) + 0.681f * exp(-0.5f * t6 * t6);

  return float3(x, y, z);
}

// Planck's law: spectral radiance at wavelength lambda (nm) and temperature T (K).
float planck(float lambda, float T)
{
  static const float h  = 6.62607015e-34f; // Planck constant (J*s)
  static const float c  = 2.99792458e8f;   // Speed of light (m/s)
  static const float kB = 1.380649e-23f;   // Boltzmann constant (J/K)

  float lambdaM = lambda * 1e-9f; // Convert nm to meters
  float nom   = 2.0f * h * c * c / (lambdaM * lambdaM * lambdaM * lambdaM * lambdaM);
  float denom = exp(h * c / (lambdaM * kB * T)) - 1.0f;

  return nom / denom;
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
  static const float kMinTemp  = 500.0f;
  static const float kMaxTemp  = 6500.0f;
  float temperature = kMinTemp + (float(threadID.x) / 255.0f) * (kMaxTemp - kMinTemp);

  // Integrate Planck's law over visible wavelengths (380-780nm) weighted by CIE XYZ
  // color matching functions using a simple Riemann sum with 5nm steps.
  float3 xyz = float3(0.0f, 0.0f, 0.0f);
  static const float kLambdaMin  = 380.0f;
  static const float kLambdaMax  = 780.0f;
  static const float kLambdaStep = 5.0f;

  for (float lambda = kLambdaMin; lambda <= kLambdaMax; lambda += kLambdaStep)
  {
    float spectralPower  = planck(lambda, temperature);
    float3 matching      = cieXyzFromWavelength(lambda);
    xyz += matching * spectralPower * kLambdaStep;
  }

  // Convert XYZ to linear sRGB using the standard D65 matrix.
  float3 rgb;
  rgb.x =  3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z;
  rgb.y = -0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z;
  rgb.z =  0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z;

  // Normalize so the brightest channel equals 1; store relative intensity in alpha.
  // Clamp negative values that can arise from the matrix conversion.
  float maxComp = max(rgb.x, max(rgb.y, rgb.z));
  if (maxComp > 0.0f)
  {
    rgb /= maxComp;
  }
  rgb = max(rgb, float3(0.0f, 0.0f, 0.0f));

  // Write result: normalized RGB color + relative intensity (alpha).
  BlackbodyLUT[uint2(threadID.x, 0u)] = float4(rgb, maxComp);
}
