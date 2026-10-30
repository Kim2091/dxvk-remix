/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#ifndef BLOOM_H
#define BLOOM_H

#include "rtx/utility/shader_types.h"

// Energy-conserving bloom after Froyok (Lena Piquet), "Custom Bloom Post-Process in Unreal Engine" (2021):
// a chain of 13-tap downsamples, then 9-tap tent upsamples that lerp each level with the next smaller one by
// `radius`, and a final lerp of the scene towards the bloom by `intensity`. No threshold and no added energy.

#define BLOOM_DOWNSAMPLE_INPUT  0
#define BLOOM_DOWNSAMPLE_OUTPUT 1

// The upsample reads the next smaller (already combined) level and lerps it into the current level in place.
#define BLOOM_UPSAMPLE_INPUT    0
#define BLOOM_UPSAMPLE_OUTPUT   1

#define BLOOM_COMPOSITE_COLOR_INPUT_OUTPUT 0
#define BLOOM_COMPOSITE_BLOOM              1

// Push constants

struct BloomDownsampleArgs {
  float2 inputSizeInverse;
  uint2  downsampledOutputSize;
  float2 downsampledOutputSizeInverse;
  float  pad0;
  float  pad1;
};

struct BloomUpsampleArgs {
  float2 inputSizeInverse;
  uint2  upsampledOutputSize;
  float2 upsampledOutputSizeInverse;
  float  radius;  // Weight of the next smaller level against this one: higher = wider bloom.
  float  pad0;
};

struct BloomCompositeArgs {
  uint2  imageSize;
  float2 imageSizeInverse;
  float  intensity;  // Lerp of the scene towards the bloom.
};

#endif  // BLOOM_H
