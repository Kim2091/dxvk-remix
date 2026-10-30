/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <d3d9types.h>
#include "../util/util_vector.h"
#include "rtx_constants.h"

namespace dxvk {
  struct LightUtils {
    // Function to calculate a radiance value from a light.
    // This function will determine the distance from `light` that the brightness would fall below kLegacyLightEndValue, based on the attenuation function.
    // If the light would never attenuate to less than kLegacyLightEndValue, light.Range will be used instead.
    // It will then determine how bright the replacement light needs to be to have a brightness of kNewLightEndValue at the same distance.
    static float calculateIntensity(const D3DLIGHT9& light, const float radius);

    // Variant of calculateIntensity but also combine that intensity with the original light's diffuse color to determine the radiance.
    static Vector3 calculateRadiance(const D3DLIGHT9& light, const float radius);

    // Best fit light transform for a given legacy light
    static Matrix4 getLightTransform(const D3DLIGHT9& light);

    // The host's own identity of a point light, carried in the D3DLIGHT9 spot fields the point light does
    // not use: Theta = the low 16 bits, Phi = the high 16 bits, both as exact integers. 0 = none (a plain D3D9 point
    // light has both at 0; spot and directional lights never carry one). The FNV wrapper puts its NiLight's address
    // there, so a light that moves (its hash is its position) is matched to last frame's copy by identity.
    static uint32_t legacyIdentity(const D3DLIGHT9& light);

    // The host's asset key of a point light, which replacements and captures use instead of its position hash.
    // Returns kEmptyHash when rtx.scene.assetKeys is off or the light carries none.
    // The encoding is in docs/RemixApi.md (Asset keys).
    static XXH64_hash_t legacyAssetKey(const D3DLIGHT9& light);
  };
} // namespace dxvk
