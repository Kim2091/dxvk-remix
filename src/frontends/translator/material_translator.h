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
#pragma once

// MaterialTranslator (translator toolkit).
//
// Fixed-function state in, scene-interface material descriptions out:
// - MaterialData: unchanged, still through SceneManager::resolveDrawMaterial, so material hashes and mod keys stay
//   exactly today's;
// - AlphaStateDesc and LegacySurfaceDesc: the scene-interface derivations of rtx_fork_material_derivation.h, the same
//   functions the old backend path calls (decision D5 (a)), from the plain record below.
// A frontend fills the record from its own state (D3D9 texture stages and render state today; glTexEnv / glAlphaFunc /
// glBlendFunc for the fixed-function OpenGL frontend later) and hands the descriptions over with its
// ExternalDrawState (alphaState / legacySurface), so the backend packs the GPU records from them.

#include "../../dxvk/rtx_render/rtx_fork_material_derivation.h"

namespace dxvk {
  class SceneManager;
  struct DrawCallState;

  namespace translator {

    // Plain fixed-function state of one draw. No API types beyond the backend's own enums.
    struct FixedFunctionState {
      // Texture stage 0 (colour and alpha) and the texture factor.
      RtTextureArgSource textureColorArg1Source = RtTextureArgSource::Texture;
      RtTextureArgSource textureColorArg2Source = RtTextureArgSource::None;
      DxvkRtTextureOperation textureColorOperation = DxvkRtTextureOperation::Modulate;
      RtTextureArgSource textureAlphaArg1Source = RtTextureArgSource::Texture;
      RtTextureArgSource textureAlphaArg2Source = RtTextureArgSource::None;
      DxvkRtTextureOperation textureAlphaOperation = DxvkRtTextureOperation::SelectArg1;
      uint32_t tFactor = 0xffffffff;
      bool textureFactorBlend = false;
      bool vertexColorBakedLighting = true;
      // Output merger: blend factors and alpha test.
      DxvkBlendMode blendMode {};
      VkCompareOp alphaTestCompareOp = VK_COMPARE_OP_ALWAYS;
      uint8_t alphaTestReferenceValue = 0;
      // Texture coordinate state and the user clip plane.
      Matrix4 textureTransform = Matrix4();
      TexGenMode texgenMode = TexGenMode::None;
      bool clipPlaneEnabled = false;
      Vector4 clipPlane { 0.f };
      // Instance categories (texture categories, cutout / hair card / particle / decal).
      CategoryFlags categories = 0;
    };

    struct MaterialTranslation {
      MaterialData material;
      AlphaStateDesc alphaState {};
      LegacySurfaceDesc legacySurface {};
    };

    class MaterialTranslator {
    public:
      // `draw` is the draw as the frontend prepared it for the scene (material data, categories); its material goes
      // through resolveDrawMaterial unchanged. The alpha / legacy descriptions come from `state` alone.
      static MaterialTranslation translate(SceneManager& scene, const DrawCallState& draw, const FixedFunctionState& state);

      // The descriptions alone (no material resolve), from a given render material.
      static AlphaStateDesc translateAlphaState(const FixedFunctionState& state, const MaterialData& material);
      static LegacySurfaceDesc translateLegacySurface(const FixedFunctionState& state);
    };

  } // namespace translator
} // namespace dxvk
