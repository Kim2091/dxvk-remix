/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_resources.h"
#include "rtx_option.h"

namespace dxvk {

  class DxvkDevice;
  struct RtFrameFeatures;
  struct VariantTableSelfCheckResult;
  class RtxContext;

  class DxvkPathtracerIntegrateDirect : public CommonDeviceObject {
  public:
    enum class RaytraceMode {
      RayQuery = 0,
      RayQueryRayGen,
      Count
    };

    // Which compile-time slice of the direct integrator (CS RayQuery path) a frame can use.
    // The Lean levels run as a shade kernel and a trace kernel over a per-pixel record.
    enum class SlimLevel {
      Full = 0, // diffusion-profile SSS compiled in (255 regs on Blackwell)
      NoSss,    // INTEGRATE_DIRECT_SSS_PROFILE=0: no diffusion-profile SSS instance in this frame's TLAS (163 regs)
      Lean,     // + INTEGRATE_DIRECT_LEAN_VISIBILITY=1: no translucent / thin-opaque shadow-ray hits possible
      // rtx.integrateDirect.coreMaterial. Lean with the NEE cutout candidates read from the core material
      // record (VISIBILITY_CORE_MATERIAL): LeanCore applies only the TextureOps part of the legacy record (no surface with
      // other legacy state this frame), LeanLegacy the whole LegacySurface record.
      LeanCore,
      LeanLegacy,
    };

    RTX_OPTION("rtx.integrateDirect", bool, slimVariants, true,
               "Uses reduced direct lighting shaders (ray query mode) that leave out features the current frame cannot "
               "use: diffusion-profile subsurface scattering when no such instance is in the scene, and translucent and "
               "thin-opaque shadow handling when rtx.enableDirectTranslucentShadows is off and nothing needs it. They "
               "need fewer registers, which saves GPU time.\n"
               "The slimmest levels split direct lighting into a shading pass and a tracing pass, with a 48-byte record "
               "per pixel (about 83 MiB at 1707x1067) while they are in use. The record stores light radiance at half "
               "precision, so the image is nearly the same.");
    RTX_OPTION("rtx.integrateDirect", bool, coreMaterial, true,
               "With the leanest direct lighting shader (rtx.integrateDirect.slimVariants), cutout tests on shadow rays "
               "read the compact material records (rtx.coreMaterial.build and rtx.coreMaterial.legacyRecord, both "
               "needed) instead of the full material records. Not used on frames with displacement, eyes, ray portal "
               "materials, ray traced render targets or debug views. The image is the same.");

    explicit DxvkPathtracerIntegrateDirect(DxvkDevice* device);
    ~DxvkPathtracerIntegrateDirect() = default;

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    void dispatch(class RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    static const char* raytraceModeToString(RaytraceMode raytraceMode);

    // Exhaustive check of the variant table against selectSlimLevel (rtx_fork_variant_tables.h).
    static void runVariantTableSelfCheck(VariantTableSelfCheckResult& result);

  private:
    static DxvkRaytracingPipelineShaders getPipelineShaders(const bool useRayQuery, const bool ommEnabled, const bool deferredNrcTrainingSetup);
    Rc<DxvkShader> getComputeShader(const bool deferredNrcTrainingSetup, const SlimLevel slimLevel) const;
    static SlimLevel selectSlimLevel(const RtFrameFeatures& features);
    static const char* slimLevelToString(SlimLevel level);
    // Lean -> LeanCore / LeanLegacy when rtx.integrateDirect.coreMaterial applies this frame
    // (coreTableValid / legacyTableValid: RaytraceArgs::coreMaterialTable / legacySurfaceTable are set).
    static SlimLevel selectCoreMaterialLevel(const RtFrameFeatures& features, const SlimLevel level, const bool coreTableValid,
                                             const bool legacyTableValid);
    static bool coreMaterialOptionActive();

    // rtx.shader.variantTables. Picks the level from the table (checked against selectSlimLevel),
    // applies the compile fallback and writes the [IntegrateDirect] slim level line.
    SlimLevel selectVariantTableLevel(RtxContext* ctx, const bool deferredNrcTrainingSetup, const SlimLevel legacyLevel,
                                      const bool coreTableValid, const bool legacyTableValid);
    std::string m_loggedVariantInfo;
    uint32_t m_lastVariantLogFrame = UINT32_MAX;

    // The split kernels of the Lean levels.
    static bool isSplitLevel(const SlimLevel level);
    Rc<DxvkShader> getSplitShadeShader(const bool dedicatedSun) const;
    Rc<DxvkShader> getSplitTraceShader(const SlimLevel slimLevel) const;
    // Creates the kernels a level binds, for prewarming.
    void prewarmLevel(const bool deferredNrcTrainingSetup, const SlimLevel level) const;
    bool isLevelReady(RtxContext& ctx, const bool deferredNrcTrainingSetup, const SlimLevel level) const;
    // Sizes the record buffer for the camera resolution and marks it used this frame.
    // Returns false if it could not be allocated.
    bool prepareSplitRecords(const uint32_t width, const uint32_t height, const uint32_t frameId);
    // Frees the record buffer once no frame has used it for kSplitRecordIdleFrames.
    void releaseIdleSplitRecords(const uint32_t frameId);
    Rc<DxvkBuffer> m_splitRecords;
    uint32_t m_splitLastUseFrame = 0;
    VkDeviceSize m_splitFailedSize = 0;
    bool m_splitAllocationFailed = false;
  };
}
