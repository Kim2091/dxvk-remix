/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "../util/rc/util_rc_ptr.h"

namespace dxvk {

  class DxvkDevice;
  class RtxContext;
  struct RtFrameFeatures;
  struct VariantTableSelfCheckResult;

  class DxvkPathtracerGbuffer : public CommonDeviceObject {
  public:
    enum class RaytraceMode {
      RayQuery = 0,
      RayQueryRayGen,
      TraceRay,
      Count
    };

    // Which compile-time slice of the RayQuery G-buffer kernels a frame can use
    // (rtx.gbuffer.slimVariants, see gbuffer_rayquery_slim.comp.slang).
    enum class SlimLevel {
      Full = 0, // everything compiled in
      Slim,     // no POM, no thin film, no ray portal resolve
      Lean,     // Slim + no subsurface material loads
      SlimPom,  // Slim with POM kept
      LeanPom,  // Lean with POM kept
    };
    static const char* slimLevelToString(SlimLevel level);

    // Why selectSlimLevel returned Full, printed on the [GBuffer] slim level log line.
    enum class SlimFullReason {
      None = 0,                    // a slim level was selected
      OptionOff,                   // rtx.gbuffer.slimVariants off, or not RayQuery mode
      NoWboit,                     // WBOIT off
      FeatureSet,                  // not psr_prepare (with or without stochastic alpha) without NRC / psr_inline with NRC
                                   // (inline PSR with stochastic alpha, RT render targets, debug, PSR off)
      ThinFilmOverride,            // rtx.opaqueMaterial.enableThinFilmOverride
      ThinFilmMaterialCreated,     // a material with the thin-film flag was created (sticky)
      PortalsConfigured,           // ray portal texture hashes configured or portals active this frame
      PortalMaterialCreated,       // a ray portal material was created (sticky)
      IntersectionBillboardsOnPrimary, // billboard orientation correction + intersection billboards on primary rays
      BillboardDynamicMask,        // a billboard primitive with dynamic (player model) mask bits
      ThinFilmMaterialLive,        // a live instance is bound to a thin-film material (rtx.frameFeatures.liveMaterialCounts)
      PortalMaterialLive,          // a live instance is bound to a ray portal material (rtx.frameFeatures.liveMaterialCounts)
      FrameFeaturesValidate,       // rtx.frameFeatures.validate forces the Full kernels
    };
    static const char* slimFullReasonToString(SlimFullReason reason);

    RTX_OPTION("rtx.gbuffer", bool, slimVariants, true,
               "Uses reduced G-buffer shaders (primary rays, PSR, PSR prepare and deferred decals) that leave out "
               "features the current frame cannot use: displacement, thin-film materials, ray portals and subsurface "
               "materials. Saves GPU time. The image is the same. Only in ray query mode with WBOIT, and with PSR "
               "prepare or with inline PSR and NRC. Other configurations use the full shaders.");
    RTX_OPTION("rtx.gbuffer", bool, compactPsr, true,
               "The reflection and transmission PSR passes run only on the pixels that need them, from lists that PSR "
               "prepare builds, instead of on every pixel. Saves GPU time. The image is the same. Only in ray query "
               "mode with PSR prepare, and not with NRC, ray traced render targets or debug views. Uses two lists of 4 "
               "bytes per pixel.");

    explicit DxvkPathtracerGbuffer(DxvkDevice* device);
    ~DxvkPathtracerGbuffer() = default;

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;
    void dispatch(class RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    static const char* raytraceModeToString(RaytraceMode raytraceMode);

    // The gbuffer_rayquery features axis index dispatch() binds for this frame's args. The frame feature mask
    // records it.
    static uint32_t frameRayQueryFeatureVariant(const RaytraceArgs& args, const bool nrcEnabled);

    // Exhaustive check of the variant table against selectSlimLevel (rtx_fork_variant_tables.h).
    static void runVariantTableSelfCheck(VariantTableSelfCheckResult& result);

  private:
    static DxvkRaytracingPipelineShaders getPipelineShaders(const uint32_t featureVariant, const bool isPSRPass, const bool useRayQuery, const bool serEnabled, const bool ommEnabled, const bool includePortals, const bool nrcEnabled, const bool wboitEnabled);
    Rc<DxvkShader> getComputeShader(
      const uint32_t featureVariant,
      const bool isPSRPass,
      const bool nrcEnabled,
      const bool wboitEnabled,
      const SlimLevel slimLevel = SlimLevel::Full) const;

    // rtx.gbuffer.slimVariants. Reads this frame's feature mask (built from its RaytraceArgs and the scene manager)
    // and the pass's own feature set.
    static SlimLevel selectSlimLevel(const RtFrameFeatures& features, const uint32_t rayQueryFeatureVariant,
                                     const bool nrcEnabled, const bool wboitEnabled, const bool includePortals, SlimFullReason& fullReason);
    void logSlimLevel(const SlimLevel level, const SlimFullReason fullReason, const uint32_t rayQueryFeatureVariant, const bool nrcEnabled,
                      const bool compactPsrActive, const uint32_t frameId,
                      const std::string& variantInfo);
    int32_t m_loggedSlimLevel = -1;
    std::string m_loggedVariantInfo; // rtx.shader.variantTables entry text, empty while off
    int32_t m_loggedSlimFullReason = -1;
    int32_t m_loggedCompactPsr = -1;
    uint32_t m_lastSlimLogFrame = UINT32_MAX;

    // rtx.gbuffer.compactPsr. Two pixel lists (m_compactPsrCapacity entries each) and the indirect
    // dispatch args / counts (GBUFFER_COMPACT_PSR_ARGS_*). Grown to the frame's pixel count, released while the option is off.
    bool prepareCompactPsrBuffers(const uint32_t pixelCount);
    Rc<DxvkBuffer> m_compactPsrList;
    Rc<DxvkBuffer> m_compactPsrArgs;
    uint32_t m_compactPsrCapacity = 0;
    uint32_t m_compactPsrFailedPixelCount = 0;
    bool m_loggedDeferParticles = false; // The last logged state of the deferred unordered particle resolve.
  };
}
