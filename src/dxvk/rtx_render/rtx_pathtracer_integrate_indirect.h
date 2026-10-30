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
#include "rtx/pass/raytrace_args.h"

namespace dxvk {

  class DxvkDevice;
  struct RtFrameFeatures;
  struct VariantTableSelfCheckResult;
  enum class IntegrateIndirectMode;

  class DxvkPathtracerIntegrateIndirect : public CommonDeviceObject {
  public:
    enum class RaytraceMode {
      RayQuery = 0,
      RayQueryRayGen,
      TraceRay,
      Count
    };

    // Which compile-time slice of the indirect shaders a frame can use (SHARC update / query,
    // assembly). The assembly has no RIS fallback, so its Slim is INDIRECT_SLIM_NO_POM alone.
    enum class SlimLevel {
      Full = 0, // everything compiled in
      Slim,     // INDIRECT_SLIM_NO_POM + INDIRECT_SLIM_NEE_CACHE_ONLY: POM off, secondary NEE always takes the NEE cache
      Lean,     // + INDIRECT_SLIM_LEAN_VISIBILITY: no translucent / thin-opaque hit possible on the secondary NEE ray
      SlimPom,  // Slim with POM kept (rtx.integrateIndirect.slimKeepPom), for frames that use POM
      LeanPom,  // Lean with POM kept
    };
    static const char* slimLevelToString(SlimLevel level);
    static bool slimLevelIsLean(SlimLevel level) { return level == SlimLevel::Lean || level == SlimLevel::LeanPom; }
    static bool slimLevelKeepsPom(SlimLevel level) { return level == SlimLevel::Full || level == SlimLevel::SlimPom || level == SlimLevel::LeanPom; }

    RTX_OPTION("rtx.integrateIndirect", bool, slimVariants, true,
               "Uses reduced SHARC update and query shaders that leave out features the current frame cannot use: "
               "displacement, the RIS and light BVH fallback of secondary NEE (when the NEE cache handles analytic "
               "lights), and translucent and thin-opaque handling of secondary shadow rays (when "
               "rtx.enableIndirectTranslucentShadows is off and nothing needs it). Saves GPU time. The image is the "
               "same. Only with WBOIT, no portals and the 2-state micromap mode off. Other configurations use the full "
               "shaders.");
    RTX_OPTION("rtx.integrateIndirect", bool, slimAssembly, true,
               "Uses reduced indirect assembly (NEE) shaders that leave out displacement in the NEE cache shadow ray "
               "(when displacement or rtx.displacement.enableNEECache is off) and translucent and thin-opaque handling "
               "of that ray (under the same conditions as rtx.integrateIndirect.slimVariants). Works in every indirect "
               "mode. Saves GPU time. The image is the same.");
    RTX_OPTION("rtx.integrateIndirect", bool, slimNonSharc, true,
               "Applies the reduced shaders of rtx.integrateIndirect.slimVariants to the TraceRay indirect pass without "
               "SHARC (Importance Sampled, NRC and ReSTIR GI): no displacement while it is off, no RIS or light BVH "
               "fallback when the NEE cache takes every sample, and, at the leanest level, no translucent or "
               "thin-opaque handling on secondary shadow rays. Only with the NEE cache, WBOIT, no portals, the 2-state "
               "micromap mode off and TraceRay mode. The image is the same.");
    RTX_OPTION("rtx.integrateIndirect", bool, slimKeepPom, false,
               "Lets rtx.integrateIndirect.slimVariants, slimNonSharc and slimAssembly use reduced shaders with "
               "displacement compiled in while the frame uses displacement. Otherwise displacement forces the full "
               "shaders. The image is the same.");

    explicit DxvkPathtracerIntegrateIndirect(DxvkDevice* device);
    ~DxvkPathtracerIntegrateIndirect() = default;

    void prewarmShaders(DxvkPipelineManager& pipelineManager) const;

    // Requires current GBuffer, sparse mask, direct-generated rays and NEE cache.
    // Produces primary indirect lobes and raw radiance retained for secondary demodulation.
    void dispatchLighting(class RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    static const char* raytraceModeToString(RaytraceMode raytraceMode);

    // Exhaustive check of the SHARC, non-SHARC and assembly variant tables against the selectors
    // (rtx_fork_variant_tables.h).
    static void runVariantTableSelfCheck(VariantTableSelfCheckResult& sharc, VariantTableSelfCheckResult& nonSharc,
                                         VariantTableSelfCheckResult& assembly);

  private:
    struct AssemblyResources;
    void dispatch(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);
    void dispatchNEE(RtxContext* ctx, const AssemblyResources& resources, const SlimLevel slimLevel);

    static DxvkRaytracingPipelineShaders getPipelineShaders(const bool useRayQuery, const bool serEnabled, const bool ommEnabled, const bool useNeeCache, const bool includePortals, const bool pomEnabled, const bool nrcEnabled, const bool wboitEnbaled, const SlimLevel slimLevel = SlimLevel::Full);
    // rtx.integrateIndirect.slimNonSharc. Full unless the TraceRay slim variants exist for this
    // configuration and the frame cannot use what they compile out.
    static SlimLevel selectNonSharcSlimLevel(RtxContext* ctx, const RaytraceArgs& args, const bool useRayQuery, const bool ommEnabled,
                                             const bool useNeeCache, const bool includePortals, const bool wboitEnabled);
    Rc<DxvkShader> getComputeShader(const bool useNeeCache, const bool nrcEnabled, const bool wboitEnabled) const;
    static DxvkRaytracingPipelineShaders getSharcQueryPipelineShaders(const bool serEnabled, const bool ommEnabled, const bool includePortals, const bool pomEnabled, const bool wboitEnabled, const SlimLevel slimLevel = SlimLevel::Full);
    static DxvkRaytracingPipelineShaders getSharcUpdatePipelineShaders(const bool depth4, const bool ommEnabled, const bool wboitEnabled, const SlimLevel slimLevel, const bool deferredFlush);
    // True when rtx.sharc.deferredFlush is on and a _df update variant exists for this configuration.
    static bool sharcDeferredFlushPossible(const bool ommEnabled, const bool wboitEnabled);
    // True when the TraceRay + SER update variants exist for this configuration (SER,
    // deferred flush, no portals); the caller also checks the level (Full / Slim / Lean) and POM.
    static bool sharcUpdateSerPossible(const bool serEnabled, const bool ommEnabled, const bool wboitEnabled, const bool includePortals);
    static DxvkRaytracingPipelineShaders getSharcUpdateSerPipelineShaders(const bool depth4, const bool ommEnabled, const SlimLevel slimLevel);
    void prewarmSharcShaders(DxvkPipelineManager& pipelineManager) const;
    static SlimLevel selectSharcSlimLevel(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput, const bool ommEnabled, const bool includePortals, const bool wboitEnabled);
    // The selectors on a given mask (the RtxContext overloads pass this frame's), for the table check.
    static SlimLevel selectSharcSlimLevel(const RtFrameFeatures& features, const bool ommEnabled, const bool includePortals, const bool wboitEnabled);
    static SlimLevel selectNonSharcSlimLevel(const RtFrameFeatures& features, const bool useRayQuery, const bool ommEnabled,
                                             const bool useNeeCache, const bool includePortals, const bool wboitEnabled);
    static SlimLevel selectAssemblySlimLevel(const RtFrameFeatures& features);
    // The mode-independent part of the slim decision, read from this frame's feature mask (built from its
    // RaytraceArgs). pomCompiledOutOk: the caller's POM code cannot run this frame.
    // requireNeeCacheOnly: the caller's variants also compile out the RIS / light BVH fallback of secondary NEE
    // (INDIRECT_SLIM_NEE_CACHE_ONLY).
    static SlimLevel selectIndirectSlimLevel(const RtFrameFeatures& features, const bool pomCompiledOutOk, const bool requireNeeCacheOnly);
    static SlimLevel selectAssemblySlimLevel(RtxContext* ctx, const RaytraceArgs& args);
    static Rc<DxvkShader> getAssemblyShader(const bool nrcEnabled, const bool restirGiEnabled, const SlimLevel slimLevel);
    void dispatchSharc(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput, const VkExtent3D& rayDims, const bool serEnabled, const bool ommEnabled, const bool includePortals, const bool pomEnabled, const bool wboitEnabled);
    void logIntegrateIndirectMode();

    // Which slim level each indirect pass bound, logged as [IntegrateIndirect] when it changes.
    enum class SlimPass { SharcUpdate = 0, Integration, Assembly, Count };
    void noteSlimLevel(const SlimPass pass, const SlimLevel level) { m_frameSlimLevels[(uint32_t) pass] = (int32_t) level; }
    void logSlimLevels(const uint32_t frameId);

    IntegrateIndirectMode m_integrateIndirectMode;
    std::array<int32_t, (uint32_t) SlimPass::Count> m_frameSlimLevels = { -1, -1, -1 };
    std::array<int32_t, (uint32_t) SlimPass::Count> m_loggedSlimLevels = { -1, -1, -1 };
    uint32_t m_lastSlimLogFrame = UINT32_MAX;
    // rtx.shader.variantTables entry text per pass (empty while off), on the [IntegrateIndirect] line.
    std::array<std::string, (uint32_t) SlimPass::Count> m_frameVariantInfo;
    std::array<std::string, (uint32_t) SlimPass::Count> m_loggedVariantInfo;
    // The SHARC / non-SHARC / assembly selection through the tables (checked against `legacyLevel`).
    SlimLevel selectSharcVariantLevel(RtxContext* ctx, const SlimPass pass, const SlimLevel legacyLevel);
    SlimLevel selectNonSharcVariantLevel(RtxContext* ctx, const bool useNeeCache, const SlimLevel legacyLevel);
    SlimLevel selectAssemblyVariantLevel(RtxContext* ctx, const SlimLevel legacyLevel);
  };
}
