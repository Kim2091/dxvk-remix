// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#pragma once
#include <array>
#include "dxvk_gpu_query.h"
#include "dxvk_gpu_event.h"
#include "rtx_option.h"
#include "rtx_resources.h"

namespace dxvk {
  class RtxContext;

  class RtxSharc : public CommonDeviceObject {
  public:
    explicit RtxSharc(DxvkDevice* device);
    static bool isSupported(const DxvkDevice& device);
    void prepareFrame(RtxContext& ctx, RaytraceArgs& args, bool resetHistory);
    void bindResources(RtxContext& ctx) const;
    void dispatchResolve(RtxContext& ctx, const Resources::RaytracingOutput& output);
    enum class TimingPoint { Begin, UpdateEnd, ResolveEnd, QueryEnd };
    void recordTimestamp(RtxContext& ctx, TimingPoint point);
    void showImguiSettings();
    void beginQueryStats(RtxContext& ctx);
    void endQueryStats(RtxContext& ctx);
    bool queryStatsActive() const { return m_statsSlot >= 0; }
    bool isActive() const { return m_active; }
    bool isLeanActive() const { return m_active && m_leanActive; }

    // Effective lean shader selection for the current frame, resolved once per frame from the
    // options, the renderer feature flags and the scene contents. Level 0 is the original lean profile.
    struct LeanProfile {
      uint32_t level = 0;        // Query tier: 0 lean, 1 adds alpha-blended indirect shadows, 2 adds the unordered resolve.
      uint32_t updateLevel = 0;  // Tier used by the sparse update; 0 unless ray-generation deferred updates are selected.
      bool wboit = false;        // Level 2 stages use the WBOIT unordered resolver.
      bool pom = false;          // Displacement-aware closest-hit stage.
      bool operator==(const LeanProfile& o) const { return level == o.level && updateLevel == o.updateLevel && wboit == o.wboit && pom == o.pom; }
      bool operator!=(const LeanProfile& o) const { return !(*this == o); }
    };
    void resolveLeanProfile(RtxContext& ctx, const RaytraceArgs& args);
    const LeanProfile& leanProfile() const { return m_leanProfile; }

    RTX_OPTION("rtx.sharc", bool, fuseAssembly, false, "Experimental combined TraceRay query and primary NEE assembly. Enable before launch to allocate independent throughput storage. Unsupported modes use standalone assembly.");

    RTX_OPTION("rtx.sharc", bool, leanSecondary, false, "Use experimental full-resolution lean secondary shaders: no unordered particles/decals, POM, alpha-blended indirect shadows, or RTXDI sample stealing. Uses TraceRay queries; portal scenes retain the full profile.");
    RTX_OPTION("rtx.sharc", int, leanFeatureLevel, 0, "Restore full-profile features to the lean secondary profile: 0 keeps the lean shaders, 1 adds alpha-blended indirect shadows, 2 also adds the unordered particle/decal resolve (using the WBOIT resolver when WBOIT is enabled). A level is only used in frames whose scene contains the geometry it handles. Levels above 0 need ray-generation deferred updates; other update backends keep level 0 for updates. Changing the level resets the cache.");
    RTX_OPTION("rtx.sharc", bool, leanIndirectPom, false, "Use displacement-aware lean closest-hit shaders when rtx.displacement.enableIndirectHit is enabled and the scene contains displaced materials. Otherwise the lean profile ignores displacement in secondary rays.");

    RTX_OPTION("rtx.sharc", bool, allowWboit, false, "Allow SHARC while WBOIT is enabled for compatibility testing. Does not enable WBOIT.");
    RTX_OPTION("rtx.sharc", bool, allowRayPortals, false, "Allow SHARC with ray portals for compatibility testing. Does not enable ray portals.");
    RTX_OPTION("rtx.sharc", bool, allowOpacityMicromap, false, "Allow SHARC while opacity micromaps are enabled for compatibility testing. Does not enable opacity micromaps.");
    RTX_OPTION("rtx.sharc", bool, deferredUpdates, true, "Accumulate lighting locally and flush each cache vertex once per update path. Disable to compare with the original update shader.");
    RTX_OPTION("rtx.sharc", bool, queryTraceRay, true, "Run SHARC queries with separate TraceRay hit/miss shaders and the indirect pass SER setting when supported. Disable to compare the inline RayQuery backends with the same cache policy.");
    RTX_OPTION("rtx.sharc", bool, queryRayGeneration, true, "Run SHARC queries as inline RayQuery in a ray-generation shader. Disable to compare the compute backend; lighting and cache eligibility are unchanged.");
    RTX_OPTION("rtx.sharc", bool, updateRayGeneration, true, "Run sparse SHARC updates as inline RayQuery in a ray-generation shader. Disable to compare compute updates with the same sampling and estimator.");
    RTX_OPTION("rtx.sharc", bool, allowSpecularPaths, false, "Allow cache insertion and reuse at rough opaque surfaces reached by non-diffuse rays. Can soften reflected lighting; changing this resets the cache.");
    RTX_OPTION("rtx.sharc", bool, collectQueryStats, false, "Collect sampled cache reuse and rejection counts while GPU timing is enabled. Disable for timing comparisons without diagnostic atomics.");
    RTX_OPTION("rtx.sharc", bool, measureGpuTime, false, "Measure SHARC update, resolve and query GPU times. Timing boundaries can affect overlap; disable for final frame-time comparisons.");
    RTX_OPTION("rtx.sharc", int, capacityLog2, 21, "Cache capacity exponent, 18..22. 21 uses 80 MiB; 22 uses 160 MiB.");
    RTX_OPTION("rtx.sharc", int, updateTileSize, 5, "One cache update path per NxN tile, 1..16.");
    RTX_OPTION("rtx.sharc", int, updateBounces, 8, "Maximum finite cache update bounces, 1..8.");
    RTX_OPTION("rtx.sharc", int, accumulationFrames, 8, "Temporal cache accumulation, 1..64 frames.");
    RTX_OPTION("rtx.sharc", int, staleFrames, 32, "Evict unobserved entries after 8..128 frames.");
    RTX_OPTION("rtx.sharc", float, gridScale, 50.0f, "SHARC grid density; independent of rtx.sceneScale. Larger values give finer cells.");
    RTX_OPTION("rtx.sharc", float, minRoughness, 0.8f, "Minimum isotropic roughness for cached diffuse surfaces, 0.5..1.");

  private:
    friend class ImGUI;
    struct FrameTiming {
      std::array<Rc<DxvkGpuQuery>, 4> queries;
      bool pending = false;
    };
    std::array<FrameTiming, 8> m_frameTimings;
    std::array<float, 3> m_gpuTimes = {};
    int m_timingSlot = -1;
    bool m_haveGpuTimes = false;
    static constexpr uint32_t kStatsStride = 256;
    std::array<Rc<DxvkGpuEvent>, 8> m_statsReady;
    Rc<DxvkBuffer> m_statsGpu;
    Rc<DxvkBuffer> m_statsReadback;
    std::array<uint32_t, 14> m_queryStats = {};
    int m_statsSlot = -1;
    bool m_haveQueryStats = false;
    uint32_t m_cacheAge = 0;
    Rc<DxvkBuffer> m_hash;
    Rc<DxvkBuffer> m_accumulation;
    Rc<DxvkBuffer> m_resolved;
    SharcArgs m_args = {};
    bool m_active = false;
    bool m_leanActive = false;
    LeanProfile m_leanProfile;
    uint64_t m_loggedLeanProfiles = 0;
    bool m_resetRequested = true;
    bool m_allocationFailed = false;
    const char* m_status = "Inactive";
    uint32_t m_compatibilityFlags = 0;
    uint32_t m_lastFrame = ~0u;
  };
}
