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
    void logFallbackStatsIfDue();
    void beginQueryStats(RtxContext& ctx);
    void endQueryStats(RtxContext& ctx);
    bool queryStatsActive() const { return m_statsSlot >= 0; }
    bool isActive() const { return m_active; }

    RTX_OPTION("rtx.sharc", bool, allowRayPortals, false, "Allow SHARC while a ray portal is active. Vertices reached through a portal are never inserted into the cache, so portal-only geometry stays uncached; queries are unrestricted. Untested against a real portal scene.");
    RTX_OPTION("rtx.sharc", bool, deferredUpdates, true, "Accumulate lighting locally and flush each cache vertex once per update path. Disable to compare with the original update shader.");
    RTX_OPTION("rtx.sharc", bool, queryTraceRay, true, "Run SHARC queries with separate TraceRay hit/miss shaders and the indirect pass SER setting when supported. Disable to compare the inline RayQuery backends with the same cache policy.");
    RTX_OPTION("rtx.sharc", bool, queryRayGeneration, true, "Run SHARC queries as inline RayQuery in a ray-generation shader. Disable to compare the compute backend; lighting and cache eligibility are unchanged.");
    RTX_OPTION("rtx.sharc", bool, updateRayGeneration, true, "Run sparse SHARC updates as inline RayQuery in a ray-generation shader. Disable to compare compute updates with the same sampling and estimator.");
    RTX_OPTION("rtx.sharc", bool, allowSpecularPaths, false, "Allow cache insertion and reuse at rough opaque surfaces reached by non-diffuse rays. Can soften reflected lighting; changing this resets the cache.");
    RTX_OPTION("rtx.sharc", bool, footprintGate, true, "Gate cache reads on specular paths by the footprint of the lobe that launched the segment, NVIDIA's prescribed test, instead of by the roughness of the surface the path hit: footprint = segment length * sqrt(0.5 * alpha^2 / (1 - alpha^2)) must exceed the voxel size, alpha being the launching surface's GGX roughness. While on, rtx.sharc.minRoughnessSpecular is not used and specular arrivals share rtx.sharc.minRoughness with diffuse ones. Only matters with allowSpecularPaths on; changing it resets the cache.");
    RTX_OPTION("rtx.sharc", bool, collectQueryStats, false, "Collect sampled cache reuse and rejection counts while GPU timing is enabled. Disable for timing comparisons without diagnostic atomics.");
    RTX_OPTION("rtx.sharc", bool, logFallbackStats, false, "Periodically log how often SHARC was selected but fell back to importance-sampled paths, split by reason. Diagnostic only; counts cost nothing when this is disabled.");
    RTX_OPTION("rtx.sharc", bool, measureGpuTime, false, "Measure SHARC update, resolve and query GPU times. Timing boundaries can affect overlap; disable for final frame-time comparisons.");
    RTX_OPTION("rtx.sharc", int, capacityLog2, 21, "Cache capacity exponent, 18..22. 21 uses 80 MiB; 22 uses 160 MiB.");
    RTX_OPTION("rtx.sharc", int, updateTileSize, 5, "One cache update path per NxN tile, 1..16.");
    RTX_OPTION("rtx.sharc", int, updateBounces, 8, "Maximum finite cache update bounces, 1..8.");
    RTX_OPTION("rtx.sharc", int, accumulationFrames, 8, "Temporal cache accumulation, 1..64 frames.");
    RTX_OPTION("rtx.sharc", int, staleFrames, 32, "Evict unobserved entries after 8..128 frames.");
    RTX_OPTION("rtx.sharc", float, gridScale, 50.0f, "SHARC grid density; independent of rtx.sceneScale. Larger values give finer cells.");
    RTX_OPTION("rtx.sharc", float, maxEmissiveLuminance, 0.0f, "Cache surfaces whose emissive luminance is at or below this. The cache holds reflected light and the path adds emission separately, so emissive surfaces are normally excluded; at 0 any emission at all disqualifies a surface, which rejects every faint emissive map. Raise until emissive surfaces start bleeding their own light into the cache.");
    RTX_OPTION("rtx.sharc", int, minSampleCount, 2, "Ignore cache cells until they hold more than this many accumulated samples, 0..32. At 0 a single sample answers a query, so geometry coming into view for the first time can read one bright path as converged radiance and glow. Raising it trades coverage in freshly revealed areas for stability there.");
    RTX_OPTION("rtx.sharc", float, minRoughnessSpecular, 0.5f, "Minimum roughness for caching a vertex a specular lobe arrived at, used only when Reuse rough surfaces in specular paths is on. Lower values make smooth reflective materials glow, because an isotropic cache cannot stand in for a directional reflection; keep this well above rtx.sharc.minRoughness.");
    RTX_OPTION("rtx.sharc", float, minRoughness, 0.8f, "Minimum isotropic roughness for cached diffuse surfaces, 0.05..1. Below roughly 0.5 the isotropic cache starts standing in for directional reflection, so lower it until reflections visibly flatten, then back off.");

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
    // Counter slots are documented at the sharcCount call sites in sharc_integrator_hooks.slangh
    // and integrator_indirect.slangh: 0..13 the path and eligibility aggregates, 14..18 the
    // surface term behind slot 9, 19..21 the too-close splits, 22 the footprint gate.
    static constexpr uint32_t kStatsCount = 25;
    static_assert(kStatsCount * sizeof(uint32_t) <= kStatsStride);
    std::array<Rc<DxvkGpuEvent>, 8> m_statsReady;
    Rc<DxvkBuffer> m_statsGpu;
    Rc<DxvkBuffer> m_statsReadback;
    std::array<uint32_t, kStatsCount> m_queryStats = {};
    // Per-frame counters are cleared every frame, so displaying them directly flickers
    // unreadably and any frame without a lookup reads as 0%. Accumulate over a window and
    // publish ratios of sums, which is both stable and the statistically correct ratio.
    static constexpr uint32_t kStatsWindowFrames = 120;
    std::array<uint64_t, kStatsCount> m_queryStatsAccum = {};
    std::array<uint64_t, kStatsCount> m_queryStatsWindow = {};
    uint32_t m_queryStatsAccumFrames = 0;
    bool m_haveQueryStatsWindow = false;
    int m_statsSlot = -1;
    bool m_haveQueryStats = false;
    uint32_t m_cacheAge = 0;
    Rc<DxvkBuffer> m_hash;
    Rc<DxvkBuffer> m_accumulation;
    Rc<DxvkBuffer> m_resolved;
    SharcArgs m_args = {};
    bool m_active = false;
    bool m_resetRequested = true;
    bool m_allocationFailed = false;
    const char* m_status = "Inactive";
    uint32_t m_compatibilityFlags = 0;
    uint32_t m_lastFrame = ~0u;
    // Fallback accounting.  SHARC can be selected yet inactive for a whole frame; these
    // counters say how often that happens and why, so the cost of each fallback reason
    // can be judged from a real play session instead of guessed at.
    static constexpr uint32_t kFallbackLogInterval = 1800;
    uint32_t m_selectedFrames = 0;
    uint32_t m_rttFallbackFrames = 0;
    uint32_t m_otherFallbackFrames = 0;
  };
}
