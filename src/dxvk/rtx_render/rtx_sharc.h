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
    bool m_resetRequested = true;
    bool m_allocationFailed = false;
    const char* m_status = "Inactive";
    uint32_t m_compatibilityFlags = 0;
    uint32_t m_lastFrame = ~0u;
  };
}
