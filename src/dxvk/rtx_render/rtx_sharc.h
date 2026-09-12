// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#pragma once
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
    void showImguiSettings();
    bool isActive() const { return m_active; }

    RTX_OPTION("rtx.sharc", int, capacityLog2, 21, "Cache capacity exponent, 18..22. 21 uses 80 MiB; 22 uses 160 MiB.");
    RTX_OPTION("rtx.sharc", int, updateTileSize, 5, "One cache update path per NxN tile, 1..16.");
    RTX_OPTION("rtx.sharc", int, updateBounces, 8, "Maximum finite cache update bounces, 1..8.");
    RTX_OPTION("rtx.sharc", int, accumulationFrames, 8, "Temporal cache accumulation, 1..64 frames.");
    RTX_OPTION("rtx.sharc", int, staleFrames, 32, "Evict unobserved entries after 8..128 frames.");
    RTX_OPTION("rtx.sharc", float, gridScale, 50.0f, "SHARC grid density; independent of rtx.sceneScale. Larger values give finer cells.");
    RTX_OPTION("rtx.sharc", float, minRoughness, 0.8f, "Minimum isotropic roughness for cached diffuse surfaces, 0.5..1.");

  private:
    friend class ImGUI;
    Rc<DxvkBuffer> m_hash;
    Rc<DxvkBuffer> m_accumulation;
    Rc<DxvkBuffer> m_resolved;
    SharcArgs m_args = {};
    bool m_active = false;
    bool m_resetRequested = true;
    bool m_allocationFailed = false;
    const char* m_status = "Inactive";
    uint32_t m_lastFrame = ~0u;
  };
}
