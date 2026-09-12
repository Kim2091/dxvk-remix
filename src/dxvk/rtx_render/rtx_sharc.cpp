// Copyright (c) 2026, NVIDIA CORPORATION. SPDX-License-Identifier: MIT
#include <algorithm>
#include <cmath>
#include "rtx_sharc.h"
#include "dxvk_device.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_shader_manager.h"
#include "rtx_imgui.h"
#include "dxvk_scoped_annotation.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/sharc/sharc_binding_indices.h"
#include <rtx_shaders/sharc_resolve.h>

namespace dxvk {
  namespace {
    class SharcResolveShader : public ManagedShader {
      SHADER_SOURCE(SharcResolveShader, VK_SHADER_STAGE_COMPUTE_BIT, sharc_resolve)
      BEGIN_PARAMETER()
        CONSTANT_BUFFER(BINDING_CONSTANTS)
        { SHARC_BINDING_HASH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
        { SHARC_BINDING_ACCUMULATION, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
        { SHARC_BINDING_RESOLVED, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT },
      END_PARAMETER()
    };
  }

  RtxSharc::RtxSharc(DxvkDevice* device) : CommonDeviceObject(device) { }

  bool RtxSharc::isSupported(const DxvkDevice& device) {
    const auto& f = device.features();
    return f.core.features.shaderInt64 && f.vulkan12Features.shaderBufferInt64Atomics
      && f.vulkan12Features.shaderFloat16 && f.vulkan11Features.storageBuffer16BitAccess
      && f.vulkan11Features.uniformAndStorageBuffer16BitAccess && f.khrRayQueryFeatures.rayQuery;
  }

  void RtxSharc::prepareFrame(RtxContext& ctx, RaytraceArgs& args, bool resetHistory) {
    args.sharcArgs = {};
    const bool selected = RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::Sharc;
    const bool wasActive = m_active;
    m_active = false;
    if (!selected) {
      m_hash = nullptr;
      m_accumulation = nullptr;
      m_resolved = nullptr;
      m_allocationFailed = false;
      m_status = "Inactive";
      return;
    }
    if (!isSupported(*m_device)) {
      m_status = "Unsupported device features; using importance-sampled paths";
      return;
    }
    // The render-target camera changes the indirect path semantics and its
    // secondary rays are not yet represented by the SHARC estimator.  Keep a
    // conservative fallback until the cache can tag those paths explicitly.
    if (args.enableRaytracedRenderTarget) {
      m_status = "Raytraced render target active; using importance-sampled paths";
      return;
    }
    // These combinations need independent estimator validation before caching them.
    if (RtxOptions::wboitEnabled() || args.numActiveRayPortals > 0
        || !RtxOptions::rayPortalModelTextureHashes().empty()
        || RtxOptions::getEnableOpacityMicromap()) {
      m_status = "Disable WBOIT, ray portals and OMM to use SHARC; tracing normally";
      return;
    }
    if (m_allocationFailed && !m_resetRequested) {
      return;
    }

    const uint32_t frame = m_device->getCurrentFrameId();
    const uint32_t capacity = 1u << std::clamp(capacityLog2(), 18, 22);
    const float scale = std::isfinite(gridScale()) ? std::clamp(gridScale(), 1.0f, 1000.0f) : 50.0f;
    const float roughness = std::isfinite(minRoughness()) ? std::clamp(minRoughness(), 0.5f, 1.0f) : 0.8f;
    const uint32_t updateBounceLimit = std::clamp(updateBounces(), 1, 8);
    bool clear = resetHistory || m_resetRequested || !wasActive || m_lastFrame + 1 != frame
      || m_args.gridScale != scale || m_args.minRoughness != roughness
      || m_args.updateBounces != updateBounceLimit;
    if (m_hash == nullptr || m_args.capacity != capacity) {
      try {
        DxvkBufferCreateInfo info = {};
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        info.access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        info.size = VkDeviceSize(capacity) * 8;
        auto hash = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC hash");
        info.size = VkDeviceSize(capacity) * 16;
        auto accumulation = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC accumulation");
        auto resolved = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "SHARC resolved");
        m_hash = hash;
        m_accumulation = accumulation;
        m_resolved = resolved;
        clear = true;
      } catch (const DxvkError& e) {
        Logger::err(str::format("SHARC allocation failed: ", e.message()));
        m_allocationFailed = true;
        m_resetRequested = false;
        m_status = "Cache allocation failed; tracing normally. Reset to retry.";
        return;
      }
    }
    if (clear) {
      ctx.clearBuffer(m_hash, 0, m_hash->info().size, 0);
      ctx.clearBuffer(m_accumulation, 0, m_accumulation->info().size, 0);
      ctx.clearBuffer(m_resolved, 0, m_resolved->info().size, 0);
    }
    m_args.cameraPositionPrev = m_args.cameraPosition;
    const auto& camera = ctx.getSceneManager().getCamera();
    const auto position = camera.getPosition();
    m_args.cameraPosition = vec3(position.x, position.y, position.z);
    if (clear) {
      m_args.cameraPositionPrev = m_args.cameraPosition;
    }
    m_args.capacity = capacity;
    m_args.gridScale = scale;
    m_args.minRoughness = roughness;
    m_args.accumulationFrames = std::clamp(accumulationFrames(), 1, 64);
    m_args.staleFrames = std::clamp(staleFrames(), 8, 128);
    m_args.updateTileSize = std::clamp(updateTileSize(), 1, 16);
    m_args.updateBounces = updateBounceLimit;
    m_args.radianceScale = 1000.0f;
    m_args.enabled = 1;
    args.sharcArgs = m_args;
    m_lastFrame = frame;
    m_resetRequested = false;
    m_allocationFailed = false;
    m_active = true;
    m_status = "Experimental diffuse cache; compute RayQuery backend, finite update paths";
  }

  void RtxSharc::bindResources(RtxContext& ctx) const {
    ctx.bindResourceBuffer(SHARC_BINDING_HASH, DxvkBufferSlice(m_hash, 0, m_hash->info().size));
    ctx.bindResourceBuffer(SHARC_BINDING_ACCUMULATION, DxvkBufferSlice(m_accumulation, 0, m_accumulation->info().size));
    ctx.bindResourceBuffer(SHARC_BINDING_RESOLVED, DxvkBufferSlice(m_resolved, 0, m_resolved->info().size));
  }

  void RtxSharc::dispatchResolve(RtxContext& ctx, const Resources::RaytracingOutput& output) {
    ScopedGpuProfileZone(&ctx, "SHARC Resolve");
    ctx.bindCommonRayTracingResources(output);
    bindResources(ctx);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, SharcResolveShader::getShader());
    ctx.dispatch((m_args.capacity + 255) / 256, 1, 1);
  }

  void RtxSharc::showImguiSettings() {
    ImGui::TextWrapped("%s", m_status);
    RemixGui::DragInt("Capacity exponent", &capacityLog2Object(), 1.0f, 18, 22);
    RemixGui::DragInt("Update tile size", &updateTileSizeObject(), 1.0f, 1, 16);
    RemixGui::DragInt("Update bounces", &updateBouncesObject(), 1.0f, 1, 8);
    RemixGui::DragInt("Accumulation frames", &accumulationFramesObject(), 1.0f, 1, 64);
    RemixGui::DragInt("Stale frames", &staleFramesObject(), 1.0f, 8, 128);
    RemixGui::DragFloat("Grid density", &gridScaleObject(), 1.0f, 1.0f, 1000.0f);
    RemixGui::DragFloat("Minimum roughness", &minRoughnessObject(), 0.01f, 0.5f, 1.0f);
    if (ImGui::Button("Reset SHARC")) {
      m_resetRequested = true;
    }
  }
}
