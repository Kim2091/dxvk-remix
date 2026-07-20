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
#include "rtx_point_instancer_system.h"

#include "dxvk_device.h"
#include "rtx_render/rtx_shader_manager.h"
#include "dxvk_scoped_annotation.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_imgui.h"

#include <rtx_shaders/point_instancer_culling.h>

namespace dxvk {

  namespace {
    class PointInstancerCullingShader : public ManagedShader {
      SHADER_SOURCE(PointInstancerCullingShader, VK_SHADER_STAGE_COMPUTE_BIT, point_instancer_culling)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(POINT_INSTANCER_CULLING_BINDING_CONSTANTS)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER)
        RW_STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_BATCH_DESCS)
        STRUCTURED_BUFFER(POINT_INSTANCER_CULLING_BINDING_BATCH_INDICES)
      END_PARAMETER()
    };
  }

  RtxPointInstancerSystem::RtxPointInstancerSystem(DxvkDevice* device)
    : CommonDeviceObject(device) { }

  void RtxPointInstancerSystem::showImguiSettings() {
    if (RemixGui::CollapsingHeader("Point Instancer Culling")) {
      ImGui::PushID("rtx_point_instancer");
      ImGui::Dummy({ 0, 2 });
      ImGui::Indent();

      RemixGui::Checkbox("Enable Culling", &enableObject());
      ImGui::BeginDisabled(!enable());

      RemixGui::DragFloat("Culling Radius", &cullingRadiusObject(), 10.f, fadeStartRadius(), 100000.f, "%.0f");
      RemixGui::DragFloat("Fade Start Radius", &fadeStartRadiusObject(), 10.f, 0.f, cullingRadius(), "%.0f");

      ImGui::EndDisabled();
      ImGui::Unindent();
      ImGui::PopID();
    }
  }

  void RtxPointInstancerSystem::dispatchCulling(
      Rc<DxvkContext> ctx,
      const Rc<DxvkBuffer>& instanceBuffer,
      const Rc<DxvkBuffer>& surfaceBuffer,
      const Rc<DxvkBuffer>& surfaceMaterialBuffer,
      const std::vector<PointInstancerBatch>& batches,
      const Vector3& cameraPosition) {
    ScopedGpuProfileZone(ctx, "PointInstancerCulling");

    if (batches.empty()) {
      return;
    }

    // Fused single dispatch (2026-07-20). The previous implementation looped
    // over batches, re-writing ONE shared transforms buffer and issuing one
    // tiny dispatch per batch. Each iteration's write-after-read hazard on
    // that shared buffer forced a full barrier between batches, serializing
    // the queue; at thousands of small GPU-instanced batches (FO4: ~1700
    // batches averaging ~4 instances) the pass measured ~4.4ms of GPU for
    // microseconds of useful work. All batch inputs are now concatenated and
    // uploaded once, per-batch constants live in a structured buffer, and a
    // single dispatch covers every instance.

    const Rc<DxvkDevice>& dev = ctx->getDevice();

    // --- Build fused CPU staging (persistent capacity across frames) ---
    m_transformsCpu.clear();
    m_descsCpu.clear();
    m_batchIdxCpu.clear();

    uint32_t totalInstances = 0;
    for (const PointInstancerBatch& batch : batches) {
      if (batch.instanceCount != 0 && batch.transforms != nullptr) {
        totalInstances += batch.instanceCount;
      }
    }
    if (totalInstances == 0) {
      return;
    }

    m_transformsCpu.reserve(totalInstances);
    m_batchIdxCpu.reserve(totalInstances);
    m_descsCpu.reserve(batches.size());

    for (const PointInstancerBatch& batch : batches) {
      if (batch.instanceCount == 0 || batch.transforms == nullptr) {
        continue;
      }
      PointInstancerBatchDescGpu desc {};
      memcpy(&desc.objectToWorld, &batch.objectToWorld, sizeof(mat4));
      memcpy(&desc.prevObjectToWorld, &batch.prevObjectToWorld, sizeof(mat4));
      desc.firstInstanceIndex   = static_cast<uint32_t>(m_transformsCpu.size());
      desc.baseSurfaceIndex     = batch.baseSurfaceIndex;
      desc.customIndexFlags     = batch.customIndexFlags;
      desc.instanceMask         = batch.instanceMask;
      desc.sbtOffsetAndFlags    = batch.sbtOffsetAndFlags;
      desc.blasRefLo            = static_cast<uint32_t>(batch.blasReference & 0xFFFFFFFFull);
      desc.blasRefHi            = static_cast<uint32_t>(batch.blasReference >> 32);
      desc.instanceBufferOffset = batch.instanceBufferByteOffset;

      m_transformsCpu.insert(m_transformsCpu.end(),
                             batch.transforms->begin(), batch.transforms->end());
      m_batchIdxCpu.insert(m_batchIdxCpu.end(), batch.instanceCount,
                           static_cast<uint32_t>(m_descsCpu.size()));
      m_descsCpu.push_back(desc);
    }

    // --- Upload (each buffer written exactly once per frame) ---
    auto ensureStorageBuffer = [&dev](Rc<DxvkBuffer>& buf, size_t size, const char* name) {
      if (buf.ptr() == nullptr || buf->info().size < size) {
        DxvkBufferCreateInfo info;
        info.usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
        info.size   = align(size, 256);
        buf = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                DxvkMemoryStats::Category::RTXBuffer, name);
      }
    };

    ensureStorageBuffer(m_transformsGpu, m_transformsCpu.size() * sizeof(Matrix4),
                        "RTX PointInstancer - Transforms Input");
    ensureStorageBuffer(m_batchDescsGpu, m_descsCpu.size() * sizeof(PointInstancerBatchDescGpu),
                        "RTX PointInstancer - Batch Descs");
    ensureStorageBuffer(m_batchIndicesGpu, m_batchIdxCpu.size() * sizeof(uint32_t),
                        "RTX PointInstancer - Batch Indices");

    ctx->writeToBuffer(m_transformsGpu, 0, m_transformsCpu.size() * sizeof(Matrix4),
                       m_transformsCpu.data());
    ctx->writeToBuffer(m_batchDescsGpu, 0, m_descsCpu.size() * sizeof(PointInstancerBatchDescGpu),
                       m_descsCpu.data());
    ctx->writeToBuffer(m_batchIndicesGpu, 0, m_batchIdxCpu.size() * sizeof(uint32_t),
                       m_batchIdxCpu.data());

    // Allocate constant buffer (once)
    if (m_cb.ptr() == nullptr) {
      DxvkBufferCreateInfo info;
      info.usage  = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
      info.size   = sizeof(PointInstancerCullingConstants);
      m_cb = dev->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               DxvkMemoryStats::Category::RTXBuffer,
                               "RTX PointInstancer - Constant Buffer");
    }

    // When culling is disabled, use FLT_MAX so every instance passes the distance test.
    const bool cullingEnabled = enable();
    PointInstancerCullingConstants constants {};
    constants.cameraPosition     = { cameraPosition.x, cameraPosition.y, cameraPosition.z };
    constants.cullingRadius      = cullingEnabled ? cullingRadius() : FLT_MAX;
    constants.totalInstanceCount = totalInstances;
    constants.fadeStartRadius    = cullingEnabled ? fadeStartRadius() : 0.f;

    const DxvkBufferSliceHandle cSlice = m_cb->allocSlice();
    ctx->invalidateBuffer(m_cb, cSlice);
    ctx->writeToBuffer(m_cb, 0, sizeof(PointInstancerCullingConstants), &constants);

    // Bind resources
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_CONSTANTS, DxvkBufferSlice(m_cb));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_TRANSFORMS_INPUT, DxvkBufferSlice(m_transformsGpu));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_INSTANCE_BUFFER, DxvkBufferSlice(instanceBuffer));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_SURFACE_BUFFER, DxvkBufferSlice(surfaceBuffer));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_MATERIAL_BUFFER, DxvkBufferSlice(surfaceMaterialBuffer));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_BATCH_DESCS, DxvkBufferSlice(m_batchDescsGpu));
    ctx->bindResourceBuffer(POINT_INSTANCER_CULLING_BINDING_BATCH_INDICES, DxvkBufferSlice(m_batchIndicesGpu));

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PointInstancerCullingShader::getShader());

    const VkExtent3D workgroups = util::computeBlockCount(
      VkExtent3D { totalInstances, 1, 1 },
      VkExtent3D { 64, 1, 1 });

    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
}
