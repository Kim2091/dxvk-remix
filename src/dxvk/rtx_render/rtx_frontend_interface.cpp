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
#include <utility>

#include <remix/remix_c.h>

#include "../dxvk_device.h"
#include "rtx_camera_manager.h"
#include "rtx_context.h"
#include "rtx_fork_camera_origin.h"
#include "rtx_frontend_interface.h"
#include "rtx_scene_manager.h"

namespace dxvk {

  namespace {
    CameraType::Enum toCameraType(frontend::CameraView view) {
      switch (view) {
      case frontend::CameraView::ViewModel: return CameraType::ViewModel;
      case frontend::CameraView::Sky: return CameraType::Sky;
      default: return CameraType::Main;
      }
    }
  }

  // The seam between the interface and the backend's draw-call record.
  // DrawCallState and LegacyMaterialData name it a friend, so it can fill their private fields.
  struct FrontendSceneSeam {
    static void toDrawCallState(frontend::SceneInstanceDesc& desc, DrawCallState& drawCall) {
      drawCall.cameraType = toCameraType(desc.cameraView);
      drawCall.transformData.objectToWorld = desc.objectToWorld;
      drawCall.transformData.texgenMode = TexGenMode::None;
      drawCall.categories = CategoryFlags(desc.categories);

      if (desc.objectPickingValue.has_value()) {
        drawCall.drawCallID = *desc.objectPickingValue;
      }

      if (desc.boneTransforms.has_value()) {
        SkinningData& skinning = drawCall.skinningData;
        skinning.minBoneIndex = 0;
        skinning.numBones = uint32_t(desc.boneTransforms->size());
        // The mesh is attached per submesh on the render thread, so the bone count per vertex is not known here.
        // Skinning reads it from RasterGeometry::numBonesPerVertex, and the capturer and crash recorder fall back to it.
        skinning.numBonesPerVertex = 0;
        skinning.pBoneMatrices = std::move(*desc.boneTransforms);
        skinning.computeHash();
      }

      if (desc.blend.has_value()) {
        const frontend::LegacyBlendDesc& blend = *desc.blend;
        LegacyMaterialData& material = drawCall.materialData;
        material.alphaTestEnabled = blend.alphaTestEnabled != 0;
        material.alphaTestReferenceValue = blend.alphaTestReferenceValue;
        material.alphaTestCompareOp = static_cast<VkCompareOp>(blend.alphaTestCompareOp);
        material.blendMode.enableBlending = blend.alphaBlendEnabled;
        material.textureColorOperation = static_cast<DxvkRtTextureOperation>(blend.textureColorOperation);
        material.textureColorArg1Source = static_cast<RtTextureArgSource>(blend.textureColorArg1Source);
        material.textureColorArg2Source = static_cast<RtTextureArgSource>(blend.textureColorArg2Source);
        material.textureAlphaOperation = static_cast<DxvkRtTextureOperation>(blend.textureAlphaOperation);
        material.textureAlphaArg1Source = static_cast<RtTextureArgSource>(blend.textureAlphaArg1Source);
        material.textureAlphaArg2Source = static_cast<RtTextureArgSource>(blend.textureAlphaArg2Source);
        material.tFactor = blend.tFactor;
        material.isTextureFactorBlend = blend.isTextureFactorBlend != 0;
        material.isVertexColorBakedLighting = blend.isVertexColorBakedLighting;
        material.blendMode.colorSrcFactor = static_cast<VkBlendFactor>(blend.srcColorBlendFactor);
        material.blendMode.colorDstFactor = static_cast<VkBlendFactor>(blend.dstColorBlendFactor);
        material.blendMode.colorBlendOp = static_cast<VkBlendOp>(blend.colorBlendOp);
        material.blendMode.alphaSrcFactor = static_cast<VkBlendFactor>(blend.srcAlphaBlendFactor);
        material.blendMode.alphaDstFactor = static_cast<VkBlendFactor>(blend.dstAlphaBlendFactor);
        material.blendMode.alphaBlendOp = static_cast<VkBlendOp>(blend.alphaBlendOp);
        material.blendMode.writeMask = static_cast<VkColorComponentFlags>(blend.writeMask);
      }
    }

    static std::unique_ptr<ExternalDrawState> toExternalDrawState(frontend::SceneInstanceDesc&& desc) {
      auto state = std::make_unique<ExternalDrawState>();
      toDrawCallState(desc, state->drawCall);

      if (desc.particles.has_value()) {
        state->optionalParticleDesc.emplace(std::move(*desc.particles));
      }
      state->gpuInstancingTransforms = std::move(desc.gpuInstancing);
      state->materialOverride = reinterpret_cast<remixapi_MaterialHandle>(desc.materialOverride);
      state->mesh = reinterpret_cast<remixapi_MeshHandle>(desc.mesh);
      state->cameraType = state->drawCall.cameraType;
      state->categories = CategoryFlags(desc.categories);
      state->doubleSided = desc.doubleSided;
      return state;
    }
  };

  namespace frontend {

    PreparedInstance::PreparedInstance() = default;
    PreparedInstance::PreparedInstance(PreparedInstance&& other) noexcept = default;
    PreparedInstance& PreparedInstance::operator=(PreparedInstance&& other) noexcept = default;
    PreparedInstance::~PreparedInstance() = default;

    PreparedInstance prepareInstance(SceneInstanceDesc&& desc) {
      PreparedInstance prepared;
      prepared.m_state = FrontendSceneSeam::toExternalDrawState(std::move(desc));
      return prepared;
    }

    void SceneInterface::drawInstance(PreparedInstance&& instance) {
      m_ctx.commitExternalGeometryToRT(std::move(instance.m_state));
    }

    void SceneInterface::describeInstance(InstanceHandle handle, PreparedInstance&& instance) {
      m_ctx.getSceneManager().setRetainedExternalDraw(handle, std::move(*instance.m_state));
    }

    void SceneInterface::setTransform(InstanceHandle handle, const Matrix4& objectToWorld) {
      m_ctx.getSceneManager().setRetainedExternalDrawTransform(handle, objectToWorld);
    }

    void SceneInterface::setVisible(InstanceHandle handle, bool visible) {
      m_ctx.getSceneManager().setRetainedExternalDrawVisible(handle, visible);
    }

    void SceneInterface::destroyInstance(InstanceHandle handle) {
      m_ctx.getSceneManager().removeRetainedExternalDraw(handle);
    }

    void SceneInterface::drawRetainedInstances(const FrameInfo& frame) {
      m_ctx.commitRetainedExternalGeometry(frame.sceneOrigin);
    }

    void SceneInterface::setupCamera(const CameraSetup& camera) {
      SceneManager& sceneManager = m_ctx.getSceneManager();
      CameraManager& cameras = sceneManager.getCameraManager();
      const CameraType::Enum type = toCameraType(camera.view);
      const Vector3* worldOrigin = camera.worldOrigin.has_value() ? &*camera.worldOrigin : nullptr;
      if (worldOrigin != nullptr && type == CameraType::Main) {
        sceneManager.applyWorldOrigin(*worldOrigin);
      }
      if (camera.events & kFrameEventOriginShift) {
        cameras.shiftCameraHistory(type, camera.originShift);
      }
      // rtx.worldOrigin.preciseCamera keeps the camera's position in double. The float world-to-view translation is
      // rounded at world magnitude differently every frame as the camera turns, which a view model at arm's length shows.
      const bool precise = camera.preciseWorldToView.has_value() && WorldOrigin::preciseCamera();
      cameras.processExternalCamera(type, camera.worldToView, camera.viewToProjection, worldOrigin,
                                    precise ? &*camera.preciseWorldToView : nullptr);
      if ((camera.events & (kFrameEventCameraCut | kFrameEventHistoryInvalid)) && type == CameraType::Main) {
        cameras.signalCameraCut();
      }
    }

    CameraQuery SceneInterface::camera(CameraView view) const {
      const CameraType::Enum type = toCameraType(view);
      const CameraManager& cameras = m_ctx.getSceneManager().getCameraManager();
      const RtCamera& rtCamera = cameras.getCamera(type);
      CameraQuery query;
      query.valid = cameras.isCameraValid(type);
      query.worldToView = rtCamera.getWorldToViewf(false);
      query.viewToProjection = rtCamera.getViewToProjectionf();
      return query;
    }

    uint32_t SceneInterface::frameIndex() const {
      return m_ctx.getDevice()->getCurrentFrameId();
    }

  } // namespace frontend
} // namespace dxvk
