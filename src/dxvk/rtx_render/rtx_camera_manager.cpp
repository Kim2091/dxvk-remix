/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_camera_manager.h"

#include "dxvk_device.h"
#include "rtx_fork_camera_origin.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace {
  constexpr float kFovToleranceRadians = 0.001f;
}

namespace dxvk {

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    return accessCamera(*this, cameraType).isValid(m_device->getCurrentFrameId());
  }

  void CameraManager::onFrameEnd() {
    m_lastSetCameraType = CameraType::Unknown;
    m_decompositionCache.clear();
  }

  CameraType::Enum CameraManager::processCameraData(const DrawCallState& input) {
    // If theres no real camera data here - bail
    if (isIdentityExact(input.getTransformData().viewToProjection)) {
      return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }

    switch (RtxOptions::fusedWorldViewMode()) {
    case FusedWorldViewMode::None:
      if (input.getTransformData().objectToView == input.getTransformData().objectToWorld && !isIdentityExact(input.getTransformData().objectToView)) {
        return input.testCategoryFlags(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
      }
      break;
    case FusedWorldViewMode::View:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if World is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToWorld),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to View but World transform is not identity!"));
      }
      break;
    case FusedWorldViewMode::World:
      if (Logger::logLevel() >= LogLevel::Warn) {
        // Check if View is identity
        ONCE_IF_FALSE(isIdentityExact(input.getTransformData().objectToView),
                      Logger::warn("[RTX-Compatibility] Fused world-view tranform set to World but View transform is not identity!"));
      }
      break;
    }

    // Get camera params
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(input.getTransformData().viewToProjection);

    // Filter invalid cameras, extreme shearing
    static auto isFovValid = [](float fovA) {
      return fovA >= kFovToleranceRadians;
    };
    static auto areFovsClose = [](float fovA, const RtCamera& cameraB) {
      return std::abs(fovA - cameraB.getFov()) < kFovToleranceRadians;
    };

    if (std::abs(decomposeProjectionParams.shearX) > 0.01f || !isFovValid(decomposeProjectionParams.fov)) {
      ONCE(Logger::warn("[RTX] CameraManager: rejected an invalid camera"));
      return input.getCategoryFlags().test(InstanceCategories::Sky) ? CameraType::Sky : CameraType::Unknown;
    }


    auto isViewModel = [this](float fov, float maxZ, uint32_t frameId) {
      if (RtxOptions::ViewModel::enable()) {
        // Note: max Z check is the top-priority
        if (maxZ <= RtxOptions::ViewModel::maxZThreshold()) {
          return true;
        }
        if (getCamera(CameraType::Main).isValid(frameId)) {
          // FOV is different from Main camera => assume that it's a ViewModel one
          if (!areFovsClose(fov, getCamera(CameraType::Main))) {
            return true;
          }
        }
      }
      return false;
    };

    const uint32_t frameId = m_device->getCurrentFrameId();

    // The draw is classified against the host's camera when that camera is applied first.
    if (!input.isDrawingToRaytracedRenderTarget && !input.testCategoryFlags(InstanceCategories::Sky)) {
      applyHeldFallbackCamera(decomposeProjectionParams.fov, "draw");
    }

    auto cameraType = CameraType::Main;
    if (input.isDrawingToRaytracedRenderTarget) {
      cameraType = CameraType::RenderToTexture;
    } else if (input.testCategoryFlags(InstanceCategories::Sky)) {
      cameraType = CameraType::Sky;
    } else if (isViewModel(decomposeProjectionParams.fov, input.maxZ, frameId)) {
      cameraType = CameraType::ViewModel;
    }
    
    // Check fov consistency across frames
    if (frameId > 0) {
      if (getCamera(cameraType).isValid(frameId - 1) && !areFovsClose(decomposeProjectionParams.fov, getCamera(cameraType))) {
        ONCE(Logger::warn("[RTX] CameraManager: FOV of a camera changed between frames"));
      }
    }

    applyFrameCamera(cameraType, input.getTransformData().worldToView, input.getTransformData().viewToProjection);
    return cameraType;
  }

  void CameraManager::applyFrameCamera(CameraType::Enum cameraType, const Matrix4& worldToView, const Matrix4& viewToProjection) {
    const uint32_t frameId = m_device->getCurrentFrameId();
    const DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);
    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    bool shouldUpdateMainCamera = cameraType == CameraType::Main && camera.getLastUpdateFrame() != frameId;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    if (isPlaying || isBrowsing) {
      if (shouldUpdateMainCamera) {
        RtCamera::RtCameraSetting setting;
        cameraSequence->getRecord(cameraSequence->currentFrame(), setting);
        isCameraCut = camera.updateFromSetting(frameId, setting, 0);

        if (isPlaying) {
          cameraSequence->goToNextFrame();
        }
      }
    } else {
      isCameraCut = camera.update(
        frameId,
        worldToView,
        viewToProjection,
        decomposeProjectionParams.fov,
        decomposeProjectionParams.aspectRatio,
        decomposeProjectionParams.nearPlane,
        decomposeProjectionParams.farPlane,
        decomposeProjectionParams.isLHS
      );
    }


    if (shouldUpdateMainCamera && RtCameraSequence::mode() == RtCameraSequence::Mode::Record) {
      auto& setting = camera.getSetting();
      cameraSequence->addRecord(setting);
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;
  }

  void CameraManager::applyDrawCamera(CameraType::Enum cameraType, const Matrix4& worldToView, const Matrix4& viewToProjection) {
    // processCameraData applies the held camera ahead of every draw that is neither a render-to-texture nor a sky draw,
    // and only a Main or view-model draw can find Main unset.
    if (cameraType == CameraType::Main || cameraType == CameraType::ViewModel) {
      applyHeldFallbackCamera(getOrDecomposeProjection(viewToProjection).fov, "frontend");
    }
    applyFrameCamera(cameraType, worldToView, viewToProjection);
  }

  bool CameraManager::applyHeldFallbackCamera(float drawFov, const char* source) {
    // REMIXAPI_CAMERA_FRAME_EVENT_FALLBACK: the host stated this frame's view and no draw has set the Main
    // camera yet. A draw with that FOV is the world camera itself and sets Main as always; a draw with another FOV
    // (a first-person draw on a frame whose world camera the host stated) must not become the frame's camera.
    const uint32_t frameId = m_device->getCurrentFrameId();
    if (!m_fallbackCameraHeld || m_fallbackCameraFrameId != frameId || getCamera(CameraType::Main).isValid(frameId) ||
        std::abs(drawFov - m_fallbackFov) < kFovToleranceRadians) {
      return false;
    }
    m_fallbackCameraHeld = false;
    applyFrameCamera(CameraType::Main, m_fallbackWorldToView, m_fallbackViewToProjection);
    if (m_fallbackAheadLogs < 8) {
      ++m_fallbackAheadLogs;
      Logger::info(str::format("[RTX] Frame ", frameId, ": the host's fallback camera (FOV ", m_fallbackFov,
                               " rad) set the Main camera ahead of a ", source, " camera with FOV ", drawFov,
                               " rad. First 8 are logged."));
    }
    return true;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::signalCameraCut() {
    getMainCamera().invalidateViewHistory(m_device->getCurrentFrameId());
  }

  void CameraManager::shiftCameraHistory(CameraType::Enum type, const Vector3& offset) {
    if (type >= CameraType::Count) {
      return;
    }
    RtCamera& camera = getCamera(type);
    if (camera.getLastUpdateFrame() != m_device->getCurrentFrameId() && offset != Vector3(0.0f)) {
      camera.applyArtificialWorldOffset(offset);
    }
  }

  void CameraManager::holdFallbackCamera(const Matrix4& worldToView, const Matrix4& viewToProjection) {
    m_fallbackWorldToView = worldToView;
    m_fallbackViewToProjection = viewToProjection;
    m_fallbackFov = getOrDecomposeProjection(viewToProjection).fov;
    m_fallbackCameraFrameId = m_device->getCurrentFrameId();
    m_fallbackCameraHeld = true;
  }

  bool CameraManager::takeFallbackCamera(Matrix4& worldToView, Matrix4& viewToProjection) {
    if (!m_fallbackCameraHeld) {
      return false;
    }
    m_fallbackCameraHeld = false;
    // This frame's, or one that arrived after the previous frame's ray tracing.
    if (m_device->getCurrentFrameId() - m_fallbackCameraFrameId > 1) {
      return false;
    }
    worldToView = m_fallbackWorldToView;
    viewToProjection = m_fallbackViewToProjection;
    return true;
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection,
                                            const Vector3* worldOrigin,
                                            const Matrix4d* preciseWorldToView) {
    if (worldOrigin != nullptr && type < CameraType::Count) {
      if (type == CameraType::Main) {
        m_apiWorldOriginSeen = true;
      }
      applyHostWorldOrigin(type, *worldOrigin);
    }
    updateExternalCamera(type, worldToView, viewToProjection, preciseWorldToView);
  }

  bool CameraManager::applyHostWorldOrigin(CameraType::Enum type, const Vector3& worldOrigin) {
    // Ported from BetaRT (sambow23/dxvk-remix-gmod betart-numos3, commit 11761453a): shift last frame's camera
    // into this frame's render origin before update() turns it into the previous camera, so an origin move is
    // neither motion nor a camera cut. Only once per frame: update() ignores a second call in the same frame.
    RtCamera& camera = getCamera(type);
    if (camera.getLastUpdateFrame() == m_device->getCurrentFrameId()) {
      return false;
    }
    const Vector3 previousHistoryOffset = fork_camera_origin::calculatePreviousCameraHistoryOffset(
      m_externalCameraOriginValid[type], m_externalCameraOrigins[type], worldOrigin);
    if (previousHistoryOffset != Vector3(0.0f)) {
      camera.applyArtificialWorldOffset(previousHistoryOffset);
    }
    m_externalCameraOrigins[type] = worldOrigin;
    m_externalCameraOriginValid[type] = true;
    return true;
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    updateExternalCamera(type, worldToView, viewToProjection, nullptr);
  }

  void CameraManager::updateExternalCamera(CameraType::Enum type,
                                           const Matrix4& worldToView,
                                           const Matrix4& viewToProjection,
                                           const Matrix4d* preciseWorldToView) {
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      viewToProjection,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS,
      (uint32_t) RtCamera::UpdateFlag::UpdateNormal,
      preciseWorldToView);
  }

    DecomposeProjectionParams CameraManager::getOrDecomposeProjection(const Matrix4& viewToProjection) {
      XXH64_hash_t projectionHash = XXH64(&viewToProjection, sizeof(viewToProjection), 0);
      auto iter = m_decompositionCache.find(projectionHash);
      if (iter != m_decompositionCache.end()) {
        return iter->second;
      }

      DecomposeProjectionParams decomposeProjectionParams;
      decomposeProjection(viewToProjection, decomposeProjectionParams);
      m_decompositionCache.emplace(projectionHash, decomposeProjectionParams);
      return decomposeProjectionParams;
    }
}  // namespace dxvk
