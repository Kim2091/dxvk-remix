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

#include <cmath>

namespace {
  constexpr float kFovToleranceRadians = 0.001f;
  constexpr float kAspectRelDiffThreshold = 0.10f;
  constexpr float kFovDiffThreshold = 0.35f;
  constexpr float kStrongDirDotThreshold = 0.20f;
  constexpr float kSuspiciousRotationDirDotThreshold = 0.35f;
  constexpr float kPersistentDirDotThreshold = 0.85f;
  constexpr float kPersistentFovDiffThreshold = 0.10f;
  constexpr float kPersistentAspectRelDiffThreshold = 0.05f;

  struct CameraPose {
    dxvk::Vector3 position;
    dxvk::Vector3 direction;
  };

  CameraPose getCameraPose(const dxvk::Matrix4& worldToView, const bool isLHS) {
    const dxvk::Matrix4 viewToWorld = dxvk::inverseAffine(worldToView);

    CameraPose pose {
      dxvk::Vector3 { viewToWorld[3].xyz() },
      dxvk::Vector3 { viewToWorld[2].xyz() }
    };

    if (!isLHS) {
      pose.direction = -pose.direction;
    }

    const float directionLength = dxvk::length(pose.direction);
    if (directionLength > 0.0f) {
      pose.direction /= directionLength;
    }

    return pose;
  }

  float aspectRelativeDifference(const float a, const float b) {
    return b > 1e-6f ? std::abs(a - b) / b : 0.0f;
  }
}

namespace dxvk {

  CameraManager::CameraManager(DxvkDevice* device) : CommonDeviceObject(device) {
    for (int i = 0; i < CameraType::Count; i++) {
      m_cameras[i].setCameraType(CameraType::Enum(i));
    }
  }

  bool CameraManager::isCameraValid(CameraType::Enum cameraType) const {
    assert(cameraType < CameraType::Enum::Count);
    const uint32_t frameId = m_device->getCurrentFrameId();
    const RtCamera& camera = accessCamera(*this, cameraType);
    if (camera.isValid(frameId)) {
      return true;
    }

    return false;
  }

  void CameraManager::finalizeFrameCameras() {
    const uint32_t frameId = m_device->getCurrentFrameId();
    if (m_lastFinalizedFrameId == frameId) {
      return;
    }

    m_lastFinalizedFrameId = frameId;

    if (!guardMainCameraFromOutliers()) {
      return;
    }

    RtCamera& mainCamera = getCamera(CameraType::Main);
    if (mainCamera.isValid(frameId)) {
      return;
    }

    if (syncMainCameraFromRenderToTexture()) {
      const RtCamera& renderTargetCamera = getCamera(CameraType::RenderToTexture);
      if (renderTargetCamera.isValid(frameId)) {
        const RtCamera::RtCameraSetting& setting = renderTargetCamera.getSetting();
        const bool isCameraCut = mainCamera.update(
          frameId,
          setting.worldToView,
          setting.viewToProjection,
          setting.fov,
          setting.aspectRatio,
          setting.nearPlane,
          setting.farPlane,
          setting.isLHS,
          setting.flags);

        if (isCameraCut) {
          m_lastCameraCutFrameId = frameId;
        }

        m_pendingMainJumpCandidate.valid = false;
        m_lastSetCameraType = CameraType::Main;

        if (logMainCameraUpdates()) {
          Logger::debug(str::format(
            "[RTX-Compatibility] CameraManager: finalized Main camera from RenderToTexture on frame ", frameId,
            "; cameraCut=", isCameraCut ? "true" : "false"));
        }
        return;
      }
    }

    if (frameId > 0 &&
        mainCamera.isValid(frameId - 1) &&
        m_pendingMainJumpCandidate.valid &&
        m_pendingMainJumpCandidate.frameId == frameId) {
      mainCamera.holdFrame(frameId);
      m_lastSetCameraType = CameraType::Main;

      if (logMainCameraUpdates() && m_lastRejectedMainCameraLogFrameId != frameId) {
        m_lastRejectedMainCameraLogFrameId = frameId;
        Logger::debug(str::format(
          "[RTX-Compatibility] CameraManager: held previous Main camera on frame ", frameId,
          " after rejecting all candidates for this frame."));
      }
    }
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

    auto& camera = getCamera(cameraType);
    auto cameraSequence = RtCameraSequence::getInstance();
    bool shouldUpdateMainCamera = cameraType == CameraType::Main && camera.getLastUpdateFrame() != frameId;
    bool isPlaying = RtCameraSequence::mode() == RtCameraSequence::Mode::Playback;
    bool isBrowsing = RtCameraSequence::mode() == RtCameraSequence::Mode::Browse;
    bool isCameraCut = false;
    Matrix4 worldToView = input.getTransformData().worldToView;
    Matrix4 viewToProjection = input.getTransformData().viewToProjection;
    const bool hadPreviousMainCamera =
      shouldUpdateMainCamera &&
      frameId > 0 &&
      getCamera(CameraType::Main).isValid(frameId - 1);
    Vector3 previousMainPosition(0.0f);
    Vector3 previousMainDirection(0.0f);
    float previousMainFov = 0.0f;

    if (hadPreviousMainCamera) {
      const RtCamera& prevMain = getCamera(CameraType::Main);
      previousMainPosition = prevMain.getPosition(false);
      previousMainDirection = prevMain.getDirection(false);
      const float previousDirectionLength = length(previousMainDirection);
      if (previousDirectionLength > 0.0f) {
        previousMainDirection /= previousDirectionLength;
      }
      previousMainFov = prevMain.getFov();
    }

    if (guardMainCameraFromOutliers() && hadPreviousMainCamera) {
      const RtCamera& prevMain = getCamera(CameraType::Main);

      const float prevAspect = prevMain.getAspectRatio();
      const float prevFov = previousMainFov;

      const float aspectRelDiff = aspectRelativeDifference(decomposeProjectionParams.aspectRatio, prevAspect);

      const CameraPose candidatePose = getCameraPose(worldToView, decomposeProjectionParams.isLHS);

      const float dirDot = dot(candidatePose.direction, previousMainDirection);
      const float fovDiff = std::abs(decomposeProjectionParams.fov - prevFov);

      // Aspect-ratio outliers are usually utility cameras, not a real player view transition.
      const bool rejectedByAspect = aspectRelDiff > kAspectRelDiffThreshold;

      if (rejectedByAspect) {
        m_pendingMainJumpCandidate.valid = false;
        if (logMainCameraUpdates()) {
          Logger::debug(str::format(
            "[RTX-Compatibility] CameraManager: rejected Main camera candidate on frame ", frameId,
            " by aspect; aspectRelDiff=", aspectRelDiff,
            ", fovDiff=", fovDiff,
            ", dirDot=", dirDot));
        }
        ONCE(Logger::warn("[RTX-Compatibility] CameraManager: rejected outlier Main camera candidate (likely shadow/utility camera)."));
        return CameraType::Unknown;
      }

      // reject the bad camera frame spikes but accept persistent jumps
      // on the next frame so real teleports/loads still converge to the new camera
      {
        const float posJumpDistSqr = lengthSqr(candidatePose.position - previousMainPosition);
        const float suspiciousJumpThresholdSqr = RtxOptions::getUniqueObjectDistanceSqr() * 4.0f;
        const bool isSuspiciousJump = posJumpDistSqr > suspiciousJumpThresholdSqr;
        const bool isSuspiciousRotation =
          dirDot < kSuspiciousRotationDirDotThreshold ||
          (fovDiff > kFovDiffThreshold && dirDot < kStrongDirDotThreshold);

        if (isSuspiciousJump || isSuspiciousRotation) {
          bool isPersistentJump = false;
          if (m_pendingMainJumpCandidate.valid &&
              m_pendingMainJumpCandidate.frameId + 1 == frameId) {
            const float repeatPosDistSqr = lengthSqr(candidatePose.position - m_pendingMainJumpCandidate.position);
            const float repeatDirDot = dot(candidatePose.direction, m_pendingMainJumpCandidate.direction);
            const float repeatFovDiff = std::abs(decomposeProjectionParams.fov - m_pendingMainJumpCandidate.fov);
            const float repeatAspectRelDiff = aspectRelativeDifference(decomposeProjectionParams.aspectRatio, m_pendingMainJumpCandidate.aspectRatio);

            isPersistentJump =
              repeatPosDistSqr <= RtxOptions::getUniqueObjectDistanceSqr() &&
              repeatDirDot > kPersistentDirDotThreshold &&
              repeatFovDiff < kPersistentFovDiffThreshold &&
              repeatAspectRelDiff < kPersistentAspectRelDiffThreshold;
          }

          if (!isPersistentJump) {
            m_pendingMainJumpCandidate.valid = true;
            m_pendingMainJumpCandidate.frameId = frameId;
            m_pendingMainJumpCandidate.position = candidatePose.position;
            m_pendingMainJumpCandidate.direction = candidatePose.direction;
            m_pendingMainJumpCandidate.fov = decomposeProjectionParams.fov;
            m_pendingMainJumpCandidate.aspectRatio = decomposeProjectionParams.aspectRatio;
            if (logMainCameraUpdates()) {
              Logger::debug(str::format(
                "[RTX-Compatibility] CameraManager: rejected provisional Main camera candidate on frame ", frameId,
                "; posJumpDist=", std::sqrt(posJumpDistSqr),
                ", threshold=", std::sqrt(suspiciousJumpThresholdSqr),
                ", dirDot=", dirDot,
                ", fovDiff=", fovDiff,
                ", suspiciousJump=", isSuspiciousJump ? "true" : "false",
                ", suspiciousRotation=", isSuspiciousRotation ? "true" : "false"));
            }
            ONCE(Logger::warn("[RTX-Compatibility] CameraManager: rejected provisional Main camera candidate."));
            return CameraType::Unknown;
          }

          if (logMainCameraUpdates()) {
            Logger::debug(str::format(
              "[RTX-Compatibility] CameraManager: accepted persistent Main camera candidate on frame ", frameId,
              "; posJumpDist=", std::sqrt(posJumpDistSqr),
              ", threshold=", std::sqrt(suspiciousJumpThresholdSqr),
              ", dirDot=", dirDot,
              ", fovDiff=", fovDiff));
          }
        }

        m_pendingMainJumpCandidate.valid = false;
      }
    }

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

    if (shouldUpdateMainCamera && logMainCameraUpdates() && camera.isValid(frameId)) {
      float positionDelta = 0.0f;
      float directionDot = 1.0f;
      float fovDiff = 0.0f;
      if (hadPreviousMainCamera) {
        Vector3 currentDirection = camera.getDirection(false);
        const float currentDirectionLength = length(currentDirection);
        if (currentDirectionLength > 0.0f) {
          currentDirection /= currentDirectionLength;
        }

        positionDelta = std::sqrt(lengthSqr(camera.getPosition(false) - previousMainPosition));
        directionDot = dot(currentDirection, previousMainDirection);
        fovDiff = std::abs(camera.getFov() - previousMainFov);
      }

      Logger::debug(str::format(
        "[RTX-Compatibility] CameraManager: accepted Main camera on frame ", frameId,
        "; positionDelta=", positionDelta,
        ", directionDot=", directionDot,
        ", fovDiff=", fovDiff,
        ", cameraCut=", isCameraCut ? "true" : "false",
        ", viewHistoryInvalidated=", camera.isViewHistoryInvalidated(frameId) ? "true" : "false",
        ", guard=", guardMainCameraFromOutliers() ? "true" : "false"));
    }

    // Register camera cut when there are significant interruptions to the view (like changing level, or opening a menu)
    if (isCameraCut && cameraType == CameraType::Main) {
      m_lastCameraCutFrameId = m_device->getCurrentFrameId();
    }
    m_lastSetCameraType = cameraType;

    return cameraType;
  }

  bool CameraManager::isCameraCutThisFrame() const {
    return m_lastCameraCutFrameId == m_device->getCurrentFrameId();
  }

  void CameraManager::processExternalCamera(CameraType::Enum type,
                                            const Matrix4& worldToView,
                                            const Matrix4& viewToProjection) {
    DecomposeProjectionParams decomposeProjectionParams = getOrDecomposeProjection(viewToProjection);

    getCamera(type).update(
      m_device->getCurrentFrameId(),
      worldToView,
      viewToProjection,
      decomposeProjectionParams.fov,
      decomposeProjectionParams.aspectRatio,
      decomposeProjectionParams.nearPlane,
      decomposeProjectionParams.farPlane,
      decomposeProjectionParams.isLHS);
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
