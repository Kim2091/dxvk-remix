/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_camera.h"
#include "rtx_option.h"
#include "rtx_types.h"
#include "rtx_common_object.h"
#include "rtx_matrix_helpers.h"

namespace dxvk {
  class DxvkDevice;
  class RayPortalManager;
  class Config;

  class CameraManager : public CommonDeviceObject {
  public:
    explicit CameraManager(DxvkDevice* device);
    ~CameraManager() override = default;

    CameraManager(const CameraManager& other) = delete;
    CameraManager(CameraManager&& other) noexcept = delete;
    CameraManager& operator=(const CameraManager& other) = delete;
    CameraManager& operator=(CameraManager&& other) noexcept = delete;

    const RtCamera& getCamera(CameraType::Enum cameraType) const { return accessCamera(*this, cameraType); }
    RtCamera& getCamera(CameraType::Enum cameraType) { return accessCamera(*this, cameraType); }

    const RtCamera& getMainCamera() const { return getCamera(CameraType::Main); }
    RtCamera& getMainCamera() { return getCamera(CameraType::Main); }

    CameraType::Enum getLastSetCameraType() const { return m_lastSetCameraType; }
    
    bool isCameraValid(CameraType::Enum cameraType) const;

    void onFrameEnd();

    // A camera a frontend found in its draws (SceneInterface::applyDrawCamera): updates the camera of this type (the
    // first call per camera and frame wins, as RtCamera::update does), with camera-sequence playback / browse / record,
    // the Main camera's cut registration and the last-set camera type.
    void applyFrameCamera(CameraType::Enum cameraType, const Matrix4& worldToView, const Matrix4& viewToProjection);
    void processExternalCamera(CameraType::Enum type, const Matrix4& worldToView, const Matrix4& viewToProjection);
    // rtx.worldOrigin.enable: worldOrigin is the host's render origin for this camera (render units,
    // render position + origin = world position), or null. When it moved since this camera's last update, the
    // camera's history is re-expressed around the new origin first (see rtx_fork_camera_origin.h).
    // preciseWorldToView (rtx.worldOrigin.preciseCamera): the host camera's world-to-view in double; replaces the
    // float matrix, whose translation is otherwise rounded at world magnitude every frame.
    void processExternalCamera(CameraType::Enum type, const Matrix4& worldToView, const Matrix4& viewToProjection,
                               const Vector3* worldOrigin, const Matrix4d* preciseWorldToView = nullptr);
    // The Main camera's current host world origin (zero when none has been published).
    const Vector3& getWorldOrigin() const { return m_externalCameraOrigins[CameraType::Main]; }
    bool hasWorldOrigin() const { return m_externalCameraOriginValid[CameraType::Main]; }

    uint32_t getLastCameraCutFrameId() const { return m_lastCameraCutFrameId; }
    bool isCameraCutThisFrame() const;

    // Scene interface (remixapi_CameraInfoFrameEventsEXT): a cut the host stated. Restarts temporal history this
    // frame; unlike a detected cut it never clears the scene.
    void signalCameraCut();
    // ...ORIGIN_SHIFT: re-expresses the camera's history after the host moved its render origin by offset (new
    // render position = old + offset). Before the camera's update this frame; a second call in a frame is ignored.
    void shiftCameraHistory(CameraType::Enum type, const Vector3& offset);

  private:
    template<
      typename T,
      std::enable_if_t<std::is_same_v<T, CameraManager> || std::is_same_v<T, const CameraManager>, bool> = true>
    static std::conditional_t<std::is_const_v<T>, const RtCamera&, RtCamera&> accessCamera(T& cameraManager, CameraType::Enum cameraType) {
      assert(cameraType < CameraType::Count);
      // Default Unknown to Main camera object
      // since cameras can get rejected but rtx pipeline can
      // still try to retrieve a camera corresponding to a DrawCall object.
      // In that case it will read from the Main camera.
      // This is OK as we never update Unknown camera directly.
      if (cameraType == CameraType::Unknown) {
        return cameraManager.m_cameras[CameraType::Main];
      }
      return cameraManager.m_cameras[cameraType];
    }

  private:
    std::array<RtCamera, CameraType::Count> m_cameras;
    CameraType::Enum m_lastSetCameraType = CameraType::Unknown;
    uint32_t m_lastCameraCutFrameId = -1;
    uint32_t m_lastCameraViewHistoryInvalidationFrameId = -1;
    fast_unordered_cache<DecomposeProjectionParams> m_decompositionCache;
    std::array<Vector3, CameraType::Count> m_externalCameraOrigins {};
    std::array<bool, CameraType::Count> m_externalCameraOriginValid {};

    DecomposeProjectionParams getOrDecomposeProjection(const Matrix4& viewToProjection);
    void updateExternalCamera(CameraType::Enum type, const Matrix4& worldToView, const Matrix4& viewToProjection,
                              const Matrix4d* preciseWorldToView);

    RTX_OPTION("rtx", bool, rayPortalEnabled, false, "Enables ray portal support. Note this requires portal texture hashes to be set for the ray portal geometries in rtx.rayPortalModelTextureHashes.");
  };
}  // namespace dxvk
