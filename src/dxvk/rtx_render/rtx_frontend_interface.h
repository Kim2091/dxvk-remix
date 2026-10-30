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
#pragma once

// The frontend scene interface: the one door through which frontends and
// the Remix API path hand scene content to the backend. Nothing here names a graphics API, and nothing here may name a
// draw-call record or a frontend type, so a frontend for any API can use it.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "../../util/util_matrix.h"
#include "../../util/util_vector.h"

#include "rtx/pass/particles/particle_system_common.h"

namespace dxvk {
  class RtxContext;
  struct ExternalDrawState;

  namespace frontend {

    using InstanceHandle = uint64_t;
    using MeshHandle = uint64_t;
    using MaterialHandle = uint64_t;

    // The camera an instance is drawn with, and the camera a SetupCamera call sets up.
    enum class CameraView : uint32_t {
      Main,
      ViewModel,
      Sky,
    };

    // Frame event bits. An event never touches the scene unless stated.
    // The values match the REMIXAPI_CAMERA_FRAME_EVENT_* bits of remix_c.h where those exist.
    enum FrameEventBits : uint32_t {
      kFrameEventCameraCut         = 1u << 0,  // temporal history reset
      kFrameEventHistoryInvalid    = 1u << 1,  // full temporal reset
      kFrameEventModsChanged       = 1u << 2,  // the asset resolver re-resolves
      kFrameEventOriginShift       = 1u << 3,  // one translation for the whole scene
      kFrameEventCategoriesChanged = 1u << 4,  // instance categories re-resolve per material
      kFrameEventReKeyAll          = 1u << 5,  // the key-derivation rule changed
    };

    // A fixed-function blend and texture-combiner record for instances drawn with a legacy material.
    // Each field holds the raw value of the Vulkan or Remix enum named next to it, and the backend stores it unchanged.
    struct LegacyBlendDesc {
      uint32_t alphaTestEnabled = 0;
      uint8_t alphaTestReferenceValue = 0;
      uint32_t alphaTestCompareOp = 0;      // VkCompareOp
      uint32_t alphaBlendEnabled = 0;
      uint32_t srcColorBlendFactor = 0;     // VkBlendFactor
      uint32_t dstColorBlendFactor = 0;     // VkBlendFactor
      uint32_t colorBlendOp = 0;            // VkBlendOp
      uint32_t srcAlphaBlendFactor = 0;     // VkBlendFactor
      uint32_t dstAlphaBlendFactor = 0;     // VkBlendFactor
      uint32_t alphaBlendOp = 0;            // VkBlendOp
      uint32_t writeMask = 0;               // VkColorComponentFlags
      uint32_t textureColorArg1Source = 0;  // RtTextureArgSource
      uint32_t textureColorArg2Source = 0;  // RtTextureArgSource
      uint32_t textureColorOperation = 0;   // DxvkRtTextureOperation
      uint32_t textureAlphaArg1Source = 0;  // RtTextureArgSource
      uint32_t textureAlphaArg2Source = 0;  // RtTextureArgSource
      uint32_t textureAlphaOperation = 0;   // DxvkRtTextureOperation
      uint32_t tFactor = 0;
      uint32_t isTextureFactorBlend = 0;
      bool isVertexColorBakedLighting = false;
    };

    // What a frontend states about one instance of a mesh.
    struct SceneInstanceDesc {
      MeshHandle mesh = 0;
      // Non-zero: every submesh is shaded with this material instead of its own.
      MaterialHandle materialOverride = 0;
      Matrix4 objectToWorld;
      CameraView cameraView = CameraView::Main;
      // Bits of the backend's InstanceCategories, one bit per category index.
      uint32_t categories = 0;
      bool doubleSided = false;
      std::optional<uint32_t> objectPickingValue;
      std::optional<std::vector<Matrix4>> boneTransforms;
      std::optional<LegacyBlendDesc> blend;
      std::optional<RtxParticleSystemDesc> particles;
      // Per-copy transforms of a GPU-instanced draw, relative to objectToWorld.
      std::vector<Matrix4> gpuInstancing;
    };

    // A description the backend has already converted into its own record. It is made on the calling thread, so the
    // render thread only stores it. Frontends move it into a command and never look inside.
    class PreparedInstance {
    public:
      PreparedInstance();
      PreparedInstance(PreparedInstance&& other) noexcept;
      PreparedInstance& operator=(PreparedInstance&& other) noexcept;
      PreparedInstance(const PreparedInstance&) = delete;
      PreparedInstance& operator=(const PreparedInstance&) = delete;
      ~PreparedInstance();

      explicit operator bool() const { return m_state != nullptr; }

    private:
      friend class SceneInterface;
      friend PreparedInstance prepareInstance(SceneInstanceDesc&& desc);

      std::unique_ptr<ExternalDrawState> m_state;
    };

    // Converts a description into the backend's record. Any thread.
    PreparedInstance prepareInstance(SceneInstanceDesc&& desc);

    struct CameraSetup {
      CameraView view = CameraView::Main;
      Matrix4 worldToView;
      Matrix4 viewToProjection;
      // worldToView in double precision, when the frontend has it. Used while rtx.worldOrigin.preciseCamera is on.
      std::optional<Matrix4d> preciseWorldToView;
      // The frontend's render origin for this camera (render position + origin = world position), when it publishes one.
      std::optional<Vector3> worldOrigin;
      // FrameEventBits. The camera cut and history events act on the Main camera only.
      uint32_t events = 0;
      // The translation of a kFrameEventOriginShift event.
      Vector3 originShift { 0.f, 0.f, 0.f };
    };

    struct FrameInfo {
      // Retained transforms are absolute. The backend subtracts this origin, so the frontend keeps stable coordinates
      // while the scene stays near the origin. The frontend rebases its cameras and lights by the same amount.
      Vector3 sceneOrigin { 0.f, 0.f, 0.f };
    };

    struct CameraQuery {
      bool valid = false;  // the camera was set up this frame
      Matrix4 worldToView;
      Matrix4 viewToProjection;
    };

    // The render-thread half of the interface: commands and queries, applied in call order. Callers on other threads
    // carry their commands to the render thread first, as a command-stream lambda or through the scene command queue.
    class SceneInterface {
    public:
      explicit SceneInterface(RtxContext& ctx) : m_ctx(ctx) { }

      // An instance drawn this frame only.
      void drawInstance(PreparedInstance&& instance);

      // Retained instances live from their first describe to destroyInstance. A describe of a live handle replaces its
      // description and keeps its visibility.
      void describeInstance(InstanceHandle handle, PreparedInstance&& instance);
      void setTransform(InstanceHandle handle, const Matrix4& objectToWorld);
      // A hidden instance stays registered, so a frontend's own culling can drop it for a frame without a new describe.
      void setVisible(InstanceHandle handle, bool visible);
      void destroyInstance(InstanceHandle handle);
      // Draws every retained instance for this frame.
      void drawRetainedInstances(const FrameInfo& frame);

      void setupCamera(const CameraSetup& camera);

      CameraQuery camera(CameraView view) const;
      uint32_t frameIndex() const;

    private:
      RtxContext& m_ctx;
    };

  } // namespace frontend
} // namespace dxvk
