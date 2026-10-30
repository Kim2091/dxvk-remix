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

// Translator toolkit: how long an object a frontend stopped drawing stays alive. Shared by the draw-call
// tracker (replacement instances) and the scene's transient lane (DrawInstance), so both age objects out the same way.

#include "../rtx_camera.h"
#include "../rtx_intersection_test.h"
#include "../rtx_options.h"
#include "../rtx_types.h"

namespace dxvk {
  namespace translator {

    // What the policy needs to know about an object that was not drawn this frame.
    struct UnseenObject {
      uint32_t frameLastSeen = 0;
      uint32_t frameCreated = 0;
      uint32_t categories = 0;          // raw InstanceCategories
      bool skinned = false;
      bool wasMovingWhenLastSeen = false;
      const AxisAlignedBoundingBox* geometryBounds = nullptr;  // object space, or null / invalid
      const AxisAlignedBoundingBox* lightBounds = nullptr;     // object space, or null / invalid
      const Matrix4* objectToWorld = nullptr;
    };

    // Per-collection inputs, read once.
    struct LifetimeFrame {
      RtCamera* camera = nullptr;
      uint32_t currentFrame = 0;
      uint32_t framesToKeepObjects = 0;
      uint32_t framesToKeepLights = 0;
      bool objectAntiCulling = false;
      bool lightAntiCulling = false;
      bool cameraCut = false;
      bool forceCollect = false;        // too many objects kept: anti-culling off for this collection
      bool antiCullingSupported = false;
      // The last camera cut that kept the scene (SceneManager::m_lastKeptCutFrame; ~0u = none). Set by the caller.
      uint32_t keptCutFrame = ~0u;

      // Last drawn before a camera cut that kept the scene: the view it belonged to is gone (a level change, a cutscene
      // camera), so nothing may keep it alive, as the clear the cut used to make would not have.
      bool seenBeforeKeptCut(uint32_t frameLastSeen) const {
        return keptCutFrame != ~0u && frameLastSeen < keptCutFrame;
      }

      static LifetimeFrame read(RtCamera& camera, bool antiCullingSupported, bool forceCollect) {
        LifetimeFrame frame;
        frame.camera = &camera;
        frame.framesToKeepObjects = RtxOptions::numFramesToKeepInstances();
        frame.framesToKeepLights = RtxOptions::AntiCulling::Light::numFramesToExtendLightLifetime();
        frame.objectAntiCulling = RtxOptions::AntiCulling::isObjectAntiCullingEnabled();
        frame.lightAntiCulling = RtxOptions::AntiCulling::isLightAntiCullingEnabled();
        frame.cameraCut = camera.isCameraCut();
        frame.forceCollect = forceCollect;
        frame.antiCullingSupported = antiCullingSupported;
        return frame;
      }
    };

    class LifetimePolicy {
    public:
      // Frustum check for an AABB in the space defined by objectToWorld.
      static bool aabbIntersectsFrustum(RtCamera& camera, const AxisAlignedBoundingBox& aabb, const Matrix4& objectToWorld) {
        const Matrix4 objectToView = camera.getWorldToView(false) * objectToWorld;
        if (RtxOptions::AntiCulling::Object::enableHighPrecisionAntiCulling()) {
          return boundingBoxIntersectsFrustumSAT(
            camera, aabb.minPos, aabb.maxPos, objectToView,
            RtxOptions::AntiCulling::Object::enableInfinityFarFrustum());
        }
        return boundingBoxIntersectsFrustum(camera.getFrustum(), aabb.minPos, aabb.maxPos, objectToView);
      }

      static bool isStale(const UnseenObject& object, const LifetimeFrame& frame) {
        return object.frameLastSeen + frame.framesToKeepObjects <= frame.currentFrame;
      }

      // For a stale object: true = keep it (anti-culling), false = collect it.
      static bool keepUnseen(const UnseenObject& object, const LifetimeFrame& frame) {
        const bool hasLights = object.lightBounds != nullptr && object.lightBounds->isValid();
        const bool hasMeshes = object.geometryBounds != nullptr && object.geometryBounds->isValid();

        // Only anti-cull objects that have been matched at least once after creation.
        const bool isStable = object.frameLastSeen > object.frameCreated;
        if (frame.cameraCut || !isStable || frame.seenBeforeKeptCut(object.frameLastSeen)) {
          return false;
        }

        // Skinned and player model objects are always collected when stale (anti-culling them freezes poses), as are
        // objects tagged IgnoreAntiCulling and objects that were moving when last seen (they would freeze mid-motion).
        const CategoryFlags categories(object.categories);
        const bool exempt =
          object.skinned ||
          categories.test(InstanceCategories::ThirdPersonPlayerModel) ||
          categories.test(InstanceCategories::IgnoreAntiCulling) ||
          object.wasMovingWhenLastSeen;
        if (exempt || frame.forceCollect || !frame.antiCullingSupported) {
          return false;
        }

        // Object anti-culling: keep objects outside the camera frustum (the game's own culling may differ from ours);
        // an object clearly in view that the game stopped drawing had a reason (destruction, LOD swap).
        if (frame.objectAntiCulling && hasMeshes &&
            !aabbIntersectsFrustum(*frame.camera, *object.geometryBounds, *object.objectToWorld)) {
          return true;
        }

        // Light anti-culling for mesh replacement lights: outside the frustum, within the extended lifetime.
        if (frame.lightAntiCulling && hasLights &&
            object.frameLastSeen + frame.framesToKeepLights > frame.currentFrame) {
          return !aabbIntersectsFrustum(*frame.camera, *object.lightBounds, *object.objectToWorld);
        }
        return false;
      }
    };

  } // namespace translator
} // namespace dxvk
