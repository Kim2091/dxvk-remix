/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>

#include "rtx_fork_hooks.h"
#include "rtx_light_manager.h"
#include "rtx_scene_interface.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_utils.h"

#include "../d3d9/d3d9_state.h"
#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/raytrace_args.h"
#include "math.h"
#include "rtx_lights.h"
#include "rtx_intersection_test.h"
#include "rtx_sparse_rendering.h"

/*  Light Manager (blurb)
* 
* There are 3 main types of lights we are heuristically detecting:
* 
*   A) Long-lived, static lights
*     - Typically used for scene/level lighting
*     - Many used to light up a level
*     - Many are "fill" lighting, to show/guide the player
* 
*   B) Short-lived, static lights
*     - Typically, FX lights
*     - e.g. gun muzzle flash
* 
*   C) Dynamic lights
*     - Lights that are moving with some object
*     - e.g. car lights, flash light
* 
*   For all lights, hardware limitations in the fixed-function era meant only a  handful of 
*   these lights can be enabled at any given time (max was 8 lights total back in the day).
*   
*   Many games will try to optimize and prioritize for these 8 available lights (and in many
*   cases use less for performance reasons).  That means, lights can be observed turning on
*   and off as the player moves through a level, to balance the more important (which in raster
*   means, closest) lights.
* 
*   How do we classify each light?
* 
*     A) Long-lived lights are those which haven't moved for 'getNumFramesToPutLightsToSleep' frames.
* 
*     B) A short-lived light is one which has not been seen for 'numFramesToKeepLights' frames, and before it can be put to sleep.
* 
*     C) Any light which moves, is defined as a dynamic light.
*/

namespace dxvk {

  // Note: This must be done as currently every other light index is valid, so this invalid index is the only one that can be used
  // to represent something such as a new light index.
  static_assert(LIGHT_INDEX_INVALID == kNewLightIdx, "New light index must match invalid light sentinel value");

  LightManager::LightManager(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  LightManager::~LightManager() {
  }

  void LightManager::clear() {
    if (!m_lightDebugUILock.owns_lock()) {
      m_lightDebugUILock.lock();
    }
    m_lights.clear();
    m_legacyIdentityByHash.clear();
    m_legacyHashByIdentity.clear();
    m_effectInPlaceSince.clear();
    m_linearizedLights.clear();
    m_lightDebugUILock.unlock();
  }

  void LightManager::clearFromUIThread() {
    // This needs to wait for `m_lightDebugUILock` to be unlocked, so it doesn't
    // cause crashes.
    std::lock_guard<std::mutex> lock(m_lightUIMutex);
    m_lights.clear();
    m_legacyIdentityByHash.clear();
    m_legacyHashByIdentity.clear();
    m_effectInPlaceSince.clear();
    m_linearizedLights.clear();

    // Note: Fallback light reset here so that changes to its settings will take effect, does not need to be part
    // of usual light clearing logic though.
    m_fallbackLight.reset();
  }

  void LightManager::rememberLegacyIdentity(XXH64_hash_t hash, uint32_t identity) {
    if (identity == 0) {
      return;
    }
    const auto it = m_legacyIdentityByHash.find(hash);
    if (it != m_legacyIdentityByHash.end()) {
      if (it->second == identity) {
        return;
      }
      // The same lamp under another identity now: the old reverse entry is stale.
      const auto rev = m_legacyHashByIdentity.find(it->second);
      if (rev != m_legacyHashByIdentity.end() && rev->second == hash) {
        m_legacyHashByIdentity.erase(rev);
      }
      it->second = identity;
    } else {
      m_legacyIdentityByHash.emplace(hash, identity);
    }
    m_legacyHashByIdentity[identity] = hash;
  }

  void LightManager::forgetLegacyIdentity(XXH64_hash_t hash) {
    const auto it = m_legacyIdentityByHash.find(hash);
    if (it == m_legacyIdentityByHash.end()) {
      return;
    }
    const auto rev = m_legacyHashByIdentity.find(it->second);
    if (rev != m_legacyHashByIdentity.end() && rev->second == hash) {
      m_legacyHashByIdentity.erase(rev);
    }
    m_legacyIdentityByHash.erase(it);
  }

  bool LightManager::isEffectLightInPlace(XXH64_hash_t hash, const RtLight& light) const {
    const auto since = m_effectInPlaceSince.find(hash);
    return since != m_effectInPlaceSince.end() &&
           light.getFrameLastTouched() - since->second + 1 >= RtxOptions::getNumFramesToPutLightsToSleep();
  }

  LightManager::EffectVerdict LightManager::judgeEffectLight(XXH64_hash_t hash, const RtLight& light, bool cameraCut) const {
    if (cameraCut) {
      return EffectVerdict::Cut;
    }
    if (!isEffectLightInPlace(hash, light)) {
      return EffectVerdict::Brief;
    }
    if (light.getIsInsideFrustum()) {
      return EffectVerdict::InView;
    }
    return EffectVerdict::Keep;
  }

  void LightManager::garbageCollectionInternal(bool cameraCut) {
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    const uint32_t framesToKeep = RtxOptions::numFramesToKeepLights();
    const bool gameLightListSet = m_gameLightListFrame == currentFrame;
    const bool keepEffectLights = RtxOptions::AntiCulling::isLightAntiCullingEnabled() && !suppressLightKeeping();

    for (auto it = m_lights.begin(); it != m_lights.end();) {
      const RtLight& light = it->second;
      if (light.isMarkedForGarbageCollection()) {
        forgetLegacyIdentity(it->first);
        it = m_lights.erase(it);
        continue;
      }
      const uint32_t frameLastTouched = light.getFrameLastTouched();

      // A host that names its lights sends every light it still has in each list, so a named light missing from this
      // frame's list was switched off or removed. It goes now, without the anti-culling and keep-frame grace periods.
      if (gameLightListSet && frameLastTouched != currentFrame && m_legacyIdentityByHash.count(it->first) != 0) {
        forgetLegacyIdentity(it->first);
        it = m_lights.erase(it);
        ++m_identityLightsDropped;
        continue;
      }

      // An effect light lives while the game draws its mesh, and the game stops drawing a mesh it culls.
      // One the game drew in place for a while (a bulb, a sign) is kept while its mesh is out of view, with no frame limit.
      // A brief or moving one (a muzzle flash), one whose mesh is in view, and every one at a camera cut goes now.
      if (keepEffectLights && frameLastTouched != currentFrame &&
          light.getLightAntiCullingType() == RtLightAntiCullingType::MeshReplacement) {
        const EffectVerdict verdict = judgeEffectLight(it->first, light, cameraCut);
        if (verdict != EffectVerdict::Keep) {
          ++m_effectLightsDropped[static_cast<uint32_t>(verdict)];
          forgetLegacyIdentity(it->first);
          it = m_lights.erase(it);
          continue;
        }
        ++m_keptUnsent[static_cast<uint32_t>(LightKind::Effect)];
        ++it;
        continue;
      }

      if (!RtxOptions::AntiCulling::isLightAntiCullingEnabled() || // It's always True if anti-culling is disabled
          (light.getIsInsideFrustum() ||
           frameLastTouched + RtxOptions::AntiCulling::Light::numFramesToExtendLightLifetime() <= currentFrame)) {
        if (light.isDynamic || suppressLightKeeping()) {
          if (light.getFrameLastTouched() < currentFrame) {
            forgetLegacyIdentity(it->first);
            it = m_lights.erase(it);
            continue;
          }
        } else if ((frameLastTouched + framesToKeep) <= currentFrame) {
          // A static light the game stopped sending ages out whether it slept or not.
          forgetLegacyIdentity(it->first);
          it = m_lights.erase(it);
          continue;
        }
      }
      if (frameLastTouched != currentFrame) {
        ++m_keptUnsent[static_cast<uint32_t>(lightKind(it->first, light))];
      }
      ++it;
    }

    // Lights leave the table in several places. Their in-place records go here.
    for (auto it = m_effectInPlaceSince.begin(); it != m_effectInPlaceSince.end();) {
      if (m_lights.count(it->first) == 0) {
        it = m_effectInPlaceSince.erase(it);
      } else {
        ++it;
      }
    }

    for (auto it = m_externallyTrackedLights.begin(); it != m_externallyTrackedLights.end();) {
      RtLight& light = it->second;
      if (light.isMarkedForGarbageCollection()) {
        it = m_externallyTrackedLights.erase(it);
      } else {
        ++it;
      }
    }
  }

  void LightManager::garbageCollection(RtCamera& camera) {
    if (!m_lightDebugUILock.owns_lock()) {
      m_lightDebugUILock.lock();
    }
    if (RtxOptions::AntiCulling::isLightAntiCullingEnabled()) {
      cFrustum& cameraLightAntiCullingFrustum = camera.getLightAntiCullingFrustum();
      for (auto& [lightHash, rtLight] : getLightTable()) {
        bool isLightInsideFrustum = true;

        // We have 3 situations for a light Anti-Culling:
        // 1. Game Light: We only need to check sphere light (directional light will not be culled)
        // 2. Light replaces original light: Same as 1, we only need to care about light that the ORIGINAL type is sphere
        // 3. Light replaces mesh: the light's own sphere, at the mesh's centroid
        switch (rtLight.getLightAntiCullingType()) {
        case RtLightAntiCullingType::GameLight:
          isLightInsideFrustum = sphereIntersectsFrustum(
            cameraLightAntiCullingFrustum, rtLight.getSphereLightReplacementOriginalPosition(), rtLight.getSphereLightReplacementOriginalRadius());
          break;
        case RtLightAntiCullingType::MeshReplacement:
          // In view only when its mesh surely is: a centroid in view means the mesh is, and one just outside may still have
          // a part the game drew. The light frustum is used because the object anti-culling one is only set up when object
          // anti-culling is on.
          isLightInsideFrustum = sphereIntersectsFrustum(cameraLightAntiCullingFrustum, rtLight.getPosition(),
                                                         rtLight.getType() == RtLightType::Sphere ? rtLight.getSphereLight().getRadius() : 0.f);
          break;
        case RtLightAntiCullingType::Ignore:
          break;
        }

        if (isLightInsideFrustum) {
          rtLight.markAsInsideFrustum();
        } else {
          rtLight.markAsOutsideFrustum();
        }
      }
    }

    const bool cameraCut = camera.isCameraCut();
    logLegacyLightsNearCamera(camera, cameraCut);
    garbageCollectionInternal(cameraCut);
    {
      std::lock_guard<std::mutex> snapshotLock(m_debugSnapshotMutex);
      m_debugSnapshot.clear();
      if (m_debugSnapshotWanted.load(std::memory_order_relaxed)) {
        m_debugSnapshot.reserve(m_lights.size() + m_externallyTrackedLights.size());
        for (const auto& [hash, light] : m_lights) {
          m_debugSnapshot.push_back(light);
        }
        for (const auto& [hash, light] : m_externallyTrackedLights) {
          m_debugSnapshot.push_back(light);
        }
      }
    }
    m_lightDebugUILock.unlock();
  }

  LightManager::LightKind LightManager::lightKind(XXH64_hash_t hash, const RtLight& light) const {
    if (m_legacyIdentityByHash.count(hash) != 0) {
      return LightKind::Named;
    }
    switch (light.getLightAntiCullingType()) {
    case RtLightAntiCullingType::GameLight: return LightKind::Game;
    case RtLightAntiCullingType::MeshReplacement: return LightKind::Effect;
    default: return LightKind::Other;
    }
  }

  // Every 300 frames, the legacy light table by state, the lights kept although they were not sent, and
  // every sphere light within 150 world units of the camera (the Pip-Boy light: where it is, whether it is new this
  // frame or matched to last frame's copy, and which instances' bounds contain it -- a light inside the first-person
  // arms or weapon is occluded by them).
  // Before garbageCollectionInternal, so a light created this frame still shows as NEW (its buffer index is
  // assigned by prepareSceneData later in the frame).
  void LightManager::logLegacyLightsNearCamera(const RtCamera& camera, bool cameraCut) {
    const uint32_t currentFrame = m_device->getCurrentFrameId();
    if (m_lights.empty() || currentFrame - m_nearLightLogFrame < 300) {
      return;
    }
    m_nearLightLogFrame = currentFrame;

    const Vector3 camPos = camera.getPosition();
    const uint32_t framesToSleep = RtxOptions::getNumFramesToPutLightsToSleep();
    uint32_t dynamicCount = 0, sleeping = 0, created = 0, notSent = 0, nearCount = 0, effectCount = 0, effectInPlace = 0;
    std::string nearList;

    const std::vector<RtInstance*>& instances = device()->getCommon()->getSceneManager().getInstanceTable();
    auto instancesContaining = [&instances](const Vector3& p) -> std::string {
      std::string s;
      uint32_t hits = 0;
      for (const RtInstance* inst : instances) {
        if (inst == nullptr || inst->getBlas() == nullptr) {
          continue;
        }
        const AxisAlignedBoundingBox& bb = inst->getBlas()->input.getGeometryData().boundingBox;
        if (!bb.isValid()) {
          continue;
        }
        const Matrix4 objectToWorld = inst->getTransform();
        // Cheap reject: the bound's reach around the instance's origin.
        const Vector3 extent(std::max(std::abs(bb.minPos.x), std::abs(bb.maxPos.x)),
                             std::max(std::abs(bb.minPos.y), std::abs(bb.maxPos.y)),
                             std::max(std::abs(bb.minPos.z), std::abs(bb.maxPos.z)));
        const float scale = std::max(std::max(length(objectToWorld[0].xyz()), length(objectToWorld[1].xyz())),
                                     length(objectToWorld[2].xyz()));
        const float reach = length(extent) * scale;
        if (length(p - objectToWorld[3].xyz()) > reach) {
          continue;
        }
        const Vector4 local = inverse(objectToWorld) * Vector4(p, 1.0f);
        if (!(local.x >= bb.minPos.x && local.x <= bb.maxPos.x &&
              local.y >= bb.minPos.y && local.y <= bb.maxPos.y &&
              local.z >= bb.minPos.z && local.z <= bb.maxPos.z)) {
          continue;
        }
        ++hits;
        if (hits <= 4) {
          s += str::format(hits == 1 ? " insideAABB=[" : ", ", "id=", inst->getId(), " mat=0x", std::hex,
                           inst->getMaterialDataHash(), std::dec, inst->isViewModel() ? " viewModel" : "",
                           inst->isHidden() ? " hidden" : "");
        }
      }
      if (hits > 4) {
        s += str::format(" +", hits - 4, " more");
      }
      if (hits != 0) {
        s += "]";
      }
      return s;
    };

    // Each unsent light with what it is and what keeps it: anti-culling (outside the light frustum), the static keep frames,
    // or, for an effect light drawn in place for a while, being out of view (inPlace: the frames since its run began).
    // A light that garbageCollectionInternal removes this frame is not listed.
    static constexpr const char* kKindNames[] = { "named", "game", "effect", "other" };
    const bool gameLightListSet = m_gameLightListFrame == currentFrame;
    const bool keepEffectLights = RtxOptions::AntiCulling::isLightAntiCullingEnabled() && !suppressLightKeeping();
    const uint32_t extendFrames = RtxOptions::AntiCulling::Light::numFramesToExtendLightLifetime();
    uint32_t keptCount = 0;
    std::string keptList;
    for (const auto& [hash, light] : m_lights) {
      const uint32_t unsentFrames = currentFrame - light.getFrameLastTouched();
      if (unsentFrames == 0) {
        continue;
      }
      const LightKind kind = lightKind(hash, light);
      const char* keptBy = nullptr;
      if (kind == LightKind::Named && gameLightListSet) {
        continue;
      }
      if (keepEffectLights && light.getLightAntiCullingType() == RtLightAntiCullingType::MeshReplacement) {
        if (judgeEffectLight(hash, light, cameraCut) != EffectVerdict::Keep) {
          continue;
        }
        keptBy = "outOfView";
      } else if (RtxOptions::AntiCulling::isLightAntiCullingEnabled() && !light.getIsInsideFrustum() && unsentFrames < extendFrames) {
        keptBy = "antiCulling";
      } else if (!light.isDynamic && !suppressLightKeeping() && unsentFrames < RtxOptions::numFramesToKeepLights()) {
        keptBy = "keepFrames";
      }
      if (keptBy == nullptr) {
        continue;
      }
      ++keptCount;
      if (keptCount > 6) {
        continue;
      }
      const Vector3 pos = light.getPosition();
      const auto since = m_effectInPlaceSince.find(hash);
      keptList += str::format(" | kept ", kKindNames[static_cast<uint32_t>(kind)], " hash=0x", std::hex, hash, std::dec,
                              " by=", keptBy, " unsent=", unsentFrames, "f dynamic=", light.isDynamic ? 1 : 0,
                              since != m_effectInPlaceSince.end() ? str::format(" inPlace=", currentFrame - since->second, "f") : std::string(),
                              " pos=(", pos.x, ", ", pos.y, ", ", pos.z, ") dist=", length(pos - camPos),
                              " radius=", light.getType() == RtLightType::Sphere ? light.getSphereLight().getRadius() : 0.f);
    }

    for (const auto& [hash, light] : m_lights) {
      const bool isNew = light.getBufferIdx() == kNewLightIdx;
      dynamicCount += light.isDynamic ? 1u : 0u;
      sleeping += light.isStaticCount >= framesToSleep ? 1u : 0u;
      created += isNew ? 1u : 0u;
      notSent += light.getFrameLastTouched() != currentFrame ? 1u : 0u;
      effectCount += light.getLightAntiCullingType() == RtLightAntiCullingType::MeshReplacement ? 1u : 0u;
      effectInPlace += isEffectLightInPlace(hash, light) ? 1u : 0u;
      if (light.getType() != RtLightType::Sphere) {
        continue;
      }
      const Vector3 pos = light.getPosition();
      const float dist = length(pos - camPos);
      if (dist > 150.0f) {
        continue;
      }
      ++nearCount;
      if (nearCount > 8) {
        continue;
      }
      const auto id = m_legacyIdentityByHash.find(hash);
      const Vector3 radiance = light.getRadiance();
      nearList += str::format(" | hash=0x", std::hex, hash, " id=0x", id != m_legacyIdentityByHash.end() ? id->second : 0u, std::dec,
                          " pos=(", pos.x, ", ", pos.y, ", ", pos.z, ") dist=", dist,
                          " radius=", light.getSphereLight().getRadius(),
                          " radiance=(", radiance.x, ", ", radiance.y, ", ", radiance.z, ")",
                          isNew ? " NEW" : " matched", " dynamic=", light.isDynamic ? 1 : 0, " static=", light.isStaticCount,
                          " touched=", currentFrame - light.getFrameLastTouched(), "f ago inFrustum=", light.getIsInsideFrustum() ? 1 : 0,
                          " bufferIdx=", light.getBufferIdx(), instancesContaining(pos));
    }

    Logger::info(str::format("[LegacyLights] frame=", currentFrame, " lights=", m_lights.size(), " dynamic=", dynamicCount,
                             " sleeping=", sleeping, " newThisFrame=", created, " notSentThisFrame=", notSent,
                             " identities=", m_legacyIdentityByHash.size(), " effect=", effectCount, " effectInPlace=", effectInPlace,
                             " droppedByGame=", m_identityLightsDropped,
                             " effectDropped=[brief ", m_effectLightsDropped[static_cast<uint32_t>(EffectVerdict::Brief)],
                             " inView ", m_effectLightsDropped[static_cast<uint32_t>(EffectVerdict::InView)],
                             " cut ", m_effectLightsDropped[static_cast<uint32_t>(EffectVerdict::Cut)], "]",
                             " keptUnsent(light-frames)=[named ", m_keptUnsent[0], " game ", m_keptUnsent[1], " effect ", m_keptUnsent[2],
                             " other ", m_keptUnsent[3], "] keptNow=", keptCount,
                             " camera=(", camPos.x, ", ", camPos.y, ", ", camPos.z, ") near150u=", nearCount, nearList, keptList));
    m_identityLightsDropped = 0;
    m_keptUnsent.fill(0);
    m_effectLightsDropped.fill(0);
  }

  void LightManager::dynamicLightMatching() {
    ScopedCpuProfileZone();
    // Try match up any stragglers now we have the full light list this frame.
    for (auto it = m_lights.cbegin(); it != m_lights.cend(); ) {
      const RtLight& light = it->second;
      // Only looking for instances of dynamic lights that have been updated on the previous frame
      if (light.getFrameLastTouched() + 1 != m_device->getCurrentFrameId()) {
        ++it;
        continue;
      }
      // Only interested in updating lights that have been around a while, this implicitly avoids searching for new lights that have been updated.
      if (light.getBufferIdx() == kNewLightIdx) {
        ++it;
        continue;
      }
      // An effect light drawn in place for a while has not moved. Its mesh left the view, and a new light nearby is another one.
      if (isEffectLightInPlace(it->first, light)) {
        ++it;
        continue;
      }

      float currentSimilarity = -1.f;
      // Note: Using an iterator for the found similar light is safe here because the m_lights map will not change between where
      // it is found and where it is accessed.
      std::optional<decltype(m_lights)::iterator> similarLight;
      for (auto similarLightIterator = m_lights.begin(); similarLightIterator != m_lights.end(); ++similarLightIterator) {
        const RtLight& newLight = similarLightIterator->second;
        // Skip comparing to old lights, this check implicitly avoids comparing the exact same light.
        if (newLight.getBufferIdx() != kNewLightIdx)
          continue;

        const float similarity = isSimilar(light, newLight, RtxOptions::uniqueObjectDistance());
        // Update the cached light if it's similar.
        if (similarity > currentSimilarity) {
          similarLight = similarLightIterator;
          currentSimilarity = similarity;
        }
      }

      if (currentSimilarity >= 0 && similarLight.has_value()) {
        // This is a dynamic light!
        RtLight& dynamicLight = (*similarLight)->second;
        dynamicLight.isDynamic = true;

        // This is the same light, so update our new light
        updateLight(light, dynamicLight);

        // Remove the previous frames version
        forgetLegacyIdentity(it->first);
        it = m_lights.erase(it);
      } else {
        ++it;
      }
    }
  }

  void LightManager::prepareSceneData(Rc<DxvkContext> ctx, CameraManager const& cameraManager) {
    ScopedCpuProfileZone();
    // Ensure external light updates/removals are applied before we linearize and draw UI
    fork_hooks::flushPendingLightMutations(*this);
    // Note: Early outing in this function (via returns) should be done carefully (or not at all ideally) as it may skip important
    // logic such as swapping the current/previous frame light buffer, updating light count information or allocating/updating the
    // light buffer which may cause issues in some cases (or rather already has, which is why this warning exists).

    // Create or remove a fallback light depending on if any lights are present in the game and the fallback light mode

    const auto mode = fallbackLightMode();

    const bool noLightsPresent = m_lights.empty() && m_externallyTrackedLights.empty() && m_externalActiveLightList.empty();

    if (
      mode == FallbackLightMode::Always ||
      (mode == FallbackLightMode::NoLightsPresent && noLightsPresent)
    ) {
      auto const& mainCamera = cameraManager.getMainCamera();
      const auto oldFallbackLightPresent = m_fallbackLight.has_value();

      const auto type = fallbackLightType();

      if (type == FallbackLightType::Distant) {
        // Note: Distant light does not need to be dynamic, do not recreate every frame.
        if (!oldFallbackLightPresent || s_fallbackLightDirty) {
          // Create the Distant Fallback Light
          const auto oldDirectionalLightBufferIndex = oldFallbackLightPresent ? m_fallbackLight->getBufferIdx() : 0;
          const auto oldDirectionalLightIdentity = oldFallbackLightPresent ? m_fallbackLight->getStableIdentity() : kInvalidLightIdentity;

          s_fallbackLightDirty = false;
          m_fallbackLight.emplace(RtDistantLight(
            // Note: Distant light direction must be normalized, but a non-normalized direction is provided as an option.
            normalize(fallbackLightDirection()),
            fallbackLightAngle() * kDegreesToRadians / 2.0f,
            fallbackLightRadiance(),
            fallbackLightVolumetricRadianceScale()
          ));

          if (oldFallbackLightPresent) {
            // Note: Carry buffer index over from previous frame if the fallback light was present on the last frame.
            m_fallbackLight->setBufferIdx(oldDirectionalLightBufferIndex);
            m_fallbackLight->setStableIdentity(oldDirectionalLightIdentity);
          }
        }
      } else if (type == FallbackLightType::Sphere) {
        // Create the Sphere Fallback Light

        // sphere light is always recreated every frame, so we can reset the dirty flag
        s_fallbackLightDirty = false;

        const auto oldSphereLightBufferIndex = oldFallbackLightPresent ? m_fallbackLight->getBufferIdx() : 0;
        const auto oldSphereLightIdentity = oldFallbackLightPresent ? m_fallbackLight->getStableIdentity() : kInvalidLightIdentity;

        const auto enableFallback = enableFallbackLightShaping();

        const bool shapingEnabled = enableFallback;
        Vector3 primaryAxis = Vector3(0.0f, 0.0f, 1.0f);
        float cosConeAngle = 0.0f;
        float coneSoftness = 0.0f;
        float focusExponent = 0.0f;

        if (enableFallback) {
          if (enableFallbackLightViewPrimaryAxis()) {
            primaryAxis = mainCamera.getDirection();
          } else {
            // Note: Must normalize the fallback light's primary axis as it is specified by options or ImGui and has
            // no hard requirement to be normalized.
            primaryAxis = safeNormalize(fallbackLightPrimaryAxis(), Vector3(0.0f, 0.0f, 1.0f));
          }

          cosConeAngle = std::cos(fallbackLightConeAngle() * kDegreesToRadians);
          coneSoftness = fallbackLightConeSoftness();
          focusExponent = fallbackLightFocusExponent();
        }

        // Note: Will be recreated every frame due to the need to be dynamic. Not super effecient but this is only
        // a one-off use case for debugging so performance is not super important here.
        m_fallbackLight.emplace(RtSphereLight(
          mainCamera.getPosition() + fallbackLightPositionOffset(),
          fallbackLightRadiance(),
          fallbackLightRadius(),
          RtLightShaping(shapingEnabled, primaryAxis, cosConeAngle, coneSoftness, focusExponent),
          fallbackLightVolumetricRadianceScale()
        ));

        // Update light dynamic properties

        // Note: Sphere fallback lights are dynamic due to following the camera position.
        m_fallbackLight->isDynamic = true;

        if (oldFallbackLightPresent) {
          // Note: Carry buffer index over from previous frame if the fallback light was present on the last frame.
          m_fallbackLight->setBufferIdx(oldSphereLightBufferIndex);
          m_fallbackLight->setStableIdentity(oldSphereLightIdentity);
        }
      }
    } else if (
      (mode == FallbackLightMode::Never) ||
      (mode == FallbackLightMode::NoLightsPresent && !noLightsPresent)
    ) {
      if (m_fallbackLight.has_value()) {
        m_fallbackLight.reset();
      }
    }

    // Light buffer
    const uint32_t previousLightActiveCount = m_currentActiveLightCount;
    m_currentActiveLightCount = 0;

    std::swap(m_lightBuffer, m_previousLightBuffer);
    ++m_lightPrepSerial;

    // Linearize the light list
    // Note: This is done rather than just iterating over the light list twice mostly so that the fallback light
    // can be processed like all other lights without complex logic at the cost of potentially more computational
    // cost, but it might actually work out in favor of performance since unordered map traversal done redundantly
    // may be more expensive than simple vector traversal on the linearized list.
    
    m_linearizedLights.clear();

    if (m_fallbackLight) {
      m_linearizedLights.emplace_back(&*m_fallbackLight);
    }

    for (auto&& pair : m_lights) {
      RtLight& light = pair.second;

      m_linearizedLights.emplace_back(&light);
    }

    for (auto&& pair : m_externallyTrackedLights) {
      m_linearizedLights.emplace_back(&pair.second);
    }

    for (auto& handle : m_externalActiveLightList) {
      auto found = m_externalLights.find(handle);
      if (found != m_externalLights.end()) {
        m_linearizedLights.emplace_back(&found->second);
     }
    }

    // Count the active light of each type

    m_lightTypeRanges.fill(LightRange {});
    for (auto&& linearizedLight : m_linearizedLights) {
      const RtLight& light = *linearizedLight;

      if (light.getColorAndIntensity().w <= 0) {
        continue;
      }

      ++m_lightTypeRanges[static_cast<uint32_t>(light.getType())].count;
      ++m_currentActiveLightCount;

      // Note: Highest light index reserved for the invalid index sentinel.
      if (m_currentActiveLightCount == LIGHT_INDEX_INVALID) {
        ONCE(Logger::info(str::format("[RTX-Compatibility-Info] Raytracing support more than 65535 lights currently, skipping some lights for now.")));
        break;
      }
    }

    // Arrange the ligth ranges of each types sequentially in the buffer, reset the counts

    uint offset = 0;
    for (uint lightType = 0; lightType < lightTypeCount; ++lightType) {
      LightRange& range = m_lightTypeRanges[lightType];
      range.offset = offset;
      offset += range.count;
      range.count = 0;
    }

    assert(offset == m_currentActiveLightCount);
    uint32_t lightsWritten = 0;

    const size_t lightsGPUSize = m_currentActiveLightCount * kLightGPUSize;
    const uint32_t lightMappingBufferEntries = m_currentActiveLightCount + previousLightActiveCount;

    // Resize persistent data buffers
    // Note: std::vector::shrink_to_fit may potentially be useful to call on these buffers in the future if the new desired size is much smaller
    // than the reserved capacity if support for many more lights than 2^16 is desired to allow reclaiming of some memory. For now though this
    // is not an issue and the buffers are allowed to keep whatever capacity they have allocated between calls for the sake of performance.

    // With rtx.scene.explicitLights and the same layout for the last three uploads, every light keeps its slot,
    // so the CPU copy is still valid and only changed slots need writing.
    // The buffer being written was last written two uploads ago, so a slot changed in this or the previous upload is written.
    // Uploads are counted by m_lightPrepSerial, not by frame id, because frames can skip this function.
    bool incrementalLights = false;
    if (scene::Options::explicitLights()) {
      XXH3_state_t* layoutState = XXH3_createState();
      XXH3_64bits_reset(layoutState);
      for (const RtLight* light : m_linearizedLights) {
        struct { const RtLight* light; uint32_t type; uint32_t active; } item { light, uint32_t(light->getType()),
                                                                                light->getColorAndIntensity().w > 0 ? 1u : 0u };
        XXH3_64bits_update(layoutState, &item, sizeof(item));
      }
      const uint64_t layoutHash = XXH3_64bits_digest(layoutState);
      XXH3_freeState(layoutState);
      if (layoutHash == m_lightLayoutHash && m_currentActiveLightCount == m_lightLayoutCount) {
        ++m_lightLayoutStableFrames;
      } else {
        m_lightLayoutStableFrames = 0;
      }
      m_lightLayoutHash = layoutHash;
      m_lightLayoutCount = m_currentActiveLightCount;
      incrementalLights = m_lightLayoutStableFrames >= 2 && m_lightsGPUData.size() == lightsGPUSize &&
        m_lightBuffer != nullptr && m_lightBuffer->info().size >= align(lightsGPUSize, kBufferAlignment);
    } else {
      m_lightLayoutStableFrames = 0;
      m_lightLayoutCount = ~0u;
    }
    m_lightSlotChangedSerial.resize(m_currentActiveLightCount, m_lightPrepSerial);

    m_lightsGPUData.resize(lightsGPUSize);
    if (!incrementalLights) {
      memset(m_lightsGPUData.data(), 0xff, sizeof(char)* m_lightsGPUData.size());
    }
    m_lightMappingData.resize(lightMappingBufferEntries);

    // Clear all slots to new light
    memset(m_lightMappingData.data(), kNewLightIdx, sizeof(uint16_t) * m_lightMappingData.size());

    const bool buildLightIdentityTable =
      SparseRendering::isEnabledByOptions() && !SparseRendering::Options::enableRtxdiReuseForInactivePixels();

    // Section A [0, N) maps current light index to light identity; 
    // section B [N, 2N) is sorted by identity for the GPU's identity->index binary search. 
    // Zero-initialised so any unused slot holds the "no identity" value (0).
    if (buildLightIdentityTable) {
      m_lightIdentityData.assign(static_cast<size_t>(m_currentActiveLightCount) * 2, LightIdentityGpuEntry {});
    } else {
      m_lightIdentityData.clear();
    }

    // Write the light data into the previously allocated ranges
    for (auto&& linearizedLight : m_linearizedLights) {
      RtLight& light = *linearizedLight;

      if (light.getColorAndIntensity().w > 0 && lightsWritten < m_currentActiveLightCount) {
        if (light.getStableIdentity() == kInvalidLightIdentity) {
          light.setStableIdentity(allocateLightIdentity());
        }

        // Find the buffer location for this light
        LightRange& range = m_lightTypeRanges[static_cast<uint32_t>(light.getType())];
        uint32_t newBufferIdx = range.offset + range.count;
        ++range.count;

        // RTXDI needs a mapping from previous light idx to current (to deal with light list reordering)
        if (light.getBufferIdx() != kNewLightIdx)
          m_lightMappingData[m_currentActiveLightCount + light.getBufferIdx()] = (uint16_t)newBufferIdx;

        // Also a mapping from current light idx to previous (for unbiased resampling)
        m_lightMappingData[newBufferIdx] = light.getBufferIdx();

        // Record this light's stable identity indexed by its current buffer position (section A).
        // This lets a reservoir recover the light after renumbering, regardless of its age or
        // changes to the light's transform and other properties.
        if (buildLightIdentityTable) {
          m_lightIdentityData[newBufferIdx].identity = light.getStableIdentity();
          m_lightIdentityData[newBufferIdx].index = newBufferIdx;
        }

        // Prepare data for GPU
        size_t dataOffset = newBufferIdx * kLightGPUSize;
        assert(dataOffset < lightsGPUSize);
        if (incrementalLights) {
          // Padding is not written in release builds, so it is filled the same way as the full path's memset.
          unsigned char scratch[kLightGPUSize];
          std::memset(scratch, 0xff, kLightGPUSize);
          size_t scratchOffset = 0;
          light.writeGPUData(scratch, scratchOffset);
          if (std::memcmp(scratch, m_lightsGPUData.data() + dataOffset, kLightGPUSize) != 0) {
            std::memcpy(m_lightsGPUData.data() + dataOffset, scratch, kLightGPUSize);
            m_lightSlotChangedSerial[newBufferIdx] = m_lightPrepSerial;
          }
        } else {
          light.writeGPUData(m_lightsGPUData.data(), dataOffset);
          if (newBufferIdx < m_lightSlotChangedSerial.size()) {
            m_lightSlotChangedSerial[newBufferIdx] = m_lightPrepSerial;
          }
        }

        // Update the position in buffer for next frame
        light.setBufferIdx(newBufferIdx);

        // Guard against overflowing the light buffer, in case the light counting loop above terminated early
        ++lightsWritten;
      } else {
        // This light is either disabled or didn't fit into the buffer, so set its buffer index to invalid.
        light.setBufferIdx(kNewLightIdx);
      }
    }

    // Build the sorted section (B) of the identity table from section (A) so the GPU can map a
    // reservoir's stored identity back to its current index via binary search.
    if (buildLightIdentityTable && m_currentActiveLightCount > 0) {
      std::copy_n(m_lightIdentityData.begin(), m_currentActiveLightCount,
                  m_lightIdentityData.begin() + m_currentActiveLightCount);
      std::sort(m_lightIdentityData.begin() + m_currentActiveLightCount, m_lightIdentityData.end(),
                [](const LightIdentityGpuEntry& a, const LightIdentityGpuEntry& b) {
                  return a.identity < b.identity;
                });
    }

    // Allocate the light buffer and copy its contents from host to device memory
    DxvkBufferCreateInfo info;
    info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    info.size = align(lightsGPUSize, kBufferAlignment);

    // Note: Only allocating the light buffer here, not the previous light buffer as on the first frame it is fine for it to be null as
    // no previous frame light indices can possibly exist (and thus nothing in the shader should be trying to access it). On the next frame
    // after the light buffer and previous light buffer are swapped, this code will allocate another buffer and the process will continue
    // fine swapping back and forth from that point onwards.
    if (info.size > 0 && (m_lightBuffer == nullptr || info.size > m_lightBuffer->info().size)) {
      m_lightBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Light Buffer");
    }

    info.size = align(lightMappingBufferEntries * sizeof(uint16_t), kBufferAlignment);
    if (info.size > 0 && (m_lightMappingBuffer == nullptr || info.size > m_lightMappingBuffer->info().size)) {
      m_lightMappingBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Light Mapping Buffer");
    }

    info.size = align(m_lightIdentityData.size() * sizeof(LightIdentityGpuEntry), kBufferAlignment);
    if (info.size > 0 && (m_lightIdentityBuffer == nullptr || info.size > m_lightIdentityBuffer->info().size)) {
      m_lightIdentityBuffer = m_device->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Light Identity Buffer");
    }
    
    if (!m_lightsGPUData.empty()) {
      bool written = false;
      if (incrementalLights) {
        // Slots changed in this or the previous upload are written in contiguous runs.
        // Past a few dozen runs, one full write is cheaper.
        std::vector<std::pair<uint32_t, uint32_t>> runs;
        uint32_t slots = 0;
        for (uint32_t i = 0; i < m_currentActiveLightCount && runs.size() <= 64; ++i) {
          if (m_lightPrepSerial - m_lightSlotChangedSerial[i] > 1) {
            continue;
          }
          if (!runs.empty() && runs.back().first + runs.back().second == i) {
            ++runs.back().second;
          } else {
            runs.emplace_back(i, 1u);
          }
          ++slots;
        }
        if (runs.size() <= 64) {
          for (const auto& [first, count] : runs) {
            ctx->writeToBuffer(m_lightBuffer, size_t(first) * kLightGPUSize, size_t(count) * kLightGPUSize,
                               m_lightsGPUData.data() + size_t(first) * kLightGPUSize);
          }
          m_lightSlotsUploaded = slots;
          written = true;
        }
      }
      if (!written) {
        ctx->writeToBuffer(m_lightBuffer, 0, m_lightsGPUData.size(), m_lightsGPUData.data());
        m_lightSlotsUploaded = m_currentActiveLightCount;
      }
    } else {
      m_lightSlotsUploaded = 0;
    }

    if (!m_lightMappingData.empty()) {
      ctx->writeToBuffer(m_lightMappingBuffer, 0, m_lightMappingData.size() * sizeof(uint16_t), m_lightMappingData.data());
    }

    if (!m_lightIdentityData.empty()) {
      ctx->writeToBuffer(m_lightIdentityBuffer, 0, m_lightIdentityData.size() * sizeof(LightIdentityGpuEntry), m_lightIdentityData.data());
    }

    // Builds, refits or keeps the light BVH from this frame's buffer indices.
    m_lightBvh.update(ctx, m_device, m_linearizedLights, m_currentActiveLightCount);

    // If there are no lights with >0 intensity, then clear the list...
    if (m_currentActiveLightCount == 0)
      clear();

    // Generate a GPU dome light if necessary
    {
      // reset state
      m_gpuDomeLightArgs.active = false;
      m_gpuDomeLightArgs.radiance = Vector3(0.0f);
      m_gpuDomeLightArgs.textureIndex = BINDING_INDEX_INVALID;
      DomeLight activeDomeLight;
      if (getActiveDomeLight(activeDomeLight)) {
        // Ensures a texture stays in VidMem
        SceneManager& sceneManager = device()->getCommon()->getSceneManager();
        sceneManager.trackTexture(activeDomeLight.texture, m_gpuDomeLightArgs.textureIndex, true, false);

        if (m_gpuDomeLightArgs.textureIndex != BINDING_INDEX_INVALID) {
          m_gpuDomeLightArgs.active = true;
          m_gpuDomeLightArgs.radiance = activeDomeLight.radiance;
          m_gpuDomeLightArgs.worldToLightTransform = activeDomeLight.worldToLight;
        }
      }
    }

    // Reset external active light list.
    m_externalActiveDomeLight = nullptr;
    m_externalActiveLightList.clear();
  }

  static const float kNotSimilar = -1.f;
  float LightManager::isSimilar(const RtLight& a, const RtLight& b, float distanceThreshold) {
    static const float kCosAngleSimilarityThreshold = cos(5.f * kPi / 180.f);

    // Basic similarity check.
    const bool sameType = a.getType() == b.getType();

    if (!sameType)
      return kNotSimilar;

    if (a.getType() == RtLightType::Distant) {
      // Distant lights should be compared against their direction
      const float cosAngle = dot(a.getDirection(), b.getDirection());
      const bool similarDirection = cosAngle >= kCosAngleSimilarityThreshold;

      return similarDirection ? cosAngle : kNotSimilar;
    } else {
      // This is just an epsilon, at which distance should we collapse similar lights into a single light.
      const float distNormalized = length(a.getPosition() - b.getPosition()) / distanceThreshold;
      const bool similarPosition = distNormalized <= 1.f;

      if (a.getType() == RtLightType::Sphere) {
        const RtLightShaping& aShaping = a.getSphereLight().getShaping();
        const RtLightShaping& bShaping = b.getSphereLight().getShaping();
        if (aShaping.getEnabled() != bShaping.getEnabled()) {
          return kNotSimilar;
        }

        if (aShaping.getEnabled() && bShaping.getEnabled()) {
          const float cosAxis = dot(aShaping.getDirection(), bShaping.getDirection());
          if (cosAxis < kCosAngleSimilarityThreshold) {
            return kNotSimilar;
          }
          const float coneAngleDelta = std::abs(aShaping.getCosConeAngle() - bShaping.getCosConeAngle());
          if (coneAngleDelta > 0.01f) {
            return kNotSimilar;
          }
          float coneSoftnessDelta = std::abs(aShaping.getConeSoftness() - bShaping.getConeSoftness());
          if (coneSoftnessDelta > 0.01f) {
            return kNotSimilar;
          }
        }
      }

      return similarPosition ? (1.f - distNormalized) : kNotSimilar;
    }
  }

  void LightManager::updateLight(const RtLight& in, RtLight& out) {
    // This is somewhat of a blank slate currently to allow for future improvement.
    out.isStaticCount = 0; // This light is not static anymore.
    out.setBufferIdx(in.getBufferIdx());  // We remapped this light.
    out.setStableIdentity(in.getStableIdentity());
  }

  uint32_t LightManager::allocateLightIdentity() {
    if (m_nextLightIdentity > std::numeric_limits<uint32_t>::max()) {
      throw DxvkError("Light identity space exhausted");
    }

    return static_cast<uint32_t>(m_nextLightIdentity++);
  }

  RtLight* LightManager::addLight(const RtLight& rtLight, const DrawCallState& drawCallState, const RtLightAntiCullingType antiCullingType) {
    if (drawCallState.getCategoryFlags().test(InstanceCategories::IgnoreLights))
      return nullptr;

    // Mesh->Lights Replacement
    if (antiCullingType == RtLightAntiCullingType::MeshReplacement) {
      rtLight.cacheMeshReplacementAntiCullingProperties(
        drawCallState.getTransformData().objectToWorld, drawCallState.getGeometryData().boundingBox);
    }

    return addLight(rtLight, antiCullingType);
  }

  void LightManager::addGameLight(const D3DLIGHTTYPE type, const RtLight& rtLight, uint32_t legacyIdentity) {
    switch (type) {
    case D3DLIGHT_DIRECTIONAL:
      if (ignoreGameDirectionalLights())
        return;
      break;

    case D3DLIGHT_POINT:
      if (ignoreGamePointLights())
        return;
      break;

    case D3DLIGHT_SPOT:
      if (ignoreGameSpotLights())
        return;
      break;
    default:
      assert(false && "Invalid option passed to addGameLight");
      break;
    }

    if (RtxOptions::AntiCulling::isLightAntiCullingEnabled() && type == D3DLIGHT_POINT) {
      // Cache the sphere light data into replacement properties so we can unify the game light and light replacement into a single case in LightManager::garbageCollection
      rtLight.cacheLightReplacementAntiCullingProperties(rtLight.getSphereLight());

      addLight(rtLight, RtLightAntiCullingType::GameLight, legacyIdentity);
    } else {
      addLight(rtLight, RtLightAntiCullingType::Ignore, legacyIdentity);
    }
  }

  void LightManager::beginGameLightList() {
    m_gameLightListFrame = m_device->getCurrentFrameId();
  }

  RtLight* LightManager::addLight(const RtLight& rtLight, const RtLightAntiCullingType antiCullingType, uint32_t legacyIdentity) {
    if (!m_lightDebugUILock.owns_lock()) {
      // As addLight can actually erase old lights, we need to lock the mutex starting from the first call each frame.
      m_lightDebugUILock.lock();
    }

    // This light is "off". This includes negative valued lights which in D3D games originally would act as subtractive lighting.
    const Vector3 originalRadiance = rtLight.getRadiance();
    if (originalRadiance.x < 0 || originalRadiance.y < 0 || originalRadiance.z < 0
        || (originalRadiance.x <= 0 && originalRadiance.y <= 0 && originalRadiance.z <= 0))
      return nullptr;

    RtLight* result = nullptr;
    rtLight.setLightAntiCullingType(antiCullingType);

    const XXH64_hash_t newHash = rtLight.getTransformedHash();
    const auto& foundLightIt = m_lights.find(newHash);
    if (foundLightIt != m_lights.end()) {
      rememberLegacyIdentity(newHash, legacyIdentity);
      // Ignore changes in the same frame
      if (foundLightIt->second.getFrameLastTouched() != m_device->getCurrentFrameId()) {
        if (!rtLight.isDynamic && !suppressLightKeeping()) {
          // Update the light - its an exact hash match (meaning it's static)
          const uint32_t isStaticCount = foundLightIt->second.isStaticCount;

          // If this light hasnt moved for N frames, put it to sleep.  This is a defeat device to stop games aggressively ramping up/down intensity as lights 
          if (isStaticCount < RtxOptions::getNumFramesToPutLightsToSleep()) {
            uint32_t bufferIdx = foundLightIt->second.getBufferIdx();
            uint32_t stableIdentity = foundLightIt->second.getStableIdentity();
            foundLightIt->second = rtLight;
            foundLightIt->second.setBufferIdx(bufferIdx);
            foundLightIt->second.setStableIdentity(stableIdentity);
          }

          // Still static, so increment our counter.
          foundLightIt->second.isStaticCount = isStaticCount + 1;
        } else {
          uint32_t bufferIdx = foundLightIt->second.getBufferIdx();
          uint32_t stableIdentity = foundLightIt->second.getStableIdentity();
          foundLightIt->second = rtLight;
          foundLightIt->second.setBufferIdx(bufferIdx);
          foundLightIt->second.setStableIdentity(stableIdentity);
        }

        // We saw this light so bump its frame counter.
        foundLightIt->second.setFrameLastTouched(m_device->getCurrentFrameId());
      }
      result = &foundLightIt->second;
      if (antiCullingType == RtLightAntiCullingType::MeshReplacement) {
        m_effectInPlaceSince.try_emplace(newHash, m_device->getCurrentFrameId());
      }
    } else {
      // The host's identity first. The same light at another position (its hash is its position) is
      // last frame's copy moved: carried over (buffer index, RTXDI identity) and marked dynamic, whatever the
      // distance. Without it only the 2 cm similarity below and dynamicLightMatching's nearest-new-light guess
      // (uniqueObjectDistance) tie the frames together.
      std::optional<RtLight> previous;
      bool previousWasDynamic = false;
      std::optional<uint32_t> inPlaceSince;
      if (legacyIdentity != 0) {
        const auto rev = m_legacyHashByIdentity.find(legacyIdentity);
        if (rev != m_legacyHashByIdentity.end()) {
          const XXH64_hash_t previousHash = rev->second;
          const auto previousIt = m_lights.find(previousHash);
          if (previousIt != m_lights.end() &&
              previousIt->second.getType() == rtLight.getType() &&
              previousIt->second.getFrameLastTouched() != m_device->getCurrentFrameId() &&
              previousIt->second.getPrimInstanceOwner().getReplacementInstance() == nullptr) {
            // Copied out and erased before the insert below: an unordered_map insert may rehash.
            previous.emplace(previousIt->second);
            previousWasDynamic = true;
            m_lights.erase(previousIt);
          }
          // Either way the identity names the new copy from now on.
          forgetLegacyIdentity(previousHash);
        }
      }

      if (!previous.has_value()) {
        //  Try find a similar light
        std::optional<decltype(m_lights)::iterator> similarLight;
        float bestSimilarity = kNotSimilar;
        for (auto similarLightIterator = m_lights.begin(); similarLightIterator != m_lights.end(); ++similarLightIterator) {
          const RtLight& light = similarLightIterator->second;
          if (light.getPrimInstanceOwner().getReplacementInstance() != nullptr) {
            // lights that are part of a replacement should not be considered.
            continue;
          }

          // Update the cached light if it's similar.  This should catch minor perturbations in static lights (e.g. due to precision loss)
          const float kDistanceThresholdMeters = 0.02f;
          const float kDistanceThresholdWorldUnits = kDistanceThresholdMeters * RtxOptions::getMeterToWorldUnitScale();
          const float thisLightsSimilarity = isSimilar(light, rtLight, kDistanceThresholdWorldUnits);

          if (thisLightsSimilarity >= 0.f && thisLightsSimilarity > bestSimilarity) {
            // Copy off light state.
            similarLight = similarLightIterator;
            bestSimilarity = thisLightsSimilarity;
          }
        }

        if (similarLight.has_value()) {
          // Copied out and removed before the insert below (the insert may rehash and invalidate the iterator).
          previous.emplace(similarLight.value()->second);
          forgetLegacyIdentity(similarLight.value()->first);
          // Within 2 cm it is still in place, so an effect light's run goes on.
          const auto since = m_effectInPlaceSince.find(similarLight.value()->first);
          if (since != m_effectInPlaceSince.end()) {
            inPlaceSince = since->second;
            m_effectInPlaceSince.erase(since);
          }
          m_lights.erase(similarLight.value());
        }
      }

      // Add as a new light (with/out updated data depending on if a similar light was found)
      const auto& [localLightIterator, addedSuccessfully] = m_lights.try_emplace(newHash, rtLight);
      RtLight& localLight = localLightIterator->second;

      // Note: Ensure that the new light was added successfully (meaning that no existing light existed in the light map at the
      // given Light instance hash). This should always be the case as this code is in the "else" branch of a check to see if the
      // light exists in the map already, meaning a light with this hash should not be present in this case.
      // If this fact ever changes, use insert_or_assign instead of emplace to insert or overwrite the light in the map if that is
      // the desired behavior.
      assert(addedSuccessfully);

      if (previous.has_value()) {
        // Copy/interpolate any state we like from the previous light.
        updateLight(*previous, localLight);
        if (previousWasDynamic) {
          // It moved: collected the frame the host stops sending it (garbageCollectionInternal).
          localLight.isDynamic = true;
        }
      }
      rememberLegacyIdentity(newHash, legacyIdentity);

      // Record we saw this light
      localLight.setFrameLastTouched(m_device->getCurrentFrameId());
      result = &localLight;
      if (antiCullingType == RtLightAntiCullingType::MeshReplacement) {
        m_effectInPlaceSince.try_emplace(newHash, inPlaceSince.value_or(m_device->getCurrentFrameId()));
      }
    }
    return result;
  }

  // Creates a new externally tracked light. These lights have their lifecycle managed by external systems
  // rather than LightManager's frame-to-frame tracking and anti-culling systems.
  RtLight* LightManager::createExternallyTrackedLight(const RtLight& light) {
    if (!m_lightDebugUILock.owns_lock()) {
      m_lightDebugUILock.lock();
    }
    
    auto [newLightIt, addedSuccessfully] = m_externallyTrackedLights.try_emplace(m_nextExternallyTrackedLightId, light);
    assert(addedSuccessfully);
    RtLight* newLight = &newLightIt->second;
    newLight->setExternallyTrackedLightId(m_nextExternallyTrackedLightId);
    m_nextExternallyTrackedLightId++;
    return newLight;
  }

  // Updates an existing externally tracked light with new data. The light's lifecycle is managed by external systems
  // rather than LightManager's frame-to-frame tracking and anti-culling systems.
  void LightManager::updateExternallyTrackedLight(RtLight* light, const RtLight& newLight) {
    if (!m_lightDebugUILock.owns_lock()) {
      m_lightDebugUILock.lock();
    }
    assert(light->getExternallyTrackedLightId() != kInvalidExternallyTrackedLightId && " light passed to updateExternallyTrackedLight is not actually externally tracked.");

    const uint64_t externalId = light->getExternallyTrackedLightId();
    fork_hooks::updateLightStaticSleep(light, newLight, m_device, externalId);
  }

  void LightManager::addExternalLight(remixapi_LightHandle handle, const RtLight& rtlight) {
    auto found = m_externalLights.find(handle);
    if (found != m_externalLights.end() && scene::Options::explicitLights()) {
      // rtx.scene.explicitLights: the host's update is the light's state; no static-light sleep.
      RtLight& light = found->second;
      const uint16_t bufferIdx = light.getBufferIdx();
      const uint32_t stableIdentity = light.getStableIdentity();
      light = rtlight;
      light.setBufferIdx(bufferIdx);
      light.setStableIdentity(stableIdentity);
      light.setFrameLastTouched(m_device->getCurrentFrameId());
      return;
    }
    if (found != m_externalLights.end()) {
      // Existing light - preserve temporal data for static lights
      fork_hooks::updateLightStaticSleep(
        &found->second, rtlight, m_device, kInvalidExternallyTrackedLightId);
    } else {
      // New light - copy it and set initial frame
      fork_hooks::setExternalLightEmplace(*this, handle, rtlight);
    }
  }

  void LightManager::removeExternalLight(remixapi_LightHandle handle) {
    fork_hooks::disableExternalLightQueue(*this, handle);
  }

  bool LightManager::getActiveDomeLight(DomeLight& domeLightOut) {
    if (m_externalActiveDomeLight == nullptr) {
      return false;
    }

    if (m_externalDomeLights.size() == 0) {
      m_externalActiveDomeLight = nullptr;
      return false;
    }

    auto found = m_externalDomeLights.find(m_externalActiveDomeLight);
    if (found == m_externalDomeLights.end()) {
      // Invalid active dome light, reset it
      m_externalActiveDomeLight = nullptr;
      return false;
    }

    domeLightOut = found->second;

    return true;
  }

  const DomeLightArgs& LightManager::getDomeLightArgs() {
    const remixapi_LightHandle activeDomeLightHandle = m_externalActiveDomeLight;
    DomeLight activeDomeLight;
    if (!getActiveDomeLight(activeDomeLight) && activeDomeLightHandle != nullptr) {
      m_gpuDomeLightArgs.active = false;
      m_gpuDomeLightArgs.radiance = Vector3(0.0f);
      m_gpuDomeLightArgs.textureIndex = BINDING_INDEX_INVALID;
    }

    return m_gpuDomeLightArgs;
  }

  void LightManager::addExternalDomeLight(remixapi_LightHandle handle, const DomeLight& domeLight) {
    auto found = m_externalDomeLights.find(handle);
    if (found != m_externalDomeLights.end()) {
      // TODO: warn the user about id collision,
      //       or just overwriting existing one is fine?
      found->second = domeLight;
    } else {
      m_externalDomeLights.emplace(handle, domeLight);
    }
  }

  void LightManager::addExternalLightInstance(remixapi_LightHandle enabledLight) {
    // Queue activation to be applied at frame start
    m_pendingExternalActiveLights.insert(enabledLight);
  }

  void LightManager::registerPersistentExternalLight(remixapi_LightHandle handle) {
    fork_hooks::registerPersistentLight(*this, handle);
  }

  void LightManager::unregisterPersistentExternalLight(remixapi_LightHandle handle) {
    fork_hooks::unregisterPersistentLight(*this, handle);
  }

  void LightManager::queueAutoInstancePersistent() {
    fork_hooks::queueAutoInstancePersistent(*this);
  }

  void LightManager::setRaytraceArgs(RaytraceArgs& raytraceArgs, uint32_t rtxdiInitialLightSamples, uint32_t volumeRISInitialLightSamples, uint32_t risLightSamples) const
  {
    // The algorithm below performs two tasks:
    // 1. Fills raytraceArgs.lightRanges[] with light range offsets and counts;
    // 2. Distributes the RTXDI and RIS samples statically among the light types, proportional to the light counts.
    //
    // The distribution code makes sure that there is at least one sample for each non-empty type, and the rest is
    // approximate, i.e. the sample counts set by the user are just guidelines and the actual total count can be
    // slightly different.

    raytraceArgs.rtxdiTotalSampleCount = 0;
    raytraceArgs.volumeRISTotalSampleCount = 0;
    raytraceArgs.risTotalSampleCount = 0;

    // Calculate the requested amount of samples per active light, but no more than 1
    const float rtxdiSamplesPerLight = std::min(1.f, (float)rtxdiInitialLightSamples / std::max(m_currentActiveLightCount, 1u));
    const float volumeRISSamplesPerLight = std::min(1.f, (float)volumeRISInitialLightSamples / std::max(m_currentActiveLightCount, 1u));
    const float risSamplesPerLight = std::min(1.f, (float)risLightSamples / std::max(m_currentActiveLightCount, 1u));

    // Go over all light types and ranges
    for (uint32_t lightType = 0; lightType < lightTypeCount; ++lightType) {
      const LightRange& srcRange = m_lightTypeRanges[lightType];
      LightRangeInfo& dstRange = raytraceArgs.lightRanges[lightType];

      // Copy the range info
      dstRange.offset = srcRange.offset;
      dstRange.count = srcRange.count;

      // Calculate the actual sample counts for this light type
      // Note: uint16 safe to use here as the total number of samples to take is a uint16 to begin with and thus these values
      // for per-light type sample counts should not be greater.
      dstRange.rtxdiSampleCount = (srcRange.count > 0 && rtxdiSamplesPerLight > 0.f)
        ? std::max((uint16_t)1u, (uint16_t) roundf(rtxdiSamplesPerLight * (float) srcRange.count)) : 0u;
      dstRange.volumeRISSampleCount = (srcRange.count > 0 && volumeRISSamplesPerLight > 0.f)
        ? std::max((uint16_t)1u, (uint16_t) roundf(volumeRISSamplesPerLight * (float) srcRange.count)) : 0u;
      dstRange.risSampleCount = (srcRange.count > 0 && risSamplesPerLight > 0.f)
        ? std::max((uint16_t)1u, (uint16_t) roundf(risSamplesPerLight * (float) srcRange.count)) : 0u;

      // Count the total samples - necessary to compute the correct PDF during sampling (not currently used by RTXDI anymore
      // at least due to changing how it does its sampling, still in use in other cases though).
      raytraceArgs.rtxdiTotalSampleCount += dstRange.rtxdiSampleCount;
      raytraceArgs.volumeRISTotalSampleCount += dstRange.volumeRISSampleCount;
      raytraceArgs.risTotalSampleCount += dstRange.risSampleCount;
    }

    m_lightBvh.setRaytraceArgs(raytraceArgs);
  }

  uint LightManager::getLightCount(uint type) {
    if (type >= lightTypeCount) {
      return 0;
    }
    return m_lightTypeRanges[type].count;
  }

}  // namespace dxvk
