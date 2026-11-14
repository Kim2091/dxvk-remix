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
#include "rtx_lens.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_asset_data_manager.h"
#include "rtx_auto_exposure.h"
#include "rtx_camera.h"
#include "rtx_context.h"
#include "rtx_imgui.h"
#include "rtx_options.h"
#include "rtx_texture_manager.h"
#include "../util/util_global_time.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx/pass/raytrace_args.h"

#include <rtx_shaders/lens_overlay.h>
#include <rtx_shaders/lens_peripheral_blur.h>

namespace dxvk {

  namespace {
    class LensOverlayShader : public ManagedShader {
      SHADER_SOURCE(LensOverlayShader, VK_SHADER_STAGE_COMPUTE_BIT, lens_overlay)

      PUSH_CONSTANTS(LensOverlayArgs)

      BEGIN_PARAMETER()
        CONSTANT_BUFFER(LENS_OVERLAY_BINDING_RAYTRACE_ARGS)
        RW_TEXTURE2D(LENS_OVERLAY_BINDING_COLOR_INPUT_OUTPUT)
        RW_TEXTURE2D_READONLY(LENS_OVERLAY_BINDING_COLOR_COPY_INPUT)
        RW_TEXTURE2D_READONLY(LENS_OVERLAY_BINDING_FEATURES_INPUT)
        RW_TEXTURE1D_READONLY(LENS_OVERLAY_BINDING_EXPOSURE_INPUT)
        RW_STRUCTURED_BUFFER(LENS_OVERLAY_BINDING_NIGHT_STATS)
        TEXTURE2DARRAY(LENS_OVERLAY_BINDING_BLUE_NOISE)
        RW_TEXTURE2D_READONLY(LENS_OVERLAY_BINDING_PERIPHERAL_BLUR_INPUT)
        RW_STRUCTURED_BUFFER(LENS_OVERLAY_BINDING_RANGEFINDER)
        RW_TEXTURE2D_READONLY(LENS_OVERLAY_BINDING_HIT_DISTANCE_INPUT)
        SAMPLER2D(LENS_OVERLAY_BINDING_RETICLE_TEXTURE_0)
        SAMPLER2D(LENS_OVERLAY_BINDING_RETICLE_TEXTURE_1)
        SAMPLER2D(LENS_OVERLAY_BINDING_RETICLE_TEXTURE_2)
        SAMPLER2D(LENS_OVERLAY_BINDING_RETICLE_TEXTURE_3)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(LensOverlayShader);

    class LensPeripheralBlurShader : public ManagedShader {
      SHADER_SOURCE(LensPeripheralBlurShader, VK_SHADER_STAGE_COMPUTE_BIT, lens_peripheral_blur)

      PUSH_CONSTANTS(LensBlurArgs)

      BEGIN_PARAMETER()
        RW_TEXTURE2D_READONLY(LENS_BLUR_BINDING_INPUT)
        RW_TEXTURE2D(LENS_BLUR_BINDING_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(LensPeripheralBlurShader);

    // The peripheral blur's standard deviation at rtx.lens.peripheralBlur 1, in half resolution pixels.
    constexpr float kPeripheralBlurMaxSigma = 6.f;

    constexpr uint64_t kDebugCameraLensId = 1;

    // The weight of the sun term in thermal lens images.
    constexpr float kThermalSunWeight = 0.15f;
    // The reticle of a reflex lens that has no reticle color and radiance, in displayed units.
    const Vector3 kReflexReticleColor { 1.0f, 0.08f, 0.04f };
    constexpr float kReflexReticleRadiance = 2.0f;

    // Returns the lens' arguments with its pose replaced, keeping its optics.
    LensArgs replacePose(const LensArgs& args, const LensPose& pose) {
      const Vector3 oldEyepieceCenter { args.eyepieceCenter.x, args.eyepieceCenter.y, args.eyepieceCenter.z };
      const Vector3 oldObjectiveCenter { args.objectiveCenter.x, args.objectiveCenter.y, args.objectiveCenter.z };
      const Vector3 oldAxis { args.axis.x, args.axis.y, args.axis.z };
      const float objectiveDistance = dot(oldObjectiveCenter - oldEyepieceCenter, oldAxis);

      const Vector3 axis = safeNormalize(pose.axis, Vector3(0.f, 0.f, 1.f));
      const Vector3 up = safeNormalize(pose.up - axis * dot(pose.up, axis), std::abs(axis.y) < 0.9f ? Vector3(0.f, 1.f, 0.f) : Vector3(1.f, 0.f, 0.f));
      const Vector3 right = safeNormalize(cross(axis, up), Vector3(1.f, 0.f, 0.f));

      LensArgs result = args;
      result.eyepieceCenter = pose.eyepieceCenter;
      result.axis = axis;
      result.up = up;
      result.right = right;
      result.objectiveCenter = pose.eyepieceCenter + axis * objectiveDistance;
      return result;
    }
  }

  LensManager::TextureHashLookup LensManager::s_textureHashLookup = nullptr;

  LensManager::LensManager(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  void LensManager::addLens(uint64_t id, const LensDesc& lens, const LensPose* previousPose) {
    assert(id != 0);

    // A lens with a non-finite or degenerate pose or size would put NaNs into every pass that reads it.
    auto finite = [](const Vector3& v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };
    auto validPose = [&](const LensPose& pose) {
      return finite(pose.eyepieceCenter) && finite(pose.axis) && finite(pose.up) && lengthSqr(pose.axis) > 1e-12f;
    };
    if (!validPose(lens.pose) || (previousPose != nullptr && !validPose(*previousPose)) ||
        !std::isfinite(lens.eyepieceRadius) || lens.eyepieceRadius <= 0.f ||
        !std::isfinite(lens.objectiveDistance) || !std::isfinite(lens.objectiveRadius) || !std::isfinite(lens.eyeRelief)) {
      ONCE(Logger::warn("[Lens] A lens with a non-finite or degenerate pose or size was dropped."));
      return;
    }

    PendingLens pending;
    pending.id = id;
    pending.lens = lens;
    if (id != kDebugCameraLensId) {
      pending.lens.magnification *= magnificationScale();
    }
    pending.hasPreviousPose = previousPose != nullptr;
    pending.previousPose = previousPose != nullptr ? *previousPose : lens.pose;

    std::lock_guard lock { m_pendingMutex };
    m_pendingLenses.push_back(pending);
  }

  LensArgs LensManager::createShaderArgs(const LensDesc& lens) const {
    const float worldUnitsPerCm = RtxOptions::getMeterToWorldUnitScale() / 100.f;
    const float magnification = std::max(lens.magnification, 0.01f);
    const float fieldStopDegrees = std::clamp(lens.fieldStopDegrees, 0.1f, 89.f);

    LensArgs args {};
    args.eyepieceRadius = std::max(lens.eyepieceRadius, 0.f);
    args.magnification = magnification;
    args.tanFieldStop = std::tan(fieldStopDegrees * kDegreesToRadians);
    args.eyeRelief = std::max(lens.eyeRelief, 0.f);
    args.exitPupilRadius = std::max(lens.objectiveRadius, 0.f) / magnification;
    args.tint = lens.tint;
    args.eyePupilRadius = eyePupilMm() * 0.1f * worldUnitsPerCm;
    args.reticleColor = lens.reticleColor;
    args.reticleRadiance = std::max(lens.reticleRadiance, 0.f);
    args.reticleStyle = static_cast<uint32_t>(lens.reticle);
    args.reticleLineWidth = std::max(lens.reticleLineWidth, 0.f);
    args.vignetteStrength = std::clamp(vignetteStrength(), 0.f, 1.f);
    args.flags = 0;
    args.kind = static_cast<uint32_t>(lens.kind);
    args.aperture = static_cast<uint32_t>(lens.aperture);
    args.apertureAspect = lens.apertureAspect > 0.f ? lens.apertureAspect : 1.f;
    args.visionMode = static_cast<uint32_t>(lens.visionMode);
    args.sunDirection = Vector3(0.f, 0.f, 1.f);
    args.sunWeight = 0.f;

    // A first focal plane reticle has its size at the reference magnification and grows with the magnification.
    args.reticleScale = lens.firstFocalPlane && lens.referenceMagnification > 0.f
      ? magnification / lens.referenceMagnification
      : 1.f;
    args.parallaxDistance = std::max(lens.parallaxDistance, 0.f);
    args.glareStrength = std::clamp(lens.glareStrength, 0.f, 1.f);
    args.dirtStrength = std::clamp(lens.dirtStrength, 0.f, 1.f);
    args.glareSun = 0.f;
    args.markingColor = lens.markingColor;
    args.markingRadiance = std::max(lens.markingRadiance, 0.f);
    args.markingOpacity = std::clamp(lens.markingOpacity, 0.f, 1.f);
    args.reticleAccentColor = lens.reticleAccentColor;
    args.reticleAccentRadiance = std::max(lens.reticleAccentRadiance, 0.f);
    args.edgeGlowColor = lens.edgeGlowColor;
    args.edgeGlowStrength = std::max(lens.edgeGlowStrength, 0.f);
    args.edgeTint = lens.edgeTint;
    args.edgeShade = std::clamp(lens.edgeShade, 0.f, 1.f);
    const bool hasNightTint = lens.nightTint.x > 0.f || lens.nightTint.y > 0.f || lens.nightTint.z > 0.f;
    args.nightTint = hasNightTint ? lens.nightTint : NightVision::phosphorColor();
    args.nightGain = lens.nightGain > 0.f ? lens.nightGain : 1.f;

    if (lens.reticleLabels) {
      args.flags |= lensFlagReticleLabels;
    }

    if (lens.rangefinder) {
      args.flags |= lensFlagRangefinder;
    }

    if (lens.occludeOutside) {
      args.flags |= lensFlagOccludeOutside;
    }

    if (lens.noGrain) {
      args.flags |= lensFlagNoGrain;
    }

    if (lens.highlightHostiles || highlightOnAllLenses()) {
      args.flags |= lensFlagHighlight;
    }

    if (lens.kind == LensKind::Reflex) {
      // A reflex lens without an objective distance leaves camera rays unchanged.
      if (lens.objectiveDistance <= 0.f) {
        args.flags |= lensFlagPassThrough;
      }

      if (lens.reticleRadiance <= 0.f && lens.reticleColor.x <= 0.f && lens.reticleColor.y <= 0.f && lens.reticleColor.z <= 0.f) {
        args.reticleColor = kReflexReticleColor;
        args.reticleRadiance = kReflexReticleRadiance;
      }
    }

    // replacePose reads the objective distance back from the objective center along the axis.
    args.eyepieceCenter = Vector3(0.f, 0.f, 0.f);
    args.axis = Vector3(0.f, 0.f, 1.f);
    args.objectiveCenter = Vector3(0.f, 0.f, std::max(lens.objectiveDistance, 0.f));

    return replacePose(args, lens.pose);
  }

  void LensManager::addDebugCameraLens(const RtCamera& camera) {
    const float worldUnitsPerCm = RtxOptions::getMeterToWorldUnitScale() / 100.f;
    const Vector3 eyeOffset = DebugLens::eyeOffsetCm() * worldUnitsPerCm;
    const float eyeRelief = DebugLens::eyeReliefCm() * worldUnitsPerCm;

    // The eyepiece is in front of the eye, so the eye sits at the ideal eye point plus the offset.
    auto createPose = [&](const Matrix4d& viewToWorld) {
      const Vector3 position { viewToWorld[3].xyz() };
      const Vector3 right = safeNormalize(Vector3 { viewToWorld[0].xyz() }, Vector3(1.f, 0.f, 0.f));
      const Vector3 up = safeNormalize(Vector3 { viewToWorld[1].xyz() }, Vector3(0.f, 1.f, 0.f));
      const Vector3 viewZ = safeNormalize(Vector3 { viewToWorld[2].xyz() }, Vector3(0.f, 0.f, 1.f));
      const Vector3 forward = camera.isLHS() ? viewZ : -viewZ;

      LensPose pose;
      pose.eyepieceCenter = position - right * eyeOffset.x - up * eyeOffset.y + forward * (eyeRelief + eyeOffset.z);
      pose.axis = forward;
      pose.up = up;
      return pose;
    };

    // The eyepiece radius keeps the lens at the same size on screen whatever the eye distance.
    const float tanHalfFovY = 1.f / std::max(static_cast<float>(std::abs(camera.getViewToProjection()[1][1])), 1e-6f);

    LensDesc lens;
    lens.pose = createPose(camera.getViewToWorld(true));
    lens.eyepieceRadius = std::max(eyeRelief + eyeOffset.z, 0.f) * tanHalfFovY * DebugLens::screenRadius();
    lens.objectiveDistance = DebugLens::objectiveDistanceCm() * worldUnitsPerCm;
    lens.objectiveRadius = DebugLens::objectiveRadiusCm() * worldUnitsPerCm;
    lens.magnification = DebugLens::magnification();
    lens.fieldStopDegrees = DebugLens::fieldStopDegrees();
    lens.eyeRelief = eyeRelief;
    lens.tint = DebugLens::tint();
    lens.reticle = DebugLens::reticle();
    lens.reticleColor = DebugLens::reticleColor();
    lens.reticleRadiance = DebugLens::reticleRadiance();
    lens.reticleLineWidth = DebugLens::reticleLineWidth();
    lens.kind = DebugLens::kind();
    lens.aperture = DebugLens::aperture();
    lens.apertureAspect = DebugLens::apertureAspect();
    lens.visionMode = DebugLens::visionMode();
    lens.highlightHostiles = DebugLens::highlightHostiles();
    lens.reticleAccentColor = DebugLens::reticleAccentColor();
    lens.reticleAccentRadiance = DebugLens::reticleAccentRadiance();
    lens.edgeGlowColor = DebugLens::edgeGlowColor();
    lens.edgeGlowStrength = DebugLens::edgeGlowStrength();
    lens.edgeShade = DebugLens::edgeShade();
    lens.edgeTint = DebugLens::edgeTint();
    lens.reticleLabels = DebugLens::reticleLabels();
    lens.rangefinder = DebugLens::rangefinder();
    lens.occludeOutside = DebugLens::occludeOutside();
    lens.noGrain = DebugLens::noGrain();
    lens.firstFocalPlane = DebugLens::firstFocalPlane();
    lens.markingColor = DebugLens::markingColor();
    lens.parallaxDistance = DebugLens::parallaxDistanceCm() * worldUnitsPerCm;
    lens.glareStrength = DebugLens::glareStrength();
    lens.dirtStrength = DebugLens::dirtStrength();
    lens.markingRadiance = DebugLens::markingRadiance();
    lens.markingOpacity = DebugLens::markingOpacity();
    lens.referenceMagnification = DebugLens::referenceMagnification();
    lens.reticleTexture = DebugLens::reticleTexture();

    const LensPose previousPose = createPose(camera.getPreviousViewToWorld(true));

    addLens(kDebugCameraLensId, lens, &previousPose);
  }

  Rc<DxvkImageView> LensManager::getReticleTexture(RtxContext* ctx, const std::string& path) {
    auto found = m_reticleTextures.find(path);

    if (found == m_reticleTextures.end()) {
      TextureRef texture;

      // "0x<hex>" names a texture made with remixapi_CreateTexture. It may not exist yet, so a miss is not kept.
      if (s_textureHashLookup != nullptr && s_textureHashLookup(*ctx, path, texture)) {
        found = m_reticleTextures.emplace(path, texture).first;
      } else if (path.size() > 2 && path[0] == '0' && (path[1] == 'x' || path[1] == 'X')) {
        return nullptr;
      } else {
        // A file loads in the background. The texture manager keeps it resident once it is tracked.
        auto assetData = AssetDataManager::get().findAsset(path);
        if (assetData != nullptr) {
          texture = TextureRef { ctx->getCommonObjects()->getTextureManager().preloadTextureAsset(assetData, ColorSpace::AUTO, true) };
          uint32_t unused;
          ctx->getSceneManager().trackTexture(texture, unused, true, true);
        } else {
          Logger::warn(str::format("[Lens] The reticle texture ", path, " cannot be found."));
        }
        found = m_reticleTextures.emplace(path, texture).first;
      }
    }

    DxvkImageView* view = found->second.getImageView();
    return view != nullptr ? Rc<DxvkImageView>(view) : nullptr;
  }

  bool LensManager::findSunDirection(Vector3& direction) const {
    const LightManager& lightManager = device()->getCommon()->getSceneManager().getLightManager();
    float brightest = 0.f;

    for (const auto& [hash, light] : lightManager.getLightTable()) {
      if (light.getType() != RtLightType::Distant) {
        continue;
      }

      const Vector3 radiance = light.getRadiance();
      const float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;

      if (luminance > brightest) {
        brightest = luminance;
        // A distant light's direction is the direction its light travels.
        direction = -light.getDistantLight().getDirection();
      }
    }

    return brightest > 0.f;
  }

  void LensManager::writeShaderConstants(RaytraceArgs& constants, const RtCamera& camera, uint32_t frameIdx) {
    ScopedCpuProfileZone();

    if (frameIdx != m_lastFrameIdx) {
      m_lastFrameIdx = frameIdx;

      if (enable() && debugCameraLens()) {
        addDebugCameraLens(camera);
      }

      std::vector<PendingLens> lenses;
      {
        std::lock_guard lock { m_pendingMutex };
        lenses.swap(m_pendingLenses);
      }

      if (!enable()) {
        lenses.clear();
      }

      // The nearest lenses win when there are more than the slots allow.
      const Vector3 cameraPosition = camera.getPosition(true);

      // With a lens debug view on, the lenses' placement is logged every few seconds.
      if (debugView() != LensDebugView::Off && frameIdx >= m_nextPlacementLogFrame) {
        m_nextPlacementLogFrame = frameIdx + 300;
        Logger::info(str::format("[Lens] ", lenses.size(), " lenses, camera (", cameraPosition.x, ", ", cameraPosition.y, ", ", cameraPosition.z, ")"));
        for (const PendingLens& pending : lenses) {
          const LensDesc& lens = pending.lens;
          const Vector3& c = lens.pose.eyepieceCenter;
          const Vector3& a = lens.pose.axis;
          Logger::info(str::format("[Lens]   id ", std::hex, pending.id, std::dec, " eyepiece (", c.x, ", ", c.y, ", ", c.z,
                                   ") axis (", a.x, ", ", a.y, ", ", a.z, ") radius ", lens.eyepieceRadius,
                                   " objective ", lens.objectiveDistance, " / ", lens.objectiveRadius,
                                   " eye relief ", lens.eyeRelief, " magnification ", lens.magnification,
                                   " eye distance ", length(c - cameraPosition)));
        }
      }
      std::stable_sort(lenses.begin(), lenses.end(), [&](const PendingLens& a, const PendingLens& b) {
        return lengthSqr(a.lens.pose.eyepieceCenter - cameraPosition) < lengthSqr(b.lens.pose.eyepieceCenter - cameraPosition);
      });

      const size_t lensLimit = static_cast<size_t>(std::clamp(maxLenses(), 1, static_cast<int>(kMaxLenses)));
      if (lenses.size() > lensLimit) {
        lenses.resize(lensLimit);
      }

      // A lens that existed on the previous frame keeps its slot. A new lens prefers a slot that was empty,
      // so the previous lens of a reused slot still marks its old pixels for a history discard.
      std::array<uint32_t, kMaxLenses> lensSlots;
      lensSlots.fill(kLensSlotInvalid);
      std::array<bool, kMaxLenses> slotTaken {};

      for (size_t i = 0; i < lenses.size(); ++i) {
        for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
          if (!slotTaken[slot] && m_slots[slot].active && m_slots[slot].id == lenses[i].id) {
            lensSlots[i] = slot;
            slotTaken[slot] = true;
            break;
          }
        }
      }

      for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < lenses.size(); ++i) {
          if (lensSlots[i] != kLensSlotInvalid) {
            continue;
          }

          for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
            if (!slotTaken[slot] && (pass == 1 || !m_slots[slot].active)) {
              lensSlots[i] = slot;
              slotTaken[slot] = true;
              break;
            }
          }
        }
      }

      // A camera cut or other view history reset leaves every lens without history.
      const bool viewHistoryInvalidated = camera.isViewHistoryInvalidated(frameIdx);

      m_shaderLenses = {};
      m_lensMask = 0;
      m_previousLensMask = 0;

      for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
        if (m_slots[slot].active && !viewHistoryInvalidated) {
          m_shaderLenses[kMaxLenses + slot] = m_slots[slot].args;
          m_previousLensMask |= 1u << slot;
        }
      }

      const float zoomStep = historyDiscardZoomStep();

      // The sun direction is looked up once, for the first thermal lens.
      bool sunLookedUp = false;
      bool hasSun = false;
      Vector3 sunDirection { 0.f, 0.f, 1.f };

      m_lensFeatureMask = 0;
      m_nightVisionMask = 0;
      m_telescopicMask = 0;
      m_rangefinderMask = 0;
      m_glareMask = 0;
      m_occludeOutsideMask = 0;
      for (std::string& path : m_slotReticleTextures) {
        path.clear();
      }

      for (size_t i = 0; i < lenses.size(); ++i) {
        const uint32_t slot = lensSlots[i];
        const PendingLens& pending = lenses[i];
        LensArgs args = createShaderArgs(pending.lens);

        if (pending.lens.visionMode == LensVisionMode::Thermal || pending.lens.glareStrength > 0.f) {
          if (!sunLookedUp) {
            sunLookedUp = true;
            hasSun = findSunDirection(sunDirection);
          }

          args.sunDirection = sunDirection;
        }

        if (pending.lens.visionMode == LensVisionMode::Thermal) {
          args.sunWeight = hasSun ? kThermalSunWeight : 0.f;
          m_lensFeatureMask |= 1u << slot;
        }

        // Glare and dirt light the glass of lenses without a vision mode, from the lens' own average brightness.
        if (pending.lens.visionMode == LensVisionMode::None && pending.lens.glareStrength > 0.f) {
          args.glareSun = hasSun ? 1.f : 0.f;
          m_glareMask |= 1u << slot;
        }

        if ((args.flags & lensFlagHighlight) != 0) {
          m_lensFeatureMask |= 1u << slot;
        }

        if (pending.lens.visionMode == LensVisionMode::Night) {
          m_nightVisionMask |= 1u << slot;
        }

        if (pending.lens.kind == LensKind::Telescopic) {
          m_telescopicMask |= 1u << slot;
        }

        // Parallax reads the reticle center's distance that the rangefinder measures.
        if (pending.lens.rangefinder || pending.lens.parallaxDistance > 0.f) {
          m_rangefinderMask |= 1u << slot;
        }

        if (pending.lens.occludeOutside) {
          m_occludeOutsideMask |= 1u << slot;
        }

        if (pending.lens.reticle == LensReticle::Texture) {
          m_slotReticleTextures[slot] = pending.lens.reticleTexture;
        }

        const bool hasHistory = (m_previousLensMask & (1u << slot)) != 0 && m_slots[slot].id == pending.id;

        if (hasHistory) {
          const LensArgs& previousArgs = m_slots[slot].args;

          if (pending.hasPreviousPose) {
            m_shaderLenses[kMaxLenses + slot] = replacePose(previousArgs, pending.previousPose);
          }

          if (zoomStep > 0.f && std::abs(args.magnification / previousArgs.magnification - 1.f) > zoomStep) {
            args.flags |= lensFlagDiscardHistory;
          }
        } else {
          args.flags |= lensFlagDiscardHistory;
        }

        m_shaderLenses[slot] = args;
        m_lensMask |= 1u << slot;
      }

      for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
        m_slots[slot] = Slot {};
      }

      for (size_t i = 0; i < lenses.size(); ++i) {
        Slot& slot = m_slots[lensSlots[i]];
        slot.id = lenses[i].id;
        slot.active = true;
        slot.args = m_shaderLenses[lensSlots[i]];
        slot.args.flags &= ~lensFlagDiscardHistory;
      }
    }

    for (uint32_t i = 0; i < kMaxLenses * 2; ++i) {
      constants.lenses[i] = m_shaderLenses[i];
    }

    constants.lensMask = m_lensMask;
    constants.previousLensMask = m_previousLensMask;
    constants.lensFeatureMask = m_lensFeatureMask;
  }

  Rc<DxvkImageView> LensManager::prepareFeatures(RtxContext* ctx, const VkExtent3D& renderExtent) {
    if (!enable()) {
      m_features = {};
      return nullptr;
    }

    if (m_lensFeatureMask == 0) {
      return nullptr;
    }

    const bool sizeChanged = m_features.isValid() &&
      (m_features.image->info().extent.width != renderExtent.width || m_features.image->info().extent.height != renderExtent.height);

    if (!m_features.isValid() || sizeChanged) {
      Rc<DxvkContext> context { ctx };
      m_features = Resources::createImageResource(context, "lens features", renderExtent, VK_FORMAT_R16_UINT);
    }

    return m_features.view;
  }

  void LensManager::dispatchOverlay(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    if (m_lensMask == 0) {
      return;
    }

    ScopedGpuProfileZone(ctx, "Lens Overlay");

    Rc<DxvkContext> context { ctx };
    const Resources::Resource& colorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D extent = colorBuffer.view->imageInfo().extent;

    uint32_t flags = 0;

    // Night vision, highlight outlines, thermal images and reflex reticles are shown at a displayed brightness,
    // so the overlay divides them by the exposure the tone mapper applies.
    DxvkAutoExposure& autoExposure = ctx->getCommonObjects()->metaAutoExposure();
    autoExposure.createResources(context);

    if (autoExposure.enabled()) {
      flags |= lensOverlayFlagAutoExposure;
    }

    // Night vision bloom, the glare halo and the field stop's fringe and blur read neighboring pixels, so the overlay
    // reads a copy of the output while it writes the output.
    const bool edgeOptics = m_telescopicMask != 0 && (chromaticFringe() > 0.f || edgeBlur() > 0.f);

    // Night vision and glare meter the lens' average brightness in the statistics buffer.
    if ((m_nightVisionMask != 0 || m_glareMask != 0) && m_nightStats == nullptr) {
      DxvkBufferCreateInfo info {};
      info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
      info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      info.size = kLensNightStatsFrames * kMaxLenses * kLensNightStatsWords * sizeof(uint32_t);
      m_nightStats = device()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Lens Brightness Stats");
      ctx->clearBuffer(m_nightStats, 0, info.size, 0);
    }

    if (m_nightVisionMask != 0 || m_glareMask != 0) {
      flags |= lensOverlayFlagNightStats;
    }

    if (m_nightVisionMask != 0 || m_glareMask != 0 || edgeOptics) {
      const VkFormat format = colorBuffer.image->info().format;
      const bool copyChanged = m_colorCopy.isValid() &&
        (m_colorCopy.image->info().extent.width != extent.width || m_colorCopy.image->info().extent.height != extent.height ||
         m_colorCopy.image->info().format != format);

      if (!m_colorCopy.isValid() || copyChanged) {
        m_colorCopy = Resources::createImageResource(context, "lens overlay color copy", extent, format);
      }

      const VkImageSubresourceLayers layers { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      ctx->copyImage(m_colorCopy.image, layers, VkOffset3D { 0, 0, 0 }, colorBuffer.image, layers, VkOffset3D { 0, 0, 0 }, extent);

      flags |= lensOverlayFlagColorCopy;
    } else {
      m_colorCopy = {};
    }

    if (m_lensFeatureMask != 0 && m_features.isValid()) {
      flags |= lensOverlayFlagFeatures;
    }

    if (m_occludeOutsideMask != 0) {
      flags |= lensOverlayFlagOccludeOutside;
    }

    // Each slot with a loaded reticle texture binds it and sets its flag bit.
    std::array<Rc<DxvkImageView>, kMaxLenses> reticleTextures;
    for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
      if (((m_lensMask >> slot) & 1u) != 0 && !m_slotReticleTextures[slot].empty()) {
        reticleTextures[slot] = getReticleTexture(ctx, m_slotReticleTextures[slot]);
        if (reticleTextures[slot] != nullptr) {
          flags |= 1u << (lensOverlayFlagReticleTextureShift + slot);
        }
      }
    }

    if (m_rangefinderMask != 0) {
      if (m_rangefinder == nullptr) {
        DxvkBufferCreateInfo info {};
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        info.size = 2 * kMaxLenses * kLensRangefinderWords * sizeof(uint32_t);
        m_rangefinder = device()->createBuffer(info, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, DxvkMemoryStats::Category::RTXBuffer, "Lens Rangefinder");
        ctx->clearBuffer(m_rangefinder, 0, info.size, 0);
      }

      flags |= lensOverlayFlagRangefinder;
    }

    // The peripheral blur halves the output while blurring along x, then blurs along y at half resolution.
    const float peripheralBlurAmount = std::clamp(peripheralBlur(), 0.f, 1.f);
    const VkExtent3D blurExtent { std::max(extent.width / 2, 1u), std::max(extent.height / 2, 1u), 1 };

    if (peripheralBlurAmount > 0.f) {
      ScopedGpuProfileZone(ctx, "Lens Peripheral Blur");

      for (Resources::Resource* resource : { &m_peripheralBlur[0], &m_peripheralBlur[1] }) {
        if (!resource->isValid() || resource->image->info().extent.width != blurExtent.width ||
            resource->image->info().extent.height != blurExtent.height) {
          *resource = Resources::createImageResource(context, "lens peripheral blur", blurExtent, VK_FORMAT_R16G16B16A16_SFLOAT);
        }
      }

      LensBlurArgs blurArgs {};
      blurArgs.sigma = kPeripheralBlurMaxSigma * peripheralBlurAmount;
      ctx->setPushConstantBank(DxvkPushConstantBank::RTX);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensPeripheralBlurShader::getShader());

      const VkExtent3D blurGroups = util::computeBlockCount(blurExtent, VkExtent3D { 16, 16, 1 });

      blurArgs.inputResolution = uvec2 { extent.width, extent.height };
      blurArgs.outputResolution = uvec2 { blurExtent.width, blurExtent.height };
      blurArgs.tapStep = vec2 { 2.f, 0.f };
      blurArgs.downsample = 2;
      ctx->bindResourceView(LENS_BLUR_BINDING_INPUT, colorBuffer.view, nullptr);
      ctx->bindResourceView(LENS_BLUR_BINDING_OUTPUT, m_peripheralBlur[0].view, nullptr);
      ctx->pushConstants(0, sizeof(blurArgs), &blurArgs);
      ctx->dispatch(blurGroups.width, blurGroups.height, blurGroups.depth);

      blurArgs.inputResolution = uvec2 { blurExtent.width, blurExtent.height };
      blurArgs.tapStep = vec2 { 0.f, 1.f };
      blurArgs.downsample = 1;
      ctx->bindResourceView(LENS_BLUR_BINDING_INPUT, m_peripheralBlur[0].view, nullptr);
      ctx->bindResourceView(LENS_BLUR_BINDING_OUTPUT, m_peripheralBlur[1].view, nullptr);
      ctx->pushConstants(0, sizeof(blurArgs), &blurArgs);
      ctx->dispatch(blurGroups.width, blurGroups.height, blurGroups.depth);

      flags |= lensOverlayFlagPeripheralBlur;
    } else {
      m_peripheralBlur[0] = {};
      m_peripheralBlur[1] = {};
    }

    const float deltaSeconds = std::max(GlobalTime::get().deltaTimeMs(), 0.f) * 0.001f;

    LensOverlayArgs pushArgs = {};
    pushArgs.outputResolution = uvec2 { extent.width, extent.height };
    pushArgs.debugView = static_cast<uint32_t>(debugView());
    pushArgs.softEdgePixels = softEdgePixels();
    pushArgs.renderResolution = uvec2 { rtOutput.m_compositeOutputExtent.width, rtOutput.m_compositeOutputExtent.height };
    pushArgs.flags = flags;
    pushArgs.frameIdx = m_lastFrameIdx;
    pushArgs.nightMaxGain = std::exp2(std::max(NightVision::maxGainEv(), 0.f));
    pushArgs.highlightColor = highlightColor();
    pushArgs.nightGrain = NightVision::grain();
    pushArgs.nightBloom = NightVision::bloom();
    pushArgs.nightAdaptation = 1.f - std::exp(-deltaSeconds * std::max(NightVision::adaptationSpeed(), 0.f));
    pushArgs.highlightTint = std::clamp(highlightTint(), 0.f, 1.f);
    pushArgs.nightTargetLuminance = std::max(NightVision::targetLuminance(), 1e-3f);
    pushArgs.peripheralBlurResolution = uvec2 { blurExtent.width, blurExtent.height };
    pushArgs.peripheralBlur = std::clamp(peripheralBlurAmount * 4.f, 0.f, 1.f);
    pushArgs.chromaticFringe = std::clamp(chromaticFringe(), 0.f, 1.f);
    pushArgs.edgeBlur = std::clamp(edgeBlur(), 0.f, 1.f);
    pushArgs.frameSeconds = std::min(deltaSeconds, 0.25f);
    pushArgs.metersPerWorldUnit = 1.f / std::max(RtxOptions::getMeterToWorldUnitScale(), 1e-6f);

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    Rc<DxvkBuffer> constantsBuffer = ctx->getResourceManager().getConstantsBuffer();
    ctx->bindResourceBuffer(LENS_OVERLAY_BINDING_RAYTRACE_ARGS, DxvkBufferSlice(constantsBuffer, 0, constantsBuffer->info().size));
    ctx->bindResourceView(LENS_OVERLAY_BINDING_COLOR_INPUT_OUTPUT, colorBuffer.view, nullptr);
    ctx->bindResourceView(LENS_OVERLAY_BINDING_COLOR_COPY_INPUT, m_colorCopy.isValid() ? m_colorCopy.view : nullptr, nullptr);
    ctx->bindResourceView(LENS_OVERLAY_BINDING_FEATURES_INPUT, m_features.isValid() ? m_features.view : nullptr, nullptr);
    ctx->bindResourceView(LENS_OVERLAY_BINDING_EXPOSURE_INPUT, autoExposure.getExposureTexture().view, nullptr);
    if (m_nightStats != nullptr) {
      ctx->bindResourceBuffer(LENS_OVERLAY_BINDING_NIGHT_STATS, DxvkBufferSlice(m_nightStats, 0, m_nightStats->info().size));
    } else {
      ctx->bindResourceBuffer(LENS_OVERLAY_BINDING_NIGHT_STATS, DxvkBufferSlice());
    }
    ctx->bindResourceView(LENS_OVERLAY_BINDING_BLUE_NOISE, ctx->getResourceManager().getBlueNoiseTexture(context), nullptr);
    ctx->bindResourceView(LENS_OVERLAY_BINDING_PERIPHERAL_BLUR_INPUT,
                          m_peripheralBlur[1].isValid() ? m_peripheralBlur[1].view : nullptr, nullptr);
    if (m_rangefinder != nullptr) {
      ctx->bindResourceBuffer(LENS_OVERLAY_BINDING_RANGEFINDER, DxvkBufferSlice(m_rangefinder, 0, m_rangefinder->info().size));
    } else {
      ctx->bindResourceBuffer(LENS_OVERLAY_BINDING_RANGEFINDER, DxvkBufferSlice());
    }
    ctx->bindResourceView(LENS_OVERLAY_BINDING_HIT_DISTANCE_INPUT, rtOutput.m_primaryHitDistance.view, nullptr);

    // Trilinear filtering keeps a reticle texture smooth at every size.
    const Rc<DxvkSampler> reticleSampler = ctx->getResourceManager().getSampler(
      VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
    static_assert(kMaxLenses == 4);
    const uint32_t reticleBindings[kMaxLenses] = {
      LENS_OVERLAY_BINDING_RETICLE_TEXTURE_0, LENS_OVERLAY_BINDING_RETICLE_TEXTURE_1,
      LENS_OVERLAY_BINDING_RETICLE_TEXTURE_2, LENS_OVERLAY_BINDING_RETICLE_TEXTURE_3,
    };
    for (uint32_t slot = 0; slot < kMaxLenses; ++slot) {
      ctx->bindResourceView(reticleBindings[slot], reticleTextures[slot], nullptr);
      ctx->bindResourceSampler(reticleBindings[slot], reticleSampler);
    }

    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, LensOverlayShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);

    const VkExtent3D workgroups = util::computeBlockCount(extent, VkExtent3D { 16, 16, 1 });
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void LensManager::showImguiSettings() {
    RemixGui::Checkbox("Enable Lenses", &enableObject());
    RemixGui::DragInt("Max Lenses", &maxLensesObject(), 0.1f, 1, static_cast<int>(kMaxLenses));
    RemixGui::DragFloat("Field Stop Soft Edge (pixels)", &softEdgePixelsObject(), 0.05f, 0.f, 16.f, "%.2f");
    RemixGui::DragFloat("Eye Pupil Radius (mm)", &eyePupilMmObject(), 0.05f, 0.01f, 10.f, "%.2f");
    RemixGui::DragFloat("Vignette Strength", &vignetteStrengthObject(), 0.01f, 0.f, 1.f, "%.2f");
    RemixGui::DragFloat("History Discard Zoom Step", &historyDiscardZoomStepObject(), 0.01f, 0.f, 10.f, "%.2f");
    RemixGui::DragFloat("Magnification Scale", &magnificationScaleObject(), 0.01f, 0.01f, 100.f, "%.2f");
    RemixGui::Combo("Debug View", &debugViewObject(), "Off\0Lens Mask\0Magnified Direction\0");

    ImGui::Separator();

    RemixGui::DragFloat("Peripheral Blur", &peripheralBlurObject(), 0.01f, 0.f, 1.f, "%.2f");
    RemixGui::DragFloat("Chromatic Fringe", &chromaticFringeObject(), 0.01f, 0.f, 1.f, "%.2f");
    RemixGui::DragFloat("Edge Blur", &edgeBlurObject(), 0.01f, 0.f, 1.f, "%.2f");
    RemixGui::Checkbox("Highlight on All Lenses", &highlightOnAllLensesObject());
    RemixGui::ColorEdit3("Highlight Color", &highlightColorObject());
    RemixGui::DragFloat("Highlight Tint", &highlightTintObject(), 0.01f, 0.f, 1.f, "%.2f");

    if (RemixGui::CollapsingHeader("Night Vision")) {
      ImGui::Indent();
      RemixGui::DragFloat("Max Gain (EV)", &NightVision::maxGainEvObject(), 0.05f, 0.f, 20.f, "%.2f");
      RemixGui::DragFloat("Target Luminance", &NightVision::targetLuminanceObject(), 0.005f, 0.001f, 10.f, "%.3f");
      RemixGui::DragFloat("Adaptation Speed", &NightVision::adaptationSpeedObject(), 0.05f, 0.01f, 100.f, "%.2f");
      RemixGui::ColorEdit3("Phosphor Color", &NightVision::phosphorColorObject());
      RemixGui::DragFloat("Bloom", &NightVision::bloomObject(), 0.01f, 0.f, 4.f, "%.2f");
      RemixGui::DragFloat("Grain", &NightVision::grainObject(), 0.005f, 0.f, 1.f, "%.3f");
      ImGui::Unindent();
    }

    ImGui::Separator();

    RemixGui::Checkbox("Debug Camera Lens", &debugCameraLensObject());

    if (debugCameraLens()) {
      ImGui::Indent();
      RemixGui::DragFloat("Screen Radius", &DebugLens::screenRadiusObject(), 0.005f, 0.f, 4.f, "%.3f");
      RemixGui::DragFloat("Magnification", &DebugLens::magnificationObject(), 0.05f, 0.01f, 100.f, "%.2f");
      RemixGui::DragFloat("Field Stop (degrees)", &DebugLens::fieldStopDegreesObject(), 0.1f, 0.1f, 89.f, "%.1f");
      RemixGui::DragFloat("Eye Relief (cm)", &DebugLens::eyeReliefCmObject(), 0.1f, 0.f, 100.f, "%.1f");
      RemixGui::DragFloat3("Eye Offset (cm)", &DebugLens::eyeOffsetCmObject(), 0.01f, -20.f, 20.f, "%.2f");
      RemixGui::DragFloat("Objective Distance (cm)", &DebugLens::objectiveDistanceCmObject(), 0.5f, 0.f, 500.f, "%.1f");
      RemixGui::DragFloat("Objective Radius (cm)", &DebugLens::objectiveRadiusCmObject(), 0.05f, 0.f, 50.f, "%.2f");
      RemixGui::Combo("Kind", &DebugLens::kindObject(), "Telescopic\0Reflex\0");
      RemixGui::Combo("Aperture", &DebugLens::apertureObject(), "Circle\0Rectangle\0Binocular\0");
      RemixGui::DragFloat("Aperture Aspect", &DebugLens::apertureAspectObject(), 0.01f, 0.1f, 10.f, "%.2f");
      RemixGui::Combo("Vision Mode", &DebugLens::visionModeObject(), "None\0Night\0Thermal\0");
      RemixGui::Checkbox("Highlight", &DebugLens::highlightHostilesObject());
      RemixGui::Combo("Reticle", &DebugLens::reticleObject(),
                      "None\0Crosshair\0Duplex\0Dot\0Circle Dot\0Mil-Dot\0German\0Drop\0Stadia\0Frame\0Grid\0Dial\0Texture\0");
      RemixGui::ColorEdit3("Reticle Accent Color", &DebugLens::reticleAccentColorObject());
      RemixGui::DragFloat("Reticle Accent Radiance", &DebugLens::reticleAccentRadianceObject(), 0.05f, 0.f, 100.f, "%.2f");
      RemixGui::ColorEdit3("Edge Glow Color", &DebugLens::edgeGlowColorObject());
      RemixGui::DragFloat("Edge Glow Strength", &DebugLens::edgeGlowStrengthObject(), 0.01f, 0.f, 10.f, "%.2f");
      RemixGui::DragFloat("Edge Shade", &DebugLens::edgeShadeObject(), 0.01f, 0.f, 1.f, "%.2f");
      RemixGui::ColorEdit3("Edge Tint", &DebugLens::edgeTintObject());
      RemixGui::Checkbox("Reticle Labels", &DebugLens::reticleLabelsObject());
      RemixGui::Checkbox("Rangefinder", &DebugLens::rangefinderObject());
      RemixGui::DragFloat("Parallax Distance (cm)", &DebugLens::parallaxDistanceCmObject(), 10.f, 0.f, 1000000.f, "%.0f");
      RemixGui::DragFloat("Glare Strength", &DebugLens::glareStrengthObject(), 0.01f, 0.f, 1.f, "%.2f");
      RemixGui::DragFloat("Dirt Strength", &DebugLens::dirtStrengthObject(), 0.01f, 0.f, 1.f, "%.2f");
      RemixGui::ColorEdit3("Marking Color", &DebugLens::markingColorObject());
      RemixGui::DragFloat("Marking Radiance", &DebugLens::markingRadianceObject(), 0.05f, 0.f, 100.f, "%.2f");
      RemixGui::DragFloat("Marking Opacity", &DebugLens::markingOpacityObject(), 0.01f, 0.f, 1.f, "%.2f");
      RemixGui::Checkbox("Occlude Outside", &DebugLens::occludeOutsideObject());
      RemixGui::Checkbox("No Grain", &DebugLens::noGrainObject());
      RemixGui::Checkbox("First Focal Plane Reticle", &DebugLens::firstFocalPlaneObject());
      RemixGui::DragFloat("Reference Magnification", &DebugLens::referenceMagnificationObject(), 0.05f, 0.01f, 100.f, "%.2f");
      RemixGui::InputText("Reticle Texture", &DebugLens::reticleTextureObject(), ImGuiInputTextFlags_EnterReturnsTrue);
      RemixGui::DragFloat("Reticle Line Width", &DebugLens::reticleLineWidthObject(), 0.0005f, 0.f, 0.5f, "%.4f");
      RemixGui::DragFloat("Reticle Radiance", &DebugLens::reticleRadianceObject(), 0.05f, 0.f, 1000.f, "%.2f");
      RemixGui::ColorEdit3("Reticle Color", &DebugLens::reticleColorObject());
      RemixGui::ColorEdit3("Glass Tint", &DebugLens::tintObject());
      ImGui::Unindent();
    }
  }

}
