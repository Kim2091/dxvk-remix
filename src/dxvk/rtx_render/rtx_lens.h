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

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_lens_types.h"
#include "rtx_resources.h"
#include "rtx_texture.h"
#include "rtx/concept/lens/lens.h"

namespace dxvk {

  class RtxContext;
  class RtCamera;

  enum class LensDebugView : uint32_t {
    Off = 0,
    Mask,
    Direction,
  };

  // Collects the frame's lenses and turns them into the lens slots of the raytracing constants.
  // Lens sources add lenses during the frame. The debug camera lens is added here.
  class LensManager : public CommonDeviceObject {
  public:
    // Finds a texture named "0x<hex>" by its image hash. The Remix API sets it, since it owns API-created textures,
    // so the lens code does not link against the API.
    using TextureHashLookup = bool (*)(DxvkContext& ctx, const std::string& path, TextureRef& outRef);
    static TextureHashLookup s_textureHashLookup;

    explicit LensManager(DxvkDevice* device);

    LensManager(LensManager const&) = delete;
    LensManager& operator=(LensManager const&) = delete;

    // Adds a lens to the current frame. The id must be nonzero and stay the same for the same lens across frames.
    // rtx.lens.magnificationScale multiplies the lens' magnification.
    // previousPose is the lens' pose on the previous frame.
    // When it is null, the pose the lens had on the previous frame is used.
    void addLens(uint64_t id, const LensDesc& lens, const LensPose* previousPose);

    // Picks this frame's lenses, keeps each lens in its slot, and writes the current and previous lens slots.
    // Lenses added after this call belong to the next frame.
    void writeShaderConstants(RaytraceArgs& constants, const RtCamera& camera, uint32_t frameIdx);

    // Returns the lens features image the G-buffer pass writes, created at the render resolution.
    // Returns null while no lens needs per-pixel features.
    Rc<DxvkImageView> prepareFeatures(RtxContext* ctx, const VkExtent3D& renderExtent);

    // Applies the field stop, exit pupil, vignette, tint, vision modes, highlighting and reticle to the upscaled output.
    void dispatchOverlay(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput);

    static void showImguiSettings();

    RTX_OPTION("rtx.lens", bool, enable, false,
               "Enables lenses: discs where camera rays are traced as magnified rays, as through a telescopic sight.\n"
               "Lenses come from mesh replacements in mod USD (RemixLensAPI), from the Remix API camera and from the debug camera lens.");
    RTX_OPTION_ARGS("rtx.lens", int, maxLenses, 1,
                    "The maximum number of lenses traced in a frame, nearest to the camera first, from 1 to 4.",
                    args.minValue = 1, args.maxValue = static_cast<int>(kMaxLenses));
    RTX_OPTION_ARGS("rtx.lens", float, softEdgePixels, 1.0f,
                    "The width of the soft edge of a lens' field stop, in output pixels.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.lens", float, eyePupilMm, 2.5f,
                    "The radius of the viewer's eye pupil, in millimeters. A larger pupil softens the exit pupil falloff "
                    "when the eye is off the lens' axis.",
                    args.minValue = 0.01f);
    RTX_OPTION_ARGS("rtx.lens", float, vignetteStrength, 0.5f,
                    "How much a lens' image darkens toward its field stop, from 0 (no darkening) to 1.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.lens", float, historyDiscardZoomStep, 0.3f,
                    "The relative magnification change in one frame above which a lens' pixels discard their denoiser history. "
                    "0.3 discards on a zoom step of more than 30%. 0 never discards on zoom changes.",
                    args.minValue = 0.0f);
    RTX_OPTION_ARGS("rtx.lens", float, magnificationScale, 1.0f,
                    "Multiplies the magnification of every lens except the debug camera lens. A game integration can set it "
                    "at runtime through the Remix API's SetConfigVariable to zoom a sight without changing the mod's USD. "
                    "It applies from the next frame.",
                    args.minValue = 0.01f, args.maxValue = 100.0f);
    RTX_OPTION_ARGS("rtx.lens", float, peripheralBlur, 0.0f,
                    "Blurs the image outside the lenses while a lens is active, from 0 (off) to 1, for a depth of field "
                    "feel when looking through a sight. The blur runs at half resolution and is skipped at 0.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.lens", float, chromaticFringe, 0.5f,
                    "The lateral chromatic aberration toward the field stop of telescopic lenses, from 0 (none) to 1, "
                    "where red and blue split by two output pixels at the stop.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION_ARGS("rtx.lens", float, edgeBlur, 0.5f,
                    "The blur in the outer 15% of the field of telescopic lenses (field curvature), from 0 (none) to 1, "
                    "where it reaches three output pixels at the field stop.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION("rtx.lens", bool, highlightOnAllLenses, false,
               "Outlines highlighted surfaces (the HighlightHostile instance category) in every lens, not only in lenses "
               "that ask for highlighting.");
    RTX_OPTION("rtx.lens", Vector3, highlightColor, Vector3(1.0f, 0.2f, 0.1f),
               "The color of highlight outlines in lens images, as a displayed color independent of exposure.");
    RTX_OPTION_ARGS("rtx.lens", float, highlightTint, 0.15f,
                    "How much highlighted surfaces in lens images are tinted with the highlight color, from 0 to 1.",
                    args.minValue = 0.0f, args.maxValue = 1.0f);
    RTX_OPTION("rtx.lens", LensDebugView, debugView, LensDebugView::Off,
               "Shows lens state in the output instead of the image.\n"
               "Supported enum values are 0 = Off, 1 = Mask (each lens in its own color, scaled by its transmission, black elsewhere), "
               "2 = Direction (the magnified ray direction inside lenses).");
    RTX_OPTION("rtx.lens", bool, debugCameraLens, false,
               "Adds a lens fixed to the camera at the center of the screen, to test lens optics and motion vectors without "
               "any lens source. Needs rtx.lens.enable. Its settings are the rtx.lens.debugLens options.");

    struct NightVision {
      RTX_OPTION_ARGS("rtx.lens.nightVision", float, maxGainEv, 8.0f,
                      "The largest gain night vision applies to a lens image, in stops.",
                      args.minValue = 0.0f, args.maxValue = 20.0f);
      RTX_OPTION_ARGS("rtx.lens.nightVision", float, targetLuminance, 0.18f,
                      "The displayed luminance night vision brings a lens image's average to.",
                      args.minValue = 0.001f, args.maxValue = 10.0f);
      RTX_OPTION_ARGS("rtx.lens.nightVision", float, adaptationSpeed, 3.0f,
                      "How fast night vision adapts to the brightness in a lens, per second.",
                      args.minValue = 0.01f);
      RTX_OPTION("rtx.lens.nightVision", Vector3, phosphorColor, Vector3(0.45f, 1.0f, 0.5f),
                 "The phosphor color of night vision lens images whose lens gives none.");
      RTX_OPTION_ARGS("rtx.lens.nightVision", float, bloom, 0.5f,
                      "The strength of the bloom around bright points in night vision lens images.",
                      args.minValue = 0.0f, args.maxValue = 4.0f);
      RTX_OPTION_ARGS("rtx.lens.nightVision", float, grain, 0.08f,
                      "The strength of the grain in night vision lens images. The grain grows with the square root of the "
                      "signal, so it is relatively stronger in dark areas.",
                      args.minValue = 0.0f, args.maxValue = 1.0f);
    };

    struct DebugLens {
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, screenRadius, 0.5f,
                      "The radius of the debug camera lens, as a fraction of half the screen height.",
                      args.minValue = 0.0f, args.maxValue = 4.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, magnification, 4.0f,
                      "The magnification of the debug camera lens.",
                      args.minValue = 0.01f, args.maxValue = 100.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, fieldStopDegrees, 18.0f,
                      "The apparent field stop half-angle of the debug camera lens, in degrees.",
                      args.minValue = 0.1f, args.maxValue = 89.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, eyeReliefCm, 8.0f,
                      "The eye relief of the debug camera lens, in centimeters (see rtx.sceneScale): the ideal distance "
                      "of the eye behind the eyepiece.",
                      args.minValue = 0.0f);
      RTX_OPTION("rtx.lens.debugLens", Vector3, eyeOffsetCm, Vector3(0.0f, 0.0f, 0.0f),
                 "The offset of the eye from the debug camera lens' ideal eye point, in centimeters: x to the right, "
                 "y up and z backward. A sideways offset dims the whole image, a z offset dims it from one side.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, objectiveDistanceCm, 30.0f,
                      "The distance from the eyepiece to the objective of the debug camera lens, in centimeters. "
                      "Magnified rays start at the objective.",
                      args.minValue = 0.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, objectiveRadiusCm, 2.0f,
                      "The objective radius of the debug camera lens, in centimeters. The exit pupil radius is this "
                      "divided by the magnification.",
                      args.minValue = 0.0f);
      RTX_OPTION("rtx.lens.debugLens", LensReticle, reticle, LensReticle::Duplex,
                 "The reticle of the debug camera lens.\n"
                 "Supported enum values are 0 = None, 1 = Crosshair, 2 = Duplex, 3 = Dot, 4 = Circle Dot, 5 = Mil-Dot, "
                 "6 = German, 7 = Drop, 8 = Stadia, 9 = Frame, 10 = Grid, 11 = Dial, 12 = Texture.");
      RTX_OPTION("rtx.lens.debugLens", Vector3, reticleAccentColor, Vector3(1.0f, 0.1f, 0.05f),
                 "The color of the debug camera lens' mil-dot dots and center dots.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, reticleAccentRadiance, 0.0f,
                      "The displayed radiance of the debug camera lens' reticle accent. 0 draws it as unlit ink.",
                      args.minValue = 0.0f);
      RTX_OPTION("rtx.lens.debugLens", Vector3, edgeGlowColor, Vector3(1.0f, 0.6f, 0.2f),
                 "The color of the debug camera lens' glow ring at the field stop.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, edgeGlowStrength, 0.0f,
                      "The displayed strength of the debug camera lens' glow ring. 0 draws no ring.",
                      args.minValue = 0.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, edgeShade, 0.0f,
                      "How much the debug camera lens' image darkens toward its field stop, from 0 to 1.",
                      args.minValue = 0.0f, args.maxValue = 1.0f);
      RTX_OPTION("rtx.lens.debugLens", Vector3, edgeTint, Vector3(1.0f, 1.0f, 1.0f),
                 "The color multiplied into the debug camera lens' image toward its field stop.");
      RTX_OPTION("rtx.lens.debugLens", bool, reticleLabels, false,
                 "Draws the debug camera lens' reticle labels: the grid's range labels and the frame's scale numbers.");
      RTX_OPTION("rtx.lens.debugLens", bool, rangefinder, false,
                 "Draws a rangefinder readout below the debug camera lens' reticle center.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, parallaxDistanceCm, 0.0f,
                      "The distance at which the debug camera lens' reticle has no parallax, in centimeters "
                      "(see rtx.sceneScale). 0 puts the reticle at infinity.",
                      args.minValue = 0.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, glareStrength, 0.0f,
                      "The strength of the debug camera lens' glare from bright light near its axis, from 0 to 1.",
                      args.minValue = 0.0f, args.maxValue = 1.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, dirtStrength, 0.0f,
                      "The strength of the dirt on the debug camera lens' glass, from 0 to 1.",
                      args.minValue = 0.0f, args.maxValue = 1.0f);
      RTX_OPTION("rtx.lens.debugLens", Vector3, markingColor, Vector3(0.0f, 0.0f, 0.0f),
                 "The color of the debug camera lens' labels and rangefinder readout. Black draws them in the accent.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, markingRadiance, 0.0f,
                      "The displayed radiance of the debug camera lens' markings. 0 draws them lit or as ink, as the "
                      "accent is.",
                      args.minValue = 0.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, markingOpacity, 1.0f,
                      "How opaque the debug camera lens' markings are, from 0 to 1.",
                      args.minValue = 0.0f, args.maxValue = 1.0f);
      RTX_OPTION("rtx.lens.debugLens", bool, noGrain, false,
                 "Turns off the grain of the debug camera lens' vision mode.");
      RTX_OPTION("rtx.lens.debugLens", bool, occludeOutside, false,
                 "Paints everything outside the debug camera lens black, as behind a scope mask.");
      RTX_OPTION("rtx.lens.debugLens", bool, firstFocalPlane, false,
                 "Puts the debug camera lens' reticle in the first focal plane, so it grows with the magnification "
                 "relative to rtx.lens.debugLens.referenceMagnification.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, referenceMagnification, 4.0f,
                      "The magnification at which the debug camera lens' first focal plane reticle has its size.",
                      args.minValue = 0.01f, args.maxValue = 100.0f);
      RTX_OPTION("rtx.lens.debugLens", std::string, reticleTexture, std::string(""),
                 "The image file of the debug camera lens' texture reticle (rtx.lens.debugLens.reticle 12). Its alpha is "
                 "the ink, and it covers the square from -1 to 1 of reticle space.");
      RTX_OPTION("rtx.lens.debugLens", LensKind, kind, LensKind::Telescopic,
                 "The kind of the debug camera lens.\n"
                 "Supported enum values are 0 = Telescopic, 1 = Reflex.");
      RTX_OPTION("rtx.lens.debugLens", LensAperture, aperture, LensAperture::Circle,
                 "The window shape of the debug camera lens.\n"
                 "Supported enum values are 0 = Circle, 1 = Rectangle, 2 = Binocular.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, apertureAspect, 1.5f,
                      "The width over the height of the debug camera lens' window when it is a rectangle, or the offset of "
                      "each circle's center from the axis in radii when it is binocular.",
                      args.minValue = 0.1f, args.maxValue = 10.0f);
      RTX_OPTION("rtx.lens.debugLens", LensVisionMode, visionMode, LensVisionMode::None,
                 "The vision mode of the debug camera lens.\n"
                 "Supported enum values are 0 = None, 1 = Night, 2 = Thermal.");
      RTX_OPTION("rtx.lens.debugLens", bool, highlightHostiles, false,
                 "Outlines highlighted surfaces in the debug camera lens.");
      RTX_OPTION("rtx.lens.debugLens", Vector3, reticleColor, Vector3(0.0f, 0.0f, 0.0f),
                 "The color of the debug camera lens' reticle. Lit (radiance above 0) it is the emitted color; unlit it is ink that passes this share of the image, so black is opaque.");
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, reticleRadiance, 0.0f,
                      "The emitted radiance of the debug camera lens' reticle. 0 draws a black reticle.",
                      args.minValue = 0.0f);
      RTX_OPTION_ARGS("rtx.lens.debugLens", float, reticleLineWidth, 0.004f,
                      "The half-width of the debug camera lens' reticle lines, as a fraction of the field stop radius.",
                      args.minValue = 0.0f, args.maxValue = 0.5f);
      RTX_OPTION("rtx.lens.debugLens", Vector3, tint, Vector3(0.92f, 0.92f, 0.92f),
                 "The transmission tint of the debug camera lens' glass.");
    };

  private:
    struct PendingLens {
      uint64_t id;
      LensDesc lens;
      LensPose previousPose;
      bool hasPreviousPose;
    };

    // A slot's lens on the previous frame.
    struct Slot {
      uint64_t id = 0;
      bool active = false;
      LensArgs args {};
    };

    LensArgs createShaderArgs(const LensDesc& lens) const;
    void addDebugCameraLens(const RtCamera& camera);
    // Returns the reticle texture at a path, loading it in the background on its first use.
    // Returns null while it loads or when it cannot be loaded.
    Rc<DxvkImageView> getReticleTexture(RtxContext* ctx, const std::string& path);
    // Returns the direction toward the brightest distant light.
    // Returns false when the scene has no distant light.
    bool findSunDirection(Vector3& direction) const;

    std::mutex m_pendingMutex;
    std::vector<PendingLens> m_pendingLenses;
    std::array<Slot, kMaxLenses> m_slots {};

    uint32_t m_lastFrameIdx = UINT32_MAX;
    uint32_t m_nextPlacementLogFrame = 0;
    std::array<LensArgs, kMaxLenses * 2> m_shaderLenses {};
    uint32_t m_lensMask = 0;
    uint32_t m_previousLensMask = 0;
    // The slots that need per-pixel lens features, see RaytraceArgs.lensFeatureMask.
    uint32_t m_lensFeatureMask = 0;
    // The slots with night vision.
    uint32_t m_nightVisionMask = 0;
    // The slots with telescopic lenses.
    uint32_t m_telescopicMask = 0;
    // The slots that measure the reticle center's distance, for a rangefinder or parallax.
    uint32_t m_rangefinderMask = 0;
    // The slots with glare and dirt.
    uint32_t m_glareMask = 0;
    // The slots that occlude everything outside the lenses.
    uint32_t m_occludeOutsideMask = 0;
    // The reticle texture path of each slot this frame, empty when the slot has none.
    std::array<std::string, kMaxLenses> m_slotReticleTextures;
    // The reticle textures by path. A path that failed to load keeps an empty reference, so it is not retried every
    // frame.
    std::unordered_map<std::string, TextureRef> m_reticleTextures;

    // The per-pixel lens features at the render resolution, written by the G-buffer pass.
    Resources::Resource m_features;
    // A copy of the output that the overlay reads while it needs neighboring pixels, for night vision bloom.
    Resources::Resource m_colorCopy;
    // The night vision statistics, see kLensNightStatsFrames in lens.h.
    Rc<DxvkBuffer> m_nightStats;
    // The smoothed rangefinder distances, see kLensRangefinderWords in lens.h.
    Rc<DxvkBuffer> m_rangefinder;
    // The half resolution peripheral blur: after the x pass, and after the y pass.
    std::array<Resources::Resource, 2> m_peripheralBlur;
  };

}
