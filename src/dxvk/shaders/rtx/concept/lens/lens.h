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

#include "rtx/utility/shader_types.h"

// Lenses are analytic magnifier discs tested when the primary ray is built, see documentation/Lens.md.
// RaytraceArgs holds kMaxLenses slots for this frame followed by kMaxLenses slots for the previous frame.
// A lens keeps its slot for as long as it exists, so slot i of both halves describes the same lens.
static const uint kMaxLenses = 4;
static const uint kLensSlotInvalid = 0xFFFFFFFFu;

// The lens has no usable history this frame: it just appeared, or its zoom stepped too far.
static const uint lensFlagDiscardHistory = 1u << 0;
// Directly visible highlighted surfaces are outlined in the lens image.
static const uint lensFlagHighlight = 1u << 1;
// Camera rays pass the lens unchanged: a reflex lens without an objective distance.
static const uint lensFlagPassThrough = 1u << 2;
// The reticle style's numbers are drawn.
static const uint lensFlagReticleLabels = 1u << 3;
// The rangefinder readout is drawn.
static const uint lensFlagRangefinder = 1u << 4;
// Everything outside every lens is black while this lens is active.
static const uint lensFlagOccludeOutside = 1u << 5;
// The vision mode draws no grain.
static const uint lensFlagNoGrain = 1u << 6;

// The rangefinder buffer holds two frames of kMaxLenses slots, written on alternate frames. Each slot is the smoothed
// distance in meters as float bits and 1 when it is valid.
static const uint kLensRangefinderWords = 2;

static const uint lensReticleNone = 0;
static const uint lensReticleCrosshair = 1;
static const uint lensReticleDuplex = 2;
static const uint lensReticleDot = 3;
static const uint lensReticleCircleDot = 4;
static const uint lensReticleMilDot = 5;
static const uint lensReticleGerman = 6;
static const uint lensReticleDrop = 7;
static const uint lensReticleStadia = 8;
static const uint lensReticleFrame = 9;
static const uint lensReticleGrid = 10;
static const uint lensReticleDial = 11;
static const uint lensReticleTexture = 12;

static const uint lensKindTelescopic = 0;
static const uint lensKindReflex = 1;

static const uint lensApertureCircle = 0;
static const uint lensApertureRectangle = 1;
static const uint lensApertureBinocular = 2;

static const uint lensVisionModeNone = 0;
static const uint lensVisionModeNight = 1;
static const uint lensVisionModeThermal = 2;

// The per-pixel lens features the G-buffer pass writes while RaytraceArgs.lensFeatureMask is nonzero.
// The low 8 bits hold the heat of a thermal lens pixel, from 0 (cold) to 255 (hot).
static const uint lensFeatureHeatMask = 0xFFu;
// The pixel shows a highlighted surface through a lens that highlights.
static const uint lensFeatureHighlight = 1u << 8;

// The night vision statistics hold kLensNightStatsFrames frames of kMaxLenses slots. Each slot is
// kLensNightStatsWords uints: the sum of the quantized log2 luminance, the sample count and the adapted average
// luminance as float bits.
static const uint kLensNightStatsFrames = 3;
static const uint kLensNightStatsWords = 4;
// One output pixel in kLensNightStatsStride along each axis is metered.
static const uint kLensNightStatsStride = 4;
// The log2 luminance is offset by this and quantized to 1 / kLensNightStatsQuantization stops.
static const float kLensNightStatsLogOffset = 32.0f;
static const float kLensNightStatsQuantization = 64.0f;

static const uint lensDebugViewOff = 0;
static const uint lensDebugViewMask = 1;
static const uint lensDebugViewDirection = 2;

// All positions and lengths are in world units.
struct LensArgs
{
  vec3 eyepieceCenter;
  float eyepieceRadius;

  // The optical axis, pointing out of the objective.
  vec3 axis;
  float magnification;

  // The reticle's right and up directions, perpendicular to the axis.
  vec3 right;
  // The tangent of the apparent field stop half-angle.
  float tanFieldStop;

  vec3 up;
  // The ideal eye distance behind the eyepiece. The exit pupil sits there.
  float eyeRelief;

  vec3 objectiveCenter;
  float exitPupilRadius;

  vec3 tint;
  float eyePupilRadius;

  vec3 reticleColor;
  // The reticle's emitted radiance scale. 0 draws a black reticle.
  float reticleRadiance;

  uint reticleStyle;
  // The half-width of a reticle line, as a fraction of the field stop radius.
  float reticleLineWidth;
  float vignetteStrength;
  uint flags;

  uint kind;
  uint aperture;
  // The rectangle's width over its height.
  // For a binocular aperture, the offset of each circle's center from the axis, in radii.
  float apertureAspect;
  uint visionMode;

  // The direction toward the brightest distant light, for thermal lenses.
  vec3 sunDirection;
  // The weight of the sun term in thermal lens images. 0 when there is no distant light.
  float sunWeight;

  // The color of the mil-dot dots and center dots.
  vec3 reticleAccentColor;
  // The accent's displayed radiance. 0 draws the accent like the rest of the reticle.
  float reticleAccentRadiance;

  vec3 edgeGlowColor;
  // The displayed strength of the glow ring at the field stop.
  float edgeGlowStrength;

  // The color multiplied in toward the field stop.
  vec3 edgeTint;
  // The darkening toward the field stop, from 0 to 1.
  float edgeShade;

  // The night vision phosphor color.
  vec3 nightTint;
  // A multiplier on the metered night vision gain.
  float nightGain;

  // The reticle's size relative to the field stop: magnification over the reference magnification for a first focal
  // plane reticle, 1 otherwise.
  float reticleScale;
  // The markings' coverage multiplier.
  float markingOpacity;
  // The distance at which the reticle has no parallax. 0 puts the reticle at infinity.
  float parallaxDistance;
  // The strength of the glare from bright light near the axis, from 0 to 1.
  float glareStrength;

  // The labels' and rangefinder's color. Zero draws them in the accent.
  vec3 markingColor;
  // The markings' displayed radiance. 0 draws them lit or as ink, as the accent is.
  float markingRadiance;

  // The strength of the dirt on the glass, from 0 to 1.
  float dirtStrength;
  // 1 when sunDirection is the direction toward a distant light that can cause glare, 0 otherwise.
  float glareSun;
  uint pad1;
  uint pad2;
};

// The overlay reads the color from the copy instead of the output, so it can read neighbors, for night vision bloom
// and the chromatic fringe and blur at the field stop.
static const uint lensOverlayFlagColorCopy = 1u << 0;
// The G-buffer pass wrote the lens features this frame.
static const uint lensOverlayFlagFeatures = 1u << 1;
// The exposure input holds the auto exposure the tone mapper applies.
static const uint lensOverlayFlagAutoExposure = 1u << 2;
// A night vision lens is active, so the night vision statistics are bound.
static const uint lensOverlayFlagNightStats = 1u << 3;

// Push constants of the lens overlay pass.
struct LensOverlayArgs
{
  uvec2 outputResolution;
  uint debugView;
  // The width of the field stop's soft edge, in output pixels.
  float softEdgePixels;

  uvec2 renderResolution;
  uint flags;
  uint frameIdx;

  // The resolution of the blurred image that pixels outside the lenses blend toward.
  uvec2 peripheralBlurResolution;
  // How far pixels outside the lenses blend toward the blurred image, from 0 to 1.
  float peripheralBlur;
  // The largest gain night vision applies to the lens image, before the lens' night gain.
  float nightMaxGain;

  vec3 highlightColor;
  float nightGrain;

  float nightBloom;
  // The fraction of the way the night vision exposure moves toward the measured average this frame.
  float nightAdaptation;
  // How much highlighted surfaces are tinted with the highlight color, from 0 to 1.
  float highlightTint;
  // The displayed luminance night vision brings the lens average to.
  float nightTargetLuminance;

  // The lateral chromatic aberration at the field stop, from 0 to 1 (two output pixels).
  float chromaticFringe;
  // The field curvature blur at the field stop, from 0 to 1 (three output pixels).
  float edgeBlur;
  // The frame time in seconds.
  float frameSeconds;
  // Meters per world unit, for the rangefinder.
  float metersPerWorldUnit;
};

// The rangefinder buffer is bound and at least one lens has a rangefinder.
static const uint lensOverlayFlagRangefinder = 1u << 5;
// An active lens occludes everything outside the lenses.
static const uint lensOverlayFlagOccludeOutside = 1u << 6;
// Bit lensOverlayFlagReticleTextureShift + i is set when lens slot i's reticle texture is loaded and bound.
static const uint lensOverlayFlagReticleTextureShift = 8;

// The overlay blends pixels outside the lenses toward the peripheral blur image.
static const uint lensOverlayFlagPeripheralBlur = 1u << 4;

// Push constants of the peripheral blur passes. Each pass is one direction of a separable Gaussian.
struct LensBlurArgs
{
  uvec2 inputResolution;
  uvec2 outputResolution;
  // The step between taps in input pixels.
  vec2 tapStep;
  // The Gaussian's standard deviation in taps.
  float sigma;
  // The input pixels per output pixel along each axis: 2 for the pass that halves the resolution, 1 otherwise.
  uint downsample;
};

// The number of taps on each side of the center in a peripheral blur pass.
static const int kLensBlurRadius = 12;

#define LENS_BLUR_BINDING_INPUT 0
#define LENS_BLUR_BINDING_OUTPUT 1

#define LENS_OVERLAY_BINDING_RAYTRACE_ARGS 0
#define LENS_OVERLAY_BINDING_COLOR_INPUT_OUTPUT 1
#define LENS_OVERLAY_BINDING_COLOR_COPY_INPUT 2
#define LENS_OVERLAY_BINDING_FEATURES_INPUT 3
#define LENS_OVERLAY_BINDING_EXPOSURE_INPUT 4
#define LENS_OVERLAY_BINDING_NIGHT_STATS 5
#define LENS_OVERLAY_BINDING_BLUE_NOISE 6
#define LENS_OVERLAY_BINDING_PERIPHERAL_BLUR_INPUT 7
#define LENS_OVERLAY_BINDING_RANGEFINDER 8
#define LENS_OVERLAY_BINDING_HIT_DISTANCE_INPUT 9
// The reticle textures of lens slots 0 to 3.
#define LENS_OVERLAY_BINDING_RETICLE_TEXTURE_0 10
#define LENS_OVERLAY_BINDING_RETICLE_TEXTURE_1 11
#define LENS_OVERLAY_BINDING_RETICLE_TEXTURE_2 12
#define LENS_OVERLAY_BINDING_RETICLE_TEXTURE_3 13
