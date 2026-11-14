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

namespace dxvk {
  // The schema identifier and property names of RemixLensAPI (src/usd-plugins/RemixLens).
  // The mod reader and test_lens_schema both use them.
  namespace lens_usd {
    inline constexpr const char* kApiName = "RemixLensAPI";

    inline constexpr const char* kEyepieceRadius = "lens:eyepieceRadius";
    inline constexpr const char* kObjectiveDistance = "lens:objectiveDistance";
    inline constexpr const char* kObjectiveRadius = "lens:objectiveRadius";
    inline constexpr const char* kMagnification = "lens:magnification";
    inline constexpr const char* kFieldStopDegrees = "lens:fieldStopDegrees";
    inline constexpr const char* kEyeRelief = "lens:eyeRelief";
    inline constexpr const char* kTint = "lens:tint";
    inline constexpr const char* kReticle = "lens:reticle";
    inline constexpr const char* kReticleColor = "lens:reticleColor";
    inline constexpr const char* kReticleRadiance = "lens:reticleRadiance";
    inline constexpr const char* kReticleLineWidth = "lens:reticleLineWidth";
    inline constexpr const char* kVisionMode = "lens:visionMode";
    inline constexpr const char* kFirstPersonOnly = "lens:firstPersonOnly";
    inline constexpr const char* kKind = "lens:kind";
    inline constexpr const char* kAperture = "lens:aperture";
    inline constexpr const char* kApertureAspect = "lens:apertureAspect";
    inline constexpr const char* kHighlightHostiles = "lens:highlightHostiles";
    inline constexpr const char* kReticleAccentColor = "lens:reticleAccentColor";
    inline constexpr const char* kReticleAccentRadiance = "lens:reticleAccentRadiance";
    inline constexpr const char* kEdgeGlowColor = "lens:edgeGlowColor";
    inline constexpr const char* kEdgeGlowStrength = "lens:edgeGlowStrength";
    inline constexpr const char* kEdgeShade = "lens:edgeShade";
    inline constexpr const char* kEdgeTint = "lens:edgeTint";
    inline constexpr const char* kNightTint = "lens:nightTint";
    inline constexpr const char* kNightGain = "lens:nightGain";
    inline constexpr const char* kMarkings = "lens:markings";
    inline constexpr const char* kOccludeOutside = "lens:occludeOutside";
    inline constexpr const char* kNoGrain = "lens:noGrain";
    inline constexpr const char* kFirstFocalPlane = "lens:firstFocalPlane";
    inline constexpr const char* kReferenceMagnification = "lens:referenceMagnification";
    inline constexpr const char* kReticleTexture = "lens:reticleTexture";
    inline constexpr const char* kMarkingColor = "lens:markingColor";
    inline constexpr const char* kMarkingRadiance = "lens:markingRadiance";
    inline constexpr const char* kMarkingOpacity = "lens:markingOpacity";
    inline constexpr const char* kParallaxDistance = "lens:parallaxDistance";
    inline constexpr const char* kGlareStrength = "lens:glareStrength";
    inline constexpr const char* kDirtStrength = "lens:dirtStrength";

    inline constexpr const char* kProperties[] = {
      kEyepieceRadius, kObjectiveDistance, kObjectiveRadius, kMagnification, kFieldStopDegrees, kEyeRelief, kTint,
      kReticle, kReticleColor, kReticleRadiance, kReticleLineWidth, kVisionMode, kFirstPersonOnly,
      kKind, kAperture, kApertureAspect, kHighlightHostiles,
      kReticleAccentColor, kReticleAccentRadiance, kEdgeGlowColor, kEdgeGlowStrength, kEdgeShade, kEdgeTint,
      kNightTint, kNightGain, kMarkings, kOccludeOutside, kFirstFocalPlane, kReferenceMagnification, kReticleTexture,
      kMarkingColor, kMarkingRadiance, kMarkingOpacity, kParallaxDistance, kGlareStrength, kDirtStrength, kNoGrain,
    };
  }
}
