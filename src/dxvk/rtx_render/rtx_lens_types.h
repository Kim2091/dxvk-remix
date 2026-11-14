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

#include <cstdint>
#include <string>

#include "../../util/util_vector.h"

namespace dxvk {

  enum class LensReticle : uint32_t {
    None = 0,
    Crosshair,
    Duplex,
    Dot,
    CircleDot,
    MilDot,
    German,
    Drop,
    Stadia,
    Frame,
    Grid,
    Dial,
    Texture,
  };

  enum class LensKind : uint32_t {
    // Camera rays through the eyepiece are traced as magnified rays from the objective.
    Telescopic = 0,
    // There is no exit pupil, field stop or vignette, and the reticle is drawn at infinity inside the window.
    // Camera rays restart at the objective when the objective distance is above 0 and pass unchanged otherwise.
    Reflex,
  };

  enum class LensAperture : uint32_t {
    // A disc with the eyepiece radius.
    Circle = 0,
    // A rectangle with half the height of the eyepiece radius and the aperture aspect as width over height.
    Rectangle,
    // Two circles with the eyepiece radius, side by side: their centers sit the aperture aspect times the radius to
    // the left and right of the axis. One image fills both, as through binoculars.
    Binocular,
  };

  enum class LensVisionMode : uint32_t {
    None = 0,
    Night,
    Thermal,
  };

  // Where a lens is.
  struct LensPose {
    Vector3 eyepieceCenter;
    // The optical axis, pointing out of the objective.
    Vector3 axis;
    // The reticle's up direction. It is made perpendicular to the axis.
    Vector3 up;
  };

  // A lens. The pose and lengths are in world space when the lens is added to LensManager. See documentation/Lens.md.
  struct LensDesc {
    LensPose pose;
    float eyepieceRadius = 0.f;
    // The distance from the eyepiece to the objective along the axis.
    float objectiveDistance = 0.f;
    float objectiveRadius = 0.f;
    float magnification = 1.f;
    // The apparent field stop half-angle.
    float fieldStopDegrees = 18.f;
    // The ideal eye distance behind the eyepiece.
    float eyeRelief = 0.f;
    Vector3 tint { 1.f, 1.f, 1.f };
    LensReticle reticle = LensReticle::None;
    Vector3 reticleColor { 0.f, 0.f, 0.f };
    float reticleRadiance = 0.f;
    // The half-width of a reticle line, as a fraction of the field stop radius.
    float reticleLineWidth = 0.004f;
    LensKind kind = LensKind::Telescopic;
    LensAperture aperture = LensAperture::Circle;
    // The rectangle's width over its height. For a binocular aperture, the offset of each circle's center from the axis,
    // in radii.
    float apertureAspect = 1.f;
    LensVisionMode visionMode = LensVisionMode::None;
    // Directly visible surfaces of highlighted instances get an outline in the lens image.
    bool highlightHostiles = false;
    // The color of the mil-dot dots and center dots. A radiance of 0 draws them as unlit ink.
    Vector3 reticleAccentColor { 0.f, 0.f, 0.f };
    float reticleAccentRadiance = 0.f;
    // The glow ring at the field stop.
    Vector3 edgeGlowColor { 0.f, 0.f, 0.f };
    float edgeGlowStrength = 0.f;
    // The darkening toward the field stop, from 0 to 1.
    float edgeShade = 0.f;
    // The color multiplied in toward the field stop.
    Vector3 edgeTint { 1.f, 1.f, 1.f };
    // The night vision phosphor color. Zero uses rtx.lens.nightVision.phosphorColor.
    Vector3 nightTint { 0.f, 0.f, 0.f };
    // A multiplier on the metered night vision gain.
    float nightGain = 1.f;
    // The reticle style's numbers: the range labels of the grid and the scale numbers of the frame.
    bool reticleLabels = false;
    // A readout of the distance along the axis to the surface at the reticle center, in meters.
    bool rangefinder = false;
    // Everything outside every lens is black while this lens is active.
    bool occludeOutside = false;
    // The vision mode draws no grain.
    bool noGrain = false;
    // The reticle grows with the magnification, relative to the reference magnification.
    bool firstFocalPlane = false;
    // The magnification a first focal plane reticle has its size at. 0 keeps its size.
    float referenceMagnification = 0.f;
    // The image of a texture reticle: a resolved file path, or "0x<hex>" for an API-created texture.
    std::string reticleTexture;
    // The color of the labels and the rangefinder readout. Zero draws them in the accent.
    Vector3 markingColor { 0.f, 0.f, 0.f };
    // The markings' displayed radiance. 0 draws them lit or as ink, as the accent is.
    float markingRadiance = 0.f;
    // The markings' coverage multiplier, from 0 to 1.
    float markingOpacity = 1.f;
    // The distance at which the reticle has no parallax, in world units. 0 puts the reticle at infinity.
    float parallaxDistance = 0.f;
    // The strength of the glare from bright light near the axis, from 0 to 1.
    float glareStrength = 0.f;
    // The strength of the dirt on the glass, from 0 to 1.
    float dirtStrength = 0.f;
  };

  // A lens on a mesh replacement, read from RemixLensAPI. See documentation/Lens.md, "Lenses in mod USD".
  struct LensReplacement {
    // The pose and lengths are in the lens prim's space: the eyepiece is at its origin, the axis is its -Z axis and
    // up is its +Y axis.
    LensDesc desc;
    // Only first-person (view model) instances of the mesh get the lens.
    bool firstPersonOnly = false;
  };

}
