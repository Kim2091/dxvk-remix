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

// The lens overlay runs on the upscaled linear output, after denoising and upscaling, so the field stop, exit pupil
// falloff, vignette, tint, vision modes, highlighting and reticle never enter any history (documentation/Lens.md,
// section Overlay).

#include "rtx/pass/raytrace_args.h"
#include "rtx/utility/math.slangh"
#include "rtx/utility/color.slangh"

layout(binding = LENS_OVERLAY_BINDING_RAYTRACE_ARGS)
ConstantBuffer<RaytraceArgs> cb;

layout(binding = LENS_OVERLAY_BINDING_COLOR_INPUT_OUTPUT)
RWTexture2D<float4> InOutColor;

layout(binding = LENS_OVERLAY_BINDING_COLOR_COPY_INPUT)
RWTexture2D<float4> InColorCopy;

layout(r16ui, binding = LENS_OVERLAY_BINDING_FEATURES_INPUT)
RWTexture2D<uint> InLensFeatures;

layout(r32f, binding = LENS_OVERLAY_BINDING_EXPOSURE_INPUT)
RWTexture1D<float> InExposure;

layout(binding = LENS_OVERLAY_BINDING_NIGHT_STATS)
RWStructuredBuffer<uint> NightStats;

layout(binding = LENS_OVERLAY_BINDING_BLUE_NOISE)
Texture2DArray BlueNoise;

layout(binding = LENS_OVERLAY_BINDING_PERIPHERAL_BLUR_INPUT)
RWTexture2D<float4> InPeripheralBlur;

layout(binding = LENS_OVERLAY_BINDING_RANGEFINDER)
RWStructuredBuffer<uint> LensRangefinder;

layout(r32f, binding = LENS_OVERLAY_BINDING_HIT_DISTANCE_INPUT)
RWTexture2D<float> InHitDistance;

layout(binding = LENS_OVERLAY_BINDING_RETICLE_TEXTURE_0)
Sampler2D<float4> ReticleTexture0;

layout(binding = LENS_OVERLAY_BINDING_RETICLE_TEXTURE_1)
Sampler2D<float4> ReticleTexture1;

layout(binding = LENS_OVERLAY_BINDING_RETICLE_TEXTURE_2)
Sampler2D<float4> ReticleTexture2;

layout(binding = LENS_OVERLAY_BINDING_RETICLE_TEXTURE_3)
Sampler2D<float4> ReticleTexture3;

layout(push_constant)
ConstantBuffer<LensOverlayArgs> push;

#include "rtx/utility/noise.slangh"
#include "rtx/concept/lens/lens.slangh"

static const vec3 kLensDebugSlotColors[4] = {
  vec3(1.0f, 0.3f, 0.3f),
  vec3(0.3f, 1.0f, 0.3f),
  vec3(0.3f, 0.5f, 1.0f),
  vec3(1.0f, 1.0f, 0.3f)
};

// How much of the glass tint a reflex lens' thin window shows.
static const float kReflexTintStrength = 0.35f;

// Returns the coverage of a line with the given half-width at a distance from its center line.
// The edge is one pixel wide.
float lensLineCoverage(float distance, float halfWidth, float pixelSize)
{
  return saturate((halfWidth - abs(distance)) / pixelSize + 0.5f);
}

// Returns the coverage of a filled disc with the given radius at a distance from its center.
float lensDotCoverage(float distance, float radius, float pixelSize)
{
  return saturate((radius - distance) / pixelSize + 0.5f);
}

// Returns the coverage of an axis-aligned box with the given center and half size.
float lensBoxCoverage(vec2 position, vec2 center, vec2 halfSize, float pixelSize)
{
  return lensLineCoverage(position.x - center.x, halfSize.x, pixelSize) * lensLineCoverage(position.y - center.y, halfSize.y, pixelSize);
}

// Returns the reticle's coverage at a position in reticle space, where the field stop radius is 1.
// x is the coverage of the reticle's main ink and y the coverage of its accent (dots and center dots).
// The reticle is in the second focal plane, so it keeps its apparent size at every magnification.
vec2 lensReticleCoverage(LensArgs lens, vec2 reticlePosition, float pixelSize)
{
  const float thinHalfWidth = lens.reticleLineWidth;
  const vec2 mirrored = abs(reticlePosition);
  const float centerDistance = length(reticlePosition);
  const float thinCross = max(
    lensLineCoverage(reticlePosition.x, thinHalfWidth, pixelSize),
    lensLineCoverage(reticlePosition.y, thinHalfWidth, pixelSize));
  const float centerDot = lensDotCoverage(centerDistance, 3.0f * thinHalfWidth, pixelSize);

  if (lens.reticleStyle == lensReticleCrosshair)
  {
    return vec2(thinCross, 0.0f);
  }

  if (lens.reticleStyle == lensReticleDuplex)
  {
    // Thick posts run in from the field stop and turn into thin lines near the center.
    const float postStart = 0.35f;
    const float postHalfWidth = 4.0f * thinHalfWidth;
    const float verticalHalfWidth = mirrored.y > postStart ? postHalfWidth : thinHalfWidth;
    const float horizontalHalfWidth = mirrored.x > postStart ? postHalfWidth : thinHalfWidth;

    return vec2(max(
      lensLineCoverage(reticlePosition.x, verticalHalfWidth, pixelSize),
      lensLineCoverage(reticlePosition.y, horizontalHalfWidth, pixelSize)), 0.0f);
  }

  if (lens.reticleStyle == lensReticleDot)
  {
    return vec2(0.0f, centerDot);
  }

  if (lens.reticleStyle == lensReticleCircleDot)
  {
    // A center dot inside a ring at half the field stop radius.
    const float ringRadius = 0.5f;

    return vec2(lensLineCoverage(centerDistance - ringRadius, thinHalfWidth, pixelSize), centerDot);
  }

  if (lens.reticleStyle == lensReticleMilDot)
  {
    // A crosshair with dots every tenth of the field stop radius along both lines.
    const float dotSpacing = 0.1f;
    const float dotRadius = 2.5f * thinHalfWidth;
    const vec2 nearestDot = round(reticlePosition / dotSpacing) * dotSpacing;
    const float horizontalDot = lensDotCoverage(length(reticlePosition - vec2(nearestDot.x, 0.0f)), dotRadius, pixelSize);
    const float verticalDot = lensDotCoverage(length(reticlePosition - vec2(0.0f, nearestDot.y)), dotRadius, pixelSize);

    return vec2(thinCross, max(horizontalDot, verticalDot));
  }

  if (lens.reticleStyle == lensReticleGerman)
  {
    // A fine cross with heavy posts from the left, the right and the bottom, ending short of the center.
    const float postStart = 0.3f;
    const float postHalfWidth = 5.0f * thinHalfWidth;
    const float sidePosts = mirrored.x > postStart ? lensLineCoverage(reticlePosition.y, postHalfWidth, pixelSize) : 0.0f;
    const float bottomPost = reticlePosition.y < -postStart ? lensLineCoverage(reticlePosition.x, postHalfWidth, pixelSize) : 0.0f;

    return vec2(max(thinCross, max(sidePosts, bottomPost)), 0.0f);
  }

  if (lens.reticleStyle == lensReticleDrop)
  {
    // Posts from half the field stop out, a fine cross inside, ticks every tenth below the center,
    // and wider holdover bars at the second and fourth tick.
    const float postStart = 0.5f;
    const float postHalfWidth = 4.0f * thinHalfWidth;
    const float verticalHalfWidth = mirrored.y > postStart ? postHalfWidth : thinHalfWidth;
    const float horizontalHalfWidth = mirrored.x > postStart ? postHalfWidth : thinHalfWidth;
    float coverage = max(
      lensLineCoverage(reticlePosition.x, verticalHalfWidth, pixelSize),
      lensLineCoverage(reticlePosition.y, horizontalHalfWidth, pixelSize));

    for (int tick = 1; tick <= 4; ++tick)
    {
      const float tickHalfLength = (tick % 2) == 0 ? 0.08f : 0.025f;
      const vec2 tickCenter = vec2(0.0f, -0.1f * float(tick));
      coverage = max(coverage, lensBoxCoverage(reticlePosition, tickCenter, vec2(tickHalfLength, thinHalfWidth), pixelSize));
    }

    return vec2(coverage, 0.0f);
  }

  if (lens.reticleStyle == lensReticleStadia)
  {
    // Posts that taper from six line widths at the field stop to one at a fifth of it, a fine cross inside,
    // and range bars above and below the center.
    const float postStart = 0.2f;
    const float horizontalHalfWidth = thinHalfWidth * lerp(1.0f, 6.0f, saturate((mirrored.x - postStart) / (1.0f - postStart)));
    const float verticalHalfWidth = thinHalfWidth * lerp(1.0f, 6.0f, saturate((mirrored.y - postStart) / (1.0f - postStart)));
    const float posts = max(
      lensLineCoverage(reticlePosition.y, horizontalHalfWidth, pixelSize),
      lensLineCoverage(reticlePosition.x, verticalHalfWidth, pixelSize));
    const float rangeBars = lensBoxCoverage(mirrored, vec2(0.0f, 0.1f), vec2(0.12f, thinHalfWidth), pixelSize);

    return vec2(max(posts, rangeBars), 0.0f);
  }

  if (lens.reticleStyle == lensReticleFrame)
  {
    // Corner brackets of a wide rectangle, ticks pointing in from the middle of each side, and a center dot.
    const vec2 corner = vec2(0.5f, 0.35f);
    const float armLength = 0.15f;
    const float tickLength = 0.06f;
    const float frameHalfWidth = 1.5f * thinHalfWidth;
    const float horizontalArm = lensBoxCoverage(mirrored, vec2(corner.x - 0.5f * armLength, corner.y), vec2(0.5f * armLength + frameHalfWidth, frameHalfWidth), pixelSize);
    const float verticalArm = lensBoxCoverage(mirrored, vec2(corner.x, corner.y - 0.5f * armLength), vec2(frameHalfWidth, 0.5f * armLength + frameHalfWidth), pixelSize);
    const float sideTicks = lensBoxCoverage(mirrored, vec2(corner.x - 0.5f * tickLength, 0.0f), vec2(0.5f * tickLength, frameHalfWidth), pixelSize);
    const float topTicks = lensBoxCoverage(mirrored, vec2(0.0f, corner.y - 0.5f * tickLength), vec2(frameHalfWidth, 0.5f * tickLength), pixelSize);

    return vec2(max(max(horizontalArm, verticalArm), max(sideTicks, topTicks)), lensDotCoverage(centerDistance, 2.0f * thinHalfWidth, pixelSize));
  }

  if (lens.reticleStyle == lensReticleGrid)
  {
    // A fine grid every fifth of the field stop radius inside 0.8 of it, a heavier center cross with range marks
    // every twentieth, and a center dot.
    const float gridSpacing = 0.2f;
    const float gridExtent = 0.8f;
    const float gridHalfWidth = 0.6f * thinHalfWidth;
    const vec2 gridDistance = abs(mirrored - round(mirrored / gridSpacing) * gridSpacing);
    const float insideGrid = all(mirrored <= gridExtent + gridHalfWidth) ? 1.0f : 0.0f;
    const float grid = insideGrid * max(
      lensLineCoverage(gridDistance.x, gridHalfWidth, pixelSize),
      lensLineCoverage(gridDistance.y, gridHalfWidth, pixelSize));

    const float markSpacing = 0.05f;
    const float markHalfLength = 0.015f;
    const vec2 markDistance = abs(mirrored - round(mirrored / markSpacing) * markSpacing);
    const float marks = max(
      lensLineCoverage(markDistance.x, thinHalfWidth, pixelSize) * lensLineCoverage(reticlePosition.y, markHalfLength, pixelSize),
      lensLineCoverage(markDistance.y, thinHalfWidth, pixelSize) * lensLineCoverage(reticlePosition.x, markHalfLength, pixelSize));

    return vec2(max(max(grid, thinCross), marks), lensDotCoverage(centerDistance, 2.0f * thinHalfWidth, pixelSize));
  }

  if (lens.reticleStyle == lensReticleDial)
  {
    // A ring at 0.6 of the field stop radius with a tick every 30 degrees, longer every 90 degrees, and a center dot.
    const float ringRadius = 0.6f;
    const float tickAngle = pi / 6.0f;
    const float angle = atan2(reticlePosition.y, reticlePosition.x);
    const float tickIndex = round(angle / tickAngle);
    const bool isMajor = (int(abs(tickIndex)) % 3) == 0;
    const float tickStart = isMajor ? 0.52f : 0.6f;
    const float tickEnd = isMajor ? 0.78f : 0.7f;
    const float tickDistance = abs(centerDistance * sin(angle - tickIndex * tickAngle));
    const float tick = lensLineCoverage(tickDistance, thinHalfWidth, pixelSize) *
      lensLineCoverage(centerDistance - 0.5f * (tickStart + tickEnd), 0.5f * (tickEnd - tickStart), pixelSize);
    const float ring = lensLineCoverage(centerDistance - ringRadius, thinHalfWidth, pixelSize);

    return vec2(max(ring, tick), lensDotCoverage(centerDistance, 2.0f * thinHalfWidth, pixelSize));
  }

  return vec2(0.0f);
}

// Returns the ink coverage of a lens slot's reticle texture at a position in reticle space.
// The texture covers the square from -1 to 1, or from -aspect to aspect across for a rectangular aperture. Its mip
// level follows the size of an output pixel on it, so it stays smooth at every size.
float lensReticleTextureCoverage(uint slot, LensArgs lens, vec2 reticlePosition, float pixelSize)
{
  if ((push.flags & (1u << (lensOverlayFlagReticleTextureShift + slot))) == 0u)
  {
    return 0.0f;
  }

  const float halfWidth = lens.aperture == lensApertureRectangle ? lens.apertureAspect : 1.0f;
  const vec2 uv = vec2(0.5f + 0.5f * reticlePosition.x / halfWidth, 0.5f - 0.5f * reticlePosition.y);

  if (any(uv < 0.0f) || any(uv > 1.0f))
  {
    return 0.0f;
  }

  uint width;
  uint height;
  float alpha = 0.0f;

  // Separate bindings per slot keep the textures out of the bindless table.
  switch (slot)
  {
  case 0:
    ReticleTexture0.GetDimensions(width, height);
    alpha = ReticleTexture0.SampleLevel(uv, log2(max(pixelSize * 0.5f * float(max(width, height)), 1.0f))).a;
    break;
  case 1:
    ReticleTexture1.GetDimensions(width, height);
    alpha = ReticleTexture1.SampleLevel(uv, log2(max(pixelSize * 0.5f * float(max(width, height)), 1.0f))).a;
    break;
  case 2:
    ReticleTexture2.GetDimensions(width, height);
    alpha = ReticleTexture2.SampleLevel(uv, log2(max(pixelSize * 0.5f * float(max(width, height)), 1.0f))).a;
    break;
  default:
    ReticleTexture3.GetDimensions(width, height);
    alpha = ReticleTexture3.SampleLevel(uv, log2(max(pixelSize * 0.5f * float(max(width, height)), 1.0f))).a;
    break;
  }

  return saturate(alpha);
}

// Returns the render resolution pixel under an output pixel.
ivec2 lensRenderPixel(vec2 screenUV)
{
  return clamp(ivec2(screenUV * vec2(push.renderResolution)), ivec2(0), ivec2(push.renderResolution) - 1);
}

uint lensLoadFeatures(ivec2 renderPixel)
{
  return InLensFeatures[clamp(renderPixel, ivec2(0), ivec2(push.renderResolution) - 1)];
}

// Returns the heat at an output pixel, bilinearly filtered from the render resolution features.
float lensLoadHeat(vec2 screenUV)
{
  const vec2 position = screenUV * vec2(push.renderResolution) - 0.5f;
  const ivec2 base = ivec2(floor(position));
  const vec2 weight = position - vec2(base);

  const float heat00 = float(lensLoadFeatures(base) & lensFeatureHeatMask);
  const float heat10 = float(lensLoadFeatures(base + ivec2(1, 0)) & lensFeatureHeatMask);
  const float heat01 = float(lensLoadFeatures(base + ivec2(0, 1)) & lensFeatureHeatMask);
  const float heat11 = float(lensLoadFeatures(base + ivec2(1, 1)) & lensFeatureHeatMask);

  return lerp(lerp(heat00, heat10, weight.x), lerp(heat01, heat11, weight.x), weight.y) / 255.0f;
}

// Returns the peripheral blur at an output pixel, bilinearly filtered from the half resolution image.
vec3 lensLoadPeripheralBlur(vec2 screenUV)
{
  const vec2 position = screenUV * vec2(push.peripheralBlurResolution) - 0.5f;
  const ivec2 base = ivec2(floor(position));
  const vec2 weight = position - vec2(base);
  const ivec2 maxPixel = ivec2(push.peripheralBlurResolution) - 1;

  const vec3 color00 = InPeripheralBlur[clamp(base, ivec2(0), maxPixel)].rgb;
  const vec3 color10 = InPeripheralBlur[clamp(base + ivec2(1, 0), ivec2(0), maxPixel)].rgb;
  const vec3 color01 = InPeripheralBlur[clamp(base + ivec2(0, 1), ivec2(0), maxPixel)].rgb;
  const vec3 color11 = InPeripheralBlur[clamp(base + ivec2(1, 1), ivec2(0), maxPixel)].rgb;

  return lerp(lerp(color00, color10, weight.x), lerp(color01, color11, weight.x), weight.y);
}

vec3 lensLoadColor(ivec2 pixel)
{
  const ivec2 clampedPixel = clamp(pixel, ivec2(0), ivec2(push.outputResolution) - 1);

  return (push.flags & lensOverlayFlagColorCopy) != 0u ? InColorCopy[clampedPixel].rgb : InOutColor[clampedPixel].rgb;
}

// Returns the color at a position in output pixels, bilinearly filtered. Pixel centers are at half-integer positions.
vec3 lensLoadColorBilinear(vec2 position)
{
  const vec2 texel = position - 0.5f;
  const ivec2 base = ivec2(floor(texel));
  const vec2 weight = texel - vec2(base);

  const vec3 color00 = lensLoadColor(base);
  const vec3 color10 = lensLoadColor(base + ivec2(1, 0));
  const vec3 color01 = lensLoadColor(base + ivec2(0, 1));
  const vec3 color11 = lensLoadColor(base + ivec2(1, 1));

  return lerp(lerp(color00, color10, weight.x), lerp(color01, color11, weight.x), weight.y);
}

// Returns where a camera ray crosses the lens' window, between its center (0) and its outline (1).
float lensWindowPosition(LensArgs lens, vec3 origin, vec3 direction)
{
  const float planeDistance = max(lensEyepiecePlaneDistance(lens, origin, direction), 0.0f);
  const vec3 offset = origin + direction * planeDistance - lens.eyepieceCenter;
  const vec2 windowPosition = vec2(dot(offset, lens.right), dot(offset, lens.up)) / max(lens.eyepieceRadius, 1e-6f);

  return lensApertureDistance(lens, windowPosition);
}

// Returns where a camera ray is between the lens' center (0) and its edge (1): the field stop of a telescopic lens,
// the window of a reflex lens. The field stop has the aperture's shape.
float lensEdgePosition(LensArgs lens, vec3 origin, vec3 direction)
{
  if (lens.kind == lensKindReflex)
  {
    return lensWindowPosition(lens, origin, direction);
  }

  const float axialDirection = max(dot(direction, lens.axis), 1e-6f);
  const vec3 transverseDirection = direction - axialDirection * lens.axis;
  const vec2 apparentTangent = vec2(dot(transverseDirection, lens.right), dot(transverseDirection, lens.up)) / axialDirection;

  return lensApertureDistance(lens, apparentTangent / lens.tanFieldStop);
}

// The built-in font: digits 0 to 9, '+', '-', '.', 'm' and a space, drawn as anti-aliased strokes on a seven-segment
// layout. A glyph box is 0.55 wide and 1 high, with y up.
static const uint kLensGlyphPlus = 10;
static const uint kLensGlyphMinus = 11;
static const uint kLensGlyphPoint = 12;
static const uint kLensGlyphMeter = 13;
static const uint kLensGlyphSpace = 14;
static const float kLensGlyphAdvance = 0.75f;
static const float kLensGlyphStrokeHalfWidth = 0.075f;

// The segments of each digit: bit 0 top, 1 upper right, 2 lower right, 3 bottom, 4 lower left, 5 upper left, 6 middle.
static const uint kLensDigitSegments[10] = { 0x3Fu, 0x06u, 0x5Bu, 0x4Fu, 0x66u, 0x6Du, 0x7Du, 0x07u, 0x7Fu, 0x6Fu };

// Returns the distance from a point to the segment between a and b.
float lensSegmentDistance(vec2 position, vec2 a, vec2 b)
{
  const vec2 ab = b - a;
  const float t = saturate(dot(position - a, ab) / max(dot(ab, ab), 1e-8f));

  return length(position - (a + ab * t));
}

// Returns the distance from a point in glyph space to the strokes of a glyph.
float lensGlyphDistance(uint glyph, vec2 position)
{
  const float left = 0.05f;
  const float right = 0.5f;
  const float bottom = 0.05f;
  const float middle = 0.5f;
  const float top = 0.95f;
  const float center = 0.5f * (left + right);
  float distance = 1e6f;

  if (glyph < 10u)
  {
    const uint segments = kLensDigitSegments[glyph];
    const vec2 ends[7][2] = {
      { vec2(left, top), vec2(right, top) },
      { vec2(right, top), vec2(right, middle) },
      { vec2(right, middle), vec2(right, bottom) },
      { vec2(left, bottom), vec2(right, bottom) },
      { vec2(left, middle), vec2(left, bottom) },
      { vec2(left, top), vec2(left, middle) },
      { vec2(left, middle), vec2(right, middle) },
    };

    for (uint segment = 0u; segment < 7u; ++segment)
    {
      if (((segments >> segment) & 1u) != 0u)
      {
        distance = min(distance, lensSegmentDistance(position, ends[segment][0], ends[segment][1]));
      }
    }
  }
  else if (glyph == kLensGlyphPlus)
  {
    distance = min(lensSegmentDistance(position, vec2(left, middle), vec2(right, middle)),
                   lensSegmentDistance(position, vec2(center, 0.25f), vec2(center, 0.75f)));
  }
  else if (glyph == kLensGlyphMinus)
  {
    distance = lensSegmentDistance(position, vec2(left, middle), vec2(right, middle));
  }
  else if (glyph == kLensGlyphPoint)
  {
    distance = length(position - vec2(center, bottom)) - 0.03f;
  }
  else if (glyph == kLensGlyphMeter)
  {
    // A lower-case m: three posts under a bar at half height.
    distance = min(min(lensSegmentDistance(position, vec2(left, bottom), vec2(left, middle)),
                       lensSegmentDistance(position, vec2(center, bottom), vec2(center, middle))),
                   min(lensSegmentDistance(position, vec2(right, bottom), vec2(right, middle)),
                       lensSegmentDistance(position, vec2(left, middle), vec2(right, middle))));
  }

  return distance;
}

// A short line of text, up to 8 glyphs.
struct LensText
{
  uint glyphs[8];
  uint count;
};

void lensTextAppend(inout LensText text, uint glyph)
{
  if (text.count < 8u)
  {
    text.glyphs[text.count] = glyph;
    text.count += 1u;
  }
}

// Appends the digits of a whole number, without leading zeros.
void lensTextAppendNumber(inout LensText text, uint value)
{
  uint divisor = 1u;
  while (divisor <= value / 10u)
  {
    divisor *= 10u;
  }

  while (divisor > 0u)
  {
    lensTextAppend(text, (value / divisor) % 10u);
    divisor /= 10u;
  }
}

// Returns the coverage of a line of text at a position in reticle space. origin is the bottom left corner of the
// first glyph, or the bottom center of the line when centered is set, and height is the glyph height.
float lensTextCoverage(LensText text, vec2 position, vec2 origin, float height, bool centered, float pixelSize)
{
  const float width = float(text.count) * kLensGlyphAdvance * height;
  const vec2 start = centered ? origin - vec2(0.5f * width, 0.0f) : origin;
  const vec2 local = (position - start) / height;

  // Pixels outside the line's box skip the glyphs.
  if (local.x < -0.2f || local.y < -0.2f || local.y > 1.2f || local.x > float(text.count) * kLensGlyphAdvance + 0.2f)
  {
    return 0.0f;
  }

  const uint index = min(uint(max(local.x, 0.0f) / kLensGlyphAdvance), max(text.count, 1u) - 1u);
  const uint glyph = text.glyphs[index];

  if (glyph == kLensGlyphSpace)
  {
    return 0.0f;
  }

  const vec2 glyphPosition = local - vec2(float(index) * kLensGlyphAdvance, 0.0f);
  const float distance = lensGlyphDistance(glyph, glyphPosition) * height;

  return saturate((kLensGlyphStrokeHalfWidth * height - distance) / pixelSize + 0.5f);
}

LensText lensTextCreate()
{
  LensText text;
  text.count = 0u;
  for (uint i = 0u; i < 8u; ++i)
  {
    text.glyphs[i] = kLensGlyphSpace;
  }
  return text;
}

// Returns a signed label such as "+40" or "-40".
LensText lensTextSignedNumber(int value)
{
  LensText text = lensTextCreate();
  lensTextAppend(text, value < 0 ? kLensGlyphMinus : kLensGlyphPlus);
  lensTextAppendNumber(text, uint(abs(value)));
  return text;
}

uint lensRangefinderIndex(uint frame, uint slot)
{
  return ((frame & 1u) * kMaxLenses + slot) * kLensRangefinderWords;
}

// Returns the coverage of the reticle style's labels at a position in reticle space.
// The grid gets range labels next to its horizontal lines, the frame gets scale numbers at its side ticks.
float lensReticleLabelCoverage(LensArgs lens, vec2 reticlePosition, float pixelSize)
{
  const float labelHeight = 0.045f;
  float coverage = 0.0f;

  if (lens.reticleStyle == lensReticleGrid)
  {
    // Labels right of the center line, just above each horizontal line from -0.8 to 0.8.
    if (reticlePosition.x > 0.0f && reticlePosition.x < 0.25f && abs(reticlePosition.y) < 0.9f)
    {
      const int line = int(round((reticlePosition.y - 0.02f) / 0.2f));

      if (line != 0 && abs(line) <= 4)
      {
        const LensText text = lensTextSignedNumber(line * 20);
        coverage = lensTextCoverage(text, reticlePosition, vec2(0.03f, float(line) * 0.2f + 0.015f), labelHeight, false, pixelSize);
      }
    }
  }
  else if (lens.reticleStyle == lensReticleFrame)
  {
    // The side ticks' offsets from the center, in hundredths of the field stop radius.
    const LensText sideText = lensTextSignedNumber(50);
    const LensText sideTextNegative = lensTextSignedNumber(-50);
    const LensText topText = lensTextSignedNumber(35);
    const LensText bottomText = lensTextSignedNumber(-35);
    coverage = max(
      max(lensTextCoverage(sideText, reticlePosition, vec2(0.38f, 0.03f), labelHeight, true, pixelSize),
          lensTextCoverage(sideTextNegative, reticlePosition, vec2(-0.38f, 0.03f), labelHeight, true, pixelSize)),
      max(lensTextCoverage(topText, reticlePosition, vec2(0.1f, 0.25f), labelHeight, true, pixelSize),
          lensTextCoverage(bottomText, reticlePosition, vec2(0.1f, -0.3f), labelHeight, true, pixelSize)));
  }

  return coverage;
}

// Returns the coverage of the rangefinder readout below the reticle center, from the previous frame's distance.
float lensRangefinderCoverage(uint slot, vec2 reticlePosition, float pixelSize)
{
  const float readoutHeight = 0.06f;
  const vec2 readoutOrigin = vec2(0.0f, -0.2f);

  // Pixels outside the readout's box skip the buffer read.
  if (abs(reticlePosition.x) > 0.3f || abs(reticlePosition.y - readoutOrigin.y - 0.5f * readoutHeight) > readoutHeight)
  {
    return 0.0f;
  }

  const uint index = lensRangefinderIndex(push.frameIdx + 1u, slot);
  LensText text = lensTextCreate();

  if (LensRangefinder[index + 1] != 0u)
  {
    lensTextAppendNumber(text, min(uint(round(asfloat(LensRangefinder[index]))), 9999u));
    lensTextAppend(text, kLensGlyphMeter);
  }
  else
  {
    lensTextAppend(text, kLensGlyphMinus);
    lensTextAppend(text, kLensGlyphMinus);
    lensTextAppend(text, kLensGlyphMinus);
  }

  return lensTextCoverage(text, reticlePosition, readoutOrigin, readoutHeight, true, pixelSize);
}

// The thread of the output pixel under the reticle center measures the distance along the axis to the surface there,
// smooths it and stores it for the next frame's readout.
void lensRangefinderMeasure(uint slot, LensArgs lens, uint2 pixel)
{
  // The reticle center is where the camera ray runs along the axis.
  const vec3 viewAxis = mul(cb.camera.worldToView, vec4(lens.axis, 0.0f)).xyz;
  const vec4 clip = mul(cb.camera.viewToProjection, vec4(viewAxis, 1.0f));

  if (clip.w <= 0.0f)
  {
    return;
  }

  const vec2 centerUV = cameraNDCToScreenUV(cb.camera, clip.xy / clip.w);

  if (any(uint2(centerUV * vec2(push.outputResolution)) != pixel))
  {
    return;
  }

  const ivec2 renderPixel = lensRenderPixel(centerUV);
  const float hitDistance = InHitDistance[renderPixel];
  const uint previousIndex = lensRangefinderIndex(push.frameIdx + 1u, slot);
  const uint currentIndex = lensRangefinderIndex(push.frameIdx, slot);

  if (!(hitDistance > 0.0f))
  {
    LensRangefinder[currentIndex + 0] = asuint(0.0f);
    LensRangefinder[currentIndex + 1] = 0u;
    return;
  }

  // The primary hit distance of a lens pixel runs from the eye to the eyepiece plane, then from the objective.
  const vec3 origin = cameraGetWorldPosition(cb.camera);
  const float eyeDistance = max(lensEyepiecePlaneDistance(lens, origin, lens.axis), 0.0f);
  const float objectiveDistance = (lens.flags & lensFlagPassThrough) != 0u ? 0.0f : dot(lens.objectiveCenter - lens.eyepieceCenter, lens.axis);
  const float meters = max(hitDistance - eyeDistance + objectiveDistance, 0.0f) * push.metersPerWorldUnit;

  // A jump of more than a tenth snaps, and smaller changes settle within a quarter second, so the digits do not
  // flicker.
  const bool previousValid = LensRangefinder[previousIndex + 1] != 0u;
  const float previousMeters = asfloat(LensRangefinder[previousIndex + 0]);
  const bool snap = !previousValid || abs(meters - previousMeters) > 0.1f * max(previousMeters, 1.0f);
  const float smoothed = snap ? meters : lerp(previousMeters, meters, 1.0f - exp(-push.frameSeconds * 12.0f));

  LensRangefinder[currentIndex + 0] = asuint(smoothed);
  LensRangefinder[currentIndex + 1] = 1u;
}

uint lensNightStatsIndex(uint frame, uint slot)
{
  return ((frame % kLensNightStatsFrames) * kMaxLenses + slot) * kLensNightStatsWords;
}

// Returns the night vision exposure average of the lens in the slot: the previous frame's measured average luminance,
// blended into the average adapted so far.
// Meters this pixel into the current frame's statistics when it is on the metering grid.
float lensNightVisionAverage(uint slot, LensArgs lens, uint2 pixel, float luminance)
{
  const uint currentIndex = lensNightStatsIndex(push.frameIdx, slot);
  const uint previousIndex = lensNightStatsIndex(push.frameIdx + kLensNightStatsFrames - 1u, slot);
  const uint nextIndex = lensNightStatsIndex(push.frameIdx + 1u, slot);

  const uint previousSum = NightStats[previousIndex + 0];
  const uint previousCount = NightStats[previousIndex + 1];
  const float previousAdapted = asfloat(NightStats[previousIndex + 2]);

  const float measured = previousCount > 0u
    ? exp2(float(previousSum) / (float(previousCount) * kLensNightStatsQuantization) - kLensNightStatsLogOffset)
    : 0.0f;
  const bool hasHistory = previousAdapted > 0.0f && (lens.flags & lensFlagDiscardHistory) == 0u;

  float adapted = previousAdapted;

  if (measured > 0.0f)
  {
    adapted = hasHistory ? lerp(previousAdapted, measured, push.nightAdaptation) : measured;
  }

  if (all((pixel % kLensNightStatsStride) == 0u))
  {
    const float logLuminance = clamp(log2(max(luminance, 1e-10f)) + kLensNightStatsLogOffset, 0.0f, 2.0f * kLensNightStatsLogOffset - 1e-3f);
    InterlockedAdd(NightStats[currentIndex + 0], uint(logLuminance * kLensNightStatsQuantization));
    InterlockedAdd(NightStats[currentIndex + 1], 1u);

    // Every metering pixel of the slot writes the same adapted value, and clears the slot of the frame after this one,
    // which no pixel reads or meters this frame.
    NightStats[currentIndex + 2] = asuint(adapted);
    NightStats[nextIndex + 0] = 0u;
    NightStats[nextIndex + 1] = 0u;
  }

  return adapted;
}

// Returns the night vision image of a lens pixel: amplified to the lens' own exposure, with bloom on bright points,
// overload, grain and the phosphor color. color is in the output's linear units, exposure is the tone mapper's.
// Returns a value noise in [0, 1) at a position, smooth between the integer lattice points.
float lensValueNoise(vec2 position)
{
  const vec2 cell = floor(position);
  const vec2 local = position - cell;
  const vec2 weight = local * local * (3.0f - 2.0f * local);
  const float n00 = frac(sin(dot(cell, vec2(127.1f, 311.7f))) * 43758.5453f);
  const float n10 = frac(sin(dot(cell + vec2(1.0f, 0.0f), vec2(127.1f, 311.7f))) * 43758.5453f);
  const float n01 = frac(sin(dot(cell + vec2(0.0f, 1.0f), vec2(127.1f, 311.7f))) * 43758.5453f);
  const float n11 = frac(sin(dot(cell + vec2(1.0f, 1.0f), vec2(127.1f, 311.7f))) * 43758.5453f);

  return lerp(lerp(n00, n10, weight.x), lerp(n01, n11, weight.x), weight.y);
}

// Returns the dirt on the eyepiece at a position on it, in units of its radius: soft smudges and a few specks.
float lensDirtMask(vec2 windowPosition)
{
  const float smudges = smoothstep(0.55f, 0.85f, 0.65f * lensValueNoise(windowPosition * 2.5f) + 0.35f * lensValueNoise(windowPosition * 6.0f + 17.0f));
  const float specks = smoothstep(0.9f, 0.97f, lensValueNoise(windowPosition * 28.0f + 41.0f));

  return saturate(0.7f * smudges + specks);
}

// Returns the lens image with glare and dirt: a veiling lift from the lens' average brightness, light from a distant
// light near the axis with a faint ghost opposite it, and dirt on the eyepiece that the glare lights. All of it stays
// small, so the target stays clear. color and average are in the output's linear units, exposure is the tone mapper's.
vec3 lensGlass(LensArgs lens, vec3 origin, vec3 direction, uint2 pixel, vec2 stopPosition, vec3 color, float average, float exposure)
{
  // Veiling glare scatters a few percent of the light that enters the lens over its whole image.
  vec3 glare = vec3(lens.glareStrength * 0.07f * average);

  // Bright pixels in the lens image, such as the sky or a lamp, scatter a soft halo around them: eight taps on a ring
  // of the copy, counting only their displayed brightness above white.
  if ((push.flags & lensOverlayFlagColorCopy) != 0u)
  {
    const float radius = 0.02f * float(push.outputResolution.y);
    float halo = 0.0f;

    for (uint tap = 0u; tap < 8u; ++tap)
    {
      const float angle = float(tap) * (0.25f * pi);
      const ivec2 tapPixel = ivec2(pixel) + ivec2(round(vec2(cos(angle), sin(angle)) * radius));
      halo += max(calcBt709Luminance(lensLoadColor(tapPixel)) * exposure - 1.0f, 0.0f);
    }

    glare += vec3(lens.glareStrength * 0.15f * min(halo / 8.0f, 2.0f) / exposure);
  }

  // A distant light near the axis adds stray light while the lens sees a lit scene. It starts about 38 degrees from
  // the axis and grows smoothly toward it. A dark room, whose displayed average stays low even after auto exposure,
  // gets none, while a daylight scene near the exposure target gets all of it.
  float lightGlare = 0.0f;

  if (lens.glareSun > 0.0f)
  {
    const float onset = smoothstep(cos(38.0f * pi / 180.0f), 1.0f, dot(lens.axis, lens.sunDirection));
    const float axisAlignment = onset * onset;
    lightGlare = lens.glareStrength * axisAlignment * smoothstep(0.02f, 0.12f, average * exposure);

    if (lightGlare > 0.0f)
    {
      glare += vec3(0.22f * lightGlare / exposure);

      // The ghost sits opposite the light's apparent position, mirrored through the center at 60% of its distance.
      vec3 eyeDirection = lens.sunDirection;
      if ((lens.flags & lensFlagPassThrough) == 0u)
      {
        lensUnmagnifyDirection(lens, lens.sunDirection, eyeDirection);
      }
      const float axialDirection = max(dot(eyeDirection, lens.axis), 1e-3f);
      const vec3 transverseDirection = eyeDirection - axialDirection * lens.axis;
      const vec2 lightPosition = vec2(dot(transverseDirection, lens.right), dot(transverseDirection, lens.up)) / (axialDirection * lens.tanFieldStop);
      const float ghost = exp(-square(length(stopPosition + 0.6f * lightPosition) / 0.12f));
      glare += vec3(1.0f, 0.85f, 0.6f) * (0.08f * lightGlare * ghost / exposure);
    }
  }

  if (lens.dirtStrength > 0.0f)
  {
    // The dirt is on the eyepiece, so it is fixed to the lens. It scatters the glare toward the eye and absorbs a
    // little of the image.
    const float planeDistance = max(lensEyepiecePlaneDistance(lens, origin, direction), 0.0f);
    const vec3 offset = origin + direction * planeDistance - lens.eyepieceCenter;
    const vec2 windowPosition = vec2(dot(offset, lens.right), dot(offset, lens.up)) / max(lens.eyepieceRadius, 1e-6f);
    const float dirt = lens.dirtStrength * lensDirtMask(windowPosition);

    color *= 1.0f - 0.06f * dirt;
    glare *= 1.0f + 4.0f * dirt;
  }

  return color + glare;
}

vec3 lensNightVision(uint slot, LensArgs lens, uint2 pixel, vec3 color, float exposure)
{
  const float luminance = calcBt709Luminance(color);
  const float average = lensNightVisionAverage(slot, lens, pixel, luminance);
  const float displayedAverage = average * exposure;
  // The lens' night gain multiplies the metered gain, so 2.5 shows the image 2.5 times as bright as the metering alone.
  const float meteredGain = displayedAverage > 0.0f ? clamp(push.nightTargetLuminance / displayedAverage, 1.0f, push.nightMaxGain) : 1.0f;
  const float gain = meteredGain * lens.nightGain;
  const float displayScale = gain * exposure;

  float signal = luminance * displayScale;

  // Bloom: the displayed luminance above white of eight taps on two rings around the pixel.
  if (push.nightBloom > 0.0f)
  {
    const float radius = 0.006f * float(push.outputResolution.y);
    float bloom = 0.0f;

    for (uint tap = 0u; tap < 8u; ++tap)
    {
      const float angle = float(tap) * (0.25f * pi);
      const float tapRadius = (tap & 1u) != 0u ? radius : 0.5f * radius;
      const ivec2 tapPixel = ivec2(pixel) + ivec2(round(vec2(cos(angle), sin(angle)) * tapRadius));
      bloom += max(calcBt709Luminance(lensLoadColor(tapPixel)) * displayScale - 1.0f, 0.0f);
    }

    signal += push.nightBloom * bloom / 8.0f;
  }

  // Overload: from two stops above the target, the tube washes out until the average falls again.
  const float overload = displayedAverage > 0.0f ? saturate(log2(displayedAverage / push.nightTargetLuminance) - 2.0f) : 0.0f;
  signal = signal * (1.0f + 4.0f * overload) + overload;

  // Grain grows with the square root of the signal, so it is relatively stronger in dark areas. A lens can turn it off.
  if ((lens.flags & lensFlagNoGrain) == 0u)
  {
    RNG rng = createRNG(pixel, push.frameIdx);
    const float noise = getNextSampleBlueNoise(rng) - 0.5f;
    signal = max(signal + 2.0f * push.nightGrain * noise * sqrt(max(signal, 1e-4f)), 0.0f);
  }

  return lens.nightTint * signal / exposure;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  if (any(threadId >= push.outputResolution))
  {
    return;
  }

  // The unjittered camera ray through the center of this output pixel.
  const vec2 screenUV = (vec2(threadId) + vec2(0.5f)) / vec2(push.outputResolution);
  const vec3 origin = cameraGetWorldPosition(cb.camera);
  const vec3 direction = cameraScreenUVToDirection(cb.camera, screenUV, false).worldDirection;

  const uint slot = lensFindSlotOfKind(origin, direction, false, true);

  if (slot == kLensSlotInvalid)
  {
    // Pixels outside the lenses are black behind the mask, or blend toward the blurred image.
    if ((push.flags & lensOverlayFlagOccludeOutside) != 0u)
    {
      InOutColor[threadId] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    else if ((push.flags & lensOverlayFlagPeripheralBlur) != 0u)
    {
      const vec3 color = InOutColor[threadId].rgb;
      InOutColor[threadId] = vec4(lerp(color, lensLoadPeripheralBlur(screenUV), push.peripheralBlur), 1.0f);
    }

    if (push.debugView == lensDebugViewMask)
    {
      InOutColor[threadId] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    else if (push.debugView == lensDebugViewDirection)
    {
      InOutColor[threadId] = vec4((0.5f + 0.5f * direction) * 0.25f, 1.0f);
    }

    return;
  }

  const LensArgs lens = cb.lenses[slot];
  const bool isReflex = lens.kind == lensKindReflex;
  const bool isRectangle = lens.aperture == lensApertureRectangle;
  const float apertureAspect = isRectangle ? lens.apertureAspect : 1.0f;

  // The apparent angle of the camera ray to the optical axis, as tangents along the reticle's right and up directions.
  const float axialDirection = max(dot(direction, lens.axis), 1e-6f);
  const vec3 transverseDirection = direction - axialDirection * lens.axis;
  const vec2 apparentTangent = vec2(dot(transverseDirection, lens.right), dot(transverseDirection, lens.up)) / axialDirection;
  const float apparentTangentLength = length(apparentTangent);

  // The size of an output pixel in tangent units near the screen center.
  const float tanHalfFovY = 1.0f / max(abs(cb.camera.viewToProjection[1][1]), 1e-6f);
  const float tangentPerPixel = 2.0f * tanHalfFovY / float(push.outputResolution.y);

  // A reflex lens has no tube, exit pupil or vignette: the eye sees the window, which the slot test already clips to.
  float fieldStopTransmission = 1.0f;
  float transmission = 1.0f;
  const float edgePosition = lensEdgePosition(lens, origin, direction);

  // Near the edge, the edge position's change per output pixel gives the direction toward the edge in pixels and the
  // distance to it, for the edge effects below.
  const float kEdgeEffectStart = 0.7f;
  const bool nearEdge = edgePosition > kEdgeEffectStart;
  vec2 outward = vec2(0.0f);
  float edgeDistancePixels = 1e6f;

  if (nearEdge)
  {
    const vec2 pixelUV = 1.0f / vec2(push.outputResolution);
    const vec3 directionRight = cameraScreenUVToDirection(cb.camera, screenUV + vec2(pixelUV.x, 0.0f), false).worldDirection;
    const vec3 directionDown = cameraScreenUVToDirection(cb.camera, screenUV + vec2(0.0f, pixelUV.y), false).worldDirection;
    const vec2 edgeGradient = vec2(
      lensEdgePosition(lens, origin, directionRight) - edgePosition,
      lensEdgePosition(lens, origin, directionDown) - edgePosition);
    const float edgeGradientLength = max(length(edgeGradient), 1e-6f);

    outward = edgeGradient / edgeGradientLength;
    edgeDistancePixels = (1.0f - edgePosition) / edgeGradientLength;
  }

  if (!isReflex)
  {
    // Outside the field stop the eye sees the black tube wall. The field stop has the aperture's shape: a rectangle's
    // edges are soft along each axis, a circle's and a binocular pair's across their outline.
    const float softEdgeWidth = max(push.softEdgePixels, 1e-3f) * tangentPerPixel;
    fieldStopTransmission = isRectangle
      ? saturate((lens.tanFieldStop * apertureAspect - abs(apparentTangent.x)) / softEdgeWidth + 0.5f) *
        saturate((lens.tanFieldStop - abs(apparentTangent.y)) / softEdgeWidth + 0.5f)
      : saturate((1.0f - edgePosition) * lens.tanFieldStop / softEdgeWidth + 0.5f);

    // The overlap of the eye's pupil with the exit pupil, where the camera ray crosses the exit pupil plane.
    // An eye off to the side dims the whole image. An eye too close or too far dims one side first.
    const vec3 exitPupilCenter = lens.eyepieceCenter - lens.axis * lens.eyeRelief;
    const float exitPupilPlaneDistance = dot(exitPupilCenter - origin, lens.axis) / axialDirection;
    const float exitPupilOffset = length(origin + direction * exitPupilPlaneDistance - exitPupilCenter);
    const float eyePupilRadius = max(lens.eyePupilRadius, 1e-6f);
    const float exitPupilTransmission =
      smoothstep(0.0f, 1.0f, saturate((lens.exitPupilRadius + eyePupilRadius - exitPupilOffset) / (2.0f * eyePupilRadius)));

    // A cos^4 falloff toward the field stop, where the field stop maps to 45 degrees.
    const float fieldPosition = saturate(edgePosition);
    const float edgeFalloff = pow(cos(fieldPosition * 0.25f * pi), 4.0f);
    const float vignette = lerp(1.0f, edgeFalloff, saturate(lens.vignetteStrength));

    // The field stop sits slightly out of focus and cuts into the eye's pupil, so the image fades out with a cos^2
    // over the outer tenth of the field instead of ending at a hard edge.
    const float kFieldStopShadowWidth = 0.1f;
    const float shadowPosition = saturate((edgePosition - (1.0f - kFieldStopShadowWidth)) / kFieldStopShadowWidth);
    const float fieldStopShadow = square(cos(0.5f * pi * shadowPosition));

    transmission = fieldStopTransmission * exitPupilTransmission * vignette * fieldStopShadow;
  }

  // The exposure the tone mapper applies, so vision modes, outlines and reflex reticles can set displayed values.
  const float exposure = (push.flags & lensOverlayFlagAutoExposure) != 0u ? max(InExposure[0], 1e-6f) : 1.0f;

  vec3 color = lensLoadColor(ivec2(threadId));

  // Toward the field stop of a telescopic lens the image goes out of focus (field curvature, in the outer 15% of the
  // field) and its colors split radially (lateral chromatic aberration, red outward and blue inward). Both sample the
  // lens image from the copy of the output.
  if (!isReflex && nearEdge && (push.flags & lensOverlayFlagColorCopy) != 0u)
  {
    const vec2 pixelCenter = vec2(threadId) + 0.5f;
    const vec3 center = color;
    const float blurRadius = push.edgeBlur * 3.0f * smoothstep(0.85f, 1.0f, edgePosition);
    const float fringe = push.chromaticFringe * 2.0f * smoothstep(kEdgeEffectStart, 1.0f, edgePosition);

    if (blurRadius > 0.05f)
    {
      const vec2 along = vec2(-outward.y, outward.x);
      color = (center +
        lensLoadColorBilinear(pixelCenter + outward * blurRadius) + lensLoadColorBilinear(pixelCenter - outward * blurRadius) +
        lensLoadColorBilinear(pixelCenter + along * blurRadius) + lensLoadColorBilinear(pixelCenter - along * blurRadius)) / 5.0f;
    }

    if (fringe > 0.01f)
    {
      color.r = max(color.r + lensLoadColorBilinear(pixelCenter - outward * fringe).r - center.r, 0.0f);
      color.b = max(color.b + lensLoadColorBilinear(pixelCenter + outward * fringe).b - center.b, 0.0f);
    }
  }

  if (lens.visionMode == lensVisionModeNight && (push.flags & lensOverlayFlagNightStats) != 0u)
  {
    color = lensNightVision(slot, lens, threadId, color, exposure);
  }
  else if (lens.visionMode == lensVisionModeThermal && (push.flags & lensOverlayFlagFeatures) != 0u)
  {
    // White hot: the heat is the displayed gray level.
    color = vec3(lensLoadHeat(screenUV)) / exposure;
  }

  // Highlighted surfaces get a soft tint and an outline along the inner edge of their mask, from the render
  // resolution pixel and its 3x3 neighbors.
  if ((lens.flags & lensFlagHighlight) != 0u && (push.flags & lensOverlayFlagFeatures) != 0u)
  {
    const ivec2 renderPixel = lensRenderPixel(screenUV);

    if ((lensLoadFeatures(renderPixel) & lensFeatureHighlight) != 0u)
    {
      bool isEdge = false;

      for (int y = -1; y <= 1; ++y)
      {
        for (int x = -1; x <= 1; ++x)
        {
          isEdge = isEdge || (lensLoadFeatures(renderPixel + ivec2(x, y)) & lensFeatureHighlight) == 0u;
        }
      }

      const vec3 highlight = push.highlightColor / exposure;
      color = isEdge ? highlight : lerp(color, highlight, push.highlightTint);
    }
  }

  const vec3 tint = isReflex ? lerp(vec3(1.0f), lens.tint, kReflexTintStrength) : lens.tint;
  color *= tint * transmission;

  // Glare and dirt light the glass of a lens without a vision mode, inside its field stop.
  if (lens.visionMode == lensVisionModeNone && (lens.glareStrength > 0.0f || lens.dirtStrength > 0.0f))
  {
    const float average = (push.flags & lensOverlayFlagNightStats) != 0u && lens.glareStrength > 0.0f
      ? lensNightVisionAverage(slot, lens, threadId, calcBt709Luminance(lensLoadColor(ivec2(threadId))))
      : 0.0f;
    const vec3 glassColor = lensGlass(lens, origin, direction, threadId, apparentTangent / lens.tanFieldStop, color, average, exposure);
    color = lerp(color, glassColor, fieldStopTransmission);
  }

  // The edge tint and shade grow toward the edge. The edge glow is a thin rim reflection just inside the edge, about a
  // pixel and a half wide and at a third of its displayed strength.
  const float kRimWidthPixels = 1.5f;
  const float kRimStrength = 0.3f;
  color *= lerp(vec3(1.0f), lens.edgeTint, smoothstep(0.3f, 1.0f, edgePosition));
  color *= 1.0f - lens.edgeShade * smoothstep(0.5f, 1.0f, edgePosition);
  const float rim = edgeDistancePixels >= 0.0f ? exp(-square(edgeDistancePixels / kRimWidthPixels)) : 0.0f;
  color += lens.edgeGlowColor * (lens.edgeGlowStrength * kRimStrength * rim) / exposure;

  // The reticle is at infinity along the axis: it depends only on the ray direction, so it stays on target as the eye
  // moves. A reflex reticle's radiance is a displayed value, and so is the accent's.
  // A first focal plane reticle and its labels are scaled with the magnification. The rangefinder readout is not.
  const vec2 stopPosition = apparentTangent / lens.tanFieldStop;
  const float stopPixelSize = tangentPerPixel / lens.tanFieldStop;
  const float reticleScale = max(lens.reticleScale, 1e-3f);
  // A reticle focused at the parallax distance shifts against a target at another distance when the eye is off the axis.
  // With the eye's offset h from the axis, the entrance pupil sees from M * h, so the target moves against the reticle
  // by M * h * (1 / distance - 1 / parallax distance) in object space, which the eyepiece magnifies M times. The whole
  // reticle moves by that, from the reticle center's distance measured one frame earlier, inside the field stop.
  vec2 parallaxShift = vec2(0.0f);

  if (lens.parallaxDistance > 0.0f && (push.flags & lensOverlayFlagRangefinder) != 0u)
  {
    const uint index = lensRangefinderIndex(push.frameIdx + 1u, slot);
    const float targetDistance = LensRangefinder[index + 1] != 0u
      ? asfloat(LensRangefinder[index]) / max(push.metersPerWorldUnit, 1e-9f)
      : 0.0f;
    const float inverseTargetDistance = targetDistance > 0.0f ? 1.0f / targetDistance : 0.0f;
    const vec3 eyeOffset = origin - lens.eyepieceCenter;
    const vec3 transverseOffset = eyeOffset - dot(eyeOffset, lens.axis) * lens.axis;
    const vec2 offset = vec2(dot(transverseOffset, lens.right), dot(transverseOffset, lens.up));
    const float magnification = (lens.flags & lensFlagPassThrough) != 0u ? 1.0f : lens.magnification;

    parallaxShift = magnification * magnification * offset * (inverseTargetDistance - 1.0f / lens.parallaxDistance) / lens.tanFieldStop;

    const float shiftLength = length(parallaxShift);
    const float kMaxParallaxShift = 0.9f;
    if (shiftLength > kMaxParallaxShift)
    {
      parallaxShift *= kMaxParallaxShift / shiftLength;
    }
  }

  const vec2 reticlePosition = (stopPosition - parallaxShift) / reticleScale;
  const float reticlePixelSize = stopPixelSize / reticleScale;
  vec2 reticle = lens.reticleStyle == lensReticleTexture
    ? vec2(lensReticleTextureCoverage(slot, lens, reticlePosition, reticlePixelSize), 0.0f)
    : lensReticleCoverage(lens, reticlePosition, reticlePixelSize);

  // The markings are text in the reticle plane, scaled by their opacity. With a marking color they have their own
  // channel. Without one, the labels are in the accent when the lens has a lit accent, else in the reticle's ink, and
  // the rangefinder readout is in the accent.
  const bool hasMarkingColor = any(lens.markingColor > 0.0f);
  float markings = 0.0f;

  if ((lens.flags & lensFlagReticleLabels) != 0u)
  {
    const float labels = lensReticleLabelCoverage(lens, reticlePosition, reticlePixelSize) * lens.markingOpacity;
    if (hasMarkingColor)
    {
      markings = max(markings, labels);
    }
    else if (lens.reticleAccentRadiance > 0.0f)
    {
      reticle.y = max(reticle.y, labels);
    }
    else
    {
      reticle.x = max(reticle.x, labels);
    }
  }

  // The rangefinder and parallax both measure the reticle center's distance.
  if (((lens.flags & lensFlagRangefinder) != 0u || lens.parallaxDistance > 0.0f) && (push.flags & lensOverlayFlagRangefinder) != 0u)
  {
    lensRangefinderMeasure(slot, lens, threadId);
  }

  if ((lens.flags & lensFlagRangefinder) != 0u && (push.flags & lensOverlayFlagRangefinder) != 0u)
  {
    const float readout = lensRangefinderCoverage(slot, stopPosition, stopPixelSize) * lens.markingOpacity;
    if (hasMarkingColor)
    {
      markings = max(markings, readout);
    }
    else
    {
      reticle.y = max(reticle.y, readout);
    }
  }

  reticle *= fieldStopTransmission;
  markings *= fieldStopTransmission;
  const float reticleCoverage = max(max(reticle.x, reticle.y), markings);
  // A lit reticle emits its color. An unlit one is ink that passes its color's share of the image, so a black reticle
  // is opaque and a gray one darkens what is behind it.
  const vec3 reticleEmission = lens.reticleRadiance > 0.0f
    ? lens.reticleColor * lens.reticleRadiance / (isReflex ? exposure : 1.0f)
    : color * saturate(lens.reticleColor);
  const vec3 accentEmission = lens.reticleAccentRadiance > 0.0f
    ? lens.reticleAccentColor * lens.reticleAccentRadiance / exposure
    : reticleEmission;
  // A marking color is emitted at the marking radiance, or follows the accent: lit at its radiance, else ink.
  const float markingRadiance = lens.markingRadiance > 0.0f ? lens.markingRadiance : lens.reticleAccentRadiance;
  const vec3 markingEmission = markingRadiance > 0.0f
    ? lens.markingColor * markingRadiance / exposure
    : color * saturate(lens.markingColor);
  color = lerp(color, reticleEmission, reticle.x);
  color = lerp(color, accentEmission, reticle.y);
  color = lerp(color, markingEmission, markings);

  if (push.debugView == lensDebugViewMask)
  {
    color = kLensDebugSlotColors[slot] * transmission * (1.0f - reticleCoverage);
  }
  else if (push.debugView == lensDebugViewDirection)
  {
    color = 0.5f + 0.5f * ((lens.flags & lensFlagPassThrough) != 0u ? direction : lensMagnifyDirection(lens, direction));
  }

  // Behind a mask, each lens window fades into the black around it the way a field stop does: a cos^2 over the outer
  // tenth of the window and the soft edge, measured in output pixels from the window's change per pixel.
  if ((push.flags & lensOverlayFlagOccludeOutside) != 0u && push.debugView == lensDebugViewOff)
  {
    const float windowPosition = lensWindowPosition(lens, origin, direction);
    const float kWindowShadowWidth = 0.1f;

    if (windowPosition > 1.0f - kWindowShadowWidth)
    {
      const vec2 pixelUV = 1.0f / vec2(push.outputResolution);
      const vec3 directionRight = cameraScreenUVToDirection(cb.camera, screenUV + vec2(pixelUV.x, 0.0f), false).worldDirection;
      const vec3 directionDown = cameraScreenUVToDirection(cb.camera, screenUV + vec2(0.0f, pixelUV.y), false).worldDirection;
      const vec2 windowGradient = vec2(
        lensWindowPosition(lens, origin, directionRight) - windowPosition,
        lensWindowPosition(lens, origin, directionDown) - windowPosition);
      const float windowDistancePixels = (1.0f - windowPosition) / max(length(windowGradient), 1e-6f);
      const float softEdge = saturate(windowDistancePixels / max(push.softEdgePixels, 1e-3f) - 0.5f);
      const float shadow = square(cos(0.5f * pi * saturate((windowPosition - (1.0f - kWindowShadowWidth)) / kWindowShadowWidth)));

      color *= softEdge * shadow;
    }
  }

  InOutColor[threadId] = vec4(color, 1.0f);
}
