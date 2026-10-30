// Checks the colour grade math (src/dxvk/rtx_render/rtx_color_grade_math.h):
// - the built-in presets fold into exactly the matrices the tone map shader hard-coded before grades became data,
//   and the tone curve chart's CPU mirror gives the same values through either path;
// - the built-ins' anchor and shoulder give exactly the curve the shader hard-coded (anchor 0.18, shoulder 0.01),
//   and any anchor and shoulder in range give the shader's curve, which rises monotonically below the peak;
// - the neutral grade is the identity;
// - blends converge to their target, end exactly on it, and a new target during a blend does not jump;
// - a two-preset mix is exact at its ends.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../../src/dxvk/rtx_render/rtx_color_grade_math.h"

namespace {
  using namespace dxvk::color_grade;

  int g_failures = 0;

  void check(bool ok, const char* what) {
    if (!ok) {
      std::printf("FAIL: %s\n", what);
      ++g_failures;
    }
  }

  // The grade matrices tonemap_operator.slangh held before the grade became data (rtx.tonemap.grade 0 and 1).
  const Mat3 kOldWarmAmber { { { 1.095973f, -0.059536f, 0.149696f },
                               { -0.028776f, 1.016891f, -0.009769f },
                               { 0.002389f, 0.008034f, 0.655857f } } };
  const Mat3 kOldOliveGreen { { { 0.857235f, 0.192498f, 0.086365f },
                                { 0.033934f, 0.914132f, 0.056143f },
                                { 0.084339f, 0.283658f, 0.189434f } } };

  // The chart's mirror of the tone curve (rtx_tone_mapping.cpp, tone_curve).
  const Mat3 kPrimaries1 { { { 0.726346493f, 0.166654825f, 0.106998682f },
                             { 0.134236977f, 0.700138867f, 0.165624142f },
                             { 0.072639547f, 0.060120508f, 0.867239952f } } };
  const Mat3 kPrimariesInverse1 { { { 1.450212359f, -0.335329264f, -0.114883065f },
                                    { -0.253468722f, 1.510707617f, -0.257238865f },
                                    { -0.103896290f, -0.076639965f, 1.180536270f } } };

  Vec3 mul(const Mat3& m, const Vec3& v) {
    return { m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
             m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
             m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] };
  }

  // The curve as the shader hard-coded it, with an anchor of 0.18 and a shoulder of 0.01.
  float curve(float x, float peak) {
    if (x <= 0.18f) {
      return x;
    }
    const float range = peak - 0.18f;
    const float t = (x - 0.18f) / range;
    return 0.18f + range * t / (1.0f + t * std::exp(-0.01f / (t * t)));
  }

  float toneMapGraded(Vec3 color, float peak, bool basis) {
    if (basis) {
      color = mul(kPrimaries1, color);
    }
    for (float& c : color) {
      c = curve(c, peak);
    }
    if (basis) {
      color = mul(kPrimariesInverse1, color);
    }
    return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
  }

  // The chart's value before: the grade switch on the enum. 2 was None.
  float oldChart(float exposed, float peak, int grade, bool basis) {
    Vec3 color { exposed, exposed, exposed };
    if (grade == 0) {
      color = mul(kOldWarmAmber, color);
    } else if (grade == 1) {
      color = mul(kOldOliveGreen, color);
    }
    return toneMapGraded(color, peak, basis);
  }

  // The chart's value now: the grade's constants, then color_grade::toneCurve with the grade's anchor and shoulder.
  float newChart(float exposed, float peak, const Params& grade, bool basis) {
    const Gpu g = toGpu(grade);
    Vec3 color = apply(g, { exposed, exposed, exposed });
    if (basis) {
      color = mul(kPrimaries1, color);
    }
    for (float& c : color) {
      c = toneCurve(c, peak, g.anchor, g.shoulder);
    }
    if (basis) {
      color = mul(kPrimariesInverse1, color);
    }
    return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
  }

  // reinhardPiecewiseSmooth in tonemap_operator.slangh, line for line.
  float shaderCurve(float x, float peak, float anchor, float shoulder) {
    if (x <= anchor) {
      return x;
    }
    float range = peak - anchor;
    float t = (x - anchor) / range;
    float s = std::exp(-shoulder / (t * t));
    return anchor + range * t / (1.0f + t * s);
  }

  float distance(const Params& a, const Params& b) {
    float d = 0.0f;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        d = std::max(d, std::abs(a.matrix[i][j] - b.matrix[i][j]));
      }
      d = std::max(d, std::abs(a.balance[i] - b.balance[i]));
    }
    d = std::max(d, std::abs(a.saturation - b.saturation));
    d = std::max(d, std::abs(a.contrast - b.contrast));
    d = std::max(d, std::abs(a.anchor - b.anchor));
    d = std::max(d, std::abs(a.shoulder - b.shoulder));
    return d;
  }

  Params someGrade() {
    Params p = warmAmber();
    p.balance = { 1.1f, 0.95f, 0.8f };
    p.saturation = 1.3f;
    p.contrast = 1.2f;
    p.anchor = 0.25f;
    p.shoulder = 0.2f;
    return p;
  }
}

int main() {
  // Built-ins: the folded constants are bit for bit the old matrices, with no contrast.
  const auto& presets = builtIns();
  check(presets.size() == 3, "three built-in presets");
  const Mat3 identity = Params {}.matrix;
  check(toGpu(presets[0].params).rows == identity, "None folds to the identity");
  check(toGpu(presets[1].params).rows == kOldWarmAmber, "Warm Amber folds to the old matrix");
  check(toGpu(presets[2].params).rows == kOldOliveGreen, "Olive Green folds to the old matrix");
  for (const auto& b : presets) {
    check(toGpu(b.params).contrast == 1.0f, "built-ins have no contrast");
    check(toGpu(b.params).anchor == 0.18f, "built-ins have the anchor the shader hard-coded");
    check(toGpu(b.params).shoulder == 0.01f, "built-ins have the shoulder the shader hard-coded");
  }

  // The default anchor and shoulder give the hard-coded curve bit for bit.
  for (float peak : { 1.0f, 4.926f, 125.0f }) {
    for (int i = 0; i < 400; ++i) {
      const float x = 0.18f * std::exp2(-10.0f + 30.0f * static_cast<float>(i) / 399.0f);
      if (toneCurve(x, peak, Params {}.anchor, Params {}.shoulder) != curve(x, peak)) {
        std::printf("default curve differs: x %.6g peak %.3f\n", x, peak);
        ++g_failures;
      }
    }
  }

  // Any anchor and shoulder in range: the CPU curve is the shader's, it is continuous at the anchor,
  // rises monotonically and stays at or below the peak.
  for (float anchor : { kAnchorMin, 0.05f, 0.18f, 0.3f, kAnchorMax }) {
    for (float shoulder : { kShoulderMin, 0.01f, 0.1f, 0.5f, kShoulderMax }) {
      for (float peak : { 1.0f, 4.926f, 125.0f }) {
        float previous = 0.0f;
        bool same = true;
        bool monotonic = true;
        bool belowPeak = true;
        for (int i = 0; i < 2000; ++i) {
          const float x = anchor * std::exp2(-6.0f + 26.0f * static_cast<float>(i) / 1999.0f);
          const float y = toneCurve(x, peak, anchor, shoulder);
          same = same && y == shaderCurve(x, peak, anchor, shoulder);
          // For inputs far above the peak the curve is within float rounding of it, so a step down of an ulp or two is allowed.
          monotonic = monotonic && y >= previous - 1e-6f * peak && std::isfinite(y);
          belowPeak = belowPeak && y <= peak;
          previous = y;
        }
        const float justAbove = toneCurve(anchor * 1.0001f, peak, anchor, shoulder);
        if (!same || !monotonic || !belowPeak || std::abs(justAbove - anchor * 1.0001f) > anchor * 1e-4f) {
          std::printf("curve fails: anchor %.3f shoulder %.4f peak %.3f (same %d, monotonic %d, below peak %d)\n",
                      anchor, shoulder, peak, same, monotonic, belowPeak);
          ++g_failures;
        }
      }
    }
  }

  // The chart and the shader agree for a grade with its own anchor and shoulder.
  {
    const Params p = someGrade();
    const Gpu g = toGpu(p);
    for (int i = 0; i < 192; ++i) {
      const float x = 0.18f * std::exp2(-8.0f + 20.0f * static_cast<float>(i) / 191.0f);
      Vec3 color = mul(kPrimaries1, apply(g, { x, x, x }));
      for (float& c : color) {
        c = shaderCurve(c, 4.926f, 0.25f, 0.2f);
      }
      color = mul(kPrimariesInverse1, color);
      const float shader = 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
      check(newChart(x, 4.926f, p, true) == shader, "the chart matches the shader for a non-default anchor and shoulder");
    }
  }

  // Clamping keeps the anchor and shoulder in range, and a non-finite value takes the default.
  {
    Params p;
    p.anchor = 0.0f;
    p.shoulder = 5.0f;
    check(clamped(p).anchor == kAnchorMin && clamped(p).shoulder == kShoulderMax, "anchor and shoulder clamp to range");
    p.anchor = NAN;
    p.shoulder = INFINITY;
    check(clamped(p).anchor == 0.18f && clamped(p).shoulder == 0.01f, "a non-finite anchor or shoulder takes the default");
  }

  // The chart through the old switch and through the constants, for every built-in, with and without the basis.
  const int oldIndex[3] = { 2, 0, 1 };
  for (int b = 0; b < 3; ++b) {
    for (int basis = 0; basis < 2; ++basis) {
      for (float peak : { 1.0f, 4.926f }) {
        for (int i = 0; i < 192; ++i) {
          const float stops = -8.0f + 20.0f * static_cast<float>(i) / 191.0f;
          const float x = 0.18f * std::exp2(stops);
          if (oldChart(x, peak, oldIndex[b], basis != 0) != newChart(x, peak, presets[b].params, basis != 0)) {
            std::printf("chart differs: %s stops %.3f basis %d peak %.3f\n", presets[b].name, stops, basis, peak);
            ++g_failures;
          }
        }
      }
    }
  }

  // Neutral is the identity.
  const Vec3 colors[] = { { 0.0f, 0.0f, 0.0f }, { 0.18f, 0.18f, 0.18f }, { 3.0f, 0.2f, 0.01f }, { 0.5f, 7.0f, 120.0f } };
  for (const Vec3& c : colors) {
    check(apply(toGpu(Params {}), c) == c, "the neutral grade leaves colours unchanged");
  }
  check(clamped(Params {}) == Params {}, "clamping keeps the neutral grade");

  // Contrast pivots at mid grey and saturation 0 gives grey.
  {
    Params p;
    p.contrast = 1.5f;
    const Vec3 g = apply(toGpu(p), { 0.18f, 0.18f, 0.18f });
    check(std::abs(g[0] - 0.18f) < 1e-6f, "contrast keeps mid grey");
    p = Params {};
    p.saturation = 0.0f;
    const Vec3 s = apply(toGpu(p), { 1.0f, 0.0f, 0.0f });
    check(std::abs(s[0] - s[1]) < 1e-6f && std::abs(s[1] - s[2]) < 1e-6f, "saturation 0 is grey");
  }

  // Mixes are exact at their ends.
  const Params a = presets[0].params;
  const Params b = someGrade();
  check(mix(a, b, 0.0f) == a, "mix at 0 is A");
  check(mix(a, b, 1.0f) == b, "mix at 1 is B");
  check(mix(Params {}, Params {}, 0.37f) == Params {}, "a mix of neutral grades is neutral");
  {
    const Params m = mix(a, b, 0.5f);
    check(std::abs(m.anchor - 0.5f * (a.anchor + b.anchor)) < 1e-6f, "a mix interpolates the anchor");
    check(std::abs(m.shoulder - 0.5f * (a.shoulder + b.shoulder)) < 1e-6f, "a mix interpolates the shoulder");
  }

  // A blend converges, ends exactly on the target and changes only a little each frame.
  {
    Blend blend;
    blend.reset(a);
    blend.retarget(2.0f);
    float last = distance(blend.current(), b);
    Params previous = blend.current();
    float largestStep = 0.0f;
    for (int frame = 0; frame < 300; ++frame) {
      const Params& p = blend.step(b, 1.0f / 120.0f);
      const float d = distance(p, b);
      check(d <= last + 1e-7f, "a blend never moves away from its target");
      largestStep = std::max(largestStep, distance(p, previous));
      last = d;
      previous = p;
    }
    check(blend.current() == b, "a blend ends exactly on its target");
    check(!blend.blending() && blend.secondsLeft() == 0.0f, "a finished blend reports no time left");
    check(largestStep < 0.02f, "a blend moves a little each frame");
  }

  // A new target during a blend starts from where the blend is.
  {
    Blend blend;
    blend.reset(a);
    blend.retarget(1.0f);
    for (int frame = 0; frame < 60; ++frame) {
      blend.step(b, 1.0f / 120.0f);
    }
    const Params before = blend.current();
    const Params c = presets[2].params;
    blend.retarget(1.0f);
    const Params after = blend.step(c, 1.0f / 120.0f);
    check(distance(before, after) < 0.01f, "a new target during a blend does not jump");
    for (int frame = 0; frame < 200; ++frame) {
      blend.step(c, 1.0f / 120.0f);
    }
    check(blend.current() == c, "the second blend ends exactly on its target");
  }

  // A blend time of 0 shows the target at once.
  {
    Blend blend;
    blend.reset(a);
    blend.retarget(0.0f);
    check(blend.step(b, 0.0f) == b, "a blend time of 0 jumps");
  }

  if (g_failures != 0) {
    std::printf("%d failures\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("test_color_grade: all checks passed\n");
  return EXIT_SUCCESS;
}
