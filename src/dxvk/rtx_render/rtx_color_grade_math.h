#pragma once

// The colour grade's parameters, the built-in grades, blending and the constants the tone map shaders read.
// It has no runtime dependencies, so the unit test (tests/rtx/unit/test_color_grade.cpp) includes it as is.
// docs/RemixApi.md, "Colour grading", describes the fields for API users.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace dxvk {
namespace color_grade {

  using Mat3 = std::array<std::array<float, 3>, 3>;
  using Vec3 = std::array<float, 3>;

  // The grade runs on linear scene colour before the tone curve, in this order:
  //   color = matrix * color
  //   color = balance * color (per channel)
  //   color = lerp(luminance(color), color, saturation)
  //   color = 0.18 * (max(color, 0) / 0.18) ^ contrast (per channel, around mid gray)
  // The default values are the neutral grade, which leaves the color unchanged.
  // The anchor and the shoulder shape the tone curve after the grade (toneCurve below).
  struct Params {
    Mat3  matrix     { { { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } } };
    Vec3  balance    { 1.0f, 1.0f, 1.0f };
    float saturation = 1.0f;
    float contrast   = 1.0f;
    float anchor     = 0.18f;
    float shoulder   = 0.01f;

    bool operator==(const Params& o) const {
      return matrix == o.matrix && balance == o.balance && saturation == o.saturation && contrast == o.contrast &&
             anchor == o.anchor && shoulder == o.shoulder;
    }
    bool operator!=(const Params& o) const { return !(*this == o); }
  };

  // The BT.709 luminance weights the saturation control uses.
  constexpr Vec3 kLuminance { 0.2126f, 0.7152f, 0.0722f };

  // The contrast pivot is mid gray, so contrast leaves mid gray where it is. It does not move with the anchor.
  constexpr float kContrastPivot = 0.18f;

  // Slider and API ranges. Values outside are clamped.
  constexpr float kMatrixMin = -4.0f;
  constexpr float kMatrixMax = 4.0f;
  constexpr float kBalanceMax = 4.0f;
  constexpr float kSaturationMax = 4.0f;
  constexpr float kContrastMin = 0.1f;
  constexpr float kContrastMax = 4.0f;
  // The anchor stays below the lowest peak (1, SDR), so the shoulder always has a range to roll off in.
  constexpr float kAnchorMin = 0.01f;
  constexpr float kAnchorMax = 0.5f;
  // The curve rises monotonically for shoulders up to e / 2 (about 1.36), so 1 keeps a margin. Above 0 the
  // shoulder term cannot divide 0 by 0 when t * t underflows.
  constexpr float kShoulderMin = 0.001f;
  constexpr float kShoulderMax = 1.0f;

  inline Params clamped(Params p) {
    for (auto& row : p.matrix) {
      for (float& v : row) {
        v = std::isfinite(v) ? std::clamp(v, kMatrixMin, kMatrixMax) : 0.0f;
      }
    }
    for (float& v : p.balance) {
      v = std::isfinite(v) ? std::clamp(v, 0.0f, kBalanceMax) : 1.0f;
    }
    p.saturation = std::isfinite(p.saturation) ? std::clamp(p.saturation, 0.0f, kSaturationMax) : 1.0f;
    p.contrast = std::isfinite(p.contrast) ? std::clamp(p.contrast, kContrastMin, kContrastMax) : 1.0f;
    p.anchor = std::isfinite(p.anchor) ? std::clamp(p.anchor, kAnchorMin, kAnchorMax) : Params {}.anchor;
    p.shoulder = std::isfinite(p.shoulder) ? std::clamp(p.shoulder, kShoulderMin, kShoulderMax) : Params {}.shoulder;
    return p;
  }

  // The built-in grades.
  struct BuiltIn {
    const char* name;
    Params params;
  };

  inline Params warmAmber() {
    Params p;
    p.matrix = { { { 1.095973f, -0.059536f, 0.149696f },
                   { -0.028776f, 1.016891f, -0.009769f },
                   { 0.002389f, 0.008034f, 0.655857f } } };
    return p;
  }

  inline Params oliveGreen() {
    Params p;
    p.matrix = { { { 0.857235f, 0.192498f, 0.086365f },
                   { 0.033934f, 0.914132f, 0.056143f },
                   { 0.084339f, 0.283658f, 0.189434f } } };
    return p;
  }

  inline const std::array<BuiltIn, 3>& builtIns() {
    static const std::array<BuiltIn, 3> kBuiltIns { {
      { "None", Params {} },
      { "Warm Amber", warmAmber() },
      { "Olive Green", oliveGreen() },
    } };
    return kBuiltIns;
  }

  // Interpolates every field. At w = 0 and w = 1 the result is exactly a or b.
  // A lerp of two matrices keeps the identity when both are the identity, so neutral stays neutral.
  inline Params mix(const Params& a, const Params& b, float w) {
    if (!(w > 0.0f)) {
      return a;
    }
    if (w >= 1.0f) {
      return b;
    }
    const auto lerp = [w](float x, float y) { return x + (y - x) * w; };
    Params r;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r.matrix[i][j] = lerp(a.matrix[i][j], b.matrix[i][j]);
      }
      r.balance[i] = lerp(a.balance[i], b.balance[i]);
    }
    r.saturation = lerp(a.saturation, b.saturation);
    r.contrast = lerp(a.contrast, b.contrast);
    r.anchor = lerp(a.anchor, b.anchor);
    r.shoulder = lerp(a.shoulder, b.shoulder);
    return r;
  }

  // A blend from a start grade to a target over a time, eased with smoothstep so it starts and ends gently.
  class Blend {
  public:
    // Starts a blend from the current value. A blend time of 0 or less jumps to the target.
    void retarget(float seconds) {
      m_start = m_current;
      m_duration = std::max(seconds, 0.0f);
      m_elapsed = 0.0f;
    }

    // Jumps to `p` with no blend.
    void reset(const Params& p) {
      m_start = p;
      m_current = p;
      m_duration = 0.0f;
      m_elapsed = 0.0f;
    }

    // Advances by `deltaSeconds` towards `target`, which may change from call to call while a blend runs.
    const Params& step(const Params& target, float deltaSeconds) {
      if (std::isfinite(deltaSeconds) && deltaSeconds > 0.0f) {
        m_elapsed += deltaSeconds;
      }
      if (m_elapsed >= m_duration) {
        m_current = target;
        m_start = target;
      } else {
        const float t = m_elapsed / m_duration;
        m_current = mix(m_start, target, t * t * (3.0f - 2.0f * t));
      }
      return m_current;
    }

    const Params& current() const { return m_current; }
    bool blending() const { return m_elapsed < m_duration; }
    float secondsLeft() const { return blending() ? m_duration - m_elapsed : 0.0f; }

  private:
    Params m_start;
    Params m_current;
    float m_duration = 0.0f;
    float m_elapsed = 0.0f;
  };

  // The constants the tone map shaders read: the matrix, balance and saturation folded into one matrix (rows),
  // the contrast, and the tone curve's anchor and shoulder. With balance 1 and saturation 1 the folded matrix is
  // exactly `matrix`.
  struct Gpu {
    Mat3 rows;
    float contrast;
    float anchor;
    float shoulder;
  };

  inline Mat3 multiply(const Mat3& a, const Mat3& b) {
    Mat3 r {};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
      }
    }
    return r;
  }

  inline Gpu toGpu(const Params& p) {
    // The saturation matrix: s * I + (1 - s) * (each row = the luminance weights).
    Mat3 saturation {};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        saturation[i][j] = (i == j ? p.saturation : 0.0f) + (1.0f - p.saturation) * kLuminance[j];
      }
    }
    Mat3 balance {};
    for (int i = 0; i < 3; ++i) {
      balance[i][i] = p.balance[i];
    }
    Gpu g;
    g.rows = multiply(saturation, multiply(balance, p.matrix));
    g.contrast = p.contrast;
    g.anchor = p.anchor;
    g.shoulder = p.shoulder;
    return g;
  }

  // Applies the grade to one colour as the shader does (tonemap_operator.slangh, applyGrade).
  inline Vec3 apply(const Gpu& g, const Vec3& c) {
    Vec3 r {};
    for (int i = 0; i < 3; ++i) {
      r[i] = g.rows[i][0] * c[0] + g.rows[i][1] * c[1] + g.rows[i][2] * c[2];
    }
    if (g.contrast != 1.0f) {
      for (float& v : r) {
        v = kContrastPivot * std::pow(std::max(v, 0.0f) / kContrastPivot, g.contrast);
      }
    }
    return r;
  }

  // The tone curve for one channel, as the shader computes it (tonemap_operator.slangh, reinhardPiecewiseSmooth).
  // Values up to the anchor pass through. Above it a smooth Reinhard shoulder rolls off towards `peak`.
  // A larger shoulder keeps the curve straight further above the anchor and bends it later and harder.
  inline float toneCurve(float x, float peak, float anchor, float shoulder) {
    if (x <= anchor) {
      return x;
    }
    const float range = peak - anchor;
    const float t = (x - anchor) / range;
    const float s = std::exp(-shoulder / (t * t));
    return anchor + range * t / (1.0f + t * s);
  }

}
}
