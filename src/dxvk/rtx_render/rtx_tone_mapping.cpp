/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_pre_exposure.h"
#include "rtx_tone_mapping.h"
#include "rtx_dlss_neural_rendering.h"
#include "dxvk_device.h"
#include "dxvk_scoped_annotation.h"
#include "rtx_render/rtx_shader_manager.h"
#include "rtx.h"
#include "rtx/pass/tonemap/tonemapping.h"
#include "rtx_fork_hooks.h"
#include "rtx_color_grade.h"
#include "../../util/util_global_time.h"

#include <rtx_shaders/tonemapping_apply_tonemapping.h>
#include <rtx_shaders/tonemapping_fast_tonemapping.h>
#include <rtx_shaders/tonemapping_inverse_tonemapping.h>
#include "rtx_imgui.h"
#include "rtx/utility/debug_view_indices.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>

namespace dxvk {
  // Defined within an unnamed namespace to ensure unique definition across binary.
  namespace {
    class ApplyTonemappingShader : public ManagedShader
    {
      SHADER_SOURCE(ApplyTonemappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_apply_tonemapping)

      PUSH_CONSTANTS(ToneMappingApplyToneMappingArgs)

      BEGIN_PARAMETER()
        TEXTURE2DARRAY(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)
      END_PARAMETER()
    };

    class FastTonemappingShader : public ManagedShader {
      SHADER_SOURCE(FastTonemappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_fast_tonemapping)

      PUSH_CONSTANTS(FastToneMappingArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)
      END_PARAMETER()
    };

    class InverseTonemappingShader : public ManagedShader {
      SHADER_SOURCE(InverseTonemappingShader, VK_SHADER_STAGE_COMPUTE_BIT, tonemapping_inverse_tonemapping)

      PUSH_CONSTANTS(FastToneMappingArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT)
        RW_TEXTURE1D_READONLY(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT)
        RW_TEXTURE2D(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(ApplyTonemappingShader);
  }

  namespace {
    std::atomic<bool> s_hdrOutputActive { false };

    // A CPU mirror of tonemap() in tonemap_operator.slangh for a neutral gray, for the tone curve chart. The
    // primaries must change with the shader's. The grade is color_grade::apply, the curve color_grade::toneCurve.
    namespace tone_curve {
      using Mat3 = std::array<std::array<float, 3>, 3>;
      using Color = std::array<float, 3>;

      // The chart's input axis counts stops from mid gray.
      constexpr float kMidGray = 0.18f;

      constexpr Mat3 kPrimaries1 { { { 0.726346493f, 0.166654825f, 0.106998682f },
                                     { 0.134236977f, 0.700138867f, 0.165624142f },
                                     { 0.072639547f, 0.060120508f, 0.867239952f } } };
      constexpr Mat3 kPrimariesInverse1 { { { 1.450212359f, -0.335329264f, -0.114883065f },
                                            { -0.253468722f, 1.510707617f, -0.257238865f },
                                            { -0.103896290f, -0.076639965f, 1.180536270f } } };
      constexpr Mat3 kPrimaries2 { { { 0.737337414f, 0.150990499f, 0.111672185f },
                                     { 0.310830779f, 0.413497919f, 0.275671278f },
                                     { 0.129649707f, 0.116560894f, 0.753789354f } } };
      constexpr Mat3 kPrimariesInverse2 { { { 1.604745378f, -0.578613002f, -0.026132549f },
                                            { -1.139794955f, 3.107331527f, -0.967536430f },
                                            { -0.099761623f, -0.380976904f, 1.480738594f } } };

      Color multiply(const Mat3& m, const Color& v) {
        return { m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                 m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                 m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] };
      }

      // Returns the BT.709 luminance of tonemap() for a gray of `exposed` in every channel.
      // Without the shoulder, only the grade is applied.
      float toneMapNeutral(float exposed, float peak, uint32_t primaries, const color_grade::Gpu& grade, bool withShoulder) {
        Color color = color_grade::apply(grade, { exposed, exposed, exposed });
        if (withShoulder) {
          const Mat3* basis = nullptr;
          const Mat3* inverse = nullptr;
          if (primaries == tonemapPrimariesCustom1) {
            basis = &kPrimaries1;
            inverse = &kPrimariesInverse1;
          } else if (primaries == tonemapPrimariesCustom2) {
            basis = &kPrimaries2;
            inverse = &kPrimariesInverse2;
          }
          if (basis != nullptr) {
            color = multiply(*basis, color);
          }
          for (float& c : color) {
            c = color_grade::toneCurve(c, peak, grade.anchor, grade.shoulder);
          }
          if (inverse != nullptr) {
            color = multiply(*inverse, color);
          }
        }
        return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
      }
    }

    // Draws dashes along the polyline through points. The dash pattern continues across segments.
    void addDashedPolyline(ImDrawList* drawList, const ImVec2* points, int count, ImU32 color, float thickness) {
      constexpr float kDash = 6.f;
      constexpr float kPeriod = 10.f;
      float phase = 0.f;
      for (int i = 0; i + 1 < count; ++i) {
        const ImVec2 a = points[i];
        const ImVec2 b = points[i + 1];
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float length = std::sqrt(dx * dx + dy * dy);
        if (!std::isfinite(length)) {
          continue;
        }
        float t = 0.f;
        while (t < length) {
          const bool on = phase < kDash;
          const float step = std::min(length - t, (on ? kDash : kPeriod) - phase);
          if (on) {
            drawList->AddLine(ImVec2(a.x + dx * (t / length), a.y + dy * (t / length)),
                              ImVec2(a.x + dx * ((t + step) / length), a.y + dy * ((t + step) / length)),
                              color, thickness);
          }
          t += step;
          phase += step;
          if (phase >= kPeriod) {
            phase -= kPeriod;
          }
        }
      }
    }

    // The tone curve chart's handles. Each one edits a single setting.
    enum class CurveHandle : int {
      None = 0,
      Exposure,
      Contrast,
      Anchor,
      Shoulder,
      Peak,
    };

    // The input, in stops from mid gray, where the contrast handle sits on the curve. It is below the default
    // anchor, where the shoulder does not act, so the curve there moves with the grade alone.
    constexpr float kContrastHandleStops = -3.f;
    // The input where the peak handle sits on the peak line, far enough right that the curve is close to it.
    constexpr float kPeakHandleStops = 10.f;
    constexpr float kHandleRadius = 4.5f;
    constexpr float kHandleGrabRadius = 8.f;
    // The range of the Global Exposure control.
    constexpr float kExposureMin = -4.f;
    constexpr float kExposureMax = 4.f;

    // Returns the contrast that puts the chart's curve at `target` at the contrast handle's input.
    // Below mid gray a higher contrast gives a darker value, so a bisection finds it.
    float contrastThrough(float target, color_grade::Params grade, float peak, uint32_t primaries) {
      const float input = tone_curve::kMidGray * std::exp2(kContrastHandleStops);
      float low = color_grade::kContrastMin;
      float high = color_grade::kContrastMax;
      for (int i = 0; i < 32; ++i) {
        grade.contrast = 0.5f * (low + high);
        if (tone_curve::toneMapNeutral(input, peak, primaries, color_grade::toGpu(grade), true) > target) {
          low = grade.contrast;
        } else {
          high = grade.contrast;
        }
      }
      return 0.5f * (low + high);
    }

    // Returns the input, in stops from mid gray, at which the chart's curve reaches `target`.
    // The curve rises with its input, so a bisection finds it.
    float inputStopsFor(float target, float peak, uint32_t primaries, const color_grade::Gpu& grade) {
      float low = -16.f;
      float high = 20.f;
      for (int i = 0; i < 32; ++i) {
        const float mid = 0.5f * (low + high);
        if (tone_curve::toneMapNeutral(tone_curve::kMidGray * std::exp2(mid), peak, primaries, grade, true) < target) {
          low = mid;
        } else {
          high = mid;
        }
      }
      return 0.5f * (low + high);
    }

    // The shoulder handle sits on the knee, where t = (x - anchor) / (peak - anchor) is the square root of the
    // shoulder, so the shoulder term there is exp(-1).
    // Returns the fraction of the way from the anchor to the peak that the curve's output has reached at the knee.
    float kneeOutputFraction(float shoulder) {
      const float u = std::sqrt(shoulder);
      return u / (1.f + u * std::exp(-1.f));
    }

    // Inverts kneeOutputFraction, within the shoulder's range.
    float shoulderForKnee(float fraction) {
      const float low = kneeOutputFraction(color_grade::kShoulderMin);
      const float high = kneeOutputFraction(color_grade::kShoulderMax);
      const float g = std::clamp(fraction, low, high);
      const float u = g / (1.f - g * std::exp(-1.f));
      return std::clamp(u * u, color_grade::kShoulderMin, color_grade::kShoulderMax);
    }
  }

  // Plots the neutral tone curve that reaches the screen, against input stops from mid gray (0.18).
  // The output axis is the display-linear value before the output encode, on a log10 scale.
  // Handles on the chart edit the exposure, the edited grade preset's contrast, anchor and shoulder and, with HDR
  // output, the peak.
  void DxvkToneMapping::showToneCurve() {
    if (!ImGui::TreeNode("Tone Curve")) {
      return;
    }

    const bool hdr = hdrOutputActive();
    const float peak = tonemapPeak();
    const float paperWhite = std::clamp(hdrPaperWhiteNits(), 80.0f, 1000.0f);
    const uint32_t primaries = static_cast<uint32_t>(DxvkToneMapping::primaries());
    const color_grade::Gpu grade = color_grade::toGpu(ColorGrade::current());
    color_grade::Params edited;
    bool editedBuiltIn = false;
    const bool hasEdited = ColorGrade::editedPreset(edited, editedBuiltIn);

    // The drag state lives in ImGui's storage. The output axis stays fixed while a handle is held, so that
    // dragging the peak does not rescale the chart under the cursor.
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const ImGuiID activeKey = ImGui::GetID("activeHandle");
    const ImGuiID offsetKey = ImGui::GetID("grabOffset");
    const ImGuiID movedKey = ImGui::GetID("handleMoved");
    const ImGuiID decadeKey = ImGui::GetID("heldMaxDecade");
    CurveHandle active = static_cast<CurveHandle>(storage->GetInt(activeKey, 0));

    constexpr int kSamples = 192;
    constexpr float kMinStops = -8.f;
    constexpr float kMaxStops = 12.f;
    constexpr float kMinDecade = -4.f;
    const float maxDecade = active != CurveHandle::None
      ? storage->GetFloat(decadeKey, 0.3f)
      : std::log10(std::max(peak, 1.f)) + 0.3f;

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(ImGui::GetContentRegionAvail().x, 240.f);
    const float height = 220.f;
    const float axisLeft = ImGui::CalcTextSize("0.0001").x + 8.f;
    const float axisBottom = ImGui::GetTextLineHeight() + 4.f;
    ImGui::InvisibleButton("##toneCurve", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    const bool held = ImGui::IsItemActive();
    const bool pressed = ImGui::IsItemActivated();

    const ImVec2 plotMin(origin.x + axisLeft, origin.y + 4.f);
    const ImVec2 plotMax(origin.x + width - 12.f, origin.y + height - axisBottom);
    auto toX = [&](float stops) {
      return plotMin.x + (stops - kMinStops) / (kMaxStops - kMinStops) * (plotMax.x - plotMin.x);
    };
    auto toY = [&](float value) {
      float decade = value > 0.f && std::isfinite(value) ? std::log10(value) : kMinDecade - 1.f;
      decade = std::clamp(decade, kMinDecade - 1.f, maxDecade + 1.f);
      return plotMax.y - (decade - kMinDecade) / (maxDecade - kMinDecade) * (plotMax.y - plotMin.y);
    };
    auto stopsAt = [&](float x) {
      return kMinStops + (x - plotMin.x) / (plotMax.x - plotMin.x) * (kMaxStops - kMinStops);
    };
    auto valueAt = [&](float y) {
      return std::pow(10.f, kMinDecade + (plotMax.y - y) / (plotMax.y - plotMin.y) * (maxDecade - kMinDecade));
    };

    // The handles: where each one is drawn, whether it shows, and whether it can be dragged.
    // The peak is 1 without HDR output, so its handle shows only with HDR output.
    struct Handle {
      CurveHandle id;
      ImVec2 pos;
      bool visible;
      bool editable;
    };
    const float contrastInput = tone_curve::kMidGray * std::exp2(kContrastHandleStops);
    const bool gradeEditable = hasEdited && !editedBuiltIn;
    auto onCurve = [&](float stops, float value) {
      return ImVec2(std::clamp(toX(stops), plotMin.x, plotMax.x), std::clamp(toY(value), plotMin.y, plotMax.y));
    };
    const float kneeOutput = grade.anchor + (peak - grade.anchor) * kneeOutputFraction(grade.shoulder);
    const std::array<Handle, 5> handles { {
      { CurveHandle::Exposure, ImVec2(toX(std::clamp(exposureBias(), kExposureMin, kExposureMax)), plotMax.y), true, true },
      { CurveHandle::Contrast,
        onCurve(kContrastHandleStops, tone_curve::toneMapNeutral(contrastInput, peak, primaries, grade, true)),
        hasEdited, gradeEditable },
      { CurveHandle::Anchor, onCurve(inputStopsFor(grade.anchor, peak, primaries, grade), grade.anchor), hasEdited, gradeEditable },
      { CurveHandle::Shoulder, onCurve(inputStopsFor(kneeOutput, peak, primaries, grade), kneeOutput), hasEdited, gradeEditable },
      { CurveHandle::Peak, ImVec2(toX(kPeakHandleStops), std::clamp(toY(peak), plotMin.y, plotMax.y)), hdr, true },
    } };
    auto findHandle = [&](CurveHandle id) -> const Handle& {
      return handles[static_cast<int>(id) - 1];
    };

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    CurveHandle pointed = CurveHandle::None;
    if (active != CurveHandle::None) {
      pointed = active;
    } else if (hovered) {
      float nearest = kHandleGrabRadius * kHandleGrabRadius;
      for (const Handle& h : handles) {
        const float dx = mouse.x - h.pos.x;
        const float dy = mouse.y - h.pos.y;
        if (h.visible && dx * dx + dy * dy <= nearest) {
          nearest = dx * dx + dy * dy;
          pointed = h.id;
        }
      }
    }
    const bool pointedEditable = pointed != CurveHandle::None && findHandle(pointed).editable;

    // A double-click resets the handle's setting. A press starts a drag that keeps the grab offset, so the
    // value does not jump to the cursor.
    if (hovered && pointedEditable && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      switch (pointed) {
      case CurveHandle::Exposure:
        RemixGui::CheckRtxOptionPopups(&exposureBiasObject());
        exposureBiasObject().resetToDefault();
        break;
      case CurveHandle::Contrast:
        edited.contrast = color_grade::Params {}.contrast;
        ColorGrade::editPreset(edited);
        break;
      case CurveHandle::Anchor:
        edited.anchor = color_grade::Params {}.anchor;
        ColorGrade::editPreset(edited);
        break;
      case CurveHandle::Shoulder:
        edited.shoulder = color_grade::Params {}.shoulder;
        ColorGrade::editPreset(edited);
        break;
      case CurveHandle::Peak:
        RemixGui::CheckRtxOptionPopups(&hdrPeakNitsObject());
        hdrPeakNitsObject().resetToDefault();
        break;
      default:
        break;
      }
      active = CurveHandle::None;
    } else if (pressed && pointedEditable) {
      active = pointed;
      const Handle& h = findHandle(pointed);
      storage->SetFloat(offsetKey, pointed == CurveHandle::Exposure ? h.pos.x - mouse.x : h.pos.y - mouse.y);
      storage->SetBool(movedKey, false);
      storage->SetFloat(decadeKey, maxDecade);
    }
    if (!held) {
      active = CurveHandle::None;
    }

    if (active != CurveHandle::None) {
      const ImVec2 dragged = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
      const bool moved = storage->GetBool(movedKey, false) || dragged.x != 0.f || dragged.y != 0.f;
      storage->SetBool(movedKey, moved);
      const float offset = storage->GetFloat(offsetKey, 0.f);
      if (moved) {
        switch (active) {
        case CurveHandle::Exposure: {
          const float bias = std::clamp(stopsAt(mouse.x + offset), kExposureMin, kExposureMax);
          if (bias != exposureBias()) {
            RemixGui::CheckRtxOptionPopups(&exposureBiasObject());
            exposureBiasObject().setDeferred(bias);
          }
          break;
        }
        case CurveHandle::Contrast: {
          const float contrast = contrastThrough(valueAt(mouse.y + offset), edited, peak, primaries);
          if (contrast != edited.contrast) {
            edited.contrast = contrast;
            ColorGrade::editPreset(edited);
          }
          break;
        }
        case CurveHandle::Anchor: {
          const float anchor = std::clamp(valueAt(mouse.y + offset), color_grade::kAnchorMin, color_grade::kAnchorMax);
          if (anchor != edited.anchor) {
            edited.anchor = anchor;
            ColorGrade::editPreset(edited);
          }
          break;
        }
        case CurveHandle::Shoulder: {
          const float shoulder = shoulderForKnee((valueAt(mouse.y + offset) - edited.anchor) / (peak - edited.anchor));
          if (shoulder != edited.shoulder) {
            edited.shoulder = shoulder;
            ColorGrade::editPreset(edited);
          }
          break;
        }
        case CurveHandle::Peak: {
          const float nits = std::clamp(valueAt(mouse.y + offset) * paperWhite, std::max(paperWhite, 80.f), 10000.f);
          if (nits != hdrPeakNits()) {
            RemixGui::CheckRtxOptionPopups(&hdrPeakNitsObject());
            hdrPeakNitsObject().setDeferred(nits);
          }
          break;
        }
        default:
          break;
        }
      }
    }
    storage->SetInt(activeKey, static_cast<int>(active));

    if (pointed != CurveHandle::None) {
      if (!findHandle(pointed).editable) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_NotAllowed);
      } else {
        ImGui::SetMouseCursor(pointed == CurveHandle::Exposure ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
      }
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImU32 gridColor = ImGui::GetColorU32(ImGuiCol_Border);
    const ImU32 labelColor = ImGui::GetColorU32(ImGuiCol_TextDisabled);
    const ImU32 activeColor = ImGui::GetColorU32(ImGuiCol_SliderGrabActive);
    const ImU32 mutedColor = ImGui::GetColorU32(ImGuiCol_TextDisabled, 0.8f);
    const ImU32 handleColor = ImGui::GetColorU32(ImGuiCol_CheckMark);
    const ImU32 handleReadOnlyColor = ImGui::GetColorU32(ImGuiCol_CheckMark, 0.5f);
    const ImU32 handleOutlineColor = ImGui::GetColorU32(ImGuiCol_Text);
    drawList->AddRectFilled(plotMin, plotMax, ImGui::GetColorU32(ImGuiCol_FrameBg));

    // Stop grid with labels below the plot. The 0-stop line marks mid gray.
    for (int stops = static_cast<int>(kMinStops); stops <= static_cast<int>(kMaxStops); stops += 2) {
      const float x = toX(static_cast<float>(stops));
      drawList->AddLine(ImVec2(x, plotMin.y), ImVec2(x, plotMax.y), gridColor, stops == 0 ? 2.f : 1.f);
      char label[8];
      std::snprintf(label, sizeof(label), stops > 0 ? "+%d" : "%d", stops);
      const float labelWidth = ImGui::CalcTextSize(label).x;
      drawList->AddText(ImVec2(x - labelWidth * 0.5f, plotMax.y + 2.f), labelColor, label);
    }

    // Decade grid with labels left of the plot.
    for (int decade = static_cast<int>(kMinDecade); decade <= static_cast<int>(std::floor(maxDecade)); ++decade) {
      const float y = toY(std::pow(10.f, static_cast<float>(decade)));
      drawList->AddLine(ImVec2(plotMin.x, y), ImVec2(plotMax.x, y), gridColor, 1.f);
      char label[16];
      std::snprintf(label, sizeof(label), "%g", std::pow(10.0, decade));
      const ImVec2 labelSize = ImGui::CalcTextSize(label);
      drawList->AddText(ImVec2(plotMin.x - labelSize.x - 4.f, y - labelSize.y * 0.5f), labelColor, label);
    }

    std::array<ImVec2, kSamples> curvePoints;
    std::array<ImVec2, kSamples> unshoulderedPoints;
    for (int i = 0; i < kSamples; ++i) {
      const float stops = kMinStops + (kMaxStops - kMinStops) * static_cast<float>(i) / static_cast<float>(kSamples - 1);
      const float input = tone_curve::kMidGray * std::exp2(stops);
      const float x = toX(stops);
      curvePoints[i] = ImVec2(x, toY(tone_curve::toneMapNeutral(input, peak, primaries, grade, true)));
      unshoulderedPoints[i] = ImVec2(x, toY(tone_curve::toneMapNeutral(input, peak, primaries, grade, false)));
    }

    const float midGrayOnScreen = tone_curve::toneMapNeutral(tone_curve::kMidGray, peak, primaries, grade, true);

    drawList->PushClipRect(plotMin, plotMax, true);
    addDashedPolyline(drawList, unshoulderedPoints.data(), kSamples, mutedColor, 1.5f);
    const float peakY = toY(peak);
    const ImVec2 peakLine[2] = { ImVec2(plotMin.x, peakY), ImVec2(plotMax.x, peakY) };
    addDashedPolyline(drawList, peakLine, 2, mutedColor, 1.f);

    // The exposure handle's line marks the input that mid gray lands on after the exposure.
    const float exposureX = findHandle(CurveHandle::Exposure).pos.x;
    const ImVec2 exposureLine[2] = { ImVec2(exposureX, plotMin.y), ImVec2(exposureX, plotMax.y) };
    addDashedPolyline(drawList, exposureLine, 2, handleReadOnlyColor, 1.f);

    for (int i = 0; i + 1 < kSamples; ++i) {
      drawList->AddLine(curvePoints[i], curvePoints[i + 1], activeColor, 2.5f);
    }
    drawList->AddCircleFilled(ImVec2(toX(0.f), toY(midGrayOnScreen)), 3.f, activeColor);

    const bool cursorInPlot = hovered && mouse.x >= plotMin.x && mouse.x <= plotMax.x;
    if (cursorInPlot && active == CurveHandle::None) {
      drawList->AddLine(ImVec2(mouse.x, plotMin.y), ImVec2(mouse.x, plotMax.y), labelColor, 1.f);
    }
    drawList->PopClipRect();

    // The exposure handle is a triangle on the input axis, the others are dots. A read-only handle is a ring.
    for (const Handle& h : handles) {
      if (!h.visible) {
        continue;
      }
      const bool highlighted = h.id == pointed;
      const float r = highlighted ? kHandleRadius + 1.5f : kHandleRadius;
      if (h.id == CurveHandle::Exposure) {
        const ImVec2 a(h.pos.x, h.pos.y - r * 1.8f);
        const ImVec2 b(h.pos.x - r * 1.2f, h.pos.y);
        const ImVec2 c(h.pos.x + r * 1.2f, h.pos.y);
        drawList->AddTriangleFilled(a, b, c, handleColor);
        if (highlighted) {
          drawList->AddTriangle(a, b, c, handleOutlineColor, 1.f);
        }
      } else if (!h.editable) {
        drawList->AddCircle(h.pos, r, handleReadOnlyColor, 0, 2.f);
      } else {
        drawList->AddCircleFilled(h.pos, r, handleColor);
        if (highlighted) {
          drawList->AddCircle(h.pos, r, handleOutlineColor, 0, 1.f);
        }
      }
    }

    // The readout. Over a handle, or while dragging one, it starts with the handle's setting.
    if (pointed != CurveHandle::None || cursorInPlot) {
      ImGui::BeginTooltip();
      switch (pointed) {
      case CurveHandle::Exposure:
        ImGui::Text("Global Exposure: %+.2f stops", exposureBias());
        ImGui::TextDisabled("Mid gray lands on this input. Drag sideways to change it.");
        break;
      case CurveHandle::Contrast:
        ImGui::Text("Contrast: %.3f (grade preset %s)", edited.contrast, ColorGrade::gradePreset().c_str());
        break;
      case CurveHandle::Anchor:
        ImGui::Text("Anchor: %.3f (grade preset %s)", edited.anchor, ColorGrade::gradePreset().c_str());
        ImGui::TextDisabled("Values below the anchor pass through. The shoulder starts here.");
        break;
      case CurveHandle::Shoulder:
        ImGui::Text("Shoulder: %.4f (grade preset %s)", edited.shoulder, ColorGrade::gradePreset().c_str());
        ImGui::TextDisabled("The dot marks the knee. Higher keeps the curve straight longer and bends it later and harder.");
        break;
      case CurveHandle::Peak:
        ImGui::Text("Peak Brightness: %.0f nits (%.2f x paper white)", hdrPeakNits(), peak);
        ImGui::TextDisabled("Drag up or down to change it.");
        break;
      default:
        break;
      }
      if (pointed == CurveHandle::Contrast || pointed == CurveHandle::Anchor || pointed == CurveHandle::Shoulder) {
        if (editedBuiltIn) {
          ImGui::TextDisabled("A built-in preset is read-only. Save As in Color Grade makes an editable copy.");
        } else {
          ImGui::TextDisabled("Drag up or down to change it. The preset shows as modified until Save.");
        }
      }
      if (pointedEditable) {
        ImGui::TextDisabled("Double-click to reset it to its default.");
      }
      if (cursorInPlot && active == CurveHandle::None) {
        if (pointed != CurveHandle::None) {
          ImGui::Separator();
        }
        const float stops = stopsAt(mouse.x);
        const float input = tone_curve::kMidGray * std::exp2(stops);
        const float toned = tone_curve::toneMapNeutral(input, peak, primaries, grade, true);
        ImGui::Text("Input: %+.2f stops (%.4g)", stops, input);
        if (hdr) {
          ImGui::Text("Tone curve: %.4g (%.0f nits)", toned, toned * paperWhite);
        } else {
          ImGui::Text("Tone curve: %.4g", toned);
        }
        ImGui::Text("No shoulder: %.4g", tone_curve::toneMapNeutral(input, peak, primaries, grade, false));
      }
      ImGui::EndTooltip();
    }

    if (hdr) {
      ImGui::Text("Peak %.3f (%.0f nits)   Mid gray %.3f -> %.3f", peak, peak * paperWhite, tone_curve::kMidGray, midGrayOnScreen);
    } else {
      ImGui::Text("Peak %.3f   Mid gray %.3f -> %.3f", peak, tone_curve::kMidGray, midGrayOnScreen);
    }
    ImGui::Text("Anchor %.3f   Shoulder %.4f", grade.anchor, grade.shoulder);
    ImGui::TextDisabled("Solid: the tone curve with the grade and primaries. Dashed: the grade only, no shoulder.");
    ImGui::TextDisabled(hdr
      ? "X: stops from mid gray (0.18). Y: display-linear output before the output encode, log scale, 1 = paper white."
      : "X: stops from mid gray (0.18). Y: display-linear output before the output encode, log scale, 1 = white.");
    ImGui::TextDisabled(hdr
      ? "Handles: the triangle sets the exposure. The dots on the curve set the grade's contrast, anchor and shoulder, left "
        "to right, and the dot on the peak line the peak brightness."
      : "Handles: the triangle sets the exposure. The dots on the curve set the grade's contrast, anchor and shoulder, left "
        "to right.");
    ImGui::TreePop();
  }

  namespace {
    // The grade on screen this frame, as shader constants.
    ToneMappingGradeArgs gradeArgs(Rc<RtxContext>& ctx) {
      const color_grade::Gpu g = ColorGrade::frameConstants(ctx->getDevice()->getCurrentFrameId(),
                                                            GlobalTime::get().deltaTimeMs() * 0.001f);
      ToneMappingGradeArgs args;
      args.row0 = vec4(g.rows[0][0], g.rows[0][1], g.rows[0][2], g.contrast);
      args.row1 = vec4(g.rows[1][0], g.rows[1][1], g.rows[1][2], g.anchor);
      args.row2 = vec4(g.rows[2][0], g.rows[2][1], g.rows[2][2], g.shoulder);
      return args;
    }
  }

  bool DxvkToneMapping::hdrOutputActive() {
    return s_hdrOutputActive.load(std::memory_order_relaxed);
  }

  void DxvkToneMapping::setHdrOutputActive(bool active) {
    if (s_hdrOutputActive.exchange(active, std::memory_order_relaxed) != active) {
      Logger::info(str::format("[HDR] HDR10 output ", active ? "active" : "inactive"));
    }
  }

  float DxvkToneMapping::tonemapPeak() {
    if (!hdrOutputActive()) {
      return 1.0f;
    }
    const float paperWhite = std::clamp(hdrPaperWhiteNits(), 80.0f, 1000.0f);
    const float peak = std::clamp(hdrPeakNits(), 80.0f, 10000.0f);
    return std::max(peak / paperWhite, 1.0f);
  }

  float DxvkToneMapping::hdrUiNitsClamped() {
    return std::clamp(hdrUiNits(), 80.0f, 1000.0f);
  }

  float DxvkToneMapping::hdrSceneToBackBufferScale() {
    return std::clamp(hdrPaperWhiteNits(), 80.0f, 1000.0f) / hdrUiNitsClamped();
  }

  DxvkToneMapping::DxvkToneMapping(DxvkDevice* device)
  : CommonDeviceObject(device), m_vkd(device->vkd()) { }

  DxvkToneMapping::~DxvkToneMapping() { }

  void DxvkToneMapping::showEffectSettings() {
    RemixGui::DragFloat("Global Exposure", &exposureBiasObject(), 0.01f, -4.f, 4.f);
    RemixGui::Combo("Primaries", &primariesObject(), "Custom Primaries 1\0Custom Primaries 2\0BT.709\0");

    if (RemixGui::CollapsingHeader("Colour Grade", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Indent();
      ColorGrade::showImguiSettings();
      ImGui::Unindent();
    }

    // HDR output follows Windows on its own; rtx.tonemap.hdrEnabled only allows it, from the user menu's Other
    // Settings. The HDR controls exist only while HDR output is running: in SDR the curve peaks at 1 and none apply.
    if (hdrOutputActive()) {
      RemixGui::DragFloat("Peak Brightness", &hdrPeakNitsObject(), 1.0f, 80.0f, 10000.0f, "%.0f nits", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("Paper White", &hdrPaperWhiteNitsObject(), 1.0f, 80.0f, 1000.0f, "%.0f nits", ImGuiSliderFlags_AlwaysClamp);
      RemixGui::DragFloat("UI Brightness", &hdrUiNitsObject(), 1.0f, 80.0f, 1000.0f, "%.0f nits", ImGuiSliderFlags_AlwaysClamp);
    }
    showToneCurve();

    // Grey the dependent controls rather than removing them: the enable toggle
    // sits on the stack row header, so hiding the body made an expanded but
    // disabled section collapse to an empty indent.
    ImGui::Indent();
    ImGui::BeginDisabled(!tonemappingEnabled());
    RemixGui::Combo("Dither Mode", &ditherModeObject(), "Disabled\0Spatial\0Spatial + Temporal\0");
    ImGui::EndDisabled();

    RemixGui::Separator();
    ImGui::Unindent();
  }

  void DxvkToneMapping::dispatchApplyToneMapping(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::Resource& inputBuffer,
    const Resources::Resource& colorBuffer,
    bool performSRGBConversion,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Apply Tone Mapping");

    const VkExtent3D workgroups = util::computeBlockCount(colorBuffer.view->imageInfo().extent, VkExtent3D{ 16 , 16, 1 });

    // Prepare shader arguments.
    ToneMappingApplyToneMappingArgs pushArgs = {};
    pushArgs.toneMappingEnabled    = tonemappingEnabled();
    pushArgs.performSRGBConversion = performSRGBConversion;
    pushArgs.enableAutoExposure    = autoExposureEnabled;
    // / P: the image is pre-exposed (rtx_pre_exposure.h); the exposure texture stays physical.
    pushArgs.exposureFactor        = exp2f(exposureBias()) * PreExposure::inverse(); // ev100
    pushArgs.tonemapPeak           = tonemapPeak();
    pushArgs.hdrOutput             = hdrOutputActive() ? 1u : 0u;
    pushArgs.tonemapPrimaries      = static_cast<uint32_t>(primaries());
    pushArgs.grade                 = gradeArgs(ctx);

    // Dither args.
    switch (ditherMode()) {
    case DitherMode::None:            pushArgs.ditherMode = ditherModeNone;            break;
    case DitherMode::Spatial:         pushArgs.ditherMode = ditherModeSpatialOnly;     break;
    case DitherMode::SpatialTemporal: pushArgs.ditherMode = ditherModeSpatialTemporal; break;
    }
    pushArgs.frameIndex = ctx->getDevice()->getCurrentFrameId();

    ctx->bindResourceView(TONEMAPPING_APPLY_BLUE_NOISE_TEXTURE_INPUT, ctx->getResourceManager().getBlueNoiseTexture(ctx), nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT, inputBuffer.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT, colorBuffer.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, ApplyTonemappingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatch(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    bool performSRGBConversion,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Tone Mapping");

    ctx->setPushConstantBank(DxvkPushConstantBank::RTX);

    const Resources::Resource& inputColorBuffer = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
    dispatchApplyToneMapping(ctx, exposureView, inputColorBuffer,
                             rtOutput.m_finalOutput.resource(Resources::AccessType::Write),
                             performSRGBConversion, autoExposureEnabled);
  }

  void DxvkToneMapping::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    const auto& dlssNeuralRendering = m_device->getCommon()->metaDlssNeuralRendering();
    if (DlssNeuralRendering::enable() && dlssNeuralRendering.supportsDlssNeuralRendering()) {
      FastTonemappingShader::getShader();
      InverseTonemappingShader::getShader();
    }

    // The apply pass is always prewarmed.
    ApplyTonemappingShader::getShader();
  }

  void DxvkToneMapping::dispatchFastToneMapping(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Fast Tone Mapping");

    // Prepare shader arguments
    FastToneMappingArgs pushArgs = {};
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.exposureFactor = exp2f(exposureBias()) * PreExposure::inverse(); // ev100, / P
    pushArgs.tonemapPrimaries = static_cast<uint32_t>(primaries());
    pushArgs.grade = gradeArgs(ctx);

    const Resources::Resource& inputColorTexture = rtOutput.m_finalOutput.resource(Resources::AccessType::Read);
    const VkExtent3D workgroups = util::computeBlockCount(inputColorTexture.view->imageInfo().extent, VkExtent3D { 16 , 16, 1 });

    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT, inputColorTexture.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT, rtOutput.m_neuralRenderingInput.resource(Resources::AccessType::Write).view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, FastTonemappingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }

  void DxvkToneMapping::dispatchInverseToneMapping(
    Rc<RtxContext> ctx,
    Rc<DxvkImageView> exposureView,
    const Resources::RaytracingOutput& rtOutput,
    bool autoExposureEnabled) {

    ScopedGpuProfileZone(ctx, "Inverse Tone Mapping");

    // Prepare shader arguments
    FastToneMappingArgs pushArgs = {};
    pushArgs.enableAutoExposure = autoExposureEnabled;
    pushArgs.exposureFactor = exp2f(exposureBias()) * PreExposure::inverse(); // ev100, / P
    pushArgs.tonemapPrimaries = static_cast<uint32_t>(primaries());
    pushArgs.grade = gradeArgs(ctx);
    // Highlight recovery (inverse-tonemap only path)
    pushArgs.enableHighlightRecovery = DlssNeuralRendering::enableHighlightRecovery();
    {
      const Vector4& t = DlssNeuralRendering::highlightRecoveryThresholds();
      pushArgs.highlightRecoveryThresholds = vec4(t.x, t.y, t.z, t.w);
    }

    const Resources::Resource& inputColorTexture = rtOutput.m_neuralRenderingOutput.resource(Resources::AccessType::Read);
    const Resources::Resource& inOutColorTexture = rtOutput.m_finalOutput.resource(Resources::AccessType::ReadWrite);
    const VkExtent3D workgroups = util::computeBlockCount(inputColorTexture.view->imageInfo().extent, VkExtent3D { 16 , 16, 1 });

    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_INPUT, inputColorTexture.view, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_EXPOSURE_INPUT, exposureView, nullptr);
    ctx->bindResourceView(TONEMAPPING_APPLY_TONEMAPPING_COLOR_OUTPUT, inOutColorTexture.view, nullptr);
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, InverseTonemappingShader::getShader());
    ctx->pushConstants(0, sizeof(pushArgs), &pushArgs);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
  }
}
