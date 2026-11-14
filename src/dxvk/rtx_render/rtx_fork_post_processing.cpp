#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>

#include "dxvk_objects.h"
#include "rtx_fork_hooks.h"
#include "rtx_context.h"
#include "rtx_bloom.h"
#include "rtx_external_effects.h"
#include "rtx_postFx.h"
#include "rtx_tone_mapping.h"
#include "rtx_srgb_dither.h"
#include "rtx_imgui.h"
#include "rtx_fork_post_processing.h"

#include "../imgui/imgui.h"

namespace dxvk {

  namespace {
    constexpr size_t kInvalidIndex = static_cast<size_t>(-1);
    constexpr size_t kEffectCount = 7;
    constexpr const char* kExternalPrefix = "external:";

    std::string joinConfigIds(const std::vector<std::string>& ids) {
      std::string result;
      for (const std::string& id : ids) {
        if (!result.empty()) {
          result += ",";
        }
        result += id;
      }
      return result;
    }

    // The drag handle is drawn with the draw list, since the menu fonts may not carry a grip glyph.
    void drawDragGrip(const ImVec2& boxMin, const ImVec2& boxMax, ImU32 color) {
      ImDrawList* drawList = ImGui::GetWindowDrawList();
      const float fontSize = ImGui::GetFontSize();
      const float radius = ImMax(1.0f, ImFloor(fontSize * 0.09f));
      const float stepX = ImMax(2.0f, ImFloor(fontSize * 0.28f));
      const float stepY = ImMax(2.0f, ImFloor(fontSize * 0.28f));
      const ImVec2 center((boxMin.x + boxMax.x) * 0.5f, (boxMin.y + boxMax.y) * 0.5f);

      for (int row = -1; row <= 1; row++) {
        for (int column = 0; column < 2; column++) {
          const ImVec2 dot(
            center.x + (column == 0 ? -stepX : stepX) * 0.5f,
            center.y + stepY * static_cast<float>(row));
          drawList->AddCircleFilled(dot, radius, color);
        }
      }
    }

    // Anchors get a dimmed static rail instead of a disabled button, which still reads as pinned.
    void drawAnchorMark(const ImVec2& boxMin, const ImVec2& boxMax, ImU32 color) {
      ImDrawList* drawList = ImGui::GetWindowDrawList();
      const float fontSize = ImGui::GetFontSize();
      const float halfWidth = ImMax(2.0f, ImFloor(fontSize * 0.30f));
      const float halfHeight = ImMax(1.0f, ImFloor(fontSize * 0.07f));
      const ImVec2 center((boxMin.x + boxMax.x) * 0.5f, (boxMin.y + boxMax.y) * 0.5f);

      drawList->AddRectFilled(
        ImVec2(center.x - halfWidth, center.y - halfHeight),
        ImVec2(center.x + halfWidth, center.y + halfHeight), color);
    }

    // The search box filters the view only. It has no access to the stored order, so it cannot
    // change what the pipeline runs or what is written back into rtx.postfx.stackOrder.
    bool matchesFilter(const std::string& text, const std::string& filter) {
      if (filter.empty()) {
        return true;
      }

      const auto match = std::search(
        text.begin(), text.end(), filter.begin(), filter.end(),
        [](char lhs, char rhs) {
          return std::tolower(static_cast<unsigned char>(lhs))
              == std::tolower(static_cast<unsigned char>(rhs));
        });
      return match != text.end();
    }
  }

  const RtxPostProcessingStack::EffectDescriptor& RtxPostProcessingStack::descriptor(EffectId id) {
    static constexpr std::array<EffectDescriptor, kEffectCount> descriptors = {{
      { EffectId::Bloom,       "Bloom",          "bloom",        EffectDomain::HDR,      true  },
      { EffectId::MotionBlur,  "Motion Blur",    "motion_blur",  EffectDomain::HDR,      true  },
      { EffectId::DepthOfField, "Depth of Field", "depth_of_field", EffectDomain::HDR,    true  },
      { EffectId::Tonemapping, "Tonemapping",    "tonemapping",  EffectDomain::HDR,      false },
      { EffectId::NtscVhs,     "NTSC / VHS",     "ntsc_vhs",     EffectDomain::Display, true  },
      { EffectId::LensEffects, "Lens Effects",   "lens_effects", EffectDomain::Display,  true  },
      { EffectId::SRGBDither,  "sRGB + Dither",  "srgb_dither",  EffectDomain::Terminal, false },
    }};

    return descriptors[static_cast<size_t>(id)];
  }

  std::string RtxPostProcessingStack::configId(const EffectEntry& entry) {
    return entry.external
      ? "external:" + entry.externalId
      : descriptor(entry.builtInId).configId;
  }

  std::string RtxPostProcessingStack::name(const EffectEntry& entry) {
    if (!entry.external) {
      return descriptor(entry.builtInId).name;
    }

    const auto infos = RtxExternalEffects::instance().effectInfos();
    const auto info = std::find_if(infos.begin(), infos.end(), [&](const RtxExternalEffectInfo& candidate) {
      return candidate.id == entry.externalId;
    });
    return info == infos.end() ? entry.externalId : info->name;
  }

  RtxPostProcessingStack::EffectDomain RtxPostProcessingStack::domain(const EffectEntry& entry) {
    if (!entry.external) {
      return descriptor(entry.builtInId).domain;
    }

    const auto infos = RtxExternalEffects::instance().effectInfos();
    const auto info = std::find_if(infos.begin(), infos.end(), [&](const RtxExternalEffectInfo& candidate) {
      return candidate.id == entry.externalId;
    });
    return info != infos.end() && info->domain == RtxExternalEffectDomain::HDR
      ? EffectDomain::HDR
      : EffectDomain::Display;
  }

  bool RtxPostProcessingStack::reorderable(const EffectEntry& entry) {
    return entry.external || descriptor(entry.builtInId).reorderable;
  }

  std::vector<RtxPostProcessingStack::EffectEntry> RtxPostProcessingStack::defaultOrder() {
    return {
      { false, EffectId::Bloom, {} },
      { false, EffectId::MotionBlur, {} },
      { false, EffectId::DepthOfField, {} },
      { false, EffectId::Tonemapping, {} },
      { false, EffectId::NtscVhs, {} },
      { false, EffectId::LensEffects, {} },
      { false, EffectId::SRGBDither, {} },
    };
  }

  size_t RtxPostProcessingStack::findEffect(
    const std::vector<EffectEntry>& order,
    const std::string& effectConfigId) {
    const auto it = std::find_if(order.begin(), order.end(), [&](const EffectEntry& entry) {
      return configId(entry) == effectConfigId;
    });
    return it == order.end() ? kInvalidIndex : static_cast<size_t>(it - order.begin());
  }

  std::string RtxPostProcessingStack::trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return {};
    }

    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
  }

  RtxPostProcessingStack::EffectEntry RtxPostProcessingStack::effectFromConfig(
    const std::string& configId,
    bool& valid) {
    for (size_t i = 0; i < kEffectCount; i++) {
      const EffectId id = static_cast<EffectId>(i);
      if (configId == descriptor(id).configId) {
        valid = true;
        return { false, id, {} };
      }
    }

    if (configId.rfind(kExternalPrefix, 0) == 0) {
      const std::string id = configId.substr(std::char_traits<char>::length(kExternalPrefix));
      if (RtxExternalEffects::instance().hasEffect(id)) {
        valid = true;
        return { true, EffectId::Bloom, id };
      }
    }

    valid = false;
    return {};
  }

  std::vector<std::string> RtxPostProcessingStack::storedOrderTokens() {
    std::vector<std::string> tokens;
    std::stringstream stream(stackOrder());
    std::string token;

    while (std::getline(stream, token, ',')) {
      const std::string trimmed = trim(token);
      if (!trimmed.empty()) {
        tokens.push_back(trimmed);
      }
    }

    return tokens;
  }

  std::vector<RtxPostProcessingStack::EffectEntry> RtxPostProcessingStack::resolvedOrder() {
    std::vector<EffectEntry> parsedOrder;

    for (const std::string& token : storedOrderTokens()) {
      bool valid = false;
      const EffectEntry entry = effectFromConfig(token, valid);
      if (valid && findEffect(parsedOrder, configId(entry)) == kInvalidIndex) {
        parsedOrder.push_back(entry);
      }
    }

    for (const EffectEntry& entry : defaultOrder()) {
      if (findEffect(parsedOrder, configId(entry)) == kInvalidIndex) {
        parsedOrder.push_back(entry);
      }
    }

    for (const RtxExternalEffectInfo& info : RtxExternalEffects::instance().effectInfos()) {
      const EffectEntry entry { true, EffectId::Bloom, info.id };
      if (findEffect(parsedOrder, configId(entry)) == kInvalidIndex) {
        parsedOrder.push_back(entry);
      }
    }

    std::vector<EffectEntry> result;
    result.reserve(parsedOrder.size());

    for (const EffectEntry& entry : parsedOrder) {
      if (domain(entry) == EffectDomain::HDR
       && (entry.external || entry.builtInId != EffectId::Tonemapping)) {
        result.push_back(entry);
      }
    }

    // Tonemapping is the fixed HDR-to-display boundary. It is never allowed
    // to move into either reorderable lane.
    result.push_back({ false, EffectId::Tonemapping, {} });

    for (const EffectEntry& entry : parsedOrder) {
      if (domain(entry) == EffectDomain::Display) {
        result.push_back(entry);
      }
    }

    result.push_back({ false, EffectId::SRGBDither, {} });
    return result;
  }

  std::vector<std::string> RtxPostProcessingStack::configIds(
    const std::vector<EffectEntry>& order) {
    std::vector<std::string> ids;
    ids.reserve(order.size());
    for (const EffectEntry& entry : order) {
      ids.push_back(configId(entry));
    }
    return ids;
  }

  std::string RtxPostProcessingStack::serializeOrder(const std::vector<EffectEntry>& order) {
    return joinConfigIds(configIds(order));
  }

  // serializeOrder, plus every stored token resolvedOrder() could not turn into an entry.
  // External effects load on a worker and can be missing for a while (still compiling, moved,
  // failing to parse), so writing back only the visible stack would drop them from the saved order.
  //
  // A missing effect is put back after the nearest preceding stored token that survived. That is the
  // only position information there is, since an unloaded effect has no domain to sort it into.
  std::string RtxPostProcessingStack::serializeOrderPreservingUnloaded(
    const std::vector<EffectEntry>& order) {
    std::vector<std::string> ids = configIds(order);
    const std::vector<std::string> stored = storedOrderTokens();

    for (size_t i = 0; i < stored.size(); i++) {
      bool valid = false;
      effectFromConfig(stored[i], valid);
      if (valid || std::find(ids.begin(), ids.end(), stored[i]) != ids.end()) {
        continue;
      }
      // Only an external id is carried. An unrecognised built-in name is a typo or comes from another
      // build, and keeping it would make the option impossible to clean up from the panel.
      const size_t prefixLength = std::char_traits<char>::length(kExternalPrefix);
      if (stored[i].rfind(kExternalPrefix, 0) != 0
       || !isValidRtxExternalEffectId(stored[i].substr(prefixLength))) {
        continue;
      }

      size_t position = 0;
      for (size_t j = i; j-- > 0;) {
        const auto anchor = std::find(ids.begin(), ids.end(), stored[j]);
        if (anchor != ids.end()) {
          position = static_cast<size_t>(anchor - ids.begin()) + 1;
          break;
        }
      }
      // A token restored here anchors the next one, which is what keeps a run
      // of unloaded effects in the order they were written.
      ids.insert(ids.begin() + position, stored[i]);
    }

    return joinConfigIds(ids);
  }

  bool RtxPostProcessingStack::canMove(
    const std::vector<EffectEntry>& order,
    size_t from,
    size_t to) {
    if (from >= order.size() || to >= order.size() || from == to) {
      return false;
    }

    return reorderable(order[from])
        && reorderable(order[to])
        && domain(order[from]) == domain(order[to]);
  }

  bool RtxPostProcessingStack::moveEffect(
    std::vector<EffectEntry>& order,
    size_t from,
    size_t to) {
    if (!canMove(order, from, to)) {
      return false;
    }

    const EffectEntry entry = order[from];
    order.erase(order.begin() + from);
    order.insert(order.begin() + to, entry);
    return true;
  }

  void RtxPostProcessingStack::dispatch(
    Rc<RtxContext> ctx,
    Resources::RaytracingOutput& rtOutput,
    bool performSRGBConversion) {
    ScopedCpuProfileZone();
    RtxExternalEffects::instance().ensureLoaded(ctx->getDevice().ptr());
    // A finished recompile is swapped in here, before the loop, so the two dispatches of one effect
    // never see different layouts within a frame.
    RtxExternalEffects::instance().pumpHotReload(ctx->getDevice().ptr());

    // The post-FX option switches the optional effects. Tonemapping and the terminal sRGB/dither
    // conversion always run, because skipping either would change the output format.
    const bool optionalEffectsEnabled = ctx->getCommonObjects()->metaPostFx().enable();

    for (const EffectEntry& entry : resolvedOrder()) {
      if (entry.external) {
        if (optionalEffectsEnabled) {
          RtxExternalEffects::instance().dispatch(ctx, rtOutput, entry.externalId);
        }
        continue;
      }

      const EffectId id = entry.builtInId;
      if (!optionalEffectsEnabled
          && id != EffectId::Tonemapping
          && id != EffectId::SRGBDither) {
        continue;
      }

      switch (id) {
      case EffectId::Bloom:
        ctx->dispatchBloom(rtOutput);
        break;
      case EffectId::MotionBlur:
        ctx->dispatchPostFxMotionBlur(rtOutput);
        break;
      case EffectId::DepthOfField:
        ctx->dispatchPostFxDof(rtOutput);
        break;
      case EffectId::Tonemapping:
        ctx->dispatchToneMapping(rtOutput);
        break;
      case EffectId::NtscVhs:
        ctx->dispatchPostFxNtsc(rtOutput);
        break;
      case EffectId::LensEffects:
        ctx->dispatchPostFxLensEffects(rtOutput);
        break;
      case EffectId::SRGBDither:
        ctx->dispatchSRGBDither(rtOutput, performSRGBConversion);
        break;
      }
    }
  }

  void RtxPostProcessingStack::showSettings(const Rc<DxvkContext>& ctx) {
    auto common = ctx->getCommonObjects();
    auto& postFx = common->metaPostFx();
    auto& externalEffects = RtxExternalEffects::instance();
    externalEffects.ensureLoaded(ctx->getDevice().ptr());

    // The panel is a fixed-height list plus one settings pane, so rows never move when one is selected.
    // Lanes are shown as captions in the list. "Post FX Enabled" skips the optional members only, and
    // the two anchors show this with a pinned handle and a disabled, always checked box.
    const bool optionalEffectsEnabled = postFx.enable();
    std::vector<EffectEntry> order = resolvedOrder();

    // Auto Exposure is not a stack member, but it gets a row and a settings panel. It uses a pseudo
    // config id the registry never produces (no built-in has this name, and external ids carry an
    // "external:" prefix), so the selection stays one plain string.
    constexpr const char* kAutoExposureId = "auto_exposure";

    // Selection is keyed by config id, never by index, because drag-and-drop reorders the list.
    // Both values are view state of this single panel, so they are statics instead of options.
    static std::string s_selectedEffectId;
    static char s_filterText[64] = {};

    // --- Global controls ---------------------------------------------------
    // Outside the scrolling child, since they act on the whole stack.
    RemixGui::Checkbox("Post FX Enabled", &postFx.enableObject());
    ImGui::SameLine();
    if (ImGui::Button("Reset to Default Order")) {
      // The plain serializer: a reset clears external effects out of the stored order too, and
      // resolvedOrder() appends each one at the end as it loads, whether it is loaded now or not.
      stackOrderObject().setDeferred(serializeOrder(defaultOrder()));
    }

    ImGui::TextWrapped(
      "Effects run top to bottom. \"Post FX Enabled\" is the master switch for the optional "
      "effects only - tonemapping and the terminal sRGB/dither conversion keep running either "
      "way, and ordering stays editable while the master switch is off.");
    if (!optionalEffectsEnabled) {
      ImGui::TextDisabled("Post FX Enabled is off, so only the always-on stages are running.");
    }

    ImGui::Spacing();

    // --- Filter ------------------------------------------------------------
    // Drawn before the filter string is sampled so a keystroke takes effect on
    // the frame it was typed rather than the one after it.
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##search", "Search effects...", s_filterText, sizeof(s_filterText));

    const std::string filter(s_filterText);
    const bool filterActive = !filter.empty();

    if (filterActive) {
      ImGui::TextDisabled("Search is active - drag to reorder is disabled. Clear the box to reorder.");
    } else {
      ImGui::TextDisabled("Drag a grip to reorder effects inside their own group.");
    }

    // An effect can vanish between frames when an external effect file is
    // removed, so a stale selection falls back to the head of the pipeline.
    // resolvedOrder() always emits the two anchors, so order[0] exists.
    if (s_selectedEffectId != kAutoExposureId
     && findEffect(order, s_selectedEffectId) == kInvalidIndex) {
      s_selectedEffectId = configId(order[0]);
    }

    // Shared row renderer. Anchors and optional effects differ only in the handle mark and the dimming.
    // The rows the master switch governs are exactly the reorderable ones, so reorderable() decides both.
    auto showEffectRow = [&](size_t index) {
      const EffectEntry& entry = order[index];
      const bool effectReorderable = reorderable(entry);
      const std::string effectConfigId = configId(entry);
      const std::string effectName = name(entry);
      const bool effectRunning = !effectReorderable || optionalEffectsEnabled;

      // Filtering hides rows and nothing else: `order` is untouched, so the
      // reorder payload logic below still sees the real stack.
      if (!matchesFilter(effectName, filter)) {
        return;
      }

      ImGui::PushID(effectConfigId.c_str());

      RtxOption<bool>* enabledOption = nullptr;
      RtxExternalEffectInfo externalInfo = {};
      if (entry.external) {
        const auto infos = externalEffects.effectInfos();
        const auto info = std::find_if(infos.begin(), infos.end(), [&](const RtxExternalEffectInfo& candidate) {
          return candidate.id == entry.externalId;
        });
        if (info != infos.end()) {
          externalInfo = *info;
        }
      } else {
        switch (entry.builtInId) {
        case EffectId::Bloom:
          enabledOption = &common->metaBloom().enableObject();
          break;
        case EffectId::MotionBlur:
          enabledOption = &postFx.enableMotionBlurObject();
          break;
        case EffectId::DepthOfField:
          enabledOption = &postFx.dofEnableObject();
          break;
        case EffectId::Tonemapping:
          enabledOption = &common->metaToneMapping().tonemappingEnabledObject();
          break;
        case EffectId::NtscVhs:
          enabledOption = &postFx.ntscEnableObject();
          break;
        case EffectId::LensEffects:
          enabledOption = &postFx.enableLensEffectsObject();
          break;
        case EffectId::SRGBDither:
          break;
        }
      }

      ImGui::BeginGroup();

      const float handleExtent = ImGui::GetFrameHeight();
      const ImVec2 handleMin = ImGui::GetCursorScreenPos();
      const ImVec2 handleMax(handleMin.x + handleExtent, handleMin.y + handleExtent);
      ImGui::InvisibleButton("##handle", ImVec2(handleExtent, handleExtent));
      const bool handleHovered = ImGui::IsItemHovered();

      if (effectReorderable) {
        // Dragging is refused while the list is filtered, since the visible neighbours are not the stored
        // ones. The grip stops looking like a handle, so the refusal is visible before the click.
        const bool dragAllowed = !filterActive;
        drawDragGrip(handleMin, handleMax,
                     ImGui::GetColorU32(dragAllowed && handleHovered ? ImGuiCol_Text : ImGuiCol_TextDisabled));
        if (handleHovered) {
          ImGui::SetTooltip(dragAllowed
            ? "Drag to reorder within this group"
            : "Clear the search box to reorder effects");
        }

        // Ordering is configuration, not runtime state, so it stays editable
        // even when the master switch is off.
        if (dragAllowed && ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoPreviewTooltip)) {
          ImGui::SetDragDropPayload(
            "RTX_POSTFX_EFFECT_ID", effectConfigId.c_str(), effectConfigId.size() + 1);
          ImGui::Text("Move %s", effectName.c_str());
          ImGui::EndDragDropSource();
        }
      } else {
        drawAnchorMark(handleMin, handleMax, ImGui::GetColorU32(ImGuiCol_TextDisabled));
        if (handleHovered) {
          ImGui::SetTooltip("Fixed pipeline stage: it cannot be moved or switched off");
        }
      }

      ImGui::SameLine();
      if (entry.external) {
        bool enabled = externalInfo.enabled;
        if (RemixGui::CheckboxNoLabel("##enabled", &enabled)) {
          externalEffects.setEffectEnabled(entry.externalId, enabled);
        }
        if (!externalInfo.ready && ImGui::IsItemHovered()) {
          ImGui::SetTooltip("This effect has no loadable shader.");
        }
        ImGui::SameLine();
      } else if (enabledOption != nullptr) {
        RemixGui::CheckboxNoLabel("##enabled", enabledOption);
        ImGui::SameLine();
      } else {
        bool required = true;
        ImGui::BeginDisabled();
        RemixGui::CheckboxNoLabel("##required", &required);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
          ImGui::SetTooltip("Always on: this conversion defines the output format");
        }
        ImGui::SameLine();
      }

      // The label is painted by hand instead of being passed to Selectable, because effect names come
      // from external manifest files and ImGui would hide the rest of any name containing "##".
      const ImVec2 labelMin = ImGui::GetCursorScreenPos();
      const float rowHeight = ImGui::GetFrameHeight();
      if (ImGui::Selectable("##row", s_selectedEffectId == effectConfigId,
                            ImGuiSelectableFlags_None, ImVec2(0.0f, rowHeight))) {
        s_selectedEffectId = effectConfigId;
      }
      ImGui::GetWindowDrawList()->AddText(
        ImVec2(labelMin.x + ImGui::GetStyle().FramePadding.x,
               labelMin.y + (rowHeight - ImGui::GetFontSize()) * 0.5f),
        ImGui::GetColorU32(effectRunning ? ImGuiCol_Text : ImGuiCol_TextDisabled),
        effectName.c_str());

      ImGui::EndGroup();

      // The whole row is a drop target. Payloads name an effect, not a row index, and are applied only on
      // delivery, so hovering cannot reshuffle the stack.
      if (effectReorderable && !filterActive && ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("RTX_POSTFX_EFFECT_ID")) {
          if (payload->IsDelivery() && payload->DataSize > 1
           && static_cast<const char*>(payload->Data)[payload->DataSize - 1] == '\0') {
            const std::string sourceConfigId(static_cast<const char*>(payload->Data));
            const size_t sourceIndex = findEffect(order, sourceConfigId);
            if (sourceIndex != kInvalidIndex && moveEffect(order, sourceIndex, index)) {
              stackOrderObject().setDeferred(serializeOrderPreservingUnloaded(order));
            }
          }
        }
        ImGui::EndDragDropTarget();
      }

      ImGui::PopID();
    };

    // Draws the settings of the selected entry.
    auto showEntrySettings = [&](const EffectEntry& entry) {
      if (entry.external) {
        externalEffects.showEffectSettings(entry.externalId);
        return;
      }

      switch (entry.builtInId) {
      case EffectId::Bloom:
        common->metaBloom().showEffectSettings();
        break;
      case EffectId::MotionBlur:
        postFx.showMotionBlurImguiSettings();
        break;
      case EffectId::DepthOfField:
        postFx.showDofImguiSettings();
        break;
      case EffectId::Tonemapping:
        common->metaToneMapping().showEffectSettings();
        break;
      case EffectId::NtscVhs:
        postFx.showNtscImguiSettings();
        break;
      case EffectId::LensEffects:
        postFx.showLensEffectsImguiSettings();
        break;
      case EffectId::SRGBDither:
        common->metaSRGBDither().showImguiSettings();
        break;
      }
    };

    // The captions name the image a lane operates on, because Tonemapping itself sits in the HDR lane.
    auto domainCaption = [](EffectDomain effectDomain) -> const char* {
      switch (effectDomain) {
      case EffectDomain::HDR:      return "HDR - operates on the pre-tonemap image";
      case EffectDomain::Display:  return "Display - operates on the tonemapped image";
      case EffectDomain::Terminal: return "Terminal - runs last";
      }
      return "Other";
    };

    // Auto Exposure's row has the pinned handle and a spacer instead of a checkbox, since its switch is
    // in its own panel. It is drawn directly above Tonemapping, which uses its exposure.
    auto showAutoExposureRow = [&]() {
      if (!matchesFilter("Auto Exposure", filter)) {
        return;
      }

      ImGui::PushID(kAutoExposureId);
      ImGui::BeginGroup();

      const float handleExtent = ImGui::GetFrameHeight();
      const ImVec2 handleMin = ImGui::GetCursorScreenPos();
      ImGui::InvisibleButton("##handle", ImVec2(handleExtent, handleExtent));
      const bool handleHovered = ImGui::IsItemHovered();
      drawAnchorMark(handleMin, ImVec2(handleMin.x + handleExtent, handleMin.y + handleExtent),
                     ImGui::GetColorU32(ImGuiCol_TextDisabled));
      if (handleHovered) {
        ImGui::SetTooltip("Measures scene brightness for tonemapping; not a reorderable stack stage");
      }

      ImGui::SameLine();
      ImGui::Dummy(ImVec2(handleExtent, handleExtent));
      ImGui::SameLine();

      const ImVec2 labelMin = ImGui::GetCursorScreenPos();
      if (ImGui::Selectable("##row", s_selectedEffectId == kAutoExposureId,
                            ImGuiSelectableFlags_None, ImVec2(0.0f, handleExtent))) {
        s_selectedEffectId = kAutoExposureId;
      }
      ImGui::GetWindowDrawList()->AddText(
        ImVec2(labelMin.x + ImGui::GetStyle().FramePadding.x,
               labelMin.y + (handleExtent - ImGui::GetFontSize()) * 0.5f),
        ImGui::GetColorU32(ImGuiCol_Text), "Auto Exposure");

      ImGui::EndGroup();
      ImGui::PopID();
    };

    // A caption is printed only when a row under it survived the filter. resolvedOrder() emits every lane
    // contiguously, so the look-ahead stops at the first row of the next lane.
    auto laneHasVisibleRow = [&](size_t start, EffectDomain lane) {
      for (size_t i = start; i < order.size() && domain(order[i]) == lane; i++) {
        if (matchesFilter(name(order[i]), filter)) {
          return true;
        }
      }
      return lane == EffectDomain::HDR && matchesFilter("Auto Exposure", filter);
    };

    // --- Effect list -------------------------------------------------------
    // A fixed-height child, since the panel sits in a CollapsingHeader with no window height to split.
    // It is sized in rows so it follows the menu font.
    ImGui::BeginChild("##effectList", ImVec2(0.0f, ImGui::GetFrameHeightWithSpacing() * 14.0f), true);

    // One pass over the resolved order, emitting a caption whenever the lane changes, so no row can be
    // missed by an unexpected domain. The anchors stay inline at their pipeline position.
    bool wroteAnyRow = false;
    bool wroteAnyCaption = false;
    EffectDomain previousDomain = EffectDomain::HDR;

    for (size_t i = 0; i < order.size(); i++) {
      const EffectDomain rowDomain = domain(order[i]);
      if ((!wroteAnyCaption || rowDomain != previousDomain) && laneHasVisibleRow(i, rowDomain)) {
        if (wroteAnyCaption) {
          ImGui::Spacing();
        }
        ImGui::TextDisabled("%s", domainCaption(rowDomain));
        previousDomain = rowDomain;
        wroteAnyCaption = true;
      }

      if (!order[i].external && order[i].builtInId == EffectId::Tonemapping) {
        showAutoExposureRow();
      }

      if (matchesFilter(name(order[i]), filter)) {
        wroteAnyRow = true;
      }
      showEffectRow(i);
    }

    if (!wroteAnyRow && !matchesFilter("Auto Exposure", filter)) {
      ImGui::TextDisabled("No effects match the search.");
    }

    ImGui::EndChild();

    // --- Settings for the selected row -------------------------------------
    // One pane for the selected row, so the list above keeps its height.
    RemixGui::Separator();

    if (s_selectedEffectId == kAutoExposureId) {
      ImGui::Text("Auto Exposure");
      ImGui::Spacing();
      common->metaAutoExposure().showImguiSettings();
    } else {
      // Re-resolved, because a drop delivered while drawing the list may have reordered `order`.
      const size_t selectedIndex = findEffect(order, s_selectedEffectId);
      if (selectedIndex != kInvalidIndex) {
        const EffectEntry& selected = order[selectedIndex];

        // Scoped by the effect id, so the option widgets keep stable ids.
        ImGui::PushID(s_selectedEffectId.c_str());
        ImGui::Text("%s", name(selected).c_str());
        if (reorderable(selected) && !optionalEffectsEnabled) {
          ImGui::TextDisabled("Not running: Post FX Enabled is off.");
        }
        ImGui::Spacing();
        showEntrySettings(selected);
        ImGui::PopID();
      }
    }

    // --- External effect files ---------------------------------------------
    // Loader configuration for the whole effect directory, so it sits outside the settings pane.
    RemixGui::Separator();
    if (RemixGui::CollapsingHeader("External Effect Files")) {
      ImGui::Indent();
      externalEffects.showGlobalSettings(ctx->getDevice().ptr());
      ImGui::Unindent();
    }

    ImGui::Separator();
  }

  namespace fork_hooks {

    void dispatchPostProcessingStack(
      Rc<RtxContext> ctx,
      Resources::RaytracingOutput& rtOutput,
      bool performSRGBConversion) {
      RtxPostProcessingStack::dispatch(ctx, rtOutput, performSRGBConversion);
    }

    void showPostProcessingStackSettings(const Rc<DxvkContext>& ctx) {
      RtxPostProcessingStack::showSettings(ctx);
    }

  } // namespace fork_hooks

} // namespace dxvk
