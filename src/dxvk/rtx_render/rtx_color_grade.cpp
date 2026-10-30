#include "rtx_color_grade.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

#include "imgui/imgui.h"
#include "rtx_imgui.h"
#include "../../util/util_env.h"
#include "../../util/util_string.h"
#include "../../util/log/log.h"

namespace dxvk {

  namespace {
    namespace fs = std::filesystem;
    using color_grade::Params;

    struct Preset {
      std::string name;
      Params params;
      Params saved;        // the file's values as last loaded or saved
      bool builtIn = false;
      bool dirty = false;  // edited since the last load or save
    };

    enum class Source { Preset, Mix, Grade };

    struct State {
      std::mutex mutex;
      bool loaded = false;
      std::vector<Preset> presets;

      // What the blend heads for.
      Source source = Source::Preset;
      std::string presetA = "None";
      std::string presetB;
      float weight = 0.0f;
      Params grade;

      color_grade::Blend blend;
      std::string fromLabel = "None";
      std::string toLabel = "None";
      bool started = false;
      std::string lastOptionPreset;
      uint64_t lastFrameId = UINT64_MAX;
      color_grade::Gpu frame = color_grade::toGpu(Params {});

      // The editor's Save As name and its last message.
      char saveAsName[64] = {};
      std::string message;
    };

    State& state() {
      static State s;
      return s;
    }

    std::string trim(const std::string& s) {
      const size_t b = s.find_first_not_of(" \t\r\n");
      if (b == std::string::npos) {
        return {};
      }
      const size_t e = s.find_last_not_of(" \t\r\n");
      return s.substr(b, e - b + 1);
    }

    std::string lower(std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return s;
    }

    bool isBuiltInName(const std::string& name) {
      for (const auto& b : color_grade::builtIns()) {
        if (lower(name) == lower(b.name)) {
          return true;
        }
      }
      return false;
    }

    // A name a preset file can have: not empty, no path or reserved characters, and not a built-in's name in
    // any case, since Windows file names ignore case.
    bool isValidFileName(const std::string& name) {
      if (name.empty() || name != trim(name) || name == "." || name == ".." || name.size() > 60) {
        return false;
      }
      if (name.find_first_of("\\/:*?\"<>|") != std::string::npos) {
        return false;
      }
      for (const char c : name) {
        if (static_cast<unsigned char>(c) < 32) {
          return false;
        }
      }
      return !isBuiltInName(name);
    }

    fs::path directory() {
      fs::path p(ColorGrade::gradePresetDirectory());
      if (p.is_relative()) {
        p = fs::path(env::getExePath()).parent_path() / p;
      }
      return p.lexically_normal();
    }

    // Reads up to `count` comma-separated numbers. Returns false when fewer are present or one does not parse.
    bool parseFloats(const std::string& value, float* out, size_t count) {
      std::istringstream in(value);
      std::string item;
      size_t n = 0;
      while (n < count && std::getline(in, item, ',')) {
        try {
          size_t used = 0;
          const std::string t = trim(item);
          out[n] = std::stof(t, &used);
          if (used != t.size()) {
            return false;
          }
        } catch (...) {
          return false;
        }
        ++n;
      }
      return n == count;
    }

    bool readPresetFile(const fs::path& path, Preset& p) {
      std::ifstream f(path, std::ios::binary);
      if (!f) {
        return false;
      }
      p.name = path.stem().string();
      std::string line;
      while (std::getline(f, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
          line.resize(hash);
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
          continue;
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        float v[9];
        bool ok = true;
        if (key == "grade.matrix") {
          ok = parseFloats(value, v, 9);
          if (ok) {
            for (int i = 0; i < 9; ++i) {
              p.params.matrix[i / 3][i % 3] = v[i];
            }
          }
        } else if (key == "grade.balance") {
          ok = parseFloats(value, v, 3);
          if (ok) {
            p.params.balance = { v[0], v[1], v[2] };
          }
        } else if (key == "grade.saturation") {
          ok = parseFloats(value, v, 1);
          if (ok) {
            p.params.saturation = v[0];
          }
        } else if (key == "grade.contrast") {
          ok = parseFloats(value, v, 1);
          if (ok) {
            p.params.contrast = v[0];
          }
        } else if (key == "grade.anchor") {
          ok = parseFloats(value, v, 1);
          if (ok) {
            p.params.anchor = v[0];
          }
        } else if (key == "grade.shoulder") {
          ok = parseFloats(value, v, 1);
          if (ok) {
            p.params.shoulder = v[0];
          }
        } else if (!key.empty()) {
          Logger::warn(str::format("[ColorGrade] ", path.string(), ": unknown key ", key, "; ignored"));
        }
        if (!ok) {
          Logger::warn(str::format("[ColorGrade] ", path.string(), ": ", key, " = ", value, " does not parse; ignored"));
        }
      }
      p.params = color_grade::clamped(p.params);
      p.saved = p.params;
      return true;
    }

    std::string formatFloats(const float* v, size_t count) {
      std::string s;
      char buffer[32];
      for (size_t i = 0; i < count; ++i) {
        // 9 significant digits read back as the same float.
        std::snprintf(buffer, sizeof(buffer), "%.9g", v[i]);
        s += (i ? ", " : "");
        s += buffer;
      }
      return s;
    }

    bool writePresetFile(const std::string& name, const Params& p) {
      std::error_code ec;
      fs::create_directories(directory(), ec);
      const fs::path path = directory() / (name + ".conf");
      std::ofstream f(path, std::ios::binary | std::ios::trunc);
      if (!f) {
        Logger::err(str::format("[ColorGrade] cannot write ", path.string()));
        return false;
      }
      float m[9];
      for (int i = 0; i < 9; ++i) {
        m[i] = p.matrix[i / 3][i % 3];
      }
      f << "# Remix color grade preset \"" << name << "\". Edit it here or in the Remix developer menu (Post-Processing > Tonemapping).\n"
        << "# The grade applies to the linear scene color before the tone curve, in this order. Missing keys are neutral.\n"
        << "# grade.matrix: a row-major 3x3 color matrix (identity = 1, 0, 0, 0, 1, 0, 0, 0, 1).\n"
        << "# grade.balance: a gain per channel (1, 1, 1). grade.saturation: 1 = unchanged, 0 = gray.\n"
        << "# grade.contrast: a power around mid gray (0.18), 1 = unchanged.\n"
        << "# grade.anchor: where the tone curve's shoulder starts (0.18). Values below pass through. Range 0.01 to 0.5.\n"
        << "# grade.shoulder: how late and hard the shoulder bends (0.01). Range 0.001 to 1.\n"
        << "grade.matrix = " << formatFloats(m, 9) << "\n"
        << "grade.balance = " << formatFloats(p.balance.data(), 3) << "\n"
        << "grade.saturation = " << formatFloats(&p.saturation, 1) << "\n"
        << "grade.contrast = " << formatFloats(&p.contrast, 1) << "\n"
        << "grade.anchor = " << formatFloats(&p.anchor, 1) << "\n"
        << "grade.shoulder = " << formatFloats(&p.shoulder, 1) << "\n";
      return static_cast<bool>(f);
    }

    void reloadLocked(State& s) {
      s.presets.clear();
      for (const auto& b : color_grade::builtIns()) {
        Preset p;
        p.name = b.name;
        p.params = b.params;
        p.saved = b.params;
        p.builtIn = true;
        s.presets.push_back(p);
      }
      std::error_code ec;
      std::vector<fs::path> files;
      for (const auto& entry : fs::directory_iterator(directory(), ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".conf") {
          files.push_back(entry.path());
        }
      }
      std::sort(files.begin(), files.end());
      for (const fs::path& path : files) {
        Preset p;
        if (!readPresetFile(path, p)) {
          continue;
        }
        if (isBuiltInName(p.name)) {
          Logger::warn(str::format("[ColorGrade] ", path.string(), " has a built-in preset's name; ignored"));
          continue;
        }
        s.presets.push_back(p);
      }
      Logger::info(str::format("[ColorGrade] ", s.presets.size() - color_grade::builtIns().size(), " presets in ",
                               directory().string()));
    }

    void ensureLoadedLocked(State& s) {
      if (!s.loaded) {
        s.loaded = true;
        reloadLocked(s);
      }
    }

    Preset* findLocked(State& s, const std::string& name) {
      ensureLoadedLocked(s);
      for (Preset& p : s.presets) {
        if (p.name == name) {
          return &p;
        }
      }
      return nullptr;
    }

    std::string mixLabel(const std::string& a, const std::string& b, float w) {
      char buffer[16];
      std::snprintf(buffer, sizeof(buffer), "%.2f", w);
      return str::format("mix of ", a, " and ", b, " at ", buffer);
    }

    // The grade the blend heads for, resolved every frame so edits to a preset show at once.
    Params targetLocked(State& s) {
      switch (s.source) {
      case Source::Preset: {
        const Preset* p = findLocked(s, s.presetA);
        return p ? p->params : s.blend.current();
      }
      case Source::Mix: {
        const Preset* a = findLocked(s, s.presetA);
        const Preset* b = findLocked(s, s.presetB);
        if (a == nullptr || b == nullptr) {
          return s.blend.current();
        }
        return color_grade::mix(a->params, b->params, s.weight);
      }
      case Source::Grade:
        return s.grade;
      }
      return Params {};
    }

    void retargetLocked(State& s, const std::string& label, float blendSeconds) {
      s.fromLabel = s.blend.blending() ? "the blend in progress" : s.toLabel;
      s.toLabel = label;
      s.blend.retarget(blendSeconds);
    }

    void setPresetLocked(State& s, const std::string& name, float blendSeconds) {
      s.source = Source::Preset;
      s.presetA = name;
      retargetLocked(s, name, blendSeconds);
    }

    // Follows rtx.tonemap.gradePreset: its value at the first frame shows at once, a later change blends.
    void followOptionLocked(State& s) {
      const std::string& option = ColorGrade::gradePreset();
      if (s.started && option == s.lastOptionPreset) {
        return;
      }
      s.lastOptionPreset = option;
      if (findLocked(s, option) == nullptr) {
        Logger::warn(str::format("[ColorGrade] rtx.tonemap.gradePreset names no preset: \"", option, "\""));
        s.started = true;
        return;
      }
      setPresetLocked(s, option, s.started ? std::max(ColorGrade::gradeBlendSeconds(), 0.0f) : 0.0f);
      s.started = true;
    }

    // An edit of a preset file's values. It shows the edited preset, whatever an API call showed before.
    void editLocked(State& s, Preset& edited, const Params& p) {
      if (edited.builtIn) {
        return;
      }
      edited.params = color_grade::clamped(p);
      edited.dirty = edited.params != edited.saved;
      if (s.source != Source::Preset || s.presetA != edited.name) {
        setPresetLocked(s, edited.name, 0.0f);
      }
    }
  }

  color_grade::Gpu ColorGrade::frameConstants(uint64_t frameId, float deltaSeconds) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (frameId == s.lastFrameId) {
      return s.frame;
    }
    s.lastFrameId = frameId;
    followOptionLocked(s);
    s.frame = color_grade::toGpu(s.blend.step(targetLocked(s), deltaSeconds));
    return s.frame;
  }

  bool ColorGrade::setPreset(const std::string& name, float blendSeconds) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (findLocked(s, name) == nullptr) {
      return false;
    }
    setPresetLocked(s, name, blendSeconds);
    return true;
  }

  void ColorGrade::setGrade(const Params& params, float blendSeconds) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.source = Source::Grade;
    s.grade = color_grade::clamped(params);
    retargetLocked(s, "a grade set through the Remix API", blendSeconds);
  }

  bool ColorGrade::mixPresets(const std::string& nameA, const std::string& nameB, float weight) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (findLocked(s, nameA) == nullptr || findLocked(s, nameB) == nullptr) {
      return false;
    }
    s.source = Source::Mix;
    s.presetA = nameA;
    s.presetB = nameB;
    s.weight = std::clamp(weight, 0.0f, 1.0f);
    s.fromLabel = s.toLabel;
    s.toLabel = mixLabel(nameA, nameB, s.weight);
    s.blend.retarget(0.0f);
    return true;
  }

  Params ColorGrade::current() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.blend.current();
  }

  ColorGrade::SaveResult ColorGrade::savePreset(const std::string& name, const Params& params) {
    if (!isValidFileName(name)) {
      return SaveResult::InvalidName;
    }
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const Params p = color_grade::clamped(params);
    if (!writePresetFile(name, p)) {
      return SaveResult::WriteFailed;
    }
    Preset* existing = findLocked(s, name);
    if (existing == nullptr) {
      Preset added;
      added.name = name;
      s.presets.push_back(added);
      existing = &s.presets.back();
    }
    existing->params = p;
    existing->saved = p;
    existing->dirty = false;
    return SaveResult::Saved;
  }

  bool ColorGrade::editedPreset(Params& params, bool& builtIn) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const Preset* p = findLocked(s, gradePreset());
    if (p == nullptr) {
      return false;
    }
    params = p->params;
    builtIn = p->builtIn;
    return true;
  }

  void ColorGrade::editPreset(const Params& params) {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    Preset* p = findLocked(s, gradePreset());
    if (p != nullptr) {
      editLocked(s, *p, params);
    }
  }

  void ColorGrade::showImguiSettings() {
    State& s = state();
    std::unique_lock<std::mutex> lock(s.mutex);
    ensureLoadedLocked(s);

    const std::string selected = gradePreset();
    Preset* edited = findLocked(s, selected);

    if (ImGui::BeginCombo("Grade Preset", selected.c_str())) {
      for (const Preset& p : s.presets) {
        const std::string label = p.dirty ? p.name + " *" : p.name;
        if (ImGui::Selectable(label.c_str(), p.name == selected)) {
          gradePresetObject().setDeferred(p.name);
        }
      }
      ImGui::EndCombo();
    }
    RemixGui::SetTooltipToLastWidgetOnHover("The colour grade preset (rtx.tonemap.gradePreset). The sliders below edit it, "
                                            "and the image shows the edits at once. A * marks unsaved edits.");

    RemixGui::DragFloat("Grade Blend Time", &gradeBlendSecondsObject(), 0.05f, 0.0f, 60.0f, "%.2f s", ImGuiSliderFlags_AlwaysClamp);

    if (s.blend.blending()) {
      ImGui::Text("Blending %s -> %s, %.1f s left", s.fromLabel.c_str(), s.toLabel.c_str(), s.blend.secondsLeft());
    } else {
      ImGui::Text("Showing %s", s.toLabel.c_str());
    }

    if (edited == nullptr) {
      ImGui::TextDisabled("No preset is named \"%s\".", selected.c_str());
    } else {
      if (edited->builtIn) {
        ImGui::TextDisabled("A built-in preset: Save As makes an editable copy.");
      }
      ImGui::BeginDisabled(edited->builtIn);
      Params p = edited->params;
      bool changed = false;
      changed |= ImGui::DragFloat3("Matrix Row 1", p.matrix[0].data(), 0.002f, color_grade::kMatrixMin, color_grade::kMatrixMax, "%.4f");
      changed |= ImGui::DragFloat3("Matrix Row 2", p.matrix[1].data(), 0.002f, color_grade::kMatrixMin, color_grade::kMatrixMax, "%.4f");
      changed |= ImGui::DragFloat3("Matrix Row 3", p.matrix[2].data(), 0.002f, color_grade::kMatrixMin, color_grade::kMatrixMax, "%.4f");
      RemixGui::SetTooltipToLastWidgetOnHover("The colour matrix: each output channel is its row times the input red, green and "
                                              "blue. The identity (1 0 0 / 0 1 0 / 0 0 1) leaves the colour unchanged.");
      changed |= ImGui::DragFloat3("Colour Balance", p.balance.data(), 0.002f, 0.0f, color_grade::kBalanceMax, "%.3f");
      RemixGui::SetTooltipToLastWidgetOnHover("A gain on red, green and blue after the matrix. 1, 1, 1 is neutral.");
      changed |= ImGui::DragFloat("Saturation", &p.saturation, 0.005f, 0.0f, color_grade::kSaturationMax, "%.3f");
      RemixGui::SetTooltipToLastWidgetOnHover("1 is unchanged, 0 is grey, above 1 more saturated.");
      changed |= ImGui::DragFloat("Contrast", &p.contrast, 0.005f, color_grade::kContrastMin, color_grade::kContrastMax, "%.3f");
      RemixGui::SetTooltipToLastWidgetOnHover("A power around mid gray (0.18), before the tone curve. 1 is unchanged.");
      changed |= ImGui::DragFloat("Curve Anchor", &p.anchor, 0.001f, color_grade::kAnchorMin, color_grade::kAnchorMax, "%.3f");
      RemixGui::SetTooltipToLastWidgetOnHover("Where the tone curve's shoulder starts. Values below it pass through unchanged. "
                                              "0.18 (mid gray) is the default. Lower rolls off more of the image.");
      changed |= ImGui::DragFloat("Curve Shoulder", &p.shoulder, 0.0005f, color_grade::kShoulderMin, color_grade::kShoulderMax, "%.4f");
      RemixGui::SetTooltipToLastWidgetOnHover("How the shoulder bends above the anchor. Higher keeps the curve straight longer and "
                                              "bends it later and harder. Lower is a softer roll-off. 0.01 is the default.");
      if (ImGui::Button("Reset Fields to Neutral")) {
        p = Params {};
        changed = true;
      }
      ImGui::EndDisabled();

      if (changed) {
        editLocked(s, *edited, p);
      }

      ImGui::BeginDisabled(edited->builtIn || !edited->dirty);
      if (ImGui::Button("Save")) {
        if (writePresetFile(edited->name, edited->params)) {
          edited->saved = edited->params;
          edited->dirty = false;
          s.message = "Saved " + edited->name;
        } else {
          s.message = "Cannot write the preset file";
        }
      }
      ImGui::SameLine();
      if (ImGui::Button("Revert")) {
        edited->params = edited->saved;
        edited->dirty = false;
      }
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(edited->builtIn);
      if (ImGui::Button("Delete")) {
        std::error_code ec;
        fs::remove(directory() / (edited->name + ".conf"), ec);
        if (ec) {
          s.message = "Cannot delete: " + ec.message();
        } else {
          s.message = "Deleted " + edited->name;
          const std::string name = edited->name;
          s.presets.erase(std::remove_if(s.presets.begin(), s.presets.end(),
                                         [&](const Preset& q) { return q.name == name; }),
                          s.presets.end());
          edited = nullptr;
          gradePresetObject().setDeferred(std::string("None"));
        }
      }
      ImGui::EndDisabled();

      if (edited != nullptr) {
        ImGui::InputText("##gradeSaveAsName", s.saveAsName, sizeof(s.saveAsName));
        ImGui::SameLine();
        if (ImGui::Button("Save As")) {
          const std::string name = trim(s.saveAsName);
          if (!isValidFileName(name)) {
            s.message = "Choose a name that is not a built-in preset's and has no \\ / : * ? \" < > |";
          } else if (writePresetFile(name, edited->params)) {
            const Params values = edited->params;
            Preset* target = findLocked(s, name);
            if (target == nullptr) {
              Preset added;
              added.name = name;
              s.presets.push_back(added);
              target = &s.presets.back();
            }
            target->params = values;
            target->saved = values;
            target->dirty = false;
            s.message = "Saved " + name;
            gradePresetObject().setDeferred(name);
          } else {
            s.message = "Cannot write the preset file";
          }
        }
        RemixGui::SetTooltipToLastWidgetOnHover("Saves the fields above as a new preset file with this name, and picks it.");
      }
    }

    RemixGui::InputText("Grade Preset Folder", &gradePresetDirectoryObject());
    if (ImGui::Button("Reload Grade Presets")) {
      reloadLocked(s);
      s.message = "Reloaded";
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", directory().string().c_str());
    if (!s.message.empty()) {
      ImGui::TextUnformatted(s.message.c_str());
    }
  }

  void ColorGrade::showUserPresetPicker() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    ensureLoadedLocked(s);
    const std::string selected = gradePreset();
    if (ImGui::BeginCombo("Colour Grade", selected.c_str())) {
      for (const Preset& p : s.presets) {
        if (ImGui::Selectable(p.name.c_str(), p.name == selected)) {
          gradePresetObject().setDeferred(p.name);
        }
      }
      ImGui::EndCombo();
    }
  }

}
