#pragma once

// The colour grade controller: presets, the grade on screen, and blends between grades.
//
// A preset is a text file in rtx.tonemap.gradePresetDirectory (default rtx-remix/colorgrade in the game folder),
// one per preset, named <preset name>.conf, in the same key = value syntax as rtx.conf and the weather presets:
//   grade.matrix = m00, m01, m02, m10, m11, m12, m20, m21, m22   (row-major, default the identity)
//   grade.balance = r, g, b                                       (default 1, 1, 1)
//   grade.saturation = s                                          (default 1)
//   grade.contrast = c                                            (default 1)
//   grade.anchor = a                                              (default 0.18)
//   grade.shoulder = s                                            (default 0.01)
// A key that is missing takes its neutral value. The built-in presets ("None", "Warm Amber", "Olive Green") are not
// files and cannot be overwritten. rtx_color_grade_math.h holds the fields and the order they apply in.
//
// What shows: the grade the last request asked for, where a request is a change of rtx.tonemap.gradePreset (the
// menus or rtx.conf), a Remix API call (remixapi_SetColorGradePreset, SetColorGrade, MixColorGradePresets), or an
// edit in the developer menu's editor. A request blends from the grade on screen to the new one, so a request in
// the middle of a blend continues from where that blend was.
//
// Every function locks one mutex: the API calls come from the game's threads, the editor from the UI thread and
// frameConstants() from the render thread.

#include <string>

#include "rtx_option.h"
#include "rtx_color_grade_math.h"

namespace dxvk {

  class ColorGrade {
  public:
    RTX_OPTION_ARGS("rtx.tonemap", std::string, gradePreset, std::string("None"),
                    "The colour grade preset the tone mapper applies: a built-in one (None, Warm Amber, Olive Green) or a "
                    "preset file in rtx.tonemap.gradePresetDirectory, by name. A game's rtx.conf can set the default. "
                    "The Remix API can show other grades while the game runs; changing this option shows this preset again.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION("rtx.tonemap", float, gradeBlendSeconds, 0.0f,
               "How long the grade takes to blend to a newly picked rtx.tonemap.gradePreset, in seconds. 0 = at once. "
               "Remix API calls give their own blend time.");
    RTX_OPTION("rtx.tonemap", std::string, gradePresetDirectory, std::string("rtx-remix/colorgrade"),
               "Folder of the colour grade preset files (<name>.conf), relative to the game's folder unless absolute.");

    // The constants of the grade on screen for this frame. The first call of a frame advances the blend by the
    // frame's time, later calls of the same frame return the same constants.
    static color_grade::Gpu frameConstants(uint64_t frameId, float deltaSeconds);

    // Remix API requests. Each returns false, and changes nothing, for an unknown preset name.
    static bool setPreset(const std::string& name, float blendSeconds);
    static void setGrade(const color_grade::Params& params, float blendSeconds);
    static bool mixPresets(const std::string& nameA, const std::string& nameB, float weight);
    static color_grade::Params current();

    enum class SaveResult { Saved, InvalidName, WriteFailed };
    static SaveResult savePreset(const std::string& name, const color_grade::Params& params);

    // The preset the editor edits, which is rtx.tonemap.gradePreset, with its current (possibly unsaved) values.
    // Returns false when no preset has that name.
    static bool editedPreset(color_grade::Params& params, bool& builtIn);
    // Writes `params` into the edited preset as an unsaved edit, as the editor's fields do, and shows it.
    // Built-in presets are read-only, so an edit to one changes nothing.
    static void editPreset(const color_grade::Params& params);

    // The developer menu's preset editor, in the tone mapping settings.
    static void showImguiSettings();
    // The user menu's preset picker.
    static void showUserPresetPicker();
  };

}
