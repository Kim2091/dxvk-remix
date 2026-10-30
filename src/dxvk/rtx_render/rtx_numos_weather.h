#pragma once

// Weather presets for the NUMOS sky provider. One system, in files, for every game.
//
// A preset is a text file in rtx.weather.presetDirectory (default rtx-remix/weather in the game
// folder), one per preset, named <preset name>.conf. It holds:
//   - any NUMOS settings, in rtx.conf syntax (rtx.atmosphere.cloudCoverageMean = 0.6, ...): the
//     preset overrides exactly those keys over the user's settings, weather fields and others alike
//     (sun, cloud colour, a dust layer);
//   - a float or colour value written "x<value>" (x1.6, or x0.75, 0.9, 0.6 per channel) is a multiplier
//     on the user's own setting, so the preset follows the user's tuning;
//   - weather.bindings = id, id, ...: the game weathers it stands for (any strings the game reports);
//   - weather.transitionSeconds = s: how long the sky takes to blend into it (optional);
//   - weather.enabled = false: the preset no longer answers to its bindings (optional).
// NUMOS's own presets (Clear ... Altostratus) are written to the folder as ordinary files once per
// folder (.numos-presets-written records which); from then on they are edited, renamed and deleted
// like any other, and a deleted one stays deleted.
//
// rtx.weather.legacyPresetFile names a preset file in the older single-file format ([name] sections
// of coverage, density, layer2..., weather = <hex id> <name> lines). Each section is converted into a
// preset file once (.legacy-presets-imported records which). A preset of the same name is replaced,
// keeps its bindings and is backed up as <name>.conf.bak. docs/RemixSkyAPI.md, "Older preset files",
// lists how each old key converts.
//
// The game drives it through options it sets with the remix API's SetConfigVariable, all NoSave:
//   rtx.weather.gameWeather          the current game weather: one id, or several separated by commas,
//                                    most specific first (e.g. "0002B52A, CommonwealthClear, pleasant"):
//                                    the first id a preset is bound to wins, so a preset bound to a broad
//                                    class covers every weather no other preset names
//   rtx.weather.gameWeatherOutgoing  the weather it is leaving, when the game blends weathers itself
//   rtx.weather.gameWeatherBlend     0..1: the game's own blend from outgoing to current; < 0 (default):
//                                    the runtime blends over the preset's transition time
// What shows, strongest first: the tuned look while tuning, the previewed preset, a held seed, the
// preset bound to the current id while rtx.weather.followGame is on, rtx.weather.preset (chosen in the
// dev menu), else the settings alone.
//
// Blending: the weather fields follow NUMOS's rules (blendWeather: extinction for visibilities, the
// short way round for angles, layer 2 as a unit); other floats and colours are interpolated, integers
// and switches change halfway.

#include <memory>
#include <string>
#include <vector>

#include "rtx_option.h"

namespace numos {
  struct Settings;
  struct WeatherSnapshot;
}

namespace dxvk {

  class DxvkDevice;

  class NumosWeather {
  public:
    RTX_OPTION("rtx.weather", std::string, presetDirectory, std::string("rtx-remix/weather"),
      "Folder of the weather preset files (<name>.conf), relative to the game's folder unless absolute.");
    RTX_OPTION("rtx.weather", std::string, preset, std::string(""),
      "The weather preset the sky shows while the game reports no weather bound to a preset (or rtx.weather.followGame "
      "is off). Empty: the NUMOS settings alone.");
    RTX_OPTION("rtx.weather", bool, followGame, true,
      "Show the preset bound to the weather the game reports (rtx.weather.gameWeather). Off: rtx.weather.preset only.");
    RTX_OPTION("rtx.weather", float, transitionSeconds, 10.0f,
      "Blend time into a new weather, for presets that set no weather.transitionSeconds of their own. 0 = instant.");
    RTX_OPTION_FLAG("rtx.weather", std::string, gameWeather, std::string(""), RtxOptionFlags::NoSave,
      "Set by the game integration (remix API SetConfigVariable): the game's current weather, as one id or several "
      "separated by commas, most specific first. The first id a preset's weather.bindings lists picks that preset.");
    RTX_OPTION_FLAG("rtx.weather", std::string, gameWeatherOutgoing, std::string(""), RtxOptionFlags::NoSave,
      "Set by the game integration: the weather the game is blending away from, with rtx.weather.gameWeatherBlend.");
    RTX_OPTION_FLAG("rtx.weather", float, gameWeatherBlend, -1.0f, RtxOptionFlags::NoSave,
      "Set by the game integration: 0..1 = the game's own blend from rtx.weather.gameWeatherOutgoing to "
      "rtx.weather.gameWeather; negative = the runtime blends over the preset's transition time.");
    RTX_OPTION_FLAG("rtx.weather", std::string, legacyPresetFile, std::string(""), RtxOptionFlags::NoSave,
      "A weather preset file in the older single-file format ([name] sections with coverage, density, layer2... keys) "
      "to convert into preset files in rtx.weather.presetDirectory. Each section is converted once; a preset of the same "
      "name is replaced and backed up as <name>.conf.bak. Usually set by the game integration. Empty: nothing is imported.");

    struct Preset {
      std::string name;                  // the file name without .conf
      std::vector<std::string> bindings; // game weather ids
      float transitionSeconds = -1.0f;   // < 0: rtx.weather.transitionSeconds
      bool enabled = true;               // false: the bindings are ignored
      std::unique_ptr<numos::Settings> values;  // only the keys in `set` mean anything
      std::vector<bool> set;             // indexed like numos::settingInfos()
      std::vector<bool> relative;        // indexed like `set`: the value multiplies the user's setting
      bool dirty = false;                // edited since the last load or save
      bool numosBuiltIn = false;         // the name of one of NUMOS's presets (it can be reset)

      Preset();
      Preset(const Preset&);
      Preset& operator=(const Preset&);
      ~Preset();
    };

    explicit NumosWeather(DxvkDevice* device);
    ~NumosWeather();

    // Once per rendered frame, render thread: the settings NUMOS renders this frame, the presets
    // (and any held seed weather) applied and blended over `base`.
    const numos::Settings& update(const numos::Settings& base, float deltaSeconds);

    // The weather fields of what update() last returned (for seeds).
    const numos::WeatherSnapshot* currentSnapshot() const;
    // Shows a decoded seed's weather until the preset or the game weather changes.
    void holdSnapshot(const numos::WeatherSnapshot& weather);

    // Presets, as loaded. Editing one changes the sky at once while it is the target.
    std::vector<Preset>& presets() { return m_presets; }
    std::string directory() const;
    void reload();
    bool save(Preset& preset, std::string* error = nullptr);
    // A new preset (unsaved until save()). `from`: copies its keys; null: the weather fields of the
    // current sky.
    Preset& create(const std::string& name, const Preset* from);
    // A new preset (unsaved) that sets every weather field and every key in `keys` to its value in `look`.
    Preset& createFromLook(const std::string& name, const numos::Settings& look, const std::vector<bool>& keys);
    // Reads a preset file from any path into the preset folder, under a name no preset has, and saves it there.
    // Returns null, with `error` set, when the file cannot be read or written.
    Preset* loadFile(const std::string& path, std::string* error = nullptr);
    bool rename(Preset& preset, const std::string& newName, std::string* error = nullptr);
    bool remove(const std::string& name, std::string* error = nullptr);
    // A NUMOS preset back to NUMOS's definition (unsaved).
    void resetToNumos(Preset& preset);
    // The preset back to its file as last loaded or saved.
    // Returns false when it has never been saved.
    bool revert(Preset& preset);
    bool hasSavedCopy(const Preset& preset) const;
    // Sets the key in the preset to the sky's current value of it.
    void addKey(Preset& preset, size_t settingIndex);
    void removeKey(Preset& preset, size_t settingIndex);
    Preset* find(const std::string& name);
    // The enabled preset bound to a reported weather (one id or a comma list, most specific first), or null.
    Preset* boundPreset(const std::string& gameIds);
    // A name no preset has: `base`, else `base 2`, `base 3`, ...
    std::string uniqueName(const std::string& base);

    // Shows the named preset whatever the game reports. Empty: no preview.
    void setPreview(const std::string& name) { m_preview = name; }
    const std::string& preview() const { return m_preview; }

    // Stores what the sky shows now into `preset`: every weather field and every key it already sets,
    // as absolute values. Null: a new preset named after `weatherId` and bound to it. Unsaved.
    Preset& storeLook(Preset* preset, const std::string& weatherId);

    // Tuning: the sky's current look is written into the user's settings and the presets are paused, so
    // the dev-menu sliders show exactly what will be stored. finishTuning stores the tuned settings into
    // the preset (or a new one bound to the weather) when `keep` is set, and restores the user's settings
    // either way. Returns the stored preset, or null.
    void beginTuning(const std::string& presetName, const std::string& weatherId);
    Preset* finishTuning(bool keep);
    bool tuning() const { return m_tuning; }
    const std::string& tuningPreset() const { return m_tuningPreset; }
    const std::string& tuningWeather() const { return m_tuningWeather; }

    // Converts the sections of an older single-file preset file (rtx.weather.legacyPresetFile) into
    // preset files. `again`: also the sections imported before. Returns the number converted.
    size_t importLegacy(const std::string& path, bool again, std::string* report = nullptr);

    // Status for the dev menu.
    const std::string& activePreset() const { return m_targetName; }
    float transitionProgress() const;
    const numos::Settings& lastBase() const;
    const numos::Settings& current() const;

  private:
    void ensureLoaded();
    void writeMissingNumosPresets();
    void applyPreset(const Preset* preset, numos::Settings& settings) const;
    void rememberSaved(const Preset& preset);
    static void blend(const numos::Settings& a, const numos::Settings& b, float t, numos::Settings& out);

    struct State;
    std::unique_ptr<State> m_state;
    std::vector<Preset> m_presets;
    std::vector<Preset> m_saved;  // each preset as last loaded or saved, by name
    bool m_loaded = false;
    std::string m_targetName;  // preset (or "(seed)") the sky is blending to / showing
    std::string m_preview;
    bool m_tuning = false;
    std::string m_tuningPreset;
    std::string m_tuningWeather;
  };

}
