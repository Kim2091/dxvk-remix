#pragma once

// NUMOS's settings (submodules/numos, settings_fields.inl) as Remix options.
//
// Every numos::settingInfos() entry is registered at static initialisation as an RtxOption under its
// own key (rtx.atmosphere.*, rtx.atmosphere.moonN.*), with its default, range, description and
// NUMOS_SETTING_NO_SAVE as NoSave. So rtx.conf, user.conf, the option layers, saving, the remix API's
// SetConfigVariable and the RemixGui widgets all work on them like on any built-in option, and the
// existing configs keep loading. The list itself is NUMOS's: nothing here names a setting.

#include <cstddef>

#include "numos/settings.h"
#include "rtx_option.h"

namespace dxvk {

  namespace NumosSettings {

    // One registered setting. Exactly one option pointer is set (by info->type); all are null when
    // the key collided with an option registered elsewhere.
    struct Entry {
      const numos::SettingInfo* info = nullptr;
      RtxOption<bool>* boolOption = nullptr;
      RtxOption<int32_t>* intOption = nullptr;
      RtxOption<uint32_t>* uintOption = nullptr;
      RtxOption<float>* floatOption = nullptr;
      RtxOption<Vector3>* vec3Option = nullptr;

      RtxOptionImpl* option() const;
    };

    // In numos::settingInfos() order.
    size_t count();
    const Entry& entry(size_t index);
    const Entry* find(const char* key);

    // The settings the options resolve to this frame, with the deprecated settings migrated into
    // their replacements (numos::migrateDeprecated; a setting counts as set when an option layer other
    // than the defaults holds it, whatever its value). A deprecated key found in a layer is consumed
    // once: the changed replacements are written into the strongest such layer and the deprecated keys
    // are removed from every layer, so they never override the options again and a save drops them.
    // Recomputed only when an option changed. Render thread.
    const numos::Settings& resolved();

    // Writes every option whose value differs from `settings` (a decoded seed, a preset baked into
    // the options), through the current RtxOptionLayerTarget like any dev-menu edit.
    void write(const numos::Settings& settings);

    // Routes NUMOS's log callback to the Remix log (each distinct message once). Idempotent.
    void installLog();
  }

}
