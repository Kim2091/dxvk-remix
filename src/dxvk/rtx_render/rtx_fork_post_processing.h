#pragma once

// Post-processing stack orchestration. The effects live in their own classes. This module owns the
// stable effect IDs, ordering rules, configuration, and the single dispatch boundary of the frame renderer.

#include <cstddef>
#include <string>
#include <vector>

#include "rtx_resources.h"

namespace dxvk {

  class DxvkContext;
  class RtxContext;

  class RtxPostProcessingStack {
  public:
    static void dispatch(
      Rc<RtxContext> ctx,
      Resources::RaytracingOutput& rtOutput,
      bool performSRGBConversion);

    static void showSettings(const Rc<DxvkContext>& ctx);

  private:
    enum class EffectId {
      Bloom,
      MotionBlur,
      DepthOfField,
      Tonemapping,
      NtscVhs,
      LensEffects,
      SRGBDither,
    };

    enum class EffectDomain {
      HDR,
      Display,
      Terminal,
    };

    struct EffectDescriptor {
      EffectId id;
      const char* name;
      const char* configId;
      EffectDomain domain;
      bool reorderable;
    };

    struct EffectEntry {
      bool external = false;
      EffectId builtInId = EffectId::Bloom;
      std::string externalId;
    };

    // A compile-time registry, not a plugin ABI. An effect is added only with a real dispatch adapter,
    // so a saved order never names an effect that does not run.

    static const EffectDescriptor& descriptor(EffectId id);
    static std::vector<EffectEntry> defaultOrder();
    static std::vector<EffectEntry> resolvedOrder();
    static size_t findEffect(const std::vector<EffectEntry>& order, const std::string& configId);
    static std::vector<std::string> storedOrderTokens();
    static std::vector<std::string> configIds(const std::vector<EffectEntry>& order);
    static std::string serializeOrder(const std::vector<EffectEntry>& order);
    // What a reorder writes back. Unlike serializeOrder, it keeps the stored effects that are not loaded.
    static std::string serializeOrderPreservingUnloaded(const std::vector<EffectEntry>& order);
    static bool moveEffect(std::vector<EffectEntry>& order, size_t from, size_t to);
    static bool canMove(const std::vector<EffectEntry>& order, size_t from, size_t to);
    static EffectEntry effectFromConfig(const std::string& configId, bool& valid);
    static std::string configId(const EffectEntry& entry);
    static std::string name(const EffectEntry& entry);
    static EffectDomain domain(const EffectEntry& entry);
    static bool reorderable(const EffectEntry& entry);
    static std::string trim(const std::string& value);

    RTX_OPTION("rtx.postfx", std::string, stackOrder,
               std::string("bloom,motion_blur,depth_of_field,tonemapping,ntsc_vhs,lens_effects,srgb_dither"),
               "Comma-separated post-processing effect order. Effects may be reordered within their color domain; tonemapping and sRGB/dither remain fixed pipeline anchors.");
  };

} // namespace dxvk
