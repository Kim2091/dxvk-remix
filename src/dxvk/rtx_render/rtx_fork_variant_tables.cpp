// Per-pass shader variant tables. See rtx_fork_variant_tables.h.

#include "rtx_fork_variant_tables.h"

#include "rtx_fork_frame_features.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_pathtracer_gbuffer.h"
#include "rtx_pathtracer_integrate_direct.h"
#include "rtx_pathtracer_integrate_indirect.h"
#include "../dxvk_device.h"

namespace dxvk {

  namespace {
    constexpr uint32_t kPassCount = (uint32_t) VariantTablePass::Count;
    constexpr uint32_t kRateFrames = 60;
    constexpr uint32_t kFallbackWarnFrames = 300;
    constexpr uint32_t kSelfCheckLinesPerPass = 8;

    const char* const kPassNames[kPassCount] = { "gbuffer", "direct", "sharcUpdate", "integration", "assembly" };

    struct MismatchState {
      uint32_t lastLogFrame = UINT32_MAX;
      uint32_t suppressed = 0;
    } s_mismatch[kPassCount];

    struct FallbackState {
      const char* chosen = nullptr; // entry names (static strings) compared by pointer
      const char* bound = nullptr;
      uint32_t lastLogFrame = UINT32_MAX;
      uint32_t lastFallbackFrame = UINT32_MAX;
      uint32_t streak = 0;
      bool warned = false;
    } s_fallback[kPassCount];

    struct SelfCheckState {
      bool valid = false;
      uint64_t signature = 0;
      uint32_t lines[kPassCount] = {};
    } s_selfCheck;

    // Options the legacy selectors and the tables read. A change re-runs the exhaustive check.
    uint64_t selfCheckSignature() {
      uint64_t s = 0;
      uint32_t bit = 0;
      auto add = [&](const bool v) { s |= (v ? 1ull : 0ull) << bit; bit++; };
      add(DxvkPathtracerGbuffer::slimVariants());
      add(DxvkPathtracerIntegrateDirect::slimVariants());
      add(DxvkPathtracerIntegrateIndirect::slimVariants());
      add(DxvkPathtracerIntegrateIndirect::slimAssembly());
      add(DxvkPathtracerIntegrateIndirect::slimNonSharc());
      add(DxvkPathtracerIntegrateIndirect::slimKeepPom());
      add(RtxOptions::FrameFeatures::validate());
      add(RtxOptions::foliageRaysOmm2StateActive());
      // The direct core forms.
      add(DxvkPathtracerIntegrateDirect::coreMaterial());
      add(RtxOptions::CoreMaterial::build());
      add(RtxOptions::CoreMaterial::legacyRecord());
      return s;
    }

    uint32_t lowestBit(const uint32_t bits) {
      for (uint32_t i = 0; i < 32; i++) {
        if ((bits & (1u << i)) != 0) {
          return 1u << i;
        }
      }
      return 0;
    }
  }

  VariantSelection selectVariant(const VariantTable& table, const uint32_t modeKey, const uint32_t scene) {
    VariantSelection selection;
    int32_t previousMatch = -1;
    bool anyKeyedEntry = false;
    for (uint32_t i = 0; i < table.count; i++) {
      const VariantTableEntry& entry = table.entries[i];
      if (!variantEntryMatchesModeKey(entry, modeKey)) {
        continue;
      }
      anyKeyedEntry |= entry.modeMask != 0;
      if ((entry.removes & scene) == 0) {
        selection.entry = i;
        selection.blockedEntry = previousMatch;
        selection.blockedBits = previousMatch >= 0 ? (table.entries[previousMatch].removes & scene) : 0u;
        selection.noEntryForModeKey = !anyKeyedEntry;
        return selection;
      }
      previousMatch = (int32_t) i;
    }
    // Unreachable with a Full entry (empty mode mask, removes nothing) last.
    assert(false && "variant table without a Full entry");
    selection.entry = table.count - 1;
    selection.noEntryForModeKey = !anyKeyedEntry;
    return selection;
  }

  uint32_t nextMatchingVariant(const VariantTable& table, const uint32_t index, const uint32_t modeKey, const uint32_t scene) {
    for (uint32_t i = index + 1; i < table.count; i++) {
      const VariantTableEntry& entry = table.entries[i];
      if (variantEntryMatchesModeKey(entry, modeKey) && (entry.removes & scene) == 0) {
        return i;
      }
    }
    return UINT32_MAX;
  }

  std::string variantModeKeyToString(const VariantTable& table, const uint32_t modeKey) {
    std::string out = str::format("0x", std::hex, modeKey, std::dec, "(");
    bool first = true;
    for (uint32_t i = 0; i < table.modeKeyBitCount; i++) {
      if ((modeKey & (1u << i)) != 0) {
        out += first ? "" : "|";
        out += table.modeKeyNames[i];
        first = false;
      }
    }
    out += first ? "none)" : ")";
    return out;
  }

  std::string variantKeyBlockToString(const VariantTable& table, const VariantSelection& selection, const uint32_t modeKey,
                                      const uint32_t scene) {
    int32_t best = -1;
    uint32_t bestBits = 0;
    uint32_t bestCount = UINT32_MAX;
    for (uint32_t i = 0; i < selection.entry && i < table.count; i++) {
      const VariantTableEntry& entry = table.entries[i];
      if ((entry.removes & scene) != 0 || variantEntryMatchesModeKey(entry, modeKey)) {
        continue;
      }
      const uint32_t bits = (modeKey & entry.modeMask) ^ entry.modeKey;
      uint32_t count = 0;
      for (uint32_t b = bits; b != 0; b &= b - 1) {
        count++;
      }
      if (count < bestCount) {
        best = (int32_t) i;
        bestBits = bits;
        bestCount = count;
      }
    }
    if (best < 0) {
      return std::string();
    }
    std::string names;
    for (uint32_t i = 0; i < table.modeKeyBitCount; i++) {
      if ((bestBits & (1u << i)) != 0) {
        names += names.empty() ? "" : "|";
        names += table.modeKeyNames[i];
      }
    }
    return str::format(" keyBlockedBy=", table.entries[best].name, ":", names);
  }

  std::string variantSelectionToString(const VariantTable& table, const VariantSelection& selection, const uint32_t modeKey,
                                       const uint32_t boundEntry) {
    const uint32_t bound = boundEntry < table.count ? boundEntry : selection.entry;
    std::string out = str::format(" table entry=", table.entries[bound].name);
    if (bound != selection.entry) {
      out += str::format(" fallbackFrom=", table.entries[selection.entry].name);
    }
    if (selection.noEntryForModeKey) {
      out += str::format(" noEntryForModeKey modeKey=", variantModeKeyToString(table, modeKey));
    } else if (selection.blockedEntry >= 0) {
      const uint32_t bit = lowestBit(selection.blockedBits);
      std::string bitName = frameFeaturesSceneToString(bit);
      // frameFeaturesSceneToString gives "0x<hex>(<name>)"; keep the name.
      const size_t open = bitName.find('(');
      if (open != std::string::npos && bitName.size() > open + 1) {
        bitName = bitName.substr(open + 1, bitName.size() - open - 2);
      }
      out += str::format(" blockedBy=", table.entries[selection.blockedEntry].name, ":", bitName);
    } else {
      out += " blockedBy=none";
    }
    return out;
  }

  const char* variantTablePassName(const VariantTablePass pass) {
    const uint32_t i = (uint32_t) pass;
    return i < kPassCount ? kPassNames[i] : "?";
  }

  void reportVariantTableMismatch(const VariantTablePass pass, const char* tableEntry, const char* legacyLevel,
                                  const uint32_t modeKey, const uint32_t scene, const uint32_t frameId) {
    MismatchState& m = s_mismatch[(uint32_t) pass];
    if (m.lastLogFrame != UINT32_MAX && frameId - m.lastLogFrame < kRateFrames) {
      m.suppressed++;
      return;
    }
    Logger::err(str::format("[VariantTable] mismatch pass=", variantTablePassName(pass), " table=", tableEntry,
                            " legacy=", legacyLevel, " modeKey=0x", std::hex, modeKey, std::dec,
                            " scene=", frameFeaturesSceneToString(scene),
                            m.suppressed > 0 ? str::format(" (+", m.suppressed, " since last line)") : std::string(),
                            " (frame ", frameId, ")"));
    m.lastLogFrame = frameId;
    m.suppressed = 0;
  }

  void reportVariantTableSelfCheckMismatch(const VariantTablePass pass, VariantTableSelfCheckResult& result, const char* tableEntry,
                                           const char* legacyLevel, const std::string& inputs) {
    result.mismatches++;
    uint32_t& lines = s_selfCheck.lines[(uint32_t) pass];
    if (lines >= kSelfCheckLinesPerPass) {
      return;
    }
    lines++;
    Logger::err(str::format("[VariantTable] mismatch pass=", variantTablePassName(pass), " table=", tableEntry,
                            " legacy=", legacyLevel, " ", inputs, " (self-check)"));
  }

  void variantTablesSelfCheckIfNeeded() {
    const uint64_t signature = selfCheckSignature();
    if (s_selfCheck.valid && s_selfCheck.signature == signature) {
      return;
    }
    s_selfCheck.valid = true;
    s_selfCheck.signature = signature;
    for (uint32_t& lines : s_selfCheck.lines) {
      lines = 0;
    }

    VariantTableSelfCheckResult results[kPassCount];
    DxvkPathtracerGbuffer::runVariantTableSelfCheck(results[(uint32_t) VariantTablePass::GBuffer]);
    DxvkPathtracerIntegrateDirect::runVariantTableSelfCheck(results[(uint32_t) VariantTablePass::Direct]);
    DxvkPathtracerIntegrateIndirect::runVariantTableSelfCheck(results[(uint32_t) VariantTablePass::SharcUpdate],
                                                             results[(uint32_t) VariantTablePass::IndirectIntegration],
                                                             results[(uint32_t) VariantTablePass::Assembly]);
    // sharcUpdate covers the SHARC table (update and query), integration the non-SHARC TraceRay table.
    uint64_t total = 0;
    std::string perPass;
    for (uint32_t i = 0; i < kPassCount; i++) {
      perPass += str::format(" ", kPassNames[i], "=", results[i].checked, "/", results[i].mismatches);
      total += results[i].mismatches;
    }
    const std::string line = str::format("[VariantTable] self-check (masks/mismatches):", perPass,
                                         " options=0x", std::hex, signature, std::dec);
    if (total > 0) {
      Logger::err(line);
    } else {
      Logger::info(line);
    }
  }

  bool variantFallbackActive(RtxContext& ctx) {
    // Without the state cache nothing compiles in the background, so a cold entry would never become ready.
    return RtxOptions::Shader::slimFallbackWhileCompiling() && RtxOptions::Shader::variantTables() &&
           ctx.getCommonObjects()->pipelineManager().hasStateCache();
  }

  bool RtxContext::isComputeShaderPipelineReady(const Rc<DxvkShader>& shader) {
    // The same pipeline object the context binds (pipelines are keyed by the shader alone). Created here if this is the
    // first request, with forceNoSpecConstants like the state cache worker that compiles registered Remix shaders.
    DxvkComputePipelineShaders shaders;
    shaders.cs = shader;
    shaders.forceNoSpecConstants = true;
    DxvkComputePipeline* pipeline = m_common->pipelineManager().createComputePipeline(shaders);
    if (pipeline == nullptr) {
      return false;
    }
    // An object made by the context's own lookup (bound before this was asked) matches instances by the full state
    // vector, binding mask included, which is only final at dispatch. That shader was compiled on bind already, so binding
    // it again behaves as without the fallback.
    if (!pipeline->shaders().forceNoSpecConstants) {
      return true;
    }
    return pipeline->hasInstance(m_state.cp.state);
  }

  bool isVariantComputeShaderReady(RtxContext& ctx, const Rc<DxvkShader>& shader) {
    return shader != nullptr && ctx.isComputeShaderPipelineReady(shader);
  }

  void noteVariantFallback(const VariantTablePass pass, const char* chosen, const char* bound, const uint32_t frameId) {
    FallbackState& f = s_fallback[(uint32_t) pass];
    // Consecutive fallback frames, for the warning.
    f.streak = (f.lastFallbackFrame != UINT32_MAX && frameId - f.lastFallbackFrame <= 1) ? f.streak + 1 : 1;
    f.lastFallbackFrame = frameId;
    if (f.streak < kFallbackWarnFrames) {
      f.warned = false;
    } else if (!f.warned) {
      f.warned = true;
      Logger::warn(str::format("[VariantTable] fallback pass=", variantTablePassName(pass), " chosen=", chosen,
                               " still not compiled after ", kFallbackWarnFrames, " frames; bound=", bound,
                               " (frame ", frameId, ")"));
    }
    // Logged once per streak and entry pair (a new streak is logged again even with the same entries).
    if (f.streak == 1) {
      f.chosen = nullptr;
      f.bound = nullptr;
    }
    if (f.chosen == chosen && f.bound == bound) {
      return;
    }
    if (f.lastLogFrame != UINT32_MAX && frameId - f.lastLogFrame < kRateFrames) {
      return;
    }
    Logger::info(str::format("[VariantTable] fallback pass=", variantTablePassName(pass), " chosen=", chosen,
                             " bound=", bound, " (chosen not compiled yet) (frame ", frameId, ")"));
    f.chosen = chosen;
    f.bound = bound;
    f.lastLogFrame = frameId;
  }

  void showVariantTablesImguiSettings() {
    ImGui::BeginDisabled(!RtxOptions::Shader::variantTables());
    RemixGui::Checkbox("Shader Variants: Fall Back While Compiling", &RtxOptions::Shader::slimFallbackWhileCompilingObject());
    RemixGui::SetTooltipToLastWidgetOnHover(
      "rtx.shader.slimFallbackWhileCompiling (needs Variant Tables). A compute pass binds the chosen variant only once its\n"
      "pipeline is compiled; until then it binds the next matching (larger) table entry, so a cold variant does not stall the\n"
      "frame. Same image. [VariantTable] fallback in the log names the entries.");
    ImGui::EndDisabled();
  }

} // namespace dxvk
