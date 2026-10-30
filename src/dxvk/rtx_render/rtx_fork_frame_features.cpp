// Per-frame feature mask. See rtx_fork_frame_features.h.

#include "rtx_fork_frame_features.h"

#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_imgui.h"
#include "rtx_scene_manager.h"
#include "rtx_neural_radiance_cache.h"
#include "rtx_sparse_rendering.h"
#include "rtx_fork_light_bvh.h"
#include "rtx_pathtracer_gbuffer.h"
#include "rtx_pathtracer_integrate_direct.h"
#include "rtx_pathtracer_integrate_indirect.h"
#include "rtx/pass/raytrace_args.h"
#include "rtx/pass/frame_features_validate.h"
#include "rtx_fork_light_bvh_pass.h"
#include "rtx/pass/instance_definitions.h"
#include "../dxvk_device.h"

namespace dxvk {

  namespace {
    RtFrameFeatures s_lastFeatures;
    // classSss-vs-sssTlas log state (-1 = never logged, 0 = match, 1 = mismatch).
    int32_t s_classSssState = -1;
    uint32_t s_classSssLogFrame = UINT32_MAX;

    // Logged state (logSlimLevel pattern, rtx_pathtracer_gbuffer.cpp).
    struct LoggedState {
      bool valid = false;
      uint32_t scene = 0;
      uint32_t mode = 0;
      uint32_t lightTypes = 0;
      uint32_t sticky = 0;
      bool live = false;
      uint32_t lastLogFrame = UINT32_MAX;
    } s_logged;

    template<typename Bit>
    struct BitName {
      Bit bit;
      const char* name;
    };

    const BitName<RtFrameFeatures::SceneBit> kSceneNames[] = {
      { RtFrameFeatures::kSceneThinFilm, "thinFilm" },
      { RtFrameFeatures::kScenePortalsActive, "portalsActive" },
      { RtFrameFeatures::kScenePortalMaterial, "portalMaterial" },
      { RtFrameFeatures::kSceneBillboardDynamic, "billboardDynamic" },
      { RtFrameFeatures::kSceneThinOpaque, "thinOpaque" },
      { RtFrameFeatures::kSceneSssTlas, "sssTlas" },
      { RtFrameFeatures::kScenePom, "pom" },
      { RtFrameFeatures::kSceneTranslucentDynamic, "translucentDynamic" },
      { RtFrameFeatures::kSceneUnorderedEmpty, "unorderedEmpty" },
      { RtFrameFeatures::kSceneTranslucentInstance, "translucentInstance" },
      { RtFrameFeatures::kSceneRaytracedRenderTarget, "raytracedRenderTarget" },
      { RtFrameFeatures::kSceneLegacySurface, "legacySurface" },
      { RtFrameFeatures::kSceneEye, "eye" },
      { RtFrameFeatures::kSceneClassTranslucent, "classTranslucent" },
      { RtFrameFeatures::kSceneClassSss, "classSss" },
      { RtFrameFeatures::kSceneClassFallback, "classFallback" },
      { RtFrameFeatures::kSceneTextureOps, "textureOps" },
    };

    const BitName<RtFrameFeatures::ModeBit> kModeNames[] = {
      { RtFrameFeatures::kModeWboit, "wboit" },
      { RtFrameFeatures::kModeNrc, "nrc" },
      { RtFrameFeatures::kModeNrcDeferred, "nrcDeferred" },
      { RtFrameFeatures::kModeThinFilmOverride, "thinFilmOverride" },
      { RtFrameFeatures::kModeBillboardsPrimary, "billboardsPrimary" },
      { RtFrameFeatures::kModeSssThinOpaqueOpt, "sssThinOpaqueOpt" },
      { RtFrameFeatures::kModeSssProfileOpt, "sssProfileOpt" },
      { RtFrameFeatures::kModePomNeeCache, "pomNeeCache" },
      { RtFrameFeatures::kModeDirectTranslucentShadows, "directTranslucentShadows" },
      { RtFrameFeatures::kModeIndirectTranslucentShadows, "indirectTranslucentShadows" },
      { RtFrameFeatures::kModeNeeCacheOnly, "neeCacheOnly" },
      { RtFrameFeatures::kModeOmm2State, "omm2State" },
      { RtFrameFeatures::kModeDedicatedSun, "dedicatedSun" },
      { RtFrameFeatures::kModeSamplerBvh, "samplerBvh" },
      { RtFrameFeatures::kModeBvhSelectPass, "bvhSelectPass" },
      { RtFrameFeatures::kModeAtmosphereNee, "atmosphereNee" },
      { RtFrameFeatures::kModeDebugView, "debugView" },
    };

    template<typename Bit, size_t N>
    std::string bitsToString(uint32_t bits, const BitName<Bit> (&names)[N]) {
      std::string out = str::format("0x", std::hex, bits, std::dec, "(");
      bool first = true;
      for (const auto& entry : names) {
        if ((bits & entry.bit) != 0) {
          if (!first) {
            out += "|";
          }
          out += entry.name;
          first = false;
        }
      }
      if (first) {
        out += "none";
      }
      out += ")";
      return out;
    }

    uint32_t stickyBits(const RtFrameFeatures& f) {
      return (f.counts.stickyThinOpaque ? 1u : 0u) | (f.counts.stickyThinFilm ? 2u : 0u) | (f.counts.stickyRayPortal ? 4u : 0u);
    }

    bool frameFeaturesLogWanted() {
      return DxvkPathtracerGbuffer::slimVariants() ||
             DxvkPathtracerIntegrateDirect::slimVariants() ||
             DxvkPathtracerIntegrateIndirect::slimVariants() ||
             DxvkPathtracerIntegrateIndirect::slimNonSharc() ||
             DxvkPathtracerIntegrateIndirect::slimAssembly() ||
             RtxOptions::FrameFeatures::liveMaterialCounts() ||
             RtxOptions::FrameFeatures::validate() ||
             RtxOptions::CoreMaterial::census();
    }
  }

  RtFrameFeatures computeFrameFeatures(RtxContext& ctx, const RaytraceArgs& args) {
    RtFrameFeatures f;
    const SceneManager& sceneManager = ctx.getSceneManager();
    const uint32_t frameId = ctx.getDevice()->getCurrentFrameId();
    f.frameId = frameId;

    // Counts (log) and the two sources of the material scene bits.
    f.counts.thinOpaque = sceneManager.getLiveThinOpaqueCount();
    f.counts.thinFilm = sceneManager.getLiveThinFilmCount();
    f.counts.rayPortal = sceneManager.getLiveRayPortalCount();
    f.counts.sss = sceneManager.getLiveSssCount();
    f.counts.pom = sceneManager.getLivePomCount();
    f.counts.translucent = sceneManager.getLiveTranslucentCount();
    f.counts.raytracedRenderTarget = sceneManager.getLiveRaytracedRenderTargetCount();
    f.counts.stickyThinOpaque = sceneManager.hasCreatedThinOpaqueMaterial();
    f.counts.stickyThinFilm = sceneManager.hasCreatedThinFilmMaterial();
    f.counts.stickyRayPortal = sceneManager.hasCreatedRayPortalMaterial();

    // With liveMaterialCounts, rows 5, 7 and 11 come from live instance counts instead of the sticky flags. A live count is a superset of what
    // the TLAS holds (an instance alive but not drawn still counts), so a clear bit is safe.
    f.liveMaterialCounts = RtxOptions::FrameFeatures::liveMaterialCounts();
    const bool thinFilm = f.liveMaterialCounts ? f.counts.thinFilm > 0 : f.counts.stickyThinFilm;
    const bool portalMaterial = f.liveMaterialCounts ? f.counts.rayPortal > 0 : f.counts.stickyRayPortal;
    const bool thinOpaque = f.liveMaterialCounts ? f.counts.thinOpaque > 0 : f.counts.stickyThinOpaque;

    uint32_t scene = 0;
    scene |= thinFilm ? RtFrameFeatures::kSceneThinFilm : 0u;
    // Same test as includePortals in the G-buffer and indirect passes.
    scene |= (RtxOptions::rayPortalModelTextureHashes().size() > 0 || args.numActiveRayPortals > 0) ? RtFrameFeatures::kScenePortalsActive : 0u;
    scene |= portalMaterial ? RtFrameFeatures::kScenePortalMaterial : 0u;
    if (args.enableBillboardOrientationCorrection) {
      for (const IntersectionBillboard& billboard : sceneManager.getInstanceManager().getBillboards()) {
        if ((billboard.instanceMask & OBJECT_MASK_ALL_DYNAMIC) != 0) {
          scene |= RtFrameFeatures::kSceneBillboardDynamic;
          break;
        }
      }
    }
    scene |= thinOpaque ? RtFrameFeatures::kSceneThinOpaque : 0u;
    scene |= sceneManager.getAccelManager().isSssTlasActive() ? RtFrameFeatures::kSceneSssTlas : 0u;
    scene |= args.pomMode != DisplacementMode::Off ? RtFrameFeatures::kScenePom : 0u;
    scene |= sceneManager.getInstanceManager().hasTranslucentDynamicInstance(frameId) ? RtFrameFeatures::kSceneTranslucentDynamic : 0u;
    scene |= sceneManager.getAccelManager().isUnorderedTlasEmpty() ? RtFrameFeatures::kSceneUnorderedEmpty : 0u;
    // Material bits. The direct integrator's variant table reads kSceneRaytracedRenderTarget.
    scene |= f.counts.translucent > 0 ? RtFrameFeatures::kSceneTranslucentInstance : 0u;
    scene |= f.counts.raytracedRenderTarget > 0 ? RtFrameFeatures::kSceneRaytracedRenderTarget : 0u;
    // Exact from the surfaces with rtx.coreMaterial.legacyRecord (kept on frames that skip the upload). Otherwise
    // conservative, every frame may have legacy surfaces and eyes.
    bool anyLegacy = true, anyTextureOps = true, anyEye = true;
    if (!sceneManager.getCoreMaterialLegacyBits(anyLegacy, anyTextureOps, anyEye)) {
      anyLegacy = true;
      anyTextureOps = true;
      anyEye = true;
    }
    scene |= anyLegacy ? RtFrameFeatures::kSceneLegacySurface : 0u;
    scene |= anyTextureOps ? RtFrameFeatures::kSceneTextureOps : 0u;
    scene |= anyEye ? RtFrameFeatures::kSceneEye : 0u;
    // Class bits from the core upload's class counts (rtx.coreMaterial.build), kept on frames that skip the upload.
    f.counts.coreClassesValid = sceneManager.getCoreMaterialClassCounts(f.counts.coreClasses);
    if (f.counts.coreClassesValid) {
      scene |= f.counts.coreClasses[1] > 0 ? RtFrameFeatures::kSceneClassTranslucent : 0u;
      scene |= f.counts.coreClasses[2] > 0 ? RtFrameFeatures::kSceneClassSss : 0u;
      scene |= f.counts.coreClasses[3] > 0 ? RtFrameFeatures::kSceneClassFallback : 0u;
    }
    f.scene = scene;

    for (uint32_t t = 0; t < RtFrameFeatures::kLightTypeCount; t++) {
      f.counts.lights[t] = args.lightRanges[t].count;
      f.lightTypes |= args.lightRanges[t].count > 0 ? (1u << t) : 0u;
    }

    const bool nrcEnabled = ctx.getCommonObjects()->metaNeuralRadianceCache().isActive();
    f.gbufferFeatureVariant = DxvkPathtracerGbuffer::frameRayQueryFeatureVariant(args, nrcEnabled);

    uint32_t mode = 0;
    mode |= RtxOptions::wboitEnabled() ? RtFrameFeatures::kModeWboit : 0u;
    mode |= nrcEnabled ? RtFrameFeatures::kModeNrc : 0u;
    mode |= SparseRendering::shouldDeferNrcTrainingSetup(args.sparseRenderingArgs) ? RtFrameFeatures::kModeNrcDeferred : 0u;
    mode |= args.opaqueMaterialArgs.enableThinFilmOverride ? RtFrameFeatures::kModeThinFilmOverride : 0u;
    mode |= (args.enableBillboardOrientationCorrection && args.useIntersectionBillboardsOnPrimaryRays) ? RtFrameFeatures::kModeBillboardsPrimary : 0u;
    // args.sssArgs carries this frame's rtx.subsurface.* values (updateRaytraceArgsConstantBuffer).
    mode |= args.sssArgs.enableThinOpaque ? RtFrameFeatures::kModeSssThinOpaqueOpt : 0u;
    mode |= args.sssArgs.enableDiffusionProfile ? RtFrameFeatures::kModeSssProfileOpt : 0u;
    mode |= args.pomEnableNEECache != 0 ? RtFrameFeatures::kModePomNeeCache : 0u;
    mode |= RtxOptions::enableDirectTranslucentShadows() ? RtFrameFeatures::kModeDirectTranslucentShadows : 0u;
    mode |= args.enableIndirectTranslucentShadows != 0 ? RtFrameFeatures::kModeIndirectTranslucentShadows : 0u;
    mode |= (args.neeCacheArgs.enable && args.neeCacheArgs.enableAnalyticalLight && args.lightBvhOverrideNeeCacheAnalytic == 0)
      ? RtFrameFeatures::kModeNeeCacheOnly : 0u;
    mode |= (RtxOptions::getEnableOpacityMicromap() && RtxOptions::foliageRaysOmm2StateActive()) ? RtFrameFeatures::kModeOmm2State : 0u;
    mode |= LightBvh::dedicatedSunActive() ? RtFrameFeatures::kModeDedicatedSun : 0u;
    mode |= LightBvh::active() ? RtFrameFeatures::kModeSamplerBvh : 0u;
    mode |= (LightBvh::active() && LightBvh::bvhSeparateSelectPass()) ? RtFrameFeatures::kModeBvhSelectPass : 0u;
    // The sky provider folds its cloud shadow whenever skyMode is NUMOS (sky_provider.slangh skyProviderActive, and
    // NUMOS decides where the shadow applies). The option, not args.skyMode, so a frame NUMOS forces off still counts.
    mode |= RtxOptions::skyMode() == SkyMode::Numos ? RtFrameFeatures::kModeAtmosphereNee : 0u;
    mode |= args.debugView != 0 ? RtFrameFeatures::kModeDebugView : 0u;
    f.mode = mode;

    // With diffusion profiles on, the SSS extension (and so class 2) exists exactly when the SSS TLAS is built.
    // Logged when the agreement changes, at most once per 300 frames.
    if (f.counts.coreClassesValid && args.sssArgs.enableDiffusionProfile) {
      const bool classSss = f.counts.coreClasses[2] > 0;
      const bool sssTlas = (scene & RtFrameFeatures::kSceneSssTlas) != 0;
      const int32_t state = classSss == sssTlas ? 0 : 1;
      if (state != s_classSssState && (s_classSssLogFrame == UINT32_MAX || frameId - s_classSssLogFrame >= 300)) {
        Logger::info(str::format("[CoreMaterial] classSss=", f.counts.coreClasses[2], " sssTlas=", sssTlas ? 1 : 0,
                                 state != 0 ? " MISMATCH" : " match", " (frame ", frameId, ")"));
        s_classSssState = state;
        s_classSssLogFrame = frameId;
      }
    }

    s_lastFeatures = f;
    return f;
  }

  std::string frameFeaturesSceneToString(uint32_t scene) {
    return bitsToString(scene, kSceneNames);
  }

  std::string frameFeaturesModeToString(uint32_t mode) {
    return bitsToString(mode, kModeNames);
  }

  void logFrameFeatures(const RtFrameFeatures& f) {
    if (!frameFeaturesLogWanted()) {
      return;
    }
    const uint32_t sticky = stickyBits(f);
    if (s_logged.valid && s_logged.scene == f.scene && s_logged.mode == f.mode && s_logged.lightTypes == f.lightTypes &&
        s_logged.sticky == sticky && s_logged.live == f.liveMaterialCounts) {
      return;
    }
    if (s_logged.lastLogFrame != UINT32_MAX && f.frameId - s_logged.lastLogFrame < 60) {
      return;
    }
    Logger::info(str::format(
      "[FrameFeatures] scene=", frameFeaturesSceneToString(f.scene),
      " lights=sphere:", f.counts.lights[0], ",rect:", f.counts.lights[1], ",disk:", f.counts.lights[2],
      ",cylinder:", f.counts.lights[3], ",distant:", f.counts.lights[4],
      " mode=", frameFeaturesModeToString(f.mode),
      " gbufferFeatures=", f.gbufferFeatureVariant,
      " materialBits=", f.liveMaterialCounts ? "live" : "sticky",
      " live=thinOpaque:", f.counts.thinOpaque, ",thinFilm:", f.counts.thinFilm, ",rayPortal:", f.counts.rayPortal,
      ",sss:", f.counts.sss, ",pom:", f.counts.pom, ",translucent:", f.counts.translucent,
      ",raytracedRenderTarget:", f.counts.raytracedRenderTarget,
      " sticky=thinOpaque:", f.counts.stickyThinOpaque ? 1 : 0, ",thinFilm:", f.counts.stickyThinFilm ? 1 : 0,
      ",rayPortal:", f.counts.stickyRayPortal ? 1 : 0,
      f.counts.coreClassesValid ? str::format(" coreClasses=", f.counts.coreClasses[0], ",", f.counts.coreClasses[1], ",",
                                              f.counts.coreClasses[2], ",", f.counts.coreClasses[3]) : std::string(),
      " (frame ", f.frameId, ")"));
    s_logged.valid = true;
    s_logged.scene = f.scene;
    s_logged.mode = f.mode;
    s_logged.lightTypes = f.lightTypes;
    s_logged.sticky = sticky;
    s_logged.live = f.liveMaterialCounts;
    s_logged.lastLogFrame = f.frameId;
  }

  const RtFrameFeatures& lastFrameFeatures() {
    return s_lastFeatures;
  }

  // ---- rtx.frameFeatures.validate -----------------------------------------------------------------------------------

  namespace {
    constexpr uint32_t kValidateSlots = 4;           // read-backs in flight
    constexpr uint32_t kValidateSummaryFrames = 300;

    struct ValidateSlot {
      Rc<DxvkBuffer> readback;
      Rc<DxvkGpuQuery> ready;
      bool pending = false;
      RtFrameFeatures mask;
      bool gbuffer = false;
      bool direct = false;
    };

    struct SeenName {
      uint32_t bit;
      const char* name;
    };
    const SeenName kSeenNames[] = {
      { FRAME_FEATURES_SEEN_THIN_OPAQUE, "thinOpaque" },
      { FRAME_FEATURES_SEEN_THIN_FILM, "thinFilm" },
      { FRAME_FEATURES_SEEN_PORTAL, "portal" },
      { FRAME_FEATURES_SEEN_POM, "pom" },
      { FRAME_FEATURES_SEEN_SSS_PROFILE, "sssProfile" },
      { FRAME_FEATURES_SEEN_UNORDERED, "unordered" },
      { FRAME_FEATURES_SEEN_LEGACY, "legacySurface" },
      { FRAME_FEATURES_SEEN_EYE, "eye" },
      { 1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + 0), "lightSphere" },
      { 1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + 1), "lightRect" },
      { 1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + 2), "lightDisk" },
      { 1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + 3), "lightCylinder" },
      { 1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + 4), "lightDistant" },
    };
    constexpr uint32_t kSeenNameCount = sizeof(kSeenNames) / sizeof(kSeenNames[0]);

    struct ValidateState {
      ValidateSlot slots[kValidateSlots];
      bool wasEnabled = false;
      bool frameGbuffer = false;
      bool frameDirect = false;
      // Summary over kValidateSummaryFrames.
      uint32_t summaryStartFrame = UINT32_MAX;
      uint32_t checkedFrames = 0;
      uint32_t gbufferFrames = 0;
      uint32_t directFrames = 0;
      uint32_t seen = 0;
      uint32_t violations = 0;
      uint32_t lastViolationLog[kSeenNameCount] = {};
      bool violationLogged[kSeenNameCount] = {};
    } s_validate;

    // Whether the mask allows what a seen bit reports: the bit the slim selectors test for it.
    bool maskAllows(const RtFrameFeatures& f, uint32_t seenBit) {
      switch (seenBit) {
      case FRAME_FEATURES_SEEN_THIN_OPAQUE: return f.has(RtFrameFeatures::kSceneThinOpaque);
      case FRAME_FEATURES_SEEN_THIN_FILM: return f.has(RtFrameFeatures::kSceneThinFilm) || f.hasMode(RtFrameFeatures::kModeThinFilmOverride);
      case FRAME_FEATURES_SEEN_PORTAL: return f.has(RtFrameFeatures::kScenePortalsActive) || f.has(RtFrameFeatures::kScenePortalMaterial);
      case FRAME_FEATURES_SEEN_POM: return f.has(RtFrameFeatures::kScenePom);
      case FRAME_FEATURES_SEEN_SSS_PROFILE: return f.has(RtFrameFeatures::kSceneSssTlas);
      case FRAME_FEATURES_SEEN_UNORDERED: return !f.has(RtFrameFeatures::kSceneUnorderedEmpty);
      // The validation kernels test the legacy union (texture ops included), so either bit allows it.
      case FRAME_FEATURES_SEEN_LEGACY: return f.has(RtFrameFeatures::kSceneLegacySurface) || f.has(RtFrameFeatures::kSceneTextureOps);
      case FRAME_FEATURES_SEEN_EYE: return f.has(RtFrameFeatures::kSceneEye);
      default:
        for (uint32_t t = 0; t < FRAME_FEATURES_SEEN_LIGHT_TYPE_COUNT; t++) {
          if (seenBit == (1u << (FRAME_FEATURES_SEEN_LIGHT_TYPE_SHIFT + t))) {
            return (f.lightTypes & (1u << t)) != 0;
          }
        }
        return true;
      }
    }

    std::string seenToString(uint32_t seen) {
      std::string out = str::format("0x", std::hex, seen, std::dec, "(");
      bool first = true;
      for (const SeenName& entry : kSeenNames) {
        if ((seen & entry.bit) != 0) {
          out += first ? "" : "|";
          out += entry.name;
          first = false;
        }
      }
      out += first ? "none)" : ")";
      return out;
    }

    void checkSlot(ValidateSlot& slot, uint32_t frameId) {
      const uint32_t seen = *static_cast<const uint32_t*>(slot.readback->mapPtr(0));
      ValidateState& v = s_validate;
      v.checkedFrames++;
      v.gbufferFrames += slot.gbuffer ? 1u : 0u;
      v.directFrames += slot.direct ? 1u : 0u;
      v.seen |= seen;
      for (uint32_t i = 0; i < kSeenNameCount; i++) {
        const uint32_t bit = kSeenNames[i].bit;
        if ((seen & bit) == 0 || maskAllows(slot.mask, bit)) {
          continue;
        }
        v.violations++;
        // Each bit at most once per summary interval.
        if (v.violationLogged[i] && frameId - v.lastViolationLog[i] < kValidateSummaryFrames) {
          continue;
        }
        v.violationLogged[i] = true;
        v.lastViolationLog[i] = frameId;
        Logger::warn(str::format("[FrameFeatures] VIOLATION hit=", kSeenNames[i].name,
                                 " mask=scene=", frameFeaturesSceneToString(slot.mask.scene),
                                 ",mode=", frameFeaturesModeToString(slot.mask.mode),
                                 ",lights=0x", std::hex, slot.mask.lightTypes, std::dec,
                                 " materialBits=", slot.mask.liveMaterialCounts ? "live" : "sticky",
                                 " (frame ", slot.mask.frameId, ")"));
      }
    }
  }

  void noteFrameFeaturesValidateKernel(FrameFeaturesValidateKernel kernel) {
    if (kernel == FrameFeaturesValidateKernel::GBuffer) {
      s_validate.frameGbuffer = true;
    } else {
      s_validate.frameDirect = true;
    }
  }

  void frameFeaturesValidateAfterPasses(RtxContext& ctx) {
    ValidateState& v = s_validate;
    const uint32_t frameId = ctx.getDevice()->getCurrentFrameId();
    const bool enabled = RtxOptions::FrameFeatures::validate();

    // Check finished read-backs (in order of their slots; one frame each).
    for (ValidateSlot& slot : v.slots) {
      if (!slot.pending) {
        continue;
      }
      DxvkQueryData queryData;
      const DxvkGpuQueryStatus status = slot.ready->getData(queryData);
      if (status == DxvkGpuQueryStatus::Pending) {
        continue;
      }
      if (status == DxvkGpuQueryStatus::Available && enabled) {
        checkSlot(slot, frameId);
      }
      slot.pending = false;
    }

    if (!enabled) {
      v.wasEnabled = false;
      v.frameGbuffer = false;
      v.frameDirect = false;
      return;
    }

    DxvkDevice* device = ctx.getDevice().ptr();
    const Rc<DxvkBuffer> gpu = getLightSamplingStatsBuffer(device);
    constexpr VkDeviceSize kSeenOffset = FRAME_FEATURES_SEEN_SLOT * sizeof(uint32_t);
    constexpr VkDeviceSize kSeenBytes = FRAME_FEATURES_SEEN_SLOTS * sizeof(uint32_t);
    if (!v.wasEnabled) {
      // Start from zero; this frame's bits are discarded.
      ctx.clearBuffer(gpu, kSeenOffset, kSeenBytes, 0u);
      v.wasEnabled = true;
      v.frameGbuffer = false;
      v.frameDirect = false;
      v.summaryStartFrame = frameId;
      return;
    }

    ValidateSlot* free = nullptr;
    for (ValidateSlot& slot : v.slots) {
      if (!slot.pending) {
        free = &slot;
        break;
      }
    }
    if (free != nullptr) {
      if (free->readback == nullptr) {
        DxvkBufferCreateInfo info = {};
        info.size = sizeof(uint32_t);
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
        info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        free->readback = device->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                              DxvkMemoryStats::Category::RTXBuffer, "Frame Features Validate Readback");
      }
      if (free->ready == nullptr) {
        free->ready = device->createGpuQuery(VK_QUERY_TYPE_TIMESTAMP, 0, 0);
      }
      ctx.copyBuffer(free->readback, 0, gpu, kSeenOffset, sizeof(uint32_t));
      ctx.emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
      ctx.writeTimestamp(free->ready);
      free->pending = true;
      free->mask = ctx.getFrameFeatures();
      free->gbuffer = v.frameGbuffer;
      free->direct = v.frameDirect;
    }
    // The next frame starts from zero (a frame without a free read-back slot goes unchecked).
    ctx.clearBuffer(gpu, kSeenOffset, kSeenBytes, 0u);
    v.frameGbuffer = false;
    v.frameDirect = false;

    if (v.summaryStartFrame != UINT32_MAX && frameId - v.summaryStartFrame >= kValidateSummaryFrames) {
      Logger::info(str::format("[FrameFeatures] validate checkedFrames=", v.checkedFrames,
                               " gbufferValidated=", v.gbufferFrames, " directValidated=", v.directFrames,
                               " seen=", seenToString(v.seen), " violations=", v.violations,
                               " (frames ", v.summaryStartFrame, "-", frameId, ")"));
      v.summaryStartFrame = frameId;
      v.checkedFrames = 0;
      v.gbufferFrames = 0;
      v.directFrames = 0;
      v.seen = 0;
      v.violations = 0;
    }
  }

  void showFrameFeaturesImguiSettings() {
    RemixGui::Checkbox("Frame Features: Live Material Counts", &RtxOptions::FrameFeatures::liveMaterialCountsObject());
    RemixGui::SetTooltipToLastWidgetOnHover(
      "rtx.frameFeatures.liveMaterialCounts. The slim shader selectors treat thin-film, ray portal and thin-opaque\n"
      "materials as present only while an instance bound to one is alive, instead of from the first time such a material\n"
      "was ever created. Same image; slimmer levels in sessions where such a material once existed.\n"
      "[FrameFeatures] in the log shows the mask, live counts and sticky flags while a slim option is on.");
    RemixGui::Checkbox("Frame Features: Validate (forces Full kernels)", &RtxOptions::FrameFeatures::validateObject());
    RemixGui::SetTooltipToLastWidgetOnHover(
      "rtx.frameFeatures.validate. Diagnostic: forces the Full kernels and records which features the frame's pixels used;\n"
      "[FrameFeatures] VIOLATION hit=<bit> in the log means the mask missed a feature the frame needed. Slower while on.");
    const RtFrameFeatures& f = s_lastFeatures;
    if (f.frameId == UINT32_MAX) {
      ImGui::TextDisabled("Frame features: not computed yet");
      return;
    }
    ImGui::TextWrapped("Frame features (frame %u, material bits %s): scene=%s mode=%s lights=%u/%u/%u/%u/%u",
                       f.frameId, f.liveMaterialCounts ? "live" : "sticky",
                       frameFeaturesSceneToString(f.scene).c_str(), frameFeaturesModeToString(f.mode).c_str(),
                       f.counts.lights[0], f.counts.lights[1], f.counts.lights[2], f.counts.lights[3], f.counts.lights[4]);
  }

} // namespace dxvk
