#pragma once

// Per-frame feature mask RtFrameFeatures.
//
// One snapshot per frame, taken in RtxContext right after updateRaytraceArgsConstantBuffer: after the TLAS build
// and the light upload, before Volumetrics and the G-buffer. The slim selectors of the G-buffer, direct and indirect
// passes test its bits instead of reading the scene manager or live options at dispatch time, so every pass of a
// frame sees the same state, and the state matches this frame's RaytraceArgs.
//
// Scene bits that come from materials (thin film, ray portal material, thin opaque) use the sticky "ever created"
// flags, or with rtx.frameFeatures.liveMaterialCounts the live instance counts kept by SceneManager's
// retainSurfaceMaterial / releaseSurfaceMaterial.

#include <cstdint>
#include <string>

struct RaytraceArgs;

namespace dxvk {

  class RtxContext;

  struct RtFrameFeatures {
    // Scene content of this frame.
    enum SceneBit : uint32_t {
      kSceneThinFilm             = 1u << 0,  // row 5: an opaque material with the thin-film layer (sticky or live)
      kScenePortalsActive        = 1u << 1,  // row 6: portal textures configured or portals active (args)
      kScenePortalMaterial       = 1u << 2,  // row 7: a ray portal material (sticky or live)
      kSceneBillboardDynamic     = 1u << 3,  // row 9: orientation-corrected billboard with a dynamic mask bit
      kSceneThinOpaque           = 1u << 4,  // row 11: a thin-opaque subsurface material (sticky or live)
      kSceneSssTlas              = 1u << 5,  // row 12: the SSS TLAS was built this frame
      kScenePom                  = 1u << 6,  // row 13: cb.pomMode != Off
      kSceneTranslucentDynamic   = 1u << 7,  // row 15: a translucent view-model / player-model instance
      kSceneUnorderedEmpty       = 1u << 8,  // row 19: the unordered TLAS is empty
      // Material bits. The direct integrator's variant table reads the render-target,
      // legacy, eye and texture-ops bits.
      kSceneTranslucentInstance  = 1u << 16, // live count of instances bound to a translucent material
      kSceneRaytracedRenderTarget = 1u << 17, // live count of instances bound to a raytraced-render-target material
      kSceneLegacySurface        = 1u << 18, // a surface with non-identity legacy state other than texture ops (exact with rtx.coreMaterial.legacyRecord, else always set)
      kSceneEye                  = 1u << 19, // a surface with the eye bit (same)
      kSceneClassTranslucent     = 1u << 20, // a core class 1 surface (rtx.coreMaterial.build)
      kSceneClassSss             = 1u << 21, // a core class 2 surface (rtx.coreMaterial.build)
      kSceneClassFallback        = 1u << 22, // a core class 3 surface (rtx.coreMaterial.build)
      kSceneTextureOps           = 1u << 23, // a surface with non-default texture ops (exact with rtx.coreMaterial.legacyRecord, else always set)
    };

    // Options and modes that select kernels. Row numbers as above.
    enum ModeBit : uint32_t {
      kModeWboit                      = 1u << 0,  // row 1
      kModeNrc                        = 1u << 1,  // row 2
      kModeNrcDeferred                = 1u << 2,  // row 3
      kModeThinFilmOverride           = 1u << 3,  // row 4
      kModeBillboardsPrimary          = 1u << 4,  // row 8
      kModeSssThinOpaqueOpt           = 1u << 5,  // row 10
      kModeSssProfileOpt              = 1u << 6,  // row 10
      kModePomNeeCache                = 1u << 7,  // row 13
      kModeDirectTranslucentShadows   = 1u << 8,  // row 14
      kModeIndirectTranslucentShadows = 1u << 9,  // row 14
      kModeNeeCacheOnly               = 1u << 10, // row 16
      kModeOmm2State                  = 1u << 11, // row 17
      kModeDedicatedSun               = 1u << 12, // row 18
      kModeSamplerBvh                 = 1u << 13, // row 20
      kModeBvhSelectPass              = 1u << 14, // row 20
      kModeAtmosphereNee              = 1u << 15, // row 21
      // 1u << 16 is unused.
      kModeDebugView                  = 1u << 17, // row 23
      // 1u << 18 and 1u << 19 are unused.
    };

    static constexpr uint32_t kLightTypeCount = 5; // sphere, rect, disk, cylinder, distant (light_types.h)

    uint32_t frameId = UINT32_MAX;
    uint32_t scene = 0;
    uint32_t mode = 0;
    uint32_t lightTypes = 0;            // row 25: bit t set when cb.lightRanges[t].count > 0
    uint32_t gbufferFeatureVariant = 0; // gbuffer_rayquery features axis index this frame's G-buffer binds
    bool liveMaterialCounts = false;    // scene bits 5, 7, 11 came from live counts (rtx.frameFeatures.liveMaterialCounts)

    // For the log only; not part of the change test.
    struct Counts {
      uint32_t lights[kLightTypeCount] = {};
      uint32_t thinOpaque = 0;
      uint32_t thinFilm = 0;
      uint32_t rayPortal = 0;
      uint32_t sss = 0;
      uint32_t pom = 0;
      uint32_t translucent = 0;
      uint32_t raytracedRenderTarget = 0;
      bool stickyThinOpaque = false;
      bool stickyThinFilm = false;
      bool stickyRayPortal = false;
      // Surfaces per core class (opaque, translucent, skin SSS, fallback) while the core table is current.
      bool coreClassesValid = false;
      uint32_t coreClasses[4] = {};
    } counts;

    bool has(SceneBit bit) const { return (scene & bit) != 0; }
    bool hasMode(ModeBit bit) const { return (mode & bit) != 0; }
  };

  // Builds this frame's mask from `args` (this frame's RaytraceArgs) and the scene, accel and instance managers.
  RtFrameFeatures computeFrameFeatures(RtxContext& ctx, const RaytraceArgs& args);

  // [FrameFeatures] line: when the scene bits, mode bits, light-type presence or sticky flags change, at most once per
  // 60 frames (the settled state is still logged), while any slim-variant option, liveMaterialCounts or the material
  // census is on.
  void logFrameFeatures(const RtFrameFeatures& features);

  std::string frameFeaturesSceneToString(uint32_t scene);
  std::string frameFeaturesModeToString(uint32_t mode);

  // The last computed mask (render thread), for the dev menu.
  const RtFrameFeatures& lastFrameFeatures();

  // Dev menu: rtx.frameFeatures.liveMaterialCounts checkbox and a read-only line with the current mask.
  void showFrameFeaturesImguiSettings();

  // rtx.frameFeatures.validate. The G-buffer / direct passes report that they bound a validation variant
  // this frame; frameFeaturesValidateAfterPasses (after dispatchPathTracing) copies the frame's seen bits out of the
  // LightSamplingStats buffer, clears them, and checks finished copies against the mask of their frame.
  enum class FrameFeaturesValidateKernel { GBuffer, Direct };
  void noteFrameFeaturesValidateKernel(FrameFeaturesValidateKernel kernel);
  void frameFeaturesValidateAfterPasses(RtxContext& ctx);

} // namespace dxvk
