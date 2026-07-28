/*
* Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "rtx/utility/shader_types.h"
#ifdef __cplusplus
#include "rtx/concept/camera/camera.h"
#include "rtx/concept/ray_portal/ray_portal.h"
#else
#include "rtx/concept/camera/camera.slangh"
#include "rtx/concept/ray_portal/ray_portal.slangh"
#endif

#include "rtx/pass/nrd_args.h"
#include "rtx/pass/nrc_args.h"
#include "rtx/pass/volume_args.h"
#include "rtx/pass/material_args.h"
#include "rtx/pass/view_distance_args.h"
#include "rtx/pass/atmosphere/atmosphere_args.h"
#include "rtx/concept/light/light_types.h"
#include "rtx/concept/surface/surface_shared.h"
#include "rtx/algorithm/nee_cache_data.h"
#include "rtx/pass/sparse_rendering/sparse_rendering.h"

struct LightRangeInfo {
  uint offset;
  uint count;
  uint16_t rtxdiSampleCount;
  uint16_t volumeRISSampleCount;
  uint16_t risSampleCount;
  uint16_t pad;
};

// Note: ensure 16B alignment
struct TerrainArgs {
  uint2 cascadeMapSize;    // Number of cascade tiles in each dimension
  float2 rcpCascadeMapSize;

  uint maxCascadeLevel;
  float lastCascadeScale;
  float displaceIn;
  uint pad0;
};

struct NeeCacheArgs {
  uint enable;
  uint enableImportanceSampling;
  uint enableMIS;
  uint enableOnFirstBounce;

  uint enableAnalyticalLight;
  float specularFactor;
  float uniformSamplingProbability;
  float cullingThreshold;

  NeeEnableMode enableModeAfterFirstBounce;
  float ageCullingSpeed;
  float emissiveTextureSampleFootprintScale;
  uint approximateParticleLighting;

  float resolution;
  float minRange;
  float learningRate;
  uint clearCache;

  float triangleExplorationRangeRatio;
  uint  triangleExplorationMaxRange;
  float triangleExplorationProbability;
  float triangleExplorationAcceptRangeRatio;

  uint padding;
  uint enableReshuffleResilience;
  uint reshuffleMaxAge;
  uint enableSpatialReuse;
};

struct DomeLightArgs {
  mat4 worldToLightTransform;

  vec3 radiance;
  uint active;

  uint3 pad0;
  uint textureIndex;
};

struct SssArgs {
  uint enableThinOpaque;
  uint enableDiffusionProfile;
  float diffusionProfileScale;
  u16vec2 diffusionProfileDebuggingPixel;
};

struct EyeArgs {
  uint  enableEyes;
  float normalBendingEyeball;
  float normalBendingCornea;
  float whitesAlbedoScale;

  float irisRadius;
  float irisDepth;
  uint  pad0;
  uint  pad1;
};

struct ShadowTerminatorArgs
{
  uint  enableOffset;
  uint  soften;
  float maxArea;
  float maxLength;
};

#define OBJECT_PICKING_INVALID (cb.clearColorPicking)

// Constant buffer
struct RaytraceArgs {
  // NOTE: this class should be kept as all structs, then all non-structs.  This is because the padding rules are different between C++ and shaders.
  Camera camera;

  // Note: Primary combined variant used in place of the primary direct denoiser when seperated direct/indirect
  // lighting is not used.
  NrdArgs primaryDirectNrd;
  NrdArgs primaryIndirectNrd;
  NrdArgs secondaryCombinedNrd;

  // Note: Not tightly packed, meaning these indices will align with the Ray Portal Index in the
  // Surface Material. Do note however due to elements being potentially "empty" each Ray Portal Hit Info
  // must be checked to be empty or not before usage. Additionally both Ray Portals in a pair will match
  // in state, either being present or not.
  // The first `maxRayPortalCount` portals are for this frame, the second `maxRayPortalCount` are for the previous frame.
  RayPortalHitInfo rayPortalHitInfos[maxRayPortalCount * 2];

  VolumeArgs volumeArgs;
  OpaqueMaterialArgs opaqueMaterialArgs;
  TranslucentMaterialArgs translucentMaterialArgs;
  ViewDistanceArgs viewDistanceArgs;

  LightRangeInfo lightRanges[lightTypeCount];

  TerrainArgs terrainArgs;
  NeeCacheArgs neeCacheArgs;
  DomeLightArgs domeLightArgs;
  NrcArgs nrcArgs;
  SssArgs sssArgs;
  EyeArgs eyeArgs;
  ShadowTerminatorArgs shadowTerminatorArgs;
  AtmosphereArgs atmosphereArgs;

  Camera renderTargetCamera;

  SparseRenderingArgs sparseRenderingArgs;

  // ------------------------- Structs above this line, non structs below this line -----------------------------------

  uint frameIdx;
  float ambientIntensity;
  uint16_t lightCount;
  uint16_t risTotalSampleCount;
  uint16_t volumeRISTotalSampleCount;
  uint16_t rtxdiTotalSampleCount;

  // The maximum probability of continuing a path when Russian Roulette is being used.
  RussianRouletteMode russianRouletteMode;
  float russianRouletteDistanceFactor;
  float russianRouletteDiffuseContinueProbability;
  float russianRouletteSpecularContinueProbability;

  float russianRouletteMaxContinueProbability;
  float russianRoulette1stBounceMinContinueProbability;
  float russianRoulette1stBounceMaxContinueProbability;
  float fireflyFilteringLuminanceThreshold;

  // The minimum number of indirect bounces the path must complete before Russian Roulette can be used. Must be < 16.
  uint8_t pathMinBounces;
  // The maximum number of indirect bounces the path will be allowed to complete. Must be < 16.
  uint8_t pathMaxBounces;
  // The number of samples to clamp temporal reservoirs to. Note this is not the same as RTXDI's history length as it is not scaled
  // by the number of samples the current reservoir performs (due to variability in how many actual current reservoir samples are done).
  uint16_t volumeTemporalReuseMaxSampleCount;
  // The maximum number of resolve interactions for primary (geometry resolver) rays.
  uint8_t primaryRayMaxInteractions;
  // The maximum number of resolve interactions for PSR (geometry resolver) rays.
  uint8_t psrRayMaxInteractions;
  // The maximum number of resolve interactions for secondary (integrator) rays.
  uint8_t secondaryRayMaxInteractions;
  // The number of active Ray Portals (Used for Ray Portal sampling). Always <= RAY_PORTAL_MAX_COUNT
  uint8_t numActiveRayPortals;
  float secondarySpecularFireflyFilteringThreshold;
  uint  outputParticleLayer;

  // Note: Packed as float16, uses uint16_t due to being shared on C++ side
  uint16_t emissiveBlendOverrideEmissiveIntensity;
  // The maximum number of bounces to evaluate reflection PSR over.
  uint8_t psrrMaxBounces;
  // The maximum number of bounces to evaluate transmission PSR over.
  uint8_t pstrMaxBounces;
  float viewModelRayTMax;
  uint16_t particleSoftnessFactor;
  uint16_t emissiveIntensity;
  uint8_t rtxdiSpatialSamples;
  uint8_t rtxdiDisocclusionSamples;
  uint8_t rtxdiMaxHistoryLength;
  uint8_t virtualInstancePortalIndex; // portal space for which virtual view model or player model instances were generated for

  float indirectRaySpreadAngleFactor;
  // Half the angle of the cone spawned by each pixel to use for ray cone texture filtering.
  float screenSpacePixelSpreadHalfAngle;
  uint debugView;
  float primaryDirectMissLinearViewZ;


  vec4 debugKnob;     // For temporary tuning in shaders, has a dedicated UI widget.

  // Values to use on a ray miss
  vec3 clearColorNormal;
  float clearColorDepth;

  float2 upscaleFactor;   // Displayed(upscaled) / RT resolution
  uint32_t clearColorPicking;

  uint enableDLSSRR;
  uint setLogValueForDisocclusionMaskForDLSSRR;

  // NOTE: Variables need to be in groups of 4x32 bits above this comment.

  uint uniformRandomNumber;
  uint16_t opaqueDiffuseLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueDiffuseLobeSamplingProbability;
  uint16_t opaqueSpecularLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueSpecularLobeSamplingProbability;
  uint16_t opaqueOpacityTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueOpacityTransmissionLobeSamplingProbability;
  uint16_t opaqueDiffuseTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minOpaqueDiffuseTransmissionLobeSamplingProbability;

  uint16_t translucentSpecularLobeSamplingProbabilityZeroThreshold;
  uint16_t minTranslucentSpecularLobeSamplingProbability;
  uint16_t translucentTransmissionLobeSamplingProbabilityZeroThreshold;
  uint16_t minTranslucentTransmissionLobeSamplingProbability;
  float roughnessDemodulationOffset;
  float timeSinceStartSeconds;

  uint enableCalculateVirtualShadingNormals;
  uint enableDirectLighting;
  uint enableEmissiveBlendEmissiveOverride;
  uint enablePortalFadeInEffect;
  uint enableRussianRoulette;
  uint enableSecondaryBounces;
  uint enableSeparateUnorderedApproximations;
  uint enableStochasticAlphaBlend;
  uint16_t enableDirectTranslucentShadows;
  uint16_t enableDirectAlphaBlendShadows;
  uint16_t enableIndirectTranslucentShadows;
  uint16_t enableIndirectAlphaBlendShadows;
  uint enableFirstBounceLobeProbabilityDithering;
  uint enableUnorderedResolveInIndirectRays;
  uint enableProbabilisticUnorderedResolveInIndirectRays;
  uint enableUnorderedEmissiveParticlesInIndirectRays;
  uint enableTransmissionApproximationInIndirectRays;
  uint enableDecalMaterialBlending;
  uint enableLegacyRectLightConeShaping;
  uint enableRectLightConeShapingRatioScaling;
  uint enableBillboardOrientationCorrection;
  uint enablePlayerModelInPrimarySpace;
  uint enablePlayerModelPrimaryShadows;
  uint enablePreviousTLAS;
  uint useIntersectionBillboardsOnPrimaryRays;

  uint enableRtxdi;
  uint enableRtxdiPermutationSampling;
  uint enableRtxdiRayTracedBiasCorrection;
  uint enableRtxdiSampleStealing;
  uint enableRtxdiStealBoundaryPixelSamplesWhenOutsideOfScreen;
  uint enableRtxdiCrossPortalLight;
  uint enableRtxdiTemporalBiasCorrection;
  uint enableRtxdiInitialVisibility;
  uint enableRtxdiTemporalReuse;
  uint enableRtxdiSpatialReuse;
  uint enableRtxdiDiscardInvisibleSamples;
  uint enableRtxdiDiscardEnlargedPixels;
  uint enableDirectLightBoilingFilter;
  uint enableRtxdiBestLightSampling;
  float directLightBoilingThreshold;
  float rtxdiDisocclusionFrames;

  uint enableDemodulateRoughness;
  uint enableHitTFiltering;
  uint enableReplaceDirectSpecularHitTWithIndirectSpecularHitT;
  uint enableSeparatedDenoisers;

  uint enableViewModelVirtualInstances;

  uint enablePSRR;
  uint enablePSTR;
  uint enablePSTROutgoingSplitApproximation;
  uint enablePSTRSecondaryIncidentSplitApproximation;
  float psrrNormalDetailThreshold;
  float pstrNormalDetailThreshold;

  uint enableEnhanceBSDFDetail;
  uint enhanceBSDFIndirectMode;
  float enhanceBSDFDirectLightPower;
  float enhanceBSDFIndirectLightPower;
  float enhanceBSDFDirectLightMaxValue;
  float enhanceBSDFIndirectLightMaxValue;
  float enhanceBSDFIndirectLightMinRoughness;

  uint startInMediumMaterialIndex;
  uint enableReSTIRGI;
  uint enableReSTIRGIFinalVisibility;
  uint enableReSTIRGIReflectionReprojection;
  float restirGIReflectionMinParallax;
  uint enableReSTIRGIVirtualSample;
  float reSTIRGIVirtualSampleLuminanceThreshold;
  float reSTIRGIVirtualSampleRoughnessThreshold;
  float reSTIRGIVirtualSampleSpecularThreshold;
  float reSTIRGIVirtualSampleMaxDistanceRatio;
  uint reSTIRGIMISMode;
  float reSTIRGIMISModePairwiseMISCentralWeight;
  uint enableReSTIRGIPermutationSampling;
  uint enableReSTIRGIDLSSRRCompatibilityMode;
  float reSTIRGIDLSSRRTemporalRandomizationRadius;
  uint enableReSTIRGISampleStealing;
  float reSTIRGISampleStealingJitter;
  uint enableReSTIRGIStealBoundaryPixelSamplesWhenOutsideOfScreen;
  uint enableReSTIRGISpatialReuse;
  uint enableReSTIRGITemporalReuse;
  uint reSTIRGIBiasCorrectionMode;
  uint enableReSTIRGIBoilingFilter;
  float boilingFilterLowerThreshold;
  float boilingFilterHigherThreshold;
  float boilingFilterRemoveReservoirThreshold;
  uint temporalHistoryLength;
  uint permutationSamplingSize;
  uint enableReSTIRGITemporalBiasCorrection;
  uint enableReSTIRGIDiscardEnlargedPixels;
  float reSTIRGIHistoryDiscardStrength;
  uint enableReSTIRGITemporalJacobian;
  float reSTIRGIFireflyThreshold;
  float reSTIRGIRoughnessClamp;
  float reSTIRGIMISRoughness;
  float reSTIRGIMISParallaxAmount;
  uint enableReSTIRGIDemodulatedTargetFunction;
  uint enableReSTIRGILightingValidation;
  uint enableReSTIRGIVisibilityValidation;
  float reSTIRGISampleValidationThreshold;
  float reSTIRGIVisibilityValidationRange;

  uint surfaceCount;
  uint teleportationPortalIndex; // 0 means no teleportation, 1+ means portal 0+

  float resolveTransparencyThreshold;
  float resolveOpaquenessThreshold;
  float resolveStochasticAlphaBlendThreshold;
  float translucentDecalAlbedoFactor;

  uint enableHeuristicSingleScatteringTransmission;

  float skyBrightness;
  uint skyMode;  // 0 = skybox rasterization, 1 = physical atmosphere

  uint isLastCompositeOutputValid;
  uint isZUp; // Note: Indicates if the Z axis is the "up" axis in world space if true, otherwise the Y axis if false.
  uint enableCullingSecondaryRays;

  u16vec2 gpuPrintThreadIndex;
  uint gpuPrintElementIndex;
  uint enableObjectPicking;

  DisplacementMode pomMode;
  uint pomEnableDirectLighting;
  uint pomEnableIndirectLighting;
  uint pomEnableNEECache;
  uint pomEnableReSTIRGI;
  uint pomEnablePSR;
  uint pomMaxIterations;
  uint enableSssTransmission;
  uint enableSssTransmissionSingleScattering;
  uint sssTransmissionBsdfSampleCount;
  uint sssTransmissionSingleScatteringSampleCount;
  uint enableTransmissionDiffusionProfileCorrection;
  float totalMipBias;

  uint forceFirstHitInGBufferPass;

  uint enableRaytracedRenderTarget;
  // NRC enablement is controlled by global macros being defined.
  // When macros are not used (i.e. in some passes) this variable controls the NRC enablement.
  uint enableNrc;

  // Debug override to disallow NRC training when it is enabled in the first place,
  // hence why it is not named enableNrcTraining here
  uint allowNrcTraining;

  float vertexColorStrength;
  float alphaBlendSurfacePackMult; // for packing/unpacking hitT into Float16 in AlphaBlendSurface

  float wboitEnergyLossCompensation;
  float wboitDepthWeightTuning;
  uint wboitEnabled;

  // Fork (2026-07-26): scale on the sky-ambient term that the resolver's
  // opacity lighting approximation adds for sky-lit particle materials
  // (weather precipitation). 0 disables the term entirely. Appended at the
  // END of the struct so no existing field offsets move.
  float particleSkyAmbientScale;

  // Fork (2026-07-27): ReSTIR PT (Lin et al. 2022) trace kernel parameters.
  // Appended at the END of the struct as ONE COMPLETE 4-SCALAR (16-byte) GROUP
  // so no existing field offsets move and the struct's 16-byte alignment is
  // preserved -- this fork has GPU-hung twice on RaytraceArgs misalignment, so
  // this group must stay exactly four scalars wide. Consumed by
  // src/dxvk/shaders/rtx/pass/fork_restir_pt/ and
  // src/dxvk/shaders/rtx/algorithm/fork_restir_pt/.
  uint restirPtMaxBounces;                   // Path length cap; 0 disables tracing past the primary scatter.
  uint restirPtFlags;                        // RESTIR_PT_FLAG_* bitfield below.
  float restirPtSpecularRoughnessThreshold;  // Perceptual roughness at or below which a non-diffuse lobe counts as a specular bounce.
  float restirPtDeltaRoughnessThreshold;     // Perceptual roughness below which opaque specular is classified as a delta event (reconnection forbidden).

  // Fork (2026-07-28): ReSTIR PT phase 3, spatial reuse. ONE MORE COMPLETE
  // 4-SCALAR (16-byte) GROUP, appended for the same reason and with the same
  // constraint as the group above -- do not add a fifth scalar here, add the next
  // complete group.
  uint restirPtSpatialNeighborCount;         // Neighbours considered per spatial round.
  uint restirPtSpatialRounds;                // Spatial reuse rounds per frame (each is one dispatch).
  float restirPtSpatialRadius;               // Neighbour gather radius, in pixels.
  float restirPtJacobianRejectionThreshold;  // Discard a shift when max(J, 1/J) > 1 + this. <= 0 disables (reference default).

  // Fork (2026-07-28): ReSTIR PT phase 4, temporal reuse. ONE MORE COMPLETE
  // 4-SCALAR (16-byte) GROUP, same constraint again. The two spare scalars are
  // deliberately left here rather than trimmed: phase 5's retrace passes want
  // scalars in this block, and reserving them now costs nothing while adding a
  // fifth scalar later would cost a re-audit of the whole struct's alignment.
  float restirPtTemporalHistoryLength;       // M-cap: history M is clamped to this x the current reservoir's M.
  float restirPtLightingValidationThreshold; // RTXDI gradient magnitude above which a temporal sample is discarded as stale.
  // The two scalars this group reserved, spent. Outlier suppression: ReSTIR PT had
  // none at all, while ReSTIR GI beside it ships three separate mechanisms for it
  // (rtx.restirGI.useBoilingFilter, fireflyThreshold, and a firefly clamp on the
  // initial sample). A resampling loop RETAINS and SPREADS a firefly rather than
  // averaging it away, so this is not optional polish.
  float restirPtBoilingFilterThreshold;      // Clear a reservoir whose radiance exceeds this x the workgroup average.
  float restirPtFireflyThreshold;            // Absolute luminance clamp on a reservoir's radiance. <= 0 disables.

  // NOTE: Add structs to the top section of RaytraceArgs, not the bottom.
  // NOTE: bool does not work in debug builds, use uint instead.
};

// Fork (2026-07-27): bit layout of RaytraceArgs::restirPtFlags. Frame-constant
// bits only -- the trace/replay mode selector is per *dispatch* (the pass runs
// twice inside one frame, and this buffer is uploaded once), so it lives in the
// pass's push constants instead; see ForkReSTIRPTArgs in
// rtx/pass/fork_restir_pt/fork_restir_pt_binding_indices.h.
// Bit 0 is reserved so a later phase can reclaim it without renumbering.
//
// Phase 2 adds bits only -- NO new scalars. Everything it needed (a mode-active
// flag, a task-feedback toggle and a three-way emissive policy) fits in the
// spare bits of this word, which is why the 4-scalar group above is untouched.
#define RESTIR_PT_FLAG_RESERVED0                (1u << 0)
#define RESTIR_PT_FLAG_ENABLE_RUSSIAN_ROULETTE  (1u << 1)
// Set when IntegrateIndirectMode::ReSTIRPT is the live indirect mode, i.e. when
// the fork trace kernel is running IN PLACE OF integrate_indirect rather than as
// the debug-only harness alongside it.
#define RESTIR_PT_FLAG_MODE_ACTIVE              (1u << 2)
// Emissive-hit task insertions into the NEE cache. Keeps the cache DISCOVERING
// emissive triangles once integrate_indirect stops running; see the task
// feedback block in restir_pt_trace_core.slangh.
#define RESTIR_PT_FLAG_NEE_CACHE_TASK_FEEDBACK  (1u << 3)

// Emissive accounting policy, 2 bits at 4-5. The A/B knob for the one phase-2
// change that moves energy (the removal of the reference's `suppressAsDirect`).
// Modes 0 and 1 bracket the default from below and above -- see the EMISSIVE
// ACCOUNTING POLICY block in restir_pt_trace_core.slangh.
#define RESTIR_PT_EMISSIVE_MIS_MODE_SHIFT       4u
#define RESTIR_PT_EMISSIVE_MIS_MODE_MASK        0x3u
#define RESTIR_PT_EMISSIVE_MIS_SUPPRESS         0u  // Reference behaviour: drop length-1 emissive as "direct".
#define RESTIR_PT_EMISSIVE_MIS_NONE             1u  // Weight 1 everywhere: double-counts against integrate_nee.
#define RESTIR_PT_EMISSIVE_MIS_NEE_CACHE        2u  // Default: BSDF-side MIS against the NEE cache at length 1.

// Phase 3. Bit 6 gates the spatial reuse pass; the trace kernel is unaffected by
// it (it always records reconnection data, which is what keeps reuse-off equal to
// phase 2). Bit 7 gates the sky-reconnection deviation -- promoting a bounce-1 sky
// escape to a reconnection vertex, which the reference only does under the hybrid
// shift; off reproduces strict reference behaviour outdoors.
#define RESTIR_PT_FLAG_SPATIAL_REUSE            (1u << 6)
#define RESTIR_PT_FLAG_SPATIAL_SKY_RECONNECTION (1u << 7)

// Phase 4. Bit 8 gates the temporal reuse pass. Bit 9 jitters the reprojected
// pixel within its footprint (the reference's sampleNext2D at
// TemporalReuse.cs.slang:126), which decorrelates a slow camera pan from its own
// history. Bit 10 gates gradient-based history invalidation; it is set only when
// the RTXDI gradient pass is actually producing gradients this frame, so the
// shader never reads a stale or unwritten gradient texture.
#define RESTIR_PT_FLAG_TEMPORAL_REUSE           (1u << 8)
#define RESTIR_PT_FLAG_TEMPORAL_JITTER          (1u << 9)
#define RESTIR_PT_FLAG_LIGHTING_VALIDATION      (1u << 10)

// Bit 11 gates the boiling filter in final shading. Deliberately a separate bit
// from the firefly clamp (which is keyed off its threshold being positive): the
// two suppress different things and want to be A/B'd independently -- the clamp is
// per-pixel and absolute, the filter is neighbourhood-relative.
#define RESTIR_PT_FLAG_BOILING_FILTER           (1u << 11)
