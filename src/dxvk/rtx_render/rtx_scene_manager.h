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

#include <atomic>
#include <mutex>
#include <optional>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <variant>

#include "../dxvk_buffer.h"
#include "../dxvk_image.h"
#include "../dxvk_staging.h"
#include "../dxvk_bind_mask.h"
#include "../util/util_hashtable.h"

#include "rtx_globals.h"
#include "rtx_retained_buffer_table.h"
#include "rtx_types.h"
#include "rtx_fork_material_derivation.h" // ExternalDrawState alpha / legacy descriptions
#include "rtx_common_object.h"
#include "rtx_camera_manager.h"
#include "rtx_draw_call_cache.h"
#include "rtx_draw_call_tracker.h"
#include "rtx_sparse_unique_cache.h"
#include "rtx_light_manager.h"
#include "rtx_instance_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_ray_portal_manager.h"
#include "rtx_lens.h"
#include "rtx_bindless_resource_manager.h"
#include "rtx_objectpicking.h"
#include "rtx_mod_manager.h"
#include "graph/rtx_graph_manager.h"
#include "rtx_particle_system.h"
#include <d3d9types.h>

namespace dxvk 
{
class DxvkContext;
class DxvkDevice;
struct AssetReplacement;
struct AssetReplacer;
class OpacityMicromapManager;
class CoreMaterialBuilder;
class TerrainBaker;
class RetainedInstanceStore;
class RetainedScene;

// The resource cache can be *searched* by other users
class ResourceCache {
public:
  bool find(const RtSurfaceMaterial& surf, uint32_t& outIdx) const { return m_surfaceMaterialCache.find(surf, outIdx); }
  const RtSurfaceMaterial& get(const uint32_t index) const { return m_surfaceMaterialCache.getObjectTable()[index]; }

protected:
  RetainedBufferTable<RaytraceBuffer> m_bufferCache;

  struct SurfaceMaterialHashFn {
    size_t operator() (const RtSurfaceMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialCache;
  SparseUniqueCache<RtSurfaceMaterial, SurfaceMaterialHashFn> m_surfaceMaterialExtensionCache;
  fast_unordered_cache<uint32_t> m_preCreationSurfaceMaterialMap;

  struct VolumeMaterialHashFn {
    size_t operator() (const RtVolumeMaterial& mat) const {
      return (size_t)mat.getHash();
    }
  };
  SparseUniqueCache<RtVolumeMaterial, VolumeMaterialHashFn> m_volumeMaterialCache;

  struct SamplerHashFn {
    size_t operator() (const Rc<DxvkSampler>& sampler) const {
      return (size_t) sampler->hash();
    }
  };

  struct SamplerKeyEqual {
    bool operator()(const Rc<DxvkSampler>& lhs, const Rc<DxvkSampler>& rhs) const {
      return lhs->info() == rhs->info();
    }
  };

  SparseUniqueCache<Rc<DxvkSampler>, SamplerHashFn, SamplerKeyEqual> m_samplerCache;
};

struct ExternalDrawState {
  DrawCallState drawCall {};
  remixapi_MeshHandle mesh {};
  CameraType::Enum cameraType {};
  CategoryFlags categories {};
  bool doubleSided {};
  std::optional<RtxParticleSystemDesc> optionalParticleDesc {};
  std::vector<Matrix4> gpuInstancingTransforms {};
  // remixapi_InstanceInfoMaterialOverrideEXT. Non-null: every submesh is shaded with
  // this material instead of its own (falls back to the submesh's if it is not live).
  remixapi_MaterialHandle materialOverride {};

  // Scene interface for frontends. inlineSubmeshes: the mesh travels with the
  // description instead of being registered as an API mesh under `mesh` (which then only keys geometry sharing).
  // resolvedMaterial: the render material, already through the frontend's material path. frontendResolved: drawCall is
  // complete as the frontend made it (samplers, cull mode, material hash, categories); the API preparation steps that
  // would change it are skipped, and the entry never goes to the legacy lane (it is rejected instead).
  std::shared_ptr<const std::vector<RasterGeometry>> inlineSubmeshes {};
  std::shared_ptr<const MaterialData> resolvedMaterial {};
  bool frontendResolved = false;
  // A frontend MaterialTranslator's alpha state and legacy surface
  // state for this draw. Only with frontendResolved; when set, InstanceManager uses them instead of deriving them from
  // drawCall (same functions, so the values match: [MaterialTranslator] parity).
  std::optional<AlphaStateDesc> alphaState {};
  std::optional<LegacySurfaceDesc> legacySurface {};
  // An expanded USD mesh replacement, one part per inline submesh (same
  // order and count). Only with frontendResolved and inlineSubmeshes. Each submesh is drawn at objectToWorld *
  // submeshToObject, with its own render material and categories; meshReplacement parts reset the texture transform
  // and texgen as SceneManager::buildReplacementMeshDrawCallState does. alphaState / legacySurface are not used then.
  struct SubmeshPart {
    Matrix4 submeshToObject;
    std::shared_ptr<const MaterialData> material;
    CategoryFlags categories {};
    bool meshReplacement = true;
    // The replacement prim's own particle system, which the scene's particle service spawns from this submesh.
    std::optional<RtxParticleSystemDesc> particleSystem;
  };
  std::shared_ptr<const std::vector<SubmeshPart>> inlineSubmeshParts {};
  // The geometry changes from description to description (CPU skinning, rewritten or dynamic buffers).
  // RetainedScene keys it by the entry (never shared), rebuilds it in place when its topology changes,
  // and gives it its own BLAS at once.
  bool deforming = false;
  // The decal draw order of this object's decal instances (8 bits on the GPU). Unset: the direct lane's
  // creation order. RetainedScene::setSortOrder changes it without a rebuild.
  std::optional<uint8_t> sortOrder;

  // Retained draws (non-zero handle) are identified by their handle rather
  // than by their transforms, so a moved placement or a changed instancing
  // list updates the existing instance instead of minting a new one while
  // the old one waits for garbage collection. forceFullUpdate makes the
  // draw take the full path once after its description was replaced.
  uint64_t retainedHandle = 0;
  bool forceFullUpdate = false;

  // Draw-instance identity for ReplacementInstance lookup. Excludes per-frame camera matrices
  // (worldToView / viewToProjection / objectToView) filled in submitExternalDraw after hashing.
  XXH64_hash_t computeExternalDrawIdentityHash() const;
};

// Scene manager is a super manager, it's the interface between rendering and world state
// along with managing the operation of other caches, scene manager also manages the cache
// directly for "SceneObject"'s - which are "unique meshes/geometry", which map 1-to-1 with
// BLAS entries in raytracing terminology.
class SceneManager : public CommonDeviceObject, public ResourceCache {
  friend class ImGUI;
  // rtx.retained.useV2: the second retained-instance implementation drives the same preserve and
  // external-draw machinery the first one uses.
  friend class RetainedInstanceStore;
  // rtx.scene.directRetained: the scene's own retained instances build and keep RT instances themselves.
  friend class RetainedScene;
  friend class TransientLane;
  friend class CoreMaterialBuilder;
  friend class OpacityMicromapManager; // Software micromap stores in m_bufferCache
  RTX_OPTION("rtx.geometry", bool, optimizeAnimatedTexcoords, false,
    "Updates UV-only animation of compatible fixed-function geometry without a BLAS refit, which saves GPU time. Shader "
    "capture, skinning, smooth normals and active opacity micromaps keep the refit. Turn it off to always refit "
    "animated geometry.");

public:
  SceneManager(SceneManager const&) = delete;
  SceneManager& operator=(SceneManager const&) = delete;

  explicit SceneManager(DxvkDevice* device);
  ~SceneManager();

  void initialize(Rc<DxvkContext> ctx);
  void logStatistics();

  void onDestroy();

  void submitDrawState(Rc<DxvkContext> ctx, const DrawCallState& input, const MaterialData* overrideMaterialData);
  void submitExternalDraw(const Rc<DxvkContext>& ctx, std::unique_ptr<ExternalDrawState> state);

  // D3D9 frontend: the parts of submitDrawState a frontend needs for draws it describes to
  // the scene instead of submitting. Fog discovery for one draw (what submitDrawState does first); whether the draw's
  // asset hashes (active rule, then the legacy rules) have USD mesh replacements, without tracking them; its render
  // material (material replacement, ray portal or legacy conversion); scene clears so far.
  void discoverFogState(const DrawCallState& input);
  // rtx.d3d9v2.frameEvents: fog discovery from the frontend's frame order. input: a draw of that fog for a
  // fog replacement material (null: a replaced fog is left undiscovered for this frame). While the frontend discovers
  // fog, submitDrawState does not.
  void discoverFog(const FogState& fog, const DrawCallState* input);
  void setFrontendFogDiscovery(bool on) { m_frontendFogDiscovery = on; }
  bool hasMeshReplacements(const DrawCallState& input) const;
  MaterialData resolveDrawMaterial(const DrawCallState& input) { return determineMaterialData(nullptr, input); }
  uint64_t clearEpoch() const { return m_clearEpoch; }
  // The USD mesh replacements submitDrawState would draw for this draw (same
  // hashes, same order: the active asset hash, then the legacy rules), null if none; *matchedRule is 0 (active hash),
  // 1 or 2 (legacy hash 0 / 1). Nothing is tracked.
  std::shared_ptr<const ReplacementBucket> findMeshReplacements(const DrawCallState& input, uint32_t* matchedRule) const;
  // While set, submitDrawState marks the replacement instance it matches as handed to the scene
  // (ReplacementInstance::handedToScene); every other submission clears that mark.
  void setSubmittingHandedOver(bool handedOver) { m_submittingHandedOver = handedOver; }

  // Host-retained draws (after doodlum/dxvk-remix 5e39a218a). A host that
  // describes its scene through the API would otherwise re-send every
  // instance every frame -- thousands of DrawInstance calls, each taking the
  // API lock, converting a draw state and queueing a command-stream lambda
  // on the host's thread. It registers a draw once instead, reports only
  // genuine changes, and asks for one replay per frame. The replay goes
  // through submitExternalDraw, so preserve-path identity and the unseen-
  // instance garbage collection behave exactly as for per-frame draws.
  void setRetainedExternalDraw(uint64_t handle, ExternalDrawState&& state);
  void setRetainedExternalDrawTransform(uint64_t handle, const Matrix4& objectToWorld);
  // Hidden draws stay registered, so a host's own culling can drop them for a
  // frame without re-registering on return. What a hidden draw does in the
  // meantime is rtx.retainedHiddenPolicy (stay ray traced, stay resident but
  // leave the TLAS, or stop replaying and age out).
  void setRetainedExternalDrawVisible(uint64_t handle, bool visible);
  void removeRetainedExternalDraw(uint64_t handle);
  // [PerfCpu] retained replay split (optional; null = not collected).
  struct RetainedReplayStats {
    double fullMs = 0.0;
    // pinned (rtx.retainedPinSteadyEntries) entries are also counted in preserved and steady.
    uint32_t full = 0, preserved = 0, skipped = 0, steady = 0, pinned = 0;
  };
  void submitRetainedExternalDraws(const Rc<DxvkContext>& ctx, RetainedReplayStats* stats = nullptr);
  // rtx.retained.diagnosticFrames: called after a retained entry's full replay (either implementation)
  // with what that replay recorded; counts what went wrong with it. logRetainedDiagnostics writes and
  // resets the counts (lateReplays: replays that arrived after the frame's RTX injection).
  void noteRetainedFullReplay(uint64_t handle, remixapi_MeshHandle mesh, XXH64_hash_t identity,
                              bool redescribed, bool visible);
  void logRetainedDiagnostics(uint32_t replays, uint32_t lateReplays, uint32_t deferredReplays);
  size_t retainedExternalDrawCount() const { return m_retainedExternalDraws.size(); }
  // Registered transforms are absolute; the replay subtracts this, so the
  // host keeps stable coordinates while the scene stays near the origin.
  // The host must rebase its camera and lights by the same amount.
  void setRetainedSceneOrigin(const Vector3& origin) { m_retainedSceneOrigin = origin; }
  // rtx.worldOrigin.enable: the host's render origin for the frame whose Main camera is being set up.
  // When it moved, every instance's current and previous transforms are re-expressed around the new origin
  // (render position + origin = world position), so this frame's draws move them from there and motion
  // vectors see no jump. Called before the Main camera update, ahead of the frame's draws.
  void applyWorldOrigin(const Vector3& origin);
  // rtx.worldOrigin.enable, D3D9 camera path: a host whose camera arrives as D3D9 transforms publishes
  // __mcrtx.world_origin.* with remixapi_SetGameValue, which records it here through the command stream, in
  // order with the host's draws. applyPendingHostWorldOrigin runs ahead of each draw's camera processing and,
  // before the frame's Main camera update, does what SetupCamera does for an API camera: applyWorldOrigin plus
  // the Main camera's history shift. Held over to the next frame when the Main camera was already updated;
  // ignored for a host that sends its origin through SetupCamera.
  void setPendingHostWorldOrigin(const Vector3& origin) {
    m_pendingHostWorldOrigin = origin;
    m_pendingHostWorldOriginValid = true;
  }
  void applyPendingHostWorldOrigin();
  // REMIXAPI_CAMERA_FRAME_EVENT_FALLBACK: at the frame's ray tracing, when no draw set the Main camera
  // this frame, the host's held fallback camera becomes it -- through the D3D9 draw path's own steps (pending
  // host origin, then CameraManager::applyFrameCamera), so it is the camera a world draw would have given.
  void applyFallbackMainCamera();

  // rtx.retained.useV2: the second retained-instance implementation (rtx_retained_v2.h). Exactly one
  // implementation owns the registered instances at a time. selectRetainedImplementation hands them
  // (with what their last replay produced) to the other one when the option changed; it runs at the
  // start of a DrawRetainedInstances replay. The retained commands above pass each command on to the
  // implementation that owns the instances, so commands queued around a switch land where the instances are.
  void selectRetainedImplementation(bool useV2);
  bool retainedV2Active() const { return m_retainedV2Active; }
  RetainedInstanceStore& retainedV2() { return *m_retainedV2; }

  // rtx.scene.directRetained (rtx_retained_scene.h). While the scene owns the retained instances, the retained
  // commands above go to it; it keeps what needs per-frame work on the first implementation (the *Legacy functions).
  // selectSceneImplementation runs at the start of a DrawRetainedInstances replay, after selectRetainedImplementation.
  void selectSceneImplementation(bool direct);
  RetainedScene& retainedScene() { return *m_retainedScene; }
  // rtx.scene.noImplicitClears: false for an API host (external meshes registered) while the option is on; a camera
  // cut or a frame without a valid camera then only resets temporal history. rtx.scene.d3d9CutAsEvent: the same for
  // every other game.
  bool implicitClearsAllowed() const;
  // The last camera cut that kept the scene (implicitClearsAllowed false), or kNoKeptCut. Lifetime never keeps an
  // object unseen since before it (translator::LifetimeFrame::keptCutFrame).
  static constexpr uint32_t kNoKeptCut = ~0u;
  uint32_t m_lastKeptCutFrame = kNoKeptCut;
  uint64_t m_clearEpoch = 0;  // clear() calls (clearEpoch)
  bool m_submittingHandedOver = false;  // See setSubmittingHandedOver
  // External-draw preserve-path accounting; see the definition for why it is worth having.
  void reportExternalPreserveOutcome(bool preserved);
  uint64_t m_externalDrawsPreserved = 0;
  uint64_t m_externalDrawsResubmitted = 0;
  uint32_t m_externalPreserveReportFrame = 0;
  void setStartInMediumMaterial(const MaterialData& translucentMaterial);
  void clearStartInMediumMaterial();

  // Remove an externally created mesh and all associated replacement instances.
  // Note: this is only safe to call from the dxvk-cs thread.
  // When the last external mesh goes away, tears the scene down and releases unused
  // DXVK chunks (see REMIX-5845) because an API client may never present again.
  void destroyExternalMesh(const Rc<DxvkContext>& ctx, remixapi_MeshHandle handle);
  
  void setExternalStartInMediumMaterial(const MaterialData& translucentMaterial);
  void clearExternalStartInMediumMaterial();
  
  bool hasAnyMods() const;
  std::vector<Mod::State> getReplacementStates() const;

  RtxGlobals& getGlobals() { return m_globals; }

  Rc<DxvkBuffer> getSurfaceMaterialBuffer() { return m_surfaceMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceMaterialExtensionBuffer() { return m_surfaceMaterialExtensionBuffer; }
  Rc<DxvkBuffer> getVolumeMaterialBuffer() { return m_volumeMaterialBuffer; }
  Rc<DxvkBuffer> getSurfaceBuffer() const { return m_accelManager.getSurfaceBuffer(); }
  Rc<DxvkBuffer> getSurfaceMappingBuffer() const { return m_accelManager.getSurfaceMappingBuffer(); }
  Rc<DxvkBuffer> getCurrentFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getCurrentFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getLastFramePrimitiveIDPrefixSumBuffer() const { return m_accelManager.getLastFramePrimitiveIDPrefixSumBuffer(); }
  Rc<DxvkBuffer> getBillboardsBuffer() const { return m_accelManager.getBillboardsBuffer(); }
  bool isPreviousFrameSceneAvailable() const { return m_previousFrameSceneAvailable && getSurfaceMappingBuffer().ptr() != nullptr; }

  const std::vector<Rc<DxvkSampler>>& getSamplerTable() const { return m_samplerCache.getObjectTable(); }
  const std::vector<RaytraceBuffer>& getBufferTable() const { return m_bufferCache.getObjectTable(); }
  const std::vector<RtInstance*>& getInstanceTable() const { return m_instanceManager.getInstanceTable(); }
  // For SccrShellManager: a material of the surface material cache, or null for an index outside it.
  const RtSurfaceMaterial* getSurfaceMaterialOrNull(uint32_t index) const {
    return index < m_surfaceMaterialCache.getTotalCount() ? &m_surfaceMaterialCache.getObjectTable()[index] : nullptr;
  }
  // The same for the surface-material extension cache (terrain-layers entries: SccrShellManager tells blended terrain
  // from a one-layer static by the entry's height material).
  const RtSurfaceMaterial* getSurfaceMaterialExtensionOrNull(uint32_t index) const {
    return index < m_surfaceMaterialExtensionCache.getTotalCount() ? &m_surfaceMaterialExtensionCache.getObjectTable()[index] : nullptr;
  }
  // For SccrShellManager: a bindless buffer slot of its own for each cached shell buffer, held from its build until the
  // cache drops it, independent of any geometry's slots.
  uint32_t acquireBufferSlot(const RaytraceBuffer& buffer) { return m_bufferCache.acquire(buffer); }
  void releaseBufferSlot(uint32_t slot) { m_bufferCache.release(slot); }
  // RtInstance::isRealGeometry changed on some instance (it decides AccelManager's BLAS routing), or a real geometry
  // binding changed: the next mergeInstancesIntoBlas must not take the unchanged-scene path.
  void notifyAccelSceneChanged() { m_instanceManager.notifySceneChanged(); }
  
  const InstanceManager& getInstanceManager() const { return m_instanceManager; }
  const AccelManager& getAccelManager() const { return m_accelManager; }
  // Sticky, set once any material with thin-opaque subsurface data (measurement distance > 0)
  // is created while rtx.subsurface.enableThinOpaque is on. Until then no surface can be thin-opaque.
  bool hasCreatedThinOpaqueMaterial() const { return m_thinOpaqueMaterialCreated.load(std::memory_order_relaxed); }
  // Sticky, set once any opaque material with the thin-film layer enabled / any ray portal material
  // is created. Until then no surface can carry a material thin film / be a portal (rtx.gbuffer.slimVariants).
  bool hasCreatedThinFilmMaterial() const { return m_thinFilmMaterialCreated.load(std::memory_order_relaxed); }
  bool hasCreatedRayPortalMaterial() const { return m_rayPortalMaterialCreated.load(std::memory_order_relaxed); }
  const LightManager& getLightManager() const { return m_lightManager; }
  const GraphManager& getGraphManager() const { return m_graphManager; }
  const RayPortalManager& getRayPortalManager() const { return m_rayPortalManager; }
  const LensManager& getLensManager() const { return m_lensManager; }
  LensManager& getLensManager() { return m_lensManager; }
  const BindlessResourceManager& getBindlessResourceManager() const { return m_bindlessResourceManager; }
  OpacityMicromapManager* getOpacityMicromapManager() const { return m_opacityMicromapManager.get(); }
  // Bindless index of the per-surface software micromap table, UINT32_MAX = none.
  uint32_t getSoftwareOmmTableIndex() const;
  // RaytraceArgs::coreMaterialTable, UINT32_MAX while rtx.coreMaterial.build is off or the table is not current.
  uint32_t getCoreMaterialTableIndex() const;
  // RaytraceArgs::coreMaterialExtTable, and the core table's surfaces per class (false while not current).
  uint32_t getCoreMaterialExtTableIndex() const;
  bool getCoreMaterialClassCounts(uint32_t (&counts)[4]) const;
  // RaytraceArgs::legacySurfaceTable, and whether any surface has legacy state / the eye (false while
  // rtx.coreMaterial.legacyRecord is off or the records are not current).
  uint32_t getLegacySurfaceTableIndex() const;
  bool getCoreMaterialLegacyBits(bool& anyLegacy, bool& anyTextureOps, bool& anyEye) const;
  LightManager& getLightManager() { return m_lightManager; }
  GraphManager& getGraphManager() { return m_graphManager; }
  std::unique_ptr<AssetReplacer>& getAssetReplacer() { return m_pReplacer; }
  TerrainBaker& getTerrainBaker() { return *m_terrainBaker.get(); }

  // Scene utility functions
  static Vector3 getSceneUp();
  static Vector3 getSceneForward();
  static Vector3 calculateSceneRight();

  // Reswizzles input vector to an output that has xy coordinates on scene's horizontal axes and z coordinate to be on the scene's vertical axis
  static Vector3 worldToSceneOrientedVector(const Vector3& worldVector); 

  static Vector3 sceneToWorldOrientedVector(const Vector3& sceneVector);

  void addLight(const D3DLIGHT9& light);

  const CameraManager& getCameraManager() const { return m_cameraManager; }
  CameraManager& getCameraManager() { return m_cameraManager; }
  const RtCamera& getCamera() const { return m_cameraManager.getMainCamera(); }
  RtCamera& getCamera() { return m_cameraManager.getMainCamera(); }

  const FogState& getFogState() const { return m_fog; }
  FogState& getFogState() { return m_fog; }
  const fast_unordered_cache<FogState>& getFogStates() const { return m_fogStates; }

  uint32_t getStartInMediumMaterialIndex() { return m_startInMediumMaterialIndex; }
  
  uint32_t getActivePOMCount() {return m_activePOMCount;}

  float getTotalMipBias();
  float getCalculatedUpscalingMipBias();

  // ISceneManager but not really
  void clear(Rc<DxvkContext> ctx, bool needWfi);
  // Clears only the ReplacementInstances whose buckets changed; unchanged RIs keep their
  // RtInstances, RtLights and graphs. Every consumer owns what it points at, so nothing
  // global needs clearing.
  void invalidateChangedReplacements(const AssetChanges& changes);
  void garbageCollection();
  void prepareSceneData(Rc<RtxContext> ctx, class DxvkBarrierSet& execBarriers);

  // CPU time of prepareSceneData by phase, summed since the last call, for [FrameTiming].
  // The first 9 partition prepareSceneData; the merge.* ones (from AccelManager) are nested
  // inside mergeInstancesIntoBlas and must not be added to the others.
  static constexpr uint32_t kPrepTopPhaseCount = 9;
  static constexpr uint32_t kPrepPhaseCount = 27;
  // Nested phases timed inside phases 0 and 1 (not to be added to them).
  static constexpr uint32_t kPrepNestedCount = 6;
  static constexpr uint32_t kPrepNestedLightMatch = 21;
  static constexpr uint32_t kPrepNestedGcGeometry = 22;
  static constexpr uint32_t kPrepNestedGcReplacementInstances = 23;
  static constexpr uint32_t kPrepNestedGcInstancesAndRest = 24;
  static constexpr uint32_t kPrepNestedAntiCull = 25;
  static constexpr uint32_t kPrepNestedBindless = 26;
  static constexpr const char* kPrepPhaseNames[kPrepPhaseCount] = {
    "lightMatch+gc", "antiCull+graph+terrain+bindless", "portals+omm+particles+virtual", "mergeInstancesIntoBlas",
    "accelPrepare(surfaces)", "lights", "surfaceMaterials", "piCulling+tlas", "instanceBuffers+rest",
    "merge.dirtyScan", "merge.route", "merge.dynBlas", "merge.restore+prefix", "merge.surfaceUpload", "merge.buildBlases",
    "merge.cache",
    // Nested inside merge.dynBlas, merge.surfaceUpload and merge.route respectively.
    "merge.dynBlas.omm", "merge.surfaceUpload.gpuData", "merge.route.scan", "merge.buildBlases.omm",
    "merge.restore+prefix.dynamicEmit",
    // Nested inside lightMatch+gc and antiCull+graph+terrain+bindless.
    "lightMatch+gc.lightMatch", "lightMatch+gc.gcGeometry", "lightMatch+gc.gcReplacementInstances",
    "lightMatch+gc.gcInstances+rest", "antiCull+graph+terrain+bindless.antiCull", "antiCull+graph+terrain+bindless.bindless" };
  uint32_t takeScenePrepPhases(double (&outMs)[kPrepPhaseCount]) {
    const uint32_t frames = m_prepPhaseFrames;
    for (uint32_t i = 0; i < kPrepPhaseCount; ++i) {
      outMs[i] = m_prepPhaseMs[i];
      m_prepPhaseMs[i] = 0.0;
    }
    m_prepPhaseFrames = 0;
    return frames;
  }
  // The most recent prepareSceneData call's phases (rtx.profile.detailedLogging).
  const double* getLastScenePrepPhases() const { return m_prepPhaseLastMs; }
private:
  double m_prepPhaseMs[kPrepPhaseCount] = {};
  double m_prepPhaseLastMs[kPrepPhaseCount] = {};
  void addPrepNestedMs(uint32_t index, double ms) {
    m_prepPhaseMs[index] += ms;
    m_prepPhaseLastMs[index] += ms;
  }
  uint32_t m_prepPhaseFrames = 0;
public:

  void onFrameEnd(Rc<DxvkContext> ctx, bool raytracedThisFrame);

  // GameCapturer
  void triggerUsdCapture() const;
  bool isGameCapturerIdle() const;

  using SamplerIndex = uint32_t;

  void trackTexture(const TextureRef& inputTexture,
                    uint32_t& textureIndex,
                    bool hasTexcoords,
                    bool async = true,
                    uint16_t* inout_samplerFeedbackStamp = nullptr);
  [[nodiscard]] SamplerIndex trackSampler(Rc<DxvkSampler> sampler);

  std::optional<XXH64_hash_t> findLegacyTextureHashByObjectPickingValue(uint32_t objectPickingValue);
  std::vector<ObjectPickingValue> gatherObjectPickingValuesByTextureHash(XXH64_hash_t texHash);

  // Replacement material hash tracking
  void trackReplacementMaterialHash(XXH64_hash_t materialHash);
  bool isReplacementMaterialHashUsedThisFrame(XXH64_hash_t materialHash) const;
  uint32_t getReplacementMaterialHashUsageCount(XXH64_hash_t materialHash) const;
  void clearFrameReplacementMaterialHashes();

  // Mesh hash tracking
  void trackMeshHash(XXH64_hash_t meshHash);
  bool isMeshHashUsedThisFrame(XXH64_hash_t meshHash) const;
  uint32_t getMeshHashUsageCount(XXH64_hash_t meshHash) const;
  void clearFrameMeshHashes();

  // rtx.trackHashUsageOnlyWhenNeeded: the hash checker components call this when they read the
  // usage maps; s_hashUsageTrackingWanted (also read by the D3D9 thread) is updated per frame.
  void noteHashUsageConsumer() const { m_hashUsageConsumerSeen.store(true, std::memory_order_relaxed); }
  inline static std::atomic<bool> s_hashUsageTrackingWanted { true };

  Rc<DxvkSampler> patchSampler( const VkFilter filterMode,
                                const VkSamplerAddressMode addressModeU,
                                const VkSamplerAddressMode addressModeV,
                                const VkSamplerAddressMode addressModeW,
                                const VkClearColorValue borderColor);

  void requestTextureVramFree();
  void requestVramCompaction();
  void manageTextureVram();

  bool isThinOpaqueMaterialExist() const { return m_thinOpaqueCount > 0; }
  bool isSssMaterialExist() const { return m_sssCount > 0; }

  // Live instance counts per material feature, kept by
  // retainSurfaceMaterial / releaseSurfaceMaterial. An instance that is alive but not drawn still counts.
  uint32_t getLiveThinOpaqueCount() const { return m_thinOpaqueCount; }
  uint32_t getLiveSssCount() const { return m_sssCount; }
  uint32_t getLivePomCount() const { return m_activePOMCount; }
  uint32_t getLiveThinFilmCount() const { return m_thinFilmCount; }
  uint32_t getLiveRayPortalCount() const { return m_rayPortalCount; }
  uint32_t getLiveTranslucentCount() const { return m_translucentCount; }
  uint32_t getLiveRaytracedRenderTargetCount() const { return m_raytracedRenderTargetCount; }

  bool isAntiCullingSupported() const { return m_isAntiCullingSupported; }

private:
  enum class ObjectCacheState
  {
    kUpdateInstance = 0,
    kUpdateBVH = 1,
    KBuildBVH = 2,
    kInvalid = -1
  };
  // Handles conversion of geometry data coming from a draw call, to the data used by the raytracing backend
  template<bool isNew>
  ObjectCacheState processGeometryInfo(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);

  // Consumes a draw call state and updates the scene state accordingly
  RtInstance* processDrawCallState(const Rc<DxvkContext>& ctx, 
                                   const DrawCallState& blasInput, 
                                   const MaterialData& materialData,
                                   ReplacementInstance& replacementInstance,
                                   RtInstance* existingInstance = nullptr,
                                   const RtxParticleSystemDesc* pParticleSystemDesc = nullptr);

  const RtSurfaceMaterial& createSurfaceMaterial(const MaterialData& renderMaterialData,
                                                 const DrawCallState& drawCallState,
                                                 uint32_t* out_indexInCache = nullptr);

  // Retain / release all resources associated with the surface material at the given cache index:
  // texture ref counts (via RtxTextureManager) and scene-wide feature counts (POM, SSS, thin-opaque).
  // Called when an instance's bound material changes or the instance is destroyed.
  void retainSurfaceMaterial(uint32_t matIdx);
  void releaseSurfaceMaterial(uint32_t matIdx);

  RtTranslucentSurfaceMaterial createTranslucentSurfaceMaterial(const TranslucentMaterialData& translucentMaterialData,
                                                                uint32_t samplerIndex,
                                                                bool hasTexcoords);
  Rc<DxvkSampler> getOrCreateExternalSampler();

  // Acquire/update stable buffer-cache slots for all geometry buffers in geo.
  // Called on the dynamic path (KBuildBVH / kUpdateBVH / kUpdateInstance) only.
  void updateBufferCache(BlasEntry* pBlas);

  // Propagate the BLAS's current geometry buffer indices/strides/formats to all
  // currently-linked instances.
  void syncLinkedInstances(BlasEntry* pBlas);

  // Release all buffer-cache slots held by geo. Called when a BlasEntry is GC'd.
  void unregisterGeometryBuffers(RaytraceGeometry& geo);

  // Release every BlasEntry's buffer slots and assert the active count reaches zero.
  // Called from clear() to detect registration/release imbalances before wiping the table.
  void verifyAndReleaseBufferCache();

  // Called whenever a new BLAS scene object is added to the cache
  ObjectCacheState onSceneObjectAdded(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas);
  // Called whenever a BLAS scene object is updated
  // rebuildTopologyInPlace (RetainedScene's owner-keyed geometry): the entry is rebuilt in place when the draw's
  // topology differs from the one it holds. Its instances then lose their opacity micromap data
  // (OpacityMicromapManager::onInstanceTopologyChanged).
  ObjectCacheState onSceneObjectUpdated(Rc<DxvkContext> ctx, const DrawCallState& drawCallState, BlasEntry* pBlas,
                                        bool rebuildTopologyInPlace = false);
  // Called whenever a new instance has been added to the database
  void onInstanceAdded(RtInstance& instance);
  // Called whenever instance metadata is updated
  void onInstanceUpdated(RtInstance& instance, const DrawCallState& drawCall, const MaterialData* material, const bool hasTransformChanged, const bool hasVerticesdChanged, const bool isFirstUpdateThisFrame);
  // Called whenever an instance has been removed from the database
  void onInstanceDestroyed(RtInstance& instance);

  void drawReplacements(Rc<DxvkContext> ctx, const DrawCallState* input, const std::shared_ptr<const ReplacementBucket>& pReplacements, MaterialData& renderMaterialData, ReplacementInstance* replacementInstance);

  // Registers the lenses of a mesh replacement on one of the mesh instances it drew, its anchor.
  // See documentation/Lens.md, "Lenses in mod USD".
  void updateLensAnchor(const DrawCallState& input, const std::shared_ptr<const ReplacementBucket>& pReplacements,
                        const ReplacementInstance& replacementInstance);
  // Adds the lenses of every live anchor to the lens manager, placed as their anchor instance is traced.
  void addReplacementLenses();

  // Build the per-replacement DrawCallState used by both the dynamic (drawReplacements) and
  // preserve (syncPreservedReplacementMeshesState) paths so they always feed the same input
  // into the BlasEntry / processDrawCallState. Returns std::nullopt for replacement types
  // that don't drive a mesh draw (lights, graphs).
  // Static member so the friendship of SceneManager with DrawCallState carries through.
  // Host (Remix API) GPU instancing on the input carries over to the replacement: every host
  // placement draws the replacement (see composeHostInstancesForReplacement). owner/index/frameId
  // key the cache of composed transform lists.
  static std::optional<DrawCallState> buildReplacementMeshDrawCallState(
      const DrawCallState& input,
      const AssetReplacement& replacement,
      const ReplacementInstance* owner,
      size_t index,
      uint32_t frameId);

  // The transform list a mesh replacement is instanced with when the draw it replaces is
  // host-instanced. world(k) = objectToWorld * host[i] * replacementToObject, or, for a replacement
  // that is itself a USD PointInstancer, objectToWorld * host[i] * proto[j] (every pair). Returns
  // the host list itself when there is nothing to compose. Results are cached per (owner, index)
  // and reused while the host list's contents and the replacement do not change, so a steady
  // instanced draw does no per-instance CPU work beyond one comparison of the list.
  static std::shared_ptr<const std::vector<Matrix4>> composeHostInstancesForReplacement(
      const std::shared_ptr<const std::vector<Matrix4>>& hostInstances,
      const AssetReplacement& replacement,
      const ReplacementInstance* owner,
      size_t index,
      uint32_t frameId);

  // Minimal per-frame work for the preserve path (unchanged / anti-culled draws).
  // When pInput is set (preserve replacement draw path), also runs the ray-portal refresh
  // via processRayPortalData on the cached RtSurfaceMaterial.
  void preserveInstance(
      RtInstance& instance,
      const DrawCallState* pInput = nullptr);

  // Lights and graph prims for mesh replacements (drawReplacements / dynamic path).
  void processReplacementLights(const DrawCallState* input, const ReplacementBucket* pReplacements, ReplacementInstance* replacementInstance);
  void processReplacementGraphs(Rc<DxvkContext> ctx, const DrawCallState* input, const ReplacementBucket* pReplacements, ReplacementInstance* replacementInstance);

  // Preserve path: minimal work to keep replacement meshes, lights, and graphs alive (buffer cache, textures, light touch, bbox).
  // Returns `input` (the draw state to keep reading from).
  const DrawCallState& preserveReplacementInstance(
      Rc<DxvkContext> ctx,
      const DrawCallState& input,
      const std::shared_ptr<const ReplacementBucket>& pReplacements,
      ReplacementInstance* replacementInstance);

  void trackObjectPickingMeta(const DrawCallState& drawCallState, ObjectPickingValue objectPickingValue);
  void trackObjectPickingMeta(ObjectPickingValue objectPickingValue, XXH64_hash_t textureHash, XXH64_hash_t textureHash2 = kEmptyHash);

  // Remix API draws: the picking value of the draw's instances -- the host's
  // (remixapi_InstanceInfoObjectPickingEXT) when it gave one, else the replacement instance's own
  // (ReplacementInstance::externalPickingValue, assigned on first use).
  ObjectPickingValue externalObjectPickingValue(const DrawCallState& drawCall, ReplacementInstance& replacementInstance);

  // Refreshes BlasEntry::input and per-instance draw state on the preserve path (matches drawReplacements'
  // DrawCallState wiring).
  void syncPreservedReplacementMeshesState(
      const DrawCallState& input,
      const ReplacementBucket* pReplacements,
      ReplacementInstance* replacementInstance);

  void createEffectLight(Rc<DxvkContext> ctx, const DrawCallState& input, const RtInstance* instance);
  // The materials that have made an effect light, each logged once.
  std::unordered_set<XXH64_hash_t> m_effectLightMaterialsLogged;

  // Print all RtInstances for debugging
  void printAllRtInstances();

  // rtx.accelMemoryLogIntervalFrames: logs what the acceleration-structure memory is made of
  // and how fragmented the device-local heap's chunks are.
  void logAccelMemoryBreakdown();
  
  MaterialData determineMaterialData(const MaterialData* overrideMaterialData, const DrawCallState& input);
  
  const uint32_t kInvalidMaterialCacheIndex = UINT32_MAX;
  uint32_t m_beginUsdExportFrameNum = -1;
  bool m_enqueueDelayedClear = false;
  bool m_previousFrameSceneAvailable = false;

  RtxGlobals m_globals;

  // Hash/Cache's
  InstanceManager m_instanceManager;
  AccelManager m_accelManager;
  std::atomic<bool> m_thinOpaqueMaterialCreated { false }; // See hasCreatedThinOpaqueMaterial()
  std::atomic<bool> m_thinFilmMaterialCreated { false };   // See hasCreatedThinFilmMaterial()
  std::atomic<bool> m_rayPortalMaterialCreated { false };  // See hasCreatedRayPortalMaterial()
  LightManager m_lightManager;
  GraphManager m_graphManager;
  RayPortalManager m_rayPortalManager;
  LensManager m_lensManager;

  // The lenses carried by a mesh replacement instance, keyed by its anchor instance.
  struct LensAnchor {
    std::shared_ptr<const ReplacementBucket> bucket;
    // The index of each lens in the bucket, with its transform to the anchor instance's object space.
    std::vector<std::pair<uint32_t, Matrix4d>> lensToAnchor;
  };
  std::mutex m_lensAnchorMutex;
  std::unordered_map<const RtInstance*, LensAnchor> m_lensAnchors;
  uint32_t m_lensesAddedFrameId = kInvalidFrameIndex;

  BindlessResourceManager m_bindlessResourceManager;
  std::unique_ptr<OpacityMicromapManager> m_opacityMicromapManager;
  std::unique_ptr<CoreMaterialBuilder> m_coreMaterials; // rtx.coreMaterial.build, null while off

  DrawCallCache m_drawCallCache;

  CameraManager m_cameraManager;

  std::unique_ptr<AssetReplacer> m_pReplacer;

  std::unique_ptr<TerrainBaker> m_terrainBaker;

  FogState m_fog;
  fast_unordered_cache<FogState> m_fogStates;
  bool m_frontendFogDiscovery = false;
  std::mutex m_startInMediumMaterialMutex;
  std::optional<MaterialData> m_pendingStartInMediumMaterial;
  bool m_pendingClearStartInMediumMaterial = false;
  std::optional<MaterialData> m_persistentStartInMediumMaterial;
  uint32_t m_startInMediumMaterialIndex = SURFACE_INDEX_INVALID;
  uint32_t m_fogStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_externalStartInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_startInMediumMaterialIndex_inCache = kInvalidMaterialCacheIndex;
  uint32_t m_lastResolvedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;
  uint32_t m_lastUploadedStartInMediumMaterialIndexInCache = kInvalidMaterialCacheIndex;

  // TODO: Move the following resources and getters to RtResources class
  Rc<DxvkBuffer> m_surfaceMaterialBuffer;
  // rtx.incrementalSurfaceMaterialUpload: the table being built, what the buffer holds (shadow), and
  // per-record dirty / GPU-owned flags.
  std::vector<unsigned char> m_surfaceMaterialsGPUData;
  std::vector<unsigned char> m_uploadedSurfaceMaterialData;
  std::vector<uint8_t> m_surfaceMaterialSlotFlags;
  Rc<DxvkBuffer> m_uploadedSurfaceMaterialBuffer;
  void uploadSurfaceMaterialsIncremental(const Rc<RtxContext>& ctx, size_t surfaceMaterialsGPUSize);
  Rc<DxvkBuffer> m_surfaceMaterialExtensionBuffer;
  Rc<DxvkBuffer> m_volumeMaterialBuffer;

  uint32_t m_activePOMCount = 0;
  
  float m_uniqueObjectSearchDistance = 1.f;

  struct DrawCallMetaInfo {
    XXH64_hash_t legacyTextureHash { kEmptyHash };
    XXH64_hash_t legacyTextureHash2 { kEmptyHash };
  };
  struct DrawCallMeta {
    constexpr static inline uint8_t MaxTicks = 2;
    std::unordered_map<ObjectPickingValue, DrawCallMetaInfo> infos[MaxTicks] {};
    bool ready[MaxTicks] {};
    uint8_t ticker {};
    dxvk::mutex mutex {};
  } m_drawCallMeta {};
  // externalObjectPickingValue: the next value handed out (in the upper half of the range, which
  // the D3D9 per-frame draw call IDs never reach).
  uint32_t m_nextExternalPickingValue = 0;

  // TODO: expand to many different
  Rc<DxvkSampler> m_externalSampler = nullptr;

  std::atomic_bool m_forceFreeTextureMemory = false;
  std::atomic_bool m_forceFreeUnusedDxvkAllocatorChunks = false;

  uint32_t m_thinOpaqueCount = 0;
  uint32_t m_sssCount = 0;
  // See getLiveThinFilmCount() and the others.
  uint32_t m_thinFilmCount = 0;
  uint32_t m_rayPortalCount = 0;
  uint32_t m_translucentCount = 0;
  uint32_t m_raytracedRenderTargetCount = 0;

  bool m_isAntiCullingSupported = true;

  // Matches RtxTextureManager::m_textureCacheGeneration after a completed frame's texture
  // registrations. Updated once per frame in onFrameEnd (before manageTextureVram). Left
  // stale when manageTextureVram clears the cache so the next frame is all-dynamic.
  uint32_t m_textureCacheGenerationValidForPreserve = 0;

  // Replacement material hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameReplacementMaterialHashes;

  // Mesh hash tracking for current frame (hash -> count)
  std::unordered_map<XXH64_hash_t, uint32_t> m_currentFrameMeshHashes;

  DrawCallTracker m_drawCallTracker;

  // The absolute transform is kept apart from the state because the replay
  // rebases a copy of the state in place.
  struct RetainedExternalDraw {
    uint64_t handle = 0;
    ExternalDrawState state {};
    Matrix4 absoluteObjectToWorld;
    bool visible = true;
    bool described = true;   // description replaced since the last replay

    // What the last full replay produced, so an unchanged entry can take the
    // retained preserve path (rtx.enableRetainedPreservePath) instead of copying
    // its state and re-running submitExternalDraw.
    bool hasFullReplay = false;
    XXH64_hash_t identityHash = 0;
    Matrix4 replayedObjectToWorld;   // rebased transform of the last replay
    std::weak_ptr<const std::vector<RasterGeometry>> replayedSubmeshes;
    XXH64_hash_t replayedMaterialHash = 0;
    // rtx.retainedHiddenPolicy 2 hid its instances from the TLAS.
    bool hiddenFromTlas = false;
    // rtx.retainedSteadyFastPath: the replacement instance and submesh bounds the last full
    // preserve check accepted, valid while both epochs below still match.
    ReplacementInstance* steadyRI = nullptr;
    uint64_t steadyTrackerEpoch = 0;   // retainedSteadyTrackerEpoch()
    uint64_t steadyAssetEpoch = 0;     // SceneManager::m_retainedSteadyAssetEpoch
    // rtx.retainedPreciseInvalidation: the asset handles this entry is indexed under.
    uint64_t indexedMesh = 0;
    uint64_t indexedMaterial = 0;
    uint64_t indexedOverride = 0;
    AxisAlignedBoundingBox steadyBbox {};
  };
  // Bumped when an input of the skipped retained checks other than the replacement instance
  // changed since the last replay (asset epoch, light-conversion list, replacement options).
  uint64_t m_retainedSteadyAssetEpoch = 1;
  struct RetainedSteadyInputs {
    uint64_t assetEpoch = ~0ull;
    uint64_t lightConverterSig = ~0ull;
    size_t lightConverterCount = ~size_t(0);
    bool replacementMeshes = false;
    bool operator==(const RetainedSteadyInputs& o) const {
      return assetEpoch == o.assetEpoch && lightConverterSig == o.lightConverterSig &&
             lightConverterCount == o.lightConverterCount && replacementMeshes == o.replacementMeshes;
    }
  } m_retainedSteadyInputs;
  uint32_t m_retainedSteadyMismatches = 0;
  bool m_retainedPreserveOptionsOn = false; // set per replay in submitRetainedExternalDraws
  std::vector<RetainedExternalDraw> m_retainedExternalDraws;
  std::unordered_map<uint64_t, size_t> m_retainedExternalIndex;
  Vector3 m_retainedSceneOrigin { 0.f, 0.f, 0.f };
  Vector3 m_worldOrigin { 0.f, 0.f, 0.f };
  bool m_worldOriginValid = false;
  Vector3 m_pendingHostWorldOrigin { 0.f, 0.f, 0.f };
  bool m_pendingHostWorldOriginValid = false;

  // Identity hash computed by the most recent submitExternalDraw (0 when it bailed
  // out before computing one); the retained replay records it per entry.
  XXH64_hash_t m_lastExternalDrawIdentityHash = 0;

  // rtx.retained.diagnosticFrames: per-window counts of full replays that went wrong, by reason.
  struct RetainedDiag {
    enum Reason : uint32_t { NoMesh, NoIdentity, NoReplacement, Recreated, NoInstance, Hidden, MarkedForGC, Count };
    uint32_t fullReplays = 0;
    uint32_t counts[Count] = {};
    uint64_t examples[Count][4] = {};
  } m_retainedDiag;

  // rtx.retainedPinSteadyEntries: per entry (same index as m_retainedExternalDraws), the instance,
  // geometry and replacement instance a pinned entry keeps alive each frame; all null = not pinned.
  // Pointers are only dereferenced while every invalidation source is unchanged (checked at the
  // start of each replay and before each pinned entry), so dropping a pin never touches them.
  struct RetainedPin {
    RtInstance* instance = nullptr;
    BlasEntry* blas = nullptr;
    ReplacementInstance* replacementInstance = nullptr;
  };
  std::vector<RetainedPin> m_retainedPins;
  uint32_t m_retainedPinCount = 0;
  // Bumped when every pin is dropped at once; stamps on RtInstance / BlasEntry from older
  // generations mean "not pinned".
  uint64_t m_retainedPinGeneration = 1;
  // What the pins were made under; any change drops all pins.
  struct RetainedPinInputs {
    uint64_t trackerEpoch = ~0ull;
    uint64_t assetEpoch = ~0ull;
    uint64_t textureCacheGeneration = ~0ull;
    Vector3 sceneOrigin { 0.f, 0.f, 0.f };
    uint32_t hiddenPolicy = ~0u;
  } m_retainedPinInputs;
  uint32_t m_retainedPinMismatches = 0;
public:
  // [PerfCpu] apiDraws split: geometry (draw-call cache, geometry processing, skinning) and
  // instance (instance / material update) time of per-frame DrawInstance calls.
  bool m_timeApiDraws = false;
  double m_apiDrawGeometryMs = 0.0;
  double m_apiDrawInstanceMs = 0.0;
  // [PerfCpu] d3d9.preserve / d3d9.dynamic: submitDrawState time split by path, while m_timeD3D9Submit
  // is set by RtxContext::commitGeometryToRT (detailed logging only). Drained by RtxContext.
  bool m_timeD3D9Submit = false;
  double m_d3d9PreserveMs = 0.0;
  double m_d3d9DynamicMs = 0.0;
private:
  // rtx.trackHashUsageOnlyWhenNeeded
  mutable std::atomic<bool> m_hashUsageConsumerSeen { false };
  uint32_t m_framesSinceHashUsageConsumer = 0;
  // rtx.retainedPreciseInvalidation
  bool m_retainedPreciseOn = false;
  std::unordered_map<const ReplacementInstance*, uint64_t> m_retainedSteadyByRI;
  std::unordered_multimap<uint64_t, uint64_t> m_retainedSteadyByAsset;
  size_t m_retainedInvalidationsConsumed = 0;
  std::vector<uint64_t> m_retainedChangedAssetsScratch;
  void dropRetainedSteady(uint64_t handle, const ReplacementInstance* ri);
  void consumeRetainedSteadyInvalidations();
  uint64_t retainedSteadyTrackerEpoch() const {
    return m_retainedPreciseOn ? m_drawCallTracker.getRetainedSteadyClearEpoch() : m_drawCallTracker.getRetainedSteadyEpoch();
  }
  // Entries still pinned at the end of a replay's loop (stamped after it), and the per-entry
  // "needs the serial stamp" flags of rtx.retainedParallelPinnedStamps.
  std::vector<uint32_t> m_retainedDeferredPins;
  std::vector<uint8_t> m_retainedStampSerialScratch;
  void unpinRetained(size_t index) {
    if (index < m_retainedPins.size() && m_retainedPins[index].instance != nullptr) {
      m_retainedPins[index] = RetainedPin {};
      --m_retainedPinCount;
    }
  }
  void unpinAllRetained() {
    if (m_retainedPinCount != 0) {
      std::fill(m_retainedPins.begin(), m_retainedPins.end(), RetainedPin {});
      m_retainedPinCount = 0;
    }
    ++m_retainedPinGeneration;
  }
  // A draw updated the vertex data of geometry that pinned entries draw: they take the full
  // checks again (which send them to a full replay this frame, as without pinning).
  void wakeRetainedPinsOf(BlasEntry& blas);

  bool tryPreserveRetainedExternalDraw(RetainedExternalDraw& entry, const Matrix4& objectToWorld,
                                       bool steadyOn = false, bool verifySteady = false, bool* tookSteady = nullptr);
  // Submesh material, honouring ExternalDrawState::materialOverride.
  const MaterialData* resolveExternalDrawMaterial(const ExternalDrawState& state, const RasterGeometry& submesh) const;
  // Scene interface: the external-draw setup shared by submitExternalDraw and the scene's direct
  // lane. The first runs once per draw (samplers, camera matrices, host instancing list); the second once per submesh
  // and returns the material to render it with. keepAlive owns a USD replacement material while the caller uses it.
  void prepareExternalDrawCall(ExternalDrawState& state);
  // The description's submeshes: its inline mesh (frontends), else the API mesh registered under state.mesh.
  std::shared_ptr<const std::vector<RasterGeometry>> externalSubmeshes(const ExternalDrawState& state) const;
  const MaterialData& prepareExternalSubmeshDrawCall(ExternalDrawState& state,
                                                     const std::shared_ptr<const std::vector<RasterGeometry>>& submeshesRef,
                                                     size_t index,
                                                     std::shared_ptr<MaterialData>& keepAlive);
  void setRetainedExternalDrawHiddenFromTlas(RetainedExternalDraw& entry, bool hidden);

  // rtx.retained.useV2 (see selectRetainedImplementation).
  std::unique_ptr<RetainedInstanceStore> m_retainedV2;
  bool m_retainedV2Active = false;

  // rtx.scene.directRetained: the first implementation's side of the scene's legacy lane.
  std::unique_ptr<RetainedScene> m_retainedScene;
  void setRetainedExternalDrawLegacy(uint64_t handle, ExternalDrawState&& state);
  void setRetainedExternalDrawTransformLegacy(uint64_t handle, const Matrix4& objectToWorld);
  void setRetainedExternalDrawVisibleLegacy(uint64_t handle, bool visible);
  void removeRetainedExternalDrawLegacy(uint64_t handle);
  // Removes the entry and its RT instances now (collected before this frame's TLAS), so another lane can take
  // the handle in the same frame without a duplicate or a gap.
  void dropRetainedExternalDrawNow(uint64_t handle);
  void listRetainedExternalDrawsLegacy(std::vector<std::pair<uint64_t, bool>>& out) const;
  // The registered description with its current absolute transform; false if the handle is not registered.
  bool copyRetainedExternalDrawLegacy(uint64_t handle, ExternalDrawState& out) const;
};

}  // namespace nvvk
