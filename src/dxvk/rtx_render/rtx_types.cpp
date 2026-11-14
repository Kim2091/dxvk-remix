/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>

#include "rtx_types.h"
#include "rtx_asset_replacer.h"
#include "rtx_options.h"
#include "rtx_terrain_baker.h"
#include "rtx_instance_manager.h"
#include "rtx_light_manager.h"
#include "graph/rtx_graph_instance.h"
#include "dxvk_scoped_annotation.h"

namespace dxvk {

  // Instance constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtInstance* instance) : m_type(Type::Instance) {
    m_ptr.instance = instance;
  }
  RtInstance* PrimInstance::getInstance() const {
    if (m_type != Type::Instance) {
      return nullptr;
    }
    return m_ptr.instance;
  }

  // Light constructor, getter, and assignment operator
  PrimInstance::PrimInstance(RtLight* light) : m_type(Type::Light) {
    m_ptr.light = light;
  }
  RtLight* PrimInstance::getLight() const {
    if (m_type != Type::Light) {
      return nullptr;
    }
    return m_ptr.light;
  }

  // Graph constructor, getter, and assignment operator
  PrimInstance::PrimInstance(GraphInstance* graph) : m_type(Type::Graph) {
    m_ptr.graph = graph;
  }
  GraphInstance* PrimInstance::getGraph() const {
    if (m_type != Type::Graph) {
      return nullptr;
    }
    return m_ptr.graph;
  }

  PrimInstance::Type PrimInstance::getType() const {
    if (m_ptr.untyped == nullptr) {
      return Type::None;
    }
    return m_type;
  }

  PrimInstance::PrimInstance(void* owner, Type type) : m_type(type) {
    m_ptr.untyped = owner;
  }

  void* PrimInstance::getUntyped() const {
    return m_ptr.untyped;
  }

  void PrimInstance::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex) {
    PrimInstanceOwner* prim = nullptr;
    if (m_type == Type::Instance) {
      prim = &m_ptr.instance->getPrimInstanceOwner();
    } else if (m_type == Type::Light) {
      prim = &m_ptr.light->getPrimInstanceOwner();
    } else if (m_type == Type::Graph) {
      prim = &m_ptr.graph->getPrimInstanceOwner();
    }

    if (prim) {
      prim->setReplacementInstance(replacementInstance, replacementIndex, m_ptr.untyped, m_type);
    }
  }

  std::ostream& operator << (std::ostream& os, PrimInstance::Type type) {
    switch (type) {
      ENUM_NAME(PrimInstance::Type::Instance);
      ENUM_NAME(PrimInstance::Type::Light);
      ENUM_NAME(PrimInstance::Type::Graph);
      ENUM_NAME(PrimInstance::Type::None);
    }
    return os << static_cast<uint8_t>(type);
  }

  ReplacementInstance::~ReplacementInstance() {
    clear();
  }

  void ReplacementInstance::clear() {
    // Mark all prim entities for GC and detach their back-pointers, then drop
    // the prim/root slots, the active-replacements tracking pointer, and the
    // cached aggregate bounding boxes so the RI is in a clean "no replacement
    // attached" state. setup() is the matching re-init.
    for (size_t i = 0; i < prims.size(); i++) {
      RtInstance* subInstance = prims[i].getInstance();
      if (subInstance) {
        subInstance->markForGarbageCollection();
      }
      GraphInstance* graphInstance = prims[i].getGraph();
      if (graphInstance) {
        graphInstance->removeInstance();
      }
      RtLight* light = prims[i].getLight();
      if (light) {
        light->markForGarbageCollection();
      }
      prims[i].setReplacementInstance(nullptr, kInvalidReplacementIndex);
    }
    prims.clear();
    root = PrimInstance();
    activeReplacements.reset();
    legacyMaterialIdentityHash = kEmptyHash;
    geometryBoundingBox.invalidate();
    lightBoundingBox.invalidate();
    boundingBoxDirty = true;
    dirtyFlags = kAllDirtyFlags;
  }

  ReplacementInstance::ReplacementInstance(const LookupKey& key, uint32_t newId, uint32_t frameId)
      : id(newId)
      , identityHash(key.identityHash)
      , spatialMapHash(key.spatialMapHash)
      , materialHash(key.materialHash)
      , vertexPositionHash(key.vertexPositionHash)
      , centroid(key.worldPos)
      , frameCreated(frameId)
      , textureTransform(key.textureTransform)
      , texgenMode(key.texgenMode) {
    // No prior data to diff against; every field is effectively new. Set all
    // dirty bits so downstream update logic that gates individual steps on
    // specific bits runs the full update on the RI's first submission.
    dirtyFlags = kAllDirtyFlags;
  }

  void ReplacementInstance::setup(PrimInstance newRoot, size_t numPrims,
                                  std::shared_ptr<const ReplacementBucket> replacements) {
    clear();
    prims.resize(numPrims);
    root = newRoot;
    activeReplacements = std::move(replacements);
  }

  void ReplacementInstance::recalculateBoundingBox(
      const Matrix4& newObjectToWorld,
      const AxisAlignedBoundingBox* originalGeometryBBox) {
    objectToWorld = newObjectToWorld;

    if (!boundingBoxDirty) {
      return;
    }

    AxisAlignedBoundingBox geoBBox;
    AxisAlignedBoundingBox litBBox;

    if (activeReplacements == nullptr) {
      geoBBox = *originalGeometryBBox;
    } else {
      for (const auto& replacement : activeReplacements->replacements) {
        if (replacement.includeOriginal && originalGeometryBBox != nullptr) {
          geoBBox.unionWith(*originalGeometryBBox);
        } else if (replacement.type == AssetReplacement::eMesh && replacement.geometry != nullptr) {
          const AxisAlignedBoundingBox& srcBBox = replacement.geometry->data.boundingBox;
          if (srcBBox.isValid()) {
            const Vector3& mn = srcBBox.minPos;
            const Vector3& mx = srcBBox.maxPos;
            const Vector3 corners[8] = {
              Vector3(mn.x, mn.y, mn.z), Vector3(mx.x, mn.y, mn.z),
              Vector3(mn.x, mx.y, mn.z), Vector3(mn.x, mn.y, mx.z),
              Vector3(mx.x, mx.y, mn.z), Vector3(mn.x, mx.y, mx.z),
              Vector3(mx.x, mn.y, mx.z), Vector3(mx.x, mx.y, mx.z)
            };
            for (const Vector3& corner : corners) {
              const Vector3 transformed = (replacement.replacementToObject * Vector4(corner, 1.0f)).xyz();
              for (uint32_t j = 0; j < 3; j++) {
                geoBBox.minPos[j] = std::min(geoBBox.minPos[j], transformed[j]);
                geoBBox.maxPos[j] = std::max(geoBBox.maxPos[j], transformed[j]);
              }
            }
          }
        } else if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
          RtLight objectSpaceLight = replacement.lightData->toRtLight();
          const Vector3 pos = objectSpaceLight.getPosition();
          float lightRadius = 0.f;
          if (objectSpaceLight.getType() == RtLightType::Sphere) {
            lightRadius = objectSpaceLight.getSphereLight().getRadius();
          }
          for (uint32_t j = 0; j < 3; j++) {
            litBBox.minPos[j] = std::min(litBBox.minPos[j], pos[j] - lightRadius);
            litBBox.maxPos[j] = std::max(litBBox.maxPos[j], pos[j] + lightRadius);
          }
        }
      }
    }

    if (geoBBox.isValid()) {
      geometryBoundingBox = geoBBox;
    }
    if (litBBox.isValid()) {
      lightBoundingBox = litBBox;
    }
    boundingBoxDirty = false;
  }

  bool PrimInstanceOwner::isRoot(const void* owner) const {
    return m_replacementInstance != nullptr
      && m_replacementIndex != ReplacementInstance::kInvalidReplacementIndex
      && m_replacementInstance->root.getUntyped() == owner;
  }

  void PrimInstanceOwner::setReplacementInstance(ReplacementInstance* replacementInstance, size_t replacementIndex, void* owner, PrimInstance::Type type) {
    // No-op if already linked to the same slot
    if (m_replacementInstance == replacementInstance && m_replacementIndex == replacementIndex) {
      return;
    }

    // Unlink from current ReplacementInstance
    if (m_replacementInstance != nullptr &&
        m_replacementIndex < m_replacementInstance->prims.size()) {
      PrimInstance& currentSlot = m_replacementInstance->prims[m_replacementIndex];
      if (currentSlot.getUntyped() == owner) {
        currentSlot = PrimInstance();
      }
      if (m_replacementInstance->root.getUntyped() == owner) {
        m_replacementInstance->root = PrimInstance();
      }
    }

    // Link to new ReplacementInstance
    m_replacementInstance = replacementInstance;
    m_replacementIndex = replacementIndex;

    if (m_replacementInstance != nullptr &&
        m_replacementIndex < m_replacementInstance->prims.size()) {
      PrimInstance& targetSlot = m_replacementInstance->prims[m_replacementIndex];
      if (targetSlot.getUntyped() != nullptr && targetSlot.getUntyped() != owner) {
        targetSlot.setReplacementInstance(nullptr, ReplacementInstance::kInvalidReplacementIndex);
      }
      targetSlot = PrimInstance(owner, type);
    }
  }

  uint32_t RasterGeometry::calculatePrimitiveCount() const {
    const uint32_t elementCount = usesIndices() ? indexCount : vertexCount;
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return elementCount / 3;

    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return elementCount >= 3
        ? elementCount - 2
        : 0;

    default:
      assert(!"Unsupported primitive topology");
      return UINT32_MAX;
    }
  }

  bool DrawCallState::finalizePendingFutures(const RtCamera* pLastCamera) {
    ScopedCpuProfileZone();
    // Geometry hashes are vital, and cannot be disabled, so its important we get valid data (hence the return type)
    const bool valid = finalizeGeometryHashes();
    if (valid) {
      // Bounding boxes (if enabled) will be finalized here, default is FLT_MAX bounds
      finalizeGeometryBoundingBox();

      // Skinning processing will be finalized here, if object requires skinning
      finalizeSkinningData(pLastCamera);

      // Update any categories that require geometry hash
      setupCategoriesForGeometry();

      return true;
    }

    return false;
  }

  bool DrawCallState::isEye() const {
    if (RtxOptions::Eye::enable() && RtxOptions::Eye::assumeViewTexgenModeAsEye()) {
      return getTransformData().texgenMode == TexGenMode::ViewPositions;
    }
    return false;
  }

  bool DrawCallState::finalizeGeometryHashes() {
    if (geometryData.hashesPrecomputed) {
      // rtx.geometryHashMemoInline: served from the memo on the D3D9 thread, no worker task.
      geometryData.hashesPrecomputed = false;
    } else {
      if (!geometryData.futureGeometryHashes.valid()) {
        return false;
      }

      geometryData.hashes = geometryData.futureGeometryHashes.get();
    }

    if (geometryData.hashes[HashComponents::VertexPosition] == kEmptyHash) {
      throw DxvkError("Position hash should never be empty");
    }

    return true;
  }

  void DrawCallState::finalizeGeometryBoundingBox() {
    if (geometryData.futureBoundingBox.valid())
      geometryData.boundingBox = geometryData.futureBoundingBox.get();
  }

  void DrawCallState::finalizeSkinningData(const RtCamera* pLastCamera) {
    if (futureSkinningData.valid()) {
      skinningData = futureSkinningData.get();

      assert(geometryData.blendWeightBuffer.defined());
      assert(skinningData.numBonesPerVertex <= 4);

      if (pLastCamera != nullptr) {
        const auto fusedMode = RtxOptions::fusedWorldViewMode();
        if (likely(fusedMode == FusedWorldViewMode::None)) {
          transformData.objectToView = transformData.worldToView;
          // Do not bother when transform is fused. Camera matrices are identity and so is worldToView.
        }
        transformData.objectToWorld = pLastCamera->getViewToWorld(false) * transformData.objectToView;
        transformData.worldToView = pLastCamera->getWorldToView(false);
      } else {
        ONCE(Logger::warn("[RTX-Compatibility-Warn] Cannot decompose the matrices for a skinned mesh because the camera is not set."));
      }

      // In rare cases when the mesh is skinned but has only one active bone, skip the skinning pass
      // and bake that single bone into the objectToWorld/View matrices.
      if (skinningData.minBoneIndex + 1 == skinningData.numBones) {
        const Matrix4& skinningMatrix = skinningData.pBoneMatrices[skinningData.minBoneIndex];

        transformData.objectToWorld = transformData.objectToWorld * skinningMatrix;
        transformData.objectToView = transformData.objectToView * skinningMatrix;

        skinningData.boneHash = 0;
        skinningData.numBones = 0;
        skinningData.numBonesPerVertex = 0;
      }

      // Store the numBonesPerVertex in the RasterGeometry as well to allow it to be overridden
      geometryData.numBonesPerVertex = skinningData.numBonesPerVertex;
    }
  }

  void DrawCallState::setCategory(InstanceCategories category, bool doSet) {
    if (doSet) {
      categories.set(category);
    }
  }

  void DrawCallState::removeCategory(InstanceCategories category) {
    categories.clr(category);
  }

  CategoryFlags DrawCallState::computeTextureListCategories(const XXH64_hash_t& textureHash) {
    // The texture-list part of setupCategoriesForTexture(): depends only on the texture hash and the
    // option values, so rtx.textureCategoryCache can cache it per texture hash.
    CategoryFlags result = 0;
    const auto setIf = [&result](InstanceCategories category, bool doSet) {
      if (doSet) {
        result.set(category);
      }
    };

    setIf(InstanceCategories::WorldUI, lookupHash(RtxOptions::worldSpaceUiTextures(), textureHash));
    setIf(InstanceCategories::WorldMatte, lookupHash(RtxOptions::worldSpaceUiBackgroundTextures(), textureHash));

    setIf(InstanceCategories::Ignore, lookupHash(RtxOptions::ignoreTextures(), textureHash));
    setIf(InstanceCategories::IgnoreLights, lookupHash(RtxOptions::ignoreLights(), textureHash));
    setIf(InstanceCategories::IgnoreAntiCulling, lookupHash(RtxOptions::antiCullingTextures(), textureHash));
    setIf(InstanceCategories::IgnoreMotionBlur, lookupHash(RtxOptions::motionBlurMaskOutTextures(), textureHash));
    setIf(InstanceCategories::IgnoreOpacityMicromap, lookupHash(RtxOptions::opacityMicromapIgnoreTextures(), textureHash));
    setIf(InstanceCategories::IgnoreAlphaChannel, lookupHash(RtxOptions::ignoreAlphaOnTextures(), textureHash));
    setIf(InstanceCategories::IgnoreBakedLighting, lookupHash(RtxOptions::ignoreBakedLightingTextures(), textureHash));

    setIf(InstanceCategories::Hidden, lookupHash(RtxOptions::hideInstanceTextures(), textureHash));

    setIf(InstanceCategories::Particle, lookupHash(RtxOptions::particleTextures(), textureHash));
    setIf(InstanceCategories::Beam, lookupHash(RtxOptions::beamTextures(), textureHash));

    setIf(InstanceCategories::DecalStatic, lookupHash(RtxOptions::decalTextures(), textureHash));
    setIf(InstanceCategories::DecalDynamic, lookupHash(RtxOptions::dynamicDecalTextures(), textureHash));
    setIf(InstanceCategories::DecalSingleOffset, lookupHash(RtxOptions::singleOffsetDecalTextures(), textureHash));
    setIf(InstanceCategories::DecalNoOffset, lookupHash(RtxOptions::nonOffsetDecalTextures(), textureHash));

    setIf(InstanceCategories::AnimatedWater, lookupHash(RtxOptions::animatedWaterTextures(), textureHash));

    setIf(InstanceCategories::ThirdPersonPlayerModel, lookupHash(RtxOptions::playerModelTextures(), textureHash));
    setIf(InstanceCategories::ThirdPersonPlayerBody, lookupHash(RtxOptions::playerModelBodyTextures(), textureHash));

    setIf(InstanceCategories::Terrain, lookupHash(RtxOptions::terrainTextures(), textureHash));
    setIf(InstanceCategories::Sky, lookupHash(RtxOptions::skyBoxTextures(), textureHash));

    setIf(InstanceCategories::ParticleEmitter, lookupHash(RtxOptions::particleEmitterTextures(), textureHash));
    setIf(InstanceCategories::HairCards, lookupHash(RtxOptions::hairCardTextures(), textureHash));
    setIf(InstanceCategories::HighlightHostile, lookupHash(RtxOptions::highlightHostileTextures(), textureHash));
    setIf(InstanceCategories::HeatSource, lookupHash(RtxOptions::heatSourceTextures(), textureHash));
    return result;
  }

  void DrawCallState::setupCategoriesForTexture(const CategoryFlags* pCachedTextureListCategories) {
    // TODO (REMIX-231): It would probably be much more efficient to use a map of texture hash to category flags, rather
    //                   than doing N lookups per texture hash for each category.
    //                   (rtx.textureCategoryCache: the D3D9 layer passes the cached result for this texture.)
    const XXH64_hash_t& textureHash = materialData.getColorTexture().getImageHash();

    // setCategory() only ever sets bits, so OR-ing the list result in is the same as setting each bit in turn.
    categories.set(pCachedTextureListCategories != nullptr ? *pCachedTextureListCategories
                                                           : computeTextureListCategories(textureHash));
    setCategory(InstanceCategories::IgnoreOpacityMicromap, isUsingRaytracedRenderTarget);
  }

  void DrawCallState::setupCategoriesForGeometry() {
    const XXH64_hash_t assetReplacementHash = getHash(RtxOptions::geometryAssetHashRule());
    setCategory(InstanceCategories::Sky, lookupHash(RtxOptions::skyBoxGeometries(), assetReplacementHash));
  }

  static std::optional<Vector3> makeCameraPosition(const Matrix4& worldToView,
                                                   bool zWrite,
                                                   bool alphaBlend,
                                                   bool hasSkinning) {
    if (hasSkinning) {
      return std::nullopt;
    }
    // particles
    if (!zWrite && alphaBlend) {
      return std::nullopt;
    }
    // identity matrix
    if (isIdentityExact(worldToView)) {
      return std::nullopt;
    }

#define USE_TRUE_CAMERA_POSITION_FOR_COMPARISON 0

#if USE_TRUE_CAMERA_POSITION_FOR_COMPARISON
    return (inverse(worldToView))[3].xyz();
#else
    // as we compare the cameras relatively and don't need precise camera position:
    // just return a position-like vector, to avoid calculating heavy matrix inverse operation
    return worldToView[3].xyz();
#endif
  }

  static bool areCamerasClose(const Vector3& a, const Vector3& b) {
    const float distanceThreshold = RtxOptions::skyAutoDetectUniqueCameraDistance();
    return lengthSqr(a - b) < distanceThreshold * distanceThreshold;
  }

  bool checkSkyAutoDetect(bool depthTestEnable,
                          const std::optional<Vector3>& newCameraPos,
                          uint32_t prevFrameSeenCamerasCount,
                          const std::vector<Vector3>& seenCameraPositions) {

    if (RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPositionAndDepthFlags &&
        RtxOptions::skyAutoDetect() != SkyAutoDetectMode::CameraPosition) {
      return false;
    }
    const bool withDepthFlags = (RtxOptions::skyAutoDetect() == SkyAutoDetectMode::CameraPositionAndDepthFlags);


    const bool searchingForSkyCamera             = (seenCameraPositions.size() == 0);
    const bool skyFoundAndSearchingForMainCamera = (seenCameraPositions.size() == 1);
    const bool skyAndMainCameraFound             = (seenCameraPositions.size() >= 2);

    if (skyAndMainCameraFound) {
      // assume that subsequent draw calls can not be sky
      return false;
    }

    if (searchingForSkyCamera) {
      if (withDepthFlags) {
        // no depth test: frame starts with a sky
        // depth test: frame starts with a world, not a sky
        return !depthTestEnable;
      }
      // assume the first camera to be sky
      return true;
    }

    {
      // corner case: if there was no sky camera at all, fallback, but this would also
      // involve a one-frame (preceding to the current one) being rasterized (like a flicker)
      if (prevFrameSeenCamerasCount < 2) {
        if (withDepthFlags) {
          // no depth test: sky
          // depth test: world
          return !depthTestEnable;
        }
        // assume no sky
        return false;
      }
    }

    if (skyFoundAndSearchingForMainCamera) {
      // if draw call doesn't have a camera position
      if (!newCameraPos) {
        // it can't contain main camera, so assume that it's still a sky
        return true;
      }

      // if same as the existing sky camera
      if (areCamerasClose(seenCameraPositions[0], *newCameraPos)) {
        // still sky
        return true;
      }

      // found a new unique camera, which should be a main camera
      return false;
    }

    assert(0);
    return false;
  }

  enum class SkyDetectionSource {
    None,
    Explicit,   // minZ, texHash, geoHash
    ExplicitDrawCallId, // dcIdThreshold (explicit; told apart only for rtx.d3d9Identity.log)
    AutoDetect  // checkSkyAutoDetect
  };

  SkyDetectionSource shouldBakeSky(const DrawCallState& drawCallState,
                     bool hasSkinning,
                     uint32_t prevFrameSeenCamerasCount,
                     std::vector<Vector3>& seenCameraPositions) {           
    const auto drawCallCameraPos =
      drawCallState.isDrawingToRaytracedRenderTarget
        ? std::optional<Vector3>{}
        : makeCameraPosition(
            drawCallState.getTransformData().worldToView,
            drawCallState.zWriteEnable,
            drawCallState.getMaterialData().blendMode.enableBlending,
            hasSkinning);

    auto l_addIfUnique = [&seenCameraPositions](const std::optional<Vector3>& newCameraPos) {
      if (!newCameraPos) {
        return;
      }
      for (const Vector3& seen : seenCameraPositions) {
        if (areCamerasClose(seen, *newCameraPos)) {
          return;
        }
      }
      seenCameraPositions.push_back(*newCameraPos);
    };
    l_addIfUnique(drawCallCameraPos);


    if (drawCallState.minZ >= RtxOptions::skyMinZThreshold()) {
      return SkyDetectionSource::Explicit;
    }

    // NOTE: we use color texture hash for sky detection, however the replacement is hashed with
    // the whole legacy material hash (which, as of 12/9/2022, equals to color texture hash). Adding a check just in case.
    assert(drawCallState.getMaterialData().getColorTexture().getImageHash() == drawCallState.getMaterialData().getHash() && "Texture or material hash method changed!");

    if (drawCallState.getMaterialData().usesTexture()) {
      if (lookupHash(RtxOptions::skyBoxTextures(), drawCallState.getMaterialData().getHash())) {
        return SkyDetectionSource::Explicit;
      }
    } else {
      if (drawCallState.drawCallID < RtxOptions::skyDrawcallIdThreshold()) {
        return SkyDetectionSource::ExplicitDrawCallId;
      }
    }

    // don't track camera positions for Raytraced Render Targets, as they are a different camera position from main view
    const static auto renderTargetCameraPositions = std::vector<Vector3>{};

    if (checkSkyAutoDetect(drawCallState.zEnable,
                           drawCallCameraPos,
                           prevFrameSeenCamerasCount,
                           drawCallState.isDrawingToRaytracedRenderTarget ? renderTargetCameraPositions : seenCameraPositions)) {
      return SkyDetectionSource::AutoDetect;
    }

    return SkyDetectionSource::None;
  }

  bool shouldBakeTerrain(const DrawCallState& drawCallState) {
    if (!TerrainBaker::needsTerrainBaking())
      return false;

    return lookupHash(RtxOptions::terrainTextures(), drawCallState.getMaterialData().getHash());
  }

  void DrawCallState::trackSkyCameraPosition(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                             bool drawingToRaytracedRenderTarget, std::vector<Vector3>& seenCameraPositions) {
    // As shouldBakeSky: no camera position for render-to-texture draws, then add it unless one close to it is known.
    if (drawingToRaytracedRenderTarget) {
      return;
    }
    const std::optional<Vector3> position = makeCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning);
    if (!position) {
      return;
    }
    for (const Vector3& seen : seenCameraPositions) {
      if (areCamerasClose(seen, *position)) {
        return;
      }
    }
    seenCameraPositions.push_back(*position);
  }

  bool DrawCallState::predictSkyAutoDetect(const Matrix4& worldToView, bool zWriteEnable, bool alphaBlend, bool hasSkinning,
                                           bool drawingToRaytracedRenderTarget, bool zEnable,
                                           uint32_t prevFrameSeenCamerasCount,
                                           const std::vector<Vector3>& seenCameraPositions) {
    // As shouldBakeSky: render-to-texture draws are judged against no cameras, others after their own camera is added.
    if (drawingToRaytracedRenderTarget) {
      static const std::vector<Vector3> kNoCameras;
      return checkSkyAutoDetect(zEnable, std::nullopt, prevFrameSeenCamerasCount, kNoCameras);
    }
    const std::optional<Vector3> position = makeCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning);
    if (seenCameraPositions.size() >= 2) {
      // Adding a camera cannot change the result once two are known.
      return checkSkyAutoDetect(zEnable, position, prevFrameSeenCamerasCount, seenCameraPositions);
    }
    std::vector<Vector3> withDraw(seenCameraPositions);
    trackSkyCameraPosition(worldToView, zWriteEnable, alphaBlend, hasSkinning, false, withDraw);
    return checkSkyAutoDetect(zEnable, position, prevFrameSeenCamerasCount, withDraw);
  }

  void DrawCallState::setupCategoriesForHeuristics(uint32_t prevFrameSeenCamerasCount,
                                                   std::vector<Vector3>& seenCameraPositions) {
    const SkyDetectionSource skySource = shouldBakeSky(*this,
                                                       futureSkinningData.valid(),
                                                       prevFrameSeenCamerasCount,
                                                       seenCameraPositions);
    setCategory(InstanceCategories::Sky, skySource != SkyDetectionSource::None);
    skyAutoDetected = (skySource == SkyDetectionSource::AutoDetect);
    skyFromDrawCallId = (skySource == SkyDetectionSource::ExplicitDrawCallId);

    setCategory(InstanceCategories::Terrain, shouldBakeTerrain(*this));
  }

  BlasEntry::BlasEntry(const DrawCallState& input_)
    : input(input_) {
    }

  void BlasEntry::unlinkInstance(RtInstance* instance) {
    if (m_linkedInstances.erase(instance) == 0) {
      ONCE(Logger::err("Tried to unlink an instance, which was never linked!"));
    }
  }

} // namespace dxvk
