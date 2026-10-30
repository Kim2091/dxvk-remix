/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include "d3d9_frontend.h"

#include "../../dxvk/rtx_render/rtx_context.h"

#ifdef REMIX_FRONTEND_D3D9_V2
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../dxvk/dxvk_device.h"
#include "../../dxvk/rtx_render/rtx_asset_replacer.h"
#include "../../dxvk/rtx_render/rtx_frontend_ui.h"
#include "../../dxvk/rtx_render/rtx_imgui.h"
#include "../../dxvk/rtx_render/rtx_instance_manager.h"
#include "../../dxvk/rtx_render/rtx_light_utils.h"
#include "../../dxvk/rtx_render/rtx_lights_data.h"
#include "../../dxvk/rtx_render/rtx_matrix_helpers.h"
#include "../../dxvk/rtx_render/rtx_option.h"
#include "../../dxvk/rtx_render/rtx_option_manager.h"
#include "../../dxvk/rtx_render/rtx_options.h"
#include "../../dxvk/rtx_render/rtx_retained_scene.h"
#include "../../dxvk/rtx_render/rtx_scene_interface.h"
#include "../../dxvk/rtx_render/rtx_scene_manager.h"
#include "../../dxvk/rtx_render/rtx_terrain_baker.h"
#include "../../dxvk/rtx_render/translator/rtx_translator_lifetime.h"
#include "../translator/material_translator.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"
#endif

namespace dxvk {
  namespace frontends {
    namespace d3d9 {

#ifdef REMIX_FRONTEND_D3D9_V2

      namespace {
        class D3D9V2Options {
        public:
          RTX_OPTION("rtx.d3d9v2", bool, log, false,
                     "D3D9 frontend diagnostic: describes every D3D9 draw sent to ray tracing by its resources and "
                     "their versions, and every 300 frames logs how many draws repeat last frame's exactly and what "
                     "rtx.d3d9v2.direct did. Counters only.");
          RTX_OPTION("rtx.d3d9v2", bool, direct, true,
                     "Hands objects the game draws unchanged for rtx.d3d9v2.handoverFrames frames to the scene, which "
                     "keeps them. Their later draws skip the render thread's per-draw work (hashing, tracking and the "
                     "preserve path), which saves CPU time. A moved object is moved in place, and any other change "
                     "hands it back to the standard path.\n"
                     "Not handed over by this option alone: vertex-captured, skinned, blended, user-pointer, particle, "
                     "sky, terrain, decal and other special draws, and draws with USD mesh replacements. The other "
                     "rtx.d3d9v2 options extend it to those. Needs rtx.scene.directRetained.");
          RTX_OPTION("rtx.d3d9v2", uint32_t, handoverFrames, 8,
                     "With rtx.d3d9v2.direct: how many frames an object must be drawn unchanged before it is handed to "
                     "the scene.");
          RTX_OPTION("rtx.d3d9v2", bool, materialTranslator, true,
                     "With rtx.d3d9v2.direct: objects handed to the scene carry their alpha state and fixed-function "
                     "surface state from the frontend, instead of the backend deriving them from the draw. Material "
                     "hashes, mod keys and the image are unchanged. With rtx.coreMaterial.selfCheck, "
                     "[MaterialTranslator] parity lines compare the two for every handover.");
          RTX_OPTION("rtx.d3d9v2", bool, directReplacements, true,
                     "With rtx.d3d9v2.direct: objects with a USD mesh replacement are handed to the scene too. The "
                     "replacement's meshes become the object's parts, each with its own transform and material, and its "
                     "lights and graphs are created, moved and removed with the object. A mod change hands every object "
                     "back.\n"
                     "Not handed over: replacements with point instancers, replacements with particle systems unless "
                     "rtx.d3d9v2.specialCategories is on, and everything rtx.d3d9v2.direct leaves out.");
          RTX_OPTION("rtx.d3d9v2", bool, explicitCamera, true,
                     "With rtx.d3d9v2.direct: the frontend classifies each draw's camera (main, view model, sky or "
                     "render-to-texture, with the same rules as the render thread) and sends each camera once per "
                     "frame, instead of sending a camera-only draw whenever a skipped draw's camera differs. With "
                     "rtx.fusedWorldViewMode it sends the corrected camera.");
          RTX_OPTION("rtx.d3d9v2", bool, earlyChangeTest, true,
                     "With rtx.d3d9v2.direct and rtx.d3d9v2.explicitCamera: recognises draws of objects the scene owns "
                     "from their raw D3D9 state (buffers, render, stage and sampler states, textures, material, "
                     "transforms, viewport and clip planes) before any preparation. Unchanged or only moved draws then "
                     "skip the vertex and index copies, texture processing and hashing on the game thread, which saves "
                     "CPU time. Other draws are prepared as usual. See rtx.d3d9v2.earlyChangeTestSelfCheckFrames to "
                     "verify it.");
          RTX_OPTION("rtx.d3d9v2", bool, firstDrawHandover, true,
                     "With rtx.d3d9v2.direct and rtx.d3d9v2.explicitCamera: hands an eligible object to the scene at "
                     "its first draw instead of after rtx.d3d9v2.handoverFrames repeats. A draw the scene does not "
                     "accept is drawn by the standard path. Objects that change within 8 frames of their handover wait "
                     "longer before the next attempt.");
          RTX_OPTION("rtx.d3d9v2", bool, dynamicGeometry, true,
                     "With rtx.d3d9v2.firstDrawHandover: objects the scene owns are updated in place when their draw "
                     "changes (rewritten or dynamic buffers, fixed-function skinning, user-pointer draws), instead of "
                     "being handed back to the standard path, and such draws are handed over too. Vertex-captured draws "
                     "also need rtx.d3d9v2.capturedGeometry, and blended, decal and particle draws "
                     "rtx.d3d9v2.unordered.");
          RTX_OPTION("rtx.d3d9v2", bool, capturedGeometry, true,
                     "With rtx.d3d9v2.dynamicGeometry: vertex-captured draws (vertex shader draws and GPU skinning) are "
                     "handed to the scene too, and updated in place from their capture when they change. With "
                     "rtx.d3d9v2.earlyChangeTest, a captured draw whose shader constants, camera and inputs did not "
                     "change skips its preparation and its capture. The geometry is still refit on frames where it "
                     "changes.");
          RTX_OPTION("rtx.d3d9v2", bool, unordered, true,
                     "With rtx.d3d9v2.dynamicGeometry and rtx.scene.directUnordered: blended draws, decals, particles "
                     "and beams are handed to the scene too, which refreshes their billboards. Decals keep the game's "
                     "draw order. Particle emitters (particle systems) stay on the standard path.");
          RTX_OPTION("rtx.d3d9v2", bool, frameEvents, true,
                     "With rtx.d3d9v2.direct: finds the frame's fog from the frontend's draw order, including skipped "
                     "draws, at the end of the frame instead of per submitted draw, so the first unreplaced fog of the "
                     "frame is used whichever path drew it. Takes effect from the next frame.");
          RTX_OPTION("rtx.d3d9v2", bool, lights, true,
                     "With rtx.d3d9v2.direct: the frontend translates the game's fixed-function lights (SetLight and "
                     "LightEnable). Lights with a USD light replacement become lights the frontend owns, created, "
                     "updated and aged out like replacement instances, anti-culling included. Other lights go to the "
                     "usual game-light handling.");
          RTX_OPTION("rtx.d3d9v2", bool, skyService, true,
                     "With rtx.d3d9v2.specialCategories: sky-camera draws are owned by the scene, and their "
                     "rasterization into the sky matte and probe runs every frame without the standard path. 3D-skybox "
                     "draws reprojected into the main camera (rtx.skyReprojectToMainCameraSpace) need "
                     "rtx.d3d9v2.skybox3d.");
          RTX_OPTION("rtx.d3d9v2", bool, terrain, true,
                     "With rtx.d3d9v2.specialCategories: terrain draws (the Terrain category, with terrain baking on) "
                     "are owned by the scene with the baked terrain material. The bake into the terrain cascades still "
                     "runs every frame, and a cascade change updates the material.");
          RTX_OPTION("rtx.d3d9v2", bool, skybox3d, true,
                     "With rtx.d3d9v2.skyService: 3D-skybox draws (sky-camera draws reprojected into the main camera, "
                     "rtx.skyReprojectToMainCameraSpace) are owned by the scene too, and the frontend reprojects them "
                     "every frame from the sky camera's transforms.");
          RTX_OPTION("rtx.d3d9v2", bool, renderTargets, true,
                     "With rtx.d3d9v2.specialCategories: draws to or sampling a ray traced render target "
                     "(rtx.raytracedRenderTargetTextures) are handed to the scene too, with the render-to-texture "
                     "camera updated every frame.");
          RTX_OPTION("rtx.d3d9v2", bool, unknownCamera, true,
                     "With rtx.d3d9v2.firstDrawHandover: draws whose camera could not be classified are handed over as "
                     "main-camera draws, as the standard path treats them, unless rtx.skipObjectsWithUnknownCamera is "
                     "on.");
          RTX_OPTION("rtx.d3d9v2", bool, specialCategories, true,
                     "With rtx.d3d9v2.dynamicGeometry and rtx.scene.directServices: special draws are handed to the "
                     "scene too, which does their per-frame work itself: view-model and portal cameras, the player "
                     "model and body, ray portals, light-converted materials, particle emitters, animated water, hair "
                     "cards, world UI and world matte.\n"
                     "Sky, the 3D skybox, terrain, ray traced render targets and draws with an unknown camera have "
                     "their own options (rtx.d3d9v2.skyService, skybox3d, terrain, renderTargets and unknownCamera).");
          RTX_OPTION("rtx.d3d9v2", uint32_t, earlyChangeTestSelfCheckFrames, 0,
                     "Diagnostic for rtx.d3d9v2.earlyChangeTest: every this many frames, the draws it would skip are "
                     "prepared in full and compared with what was stored. Differences are logged as [D3D9v2] "
                     "preKeyMismatch. 0 = off.");
        };

        // rtx.d3d9v2.explicitCamera in effect (needs rtx.d3d9v2.direct).
        bool explicitCameraActive() {
          return D3D9V2Options::direct() && D3D9V2Options::explicitCamera();
        }

        // rtx.d3d9v2.firstDrawHandover in effect (R4 decides eligibility from R2's camera type, so it needs R2).
        bool firstDrawHandoverActive() {
          return explicitCameraActive() && D3D9V2Options::firstDrawHandover();
        }

        // rtx.d3d9v2.dynamicGeometry in effect (R5 builds on R4's synchronous handover).
        bool dynamicGeometryActive() {
          return firstDrawHandoverActive() && D3D9V2Options::dynamicGeometry();
        }

        // rtx.d3d9v2.capturedGeometry in effect (R6 updates captured objects the R5 way).
        bool capturedGeometryActive() {
          return dynamicGeometryActive() && D3D9V2Options::capturedGeometry();
        }

        // rtx.d3d9v2.unordered in effect (R7: blended / decal / particle draws, updated in place the R5 way; the scene's
        // unordered handling is rtx.scene.directUnordered).
        bool unorderedActive() {
          return dynamicGeometryActive() && D3D9V2Options::unordered() && scene::Options::directUnordered();
        }

        // rtx.d3d9v2.specialCategories in effect (R8: the scene's per-frame services, rtx.scene.directServices, replace the
        // old path's per-draw work).
        bool specialCategoriesActive() {
          return dynamicGeometryActive() && D3D9V2Options::specialCategories() && scene::Options::directServices();
        }

        // R8: camera types the scene takes with rtx.d3d9v2.specialCategories (stamped per frame, kServiceStamp). Sky,
        // RenderToTexture and Unknown need their R8b options.
        bool serviceCamera(uint32_t cameraType) {
          return cameraType == CameraType::ViewModel || cameraType == CameraType::Portal0 || cameraType == CameraType::Portal1;
        }

        // R8b options in effect.
        bool skyServiceActive() {
          return specialCategoriesActive() && D3D9V2Options::skyService();
        }
        bool renderTargetsActive() {
          return specialCategoriesActive() && D3D9V2Options::renderTargets();
        }
        bool skybox3dActive() {
          return skyServiceActive() && D3D9V2Options::skybox3d();
        }
        bool terrainActive() {
          return specialCategoriesActive() && D3D9V2Options::terrain();
        }
        bool unknownCameraActive() {
          return firstDrawHandoverActive() && D3D9V2Options::unknownCamera() && !RtxOptions::skipObjectsWithUnknownCamera();
        }

        // R8 / R8b: the camera types a handover accepts.
        bool acceptedCamera(uint32_t cameraType) {
          if (cameraType == CameraType::Main) {
            return true;
          }
          if (serviceCamera(cameraType)) {
            return specialCategoriesActive();
          }
          if (cameraType == CameraType::Sky) {
            return skyServiceActive();
          }
          if (cameraType == CameraType::RenderToTexture) {
            return renderTargetsActive();
          }
          if (cameraType == CameraType::Unknown) {
            return unknownCameraActive();
          }
          return false;
        }

        // A terrain draw the terrain bake service takes (rtx.d3d9v2.terrain with baking on): replayed every frame
        // (RasterOnly).
        bool bakedTerrainCategory(uint32_t categories) {
          return terrainActive() && TerrainBaker::needsTerrainBaking() && CategoryFlags(categories).test(InstanceCategories::Terrain);
        }

        // rtx.d3d9v2.frameEvents in effect (R10).
        bool frameEventsActive() {
          return D3D9V2Options::direct() && D3D9V2Options::frameEvents();
        }

        FogState toFogState(const FogInputs& in) {
          FogState fog;
          fog.mode = in.mode;
          fog.color = Vector3(in.color[0], in.color[1], in.color[2]);
          fog.scale = in.scale;
          fog.end = in.end;
          fog.density = in.density;
          return fog;
        }

        // rtx.d3d9v2.lights in effect (R9).
        bool lightsActive() {
          return D3D9V2Options::direct() && D3D9V2Options::lights();
        }

        // rtx.d3d9v2.earlyChangeTest in effect (R1 sends cameras the R2 way, so it needs R2).
        bool earlyChangeTestActive() {
          return explicitCameraActive() && D3D9V2Options::earlyChangeTest();
        }

        // CameraManager::processCameraData's classification (rtx_camera_manager.cpp), on the app thread. "The Main camera
        // was updated this frame" is the first Main classification of this D3D9 frame; draw order is the same on both
        // threads, so the result is the same.
        class CameraClassifier {
        public:
          // updatesCamera: false where processCameraData returns before updating a camera (rejected matrices: the type is
          // then Unknown, or Sky for Sky-category draws, and no camera may be sent).
          CameraType::Enum classify(const CameraInputs& c, bool& updatesCamera) {
            updatesCamera = false;
            const CameraType::Enum rejected = c.sky ? CameraType::Sky : CameraType::Unknown;
            if (isIdentityExact(c.viewToProjection)) {
              return rejected;
            }
            if (RtxOptions::fusedWorldViewMode() == FusedWorldViewMode::None &&
                c.objectToView == c.objectToWorld && !isIdentityExact(c.objectToView)) {
              return rejected;
            }
            if (!m_hasLast || std::memcmp(&m_lastProjection, &c.viewToProjection, sizeof(Matrix4)) != 0) {
              decomposeProjection(c.viewToProjection, m_lastParams);
              m_lastProjection = c.viewToProjection;
              m_hasLast = true;
            }
            if (std::abs(m_lastParams.shearX) > 0.01f || !(m_lastParams.fov >= kFovToleranceRadians)) {
              return rejected;
            }
            updatesCamera = true;
            const CameraType::Enum type = classifyValid(c);
            m_views[type] = c.worldToView;
            m_viewFrames[type] = m_frame + 1;
            return type;
          }

          // The camera RtxContext::prepareGeometryForRTImpl rebuilds a pre-combined draw's world matrix from: the camera
          // of the draw's type when it was set this frame or the last, otherwise Main.
          // Returns null when neither was ever set.
          const Matrix4* referenceView(CameraType::Enum type) const {
            // m_viewFrames holds the frame plus one, so the camera was set this frame or the last when it is at least
            // m_frame.
            if (type >= CameraType::Unknown || m_viewFrames[type] == 0 || m_viewFrames[type] < m_frame) {
              type = CameraType::Main;
            }
            return m_viewFrames[type] != 0 ? &m_views[type] : nullptr;
          }

          // True the first time a camera type is seen this frame.
          bool firstOfFrame(CameraType::Enum type) {
            const uint32_t bit = 1u << uint32_t(type);
            const bool first = (m_fed & bit) == 0;
            m_fed |= bit;
            return first;
          }

          void endFrame() {
            m_mainSet = false;
            m_fed = 0;
            ++m_frame;
          }

        private:
          CameraType::Enum classifyValid(const CameraInputs& c) {
            if (c.drawingToRaytracedRenderTarget) {
              return CameraType::RenderToTexture;
            }
            if (c.sky) {
              return CameraType::Sky;
            }
            if (RtxOptions::ViewModel::enable()) {
              if (c.maxZ <= RtxOptions::ViewModel::maxZThreshold()) {
                return CameraType::ViewModel;
              }
              if (m_mainSet && !(std::abs(m_lastParams.fov - m_mainFov) < kFovToleranceRadians)) {
                return CameraType::ViewModel;
              }
            }
            if (!m_mainSet) {
              m_mainSet = true;
              m_mainFov = m_lastParams.fov;
            }
            return CameraType::Main;
          }

          static constexpr float kFovToleranceRadians = 0.001f;  // rtx_camera_manager.cpp
          Matrix4 m_lastProjection;
          DecomposeProjectionParams m_lastParams;
          bool m_hasLast = false;
          bool m_mainSet = false;
          float m_mainFov = 0.f;
          uint32_t m_fed = 0;
          // The last camera set per type and the frame plus one it was set in (0 = never), for referenceView.
          Matrix4 m_views[CameraType::Count];
          uint32_t m_viewFrames[CameraType::Count] = {};
          uint32_t m_frame = 0;
        };

        constexpr uint32_t kLogIntervalFrames = 300;
        // R4: an object handed back as changed within this many frames of its first-draw handover backs off.
        constexpr uint32_t kChurnFrames = 8;

        // Scene handles of D3D9 objects (bit 62 is the transient lane's, rtx_scene_transient.h).
        constexpr uint64_t kHandleNamespace = uint64_t(1) << 61;
        // Geometry-sharing keys (ExternalDrawState::mesh) of D3D9 objects: identical geometry shares one BLAS entry.
        constexpr uint64_t kMeshNamespace = uint64_t(1) << 60;

        // A scene-owned object's object-picking value, unique per handle. Old-path draws use their draw-call ID, which
        // counts from 0 each frame, so the top bit keeps the two apart.
        ObjectPickingValue pickingValue(uint64_t handle) {
          return ObjectPickingValue(0x80000000u | (uint32_t(handle) & 0x7FFFFFFFu));
        }

        void drawPanel() {
          RemixGui::Checkbox("D3D9 v2: Stable Objects Owned by the Scene", &D3D9V2Options::directObject());
          if (D3D9V2Options::direct() && !scene::Options::directRetained()) {
            ImGui::TextUnformatted("  (needs \"Scene: Direct Retained Instances\")");
          }
        }

        // The categories rtx.d3d9v2.direct never hands over (the old preserve-path exclusions plus the direct lane's own).
        // unordered (R7, rtx.d3d9v2.unordered): decals, particles and beams are the scene's unordered instances, not special.
        // services (R8, rtx.d3d9v2.specialCategories): emitters, player model, world UI / matte, water and hair are the
        // scene's services. Sky and terrain stay special (old path) in v1.
        // Terrain is special only while terrain baking is on: otherwise the old path draws it as a plain draw.
        bool specialCategory(const CategoryFlags& categories, bool unordered, bool services) {
          if ((categories.test(InstanceCategories::Terrain) && TerrainBaker::needsTerrainBaking() && !terrainActive()) ||  // R8b
              (categories.test(InstanceCategories::Sky) && !skyServiceActive())) {  // R8b
            return true;
          }
          if (!services && categories.any(InstanceCategories::ParticleEmitter,
                                          InstanceCategories::ThirdPersonPlayerModel, InstanceCategories::ThirdPersonPlayerBody,
                                          InstanceCategories::WorldUI, InstanceCategories::WorldMatte,
                                          InstanceCategories::AnimatedWater, InstanceCategories::HairCards)) {
            return true;
          }
          return !unordered && categories.any(InstanceCategories::Particle, InstanceCategories::Beam, DECAL_CATEGORY_FLAGS);
        }

        // The D3D9 draw's fixed-function state as the MaterialTranslator's plain record.
        translator::FixedFunctionState fixedFunctionState(const DrawCallState& state) {
          const LegacyMaterialData& legacy = state.getMaterialData();
          const DrawCallTransforms& transforms = state.getTransformData();
          translator::FixedFunctionState ff;
          ff.textureColorArg1Source = legacy.textureColorArg1Source;
          ff.textureColorArg2Source = legacy.textureColorArg2Source;
          ff.textureColorOperation = legacy.textureColorOperation;
          ff.textureAlphaArg1Source = legacy.textureAlphaArg1Source;
          ff.textureAlphaArg2Source = legacy.textureAlphaArg2Source;
          ff.textureAlphaOperation = legacy.textureAlphaOperation;
          ff.tFactor = legacy.tFactor;
          ff.textureFactorBlend = legacy.isTextureFactorBlend;
          ff.vertexColorBakedLighting = legacy.isVertexColorBakedLighting;
          ff.blendMode = legacy.blendMode;
          ff.alphaTestCompareOp = legacy.alphaTestCompareOp;
          ff.alphaTestReferenceValue = legacy.alphaTestReferenceValue;
          ff.textureTransform = transforms.textureTransform;
          ff.texgenMode = transforms.texgenMode;
          ff.clipPlaneEnabled = transforms.enableClipPlane;
          ff.clipPlane = transforms.clipPlane;
          ff.categories = state.getCategoryFlags();
          return ff;
        }

        bool sameMatrix(const Matrix4& a, const Matrix4& b) {
          return std::memcmp(&a, &b, sizeof(Matrix4)) == 0;
        }
      }

      // ---- shared between the two halves -------------------------------------------------------------------------

      // CS -> app: what became of a handed-over object.
      struct Channel {
        enum class Kind : uint8_t {
          Accepted,  // the scene owns it (Pending -> Direct)
          Rejected,  // it cannot be direct; back to the old path (with a back-off)
          Released,  // the CS thread wants it back (aged out, a reset, rebuilt and rejected): the app thread sends a
                     // destroy and routes the draw to the old path. The scene keeps the entry until that destroy, so
                     // nothing disappears while the app thread still skips the draw.
          Retry,     // it cannot be built yet (a 3D-skybox draw before the first main camera): handed back without a
                     // back-off, so the next draw is handed over again.
        };
        struct Result {
          uint64_t handle;
          Kind kind;
        };
        std::mutex mutex;
        std::vector<Result> results;
        // An option that invalidates draw-call translation changed: every Direct object is prepared and described again
        // at its next draw.
        bool reprepare = false;

        void push(uint64_t handle, Kind kind) {
          std::lock_guard<std::mutex> lock(mutex);
          results.push_back({ handle, kind });
        }

        void requestReprepare() {
          std::lock_guard<std::mutex> lock(mutex);
          reprepare = true;
        }
      };

      struct FrameBatch {
        struct Move {
          uint64_t handle;
          Matrix4 objectToWorld;
        };
        std::vector<uint64_t> seen;        // direct objects drawn this frame (unchanged or moved): not on the old path
        std::vector<uint64_t> pendingSeen; // pending objects drawn unchanged this frame (still on the old path)
        std::vector<Move> moves;
        std::vector<uint64_t> destroys;    // direct or pending objects whose draw changed
        // R7: direct decals whose rank in this frame's draw order changed, or that were (re)built (RetainedScene::setSortOrder).
        struct SortOrder {
          uint64_t handle;
          uint8_t order;
        };
        std::vector<SortOrder> sortOrders;
        // R10 (rtx.d3d9v2.frameEvents): the frame's fog states in draw order (each once, D3DFOG_NONE left out).
        std::vector<FogState> fogOrder;
        bool frameEvents = false;
        bool releaseAll = false;           // rtx.d3d9v2.direct switched off

        bool empty() const {
          return seen.empty() && pendingSeen.empty() && moves.empty() && destroys.empty() && sortOrders.empty() &&
                 fogOrder.empty() && !releaseAll;
        }
      };

      // ---- CS thread -------------------------------------------------------------------------------------------

      struct D3D9Frontend::Impl {
        // V4: what one prim index of a mesh replacement is in the scene's copy: a submesh, one of the object's lights
        // (index into Object::lights), a graph, or nothing.
        struct PrimSlot {
          PrimInstance::Type type = PrimInstance::Type::None;
          uint32_t index = 0;
          bool operator==(const PrimSlot& o) const { return type == o.type && index == o.index; }
        };

        struct Object {
          XXH64_hash_t assetHash = 0;   // mesh hash for the hash checker / hash dumps (trackMeshHash)
          XXH64_hash_t legacyHash0 = 0; // the legacy asset hashes submitDrawState also tracks (0 = rule off)
          XXH64_hash_t legacyHash1 = 0;
          XXH64_hash_t fogHash = 0;     // 0 = no fog
          XXH64_hash_t described = 0;   // descriptionKey of the last description
          AxisAlignedBoundingBox bounds;
          Matrix4 objectToWorld;
          uint32_t categories = 0;
          uint32_t frameCreated = 0;
          uint32_t frameLastSeen = 0;
          uint32_t frameMoved = ~0u;
          uint32_t frameMergedPass = 0; // A4: the last frame mergePass ran for the object
          bool pending = true;          // described, not yet built
          bool releasing = false;       // Released sent, waiting for the app thread's destroy
          // V4 (rtx.d3d9v2.directReplacements): an expanded USD mesh replacement and its lights. The lights are the
          // scene's from the first frame the app thread skips the draw (the old path's replacement instance, which had
          // them until then, is collected in that frame: ReplacementInstance::handedToScene).
          bool replaced = false;
          AxisAlignedBoundingBox lightBounds;   // object space, as ReplacementInstance::lightBoundingBox
          std::vector<RtLight> lightTemplates;  // the replacement's lights in object space
          std::vector<RtLight*> lights;         // externally tracked lights, one per template once created
          bool lightsCreated = false;
          // V4: the replacement's prims (what each prim index is in the entry) and its graphs. Each graph instance
          // (GraphManager id, 0 = none) is made with the lights, at the slot of its template.
          std::vector<PrimSlot> primSlots;
          std::vector<std::shared_ptr<const RtGraphState>> graphTemplates;
          std::vector<uint64_t> graphs;
          bool skinned = false;         // R5: never kept unseen (LifetimePolicy), as the tracker does for skinned instances
          bool nonMainCamera = false;   // R8: never kept unseen either (the anti-culling frustum is the main camera's)
          bool captured = false;        // R6: its geometry is a vertex capture (rtx.d3d9v2.log lines)
          // R8b (rtx.d3d9v2.skybox3d): a 3D-skybox object, described as a Main-camera entry; skyTransforms are its latest
          // sky-camera transforms, reprojected at every flush.
          bool skybox3d = false;
          bool terrain = false;         // R8b (rtx.d3d9v2.terrain): baked every frame by RasterOnly, material from the baker
          DrawCallTransforms skyTransforms;
          uint32_t reprepareEpoch = 0;  // the epoch of its description: an older one is described again even if unchanged
        };
        std::unordered_map<uint64_t, Object> objects;
        // A4: objects whose scene entry has a merged pass (RetainedScene::setMergedPass).
        std::unordered_set<uint64_t> mergedPasses;
        // One draw state per fog state of a direct object: fog discovery (SceneManager::discoverFogState) needs one.
        // R10: also of old-path draws (keepFogDraw); a fog replacement material is created from it.
        std::unordered_map<XXH64_hash_t, DrawCallState> fogDraws;

        // R10: a draw of each fog state, for the fog replacement material at the flush. Bounded: a game whose fog changes
        // every frame refills it from the frame's draws.
        void keepFogDraw(const DrawCallState& state) {
          if (state.getFogState().mode == D3DFOG_NONE) {
            return;
          }
          const XXH64_hash_t fogHash = state.getFogState().getHash();
          if (fogDraws.find(fogHash) != fogDraws.end()) {
            return;
          }
          if (makeRoomForFogDraw()) {
            fogDraws.emplace(fogHash, state);
          }
        }

        // At the bound, the fog draws no scene-owned object uses go. Draws of scene-owned objects are skipped, so nothing
        // would keep their fog again, while old-path draws keep theirs at their next draw.
        // Returns false when every kept draw is still in use.
        bool makeRoomForFogDraw() {
          constexpr size_t kMaxFogDraws = 256;
          if (fogDraws.size() < kMaxFogDraws) {
            return true;
          }
          std::unordered_set<XXH64_hash_t> used;
          for (const auto& [handle, object] : objects) {
            if (object.fogHash != 0) {
              used.insert(object.fogHash);
            }
          }
          for (auto it = fogDraws.begin(); it != fogDraws.end();) {
            it = used.count(it->first) != 0 ? std::next(it) : fogDraws.erase(it);
          }
          return fogDraws.size() < kMaxFogDraws;
        }
        std::shared_ptr<Channel> channel = std::make_shared<Channel>();
        // Counts the re-preparations requested from the app thread (Channel::reprepare).
        uint32_t reprepareEpoch = 0;
        uint64_t clearEpoch = ~0ull;
        uint64_t assetEpoch = ~0ull;
        bool v4WasOn = false;  // rtx.d3d9v2.directReplacements at the last flush

        // R9 (rtx.d3d9v2.lights): D3D9 lights with a USD light replacement, keyed as SceneManager::addLight keyed their
        // replacement instance (the light's hash and position). The entry owns the replacement's lights.
        struct LightEntry {
          std::shared_ptr<const ReplacementBucket> bucket;
          std::vector<RtLight*> lights;     // one per light replacement, in bucket order (null = not created)
          Matrix4 transform;                // LightUtils::getLightTransform of the D3D9 light
          AxisAlignedBoundingBox lightBounds;
          uint32_t frameCreated = 0;
          uint32_t frameLastSeen = 0;
        };
        std::unordered_map<XXH64_hash_t, LightEntry> lightTable;
        uint32_t lightCollectFrame = ~0u;
        uint64_t lightClearEpoch = ~0ull;
        struct LightTotals {
          uint64_t game = 0;       // emitted lights handed to LightManager::addGameLight
          uint64_t replaced = 0;   // emitted lights with a replacement (entry updated)
          uint64_t created = 0;    // entries created
          uint64_t collected = 0;  // entries aged out
        } lightTotals;

        static void destroyLightEntry(LightEntry& entry) {
          for (RtLight* light : entry.lights) {
            if (light != nullptr) {
              light->markForGarbageCollection();
            }
          }
          entry.lights.clear();
        }

        // R9: SceneManager::addLight's work for one emission of the enabled lights, with the replacement lights owned by
        // lightTable instead of tracker replacement instances.
        void translateLights(RtxContext* ctx, const D3DLIGHT9* lights, uint32_t count) {
          SceneManager& sm = ctx->getSceneManager();
          LightManager& lightManager = sm.getLightManager();
          const uint32_t frame = sm.device()->getCurrentFrameId();
          lightManager.beginGameLightList();
          for (uint32_t l = 0; l < count; ++l) {
            const D3DLIGHT9& light = lights[l];
            const std::optional<LightData> lightData = LightData::tryCreate(light);
            if (!lightData.has_value()) {
              continue;  // malformed, as the old path skips it
            }
            RtLight rtLight = lightData->toRtLight();
            const XXH64_hash_t assetKey = LightUtils::legacyAssetKey(light);
            if (assetKey != kEmptyHash) {
              rtLight.setAssetKey(assetKey);
            }
            const XXH64_hash_t assetHash = rtLight.getInitialHash();
            const Vector3 position = rtLight.getPosition();
            // A light with an asset key keeps its entry while it moves.
            const XXH64_hash_t key = assetKey != kEmptyHash ? assetKey : XXH64(&position, sizeof(Vector3), assetHash);
            const std::shared_ptr<const ReplacementBucket> bucket = sm.getAssetReplacer()->getReplacementsForLight(assetHash);
            if (bucket == nullptr) {
              // A light that had a replacement (replacement lights toggled off): its replacement lights go first.
              const auto stale = lightTable.find(key);
              if (stale != lightTable.end()) {
                destroyLightEntry(stale->second);
                lightTable.erase(stale);
              }
              lightManager.addGameLight(light.Type, rtLight, LightUtils::legacyIdentity(light));
              lightTotals.game++;
              continue;
            }

            const Matrix4 transform = LightUtils::getLightTransform(light);
            auto [found, inserted] = lightTable.try_emplace(key);
            LightEntry& entry = found->second;
            if (inserted) {
              entry.frameCreated = frame;
              lightTotals.created++;
            }
            if (entry.bucket != bucket) {
              // New or changed bucket (unchanged buckets keep their address): its lights are created again.
              destroyLightEntry(entry);
              entry.bucket = bucket;
              entry.lightBounds = AxisAlignedBoundingBox();
            }
            const bool boundsNeeded = !entry.lightBounds.isValid();
            AxisAlignedBoundingBox litBounds;
            size_t slot = 0;
            for (const AssetReplacement& replacement : bucket->replacements) {
              if (replacement.type != AssetReplacement::eLight || !replacement.lightData.has_value()) {
                continue;  // meshes under lights are not supported (the old path asserts)
              }
              LightData replacementLight = replacement.lightData.value();
              // Merged before the bounds: some entries take type, position and radius from the D3D9 light.
              replacementLight.merge(light);
              RtLight rtReplacement = replacementLight.toRtLight();
              if (boundsNeeded) {
                const Vector3 p = rtReplacement.getPosition();
                const float radius = rtReplacement.getType() == RtLightType::Sphere ? rtReplacement.getSphereLight().getRadius() : 0.f;
                for (uint32_t j = 0; j < 3; ++j) {
                  litBounds.minPos[j] = std::min(litBounds.minPos[j], p[j] - radius);
                  litBounds.maxPos[j] = std::max(litBounds.maxPos[j], p[j] + radius);
                }
              }
              if (replacementLight.relativeTransform()) {
                rtReplacement.applyTransform(transform);
              }
              if (slot < entry.lights.size() && entry.lights[slot] != nullptr) {
                lightManager.updateExternallyTrackedLight(entry.lights[slot], rtReplacement);
              } else {
                RtLight* created = lightManager.createExternallyTrackedLight(rtReplacement);
                if (slot < entry.lights.size()) {
                  entry.lights[slot] = created;
                } else {
                  entry.lights.push_back(created);
                }
              }
              ++slot;
            }
            if (boundsNeeded && litBounds.isValid()) {
              entry.lightBounds = litBounds;
            }
            entry.transform = transform;
            entry.frameLastSeen = frame;
            lightTotals.replaced++;
          }
        }

        // R9: once per frame, the light entries age out as the tracker's light replacement instances did (LifetimePolicy
        // with the lights' bounds, anti-culling included); all of them go on a scene clear or when the option is off.
        void collectLights(RtxContext* ctx) {
          SceneManager& sm = ctx->getSceneManager();
          const uint32_t frame = sm.device()->getCurrentFrameId();
          if (lightCollectFrame == frame) {
            return;
          }
          lightCollectFrame = frame;
          const uint64_t clears = sm.clearEpoch();
          const bool cleared = lightClearEpoch != ~0ull && clears != lightClearEpoch;
          lightClearEpoch = clears;
          if (lightTable.empty()) {
            return;
          }
          if (cleared || !lightsActive()) {
            for (auto& [key, entry] : lightTable) {
              destroyLightEntry(entry);
            }
            lightTotals.collected += lightTable.size();
            lightTable.clear();
            return;
          }
          // As the old path: anti-culling only when the game sets a view matrix (SceneManager::prepareSceneData).
          translator::LifetimeFrame lifetime = translator::LifetimeFrame::read(
            sm.getCamera(), sm.isAntiCullingSupported(),
            objects.size() + lightTable.size() >= RtxOptions::AntiCulling::Object::numObjectsToKeep());
          lifetime.currentFrame = frame;
          lifetime.keptCutFrame = sm.m_lastKeptCutFrame;
          for (auto it = lightTable.begin(); it != lightTable.end();) {
            LightEntry& entry = it->second;
            if (entry.frameLastSeen == frame) {
              ++it;
              continue;
            }
            translator::UnseenObject unseen;
            unseen.frameLastSeen = entry.frameLastSeen;
            unseen.frameCreated = entry.frameCreated;
            unseen.categories = 0;
            unseen.wasMovingWhenLastSeen = false;
            unseen.skinned = false;
            unseen.geometryBounds = nullptr;
            unseen.lightBounds = &entry.lightBounds;
            unseen.objectToWorld = &entry.transform;
            if (!translator::LifetimePolicy::isStale(unseen, lifetime) || translator::LifetimePolicy::keepUnseen(unseen, lifetime)) {
              ++it;
              continue;
            }
            destroyLightEntry(entry);
            lightTotals.collected++;
            it = lightTable.erase(it);
          }
        }

        // Why the CS thread asked for an object back ([D3D9v2] direct "released (...)").
        enum class ReleaseReason : uint8_t { Aged, Rebuilt, Reset };
        // Which event handed everything back ([D3D9v2] direct "resets (...)").
        enum class ResetReason : uint8_t { Clear, Mods, Off };

        struct Totals {
          uint64_t accepted = 0;
          uint64_t rejected = 0;
          uint64_t released = 0;
          uint64_t resets = 0;
          uint64_t cameraOnly = 0;
          uint64_t handovers = 0;
          uint64_t releasedBy[3] = {};   // ReleaseReason
          uint64_t resetsBy[3] = {};     // ResetReason
          uint64_t reprepares = 0;       // re-preparations of every Direct object after a translation-invalidating option
          uint64_t replacedAccepted = 0; // V4
          uint64_t lightsCreated = 0;    // V4
          uint64_t updated = 0;          // R5: objects described and built again in place
          uint64_t unchangedUpdates = 0; // R5: updates whose description matched the scene's, kept as they were
          uint64_t mergedPasses = 0;     // A4: merged passes described to the scene
          uint64_t cameraChecks = 0;     // R2: old-path draws whose camera type was compared
          uint64_t cameraMismatches = 0;
          uint64_t fallbackDraws = 0;    // R4: rejected first-draw handovers replayed through the old path
          uint64_t rasterMisses = 0;     // R8b: RasterOnly draws that were not rasterized sky draws (expected 0)
          uint64_t skyboxMoves = 0;      // R8b: 3D-skybox reprojections applied (rtx.d3d9v2.skybox3d)
          uint64_t skyboxRetries = 0;    // 3D-skybox handovers tried again because no main camera was known yet
          uint64_t terrainBakes = 0;     // R8b: terrain bakes of scene-owned objects (rtx.d3d9v2.terrain)
          uint64_t terrainMisses = 0;    // ... that baked nothing
          uint64_t terrainFallbacks = 0; // R8b: draws of rejected terrain objects drawn by the old path (rejectedTerrainDraw)
          uint64_t terrainRematerials = 0;  // ... that re-described the material after a cascade change
        } totals;
        std::string firstCameraMismatch = "none";

        // rtx.d3d9v2.log: per-frame lines of the objects built and aged out in a frame (at most kFrameLogLines of each per
        // log interval), and of World UI objects (in-world screens).
        static constexpr uint32_t kFrameLogLines = 24;
        std::vector<uint64_t> frameBuilt;  // first-draw handovers built this frame
        uint32_t builtLogLines = 0;
        uint32_t agedLogLines = 0;
        uint32_t worldUILogLines = 0;

        static std::string describeObject(uint64_t handle, const Object& object, uint32_t frame) {
          const Vector3 p = object.objectToWorld[3].xyz();
          return str::format(" 0x", std::hex, handle, " cat=0x", object.categories, " asset=0x", object.assetHash, std::dec,
                             " pos=(", p.x, ", ", p.y, ", ", p.z, ") age=", frame - object.frameCreated,
                             object.captured ? " captured" : "", object.skinned ? " skinned" : "",
                             object.nonMainCamera ? " nonMain" : "");
        }

        // One line for a frame that built or aged out objects: how many, how many were World UI or vertex-captured, and
        // up to four of them.
        void logFrameObjects(const char* what, const std::vector<uint64_t>& handles, uint32_t frame, uint32_t& lines) {
          if (handles.empty() || lines >= kFrameLogLines) {
            return;
          }
          ++lines;
          uint32_t worldUI = 0, captured = 0;
          std::string samples;
          for (size_t i = 0; i < handles.size(); ++i) {
            const auto found = objects.find(handles[i]);
            if (found == objects.end()) {
              continue;
            }
            worldUI += CategoryFlags(found->second.categories).test(InstanceCategories::WorldUI) ? 1 : 0;
            captured += found->second.captured ? 1 : 0;
            if (i < 4) {
              samples += describeObject(handles[i], found->second, frame);
            }
          }
          Logger::info(str::format("[D3D9v2] ", what, " frame=", frame, " count=", handles.size(), " worldUI=", worldUI,
                                   " captured=", captured, " |", samples));
        }

        // Every fourth frame, every World UI object after the frame's moves: where the frontend placed it and where its
        // instance is. At most kWorldUILogLines per log interval.
        static constexpr uint32_t kWorldUILogLines = 64;
        void logWorldUIObjects(RetainedScene& scene, uint32_t frame) {
          if ((frame & 3) != 0) {
            return;
          }
          for (const auto& [handle, object] : objects) {
            if (!CategoryFlags(object.categories).test(InstanceCategories::WorldUI) || worldUILogLines >= kWorldUILogLines) {
              continue;
            }
            ++worldUILogLines;
            const RtInstance* instance = scene.submeshInstance(handle, 0);
            const RetainedScene::EntryState state = scene.entryState(handle);
            const char* stateName = state == RetainedScene::EntryState::Direct ? "direct"
                                  : state == RetainedScene::EntryState::Pending ? "pending"
                                  : state == RetainedScene::EntryState::Legacy ? "legacy"
                                  : state == RetainedScene::EntryState::Rejected ? "rejected" : "none";
            std::string placed = " instance=none";
            if (instance != nullptr) {
              const Vector3 q = instance->surface.objectToWorld[3].xyz();
              placed = str::format(" instance=(", q.x, ", ", q.y, ", ", q.z, ")");
            }
            Logger::info(str::format("[D3D9v2] worldUI object frame=", frame, describeObject(handle, object, frame),
                                     " lastSeen=", object.frameLastSeen, " moved=", object.frameMoved == ~0u ? -1 : int64_t(object.frameMoved),
                                     " entry=", stateName, placed, object.releasing ? " releasing" : ""));
          }
        }

        // R4: a first-draw handover the scene rejected. The app thread keeps skipping the draw until it hears of the
        // rejection (1-2 frames), so the CS thread replays the kept draw through the old path meanwhile.
        struct Fallback {
          DrawCallState state;     // the processed draw, its geometry snapshotted
          uint32_t lastFrame = 0;
        };
        std::unordered_map<uint64_t, Fallback> fallbacks;

        // R6: the capture buffer of a vertex-captured draw (its positions), or null.
        static const DxvkBuffer* captureBufferOf(const DrawCallState& state) {
          return state.vertexCaptured && state.getGeometryData().positionBuffer.defined()
            ? state.getGeometryData().positionBuffer.buffer().ptr() : nullptr;
        }

        // What a description of a processed draw depends on besides the world matrix: the geometry (FullGeometryHash, the
        // hash acquireGeometry reuses geometry by), the bones, both materials, the texture transform, the categories, the
        // camera, the fog and the replacement bucket. The world matrix is compared with Object::objectToWorld, which
        // follows the moves applied since the description.
        static XXH64_hash_t descriptionKey(const DrawCallState& state, const MaterialData& material,
                                           const ReplacementBucket* bucket) {
          const DrawCallTransforms& transforms = state.getTransformData();
          const struct {
            XXH64_hash_t geometry, bones, material, legacyMaterial, fog;
            uint64_t bucket;
            uint32_t categories, cameraType, texgenMode, pad;
          } k = {
            state.getGeometryData().getHashForRule<rules::FullGeometryHash>(), state.getSkinningState().boneHash,
            material.getHash(), state.getMaterialData().computeIdentityHash(),
            state.getFogState().mode != D3DFOG_NONE ? state.getFogState().getHash() : 0,
            uint64_t(reinterpret_cast<uintptr_t>(bucket)), state.getCategoryFlags().raw(), uint32_t(state.cameraType),
            uint32_t(transforms.texgenMode), 0u
          };
          static_assert(sizeof(k) == 64, "no implicit padding");
          const XXH64_hash_t h = XXH3_64bits(&k, sizeof(k));
          return XXH3_64bits_withSeed(&transforms.textureTransform, sizeof(transforms.textureTransform), h);
        }

        // A4: an opaque draw of a scene-owned object after a non-opaque draw of it in the same frame. The old path merges
        // it into the first draw's instance, so its prepared state becomes the scene entry's merged pass.
        void mergePass(RtxContext* ctx, const DrawParameters& params, DrawCallState& state, uint64_t handle) {
          const auto found = objects.find(handle);
          if (found == objects.end() || found->second.pending || found->second.releasing) {
            return;
          }
          SceneManager& sm = ctx->getSceneManager();
          const MaterialData* overrideMaterial = nullptr;
          if (!ctx->prepareGeometryForRT(params, state, overrideMaterial, skinningCamera(state))) {
            return;
          }
          RTX_OPTION_INVALIDATION_SCOPE(RtxOptionFlags::InvalidatesDrawcallTranslation);
          auto pass = std::make_shared<RetainedScene::MergedPass>();
          if (D3D9V2Options::materialTranslator()) {
            translator::MaterialTranslation translation =
              translator::MaterialTranslator::translate(sm, state, fixedFunctionState(state));
            pass->material = std::make_shared<const MaterialData>(std::move(translation.material));
            pass->alphaState = translation.alphaState;
            pass->legacySurface = translation.legacySurface;
          } else {
            pass->material = std::make_shared<const MaterialData>(sm.resolveDrawMaterial(state));
          }
          pass->key = descriptionKey(state, *pass->material, nullptr);
          pass->drawCall = std::move(state);
          // The build draws the pass with the entry's geometry, so the pass keeps no reference to this frame's buffers.
          pass->drawCall.overrideGeometryData(std::make_shared<const RasterGeometry>());
          found->second.frameMergedPass = sm.device()->getCurrentFrameId();
          if (mergedPasses.insert(handle).second) {
            totals.mergedPasses++;
          }
          sm.retainedScene().setMergedPass(handle, std::move(pass));
        }

        void rejectFirstDraw(RtxContext* ctx, DrawCallState& state, const MaterialData* overrideMaterial, uint64_t handle,
                             const char* reason) {
          SceneManager& sm = ctx->getSceneManager();
          // R5: an object being updated loses its scene copy (and lights) first; a no-op for a first-draw handover.
          eraseObject(sm, handle);
          // The processed state, copied before the submit (which may move it); its geometry still points at this
          // frame's staging data, so the snapshot is taken now.
          Fallback fallback;
          bool keep = overrideMaterial == nullptr;  // per-frame override materials (terrain bake) are not replayed
          if (keep) {
            fallback.state = state;
            RasterGeometry geometry = fallback.state.getGeometryData();
            keep = snapshotGeometry(ctx, sm.device(), geometry, captureBufferOf(fallback.state));
            if (keep) {
              geometry.futureGeometryHashes = Future<GeometryHashes>();
              geometry.futureBoundingBox = Future<AxisAlignedBoundingBox>();
              geometry.hashesPrecomputed = true;
              fallback.state.overrideGeometryData(std::make_shared<const RasterGeometry>(std::move(geometry)));
            }
          }
          ctx->submitPreparedGeometry(state, overrideMaterial);
          if (keep) {
            fallback.lastFrame = sm.device()->getCurrentFrameId();
            fallbacks[handle] = std::move(fallback);
          }
          reject(handle, reason);
        }

        // R2 parity: the frontend's camera type (written into the queued state) against processCameraData's (written by
        // prepareGeometryForRTImpl, Unknown mapped to Main unless rtx.skipObjectsWithUnknownCamera).
        void checkCamera(CameraType::Enum expected, const DrawCallState& state) {
          if (expected == CameraType::Unknown && !RtxOptions::skipObjectsWithUnknownCamera()) {
            expected = CameraType::Main;
          }
          totals.cameraChecks++;
          if (state.cameraType != expected) {
            if (totals.cameraMismatches == 0) {
              firstCameraMismatch = str::format("frontend ", uint32_t(expected), " render thread ", uint32_t(state.cameraType));
            }
            totals.cameraMismatches++;
          }
        }
        std::unordered_map<std::string, uint64_t> rejectReasons;
        uint32_t logFrames = 0;

        // [MaterialTranslator] parity (rtx.d3d9v2.materialTranslator with rtx.coreMaterial.selfCheck).
        struct Parity {
          uint64_t draws = 0;
          uint64_t mismatches = 0;
          std::string first = "none";
          uint32_t lastLogFrame = ~0u;
        } parity;

        void checkParity(const translator::MaterialTranslation& translation, const DrawCallState& state) {
          // The backend's derivation for the same draw and render material (InstanceManager::calculateAlphaState and the
          // legacy fields updateInstance reads).
          const AlphaStateDesc backendAlpha = InstanceManager::predictAlphaState(state, translation.material);
          const LegacySurfaceDesc backendLegacy = deriveLegacySurface(state.getMaterialData(), state.getTransformData());
          const char* alphaField = firstAlphaStateDifference(translation.alphaState, backendAlpha);
          const char* legacyField = firstLegacySurfaceDifference(translation.legacySurface, backendLegacy);
          parity.draws++;
          const bool alphaDiffers = std::strcmp(alphaField, "none") != 0;
          const bool legacyDiffers = std::strcmp(legacyField, "none") != 0;
          if (alphaDiffers || legacyDiffers) {
            if (parity.mismatches == 0) {
              parity.first = alphaDiffers ? str::format("alphaState.", alphaField) : str::format("legacySurface.", legacyField);
            }
            parity.mismatches++;
          }
        }

        void logParity(uint32_t frame) {
          if (parity.lastLogFrame == ~0u) {
            parity.lastLogFrame = frame;
            return;
          }
          if (frame - parity.lastLogFrame < kLogIntervalFrames) {
            return;
          }
          if (parity.draws != 0 || parity.mismatches != 0) {
            Logger::info(str::format("[MaterialTranslator] parity draws=", parity.draws, " mismatches=", parity.mismatches,
                                     " first=", parity.first, " (frames ", parity.lastLogFrame, "-", frame, ")"));
          }
          parity = Parity();
          parity.lastLogFrame = frame;
        }

        void reject(uint64_t handle, const char* reason) {
          totals.rejected++;
          rejectReasons[reason]++;
          channel->push(handle, Channel::Kind::Rejected);
        }

        void requestRelease(uint64_t handle, Object& object, ReleaseReason reason) {
          if (!object.releasing) {
            object.releasing = true;
            totals.released++;
            totals.releasedBy[uint32_t(reason)]++;
            channel->push(handle, Channel::Kind::Released);
          }
        }

        // V4: the replacement's lights at the object's transform, as SceneManager::processReplacementLights places them.
        static void createLights(LightManager& lightManager, Object& object) {
          for (const RtLight& objectSpace : object.lightTemplates) {
            RtLight light = objectSpace;
            light.applyTransform(object.objectToWorld);
            object.lights.push_back(lightManager.createExternallyTrackedLight(light));
          }
          object.lightsCreated = true;
        }

        static void moveLights(LightManager& lightManager, Object& object) {
          for (size_t i = 0; i < object.lights.size() && i < object.lightTemplates.size(); ++i) {
            RtLight light = object.lightTemplates[i];
            light.applyTransform(object.objectToWorld);
            lightManager.updateExternallyTrackedLight(object.lights[i], light);
          }
        }

        // Externally tracked lights live until marked; the light manager drops them at its next collection.
        static void destroyLights(Object& object) {
          for (RtLight* light : object.lights) {
            light->markForGarbageCollection();
          }
          object.lights.clear();
        }

        // V4: the replacement's graph instances, owned by the object (ScenePrimOwner resolves their targets through the
        // scene by handle and submesh). Made once the object's lights exist, and again for any the graph manager dropped
        // since (a graph reset, rtx.graph.enable switched off and on), as drawReplacements makes them again.
        static void ensureGraphs(SceneManager& sm, RtxContext* ctx, uint64_t handle, Object& object) {
          if (object.graphTemplates.empty() || !object.lightsCreated) {
            return;
          }
          GraphManager& graphManager = sm.getGraphManager();
          object.graphs.resize(object.graphTemplates.size(), 0);
          ScenePrimOwner owner;
          bool ownerReady = false;
          for (size_t i = 0; i < object.graphTemplates.size(); ++i) {
            if (object.graphs[i] != 0 && graphManager.getGraphInstances().count(object.graphs[i]) != 0) {
              continue;
            }
            if (!ownerReady) {
              owner.scene = &sm.retainedScene();
              owner.handle = handle;
              owner.slots.resize(object.primSlots.size());
              for (size_t k = 0; k < object.primSlots.size(); ++k) {
                const PrimSlot& slot = object.primSlots[k];
                owner.slots[k].type = slot.type;
                if (slot.type == PrimInstance::Type::Instance) {
                  owner.slots[k].submesh = slot.index;
                } else if (slot.type == PrimInstance::Type::Light) {
                  owner.slots[k].light = slot.index < object.lights.size() ? object.lights[slot.index] : nullptr;
                }
              }
              ownerReady = true;
            }
            const GraphInstance* graph = graphManager.addInstance(Rc<DxvkContext>(ctx), object.graphTemplates[i], &owner);
            object.graphs[i] = graph != nullptr ? graph->getId() : 0;
          }
        }

        static void destroyGraphs(SceneManager& sm, Object& object) {
          GraphManager& graphManager = sm.getGraphManager();
          for (const uint64_t id : object.graphs) {
            if (id != 0 && graphManager.getGraphInstances().count(id) != 0) {
              graphManager.removeInstance(id);
            }
          }
          object.graphs.clear();
        }

        // The scene's copy of the object goes (its entry and, V4, its lights and graphs).
        void eraseObject(SceneManager& sm, uint64_t handle) {
          const auto found = objects.find(handle);
          if (found == objects.end()) {
            return;
          }
          destroyGraphs(sm, found->second);
          destroyLights(found->second);
          objects.erase(found);
          sm.retainedScene().destroy(handle);
        }

        // replacementsAllowed (rtx.d3d9v2.directReplacements): draws with USD mesh replacements pass here and are checked
        // by expandReplacement; otherwise they are rejected here.
        static const char* eligibility(SceneManager& sm, const DrawCallState& state, const MaterialData* overrideMaterial,
                                       const MaterialData& material, bool replacementsAllowed, bool skinnedAllowed) {
          const bool unordered = unorderedActive();  // R7
          const bool services = specialCategoriesActive();  // R8
          if (!sm.retainedScene().active()) {
            return "rtx.scene.directRetained off";
          }
          if (!acceptedCamera(state.cameraType)) {  // R8 / R8b
            return "camera";
          }
          // R8b (rtx.d3d9v2.terrain): the terrain baker's material is the object's material.
          if (overrideMaterial != nullptr && !(terrainActive() && state.getCategoryFlags().test(InstanceCategories::Terrain))) {
            return "override material";
          }
          if (state.getSkinningState().numBones > 0 && !skinnedAllowed) {
            return "skinned";
          }
          if (specialCategory(state.getCategoryFlags(), unordered, services)) {
            return "category";
          }
          if ((state.isDrawingToRaytracedRenderTarget || state.isUsingRaytracedRenderTarget) && !renderTargetsActive()) {
            return "render target";
          }
          if (!services && RtxOptions::shouldConvertToLight(state.getMaterialData().getHash())) {
            return "light conversion";
          }
          if (state.getMaterialData().blendMode.enableBlending && !unordered) {
            return "blended";
          }
          if (!replacementsAllowed && sm.hasMeshReplacements(state)) {
            return "mesh replacement";
          }
          // An ignored material is accepted: the scene builds no instance for it, as processDrawCallState draws nothing.
          if (!services && material.getType() == MaterialDataType::RayPortal) {
            return "portal";
          }
          if (!services && material.getParticleSystemDesc() != nullptr) {
            return "particle material";
          }
          const RtSurface::AlphaState alpha = InstanceManager::predictAlphaState(state, material);
          if ((alpha.isDecal || alpha.emissiveBlend) && !unordered) {
            return "unordered";
          }
          if (!state.getGeometryData().positionBuffer.defined()) {
            return "geometry";
          }
          return nullptr;
        }

        // The draw's buffers are D3D9-owned or staging memory that is reused after the frame: copy every referenced
        // range into buffers the snapshot owns (once per distinct source range) and point the slices at them.
        // keep (R6): a buffer referenced as it is instead of copied: a vertex-capture buffer (positions and any captured
        // normals / texcoords). It is a new buffer per draw (D3D9Rtx::allocVertexCaptureBuffer), written once by the
        // raster draw recorded before this commit, so holding the reference keeps the data and saves the GPU copy.
        static bool snapshotGeometry(RtxContext* ctx, DxvkDevice* device, RasterGeometry& geometry, const DxvkBuffer* keep = nullptr) {
          struct Copied {
            const DxvkBuffer* source;
            VkDeviceSize offset;
            VkDeviceSize length;
            DxvkBufferSlice copy;
          };
          std::vector<Copied> copies;
          bool ok = true;

          const auto copyRange = [&](const RasterBuffer& from) -> DxvkBufferSlice {
            if (keep != nullptr && from.buffer().ptr() == keep) {
              return DxvkBufferSlice(from.buffer(), from.offset(), from.length());
            }
            for (const Copied& c : copies) {
              if (c.source == from.buffer().ptr() && c.offset == from.offset() && c.length == from.length()) {
                return c.copy;
              }
            }
            DxvkBufferCreateInfo info {};
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
            info.access = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            info.size = align(from.length(), CACHE_LINE_SIZE);
            Rc<DxvkBuffer> buffer = device->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                                         DxvkMemoryStats::Category::RTXBuffer, "D3D9 v2 geometry snapshot");
            if (buffer == nullptr) {
              ok = false;
              return DxvkBufferSlice();
            }
            const void* src = from.mapPtr(0);
            void* dst = buffer->mapPtr(0);
            if (src != nullptr && dst != nullptr && !from.isPendingGpuWrite()) {
              std::memcpy(dst, src, from.length());
            } else {
              ctx->copyBuffer(buffer, 0, from.buffer(), from.offset(), from.length());
            }
            DxvkBufferSlice slice(buffer, 0, from.length());
            copies.push_back({ from.buffer().ptr(), from.offset(), from.length(), slice });
            return slice;
          };
          const auto vertex = [&](RasterBuffer& b) {
            if (b.defined()) {
              b = RasterBuffer(copyRange(b), b.offsetFromSlice(), b.stride(), b.vertexFormat());
            }
          };
          vertex(geometry.positionBuffer);
          vertex(geometry.normalBuffer);
          vertex(geometry.texcoordBuffer);
          vertex(geometry.color0Buffer);
          vertex(geometry.blendWeightBuffer);
          vertex(geometry.blendIndicesBuffer);
          if (geometry.indexBuffer.defined()) {
            geometry.indexBuffer = RasterBuffer(copyRange(geometry.indexBuffer), geometry.indexBuffer.offsetFromSlice(),
                                                geometry.indexBuffer.stride(), geometry.indexBuffer.indexType());
          }
          return ok;
        }

        // V4: a USD mesh replacement expanded the way SceneManager::drawReplacements draws it.
        struct Expansion {
          std::vector<RasterGeometry> submeshes;            // one per mesh prim
          std::vector<ExternalDrawState::SubmeshPart> parts; // same order
          std::vector<size_t> originalSlots;                // submeshes that are the draw's own geometry (includeOriginal)
          std::vector<RtLight> lights;                      // light prims, object space
          std::vector<PrimSlot> primSlots;                  // one per prim of the bucket
          std::vector<std::shared_ptr<const RtGraphState>> graphs;  // graph prims, in prim order
        };

        // Fills `out` from the bucket, or says why the object stays on the old path. Same prims, order, transforms,
        // categories and materials as drawReplacements / buildReplacementMeshDrawCallState, including its running
        // material: a mesh prim with its own material sets it for the prims after it that have none.
        // A replacement of a vertex-captured draw is placed at the draw's transform as on the old path, and its original
        // (includeOriginal) is the captured geometry. With rtx.d3d9v2.specialCategories, replacement prims with particle
        // systems, particle materials or portal materials are allowed, as they are for plain draws: the scene's services
        // spawn the particles and stamp the portals.
        static const char* expandReplacement(const DrawCallState& state, const ReplacementBucket& bucket,
                                             const MaterialData& drawMaterial, Expansion& out) {
          const bool services = specialCategoriesActive();
          std::shared_ptr<const MaterialData> running = std::make_shared<const MaterialData>(drawMaterial);
          uint32_t drawnParts = 0;
          out.primSlots.resize(bucket.replacements.size());
          for (size_t prim = 0; prim < bucket.replacements.size(); ++prim) {
            const AssetReplacement& replacement = bucket.replacements[prim];
            if (replacement.type == AssetReplacement::eGraph) {
              // As processReplacementGraphs: a graph prim without a graph state is an authoring error.
              if (!replacement.graphState) {
                return "replacement: graph state";
              }
              out.primSlots[prim].type = PrimInstance::Type::Graph;
              out.graphs.push_back(replacement.graphState);
              continue;
            }
            if (replacement.particleSystem.has_value() && !services) {
              return "replacement: particle system";
            }
            ExternalDrawState::SubmeshPart part;
            part.particleSystem = replacement.particleSystem;
            if (replacement.includeOriginal) {
              part.submeshToObject = Matrix4();
              part.meshReplacement = false;
              out.primSlots[prim] = { PrimInstance::Type::Instance, uint32_t(out.submeshes.size()) };
              out.originalSlots.push_back(out.submeshes.size());
              out.submeshes.push_back(RasterGeometry());  // the snapshot of the draw's geometry, filled by handover
            } else if (replacement.type == AssetReplacement::eMesh) {
              if (replacement.instancesToObject != nullptr && !replacement.instancesToObject->empty()) {
                return "replacement: point instancer";
              }
              if (replacement.geometry == nullptr || !replacement.geometry->data.positionBuffer.defined()) {
                return "replacement: geometry";
              }
              if (replacement.materialData != nullptr) {
                running = std::make_shared<const MaterialData>(*replacement.materialData);
              }
              part.submeshToObject = replacement.replacementToObject;
              part.meshReplacement = true;
              out.primSlots[prim] = { PrimInstance::Type::Instance, uint32_t(out.submeshes.size()) };
              RasterGeometry geometry = replacement.geometry->data;
              geometry.externalMaterial = nullptr;
              geometry.externalMesh = nullptr;
              out.submeshes.push_back(std::move(geometry));
            } else {
              if (replacement.type == AssetReplacement::eLight && replacement.lightData.has_value()) {
                out.primSlots[prim] = { PrimInstance::Type::Light, uint32_t(out.lights.size()) };
                out.lights.push_back(replacement.lightData->toRtLight());
              }
              continue;
            }
            part.material = running;
            part.categories = replacement.categories.applyCategoryFlags(state.getCategoryFlags());
            // What eligibility checks on the draw, for the prim as drawn.
            if (specialCategory(part.categories, unorderedActive(), specialCategoriesActive())) {
              return "replacement: category";
            }
            if (running->getType() == MaterialDataType::RayPortal && !services) {
              return "replacement: portal";
            }
            if (running->getParticleSystemDesc() != nullptr && !services) {
              return "replacement: particle material";
            }
            DrawCallState probe = state;
            probe.modifyCategoryFlags() = part.categories;
            const RtSurface::AlphaState alpha = InstanceManager::predictAlphaState(probe, *running);
            if (alpha.isDecal || alpha.emissiveBlend) {
              return "replacement: unordered";
            }
            drawnParts += running->getIgnored() ? 0 : 1;
            out.parts.push_back(std::move(part));
          }
          // drawReplacements places lights only on a replacement that drew a mesh.
          if (drawnParts == 0) {
            return "replacement: no meshes";
          }
          return nullptr;
        }

        // With rtx.d3d9v2.explicitCamera the queued state carries the draw's camera type, and its camera decomposes the
        // draw's fixed-function skinned matrices. A skipped draw sets no camera, so the last camera set may be another
        // type's. Unknown is mapped to Main, as prepareGeometryForRTImpl maps it.
        static CameraType::Enum skinningCamera(const DrawCallState& state) {
          if (!explicitCameraActive() || state.cameraType >= CameraType::Count) {
            return CameraType::Count;
          }
          return state.cameraType == CameraType::Unknown ? CameraType::Main : state.cameraType;
        }

        // firstDraw (R4, CommitMode::HandoverFirstDraw): the old path does not run for this draw; the scene builds the
        // object at once, and a rejection draws it through the old path and keeps a fallback (rejectFirstDraw).
        // update (R5, CommitMode::Update): the same as firstDraw for an object the scene already owns under `handle`: the
        // entry is described and built again in place (same instances; geometry refit or rebuilt in place).
        void handover(RtxContext* ctx, const DrawParameters& params, DrawCallState& state, uint64_t handle, bool firstDraw,
                      bool update = false) {
          totals.handovers++;
          SceneManager& sm = ctx->getSceneManager();
          const MaterialData* overrideMaterial = nullptr;
          // R8b (rtx.d3d9v2.skybox3d): a 3D-skybox draw is described as the Main-camera object tryHandleSky would submit
          // (Sky category cleared, transforms reprojected); its sky-space transforms are kept for the per-frame reprojection.
          // (NUMOS draws no game sky: tryHandleSky skips these draws, and so does this.)
          const bool skybox = skybox3dActive() && state.cameraType == CameraType::Sky &&
                              RtxOptions::skyMode() != SkyMode::Numos && !ctx->skyDrawRasterizes(state);
          DrawCallTransforms skyTransforms;
          if (skybox) {
            if (!ctx->prepareSkyboxGeometry(state)) {
              reject(handle, "not submitted");
              return;
            }
            skyTransforms = state.getTransformData();
            state.cameraType = CameraType::Main;
            state.modifyCategoryFlags().clr(InstanceCategories::Sky);
            if (!ctx->reprojectSkyToMain(state.modifyTransformData(), true)) {
              // Nothing is ray traced before the first main camera, so the draw is only tried again.
              totals.skyboxRetries++;
              channel->push(handle, Channel::Kind::Retry);
              return;
            }
          } else if (!ctx->prepareGeometryForRT(params, state, overrideMaterial, skinningCamera(state))) {
            reject(handle, "not submitted");
            return;
          }
          // As around submitDrawState's dynamic update: every option the translation and the build read is tagged, so
          // changing it re-prepares the scene's objects (flush, Channel::reprepare).
          RTX_OPTION_INVALIDATION_SCOPE(RtxOptionFlags::InvalidatesDrawcallTranslation);
          // Phase 4 M3b: rtx.d3d9v2.materialTranslator. The render material is the same resolveDrawMaterial result
          // either way; the translator adds the alpha / legacy descriptions.
          // R8b: a baked terrain draw's material is the baker's (overrideMaterial); the translator does not describe it.
          const bool bakedTerrain = overrideMaterial != nullptr && terrainActive() &&
                                    state.getCategoryFlags().test(InstanceCategories::Terrain);
          const bool useTranslator = D3D9V2Options::materialTranslator() && !bakedTerrain;
          std::optional<translator::MaterialTranslation> translation;
          if (useTranslator) {
            translation.emplace(translator::MaterialTranslator::translate(sm, state, fixedFunctionState(state)));
            if (RtxOptions::CoreMaterial::selfCheck()) {
              checkParity(*translation, state);
            }
          }
          MaterialData material = bakedTerrain ? *overrideMaterial
                                : translation.has_value() ? translation->material : sm.resolveDrawMaterial(state);
          // V4: the replacement submitDrawState draws for this draw (none looked up while the option is off).
          const bool replacementsAllowed = D3D9V2Options::directReplacements();
          uint32_t matchedRule = 0;
          const std::shared_ptr<const ReplacementBucket> bucket =
            replacementsAllowed ? sm.findMeshReplacements(state, &matchedRule) : nullptr;
          const char* reason = eligibility(sm, state, overrideMaterial, material, replacementsAllowed, update || dynamicGeometryActive());
          Expansion expansion;
          if (reason == nullptr && bucket != nullptr) {
            reason = expandReplacement(state, *bucket, material, expansion);
          }
          // The old path runs as for any draw (the object is still on it until the scene has built its copy); the
          // processed state is kept first, since the submit may move it. A replaced object's replacement instance is
          // marked handed to the scene (collected once the old path stops drawing it, lights included).
          DrawCallState kept;
          if (reason == nullptr) {
            kept = state;
          }
          if (firstDraw) {
            if (reason != nullptr) {
              rejectFirstDraw(ctx, state, overrideMaterial, handle, reason);
              return;
            }
          } else {
            sm.setSubmittingHandedOver(reason == nullptr && bucket != nullptr);
            ctx->submitPreparedGeometry(state, overrideMaterial);
            sm.setSubmittingHandedOver(false);
            if (reason != nullptr) {
              reject(handle, reason);
              return;
            }
          }

          // An update that would describe the object as the scene already has it counts as seen, without the snapshot and
          // the rebuild. Captured, 3D-skybox and baked-terrain draws are always described again, since their positions or
          // transforms come from more than the geometry hash and the draw's own transforms show.
          const XXH64_hash_t described = descriptionKey(kept, material, bucket.get());
          if (update && !kept.vertexCaptured && !skybox && !bakedTerrain) {
            const auto previous = objects.find(handle);
            const Matrix4& objectToWorld = kept.getTransformData().objectToWorld;
            if (previous != objects.end() && !previous->second.pending && !previous->second.releasing &&
                previous->second.described == described && previous->second.reprepareEpoch == reprepareEpoch &&
                std::memcmp(&previous->second.objectToWorld, &objectToWorld, sizeof(Matrix4)) == 0) {
              previous->second.frameLastSeen = sm.device()->getCurrentFrameId();
              totals.unchangedUpdates++;
              return;
            }
          }

          // The draw's own geometry: the whole object (V2), or the original kept by a replacement (V4 includeOriginal).
          RasterGeometry geometry = kept.getGeometryData();
          const AxisAlignedBoundingBox originalBounds = geometry.boundingBox;
          const XXH64_hash_t originalFullHash = geometry.getHashForRule<rules::FullGeometryHash>();
          // Geometry that changes between descriptions is owned by the entry (GeomKey::owner), so its key only has to stay
          // the same across updates: the scene then refits it with motion history when the indices match and rebuilds it
          // in place when they do not. The vertex layout is part of it, so a layout change builds new geometry.
          const bool deforming = kept.getSkinningState().numBones > 0 || kept.dynamicSource || kept.vertexCaptured || update;
          XXH64_hash_t layoutHash = geometry.hashes[HashComponents::VertexLayout];
          const uint32_t layout[4] = { uint32_t(geometry.topology), uint32_t(geometry.indexBuffer.indexType()),
                                       uint32_t(geometry.positionBuffer.stride()), uint32_t(geometry.positionBuffer.vertexFormat()) };
          layoutHash = XXH3_64bits_withSeed(layout, sizeof(layout), layoutHash);
          const bool needOriginal = bucket == nullptr || !expansion.originalSlots.empty();
          if (needOriginal) {
            if (!snapshotGeometry(ctx, sm.device(), geometry, captureBufferOf(kept))) {
              if (firstDraw) {
                rejectFirstDraw(ctx, state, overrideMaterial, handle, "snapshot");
              } else {
                reject(handle, "snapshot");
              }
              return;
            }
            // Hashes and bounds were finalized by prepareGeometryForRT; the snapshot carries them as values.
            geometry.futureGeometryHashes = Future<GeometryHashes>();
            geometry.futureBoundingBox = Future<AxisAlignedBoundingBox>();
            geometry.hashesPrecomputed = true;
            geometry.externalMaterial = nullptr;
            geometry.externalMesh = nullptr;
          }

          const uint32_t frame = sm.device()->getCurrentFrameId();
          Object object;
          object.assetHash = kept.getHash(RtxOptions::geometryAssetHashRule());
          // The legacy hashes submitDrawState tracks: all with the rule on, but for a replaced draw only those it
          // looked at before the replacement was found.
          if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash0) == rules::LegacyAssetHash0 &&
              (bucket == nullptr || matchedRule >= 1)) {
            object.legacyHash0 = kept.getHashLegacy(rules::LegacyAssetHash0);
          }
          if ((RtxOptions::geometryHashGenerationRule() & rules::LegacyAssetHash1) == rules::LegacyAssetHash1 &&
              (bucket == nullptr || matchedRule >= 2)) {
            object.legacyHash1 = kept.getHashLegacy(rules::LegacyAssetHash1);
          }
          object.bounds = originalBounds;
          object.objectToWorld = kept.getTransformData().objectToWorld;
          object.categories = kept.getCategoryFlags().raw();
          object.frameCreated = frame;
          object.frameLastSeen = frame;
          object.described = described;
          object.reprepareEpoch = reprepareEpoch;
          if (kept.getFogState().mode != D3DFOG_NONE) {
            object.fogHash = kept.getFogState().getHash();
          }

          ExternalDrawState desc;
          if (bucket == nullptr) {
            const uint64_t meshKey = kMeshNamespace | ((deforming ? layoutHash : originalFullHash) & (kMeshNamespace - 1));
            desc.mesh = reinterpret_cast<remixapi_MeshHandle>(static_cast<uintptr_t>(meshKey));
            desc.inlineSubmeshes = std::make_shared<const std::vector<RasterGeometry>>(1, std::move(geometry));
            if (translation.has_value()) {
              desc.alphaState = translation->alphaState;
              desc.legacySurface = translation->legacySurface;
            }
          } else {
            // V4: one submesh per replacement mesh prim. Objects replaced by the same bucket share the prims' geometry
            // (the key includes the original geometry when the replacement keeps it).
            for (const size_t slot : expansion.originalSlots) {
              expansion.submeshes[slot] = geometry;
            }
            const XXH64_hash_t lookupHash = matchedRule == 0 ? object.assetHash
                                          : kept.getHashLegacy(matchedRule == 1 ? rules::LegacyAssetHash0 : rules::LegacyAssetHash1);
            constexpr uint64_t kV4Seed = 0x56344d6573685265ull;  // "V4MeshRe"
            XXH64_hash_t key = XXH3_64bits_withSeed(&lookupHash, sizeof(lookupHash), kV4Seed);
            if (!expansion.originalSlots.empty()) {
              key = XXH3_64bits_withSeed(&originalFullHash, sizeof(originalFullHash), key);
            }
            desc.mesh = reinterpret_cast<remixapi_MeshHandle>(static_cast<uintptr_t>(kMeshNamespace | (key & (kMeshNamespace - 1))));
            desc.inlineSubmeshes = std::make_shared<const std::vector<RasterGeometry>>(std::move(expansion.submeshes));
            desc.inlineSubmeshParts = std::make_shared<const std::vector<ExternalDrawState::SubmeshPart>>(std::move(expansion.parts));

            // Anti-culling bounds as the replacement instance computes them (meshes and lights, object space).
            ReplacementInstance::LookupKey scratchKey {};
            ReplacementInstance scratch(scratchKey, 0, frame);
            scratch.activeReplacements = bucket;
            scratch.recalculateBoundingBox(object.objectToWorld, &originalBounds);
            scratch.activeReplacements.reset();
            object.bounds = scratch.geometryBoundingBox;
            object.lightBounds = scratch.lightBoundingBox;
            object.replaced = true;
            object.lightTemplates = std::move(expansion.lights);
            object.primSlots = std::move(expansion.primSlots);
            object.graphTemplates = std::move(expansion.graphs);
          }
          desc.resolvedMaterial = std::make_shared<const MaterialData>(std::move(material));
          desc.frontendResolved = true;
          desc.cameraType = kept.cameraType;
          desc.categories = kept.getCategoryFlags();
          // R5: geometry that changes between descriptions (skinned, dynamic / user-pointer data, or an object updated in
          // place) is owned by the entry.
          object.skinned = kept.getSkinningState().numBones > 0;
          object.nonMainCamera = kept.cameraType != CameraType::Main;  // R8
          object.captured = kept.vertexCaptured;
          object.skybox3d = skybox;  // R8b
          object.terrain = bakedTerrain;
          object.skyTransforms = skyTransforms;
          desc.deforming = deforming;
          desc.drawCall = std::move(kept);
          // The instance's object-picking value (InstanceManager::updateInstance reads drawCallID) stays the same in every
          // frame, and RetainedScene::trackPickingMeta registers it every frame, so it is never a stale draw ID.
          desc.drawCall.drawCallID = pickingValue(handle);
          // The stored draw state points at owned geometry too (the snapshot, or V4's first submesh), not at the D3D9 /
          // staging buffers it was made from.
          desc.drawCall.overrideGeometryData(std::shared_ptr<const RasterGeometry>(desc.inlineSubmeshes, &desc.inlineSubmeshes->front()));
          if (object.fogHash != 0 && fogDraws.find(object.fogHash) == fogDraws.end() && makeRoomForFogDraw()) {
            fogDraws.emplace(object.fogHash, desc.drawCall);
          }
          if (!firstDraw) {
            sm.retainedScene().describe(handle, std::move(desc));
            objects[handle] = std::move(object);
            return;
          }
          // R4: built now; the scene owns the object from this draw on (no Accepted message, no pending state).
          RetainedScene& scene = sm.retainedScene();
          if (scene.describeAndBuild(ctx, handle, std::move(desc)) == RetainedScene::EntryState::Direct) {
            object.pending = false;
            const auto previous = objects.find(handle);
            if (previous != objects.end()) {
              // R5 update: same object; its age and (when the light set kept its size) its lights carry over.
              object.frameCreated = previous->second.frameCreated;
              // An update that moves the object counts as a move, so it is not anti-culled mid-motion.
              object.frameMoved = sameMatrix(previous->second.objectToWorld, object.objectToWorld)
                ? previous->second.frameMoved
                : frame;
              if (object.replaced && previous->second.lights.size() == object.lightTemplates.size()) {
                object.lights = std::move(previous->second.lights);
                object.lightsCreated = true;
                moveLights(sm.getLightManager(), object);
              } else {
                destroyLights(previous->second);
              }
              // The graphs keep their state through an update of the same replacement. Their owners point at the
              // carried lights and resolve the rebuilt submeshes through the scene.
              if (object.lightsCreated && previous->second.graphTemplates == object.graphTemplates &&
                  previous->second.primSlots == object.primSlots) {
                object.graphs = std::move(previous->second.graphs);
              } else {
                destroyGraphs(sm, previous->second);
              }
              totals.updated++;
            } else {
              totals.accepted++;
              totals.replacedAccepted += object.replaced ? 1 : 0;
              if (D3D9V2Options::log()) {
                frameBuilt.push_back(handle);
              }
            }
            if (object.replaced && !object.lightsCreated) {
              // No old replacement instance exists (the old path never drew it), so the scene's lights start now.
              createLights(sm.getLightManager(), object);
              totals.lightsCreated += object.lights.size();
            }
            ensureGraphs(sm, ctx, handle, object);
            objects[handle] = std::move(object);
            return;
          }
          scene.destroy(handle);
          rejectFirstDraw(ctx, state, overrideMaterial, handle, "scene");
        }

        // An old-path draw of a pending object (handle != 0). A replaced object's replacement instance is marked handed
        // to the scene, so it goes in the first frame the app thread skips the draw.
        // R8b: a 3D-skybox object's per-frame draw (RasterOnly): its sky-camera transforms for this frame's reprojection.
        // R8b: a terrain object's per-frame draw (RasterOnly): the bake into the cascades, then the baked texture transform
        // and, after a cascade change, the baker's material for the scene's copy.
        bool terrainDraw(RtxContext* ctx, const DrawParameters& params, DrawCallState& state, uint64_t handle) {
          const auto found = objects.find(handle);
          if (found == objects.end() || !found->second.terrain) {
            return false;
          }
          SceneManager& sm = ctx->getSceneManager();
          const MaterialData* bakerMaterial = ctx->bakeTerrainDraw(params, state);
          if (bakerMaterial == nullptr) {
            totals.terrainMisses++;  // baking off or failed this frame: the scene keeps its last texture transform
            return true;
          }
          RetainedScene& scene = sm.retainedScene();
          const DrawCallTransforms& transforms = state.getTransformData();
          scene.setTextureTransform(handle, transforms.textureTransform, transforms.texgenMode);
          if (sm.getTerrainBaker().cascadeCompositionChangedThisFrame()) {
            scene.setResolvedMaterial(handle, std::make_shared<const MaterialData>(*bakerMaterial));
            totals.terrainRematerials++;
          }
          totals.terrainBakes++;
          return true;
        }

        // R4: a baked terrain object the scene rejected keeps no fallback (its baker material is made per frame), but the
        // app thread replays its draw as a raster service until it hears of the rejection. The old path draws those
        // draws, bake included, so the terrain is not missing for a frame or two.
        bool rejectedTerrainDraw(RtxContext* ctx, const DrawParameters& params, DrawCallState& state, uint64_t handle) {
          if (objects.find(handle) != objects.end() || !state.testCategoryFlags(InstanceCategories::Terrain) ||
              !TerrainBaker::needsTerrainBaking()) {
            return false;
          }
          commitOldPath(ctx, params, state);
          totals.terrainFallbacks++;
          return true;
        }

        bool skyboxDraw(uint64_t handle, const DrawCallState& state) {
          const auto found = objects.find(handle);
          if (found == objects.end() || !found->second.skybox3d) {
            return false;
          }
          found->second.skyTransforms = state.getTransformData();
          return true;
        }

        void oldPathDraw(RtxContext* ctx, const DrawParameters& params, DrawCallState& state, uint64_t handle) {
          const auto found = objects.find(handle);
          const bool handedOver = found != objects.end() && found->second.replaced && !found->second.releasing;
          SceneManager& sm = ctx->getSceneManager();
          sm.setSubmittingHandedOver(handedOver);
          commitOldPath(ctx, params, state);
          sm.setSubmittingHandedOver(false);
        }

        // The old path for one draw; with rtx.d3d9v2.explicitCamera the frontend's camera type is checked against it.
        void commitOldPath(RtxContext* ctx, const DrawParameters& params, DrawCallState& state) {
          if (!explicitCameraActive()) {
            ctx->commitGeometryToRT(params, state);
            return;
          }
          const CameraType::Enum expected = state.cameraType;
          ctx->commitGeometryToRT(params, state);
          checkCamera(expected, state);
        }

        void releaseEverything(SceneManager& sm) {
          if (objects.empty()) {
            return;
          }
          for (auto& [handle, object] : objects) {
            destroyGraphs(sm, object);
            destroyLights(object);
            sm.retainedScene().destroy(handle);
            channel->push(handle, Channel::Kind::Released);
            totals.released++;
            totals.releasedBy[uint32_t(ReleaseReason::Reset)]++;
          }
          objects.clear();
          fogDraws.clear();
          fallbacks.clear();
          totals.resets++;
          totals.resetsBy[uint32_t(ResetReason::Off)]++;
        }

        void flush(RtxContext* ctx, const FrameBatch& batch) {
          SceneManager& sm = ctx->getSceneManager();
          // D3D9-only games have no DrawRetainedInstances to switch the direct lane on (rtx.scene.directRetained).
          if (!sm.retainedV2Active()) {
            sm.selectSceneImplementation(scene::Options::directRetained());
          }
          RetainedScene& scene = sm.retainedScene();
          LightManager& lightManager = sm.getLightManager();
          const uint32_t frame = sm.device()->getCurrentFrameId();

          // Events that end the scene's copies: a scene clear, mods changed, the direct lane or rtx.d3d9v2.direct
          // switched off.
          const uint64_t clears = sm.clearEpoch();
          const uint64_t assets = sm.getAssetReplacer()->getExternalAssetGlobalEpoch();
          const bool resetNow = batch.releaseAll || !scene.active();
          const bool clearChanged = clearEpoch != ~0ull && clears != clearEpoch;
          const bool modsChanged = assetEpoch != ~0ull && assets != assetEpoch;
          const bool resetRequested = clearChanged || modsChanged;
          clearEpoch = clears;
          assetEpoch = assets;
          if (resetNow) {
            // The app thread dropped its handles already (switched off), or the scene lost its entries.
            releaseEverything(sm);
          } else if (resetRequested && !objects.empty()) {
            // Everything goes back to the old path; the entries stay until the app thread's destroys arrive.
            for (auto& [handle, object] : objects) {
              requestRelease(handle, object, ReleaseReason::Reset);
            }
            totals.resets++;
            // The first event that applies ([D3D9v2] direct "resets (...)", for the churn hunt).
            totals.resetsBy[uint32_t(clearChanged ? ResetReason::Clear : ResetReason::Mods)]++;
          } else {
            if (RtxOptionManager::isDrawcallTranslationInvalid() && !objects.empty()) {
              // As the old path translates every draw once more: Direct objects are prepared and described again in
              // place at their next draw. Pending ones are not built yet, so they go back to the old path.
              bool anyDirect = false;
              for (auto& [handle, object] : objects) {
                if (object.pending) {
                  requestRelease(handle, object, ReleaseReason::Reset);
                } else {
                  anyDirect = true;
                }
              }
              if (anyDirect) {
                ++reprepareEpoch;
                channel->requestReprepare();
                totals.reprepares++;
              }
            }
            if (v4WasOn && !D3D9V2Options::directReplacements()) {
              // V4 switched off: replaced objects go back to the old path.
              for (auto& [handle, object] : objects) {
                if (object.replaced) {
                  requestRelease(handle, object, ReleaseReason::Reset);
                }
              }
            }
          }
          v4WasOn = D3D9V2Options::directReplacements();

          for (const uint64_t handle : batch.destroys) {
            eraseObject(sm, handle);
            fallbacks.erase(handle);  // R4: the app thread heard of the rejection
          }
          for (const FrameBatch::Move& move : batch.moves) {
            const auto found = objects.find(move.handle);
            if (found != objects.end() && found->second.skybox3d) {
              found->second.skyTransforms.objectToWorld = move.objectToWorld;  // R8b: sky space, reprojected below
              found->second.frameMoved = frame;
            } else if (found != objects.end()) {
              scene.setTransform(move.handle, move.objectToWorld);
              found->second.objectToWorld = move.objectToWorld;
              found->second.frameMoved = frame;
              // V4: the replacement's lights follow the object.
              if (found->second.lightsCreated) {
                moveLights(lightManager, found->second);
              }
            }
          }
          // R7: decal order after this frame's builds (a rebuilt entry has taken its creation order meanwhile).
          for (const FrameBatch::SortOrder& order : batch.sortOrders) {
            if (objects.find(order.handle) != objects.end()) {
              scene.setSortOrder(order.handle, order.order);
            }
          }
          // Pending objects drawn unchanged (on the old path until the app thread hears they were accepted): kept alive.
          for (const uint64_t handle : batch.pendingSeen) {
            const auto found = objects.find(handle);
            if (found != objects.end()) {
              found->second.frameLastSeen = frame;
            }
          }
          std::vector<XXH64_hash_t> fogSeen;
          const auto feedSideEffects = [&](const Object& object) {
            // What submitDrawState would have done for the draw: mesh-hash tracking and fog discovery.
            sm.trackMeshHash(object.assetHash);
            if (object.legacyHash0 != 0) {
              sm.trackMeshHash(object.legacyHash0);
            }
            if (object.legacyHash1 != 0) {
              sm.trackMeshHash(object.legacyHash1);
            }
            if (object.fogHash != 0 && std::find(fogSeen.begin(), fogSeen.end(), object.fogHash) == fogSeen.end()) {
              fogSeen.push_back(object.fogHash);
            }
          };
          for (const uint64_t handle : batch.pendingSeen) {
            const auto found = objects.find(handle);
            if (found != objects.end()) {
              feedSideEffects(found->second);
            }
          }
          for (const uint64_t handle : batch.seen) {
            const auto found = objects.find(handle);
            if (found == objects.end()) {
              continue;
            }
            Object& object = found->second;
            object.frameLastSeen = frame;
            feedSideEffects(object);
            // V4: the first frame the old path did not draw the object, its replacement instance is collected
            // (handedToScene) and the scene's lights take over.
            if (object.replaced && !object.lightsCreated && !object.pending) {
              createLights(lightManager, object);
              totals.lightsCreated += object.lights.size();
            }
            ensureGraphs(sm, ctx, handle, object);
          }
          // A4: an object drawn this frame without its merged pass is the first draw's alone again, as on the old path.
          for (auto it = mergedPasses.begin(); it != mergedPasses.end();) {
            const auto found = objects.find(*it);
            if (found == objects.end()) {
              it = mergedPasses.erase(it);
            } else if (found->second.frameLastSeen == frame && found->second.frameMergedPass != frame) {
              scene.setMergedPass(*it, nullptr);
              it = mergedPasses.erase(it);
            } else {
              ++it;
            }
          }
          // R4: rejected first-draw handovers the app thread still skips are drawn by the old path from their kept state;
          // fallbacks not seen for 2 frames (the app thread draws them itself again) are dropped.
          if (!fallbacks.empty()) {
            for (const uint64_t handle : batch.seen) {
              const auto found = fallbacks.find(handle);
              if (found == fallbacks.end() || found->second.lastFrame == frame) {
                continue;
              }
              found->second.lastFrame = frame;
              sm.submitDrawState(Rc<DxvkContext>(ctx), found->second.state, nullptr);
              totals.fallbackDraws++;
            }
            for (auto it = fallbacks.begin(); it != fallbacks.end();) {
              it = frame - it->second.lastFrame > 2 ? fallbacks.erase(it) : std::next(it);
            }
          }
          if (batch.frameEvents) {
            // R10: the frame's fog in the app thread's draw order (old-path and scene-owned draws alike); submitDrawState
            // no longer discovers it (setFrontendFogDiscovery, from the next frame).
            for (const FogState& fog : batch.fogOrder) {
              const auto draw = fogDraws.find(fog.getHash());
              sm.discoverFog(fog, draw != fogDraws.end() ? &draw->second : nullptr);
            }
          } else {
            for (const XXH64_hash_t fog : fogSeen) {
              const auto draw = fogDraws.find(fog);
              if (draw != fogDraws.end()) {
                sm.discoverFogState(draw->second);
              }
            }
          }
          sm.setFrontendFogDiscovery(batch.frameEvents);

          // R8b (rtx.d3d9v2.skybox3d): 3D-skybox objects follow the main camera; reprojected every frame, as tryHandleSky's
          // delayed submit did.
          for (auto& [handle, object] : objects) {
            if (!object.skybox3d || object.pending || object.releasing) {
              continue;
            }
            DrawCallTransforms transforms = object.skyTransforms;
            if (ctx->reprojectSkyToMain(transforms, false)) {
              scene.setTransform(handle, transforms.objectToWorld);
              object.objectToWorld = transforms.objectToWorld;
              totals.skyboxMoves++;
            }
          }

          // Builds this frame's descriptions unless a host's DrawRetainedInstances already ran the scene's frame.
          scene.frontendFrame(ctx);

          std::vector<uint64_t> dead;
          for (auto& [handle, object] : objects) {
            switch (scene.entryState(handle)) {
            case RetainedScene::EntryState::Direct:
              if (object.pending) {
                object.pending = false;
                // Aging starts now: the app thread may not have reported this object seen while it was pending.
                object.frameLastSeen = frame;
                totals.accepted++;
                totals.replacedAccepted += object.replaced ? 1 : 0;
                channel->push(handle, Channel::Kind::Accepted);
              }
              break;
            case RetainedScene::EntryState::Pending:
              break;
            default:
              if (object.pending) {
                // Never accepted: the app thread still draws it on the old path.
                dead.push_back(handle);
                reject(handle, "scene");
              } else {
                // Rebuilt later (lane options, texture cache, scene clear) and no longer direct.
                requestRelease(handle, object, ReleaseReason::Rebuilt);
              }
              break;
            }
          }

          // Direct objects the game stopped drawing age out like replacement instances (anti-culling included).
          // As the old path: anti-culling only when the game sets a view matrix (SceneManager::prepareSceneData).
          translator::LifetimeFrame lifetime = translator::LifetimeFrame::read(
            sm.getCamera(), sm.isAntiCullingSupported(), objects.size() >= RtxOptions::AntiCulling::Object::numObjectsToKeep());
          lifetime.currentFrame = frame;
          lifetime.keptCutFrame = sm.m_lastKeptCutFrame;
          const bool logObjects = D3D9V2Options::log();
          std::vector<uint64_t> aged;
          for (auto& [handle, object] : objects) {
            if (object.pending || object.releasing || object.frameLastSeen == frame) {
              continue;
            }
            translator::UnseenObject unseen;
            unseen.frameLastSeen = object.frameLastSeen;
            unseen.frameCreated = object.frameCreated;
            unseen.categories = object.categories;
            unseen.wasMovingWhenLastSeen = object.frameMoved == object.frameLastSeen;
            unseen.skinned = object.skinned || object.nonMainCamera;  // R8: both exempt from anti-culling
            unseen.geometryBounds = &object.bounds;
            // V4: replacement lights keep the object alive as they keep its replacement instance (light anti-culling).
            unseen.lightBounds = object.replaced ? &object.lightBounds : nullptr;
            unseen.objectToWorld = &object.objectToWorld;
            if (!translator::LifetimePolicy::isStale(unseen, lifetime) || translator::LifetimePolicy::keepUnseen(unseen, lifetime)) {
              continue;
            }
            requestRelease(handle, object, ReleaseReason::Aged);
            if (logObjects) {
              aged.push_back(handle);
            }
          }
          if (logObjects) {
            logFrameObjects("built", frameBuilt, frame, builtLogLines);
            logFrameObjects("aged", aged, frame, agedLogLines);
            logWorldUIObjects(scene, frame);
          }
          frameBuilt.clear();
          for (const uint64_t handle : dead) {
            eraseObject(sm, handle);
          }

          if (D3D9V2Options::materialTranslator() && RtxOptions::CoreMaterial::selfCheck()) {
            logParity(frame);
          }

          if (D3D9V2Options::log() && ++logFrames >= kLogIntervalFrames) {
            uint32_t pending = 0, releasing = 0, replaced = 0, lights = 0;
            for (const auto& [handle, object] : objects) {
              pending += object.pending ? 1 : 0;
              releasing += object.releasing ? 1 : 0;
              replaced += object.replaced ? 1 : 0;
              lights += uint32_t(object.lights.size());
            }
            std::string reasons;
            for (const auto& [reason, count] : rejectReasons) {
              reasons += str::format(" ", reason, "=", count);
            }
            // objects / pending / releasing / replaced / lights: now; the rest: totals over the interval. released and
            // resets are split by cause (the accept / release churn hunt).
            Logger::info(str::format(
              "[D3D9v2] direct (totals over ", logFrames, " frames) objects=", objects.size(), " pending=", pending,
              " releasing=", releasing, " replaced=", replaced, " lights=", lights,
              " handovers=", totals.handovers, " accepted=", totals.accepted, " (replaced ", totals.replacedAccepted, ")",
              " rejected=", totals.rejected,
              " released=", totals.released, " (aged ", totals.releasedBy[uint32_t(ReleaseReason::Aged)],
              " rebuilt ", totals.releasedBy[uint32_t(ReleaseReason::Rebuilt)],
              " reset ", totals.releasedBy[uint32_t(ReleaseReason::Reset)], ")",
              " resets=", totals.resets, " (clear ", totals.resetsBy[uint32_t(ResetReason::Clear)],
              " mods ", totals.resetsBy[uint32_t(ResetReason::Mods)],
              " off ", totals.resetsBy[uint32_t(ResetReason::Off)], ")", " reprepares=", totals.reprepares,
              " lightsCreated=", totals.lightsCreated, " cameraOnly=", totals.cameraOnly,
              " fallbackDraws=", totals.fallbackDraws, " (live ", fallbacks.size(), ")", " updated=", totals.updated,
              " unchangedUpdates=", totals.unchangedUpdates, " mergedPasses=", totals.mergedPasses,
              " rasterMisses=", totals.rasterMisses, " skyboxMoves=", totals.skyboxMoves, " skyboxRetries=", totals.skyboxRetries,
              " terrain: bakes=", totals.terrainBakes, " misses=", totals.terrainMisses, " fallbacks=", totals.terrainFallbacks,
              " rematerials=", totals.terrainRematerials,
              " cameraCheck=", totals.cameraMismatches, "/", totals.cameraChecks, " (first: ", firstCameraMismatch, ")",
              " | rejected:", reasons.empty() ? std::string(" none") : reasons,
              " | lights (R9): game=", lightTotals.game, " replaced=", lightTotals.replaced, " entries=", lightTable.size(),
              " created=", lightTotals.created, " collected=", lightTotals.collected));
            totals = Totals();
            lightTotals = LightTotals();
            builtLogLines = 0;
            agedLogLines = 0;
            worldUILogLines = 0;
            firstCameraMismatch = "none";
            rejectReasons.clear();
            logFrames = 0;
          }
        }
      };

      D3D9Frontend::D3D9Frontend()
        : m_impl(std::make_shared<Impl>()) {
      }

      void D3D9Frontend::commit(RtxContext* ctx, const DrawParameters& params, DrawCallState& drawCallState,
                                CommitMode mode, uint64_t handle) {
        if (frameEventsActive()) {
          m_impl->keepFogDraw(drawCallState);  // R10
        }
        switch (mode) {
        case CommitMode::Handover:
          m_impl->handover(ctx, params, drawCallState, handle, false);
          return;
        case CommitMode::HandoverFirstDraw:
          m_impl->handover(ctx, params, drawCallState, handle, true);
          return;
        case CommitMode::Update:
          m_impl->handover(ctx, params, drawCallState, handle, true, true);
          return;
        case CommitMode::MergePass:
          m_impl->mergePass(ctx, params, drawCallState, handle);
          return;
        case CommitMode::RasterOnly:
          // R8b: the scene owns the object; only the game's raster draw into the sky matte / probe is replayed.
          // R8b: a terrain object's bake, else the sky matte / probe replay, else a 3D-skybox object's transforms.
          if (!m_impl->terrainDraw(ctx, params, drawCallState, handle) &&
              !m_impl->rejectedTerrainDraw(ctx, params, drawCallState, handle) &&
              !(drawCallState.cameraType == CameraType::Sky && ctx->rasterizeSkyDraw(params, drawCallState)) &&
              !m_impl->skyboxDraw(handle, drawCallState)) {
            m_impl->totals.rasterMisses++;
          }
          return;
        case CommitMode::CameraOnly: {
          // The scene owns the object; only the camera the old path derives from draws is fed.
          const MaterialData* overrideMaterial = nullptr;
          ctx->prepareGeometryForRT(params, drawCallState, overrideMaterial);
          m_impl->totals.cameraOnly++;
          return;
        }
        default:
          // The old path (the fallback until D3D9 v2 matches it; deleted in Phase 6). handle: a pending object's draw.
          if (handle != 0) {
            m_impl->oldPathDraw(ctx, params, drawCallState, handle);
            return;
          }
          m_impl->commitOldPath(ctx, params, drawCallState);
          return;
        }
      }

      void D3D9Frontend::lights(RtxContext* ctx, const D3DLIGHT9* lights, uint32_t count, bool translate) {
        if (translate) {
          m_impl->translateLights(ctx, lights, count);
        } else {
          ctx->addLights(lights, count);  // the old path
        }
      }

      void D3D9Frontend::flush(RtxContext* ctx, const std::shared_ptr<FrameBatch>& batch) {
        m_impl->collectLights(ctx);  // R9
        if (batch != nullptr) {
          m_impl->flush(ctx, *batch);
        } else if (!m_impl->objects.empty()) {
          ctx->getSceneManager().retainedScene().frontendFrame(ctx);
        }
      }

      // ---- app thread ------------------------------------------------------------------------------------------

      struct D3D9FrontendApp::Impl {
        // V1 statistics (rtx.d3d9v2.log): per object key, the draws of that key in draw order.
        struct Seen {
          XXH64_hash_t inputs = 0;
          XXH64_hash_t inputsNoVsConstants = 0;
          bool versioned = false;
        };
        std::unordered_map<XXH64_hash_t, std::vector<Seen>> previous;
        std::unordered_map<XXH64_hash_t, std::vector<Seen>> current;

        struct Totals {
          uint64_t draws = 0;
          uint64_t versioned = 0;
          uint64_t userPointer = 0;
          uint64_t programmableVS = 0;
          uint64_t capture = 0;
          uint64_t skinned = 0;
          uint64_t repeat = 0;            // same object, same occurrence, same inputs as last frame
          uint64_t repeatNoVsConst = 0;   // the same, ignoring VS constants
          uint64_t newObject = 0;         // no draw of this object key (and occurrence) last frame
          // rtx.d3d9v2.direct
          uint64_t skipped = 0;           // direct, unchanged: nothing queued
          uint64_t moved = 0;             // direct, moved: setTransform
          uint64_t changed = 0;           // direct, changed: handed back
          uint64_t handovers = 0;
          uint64_t cameraOnly = 0;
          uint64_t pendingRepeats = 0;    // pending, unchanged: still on the old path
          uint64_t cameraFeeds = 0;       // R2: cameras sent (first draw of a camera type in a frame)
          // R1 (rtx.d3d9v2.earlyChangeTest)
          uint64_t preKeyHits = 0;        // skipped before preparation, unchanged
          uint64_t preKeyMoves = 0;       // skipped before preparation, moved
          uint64_t preKeyMisses = 0;      // pre-checked, prepared
          uint64_t selfChecks = 0;        // hits prepared in full for the self-check
          uint64_t preKeyMismatches = 0;  // self-check found a change the pre-key missed
          uint64_t preKeyNs = 0;          // time building pre-keys (D3D9 thread)
          uint64_t preKeyTimed = 0;
          uint64_t churnBackoffs = 0;     // R4: objects handed back as changed within 8 frames of their handover
          uint64_t updates = 0;           // R5: changed draws of Direct objects updated in place
          uint64_t sortOrders = 0;        // R7: decal orders sent
          uint64_t rasterOnly = 0;        // R8b: raster-service draws of scene-owned objects (sky)
          uint64_t derivedChanges = 0;    // draws with unchanged inputs whose categories or camera type changed
          uint64_t reprepared = 0;        // unchanged draws of Direct objects updated after an option change
          uint64_t duplicates = 0;        // A4: repeated draws of an object already drawn this frame
          uint64_t skinDeferrals = 0;     // new skinned objects held back a frame to take over an undrawn character
          uint64_t skinPairs = 0;         // ... that took it over
          uint64_t skinCancels = 0;       // ... whose partner was drawn later in the frame (the held-back draw was not drawn)
          uint64_t skinAmbiguous = 0;     // held back with more than one candidate
          uint64_t skinEvents = 0;        // skinned lifecycle events (skinEvent), logged or not
          uint64_t skinSwaps = 0;         // skinned draws that took a slot whose last material or position differs
          uint64_t matchMaterialRejects = 0;  // slot candidates skipped for another material (lookupNearest)
          uint64_t resetPairs = 0;        // new objects after a device Reset that took over a scene object of the same draw
        } totals;
        uint32_t mismatchLogs = 0;        // R1: [D3D9v2] preKeyMismatch lines written (first 20)
        // Skinned lifecycle lines written in this log interval.
        static constexpr uint32_t kSkinEventLinesPerInterval = 40;
        uint32_t skinEventLines = 0;

        // rtx.d3d9v2.log: World UI draws (in-world screens), one line per draw with what the frontend did with it, the
        // object's handle, its world translation and its camera's vertical field of view. At most kWorldUILinesPerInterval
        // lines per log interval.
        static constexpr uint32_t kWorldUILinesPerInterval = 96;
        uint32_t worldUILines = 0;
        void worldUIEvent(uint32_t categories, const char* event, uint64_t handle, const Matrix4& objectToWorld,
                          const Matrix4& viewToProjection) {
          if (!D3D9V2Options::log() || !CategoryFlags(categories).test(InstanceCategories::WorldUI) ||
              worldUILines >= kWorldUILinesPerInterval) {
            return;
          }
          ++worldUILines;
          const Vector3 p = objectToWorld[3].xyz();
          const float scaleY = std::abs(viewToProjection[1][1]);
          const float fov = scaleY > 0.f ? 2.f * std::atan(1.f / scaleY) : 0.f;
          Logger::info(str::format("[D3D9v2] worldUI draw frame=", frame, " ", event, " handle=0x", std::hex, handle,
                                   std::dec, " pos=(", p.x, ", ", p.y, ", ", p.z, ") fov=", fov));
        }

        // R8 step 0: why draws of objects on the old path were not handed over at their first-draw check, per class. A draw
        // counts once in every class it is in (not exclusive). Collected only while rtx.d3d9v2.log is on.
        enum NotEligible : uint32_t {
          kNeInputs, kNeBackoff, kNeCameraViewModel, kNeCameraPortal, kNeCameraSky, kNeCameraRenderToTexture,
          kNeCameraUnknown, kNeSky, kNeTerrain, kNeParticle, kNeParticleEmitter, kNeBeam, kNeDecal, kNePlayerModel,
          kNeWorldUI, kNeWorldMatte, kNeAnimatedWater, kNeHairCards, kNeLightConversion, kNeRenderTarget, kNeCount
        };
        static constexpr const char* kNotEligibleNames[kNeCount] = {
          "inputs", "backoff", "camViewModel", "camPortal", "camSky", "camRenderToTexture",
          "camUnknown", "sky", "terrain", "particle", "particleEmitter", "beam", "decal", "playerModel",
          "worldUI", "worldMatte", "animatedWater", "hairCards", "lightConversion", "renderTarget"
        };
        uint64_t notEligible[kNeCount] = {};

        void countNotEligible(bool eligible, bool backoff, uint32_t cameraType, const PreparedState& prepared, bool services,
                              bool unordered) {
          if (!eligible) {
            notEligible[kNeInputs]++;
          }
          if (backoff) {
            notEligible[kNeBackoff]++;
          }
          if (!acceptedCamera(cameraType)) {
            notEligible[cameraType == CameraType::ViewModel ? kNeCameraViewModel
                      : cameraType == CameraType::Portal0 || cameraType == CameraType::Portal1 ? kNeCameraPortal
                      : cameraType == CameraType::Sky ? kNeCameraSky
                      : cameraType == CameraType::RenderToTexture ? kNeCameraRenderToTexture : kNeCameraUnknown]++;
          }
          const CategoryFlags c(prepared.categories);
          const auto note = [&](bool in, NotEligible what) { notEligible[what] += in ? 1 : 0; };
          note(!skyServiceActive() && c.test(InstanceCategories::Sky), kNeSky);
          note(!terrainActive() && TerrainBaker::needsTerrainBaking() && c.test(InstanceCategories::Terrain), kNeTerrain);
          note(!unordered && c.test(InstanceCategories::Particle), kNeParticle);
          note(!services && c.test(InstanceCategories::ParticleEmitter), kNeParticleEmitter);
          note(!unordered && c.test(InstanceCategories::Beam), kNeBeam);
          note(!unordered && c.any(DECAL_CATEGORY_FLAGS), kNeDecal);
          note(!services && c.any(InstanceCategories::ThirdPersonPlayerModel, InstanceCategories::ThirdPersonPlayerBody), kNePlayerModel);
          note(!services && c.test(InstanceCategories::WorldUI), kNeWorldUI);
          note(!services && c.test(InstanceCategories::WorldMatte), kNeWorldMatte);
          note(!services && c.test(InstanceCategories::AnimatedWater), kNeAnimatedWater);
          note(!services && c.test(InstanceCategories::HairCards), kNeHairCards);
          note(!services && prepared.convertsToLight, kNeLightConversion);
          note(!renderTargetsActive() && prepared.raytracedRenderTarget, kNeRenderTarget);
        }

        // R1: the object an early check looked up, handed to decide when the draw is prepared after all.
        struct Pending {
          bool valid = false;
          XXH64_hash_t objectKey = 0;      // the draw it belongs to (a draw the preparation then ignored never calls decide)
          XXH64_hash_t key = 0;
          EarlyKey early;
          bool wouldHit = false;           // self-check: the pre-key matched, the draw is prepared to verify it
        } pending;
        CameraClassifier cameras;         // R2 (rtx.d3d9v2.explicitCamera)
        uint32_t frames = 0;
        bool wasActive = false;

        // rtx.d3d9v2.direct
        enum class State : uint8_t { Legacy, Pending, Direct };
        struct Object {
          XXH64_hash_t inputs = 0;
          XXH64_hash_t inputsNoWorld = 0;
          uint64_t handle = 0;
          uint32_t streak = 0;
          uint32_t lastFrame = 0;
          uint32_t backoffUntil = 0;
          uint32_t rejects = 0;
          State state = State::Legacy;
          bool hasLast = false;
          uint32_t handoverFrame = 0;     // R4: frame of the last handover (churn guard)
          int32_t sortOrder = -1;         // R7: the decal order last sent to the scene (-1 = none)
          bool rasterService = false;     // R8b: a sky-camera object: its draw is replayed every frame (RasterOnly)
          // The categories and camera type of the last prepared draw. A draw that differs in either is not a repeat.
          uint32_t derivedCategories = 0;
          uint32_t derivedCamera = Decision::kNoCamera;
          bool derivedTerrainBaking = false;
          bool derivedFreeCamera = false;  // rtx.camera.enableFreeCamera at the last prepared draw
          // The re-preparation epoch its scene copy was described in. A Direct object of an older one is updated at its
          // next draw.
          uint32_t reprepareEpoch = 0;
          // Fixed-function skinned draws: the class skinnedByClass lists the object under (0 = not listed), and its last
          // draw's bone centroid, drawn counts, bone extent, used bone range, material and object key.
          XXH64_hash_t skinClass = 0;
          Vector3 skinPosition;
          uint64_t skinCounts = 0;
          float skinRadius = 0.f;
          uint32_t skinFirstBone = 0;
          uint32_t skinBoneCount = 0;
          XXH64_hash_t skinMaterial = 0;
          XXH64_hash_t skinObjectKey = 0;
          XXH64_hash_t resetSignature = 0;  // D3D9DrawInputs::resetSignature of the last draw (0 = coarse-keyed)
          // R1: the last full preparation's pre-keys, record and derived state.
          bool hasEarly = false;
          XXH64_hash_t preKey = 0;
          XXH64_hash_t preKeyNoWorld = 0;
          D3D9DrawInputs last;
          PreparedState prepared;
        };
        std::unordered_map<XXH64_hash_t, Object> objects;       // by object key + slot (lookupNearest)
        std::unordered_map<uint64_t, XXH64_hash_t> byHandle;    // pending and direct objects
        std::unordered_map<XXH64_hash_t, uint32_t> occurrence;  // this frame, per object key
        // lookupNearest: per key, the last draw of each slot (the index in the object's lookup key). Slots outlive frames
        // without a draw, so an object that was culled for a while finds its own slot again, and are freed by endFrame
        // once their object is forgotten.
        struct PositionSlot {
          Vector3 position;
          XXH64_hash_t signature = 0;
          XXH64_hash_t material = 0;  // the legacy material identity of its last prepared draw (0 = not known)
          uint32_t lastFrame = 0;  // the slot is taken in this frame when lastFrame equals it
          bool valid = false;
        };
        std::unordered_map<XXH64_hash_t, std::vector<PositionSlot>> slots;
        std::unordered_map<XXH64_hash_t, uint32_t> slotCursors;  // this frame, per key: no free slot is below it
        // A4: this frame's draws (frameDrawKey) to the object of the first one, by inputs and by pre-key.
        std::unordered_map<XXH64_hash_t, XXH64_hash_t> frameDraws;
        std::unordered_map<XXH64_hash_t, XXH64_hash_t> framePreKeys;
        // A4: this frame's draws by pass key (frameDrawKey of passInputsOf) to the object of the first one, and whether a
        // later pass was merged into it.
        struct FramePass {
          XXH64_hash_t object = 0;
          bool merged = false;
        };
        std::unordered_map<XXH64_hash_t, FramePass> framePasses;
        // Skinned objects by material and primitive type (skinClassOf), and this frame's new skinned objects waiting to
        // take over an undrawn one: the waiting draw's object key and the undrawn object's.
        std::unordered_map<XXH64_hash_t, std::vector<XXH64_hash_t>> skinnedByClass;
        struct SkinPairing {
          XXH64_hash_t waiting;
          XXH64_hash_t partner;
        };
        std::vector<SkinPairing> skinPairings;
        // After a device Reset: the scene-owned objects by resetSignature, each with its last draw frame at the Reset (it
        // is a candidate only while it has not been drawn since). Kept for kResetFrames frames.
        struct ResetCandidate {
          XXH64_hash_t key;
          uint32_t lastFrame;
        };
        static constexpr uint32_t kResetFrames = 16;
        std::unordered_multimap<XXH64_hash_t, ResetCandidate> resetCandidates;
        uint32_t resetFrame = 0;
        std::shared_ptr<FrameBatch> batch = std::make_shared<FrameBatch>();
        bool batchTaken = false;
        std::shared_ptr<Channel> channel;
        uint32_t reprepareEpoch = 0;      // Channel::reprepare requests heard so far
        uint32_t frame = 0;
        uint64_t nextHandle = 1;
        Matrix4 queuedView;
        Matrix4 queuedProjection;
        bool queuedCameraValid = false;
        uint32_t decalRank = 0;           // R7: Direct decals drawn so far this frame
        std::vector<XXH64_hash_t> fogSeenThisFrame;  // R10

        // R10: the draw's fog, once per frame, in draw order.
        void noteFog(const FogInputs& in) {
          if (in.mode == D3DFOG_NONE || !frontends::d3d9::frameEventsActive()) {
            return;
          }
          const FogState fog = toFogState(in);
          const XXH64_hash_t hash = fog.getHash();
          if (std::find(fogSeenThisFrame.begin(), fogSeenThisFrame.end(), hash) != fogSeenThisFrame.end()) {
            return;
          }
          fogSeenThisFrame.push_back(hash);
          batch->fogOrder.push_back(fog);
        }

        void resetStats() {
          previous.clear();
          current.clear();
          totals = Totals();
          skinEventLines = 0;
          worldUILines = 0;
          frames = 0;
        }

        // The key decide identifies the draw's object by.
        // R5: draws whose data moves or changes size every frame are identified by what they draw (coarse key).
        static bool usesCoarseKey(const D3D9DrawInputs& in) {
          return dynamicGeometryActive() && (in.flags & (D3D9DrawInputs::kDynamicSource | D3D9DrawInputs::kUserPointer));
        }
        static XXH64_hash_t identityKey(const D3D9DrawInputs& in) {
          return usesCoarseKey(in) ? in.coarseKey() : in.objectKey();
        }

        void observe(const D3D9DrawInputs& in) {
          const XXH64_hash_t key = identityKey(in);
          Seen seen;
          seen.inputs = in.inputsKey(true);
          seen.inputsNoVsConstants = in.inputsKey(false);
          seen.versioned = in.versioned();

          std::vector<Seen>& mine = current[key];
          const size_t index = mine.size();
          mine.push_back(seen);

          totals.draws++;
          totals.versioned += seen.versioned ? 1 : 0;
          totals.userPointer += (in.flags & D3D9DrawInputs::kUserPointer) ? 1 : 0;
          totals.programmableVS += (in.flags & D3D9DrawInputs::kProgrammableVS) ? 1 : 0;
          totals.capture += (in.flags & D3D9DrawInputs::kVertexCapture) ? 1 : 0;
          totals.skinned += (in.flags & D3D9DrawInputs::kSkinned) ? 1 : 0;

          const auto it = previous.find(key);
          if (it == previous.end() || index >= it->second.size()) {
            totals.newObject++;
            return;
          }
          // Unversioned inputs compare equal (version 0) whatever their data, and fixed-function bones are not in the
          // record: neither can be a repeat.
          if (!seen.versioned || (in.flags & D3D9DrawInputs::kSkinned)) {
            return;
          }
          const Seen& last = it->second[index];
          if (last.inputs == seen.inputs) {
            totals.repeat++;
          }
          if (last.inputsNoVsConstants == seen.inputsNoVsConstants) {
            totals.repeatNoVsConst++;
          }
        }

        void endStatsFrame() {
          previous.swap(current);
          for (auto it = current.begin(); it != current.end();) {
            // Keep the vectors' storage for keys drawn every frame; drop keys not drawn this frame.
            if (previous.find(it->first) == previous.end()) {
              it = current.erase(it);
            } else {
              it->second.clear();
              ++it;
            }
          }
          if (++frames < kLogIntervalFrames) {
            return;
          }
          const double n = double(frames);
          const auto perFrame = [n](uint64_t v) { return uint64_t(double(v) / n + 0.5); };
          // live / direct / pending: this thread's objects at this moment (direct = drawn by the scene, pending =
          // handed over, still drawn on the old path). The per-frame values are rounded averages over the interval, so
          // the interval totals of skipped and handovers are printed too: a handful of skips rounds to 0.
          uint32_t directNow = 0, pendingNow = 0;
          for (const auto& [handle, key] : byHandle) {
            const auto found = objects.find(key);
            if (found != objects.end()) {
              directNow += found->second.state == State::Direct ? 1 : 0;
              pendingNow += found->second.state == State::Pending ? 1 : 0;
            }
          }
          Logger::info(str::format(
            "[D3D9v2] inputs frames=", frames, " (per frame) draws=", perFrame(totals.draws),
            " versioned=", perFrame(totals.versioned), " up=", perFrame(totals.userPointer),
            " programmableVS=", perFrame(totals.programmableVS), " capture=", perFrame(totals.capture),
            " skinned=", perFrame(totals.skinned), " newObject=", perFrame(totals.newObject),
            " repeat=", perFrame(totals.repeat), " repeatNoVsConst=", perFrame(totals.repeatNoVsConst),
            " | direct: live=", byHandle.size(), " (direct ", directNow, " pending ", pendingNow, ")",
            " skipped=", perFrame(totals.skipped), " moved=", perFrame(totals.moved),
            " changed=", perFrame(totals.changed), " handovers=", perFrame(totals.handovers),
            " cameraOnly=", perFrame(totals.cameraOnly), " cameraFeeds=", perFrame(totals.cameraFeeds),
            " pendingRepeats=", perFrame(totals.pendingRepeats),
            " | early: hits=", perFrame(totals.preKeyHits), " moves=", perFrame(totals.preKeyMoves),
            " misses=", perFrame(totals.preKeyMisses), " selfChecks=", totals.selfChecks,
            " preKeyMismatch=", totals.preKeyMismatches,
            " preKeyUs/frame=", str::format(double(totals.preKeyNs) / 1000.0 / n),
            " preKeyNs/draw=", totals.preKeyTimed != 0 ? totals.preKeyNs / totals.preKeyTimed : 0,
            " | churnBackoffs=", totals.churnBackoffs, " updates=", perFrame(totals.updates),
            " sortOrders=", perFrame(totals.sortOrders), " rasterOnly=", perFrame(totals.rasterOnly),
            " derivedChanges=", totals.derivedChanges, " reprepared=", totals.reprepared,
            " duplicates=", perFrame(totals.duplicates), " skinPairs=", totals.skinPairs, "/", totals.skinDeferrals,
            " skinCancels=", totals.skinCancels, " skinAmbiguous=", totals.skinAmbiguous, " skinSwaps=", totals.skinSwaps,
            " matchMaterialRejects=", totals.matchMaterialRejects,
            " skinEvents=", skinEventLines, "/", totals.skinEvents, " (logged/all)",
            " resetPairs=", totals.resetPairs,
            " | interval totals: skipped=", totals.skipped, " handovers=", totals.handovers, " changed=", totals.changed));
          // R8 step 0: draws per frame not handed over at their first-draw check, per class (non-exclusive).
          std::string notEligibleLine;
          for (uint32_t i = 0; i < kNeCount; ++i) {
            if (notEligible[i] != 0) {
              notEligibleLine += str::format(" ", kNotEligibleNames[i], "=", perFrame(notEligible[i]));
            }
            notEligible[i] = 0;
          }
          if (!notEligibleLine.empty()) {
            Logger::info(str::format("[D3D9v2] notEligible (per frame, a draw counts in each class it is in):", notEligibleLine));
          }
          totals = Totals();
          skinEventLines = 0;
          worldUILines = 0;
          frames = 0;
        }

        // The CS thread derives the camera from queued draws: a skipped draw whose camera differs from the last one
        // queued this frame is queued for that alone.
        bool cameraNeedsFeed(const Matrix4& view, const Matrix4& projection) {
          if (queuedCameraValid && sameMatrix(view, queuedView) && sameMatrix(projection, queuedProjection)) {
            return false;
          }
          noteQueuedCamera(view, projection);
          return true;
        }

        void noteQueuedCamera(const Matrix4& view, const Matrix4& projection) {
          queuedView = view;
          queuedProjection = projection;
          queuedCameraValid = true;
        }

        void dropHandle(Object& object) {
          byHandle.erase(object.handle);
          object.handle = 0;
          object.state = State::Legacy;
          object.streak = 0;
          object.sortOrder = -1;
          object.rasterService = false;
        }

        // R7: a Direct decal's rank among this frame's Direct decals, in draw order (8 bits on the GPU, so it wraps, as the
        // old path's counter does). Sent when it changed, or after a (re)build, which assigns a creation order.
        void noteSortOrder(Object& object, uint32_t categories, bool built) {
          if (!frontends::d3d9::unorderedActive() || !CategoryFlags(categories).any(DECAL_CATEGORY_FLAGS)) {
            return;
          }
          const uint8_t rank = uint8_t(decalRank++ & 0xFFu);
          if (built || object.sortOrder != int32_t(rank)) {
            batch->sortOrders.push_back({ object.handle, rank });
            object.sortOrder = rank;
            totals.sortOrders++;
          }
        }

        // lookupNearest's limits for one draw, and what the slot it took held before.
        struct SlotQuery {
          // LegacyMaterialData::computeIdentityHash of the draw, or 0 when not known (earlyCheck): slots are then
          // compared by position and signature only.
          XXH64_hash_t material = 0;
          // A slot not drawn in the previous frame is taken only this close.
          float staleDistanceSqr = 0.f;
          // A slot whose last material differs is taken only when it was drawn in the previous frame and is this close:
          // the same object, whose material the game changed.
          float otherMaterialDistanceSqr = 0.f;
          // Out: whether a known slot was taken, with its last position and material.
          bool reused = false;
          Vector3 previousPosition;
          XXH64_hash_t previousMaterial = 0;
        };

        // The limits for a draw: a fixed-function skinned draw takes a stale slot or one of another material only within
        // its bone extent (skinPairBound), since another character of the same mesh may have left the view nearby. Other
        // draws take a stale slot within rtx.uniqueObjectDistance and one of another material within a twentieth of it.
        static SlotQuery slotQuery(const D3D9DrawInputs& in) {
          SlotQuery query;
          query.material = in.material;
          const float tight = skinPairBound(in.skinRadius, 0.f);
          query.otherMaterialDistanceSqr = tight * tight;
          query.staleDistanceSqr = (in.flags & D3D9DrawInputs::kSkinned) ? tight * tight : RtxOptions::getUniqueObjectDistanceSqr();
          return query;
        }

        // The object of this draw: a slot of its key, told apart from the other draws of the key by position (the world
        // translation, or the bone centroid for fixed-function skinned draws). Draws of one key can change order between
        // frames, so each takes the free slot whose last position is nearest to its own and within rtx.uniqueObjectDistance,
        // searched within kWindow slots of its occurrence index. A slot of the draw's material ranks above every slot of
        // another material, and SlotQuery limits stale slots and slots of another material. Equal distances (identity
        // world matrices among them) prefer the slot with the same content signature, then the occurrence order. A draw
        // without a candidate takes a slot no known object holds, so it never inherits the history of another object.
        XXH64_hash_t lookupNearest(XXH64_hash_t groupKey, const Vector3& position, XXH64_hash_t signature, SlotQuery& query) {
          constexpr uint32_t kWindow = 32;
          const uint32_t index = occurrence[groupKey]++;
          std::vector<PositionSlot>& known = slots[groupKey];
          const uint32_t count = uint32_t(known.size());
          const auto claimable = [&](uint32_t slot) { return known[slot].valid && known[slot].lastFrame != frame; };
          const auto sameMaterialAs = [&](uint32_t slot) {
            return query.material == 0 || known[slot].material == 0 || known[slot].material == query.material;
          };

          uint32_t chosen = ~0u;
          if (index < count && claimable(index) && known[index].position == position && known[index].signature == signature &&
              sameMaterialAs(index)) {
            chosen = index;
          } else if (count != 0) {
            const float maxDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();
            float best = std::numeric_limits<float>::infinity();
            bool bestSameMaterial = false;
            bool bestSameSignature = false;
            uint32_t bestOffset = ~0u;
            const uint32_t first = index > kWindow ? index - kWindow : 0u;
            const uint32_t last = std::min(count, index + kWindow + 1);
            for (uint32_t slot = first; slot < last; ++slot) {
              if (!claimable(slot)) {
                continue;
              }
              const float distance = lengthSqr(known[slot].position - position);
              const bool recent = known[slot].lastFrame + 1 == frame;
              if (distance > maxDistanceSqr || (!recent && distance > query.staleDistanceSqr)) {
                continue;
              }
              const bool sameMaterial = sameMaterialAs(slot);
              if (!sameMaterial && (!recent || distance > query.otherMaterialDistanceSqr)) {
                totals.matchMaterialRejects++;
                continue;
              }
              const bool sameSignature = known[slot].signature == signature;
              const uint32_t offset = slot > index ? slot - index : index - slot;
              const bool better = chosen == ~0u ||
                (sameMaterial != bestSameMaterial ? sameMaterial
                 : distance != best ? distance < best
                 : sameSignature != bestSameSignature ? sameSignature : offset < bestOffset);
              if (better) {
                best = distance;
                bestSameMaterial = sameMaterial;
                bestSameSignature = sameSignature;
                bestOffset = offset;
                chosen = slot;
              }
            }
          }
          query.reused = false;
          if (chosen == ~0u) {
            uint32_t& cursor = slotCursors[groupKey];
            while (cursor < count && known[cursor].valid) {
              ++cursor;
            }
            chosen = cursor;
          } else {
            query.reused = true;
            query.previousPosition = known[chosen].position;
            query.previousMaterial = known[chosen].material;
          }
          if (chosen >= known.size()) {
            known.resize(chosen + 1);
          }
          PositionSlot& taken = known[chosen];
          taken.position = position;
          taken.signature = signature;
          if (query.material != 0 || !query.reused) {
            taken.material = query.material;
          }
          taken.lastFrame = frame;
          taken.valid = true;
          return slotKey(groupKey, chosen);
        }

        static XXH64_hash_t slotKey(XXH64_hash_t groupKey, uint32_t slot) {
          return XXH3_64bits_withSeed(&slot, sizeof(slot), groupKey);
        }

        // What stays the same when a character's draw moves to other buffers or another key: its material and topology.
        static XXH64_hash_t skinClassOf(const D3D9DrawInputs& in) {
          return XXH3_64bits_withSeed(&in.primitiveType, sizeof(in.primitiveType), in.material) | 1;
        }

        static uint64_t drawnCounts(const D3D9DrawInputs& in) {
          return (uint64_t(in.primitiveCount) << 32) | in.numVertices;
        }

        // rtx.d3d9v2.log: one line per lifecycle event of a fixed-function skinned object, with the object's last draw.
        // At most kSkinEventLinesPerInterval lines per log interval. The interval line counts all events.
        void skinEvent(const char* event, const Object& object, const std::string& detail = std::string()) {
          if (!D3D9V2Options::log()) {
            return;
          }
          totals.skinEvents++;
          if (skinEventLines >= kSkinEventLinesPerInterval) {
            return;
          }
          ++skinEventLines;
          const Vector3& p = object.skinPosition;
          Logger::info(str::format(
            "[D3D9v2] skin ", event, " frame=", frame, " handle=0x", std::hex, object.handle, " material=0x", object.skinMaterial,
            " key=0x", object.skinObjectKey, std::dec, " pos=(", p.x, ", ", p.y, ", ", p.z, ") radius=", object.skinRadius,
            " bones=", object.skinFirstBone, "+", object.skinBoneCount, " counts=", uint32_t(object.skinCounts >> 32), "/",
            uint32_t(object.skinCounts), detail.empty() ? "" : " ", detail));
        }

        // What differs between two skinned objects' last draws (a character whose draw moved to another key).
        static std::string skinDifferences(const Object& a, const Object& b) {
          std::string what;
          const auto add = [&](bool differs, const char* name) {
            if (differs) {
              what += what.empty() ? std::string(name) : std::string(",") + name;
            }
          };
          add(a.skinObjectKey != b.skinObjectKey, "buffers");
          add(a.skinMaterial != b.skinMaterial, "material");
          add(a.skinBoneCount != b.skinBoneCount, "boneCount");
          add(a.skinFirstBone != b.skinFirstBone, "boneRange");
          add(a.skinCounts != b.skinCounts, "counts");
          return what.empty() ? std::string("none") : what;
        }

        void noteSkinned(Object& object, XXH64_hash_t key, const D3D9DrawInputs& in) {
          const bool known = object.skinClass != 0;
          const bool changed = known &&
            (object.skinMaterial != in.material || object.skinBoneCount != in.skinBoneCount ||
             object.skinFirstBone != in.skinFirstBone || object.skinCounts != drawnCounts(in));
          std::string what;
          if (changed) {
            what = str::format("was material=0x", std::hex, object.skinMaterial, std::dec, " bones=", object.skinFirstBone,
                               "+", object.skinBoneCount, " counts=", uint32_t(object.skinCounts >> 32), "/",
                               uint32_t(object.skinCounts));
          }
          object.skinPosition = in.skinPosition;
          object.skinCounts = drawnCounts(in);
          object.skinRadius = in.skinRadius;
          object.skinFirstBone = in.skinFirstBone;
          object.skinBoneCount = in.skinBoneCount;
          object.skinMaterial = in.material;
          object.skinObjectKey = identityKey(in);
          const XXH64_hash_t skinClass = skinClassOf(in);
          if (object.skinClass != skinClass) {
            object.skinClass = skinClass;
            skinnedByClass[skinClass].push_back(key);
          }
          if (changed) {
            skinEvent("changed", object, what);
          }
        }

        // findSkinnedPartner's result: the chosen object (0 = none), how many qualified, and why the others did not.
        struct SkinSearch {
          XXH64_hash_t chosen = 0;
          float distance = 0.f;
          float bound = 0.f;
          uint32_t qualified = 0;
          uint32_t drawnThisFrame = 0;  // drawn in this frame already
          uint32_t notRecent = 0;       // not drawn in the previous frame
          uint32_t tooFar = 0;
          uint32_t claimed = 0;
          uint32_t notDirect = 0;
          float nearestTooFar = -1.f;   // the nearest candidate that was too far (-1 = none)

          std::string describe() const {
            return str::format("candidates: qualified=", qualified, " drawnThisFrame=", drawnThisFrame, " notRecent=", notRecent,
                               " tooFar=", tooFar, " claimed=", claimed, " notDirect=", notDirect,
                               " nearestTooFar=", nearestTooFar);
          }
        };

        // A character is paired only with one this close: the larger bone extent of the two (how far a body's parts
        // reach from its centroid), at least a twentieth of rtx.uniqueObjectDistance and at most all of it.
        static float skinPairBound(float radiusA, float radiusB) {
          const float unique = std::sqrt(RtxOptions::getUniqueObjectDistanceSqr());
          return std::min(unique, std::max(std::max(radiusA, radiusB), 0.05f * unique));
        }

        // The nearest scene-owned skinned object of the draw's class that was drawn in the previous frame but not yet in
        // this one, is not claimed yet and lies within skinPairBound. Drops list entries whose object is gone or is
        // listed under another class.
        SkinSearch findSkinnedPartner(const D3D9DrawInputs& in, XXH64_hash_t self) {
          SkinSearch search;
          const XXH64_hash_t skinClass = skinClassOf(in);
          const auto list = skinnedByClass.find(skinClass);
          if (list == skinnedByClass.end()) {
            return search;
          }
          std::vector<XXH64_hash_t>& keys = list->second;
          size_t kept = 0;
          for (size_t i = 0; i < keys.size(); ++i) {
            const XXH64_hash_t candidateKey = keys[i];
            const auto found = objects.find(candidateKey);
            if (found == objects.end() || found->second.skinClass != skinClass) {
              continue;
            }
            keys[kept++] = candidateKey;
            const Object& candidate = found->second;
            if (candidateKey == self) {
              continue;
            }
            if (candidate.state != State::Direct) {
              search.notDirect++;
              continue;
            }
            if (candidate.lastFrame == frame) {
              search.drawnThisFrame++;
              continue;
            }
            if (candidate.lastFrame + 1 != frame) {
              search.notRecent++;
              continue;
            }
            if (std::any_of(skinPairings.begin(), skinPairings.end(), [&](const SkinPairing& p) { return p.partner == candidateKey; })) {
              search.claimed++;
              continue;
            }
            const float distance = length(candidate.skinPosition - in.skinPosition);
            const float bound = skinPairBound(in.skinRadius, candidate.skinRadius);
            if (distance > bound) {
              search.tooFar++;
              if (search.nearestTooFar < 0.f || distance < search.nearestTooFar) {
                search.nearestTooFar = distance;
              }
              continue;
            }
            search.qualified++;
            if (search.chosen == 0 || distance < search.distance) {
              search.chosen = candidateKey;
              search.distance = distance;
              search.bound = bound;
            }
          }
          keys.resize(kept);
          return search;
        }

        // At the end of the frame, a waiting draw whose partner stayed undrawn takes the partner over: the partner's
        // object moves to the waiting draw's key, so the next draw updates the same scene object in place.
        void pairSkinned() {
          for (const SkinPairing& pairing : skinPairings) {
            const auto partner = objects.find(pairing.partner);
            const auto waiting = objects.find(pairing.waiting);
            if (partner == objects.end() || waiting == objects.end()) {
              continue;
            }
            if (partner->second.state != State::Direct || partner->second.lastFrame == frame ||
                waiting->second.state != State::Legacy) {
              // The held-back draw was not drawn in this frame. It is handed over at its next draw.
              totals.skinCancels++;
              skinEvent("cancelled", waiting->second,
                        str::format("partner=0x", std::hex, partner->second.handle, std::dec,
                                    partner->second.lastFrame == frame ? " drawn later in the frame" : " no longer scene-owned"));
              continue;
            }
            const float distance = length(partner->second.skinPosition - waiting->second.skinPosition);
            const std::string differences = skinDifferences(partner->second, waiting->second);
            Object moved = std::move(partner->second);
            objects.erase(partner);
            moved.lastFrame = frame;
            moved.skinPosition = waiting->second.skinPosition;
            moved.skinCounts = waiting->second.skinCounts;
            moved.skinRadius = waiting->second.skinRadius;
            moved.skinFirstBone = waiting->second.skinFirstBone;
            moved.skinBoneCount = waiting->second.skinBoneCount;
            moved.skinMaterial = waiting->second.skinMaterial;
            moved.skinObjectKey = waiting->second.skinObjectKey;
            byHandle[moved.handle] = pairing.waiting;
            waiting->second = std::move(moved);
            totals.skinPairs++;
            skinEvent("paired", waiting->second, str::format("moved=", distance, " differs=", differences));
          }
          skinPairings.clear();
        }

        // M1: a device Reset makes the game create its default-pool buffers again, so their draws get new object keys.
        // Each scene-owned object becomes a candidate for a new draw with the same draw apart from its buffers.
        void onDeviceReset() {
          resetCandidates.clear();
          resetFrame = frame;
          for (const auto& [handle, key] : byHandle) {
            const auto found = objects.find(key);
            if (found != objects.end() && found->second.state == State::Direct && found->second.resetSignature != 0) {
              resetCandidates.emplace(found->second.resetSignature, ResetCandidate { key, found->second.lastFrame });
            }
          }
        }

        // M1: a new object whose draw matches a candidate that has not been drawn since the Reset takes that object over,
        // so the draw is an Update of the same scene object. The CS thread then compares the geometry hash and the world
        // matrix, and keeps the scene's copy as it is when both match (unchangedUpdates).
        void takeOverAfterReset(Object& object, XXH64_hash_t key, const D3D9DrawInputs& in) {
          const auto range = resetCandidates.equal_range(in.resetSignature());
          for (auto it = range.first; it != range.second; ++it) {
            const XXH64_hash_t candidateKey = it->second.key;
            const auto candidate = objects.find(candidateKey);
            if (candidateKey == key || candidate == objects.end() || candidate->second.state != State::Direct ||
                candidate->second.lastFrame != it->second.lastFrame) {
              continue;
            }
            object = std::move(candidate->second);
            objects.erase(candidate);
            byHandle[object.handle] = key;
            resetCandidates.erase(it);
            totals.resetPairs++;
            return;
          }
        }

        // R2: the frontend's camera for this draw, sent at the first draw of each camera type in the frame.
        void classifyCamera(const CameraInputs& camera, Decision& decision) {
          bool updatesCamera = false;
          const CameraType::Enum type = cameras.classify(camera, updatesCamera);
          useCamera(type, updatesCamera, decision);
        }

        // classify gives the same result when called again for the same draw, so earlyCheck can classify a draw that
        // decide classifies again after a miss. Only this step is once per draw.
        void useCamera(CameraType::Enum type, bool updatesCamera, Decision& decision) {
          decision.cameraType = uint32_t(type);
          if (updatesCamera && cameras.firstOfFrame(type)) {
            decision.feedCamera = true;
            totals.cameraFeeds++;
          }
        }

        // True when the sky heuristic would now classify the draw differently from its last preparation. Explicit sky
        // rules read only inputs the pre-key holds.
        static bool skyDetectionChanged(const PreparedState& prepared, const CameraInputs& camera, const SkyDetectInputs& sky) {
          const bool explicitSky = CategoryFlags(prepared.categories).test(InstanceCategories::Sky) && !prepared.skyAutoDetected;
          if (explicitSky || sky.seenCameraPositions == nullptr) {
            return false;
          }
          return DrawCallState::predictSkyAutoDetect(sky.worldToView, prepared.zWriteEnable, prepared.blending, prepared.skinned,
                                                     camera.drawingToRaytracedRenderTarget, sky.zEnable,
                                                     sky.prevFrameSeenCameras, *sky.seenCameraPositions) != prepared.skyAutoDetected;
        }

        // True when the draw's texture transform is corrected for the free camera (RtxContext::prepareGeometryForRTImpl):
        // view-space texgen while the free camera is on.
        static bool followsFreeCamera(uint32_t texgenMode, bool freeCamera) {
          return freeCamera && (texgenMode == uint32_t(TexGenMode::ViewPositions) || texgenMode == uint32_t(TexGenMode::ViewNormals));
        }

        // A4: a draw's inputs with the old path's material identity, the legacy material hash and the categories.
        static XXH64_hash_t passInputsOf(const D3D9DrawInputs& in, const PreparedState& prepared) {
          const uint64_t passMaterial[2] = { prepared.legacyMaterialHash, prepared.categories };
          return in.passKey(XXH3_64bits(passMaterial, sizeof(passMaterial)));
        }

        // A4: the key of one draw in this frame, from its object and inputs (decide) or its pre-key (earlyCheck), and the
        // camera it is drawn with.
        static XXH64_hash_t frameDrawKey(XXH64_hash_t object, XXH64_hash_t inputs, const CameraInputs& camera) {
          const XXH64_hash_t parts[2] = { object, inputs };
          XXH64_hash_t h = XXH3_64bits(parts, sizeof(parts));
          h = XXH3_64bits_withSeed(&camera.worldToView, sizeof(Matrix4), h);
          return XXH3_64bits_withSeed(&camera.viewToProjection, sizeof(Matrix4), h);
        }

        // A4: a repeated draw of an object already drawn this frame. A scene-owned object is not drawn again (a raster
        // service still replays the draw), and the old path merges the draw of any other object into its instance.
        Decision duplicate(XXH64_hash_t firstKey, Decision decision) {
          totals.duplicates++;
          const auto first = objects.find(firstKey);
          if (first == objects.end() || first->second.state != State::Direct) {
            return decision;
          }
          if (first->second.rasterService) {
            decision.mode = CommitMode::RasterOnly;
            decision.handle = first->second.handle;
            return decision;
          }
          decision.skip = true;
          return decision;
        }

        // A4: a second pass of an object drawn this frame that differs from the first only in its blend or alpha state. The
        // old path merges it into the first draw's instance with a second update in the frame (its alpha state wins when it
        // is opaque), so a scene-owned object gets it as its merged pass. One pass is merged per frame, and later ones are
        // skipped.
        Decision mergedPass(FramePass& pass, Decision decision) {
          const auto first = objects.find(pass.object);
          if (pass.merged || first == objects.end() || first->second.state != State::Direct || first->second.rasterService) {
            return duplicate(pass.object, decision);
          }
          totals.duplicates++;
          pass.merged = true;
          decision.mode = CommitMode::MergePass;
          decision.handle = first->second.handle;
          return decision;
        }

        bool earlyCheck(const D3D9DrawInputs& identity, const EarlyKey& key, CameraInputs camera, const SkyDetectInputs& sky,
                        Decision& decision, PreparedState& stored, bool collectStats) {
          pending = Pending();
          if (key.preKeyNs != 0) {
            totals.preKeyNs += key.preKeyNs;
            totals.preKeyTimed++;
          }
          // A4: a repeated draw is checked against the first draw's object without taking a slot of its own.
          const XXH64_hash_t drawKey = frameDrawKey(key.preKey, 0, camera);
          const auto repeated = framePreKeys.find(drawKey);
          if (repeated != framePreKeys.end()) {
            const auto first = objects.find(repeated->second);
            if (first == objects.end() || first->second.state != State::Direct || !first->second.hasEarly ||
                first->second.rasterService) {
              totals.preKeyMisses++;
              return false;  // decide finds the first draw by its inputs
            }
            totals.duplicates++;
            totals.preKeyHits++;
            camera.sky = CategoryFlags(first->second.prepared.categories).test(InstanceCategories::Sky);
            classifyCamera(camera, decision);
            decision.skip = true;
            stored = first->second.prepared;
            worldUIEvent(stored.categories, "earlyDuplicate", first->second.handle, key.objectToWorld, camera.viewToProjection);
            return true;
          }
          pending.valid = true;
          pending.objectKey = identity.objectKey();
          SlotQuery query;
          query.staleDistanceSqr = RtxOptions::getUniqueObjectDistanceSqr();
          pending.key = lookupNearest(pending.objectKey, key.objectToWorld[3].xyz(), key.signature, query);
          pending.early = key;
          const auto found = objects.find(pending.key);
          if (found == objects.end()) {
            totals.preKeyMisses++;
            return false;
          }
          Object& object = found->second;
          // R8b: raster-service objects are never skipped (their draw is replayed every frame).
          if (object.state != State::Direct || !object.hasEarly || object.rasterService ||
              object.preKeyNoWorld != key.preKeyNoWorld || object.reprepareEpoch != reprepareEpoch) {
            totals.preKeyMisses++;
            return false;
          }
          // The sky heuristic, the free camera's texgen correction and the camera type depend on more than the pre-key. A
          // change is prepared, so decide sees it. The sky test comes first: classify keeps the frame's first Main FOV, so it must see the draw's real Sky
          // category.
          if (skyDetectionChanged(object.prepared, camera, sky) ||
              followsFreeCamera(object.last.texgenMode, RtCamera::enableFreeCamera() || object.derivedFreeCamera)) {
            totals.derivedChanges++;
            totals.preKeyMisses++;
            return false;
          }
          camera.sky = CategoryFlags(object.prepared.categories).test(InstanceCategories::Sky);
          bool updatesCamera = false;
          const CameraType::Enum cameraType = cameras.classify(camera, updatesCamera);
          if (uint32_t(cameraType) != object.derivedCamera) {
            totals.derivedChanges++;
            totals.preKeyMisses++;
            return false;
          }
          const uint32_t selfCheckFrames = D3D9V2Options::earlyChangeTestSelfCheckFrames();
          if (selfCheckFrames != 0 && frame % selfCheckFrames == 0) {
            pending.wouldHit = true;
            totals.selfChecks++;
            return false;
          }

          // As decide's Direct branch for a repeat or a move.
          const bool moved = object.preKey != key.preKey;
          if (moved) {
            batch->moves.push_back({ object.handle, key.objectToWorld });
            object.last.objectToWorld = key.objectToWorld;
            object.preKey = key.preKey;
            object.inputs = object.last.inputsKey(true, true);
            totals.moved++;
            totals.preKeyMoves++;
          } else {
            totals.skipped++;
            totals.preKeyHits++;
          }
          batch->seen.push_back(object.handle);
          noteSortOrder(object, object.prepared.categories, false);
          noteFog(object.prepared.fog);  // R10
          object.lastFrame = frame;
          if (collectStats) {
            wasActive = true;
            observe(object.last);
          }
          useCamera(cameraType, updatesCamera, decision);
          decision.skip = true;
          stored = object.prepared;
          worldUIEvent(stored.categories, moved ? "earlyMove" : "earlySkip", object.handle, key.objectToWorld,
                       camera.viewToProjection);
          framePreKeys.emplace(drawKey, pending.key);
          framePasses.emplace(frameDrawKey(object.last.objectKey(), passInputsOf(object.last, object.prepared), camera),
                              FramePass { pending.key });
          pending = Pending();
          return true;
        }

        // R1 self-check: the pre-key matched (wouldHit) and the draw was prepared in full to verify it.
        void checkEarly(const Object& object, bool repeat, bool moved, XXH64_hash_t storedPreKey, const PreparedState& prepared) {
          const char* what = nullptr;
          if (storedPreKey == pending.early.preKey ? !repeat : !(repeat || moved)) {
            what = "inputs";
          } else if (prepared.categories != object.prepared.categories) {
            what = "categories";
          } else if (prepared.applyDrawState != object.prepared.applyDrawState) {
            what = "applyDrawState";
          } else if (prepared.colorTextureHash != object.prepared.colorTextureHash) {
            what = "colorTexture";
          }
          if (what == nullptr) {
            return;
          }
          totals.preKeyMismatches++;
          if (mismatchLogs < 20) {
            ++mismatchLogs;
            Logger::warn(str::format("[D3D9v2] preKeyMismatch: the pre-key matched but the full preparation differs in ", what,
                                     " (object 0x", std::hex, pending.key, std::dec, "). First 20 are logged."));
          }
        }

        Decision decide(const D3D9DrawInputs& in, const CameraInputs& camera, bool neverHandOver,
                        const PreparedState* prepared) {
          Decision decision;
          const bool direct = D3D9V2Options::direct();
          if (!direct && byHandle.empty()) {
            pending = Pending();
            return decision;
          }
          const Matrix4& view = camera.worldToView;
          const Matrix4& projection = camera.viewToProjection;
          // R2: the frontend's camera, sent at the first draw of each camera type in the frame.
          const bool explicitCamera = explicitCameraActive();
          if (explicitCamera) {
            classifyCamera(camera, decision);
          }

          // R1: an early check looked this draw up already (the occurrence index is counted once).
          const bool dynamicGeometry = dynamicGeometryActive();
          const XXH64_hash_t objectKey = identityKey(in);
          XXH64_hash_t inputs = in.inputsKey(true, true);
          XXH64_hash_t inputsNoWorld = in.inputsKey(true, false);
          XXH64_hash_t passInputs = prepared != nullptr ? passInputsOf(in, *prepared) : 0;
          Vector3 position = in.objectToWorld[3].xyz();
          // A pre-combined draw's world matrix holds world-view, which the CS thread turns into a world matrix with the
          // reference camera. A view-locked object keeps its world-view while the camera moves, so the reference camera
          // is part of its inputs, and its position is looked up in world space. Without the frontend's camera there is
          // no reference, so the draw stays on the old path.
          bool stayOnOldPath = neverHandOver;
          if (camera.preCombined) {
            const Matrix4* reference = explicitCamera ? cameras.referenceView(CameraType::Enum(decision.cameraType)) : nullptr;
            if (reference == nullptr) {
              stayOnOldPath = true;
            } else {
              inputs = XXH3_64bits_withSeed(reference, sizeof(Matrix4), inputs);
              inputsNoWorld = XXH3_64bits_withSeed(reference, sizeof(Matrix4), inputsNoWorld);
              passInputs = XXH3_64bits_withSeed(reference, sizeof(Matrix4), passInputs);
              position = (inverseAffine(*reference) * in.objectToWorld)[3].xyz();
            }
          }
          // A4: a second draw of the same object with the same inputs, world and camera in this frame is the first
          // draw's object, as the old path merges it into one instance. Unversioned data may differ, so it is never one.
          // The draw is keyed by its range even when its object has a coarse key, since two ranges of one dynamic buffer
          // are two objects.
          // A draw that differs from the first only in its blend or alpha state is merged too, since the old path's instance
          // identity holds the legacy material hash, not that state (mergedPass).
          XXH64_hash_t drawKey = 0;
          XXH64_hash_t passKey = 0;
          if (in.versioned()) {
            const XXH64_hash_t rangeKey = usesCoarseKey(in) ? in.objectKey() : objectKey;
            drawKey = frameDrawKey(rangeKey, inputs, camera);
            const uint32_t preparedCategories = prepared != nullptr ? prepared->categories : 0;
            const auto first = frameDraws.find(drawKey);
            if (first != frameDraws.end()) {
              pending = Pending();
              decision = duplicate(first->second, decision);
              worldUIEvent(preparedCategories, decision.skip ? "duplicateSkipped" : "duplicate", decision.handle,
                           in.objectToWorld, projection);
              return decision;
            }
            if (prepared != nullptr) {
              passKey = frameDrawKey(rangeKey, passInputs, camera);
              const auto firstPass = framePasses.find(passKey);
              if (firstPass != framePasses.end()) {
                pending = Pending();
                decision = mergedPass(firstPass->second, decision);
                worldUIEvent(preparedCategories, decision.mode == CommitMode::MergePass ? "mergedPass" : "passDuplicate",
                             decision.handle, in.objectToWorld, projection);
                return decision;
              }
            }
          }
          const bool early = pending.valid && pending.objectKey == objectKey;
          XXH64_hash_t key = pending.key;
          SlotQuery slotMatch = slotQuery(in);
          if (!early) {
            const bool skinned = (in.flags & D3D9DrawInputs::kSkinned) != 0;
            key = lookupNearest(objectKey, skinned ? in.skinPosition : position, in.contentSignature(), slotMatch);
          }
          if (drawKey != 0) {
            frameDraws.emplace(drawKey, key);
          }
          if (passKey != 0) {
            framePasses.emplace(passKey, FramePass { key });
          }
          if (early) {
            framePreKeys.emplace(frameDrawKey(pending.early.preKey, 0, camera), key);
          }
          Object& object = objects[key];
          if (!object.hasLast && !resetCandidates.empty() && dynamicGeometry && direct && !usesCoarseKey(in)) {
            takeOverAfterReset(object, key, in);
          }
          if (!usesCoarseKey(in)) {
            object.resetSignature = in.resetSignature();
          }
          const bool fresh = !object.hasLast;

          // Unversioned inputs (user-pointer data, version-0 buffers) compare equal whatever their data: never a repeat.
          // The categories and the camera type also come from sky auto-detection, the draw-call-ID sky rule, texture
          // tagging and camera reclassification, which can change while the inputs stay the same.
          // Terrain baking switched on or off moves terrain between the bake service and plain draws.
          // A view-space texgen draw's texture transform follows the free camera, so it changes every frame the free
          // camera is on, and once more when it is switched off.
          const bool terrainBaking = TerrainBaker::needsTerrainBaking();
          const bool freeCamera = RtCamera::enableFreeCamera();
          const bool derivedSame = prepared == nullptr ||
            (prepared->categories == object.derivedCategories && decision.cameraType == object.derivedCamera &&
             (!CategoryFlags(prepared->categories).test(InstanceCategories::Terrain) || terrainBaking == object.derivedTerrainBaking) &&
             !followsFreeCamera(in.texgenMode, freeCamera || object.derivedFreeCamera));
          const bool inputsSame = object.hasLast && in.versioned() && !(in.flags & D3D9DrawInputs::kVertexTextures) &&
                                  inputsNoWorld == object.inputsNoWorld;
          if (inputsSame && !derivedSame) {
            totals.derivedChanges++;
          }
          // A scene copy described before the last translation-invalidating option change is updated once.
          if (object.state != State::Direct) {
            object.reprepareEpoch = reprepareEpoch;
          }
          const bool current = object.reprepareEpoch == reprepareEpoch;
          if (inputsSame && derivedSame && !current) {
            totals.reprepared++;
          }
          const bool repeat = inputsSame && derivedSame && current && inputs == object.inputs;
          // A move would send a pre-combined draw's world-view to setTransform, so such a change is an Update.
          const bool moved = inputsSame && derivedSame && current && !repeat && !camera.preCombined;
          object.inputs = inputs;
          object.inputsNoWorld = inputsNoWorld;
          object.hasLast = true;
          if (prepared != nullptr) {
            object.derivedCategories = prepared->categories;
            object.derivedCamera = decision.cameraType;
            object.derivedTerrainBaking = terrainBaking;
            object.derivedFreeCamera = freeCamera;
          }
          object.lastFrame = frame;
          if (early && prepared != nullptr) {
            if (pending.wouldHit) {
              checkEarly(object, repeat, moved, object.preKey, *prepared);
            }
            object.hasEarly = true;
            object.preKey = pending.early.preKey;
            object.preKeyNoWorld = pending.early.preKeyNoWorld;
            object.last = in;
            object.prepared = *prepared;
          } else {
            object.hasEarly = false;
          }
          pending = Pending();
          const uint32_t categories = prepared != nullptr ? prepared->categories : object.prepared.categories;  // R7
          noteFog(prepared != nullptr ? prepared->fog : object.prepared.fog);  // R10

          // Vertex-captured draws are handed over only with rtx.d3d9v2.capturedGeometry (R6): their VS constants change with
          // the camera, so they are updated in place like R5's dynamic draws.
          // R5 (rtx.d3d9v2.dynamicGeometry): fixed-function skinned and user-pointer draws too (they are updated in place).
          const bool eligible = dynamicGeometry
            ? (in.versioned() || (in.flags & D3D9DrawInputs::kUserPointer)) && !stayOnOldPath &&
              (!(in.flags & D3D9DrawInputs::kVertexCapture) || frontends::d3d9::capturedGeometryActive())  // R6
            : in.versioned() && !stayOnOldPath &&
              !(in.flags & (D3D9DrawInputs::kUserPointer | D3D9DrawInputs::kSkinned | D3D9DrawInputs::kVertexCapture));
          const bool firstDraw = firstDrawHandoverActive();  // R4

          const bool logSkinned = (in.flags & D3D9DrawInputs::kSkinned) && fresh && D3D9V2Options::log();
          std::string skinSearchDetail;
          if (in.flags & D3D9DrawInputs::kSkinned) {
            // A slot whose last draw had another material, or lay farther than the draw's bone extent, held another
            // object or this one changed. Logged before noteSkinned overwrites the object's last draw.
            if (!early && slotMatch.reused && (slotMatch.previousMaterial != in.material ||
                lengthSqr(slotMatch.previousPosition - in.skinPosition) > slotMatch.otherMaterialDistanceSqr)) {
              totals.skinSwaps++;
              if (D3D9V2Options::log()) {
                const Vector3& p = slotMatch.previousPosition;
                skinEvent("swapped", object, str::format("to material=0x", std::hex, in.material, std::dec, " pos=(", in.skinPosition.x,
                                                         ", ", in.skinPosition.y, ", ", in.skinPosition.z, ") from slot material=0x",
                                                         std::hex, slotMatch.previousMaterial, std::dec, " pos=(", p.x, ", ", p.y, ", ",
                                                         p.z, ") moved=", length(in.skinPosition - p)));
              }
            }
            noteSkinned(object, key, in);
            // A new skinned object close to a scene-owned character of the same material that was drawn in the previous
            // frame but not yet in this one (its draw moved to other buffers or another key, as when a character becomes
            // a ragdoll) waits one frame, and that character is kept for it. If the character stays undrawn, endFrame
            // moves it to this draw's slot, so the next draw updates it in place with its history instead of building a
            // second copy. A new skinned object without such a character is a new object at once.
            if (fresh && direct && dynamicGeometry && eligible) {
              const SkinSearch search = findSkinnedPartner(in, key);
              if (search.chosen != 0) {
                const Object& partner = objects.find(search.chosen)->second;
                skinPairings.push_back({ key, search.chosen });
                batch->seen.push_back(partner.handle);
                totals.skinDeferrals++;
                totals.skinAmbiguous += search.qualified > 1 ? 1 : 0;
                if (logSkinned) {
                  skinEvent("held", object, str::format("partner=0x", std::hex, partner.handle, std::dec, " distance=", search.distance,
                                                        " bound=", search.bound, " differs=", skinDifferences(partner, object), " ",
                                                        search.describe()));
                }
                decision.skip = true;
                return decision;
              }
              if (logSkinned) {
                skinSearchDetail = search.describe();
              }
            }
          }

          if (object.state == State::Direct) {
            if (direct && eligible && (repeat || moved)) {
              if (moved) {
                batch->moves.push_back({ object.handle, in.objectToWorld });
                totals.moved++;
              } else {
                totals.skipped++;
              }
              batch->seen.push_back(object.handle);
              noteSortOrder(object, categories, false);
              if (object.rasterService) {
                // R8b: the scene owns the object; its draw still feeds the sky matte / probe.
                decision.mode = CommitMode::RasterOnly;
                decision.handle = object.handle;
                totals.rasterOnly++;
              } else if (!explicitCamera && cameraNeedsFeed(view, projection)) {
                decision.mode = CommitMode::CameraOnly;
                totals.cameraOnly++;
              } else {
                decision.skip = true;
              }
              worldUIEvent(categories, moved ? "move" : "skip", object.handle, in.objectToWorld, projection);
              return decision;
            }
            if (dynamicGeometry && direct && eligible) {
              // R5: changed but still eligible: described and built again in place under the same handle (the CS thread
              // hands it back itself if the scene no longer takes it).
              decision.mode = CommitMode::Update;
              decision.handle = object.handle;
              object.reprepareEpoch = reprepareEpoch;
              // A reclassified draw (a world object detected as sky, or the reverse) changes its raster service too.
              object.rasterService = decision.cameraType == uint32_t(CameraType::Sky) ||
                                     bakedTerrainCategory(categories);
              totals.updates++;
              batch->seen.push_back(object.handle);  // fed like a skipped draw at the flush (mesh-hash tracking, fog)
              noteSortOrder(object, categories, true);
              noteQueuedCamera(view, projection);
              worldUIEvent(categories, "update", object.handle, in.objectToWorld, projection);
              return decision;
            }
            worldUIEvent(categories, "handedBack", object.handle, in.objectToWorld, projection);
            // Changed (or v2 switched off): the scene's copy goes, the old path takes the draw again.
            if (object.skinClass != 0) {
              skinEvent("handedBack", object);
            }
            batch->destroys.push_back(object.handle);
            dropHandle(object);
            totals.changed++;
            // R4 churn guard: an object that changes soon after its first-draw handover backs off as a rejected one does,
            // so a draw that changes every frame is not described, built and destroyed every frame.
            if (firstDraw && frame - object.handoverFrame < kChurnFrames) {
              object.rejects++;
              object.backoffUntil = frame + std::min(2048u, 32u << std::min(object.rejects, 6u));
              totals.churnBackoffs++;
            }
          } else if (object.state == State::Pending) {
            if (repeat) {
              // Drawn on the old path until accepted; the scene's copy counts as seen too, so it does not age out. The
              // draw carries the handle so the CS thread knows whose old-path draw it is (V4: handedToScene).
              batch->pendingSeen.push_back(object.handle);
              decision.handle = object.handle;
              totals.pendingRepeats++;
            } else {
              // Changed before the scene reported back: its description is stale.
              batch->destroys.push_back(object.handle);
              dropHandle(object);
            }
          }

          if (object.state == State::Legacy && direct && firstDraw) {
            // R4: handed over at its first eligible draw, without an old-path copy: Direct from this draw on (the CS
            // thread builds it at once and draws it through the old path if the scene rejects it). The app thread checks
            // what it knows (camera, categories, light conversion, raytraced render targets); the CS thread the rest.
            const bool services = frontends::d3d9::specialCategoriesActive();  // R8
            const bool unordered = frontends::d3d9::unorderedActive();         // R7
            const bool sceneEligible = eligible && prepared != nullptr &&
              acceptedCamera(decision.cameraType) &&  // R8 / R8b
              !specialCategory(CategoryFlags(prepared->categories), unordered, services) &&
              (services || !prepared->convertsToLight) && (renderTargetsActive() || !prepared->raytracedRenderTarget);
            if ((!sceneEligible || frame < object.backoffUntil) && prepared != nullptr && D3D9V2Options::log()) {
              countNotEligible(eligible, sceneEligible && frame < object.backoffUntil, decision.cameraType, *prepared,
                               services, unordered);
            }
            if (sceneEligible && frame >= object.backoffUntil) {
              object.state = State::Direct;
              object.handle = kHandleNamespace | nextHandle++;
              object.handoverFrame = frame;
              byHandle[object.handle] = key;
              decision.mode = CommitMode::HandoverFirstDraw;
              decision.handle = object.handle;
              object.rasterService = decision.cameraType == uint32_t(CameraType::Sky) ||  // R8b: sky matte / probe
                                     bakedTerrainCategory(categories);  // bake
              totals.handovers++;
              // Fed like a skipped draw at the flush (mesh-hash tracking, fog).
              batch->seen.push_back(object.handle);
              noteSortOrder(object, categories, true);
            }
          } else if (object.state == State::Legacy && direct) {
            // Probation path.
            object.streak = (repeat && eligible) ? object.streak + 1 : 0;
            if (object.streak >= std::max(1u, D3D9V2Options::handoverFrames()) && frame >= object.backoffUntil) {
              object.state = State::Pending;
              object.handle = kHandleNamespace | nextHandle++;
              byHandle[object.handle] = key;
              decision.mode = CommitMode::Handover;
              decision.handle = object.handle;
              totals.handovers++;
            }
          }
          if (logSkinned) {
            skinEvent("created", object, str::format(object.state == State::Direct ? "scene-owned" : "old path",
                                                     skinSearchDetail.empty() ? "" : " ", skinSearchDetail));
          }
          noteQueuedCamera(view, projection);
          worldUIEvent(categories,
                       decision.mode == CommitMode::HandoverFirstDraw ? "handover"
                       : decision.mode == CommitMode::Handover ? "handoverAfterRepeats"
                       : object.state == State::Pending ? "pendingOldPath" : "oldPath",
                       decision.handle, in.objectToWorld, projection);
          return decision;
        }

        std::shared_ptr<FrameBatch> takeBatch() {
          if (batchTaken) {
            return nullptr;
          }
          batchTaken = true;
          batch->frameEvents = frontends::d3d9::frameEventsActive();  // R10
          if (!D3D9V2Options::direct() && !byHandle.empty()) {
            // Switched off: every object goes back to the old path; the CS thread drops the scene's copies.
            batch->releaseAll = true;
            for (auto& [handle, key] : byHandle) {
              const auto found = objects.find(key);
              if (found != objects.end()) {
                found->second.handle = 0;
                found->second.state = State::Legacy;
                found->second.streak = 0;
              }
            }
            byHandle.clear();
          }
          if (batch->empty() && byHandle.empty() && !D3D9V2Options::direct()) {
            return nullptr;
          }
          std::shared_ptr<FrameBatch> taken = std::move(batch);
          batch = std::make_shared<FrameBatch>();
          return taken;
        }

        void applyResults() {
          if (channel == nullptr) {
            return;
          }
          std::vector<Channel::Result> results;
          {
            std::lock_guard<std::mutex> lock(channel->mutex);
            results.swap(channel->results);
            if (channel->reprepare) {
              channel->reprepare = false;
              ++reprepareEpoch;
            }
          }
          for (const Channel::Result& result : results) {
            const auto found = byHandle.find(result.handle);
            if (found == byHandle.end()) {
              continue;  // dropped here already (changed, or switched off)
            }
            const auto object = objects.find(found->second);
            if (object == objects.end()) {
              byHandle.erase(found);
              continue;
            }
            Object& o = object->second;
            if (o.skinClass != 0 && result.kind != Channel::Kind::Accepted) {
              skinEvent(result.kind == Channel::Kind::Rejected ? "rejected"
                        : result.kind == Channel::Kind::Released ? "released" : "retry", o);
            }
            switch (result.kind) {
            case Channel::Kind::Accepted:
              if (o.state == State::Pending) {
                o.state = State::Direct;
              }
              break;
            case Channel::Kind::Rejected:
              // R4: the destroy also tells the CS thread to drop its fallback draw (a no-op for other handovers).
              batch->destroys.push_back(result.handle);
              dropHandle(o);
              o.rejects++;
              // Back off: a draw the scene keeps rejecting is not handed over every few frames.
              o.backoffUntil = frame + std::min(2048u, 32u << std::min(o.rejects, 6u));
              break;
            case Channel::Kind::Released:
              // The CS thread keeps the entry until this destroy, so the object never goes missing in between.
              batch->destroys.push_back(result.handle);
              dropHandle(o);
              break;
            case Channel::Kind::Retry:
              batch->destroys.push_back(result.handle);
              dropHandle(o);
              break;
            }
          }
        }

        void endFrame() {
          applyResults();
          pairSkinned();
          if (!resetCandidates.empty() && frame - resetFrame >= kResetFrames) {
            resetCandidates.clear();
          }
          batchTaken = false;
          occurrence.clear();
          slotCursors.clear();
          frameDraws.clear();
          framePreKeys.clear();
          framePasses.clear();
          queuedCameraValid = false;
          decalRank = 0;
          fogSeenThisFrame.clear();
          cameras.endFrame();
          ++frame;
          // Objects the game no longer draws are forgotten after a while (scene-owned ones only once released).
          if ((frame & 63) == 0) {
            for (auto it = objects.begin(); it != objects.end();) {
              if (it->second.state == State::Legacy && it->second.lastFrame + 120 < frame) {
                if (it->second.skinClass != 0) {
                  skinEvent("forgotten", it->second);
                }
                it = objects.erase(it);
              } else {
                ++it;
              }
            }
            // A slot is freed once nothing has drawn it for as long and its object is forgotten, so an anti-culled
            // object the scene still owns keeps its slot.
            for (auto it = slots.begin(); it != slots.end();) {
              bool anyValid = false;
              for (uint32_t i = 0; i < uint32_t(it->second.size()); ++i) {
                PositionSlot& slot = it->second[i];
                if (slot.valid && slot.lastFrame + 120 < frame && objects.find(slotKey(it->first, i)) == objects.end()) {
                  slot.valid = false;
                }
                anyValid |= slot.valid;
              }
              it = anyValid ? std::next(it) : slots.erase(it);
            }
            for (auto it = skinnedByClass.begin(); it != skinnedByClass.end();) {
              std::vector<XXH64_hash_t>& keys = it->second;
              const XXH64_hash_t skinClass = it->first;
              keys.erase(std::remove_if(keys.begin(), keys.end(), [&](XXH64_hash_t key) {
                const auto found = objects.find(key);
                return found == objects.end() || found->second.skinClass != skinClass;
              }), keys.end());
              it = keys.empty() ? skinnedByClass.erase(it) : std::next(it);
            }
          }
        }
      };

      D3D9FrontendApp::D3D9FrontendApp()
        : m_impl(std::make_shared<Impl>()) {
        frontend_ui::registerPanel(frontend_ui::Group::Geometry, &drawPanel);
      }

      void D3D9FrontendApp::connect(D3D9Frontend& cs) {
        m_impl->channel = cs.m_impl->channel;
      }

      bool D3D9FrontendApp::wantsInputs() const {
        return D3D9V2Options::log() || D3D9V2Options::direct() || !m_impl->byHandle.empty();
      }

      Decision D3D9FrontendApp::decide(const D3D9DrawInputs& inputs, const CameraInputs& camera, bool neverHandOver,
                                       const PreparedState* prepared) {
        if (D3D9V2Options::log()) {
          m_impl->wasActive = true;
          m_impl->observe(inputs);
        }
        return m_impl->decide(inputs, camera, neverHandOver, prepared);
      }

      bool D3D9FrontendApp::earlyChangeTestActive() const {
        return frontends::d3d9::earlyChangeTestActive();
      }

      bool D3D9FrontendApp::statsActive() const {
        return D3D9V2Options::log();
      }

      bool D3D9FrontendApp::capturedGeometryActive() const {
        return frontends::d3d9::capturedGeometryActive();
      }

      bool D3D9FrontendApp::unorderedActive() const {
        return frontends::d3d9::unorderedActive();
      }

      bool D3D9FrontendApp::lightsActive() const {
        return frontends::d3d9::lightsActive();
      }

      bool D3D9FrontendApp::earlyCheck(const D3D9DrawInputs& identity, const EarlyKey& key, CameraInputs camera,
                                       const SkyDetectInputs& sky, Decision& decision, PreparedState& stored) {
        return m_impl->earlyCheck(identity, key, camera, sky, decision, stored, D3D9V2Options::log());
      }

      void D3D9FrontendApp::notePerDrawPath(const FogInputs& fog) {
        m_impl->noteFog(fog);
      }

      std::shared_ptr<FrameBatch> D3D9FrontendApp::takeFrameBatch() {
        return m_impl->takeBatch();
      }

      void D3D9FrontendApp::onDeviceReset() {
        m_impl->onDeviceReset();
      }

      void D3D9FrontendApp::endFrame() {
        m_impl->endFrame();
        if (!D3D9V2Options::log()) {
          // Switched off: forget the statistics, so switching on again starts clean.
          if (m_impl->wasActive) {
            m_impl->resetStats();
            m_impl->wasActive = false;
          }
          return;
        }
        m_impl->endStatsFrame();
      }

#else

      struct D3D9Frontend::Impl { };
      struct D3D9FrontendApp::Impl { };
      struct FrameBatch { };

      D3D9Frontend::D3D9Frontend() { }

      void D3D9Frontend::commit(RtxContext* ctx, const DrawParameters& params, DrawCallState& drawCallState,
                                CommitMode, uint64_t) {
        // The old path (the fallback until D3D9 v2 matches it; deleted in Phase 6).
        ctx->commitGeometryToRT(params, drawCallState);
      }

      void D3D9Frontend::flush(RtxContext*, const std::shared_ptr<FrameBatch>&) { }

      void D3D9Frontend::lights(RtxContext* ctx, const D3DLIGHT9* lights, uint32_t count, bool) {
        ctx->addLights(lights, count);
      }

      D3D9FrontendApp::D3D9FrontendApp() { }
      void D3D9FrontendApp::connect(D3D9Frontend&) { }
      bool D3D9FrontendApp::wantsInputs() const { return false; }
      Decision D3D9FrontendApp::decide(const D3D9DrawInputs&, const CameraInputs&, bool, const PreparedState*) { return Decision(); }
      bool D3D9FrontendApp::earlyChangeTestActive() const { return false; }
      bool D3D9FrontendApp::statsActive() const { return false; }
      bool D3D9FrontendApp::capturedGeometryActive() const { return false; }
      bool D3D9FrontendApp::unorderedActive() const { return false; }
      bool D3D9FrontendApp::lightsActive() const { return false; }
      bool D3D9FrontendApp::earlyCheck(const D3D9DrawInputs&, const EarlyKey&, CameraInputs, const SkyDetectInputs&, Decision&,
                                       PreparedState&) { return false; }
      void D3D9FrontendApp::notePerDrawPath(const FogInputs&) { }
      std::shared_ptr<FrameBatch> D3D9FrontendApp::takeFrameBatch() { return nullptr; }
      void D3D9FrontendApp::endFrame() { }
      void D3D9FrontendApp::onDeviceReset() { }

#endif

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
