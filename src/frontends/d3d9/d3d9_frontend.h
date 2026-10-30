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
#pragma once

// The D3D9 frontend.
//
// Two halves:
//   D3D9FrontendApp (D3D9 / app thread) sees every draw D3D9 commits to ray tracing, before it is queued for the CS
//     thread, and decides per draw: queue it for the old path, queue it as a handover to the scene, queue it only to
//     feed the camera, or skip it (the scene already owns the object and nothing changed).
//   D3D9Frontend (CS thread) runs the old path for queued draws, describes handed-over objects to the scene, applies
//     the frame's commands and ages out direct objects the game stopped drawing.
//
// The backend never includes this file directly (scripts-common/check_frontend_deps.py); it sees it only through
// d3d9_rtx.h, so nothing here may depend on REMIX_FRONTEND_D3D9_V2 for its layout.

#include <cstdint>
#include <memory>
#include <vector>

#include "d3d9_draw_inputs.h"

struct _D3DLIGHT9;  // D3DLIGHT9 (d3d9types.h)

namespace dxvk {
  class RtxContext;
  struct DrawParameters;
  struct DrawCallState;

  namespace frontends {
    namespace d3d9 {

      // How a queued draw is handled on the CS thread.
      enum class CommitMode : uint8_t {
        OldPath,     // the old path (SceneManager::submitDrawState)
        Handover,    // the old path, then described to the scene under `handle` if it qualifies
        CameraOnly,  // only the old path's preparation (camera detection); the scene owns the object
        HandoverFirstDraw,  // R4: no old path; described and built at once; on rejection the old path and a fallback
        Update,             // R5: a Direct object's draw changed: described and built again in place under its handle
        RasterOnly,         // R8b: a Direct object whose draw is also a raster service (sky): replayed, not described
        MergePass,          // A4: an opaque draw after a non-opaque draw of a Direct object: its merged pass
      };

      // What the frontend's camera classification reads from a draw (rtx.d3d9v2.explicitCamera):
      // the same inputs CameraManager::processCameraData reads. worldToView and
      // objectToWorld are as the CS thread's preparation sees them, after rtx.fusedWorldViewMode's fix-up.
      struct CameraInputs {
        Matrix4 worldToView;
        Matrix4 viewToProjection;
        Matrix4 objectToWorld;
        Matrix4 objectToView;
        float maxZ = 1.f;                         // viewport max depth (view model detection)
        bool sky = false;                         // Sky category (texture lists and draw-order heuristics)
        bool drawingToRaytracedRenderTarget = false;
        // rtx.resolvePreCombinedMatrices with an identity view: the CS thread rebuilds the world matrix from the
        // reference camera, so the draw's world matrix holds world-view.
        bool preCombined = false;
      };

      // R1 (rtx.d3d9v2.earlyChangeTest): the draw's pre-key, hashed from raw D3D9 state
      // before any preparation. preKey covers every input the preparation reads; preKeyNoWorld the same without the
      // world matrix (a move). preKeyNs: the time the caller spent building it (0 = not timed).
      struct EarlyKey {
        XXH64_hash_t preKey = 0;
        XXH64_hash_t preKeyNoWorld = 0;
        // The bound textures' hashes, the D3D9 material and the drawn counts: what tells apart draws of one object key
        // at the same position (D3D9DrawInputs::contentSignature without a prepared material).
        XXH64_hash_t signature = 0;
        Matrix4 objectToWorld;
        uint64_t preKeyNs = 0;
      };

      // R10 (rtx.d3d9v2.frameEvents): the draw's fixed-function fog, FogState's fields (mode 0 = D3DFOG_NONE).
      struct FogInputs {
        uint32_t mode = 0;
        float color[3] = {};
        float scale = 0.f;
        float end = 0.f;
        float density = 0.f;
      };

      // R1: what a full preparation derived for a draw, kept per object so a skipped draw can repeat its D3D9-thread
      // side effects (sky camera tracking, texture hash tracking, the ApplyDrawState flag) without preparing.
      struct PreparedState {
        uint32_t categories = 0;           // CategoryFlags raw
        uint64_t colorTextureHash = 0;     // the colour texture's image hash (texture hash tracking), 0 = none
        bool applyDrawState = false;       // PrepareDrawFlag::ApplyDrawState
        bool zWriteEnable = false;         // sky camera tracking inputs (DrawCallState::trackSkyCameraPosition)
        bool blending = false;
        bool skinned = false;
        // A4: the legacy material hash, which the old path's instance identity holds instead of the full material.
        uint64_t legacyMaterialHash = 0;
        // R4: what the app thread checks before a first-draw handover (the CS thread's eligibility checks the rest).
        bool convertsToLight = false;      // rtx.lightConverter lists the legacy material
        bool raytracedRenderTarget = false;// draws to or samples a raytraced render target
        bool skyAutoDetected = false;      // the Sky category came from rtx.skyAutoDetect
        FogInputs fog;                     // R10: the frame's fog order
      };

      // R1: the sky auto-detection state at a draw (rtx.skyAutoDetect), so a skip can tell whether the heuristic would
      // now classify the draw differently.
      struct SkyDetectInputs {
        Matrix4 worldToView;  // the D3D9 view matrix, which sky detection reads without rtx.fusedWorldViewMode's fix-up
        const std::vector<Vector3>* seenCameraPositions = nullptr;  // this frame's cameras so far
        uint32_t prevFrameSeenCameras = 0;
        bool zEnable = false;
      };

      // The app thread's verdict for one draw committed to ray tracing.
      struct Decision {
        static constexpr uint32_t kNoCamera = ~0u;
        bool skip = false;               // nothing is queued for this draw
        CommitMode mode = CommitMode::OldPath;
        uint64_t handle = 0;             // Handover: the scene handle; OldPath: the pending object's handle, or 0
        // rtx.d3d9v2.explicitCamera: the draw's camera type as the frontend classified it (CameraType::Enum value, Unknown
        // included), or kNoCamera while the option is inactive; feedCamera: this is the first draw of that camera type
        // in the D3D9 frame, so the caller sends the camera (CameraManager::applyFrameCamera) before the draw's commit.
        uint32_t cameraType = kNoCamera;
        bool feedCamera = false;
      };

      // Commands and per-frame facts the app thread hands to the CS thread once per D3D9 frame (inside the inject or
      // end-frame command). Defined in d3d9_frontend.cpp.
      struct FrameBatch;
      class D3D9FrontendApp;

      // CS thread only.
      class D3D9Frontend {
      public:
        D3D9Frontend();

        // One committed draw. drawCallState may be consumed.
        void commit(RtxContext* ctx, const DrawParameters& params, DrawCallState& drawCallState,
                    CommitMode mode = CommitMode::OldPath, uint64_t handle = 0);

        // Hands the frame's batch to the scene (null: nothing new, e.g. the second call in a frame). Called before
        // injectRTX and before RtxContext::endFrame (whose fallback inject builds the scene).
        void flush(RtxContext* ctx, const std::shared_ptr<FrameBatch>& batch = nullptr);

        // The enabled D3D9 lights, emitted when they changed (D3D9Rtx::emitDirtyLights). translate (R9,
        // rtx.d3d9v2.lights, decided on the app thread): the frontend's LightTranslator takes them; otherwise the old
        // path (RtxContext::addLights -> SceneManager::addLight).
        void lights(RtxContext* ctx, const _D3DLIGHT9* lights, uint32_t count, bool translate);

      private:
        friend class D3D9FrontendApp;
        struct Impl;
        std::shared_ptr<Impl> m_impl;  // shared_ptr: destructible where Impl is incomplete
      };

      // D3D9 (app) thread only.
      class D3D9FrontendApp {
      public:
        D3D9FrontendApp();

        // Shares the result channel with the CS half (D3D9Rtx::Initialize).
        void connect(D3D9Frontend& cs);

        // True when the caller should build a D3D9DrawInputs record for each committed draw.
        bool wantsInputs() const;

        // One draw committed to ray tracing. camera: the draw's camera inputs (rtx.d3d9v2.explicitCamera classifies them;
        // otherwise the CS thread derives the camera from queued draws); neverHandOver: the draw must stay on the old path
        // (blending without rtx.d3d9v2.unordered).
        // prepared (R1): what the preparation derived, stored for later skips; null when not collected.
        Decision decide(const D3D9DrawInputs& inputs, const CameraInputs& camera, bool neverHandOver,
                        const PreparedState* prepared = nullptr);

        // R1: rtx.d3d9v2.earlyChangeTest in effect (needs rtx.d3d9v2.direct and rtx.d3d9v2.explicitCamera).
        bool earlyChangeTestActive() const;
        // R6: rtx.d3d9v2.capturedGeometry in effect (vertex-captured draws can be owned by the scene, so R1 pre-checks them).
        bool capturedGeometryActive() const;
        // R7: rtx.d3d9v2.unordered in effect (blended draws can be owned by the scene).
        bool unorderedActive() const;
        // R9: rtx.d3d9v2.lights in effect (D3D9 lights go to the frontend's LightTranslator).
        bool lightsActive() const;
        // True while the [D3D9v2] counters are collected (the caller times the pre-key only then).
        bool statsActive() const;
        // R1, before preparation. identity: the draw's ids, versions, range, shaders and extras (no material); camera:
        // its camera inputs (sky is filled from the stored state). True = the scene owns the object and nothing but the
        // world matrix changed, neither in the inputs nor in the camera type and sky detection derived from them:
        // decision says skip (and the camera to send), stored is what the last full preparation derived.
        // False = prepare as usual and call decide, which reuses this lookup.
        bool earlyCheck(const D3D9DrawInputs& identity, const EarlyKey& key, CameraInputs camera, const SkyDetectInputs& sky,
                        Decision& decision, PreparedState& stored);

        // A draw committed to ray tracing that the host keeps on the per-draw path (FnvLegacyProtocol::kPerDrawPath):
        // decide and earlyCheck do not see it, so only its fog joins the frame's fog order (R10).
        void notePerDrawPath(const FogInputs& fog);

        // The frame's commands for the CS thread: the first call in a D3D9 frame returns them, later calls null.
        std::shared_ptr<FrameBatch> takeFrameBatch();

        // Once per D3D9 frame (D3D9Rtx::EndFrame), after takeFrameBatch.
        void endFrame();

        // At a device Reset (D3D9Rtx::onReset), before the game creates its default-pool buffers again.
        void onDeviceReset();

      private:
        struct Impl;
        std::shared_ptr<Impl> m_impl;
      };

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
