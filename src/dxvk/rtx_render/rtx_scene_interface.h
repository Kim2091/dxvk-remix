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

// Scene interface.
// The types an API frontend hands to the backend scene, and the options that switch the new path on.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <remix/remix_c.h>

#include "rtx_frontend_interface.h"
#include "rtx_option.h"
#include "rtx_scene_manager.h"

namespace dxvk {
  namespace scene {

    class Options {
    public:
      RTX_OPTION("rtx.scene", bool, directRetained, false,
                 "Remix API retained instances that need no per-frame work (static, main camera, and without bones, "
                 "particles, USD mesh replacements, light conversion, billboards, decals or portals) are owned by the "
                 "scene directly. Their instances and geometry are built once at creation, moved and shown or hidden in "
                 "place, and shifted without a replay when the origin changes, which saves CPU time. Everything else "
                 "stays on the retained path. Takes effect at the next DrawRetainedInstances call. Needs "
                 "rtx.retained.useV2 off. On by default in D3D9 games, which need it for rtx.d3d9v2.direct, and off "
                 "by default for Remix API hosts.");
      RTX_OPTION("rtx.scene", bool, commandQueue, false,
                 "With rtx.scene.directRetained: retained create, update and destroy calls go into a lock-free queue "
                 "that the render thread empties once per frame, instead of taking the API lock for every call.");
      RTX_OPTION("rtx.scene", bool, noImplicitClears, false,
                 "Remix API hosts: a camera cut or a frame without a valid camera only resets temporal history, and no "
                 "longer clears the scene (geometry, instances and micromaps). No effect on D3D9 games.");
      RTX_OPTION("rtx.scene", bool, d3d9CutAsEvent, true,
                 "D3D9 games: a detected camera cut or a frame without a valid camera only resets temporal history, and "
                 "no longer clears the scene. Objects the game stops drawing age out as usual, and anti-culling never "
                 "keeps an object last drawn before the cut, so a level change still frees the old level.");
      RTX_OPTION("rtx.scene", bool, originShiftEvent, false,
                 "With rtx.scene.directRetained: a scene-origin change shifts the scene's instances as one event, which "
                 "keeps their history, instead of moving each one. The moved instances keep their merged BLAS.");
      RTX_OPTION("rtx.scene", bool, resolveCategoriesOnOptionChange, false,
                 "With rtx.scene.directRetained: when texture tagging or category options change, instances the scene "
                 "owns resolve their categories and materials again, so tagging textures live also reaches API "
                 "geometry.");
      RTX_OPTION("rtx.scene", bool, directUnordered, false,
                 "With rtx.scene.directRetained: retained decals, emissive-blend and blended particle instances are "
                 "owned by the scene too. Decals get a stable draw order when they are created, and billboards are "
                 "refreshed every frame from the camera. On by default in D3D9 games and off by default for Remix API "
                 "hosts.");
      RTX_OPTION("rtx.scene", bool, directServices, false,
                 "With rtx.scene.directRetained: retained instances that need per-frame work are owned by the scene "
                 "too, which provides it: light conversion (rtx.lightConverter), particle emitters, non-main cameras "
                 "(view model and sky), the third-person player model and ray portals. USD mesh replacements become "
                 "child instances that move with their entry. On by default in D3D9 games and off by default for Remix "
                 "API hosts.");
      RTX_OPTION("rtx.scene", bool, explicitLights, false,
                 "Remix API lights are kept as explicit records: a light the host updates is always updated, even when "
                 "it moves slowly. While the layout of the light list is unchanged, only lights whose data changed are "
                 "uploaded, instead of the whole light buffer every frame.");
      RTX_OPTION("rtx.scene", bool, selectiveModChanges, false,
                 "Remix API hosts: a mod reload or change (mods enabled or disabled, files changed) no longer clears "
                 "the whole scene (textures, geometry, BLASes and micromaps). Replacement instances are rebuilt "
                 "instead, and the scene's own instances are rebuilt only where a reload changed their mesh or material "
                 "replacement.");
      RTX_OPTION("rtx.scene", bool, assetKeys, false,
                 "Uses host asset keys: remixapi_AssetKeyEXT on API meshes and materials, and the key a D3D9 point light "
                 "carries in its unused spot fields. USD replacements are looked up by the host's asset key (and then its "
                 "aliases) instead of by the mesh, material or light hash, and captures write the key.");
      RTX_OPTION("rtx.scene", bool, directTransient, false,
                 "Per-frame Remix API DrawInstance calls (skinned actors, effects and the view model) become "
                 "scene-owned instances, matched from frame to frame. Unchanged draws are kept, moved draws are moved "
                 "in place, and new poses update skinned geometry in place, which saves CPU time. Draws the host stops "
                 "making age out like other instances, anti-culling included. Particles, USD mesh replacements and "
                 "light conversion use the standard path. Takes effect from the next frame.");
      RTX_OPTION("rtx.scene", uint32_t, selfCheckFrames, 0,
                 "Diagnostic for rtx.scene.directRetained: every this many frames, compares every instance the scene "
                 "owns with the host's description (transform, hidden state and material). Mismatches are logged as "
                 "[SceneCheck] and the instance is rebuilt. 0 = off.");

      // Makes directRetained, directUnordered and directServices default on.
      // The D3D9 frontend calls it when it creates a device whose draws are ray traced.
      // A value set in a config file or the menu still wins, because it sits in a stronger layer.
      static void useD3D9Defaults() {
        const RtxOptionLayer* defaults = RtxOptionLayer::getDefaultLayer();
        directRetainedObject().setImmediately(true, defaults);
        directUnorderedObject().setImmediately(true, defaults);
        directServicesObject().setImmediately(true, defaults);
      }
    };

    using Handle = uint64_t;

    // UpdateRetainedInstances, one entry. flags: REMIXAPI_RETAINED_UPDATE_TRANSFORM / _VISIBILITY.
    struct InstanceUpdate {
      Handle handle = 0;
      uint32_t flags = 0;
      Matrix4 objectToWorld;
      bool visible = true;
    };

    // One command for the backend scene. Create / Describe carry the prepared description (the expensive half of a
    // create, done on the calling thread).
    struct SceneCommand {
      enum class Type : uint8_t { Create, Describe, Update, Destroy, Frame };
      Type type = Type::Frame;
      Handle handle = 0;
      frontend::PreparedInstance instance;
      InstanceUpdate update;
      Vector3 origin { 0.f, 0.f, 0.f };   // Frame: DrawRetainedInstances' scene origin
      uint32_t events = 0;                // Frame: frontend::FrameEventBits
    };

  } // namespace scene
} // namespace dxvk
