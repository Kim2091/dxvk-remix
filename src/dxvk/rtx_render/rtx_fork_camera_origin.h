#pragma once

// The world origin published by a host that submits everything relative to a moving render origin.
//
// Ported from BetaRT (sambow23/dxvk-remix-gmod, branch betart-numos3): rtx_fork_camera_origin.{h,cpp}
// added in commit 11761453a ("add png screenshot support"), used by rtx_camera_manager.cpp and
// rtx_atmosphere.cpp (11761453a, 963c402f5, 92ea99a7b). Those files carry no licence header; that
// repository is MIT (NVIDIA LICENSE-MIT) / zlib (DXVK LICENSE) like this one. See CREDITS.md.
// Kept: the game-state keys (__mcrtx.world_origin.x/y/z), their parsing and the previous-camera
// history offset. Added here: the rtx.worldOrigin.enable gate, the origin travelling through the
// command stream with SetupCamera (read on the API thread, in order with the host's draws), and the
// instance-history shift on an origin move (SceneManager::applyWorldOrigin).
//
// Convention: render position + world origin = world position, both in render units.

#include <string>

#include "rtx_option.h"
#include "../util/util_vector.h"

namespace dxvk {

  class WorldOrigin {
  public:
    RTX_OPTION("rtx.worldOrigin", bool, enable, false,
               "For hosts that submit the scene relative to a moving render origin and publish that origin, in render "
               "units, in the game values __mcrtx.world_origin.x/y/z (remixapi_SetGameValue) before each SetupCamera. "
               "For a camera that arrives as D3D9 transforms, the host publishes them before the frame's first draw. "
               "When the origin moves, the previous camera and every instance's previous transform are moved with it, "
               "so motion vectors, camera-cut detection and temporal history see no jump, and NUMOS clouds stay "
               "anchored to the world. When off, those game values are ignored.");

    RTX_OPTION("rtx.worldOrigin", bool, preciseCamera, false,
               "Keeps a Remix API parameterized camera (SetupCamera with CameraInfoParameterizedEXT) in double "
               "precision instead of rounding its view matrix to float. At world coordinates around 100,000 units, "
               "float rounding changes every frame as the camera turns, which makes a nearby view model wobble.");

    static void showImguiSettings();
  };

  namespace fork_camera_origin {

    inline constexpr const char* kWorldOriginGameValueX = "__mcrtx.world_origin.x";
    inline constexpr const char* kWorldOriginGameValueY = "__mcrtx.world_origin.y";
    inline constexpr const char* kWorldOriginGameValueZ = "__mcrtx.world_origin.z";

    bool tryParseWorldOriginComponent(const std::string& raw, float& outValue);
    // False when any component is missing or unparsable.
    bool tryReadWorldOriginOffsetFromGameState(Vector3& outOrigin);
    // Offset that re-expresses last frame's render positions around the current origin.
    Vector3 calculatePreviousCameraHistoryOffset(
      bool hasPreviousOrigin,
      const Vector3& previousOrigin,
      const Vector3& currentOrigin);

  } // namespace fork_camera_origin
} // namespace dxvk
