#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "rtx_option.h"

namespace dxvk {
  class DxvkImage;
  class RtxContext;

  // rtx.debug.allocationTrace.
  // While on, every image allocated with DxvkMemoryStats::Category::AppTexture is tracked, keyed by its DxvkImage.
  // The D3D9 frontend adds the texture or surface it belongs to, its D3D9 description and its COM reference counts.
  // Lines go to remix-dxvk.log, tagged [AllocTrace]:
  //   + #<id> ...      a D3D9 render target, depth stencil or texture of at least 8 MiB was created,
  //                    or another application image of at least 8 MiB was allocated.
  //   - #<id> ...      the application released the D3D9 object that owns the image.
  //   x #<id> ...      the image itself was freed, with how many frames after the application released it.
  //   screen ...       the set of render targets and depth stencils at the back buffer's size changed.
  //   backbuffers ...  a swap chain made its back buffers.
  //   summary / group / held / screen-sized lines, every rtx.debug.allocationTraceSummarySeconds and when the trace
  //   is turned off: the live set, grouped by shape, next to the AppTexture memory category, and for each image
  //   still alive after the application released it, what in Remix references it.
  // Lines for images of at least 8 MiB are never dropped. Lines for smaller ones are limited to a few per second.
  // Images allocated before the trace was turned on are not tracked.
  // Off: DxvkImage creation and destruction and the D3D9 hooks each test one bool.
  class AllocationTrace {
  public:
    RTX_OPTION("rtx.debug", bool, allocationTrace, false,
               "Logs the creation and destruction of the application's textures and surfaces to remix-dxvk.log, as "
               "[AllocTrace] lines: every render target and depth stencil, and every texture of at least 8 MiB, with "
               "its size, format, usage and frame. At intervals it also logs a summary of the live application "
               "images next to the app texture memory, and for each image Remix still holds after the game released "
               "it, what holds it. Images created before the trace is turned on are not tracked, so turn it on in "
               "rtx.conf to trace a whole session. Costs CPU time and log space while on.");
    RTX_OPTION("rtx.debug", uint32_t, allocationTraceSummarySeconds, 10,
               "Seconds between the summaries rtx.debug.allocationTrace logs. 0 logs a summary only when the trace "
               "is turned off.");

    // The D3D9 description of a texture or surface, filled by the D3D9 frontend.
    struct D3D9Info {
      // "texture", "cube", "volume", "surface", "render target", "depth stencil" or "back buffer".
      const char* kind = "";
      std::string format;
      std::string pool;
      std::string usage;
      uint32_t width = 0;
      uint32_t height = 0;
      uint32_t depth = 1;
      uint32_t mipLevels = 1;
      bool renderTarget = false;
      bool depthStencil = false;
      bool backBuffer = false;
    };

    // True while the trace runs. Any thread.
    static bool active() {
      return s_active.load(std::memory_order_relaxed);
    }

    // DxvkImage, any thread. name is the image's debug name, a string literal.
    static void onImageCreated(DxvkImage& image, const char* name);
    static void onImageDestroyed(const DxvkImage& image);

    // D3D9 frontend, any thread.
    // The reference counts belong to the COM object that owns the image. They must stay readable until
    // onD3D9Released returns for the image.
    static void onD3D9Created(const DxvkImage& image, const D3D9Info& info);
    static void onD3D9OwnerRefs(const DxvkImage& image, const std::atomic<uint32_t>* publicRefs,
                                const std::atomic<uint32_t>* privateRefs);
    static void onD3D9Released(const DxvkImage& image);
    static void onBackBuffers(uint32_t width, uint32_t height, const std::string& format, uint32_t count);

    // CS thread, before each present. Starts and stops the trace and logs the summaries.
    static void onPresent(RtxContext& ctx);

    // Adds the number of references a holder keeps on each image.
    using HolderCounter = void (*)(std::unordered_map<const DxvkImage*, uint32_t>& counts);

    // The Remix API registers its host-import and API-texture counters, so libdxvk does not link against it.
    static void setHolderCounters(HolderCounter hostImports, HolderCounter apiTextures);

  private:
    inline static std::atomic<bool> s_active { false };
    inline static std::atomic<HolderCounter> s_countHostImports { nullptr };
    inline static std::atomic<HolderCounter> s_countApiTextures { nullptr };
  };
}
