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

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "rtx_texture.h"
#include "../../util/thread.h"

namespace dxvk {
  class RtxContext;

  // Derived height+cone textures for the height maps of displaced materials (rtx.displacement.mode =
  // SCCR, pom_sccr.slangh). R is a copy of the height, every mip; G is the square root of the cone ratio (relaxed or
  // conservative, see ConeVariant) at mip 0, in UV per unit of depth. A material points at the derived texture from its
  // creation on, so the texture index never changes (preserved instances keep the index they were built with); until its cones are ready G is 0, which
  // the march reads as "take a linear step". Generation (pom_cone_generate.comp.slang) walks candidate rays only in a
  // small near field and finds every other texel's bound with a hierarchical search of a max-height pyramid. Square
  // power-of-two maps get coarse conservative cones (one per cell of 8x8 texels or more, pom_cone_coarse.comp.slang)
  // within a frame of their copy; generated tiles replace them as they finish. rtx.displacement.coneMap.quality =
  // CoarseOnly puts coarse cones back into every map, for comparison.
  class PomConeMapManager : public CommonDeviceObject {
  public:
    explicit PomConeMapManager(DxvkDevice* device);
    ~PomConeMapManager();

    // Which cones a derived texture holds. Relaxed (GPU Gems 3 ch. 18): a cone may contain the field once, past the
    // first surface the ray meets; the widest cones, for a single height map. Conservative: a cone contains no point of
    // the field at all (the generator without its ray walk), so cones of several maps blended with fixed weights still
    // bound the blend (terrain layers). Each variant gets its own derived texture and its own cache files, so a height
    // map used both ways is generated once per variant and the two never share a texture.
    enum class ConeVariant : uint32_t {
      Relaxed,
      Conservative
    };

    // What G of the derived textures holds (rtx.displacement.coneMap.quality). Full: generated cones, coarse ones until
    // a map's tiles are done. CoarseOnly: coarse cones in every map that allows them (square power-of-two maps of more
    // than one tile; the others keep G = 0, a linear march, as before generation), for comparing the two; generation,
    // cache reads and twin copies pause, and no coarse cone ever reaches the disk cache.
    enum class ConeQuality : uint32_t {
      Full,
      CoarseOnly
    };

    // The texture a material displaced with `height` should use; coneAvailable says whether it is a derived one.
    TextureRef resolveHeight(const TextureRef& height, bool displaced, bool& coneAvailable,
                             ConeVariant variant = ConeVariant::Relaxed);

    // Records the pending cone map work. Once per frame, after the frame's materials are created, before ray tracing.
    void update(RtxContext& ctx);

    // What is left to do, for the settings UI.
    struct Progress {
      uint32_t pendingMaps = 0;
      // Every derived texture, by what its G holds: finished cones (generated, read from the cache or copied from a
      // twin), coarse cones (a map waiting for generation, part way through it, or in CoarseOnly), or neither (G = 0
      // where no tile is generated yet: a linear march).
      uint32_t liveMaps = 0;
      uint32_t fullMaps = 0;
      uint32_t coarseMaps = 0;
      // Tiles of 64x64 still to generate, counting maps not started yet in full (a cache hit spares them).
      uint64_t tilesRemaining = 0;
      size_t derivedBytes = 0;
    };
    Progress progress();

    // Bake now: the next update generates every queued map at bakeTexelsPerFrame until the queue is empty. Called by
    // the settings UI from any thread. Not an option write: see bakeNow.
    void requestBake() {
      m_bakeRequested = true;
    }
    // A bake is running, or was requested and not yet seen by update.
    bool baking() const {
      return m_baking || m_bakeRequested;
    }

    // Every cone in G of every derived texture is conservative: coarse cones, or cones generated without a ray walk
    // (rayStepsPerCandidate = 0, or the conservative variant), or none. A march that leaves the field (shell-traced
    // self-shadows, SCCR_SHELL_MARCH_FLAG_CONSERVATIVE_CONES) may then take cone steps. As of the last update; after a
    // switch to CoarseOnly it turns true once the coarse pass has overwritten every map's generated cones.
    bool conesConservative() const {
      return m_conesConservative;
    }

    RTX_OPTION("rtx.displacement.coneMap", bool, enable, true,
               "SCCR: generates a combined height and cone texture for the height maps of displaced materials. Only in "
               "SCCR mode (rtx.displacement.mode = 3). The other modes use the source height textures and pay nothing.");
    RTX_OPTION("rtx.displacement.coneMap", ConeQuality, quality, ConeQuality::Full,
               "SCCR: which cones the march uses. 0 = Full: generated cones, with coarse cones until a map's generation "
               "is done. 1 = CoarseOnly, for comparison: every map gets coarse cones (one cone per 8x8 texels) and "
               "generation pauses. Maps that cannot have coarse cones (not square powers of two, or a single 64x64 "
               "tile) march linearly. Coarse cones are never written to the disk cache. Switching back to Full restores "
               "the generated cones from the disk cache, or resumes generation. Uses no extra memory.");
    RTX_OPTION("rtx.displacement", bool, levelHeightMaps, true,
               "SCCR: raises each height map so the top of its field is the surface: every height gets 1 - top added "
               "and is clamped to 1, where top is the brightest texel of the map's 16x16 mip (the broad top, not a few "
               "bright texels). Height maps made for a mid-grey surface otherwise sit entirely below the surface, look "
               "sunken and swim. The depth range is unchanged. Applies to height maps loaded after the change; already "
               "loaded maps keep their copy until the game loads them again. Only in SCCR mode (rtx.displacement.mode "
               "= 3) with cone maps on.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, maxResolution, 4096,
               "SCCR: height maps larger than this on either side get no cone map and are marched linearly.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, generateTexelsPerFrame, 16384,
               "SCCR: the number of cone map texels generated per frame, in 64x64 tiles (at least one tile), so "
               "generation never stalls a frame. The default of 4 tiles generates a 2048x2048 map in 256 frames, about "
               "4 seconds at 60 fps. Each map is generated once (see rtx.displacement.coneMap.enableDiskCache), and the "
               "PomConeMaps GPU stage shows the cost. Until a map is generated, coarse cones let the cone march work "
               "anyway.");
    // Never written by the runtime. An option set from the render thread lands in the Derived layer, which outranks the
    // Remix Config layer the settings UI writes, so a reset there would mask every later press of the button.
    RTX_OPTION_FLAG("rtx.displacement.coneMap", bool, bakeNow, false, RtxOptionFlags::NoSave,
                    "SCCR: changing this from false to true starts a bake, as the Bake now button in the Displacement "
                    "settings does: every queued cone map is generated at rtx.displacement.coneMap.bakeTexelsPerFrame "
                    "until the queue is empty. Maps that come into view later use the normal budget. The runtime does "
                    "not reset it, so a host must set it back to false before requesting another bake. Not saved.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, bakeTexelsPerFrame, 262144,
               "SCCR: the number of cone map texels generated per frame during a bake (Bake now, or "
               "rtx.displacement.coneMap.bakeNow), in 64x64 tiles. The default of 64 tiles generates a 2048x2048 map in "
               "16 frames, at a few ms of GPU time per frame. Work is split into short dispatches, so a bake cannot "
               "cause a GPU timeout.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, nearRadius, 2,
               "SCCR: the radius in texels (1 to 32) of the near field, where cone generation walks each candidate ray "
               "(see rtx.displacement.coneMap.rayStepsPerCandidate). A larger radius widens the cones a little, which "
               "makes the march slightly faster, at a much higher generation cost. On maps that are not square powers "
               "of two, a window of at least 8 texels is used.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, rayStepsPerCandidate, 4,
               "SCCR: how many one-texel steps cone generation walks along each near-field candidate ray (0 to 64) to "
               "find where it leaves the height field. Fewer steps give narrower cones and a slower march, but never "
               "wrong ones. 0 gives conservative cones (no point of the height field inside any cone), which generate "
               "fastest, need a few percent more march steps and also allow shell-traced self-shadows to use the cones. "
               "A change applies to maps generated afterwards.");
    RTX_OPTION("rtx.displacement.coneMap", bool, enableDiskCache, true,
               "SCCR: keeps generated cones on disk, keyed by the height map's hash, its size and the generation "
               "settings, so each map is generated only once.");
    RTX_OPTION("rtx.displacement.coneMap", std::string, cachePath, std::string(""),
               "SCCR: the folder of the cone cache. Empty = rtx-remix/cache/pom_cone next to the logs folder.");
    RTX_OPTION("rtx.displacement.coneMap", uint32_t, cacheMaxMB, 1024,
               "SCCR: the cone cache is trimmed to this size in MB at startup, oldest files first.");

  private:
    struct Entry;

    bool createImage(Entry& entry) const;
    bool beginGeneration(Entry& entry) const;
    void recordLevel(RtxContext& ctx, Entry& entry) const;
    static bool levelApplies(const Entry& entry, const Rc<DxvkImageView>& height);
    static bool hasHeight(const Entry& entry);
    static Rc<DxvkImageView> heightSource(const Entry& entry);
    Rc<DxvkImageView> mergeHeight(RtxContext& ctx, const Entry& entry) const;
    void recordCopy(RtxContext& ctx, Entry& entry, uint32_t mode) const;
    void recordPyramid(RtxContext& ctx, Entry& entry) const;
    void recordMerge(RtxContext& ctx, Entry& entry, const Rc<DxvkImageView>& height, const Rc<DxvkImageView>& cones,
                     uint32_t mode, const VkRect2D& region, uint32_t coarseShift) const;
    void recordMergeGenerated(RtxContext& ctx, Entry& entry) const;
    bool recordCoarse(RtxContext& ctx, Entry& entry) const;
    void onQualityChange();
    uint32_t recordGenerate(RtxContext& ctx, Entry& entry, uint32_t texelBudget, std::vector<VkRect2D>& runs) const;
    const Entry* findTwin(const Entry& entry, bool generating) const;
    void recordTwinCopy(RtxContext& ctx, Entry& entry, const Entry& twin) const;
    void finish(Entry& entry);
    std::string cacheDirectory() const;
    void startCacheLoad(Entry& entry) const;
    bool applyCachedCones(RtxContext& ctx, Entry& entry);
    void storeInCache(RtxContext& ctx, const Entry& entry) const;
    void trimCache() const;

    dxvk::mutex m_mutex;
    // By the derived texture's unique key.
    std::unordered_map<size_t, std::unique_ptr<Entry>> m_entries;
    std::deque<size_t> m_pending;
    size_t m_totalBytes = 0;
    uint32_t m_generatedSinceReport = 0;
    uint32_t m_loadedSinceReport = 0;
    uint32_t m_copiedSinceReport = 0;
    uint32_t m_coarseSinceReport = 0;
    bool m_cacheTrimmed = false;
    std::atomic<bool> m_bakeRequested { false };
    std::atomic<bool> m_baking { false };
    // bakeNow as update last saw it: a bake starts on its rising edge.
    bool m_bakeOptionSeen = false;
    // quality == CoarseOnly as update last applied it.
    bool m_coarseOnly = false;
    std::atomic<bool> m_conesConservative { true };
  };
}
