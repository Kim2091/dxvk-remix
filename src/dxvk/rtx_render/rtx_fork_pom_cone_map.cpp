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
#include "rtx_fork_pom_cone_map.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <vector>

#include "rtx_asset_exporter.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_shader_manager.h"
#include "rtx_texture_manager.h"
#include "../dxvk_buffer.h"
#include "../dxvk_device.h"
#include "../dxvk_format.h"
#include "../dxvk_image.h"
#include "../dxvk_scoped_annotation.h"
#include "../dxvk_util.h"
#include "../../util/log/log.h"
#include "../../util/util_filesys.h"
#include "../../util/util_string.h"
#include "../../util/xxHash/xxhash.h"

#include <gli/gli.hpp>

#include "rtx/pass/pom_cone/pom_cone_args.h"

#include <rtx_shaders/pom_cone_coarse.h>
#include <rtx_shaders/pom_cone_copy.h>
#include <rtx_shaders/pom_cone_generate.h>

namespace dxvk {
  namespace {
    class PomConeCopyShader : public ManagedShader {
      SHADER_SOURCE(PomConeCopyShader, VK_SHADER_STAGE_COMPUTE_BIT, pom_cone_copy)

      PUSH_CONSTANTS(PomConeCopyArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POM_CONE_BINDING_SOURCE)
        TEXTURE2D(POM_CONE_BINDING_CONE)
        RW_TEXTURE2D(POM_CONE_BINDING_OUTPUT)
        TEXTURE2D(POM_CONE_BINDING_LEVEL)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(PomConeCopyShader);

    class PomConeGenerateShader : public ManagedShader {
      SHADER_SOURCE(PomConeGenerateShader, VK_SHADER_STAGE_COMPUTE_BIT, pom_cone_generate)

      PUSH_CONSTANTS(PomConeGenerateArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POM_CONE_BINDING_SOURCE)
        RW_TEXTURE2D(POM_CONE_BINDING_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(PomConeGenerateShader);

    class PomConeCoarseShader : public ManagedShader {
      SHADER_SOURCE(PomConeCoarseShader, VK_SHADER_STAGE_COMPUTE_BIT, pom_cone_coarse)

      PUSH_CONSTANTS(PomConeCoarseArgs)

      BEGIN_PARAMETER()
        TEXTURE2D(POM_CONE_BINDING_SOURCE)
        TEXTURE2D(POM_CONE_BINDING_CONE)
        RW_TEXTURE2D(POM_CONE_BINDING_OUTPUT)
      END_PARAMETER()
    };
    PREWARM_SHADER_PIPELINE(PomConeCoarseShader);

    // Derived keys live in the texture manager's key space next to the hashes of every other texture.
    constexpr XXH64_hash_t kDerivedKeySeed = 0x53434352434f4e45ull; // "SCCRCONE"
    // Conservative cones (terrain layers) have keys of their own, for the texture table and the disk cache alike.
    // A source used both as a single height map and as a terrain layer gets two derived textures that never collide.
    // Relaxed keys keep the seeds above, and 0 for the cache.
    constexpr XXH64_hash_t kConservativeDerivedKeySeed = 0x53434352434f4e43ull; // "SCCRCONC"
    constexpr XXH64_hash_t kConservativeCacheKeySeed = 0x434f4e5345525600ull;   // "CONSERV"
    // Bump when pom_cone_generate.comp.slang changes what it computes, since the disk cache is keyed on it.
    constexpr uint32_t kConeCacheVersion = 2;
    constexpr uint32_t kMaxCacheLoadsInFlight = 4;

    VkFormat scratchFormat(VkFormat heightConeFormat) {
      return heightConeFormat == VK_FORMAT_R8G8_UNORM ? VK_FORMAT_R8_UNORM : VK_FORMAT_R16_UNORM;
    }

    uint32_t clampedNearRadius(uint32_t nearRadius) {
      return std::clamp(nearRadius, 1u, 32u);
    }

    uint32_t clampedRaySteps(uint32_t raySteps) {
      return std::min(raySteps, 64u);
    }

    std::string coneCacheFile(XXH64_hash_t sourceHash, const VkExtent3D& extent, VkFormat format, uint32_t nearRadius,
                              uint32_t raySteps, bool conservative) {
      struct Key {
        uint64_t sourceHash;
        uint32_t width;
        uint32_t height;
        uint32_t format;
        uint32_t version;
        uint32_t nearRadius;
        uint32_t raySteps;
      };
      static_assert(sizeof(Key) == 32, "Key must have no padding: it is hashed by memory.");
      const Key key { sourceHash, extent.width, extent.height, uint32_t(format), kConeCacheVersion, nearRadius, raySteps };
      return str::format(std::hex, XXH64(&key, sizeof(key), conservative ? kConservativeCacheKeySeed : 0), ".dds");
    }

    VkFormat derivedFormat(VkFormat sourceFormat) {
      const DxvkFormatInfo* formatInfo = imageFormatInfo(sourceFormat);
      if (formatInfo == nullptr || (formatInfo->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
        return VK_FORMAT_UNDEFINED;
      }
      switch (sourceFormat) {
      case VK_FORMAT_R8_UNORM:
      case VK_FORMAT_R8_SRGB:
      case VK_FORMAT_R8G8_UNORM:
      case VK_FORMAT_R8G8_SRGB:
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_SRGB:
      case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
      case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
      case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
      case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
      case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      case VK_FORMAT_BC2_UNORM_BLOCK:
      case VK_FORMAT_BC2_SRGB_BLOCK:
      case VK_FORMAT_BC3_UNORM_BLOCK:
      case VK_FORMAT_BC3_SRGB_BLOCK:
      case VK_FORMAT_BC7_UNORM_BLOCK:
      case VK_FORMAT_BC7_SRGB_BLOCK:
        return VK_FORMAT_R8G8_UNORM;
      default:
        // More than 8 bits of height (R16, BC4/BC5, float formats).
        return VK_FORMAT_R16G16_UNORM;
      }
    }

    Rc<DxvkImage> createScratchImage(DxvkDevice* device, VkFormat format, VkExtent3D extent, uint32_t mipLevels,
                                     const char* name) {
      DxvkImageCreateInfo imageInfo = {};
      imageInfo.type        = VK_IMAGE_TYPE_2D;
      imageInfo.format      = format;
      imageInfo.flags       = 0;
      imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      imageInfo.extent      = { extent.width, extent.height, 1 };
      imageInfo.numLayers   = 1;
      imageInfo.mipLevels   = mipLevels;
      imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                            | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
      imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;
      return device->createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 DxvkMemoryStats::Category::RTXRenderTarget, name);
    }

    Rc<DxvkImageView> createView(DxvkDevice* device, const Rc<DxvkImage>& image, VkImageUsageFlags usage,
                                 uint32_t minLevel, uint32_t numLevels) {
      DxvkImageViewCreateInfo viewInfo = {};
      viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
      viewInfo.format    = image->info().format;
      viewInfo.usage     = usage;
      viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
      viewInfo.minLevel  = minLevel;
      viewInfo.numLevels = numLevels;
      viewInfo.minLayer  = 0;
      viewInfo.numLayers = 1;
      return device->createImageView(image, viewInfo);
    }

    uint32_t tileCount(const VkExtent3D& extent) {
      const uint32_t tilesX = (extent.width + POM_CONE_TILE_SIZE - 1) / POM_CONE_TILE_SIZE;
      const uint32_t tilesY = (extent.height + POM_CONE_TILE_SIZE - 1) / POM_CONE_TILE_SIZE;
      return tilesX * tilesY;
    }

    // Coarse cones (pom_cone_coarse.comp.slang) for this many texels of maps per frame, and for at least one map: about
    // one 2048x2048 map, estimated well under 1 ms.
    constexpr uint64_t kCoarseTexelsPerFrame = 2048ull * 2048ull;

    uint32_t log2Floor(uint32_t value) {
      uint32_t result = 0;
      while ((value >> (result + 1)) != 0) {
        ++result;
      }
      return result;
    }

    // Maps without the hierarchical search bound everything outside their window by its distance alone, which is weak
    // for a small window: theirs is at least this wide.
    constexpr uint32_t kMinWindowRadius = 8;

    // The generator's settings for a map (pom_cone_generate.comp.slang), as beginGeneration uses them and the disk
    // cache keys them. Square power-of-two maps get the whole max-height pyramid (to 1x1) and the hierarchical search;
    // their near field matters only with a walk, so a map without one keys no radius. Other maps keep the window.
    struct GenerationParams {
      uint32_t nearRadius = 0;
      uint32_t raySteps = 0;
      uint32_t farLevels = 0;
    };

    GenerationParams generationParams(const VkExtent3D& extent, bool conservative, uint32_t nearRadiusOption,
                                      uint32_t rayStepsOption) {
      GenerationParams params;
      params.raySteps = conservative ? 0u : clampedRaySteps(rayStepsOption);
      const uint32_t width = extent.width;
      if (width == extent.height && width > 1 && (width & (width - 1)) == 0) {
        params.farLevels = log2Floor(width);
        params.nearRadius = params.raySteps > 0 ? clampedNearRadius(nearRadiusOption) : 0u;
      } else {
        params.nearRadius = std::max(clampedNearRadius(nearRadiusOption), kMinWindowRadius);
      }
      return params;
    }

    // The max-height pyramid of a source: level 0 a copy of the height in the pyramid's precision, every further level
    // the largest of the 2x2 texels under it. Leaves the pyramid readable by shaders.
    void recordMaxPyramid(RtxContext& ctx, const Rc<DxvkImageView>& sourceView, const Rc<DxvkImage>& pyramid,
                          const std::vector<Rc<DxvkImageView>>& sampledViews,
                          const std::vector<Rc<DxvkImageView>>& storageViews) {
      ctx.changeImageLayout(pyramid, VK_IMAGE_LAYOUT_GENERAL);
      ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
      ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeCopyShader::getShader());
      for (uint32_t level = 0; level < storageViews.size(); ++level) {
        const VkExtent3D levelExtent = pyramid->mipLevelExtent(level);
        const Rc<DxvkImageView>& source = level == 0 ? sourceView : sampledViews[level - 1];
        PomConeCopyArgs args = {};
        args.extent = { levelExtent.width, levelExtent.height };
        if (level == 0) {
          args.sourceExtent = args.extent;
          args.mode = POM_CONE_COPY_HEIGHT;
        } else {
          const VkExtent3D sourceExtent = pyramid->mipLevelExtent(level - 1);
          args.sourceExtent = { sourceExtent.width, sourceExtent.height };
          args.mode = POM_CONE_REDUCE_MAX;
        }
        args.sourceMip = 0;
        ctx.pushConstants(0, sizeof(args), &args);
        ctx.bindResourceView(POM_CONE_BINDING_SOURCE, source, nullptr);
        ctx.bindResourceView(POM_CONE_BINDING_CONE, source, nullptr);
        ctx.bindResourceView(POM_CONE_BINDING_LEVEL, source, nullptr);
        ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, storageViews[level], nullptr);
        const VkExtent3D workgroups = util::computeBlockCount(levelExtent, VkExtent3D { 16, 16, 1 });
        ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
      }
      ctx.changeImageLayout(pyramid, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  }

  struct PomConeMapManager::Entry {
    enum class State {
      Copy,        // the height is not copied yet
      CacheLookup, // copied (valid, G = 0); the disk cache is not asked yet
      CacheLoad,   // reading the disk cache on a worker thread
      Pyramid,     // copied; cone generation has not started
      Generate,    // generating tiles
      Done
    };

    // Identity only, not owned: a different image under the same key (a host replacing a texture in place) is copied
    // again.
    const DxvkImage* sourceImage = nullptr;
    // Held only while work on the entry is pending.
    Rc<DxvkImageView> sourceView;
    XXH64_hash_t sourceHash = 0;
    // The source view's format: an sRGB and a UNORM view of one image read different heights.
    VkFormat sourceFormat = VK_FORMAT_UNDEFINED;
    VkExtent3D extent = { 0, 0, 0 };
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t mipLevels = 0;
    Rc<DxvkImage> image;
    Rc<DxvkImageView> view;                  // sampled, every mip: what the texture table holds
    std::vector<Rc<DxvkImageView>> mipViews; // storage, one per mip
    size_t bytes = 0;
    bool pending = false;
    // ConeVariant::Conservative: generated without the ray walk. Fixed by the derived key.
    bool conservative = false;
    State state = State::Copy;
    // G holds finished cones (generated, read from the cache or copied from a twin), not zeros or a partial result.
    bool hasCones = false;
    // Coarse cones (pom_cone_coarse.comp.slang) were tried for G, and made (hasCoarse) unless the map does not allow them;
    // until generation finishes, G holds them where no tile is generated yet.
    bool coarse = false;
    bool hasCoarse = false;
    // G holds generated values (finished cones, or tiles merged so far) that coarse-only quality must overwrite.
    bool refinedInG = false;
    // Some of those values were generated with a ray walk (relaxed cones). Cleared when coarse cones or zeros overwrite
    // all of G (conesConservative).
    bool relaxedInG = false;
    // The walk the pending cache read is keyed with (startCacheLoad): what the cached cones hold.
    uint32_t cacheRaySteps = 0;
    // Back in the queue after it was done (coarse-only quality), without its source: the height comes from the derived
    // texture's own R, which holds the source height in the precision every pass works in (heightSource).
    bool heightFromDerived = false;
    // R is levelled (levelHeightMaps): level holds the top of the source's field, and every R written from the source
    // is raised by 1 - top. R itself, and anything made from it, is already levelled.
    bool levelled = false;
    Rc<DxvkImage> level;
    Rc<DxvkImageView> levelSampledView;
    Rc<DxvkImageView> levelStorageView;
    // The generation pyramid was made from the source, not from the levelled R.
    bool pyramidRaw = false;

    // Cone generation scratch, alive from beginGeneration to finish. pyramid: max heights, mip 0 = the height map in
    // the derived R precision; cone: the generated G.
    Rc<DxvkImage> pyramid;
    Rc<DxvkImageView> pyramidView;
    std::vector<Rc<DxvkImageView>> pyramidSampledViews;
    std::vector<Rc<DxvkImageView>> pyramidStorageViews;
    Rc<DxvkImage> cone;
    Rc<DxvkImageView> coneSampledView;
    Rc<DxvkImageView> coneStorageView;
    uint32_t farLevels = 0;
    uint32_t nearRadius = 0;
    uint32_t rayStepsPerCandidate = 0;
    uint32_t nextTile = 0;
    std::future<gli::texture> cacheLoad;
  };

  PomConeMapManager::PomConeMapManager(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  PomConeMapManager::~PomConeMapManager() = default;

  TextureRef PomConeMapManager::resolveHeight(const TextureRef& height, bool displaced, bool& coneAvailable,
                                              ConeVariant variant) {
    coneAvailable = false;
    // Only SCCR reads the cones. A material created in another mode keeps its source height texture, which SCCR
    // marches linearly if the mode changes while the material lives on (preserved instances).
    if (!displaced || !enable() || RtxOptions::Displacement::mode() != DisplacementMode::SCCR) {
      return height;
    }
    // Streamed replacement textures change their top mip while they stream, so they are not handled.
    if (!height.isValid() || height.getManagedTexture() != nullptr) {
      return height;
    }
    DxvkImageView* sourceView = height.getImageView();
    // A zero image hash marks a render target (e.g. terrain baker output), whose contents change after creation.
    if (sourceView == nullptr || height.getImageHash() == 0) {
      return height;
    }
    const DxvkImageViewCreateInfo& viewInfo = sourceView->info();
    const VkExtent3D extent = sourceView->mipLevelExtent(0);
    const VkFormat format = derivedFormat(viewInfo.format);
    if (viewInfo.type != VK_IMAGE_VIEW_TYPE_2D || viewInfo.numLayers != 1 || format == VK_FORMAT_UNDEFINED ||
        extent.width == 0 || extent.height == 0 || std::max(extent.width, extent.height) > maxResolution()) {
      return height;
    }

    std::lock_guard<dxvk::mutex> lock(m_mutex);

    const bool conservative = variant == ConeVariant::Conservative;
    const size_t sourceKey = height.getUniqueKey();
    const size_t derivedKey = XXH64(&sourceKey, sizeof(sourceKey),
                                    conservative ? kConservativeDerivedKeySeed : kDerivedKeySeed);
    std::unique_ptr<Entry>& entry = m_entries[derivedKey];
    if (entry == nullptr || entry->sourceImage != sourceView->image().ptr()) {
      const XXH64_hash_t sourceHash = height.getImageHash();
      const bool sameShape = entry != nullptr && entry->format == format && entry->mipLevels == viewInfo.numLevels &&
                             entry->extent.width == extent.width && entry->extent.height == extent.height;
      // The same contents under another image (a copy-on-write import, a texture created again with the same data):
      // the derived texture is already right, or being made from identical data. Redoing it would restart the copy,
      // the cache read or the generation every time the key's image changes. The disk cache trusts the same hash.
      if (sameShape && entry->sourceHash == sourceHash && entry->sourceFormat == viewInfo.format) {
        entry->sourceImage = sourceView->image().ptr();
        if (entry->pending) {
          entry->sourceView = Rc<DxvkImageView>(sourceView);
        }
        coneAvailable = true;
        return TextureRef(entry->view, derivedKey);
      }
      if (!sameShape) {
        auto fresh = std::make_unique<Entry>();
        fresh->extent = extent;
        fresh->format = format;
        fresh->mipLevels = viewInfo.numLevels;
        fresh->conservative = conservative;
        if (!createImage(*fresh)) {
          if (entry == nullptr) {
            m_entries.erase(derivedKey);
          }
          return height;
        }
        if (entry != nullptr) {
          // The texture table may still hold the old image under the derived key.
          m_totalBytes -= entry->bytes;
          m_device->getCommon()->getTextureManager().replaceTexture(TextureRef(fresh->view, derivedKey));
          fresh->pending = entry->pending;
        }
        m_totalBytes += fresh->bytes;
        entry = std::move(fresh);
      }
      entry->sourceImage = sourceView->image().ptr();
      entry->sourceView = Rc<DxvkImageView>(sourceView);
      entry->sourceHash = sourceHash;
      entry->sourceFormat = viewInfo.format;
      entry->state = Entry::State::Copy;
      entry->hasCones = false;
      entry->coarse = false;
      entry->hasCoarse = false;
      entry->refinedInG = false;
      // The copy this frame's update records clears G before any ray tracing.
      entry->relaxedInG = false;
      entry->heightFromDerived = false;
      entry->levelled = false;
      if (!entry->pending) {
        entry->pending = true;
        m_pending.push_back(derivedKey);
      }
    }

    coneAvailable = true;
    return TextureRef(entry->view, derivedKey);
  }

  bool PomConeMapManager::createImage(Entry& entry) const {
    DxvkImageCreateInfo imageInfo = {};
    imageInfo.type        = VK_IMAGE_TYPE_2D;
    imageInfo.format      = entry.format;
    imageInfo.flags       = 0;
    imageInfo.sampleCount = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.extent      = { entry.extent.width, entry.extent.height, 1 };
    imageInfo.numLayers   = 1;
    imageInfo.mipLevels   = entry.mipLevels;
    // Transfer destination: a twin's finished texture is copied in (recordTwinCopy).
    imageInfo.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                          | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.stages      = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
    imageInfo.access      = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
                          | VK_ACCESS_TRANSFER_WRITE_BIT;
    imageInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.layout      = VK_IMAGE_LAYOUT_UNDEFINED;

    entry.image = m_device->createImage(imageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                        DxvkMemoryStats::Category::RTXMaterialTexture, "SCCR height+cone map");
    if (entry.image == nullptr) {
      Logger::err("[SCCR] Failed to create a height+cone map image.");
      return false;
    }

    DxvkImageViewCreateInfo viewInfo = {};
    viewInfo.type      = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format    = entry.format;
    viewInfo.usage     = VK_IMAGE_USAGE_SAMPLED_BIT;
    viewInfo.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.minLevel  = 0;
    viewInfo.numLevels = entry.mipLevels;
    viewInfo.minLayer  = 0;
    viewInfo.numLayers = 1;
    entry.view = m_device->createImageView(entry.image, viewInfo);

    viewInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    viewInfo.numLevels = 1;
    entry.mipViews.clear();
    entry.bytes = 0;
    const size_t texelBytes = entry.format == VK_FORMAT_R8G8_UNORM ? 2 : 4;
    for (uint32_t mip = 0; mip < entry.mipLevels; ++mip) {
      viewInfo.minLevel = mip;
      entry.mipViews.push_back(m_device->createImageView(entry.image, viewInfo));
      const VkExtent3D mipExtent = entry.image->mipLevelExtent(mip);
      entry.bytes += size_t(mipExtent.width) * mipExtent.height * texelBytes;
    }

    entry.level = createScratchImage(m_device, VK_FORMAT_R32G32_SFLOAT, VkExtent3D { 1, 1, 1 }, 1, "SCCR height level");
    if (entry.level == nullptr) {
      Logger::err("[SCCR] Failed to create a height level image.");
      return false;
    }
    entry.levelSampledView = createView(m_device, entry.level, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1);
    entry.levelStorageView = createView(m_device, entry.level, VK_IMAGE_USAGE_STORAGE_BIT, 0, 1);
    return entry.view != nullptr;
  }

  void PomConeMapManager::recordLevel(RtxContext& ctx, Entry& entry) const {
    ctx.changeImageLayout(entry.level, VK_IMAGE_LAYOUT_GENERAL);
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeCopyShader::getShader());
    ctx.bindResourceView(POM_CONE_BINDING_SOURCE, entry.sourceView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_CONE, entry.sourceView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_LEVEL, entry.sourceView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, entry.levelStorageView, nullptr);
    PomConeCopyArgs args = {};
    args.extent = { POM_CONE_LEVEL_GRID, POM_CONE_LEVEL_GRID };
    args.sourceExtent = args.extent;
    args.mode = POM_CONE_LEVEL;
    ctx.pushConstants(0, sizeof(args), &args);
    ctx.dispatch(1, 1, 1);
    ctx.changeImageLayout(entry.level, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  bool PomConeMapManager::levelApplies(const Entry& entry, const Rc<DxvkImageView>& height) {
    // Only a height read from the source is raised: one read from R, or from a pyramid made from R, is levelled already.
    if (!entry.levelled || height == nullptr) {
      return false;
    }
    return height == entry.sourceView ||
           (entry.pyramidRaw && !entry.pyramidSampledViews.empty() && height == entry.pyramidSampledViews[0]);
  }

  bool PomConeMapManager::beginGeneration(Entry& entry) const {
    // Square power-of-two maps: the pyramid reaches 1x1 for the hierarchical search (farLevels = log2(width)).
    const GenerationParams params = generationParams(entry.extent, entry.conservative, nearRadius(), rayStepsPerCandidate());
    entry.nearRadius = params.nearRadius;
    entry.rayStepsPerCandidate = params.raySteps;
    entry.farLevels = params.farLevels;
    entry.nextTile = 0;

    const VkFormat format = scratchFormat(entry.format);
    const uint32_t pyramidLevels = entry.farLevels + 1;
    entry.pyramid = createScratchImage(m_device, format, entry.extent, pyramidLevels, "SCCR cone generation pyramid");
    entry.cone = createScratchImage(m_device, format, entry.extent, 1, "SCCR cone generation output");
    if (entry.pyramid == nullptr || entry.cone == nullptr) {
      Logger::err("[SCCR] Failed to create cone generation images.");
      return false;
    }
    entry.pyramidView = createView(m_device, entry.pyramid, VK_IMAGE_USAGE_SAMPLED_BIT, 0, pyramidLevels);
    entry.pyramidSampledViews.clear();
    entry.pyramidStorageViews.clear();
    for (uint32_t level = 0; level < pyramidLevels; ++level) {
      entry.pyramidSampledViews.push_back(createView(m_device, entry.pyramid, VK_IMAGE_USAGE_SAMPLED_BIT, level, 1));
      entry.pyramidStorageViews.push_back(createView(m_device, entry.pyramid, VK_IMAGE_USAGE_STORAGE_BIT, level, 1));
    }
    entry.coneSampledView = createView(m_device, entry.cone, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1);
    entry.coneStorageView = createView(m_device, entry.cone, VK_IMAGE_USAGE_STORAGE_BIT, 0, 1);
    return true;
  }

  bool PomConeMapManager::hasHeight(const Entry& entry) {
    return entry.sourceView != nullptr || entry.heightFromDerived;
  }

  Rc<DxvkImageView> PomConeMapManager::heightSource(const Entry& entry) {
    // Mip 0 of the derived R is the source's mip 0 in the precision every pass works in (scratchFormat), so pyramids
    // and cones made from it are the same as from the source.
    return entry.sourceView != nullptr ? entry.sourceView : entry.view;
  }

  Rc<DxvkImageView> PomConeMapManager::mergeHeight(RtxContext& ctx, const Entry& entry) const {
    // What a pass writing mip 0 of the derived texture reads R from: never the derived texture itself.
    if (entry.sourceView != nullptr) {
      return entry.sourceView;
    }
    if (!entry.pyramidSampledViews.empty()) {
      return entry.pyramidSampledViews[0];
    }
    // A re-queued entry outside generation (a cache hit, a coarse-only pass): its R copied out for this pass only.
    const Rc<DxvkImage> scratch = createScratchImage(m_device, scratchFormat(entry.format), entry.extent, 1, "SCCR height copy");
    if (scratch == nullptr) {
      Logger::err("[SCCR] Failed to create a height copy image.");
      return nullptr;
    }
    const std::vector<Rc<DxvkImageView>> sampledViews = { createView(m_device, scratch, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1) };
    const std::vector<Rc<DxvkImageView>> storageViews = { createView(m_device, scratch, VK_IMAGE_USAGE_STORAGE_BIT, 0, 1) };
    recordMaxPyramid(ctx, entry.view, scratch, sampledViews, storageViews);
    return sampledViews[0];
  }

  void PomConeMapManager::recordCopy(RtxContext& ctx, Entry& entry, uint32_t mode) const {
    // A height copy writes every mip from the source; the merge rewrites mip 0 with the cones.
    const bool withCones = mode == POM_CONE_COPY_HEIGHT_CONE;
    const uint32_t mipCount = withCones ? 1 : entry.mipLevels;
    const Rc<DxvkImageView> height = withCones ? mergeHeight(ctx, entry) : entry.sourceView;
    if (height == nullptr) {
      return;
    }
    if (withCones) {
      ctx.changeImageLayout(entry.cone, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    ctx.changeImageLayout(entry.image, VK_IMAGE_LAYOUT_GENERAL);
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeCopyShader::getShader());
    const bool level = levelApplies(entry, height);
    ctx.bindResourceView(POM_CONE_BINDING_SOURCE, height, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_CONE, withCones ? entry.coneSampledView : height, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_LEVEL, level ? entry.levelSampledView : height, nullptr);
    for (uint32_t mip = 0; mip < mipCount; ++mip) {
      const VkExtent3D mipExtent = entry.image->mipLevelExtent(mip);
      PomConeCopyArgs args = {};
      args.extent = { mipExtent.width, mipExtent.height };
      args.sourceExtent = args.extent;
      args.sourceMip = mip;
      args.mode = mode;
      args.level = level ? 1u : 0u;
      ctx.pushConstants(0, sizeof(args), &args);
      ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, entry.mipViews[mip], nullptr);
      const VkExtent3D workgroups = util::computeBlockCount(mipExtent, VkExtent3D { 16, 16, 1 });
      ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
    ctx.changeImageLayout(entry.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void PomConeMapManager::recordPyramid(RtxContext& ctx, Entry& entry) const {
    entry.pyramidRaw = entry.sourceView != nullptr;
    recordMaxPyramid(ctx, heightSource(entry), entry.pyramid, entry.pyramidSampledViews, entry.pyramidStorageViews);
  }

  void PomConeMapManager::recordMerge(RtxContext& ctx, Entry& entry, const Rc<DxvkImageView>& height,
                                      const Rc<DxvkImageView>& cones, uint32_t mode, const VkRect2D& region,
                                      uint32_t coarseShift) const {
    if (height == nullptr) {
      return;
    }
    ctx.changeImageLayout(entry.image, VK_IMAGE_LAYOUT_GENERAL);
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeCopyShader::getShader());
    const bool level = levelApplies(entry, height);
    ctx.bindResourceView(POM_CONE_BINDING_SOURCE, height, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_CONE, cones, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_LEVEL, level ? entry.levelSampledView : height, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, entry.mipViews[0], nullptr);
    PomConeCopyArgs args = {};
    args.extent = { entry.extent.width, entry.extent.height };
    args.sourceExtent = args.extent;
    args.sourceMip = 0;
    args.mode = mode;
    args.level = level ? 1u : 0u;
    args.origin = { uint32_t(region.offset.x), uint32_t(region.offset.y) };
    args.coarseShift = coarseShift;
    ctx.pushConstants(0, sizeof(args), &args);
    const VkExtent3D workgroups = util::computeBlockCount(VkExtent3D { region.extent.width, region.extent.height, 1 },
                                                          VkExtent3D { 16, 16, 1 });
    ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
    ctx.changeImageLayout(entry.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void PomConeMapManager::recordMergeGenerated(RtxContext& ctx, Entry& entry) const {
    // The tiles generated so far, back into G after a coarse-only pass overwrote them: the full tile rows, then the
    // start of the next one (tiles go in row order). The scratch holds them still.
    const uint32_t tilesX = (entry.extent.width + POM_CONE_TILE_SIZE - 1) / POM_CONE_TILE_SIZE;
    const uint32_t fullRows = entry.nextTile / tilesX;
    const uint32_t rest = entry.nextTile % tilesX;
    const Rc<DxvkImageView> height = mergeHeight(ctx, entry);
    ctx.changeImageLayout(entry.cone, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const uint32_t rowsHeight = std::min(fullRows * POM_CONE_TILE_SIZE, entry.extent.height);
    if (rowsHeight > 0) {
      const VkRect2D rows = { { 0, 0 }, { entry.extent.width, rowsHeight } };
      recordMerge(ctx, entry, height, entry.coneSampledView, POM_CONE_COPY_HEIGHT_CONE, rows, 0);
    }
    if (rest > 0 && rowsHeight < entry.extent.height) {
      const VkRect2D partial = { { 0, int32_t(rowsHeight) },
                                 { std::min(rest * POM_CONE_TILE_SIZE, entry.extent.width),
                                   std::min(uint32_t(POM_CONE_TILE_SIZE), entry.extent.height - rowsHeight) } };
      recordMerge(ctx, entry, height, entry.coneSampledView, POM_CONE_COPY_HEIGHT_CONE, partial, 0);
    }
  }

  bool PomConeMapManager::recordCoarse(RtxContext& ctx, Entry& entry) const {
    // The pyramid must map cells to texels exactly, and a map of one tile is generated in a frame anyway.
    const uint32_t width = entry.extent.width;
    if (width != entry.extent.height || (width & (width - 1)) != 0 || width <= POM_CONE_TILE_SIZE) {
      return false;
    }
    const uint32_t widthLog2 = log2Floor(width);
    const uint32_t shift = std::max<uint32_t>(POM_CONE_COARSE_MIN_SHIFT,
                                              widthLog2 > POM_CONE_COARSE_MAX_CELLS_LOG2 ? widthLog2 - POM_CONE_COARSE_MAX_CELLS_LOG2 : 0u);
    uint32_t topLevel = shift;
    while (topLevel < widthLog2 && (uint32_t(POM_CONE_COARSE_RING) << topLevel) < width) {
      ++topLevel;
    }

    // Scratch for this pass only: dropped when it returns (the command list keeps them alive until the GPU is done).
    const VkFormat format = scratchFormat(entry.format);
    const uint32_t pyramidLevels = topLevel + 1;
    const VkExtent3D cellsExtent = { width >> shift, width >> shift, 1 };
    const Rc<DxvkImage> pyramid = createScratchImage(m_device, format, entry.extent, pyramidLevels, "SCCR coarse cone pyramid");
    const Rc<DxvkImage> cells = createScratchImage(m_device, VK_FORMAT_R32G32_SFLOAT, cellsExtent, 1, "SCCR coarse cone cells");
    const Rc<DxvkImage> cones = createScratchImage(m_device, VK_FORMAT_R32_SFLOAT, cellsExtent, 1, "SCCR coarse cones");
    if (pyramid == nullptr || cells == nullptr || cones == nullptr) {
      Logger::err("[SCCR] Failed to create coarse cone images.");
      return false;
    }
    std::vector<Rc<DxvkImageView>> pyramidSampledViews;
    std::vector<Rc<DxvkImageView>> pyramidStorageViews;
    for (uint32_t level = 0; level < pyramidLevels; ++level) {
      pyramidSampledViews.push_back(createView(m_device, pyramid, VK_IMAGE_USAGE_SAMPLED_BIT, level, 1));
      pyramidStorageViews.push_back(createView(m_device, pyramid, VK_IMAGE_USAGE_STORAGE_BIT, level, 1));
    }
    const Rc<DxvkImageView> pyramidView = createView(m_device, pyramid, VK_IMAGE_USAGE_SAMPLED_BIT, 0, pyramidLevels);
    const Rc<DxvkImageView> cellsSampledView = createView(m_device, cells, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1);
    const Rc<DxvkImageView> cellsStorageView = createView(m_device, cells, VK_IMAGE_USAGE_STORAGE_BIT, 0, 1);
    const Rc<DxvkImageView> conesSampledView = createView(m_device, cones, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1);
    const Rc<DxvkImageView> conesStorageView = createView(m_device, cones, VK_IMAGE_USAGE_STORAGE_BIT, 0, 1);

    recordMaxPyramid(ctx, heightSource(entry), pyramid, pyramidSampledViews, pyramidStorageViews);

    PomConeCoarseArgs args = {};
    args.extent = { width, width };
    args.coarseShift = shift;
    args.topLevel = topLevel;
    args.coneQuantization = entry.format == VK_FORMAT_R8G8_UNORM ? 255.0f : 65535.0f;

    // Cells: one group per cell.
    ctx.changeImageLayout(cells, VK_IMAGE_LAYOUT_GENERAL);
    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeCoarseShader::getShader());
    ctx.bindResourceView(POM_CONE_BINDING_SOURCE, pyramidView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_CONE, pyramidView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, cellsStorageView, nullptr);
    args.mode = POM_CONE_COARSE_CELLS;
    ctx.pushConstants(0, sizeof(args), &args);
    ctx.dispatch(cellsExtent.width, cellsExtent.height, 1);
    ctx.changeImageLayout(cells, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Cones: one thread per cell.
    ctx.changeImageLayout(cones, VK_IMAGE_LAYOUT_GENERAL);
    ctx.bindResourceView(POM_CONE_BINDING_CONE, cellsSampledView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, conesStorageView, nullptr);
    args.mode = POM_CONE_COARSE_CONES;
    ctx.pushConstants(0, sizeof(args), &args);
    const VkExtent3D workgroups = util::computeBlockCount(
      cellsExtent, VkExtent3D { POM_CONE_COARSE_GROUP_SIZE, POM_CONE_COARSE_GROUP_SIZE, 1 });
    ctx.dispatch(workgroups.width, workgroups.height, workgroups.depth);
    ctx.changeImageLayout(cones, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Into G of mip 0, every texel. Without a source, R comes from the pyramid's level 0 (the same data).
    const VkRect2D all = { { 0, 0 }, { entry.extent.width, entry.extent.height } };
    const Rc<DxvkImageView> height = entry.sourceView != nullptr ? entry.sourceView : pyramidSampledViews[0];
    recordMerge(ctx, entry, height, conesSampledView, POM_CONE_COPY_HEIGHT_COARSE, all, shift);
    return true;
  }

  uint32_t PomConeMapManager::recordGenerate(RtxContext& ctx, Entry& entry, uint32_t texelBudget,
                                             std::vector<VkRect2D>& runs) const {
    const uint32_t tilesX = (entry.extent.width + POM_CONE_TILE_SIZE - 1) / POM_CONE_TILE_SIZE;
    const uint32_t tiles = tileCount(entry.extent);
    // The previous frame's merge left the cones readable.
    ctx.changeImageLayout(entry.cone, VK_IMAGE_LAYOUT_GENERAL);

    PomConeGenerateArgs args = {};
    args.extent = { entry.extent.width, entry.extent.height };
    args.nearRadius = entry.nearRadius;
    args.rayStepsPerCandidate = entry.rayStepsPerCandidate;
    args.farLevels = entry.farLevels;
    args.coneQuantization = entry.format == VK_FORMAT_R8G8_UNORM ? 255.0f : 65535.0f;

    ctx.setPushConstantBank(DxvkPushConstantBank::RTX);
    ctx.bindShader(VK_SHADER_STAGE_COMPUTE_BIT, PomConeGenerateShader::getShader());
    ctx.bindResourceView(POM_CONE_BINDING_SOURCE, entry.pyramidView, nullptr);
    ctx.bindResourceView(POM_CONE_BINDING_OUTPUT, entry.coneStorageView, nullptr);

    // Consecutive tiles of one tile row go out as one dispatch: each dispatch writes the cone image, so DXVK puts a
    // barrier between two of them, and a single 64x64 tile is far too small to fill the GPU on its own.
    constexpr uint32_t kTileTexels = POM_CONE_TILE_SIZE * POM_CONE_TILE_SIZE;
    constexpr uint32_t kTileGroups = POM_CONE_TILE_SIZE / POM_CONE_GENERATE_GROUP_SIZE;
    uint32_t used = 0;
    while (entry.nextTile < tiles && used < texelBudget) {
      const uint32_t remaining = texelBudget - used;
      const uint32_t budgetTiles = remaining / kTileTexels + (remaining % kTileTexels != 0 ? 1u : 0u);
      const uint32_t tileX = entry.nextTile % tilesX;
      const uint32_t run = std::min({ budgetTiles, tilesX - tileX, tiles - entry.nextTile });
      args.tileOrigin = { tileX * POM_CONE_TILE_SIZE, (entry.nextTile / tilesX) * POM_CONE_TILE_SIZE };
      ctx.pushConstants(0, sizeof(args), &args);
      ctx.dispatch(kTileGroups * run, kTileGroups, 1);
      VkRect2D region;
      region.offset = { int32_t(args.tileOrigin.x), int32_t(args.tileOrigin.y) };
      region.extent = { std::min(run * POM_CONE_TILE_SIZE, entry.extent.width - args.tileOrigin.x),
                        std::min(uint32_t(POM_CONE_TILE_SIZE), entry.extent.height - args.tileOrigin.y) };
      runs.push_back(region);
      used += run * kTileTexels;
      entry.nextTile += run;
    }
    return used;
  }

  const PomConeMapManager::Entry* PomConeMapManager::findTwin(const Entry& entry, bool generating) const {
    // Two keys can reach one height map: two views or two textures of the same data. Their derived textures are
    // identical, and so are their cache files, so only one of them is generated.
    for (const auto& item : m_entries) {
      const Entry* other = item.second.get();
      if (other == nullptr || other == &entry || other->sourceHash != entry.sourceHash ||
          other->sourceFormat != entry.sourceFormat || other->conservative != entry.conservative ||
          other->levelled != entry.levelled ||
          other->format != entry.format || other->mipLevels != entry.mipLevels ||
          other->extent.width != entry.extent.width || other->extent.height != entry.extent.height) {
        continue;
      }
      if (generating ? other->state == Entry::State::Generate : (other->state == Entry::State::Done && other->hasCones)) {
        return other;
      }
    }
    return nullptr;
  }

  void PomConeMapManager::recordTwinCopy(RtxContext& ctx, Entry& entry, const Entry& twin) const {
    // Every mip: R is the same source copy, G the same cones. copyImage handles the layouts.
    for (uint32_t mip = 0; mip < entry.mipLevels; ++mip) {
      VkImageSubresourceLayers layers = {};
      layers.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      layers.mipLevel       = mip;
      layers.baseArrayLayer = 0;
      layers.layerCount     = 1;
      ctx.copyImage(entry.image, layers, VkOffset3D { 0, 0, 0 }, twin.image, layers, VkOffset3D { 0, 0, 0 },
                    entry.image->mipLevelExtent(mip));
    }
    entry.relaxedInG = twin.relaxedInG;
  }

  void PomConeMapManager::finish(Entry& entry) {
    entry.pyramid = nullptr;
    entry.pyramidView = nullptr;
    entry.pyramidSampledViews.clear();
    entry.pyramidStorageViews.clear();
    entry.cone = nullptr;
    entry.coneSampledView = nullptr;
    entry.coneStorageView = nullptr;
    entry.sourceView = nullptr;
    entry.heightFromDerived = false;
    entry.pending = false;
    entry.state = Entry::State::Done;
  }

  void PomConeMapManager::onQualityChange() {
    if (m_coarseOnly) {
      // Every finished map goes back to the queue, where the coarse loop overwrites its G and it waits for full
      // quality: its cones then come back from the disk cache, or from generation.
      for (auto& item : m_entries) {
        Entry& entry = *item.second;
        if (entry.pending || entry.state != Entry::State::Done || !entry.hasCones) {
          continue;
        }
        entry.pending = true;
        entry.heightFromDerived = true;
        entry.coarse = false;
        entry.state = enableDiskCache() ? Entry::State::CacheLookup : Entry::State::Pyramid;
        m_pending.push_back(item.first);
      }
      return;
    }
    // Back to full quality: maps whose G kept its finished cones (the coarse pass had not reached them) are done again.
    for (const size_t derivedKey : m_pending) {
      const auto it = m_entries.find(derivedKey);
      if (it != m_entries.end() && it->second->pending && it->second->hasCones) {
        finish(*it->second);
      }
    }
  }

  std::string PomConeMapManager::cacheDirectory() const {
    if (!cachePath().empty()) {
      return cachePath();
    }
    const std::filesystem::path logs = util::RtxFileSys::isInitialized() ?
      util::RtxFileSys::path(util::RtxFileSys::Logs) : std::filesystem::path("rtx-remix/logs");
    return (logs.parent_path() / "cache" / "pom_cone").string();
  }

  void PomConeMapManager::startCacheLoad(Entry& entry) const {
    const GenerationParams params = generationParams(entry.extent, entry.conservative, nearRadius(), rayStepsPerCandidate());
    entry.cacheRaySteps = params.raySteps;
    const std::filesystem::path file = std::filesystem::path(cacheDirectory()) /
      coneCacheFile(entry.sourceHash, entry.extent, scratchFormat(entry.format), params.nearRadius, params.raySteps,
                    entry.conservative);
    entry.cacheLoad = std::async(std::launch::async, [file]() {
      std::error_code error;
      if (!std::filesystem::exists(file, error)) {
        return gli::texture();
      }
      return gli::load(file.string());
    });
  }

  bool PomConeMapManager::applyCachedCones(RtxContext& ctx, Entry& entry) {
    const gli::texture cached = entry.cacheLoad.get();
    const VkFormat format = scratchFormat(entry.format);
    const size_t texelBytes = format == VK_FORMAT_R8_UNORM ? 1 : 2;
    const size_t bytes = size_t(entry.extent.width) * entry.extent.height * texelBytes;
    if (cached.empty() || cached.format() != gli::format(format) ||
        uint32_t(cached.extent(0).x) != entry.extent.width || uint32_t(cached.extent(0).y) != entry.extent.height ||
        cached.size(0) != bytes) {
      return false;
    }

    entry.cone = createScratchImage(m_device, format, entry.extent, 1, "SCCR cached cones");
    if (entry.cone == nullptr) {
      return false;
    }
    entry.coneSampledView = createView(m_device, entry.cone, VK_IMAGE_USAGE_SAMPLED_BIT, 0, 1);

    DxvkBufferCreateInfo stagingInfo = {};
    stagingInfo.size   = bytes;
    stagingInfo.usage  = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.stages = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT;
    stagingInfo.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
    const Rc<DxvkBuffer> staging = m_device->createBuffer(
      stagingInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      DxvkMemoryStats::Category::RTXBuffer, "SCCR cached cones staging");
    if (staging == nullptr) {
      return false;
    }
    std::memcpy(DxvkBufferSlice(staging).mapPtr(0), cached.data(0, 0, 0), bytes);

    VkImageSubresourceLayers subresource = {};
    subresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource.mipLevel       = 0;
    subresource.baseArrayLayer = 0;
    subresource.layerCount     = 1;
    ctx.changeImageLayout(entry.cone, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    ctx.copyBufferToImage(entry.cone, subresource, VkOffset3D { 0, 0, 0 },
                          VkExtent3D { entry.extent.width, entry.extent.height, 1 }, staging, 0, 0, 0);
    recordCopy(ctx, entry, POM_CONE_COPY_HEIGHT_CONE);
    return true;
  }

  void PomConeMapManager::storeInCache(RtxContext& ctx, const Entry& entry) const {
    // The cache is keyed by the source: cones generated from the levelled R bound the levelled field only.
    if (entry.levelled && !entry.pyramidRaw) {
      return;
    }
    const std::string directory = cacheDirectory();
    if (!util::createDirectories(directory)) {
      return;
    }
    const std::string file = coneCacheFile(entry.sourceHash, entry.extent, scratchFormat(entry.format),
                                           entry.nearRadius, entry.rayStepsPerCandidate, entry.conservative);
    m_device->getCommon()->metaExporter().dumpImageToFile(&ctx, str::format(directory, "/"), file, entry.cone);
  }

  void PomConeMapManager::trimCache() const {
    const std::filesystem::path directory(cacheDirectory());
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) {
      return;
    }
    struct CacheFile {
      std::filesystem::path path;
      std::filesystem::file_time_type time;
      uintmax_t size;
    };
    std::vector<CacheFile> files;
    uintmax_t total = 0;
    for (const auto& item : std::filesystem::directory_iterator(directory, error)) {
      if (item.is_regular_file(error) && item.path().extension() == ".dds") {
        const uintmax_t size = item.file_size(error);
        files.push_back({ item.path(), item.last_write_time(error), size });
        total += size;
      }
    }
    const uintmax_t limit = uintmax_t(cacheMaxMB()) << 20;
    if (total <= limit) {
      return;
    }
    std::sort(files.begin(), files.end(), [](const CacheFile& a, const CacheFile& b) { return a.time < b.time; });
    uint32_t removed = 0;
    for (const CacheFile& file : files) {
      if (total <= limit) {
        break;
      }
      if (std::filesystem::remove(file.path, error)) {
        total -= file.size;
        ++removed;
      }
    }
    Logger::info(str::format("[SCCR] Cone cache trimmed: ", removed, " oldest files removed."));
  }

  void PomConeMapManager::update(RtxContext& ctx) {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    // Bake now: the button's request, or the option going from false to true (a host). The option is only read: see
    // its declaration.
    const bool bakeOption = bakeNow();
    if (m_bakeRequested.exchange(false) || (bakeOption && !m_bakeOptionSeen)) {
      m_baking = true;
    }
    m_bakeOptionSeen = bakeOption;

    // Coarse-only quality: generation, cache reads and twin copies pause, so there is nothing to bake.
    const bool coarseOnly = quality() == ConeQuality::CoarseOnly;
    if (coarseOnly != m_coarseOnly) {
      m_coarseOnly = coarseOnly;
      onQualityChange();
    }
    if (coarseOnly) {
      m_baking = false;
    }

    if (m_pending.empty()) {
      // Nothing to bake: the request does not wait for maps that come into view later.
      m_baking = false;
      return;
    }

    ScopedGpuProfileZone(&ctx, "PomConeMaps");

    const bool useCache = enableDiskCache();
    if (useCache && !m_cacheTrimmed) {
      trimCache();
      m_cacheTrimmed = true;
    }

    // Copies, all of them: they are cheap, and a copied map is valid (it marches linearly) right away. Then the disk
    // cache: finished reads are merged, misses go on to generation, and a few more reads start (not in coarse-only
    // quality: a read finished meanwhile waits for full quality).
    uint32_t loadsInFlight = 0;
    for (const size_t derivedKey : m_pending) {
      const auto it = m_entries.find(derivedKey);
      if (it == m_entries.end() || !hasHeight(*it->second)) {
        continue;
      }
      Entry& entry = *it->second;
      if (entry.state == Entry::State::Copy) {
        entry.levelled = levelHeightMaps();
        if (entry.levelled) {
          recordLevel(ctx, entry);
        }
        recordCopy(ctx, entry, POM_CONE_COPY_HEIGHT);
        entry.state = useCache ? Entry::State::CacheLookup : Entry::State::Pyramid;
      } else if (entry.state == Entry::State::CacheLoad) {
        if (entry.cacheLoad.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
          ++loadsInFlight;
        } else if (coarseOnly) {
          continue;
        } else if (applyCachedCones(ctx, entry)) {
          finish(entry);
          entry.hasCones = true;
          entry.refinedInG = true;
          entry.relaxedInG = entry.cacheRaySteps > 0;
          ++m_loadedSinceReport;
        } else {
          entry.cone = nullptr;
          entry.coneSampledView = nullptr;
          entry.state = Entry::State::Pyramid;
        }
      }
    }
    for (const size_t derivedKey : m_pending) {
      if (coarseOnly || loadsInFlight >= kMaxCacheLoadsInFlight) {
        break;
      }
      const auto it = m_entries.find(derivedKey);
      if (it != m_entries.end() && it->second->state == Entry::State::CacheLookup && hasHeight(*it->second)) {
        startCacheLoad(*it->second);
        it->second->state = Entry::State::CacheLoad;
        ++loadsInFlight;
      }
    }

    // Coarse cones for the maps waiting for generation, so the cone march applies to them within a frame or two instead
    // of linear steps until their last tile. Each map once, as soon as it waits (copied, cache missed). A map with a
    // finished twin is copied instead, whatever the budgets (a copy is cheap). In coarse-only quality every queued map
    // (new, part way through generation, or re-queued finished) gets its G overwritten once with coarse cones, or with
    // zeros where it cannot have them, unless G holds nothing generated already.
    uint64_t coarseBudget = kCoarseTexelsPerFrame;
    for (const size_t derivedKey : m_pending) {
      const auto it = m_entries.find(derivedKey);
      if (it == m_entries.end() || !it->second->pending || !hasHeight(*it->second)) {
        continue;
      }
      Entry& entry = *it->second;
      const bool waiting = coarseOnly ? entry.state != Entry::State::Copy : entry.state == Entry::State::Pyramid;
      if (!waiting) {
        continue;
      }
      if (!coarseOnly) {
        if (const Entry* twin = findTwin(entry, false)) {
          recordTwinCopy(ctx, entry, *twin);
          finish(entry);
          entry.hasCones = true;
          entry.refinedInG = true;
          ++m_copiedSinceReport;
          continue;
        }
      }
      const bool needsCoarse = !entry.coarse || (coarseOnly && entry.refinedInG);
      if (!needsCoarse || coarseBudget == 0) {
        continue;
      }
      entry.coarse = true;
      bool recorded = false;
      if (recordCoarse(ctx, entry)) {
        entry.hasCoarse = true;
        recorded = true;
        ++m_coarseSinceReport;
      } else {
        entry.hasCoarse = false;
        if (entry.refinedInG) {
          // No coarse cones for this map (shape, or a failed allocation), so G goes back to 0, a linear march.
          const Rc<DxvkImageView> height = mergeHeight(ctx, entry);
          const VkRect2D all = { { 0, 0 }, { entry.extent.width, entry.extent.height } };
          recordMerge(ctx, entry, height, height, POM_CONE_COPY_HEIGHT, all, 0);
          recorded = true;
        }
      }
      entry.refinedInG = false;
      entry.hasCones = false;
      if (recorded) {
        entry.relaxedInG = false;
        const uint64_t texels = uint64_t(entry.extent.width) * entry.extent.height;
        coarseBudget = texels >= coarseBudget ? 0 : coarseBudget - texels;
      }
    }

    // Then cone generation in queue order, within the per-frame budget (a larger one while baking). Paused in
    // coarse-only quality.
    uint32_t budget = coarseOnly ? 0u :
      m_baking ? std::max(bakeTexelsPerFrame(), generateTexelsPerFrame()) : generateTexelsPerFrame();
    std::vector<VkRect2D> runs;
    for (const size_t derivedKey : m_pending) {
      if (budget == 0) {
        break;
      }
      const auto it = m_entries.find(derivedKey);
      if (it == m_entries.end() || !hasHeight(*it->second)) {
        continue;
      }
      Entry& entry = *it->second;
      if (entry.state == Entry::State::Pyramid) {
        // A twin with finished cones is copied; a twin being generated is waited for (only one entry generates at a
        // time, so it finishes and is then copied).
        if (const Entry* twin = findTwin(entry, false)) {
          recordTwinCopy(ctx, entry, *twin);
          finish(entry);
          entry.hasCones = true;
          entry.refinedInG = true;
          ++m_copiedSinceReport;
          continue;
        }
        if (findTwin(entry, true) != nullptr) {
          continue;
        }
        if (!beginGeneration(entry)) {
          finish(entry);
          continue;
        }
        recordPyramid(ctx, entry);
        entry.state = Entry::State::Generate;
      }
      if (entry.state != Entry::State::Generate) {
        continue;
      }

      // Back from coarse-only quality part way through: the tiles generated before it replace the coarse cones again.
      if (!entry.refinedInG && entry.nextTile > 0) {
        recordMergeGenerated(ctx, entry);
        entry.relaxedInG = entry.relaxedInG || entry.rayStepsPerCandidate > 0;
      }
      runs.clear();
      const uint32_t used = recordGenerate(ctx, entry, budget, runs);
      budget = used >= budget ? 0 : budget - used;
      if (entry.nextTile < tileCount(entry.extent)) {
        // The tiles generated this frame replace their coarse cones (or zeros) in G right away; only those texels, as
        // the rest of the scratch holds nothing yet.
        const Rc<DxvkImageView> height = mergeHeight(ctx, entry);
        ctx.changeImageLayout(entry.cone, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        for (const VkRect2D& region : runs) {
          recordMerge(ctx, entry, height, entry.coneSampledView, POM_CONE_COPY_HEIGHT_CONE, region, 0);
        }
        entry.refinedInG = true;
        entry.relaxedInG = entry.relaxedInG || entry.rayStepsPerCandidate > 0;
        break;
      }

      // Every tile is generated: all of G at once (the cache gets the generated cones only, never coarse ones).
      recordCopy(ctx, entry, POM_CONE_COPY_HEIGHT_CONE);
      if (useCache) {
        storeInCache(ctx, entry);
      }
      finish(entry);
      entry.hasCones = true;
      entry.refinedInG = true;
      entry.relaxedInG = entry.rayStepsPerCandidate > 0;
      ++m_generatedSinceReport;
    }

    // Drop what is done, and what lost its source.
    m_pending.erase(std::remove_if(m_pending.begin(), m_pending.end(), [this](size_t derivedKey) {
      const auto it = m_entries.find(derivedKey);
      if (it == m_entries.end()) {
        return true;
      }
      if (it->second->pending && !hasHeight(*it->second)) {
        finish(*it->second);
      }
      return !it->second->pending;
    }), m_pending.end());

    // Whether every cone in G is conservative now (conesConservative): G changes only here, and in resolveHeight in a
    // way this frame's copies complete.
    bool conservative = true;
    for (const auto& item : m_entries) {
      if (item.second->relaxedInG) {
        conservative = false;
        break;
      }
    }
    m_conesConservative = conservative;

    if (m_pending.empty() && m_generatedSinceReport + m_loadedSinceReport + m_copiedSinceReport > 0) {
      Logger::info(str::format("[SCCR] Cone maps up to date: ", m_generatedSinceReport, " generated (",
                               m_coarseSinceReport, " had coarse cones meanwhile), ",
                               m_loadedSinceReport, " read from the disk cache and ", m_copiedSinceReport,
                               " copied from an identical map since the last report, ",
                               m_entries.size(), " in all, ", (m_totalBytes + (1u << 19)) >> 20,
                               " MB of derived height+cone textures."));
      m_generatedSinceReport = 0;
      m_loadedSinceReport = 0;
      m_copiedSinceReport = 0;
      m_coarseSinceReport = 0;
    }
    if (m_pending.empty()) {
      m_baking = false;
    }
  }

  PomConeMapManager::Progress PomConeMapManager::progress() {
    std::lock_guard<dxvk::mutex> lock(m_mutex);
    Progress result;
    result.pendingMaps = uint32_t(m_pending.size());
    for (const size_t derivedKey : m_pending) {
      const auto it = m_entries.find(derivedKey);
      if (it == m_entries.end() || !it->second->pending) {
        continue;
      }
      const Entry& entry = *it->second;
      const uint32_t tiles = tileCount(entry.extent);
      if (entry.state == Entry::State::Generate) {
        result.tilesRemaining += tiles - std::min(entry.nextTile, tiles);
      } else if (!entry.hasCones) {
        // Not started: a cache hit or an identical map may still spare it.
        result.tilesRemaining += tiles;
      }
    }
    for (const auto& item : m_entries) {
      const Entry& entry = *item.second;
      ++result.liveMaps;
      if (entry.hasCones) {
        ++result.fullMaps;
      } else if (entry.hasCoarse) {
        ++result.coarseMaps;
      }
    }
    result.derivedBytes = m_totalBytes;
    return result;
  }
}
