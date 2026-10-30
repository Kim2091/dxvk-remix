#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

// Only the formats the texture directive documents are enabled, so effects
// cannot come to depend on other formats stb_image happens to decode.
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_TGA
#define STBI_ONLY_BMP
// Files are decoded from memory, because stb's fopen path takes a narrow
// string and fails under a non-ASCII game directory.
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244)  // conversion from 'int' to 'stbi_uc', possible loss of data
#pragma warning(disable: 4996)  // 'This function or variable may be unsafe'
#endif
#include <stb/stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <gli/gli.hpp>

#include "dxvk_format.h"
#include "dxvk_util.h"
#include "rtx_external_effect_image.h"
#include "rtx_external_effect_manifest.h"
#include "rtx_texture.h"

namespace dxvk {

  namespace {
    // Files are read into memory whole, so the cap applies to the file size.
    constexpr uintmax_t kMaxImageFileBytes = 256u * 1024u * 1024u;

    std::string toLower(std::string value) {
      std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      return value;
    }

    bool readFileBytes(
      const std::filesystem::path& path, std::vector<char>& bytes, std::string& error) {
      std::error_code fileError;
      const uintmax_t size = std::filesystem::file_size(path, fileError);
      if (fileError) {
        error = "the file could not be found or opened";
        return false;
      }
      if (size == 0) {
        error = "the file is empty";
        return false;
      }
      if (size > kMaxImageFileBytes) {
        error = "the file is larger than the " + std::to_string(kMaxImageFileBytes / (1024 * 1024))
          + " MiB image limit";
        return false;
      }

      // Passing the std::filesystem::path makes MSVC open the file through the
      // wide CRT entry point, so paths outside the active code page resolve.
      std::ifstream file(path, std::ios::binary);
      bytes.resize(static_cast<size_t>(size));
      if (!file || !file.read(bytes.data(), static_cast<std::streamsize>(size))) {
        error = "the file could not be read";
        return false;
      }
      return true;
    }

    bool decodeWithStb(
      const std::vector<char>& bytes, RtxExternalEffectImage& image, std::string& error) {
      int width = 0;
      int height = 0;
      int channels = 0;
      // Every image is expanded to four channels, because RemixFX read bindings
      // return a float4 and the views carry no component swizzle.
      stbi_uc* pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(bytes.data()), static_cast<int>(bytes.size()),
        &width, &height, &channels, 4);
      if (pixels == nullptr) {
        const char* reason = stbi_failure_reason();
        error = reason != nullptr ? reason : "the image could not be decoded";
        return false;
      }

      image.extent = VkExtent3D {
        static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1u };
      image.format = VK_FORMAT_R8G8B8A8_UNORM;
      image.data.assign(pixels, pixels + static_cast<size_t>(width) * height * 4);
      stbi_image_free(pixels);
      return true;
    }

    // Rejects a DDS that holds fewer bytes than its header declares.
    // gli's load_dds checks this only with GLI_ASSERT, which is empty in release builds,
    // and then copies the whole declared mip chain out of the buffer.
    // The size is computed the same way as the asset pipeline's DdsFileParser.
    bool checkDdsIsWhole(
      const std::vector<char>& bytes, std::string& error) {
      using namespace gli::detail;

      if (bytes.size() < sizeof(FOURCC_DDS) + sizeof(dds_header)
       || std::memcmp(bytes.data(), FOURCC_DDS, sizeof(FOURCC_DDS)) != 0) {
        error = "the file is too small to be a DDS, or does not start with one";
        return false;
      }

      dds_header header;
      std::memcpy(&header, bytes.data() + sizeof(FOURCC_DDS), sizeof(header));

      dds_header10 header10;
      size_t offset = sizeof(FOURCC_DDS) + sizeof(header);
      if ((header.Format.flags & gli::dx::DDPF_FOURCC)
       && (header.Format.fourCC == gli::dx::D3DFMT_DX10
        || header.Format.fourCC == gli::dx::D3DFMT_GLI1)) {
        if (bytes.size() < offset + sizeof(header10)) {
          error = "the DX10 header is truncated";
          return false;
        }
        std::memcpy(&header10, bytes.data() + offset, sizeof(header10));
        offset += sizeof(header10);
      }

      // Uses gli's own target selection. Only a 2D target has a single face,
      // layer and slice, so the one mip chain summed below is exactly what gli copies.
      if (get_target(header, header10) != gli::target::TARGET_2D) {
        error = "only 2D images are supported; 1D, cube map, array and volume textures are not";
        return false;
      }

      const gli::format format = get_dds_format(header, header10);
      if (format == static_cast<gli::format>(gli::FORMAT_INVALID)) {
        error = "the pixel format is not one gli recognises";
        return false;
      }

      const uint32_t levels = (header.Flags & DDSD_MIPMAPCOUNT)
        ? std::max(header.MipMapLevels, 1u) : 1u;
      const size_t blockSize = gli::block_size(format);
      const glm::ivec3 blockExtent = gli::block_extent(format);

      size_t required = offset;
      for (uint32_t level = 0; level < levels; level++) {
        const uint32_t width = std::max(header.Width >> level, 1u);
        const uint32_t height = std::max(header.Height >> level, 1u);
        required += size_t(util::ceilDivide(width, uint32_t(blockExtent.x)))
                  * util::ceilDivide(height, uint32_t(blockExtent.y)) * blockSize;
      }

      if (bytes.size() < required) {
        error = "the header declares " + std::to_string(required)
          + " bytes but the file holds " + std::to_string(bytes.size());
        return false;
      }
      return true;
    }

    bool decodeWithGli(
      const std::vector<char>& bytes, RtxExternalEffectImage& image, std::string& error) {
      if (!checkDdsIsWhole(bytes, error)) {
        return false;
      }

      // Loads from memory, because gli's path overload takes a narrow string.
      const gli::texture texture = gli::load(bytes.data(), bytes.size());
      if (texture.empty()) {
        error = "the container could not be parsed as a DDS";
        return false;
      }
      if (texture.target() != gli::target::TARGET_2D) {
        error = "only 2D images are supported; cube maps, arrays and volumes are not";
        return false;
      }

      // gli's format enumeration matches VkFormat numerically, as rtx_asset_data_manager.cpp also assumes.
      image.format = static_cast<VkFormat>(texture.format());
      const gli::extent3d extent = texture.extent(0);
      image.extent = VkExtent3D {
        static_cast<uint32_t>(extent.x), static_cast<uint32_t>(extent.y), 1u };

      const size_t levelSize = texture.size(0);
      const uint8_t* levelData = static_cast<const uint8_t*>(texture.data(0, 0, 0));
      if (levelData == nullptr || levelSize == 0) {
        error = "the container declares no pixel data for its top mip level";
        return false;
      }
      image.data.assign(levelData, levelData + levelSize);
      return true;
    }
  }

  bool loadRtxExternalEffectImage(
    const std::filesystem::path& path,
    bool srgb,
    RtxExternalEffectImage& image,
    std::string& error) {
    image = {};
    error.clear();

    std::vector<char> bytes;
    if (!readFileBytes(path, bytes, error)) {
      return false;
    }

    // DDS is the only container accepted. gli also reads KTX and KMG, but those
    // cannot be size-checked like a DDS, and the asset pipeline rejects them too.
    const std::string extension = toLower(path.extension().u8string());
    const bool isContainer = extension == ".dds";
    if (!isContainer && extension != ".png" && extension != ".jpg"
     && extension != ".jpeg" && extension != ".tga" && extension != ".bmp") {
      error = "'" + extension + "' is not a supported image extension"
        " (.png, .jpg, .tga, .bmp, .dds)";
      return false;
    }

    if (!(isContainer ? decodeWithGli(bytes, image, error) : decodeWithStb(bytes, image, error))) {
      return false;
    }

    if (image.extent.width == 0 || image.extent.height == 0
     || image.extent.width > kMaxRtxExternalEffectTextureExtent
     || image.extent.height > kMaxRtxExternalEffectTextureExtent) {
      error = "the image is " + std::to_string(image.extent.width) + "x"
        + std::to_string(image.extent.height) + ", which is outside the 1.."
        + std::to_string(kMaxRtxExternalEffectTextureExtent) + " range";
      return false;
    }

    // A container that already declares sRGB keeps it whatever the flag says.
    if (srgb) {
      image.format = TextureUtils::toSRGB(image.format);
      if (!TextureUtils::isSRGB(image.format)) {
        error = "'srgb = true' was declared but the image format has no sRGB variant";
        return false;
      }
    }

    const DxvkFormatInfo* formatInfo = imageFormatInfo(image.format);
    if (formatInfo == nullptr || formatInfo->elementSize == 0) {
      error = "the image format is not one this runtime can upload";
      return false;
    }

    const VkExtent3D blocks = util::computeBlockCount(image.extent, formatInfo->blockSize);
    image.rowPitch = VkDeviceSize(blocks.width) * formatInfo->elementSize;
    image.layerPitch = image.rowPitch * blocks.height;

    // Guards updateImage against reading past the decoded data when the top level is short.
    if (image.data.size() < image.layerPitch) {
      error = "the image holds " + std::to_string(image.data.size()) + " bytes but its "
        + std::to_string(image.extent.width) + "x" + std::to_string(image.extent.height)
        + " extent needs " + std::to_string(image.layerPitch);
      return false;
    }

    return true;
  }

} // namespace dxvk
