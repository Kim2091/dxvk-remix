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
#include "rtx_screenshot.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <vector>

#include <objbase.h>
#include <shlobj.h>
#include <wincodec.h>

#include "rtx_screenshot_pq_icc.h"
#include "rtx_session_log.h"
#include "rtx_tone_mapping.h"
#include "rtx_utils.h"
#include "../dxvk_buffer.h"
#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "../../util/com/com_pointer.h"
#include "../../util/log/log.h"
#include "../../util/sync/sync_signal.h"
#include "../../util/util_env.h"
#include "../../util/util_filesys.h"
#include "../../util/util_string.h"

#pragma comment(lib, "windowscodecs")
#pragma comment(lib, "shell32")

namespace dxvk {
  namespace {
    constexpr double kGpuWaitSeconds = 10.0;
    constexpr double kSavedMessageSeconds = 3.0;
    constexpr float kDisplayGamma = 2.2f;

    // The back buffer formats a screenshot can read.
    enum class PixelLayout {
      Unsupported,
      Rgba8,
      Bgra8,
      Rgb10A2,
      Bgr10A2,
      Rgba16Float,
    };

    PixelLayout pixelLayout(VkFormat format) {
      switch (format) {
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R8G8B8A8_SRGB:
        return PixelLayout::Rgba8;
      case VK_FORMAT_B8G8R8A8_UNORM:
      case VK_FORMAT_B8G8R8A8_SRGB:
        return PixelLayout::Bgra8;
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        return PixelLayout::Rgb10A2;
      case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return PixelLayout::Bgr10A2;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
        return PixelLayout::Rgba16Float;
      default:
        return PixelLayout::Unsupported;
      }
    }

    uint32_t bytesPerPixel(PixelLayout layout) {
      return layout == PixelLayout::Rgba16Float ? 8u : 4u;
    }

    using Mat3 = std::array<std::array<float, 3>, 3>;

    Vector3 multiply(const Mat3& m, const Vector3& v) {
      return Vector3(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
    }

    // The rows match the present blit's BT.709 to BT.2020 conversion (dxvk_present_frag_blit.frag).
    constexpr Mat3 kBt709ToBt2020 { {
      { 0.6274038960f, 0.3292830384f, 0.0433130657f },
      { 0.0690972894f, 0.9195403951f, 0.0113623156f },
      { 0.0163914388f, 0.0880133078f, 0.8955952532f } } };

    constexpr double kCurveAnchor = 0.18;

    // The normalized shoulder of reinhardPiecewiseSmooth in tonemap_operator.slangh, for t > 0.
    double shoulder(double t) {
      const double s = std::exp(-0.01 / (t * t));
      return t / (1.0 + t * s);
    }

    // reinhardPiecewiseSmooth in tonemap_operator.slangh, with an anchor of 0.18.
    double toneCurve(double x, double peak) {
      if (x <= kCurveAnchor || peak <= kCurveAnchor) {
        return x;
      }
      const double range = peak - kCurveAnchor;
      return kCurveAnchor + range * shoulder((x - kCurveAnchor) / range);
    }

    // The tone mapping state at capture time, for converting an HDR capture to SDR.
    struct SdrConversion {
      float sceneScale = 1.0f;
      bool toneMapped = true;
    };

    // Converts a linear HDR back buffer color (BT.709, 1 = UI white) to SDR, as an HDR10 image is mapped for an SDR
    // display: the color is divided by the scene scale (paper white / UI nits), so 1 is paper white, and each channel
    // goes through reinhardPiecewiseSmooth with a peak of 1 (anchor 0.18). Without tone mapping it is clipped instead.
    Vector3 hdrToSdr(const Vector3& linearUi, const SdrConversion& conversion) {
      Vector3 scene = linearUi * (1.0f / conversion.sceneScale);
      if (!conversion.toneMapped) {
        return scene;
      }
      for (uint32_t c = 0; c < 3; ++c) {
        scene[c] = float(toneCurve(scene[c], 1.0));
      }
      return scene;
    }

    // Encodes linear light with 1 at white as the SDR output does (srgb_dither.comp.slang uses a 2.2 power).
    uint8_t encodeSdr(float linear) {
      const float encoded = std::pow(std::clamp(linear, 0.0f, 1.0f), 1.0f / kDisplayGamma);
      return uint8_t(encoded * 255.0f + 0.5f);
    }

    uint8_t quantize8(float encoded) {
      return uint8_t(std::clamp(encoded, 0.0f, 1.0f) * 255.0f + 0.5f);
    }

    // ST.2084 with the present blit's scaling: linear 1 = whiteNits.
    uint16_t encodePq(float linear, float whiteNits) {
      constexpr float m1 = 2610.0f / 16384.0f;
      constexpr float m2 = 128.0f * (2523.0f / 4096.0f);
      constexpr float c1 = 3424.0f / 4096.0f;
      constexpr float c2 = 32.0f * (2413.0f / 4096.0f);
      constexpr float c3 = 32.0f * (2392.0f / 4096.0f);
      const float y = std::pow(std::clamp(linear * (whiteNits / 10000.0f), 0.0f, 1.0f), m1);
      const float pq = std::pow((c1 + c2 * y) / (1.0f + c3 * y), m2);
      return uint16_t(std::clamp(pq, 0.0f, 1.0f) * 65535.0f + 0.5f);
    }

    uint32_t crc32(const uint8_t* pData, size_t size) {
      static const std::array<uint32_t, 256> kTable = [] {
        std::array<uint32_t, 256> table {};
        for (uint32_t n = 0; n < 256; ++n) {
          uint32_t c = n;
          for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
          }
          table[n] = c;
        }
        return table;
      }();
      uint32_t c = 0xFFFFFFFFu;
      for (size_t i = 0; i < size; ++i) {
        c = kTable[(c ^ pData[i]) & 0xFFu] ^ (c >> 8);
      }
      return c ^ 0xFFFFFFFFu;
    }

    uint32_t readBigEndian32(const uint8_t* p) {
      return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
    }

    void appendBigEndian32(std::vector<uint8_t>& out, uint32_t value) {
      out.push_back(uint8_t(value >> 24));
      out.push_back(uint8_t(value >> 16));
      out.push_back(uint8_t(value >> 8));
      out.push_back(uint8_t(value));
    }

    void appendChunk(std::vector<uint8_t>& out, const char (&type)[5], const std::vector<uint8_t>& data) {
      appendBigEndian32(out, uint32_t(data.size()));
      const size_t typeAt = out.size();
      out.insert(out.end(), type, type + 4);
      out.insert(out.end(), data.begin(), data.end());
      appendBigEndian32(out, crc32(&out[typeAt], data.size() + 4));
    }

    // A zlib stream of stored (uncompressed) deflate blocks.
    std::vector<uint8_t> zlibStored(const uint8_t* pData, size_t size) {
      std::vector<uint8_t> out { 0x78, 0x01 };
      size_t offset = 0;
      do {
        const size_t block = std::min<size_t>(size - offset, 0xFFFF);
        out.push_back(offset + block == size ? 1 : 0);
        out.push_back(uint8_t(block));
        out.push_back(uint8_t(block >> 8));
        out.push_back(uint8_t(~block));
        out.push_back(uint8_t(~block >> 8));
        out.insert(out.end(), pData + offset, pData + offset + block);
        offset += block;
      } while (offset < size);
      uint32_t a = 1;
      uint32_t b = 0;
      for (size_t i = 0; i < size; ++i) {
        a = (a + pData[i]) % 65521u;
        b = (b + a) % 65521u;
      }
      appendBigEndian32(out, (b << 16) | a);
      return out;
    }

    // The content light levels of an HDR screenshot (CTA-861.3), in nits of the encoded BT.2020 signal.
    struct HdrLight {
      double maxCll = 0.0;
      double maxFall = 0.0;
    };

    // Returns the PNG with the chunks of an HDR PNG every viewer shows right, Discord included, as the NVIDIA app
    // and SKIV write them. After IHDR: iCCP (the PQ profile of rtx_screenshot_pq_icc.h), cLLi, sBIT (16 bits) and
    // cHRM (BT.2020 primaries, D65 white). The encoder's gAMA, cHRM, sRGB and iCCP chunks are left out. There is no
    // cICP chunk: some viewers, Discord among them, show a PNG with cICP blown out, and the profile carries the same
    // signal in its own cicp tag. Returns an empty vector when the PNG cannot be parsed.
    std::vector<uint8_t> addHdrChunks(const std::vector<uint8_t>& png, const HdrLight& light) {
      constexpr size_t kSignatureSize = 8;
      std::vector<uint8_t> out(png.begin(), png.begin() + std::min(png.size(), kSignatureSize));
      size_t offset = kSignatureSize;
      bool wroteProfile = false;
      while (offset + 12 <= png.size()) {
        const uint32_t length = readBigEndian32(&png[offset]);
        const size_t chunkSize = size_t(length) + 12;
        if (offset + chunkSize > png.size()) {
          return {};
        }
        const char* type = reinterpret_cast<const char*>(&png[offset + 4]);
        const bool colourChunk = std::memcmp(type, "gAMA", 4) == 0 || std::memcmp(type, "cHRM", 4) == 0 ||
                                 std::memcmp(type, "sRGB", 4) == 0 || std::memcmp(type, "iCCP", 4) == 0 ||
                                 std::memcmp(type, "cICP", 4) == 0;
        if (!colourChunk) {
          out.insert(out.end(), png.begin() + offset, png.begin() + offset + chunkSize);
        }
        if (std::memcmp(type, "IHDR", 4) == 0) {
          // The profile's name (its desc), a terminator, compression method 0 and the zlib stream.
          static const char kName[] = "RGB_D65_202_Rel_PeQ";
          std::vector<uint8_t> iccp(kName, kName + sizeof(kName));
          iccp.push_back(0);
          const std::vector<uint8_t> profile = zlibStored(kRec2100PqIcc, sizeof(kRec2100PqIcc));
          iccp.insert(iccp.end(), profile.begin(), profile.end());
          appendChunk(out, "iCCP", iccp);

          std::vector<uint8_t> clli;
          appendBigEndian32(clli, uint32_t(std::lround(light.maxCll * 10000.0)));
          appendBigEndian32(clli, uint32_t(std::lround(light.maxFall * 10000.0)));
          appendChunk(out, "cLLi", clli);

          appendChunk(out, "sBIT", { 16, 16, 16 });

          // White x, y, then red, green and blue x, y, times 100000.
          std::vector<uint8_t> chrm;
          for (const uint32_t v : { 31270u, 32900u, 70800u, 29200u, 17000u, 79700u, 13100u, 4600u }) {
            appendBigEndian32(chrm, v);
          }
          appendChunk(out, "cHRM", chrm);
          wroteProfile = true;
        }
        offset += chunkSize;
      }
      return wroteProfile ? out : std::vector<uint8_t>();
    }

    // Encodes the pixels as a PNG with the Windows Imaging Component, which compresses it with zlib.
    // 8-bit pixels are BGR, and 16-bit pixels are RGB in native byte order.
    // pHdrLight marks the image as BT.2020 PQ (addHdrChunks).
    bool writePng(const std::string& path, uint32_t width, uint32_t height, const void* pPixels, bool sixteenBit,
                  const HdrLight* pHdrLight) {
      // RPC_E_CHANGED_MODE means the thread already runs COM in another mode, which WIC works with as well.
      const HRESULT coInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

      std::vector<uint8_t> png;
      {
        Com<IWICImagingFactory> factory;
        Com<IStream> stream;
        Com<IWICBitmapEncoder> encoder;
        Com<IWICBitmapFrameEncode> frame;

        const UINT stride = width * (sixteenBit ? 6u : 3u);
        const UINT bufferSize = stride * height;
        const WICPixelFormatGUID expectedFormat = sixteenBit ? GUID_WICPixelFormat48bppRGB : GUID_WICPixelFormat24bppBGR;
        WICPixelFormatGUID pixelFormat = expectedFormat;
        STATSTG stat {};
        HGLOBAL memory = nullptr;
        const bool encoded =
          SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
          SUCCEEDED(CreateStreamOnHGlobal(nullptr, TRUE, &stream)) &&
          SUCCEEDED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
          SUCCEEDED(encoder->Initialize(stream.ptr(), WICBitmapEncoderNoCache)) &&
          SUCCEEDED(encoder->CreateNewFrame(&frame, nullptr)) &&
          SUCCEEDED(frame->Initialize(nullptr)) &&
          SUCCEEDED(frame->SetSize(width, height)) &&
          SUCCEEDED(frame->SetPixelFormat(&pixelFormat)) &&
          IsEqualGUID(pixelFormat, expectedFormat) &&
          SUCCEEDED(frame->WritePixels(height, stride, bufferSize, static_cast<BYTE*>(const_cast<void*>(pPixels)))) &&
          SUCCEEDED(frame->Commit()) &&
          SUCCEEDED(encoder->Commit()) &&
          SUCCEEDED(stream->Stat(&stat, STATFLAG_NONAME)) &&
          SUCCEEDED(GetHGlobalFromStream(stream.ptr(), &memory));
        if (encoded) {
          const void* pData = GlobalLock(memory);
          if (pData != nullptr) {
            const uint8_t* pBytes = static_cast<const uint8_t*>(pData);
            png.assign(pBytes, pBytes + size_t(stat.cbSize.QuadPart));
            GlobalUnlock(memory);
          }
        }
      }

      if (SUCCEEDED(coInit)) {
        CoUninitialize();
      }

      if (png.empty()) {
        return false;
      }
      if (pHdrLight != nullptr) {
        png = addHdrChunks(png, *pHdrLight);
        if (png.empty()) {
          return false;
        }
      }

      std::ofstream file(std::filesystem::path(str::tows(path.c_str())), std::ios::binary | std::ios::trunc);
      file.write(reinterpret_cast<const char*>(png.data()), std::streamsize(png.size()));
      return file.good();
    }

    std::string fileName(const std::string& path) {
      return str::fromws(std::filesystem::path(str::tows(path.c_str())).filename().c_str());
    }

    // Writes the thumbnail file rtx.screenshot.thumbnailRequest describes, through a temporary file, so a reader
    // never sees a partly written one. bgr holds 8-bit BGR rows from the top.
    bool writeThumbnail(const std::string& request, uint32_t width, uint32_t height, const std::vector<uint8_t>& bgr) {
      std::vector<uint8_t> out;
      out.reserve(16 + request.size() + bgr.size());
      out.insert(out.end(), { uint8_t('R'), uint8_t('T'), uint8_t('H'), uint8_t('B') });
      for (const uint32_t v : { width, height, uint32_t(request.size()) }) {
        for (uint32_t k = 0; k < 4; ++k) {
          out.push_back(uint8_t(v >> (8 * k)));
        }
      }
      out.insert(out.end(), request.begin(), request.end());
      for (size_t i = 0; i + 2 < bgr.size(); i += 3) {
        out.push_back(bgr[i + 2]);
        out.push_back(bgr[i + 1]);
        out.push_back(bgr[i + 0]);
      }

      std::error_code ec;
      const std::filesystem::path folder = util::RtxFileSys::path(util::RtxFileSys::Logs).parent_path();
      std::filesystem::create_directories(folder, ec);
      const std::filesystem::path path = folder / L"thumbnail.rgb";
      const std::filesystem::path temp = folder / L"thumbnail.rgb.tmp";
      {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        if (!file.good()) {
          return false;
        }
      }
      return MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != FALSE;
    }

    bool exists(const std::string& path) {
      std::error_code ec;
      return std::filesystem::exists(std::filesystem::path(str::tows(path.c_str())), ec);
    }
  }

  struct RtxScreenshot::Job {
    Rc<DxvkBuffer> buffer;
    Rc<sync::Fence> fence;
    PixelLayout layout = PixelLayout::Unsupported;
    uint32_t width = 0;
    uint32_t height = 0;
    // Set when the back buffer holds HDR10 output, which gets an HDR file and a converted SDR file.
    bool hdr = false;
    float uiNits = 0.0f;
    SdrConversion conversion;
    // Set for a screenshot from the hotkey or the menu. Without it, only the bug report screenshot is written.
    bool user = false;
    // The local time of the capture, for the file names.
    std::tm time {};
    // The bug report screenshot's path, or empty.
    std::string markPath;
    // The rtx.screenshot.thumbnailRequest value of a thumbnail, or empty.
    std::string thumbnailRequest;
  };

  RtxScreenshot::RtxScreenshot(DxvkDevice* device)
    : CommonDeviceObject(device) {
  }

  RtxScreenshot::~RtxScreenshot() {
    {
      std::lock_guard<std::mutex> lock(m_queueMutex);
      m_stopping = true;
      if (!m_queue.empty()) {
        Logger::warn(str::format("[Screenshot] ", m_queue.size(), " screenshot(s) not saved: the renderer shut down first"));
      }
      m_queue.clear();
    }
    m_queueCondition.notify_all();
    if (m_worker.joinable()) {
      m_worker.join();
    }
  }

  void RtxScreenshot::request() {
    m_requested = true;
  }

  std::string RtxScreenshot::resolveFolder() {
    // The bridge looks up the game's executable through a process snapshot, so the folder is looked up once.
    static const std::filesystem::path s_exeFolder = std::filesystem::path(str::tows(env::getExePath().c_str())).parent_path();
    static const std::filesystem::path s_default = [] {
      PWSTR pictures = nullptr;
      std::filesystem::path path;
      if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Pictures, 0, nullptr, &pictures)) && pictures != nullptr) {
        path = std::filesystem::path(pictures) / L"RTX Remix" / str::tows(env::getExeBaseName().c_str());
      }
      CoTaskMemFree(pictures);
      return path.empty() ? s_exeFolder / L"Screenshots" : path;
    }();
    const std::string& option = folder();
    std::filesystem::path path = option.empty() ? s_default : std::filesystem::path(str::tows(option.c_str()));
    if (path.is_relative()) {
      path = s_exeFolder / path;
    }
    return str::fromws(path.lexically_normal().c_str());
  }

  void RtxScreenshot::capture(DxvkContext* ctx, const Rc<DxvkImage>& backBuffer, bool hdrOutput) {
    serviceThumbnail(ctx);

    const bool user = m_requested.exchange(false);
    const std::string markPath = RtxSessionLog::takeScreenshotRequest();
    if (!user && markPath.empty()) {
      return;
    }

    const PixelLayout layout = backBuffer != nullptr ? pixelLayout(backBuffer->info().format) : PixelLayout::Unsupported;
    if (layout == PixelLayout::Unsupported || backBuffer->info().sampleCount != VK_SAMPLE_COUNT_1_BIT ||
        (backBuffer->info().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
      const std::string reason = backBuffer == nullptr
        ? std::string("there is no back buffer")
        : str::format("the back buffer cannot be read back (Vulkan format ", uint32_t(backBuffer->info().format), ", ",
                      uint32_t(backBuffer->info().sampleCount), " samples)");
      if (user) {
        Logger::warn(str::format("[Screenshot] not saved: ", reason));
        setHudMessage("Screenshot not saved", "The frame cannot be read back. The log has the details.", kSavedMessageSeconds);
      }
      if (!markPath.empty()) {
        Logger::warn(str::format("[Mark] screenshot not saved: ", reason, " (", markPath, ")"));
      }
      return;
    }

    auto job = std::make_unique<Job>();
    job->layout = layout;
    job->width = backBuffer->info().extent.width;
    job->height = backBuffer->info().extent.height;
    job->hdr = hdrOutput && layout == PixelLayout::Rgba16Float;
    job->user = user;
    job->markPath = markPath;
    const std::time_t now = std::time(nullptr);
    localtime_s(&job->time, &now);

    // The tone mapping state is read now, so a menu change while the files are written does not affect them.
    if (job->hdr) {
      job->uiNits = DxvkToneMapping::hdrUiNitsClamped();
      job->conversion.sceneScale = DxvkToneMapping::hdrSceneToBackBufferScale();
      job->conversion.toneMapped = DxvkToneMapping::tonemappingEnabled();
    }

    DxvkBufferCreateInfo info;
    info.size = VkDeviceSize(job->width) * job->height * bytesPerPixel(layout);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    job->buffer = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                                 DxvkMemoryStats::Category::RTXBuffer, "Screenshot readback");
    job->fence = new sync::Fence(0);

    const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    ctx->copyImageToBuffer(job->buffer, 0, 0, 0, backBuffer, subresource, VkOffset3D { 0, 0, 0 }, backBuffer->info().extent);
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx->signal(job->fence, 1);

    if (user) {
      setHudMessage("Saving screenshot", resolveFolder(), 60.0);
    }

    {
      std::lock_guard<std::mutex> lock(m_queueMutex);
      if (!m_worker.joinable()) {
        m_worker = std::thread([this] { runWorker(); });
      }
      m_queue.push_back(std::move(job));
    }
    m_queueCondition.notify_one();
  }

  void RtxScreenshot::captureThumbnail(DxvkContext* ctx, const Rc<DxvkImage>& image) {
    const PixelLayout layout = image != nullptr ? pixelLayout(image->info().format) : PixelLayout::Unsupported;
    if (layout == PixelLayout::Unsupported || image->info().sampleCount != VK_SAMPLE_COUNT_1_BIT ||
        (image->info().usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
      if (!m_thumbnailWarned) {
        m_thumbnailWarned = true;
        Logger::warn(image == nullptr
          ? std::string("[Screenshot] no thumbnail: there is no render target")
          : str::format("[Screenshot] no thumbnail: the render target cannot be read back (Vulkan format ",
                        uint32_t(image->info().format), ", ", uint32_t(image->info().sampleCount), " samples)"));
      }
      return;
    }

    const VkExtent3D source = image->info().extent;
    const uint32_t width = std::clamp(thumbnailWidth(), 16u, std::max(16u, source.width));
    const uint32_t height = std::max(1u, uint32_t(std::lround(double(width) * source.height / std::max(1u, source.width))));
    if (m_thumbnailImage == nullptr || m_thumbnailImage->info().format != image->info().format ||
        m_thumbnailImage->info().extent.width != width || m_thumbnailImage->info().extent.height != height) {
      DxvkImageCreateInfo desc;
      desc.type = VK_IMAGE_TYPE_2D;
      desc.flags = 0;
      desc.format = image->info().format;
      desc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
      desc.extent = VkExtent3D { width, height, 1 };
      desc.numLayers = 1;
      desc.mipLevels = 1;
      desc.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      desc.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
      desc.access = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      desc.tiling = VK_IMAGE_TILING_OPTIMAL;
      desc.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      try {
        m_thumbnailImage = ctx->getDevice()->createImage(desc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                         DxvkMemoryStats::Category::RTXRenderTarget, "Screenshot thumbnail");
      } catch (const DxvkError& e) {
        m_thumbnailImage = nullptr;
        m_thumbnailFrame = nullptr;
        Logger::warn(str::format("[Screenshot] no thumbnail: creating its image failed (", e.message(), ")"));
        return;
      }
    }

    const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    VkImageBlit blit;
    blit.srcSubresource = subresource;
    blit.srcOffsets[0] = VkOffset3D { 0, 0, 0 };
    blit.srcOffsets[1] = VkOffset3D { int32_t(source.width), int32_t(source.height), 1 };
    blit.dstSubresource = subresource;
    blit.dstOffsets[0] = VkOffset3D { 0, 0, 0 };
    blit.dstOffsets[1] = VkOffset3D { int32_t(width), int32_t(height), 1 };
    const VkComponentMapping identity = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    ctx->blitImage(m_thumbnailImage, identity, image, identity, blit, VK_FILTER_LINEAR);

    // The tone mapping state is read now, so the conversion matches the frame even when it is written later.
    auto frame = std::make_unique<Job>();
    frame->layout = layout;
    frame->width = width;
    frame->height = height;
    frame->hdr = DxvkToneMapping::hdrOutputActive() && layout == PixelLayout::Rgba16Float;
    if (frame->hdr) {
      frame->uiNits = DxvkToneMapping::hdrUiNitsClamped();
      frame->conversion.sceneScale = DxvkToneMapping::hdrSceneToBackBufferScale();
      frame->conversion.toneMapped = DxvkToneMapping::tonemappingEnabled();
    }
    m_thumbnailFrame = std::move(frame);
  }

  void RtxScreenshot::serviceThumbnail(DxvkContext* ctx) {
    const std::string request = thumbnailRequest();
    if (request.empty() || request == m_handledThumbnailRequest || m_thumbnailFrame == nullptr) {
      return;
    }
    m_handledThumbnailRequest = request;

    auto job = std::make_unique<Job>(*m_thumbnailFrame);
    job->thumbnailRequest = request;

    DxvkBufferCreateInfo info;
    info.size = VkDeviceSize(job->width) * job->height * bytesPerPixel(job->layout);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    info.access = VK_ACCESS_TRANSFER_WRITE_BIT;
    try {
      job->buffer = ctx->getDevice()->createBuffer(info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                                   DxvkMemoryStats::Category::RTXBuffer, "Screenshot thumbnail readback");
    } catch (const DxvkError& e) {
      Logger::warn(str::format("[Screenshot] no thumbnail: creating its readback buffer failed (", e.message(), ")"));
      return;
    }

    const VkImageSubresourceLayers subresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    job->fence = new sync::Fence(0);
    ctx->copyImageToBuffer(job->buffer, 0, 0, 0, m_thumbnailImage, subresource, VkOffset3D { 0, 0, 0 },
                           m_thumbnailImage->info().extent);
    ctx->emitMemoryBarrier(0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    ctx->signal(job->fence, 1);

    {
      std::lock_guard<std::mutex> lock(m_queueMutex);
      if (!m_worker.joinable()) {
        m_worker = std::thread([this] { runWorker(); });
      }
      m_queue.push_back(std::move(job));
    }
    m_queueCondition.notify_one();
  }

  void RtxScreenshot::runWorker() {
    env::setThreadName("rtx-screenshot");
    while (true) {
      std::unique_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_queueCondition.wait(lock, [this] { return m_stopping || !m_queue.empty(); });
        if (m_stopping) {
          return;
        }
        job = std::move(m_queue.front());
        m_queue.pop_front();
      }
      write(*job);
    }
  }

  void RtxScreenshot::write(Job& job) {
    // The copy finishes with the frame's command list, usually one or two frames later.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(kGpuWaitSeconds);
    while (job.fence->value() < 1) {
      if (m_stopping || std::chrono::steady_clock::now() > deadline) {
        const char* reason = m_stopping ? "the renderer shut down first" : "the GPU did not finish the copy in time";
        if (job.user) {
          Logger::warn(str::format("[Screenshot] not saved: ", reason));
          setHudMessage("Screenshot not saved", "The log has the details.", kSavedMessageSeconds);
        }
        if (!job.markPath.empty()) {
          Logger::warn(str::format("[Mark] screenshot not saved: ", reason, " (", job.markPath, ")"));
        }
        if (!job.thumbnailRequest.empty()) {
          Logger::warn(str::format("[Screenshot] thumbnail not saved: ", reason));
        }
        return;
      }
      Sleep(1);
    }

    const uint8_t* pSrc = static_cast<const uint8_t*>(job.buffer->mapPtr(0));
    const size_t pixelCount = size_t(job.width) * job.height;
    std::vector<uint8_t> sdr(pixelCount * 3);
    std::vector<uint16_t> hdr;
    HdrLight light;

    if (job.hdr) {
      double sumMax = 0.0;
      // The bug report screenshot needs only the SDR image.
      if (job.user) {
        hdr.resize(pixelCount * 3);
      }
      const uint16_t* pHalf = reinterpret_cast<const uint16_t*>(pSrc);
      for (size_t i = 0; i < pixelCount; ++i) {
        // The back buffer is gamma 2.2 encoded with 1 at UI white (srgb_dither.comp.slang).
        Vector3 linear;
        for (uint32_t c = 0; c < 3; ++c) {
          linear[c] = std::pow(std::max(glm::unpackHalf1x16(pHalf[i * 4 + c]), 0.0f), kDisplayGamma);
        }

        if (!hdr.empty()) {
          const Vector3 linear2020 = multiply(kBt709ToBt2020, linear);
          double pixelMax = 0.0;
          for (uint32_t c = 0; c < 3; ++c) {
            hdr[i * 3 + c] = encodePq(std::max(linear2020[c], 0.0f), job.uiNits);
            pixelMax = std::max(pixelMax, std::clamp(double(linear2020[c]) * job.uiNits, 0.0, 10000.0));
          }
          light.maxCll = std::max(light.maxCll, pixelMax);
          sumMax += pixelMax;
        }

        const Vector3 sdrLinear = hdrToSdr(linear, job.conversion);
        sdr[i * 3 + 0] = encodeSdr(sdrLinear.z);
        sdr[i * 3 + 1] = encodeSdr(sdrLinear.y);
        sdr[i * 3 + 2] = encodeSdr(sdrLinear.x);
      }
      light.maxFall = pixelCount > 0 ? sumMax / double(pixelCount) : 0.0;
    } else {
      // An SDR back buffer is already display encoded, so its values are written as they are.
      for (size_t i = 0; i < pixelCount; ++i) {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        switch (job.layout) {
        case PixelLayout::Rgba8:
          r = pSrc[i * 4 + 0];
          g = pSrc[i * 4 + 1];
          b = pSrc[i * 4 + 2];
          break;
        case PixelLayout::Bgra8:
          b = pSrc[i * 4 + 0];
          g = pSrc[i * 4 + 1];
          r = pSrc[i * 4 + 2];
          break;
        case PixelLayout::Rgb10A2:
        case PixelLayout::Bgr10A2: {
          uint32_t packed;
          std::memcpy(&packed, pSrc + i * 4, sizeof(packed));
          const uint8_t low = quantize8(float(packed & 0x3FFu) / 1023.0f);
          g = quantize8(float((packed >> 10) & 0x3FFu) / 1023.0f);
          const uint8_t high = quantize8(float((packed >> 20) & 0x3FFu) / 1023.0f);
          r = job.layout == PixelLayout::Rgb10A2 ? low : high;
          b = job.layout == PixelLayout::Rgb10A2 ? high : low;
          break;
        }
        case PixelLayout::Rgba16Float: {
          const uint16_t* pHalf = reinterpret_cast<const uint16_t*>(pSrc) + i * 4;
          r = quantize8(glm::unpackHalf1x16(pHalf[0]));
          g = quantize8(glm::unpackHalf1x16(pHalf[1]));
          b = quantize8(glm::unpackHalf1x16(pHalf[2]));
          break;
        }
        case PixelLayout::Unsupported:
          break;
        }
        sdr[i * 3 + 0] = b;
        sdr[i * 3 + 1] = g;
        sdr[i * 3 + 2] = r;
      }
    }
    job.buffer = nullptr;

    if (job.user) {
      std::error_code ec;
      const std::string folderPath = resolveFolder();
      std::filesystem::create_directories(std::filesystem::path(str::tows(folderPath.c_str())), ec);

      char stamp[32];
      std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &job.time);
      const std::string base = str::format(folderPath, "\\", env::getExeBaseName(), "_", stamp);

      // A number is added when a file of that name exists, so both files of an HDR pair keep the same name.
      std::string hdrPath;
      std::string sdrPath;
      for (uint32_t n = 1; ; ++n) {
        const std::string stem = n == 1 ? base : str::format(base, "_", n);
        hdrPath = job.hdr ? stem + "_HDR.png" : std::string();
        sdrPath = job.hdr ? stem + "_SDR.png" : stem + ".png";
        if (!exists(sdrPath) && (hdrPath.empty() || !exists(hdrPath))) {
          break;
        }
      }

      bool success = writePng(sdrPath, job.width, job.height, sdr.data(), false, nullptr);
      if (success && job.hdr) {
        success = writePng(hdrPath, job.width, job.height, hdr.data(), true, &light);
      }
      if (success) {
        if (job.hdr) {
          Logger::info(str::format("[Screenshot] saved: ", hdrPath, " and ", sdrPath));
          setHudMessage("Screenshot saved", str::format(fileName(hdrPath), " and ", fileName(sdrPath)), kSavedMessageSeconds);
        } else {
          Logger::info(str::format("[Screenshot] saved: ", sdrPath));
          setHudMessage("Screenshot saved", fileName(sdrPath), kSavedMessageSeconds);
        }
      } else {
        Logger::warn(str::format("[Screenshot] not saved: writing ", job.hdr ? hdrPath + " or " + sdrPath : sdrPath, " failed"));
        setHudMessage("Screenshot not saved", "The log has the details.", kSavedMessageSeconds);
      }
    }

    if (!job.thumbnailRequest.empty() && !writeThumbnail(job.thumbnailRequest, job.width, job.height, sdr)) {
      Logger::warn("[Screenshot] thumbnail not saved: writing rtx-remix\\thumbnail.rgb failed");
    }

    if (!job.markPath.empty()) {
      if (writePng(job.markPath, job.width, job.height, sdr.data(), false, false)) {
        Logger::status(str::format("[Mark] screenshot saved: ", job.markPath));
      } else {
        Logger::warn(str::format("[Mark] screenshot not saved: writing ", job.markPath, " failed"));
      }
    }
  }

  void RtxScreenshot::setHudMessage(std::string text, std::string subText, double seconds) {
    std::lock_guard<std::mutex> lock(m_hudMutex);
    m_hudText = std::move(text);
    m_hudSubText = std::move(subText);
    m_hudUntil = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
  }

  bool RtxScreenshot::getHudMessage(std::string& text, std::string& subText) const {
    std::lock_guard<std::mutex> lock(m_hudMutex);
    if (m_hudText.empty() || std::chrono::steady_clock::now() > m_hudUntil) {
      return false;
    }
    text = m_hudText;
    subText = m_hudSubText;
    return true;
  }

}
