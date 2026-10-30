// D3D9 identity trace and dry-run replay. See rtx_fork_d3d9_identity_trace.h.

#include "rtx_fork_d3d9_identity_trace.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <type_traits>
#include <vector>

#include "rtx_draw_call_tracker.h"
#include "rtx_fork_d3d9_alpha.h"
#include "rtx_options.h"
#include "rtx_scene_interface.h"
#include "translator/rtx_translator_lifetime.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  bool D3d9IdentityTrace::s_recording = false;

  namespace {
    constexpr char kMagic[4] = { 'D', '9', 'I', 'T' };
    constexpr uint32_t kVersion = 1;

    // The options that shaped the recorded hashes and lifetimes, for the replay's report.
    struct FileHeader {
      char magic[4];
      uint32_t version;
      uint32_t drawSize;                 // sizeof(DrawRecord) of the recording build
      uint32_t firstFrame;
      float uniqueObjectDistance;
      uint32_t numFramesToKeepInstances;
      uint32_t vsStableHash;             // always 0, kept for the file layout
      uint32_t dynamicTopologyKey;       // always 0, kept for the file layout
      uint32_t stableGraceFrames;        // always 0, kept for the file layout
      uint32_t objectAntiCulling;
      uint32_t d3d9CutAsEvent;
      uint32_t trackerMaterialFallback;  // always 0, kept for the file layout
      uint32_t reserved[4];
    };

    enum class RecordType : uint32_t { Draw = 1, Collect = 2, Cut = 3, InvalidCamera = 4, Clear = 5, FrameEnd = 6 };

    struct RecordHeader {
      RecordType type;
      uint32_t frame;
    };

    // Plain data, written and read with memcpy by the same build (FileHeader::drawSize guards the layout).
    struct DrawRecord {
      ReplacementInstance::IdentityTrace trace;  // classifier inputs; trace.material is the lookup's material hash
      XXH64_hash_t identityHash;                 // L1 key as computed in game
      XXH64_hash_t topologyHash;
      XXH64_hash_t dynamicKey;                   // always kEmptyHash, kept for the file layout
      XXH64_hash_t spatialKey;                   // the spatial key the recorded lookup used
      XXH64_hash_t vertexPositionHash;
      float objectToWorld[16];
      float worldPos[3];
      float boundsMin[3];                        // object space
      float boundsMax[3];
      uint8_t boundsValid;
      uint8_t hasReplacements;
      uint8_t pad[2];
    };
    static_assert(std::is_trivially_copyable_v<ReplacementInstance::IdentityTrace>);
    static_assert(sizeof(Matrix4) == 16 * sizeof(float));

    struct CollectRecord {
      uint8_t cameraCut;
      uint8_t antiCullingSupported;
      uint8_t pad[6];
    };

    struct ClearRecord {
      uint32_t reason;  // D3d9Identity::ClearReason
      uint32_t pad;
    };

    // Recording state (CS thread).
    FILE* s_file = nullptr;
    std::vector<uint8_t> s_buffer;
    bool s_armed = true;          // a non-zero traceFrames starts a recording (cleared until the option goes back to 0)
    uint32_t s_framesLeft = 0;
    uint32_t s_framesWritten = 0;
    uint64_t s_drawsWritten = 0;
    uint64_t s_bytesWritten = 0;
    std::string s_path;

    void appendBytes(const void* data, size_t size) {
      const uint8_t* bytes = static_cast<const uint8_t*>(data);
      s_buffer.insert(s_buffer.end(), bytes, bytes + size);
    }

    void appendRecord(RecordType type, uint32_t frame, const void* payload = nullptr, size_t payloadSize = 0) {
      const RecordHeader header { type, frame };
      appendBytes(&header, sizeof(header));
      if (payloadSize != 0) {
        appendBytes(payload, payloadSize);
      }
    }

    std::string makeTracePath() {
      const std::time_t now = std::time(nullptr);
      std::tm local {};
      localtime_s(&local, &now);
      char name[64];
      std::strftime(name, sizeof(name), "identity-trace-%Y%m%d-%H%M%S.d9it", &local);
      std::error_code ec;
      std::filesystem::create_directories("rtx-remix", ec);
      return (std::filesystem::absolute(std::filesystem::path("rtx-remix") / name, ec)).string();
    }

    void closeTrace() {
      if (s_file != nullptr) {
        std::fclose(s_file);
        s_file = nullptr;
      }
      // D3d9IdentityTrace::s_recording is private: the member callers clear it.
      s_buffer.clear();
      s_buffer.shrink_to_fit();
    }

    // The texture transform and texgen mode travel as one hash; the tracker only compares them for equality, so a
    // matrix holding the hash bits keeps the comparisons the same.
    Matrix4 textureTransformFromHash(XXH64_hash_t hash) {
      float values[16] = {};
      std::memcpy(values, &hash, sizeof(hash));
      Matrix4 m;
      std::memcpy(&m, values, sizeof(values));
      return m;
    }
  }

  // ---- recording ------------------------------------------------------------------------------------------------

  void D3d9IdentityTrace::recordDraw(uint32_t frameId, const D3d9Identity::Lookup& lookup, const ReplacementInstance::LookupKey& key,
                                     const AxisAlignedBoundingBox& bounds, bool hasReplacements) {
    if (!s_recording) {
      return;
    }
    DrawRecord d {};
    d.trace = lookup.trace;
    d.identityHash = key.identityHash;
    d.topologyHash = lookup.topology;
    d.dynamicKey = kEmptyHash;
    d.spatialKey = key.spatialMapHash;
    d.vertexPositionHash = key.vertexPositionHash;
    std::memcpy(d.objectToWorld, &key.transform, sizeof(d.objectToWorld));
    for (uint32_t i = 0; i < 3; ++i) {
      d.worldPos[i] = key.worldPos[i];
      d.boundsMin[i] = bounds.minPos[i];
      d.boundsMax[i] = bounds.maxPos[i];
    }
    d.boundsValid = bounds.isValid() ? 1 : 0;
    d.hasReplacements = hasReplacements ? 1 : 0;
    appendRecord(RecordType::Draw, frameId, &d, sizeof(d));
    ++s_drawsWritten;
  }

  void D3d9IdentityTrace::recordCollect(uint32_t frameId, bool cameraCut, bool antiCullingSupported) {
    if (!s_recording) {
      return;
    }
    CollectRecord c {};
    c.cameraCut = cameraCut ? 1 : 0;
    c.antiCullingSupported = antiCullingSupported ? 1 : 0;
    appendRecord(RecordType::Collect, frameId, &c, sizeof(c));
  }

  void D3d9IdentityTrace::recordCut(uint32_t frameId) {
    if (s_recording) {
      appendRecord(RecordType::Cut, frameId);
    }
  }

  void D3d9IdentityTrace::recordInvalidCamera(uint32_t frameId) {
    if (s_recording) {
      appendRecord(RecordType::InvalidCamera, frameId);
    }
  }

  void D3d9IdentityTrace::recordClear(uint32_t frameId, D3d9Identity::ClearReason reason) {
    // Cuts and invalid cameras are recorded as events; the replay decides whether they clear.
    if (!s_recording || reason == D3d9Identity::ClearReason::CameraCut || reason == D3d9Identity::ClearReason::InvalidCamera) {
      return;
    }
    ClearRecord c {};
    c.reason = uint32_t(reason);
    appendRecord(RecordType::Clear, frameId, &c, sizeof(c));
  }

  void D3d9IdentityTrace::onFrameEnd(uint32_t frameId) {
    const uint32_t wanted = traceFrames();
    if (!s_recording) {
      if (wanted == 0) {
        s_armed = true;
        return;
      }
      if (!s_armed) {
        return;  // this setting's trace is written; 0 and back records another
      }
      s_armed = false;
      s_path = makeTracePath();
      s_file = std::fopen(s_path.c_str(), "wb");
      if (s_file == nullptr) {
        Logger::warn(str::format("[D3d9IdentityTrace] cannot create ", s_path, "; not recording"));
        return;
      }
      FileHeader header {};
      std::memcpy(header.magic, kMagic, sizeof(kMagic));
      header.version = kVersion;
      header.drawSize = uint32_t(sizeof(DrawRecord));
      header.firstFrame = frameId + 1;
      header.uniqueObjectDistance = RtxOptions::uniqueObjectDistance();
      header.numFramesToKeepInstances = RtxOptions::numFramesToKeepInstances();
      header.vsStableHash = 0;
      header.dynamicTopologyKey = 0;
      header.stableGraceFrames = 0;
      header.objectAntiCulling = RtxOptions::AntiCulling::isObjectAntiCullingEnabled() ? 1 : 0;
      header.d3d9CutAsEvent = scene::Options::d3d9CutAsEvent() ? 1 : 0;
      header.trackerMaterialFallback = 0;
      std::fwrite(&header, sizeof(header), 1, s_file);
      s_recording = true;
      s_framesLeft = wanted;
      s_framesWritten = 0;
      s_drawsWritten = 0;
      s_bytesWritten = sizeof(header);
      s_buffer.clear();
      s_buffer.reserve(size_t(1) << 20);
      Logger::info(str::format("[D3d9IdentityTrace] recording ", wanted, " frames from frame ", frameId + 1, " to ", s_path));
      return;
    }

    appendRecord(RecordType::FrameEnd, frameId);
    const bool written = std::fwrite(s_buffer.data(), 1, s_buffer.size(), s_file) == s_buffer.size();
    s_bytesWritten += s_buffer.size();
    s_buffer.clear();
    ++s_framesWritten;
    if (!written) {
      Logger::warn(str::format("[D3d9IdentityTrace] write failed after ", s_framesWritten, " frames (disk full?); ", s_path, " is cut short"));
      closeTrace();
      s_recording = false;
      return;
    }
    if (--s_framesLeft == 0 || wanted == 0) {
      closeTrace();
      s_recording = false;
      Logger::info(str::format("[D3d9IdentityTrace] wrote ", s_framesWritten, " frames, ", s_drawsWritten, " draws, ",
                               s_bytesWritten >> 20, " MiB to ", s_path));
    }
  }

  // ---- replay ---------------------------------------------------------------------------------------------------

  bool D3d9IdentityTrace::replay(DxvkDevice* device, const wchar_t* path, remixapi_fork_IdentityReplayResult& result) {
    result = remixapi_fork_IdentityReplayResult {};
    const auto fail = [&](const std::string& message) {
      std::snprintf(result.message, sizeof(result.message), "%s", message.c_str());
      Logger::err(str::format("[D3d9IdentityReplay] ", message));
      return false;
    };
    if (s_recording) {
      return fail("a trace is being recorded in this process");
    }
    FILE* file = _wfopen(path, L"rb");
    if (file == nullptr) {
      return fail("cannot open the trace file");
    }
    std::vector<char> ioBuffer(size_t(1) << 20);
    std::setvbuf(file, ioBuffer.data(), _IOFBF, ioBuffer.size());

    FileHeader header {};
    if (std::fread(&header, sizeof(header), 1, file) != 1 || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
        header.version != kVersion) {
      std::fclose(file);
      return fail("not a D3D9 identity trace, or another format version");
    }
    if (header.drawSize != sizeof(DrawRecord)) {
      std::fclose(file);
      return fail(str::format("the trace's draw records are ", header.drawSize, " bytes and this build reads ", sizeof(DrawRecord),
                              "; replay it with the build that recorded it"));
    }

    const bool cutAsEvent = scene::Options::d3d9CutAsEvent();
    const std::string recorded = str::format(
      "recorded: from frame ", header.firstFrame, ", uniqueObjectDistance ", header.uniqueObjectDistance,
      ", numFramesToKeepInstances ", header.numFramesToKeepInstances, ", vsStableHash ", header.vsStableHash,
      ", dynamicTopologyKey ", header.dynamicTopologyKey, ", stableGraceFrames ", header.stableGraceFrames,
      ", objectAntiCulling ", header.objectAntiCulling, ", d3d9CutAsEvent ", header.d3d9CutAsEvent,
      ", trackerMaterialFallback ", header.trackerMaterialFallback);
    const std::string replayed = str::format(
      "replay: uniqueObjectDistance ", RtxOptions::uniqueObjectDistance(),
      ", numFramesToKeepInstances ", RtxOptions::numFramesToKeepInstances(),
      ", objectAntiCulling 0 (no camera in a replay), d3d9CutAsEvent ", cutAsEvent ? 1 : 0);
    Logger::info(str::format("[D3d9IdentityReplay] ", recorded));
    Logger::info(str::format("[D3d9IdentityReplay] ", replayed));

    DrawCallTracker tracker(device);
    uint32_t frame = header.firstFrame;
    tracker.setFrameOverride(&frame);
    // [D3d9Identity] counts from the frame after the one that turns it on.
    D3d9Identity::onFrameEnd(frame - 1);

    uint32_t keptCutFrame = ~0u;
    bool cutClearPending = false;
    const auto clearNow = [&](D3d9Identity::ClearReason reason) {
      const size_t count = tracker.getReplacementInstanceCount();
      D3d9Identity::setPendingClearReason(reason);
      D3d9Identity::onSceneClear(frame, count);
      tracker.clear();
      result.destroyed += count;
      ++result.clears;
    };

    bool damaged = false;
    RecordHeader record {};
    DrawRecord d {};
    while (!damaged && std::fread(&record, sizeof(record), 1, file) == 1) {
      frame = record.frame;
      switch (record.type) {
      case RecordType::Draw: {
        if (std::fread(&d, sizeof(d), 1, file) != 1) {
          damaged = true;
          break;
        }
        ReplacementInstance::LookupKey key {};
        key.identityHash = d.identityHash;
        key.spatialMapHash = d.topologyHash;
        key.materialHash = d.trace.material;
        key.vertexPositionHash = d.vertexPositionHash;
        key.worldPos = Vector3(d.worldPos[0], d.worldPos[1], d.worldPos[2]);
        std::memcpy(&key.transform, d.objectToWorld, sizeof(d.objectToWorld));
        key.textureTransform = textureTransformFromHash(d.trace.textureTransform);
        key.texgenMode = TexGenMode::None;

        D3d9Identity::Lookup lookup;
        lookup.trace = d.trace;
        lookup.worldPos = key.worldPos;
        lookup.topology = d.topologyHash;

        const bool exact = tracker.findReplacementInstanceByIdentity(key.identityHash) != nullptr;
        const size_t before = tracker.getReplacementInstanceCount();
        ReplacementInstance* ri = tracker.findOrCreateKeyed(key, D3d9Identity::statsActive() ? &lookup : nullptr);
        ++result.draws;
        if (tracker.getReplacementInstanceCount() != before) {
          ++result.created;
        } else if (exact) {
          ++result.exact;
        } else {
          ++result.matched;
        }

        // What submitDrawState writes into the instance after the lookup.
        ri->frameLastSeen = frame;
        ri->categoryFlags = d.trace.categories;
        ri->isSkinned = (d.trace.flags & D3d9Identity::FlagSkinned) != 0;
        ri->textureTransform = key.textureTransform;
        ri->texgenMode = key.texgenMode;
        if (d.boundsValid != 0 && d.hasReplacements == 0) {
          ri->geometryBoundingBox.minPos = Vector3(d.boundsMin[0], d.boundsMin[1], d.boundsMin[2]);
          ri->geometryBoundingBox.maxPos = Vector3(d.boundsMax[0], d.boundsMax[1], d.boundsMax[2]);
        }
        if (d.boundsValid != 0 || d.hasReplacements != 0) {
          ri->objectToWorld = key.transform;
        }
        break;
      }
      case RecordType::Collect: {
        CollectRecord c {};
        if (std::fread(&c, sizeof(c), 1, file) != 1) {
          damaged = true;
          break;
        }
        // No camera: anti-culling stays off (objectAntiCulling and antiCullingSupported false).
        translator::LifetimeFrame lifetime;
        lifetime.framesToKeepObjects = RtxOptions::numFramesToKeepInstances();
        lifetime.framesToKeepLights = RtxOptions::AntiCulling::Light::numFramesToExtendLightLifetime();
        lifetime.cameraCut = c.cameraCut != 0;
        lifetime.keptCutFrame = keptCutFrame;
        const size_t before = tracker.getReplacementInstanceCount();
        tracker.collectReplacementInstances(lifetime);
        result.destroyed += before - tracker.getReplacementInstanceCount();
        break;
      }
      case RecordType::Cut:
        if (cutAsEvent) {
          keptCutFrame = frame;
          D3d9Identity::noteCutKept();
          ++result.cutsKept;
        } else {
          cutClearPending = true;  // SceneManager clears at the frame end
        }
        break;
      case RecordType::InvalidCamera:
        if (!cutAsEvent) {
          clearNow(D3d9Identity::ClearReason::InvalidCamera);
        }
        break;
      case RecordType::Clear: {
        ClearRecord c {};
        if (std::fread(&c, sizeof(c), 1, file) != 1 || c.reason >= uint32_t(D3d9Identity::ClearReason::Count)) {
          damaged = true;
          break;
        }
        clearNow(D3d9Identity::ClearReason(c.reason));
        break;
      }
      case RecordType::FrameEnd:
        if (cutClearPending) {
          clearNow(D3d9Identity::ClearReason::CameraCut);
          cutClearPending = false;
        }
        D3d9Identity::onFrameEnd(frame);
        ++result.frames;
        result.peakInstances = std::max<uint64_t>(result.peakInstances, tracker.getReplacementInstanceCount());
        break;
      default:
        damaged = true;
        break;
      }
    }
    std::fclose(file);
    result.endInstances = tracker.getReplacementInstanceCount();

    const std::string totals = str::format(
      "frames=", result.frames, " draws=", result.draws, " exact=", result.exact, " matched=", result.matched,
      " created=", result.created, " destroyed=", result.destroyed, " clears=", result.clears, " cutsKept=", result.cutsKept,
      " peakInstances=", result.peakInstances, " endInstances=", result.endInstances);
    Logger::info(str::format("[D3d9IdentityReplay] ", totals, damaged ? " (trace damaged or cut short; replayed up to it)" : ""));
    std::snprintf(result.message, sizeof(result.message), "%s\n%s%s", recorded.c_str(), replayed.c_str(),
                  damaged ? "\ntrace damaged or cut short; replayed up to it" : "");
    return true;
  }

} // namespace dxvk
