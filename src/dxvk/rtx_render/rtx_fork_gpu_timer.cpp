#include "rtx_fork_gpu_timer.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "../dxvk_context.h"
#include "../dxvk_device.h"
#include "rtx_options.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {
  namespace fork_hooks {

    namespace {

      // 4096 in-flight pass records (8192 timestamp queries). A frame
      // carries a few hundred annotated passes; results retire within a
      // couple of frames, so this covers deep pipelining with a wide margin.
      constexpr uint32_t kMaxRecords = 4096;
      constexpr uint32_t kQueryPoolSize = kMaxRecords * 2;
      // Give up on a record whose queries never become available (its
      // command buffer was discarded before submit): reclaim the slots.
      constexpr uint64_t kMaxAgeFrames = 90;
      // How many passes the ranked log table shows.
      constexpr size_t kLogTopCount = 28;

      struct PassAccum {
        uint64_t totalNs = 0;
        uint64_t hits = 0;
      };

      struct Record {
        const char* name = nullptr;
        uint32_t    slot = 0;      // begin query index; end = slot + 1
        uint64_t    frameId = 0;
      };

      struct TimerState {
        std::mutex mutex;
        DxvkDevice* device = nullptr;
        VkQueryPool pool = VK_NULL_HANDLE;
        bool initAttempted = false;
        bool valid = false;
        double nsPerTick = 1.0;

        // Ring of pass records: head allocates, tail drains in order.
        Record ring[kMaxRecords];
        uint64_t head = 0;
        uint64_t tail = 0;

        uint64_t lastSeenFrame = 0;
        uint64_t windowStartFrame = 0;
        uint64_t dropped = 0;   // ring full — sample not taken
        uint64_t lost = 0;      // queries never became available

        std::unordered_map<const char*, PassAccum> accum;
      };

      TimerState& state() {
        static TimerState s;
        return s;
      }

      // Per-thread nesting stack. Every gpuTimerBegin pushes exactly one
      // entry (kInvalidRecord when disabled or not sampled) and every
      // gpuTimerEnd pops exactly one, so ctor/dtor pairing survives option
      // toggles mid-zone and allocation failures.
      constexpr uint64_t kInvalidRecord = ~0ull;
      thread_local std::vector<uint64_t> t_nesting;

      void ensureInit(TimerState& s, DxvkContext* ctx) {
        if (s.initAttempted) {
          return;
        }
        s.initAttempted = true;

        DxvkDevice* device = ctx->getDevice().ptr();
        if (!device->features().vulkan12Features.hostQueryReset) {
          Logger::warn("[GPU-Timer] hostQueryReset feature unavailable -- GPU pass timer disabled");
          return;
        }

        VkQueryPoolCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = kQueryPoolSize;
        if (device->vkd()->vkCreateQueryPool(device->handle(), &info, nullptr, &s.pool) != VK_SUCCESS) {
          Logger::warn("[GPU-Timer] query pool creation failed -- GPU pass timer disabled");
          return;
        }
        device->vkd()->vkResetQueryPool(device->handle(), s.pool, 0, kQueryPoolSize);

        s.device = device;
        s.nsPerTick = device->adapter()->deviceProperties().limits.timestampPeriod;
        s.valid = true;
        Logger::info(str::format("[GPU-Timer] enabled: ", kMaxRecords,
                                 " pass records, log interval ",
                                 RtxOptions::gpuTimerLogIntervalFrames(), " frames"));
      }

      void logWindow(TimerState& s, uint64_t currentFrame) {
        const uint64_t frames = currentFrame > s.windowStartFrame
            ? currentFrame - s.windowStartFrame : 1;

        std::vector<std::pair<const char*, PassAccum>> sorted(s.accum.begin(), s.accum.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                    return a.second.totalNs > b.second.totalNs;
                  });

        Logger::info(str::format("[GPU-Timer] window frames=", frames,
                                 " passes=", sorted.size(),
                                 " dropped=", s.dropped, " lost=", s.lost,
                                 " -- avg GPU ms/frame per pass (nested zones overlap):"));
        const size_t count = std::min(sorted.size(), kLogTopCount);
        for (size_t i = 0; i < count; ++i) {
          const double msPerFrame =
              (double)sorted[i].second.totalNs / 1.0e6 / (double)frames;
          const double hitsPerFrame =
              (double)sorted[i].second.hits / (double)frames;
          // Skip sub-0.01ms noise once we're past the top entries.
          if (msPerFrame < 0.01 && i >= 8) {
            break;
          }
          char line[192];
          std::snprintf(line, sizeof(line), "[GPU-Timer] %8.3f ms  %-56s (x%.1f/frame)",
                        msPerFrame, sorted[i].first, hitsPerFrame);
          Logger::info(line);
        }

        s.accum.clear();
        s.dropped = 0;
        s.lost = 0;
        s.windowStartFrame = currentFrame;
      }

      // Retire completed records in submission order; give up on ancient
      // ones (discarded command buffers). Caller holds the mutex.
      void drain(TimerState& s, uint64_t currentFrame) {
        DxvkDevice* device = s.device;
        while (s.tail != s.head) {
          Record& r = s.ring[s.tail % kMaxRecords];
          // value + availability per query, two queries per record.
          uint64_t results[4] = {};
          device->vkd()->vkGetQueryPoolResults(
              device->handle(), s.pool, r.slot, 2,
              sizeof(results), results, 2 * sizeof(uint64_t),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
          const bool beginReady = results[1] != 0;
          const bool endReady = results[3] != 0;

          if (beginReady && endReady) {
            if (results[2] > results[0]) {
              PassAccum& a = s.accum[r.name];
              a.totalNs += (uint64_t)((double)(results[2] - results[0]) * s.nsPerTick);
              a.hits += 1;
            }
          } else if (currentFrame - r.frameId <= kMaxAgeFrames) {
            break;  // still in flight -- retire later, keep order
          } else {
            ++s.lost;  // command buffer discarded before submit
          }
          device->vkd()->vkResetQueryPool(device->handle(), s.pool, r.slot, 2);
          ++s.tail;
        }
      }

    } // namespace

    void gpuTimerBegin(DxvkContext* ctx, const char* name) {
      if (!RtxOptions::gpuTimerEnable()) {
        t_nesting.push_back(kInvalidRecord);
        return;
      }

      TimerState& s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      ensureInit(s, ctx);
      if (!s.valid) {
        t_nesting.push_back(kInvalidRecord);
        return;
      }

      const uint64_t frame = ctx->getDevice()->getCurrentFrameId();
      if (frame != s.lastSeenFrame) {
        s.lastSeenFrame = frame;
        drain(s, frame);
        const uint32_t interval = std::max(60u, RtxOptions::gpuTimerLogIntervalFrames());
        if (frame - s.windowStartFrame >= interval) {
          logWindow(s, frame);
        }
      }

      if (s.head - s.tail >= kMaxRecords) {
        ++s.dropped;
        t_nesting.push_back(kInvalidRecord);
        return;
      }

      const uint64_t recordIdx = s.head++;
      Record& r = s.ring[recordIdx % kMaxRecords];
      r.name = name;
      r.slot = (uint32_t)((recordIdx % kMaxRecords) * 2);
      r.frameId = frame;

      s.device->vkd()->vkCmdWriteTimestamp(
          ctx->getCmdBuffer(DxvkCmdBuffer::ExecBuffer),
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, r.slot);

      t_nesting.push_back(recordIdx);
    }

    void gpuTimerEnd(DxvkContext* ctx) {
      if (t_nesting.empty()) {
        return;  // begin ran before the module was live -- nothing to pair
      }
      const uint64_t recordIdx = t_nesting.back();
      t_nesting.pop_back();
      if (recordIdx == kInvalidRecord) {
        return;
      }

      TimerState& s = state();
      std::lock_guard<std::mutex> lock(s.mutex);
      if (!s.valid) {
        return;
      }
      const Record& r = s.ring[recordIdx % kMaxRecords];
      s.device->vkd()->vkCmdWriteTimestamp(
          ctx->getCmdBuffer(DxvkCmdBuffer::ExecBuffer),
          VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, r.slot + 1);
    }

  } // namespace fork_hooks
} // namespace dxvk
