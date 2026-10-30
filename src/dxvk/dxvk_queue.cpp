/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "dxvk_device.h"
#include "dxvk_queue.h"
#include "dxvk_scoped_annotation.h"
// NV-DXVK start: present pacing
#include "dxvk_perf_counters.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <utility>
// Remix-present-trace.csv lives next to remix-dxvk.log
#include "../util/util_filesys.h"
#include "../util/util_string.h"
// NV-DXVK end

// NV-DXVK start: Nsight Graphics capture
#include "rtx_render/rtx_nsight_capture.h"
// NV-DXVK end
#include "rtx_render/rtx_gpu_breadcrumbs.h"

#include "NvLowLatencyVk.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"

#include "../util/util_env.h"
// NV-DXVK start: Shared crash-report exit code / dialog helpers
#include "../util/util_crash_report.h"
#include "rtx_render/rtx_bridge_message_channel.h"
// NV-DXVK end
#include "../util/util_sentry.h"

namespace dxvk {
  
  DxvkSubmissionQueue::DxvkSubmissionQueue(DxvkDevice* device)
  : m_device(device),
    m_submitThread([this] () { submitCmdLists(); }),
    m_finishThread([this] () { finishCmdLists(); }) {
  }
  
  
  DxvkSubmissionQueue::~DxvkSubmissionQueue() {
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      m_stopped.store(true);
    }
    
    m_appendCond.notify_all();
    m_submitCond.notify_all();

    m_submitThread.join();
    m_finishThread.join();

    // Both threads are gone, so the trace state is ours.
    flushPresentTrace(true);
  }

  
  void DxvkSubmissionQueue::submit(DxvkSubmitInfo submitInfo) {
    ScopedCpuProfileZone();
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    m_finishCond.wait(lock, [this] {
      return m_submitQueue.size() + m_finishQueue.size() <= MaxNumQueuedCommandBuffers;
    });

    DxvkSubmitEntry entry = { };
    entry.submit = std::move(submitInfo);

    m_pending += 1;
    m_submitQueue.push(std::move(entry));
    m_appendCond.notify_all();
  }


  void DxvkSubmissionQueue::present(DxvkPresentInfo presentInfo, DxvkSubmitStatus* status) {
    ScopedCpuProfileZone();
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    DxvkSubmitEntry entry = { };
    entry.status  = status;
    entry.present = std::move(presentInfo);
    m_submitQueue.push(std::move(entry));
    m_appendCond.notify_all();
  }


// NV-DXVK begin: DLFG integration
  void DxvkSubmissionQueue::setupFrameInterpolation(DxvkFrameInterpolationInfo frameInterpolationInfo) {
    ZoneScoped;
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    DxvkSubmitEntry entry = { };
    entry.frameInterpolation = std::move(frameInterpolationInfo);
    m_submitQueue.push(std::move(entry));
    m_appendCond.notify_all();
  }
// NV-DXVK end

  void DxvkSubmissionQueue::synchronizeSubmission(
          DxvkSubmitStatus*   status) {
    ScopedCpuProfileZone();
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    m_submitCond.wait(lock, [status] {
      return status->result.load() != VK_NOT_READY;
    });
  }


  void DxvkSubmissionQueue::synchronize() {
    ScopedCpuProfileZone();
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    m_submitCond.wait(lock, [this] {
      return m_submitQueue.empty();
    });

    // NV-DXVK start: DLFG integration
    if (m_lastPresenter != nullptr) {
      m_lastPresenter->synchronize();
      m_lastPresenter = nullptr;
    }
    // NV-DXVK end

    // Drop the presenter reference so a swapchain being destroyed (waitForIdle) is not
    // kept alive by the pacing state. With m_mutex held and the queue empty the submit thread is not using it.
    m_presentWait.presenter = nullptr;
    m_presentWait.armed = false;
  }


  void DxvkSubmissionQueue::lockDeviceQueue() {
    ScopedCpuProfileZone();
    m_mutexQueue.lock();
  }


  void DxvkSubmissionQueue::unlockDeviceQueue() {
    ScopedCpuProfileZone();
    m_mutexQueue.unlock();
  }

  // NV-DXVK start: GPU crash diagnostics
  void DxvkSubmissionQueue::onGpuCrash(const char* reason) {
    if (m_gpuCrashHandled.exchange(true)) {
      return;
    }

    Logger::err(str::format("[Session] device lost (", reason != nullptr ? reason : "?", ")"));

    // Logged first, while the marker buffer still holds the faulting frame's passes.
    if (GpuBreadcrumbs* breadcrumbs = m_device->gpuBreadcrumbs()) {
      breadcrumbs->logDeviceLost(reason);
    }

    // The submission queue is destroyed and its threads are joined before the
    // device's common objects, so the scene manager and recorder are alive here.
    m_device->getCommon()->getSceneManager().getAccelManager().dumpCrashState(reason);

    if (m_device->config().enableAftermath) {
      // Stall the pending exception until Aftermath has finished writing or reports an error.
      constexpr uint32_t kTimeoutPreventionLimit = 5000;
      constexpr uint32_t kTimeoutPerTry = 100;
      uint32_t elapsedTime = 0;
      GFSDK_Aftermath_CrashDump_Status aftermathStatus = GFSDK_Aftermath_CrashDump_Status_NotStarted;

      while (elapsedTime < kTimeoutPreventionLimit) {
        GFSDK_Aftermath_GetCrashDumpStatus(&aftermathStatus);

        if (aftermathStatus == GFSDK_Aftermath_CrashDump_Status_Finished
         || aftermathStatus == GFSDK_Aftermath_CrashDump_Status_Unknown) {
          break;
        }

        Sleep(kTimeoutPerTry);
        elapsedTime += kTimeoutPerTry;
      }
    }
  }
  // NV-DXVK end

  // NV-DXVK start: present pacing
  namespace {
    // "p50/p95/p99/max" with two decimals, or "-" when empty.
    std::string pacingPct(std::vector<float>& v) {
      if (v.empty()) {
        return "-";
      }
      std::sort(v.begin(), v.end());
      const auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))]; };
      char buf[96];
      std::snprintf(buf, sizeof(buf), "%.2f/%.2f/%.2f/%.2f", at(0.50), at(0.95), at(0.99), v.back());
      return buf;
    }

    std::string pacingMs(float v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.1f", v);
      return buf;
    }

    // PresentMon --qpc_time_ms writes QueryPerformanceCounter * 1000 / frequency.
    // Write the same clock so the two CSVs line up without guessing an offset.
    double qpcMs() {
      static const double kMsPerTick = [] {
        LARGE_INTEGER f;
        ::QueryPerformanceFrequency(&f);
        return 1000.0 / double(f.QuadPart);
      }();
      LARGE_INTEGER c;
      ::QueryPerformanceCounter(&c);
      return double(c.QuadPart) * kMsPerTick;
    }
  }

  bool DxvkSubmissionQueue::BurstDetector::push(float ms, float* outLateMs) {
    ring[count % 16] = ms;
    count += 1;
    if (count < 16) {
      return false;
    }
    float sorted[16];
    std::copy(std::begin(ring), std::end(ring), std::begin(sorted));
    std::nth_element(std::begin(sorted), std::begin(sorted) + 8, std::end(sorted));
    const float median = sorted[8];
    const float late = ring[(count - 4) % 16];
    const float fastMax = std::max({ ring[(count - 3) % 16], ring[(count - 2) % 16], ring[(count - 1) % 16] });
    if (late > 30.f && late > 2.5f * median && fastMax < 0.5f * median) {
      *outLateMs = late;
      return true;
    }
    return false;
  }

  uint64_t DxvkSubmissionQueue::waitRemixPresentGate(uint32_t& framesInFlight, uint32_t& cmdListsInFlight, uint32_t& submitQueueLen,
                                                     uint64_t& outFrameSeq, bool& gpuDoneAtPresent) {
    const uint32_t limit = std::min(perf::g_presentGpuQueueLimit.load(std::memory_order_relaxed), 4u);
    std::unique_lock<dxvk::mutex> lock(m_mutex);

    // The command list submitted last (the swapchain blit that signals the present semaphore)
    // closes this frame. Everything with a lower or equal seq is this frame or older.
    const uint64_t frameSeq = m_submittedSeq;
    uint64_t* prev = m_pacing.prevFrameSeqs;
    const uint64_t target = limit == 1 ? frameSeq : (limit >= 2 ? prev[limit - 2] : 0);
    for (uint32_t i = 3; i > 0; --i) {
      prev[i] = prev[i - 1];
    }
    prev[0] = frameSeq;

    outFrameSeq = frameSeq;
    gpuDoneAtPresent = frameSeq <= m_completedSeq;
    if (frameSeq > m_completedSeq) {
      m_presentFrameSeqs.push_back(frameSeq);
    }
    framesInFlight = uint32_t(m_presentFrameSeqs.size());
    cmdListsInFlight = uint32_t(frameSeq - m_completedSeq);
    submitQueueLen = m_submitQueue.empty() ? 0u : uint32_t(m_submitQueue.size() - 1); // minus this present

    if (target <= m_completedSeq) {
      return 0;
    }

    // Bounded: a stuck GPU is the watchdog's business, not the present gate's.
    const uint64_t startNs = perf::nowNs();
    // rtx.present.gpuQueueLimitMaxWaitMs (default 200 ms). A short cap gives a partial drain.
    const uint32_t maxWaitUs = std::clamp(perf::g_presentGateMaxWaitUs.load(std::memory_order_relaxed), 100u, 200000u);
    const auto deadline = dxvk::high_resolution_clock::now() + std::chrono::microseconds(maxWaitUs);
    while (m_completedSeq < target && !m_stopped.load() && m_lastError.load() != VK_ERROR_DEVICE_LOST) {
      const auto now = dxvk::high_resolution_clock::now();
      if (now >= deadline) {
        m_pacing.gateTimeouts += 1;
        break;
      }
      m_finishCond.wait_for(lock, std::max(std::chrono::duration_cast<std::chrono::microseconds>(deadline - now),
                                           std::chrono::microseconds(100)));
    }
    return perf::nowNs() - startNs;
  }

  void DxvkSubmissionQueue::recordRemixPresent(const vk::Presenter& presenter, uint64_t callNs, double callQpcMs, uint64_t gateWaitNs,
                                               uint32_t framesInFlight, uint32_t cmdListsInFlight, uint32_t submitQueueLen,
                                               uint64_t frameSeq, bool gpuDoneAtPresent, uint64_t reflexFrameId, float reflexSleepMs) {
    PresentPacingDiag& d = m_pacing;
    const uint32_t interval = perf::g_presentPacingLogInterval.load(std::memory_order_relaxed);

    // The always-on [Perf] window. Capped, so it cannot grow while nothing takes it.
    {
      std::lock_guard<dxvk::mutex> lock(m_presentWindowMutex);
      PresentWindow& w = m_presentWindow;
      if (w.presents < 16384) {
        if (m_presentWindowLastNs != 0 && callNs > m_presentWindowLastNs) {
          w.intervalMs.push_back(float(double(callNs - m_presentWindowLastNs) * 1e-6));
        }
        w.acquireMs.push_back(float(double(presenter.lastPostPresentAcquireNs()) * 1e-6));
        w.gateWaitMs.push_back(float(double(gateWaitNs) * 1e-6));
        w.framesInFlightSum += framesInFlight;
        w.presents += 1;
      }
      m_presentWindowLastNs = callNs;
    }

    // Independent of the [PresentPacing] interval.
    if (!d.loggedRuntimeModules && interval != 0) {
      d.loggedRuntimeModules = true;
      logPresentRuntimeModules();
    }
    {
      PresentPacingDiag::TraceCall call;
      call.frameSeq = frameSeq;
      call.callQpcMs = callQpcMs;
      call.queuePresentMs = float(double(presenter.lastQueuePresentNs()) * 1e-6);
      call.gateWaitMs = float(double(gateWaitNs) * 1e-6);
      call.framesInFlight = framesInFlight;
      call.cmdListsInFlight = cmdListsInFlight;
      call.gpuDoneAtPresent = gpuDoneAtPresent;
      call.reflexFrameId = reflexFrameId;
      call.reflexSleepMs = reflexSleepMs;
      // This present's id, and the wait this frame did before its RT work.
      call.presentId = presenter.lastAttachedPresentId();
      call.presentWaitId = m_presentWait.frameWaitId;
      call.presentWaitMs = m_presentWait.frameWaitMs;
      call.presentDoneQpcMs = m_presentWait.frameDoneQpcMs;
      tracePresent(call);
    }
    if (m_presentWait.frameWaitMs >= 0.0f && interval != 0) {
      d.presentWaitMs.push_back(m_presentWait.frameWaitMs);
    }
    m_presentWait.frameWaitId = 0;
    m_presentWait.frameWaitMs = -1.0f;
    m_presentWait.frameDoneQpcMs = -1.0;

    if (interval == 0) {
      d.lastCallNs = 0;
      d.presents = 0;
      return;
    }

    if (d.lastCallNs != 0 && callNs > d.lastCallNs) {
      const float ms = float(double(callNs - d.lastCallNs) * 1e-6);
      d.callIntervalMs.push_back(ms);
      float lateMs = 0.f;
      if (d.callBursts.push(ms, &lateMs)) {
        d.callBurstCount += 1;
        if (d.burstLinesLogged < 3) {
          d.burstLinesLogged += 1;
          const BurstDetector& b = d.callBursts;
          std::string recent;
          for (uint32_t i = 8; i > 0; --i) {
            recent += (i == 8 ? "" : " ") + pacingMs(b.ring[(b.count - i) % 16]);
          }
          Logger::warn(str::format("[PresentBurst] vkQueuePresentKHR intervals (ms, oldest first): ", recent,
            " | one ", pacingMs(lateMs), " ms gap then 3 fast presents, as the submission thread saw it",
            " | at this present: GPU frames in flight ", framesInFlight, ", command lists in flight ", cmdListsInFlight,
            ", queued submissions ", submitQueueLen, ", gate wait ", pacingMs(float(double(gateWaitNs) * 1e-6)), " ms",
            ", vkQueuePresentKHR ", pacingMs(float(double(presenter.lastQueuePresentNs()) * 1e-6)), " ms",
            ", next acquire ", pacingMs(float(double(presenter.lastPostPresentAcquireNs()) * 1e-6)), " ms"));
        }
      }
    }
    d.lastCallNs = callNs;
    d.queuePresentMs.push_back(float(double(presenter.lastQueuePresentNs()) * 1e-6));
    d.acquireMs.push_back(float(double(presenter.lastPostPresentAcquireNs()) * 1e-6));
    d.gateWaitMs.push_back(float(double(gateWaitNs) * 1e-6));
    d.framesInFlight.push_back(float(framesInFlight));
    d.cmdListsInFlight.push_back(float(cmdListsInFlight));
    d.submitQueueLen.push_back(float(submitQueueLen));

    d.presents += 1;
    if (d.presents < interval) {
      return;
    }

    std::vector<float> doneMs;
    uint32_t doneBursts = 0;
    {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      doneMs.swap(m_frameDoneIntervalsMs);
      doneBursts = std::exchange(m_frameDoneBurstCount, 0u);
    }

    const auto p50max = [](std::vector<float>& v) {
      if (v.empty()) {
        return std::string("-");
      }
      std::sort(v.begin(), v.end());
      return str::format(uint32_t(v[v.size() / 2]), "/", uint32_t(v.back()));
    };

    Logger::info(str::format("[PresentPacing] remixPresents=", d.presents,
      " gpuQueueLimit=", perf::g_presentGpuQueueLimit.load(std::memory_order_relaxed),
      " gateMaxWaitMs=", pacingMs(float(perf::g_presentGateMaxWaitUs.load(std::memory_order_relaxed)) * 1e-3f),
      " | vkQueuePresentKHR call interval ms p50/p95/p99/max=", pacingPct(d.callIntervalMs), " bursts=", d.callBurstCount,
      " | GPU-done interval of presented frames ms=", pacingPct(doneMs), " bursts=", doneBursts,
      " | vkQueuePresentKHR ms=", pacingPct(d.queuePresentMs),
      " | acquire after present ms=", pacingPct(d.acquireMs),
      " | gate wait ms=", pacingPct(d.gateWaitMs), " timeouts=", d.gateTimeouts,
      " | at present (p50/max): GPU frames in flight=", p50max(d.framesInFlight),
      " command lists in flight=", p50max(d.cmdListsInFlight),
      " queued submissions=", p50max(d.submitQueueLen),
      " | presentWaitPacing=", perf::g_presentWaitPacing.load(std::memory_order_relaxed),
      " lag=", perf::g_presentWaitLag.load(std::memory_order_relaxed),
      " waits=", uint32_t(d.presentWaitMs.size()), " wait ms=", pacingPct(d.presentWaitMs),
      " timeouts=", m_presentWait.timeoutsInInterval, m_presentWait.suspended ? " (suspended)" : "",
      " preGateLists=", m_presentWait.preGateListsInInterval));

    d.presentWaitMs.clear();
    m_presentWait.timeoutsInInterval = 0;
    m_presentWait.preGateListsInInterval = 0;
    d.presents = 0;
    d.callBurstCount = 0;
    d.burstLinesLogged = 0;
    d.gateTimeouts = 0;
    d.callIntervalMs.clear();
    d.queuePresentMs.clear();
    d.acquireMs.clear();
    d.gateWaitMs.clear();
    d.framesInFlight.clear();
    d.cmdListsInFlight.clear();
    d.submitQueueLen.clear();
  }

  DxvkSubmissionQueue::PresentWindow DxvkSubmissionQueue::takePresentWindow() {
    std::lock_guard<dxvk::mutex> lock(m_presentWindowMutex);
    return std::exchange(m_presentWindow, PresentWindow());
  }

  void DxvkSubmissionQueue::logPresentRuntimeModules() {
    const auto loaded = [](const char* name) { return ::GetModuleHandleA(name) != nullptr ? "yes" : "no"; };
    const std::string loaderDisable = env::getEnvVar("VK_LOADER_LAYERS_DISABLE");
    Logger::info(str::format("[PresentPacing] at the first Remix present, modules in this process:",
      " NvPresent64.dll=", loaded("NvPresent64.dll"), " (NVIDIA present layer runtime: loaded means presentation goes through",
      " the layer's DXGI swapchain, whatever the loader lists)",
      ", d3d12.dll=", loaded("d3d12.dll"),
      ", nvspcap64.dll=", loaded("nvspcap64.dll"), " (NVIDIA overlay)",
      ", GameOverlayRenderer64.dll=", loaded("GameOverlayRenderer64.dll"), " (Steam overlay)",
      ", NvLowLatencyVk.dll=", loaded("NvLowLatencyVk.dll"), " (Reflex)",
      " | DISABLE_LAYER_NV_PRESENT_1=", env::getEnvVar("DISABLE_LAYER_NV_PRESENT_1").empty() ? "unset" : "set",
      " VK_LOADER_LAYERS_DISABLE=", loaderDisable.empty() ? "unset" : loaderDisable));
  }

  void DxvkSubmissionQueue::tracePresent(const PresentPacingDiag::TraceCall& call) {
    PresentPacingDiag& d = m_pacing;
    if (!perf::g_presentTraceCsv.load(std::memory_order_relaxed)) {
      if (d.traceFile != nullptr) {
        flushPresentTrace(true);
      }
      d.traceOpenFailed = false;
      return;
    }

    if (d.traceFile == nullptr) {
      if (d.traceOpenFailed) {
        return;
      }
      const auto dir = util::RtxFileSys::path(util::RtxFileSys::Logs);
      const std::string file = dir.empty() ? std::string("remix-present-trace.csv") : (dir / "remix-present-trace.csv").string();
      d.traceFile = ::_wfopen(str::tows(file.c_str()).c_str(), L"w");
      if (d.traceFile == nullptr) {
        d.traceOpenFailed = true;
        Logger::warn(str::format("[PresentPacing] rtx.present.traceCsv: cannot open ", file));
        return;
      }
      d.tracePresents = 0;
      d.traceCalls.clear();
      {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        m_traceDone.clear();
      }
      std::fprintf(d.traceFile, "present,frameSeq,callQpcMs,gpuDoneQpcMs,gpuDoneAfterCallMs,queuePresentMs,gateWaitMs,framesInFlight,cmdListsInFlight,gpuDoneAtPresent,reflexFrameId,reflexSleepMs,presentId,presentWaitMs,presentDoneQpcMs,presentWaitId\n");
      Logger::info(str::format("[PresentPacing] rtx.present.traceCsv: writing one row per Remix present to ", file,
        " (callQpcMs / gpuDoneQpcMs are QueryPerformanceCounter ms, PresentMon --qpc_time_ms clock)"));
    }

    d.traceCalls.push_back(call);
    // A frame the GPU never finishes (device loss) must not pin memory.
    while (d.traceCalls.size() > 1024) {
      d.traceCalls.pop_front();
    }
    if (d.traceCalls.size() >= 32) {
      flushPresentTrace(false);
    }
  }

  void DxvkSubmissionQueue::flushPresentTrace(bool closeFile) {
    PresentPacingDiag& d = m_pacing;
    if (d.traceFile == nullptr) {
      return;
    }

    std::vector<std::pair<uint64_t, double>> done;
    {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      done.swap(m_traceDone);
    }

    // Both lists are in frame-seq order: walk them together.
    size_t di = 0;
    while (!d.traceCalls.empty()) {
      const PresentPacingDiag::TraceCall& c = d.traceCalls.front();
      double doneQpcMs = 0.0;
      bool haveDone = false;
      if (c.gpuDoneAtPresent) {
        doneQpcMs = c.callQpcMs;
        haveDone = true;
      } else {
        while (di < done.size() && done[di].first < c.frameSeq) {
          ++di; // completed before the trace started, or belongs to a dropped row
        }
        if (di < done.size() && done[di].first == c.frameSeq) {
          doneQpcMs = done[di].second;
          haveDone = true;
          ++di;
        }
      }
      if (!haveDone) {
        break; // still on the GPU; the next flush picks it up
      }
      std::fprintf(d.traceFile, "%llu,%llu,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%d,%llu,%.3f,%llu,%.3f,%.3f,%llu\n",
        static_cast<unsigned long long>(d.tracePresents), static_cast<unsigned long long>(c.frameSeq),
        c.callQpcMs, doneQpcMs, doneQpcMs - c.callQpcMs, double(c.queuePresentMs), double(c.gateWaitMs),
        c.framesInFlight, c.cmdListsInFlight, c.gpuDoneAtPresent ? 1 : 0,
        static_cast<unsigned long long>(c.reflexFrameId), double(c.reflexSleepMs),
        static_cast<unsigned long long>(c.presentId), double(c.presentWaitMs), c.presentDoneQpcMs,
        static_cast<unsigned long long>(c.presentWaitId));
      d.tracePresents += 1;
      d.traceCalls.pop_front();
    }

    // Completions that arrived ahead of their rows go back to the front, before anything newer.
    if (di < done.size()) {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      m_traceDone.insert(m_traceDone.begin(), done.begin() + di, done.end());
    }
    std::fflush(d.traceFile);

    if (closeFile) {
      std::fclose(d.traceFile);
      d.traceFile = nullptr;
      d.traceCalls.clear();
      Logger::info(str::format("[PresentPacing] rtx.present.traceCsv: closed remix-present-trace.csv after ", d.tracePresents, " rows"));
    }
  }

  // rtx.present.presentWaitPacing.
  // Runs on the submission thread, before the queue lock, for the first command list after a Remix present
  // that holds ray tracing work. Blocking here is safe: the present being waited for was already handed to
  // vkQueuePresentKHR by this thread and its GPU work was submitted before it, so nothing it depends on sits
  // behind this wait. The CS thread keeps recording (and blocks only on the queued-submission limit), the
  // finish thread keeps retiring command lists, and the wait is bounded by presentWaitTimeoutMs.
  void DxvkSubmissionQueue::waitForPresentBeforeRtWork() {
    PresentWaitState& w = m_presentWait;
    w.armed = false;

    if (perf::g_presentWaitPacing.load(std::memory_order_relaxed) == 0) {
      w.consecutiveTimeouts = 0;
      w.suspended = false; // toggling the option off and on re-enables after a timeout suspension
      return;
    }
    const Rc<vk::Presenter> presenter = w.presenter;
    if (presenter == nullptr) {
      return;
    }
    if (!presenter->presentWaitSupported()) {
      if (!w.loggedUnsupported) {
        w.loggedUnsupported = true;
        Logger::warn("[PresentWait] rtx.present.presentWaitPacing is on, but the Remix presenter cannot wait for presents"
          " (VK_KHR_present_id / VK_KHR_present_wait not enabled on the device, or not the D3D11 host swapchain): no-op");
      }
      return;
    }
    if (w.suspended) {
      if (presenter->swapchainGeneration() == w.suspendedGeneration) {
        return;
      }
      w.suspended = false;
      w.consecutiveTimeouts = 0;
      Logger::info("[PresentWait] swapchain recreated: present waits resumed");
    }

    const uint64_t last = presenter->lastPresentId();
    const uint32_t lag = std::min(perf::g_presentWaitLag.load(std::memory_order_relaxed), 8u);
    if (last <= lag) {
      return; // first presents after enabling or after a swapchain recreate
    }
    const uint64_t target = last - lag;
    const uint64_t timeoutNs = uint64_t(std::clamp(perf::g_presentWaitTimeoutUs.load(std::memory_order_relaxed), 1000u, 1000000u)) * 1000ull;

    const uint64_t startNs = perf::nowNs();
    const VkResult vr = presenter->waitForPresent(target, timeoutNs);
    const uint64_t waitNs = perf::nowNs() - startNs;
    if (vr == VK_NOT_READY) {
      return; // id no longer belongs to the current swapchain
    }

    w.frameWaitId = target;
    w.frameWaitMs = float(double(waitNs) * 1e-6);
    w.frameDoneQpcMs = (vr == VK_SUCCESS || vr == VK_SUBOPTIMAL_KHR) ? qpcMs() : -1.0;

    if (!w.loggedFirstWait) {
      w.loggedFirstWait = true;
      Logger::info(str::format("[PresentWait] first wait: present id ", target, " (last ", last, ", lag ", lag, "), ",
        pacingMs(w.frameWaitMs), " ms, result ", vr));
    }

    if (vr == VK_TIMEOUT) {
      w.timeoutsInInterval += 1;
      w.consecutiveTimeouts += 1;
      if (w.consecutiveTimeouts >= 16) {
        w.suspended = true;
        w.suspendedGeneration = presenter->swapchainGeneration();
        Logger::warn(str::format("[PresentWait] ", w.consecutiveTimeouts, " consecutive vkWaitForPresentKHR timeouts (",
          pacingMs(float(double(timeoutNs) * 1e-6)), " ms each): present waits suspended until the swapchain is recreated",
          " or rtx.present.presentWaitPacing is toggled"));
      }
    } else {
      w.consecutiveTimeouts = 0;
      if (vr != VK_SUCCESS && vr != VK_SUBOPTIMAL_KHR && !w.loggedError) {
        w.loggedError = true;
        Logger::warn(str::format("[PresentWait] vkWaitForPresentKHR returned ", vr, " (logged once); continuing"));
      }
    }
  }
  // NV-DXVK end

  void DxvkSubmissionQueue::submitCmdLists() {
    env::setThreadName("dxvk-submit");

    std::unique_lock<dxvk::mutex> lock(m_mutex);

    while (!m_stopped.load()) {
      m_appendCond.wait(lock, [this] {
        return m_stopped.load() || !m_submitQueue.empty();
      });
      
      if (m_stopped.load())
        return;

      ScopedCpuProfileZone();

      DxvkSubmitEntry entry = std::move(m_submitQueue.front());
      lock.unlock();
      
      // Submit command buffer to device
      VkResult status = VK_NOT_READY;

      // NV-DXVK start: present pacing
      // Before taking the queue lock: the gate may wait for the GPU.
      const bool isRemixPresentEntry = entry.submit.cmdList == nullptr && !entry.frameInterpolation.valid()
        && entry.present.presenter != nullptr && entry.present.isRemixPresent;
      uint64_t pacingGateWaitNs = 0;
      uint32_t pacingFramesInFlight = 0, pacingCmdListsInFlight = 0, pacingSubmitQueueLen = 0;
      uint64_t pacingFrameSeq = 0;
      bool pacingGpuDoneAtPresent = false;
      if (isRemixPresentEntry && m_lastError != VK_ERROR_DEVICE_LOST) {
        pacingGateWaitNs = waitRemixPresentGate(pacingFramesInFlight, pacingCmdListsInFlight, pacingSubmitQueueLen,
                                                pacingFrameSeq, pacingGpuDoneAtPresent);
      }
      // NV-DXVK end

      // rtx.present.presentWaitPacing. Also before the queue lock (other components
      // take it through lockDeviceQueue) and before the Reflex RENDERSUBMIT_START marker of this command list.
      if (m_presentWait.armed && entry.submit.cmdList != nullptr && entry.submit.cmdList->perfInfo().injectRtx
       && m_lastError != VK_ERROR_DEVICE_LOST) {
        waitForPresentBeforeRtWork();
      }
      // A pre-gate list passes an armed gate without waiting; count it for [PresentPacing].
      if (m_presentWait.armed && entry.submit.cmdList != nullptr && entry.submit.cmdList->perfInfo().preGate) {
        m_presentWait.preGateListsInInterval += 1;
      }

      if (m_lastError != VK_ERROR_DEVICE_LOST) {
        // NV-DXVK start: Rename lock to lockQueue to avoid shadowing other mutex
        std::lock_guard<dxvk::mutex> lockQueue(m_mutexQueue);
        // NV-DXVK end

          // NV-DXVK start: Reflex render submit
        const auto& reflex = m_device->getCommon()->metaReflex();
        // NV-DXVK end

        if (entry.submit.cmdList != nullptr) {
          // When using Reflex with Remix, we need to wrap the queue submit for the injectRTX rendering
          // work with the reflex render_submit markers.  This is because in Remix we essentially
          // have one large cmd list of work (inject rtx) and we want the Reflex timing to prioritize
          // this work for best latency reduction while minimizing performance impact.  So we tag the submit
          // upstream (RtxContext) which contains the injectRTX call as the one we want to wrap with Reflex markers.
          // NV-DXVK start: Reflex render submit
          if (entry.submit.insertReflexRenderMarkers) {
            reflex.beginRendering(entry.submit.cachedReflexFrameId);
          }

          status = entry.submit.cmdList->submit(
            entry.submit.waitSync,
            entry.submit.wakeSync);

          if (entry.submit.insertReflexRenderMarkers) {
            reflex.endRendering(entry.submit.cachedReflexFrameId);
          }
          // NV-DXVK end
        }
        // NV-DXVK start: DLFG integration
        else if (entry.frameInterpolation.valid()) {
          // stash frame interpolation data for next present call
          m_currentFrameInterpolationData = entry.frameInterpolation;
        }
        else if (entry.present.presenter != nullptr) {
          // NV-DXVK start: Raster presents must not replace the async RTX presenter
          if (entry.present.isRemixPresent) {
            m_lastPresenter = entry.present.presenter;
          }
          // NV-DXVK end

          // NV-DXVK start: Reflex present start
          const auto insertReflexPresentMarkers = entry.present.insertReflexPresentMarkers;
          const auto cachedReflexFrameId = entry.present.cachedReflexFrameId;

          // Note: Only insert Reflex Present markers around the Presenter's present call if requested.
          if (insertReflexPresentMarkers) {
            reflex.beginPresentation(cachedReflexFrameId);
          }
          // NV-DXVK end

          // NV-DXVK start: DLFG acquired image information retrieval
          const auto cachedAcquiredImageIndex = entry.present.cachedAcquiredImageIndex;
          // NV-DXVK end

          // m_device->vkd()->vkQueueWaitIdle(m_device->queues().graphics.queueHandle);
          // NV-DXVK start: Only a Remix present consumes queued interpolation data
          const DxvkFrameInterpolationInfo noInterpolation {};
          const auto& interpolation = entry.present.isRemixPresent
            ? m_currentFrameInterpolationData : noInterpolation;
          // NV-DXVK start: present pacing diagnostics
          const uint64_t pacingCallNs = perf::nowNs();
          const double pacingCallQpcMs = isRemixPresentEntry ? qpcMs() : 0.0;
          // NV-DXVK end
          status = entry.present.presenter->presentImage(&entry.status->result, entry.present, interpolation, cachedAcquiredImageIndex);
          // NV-DXVK start: present pacing diagnostics
          if (isRemixPresentEntry && (status == VK_SUCCESS || status == VK_SUBOPTIMAL_KHR)) {
            recordRemixPresent(*entry.present.presenter, pacingCallNs, pacingCallQpcMs, pacingGateWaitNs,
              pacingFramesInFlight, pacingCmdListsInFlight, pacingSubmitQueueLen,
              pacingFrameSeq, pacingGpuDoneAtPresent,
              entry.present.cachedReflexFrameId, entry.present.reflexSleepMs);
            // The next frame's first RT command list may wait for this present.
            m_presentWait.presenter = entry.present.presenter;
            m_presentWait.armed = true;
          }
          // NV-DXVK end
          // if both submit and DLFG+present run on the same queue, then we need to wait for present to avoid racing on the queue
#if __DLFG_USE_GRAPHICS_QUEUE
          entry.present.presenter->synchronize();
#endif

          // NV-DXVK start: Reflex present end
          // Note: Only insert Reflex Present markers around the Presenter's present call if requested.
          if (insertReflexPresentMarkers) {
            reflex.endPresentation(cachedReflexFrameId);
          }
          // NV-DXVK end

          // NV-DXVK start: Preserve Remix data across interleaved raster presents
          if (entry.present.isRemixPresent) {
            m_currentFrameInterpolationData.reset();
          }
          // NV-DXVK end

          const auto presentThrottleDelay = m_device->config().presentThrottleDelay;

          if (presentThrottleDelay > 0) {
            ScopedCpuProfileZoneN("Present Throttle Delay Sleep");

            Sleep(presentThrottleDelay);
          }

          // NV-DXVK start: Nsight Graphics capture
          if (status == VK_SUCCESS) {
            NsightGraphicsCapture::signalFrameBoundary(m_device->queues().graphics.queueHandle);
            NsightGraphicsCapture::processPendingCaptureRequest();
          }
          // NV-DXVK end
        }
      } else {
        // Don't submit anything after device loss
        // so that drivers get a chance to recover
        status = VK_ERROR_DEVICE_LOST;
      }

      if (entry.status)
        // NV-DXVK start: DLFG integration
        // if we queued for interpolation, then don't touch the output status here; DLFG presenter thread will update it (and may have already done so)
        if (status != VK_EVENT_SET) {
          entry.status->result = status;
        }
        // NV-DXVK end

      // On success, pass it on to the queue thread
      lock = std::unique_lock<dxvk::mutex>(m_mutex);

      if (status == VK_SUCCESS) {
        if (entry.submit.cmdList != nullptr) {
          m_finishQueue.push(std::move(entry));
          // NV-DXVK start: present pacing -- the finish thread completes these in the same order
          m_submittedSeq += 1;
          // NV-DXVK end
        }
      } else if (status == VK_ERROR_DEVICE_LOST || entry.submit.cmdList != nullptr) {
        Logger::err(str::format("DxvkSubmissionQueue: Command submission failed: ", status));
        m_lastError = status;

        // NV-DXVK start: GPU crash diagnostics and handling on device loss
        // Aftermath only produces a GPU crash dump for device loss. Other
        // submission failures retain their existing handling without a stall.
        if (status == VK_ERROR_DEVICE_LOST) {
          onGpuCrash("queue submission");
          handleDeviceLost();
        } else {
          waitForAftermathDump();
          m_device->waitForIdle();
        }
        // NV-DXVK end
      }

      m_submitQueue.pop();
      m_submitCond.notify_all();
    }
  }
  
  
  void DxvkSubmissionQueue::finishCmdLists() {
    env::setThreadName("dxvk-queue");

    while (!m_stopped.load()) {
      std::unique_lock<dxvk::mutex> lock(m_mutex);

      if (m_finishQueue.empty()) {
        auto t0 = dxvk::high_resolution_clock::now();

        m_submitCond.wait(lock, [this] {
          return m_stopped.load() || !m_finishQueue.empty();
        });

        auto t1 = dxvk::high_resolution_clock::now();
        m_gpuIdle += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      }

      if (m_stopped.load())
        return;

      ScopedCpuProfileZone();
      
      DxvkSubmitEntry entry = std::move(m_finishQueue.front());
      lock.unlock();
      
      VkResult status = m_lastError.load();
      
      if (status != VK_ERROR_DEVICE_LOST)
        status = entry.submit.cmdList->synchronize();
      
      if (status != VK_SUCCESS) {
        Logger::err(str::format("DxvkSubmissionQueue: Failed to sync fence: ", status));
        m_lastError = status;

        // NV-DXVK start: GPU crash diagnostics and handling on device loss
        if (status == VK_ERROR_DEVICE_LOST) {
          onGpuCrash("fence synchronization");
          handleDeviceLost();
        } else {
          m_device->waitForIdle();
        }
        // NV-DXVK end
      }

      // Release resources and signal events, then immediately wake
      // up any thread that's currently waiting on a resource in
      // order to reduce delays as much as possible.
      entry.submit.cmdList->notifyObjects();

      // A zero pending count promises that command-list references have been released,
      // so the command list is recycled before finish waiters wake.
      entry.submit.cmdList->reset();
      m_device->recycleCommandList(entry.submit.cmdList);

      lock.lock();
      m_pending -= 1;

      m_finishQueue.pop();
      // NV-DXVK start: present pacing -- completion cadence of presented Remix frames
      m_completedSeq += 1;
      if (!m_presentFrameSeqs.empty() && m_presentFrameSeqs.front() <= m_completedSeq) {
        const uint64_t nowNs = perf::nowNs();
        // rtx.present.traceCsv
        const bool traceOn = perf::g_presentTraceCsv.load(std::memory_order_relaxed);
        const double doneQpcMs = traceOn ? qpcMs() : 0.0;
        while (!m_presentFrameSeqs.empty() && m_presentFrameSeqs.front() <= m_completedSeq) {
          const uint64_t doneSeq = m_presentFrameSeqs.front();
          m_presentFrameSeqs.pop_front();
          if (traceOn && m_traceDone.size() < 4096) {
            m_traceDone.emplace_back(doneSeq, doneQpcMs);
          }
          if (m_lastFrameDoneNs != 0) {
            const float ms = float(double(nowNs - m_lastFrameDoneNs) * 1e-6);
            if (m_frameDoneIntervalsMs.size() < 8192) {
              m_frameDoneIntervalsMs.push_back(ms);
            }
            float lateMs = 0.f;
            if (m_frameDoneBursts.push(ms, &lateMs)) {
              m_frameDoneBurstCount += 1;
            }
          }
          m_lastFrameDoneNs = nowNs;
        }
      }
      // NV-DXVK end
      m_finishCond.notify_all();
      lock.unlock();
    }
  }

  // NV-DXVK start: GPU crash handling (Aftermath dump wait + device-loss reporting)
  void DxvkSubmissionQueue::waitForAftermathDump() {
    if (!m_device->config().enableAftermath) {
      return;
    }

    // Stall the pending exception until Aftermath has finished writing (or hits some error).
    uint32_t counter = 0;
    GFSDK_Aftermath_CrashDump_Status aftermathStatus = GFSDK_Aftermath_CrashDump_Status_NotStarted;

    static const uint32_t kTimeoutPreventionLimit = 5000;

    while (counter < kTimeoutPreventionLimit) {
      GFSDK_Aftermath_GetCrashDumpStatus(&aftermathStatus);

      if (aftermathStatus == GFSDK_Aftermath_CrashDump_Status_Finished || aftermathStatus == GFSDK_Aftermath_CrashDump_Status_Unknown) {
        break; // Our dump was written
      }

      static const uint32_t kTimeoutPerTry = 100;
      Sleep(kTimeoutPerTry);
      counter += kTimeoutPerTry;
    }
  }

  void DxvkSubmissionQueue::handleDeviceLost() {
    // Both the submit and finish threads can observe device loss; ensure only the first one performs
    // crash handling. Any other thread blocks here until the process is terminated below.
    static std::atomic<bool> s_handlingDeviceLoss(false);
    if (s_handlingDeviceLoss.exchange(true)) {
      while (true) {
        Sleep(1000);
      }
    }

    // Notify the bridge client immediately so it can disable the Windows ghost dialog and
    // prepare the crash dialog window before the Aftermath wait below.
    BridgeMessageChannel::get().send(kGpuCrashNotifyMsgName, 0, 0);

    waitForAftermathDump();
    std::ofstream(gpuCrashMarkerPath(util::RtxFileSys::path(util::RtxFileSys::Logs))) << "GPU crash";
    sentry::showGpuCrashDialog();
    Logger::err("DxvkSubmissionQueue: Exiting after GPU device loss");
    // The GPU-crash exit code lets the bridge distinguish this from a CPU crash.
    env::killProcess(kRemixGpuCrashExitCode);
  }
  // NV-DXVK end
}
