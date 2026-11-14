// rtx.profile.detailedLogging: see rtx_perf_log.h for what each line holds.

#include "rtx_perf_log.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "rtx_context.h"
#include "rtx_scene_manager.h"
#include "rtx_accel_manager.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_fork_d3d9_alpha.h"
#include "rtx_fork_sccr_shell.h"
#include "rtx_light_manager.h"
#include "rtx_rtxdi_rayquery.h"
#include "rtx_options.h"
#include "../dxvk_device.h"
#include "../dxvk_cs.h"
#include "../../util/log/log.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    struct Pct {
      float p50 = 0.f, p95 = 0.f, p99 = 0.f, max = 0.f, avg = 0.f;
    };

    Pct percentiles(const std::vector<float>& values) {
      Pct r;
      if (values.empty()) {
        return r;
      }
      std::vector<float> v = values;
      std::sort(v.begin(), v.end());
      const auto at = [&](double q) { return v[std::min(v.size() - 1, size_t(q * double(v.size() - 1) + 0.5))]; };
      r.p50 = at(0.50);
      r.p95 = at(0.95);
      r.p99 = at(0.99);
      r.max = v.back();
      double sum = 0.0;
      for (float x : v) {
        sum += x;
      }
      r.avg = float(sum / double(v.size()));
      return r;
    }

    std::string f2(double v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", v);
      return buf;
    }

    // name=p50/p95/max (ms)
    std::string triple(const char* name, const std::vector<float>& v) {
      const Pct p = percentiles(v);
      return str::format(" ", name, "=", f2(p.p50), "/", f2(p.p95), "/", f2(p.max));
    }

    uint64_t threadCpu100ns() {
      FILETIME creationTime, exitTime, kernelTime, userTime;
      if (!GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime)) {
        return 0;
      }
      const auto toU64 = [](const FILETIME& t) { return (uint64_t(t.dwHighDateTime) << 32) | uint64_t(t.dwLowDateTime); };
      return toU64(kernelTime) + toU64(userTime);
    }

    constexpr uint32_t kApiCount = uint32_t(perf::Api::Count);
    constexpr uint32_t kD3D9CtrCount = uint32_t(perf::D3D9Ctr::Count);
  }

  void RtxPerfLog::syncGlobals() {
    const bool detailed = detailedLogging();
    perf::g_perfDetailed.store(detailed, std::memory_order_relaxed);
    perf::g_gpuStallWarnMs.store(uint32_t(std::max(0.f, gpuStallWarnSeconds()) * 1000.f), std::memory_order_relaxed);
    perf::g_gpuStallTimeoutMs.store(uint32_t(std::max(0.f, gpuStallTimeoutSeconds()) * 1000.f), std::memory_order_relaxed);
    perf::g_presentGpuQueueLimit.store(std::min(gpuQueueLimit(), 4u), std::memory_order_relaxed);
    perf::g_presentPacingLogInterval.store(pacingLogIntervalPresents(), std::memory_order_relaxed);
    perf::g_presentGateMaxWaitUs.store(uint32_t(std::clamp(gpuQueueLimitMaxWaitMs(), 0.1f, 200.f) * 1000.f), std::memory_order_relaxed);
    perf::g_presentTraceCsv.store(traceCsv(), std::memory_order_relaxed);
    perf::g_presentWaitPacing.store(std::min(presentWaitPacing(), 1u), std::memory_order_relaxed);
    perf::g_presentWaitLag.store(std::min(presentWaitLag(), 8u), std::memory_order_relaxed);
    perf::g_presentWaitTimeoutUs.store(uint32_t(std::clamp(presentWaitTimeoutMs(), 1.f, 1000.f) * 1000.f), std::memory_order_relaxed);
    g_rtxOptionLockFreeScalarReads.store(RtxOptions::lockFreeScalarOptionReads(), std::memory_order_relaxed);
    g_rtxOptionCountLocks.store(detailed, std::memory_order_relaxed);
  }

  uint16_t RtxPerfLog::labelIndex(const char* label) {
    for (size_t i = 0; i < m_labels.size(); ++i) {
      if (m_labels[i] == label || std::strcmp(m_labels[i], label) == 0) {
        return uint16_t(i);
      }
    }
    m_labels.push_back(label);
    return uint16_t(m_labels.size() - 1);
  }

  void RtxPerfLog::onGpuSample(uint32_t frameId, const char* const* labels, const double* ms, uint32_t count, double measuredMs) {
    if (!m_started) {
      return;
    }
    // Sum repeated labels (e.g. IndirectIntegration is marked twice) into one value per stage.
    std::vector<std::pair<uint16_t, float>> stages;
    stages.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint16_t idx = labelIndex(labels[i]);
      auto it = std::find_if(stages.begin(), stages.end(), [&](const auto& s) { return s.first == idx; });
      if (it == stages.end()) {
        stages.emplace_back(idx, float(ms[i]));
      } else {
        it->second += float(ms[i]);
      }
    }
    if (m_w.gpuStageMs.size() < m_labels.size()) {
      m_w.gpuStageMs.resize(m_labels.size());
    }
    // Stages absent from this sample count as 0 so percentiles stay per sampled frame.
    for (size_t idx = 0; idx < m_labels.size(); ++idx) {
      auto it = std::find_if(stages.begin(), stages.end(), [&](const auto& s) { return s.first == idx; });
      m_w.gpuStageMs[idx].push_back(it != stages.end() ? it->second : 0.f);
    }
    m_w.gpuMeasuredMs.push_back(float(measuredMs));
    m_w.gpuSamples++;

    FrameRec& r = m_ring[frameId % kRing];
    if (r.frameId == frameId) {
      r.gpuValid = true;
      r.gpuMeasuredMs = float(measuredMs);
      r.gpuStages = std::move(stages);
    }
  }

  void RtxPerfLog::readChanges(RtxContext& ctx, Changes& out) {
    SceneManager& scene = ctx.getSceneManager();
    uint64_t build[8];
    scene.getAccelManager().getCumulativeBuildCounts(build);
    const auto delta = [&](uint32_t i) { const uint64_t d = build[i] >= m_prevBuild[i] ? build[i] - m_prevBuild[i] : build[i]; m_prevBuild[i] = build[i]; return d; };
    out.mergedBuilt = uint32_t(delta(0));
    out.mergedRefit = uint32_t(delta(1));
    out.dynBuilt = uint32_t(delta(2));
    out.dynRefit = uint32_t(delta(3));
    out.addedInstances = uint32_t(delta(4));
    out.tlasBuilt = uint32_t(delta(5));
    out.primsBuilt = delta(6);
    out.surfaceBytes = delta(7);
    if (const OpacityMicromapManager* omm = scene.getOpacityMicromapManager()) {
      out.ommBakedKTris = omm->getMicroTrianglesBakedThisFrame() / 1000;
      out.ommBuiltKTris = omm->getMicroTrianglesBuiltThisFrame() / 1000;
    }
    uint64_t apiNs = 0;
    for (uint32_t i = 0; i < kApiCount; ++i) {
      const uint64_t calls = perf::g_api[i].calls.load(std::memory_order_relaxed);
      const uint64_t ns = perf::g_api[i].ns.load(std::memory_order_relaxed);
      const uint64_t items = perf::g_api[i].items.load(std::memory_order_relaxed);
      out.apiCalls[i] = uint32_t(calls - m_prevApi[i][0]);
      apiNs += ns - m_prevApi[i][1];
      out.apiItems[i] = uint32_t(items - m_prevApi[i][2]);
      m_prevApi[i][0] = calls;
      m_prevApi[i][1] = ns;
      m_prevApi[i][2] = items;
    }
    out.apiMs = double(apiNs) / 1e6;
  }

  void RtxPerfLog::onFrame(RtxContext& ctx, const CpuFrame& cpu) {
    const Rc<DxvkDevice>& device = ctx.getDevice();
    const uint32_t frameId = device->getCurrentFrameId();
    const uint64_t nowNs = perf::nowNs();
    const DxvkStatCounters counters = device->getStatCounters();
    const uint64_t gpuIdleUs = counters.getCtr(DxvkStatCounter::GpuIdleTicks);
    const uint64_t csExecUs = counters.getCtr(DxvkStatCounter::CsExecTicks);
    const uint64_t csIdleUs = counters.getCtr(DxvkStatCounter::CsIdleTicks);
    const uint64_t csSyncUs = counters.getCtr(DxvkStatCounter::CsSyncTicks);
    const uint64_t gpuSyncUs = counters.getCtr(DxvkStatCounter::GpuSyncTicks);
    const uint64_t latencyNs = perf::g_latencyWaitNs.load(std::memory_order_relaxed);

    const auto startWindow = [&]() {
      m_w = Window();
      m_w.firstFrameId = frameId;
      m_w.csThreadCpuStart = threadCpu100ns();
      for (uint32_t i = 0; i < kApiCount; ++i) {
        m_w.apiStart[i][0] = perf::g_api[i].calls.load(std::memory_order_relaxed);
        m_w.apiStart[i][1] = perf::g_api[i].ns.load(std::memory_order_relaxed);
        m_w.apiStart[i][2] = perf::g_api[i].items.load(std::memory_order_relaxed);
        perf::g_api[i].maxNs.store(0, std::memory_order_relaxed);
      }
      for (uint32_t i = 0; i < kD3D9CtrCount; ++i) {
        m_w.d3d9CtrStart[i] = perf::g_d3d9Ctr[i].load(std::memory_order_relaxed);
      }
      m_w.d3d9CpuNsStart = perf::g_d3d9ThreadCpuNs.load(std::memory_order_relaxed);
      m_w.d3d9FramesStart = perf::g_d3d9ThreadFrames.load(std::memory_order_relaxed);
      m_w.optionLocksStart = g_rtxOptionLockCount.load(std::memory_order_relaxed);
      m_w.stallCountStart = perf::g_gpuStallCount.load(std::memory_order_relaxed);
      perf::g_latencyWaitMaxNs.store(0, std::memory_order_relaxed);
      perf::g_gpuStallMaxWaitMs.store(0, std::memory_order_relaxed);
    };

    const auto snapshot = [&]() {
      m_lastFrameNs = nowNs;
      m_prevGpuIdleUs = gpuIdleUs;
      m_prevCsExecUs = csExecUs;
      m_prevCsIdleUs = csIdleUs;
      m_prevCsSyncUs = csSyncUs;
      m_prevGpuSyncUs = gpuSyncUs;
      m_prevLatencyNs = latencyNs;
    };

    if (m_started && frameId > m_lastFrameId + 2) {
      // Frames without ray tracing in between (menus, loading screens, logging switched off and on):
      // re-baseline the counters instead of booking the whole gap as one frame.
      m_lastFrameId = frameId;
      Changes discard;
      readChanges(ctx, discard);
      snapshot();
      return;
    }
    m_lastFrameId = frameId;

    if (!m_started) {
      m_started = true;
      Changes discard;
      readChanges(ctx, discard);
      snapshot();
      startWindow();
      Logger::info(str::format("[PerfFrame] detailed performance logging on: interval ", detailedLogIntervalFrames(),
        " frames, GPU sample every ", std::max(1u, detailedGpuSamplePeriod()), " frames, spike = max(", spikeThresholdMs(),
        " ms, ", spikeMedianMultiplier(), " x median), GPU stall warn ", gpuStallWarnSeconds(), " s, watchdog ", gpuStallTimeoutSeconds(), " s"));
      return;
    }

    const auto deltaMs = [](uint64_t cur, uint64_t prev) { return cur >= prev ? float(double(cur - prev) / 1000.0) : 0.f; };

    FrameRec& r = m_ring[frameId % kRing];
    r = FrameRec();
    r.frameId = frameId;
    r.frameMs = float(double(nowNs - m_lastFrameNs) / 1e6);
    r.gpuBusyMs = std::max(0.f, r.frameMs - deltaMs(gpuIdleUs, m_prevGpuIdleUs));
    r.csExecMs = deltaMs(csExecUs, m_prevCsExecUs);
    r.csIdleMs = deltaMs(csIdleUs, m_prevCsIdleUs);
    r.appOnCsMs = deltaMs(csSyncUs, m_prevCsSyncUs);
    r.cpuOnGpuMs = deltaMs(gpuSyncUs, m_prevGpuSyncUs);
    r.latencyWaitMs = latencyNs >= m_prevLatencyNs ? float(double(latencyNs - m_prevLatencyNs) / 1e6) : 0.f;
    r.cpu = cpu;
    readChanges(ctx, r.changes);
    snapshot();

    // Rolling median of the last 61 frames.
    m_recentMs[m_recentCount++ % m_recentMs.size()] = r.frameMs;
    const size_t n = std::min<size_t>(m_recentCount, m_recentMs.size());
    std::array<float, 61> tmp = m_recentMs;
    std::nth_element(tmp.begin(), tmp.begin() + n / 2, tmp.begin() + n);
    r.medianMs = tmp[n / 2];

    // Window accumulation
    Window& w = m_w;
    w.frames++;
    w.frameMs.push_back(r.frameMs);
    w.gpuBusyMs.push_back(r.gpuBusyMs);
    w.latencyWaitMs.push_back(r.latencyWaitMs);
    w.csExecMs.push_back(r.csExecMs);
    w.csIdleMs.push_back(r.csIdleMs);
    w.appOnCsMs.push_back(r.appOnCsMs);
    w.cpuOnGpuMs.push_back(r.cpuOnGpuMs);
    w.injectMs.push_back(float(cpu.injectMs));
    w.scenePrepMs.push_back(float(cpu.scenePrepMs));
    w.retainedMs.push_back(float(cpu.retainedMs));
    w.apiDrawMs.push_back(float(cpu.apiDrawMs));
    w.apiDrawGeometryMs.push_back(float(cpu.apiDrawGeometryMs));
    w.apiDrawInstanceMs.push_back(float(cpu.apiDrawInstanceMs));
    w.texUploadMs.push_back(float(cpu.texUploadMs));
    w.flushMs.push_back(float(cpu.flushMs));
    w.flushEndMs.push_back(float(cpu.flushEndMs));
    w.flushSubmitMs.push_back(float(cpu.flushSubmitMs));
    w.flushBeginMs.push_back(float(cpu.flushBeginMs));
    w.retainedFullMs.push_back(float(cpu.retainedFullMs));
    w.retainedFull += cpu.retainedFull;
    w.retainedPreserved += cpu.retainedPreserved;
    w.retainedSkipped += cpu.retainedSkipped;
    w.retainedSteady += cpu.retainedSteady;
    w.retainedPinned += cpu.retainedPinned;
    w.retainedV2Replays += cpu.retainedV2Replays;
    w.retainedMoved += cpu.retainedMoved;
    w.retainedAwake += cpu.retainedAwake;
    w.retainedReleased += cpu.retainedReleased;
    w.retainedV3Replays += cpu.retainedV3Replays;
    w.sceneDirect += cpu.sceneDirect;
    w.sceneLegacy += cpu.sceneLegacy;
    w.sceneCreated += cpu.sceneCreated;
    w.sceneSettled += cpu.sceneSettled;
    w.sceneOmmPending += cpu.sceneOmmPending;
    w.transientKept += cpu.transientKept;
    w.transientMoved += cpu.transientMoved;
    w.transientPosed += cpu.transientPosed;
    w.transientBuilt += cpu.transientBuilt;
    w.transientDeclined += cpu.transientDeclined;
    w.transientUnseenKept += cpu.transientUnseenKept;
    w.transientCollected += cpu.transientCollected;
    w.transientBuiltNew += cpu.transientBuiltNew;
    w.transientBuiltMovedFull += cpu.transientBuiltMovedFull;
    w.transientBuiltDescription += cpu.transientBuiltDescription;
    w.transientBuiltAssets += cpu.transientBuiltAssets;
    w.transientBuiltBlocked += cpu.transientBuiltBlocked;
    w.sceneServiced += cpu.sceneServiced;
    w.lightSlotsUploaded += cpu.lightSlotsUploaded;
    w.d3d9CommitMs.push_back(float(cpu.d3d9CommitMs));
    w.d3d9FutureWaitMs.push_back(float(cpu.d3d9FutureWaitMs));
    w.d3d9SkyTerrainMs.push_back(float(cpu.d3d9SkyTerrainMs));
    w.d3d9SubmitMs.push_back(float(cpu.d3d9SubmitMs));
    w.d3d9PreserveMs.push_back(float(cpu.d3d9PreserveMs));
    w.d3d9DynamicMs.push_back(float(cpu.d3d9DynamicMs));
    w.d3d9GeometryMs.push_back(float(cpu.d3d9GeometryMs));
    w.d3d9InstanceMs.push_back(float(cpu.d3d9InstanceMs));
    w.d3d9Commits += cpu.d3d9Commits;
    for (uint32_t i = 0; i < cpu.prepPhases && i < kMaxPrepPhases; ++i) {
      w.prepMs[i].push_back(float(cpu.prepMs[i]));
    }
    w.apiDraws += cpu.apiDraws;
    w.flushes += cpu.flushes;
    const Changes& c = r.changes;
    w.changes.mergedBuilt += c.mergedBuilt;
    w.changes.mergedRefit += c.mergedRefit;
    w.changes.dynBuilt += c.dynBuilt;
    w.changes.dynRefit += c.dynRefit;
    w.changes.addedInstances += c.addedInstances;
    w.changes.tlasBuilt += c.tlasBuilt;
    w.changes.primsBuilt += c.primsBuilt;
    w.changes.surfaceBytes += c.surfaceBytes;
    w.changes.ommBakedKTris += c.ommBakedKTris;
    w.changes.ommBuiltKTris += c.ommBuiltKTris;

    // An interval whose displacement settings changed part way mixes two configs (and its GPU samples,
    // read back 8+ frames late, lag the change): logInterval marks it so the A/B ladder drops it.
    {
      DispConfig d;
      d.mode = uint32_t(RtxOptions::Displacement::mode());
      d.shell = SccrShellManager::enable();
      d.shellMaxDistance = SccrShellManager::maxDistance();
      d.maxSteps = SccrShellManager::maxSteps();
      d.maxPrisms = SccrShellManager::maxPrisms();
      d.shellShadows = SccrShellManager::shadows();
      d.maxShadowPrisms = SccrShellManager::maxShadowPrisms();
      d.liveWeights = RtxOptions::Displacement::terrainLiveWeights();
      if (!w.dispSet) {
        w.disp = d;
        w.dispSet = true;
      } else if (!(d == w.disp)) {
        w.disp = d;
        w.dispChanged = true;
      }
    }

    // Spike detection (the first frames of a window have a thin median; need 15 frames of history).
    const float threshold = std::max(spikeThresholdMs(), spikeMedianMultiplier() > 0.f ? spikeMedianMultiplier() * r.medianMs : 0.f);
    if (m_recentCount > 15 && r.frameMs > threshold) {
      w.spikes++;
      if (w.spikesLogged < spikeMaxPerInterval()) {
        w.spikesLogged++;
        m_pendingSpikes.push_back(frameId);
      }
    }
    // Emit spikes once their GPU timestamps had time to come back (read 8+ frames late).
    for (auto it = m_pendingSpikes.begin(); it != m_pendingSpikes.end();) {
      if (frameId - *it >= 12 || frameId < *it) {
        const FrameRec& s = m_ring[*it % kRing];
        if (s.frameId == *it) {
          emitSpike(s, m_labels);
        }
        it = m_pendingSpikes.erase(it);
      } else {
        ++it;
      }
    }

    switch (costTestFrame()) {
    case CostTestAction::NotRunning:
      if (w.frames >= std::max(1u, detailedLogIntervalFrames())) {
        logInterval(ctx);
        startWindow();
      }
      break;
    case CostTestAction::Running:
      break;
    case CostTestAction::RestartWindow:
      startWindow();
      break;
    case CostTestAction::LogAndRestart:
      logInterval(ctx);
      if (m_ctSummaryPending) {
        m_ctSummaryPending = false;
        costTestLogSummary();
      }
      startWindow();
      break;
    }
  }

  // rtx.profile.sccrCostTest. Each step: apply its settings (deferred: they resolve at the end of this
  // frame), drop sccrCostTestSettleFrames frames (shell data rebuilds, GPU samples of the old settings read
  // back late), then measure sccrCostTestStepFrames frames as one log interval, labelled by a [SccrCostTest] line.
  namespace {
    constexpr const char* kCostTestStepNames[] = {
      "everything on", "shell self-shadows off", "shell tracing off" };
    // The A/B test's step names, written while it is not running (g_costTestProgress 0), read by the dev menu.
    char g_abStepNames[RtxPerfLog::kTestMaxSteps][96] = {};
    // 0: not running; else 1 + cycle * steps + step. g_costTestShape: steps in the low 8 bits, cycles above, and bit 31
    // for the A/B test.
    std::atomic<uint32_t> g_costTestProgress { 0 };
    std::atomic<uint32_t> g_costTestShape { 0 };
    constexpr uint32_t kShapeAb = 1u << 31;

    std::string trimmed(const std::string& s) {
      const size_t b = s.find_first_not_of(" \t");
      const size_t e = s.find_last_not_of(" \t");
      return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
    }

    void setOptionValue(RtxOptionImpl* option, const std::string& name, const std::string& value) {
      Config config;
      config.setOptionMove(std::string(name), std::string(value));
      std::lock_guard lock { RtxOptionImpl::getUpdateMutex() };
      option->readOption(config, RtxOptionLayer::getUserLayer());
    }

    float medianOf(std::vector<float> v) {
      if (v.empty()) {
        return 0.f;
      }
      std::sort(v.begin(), v.end());
      const size_t n = v.size();
      return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    }
  }

  bool RtxPerfLog::sccrCostTestProgress(uint32_t& cycle, uint32_t& cycles, uint32_t& step, uint32_t& steps,
                                        const char*& stepName) {
    const uint32_t p = g_costTestProgress.load(std::memory_order_relaxed);
    const uint32_t shape = g_costTestShape.load(std::memory_order_relaxed);
    steps = shape & 0xFFu;
    if (p == 0 || steps == 0) {
      return false;
    }
    cycles = (shape & ~kShapeAb) >> 8;
    cycle = (p - 1) / steps;
    step = (p - 1) % steps;
    stepName = (shape & kShapeAb) != 0 ? g_abStepNames[std::min(step, kTestMaxSteps - 1)] : kCostTestStepNames[step % kCostTestSteps];
    return true;
  }

  // sccrAbTestOption: "name" (a bool, False/True) or "name=A/B", separated by commas. False (logged) when an option is
  // unknown, a bare name is not a bool, or there are none or too many.
  bool RtxPerfLog::abParse(const std::string& list) {
    m_ctAbOptions.clear();
    size_t start = 0;
    while (start <= list.size()) {
      const size_t comma = std::min(list.find(',', start), list.size());
      const std::string item = trimmed(list.substr(start, comma - start));
      start = comma + 1;
      if (item.empty()) {
        continue;
      }
      AbOption o;
      const size_t eq = item.find('=');
      o.name = trimmed(item.substr(0, eq));
      RtxOptionImpl* option = RtxOptionImpl::getOptionByFullName(o.name);
      if (option == nullptr) {
        Logger::info(str::format("[SccrCostTest] not started: no option '", o.name, "'"));
        return false;
      }
      if (eq == std::string::npos) {
        if (option->getType() != OptionType::Bool) {
          Logger::info(str::format("[SccrCostTest] not started: '", o.name, "' is not a bool option (give name=A/B)"));
          return false;
        }
        o.a = "False";
        o.b = "True";
      } else {
        const std::string values = item.substr(eq + 1);
        const size_t slash = values.find('/');
        if (slash == std::string::npos) {
          Logger::info(str::format("[SccrCostTest] not started: '", item, "' needs name=A/B"));
          return false;
        }
        o.a = trimmed(values.substr(0, slash));
        o.b = trimmed(values.substr(slash + 1));
      }
      o.saved = option->getResolvedValueAsString();
      m_ctAbOptions.push_back(std::move(o));
    }
    if (m_ctAbOptions.empty() || m_ctAbOptions.size() > kAbTestMaxOptions) {
      Logger::info(str::format("[SccrCostTest] not started: sccrAbTestOption names ", m_ctAbOptions.size(), " options (1 to ",
                               kAbTestMaxOptions, ")"));
      return false;
    }
    const size_t n = m_ctAbOptions.size();
    for (uint32_t s = 0; s < steps(); ++s) {
      std::string name = n == 1 ? (s == 0 ? "A: " + m_ctAbOptions[0].a : "B: " + m_ctAbOptions[0].b)
                       : s == 0 ? std::string("A: every option at A")
                       : s == n + 1 ? std::string("all options at B")
                       : str::format("B: ", m_ctAbOptions[s - 1].name, " = ", m_ctAbOptions[s - 1].b);
      snprintf(g_abStepNames[s], sizeof(g_abStepNames[s]), "%s", name.c_str());
    }
    return true;
  }

  std::string RtxPerfLog::abSummary() const {
    std::string s;
    for (const AbOption& o : m_ctAbOptions) {
      s += str::format(s.empty() ? "" : ", ", o.name, " ", o.a, "/", o.b, " (was ", o.saved, ")");
    }
    return s;
  }

  uint32_t RtxPerfLog::cycles() const {
    return std::max(1u, m_ctAb ? sccrAbTestCycles() : sccrCostTestCycles());
  }

  void RtxPerfLog::costTestApplyStep(uint32_t step) {
    if (m_ctAb) {
      // The A/B runs on full shell tracing (the cost test's first step), whatever the menu had.
      SccrShellManager::enableObject().setDeferred(true);
      const uint32_t n = uint32_t(m_ctAbOptions.size());
      for (uint32_t i = 0; i < n; ++i) {
        const bool b = n == 1 ? step == 1 : (step == n + 1 || step == i + 1);
        if (RtxOptionImpl* option = RtxOptionImpl::getOptionByFullName(m_ctAbOptions[i].name)) {
          setOptionValue(option, m_ctAbOptions[i].name, b ? m_ctAbOptions[i].b : m_ctAbOptions[i].a);
        }
      }
    } else {
      SccrShellManager::enableObject().setDeferred(step < 2);
      SccrShellManager::shadowsObject().setDeferred(step < 1);
    }
    g_costTestShape.store(steps() | (cycles() << 8) | (m_ctAb ? kShapeAb : 0u), std::memory_order_relaxed);
    g_costTestProgress.store(1 + m_ctCycle * steps() + step, std::memory_order_relaxed);
  }

  void RtxPerfLog::costTestFinish(const char* why) {
    if (m_ctAb) {
      for (const AbOption& o : m_ctAbOptions) {
        if (RtxOptionImpl* option = RtxOptionImpl::getOptionByFullName(o.name)) {
          setOptionValue(option, o.name, o.saved);
        }
      }
    }
    SccrShellManager::enableObject().setDeferred(m_ctSavedShell);
    SccrShellManager::shadowsObject().setDeferred(m_ctSavedShadows);
    if (sccrCostTest()) {
      sccrCostTestObject().setDeferred(false);
    }
    if (sccrAbTest()) {
      sccrAbTestObject().setDeferred(false);
    }
    m_ctState = CostTestState::Idle;
    g_costTestProgress.store(0, std::memory_order_relaxed);
    if (why != nullptr) {
      Logger::info(str::format("[SccrCostTest] stopped (", why, ") in cycle ", m_ctCycle + 1, " step ", m_ctStep + 1,
                               "; shell settings restored", m_ctAb ? ", and the options' values" : ""));
    }
  }

  RtxPerfLog::CostTestAction RtxPerfLog::costTestFrame() {
    const bool wanted = m_ctState == CostTestState::Idle ? (sccrCostTest() || sccrAbTest())
                                                         : (m_ctAb ? sccrAbTest() : sccrCostTest());
    const bool sccr = RtxOptions::Displacement::mode() == DisplacementMode::SCCR;
    if (m_ctState == CostTestState::Idle) {
      if (!wanted) {
        return CostTestAction::NotRunning;
      }
      m_ctAb = !sccrCostTest();
      if (!sccr) {
        Logger::info("[SccrCostTest] not started: the displacement mode is not SCCR");
        sccrCostTestObject().setDeferred(false);
        sccrAbTestObject().setDeferred(false);
        return CostTestAction::NotRunning;
      }
      if (m_ctAb && !abParse(sccrAbTestOption())) {
        sccrAbTestObject().setDeferred(false);
        return CostTestAction::NotRunning;
      }
      m_ctSavedShell = SccrShellManager::enable();
      m_ctSavedShadows = SccrShellManager::shadows();
      for (uint32_t s = 0; s < kTestMaxSteps; ++s) {
        m_ctTotalP50[s].clear();
        m_ctStageP50[s].clear();
      }
      m_ctCycle = 0;
      m_ctStep = 0;
      m_ctFrames = 0;
      m_ctSummaryPending = false;
      m_ctState = CostTestState::Settle;
      costTestApplyStep(0);
      Logger::info(str::format("[SccrCostTest] start",
                               m_ctAb ? str::format(" A/B of ", abSummary()) : std::string(),
                               ": ", cycles(), " cycles x ", steps(), " steps, ", std::max(1u, sccrCostTestStepFrames()),
                               " measured frames each after ", sccrCostTestSettleFrames(), " settle frames"));
      return CostTestAction::Running;
    }
    if (!wanted || !sccr) {
      costTestFinish(!wanted ? "turned off" : "displacement mode left SCCR");
      return CostTestAction::RestartWindow;
    }
    ++m_ctFrames;
    if (m_ctState == CostTestState::Settle) {
      if (m_ctFrames < sccrCostTestSettleFrames()) {
        return CostTestAction::Running;
      }
      m_ctState = CostTestState::Measure;
      m_ctFrames = 0;
      return CostTestAction::RestartWindow;
    }
    if (m_ctFrames < std::max(1u, sccrCostTestStepFrames())) {
      return CostTestAction::Running;
    }

    // The step is measured: keep its p50s for the summary, label the interval logInterval writes next.
    const Window& w = m_w;
    m_ctTotalP50[m_ctStep].push_back(percentiles(w.gpuMeasuredMs).p50);
    std::vector<float> stages(w.gpuStageMs.size(), 0.f);
    for (size_t i = 0; i < w.gpuStageMs.size(); ++i) {
      stages[i] = percentiles(w.gpuStageMs[i]).p50;
    }
    m_ctStageP50[m_ctStep].push_back(std::move(stages));
    Logger::info(str::format("[SccrCostTest] cycle ", m_ctCycle + 1, "/", cycles(), " step ", m_ctStep + 1, "/", steps(),
                             " '", m_ctAb ? g_abStepNames[m_ctStep] : kCostTestStepNames[m_ctStep], "': ", w.frames,
                             " frames, ", w.gpuSamples, " GPU samples, Remix GPU p50 ", f2(m_ctTotalP50[m_ctStep].back()),
                             " ms (block below)"));
    if (++m_ctStep == steps()) {
      m_ctStep = 0;
      ++m_ctCycle;
    }
    if (m_ctCycle >= cycles()) {
      m_ctSummaryPending = true;
      costTestFinish(nullptr);
    } else {
      m_ctState = CostTestState::Settle;
      m_ctFrames = 0;
      costTestApplyStep(m_ctStep);
    }
    return CostTestAction::LogAndRestart;
  }

  void RtxPerfLog::costTestLogSummary() {
    const size_t cycles = m_ctTotalP50[0].size();
    // What each feature costs, per cycle (neighbouring steps of one cycle, so slow drift cancels), then the median and
    // the range over the cycles.
    const auto costOf = [&](uint32_t on, uint32_t off) {
      std::vector<float> d;
      for (size_t c = 0; c < cycles && c < m_ctTotalP50[off].size(); ++c) {
        d.push_back(m_ctTotalP50[on][c] - m_ctTotalP50[off][c]);
      }
      std::string s = f2(medianOf(d));
      if (!d.empty()) {
        s += str::format(" (", f2(*std::min_element(d.begin(), d.end())), " to ", f2(*std::max_element(d.begin(), d.end())), ")");
      }
      return s;
    };
    if (m_ctAb && m_ctAbOptions.size() <= 1) {
      Logger::info(str::format("[SccrCostTest] A/B done, ", cycles, " cycles of ", abSummary(),
        ": Remix GPU p50 (median over cycles) A ", f2(medianOf(m_ctTotalP50[0])), ", B ", f2(medianOf(m_ctTotalP50[1])),
        " ms; B - A per cycle, median (min to max): ", costOf(1, 0), " ms (negative: B is faster)"));
    } else if (m_ctAb) {
      const uint32_t n = uint32_t(m_ctAbOptions.size());
      Logger::info(str::format("[SccrCostTest] A/B done, ", cycles, " cycles of ", n, " options; Remix GPU p50 with every "
        "option at A (median over cycles): ", f2(medianOf(m_ctTotalP50[0])), " ms. Each at B minus every one at A, per "
        "cycle, median (min to max), negative: B is faster:"));
      for (uint32_t i = 0; i < n; ++i) {
        Logger::info(str::format("[SccrCostTest]   ", m_ctAbOptions[i].name, " ", m_ctAbOptions[i].a, " -> ",
                                 m_ctAbOptions[i].b, ": ", costOf(i + 1, 0), " ms"));
      }
      Logger::info(str::format("[SccrCostTest]   all options at B: ", costOf(n + 1, 0), " ms"));
    } else {
    Logger::info(str::format("[SccrCostTest] done, ", cycles, " cycles. Remix GPU p50 per step (median over cycles): ",
      "1 everything on ", f2(medianOf(m_ctTotalP50[0])), ", 2 shell self-shadows off ", f2(medianOf(m_ctTotalP50[1])),
      ", 3 shell off ", f2(medianOf(m_ctTotalP50[2])), " ms"));
    Logger::info(str::format("[SccrCostTest] cost in ms, median (min to max) over cycles: shell self-shadows ", costOf(0, 1),
      ", shell march ", costOf(1, 2), ", whole shell ", costOf(0, 2)));
    }

    // Per stage: the median p50 of each step, for stages whose steps differ by 0.05 ms or more, largest change first.
    size_t labels = 0;
    const uint32_t stepCount = steps();
    for (uint32_t s = 0; s < stepCount; ++s) {
      for (const auto& v : m_ctStageP50[s]) {
        labels = std::max(labels, v.size());
      }
    }
    struct Row { size_t idx; float step[kTestMaxSteps]; float spread; };
    std::vector<Row> rows;
    for (size_t i = 0; i < labels && i < m_labels.size(); ++i) {
      Row row { i, {}, 0.f };
      float lo = FLT_MAX, hi = -FLT_MAX;
      for (uint32_t s = 0; s < stepCount; ++s) {
        std::vector<float> v;
        for (const auto& cycle : m_ctStageP50[s]) {
          v.push_back(i < cycle.size() ? cycle[i] : 0.f);
        }
        row.step[s] = medianOf(v);
        lo = std::min(lo, row.step[s]);
        hi = std::max(hi, row.step[s]);
      }
      row.spread = hi - lo;
      if (row.spread >= 0.05f) {
        rows.push_back(row);
      }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.spread > b.spread; });
    std::string line = m_ctAb ? "[SccrCostTest] stages, ms per step (in step order):" : "[SccrCostTest] stages, steps 1/2/3/4/5 ms:";
    for (const Row& row : rows) {
      line += str::format(" ", m_labels[row.idx], "=", f2(row.step[0]));
      for (uint32_t s = 1; s < stepCount; ++s) {
        line += str::format("/", f2(row.step[s]));
      }
    }
    Logger::info(line);
  }

  void RtxPerfLog::emitSpike(const FrameRec& r, const std::vector<const char*>& labels) {
    std::string cpu = str::format(" | cpu: inject=", f2(r.cpu.injectMs), " scenePrep=", f2(r.cpu.scenePrepMs));
    for (uint32_t i = 0; i < r.cpu.prepPhases && i < kMaxPrepPhases; ++i) {
      if (r.cpu.prepMs[i] >= 0.5) {
        cpu += str::format(" prep.", SceneManager::kPrepPhaseNames[i], "=", f2(r.cpu.prepMs[i]));
      }
    }
    if (r.cpu.d3d9Commits != 0) {
      cpu += str::format(" d3d9Commit=", f2(r.cpu.d3d9CommitMs), "(", r.cpu.d3d9Commits, ") d3d9.futureWait=", f2(r.cpu.d3d9FutureWaitMs),
                         " d3d9.preserve=", f2(r.cpu.d3d9PreserveMs), " d3d9.dynamic=", f2(r.cpu.d3d9DynamicMs));
    }
    cpu += str::format(" retained=", f2(r.cpu.retainedMs), " apiDraws=", f2(r.cpu.apiDrawMs), "(", r.cpu.apiDraws, ")",
                       " texUpload=", f2(r.cpu.texUploadMs), " flush=", f2(r.cpu.flushMs), "(", r.cpu.flushes, ")",
                       " csExec=", f2(r.csExecMs), " csIdle=", f2(r.csIdleMs), " appOnCs=", f2(r.appOnCsMs),
                       " cpuOnGpu=", f2(r.cpuOnGpuMs), " latencyWait=", f2(r.latencyWaitMs), " hostApi=", f2(r.changes.apiMs));
    std::string gpu;
    if (r.gpuValid) {
      std::vector<std::pair<uint16_t, float>> sorted = r.gpuStages;
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
      gpu = str::format(" | gpu: measured=", f2(r.gpuMeasuredMs));
      for (size_t i = 0; i < sorted.size() && i < 10; ++i) {
        if (sorted[i].first < labels.size()) {
          gpu += str::format(" ", labels[sorted[i].first], "=", f2(sorted[i].second));
        }
      }
    } else {
      gpu = " | gpu: not sampled";
    }
    const Changes& c = r.changes;
    std::string changes = str::format(" | changes: mergedBlasBuilt=", c.mergedBuilt, " mergedBlasRefit=", c.mergedRefit,
      " primsBuilt=", c.primsBuilt, " perGeomBuilt=", c.dynBuilt, " perGeomRefit=", c.dynRefit, " addedInstances=", c.addedInstances,
      " tlasBuilt=", c.tlasBuilt, " surfaceKiB=", c.surfaceBytes >> 10, " ommBakedKTris=", c.ommBakedKTris, " ommBuiltKTris=", c.ommBuiltKTris);
    std::string api = " | api:";
    bool anyApi = false;
    for (uint32_t i = 0; i < kApiCount; ++i) {
      if (c.apiCalls[i] != 0) {
        anyApi = true;
        api += str::format(" ", perf::kApiNames[i], "=", c.apiCalls[i]);
        if (c.apiItems[i] != c.apiCalls[i]) {
          api += str::format("(", c.apiItems[i], " items)");
        }
      }
    }
    Logger::warn(str::format("[Spike] frame=", r.frameId, " ms=", f2(r.frameMs), " median=", f2(r.medianMs),
      " gpuBusyEst=", f2(r.gpuBusyMs), cpu, gpu, changes, anyApi ? api : std::string(" | api: none")));
  }

  void RtxPerfLog::logInterval(RtxContext& ctx) {
    Window& w = m_w;
    const double n = std::max(1.0, double(w.frames));
    const Pct frame = percentiles(w.frameMs);
    const Pct busy = percentiles(w.gpuBusyMs);
    const Pct measured = percentiles(w.gpuMeasuredMs);
    const Pct latency = percentiles(w.latencyWaitMs);
    double idleSum = 0.0, frameSum = 0.0;
    for (size_t i = 0; i < w.frameMs.size(); ++i) {
      frameSum += w.frameMs[i];
      idleSum += std::max(0.f, w.frameMs[i] - w.gpuBusyMs[i]);
    }
    const double idleFrac = frameSum > 0.0 ? idleSum / frameSum : 0.0;
    const char* verdict = idleFrac < 0.10 ? "GPU-bound" : (idleFrac > 0.30 ? "CPU/sync-bound" : "mixed");
    const uint32_t lastFrame = ctx.getDevice()->getCurrentFrameId();
    const uint32_t stalls = perf::g_gpuStallCount.load(std::memory_order_relaxed) - w.stallCountStart;

    Logger::info(str::format("[PerfFrame] frames=", w.frames, " ids=", w.firstFrameId, "-", lastFrame,
      " | frameMs p50=", f2(frame.p50), " p95=", f2(frame.p95), " p99=", f2(frame.p99), " max=", f2(frame.max),
      " avg=", f2(frame.avg), " fps=", f2(1000.0 / std::max(0.001f, frame.avg)),
      " | gpuBusyEstMs p50=", f2(busy.p50), " p95=", f2(busy.p95), " p99=", f2(busy.p99), " max=", f2(busy.max),
      " | gpuIdle=", f2(idleFrac * 100.0), "% verdict=", verdict,
      " | gpuRemixMs(", w.gpuSamples, " samples) p50=", f2(measured.p50), " p95=", f2(measured.p95), " p99=", f2(measured.p99),
      " max=", f2(measured.max),
      " | latencyWaitMs p50=", f2(latency.p50), " p95=", f2(latency.p95), " max=", f2(latency.max),
      " | spikes=", w.spikes, " logged=", w.spikesLogged,
      " | gpuStalls=", stalls, " longestFenceWaitMs=", perf::g_gpuStallMaxWaitMs.load(std::memory_order_relaxed)));

    // The settings at the end of the interval; mixed=1 when they changed during it.
    const DispConfig& d = w.disp;
    Logger::info(str::format("[PerfDisp] mixed=", w.dispChanged ? 1 : 0, " mode=", d.mode, " (2 Quadtree, 3 SCCR)",
      " shell=", d.shell ? 1 : 0, " shellShadows=", d.shellShadows ? 1 : 0, " liveWeights=", d.liveWeights ? 1 : 0,
      " maxDistance=", f2(d.shellMaxDistance), " maxSteps=", d.maxSteps, " maxPrisms=", d.maxPrisms,
      " maxShadowPrisms=", d.maxShadowPrisms));

    // GPU stages sorted by p50, all stages with p95 >= 0.05 ms.
    if (w.gpuSamples > 0) {
      struct Row { size_t idx; Pct p; };
      std::vector<Row> rows;
      for (size_t i = 0; i < w.gpuStageMs.size() && i < m_labels.size(); ++i) {
        if (w.gpuStageMs[i].empty()) {
          continue;
        }
        Row row { i, percentiles(w.gpuStageMs[i]) };
        if (row.p.p95 >= 0.05f) {
          rows.push_back(row);
        }
      }
      std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.p.avg > b.p.avg; });
      std::string s = str::format("[PerfGpu] samples=", w.gpuSamples, " (stage=p50/p95/max ms, sorted by avg)");
      for (const Row& row : rows) {
        s += str::format(" ", m_labels[row.idx], "=", f2(row.p.p50), "/", f2(row.p.p95), "/", f2(row.p.max));
      }
      Logger::info(s);
    }

    // CPU (Remix CS thread; injectRTX runs there together with the game's D3D11 command stream).
    const uint64_t csCpu = threadCpu100ns();
    const double csCpuMs = csCpu >= w.csThreadCpuStart ? double(csCpu - w.csThreadCpuStart) / 10000.0 / n : 0.0;
    std::string cpu = str::format("[PerfCpu] frames=", w.frames, " (name=p50/p95/max ms) | csThreadCpuMsPerFrame=", f2(csCpuMs),
      triple("inject", w.injectMs), triple("scenePrep", w.scenePrepMs));
    for (uint32_t i = 0; i < SceneManager::kPrepPhaseCount && i < kMaxPrepPhases; ++i) {
      cpu += triple(str::format("prep.", SceneManager::kPrepPhaseNames[i]).c_str(), w.prepMs[i]);
    }
    cpu += triple("retainedReplay", w.retainedMs) + triple("apiDraws", w.apiDrawMs) +
           triple("apiDraws.geometry", w.apiDrawGeometryMs) + triple("apiDraws.instance", w.apiDrawInstanceMs) +
           str::format(" apiDrawsPerFrame=", f2(double(w.apiDraws) / n)) +
           triple("texUpload", w.texUploadMs) + triple("flush", w.flushMs) +
           str::format(" flushesPerFrame=", f2(double(w.flushes) / n)) +
           triple("flush.end", w.flushEndMs) + triple("flush.submitWait", w.flushSubmitMs) + triple("flush.begin", w.flushBeginMs) +
           triple("retained.full", w.retainedFullMs) +
           triple("d3d9Commit", w.d3d9CommitMs) + str::format(" d3d9CommitsPerFrame=", f2(double(w.d3d9Commits) / n)) +
           triple("d3d9.futureWait", w.d3d9FutureWaitMs) + triple("d3d9.skyTerrain", w.d3d9SkyTerrainMs) +
           triple("d3d9.submit", w.d3d9SubmitMs) + triple("d3d9.preserve", w.d3d9PreserveMs) +
           triple("d3d9.dynamic", w.d3d9DynamicMs) + triple("d3d9.dynamic.geometry", w.d3d9GeometryMs) +
           triple("d3d9.dynamic.instance", w.d3d9InstanceMs) +
           str::format(" retainedPerFrame=full:", f2(double(w.retainedFull) / n), ",preserved:", f2(double(w.retainedPreserved) / n),
                       ",skipped:", f2(double(w.retainedSkipped) / n), ",steady:", f2(double(w.retainedSteady) / n),
                       ",pinned:", f2(double(w.retainedPinned) / n),
                       // rtx.retained.useV2: which implementation ran the window's replays, and its own counts.
                       // rtx.scene.directRetained (v3): the scene's direct lane with the first implementation as legacy.
                       ",impl:", w.retainedV3Replays * 2 >= w.frames && w.retainedV3Replays != 0 ? "v3"
                         : (w.retainedV2Replays == 0 && w.retainedV3Replays == 0 ? "v1"
                         : (w.retainedV2Replays * 2 >= w.frames ? "v2" : "mixed")),
                       ",moved:", f2(double(w.retainedMoved) / n), ",awake:", f2(double(w.retainedAwake) / n),
                       ",released:", f2(double(w.retainedReleased) / n),
                       ",direct:", f2(double(w.sceneDirect) / n), ",legacy:", f2(double(w.sceneLegacy) / n),
                       ",created:", f2(double(w.sceneCreated) / n), ",settled:", f2(double(w.sceneSettled) / n),
                       ",ommPending:", f2(double(w.sceneOmmPending) / n), ",serviced:", f2(double(w.sceneServiced) / n)) +
           str::format(" transientPerFrame=kept:", f2(double(w.transientKept) / n), ",moved:", f2(double(w.transientMoved) / n),
                       ",posed:", f2(double(w.transientPosed) / n), ",built:", f2(double(w.transientBuilt) / n),
                       ",declined:", f2(double(w.transientDeclined) / n), ",unseenKept:", f2(double(w.transientUnseenKept) / n),
                       ",collected:", f2(double(w.transientCollected) / n),
                       ",builtNew:", f2(double(w.transientBuiltNew) / n), ",builtMovedFull:", f2(double(w.transientBuiltMovedFull) / n),
                       ",builtDesc:", f2(double(w.transientBuiltDescription) / n), ",builtAssets:", f2(double(w.transientBuiltAssets) / n),
                       ",builtBlocked:", f2(double(w.transientBuiltBlocked) / n)) +
           str::format(" lightSlotsUploadedPerFrame=", f2(double(w.lightSlotsUploaded) / n)) +
           triple("csExecAll", w.csExecMs) + triple("csIdleAll", w.csIdleMs) + triple("appWaitOnCs", w.appOnCsMs) +
           triple("cpuWaitOnGpu", w.cpuOnGpuMs) + triple("latencyWait", w.latencyWaitMs);
    Logger::info(cpu);

    // Remix API calls by the host during the window.
    {
      std::string api = str::format("[PerfApi] frames=", w.frames, " (name=calls/frame,us/frame,maxUs[,items/frame])");
      double totalUs = 0.0;
      for (uint32_t i = 0; i < kApiCount; ++i) {
        const uint64_t calls = perf::g_api[i].calls.load(std::memory_order_relaxed) - w.apiStart[i][0];
        if (calls == 0) {
          continue;
        }
        const double us = double(perf::g_api[i].ns.load(std::memory_order_relaxed) - w.apiStart[i][1]) / 1000.0;
        const uint64_t items = perf::g_api[i].items.load(std::memory_order_relaxed) - w.apiStart[i][2];
        totalUs += us;
        api += str::format(" ", perf::kApiNames[i], "=", f2(double(calls) / n), ",", f2(us / n), ",",
                           f2(double(perf::g_api[i].maxNs.load(std::memory_order_relaxed)) / 1000.0));
        if (items != calls) {
          api += str::format(",", f2(double(items) / n));
        }
      }
      api += str::format(" | totalUsPerFrame=", f2(totalUs / n));
      Logger::info(api);
    }

    // Standard D3D9 draw path: per-frame averages of the D3D9Ctr counters (non-zero ones only).
    {
      const uint64_t d3d9Frames = perf::g_d3d9ThreadFrames.load(std::memory_order_relaxed) - w.d3d9FramesStart;
      const uint64_t d3d9CpuNs = perf::g_d3d9ThreadCpuNs.load(std::memory_order_relaxed) - w.d3d9CpuNsStart;
      const uint64_t optionLocks = g_rtxOptionLockCount.load(std::memory_order_relaxed) - w.optionLocksStart;
      std::string d3d9 = str::format("[PerfD3D9] frames=", w.frames, " (per ray traced frame) | d3d9ThreadCpuMsPerFrame=",
        d3d9Frames ? f2(double(d3d9CpuNs) / 1e6 / double(d3d9Frames)) : std::string("n/a"), " (", d3d9Frames, " D3D9 frames)",
        " optionMutexLocks=", f2(double(optionLocks) / n), " |");
      for (uint32_t i = 0; i < kD3D9CtrCount; ++i) {
        const uint64_t v = perf::g_d3d9Ctr[i].load(std::memory_order_relaxed) - w.d3d9CtrStart[i];
        if (v != 0) {
          d3d9 += str::format(" ", perf::kD3D9CtrNames[i], "=", f2(double(v) / n));
        }
      }
      Logger::info(d3d9);
    }

    logSceneStats(ctx);
  }

  void RtxPerfLog::logSceneStats(RtxContext& ctx) {
    SceneManager& scene = ctx.getSceneManager();
    const AccelManager& accel = scene.getAccelManager();
    AccelManager::MemoryBreakdown as;
    accel.getMemoryBreakdown(as);

    const OpacityMicromapManager* omm = scene.getOpacityMicromapManager();
    uint32_t instances = 0, hidden = 0, alphaTested = 0, ommUsers = 0, ommReady = 0, ommPending = 0;
    uint64_t tris = 0, alphaTris = 0, ommUserTris = 0, ommReadyTris = 0;
    uint32_t geomOpaque = 0, geomNonOpaque = 0, solidNonOpaque = 0;
    uint64_t solidNonOpaqueTris = 0;
    for (const RtInstance* inst : accel.getOrderedInstances()) {
      if (inst == nullptr) {
        continue;
      }
      instances++;
      if (inst->isHidden()) {
        hidden++;
      }
      const BlasEntry* blas = inst->getBlas();
      const uint32_t prims = blas ? blas->modifiedGeometryData.calculatePrimitiveCount() : 0;
      tris += prims;
      if (!inst->isOpaque()) {
        alphaTested++;
        alphaTris += prims;
      }
      // Geometry opacity audit. BLAS geometry flags are per member (merged
      // buckets copy each member's own flags), so a fully opaque surface should always carry
      // VK_GEOMETRY_OPAQUE_BIT and no FORCE_NO_OPAQUE instance flag outside the unordered TLAS.
      // solidNonOpaque* counts the ones that do not (render-target surfaces and clip planes are
      // legitimate exceptions); they would take the any-hit / cull-non-opaque paths needlessly.
      if (!inst->usesUnorderedApproximations()) {
        const bool opaqueGeom = (inst->getGeometryFlags() & VK_GEOMETRY_OPAQUE_BIT_KHR) != 0 &&
                                (inst->getVkInstance().flags & VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR) == 0;
        if (opaqueGeom) {
          geomOpaque++;
        } else {
          geomNonOpaque++;
          if (inst->surface.alphaState.isFullyOpaque && inst->getMaterialType() == MaterialDataType::Opaque) {
            solidNonOpaque++;
            solidNonOpaqueTris += prims;
          }
        }
      }
      if (omm && OpacityMicromapManager::usesOpacityMicromap(*inst)) {
        ommUsers++;
        ommUserTris += prims;
        const int readiness = omm->getOmmReadiness(*inst);
        if (readiness == 2) {
          ommReady++;
          ommReadyTris += prims;
        } else if (readiness == 1) {
          ommPending++;
        }
      }
    }

    uint32_t lightsByType[5] = {};
    for (const auto& [hash, light] : scene.getLightManager().getLightTable()) {
      const uint32_t t = uint32_t(light.getType());
      if (t < 5) {
        lightsByType[t]++;
      }
    }

    // Device-local memory: driver usage / budget and the dxvk allocator's slack.
    const Rc<DxvkDevice>& device = ctx.getDevice();
    const DxvkAdapterMemoryInfo memInfo = device->adapter()->getMemoryHeapInfo();
    const VkPhysicalDeviceMemoryProperties memProps = device->adapter()->memoryProperties();
    uint64_t budget = 0, driverUsed = 0, allocated = 0, used = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
      if (!(memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
        continue;
      }
      if (i < memInfo.heapCount) {
        budget += memInfo.heaps[i].memoryBudget;
        driverUsed += memInfo.heaps[i].memoryAllocated;
      }
      const DxvkMemoryStats stats = device->getMemoryStats(i);
      allocated += stats.totalAllocated();
      used += stats.totalUsed();
    }

    const auto pctOf = [](uint64_t a, uint64_t b) { return b ? f2(100.0 * double(a) / double(b)) : std::string("0"); };
    const RtxPerfLog::Window& w = m_w;
    const double n = std::max(1.0, double(w.frames));
    // Render/output resolution from the most recent frame in the ring.
    const FrameRec& last = m_ring[(ctx.getDevice()->getCurrentFrameId()) % kRing];

    Logger::info(str::format("[SceneStats] tlasInstances=", accel.getTlasInstanceCount(),
      " piSlots=", accel.getPointInstancerSlotCount(), " piBatches=", accel.getPointInstancerBatchCount(),
      " surfaces=", instances, " hidden=", hidden,
      " | mergedBlas=", as.bucketCount, " mergedInstances=", as.bucketInstances, " mergedPrims=", as.bucketPrims,
      " perGeomBlas=", as.dynamicCount, " pooledBlas=", as.pooledCount,
      " | tris=", tris, " alphaTestedInstances=", alphaTested, " alphaTestedTris=", alphaTris,
      " geomOpaque=", geomOpaque, " geomNonOpaque=", geomNonOpaque,
      " solidNonOpaque=", solidNonOpaque, " solidNonOpaqueTris=", solidNonOpaqueTris,
      " | ommInstances=", ommUsers, " ommReady=", ommReady, " ommPending=", ommPending,
      " ommCoveragePct=", pctOf(ommReady, ommUsers), " ommTrisCoveragePct=", pctOf(ommReadyTris, ommUserTris),
      " ommCacheItems=", omm ? omm->getCacheItemCount() : 0, " ommUnprocessed=", omm ? omm->getUnprocessedCount() : 0,
      " ommBlacklisted=", omm ? omm->getBlacklistedCount() : 0,
      " ommCacheMiB=", omm ? (omm->getCacheBytesUsed() >> 20) : 0, "/", omm ? (omm->getCacheBudgetBytes() >> 20) : 0,
      " | lights=", scene.getLightManager().getLightTable().size(), " active=", scene.getLightManager().getActiveCount(),
      " sphere=", lightsByType[uint32_t(RtLightType::Sphere)], " rect=", lightsByType[uint32_t(RtLightType::Rect)],
      " disk=", lightsByType[uint32_t(RtLightType::Disk)], " cylinder=", lightsByType[uint32_t(RtLightType::Cylinder)],
      " distant=", lightsByType[uint32_t(RtLightType::Distant)],
      " rtxdiInitialSamples=", DxvkRtxdiRayQuery::initialSampleCount(), " risLightSamples=", RtxOptions::risLightSampleCount(),
      " | portals=", scene.getRayPortalManager().getRayPortalInfoSceneData().numActiveRayPortals,
      " portalDuplicatesHidden=", scene.getRayPortalManager().getDuplicatesHidden(),
      " | bounces=", uint32_t(RtxOptions::pathMinBounces()), "-", uint32_t(RtxOptions::pathMaxBounces()),
      " indirectMode=", int(RtxOptions::integrateIndirectMode()), " (0 IS,1 ReSTIR-GI,2 NRC,3 SHaRC)",
      " | render=", last.cpu.renderW, "x", last.cpu.renderH, " output=", last.cpu.outputW, "x", last.cpu.outputH,
      " upscaler=", int(RtxOptions::upscalerType()), " rayReconstruction=", RtxOptions::isRayReconstructionEnabled() ? 1 : 0,
      " dlssProfile=", int(RtxOptions::qualityDLSS()), " dlssNR=", last.cpu.dlssNr ? 1 : 0,
      " | vramMiB driverUsed=", driverUsed >> 20, " budget=", budget >> 20,
      " dxvkAllocated=", allocated >> 20, " dxvkUsed=", used >> 20, " slack=", (allocated - std::min(allocated, used)) >> 20,
      " | perFrame: mergedBlasBuilt=", f2(w.changes.mergedBuilt / n), " mergedBlasRefit=", f2(w.changes.mergedRefit / n),
      " primsBuilt=", f2(double(w.changes.primsBuilt) / n), " perGeomBuilt=", f2(w.changes.dynBuilt / n),
      " perGeomRefit=", f2(w.changes.dynRefit / n), " addedInstances=", f2(w.changes.addedInstances / n),
      " tlasBuilt=", f2(w.changes.tlasBuilt / n), " surfaceKiB=", f2(double(w.changes.surfaceBytes >> 10) / n),
      " ommBakedKTris=", f2(w.changes.ommBakedKTris / n), " ommBuiltKTris=", f2(w.changes.ommBuiltKTris / n)));

    // Since the last [SceneStats] line: micromaps dropped because their opacity texture changed
    // (rtx.opacityMicromap.rebakeOnTextureChange), registered again after a bind miss
    // (rerequestLostMicromaps), LRU-evicted (and how many of those were younger than
    // minUsageFrameAgeBeforeEviction: a budget decrease), and merged-BLAS refreshes
    // (cache.keepMergedBlasMicromapsInUse).
    if (omm != nullptr) {
      const OpacityMicromapManager::ChurnStats churn = const_cast<OpacityMicromapManager*>(omm)->takeChurnStats();
      Logger::info(str::format("[OmmChurn] textureChangeRebakes=", churn.textureChangeRebakes,
        " (kept: slotReused=", churn.slotReusedSkips, " sameSize=", churn.sameExtentSkips, ")",
        " lostRerequests=", churn.lostRerequests, " evictions=", churn.evictions,
        " (younger than minUsageFrameAge ", churn.evictedYoungerThan900, ")",
        " mergedInUseRefreshes=", churn.mergedTouches,
        // The request filter's table: hashes waiting in it now against its cap, entries dropped as stale, and
        // requests refused because it was full (with the table at the cap nothing new is ever baked).
        " requestStats=", omm->getRequestStatisticsCount(), "/", OpacityMicromapOptions::BuildRequests::maxRequests(),
        " (purged ", churn.requestStatsPurged, ", refusedAtCap ", churn.requestsRefusedAtCap, ")",
        " | rebakeOnTextureChange=", OpacityMicromapOptions::rebakeOnTextureChange() ? 1 : 0,
        " rerequestLost=", OpacityMicromapOptions::rerequestLostMicromaps() ? 1 : 0,
        " keepMergedInUse=", OpacityMicromapOptions::Cache::keepMergedBlasMicromapsInUse() ? 1 : 0,
        " lastInteractionSkipsCutouts=", RtxOptions::resolveFinalInteractionSkipsNonOpaque() ? 1 : 0,
        // Record the foliage-ray A/B state next to the other cutout knobs.
        " foliageCullFromBounce=", RtxOptions::cullNonOpaqueFromBounce(),
        " foliageCullDiffuse=", RtxOptions::cullNonOpaqueOnDiffuseRays() ? 1 : 0,
        " foliageFastRayMode=", RtxOptions::fastRayMode()));
    }
    D3d9Alpha::logStats(omm);
  }

  // rtx.profile.liveTraceReadout. Samples arrive on the CS thread from
  // RtxContext::beginGpuStageTiming; the dev menu reads them, so the state sits behind a mutex.
  namespace {
    struct LiveTraceAccum {
      const char* label;
      double sumMs;
      uint32_t count;
    };
    struct LiveTraceState {
      std::mutex mutex;
      std::vector<LiveTraceAccum> stages;
      uint32_t samples = 0;
      double measuredSumMs = 0.0;
      uint32_t skipBeforeFrameId = 0;
    };
    LiveTraceState& liveTrace() {
      static LiveTraceState s;
      return s;
    }

    bool startsWith(const char* s, const char* prefix) {
      return std::strncmp(s, prefix, std::strlen(prefix)) == 0;
    }

    // Stages whose cost is dominated by ray traversal + hit shading. Shading is the same in an A/B of
    // BLAS layouts, so their sum moves with traversal cost.
    bool isTraceStage(const char* label) {
      return startsWith(label, "GBuffer") || startsWith(label, "DirectIntegration") ||
             std::strcmp(label, "IndirectIntegration") == 0 || std::strcmp(label, "ReSTIRGI") == 0 ||
             std::strcmp(label, "RTXDI") == 0 || std::strcmp(label, "NEECache") == 0;
    }
  }

  void RtxPerfLog::onLiveTraceSample(uint32_t frameId, const char* const* labels, const double* ms, uint32_t count, double measuredMs) {
    LiveTraceState& s = liveTrace();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (int32_t(frameId - s.skipBeforeFrameId) < 0) {
      return;
    }
    for (uint32_t i = 0; i < count; ++i) {
      auto it = std::find_if(s.stages.begin(), s.stages.end(),
        [&](const LiveTraceAccum& a) { return std::strcmp(a.label, labels[i]) == 0; });
      if (it == s.stages.end()) {
        s.stages.push_back({ labels[i], ms[i], 1 });
      } else {
        it->sumMs += ms[i];
        it->count++;
      }
    }
    s.samples++;
    s.measuredSumMs += measuredMs;
  }

  RtxPerfLog::LiveTraceSnapshot RtxPerfLog::liveTraceSnapshot() {
    LiveTraceSnapshot out;
    LiveTraceState& s = liveTrace();
    std::lock_guard<std::mutex> lock(s.mutex);
    out.samples = s.samples;
    out.settling = s.samples == 0 && s.skipBeforeFrameId != 0;
    if (s.samples == 0) {
      return out;
    }
    out.measuredMs = s.measuredSumMs / double(s.samples);
    for (const LiveTraceAccum& a : s.stages) {
      // Average over all samples: a stage missing from some frames counts as 0 there.
      const double avg = a.sumMs / double(s.samples);
      const bool trace = isTraceStage(a.label);
      out.stages.push_back({ a.label, avg, trace });
      if (trace) {
        out.traceMs += avg;
      }
    }
    std::sort(out.stages.begin(), out.stages.end(),
      [](const LiveTraceStage& a, const LiveTraceStage& b) { return a.avgMs > b.avgMs; });
    return out;
  }

  void RtxPerfLog::resetLiveTrace(uint32_t currentFrameId, uint32_t settleFrames) {
    LiveTraceState& s = liveTrace();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stages.clear();
    s.samples = 0;
    s.measuredSumMs = 0.0;
    s.skipBeforeFrameId = std::max(1u, currentFrameId + settleFrames);
  }

  RtxPerfLog::TlasCounts RtxPerfLog::gatherTlasCounts(SceneManager& scene) {
    const AccelManager& accel = scene.getAccelManager();
    AccelManager::MemoryBreakdown as;
    accel.getMemoryBreakdown(as);
    TlasCounts c;
    c.opaque = accel.getTlasInstanceCount(Tlas::Opaque);
    c.unordered = accel.getTlasInstanceCount(Tlas::Unordered);
    c.sss = accel.getTlasInstanceCount(Tlas::SSS);
    c.piSlots = accel.getPointInstancerSlotCount();
    c.mergedBlas = as.bucketCount;
    c.mergedInstances = as.bucketInstances;
    c.mergedPrims = as.bucketPrims;
    c.perGeomBlas = as.dynamicCount;
    c.tlasBytes = as.tlasBytes;
    return c;
  }

  void RtxPerfLog::logLiveTrace(SceneManager& scene) {
    const TlasCounts c = gatherTlasCounts(scene);
    const LiveTraceSnapshot snap = liveTraceSnapshot();
    std::string line = str::format("[TlasAB] minimizeBlasMerging=", RtxOptions::minimizeBlasMerging() ? 1 : 0,
      " mergedBucketCellSize=", f2(RtxOptions::effectiveMergedBucketCellSize()),
      " fastTraceStaticMerged=", RtxOptions::preferFastTraceForStaticMergedBlas() ? 1 : 0,
      " | tlasInstances opaque=", c.opaque, " unordered=", c.unordered, " sss=", c.sss, " piSlots=", c.piSlots,
      " | mergedBlas=", c.mergedBlas, " mergedInstances=", c.mergedInstances, " mergedPrims=", c.mergedPrims,
      " perGeomBlas=", c.perGeomBlas, " tlasMiB=", f2(double(c.tlasBytes) / (1024.0 * 1024.0)),
      " | samples=", snap.samples, " measuredMs=", f2(snap.measuredMs), " traceMs=", f2(snap.traceMs), " |");
    for (const LiveTraceStage& st : snap.stages) {
      if (st.avgMs >= 0.01) {
        line += str::format(" ", st.label, "=", f2(st.avgMs));
      }
    }
    Logger::info(line);
  }
}
