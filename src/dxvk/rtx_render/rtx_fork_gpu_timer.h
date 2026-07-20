#pragma once

#include <cstdint>

namespace dxvk {

  class DxvkContext;

  namespace fork_hooks {

    // GPU pass timer (fork, 2026-07-20). Hangs Vulkan timestamp-query pairs
    // on the __ScopedAnnotation brackets that already wrap every annotated
    // render pass (the same brackets Tracy/Nsight consume), aggregates
    // per-pass GPU durations CPU-side, and logs a ranked table to the log
    // every rtx.gpuTimerLogIntervalFrames frames. In-process replacement
    // for the bundled Tracy v0.8 client, whose on-demand stream stalls the
    // game under Remix's zone volume (2026-07-20 FO4 field sessions).
    //
    // Enable with rtx.gpuTimerEnable (default off; a single branch + a
    // thread-local push/pop per annotation when disabled). Requires the
    // Vulkan 1.2 hostQueryReset feature (enabled on the d3d9 device path);
    // self-disables with a log line when unavailable.
    //
    // Threading: annotations record on the CS thread; the shared query ring
    // is mutex-guarded and the per-thread nesting stack is thread_local, so
    // stray zones from other threads stay paired.
    void gpuTimerBegin(DxvkContext* ctx, const char* name);
    void gpuTimerEnd(DxvkContext* ctx);

  }
}
