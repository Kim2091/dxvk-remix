#pragma once

#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <queue>
#include <vector>

#include "../util/thread.h"

#include "../vulkan/vulkan_presenter.h"

#include "dxvk_cmdlist.h"
#include "rtx_render/rtx_camera.h"

namespace dxvk {
  
  class DxvkDevice;

  /**
   * \brief Submission status
   * 
   * Stores the result of a queue
   * submission or a present call.
   */
  struct DxvkSubmitStatus {
    std::atomic<VkResult> result = { VK_SUCCESS };
  };


  /**
   * \brief Queue submission info
   * 
   * Stores parameters used to submit
   * a command buffer to the device.
   */
  struct DxvkSubmitInfo {
    Rc<DxvkCommandList> cmdList;
    VkSemaphore         waitSync;
    VkSemaphore         wakeSync;
    // NV-DXVK start: Reflex rendering work support
    bool                insertReflexRenderMarkers;
    uint64_t            cachedReflexFrameId;
    //NV-DXVK end
  };
  
  
  /**
   * \brief Present info
   *
   * Stores parameters used to present
   * a swap chain image on the device.
   */
  struct DxvkPresentInfo {
    Rc<vk::Presenter>   presenter;
    // Represents the Reflex Frame ID at the time of image presentation for usage in presentation submission.
    uint64_t            cachedReflexFrameId;
    // Note: This flag is specifically used when the DXVK Queue should insert Reflex present
    // markers rather than the presenter currently in use. This is done because some presenters
    // (namely the DLFG Presenter) will insert their own Reflex markers due to having more complex
    // requirements.
    bool                insertReflexPresentMarkers;
    // Represents the acquired presenter image index at the time of image presentation for usage in presentation submission.
    uint32_t            cachedAcquiredImageIndex;
    // NV-DXVK start: Keep raster presents out of Remix frame interpolation
    bool                isRemixPresent = true;
    // Reflex sleep preceding this frame's simulation, for rtx.present.traceCsv (ms; -1 = none)
    float               reflexSleepMs = -1.0f;
    // NV-DXVK end
  };


  // NV-DXVK start: DLFG integration
  /**
   * \brief Frame interpolation info
   *
   * Stores parameters used to run frame
   * interpolation in the submit queue at present time.
   */
  struct DxvkFrameInterpolationInfo {
    uint32_t          frameId;
    RtCamera          camera;
    Rc<DxvkImageView> motionVectors;
    VkImageLayout     motionVectorsLayout;
    Rc<DxvkImageView> depth;
    VkImageLayout     depthLayout;
    bool              resetHistory;
    // number of requested interpolated frames; must be at least 1, actual count may be lower depending on hardware support for MFG
    uint32_t          interpolatedFrameCount = 0;
    // The Remix output in the back buffer format, for backends that compose the HUD over the generated frame. DLSS-G does not use it.
    Rc<DxvkImage>     hudLessColor;
    // The real frame time in milliseconds. DLSS-G does not use it.
    float             frameTimeMs = 0.0f;
    
    bool valid() const {
      return motionVectors.ptr() &&
        depth.ptr() &&
        interpolatedFrameCount > 0;
    }

    void reset() {
      motionVectors = nullptr;
      depth = nullptr;
      resetHistory = false;
      interpolatedFrameCount = 0;
      hudLessColor = nullptr;
      frameTimeMs = 0.0f;
    }
  };
  // NV-DXVK end

  /**
   * \brief Submission queue entry
   */
  struct DxvkSubmitEntry {
    DxvkSubmitStatus*   status;
    DxvkSubmitInfo      submit;
    DxvkPresentInfo     present;
    // NV-DXVK start: DLFG integration
    // sent down to stash frame interpolation parameters before present
    DxvkFrameInterpolationInfo frameInterpolation;
    // NV-DXVK end
  };

  /**
   * \brief Submission queue
   */
  class DxvkSubmissionQueue {

  public:
    
    DxvkSubmissionQueue(DxvkDevice* device);
    ~DxvkSubmissionQueue();

    /**
     * \brief Number of pending submissions
     * 
     * A return value of 0 indicates
     * that the GPU is currently idle.
     * \returns Pending submission count
     */
    uint32_t pendingSubmissions() const {
      return m_pending.load();
    }

    /**
     * \brief Retrieves estimated GPU idle time
     *
     * This is a monotonically increasing counter
     * which can be evaluated periodically in order
     * to calculate the GPU load.
     * \returns Accumulated GPU idle time, in us
     */
    uint64_t gpuIdleTicks() const {
      return m_gpuIdle.load();
    }

    /**
     * \brief Retrieves last submission error
     * 
     * In case an error occured during asynchronous command
     * submission, it will be returned by this function.
     * \returns Last error from command submission
     */
    VkResult getLastError() const {
      return m_lastError.load();
    }

    // The Remix presents since the last take, for the always-on [Perf] line.
    // Filled on the submit thread, taken on the CS thread.
    struct PresentWindow {
      std::vector<float> intervalMs;   // between vkQueuePresentKHR calls
      std::vector<float> acquireMs;    // the swapchain image acquire after each present (V-Sync and queue back-pressure)
      std::vector<float> gateWaitMs;   // rtx.present.gpuQueueLimit waits
      uint64_t framesInFlightSum = 0;
      uint32_t presents = 0;
    };
    PresentWindow takePresentWindow();
    
    /**
     * \brief Submits a command list asynchronously
     * 
     * Queues a command list for submission on the
     * dedicated submission thread. Use this to take
     * the submission overhead off the calling thread.
     * \param [in] submitInfo Submission parameters 
     */
    void submit(
            DxvkSubmitInfo      submitInfo);
    
    /**
     * \brief Presents an image synchronously
     *
     * Waits for queued command lists to be submitted
     * and then presents the current swap chain image
     * of the presenter. May stall the calling thread.
     * \param [in] present Present parameters
     * \returns Status of the operation
     */
    void present(
            DxvkPresentInfo     presentInfo,
            DxvkSubmitStatus*   status);

    // NV-DXVK start: DLFG integration
    /**
     * \brief Set up frame interpolation parameters for next present
     *
     * Used to send down all data required to do frame interpolation
     * at present time, except for the final output image.
     * If not called on a given frame, or called with an invalid input,
     * frame interpolation won't be done.
     */
    void setupFrameInterpolation(
      DxvkFrameInterpolationInfo frameInterpolationInfo);

    /**
     * \brief Checks if the next present will trigger frame interpolation
     * \returns true if frame interpolation is set up for the current frame
     */
    //bool frameInterpolationOnNextPresent();
    
    /**
     * \brief Does a busy-wait in the current thread
     */
    //void threadWaitUs(int64_t us);
    // NV-DXVK end

    /**
     * \brief Synchronizes with one queue submission
     * 
     * Waits for the result of the given submission
     * or present operation to become available.
     * \param [in,out] status Submission status
     */
    void synchronizeSubmission(
            DxvkSubmitStatus*   status);
    
    /**
     * \brief Synchronizes with queue submissions
     * 
     * Waits for all pending command lists to be
     * submitted to the GPU before returning.
     */
    void synchronize();

    /**
     * \brief Synchronizes until a given condition becomes true
     *
     * Useful to wait for the GPU without busy-waiting.
     * \param [in] pred Predicate to check
     */
    template<typename Pred>
    void synchronizeUntil(const Pred& pred) {
      std::unique_lock<dxvk::mutex> lock(m_mutex);
      m_finishCond.wait(lock, pred);
    }

    /**
     * \brief Locks device queue
     *
     * Locks the mutex that protects the Vulkan queue
     * that DXVK uses for command buffer submission.
     * This is needed when the app submits its own
     * command buffers to the queue.
     */
    void lockDeviceQueue();

    /**
     * \brief Unlocks device queue
     *
     * Unlocks the mutex that protects the Vulkan
     * queue used for command buffer submission.
     */
    void unlockDeviceQueue();
  
  private:

    DxvkDevice*             m_device;

    std::atomic<VkResult>   m_lastError = { VK_SUCCESS };

    // NV-DXVK start: GPU crash diagnostics
    std::atomic<bool>       m_gpuCrashHandled = { false };
    // NV-DXVK end
    
    std::atomic<bool>       m_stopped = { false };
    std::atomic<uint32_t>   m_pending = { 0u };
    std::atomic<uint64_t>   m_gpuIdle = { 0ull };

    dxvk::mutex                 m_mutex;
    dxvk::mutex                 m_mutexQueue;
    
    dxvk::condition_variable    m_appendCond;
    dxvk::condition_variable    m_submitCond;
    dxvk::condition_variable    m_finishCond;

    std::queue<DxvkSubmitEntry> m_submitQueue;
    std::queue<DxvkSubmitEntry> m_finishQueue;

    // NV-DXVK start: present pacing
    // Declared before the threads, which start in the constructor.
    // Detects "one late interval, then 3 fast ones" in a stream of intervals.
    struct BurstDetector {
      float    ring[16] = {};
      uint32_t count    = 0;
      // Returns true when the interval just pushed completes a burst.
      bool push(float ms, float* outLateMs);
    };

    // Command lists handed to / completed by the finish thread, in submission order. Under m_mutex.
    uint64_t                    m_submittedSeq = 0;
    uint64_t                    m_completedSeq = 0;
    // Per Remix present, the seq of the last command list submitted before it. Under m_mutex;
    // the finish thread pops entries as those command lists complete.
    std::deque<uint64_t>        m_presentFrameSeqs;
    // Finish thread (samples moved out under m_mutex): intervals between presented frames' GPU completion.
    uint64_t                    m_lastFrameDoneNs = 0;
    std::vector<float>          m_frameDoneIntervalsMs;
    BurstDetector               m_frameDoneBursts;
    uint32_t                    m_frameDoneBurstCount = 0;
    // rtx.present.traceCsv. Finish thread appends (frame seq, GPU-done QPC ms) as presented
    // frames complete; the submit thread drains it under m_mutex when it writes rows.
    std::vector<std::pair<uint64_t, double>> m_traceDone;

    // Submit thread only.
    struct PresentPacingDiag {
      uint64_t           lastCallNs = 0;
      uint32_t           presents = 0;
      uint64_t           prevFrameSeqs[4] = {};   // [0] = previous Remix present's frame seq
      std::vector<float> callIntervalMs, queuePresentMs, acquireMs, gateWaitMs;
      std::vector<float> presentWaitMs;   // Frames that waited only
      std::vector<float> framesInFlight, cmdListsInFlight, submitQueueLen;
      BurstDetector      callBursts;
      uint32_t           callBurstCount = 0, burstLinesLogged = 0, gateTimeouts = 0;
      bool               loggedRuntimeModules = false;
      // rtx.present.traceCsv
      struct TraceCall {
        uint64_t frameSeq;
        double   callQpcMs;           // vkQueuePresentKHR call, QueryPerformanceCounter in ms (PresentMon --qpc_time_ms clock)
        float    queuePresentMs, gateWaitMs;
        uint32_t framesInFlight, cmdListsInFlight;
        bool     gpuDoneAtPresent;    // the frame had already finished on the GPU when the present reached the submit thread
        uint64_t reflexFrameId;
        float    reflexSleepMs;       // Reflex sleep before this frame's simulation, -1 = none
        uint64_t presentId;           // VkPresentIdKHR attached to this present, 0 = none
        uint64_t presentWaitId;       // id this frame waited for before its RT work, 0 = no wait
        float    presentWaitMs;       // time in vkWaitForPresentKHR before this frame's RT work, -1 = no wait
        double   presentDoneQpcMs;    // QPC ms when that wait returned complete, -1 = no wait / timeout / error
      };
      std::deque<TraceCall> traceCalls;   // presents whose GPU-done time is not written yet
      uint64_t           tracePresents = 0; // rows written so far (the present index column)
      FILE*              traceFile = nullptr;
      bool               traceOpenFailed = false;
    } m_pacing;

    dxvk::mutex   m_presentWindowMutex;
    PresentWindow m_presentWindow;
    uint64_t      m_presentWindowLastNs = 0;

    // rtx.present.presentWaitPacing, submit thread only.
    struct PresentWaitState {
      Rc<vk::Presenter> presenter;          // the last Remix present's presenter
      bool     armed = false;               // a Remix present went out and its next frame's RT work has not been submitted
      // Result of this frame's wait, consumed by the next Remix present's trace row.
      uint64_t frameWaitId = 0;
      float    frameWaitMs = -1.0f;
      double   frameDoneQpcMs = -1.0;
      // Timeout guard: consecutive timeouts switch the wait off until the swapchain is recreated or the option toggles.
      uint32_t consecutiveTimeouts = 0;
      bool     suspended = false;
      uint32_t suspendedGeneration = 0;
      uint32_t timeoutsInInterval = 0;
      // rtx.present.presentWaitPreGateFlush lists submitted ahead of an armed gate this interval.
      uint32_t preGateListsInInterval = 0;
      bool     loggedUnsupported = false, loggedError = false, loggedFirstWait = false;
    } m_presentWait;

    // Submit thread, before the first ray tracing command list after a Remix present.
    void waitForPresentBeforeRtWork();

    // Submit thread. Registers a Remix present's frame boundary, reports the GPU backlog at that
    // moment and waits for the rtx.present.gpuQueueLimit gate. Returns the gate wait in ns.
    // frameSeq / gpuDoneAtPresent: the frame's closing command list seq and whether it had already completed.
    uint64_t waitRemixPresentGate(uint32_t& framesInFlight, uint32_t& cmdListsInFlight, uint32_t& submitQueueLen,
                                  uint64_t& frameSeq, bool& gpuDoneAtPresent);
    // Submit thread. Records one Remix present and logs [PresentPacing] / [PresentBurst].
    void recordRemixPresent(const vk::Presenter& presenter, uint64_t callNs, double callQpcMs, uint64_t gateWaitNs,
                            uint32_t framesInFlight, uint32_t cmdListsInFlight, uint32_t submitQueueLen,
                            uint64_t frameSeq, bool gpuDoneAtPresent, uint64_t reflexFrameId, float reflexSleepMs);
    // rtx.present.traceCsv, submit thread. tracePresent queues a row and opens/closes the file
    // as the option changes; flushPresentTrace writes the rows whose GPU-done time has arrived.
    void tracePresent(const PresentPacingDiag::TraceCall& call);
    void flushPresentTrace(bool closeFile);
    // Once, at the first Remix present: which presentation/overlay runtimes are loaded in this process.
    void logPresentRuntimeModules();
    // NV-DXVK end

    dxvk::thread                m_submitThread;
    dxvk::thread                m_finishThread;

    VkResult submitToQueue(
      const DxvkSubmitInfo& submission);

    // NV-DXVK start: GPU crash diagnostics
    void onGpuCrash(const char* reason);
    // NV-DXVK end

    void submitCmdLists();

    void finishCmdLists();

    // NV-DXVK start: GPU crash handling (Aftermath dump wait + device-loss reporting)
    // Stall until Aftermath finishes writing its crash dump (no-op if Aftermath is disabled).
    void waitForAftermathDump();

    // Report the GPU crash and terminate the process. Only the first thread to observe device loss
    // performs the handling; any other thread blocks here until the process is killed.
    void handleDeviceLost();
    // NV-DXVK end
    
    // NV-DXVK start: DLFG integration
    DxvkFrameInterpolationInfo m_currentFrameInterpolationData;
    // stash a reference to the last presenter object in case we need to flush
    // this is reset when we flush
    Rc<vk::Presenter> m_lastPresenter = nullptr;
    // NV-DXVK end
  };
}
