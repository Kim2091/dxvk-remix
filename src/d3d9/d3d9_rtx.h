#pragma once

#include "d3d9_state.h"
#include "../dxvk/dxvk_buffer.h"
#include "../util/util_threadpool.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>
#include <optional>
#include <unordered_map>

#include "../util/thread.h"
#include "../frontends/d3d9/d3d9_frontend.h"

namespace dxvk {
  struct D3D9BufferSlice;
  class DxvkDevice;

  enum class D3D9RtxFlag : uint32_t {
    DirtyLights,
    DirtyClipPlanes,
  };

  using D3D9RtxFlags = Flags<D3D9RtxFlag>;

  namespace PrepareDrawFlag {
    enum {
      Ignore                      = 0,
      CommitToRayTracing          = 1 << 0, // Process the current state as a part of ray tracing
      ApplyDrawState              = 1 << 1, // Submit draw state to dxvk-cs, so it can be used to issue original or non-original draw calls (e.g. terrain, sky)
      OriginalDrawCall            = 1 << 2, // Issue the original draw call
      PreserveDrawCallAndItsState = ApplyDrawState | OriginalDrawCall,
    };
  }
  using PrepareDrawFlags = uint32_t;

  //This class handles all of the RTX operations that are required from the D3D9 side.
  struct D3D9Rtx {
    friend class ImGUI; // <-- we want to modify these values directly.

    D3D9Rtx(D3D9DeviceEx* d3d9Device, bool enableDrawCallConversion = true);

    RTX_OPTION("rtx", bool, orthographicIsUI, true, "When enabled, draw calls that are orthographic will be considered as UI.");
    RTX_OPTION("rtx", bool, preTransformedVerticesIsUI, false, "When enabled, draw calls using pre-transformed (screen-space) vertices will be considered as UI. This is typical for D3D8/D3D9 games that render UI with RHW vertices.");
    RTX_OPTION("rtx", bool, allowCubemaps, false, "When enabled, cubemaps from the game are processed through Remix, but they may not render correctly.");
    RTX_OPTION("rtx", bool, useVertexCapture, true, "When enabled, injects code into the original vertex shader to capture final shaded vertex positions.  Is useful for games using simple vertex shaders, that still also set the fixed function transform matrices.");
    RTX_OPTION("rtx", bool, useVertexCapturedNormals, true, "When enabled, vertex normals are read from the input assembler and used in raytracing.  This doesn't always work as normals can be in any coordinate space, but can help sometimes.");
    RTX_OPTION("rtx", bool, useVertexCapturedTexcoords, false, "When enabled, vertex shader output texcoords always override input texcoords from the vertex declaration. Enable for games where the vertex shader applies meaningful UV transformations that should be used for ray tracing (e.g. animated UVs via shader constants).");
    RTX_OPTION("rtx", bool, useWorldMatricesForShaders, true, "When enabled, Remix will utilize the world matrices being passed from the game via D3D9 fixed function API, even when running with shaders.  Sometimes games pass these matrices and they are useful, however for some games they are very unreliable, and should be filtered out.  If you're seeing precision related issues with shader vertex capture, try disabling this setting.");
    RTX_OPTION("rtx", bool, enableIndexBufferMemoization, true, "CPU performance optimization, should generally be enabled.  Will reduce main thread time by caching processIndexBuffer operations and reusing when possible, this will come at the expense of some CPU RAM.");
    RTX_OPTION("rtx", bool, geometryHashMemoInline, false,
               "Reduces CPU cost on the standard D3D9 draw path, with the geometry hash memo: a memoized geometry hash "
               "is passed to the render thread with the draw, instead of through a worker task that only returns the "
               "stored value. The hashes are the same.");
    RTX_OPTION("rtx", uint32_t, geometryHashMemoSelfCheckFrames, 0,
               "Diagnostic for the geometry hash memo: every this many frames, every memo hit is hashed again in full "
               "and compared with the stored value. A mismatch, which means a buffer write did not update the buffer's "
               "content version, is logged as [GeometryHashMemoCheck] (the first 20) and counted as "
               "memoSelfCheckMismatch in [PerfD3D9]. The fresh hash is used on those frames. 0 = off.");
    RTX_OPTION("rtx", uint32_t, numGeometryProcessingThreads, 2, "The desired number of CPU threads to dedicate to geometry processing  Will be limited by the number of CPU cores.  There may be some advantage to lowering this number in games which are fairly simple and use a low number of draw calls per frame.  The default was determined by looking at a game with around 2000 draw calls per frame, and with a reasonably high average triangle count per draw.");

    // Copy of the parameters issued to D3D9 on DrawXXX
    struct DrawContext {
      D3DPRIMITIVETYPE PrimitiveType;
      INT              BaseVertexIndex;
      UINT             MinVertexIndex;
      UINT             NumVertices;
      UINT             StartIndex;
      UINT             PrimitiveCount;
      BOOL             Indexed;
    };
    static_assert(sizeof(DrawContext) == 28, "Please, recheck initializer usages if this changes.");

    /**
      * \brief: Initialize the D3D9 RTX interface
      */
    void Initialize();

    /**
      * \brief: Signal that an occlusion query has started for the current device
      */
    void BeginOcclusionQuery() {
      ++m_activeOcclusionQueries;
    }

    /**
      * \brief: Signal that an occlusion query has ended for the current device
      */
    void EndOcclusionQuery() {
      --m_activeOcclusionQueries;
      assert(m_activeOcclusionQueries >= 0);
    }

    /**
      * \brief: Signal that a parameter needs to be updated for RTX
      *
      * \param [in] flag: parameter that requires updating
      */
    void SetDirty(D3D9RtxFlag flag) {
      m_flags.set(flag);
    }

    /**
      * \brief: Signal that a transform has updated
      *
      * \param [in] idx: index of transform
      */
    void SetTransformDirty(const uint32_t transformIdx) {
      if (transformIdx > GetTransformIndex(D3DTS_WORLD)) {
        m_maxBone = std::max(m_maxBone, transformIdx - GetTransformIndex(D3DTS_WORLD));
      }
    }

    /**
      * \brief: This function is responsible for preparing the geometry for rendering in Direct3D 9.
      *
      * \param [in] indexed: A boolean value indicating whether or not the geometry to be rendered is indexed.
      * \param [in] state : An object of type Direct3DState9 that contains the current state of the Direct3D pipeline.
      * \param [in] context : An object of type Draw that contains the context for the draw call.
      *
      * Returns false if this drawcall should be removed from further processing, returns true otherwise.
      */
    PrepareDrawFlags PrepareDrawGeometryForRT(const bool indexed, const DrawContext& context);

    /**
      * \brief: This function is responsible for preparing the geometry for rendering in Direct3D 9 
      *         when the vertex and index data is packed into a single buffer: ||VERTICES|INDICES||
      * 
      * \param [in] indexed: A boolean value indicating whether or not the geometry to be rendered is indexed.
      * \param [in] buffer : An object of type D3D9BufferSlice that contains the packed vertex and index data.
      * \param [in] indexFormat : The format of the indices in the buffer.
      * \param [in] indexOffset : The offset of the index data in bytes
      * \param [in] indexSize : The size of the index data in bytes.
      * \param [in] vertexSize : The size of the vertex data in bytes.
      * \param [in] vertexStride : The stride of the vertex data in bytes.
      * \param [in] drawContext : An object of type Draw that contains the context for the draw call.
      *
      * Returns false if this drawcall should be removed from further processing, returns true otherwise.
      */
    PrepareDrawFlags PrepareDrawUPGeometryForRT(const bool indexed,
                                                const D3D9BufferSlice& buffer,
                                                const D3DFORMAT indexFormat,
                                                const uint32_t indexSize,
                                                const uint32_t indexOffset,
                                                const uint32_t vertexSize,
                                                const uint32_t vertexStride,
                                                const DrawContext& context);

    /**
      * \brief: Sends the pending drawcall geometry/state for raytracing, if nothing pending, does nothing.
      *
      * \param [in] drawContext : An object of type Draw that contains the context for the draw call.
      */
    void CommitGeometryToRT(const DrawContext& drawContext);

    /**
      * \brief: Signal that a swapchain has been resized or reconfigured.
      * 
      * \param [in] presentationParameters: A reference to the D3D present params.
      */
    void ResetSwapChain(const D3DPRESENT_PARAMETERS& presentationParameters);

    /**
      * \brief: Signal that we've reached the end of the frame.
      */
    void EndFrame(const Rc<DxvkImage>& targetImage, bool callInjectRtx = true);

    /**
      * \brief: Records whether the game window is inactive or minimized, where a game may keep drawing without presenting.
      * Safe to call from any thread. Under the bridge the window procedure runs on the message-channel thread,
      * which must not touch frame state, so onBeginScene and onReset end such frames on the D3D9 thread.
      */
    void setWindowInactive(bool inactive) {
      m_windowInactive.store(inactive, std::memory_order_release);
    }

    /**
      * \brief: Runs at BeginScene on the D3D9 thread with the device lock held.
      * Without presents, it ends the frame without injection once the frame has more scenes than the last presented frame had.
      */
    void onBeginScene();

    /**
      * \brief: Runs at Reset and ResetEx on the D3D9 thread with the device lock held.
      * Without presents, it ends a frame that has a draw without injection.
      */
    void onReset();

    /**
      * \brief: Signal that we're about to present the image.
      */
    void OnPresent(const Rc<DxvkImage>& targetImage);

    /**
      * \brief: Increments the Reflex frame ID. Should be called after presentation and only after every Reflex related marker
      * call for the current frame (this typically means other threads running in parallel will need to cache this value from the
      * frame they were dispatched on).
      */
    void IncrementReflexFrameId() {
      ++m_reflexFrameId;
    }

    /**
      * \brief: Gets the Reflex frame ID for the current frame on the main thread. This is incremented after each present.
      * Only intended for use with Reflex, other methods for getting a frame ID exist which may make more sense for other systems.
      */
    uint64_t GetReflexFrameId() const {
      return m_reflexFrameId;
    }

  private: 
    // Reused fixed-size blocks: once allocated, a block is never reallocated, so background
    // skinning can keep raw Matrix4* into a prior copy. m_blocks can grow, but the heap
    // Block objects and their `m_matrices` array storage stay pinned. The write cursor
    // rewinds each frame; old blocks are retained to avoid per-frame allocs.
    struct SkinningMatrixPool {
      static constexpr size_t kMatricesPerBlock = 256 * 256;
      struct Block {
        std::array<Matrix4, kMatricesPerBlock> m_matrices;
      };
      std::vector<std::unique_ptr<Block>> m_blocks;
      size_t m_blockIndex = 0;          // which block the next write will use
      size_t m_nextIndexInBlock = 0;    // next free slot in m_blocks[m_blockIndex]

      void clear();
      const Matrix4* stageBones(const Matrix4* source, size_t matrixCount);
    } m_stagedBones;

    inline static const uint32_t kMaxConcurrentDraws = 6 * 1024; // some games issuing >3000 draw calls per frame...  account for some consumer thread lag with x2
    using GeometryProcessor = WorkerThreadPool<kMaxConcurrentDraws>;
    const std::unique_ptr<GeometryProcessor> m_pGeometryWorkers;
    AtomicQueue<DrawCallState, kMaxConcurrentDraws> m_drawCallStateQueue;
    // The D3D9 frontend's CS-thread half: every committed draw goes through it.
    frontends::d3d9::D3D9Frontend m_frontend;

    DrawCallState m_activeDrawCallState;

    RtxStagingDataAlloc m_rtStagingData;
    D3D9DeviceEx* m_parent;

    std::optional<D3DPRESENT_PARAMETERS> m_activePresentParams;

    D3D9RtxFlags m_flags = 0xFFFFffff;

    uint32_t m_drawCallID = 0;
    // Note: A frame identifier the the main thread holds on to passed down into thread invocations such that
    // Reflex markers have a consistent ID despite executing in parallel (as typical methods of getting a frame ID
    // in DXVK depend on say when the submit thread's present happens which is unpredictable).
    uint64_t m_reflexFrameId = 0;

    uint32_t m_maxBone = 0;

    const bool m_enableDrawCallConversion;
    bool m_rtxInjectTriggered = false;

    // While the window is inactive or minimized a game may draw whole frames without presenting them.
    // Frames are then ended at scene boundaries until the first present after the window is active again.
    std::atomic<bool> m_windowInactive { false };
    bool m_endingFramesWithoutPresent = false;
    uint32_t m_scenesThisFrame = 0;
    uint32_t m_scenesLastPresentedFrame = 0;

    bool m_forceGeometryCopy = false;
    DWORD m_texcoordIndex = 0;

    int m_activeOcclusionQueries = 0;

    Rc<DxvkBuffer> m_vsVertexCaptureData;

    fast_unordered_cache<Rc<DxvkSampler>> m_samplerCache;

    // Memoized geometry hashes, keyed by buildGeometryHashMemoizationKey(). Written from the
    // geometry worker threads once a hash completes and read from the main thread, hence the mutex.
    static constexpr size_t kMaxGeometryHashCacheEntries = 16384;
    std::unordered_map<XXH64_hash_t, GeometryHashes> m_geometryHashCache;
    mutable dxvk::mutex m_geometryHashCacheMutex;
    uint32_t m_geometryHashCacheHits = 0;
    uint32_t m_geometryHashCacheMisses = 0;
    // D3D9 frames seen (EndFrame), for rtx.geometryHashMemoSelfCheckFrames.
    uint32_t m_d3d9FrameIndex = 0;

    DxvkBufferSlice allocVertexCaptureBuffer(const VkDeviceSize size);

    // The capture target of an original draw that tryEarlySkip keeps. Nothing reads it, so its writes leave the capture
    // buffers of other draws intact.
    DxvkBufferSlice m_vertexCaptureSink;

    // [PerfD3D9] d3d9ThreadCpuMsPerFrame (rtx.profile.detailedLogging), see sampleThreadCpuForPerfLog().
    bool m_perfThreadCpuValid = false;
    DWORD m_perfThreadId = 0;
    uint64_t m_perfThreadCpu100ns = 0;
    void sampleThreadCpuForPerfLog();
    // Returns false when there is no implicit swapchain back buffer to end the frame on.
    bool endFrameWithoutPresent();
    static PrepareDrawFlags countPrepareOutcome(const PrepareDrawFlags flags);

    // rtx.textureCategoryCache: every texture-list lookup a draw makes for one texture hash.
    struct TextureListInfo {
      CategoryFlags categories = 0;     // DrawCallState::computeTextureListCategories() + SmoothNormals
      bool lightmap = false;            // rtx.lightmapTextures
      bool ignoreBakedLighting = false; // rtx.ignoreBakedLightingTextures
      bool ui = false;                  // rtx.uiTextures
    };
    static constexpr size_t kMaxTextureListCacheEntries = 65536;
    fast_unordered_cache<TextureListInfo> m_textureListCache;
    uint64_t m_textureListCacheGeneration = ~0ull;
    // Never null. The pointer is valid until the next call.
    const TextureListInfo* getTextureListInfo(const XXH64_hash_t& textureHash);

    // NOTE: to avoid calculating matrix inverse,
    //       m_seenCameraPositions doesn't contain the actual positions,
    //       but only relative values, see USE_TRUE_CAMERA_POSITION_FOR_COMPARISON
    std::vector<Vector3> m_seenCameraPositions;
    std::vector<Vector3> m_seenCameraPositionsPrev;

    struct IndexContext {
      VkIndexType indexType = VK_INDEX_TYPE_NONE_KHR;
      D3D9CommonBuffer* ibo = nullptr;
      DxvkBufferSliceHandle indexBuffer;
    };

    struct VertexContext {
      uint32_t stride = 0;
      uint32_t offset = 0;
      DxvkBufferSlice buffer;
      DxvkBufferSliceHandle mappedSlice;
      D3D9CommonBuffer* pVBO = nullptr;
      bool canUseBuffer;
    };

    static bool isPrimitiveSupported(const D3DPRIMITIVETYPE PrimitiveType) {
      return (PrimitiveType == D3DPT_TRIANGLELIST || PrimitiveType == D3DPT_TRIANGLEFAN || PrimitiveType == D3DPT_TRIANGLESTRIP);
    }

    const Direct3DState9& d3d9State() const;

    template<typename T>
    static void copyIndices(const uint32_t indexCount, T*& pIndicesDst, T* pIndices, uint32_t& minIndex, uint32_t& maxIndex);

    template<typename T>
    DxvkBufferSlice processIndexBuffer(const uint32_t indexCount, const uint32_t startIndex, const IndexContext& indexCtx, uint32_t& minIndex, uint32_t& maxIndex);

    void prepareVertexCapture(const int vertexIndexOffset);

    void processVertices(const VertexContext vertexContext[caps::MaxStreams], int vertexIndexOffset, RasterGeometry& geoData);

    bool processRenderState();

    template<bool FixedFunction>
    bool processTextures();

    PrepareDrawFlags internalPrepareDraw(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext);
    // internalPrepareDraw = head + body (the early change test runs between them). The head (post-injection
    // gate, makeDrawCallType) runs for every draw; false = stop with `flags`.
    bool prepareDrawHead(const DrawContext& drawContext, PrepareDrawFlags& flags);
    PrepareDrawFlags prepareDrawBody(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext);
    // processRenderState's clip plane and D3D9 light steps, also run for draws R1 skips.
    void updateClipPlanes();
    void emitDirtyLights();

    // D3D9 v2: describe a draw committed to ray tracing by ids and versions for the
    // frontend's app-thread half. Called after internalPrepareDraw, while m_activeDrawCallState still holds the draw.
    // flags: what the preparation returned (R1 stores the ApplyDrawState bit).
    void observeFrontendInputs(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams],
                               const DrawContext& drawContext, bool userPointer, PrepareDrawFlags flags);
    // The identity part of the frontend record (ids, versions, range, shaders, declaration, extras; no material).
    void fillFrontendIdentity(frontends::d3d9::D3D9DrawInputs& in, const IndexContext& indexContext,
                              const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext,
                              bool userPointer);
    // Fixed-function skinning: the world matrices the draw's vertices blend with, firstBone .. firstBone + boneCount - 1.
    // Returns false when the blend indices cannot be read on the host.
    bool usedBoneRange(const VertexContext vertexContext[caps::MaxStreams], const DrawContext& drawContext,
                       uint32_t& firstBone, uint32_t& boneCount);
    // usedBoneRange's scans, by vertex buffer version and range.
    static constexpr size_t kMaxBoneRangeCacheEntries = 4096;
    std::unordered_map<XXH64_hash_t, std::pair<uint32_t, uint32_t>> m_boneRangeCache;
    // The content of the index range the draw reads, seeded with the primitive count (D3D9DrawInputs::kIndexContent).
    // Returns 0 when the range is not host-mapped or lies outside the buffer.
    XXH64_hash_t hashIndexRange(const IndexContext& indexContext, const DrawContext& drawContext);
    struct IndexRangeHash {
      uint64_t resourceId = 0;
      uint64_t version = 0;
      VkDeviceSize offset = 0;
      VkDeviceSize size = 0;
      uint32_t primitiveCount = 0;
      XXH64_hash_t hash = 0;
    } m_lastIndexRange;
    // R1: the clip planes, stream frequencies and viewport, hashed (D3D9DrawInputs::extra).
    XXH64_hash_t hashDrawExtras() const;
    // R1: the pre-key over every D3D9 input the preparation reads.
    frontends::d3d9::EarlyKey buildEarlyKey(const frontends::d3d9::D3D9DrawInputs& identity, const Matrix4& objectToWorld) const;
    // R1: skip a draw of an object the scene owns before preparing it. True = handled (flags set).
    bool tryEarlySkip(const IndexContext& indexContext, const VertexContext vertexContext[caps::MaxStreams],
                      const DrawContext& drawContext, PrepareDrawFlags& flags);
    // The host asks that this draw stay on the per-draw path (FnvLegacyProtocol::kPerDrawPath in render state 42): the
    // frontend neither early-checks it, hands it to the scene nor takes its camera.
    bool hostKeepsPerDrawPath() const;
    // A per-draw-path draw that commits: its fog joins the frontend's frame fog order.
    void notePerDrawPath();
    // R2: sends the camera the frontend chose for this draw (first draw of its type in the frame).
    void sendFrontendCamera(const frontends::d3d9::Decision& decision, const Matrix4& worldToView, const Matrix4& viewToProjection);
    frontends::d3d9::D3D9FrontendApp m_frontendApp;
    // The frontend's verdict for the draw being prepared, consumed by CommitGeometryToRT.
    frontends::d3d9::Decision m_frontendDecision;
    // The last VS constants hash, reused while no VS constant was written and the shader is the same.
    uint64_t m_frontendVsConstantsVersion = 0;
    XXH64_hash_t m_frontendVsConstantsShader = 0;
    XXH64_hash_t m_frontendVsConstantsHash = 0;

    void triggerInjectRTX();


    struct DrawCallType {
      RtxGeometryStatus status;
      bool triggerRtxInjection;
    };
    DrawCallType makeDrawCallType(const DrawContext& drawContext);

    bool checkBoundTextureCategory(const fast_unordered_set& textureCategory) const;

    bool isRenderingUI();

    Future<SkinningData> processSkinning(const RasterGeometry& geoData);

    Future<AxisAlignedBoundingBox> computeAxisAlignedBoundingBox(const RasterGeometry& geoData);

    // `memoizationKey` identifies the complete set of inputs this hash is derived from. Pass 0 to
    // force a full re-hash (the behaviour before memoization existed).
    // With rtx.geometryHashMemoInline a memo hit fills geoData.hashes / hashesPrecomputed directly
    // and returns an empty Future.
    Future<GeometryHashes> computeHash(RasterGeometry& geoData, const uint32_t maxIndexValue, const XXH64_hash_t memoizationKey);

    // Builds the memoization key for the current draw call, or returns 0 when this draw is not
    // eligible for memoization.
    XXH64_hash_t buildGeometryHashMemoizationKey(const RasterGeometry& geoData,
                                                 const VertexContext vertexContext[caps::MaxStreams],
                                                 const IndexContext& indexContext,
                                                 const uint32_t startIndex,
                                                 const int vertexIndexOffset,
                                                 const uint32_t maxIndexValue) const;

    void submitActiveDrawCallState();
  };
}
