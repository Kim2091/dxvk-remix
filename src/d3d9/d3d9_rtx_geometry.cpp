#include <vector>
#include "d3d9_device.h"
#include "d3d9_rtx.h"
#include "d3d9_rtx_utils.h"
#include "d3d9_state.h"
#include "../dxvk/dxvk_buffer.h"
#include "../dxvk/dxvk_perf_counters.h"
#include "../dxvk/rtx_render/rtx_hashing.h"
#include "../util/util_fastops.h"

namespace dxvk {
  // Geometry indices should never be signed.  Using this to handle the non-indexed case for templates.
  typedef int NoIndices;

  namespace VertexRegions {
    enum Type : uint32_t {
      Position = 0,
      Texcoord,
      Count
    };
  }

  // NOTE: Intentionally leaving the legacy hashes out of here, because they are special (REMIX-656)
  // (Was a std::map looked up twice per component per draw on the geometry workers.)
  inline bool componentToRegion(const HashComponents component, VertexRegions::Type& regionOut) {
    switch (component) {
    case HashComponents::VertexPosition: regionOut = VertexRegions::Position; return true;
    case HashComponents::VertexTexcoord: regionOut = VertexRegions::Texcoord; return true;
    default: return false;
    }
  }

  bool getVertexRegion(const RasterBuffer& buffer, const size_t vertexCount, HashQuery& outResult) {
    ScopedCpuProfileZone();

    if (!buffer.defined())
      return false;

    outResult.pBase = (uint8_t*) buffer.mapPtr(buffer.offsetFromSlice());
    outResult.elementSize = imageFormatInfo(buffer.vertexFormat())->elementSize;
    outResult.stride = buffer.stride();
    outResult.size = outResult.stride * vertexCount;
    // Make sure we hold on to this reference while the hashing is in flight
    outResult.ref = buffer.buffer().ptr();
    assert(outResult.ref);
    return true;
  }

  // Sorts and deduplicates a set of integers, storing the result in a vector
  template<typename T>
  void deduplicateSortIndices(const void* pIndexData, const size_t indexCount, const uint32_t maxIndexValue, std::vector<T>& uniqueIndicesOut) {
    // TODO (REMIX-657): Implement optimized variant of this function
    // We know there will be at most, this many unique indices
    const uint32_t indexRange = maxIndexValue + 1;

    // Initialize all to 0
    uniqueIndicesOut.resize(indexRange, (T)0);

    // Use memory as a bin table for index data
    for (uint32_t i = 0; i < indexCount; i++) {
      const T& index = ((T*) pIndexData)[i];
      assert(index <= maxIndexValue);
      uniqueIndicesOut[index] = 1;
    }

    // Repopulate the bins with contiguous index values
    uint32_t uniqueIndexCount = 0;
    for (uint32_t i = 0; i < indexRange; i++) {
      if (uniqueIndicesOut[i])
        uniqueIndicesOut[uniqueIndexCount++] = i;
    }

    // Remove any unused entries
    uniqueIndicesOut.resize(uniqueIndexCount);
  }

  template<typename T>
  void hashGeometryData(const size_t indexCount, const uint32_t maxIndexValue, const void* pIndexData,
                        DxvkBuffer* indexBufferRef, const HashQuery vertexRegions[VertexRegions::Count], GeometryHashes& hashesOut) {
    ScopedCpuProfileZone();

    const HashRule& globalHashRule = RtxOptions::geometryHashGenerationRule();

    // One vector per geometry worker thread, reused across draws. clear() + resize(n, 0) zero-fills it.
    thread_local std::vector<T> tlsUniqueIndices;
    std::vector<T>& uniqueIndices = tlsUniqueIndices;
    uniqueIndices.clear();
    if constexpr (!std::is_same<T, NoIndices>::value) {
      assert((indexCount > 0 && indexBufferRef));
      deduplicateSortIndices(pIndexData, indexCount, maxIndexValue, uniqueIndices);

      if (globalHashRule.test(HashComponents::Indices)) {
        hashesOut[HashComponents::Indices] = hashContiguousMemory(pIndexData, indexCount * sizeof(T));
      }

      // TODO (REMIX-656): Remove this once we can transition content to new hash
      if (globalHashRule.test(HashComponents::LegacyIndices)) {
        hashesOut[HashComponents::LegacyIndices] = hashIndicesLegacy<T>(pIndexData, indexCount);
      }

      // Release this memory back to the staging allocator
      indexBufferRef->release(DxvkAccess::Read);
      indexBufferRef->decRef();
    }

    // Do vertex based rules
    for (uint32_t i = 0; i < (uint32_t) HashComponents::Count; i++) {
      const HashComponents& component = (HashComponents) i;

      VertexRegions::Type region = VertexRegions::Position;
      if (globalHashRule.test(component) && componentToRegion(component, region)) {
        hashesOut[component] = hashVertexRegionIndexed(vertexRegions[(uint32_t)region], uniqueIndices);
      }
    }

    // Don't keep a huge bin table alive on the worker after an unusually large draw.
    if (uniqueIndices.capacity() > (size_t(1) << 22)) {
      std::vector<T>().swap(uniqueIndices);
    }

    // TODO (REMIX-656): Remove this once we can transition content to new hash
    if (globalHashRule.test(HashComponents::LegacyPositions0) || globalHashRule.test(HashComponents::LegacyPositions1)) {
      hashRegionLegacy(vertexRegions[VertexRegions::Position], hashesOut[HashComponents::LegacyPositions0], hashesOut[HashComponents::LegacyPositions1]);
    }

    // Release this memory back to the staging allocator
    for (uint32_t i = 0; i < VertexRegions::Count; i++) {
      const HashQuery& region = vertexRegions[i];
      if (region.size == 0)
        continue;

      if (region.ref) {
        region.ref->release(DxvkAccess::Read);
        region.ref->decRef();
      }
    }
  }

  Future<GeometryHashes> D3D9Rtx::computeHash(RasterGeometry& geoData, const uint32_t maxIndexValue, const XXH64_hash_t memoizationKey) {
    ScopedCpuProfileZone();

    // rtx.geometryHashMemoSelfCheckFrames: on a self-check frame a memo hit is re-hashed in full, and
    // the worker compares the result with the memoized entry before replacing it.
    bool selfCheck = false;

    if (memoizationKey != 0) {
      std::lock_guard<dxvk::mutex> lock(m_geometryHashCacheMutex);
      auto it = m_geometryHashCache.find(memoizationKey);
      if (it != m_geometryHashCache.end()) {
        const uint32_t selfCheckFrames = geometryHashMemoSelfCheckFrames();
        selfCheck = selfCheckFrames != 0 && (m_d3d9FrameIndex % selfCheckFrames) == 0;
        if (!selfCheck) {
          ++m_geometryHashCacheHits;
          perf::countD3D9(perf::D3D9Ctr::MemoHits);

          if (geometryHashMemoInline()) {
            // Served inline: no worker round trip. finalizeGeometryHashes() takes `hashes` as is.
            geoData.hashes = it->second;
            geoData.hashesPrecomputed = true;
            return Future<GeometryHashes>();
          }

          // Future is single-consumption - get() clears its task pointer - so a stored Future cannot
          // be handed out twice. Schedule a task that simply returns the memoized value instead: the
          // dispatch is kept, but the buffer acquisition and the content hashing are both skipped.
          const GeometryHashes cachedHashes = it->second;
          return m_pGeometryWorkers->Schedule([cachedHashes]() -> GeometryHashes {
            return cachedHashes;
          });
        }
        perf::countD3D9(perf::D3D9Ctr::MemoSelfCheckRuns);
      } else {
        ++m_geometryHashCacheMisses;
        perf::countD3D9(perf::D3D9Ctr::MemoMisses);
      }
    } else {
      perf::countD3D9(perf::D3D9Ctr::MemoIneligible);
    }

    const uint32_t indexCount = geoData.indexCount;
    const uint32_t vertexCount = geoData.vertexCount;

    HashQuery vertexRegions[VertexRegions::Count];
    memset(&vertexRegions[0], 0, sizeof(vertexRegions));

    if (!getVertexRegion(geoData.positionBuffer, vertexCount, vertexRegions[VertexRegions::Position]))
      return Future<GeometryHashes>(); //invalid

    // Acquire prevents the staging allocator from re-using this memory
    vertexRegions[VertexRegions::Position].ref->acquire(DxvkAccess::Read);
    vertexRegions[VertexRegions::Position].ref->incRef();

    if (getVertexRegion(geoData.texcoordBuffer, vertexCount, vertexRegions[VertexRegions::Texcoord])) {
      vertexRegions[VertexRegions::Texcoord].ref->acquire(DxvkAccess::Read);
      vertexRegions[VertexRegions::Texcoord].ref->incRef();
    }

    // Make sure we hold a ref to the index buffer while hashing.
    const Rc<DxvkBuffer> indexBufferRef = geoData.indexBuffer.buffer();
    if (indexBufferRef.ptr()) {
      indexBufferRef->acquire(DxvkAccess::Read);
      indexBufferRef->incRef();
    }
    const void* pIndexData = geoData.indexBuffer.defined() ? geoData.indexBuffer.mapPtr(0) : nullptr;
    const size_t indexStride = geoData.indexBuffer.stride();
    const size_t indexDataSize = indexCount * indexStride;

    // Assume the GPU changed the data via shaders, include the constant buffer data in hash
    XXH64_hash_t vertexShaderHash = kEmptyHash;
    XXH64_hash_t vertexShaderBytecodeHash = kEmptyHash; // Side field for the D3D9 identity classifier only
    if (m_parent->UseProgrammableVS() && useVertexCapture()) {
      if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
        const D3D9ConstantSets& cb = m_parent->m_consts[DxsoProgramTypes::VertexShader];
        const D3D9CommonShader* pVertexShader = d3d9State().vertexShader->GetCommonShader();
        // Hashed once when the shader was created.
        vertexShaderHash = pVertexShader->GetBytecodeHash();
        vertexShaderBytecodeHash = vertexShaderHash;
        vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.fConsts[0], cb.meta.maxConstIndexF * sizeof(float) * 4, vertexShaderHash);
        vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.iConsts[0], cb.meta.maxConstIndexI * sizeof(int) * 4, vertexShaderHash);
        vertexShaderHash = XXH3_64bits_withSeed(&d3d9State().vsConsts.bConsts[0], cb.meta.maxConstIndexB * sizeof(uint32_t)/32, vertexShaderHash);
      }
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t geometryDescriptorHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::GeometryDescriptor)) {
      geometryDescriptorHash = hashGeometryDescriptor(geoData.indexCount, 
                                                      geoData.vertexCount, 
                                                      geoData.indexBuffer.indexType(), 
                                                      geoData.topology);
    }

    // Calculate this based on the RasterGeometry input data
    XXH64_hash_t vertexLayoutHash = kEmptyHash;
    if (RtxOptions::geometryHashGenerationRule().test(HashComponents::VertexLayout)) {
      vertexLayoutHash = hashVertexLayout(geoData);
    }

    return m_pGeometryWorkers->Schedule([this, memoizationKey, vertexRegions, indexBufferRef = indexBufferRef.ptr(),
                                 pIndexData, indexStride, indexDataSize, indexCount,
                                 maxIndexValue, vertexShaderHash, geometryDescriptorHash,
                                 vertexLayoutHash, selfCheck, vertexShaderBytecodeHash]() -> GeometryHashes {
      ScopedCpuProfileZone();

      GeometryHashes hashes;

      // Finalize the descriptor hash
      hashes[HashComponents::GeometryDescriptor] = geometryDescriptorHash;
      hashes[HashComponents::VertexLayout] = vertexLayoutHash;
      hashes[HashComponents::VertexShader] = vertexShaderHash;
      hashes.vertexShaderBytecode = vertexShaderBytecodeHash;

      // Index hash
      switch (indexStride) {
      case 2:
        hashGeometryData<uint16_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      case 4:
        hashGeometryData<uint32_t>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      default:
        hashGeometryData<NoIndices>(indexCount, maxIndexValue, pIndexData, indexBufferRef, vertexRegions, hashes);
        break;
      }

      assert(hashes[HashComponents::VertexPosition] != kEmptyHash);

      hashes.precombine();

      if (memoizationKey != 0) {
        std::lock_guard<dxvk::mutex> lock(m_geometryHashCacheMutex);
        // Bounded so a scene that streams unique geometry indefinitely cannot grow this without
        // limit. Clearing wholesale is acceptable because a miss only costs the hash we were
        // computing anyway.
        if (selfCheck) {
          // The fresh hash must equal the memoized one; a mismatch means a buffer write that did not
          // bump remixContentVersion (the memo would have served a stale hash). The entry is replaced below.
          // Compare the hash components only: GeometryHashes::precombined[3/4] are left uninitialized when
          // the legacy components are empty, so a whole-struct memcmp reports false mismatches.
          auto it = m_geometryHashCache.find(memoizationKey);
          if (it != m_geometryHashCache.end()) {
            std::string differing;
            for (uint32_t i = 0; i < (uint32_t) HashComponents::Count; i++) {
              const HashComponents component = (HashComponents) i;
              if (it->second[component] != hashes[component]) {
                differing += str::format(differing.empty() ? "" : ", ", getHashComponentName(component), " ",
                                         std::hex, it->second[component], " -> ", hashes[component]);
              }
            }
            if (!differing.empty()) {
              perf::countD3D9(perf::D3D9Ctr::MemoSelfCheckMismatches);
              static std::atomic<uint32_t> s_mismatchLogs { 0 };
              if (s_mismatchLogs.fetch_add(1, std::memory_order_relaxed) < 20) {
                Logger::warn(str::format("[GeometryHashMemoCheck] memoized geometry hash is stale: key ", std::hex, memoizationKey,
                                         " | ", differing, " (first 20 mismatches are logged)"));
              }
            }
          }
        }
        if (m_geometryHashCache.size() >= kMaxGeometryHashCacheEntries) {
          m_geometryHashCache.clear();
        }
        m_geometryHashCache[memoizationKey] = hashes;
      }

      return hashes;
    });
  }

  Future<AxisAlignedBoundingBox> D3D9Rtx::computeAxisAlignedBoundingBox(const RasterGeometry& geoData) {
    ScopedCpuProfileZone();

    if (!RtxOptions::needsMeshBoundingBox()) {
      return Future<AxisAlignedBoundingBox>();
    }

    const void* pVertexData = geoData.positionBuffer.mapPtr((size_t)geoData.positionBuffer.offsetFromSlice());
    const uint32_t vertexCount = geoData.vertexCount;
    const size_t vertexStride = geoData.positionBuffer.stride();

    if (pVertexData == nullptr) {
      return Future<AxisAlignedBoundingBox>();
    }

    auto vertexBuffer = geoData.positionBuffer.buffer().ptr();
    vertexBuffer->incRef();

    return m_pGeometryWorkers->Schedule([pVertexData, vertexCount, vertexStride, vertexBuffer]()->AxisAlignedBoundingBox {
      ScopedCpuProfileZone();

#if defined(_M_ARM64) || defined(_M_ARM64EC)
      float32x4_t minPos = vdupq_n_f32(FLT_MAX);
      float32x4_t maxPos = vdupq_n_f32(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        float32x4_t vertexPos;
        vertexPos.n128_f32[0] = pVertexPos->x;
        vertexPos.n128_f32[1] = pVertexPos->y;
        vertexPos.n128_f32[2] = pVertexPos->z;

        minPos = vminq_f32(minPos, vertexPos);
        maxPos = vmaxq_f32(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox {
        Vector3{ vgetq_lane_f32(minPos, 0), vgetq_lane_f32(minPos, 1), vgetq_lane_f32(minPos, 2) },
        Vector3{ vgetq_lane_f32(maxPos, 0), vgetq_lane_f32(maxPos, 1), vgetq_lane_f32(maxPos, 2) }
      };
#else
      __m128 minPos = _mm_set_ps1(FLT_MAX);
      __m128 maxPos = _mm_set_ps1(-FLT_MAX);

      const uint8_t* pVertex = static_cast<const uint8_t*>(pVertexData);
      for (uint32_t vertexIdx = 0; vertexIdx < vertexCount; ++vertexIdx) {
        const Vector3* const pVertexPos = reinterpret_cast<const Vector3* const>(pVertex);
        __m128 vertexPos = _mm_set_ps(0.0f, pVertexPos->z, pVertexPos->y, pVertexPos->x);
        minPos = _mm_min_ps(minPos, vertexPos);
        maxPos = _mm_max_ps(maxPos, vertexPos);

        pVertex += vertexStride;
      }

      AxisAlignedBoundingBox boundingBox{
        Vector3{ minPos.m128_f32[0], minPos.m128_f32[1], minPos.m128_f32[2] },
        Vector3{ maxPos.m128_f32[0], maxPos.m128_f32[1], maxPos.m128_f32[2] }
      };
#endif

      vertexBuffer->decRef();

      return boundingBox;
    });
  }
}
