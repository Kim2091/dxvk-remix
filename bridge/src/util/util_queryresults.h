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
#pragma once

// Non-blocking D3D9 queries through the bridge.
//
// One shared block, mapped by both processes:
// - issuedProcessedUid: the client command UID of the last IDirect3DQuery9::Issue the server has executed. The client
//   answers D3DQUERYTYPE_EVENT itself: the event is signalled once this has reached the UID of its own Issue.
// - slots: results of the other query types. The client sends GetData without waiting (FlagBits::AsyncResult); the
//   server runs the real GetData and writes hresult + data, then publishes the request's sequence number. The client
//   returns S_FALSE until its sequence number appears, and sends a new request after an S_FALSE answer.
// - hostSettingsGeneration: the low 32 bits of the runtime's host-settings generation, written by the server at each
//   Present once the client has polled host settings. remixapi_PollHostSettings answers an unchanged generation from it
//   without a round trip.
//
// Layout is identical in the 32-bit client and the 64-bit server (only 32-bit fields).

#include <atomic>
#include <cstdint>

namespace bridge_util {

  struct QueryResultSlot {
    static constexpr uint32_t kMaxDataSize = 32;

    std::atomic<uint32_t> answeredSeq;  // request sequence number the slot holds the answer for (0 = none)
    int32_t hresult;
    uint32_t size;
    uint8_t data[kMaxDataSize];
  };
  static_assert(sizeof(QueryResultSlot) == 44, "QueryResultSlot must have the same layout in both processes");

  struct QuerySharedBlock {
    static constexpr uint32_t kMaxSlots = 4096;

    std::atomic<uint32_t> issuedProcessedUid;
    std::atomic<uint32_t> hostSettingsGeneration;
    std::atomic<uint32_t> hostSettingsPublished;  // nonzero once hostSettingsGeneration is written
    uint32_t pad[13];  // keep the slots off the counters' cache line
    QueryResultSlot slots[kMaxSlots];
  };

  class QueryResults {
  public:
    static constexpr uint32_t kNoSlot = ~0u;

    // Maps the shared block. Both sides, after the bridge GUID is known (initDeviceBridge).
    static void init();

    // nullptr when the mapping failed: callers fall back to blocking queries.
    static QuerySharedBlock* block();

    // Client: slot per query object, kNoSlot when all are in use (that query falls back to blocking GetData).
    static uint32_t allocSlot();
    static void freeSlot(uint32_t slot);

    // Client: request sequence numbers, never 0.
    static uint32_t nextSeq();
  };

  // Wrap-safe "a is at or after b" for 32-bit UIDs.
  inline bool uidReached(uint32_t a, uint32_t b) {
    return static_cast<int32_t>(a - b) >= 0;
  }
}
