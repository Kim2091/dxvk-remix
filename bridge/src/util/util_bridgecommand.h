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
#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <type_traits>
#include <vector>

#include "config/global_options.h"

#include "util_common.h"
#include "util_commands.h"
#include "util_circularbuffer.h"
#include "util_bridge_state.h"
#include "util_ipcchannel.h"
#include "util_perfbridge.h"
#include "util_scopedlock.h"
#include "util_singleton.h"
#include "../tracy/tracy.hpp"

extern bool gbBridgeRunning;

#define WAIT_FOR_SERVER_RESPONSE(func, value, uidVal) \
  { \
    const uint32_t timeoutMs = GlobalOptions::getAckTimeout(); \
    if (Result::Success != DeviceBridge::waitForCommand(Commands::Bridge_Response, timeoutMs, nullptr, true, uidVal)) { \
      Logger::err(func " failed with: no response from server."); \
      return value; \
    } \
  }

#define POP_BRIDGE_COMMAND_QUEUE() \
  { \
    DeviceBridge::pop_front(); \
  }

#define WAIT_FOR_OPTIONAL_SERVER_RESPONSE(func, value, uidVal) \
  { \
    if (GlobalOptions::getSendAllServerResponses()) { \
      WAIT_FOR_SERVER_RESPONSE(func, value, uidVal) \
      HRESULT res = (HRESULT) DeviceBridge::get_data(); \
      DeviceBridge::pop_front(); \
      return  res; \
    } else { \
      return D3D_OK; \
    } \
  } 

#define WAIT_FOR_OPTIONAL_CREATE_FUNCTION_SERVER_RESPONSE(func, value, uidVal) \
  { \
    if (GlobalOptions::getSendCreateFunctionServerResponses() || GlobalOptions::getSendAllServerResponses()) { \
      WAIT_FOR_SERVER_RESPONSE(func, value, uidVal) \
      HRESULT res = (HRESULT) DeviceBridge::get_data(); \
      DeviceBridge::pop_front(); \
      return  res; \
    } else { \
      return D3D_OK; \
    } \
  } 

using namespace bridge_util;

// Used structs over class enum for explicit type names while debugging
namespace BridgeId {
  struct _Bridge {
    virtual ~_Bridge() = 0; // Don't allow instantiation
  };
  struct Module : _Bridge {};
  struct Device : _Bridge {};
};
#define ASSERT_VALID_BRIDGE_ID(BRIDGE_ID) \
  static_assert(std::is_base_of<BridgeId::_Bridge, BRIDGE_ID>::value, "Must use valid BridgeId.");

template <typename BridgeId>
class Bridge {
  ASSERT_VALID_BRIDGE_ID(BridgeId);
  using DataT = uint32_t;
public:
  using ResponseTransaction = bridge_util::ResponseTransaction;

  static void init(
    const std::string baseName,
    const size_t writerChannelMemSize, const size_t writerChannelCmdQueueSize,
    const size_t writerChannelDataQueueSize, const size_t readerChannelMemSize,
    const size_t readerChannelCmdQueueSize, const size_t readerChannelDataQueueSize);
  static inline const WriterChannel& getWriterChannel() {
    return *s_pWriterChannel;
  }
  static inline const ReaderChannel& getReaderChannel() {
    return *s_pReaderChannel;
  }
  
  // [PerfBridge] (util_perfbridge.h): call once per frame on the presenting thread, after Present.
  static void perfOnFrame();

  //=========================//
  // Channel reading methods //
  //=========================//
  // Every read waits until the data it reads has been published (see "Streamed commands" below).
  static inline DataT get_data() {
    ZoneScoped;
    if (!awaitData()) {
      return 0;
    }
    size_t prevPos = get_data_pos();
    const DataT retval = getReaderChannel().data->pull();
    noteReadWrap(prevPos);
    return retval;
  }

  // Reads a variable-size item and returns its size in bytes.
  // The pointer stays valid until the command's data has been read to its end.
  static inline DataT get_data(void** obj) {
    return getItem(obj, true);
  }

  // Like get_data(void**), but the pointer stays valid only until the next read.
  // Saves a copy when a streamed command's item is consumed at once.
  static inline DataT get_data_view(void** obj) {
    return getItem(obj, false);
  }

  template<typename T>
  static inline size_t copy_data(T& obj, bool checkSize = true) {
    ZoneScoped;
    if (!awaitData()) {
      obj = T();
      return 0;
    }
    size_t prevPos = get_data_pos();
    const DataT retval = getReaderChannel().data->pull_and_copy(obj);

    if (checkSize) {
      assert((size_t) retval == sizeof(T) && "Size of source and target object does not match!");
      if ((size_t) retval != sizeof(T)) {
        Logger::err("DataQueue copy data: Size of source and target object does not match!");
      }
    }

    noteReadWrap(prevPos);
    return retval;
  }

  static inline size_t get_data_pos() {
    ZoneScoped;
    return getReaderChannel().data->get_pos();
  }

  static inline bridge_util::Result begin_read_data() {
    ZoneScoped;
    if (gbBridgeRunning) {
      return getReaderChannel().data->begin_batch();
    }
    return bridge_util::Result::Failure;
  }

  static inline size_t end_read_data() {
    ZoneScoped;
    if (gbBridgeRunning) {
      return getReaderChannel().data->end_batch();
    }
    return 0;
  }

  //===================//
  // Streamed commands //
  //===================//
  // A device bridge command whose data grows past maxItemBytes() is streamed. The writer pushes the command's header
  // with Commands::DataStreamed once the unpublished data would pass that size, and publishes every further part
  // with a Bridge_DataChunk header. The last chunk header carries Commands::DataStreamEnd and marks the end of the
  // data. The reader runs the command while it is still being written and releases each part of the data ring after
  // reading it. However much data a command carries, its unpublished part holds at most two items of the ring.
  // An item larger than maxItemBytes() travels as the words [kStreamedItem][size in bytes], followed by items of at
  // most maxItemBytes() that hold its bytes in order. The reader joins them in its own memory.
  static constexpr DataT kStreamedItem = 0xFFFFFFFFu;

  // Returns the largest item the writer puts in its data ring: 1 MB, or a sixteenth of a smaller ring.
  static size_t maxItemBytes() {
    return s_maxItemWords * sizeof(DataT);
  }

  // Sets up reading the data of the command whose header the reader has just received.
  // headerAtFront is true when that header stays at the front of the command queue until pop_front().
  static void beginReadCommand(const Header& header, bool headerAtFront);
  // Reads the rest of a streamed command's chunk headers.
  // Returns false when the reader did not read exactly the command's data.
  static bool endReadCommand();
  // Returns the data position the command's data has been published up to.
  static uint32_t readDataEnd() {
    return s_readEnd;
  }
  // Publishes the reader's data position and wakes a writer that waits for that space.
  static void releaseReadData();

  static Header pop_front();
  static void syncDataQueue(size_t expectedMemUsage, bool posResetOnLastIndex = false);
  static bridge_util::Result ensureQueueEmpty();

  //=========================//
  // Channel waiting methods //
  //=========================//
  // Waits for a command to appear in the command queue. Upon success the command will NOT be removed from queue
  // and client MUST pull the command header manually by using pop_front() or pull() methods. The queue will enter
  // into an unrecoverable state otherwise.
  static bridge_util::Result waitForCommand(const Commands::D3D9Command& command = Commands::Bridge_Any,
                                            DWORD overrideTimeoutMS = 0,
                                            std::atomic<bool>* const pbEarlyOutSignal = nullptr, bool verifyUID = false, UID uidToVerify=0);
  // Waits for a command to appear in the command queue. Upon success the command will be removed from the queue
  // and discarded.
  static bridge_util::Result waitForCommandAndDiscard(const Commands::D3D9Command& command = Commands::Bridge_Any,
                                                      DWORD overrideTimeoutMS = 0,
                                                      std::atomic<bool>* const pbEarlyOutSignal = nullptr, bool verifyUID = false, UID uidToVerify = 0) {
    auto result = waitForCommand(command, overrideTimeoutMS, pbEarlyOutSignal, verifyUID, uidToVerify);
    if (bridge_util::Result::Success == result) {
      pop_front();
    }
    return result;
  }
  
  // Writes a command's arguments into the data ring. Shared by Command and StateMessage (A3b).
  class DataWriter {
  public:
    inline void send_data(const DataT obj) {
      ZoneScoped;
      if (gbBridgeRunning) {
        prepareWrite(1);
        syncDataQueue(1, false);
        const auto result = s_pWriterChannel->data->push(obj);
        if (RESULT_FAILURE(result)) {
          // For now just log when things go wrong, but could use some robustness improvements
          Logger::err("DataQueue send_data: Failed to send data!");
        }
      }
    }

    inline void send_data(const DataT size, const void* obj) {
      ZoneScoped;
      if (gbBridgeRunning) {
        const size_t dataWords = (obj == nullptr) ? 0 : align<size_t>(size, sizeof(DataT)) / sizeof(DataT);
        if (dataWords > s_maxItemWords && s_streamable) {
          sendStreamedItem(size, obj);
          return;
        }
        const size_t memUsed = dataWords + 1;
        prepareWrite(memUsed);
        syncDataQueue(memUsed, true);
        const auto result = s_pWriterChannel->data->push(size, obj);
        if (RESULT_FAILURE(result)) {
          // For now just log when things go wrong, but could use some robustness improvements
          Logger::err("DataQueue send_data: Failed to send data object!");
        }
      }
    }

    template<typename... Ts>
    inline void send_many(const Ts... objs) {
      ZoneScoped;
      if (gbBridgeRunning) {
        size_t count = sizeof...(Ts);
        prepareWrite(count);
        syncDataQueue(count, false);
        const auto result = s_pWriterChannel->data->push_many(objs...);
        if (RESULT_FAILURE(result)) {
          // For now just log when things go wrong, but could use some robustness improvements
          Logger::err("DataQueue send_many: Failed to send multiple m_writerChanneldata items!");
        }
      }
    }

    // Returns memory for an item of the given size, to be filled before end_data_blob().
    // A blob larger than maxItemBytes() is staged in writer memory and streamed by end_data_blob().
    inline uint8_t* begin_data_blob(const size_t size) {
      ZoneScoped;
      uint8_t* blobPacketPtr = nullptr;
      if (gbBridgeRunning) {
        const size_t dataWords = align<size_t>(size, sizeof(DataT)) / sizeof(DataT);
        if (dataWords > s_maxItemWords && s_streamable) {
          s_blobStaging.resize(size);
          s_blobStaged = true;
          return s_blobStaging.data();
        }
        const size_t memUsed = dataWords + 1;
        prepareWrite(memUsed);
        syncDataQueue(memUsed, true);
        const auto result = s_pWriterChannel->data->begin_blob_push(size, blobPacketPtr);
        if (RESULT_FAILURE(result)) {
          // For now just log when things go wrong, but could use some robustness improvements
          Logger::err("DataQueue begin_data_blob: Failed to begin sending a data blob!");
        }
      }
      return blobPacketPtr;
    }

    inline void end_data_blob() {
      ZoneScoped;
      if (s_blobStaged) {
        s_blobStaged = false;
        if (gbBridgeRunning) {
          sendStreamedItem((DataT) s_blobStaging.size(), s_blobStaging.data());
        }
        std::vector<uint8_t>().swap(s_blobStaging);
        return;
      }
      if (gbBridgeRunning) {
        s_pWriterChannel->data->end_blob_push();
      }
    }

    static inline size_t get_uid() {
      return s_cmdUID;
    }

  private:
    // Sends an item larger than maxItemBytes() in pieces (see "Streamed commands").
    void sendStreamedItem(const DataT size, const void* obj) {
      send_many(kStreamedItem, size);
      const size_t pieceBytes = maxItemBytes();
      for (size_t offset = 0; offset < size && gbBridgeRunning; offset += pieceBytes) {
        const size_t bytes = std::min<size_t>(pieceBytes, size - offset);
        send_data((DataT) bytes, static_cast<const uint8_t*>(obj) + offset);
      }
    }
  };

  class Command : public DataWriter {
  public:
    Command(const Commands::D3D9Command command) :
      Command(command, NULL) {
    }
    Command(const Commands::D3D9Command command, uintptr_t pHandle) :
      Command(command, pHandle, 0) {
    }
    Command(const Commands::D3D9Command command,
            uintptr_t pHandle,
            const Commands::Flags commandFlags);
    ~Command();

    static inline size_t get_counter() {
      return s_cmdCounter;
    }

    static inline void reset_counter() {
      s_cmdCounter = 0;
    }
    
    static inline void print_data(std::string prefix, std::vector<Commands::D3D9Command>& commandList)       {
      for (const auto& currentCommand : commandList) {
        Logger::info(prefix + toString(currentCommand));
      }
    }

    static inline void print_writer_data_sent() {
      std::vector<Commands::D3D9Command> resultCommands = s_pWriterChannel->commands->getWriterQueueData();
      print_data("Command sent: ", resultCommands);
    }

    static inline void print_writer_data_received() {
      std::vector<Commands::D3D9Command> resultCommands = s_pWriterChannel->commands->getReaderQueueData();
      print_data("Command received: ", resultCommands);
    }

    static inline void print_reader_data_sent() {
      std::vector<Commands::D3D9Command> resultCommands = s_pReaderChannel->commands->getWriterQueueData();
      print_data("Command sent: ", resultCommands);
    }

    static inline void print_reader_data_received() {
      std::vector<Commands::D3D9Command> resultCommands = s_pReaderChannel->commands->getReaderQueueData();
      print_data("Command received: ", resultCommands);
    }

  private:
    const Commands::D3D9Command m_command;
    const uint32_t m_handle;
    const Commands::Flags m_commandFlags;
    int64_t m_perfStart = 0;  // [PerfBridge]: 0 when the counters are off or this is not a counted command
  };

  // One state setter, written as a record into the open state packet instead of
  // as its own command. Client and device bridge only.
  //
  // Wire format of a Bridge_StatePacket command (data ring, in order):
  //   UID, then per record: [setter command id][the setter's arguments, exactly as its own command would send them]
  // The packet's header (pushed when the packet closes) carries the device handle and the data end position; the
  // server applies records until it reaches that position, through the same handlers as the stand-alone commands
  // (Commands::IsBatchableState lists the setters allowed in a packet).
  //
  // A packet stays open across setters and closes before any other command is constructed (Command's constructor),
  // when a record for another device arrives, when it grows past kMaxPacketWords (or one item of a smaller ring) or
  // kMaxPacketRecords, or right after its record when batching is off (setStateBatching(false): one record per packet,
  // the per-command cost of before). So the server sees every setter before the next draw, Present, query or create,
  // in the original order. Records are invisible to the server until the header is pushed. A packet is never
  // streamed: a record is at most a few words plus the device's shader constant range.
  //
  // Toward path C: the packet is a typed record stream (record = type + payload) inside one command. Scene-interface
  // records can join it later as further types, with the same open / close discipline.
  class StateMessage : public DataWriter {
  public:
    StateMessage(const Commands::D3D9Command command, uintptr_t pHandle);
    ~StateMessage();
  private:
    const Commands::D3D9Command m_command;
  };

  // A3b: called once by the client device with client.batchState (and not sendAllServerResponses).
  static void setStateBatching(bool enabled) {
    s_stateBatching = enabled;
  }

private:
  Bridge() = delete;
  Bridge(const Bridge&) = delete;
  Bridge(const Bridge&&) = delete;

  // A3b: open state packet. Called with the writer channel's mutex held.
  static void openStatePacketLocked(uint32_t handle);
  static void closeStatePacketLocked();
  // Pushes a command header, retrying like ~Command; turns the bridge off when it cannot. Mutex held.
  static void pushHeaderLocked(Commands::D3D9Command command, Commands::Flags flags, uint32_t handle);

  // Returns the words written to the data ring since startPos.
  // A wrap skips the ring's tail, so this can overcount slightly.
  static size_t dataWordsSince(const int32_t startPos) {
    const size_t endPos = s_pWriterChannel->data->get_pos();
    const size_t start = (size_t) startPos;
    return endPos >= start ? endPos - start : endPos + s_pWriterChannel->data->get_total_size() - start;
  }

  // Publishes the streamable command's data written so far when the next write would make it larger than one item.
  static inline void prepareWrite(const size_t words) {
    if (s_streamable && s_curBatchStartPos >= 0) {
      const size_t pending = dataWordsSince(s_curBatchStartPos);
      if (pending != 0 && pending + words > s_maxItemWords) {
        publishLocked();
      }
    }
  }
  static void publishLocked();

  // Clears serverResetPosRequired when the read that started at prevPos wrapped around the queue.
  static inline void noteReadWrap(const size_t prevPos) {
    if (get_data_pos() < prevPos) {
      if constexpr (std::is_same_v<BridgeId, ::BridgeId::Device>) {
        // The reader has started a new lap. It publishes position 0 before it clears the flag, so a writer that
        // sees the flag clear never pairs it with a position from the reader's previous lap.
        *static_cast<volatile int64_t*>(getReaderChannel().serverDataPos) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
      }
      *static_cast<volatile bool*>(getReaderChannel().serverResetPosRequired) = false;
    }
  }

  static inline bool awaitData() {
    return !s_readStreamed || getReaderChannel().data->get_pos() != s_readEnd || awaitDataSlow();
  }
  static bool awaitDataSlow();
  static bool readNextChunk();
  static DataT readStreamedItem(void** obj);
  static void keepItem(void** obj, DataT size);
  static uint8_t* allocScratch(size_t size);
  static constexpr size_t kScratchPadding = 16;

  static inline DataT getItem(void** obj, const bool keep) {
    ZoneScoped;
    if (!awaitData()) {
      *obj = nullptr;
      return 0;
    }
    if (getReaderChannel().data->peek() == kStreamedItem) {
      return readStreamedItem(obj);
    }
    size_t prevPos = get_data_pos();
    const DataT size = getReaderChannel().data->pull(obj);
    noteReadWrap(prevPos);
    // The writer may reuse a streamed command's ring space as soon as the reader moves on to the next chunk.
    if (keep && s_readStreamed && *obj != nullptr) {
      keepItem(obj, size);
    }
    return size;
  }

  static constexpr size_t kMaxPacketWords = 64 * 1024;
  static constexpr uint32_t kMaxPacketRecords = 4096;
  // The largest item, in words, at the default ring sizes. Smaller rings use a sixteenth of the ring.
  static constexpr size_t kMaxItemWordsLimit = 256 * 1024;

  // Writer state of the command being written (see "Streamed commands").
  static inline size_t   s_maxItemWords = kMaxItemWordsLimit;
  static inline bool     s_streamable = false;
  static inline bool     s_streamHeaderPushed = false;
  static inline Commands::D3D9Command s_streamCommand = Commands::Bridge_Invalid;
  static inline Commands::Flags s_streamFlags = 0;
  static inline uint32_t s_streamHandle = 0;
  static inline std::vector<uint8_t> s_blobStaging;
  static inline bool     s_blobStaged = false;

  // Reader state of the command being read.
  static inline bool     s_readStreamed = false;
  static inline bool     s_readEnded = true;
  static inline bool     s_readHeaderAtFront = false;
  static inline uint32_t s_readEnd = 0;
  // Items of a streamed command, and joined streamed items, kept until the command's data has been read.
  static inline std::deque<std::unique_ptr<uint8_t[]>> s_readScratch;

  static inline WriterChannel* s_pWriterChannel = nullptr;
  static inline ReaderChannel* s_pReaderChannel = nullptr;
  static inline int32_t        s_curBatchStartPos = -1;
  static inline size_t         s_cmdCounter = 0;
  // UIDs are assigned to commands to tag the responses from server to allow misorder responses to be handled correctly
  static inline UID s_cmdUID = 0;
  // A3b state packet (client, device bridge)
  static inline bool     s_stateBatching = false;
  static inline bool     s_packetOpen = false;
  static inline uint32_t s_packetHandle = 0;
  static inline uint32_t s_packetRecords = 0;
#if defined(REMIX_BRIDGE_CLIENT)
  static constexpr char kWriterChannelName[] = "Client2Server";
  static constexpr char kReaderChannelName[] = "Server2Client";
#elif defined(REMIX_BRIDGE_SERVER)
  static constexpr char kWriterChannelName[] = "Server2Client";
  static constexpr char kReaderChannelName[] = "Client2Server";
#endif
};
