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

#include <atomic>
#include <memory>
#include <vector>

#include "rtx_scene_interface.h"

namespace dxvk {
  namespace scene {

    struct SceneCommandBatch {
      std::atomic<SceneCommandBatch*> next { nullptr };
      std::vector<SceneCommand> commands;
    };

    // Multi-producer, single-consumer queue of command batches (Vyukov's intrusive MPSC queue). Any API thread pushes
    // without a lock; only the render (CS) thread pops. Batches from one thread stay in that thread's order.
    class SceneCommandQueue {
    public:
      SceneCommandQueue() {
        m_head.store(&m_stub, std::memory_order_relaxed);
        m_tail = &m_stub;
      }

      ~SceneCommandQueue() {
        while (popBatch() != nullptr) {
        }
      }

      SceneCommandQueue(const SceneCommandQueue&) = delete;
      SceneCommandQueue& operator=(const SceneCommandQueue&) = delete;

      void push(std::unique_ptr<SceneCommandBatch> batch) {
        pushNode(batch.release());
        m_pushed.fetch_add(1, std::memory_order_relaxed);
      }

      // Null when empty, or when a producer is between its two stores (the batch arrives with the next pop).
      std::unique_ptr<SceneCommandBatch> popBatch() {
        SceneCommandBatch* tail = m_tail;
        SceneCommandBatch* next = tail->next.load(std::memory_order_acquire);
        if (tail == &m_stub) {
          if (next == nullptr) {
            return nullptr;
          }
          m_tail = next;
          tail = next;
          next = next->next.load(std::memory_order_acquire);
        }
        if (next != nullptr) {
          m_tail = next;
          return std::unique_ptr<SceneCommandBatch>(tail);
        }
        if (tail != m_head.load(std::memory_order_acquire)) {
          return nullptr;
        }
        pushNode(&m_stub);
        next = tail->next.load(std::memory_order_acquire);
        if (next != nullptr) {
          m_tail = next;
          return std::unique_ptr<SceneCommandBatch>(tail);
        }
        return nullptr;
      }

      // Batches pushed so far; lets a drain skip the queue without touching its nodes.
      uint64_t pushedCount() const { return m_pushed.load(std::memory_order_acquire); }

    private:
      void pushNode(SceneCommandBatch* node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        SceneCommandBatch* previous = m_head.exchange(node, std::memory_order_acq_rel);
        previous->next.store(node, std::memory_order_release);
      }

      std::atomic<SceneCommandBatch*> m_head;
      SceneCommandBatch* m_tail;  // consumer only
      SceneCommandBatch m_stub;
      std::atomic<uint64_t> m_pushed { 0 };
    };

  } // namespace scene
} // namespace dxvk
