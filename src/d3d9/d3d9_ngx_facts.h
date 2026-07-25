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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dxvk {

  /**
   * \brief NGX passthrough fact log
   *
   * A deterministic, diffable record of every decision the passthrough heuristics reached
   * about the running game - which camera provider fired, the learned determinant sign, the
   * scene targets, the injection point, the velocity route.
   *
   * The problem it solves: this mode brings up games by MEASURING what they do rather than by
   * branching on which engine they are, so a change to any heuristic can silently move the
   * answer for a game nobody re-tested. Games that are not installed cannot be re-tested at
   * all. Run a game, change code, run it again, diff two of these files: only the intended
   * line should move. That makes "did I break the other games" a question with an answer
   * instead of a hope.
   *
   * Two design rules make the diff meaningful:
   * - the [facts] block carries NO frame numbers, timings or counters, so two runs of
   *   unchanged code produce byte-identical blocks even at different framerates or route
   *   lengths;
   * - instability is reported separately, as a change count plus a coarse settle bucket,
   *   because a fact that keeps flipping is itself a bug (it is what the ME2 startup flicker
   *   looks like from here) and a run-to-run exact frame number would only add noise.
   *
   * Not thread safe: owned and driven by the D3D9 app thread.
   */
  class NgxFactLog {

  public:

    /**
      * \brief Record a fact's current value, keyed by a stable dotted name (e.g. "camera.source").
      *
      * Idempotent: re-setting the same value only refreshes liveness. A different value counts
      * as a transition and marks the log dirty.
      */
    void set(const char* key, const std::string& value, uint32_t frameId);
    void setInt(const char* key, int64_t value, uint32_t frameId);
    void setBool(const char* key, bool value, uint32_t frameId);

    /**
      * \brief Write (overwrite) the summary file in the Remix log directory.
      *
      * Overwrite-in-place rather than append-on-exit so the file is always current even when
      * the game is killed or crashes, which is the normal way these sessions end. Does nothing
      * when no fact has changed since the last write.
      */
    void flush();

    /**
      * \brief Drop everything (device reset / mode change).
      */
    void reset();

    bool empty() const {
      return m_facts.empty();
    }

    /**
      * \brief Whether any fact changed since the last flush.
      */
    bool dirty() const {
      return m_dirty;
    }

    /**
      * \brief Facts whose value has changed since it was first established.
      *
      * A settled game reports zero. Anything else is a heuristic that cannot make its mind up,
      * which is worth surfacing in the developer panel: it predicts exactly the artifacts
      * (upscaler history resets, injection-point flips) that are otherwise diagnosed by eye.
      */
    uint32_t unstableFactCount() const;

    /**
      * \brief Rendered summary, same text as the file. For the developer panel.
      */
    std::string format() const;

  private:

    struct Fact {
      std::string value;
      uint32_t firstSetFrame = 0;
      uint32_t lastChangeFrame = 0;
      // Transitions AFTER the first establishment; 0 means the heuristic decided once and held
      uint32_t changeCount = 0;
    };

    // Ordered, so the output does not depend on hash iteration order: the file has to be
    // byte-stable across runs to be worth diffing
    std::map<std::string, Fact> m_facts;

    // Chronological transition record, for reading rather than diffing. Bounded: a genuinely
    // oscillating fact would otherwise grow this without limit.
    struct Transition {
      std::string key;
      std::string from;
      std::string to;
      uint32_t frameId = 0;
    };
    std::vector<Transition> m_transitions;
    static constexpr size_t kMaxTransitions = 256;
    uint32_t m_transitionsDropped = 0;

    bool m_dirty = false;

    // How long a fact took to settle, bucketed. Exact frame numbers differ every run (loading
    // times, menus, where the player stood); the bucket is stable while still separating
    // "decided immediately" from "took a thousand frames of flip-flopping".
    static const char* settleBucket(uint32_t frameId);
  };

}
