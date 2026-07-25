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

#include "d3d9_ngx_facts.h"

#include <fstream>
#include <sstream>

#include "../util/log/log.h"
#include "../util/util_env.h"
#include "../util/util_filesys.h"
#include "../util/util_once.h"
#include "../util/util_string.h"

namespace dxvk {

  namespace {
    // Widest key the report aligns to. Keys longer than this simply push their value right,
    // which costs alignment on one line rather than truncating a name.
    constexpr size_t kKeyColumnWidth = 34;

    void appendPadded(std::stringstream& out, const std::string& text, size_t width) {
      out << text;
      for (size_t i = text.size(); i < width; ++i) {
        out << ' ';
      }
    }
  }

  const char* NgxFactLog::settleBucket(uint32_t frameId) {
    if (frameId < 10) {
      return "immediate";
    }
    if (frameId < 100) {
      return "<100 frames";
    }
    if (frameId < 1000) {
      return "<1000 frames";
    }
    return ">=1000 frames";
  }

  void NgxFactLog::set(const char* key, const std::string& value, uint32_t frameId) {
    auto [it, inserted] = m_facts.try_emplace(key);
    Fact& fact = it->second;

    if (inserted) {
      fact.value = value;
      fact.firstSetFrame = frameId;
      fact.lastChangeFrame = frameId;
      m_dirty = true;
      return;
    }

    if (fact.value == value) {
      return;
    }

    // A transition. Record it before overwriting: which value it came FROM is most of the
    // diagnostic value (pre-post -> late is a different bug from late -> pre-post).
    if (m_transitions.size() < kMaxTransitions) {
      m_transitions.push_back(Transition { key, fact.value, value, frameId });
    } else {
      ++m_transitionsDropped;
    }

    fact.value = value;
    fact.lastChangeFrame = frameId;
    ++fact.changeCount;
    m_dirty = true;
  }

  void NgxFactLog::setInt(const char* key, int64_t value, uint32_t frameId) {
    set(key, std::to_string(value), frameId);
  }

  void NgxFactLog::setBool(const char* key, bool value, uint32_t frameId) {
    set(key, value ? "yes" : "no", frameId);
  }

  void NgxFactLog::reset() {
    m_facts.clear();
    m_transitions.clear();
    m_transitionsDropped = 0;
    m_dirty = false;
  }

  uint32_t NgxFactLog::unstableFactCount() const {
    uint32_t count = 0;
    for (const auto& [key, fact] : m_facts) {
      if (fact.changeCount > 0) {
        ++count;
      }
    }
    return count;
  }

  std::string NgxFactLog::format() const {
    std::stringstream out;

    out << "# NGX passthrough fact log\n"
        << "# exe: " << env::getExeName() << "\n"
        << "#\n"
        << "# What the passthrough heuristics MEASURED about this game. Diff this file between\n"
        << "# two runs to see what a code change moved; the [facts] block below is frame-number\n"
        << "# free and byte-stable, so an unrelated change should produce no diff at all.\n"
        << "\n";

    out << "[facts]\n";
    for (const auto& [key, fact] : m_facts) {
      appendPadded(out, key, kKeyColumnWidth);
      out << "= " << fact.value << "\n";
    }
    out << "\n";

    // Stability. Informational, not a diff target: counts and buckets legitimately vary with
    // how long the session ran and where the player went.
    const uint32_t unstable = unstableFactCount();
    out << "[stability]\n";
    if (unstable == 0) {
      out << "# every fact was decided once and held.\n";
    } else {
      out << "# " << unstable << " fact(s) changed after first being established. A fact that keeps\n"
          << "# flipping is a bug in its own right: it means a heuristic is re-deciding mid-session,\n"
          << "# which is what upscaler history resets and injection-point flicker look like from here.\n";
    }
    for (const auto& [key, fact] : m_facts) {
      if (fact.changeCount == 0) {
        continue;
      }
      appendPadded(out, key, kKeyColumnWidth);
      out << "changes=" << fact.changeCount
          << " settled=" << settleBucket(fact.lastChangeFrame) << "\n";
    }
    out << "\n";

    if (!m_transitions.empty()) {
      out << "[transitions]\n";
      for (const Transition& transition : m_transitions) {
        out << "frame " << transition.frameId << "  " << transition.key
            << ": " << (transition.from.empty() ? "(unset)" : transition.from)
            << " -> " << transition.to << "\n";
      }
      if (m_transitionsDropped > 0) {
        out << "... and " << m_transitionsDropped << " more (record capped)\n";
      }
    }

    return out.str();
  }

  void NgxFactLog::flush() {
    if (!m_dirty || m_facts.empty()) {
      return;
    }
    m_dirty = false;

    const std::filesystem::path path =
      util::RtxFileSys::path(util::RtxFileSys::Logs) / "ngx-facts.log";

    std::optional<std::ofstream> file = util::createDirectoriesAndOpenFile(path);
    if (!file.has_value() || !file->is_open()) {
      // Once: a read-only install directory is a normal condition (Program Files) and must not
      // turn into a per-frame log spam
      ONCE(Logger::warn(str::format("[RTX NGX Passthrough] Could not write the fact log to ",
                                    path.string(), " - diagnostics only, rendering is unaffected.")));
      return;
    }

    *file << format();
  }

}
