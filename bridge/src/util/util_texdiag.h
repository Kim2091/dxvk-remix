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

// [TexDiag] logs a fingerprint of each texture upload on both sides of the bridge.
// This compares what the game wrote on the client with what the runtime's D3D9 device received on the server.
//
// The bridge.conf option "textureUploadDiagnostics = N" logs every IDirect3DSurface9 UnlockRect upload whose locked
// rect is at least N pixels wide. 0 turns it off and is the default. Both lines carry the command UID, so a client
// line and its server line pair up exactly:
//
//   [TexDiag] client uid=… surf=… fmt=… rect=… flags=… rows=… rowBytes=… hash=0x… sameAsPrev=… sameAsFirst=…
//             zeroRows=… firstRepeatRow=…
//   [TexDiag] server uid=… with the same fields
//
// - hash is an FNV-1a 64 hash of the rect's rows in order, rowBytes per row without the pitch padding,
//   so both sides hash the same bytes.
// - rows counts block rows for BC formats (4 pixel rows each) and pixel rows otherwise.
// - sameAsPrev counts rows that are byte-identical to the row above.
// - sameAsFirst counts rows that are identical to row 0.
// - zeroRows counts rows that are all zero.
// - firstRepeatRow is the first row that matches the row above, or -1 when there is none.
// Real texture content has almost none of these repeats. Corruption where every row looks the same shows
// sameAsPrev close to rows.
//
// When the option is off, each unlock costs one option read.

#include "log/log.h"

#include <cstdint>
#include <cstring>

namespace bridge_util {
  namespace texdiag {

    struct RowStats {
      uint64_t hash = 0xcbf29ce484222325ull;  // FNV-1a 64 offset basis
      uint32_t rows = 0;
      uint32_t sameAsPrev = 0;
      uint32_t sameAsFirst = 0;
      uint32_t zeroRows = 0;
      uint32_t firstRepeatRow = UINT32_MAX;   // first row identical to the row above
    };

    inline bool isZeroRow(const uint8_t* const row, const size_t rowBytes) {
      for (size_t i = 0; i < rowBytes; ++i) {
        if (row[i] != 0) {
          return false;
        }
      }
      return true;
    }

    // Analyzes numRows rows that start at base and lie pitch bytes apart.
    // The first rowBytes bytes of each row are hashed and compared.
    inline RowStats analyzeRows(const uint8_t* const base, const size_t pitch, const size_t rowBytes,
                                const uint32_t numRows) {
      RowStats s;
      s.rows = numRows;
      if (base == nullptr || rowBytes == 0) {
        return s;
      }
      const uint8_t* const first = base;
      for (uint32_t y = 0; y < numRows; ++y) {
        const uint8_t* const row = base + size_t(y) * pitch;
        for (size_t i = 0; i < rowBytes; ++i) {
          s.hash ^= row[i];
          s.hash *= 0x100000001b3ull;  // FNV-1a 64 prime
        }
        if (y > 0) {
          if (std::memcmp(row, row - pitch, rowBytes) == 0) {
            ++s.sameAsPrev;
            if (s.firstRepeatRow == UINT32_MAX) {
              s.firstRepeatRow = y;
            }
          }
          if (std::memcmp(row, first, rowBytes) == 0) {
            ++s.sameAsFirst;
          }
        }
        if (isZeroRow(row, rowBytes)) {
          ++s.zeroRows;
        }
      }
      return s;
    }

    inline void logRowStats(const char* const side, const uint32_t uid, const uint32_t surfaceHandle,
                            const uint32_t format, const long left, const long top, const long right,
                            const long bottom, const uint32_t flags, const size_t rowBytes, const RowStats& s) {
      // A rect whose rows are mostly repeats or zeros is likely corrupt, so it is logged as a warning.
      const bool suspicious = s.rows >= 8 && (s.sameAsPrev * 2 >= s.rows || s.zeroRows * 2 >= s.rows);
      const std::string line = format_string(
        "[TexDiag] %s uid=%u surf=0x%x fmt=0x%x rect=(%ld,%ld,%ld,%ld) flags=0x%x rows=%u rowBytes=%llu "
        "hash=0x%016llx sameAsPrev=%u sameAsFirst=%u zeroRows=%u firstRepeatRow=%d%s",
        side, uid, surfaceHandle, format, left, top, right, bottom, flags, s.rows,
        static_cast<unsigned long long>(rowBytes), static_cast<unsigned long long>(s.hash), s.sameAsPrev,
        s.sameAsFirst, s.zeroRows, s.firstRepeatRow == UINT32_MAX ? -1 : static_cast<int>(s.firstRepeatRow),
        suspicious ? " SUSPICIOUS" : "");
      if (suspicious) {
        Logger::warn(line);
      } else {
        Logger::info(line);
      }
    }

  }
}
