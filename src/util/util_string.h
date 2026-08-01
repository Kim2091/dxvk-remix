/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <string_view>
#include <sstream>
#include <locale>
#include <vector>

#include "./com/com_include.h"

// Converts a string defined by a macro e.g. #define FOO "bar" to a wide string, e.g. L"bar"
#define WIDEN_MACRO_LITERAL2(x) L ## x
#define WIDEN_MACRO_LITERAL(x) WIDEN_MACRO_LITERAL2(x)

namespace dxvk::str {

  std::string fromws(const WCHAR *ws);

  void tows(const char* mbs, WCHAR* wcs, size_t wcsLen);

  template <size_t N>
  void tows(const char* mbs, WCHAR (&wcs)[N]) {
    return tows(mbs, wcs, N);
  }

  std::wstring tows(const char* mbs);

  template<typename... Tx>
  void format1(std::stringstream& str, const WCHAR* arg, const Tx&... args);
  template<typename T, typename... Tx>
  void format1(std::stringstream& str, const T& arg, const Tx&... args);

  inline void format1(std::stringstream&) { }

  template<typename... Tx>
  void format1(std::stringstream& str, const WCHAR *arg, const Tx&... args) {
    str << fromws(arg);
    format1(str, args...);
  }

  template<typename T, typename... Tx>
  void format1(std::stringstream& str, const T& arg, const Tx&... args) {
    str << arg;
    format1(str, args...);
  }

  template<typename... Args>
  std::string format(const Args&... args) {
    std::stringstream stream;
    // A stream is constructed with the *global* locale, and a host application
    // can move that out from under us: Dolphin calls std::locale::global() with
    // the user's locale at startup (UICommon.cpp), which made every number this
    // formatter emits pick up digit grouping — hashes printed and copied out of
    // the dev menu came back as "1,AC8,CA7,5E4,0AA,123" and rtx.conf saved
    // "25,000, 25,000, 25,000". The read side is std::stoull/strtoull, which is
    // C-locale and stops at the first separator, so those values did not round
    // trip. Pin the classic locale; for a host that never touches the global
    // one this is exactly the behaviour we already had.
    stream.imbue(std::locale::classic());
    format1(stream, args...);
    return stream.str();
  }

  std::vector<std::string> split(std::string s, const char delimiter = ',');

  // Escape a string into a single-line, double-quoted string literal using
  // C-style backslash escapes. Suitable for JSON and USD string values.
  std::string escapeCStyle(std::string_view input);

  std::string sanitizeUtf8(const std::string& input);

  std::string formatBytes(size_t bytes);

  // Note: Constructs a string view including the null terminator unlike the standard library's std::basic_string_view
  // which does not include the null terminator. This makes it easier to work with APIs that expect null terminated strings
  // using string views.
  template<typename T, std::size_t L>
  constexpr auto string_viewz(const T(&t)[L]) {
    return std::basic_string_view<T>(t, L);
  }
}
