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

#include <string>

#include "../rtx_render/rtx_option.h"

struct ImFont;

namespace dxvk {

  // The problem notice tells the player in one short sentence that something most likely broke and which file
  // to send. The technical details stay in a collapsed section. Problems come from the game integration
  // (rtx.faultNotice.hostReport) and from the runtime itself: a crash or a GPU crash in the previous session.
  class FaultNotice {
  public:
    RTX_OPTION("rtx.faultNotice", bool, enable, false,
      "Shows a notice on top of the game when a problem is detected: a likely conflict with another mod, a broken "
      "install, or a crash in the previous session. The notice stays until any Remix menu is opened.");
    RTX_OPTION("rtx.faultNotice", std::string, supportUrl, std::string(""),
      "The https link the problem notice offers for getting help, such as a support forum or chat invite.");
    RTX_OPTION("rtx.faultNotice", std::string, supportChannel, std::string(""),
      "Where the user menu tells the player to send a report file, such as a forum channel name. The game integration "
      "sets it through the Remix API's SetConfigVariable. Empty leaves the place unnamed.");
    RTX_OPTION_FLAG("rtx.faultNotice", std::string, hostReport, std::string(""), RtxOptionFlags::NoSave,
      "The problems the game integration detected, set through the Remix API's SetConfigVariable. Each line holds "
      "one problem written as id|text or id|text|details. The text is one short sentence for the player; the details "
      "are technical and stay in a collapsed section. The id identifies the problem for the dismissal.");
    RTX_OPTION_FLAG("rtx.faultNotice", std::string, uploadFile, std::string(""), RtxOptionFlags::NoSave,
      "The full path of the one file to attach to a bug report, such as a zip of the previous session's logs. The game "
      "integration sets it through the Remix API's SetConfigVariable. When set, the problem section of the user menu asks "
      "the player to send this file and gets a button that shows it in Explorer.");

    // Draws the notice on top of the game.
    // Opening a menu marks the problems shown so far as seen, which hides the notice until a new one arrives.
    static void showToast(ImFont* font, bool menuOpen);

    // Draws the user menu's help section: a way to report a problem, or the current problems with the file to send,
    // and the collapsed details.
    static void showSection();
  };

}
