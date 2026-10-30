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
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rtx_common_object.h"
#include "rtx_option.h"
#include "../../util/rc/util_rc_ptr.h"

namespace dxvk {

  class DxvkContext;
  class DxvkImage;

  // Saves the presented frame as PNG files: the game's back buffer with the game's own interface, without the
  // Remix menu, the DXVK HUD and frame generation. The copy is recorded at present, and a worker thread waits for
  // the GPU, converts the pixels and writes the files, so the frame waits for neither.
  // In SDR one 8-bit PNG is written. With HDR10 output active, an HDR PNG (16-bit BT.2020 PQ with a PQ ICC profile)
  // and an SDR PNG converted from it are written.
  // It also saves the bug report screenshots that rtx.diagnostics.screenshotRequest asks for, as one SDR PNG.
  class RtxScreenshot : public CommonDeviceObject {
  public:
    inline static const VirtualKeys kDefaultHotkey { VirtualKey { VK_SNAPSHOT } };
    RTX_OPTION_ARGS("rtx.screenshot", VirtualKeys, hotkey, kDefaultHotkey,
                    "Takes a screenshot, as the user menu's Take Screenshot button does. The image is the game's frame "
                    "with its own interface, without the Remix menu. With HDR output active, an HDR PNG and an SDR PNG "
                    "are saved.\n"
                    "Example override: 'rtx.screenshot.hotkey = CTRL, F12'. Key names are listed in "
                    "src/util/util_keybind.h.",
                    args.flags = RtxOptionFlags::UserSetting);
    RTX_OPTION_ARGS("rtx.screenshot", std::string, folder, std::string(""),
                    "The folder screenshots are saved to. A relative path is relative to the game's executable. "
                    "Empty = Pictures\\RTX Remix\\<the executable's name>, outside the game folder, so a mod "
                    "manager's virtual or root file handling does not move or hide them (a Screenshots folder next "
                    "to the game's executable when the Pictures folder cannot be found).",
                    args.flags = RtxOptionFlags::UserSetting);

    RTX_OPTION_FLAG("rtx.screenshot", std::string, thumbnailRequest, std::string(""), RtxOptionFlags::NoSave,
                    "Set by the game integration (remix API SetConfigVariable): each new value asks for a thumbnail of "
                    "the latest ray traced frame, taken where the frame is written to the game's render target, so it has "
                    "none of the game's interface. While frames are not ray traced (a game menu over a frozen "
                    "background), that is the last one that was. The value is answered at the next present. "
                    "The thumbnail is written to rtx-remix\\thumbnail.rgb: the bytes "
                    "'RTHB', then the width, the height and the length of the value as 32-bit little-endian numbers, "
                    "the value, then 8-bit RGB rows from the top, SDR (an HDR frame is converted as the SDR file of an "
                    "HDR screenshot is).");
    RTX_OPTION("rtx.screenshot", uint32_t, thumbnailWidth, 512, "The width of the thumbnail rtx.screenshot.thumbnailRequest "
               "asks for. The height follows the frame's aspect ratio.");

    explicit RtxScreenshot(DxvkDevice* device);
    ~RtxScreenshot();

    // Asks for a screenshot of the next presented frame.
    void request();

    // Records the copy of the back buffer when a screenshot is asked for, and queues the files to be written.
    // Also answers a new rtx.screenshot.thumbnailRequest from the latest ray traced frame.
    // Call it at present, after the frame's own rendering and before the Remix menu or the DXVK HUD is drawn
    // into the back buffer. hdrOutput tells whether the back buffer holds HDR10 output (rtx_tone_mapping.h).
    void capture(DxvkContext* ctx, const Rc<DxvkImage>& backBuffer, bool hdrOutput);

    // Keeps a downscaled copy of the ray traced frame for rtx.screenshot.thumbnailRequest. Call it right after the
    // frame is written to the game's render target, before the game draws its interface into it.
    void captureThumbnail(DxvkContext* ctx, const Rc<DxvkImage>& image);

    // Returns the message the HUD shows while a screenshot is saved and shortly after.
    // Returns false when there is none.
    bool getHudMessage(std::string& text, std::string& subText) const;

    // Returns the folder rtx.screenshot.folder names, as an absolute path.
    static std::string resolveFolder();

  private:
    struct Job;

    void serviceThumbnail(DxvkContext* ctx);
    void runWorker();
    void write(Job& job);
    void setHudMessage(std::string text, std::string subText, double seconds);

    std::atomic<bool> m_requested { false };
    // Render thread only.
    std::string m_handledThumbnailRequest;
    // The latest ray traced frame, downscaled, and how to convert it (null until a frame was ray traced).
    Rc<DxvkImage> m_thumbnailImage;
    std::unique_ptr<Job> m_thumbnailFrame;
    bool m_thumbnailWarned = false;
    std::atomic<bool> m_stopping { false };

    // The worker thread starts with the first screenshot and writes the jobs in order.
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;
    std::deque<std::unique_ptr<Job>> m_queue;
    std::thread m_worker;

    mutable std::mutex m_hudMutex;
    std::string m_hudText;
    std::string m_hudSubText;
    std::chrono::steady_clock::time_point m_hudUntil;
  };

}
