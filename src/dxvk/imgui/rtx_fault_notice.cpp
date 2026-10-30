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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <windows.h>
#include <shellapi.h>

#include "rtx_fault_notice.h"
#include "dxvk_imgui.h"
#include "imgui.h"
#include "../rtx_render/rtx_imgui.h"
#include "../../util/log/log.h"
#include "../../util/util_crash_report.h"
#include "../../util/util_filesys.h"
#include "../../util/util_sentry.h"
#include "../../util/util_string.h"

namespace dxvk {

  namespace {
    struct Problem {
      std::string id;
      std::string text;
    };

    const ImVec4 kTitleColor(1.0f, 0.62f, 0.2f, 1.0f);
    const ImVec4 kLinkColor(0.4f, 0.7f, 1.0f, 1.0f);

    bool s_started = false;
    std::vector<Problem> s_problems;
    size_t s_runtimeCount = 0;
    std::string s_hostReport;
    std::unordered_set<std::string> s_logged;
    std::unordered_set<std::string> s_ignored;
    std::unordered_set<std::string> s_seen;
    std::string s_logFolder;
    std::string s_dumpFolder;
    std::filesystem::path s_newestDump;

    std::filesystem::path ignoredFilePath() {
      return util::RtxFileSys::path(util::RtxFileSys::Logs) / "fault-notice-ignored.txt";
    }

    // Returns the newest minidump in the folder.
    // Returns an empty path when there is none.
    std::filesystem::path newestDump(const std::filesystem::path& folder) {
      std::error_code ec;
      std::filesystem::path newest;
      std::filesystem::file_time_type newestTime {};
      for (const auto& entry : std::filesystem::directory_iterator(folder, ec)) {
        const auto time = entry.last_write_time(ec);
        if (!ec && entry.path().extension() == ".dmp" && (newest.empty() || time > newestTime)) {
          newest = entry.path();
          newestTime = time;
        }
      }
      return newest;
    }

    void addProblem(std::string id, std::string text) {
      if (s_logged.insert(id + '|' + text).second) {
        Logger::warn(str::format("[Problem] ", text));
      }
      s_problems.push_back({ std::move(id), std::move(text) });
    }

    // Runs once to check for crashes in the previous session and to find the folders the notice names.
    void start() {
      s_started = true;

      // The real path of the log file is the folder Explorer can open, even under a mod manager's virtual file system.
      const std::filesystem::path logsDir = util::RtxFileSys::path(util::RtxFileSys::Logs);
      const std::wstring& logFile = Logger::realFilePath();
      const std::filesystem::path realLogsDir = logFile.empty() ? logsDir : std::filesystem::path(logFile).parent_path();
      const std::filesystem::path dumps = crashReportsDir(realLogsDir);
      s_logFolder = str::fromws(realLogsDir.c_str());
      s_dumpFolder = str::fromws(dumps.c_str());

      std::ifstream ignored(ignoredFilePath());
      for (std::string line; std::getline(ignored, line);) {
        s_ignored.insert(line);
      }

      if (sentry::crashedLastRun()) {
        s_newestDump = newestDump(dumps);
        addProblem("crash", s_newestDump.empty()
          ? std::string("The renderer crashed in the previous session.")
          : str::format("The renderer crashed in the previous session. Its crash dump is ", str::fromws(s_newestDump.c_str()), "."));
      }

      std::error_code ec;
      if (std::filesystem::remove(gpuCrashMarkerPath(logsDir), ec)) {
        addProblem("gpu-crash", "The GPU crashed (device lost) in the previous session.");
      }
      s_runtimeCount = s_problems.size();
    }

    // Rebuilds the problem list when the game integration's report changes.
    void update() {
      if (!s_started) {
        start();
      }
      const std::string& report = FaultNotice::hostReport();
      if (report == s_hostReport) {
        return;
      }
      s_hostReport = report;
      s_problems.resize(s_runtimeCount);
      std::istringstream lines(report);
      for (std::string line; std::getline(lines, line);) {
        const size_t bar = line.find('|');
        if (bar != std::string::npos && bar > 0 && bar + 1 < line.size()) {
          addProblem(line.substr(0, bar), line.substr(bar + 1));
        }
      }
    }

    // Starts Explorer with the given arguments, such as a folder, or /select, and a file.
    void runExplorer(const std::wstring& arguments) {
      ShellExecuteW(nullptr, L"open", L"explorer.exe", arguments.c_str(), nullptr, SW_SHOWNORMAL);
    }

    void showProblemList(bool inMenu) {
      ImGui::TextColored(kTitleColor, "Problem detected");
      ImGui::TextWrapped("A problem was detected. The most likely cause is a conflict with another mod or a broken install:");
      for (const Problem& problem : s_problems) {
        const bool ignored = s_ignored.count(problem.id) != 0;
        if (!inMenu && (ignored || s_seen.count(problem.id))) {
          continue;
        }
        ImGui::Bullet();
        ImGui::TextWrapped("%s", problem.text.c_str());
        if (inMenu && !ignored) {
          ImGui::PushID(problem.id.c_str());
          if (ImGui::SmallButton("Don't Show Again")) {
            s_ignored.insert(problem.id);
            std::ofstream(ignoredFilePath(), std::ios::app) << problem.id << '\n';
          }
          RemixGui::SetTooltipToLastWidgetOnHover("Hides this problem from the notice in this and later sessions.");
          ImGui::PopID();
        }
      }
    }
  }

  void FaultNotice::showToast(ImFont* font, bool menuOpen) {
    if (!enable()) {
      return;
    }
    update();

    bool anyNew = false;
    for (const Problem& problem : s_problems) {
      if (!s_ignored.count(problem.id) && !s_seen.count(problem.id)) {
        anyNew = true;
        if (menuOpen) {
          s_seen.insert(problem.id);
        }
      }
    }
    if (!anyNew || menuOpen) {
      return;
    }

    if (font != nullptr) {
      ImGui::PushFont(font);
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float margin = ImGui::GetFontSize();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x + viewport->Size.x - margin, viewport->Pos.y + margin), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(std::min(viewport->Size.x - 2.0f * margin, ImGui::GetFontSize() * 34.0f), 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f);
    constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                                        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::Begin("##faultNotice", nullptr, kFlags)) {
      showProblemList(false);
      const std::string& upload = uploadFile();
      if (upload.empty()) {
        ImGui::TextWrapped("Logs: %s", s_logFolder.c_str());
        ImGui::TextWrapped("Crash dumps: %s", s_dumpFolder.c_str());
      } else {
        ImGui::TextWrapped("Upload this file: %s", upload.c_str());
      }
      if (!supportText().empty()) {
        ImGui::TextWrapped("%s", supportText().c_str());
      }
      if (!supportUrl().empty()) {
        ImGui::TextColored(kLinkColor, "%s", supportUrl().c_str());
      }
      const std::string menuKeys = buildKeyBindDescriptorStringForDisplay(RtxOptions::remixMenuKeyBinds());
      if (upload.empty()) {
        ImGui::TextWrapped("Press %s for buttons that open these folders and the link, and to hide this notice.", menuKeys.c_str());
      } else {
        ImGui::TextWrapped("Press %s for buttons that show this file and open the link, and to hide this notice.", menuKeys.c_str());
      }
    }
    ImGui::End();
    if (font != nullptr) {
      ImGui::PopFont();
    }
  }

  void FaultNotice::showSection() {
    RemixGui::Checkbox("Show Problem Notices", &enableObject());
    update();
    if (s_problems.empty()) {
      ImGui::TextDisabled("No problems were detected in this session.");
      return;
    }

    ImGui::PushID("faultNotice");
    showProblemList(true);

    const std::string& upload = uploadFile();
    if (!upload.empty()) {
      if (ImGui::Button("Show File to Upload")) {
        runExplorer(L"/select,\"" + str::tows(upload.c_str()) + L"\"");
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Opens the folder of the file to attach to a bug report, with the file selected.");
      ImGui::SameLine();
      if (ImGui::SmallButton("Copy Path##upload")) {
        ImGui::SetClipboardText(upload.c_str());
      }
      ImGui::TextWrapped("File to upload: %s", upload.c_str());
    }

    ImGUI::showLogFolder(false);

    if (ImGui::Button("Open Crash Dump Folder")) {
      runExplorer(s_newestDump.empty() ? L"\"" + str::tows(s_dumpFolder.c_str()) + L"\"" : L"/select,\"" + s_newestDump.wstring() + L"\"");
    }
    RemixGui::SetTooltipToLastWidgetOnHover("Opens the folder of the crash dumps (.dmp), with the newest one selected after a crash.");
    ImGui::SameLine();
    if (ImGui::SmallButton("Copy Path##dumps")) {
      ImGui::SetClipboardText(s_dumpFolder.c_str());
    }
    ImGui::TextWrapped("Crash dump folder: %s", s_dumpFolder.c_str());

    if (!supportText().empty()) {
      ImGui::TextWrapped("%s", supportText().c_str());
    }
    const std::string& url = supportUrl();
    if (url.rfind("https://", 0) == 0) {
      ImGui::PushStyleColor(ImGuiCol_Text, kLinkColor);
      if (ImGui::Button(url.c_str())) {
        ShellExecuteW(nullptr, L"open", str::tows(url.c_str()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
      }
      ImGui::PopStyleColor();
      RemixGui::SetTooltipToLastWidgetOnHover("Opens the link in the web browser.");
      ImGui::SameLine();
      if (ImGui::SmallButton("Copy Link")) {
        ImGui::SetClipboardText(url.c_str());
      }
    }
    ImGui::PopID();
  }

}
