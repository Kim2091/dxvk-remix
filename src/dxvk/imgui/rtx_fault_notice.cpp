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
      // The key the dismissal is stored under. It names the report the problem belongs to, so a later crash shows again.
      std::string key;
      std::string text;
      std::string detail;
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

    std::string fileName(const std::string& path) {
      return str::fromws(std::filesystem::path(str::tows(path.c_str())).filename().c_str());
    }

    void addProblem(std::string id, std::string text, std::string detail = std::string(), std::string key = std::string()) {
      if (s_logged.insert(id + '|' + text + '|' + detail).second) {
        Logger::warn(str::format("[Problem] ", text, detail.empty() ? "" : " ", detail));
      }
      s_problems.push_back({ std::move(id), std::move(key), std::move(text), std::move(detail) });
    }

    // Returns true for a problem that is a crash, whose report file the player is asked to send.
    bool isCrash(const Problem& problem) {
      return problem.id == "crash" || problem.id == "gpu-crash" || problem.id == "game-crashed";
    }

    // Returns the name a problem is dismissed under.
    // A crash reported by the game integration belongs to the file it names.
    std::string dismissKey(const Problem& problem) {
      if (problem.id == "game-crashed" && !FaultNotice::uploadFile().empty()) {
        return problem.id + '@' + fileName(FaultNotice::uploadFile());
      }
      return problem.key.empty() ? problem.id : problem.key;
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
        addProblem("crash", "The game crashed last time.",
          s_newestDump.empty() ? std::string() : str::format("Crash dump: ", str::fromws(s_newestDump.c_str()), "."),
          s_newestDump.empty() ? std::string() : "crash@" + str::fromws(s_newestDump.filename().c_str()));
      }

      std::error_code ec;
      const std::filesystem::path marker = gpuCrashMarkerPath(logsDir);
      const auto markerTime = std::filesystem::last_write_time(marker, ec);
      const std::string markerKey = ec ? std::string() : str::format("gpu-crash@", markerTime.time_since_epoch().count());
      if (std::filesystem::remove(marker, ec)) {
        addProblem("gpu-crash", "The GPU crashed last time.", "The graphics device was lost (device lost).", markerKey);
      }
      s_runtimeCount = s_problems.size();
    }

    // Rebuilds the problem list when the game integration's report changes.
    // Each line is id|text, or id|text|details.
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
          std::string text = line.substr(bar + 1);
          std::string detail;
          const size_t second = text.find('|');
          if (second != std::string::npos) {
            detail = text.substr(second + 1);
            text.resize(second);
          }
          if (!text.empty()) {
            addProblem(line.substr(0, bar), std::move(text), std::move(detail));
          }
        }
      }
    }

    // Returns the problems the notice shows now: not dismissed, and for the toast also not shown before.
    // A problem the runtime and the game integration both report is listed once.
    std::vector<const Problem*> visibleProblems(bool forToast) {
      std::vector<const Problem*> visible;
      for (const Problem& problem : s_problems) {
        const std::string key = dismissKey(problem);
        if (s_ignored.count(key) || (forToast && s_seen.count(key))) {
          continue;
        }
        if (std::none_of(visible.begin(), visible.end(), [&](const Problem* other) { return other->text == problem.text; })) {
          visible.push_back(&problem);
        }
      }
      return visible;
    }

    // Starts Explorer with the given arguments, such as a folder, or /select, and a file.
    void runExplorer(const std::wstring& arguments) {
      ShellExecuteW(nullptr, L"open", L"explorer.exe", arguments.c_str(), nullptr, SW_SHOWNORMAL);
    }

    void showInExplorer(const std::string& file) {
      runExplorer(L"/select,\"" + str::tows(file.c_str()) + L"\"");
    }

    // Returns the file to send with a report: the crashed session's file when the game integration named it, else the newest report.
    std::string fileToSend() {
      return FaultNotice::uploadFile().empty() ? ImGUI::reportFilePath() : FaultNotice::uploadFile();
    }

    // Returns " in <channel>", or nothing when the game integration names no channel.
    std::string whereToSend() {
      return FaultNotice::supportChannel().empty() ? std::string() : " in " + FaultNotice::supportChannel();
    }

    bool hasSupportLink() {
      return FaultNotice::supportUrl().rfind("https://", 0) == 0;
    }

    void showSupportButton() {
      ImGui::PushStyleColor(ImGuiCol_Text, kLinkColor);
      if (ImGui::Button("Discord")) {
        ShellExecuteW(nullptr, L"open", str::tows(FaultNotice::supportUrl().c_str()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
      }
      ImGui::PopStyleColor();
      RemixGui::SetTooltipToLastWidgetOnHover("Opens the support chat in the web browser.");
    }

    // Draws the technical part, collapsed by default: problem details, the report file, the folders and the link.
    void showDetails() {
      if (!ImGui::CollapsingHeader("Details")) {
        return;
      }
      RemixGui::Checkbox("Show Problem Notices", &FaultNotice::enableObject());

      for (const Problem& problem : s_problems) {
        if (!problem.detail.empty()) {
          ImGui::TextWrapped("%s", problem.detail.c_str());
        }
      }

      const std::string file = fileToSend();
      if (!file.empty()) {
        ImGui::TextWrapped("Report file: %s", file.c_str());
        if (ImGui::SmallButton("Copy Path##reportFile")) {
          ImGui::SetClipboardText(file.c_str());
        }
        ImGui::TextWrapped("The report file holds one session's logs, settings, load order, crash log and the screenshots "
                           "taken with the mark key. To report the session you are playing, use Save Report Now in the "
                           "Host Settings tab of the developer menu.");
      }

      ImGUI::showLogFolder(false);

      if (!s_dumpFolder.empty()) {
        if (ImGui::Button("Open Crash Dump Folder")) {
          runExplorer(s_newestDump.empty() ? L"\"" + str::tows(s_dumpFolder.c_str()) + L"\"" : L"/select,\"" + s_newestDump.wstring() + L"\"");
        }
        RemixGui::SetTooltipToLastWidgetOnHover("Opens the folder of the crash dumps (.dmp), with the newest one selected after a crash.");
        ImGui::TextWrapped("Crash dump folder: %s", s_dumpFolder.c_str());
      }

      if (hasSupportLink()) {
        ImGui::TextWrapped("Support link: %s", FaultNotice::supportUrl().c_str());
        if (ImGui::SmallButton("Copy Link")) {
          ImGui::SetClipboardText(FaultNotice::supportUrl().c_str());
        }
      }
    }
  }

  void FaultNotice::showToast(ImFont* font, bool menuOpen) {
    if (!enable()) {
      return;
    }
    update();

    const std::vector<const Problem*> visible = visibleProblems(true);
    if (visible.empty()) {
      return;
    }
    // Opening a menu marks the problems as seen, which hides the toast until a new one arrives.
    if (menuOpen) {
      for (const Problem* problem : visible) {
        s_seen.insert(dismissKey(*problem));
      }
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
      ImGui::TextColored(kTitleColor, "Problem detected");
      for (const Problem* problem : visible) {
        ImGui::Bullet();
        ImGui::TextWrapped("%s", problem->text.c_str());
      }
      const std::string menuKeys = buildKeyBindDescriptorStringForDisplay(RtxOptions::remixMenuKeyBinds());
      ImGui::TextWrapped("Press %s for help.", menuKeys.c_str());
    }
    ImGui::End();
    if (font != nullptr) {
      ImGui::PopFont();
    }
  }

  void FaultNotice::showSection() {
    update();
    const std::vector<const Problem*> visible = visibleProblems(false);

    ImGui::PushID("faultNotice");
    if (visible.empty()) {
      if (ImGui::Button("Report a problem")) {
        const std::string report = ImGUI::reportFilePath();
        if (report.empty()) {
          runExplorer(L"\"" + str::tows(s_logFolder.c_str()) + L"\"");
        } else {
          showInExplorer(report);
        }
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Opens the folder of the newest report file, with the file selected. Send that file when you report a problem.");
      if (hasSupportLink()) {
        ImGui::SameLine();
        showSupportButton();
      }
    } else {
      ImGui::TextColored(kTitleColor, "Problem detected");
      bool crashed = false;
      for (const Problem* problem : visible) {
        ImGui::Bullet();
        ImGui::TextWrapped("%s", problem->text.c_str());
        crashed = crashed || isCrash(*problem);
      }

      const std::string file = crashed ? fileToSend() : std::string();
      if (!file.empty()) {
        ImGui::TextWrapped("To report it: send this file%s.", whereToSend().c_str());
      } else if (crashed) {
        ImGui::TextWrapped("To report it: see Details.");
      } else if (hasSupportLink()) {
        ImGui::TextWrapped("Need help? Ask%s.", whereToSend().empty() ? " in the support chat" : whereToSend().c_str());
      }

      bool sameLine = false;
      if (!file.empty()) {
        if (ImGui::Button("Show file")) {
          showInExplorer(file);
        }
        RemixGui::SetTooltipToLastWidgetOnHover("Opens the folder of the report file, with the file selected.");
        sameLine = true;
      }
      if (hasSupportLink()) {
        if (sameLine) {
          ImGui::SameLine();
        }
        showSupportButton();
        sameLine = true;
      }
      if (sameLine) {
        ImGui::SameLine();
      }
      if (ImGui::Button("Dismiss")) {
        std::ofstream ignored(ignoredFilePath(), std::ios::app);
        for (const Problem* problem : visible) {
          s_ignored.insert(dismissKey(*problem));
          ignored << dismissKey(*problem) << '\n';
        }
      }
      RemixGui::SetTooltipToLastWidgetOnHover("Hides this notice. It stays hidden for this report; a new problem shows again.");
    }
    showDetails();
    ImGui::PopID();
  }

}
