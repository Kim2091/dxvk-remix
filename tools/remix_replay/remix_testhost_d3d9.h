// remix_testhost D3D9 scenarios (--d3d9 <name>). See remix_testhost_d3d9.cpp.

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <filesystem>
#include <string>

struct D3D9ScenarioArgs {
  int frames = 720;
  // The frame of the scenario's one event (Reset, reclassification, bone switch, the start of the unpresented stretch).
  int eventFrame = 450;
  // --no-present: how many frames are drawn without a Present.
  int hiddenFrames = 120;
  // --twice: the second pass is an additive overlay instead of an exact repeat.
  bool overlay = false;
  // --overlay-first: with --overlay, the blended pass is drawn first and the opaque pass second.
  bool overlayFirst = false;
};

// Runs one scenario on a device made by <runtime>/d3d9.dll. Returns the process exit code.
int runD3D9Scenario(const std::wstring& name, const std::filesystem::path& runtime, HWND hwnd, uint32_t width,
                    uint32_t height, const D3D9ScenarioArgs& args);

// Prints the texture hashes the scenarios' rtx.conf tags need, one "hash <name> 0x..." line each.
void printD3D9Hashes();
