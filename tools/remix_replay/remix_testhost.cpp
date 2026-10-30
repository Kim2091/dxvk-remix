// remix_testhost: a tiny Remix API host for verifying the recorder and replayer.
// Draws a textured ground, cubes (per-frame, GPU-instanced and retained), a sphere
// and a distant light under an orbiting camera, for a fixed number of frames.
//
//   remix_testhost --runtime <dir> [--frames N] [--d3d11] [--show] [--origin-shift [--camera-events] [--instanced]]
//                  [--skinned N [--reorder]] [--particles] [--usd-mod [--mod-edit-frame F]] [--light-updates N]
//                  [--asset-key] [--clusters N [--stream K]]
//   remix_testhost --runtime <dir> --d3d9 <scenario> [--frames N] [--event-frame F] [--hidden-frames N] [--overlay] [--overlay-first]
//   remix_testhost --d3d9-hashes
//
// --d3d9 <scenario>: instead of the Remix API, a D3D9 device made by the runtime's d3d9.dll draws one of the D3D9 v2
// test scenarios (remix_testhost_d3d9.cpp). --d3d9-hashes prints the texture hashes their rtx.conf tags need.
//
// --clusters N: instead of the default scene, N small cubes with N distinct meshes (so every one is a merged-BLAS
// candidate) as retained instances on a grid (1.5 units apart, centred on the origin) under a fixed camera. With
// --stream K, K of them (round robin) are destroyed and created again every frame from frame 40 until 60 frames
// before the end, like cell streaming. Prints the host's per-frame CPU time (steady / streaming / after); pair with
// rtx.mergedBucketCellSize and rtx.accelMemoryLogIntervalFrames to see bucket counts and rebuilt primitives.
// With --movers M, M of them (evenly spread) move in bursts like doors or pushed props: each moves for 3 frames every
// --mover-period P frames (default 150, staggered), in the same window as --stream (rtx.accel.moverBackoff).
// With --alpha-cutout the cubes are alpha-tested (greater than 127) with a cutout texture laid out so every cube has
// 6 opaque, 4 transparent and 2 mixed triangles (rtx.alphaBake.census should report 50 / 33 / 17 %).
//
// --asset-key: one more cube mesh (hash 0x8000000000000010) with remixapi_AssetKeyEXT {key 0x0A55E70000000001,
// alias 0x8000000000000002} and a retained instance of it; with --usd-mod and rtx.scene.assetKeys the mod (authored
// for 0x8000000000000002) reaches it through the alias. --mod-edit-frame F rewrites the mod at frame F (brighter
// light, taller pyramid) for a hot reload.
//
// --light-updates N: 1000 small sphere lights in a grid above the ground (created once, persistent), and N of them
// (round robin) moved slightly with UpdateLightDefinition every frame; the rest never change.
//
// --usd-mod: writes <runtime>/rtx-remix/mods/testhost_usd/mod.usda before startup, replacing the cube mesh (hash
// 0x8000000000000002) with a pyramid plus a sphere light prim, and removes it at exit. Retained cube 0 moves every
// frame, so the replacement's prims and light must follow it.
//
// --particles: two more retained cubes, one with remixapi_InstanceInfoParticleSystemEXT (spawning, emitter hidden),
// one with the PARTICLE_EMITTER category (global particle settings), and one with the VIEW_MODEL category.
//
// --skinned N: the default scene plus N skinned cubes drawn with DrawInstance every frame (two bones, the top bending
// over time; a ring around the scene). --reorder rotates their draw order every frame, so frame-to-frame matching
// cannot rely on the order.
//
// --origin-shift: instead of the default scene, a 40x40 grid of static retained cubes near x = 3 * 16384
// (FO4's origin grid) and a camera flying along +x. The scene origin follows the camera on a 16384-unit grid
// (DrawRetainedInstances(origin), camera in render space), so it moves once, at the frame printed as
// "origin shift". Prints the host's per-frame CPU time around that frame. With --camera-events the shift frame's
// SetupCamera also carries remixapi_CameraInfoFrameEventsEXT (ORIGIN_SHIFT), so the camera history moves with it.
// With --instanced the field is 40 GPU-instanced retained instances (one per row of 40 cubes) instead of 1600.
// With --distinct-meshes every cube of the field has its own mesh, so the field goes into merged BLASes (a shared
// mesh gets one BLAS of its own); the runtime log's [AccelMem] lines then show what the shift rebuilds.
//
// --d3d11 registers a D3D11 device made by the runtime's d3d11.dll (as the Fallout 4
// plugin does), imports the cube texture with Prepare/ImportD3D11TextureEx and
// presents through RenderD3D11 + the DXGI swapchain. Without it, the host uses
// Startup/Present and CreateTexture.

#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <remix/remix_c.h>

#include "remix_testhost_d3d9.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
  remixapi_Interface g_remix {};

  remixapi_HardcodedVertex vertex(float x, float y, float z, float nx, float ny, float nz, float u, float v) {
    remixapi_HardcodedVertex out {};
    out.position[0] = x; out.position[1] = y; out.position[2] = z;
    out.normal[0] = nx; out.normal[1] = ny; out.normal[2] = nz;
    out.texcoord[0] = u; out.texcoord[1] = v;
    out.color = 0xFFFFFFFF;
    return out;
  }

  void cube(std::vector<remixapi_HardcodedVertex>& v, std::vector<uint32_t>& idx) {
    const float n[6][3] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
    for (int f = 0; f < 6; ++f) {
      const float* N = n[f];
      // two tangent axes
      float a[3] = { N[1], N[2], N[0] };
      float b[3] = { N[1] * a[2] - N[2] * a[1], N[2] * a[0] - N[0] * a[2], N[0] * a[1] - N[1] * a[0] };
      const uint32_t base = uint32_t(v.size());
      const float s[4][2] = { {-1,-1}, {1,-1}, {1,1}, {-1,1} };
      for (int c = 0; c < 4; ++c) {
        const float x = 0.5f * (N[0] + s[c][0] * a[0] + s[c][1] * b[0]);
        const float y = 0.5f * (N[1] + s[c][0] * a[1] + s[c][1] * b[1]);
        const float z = 0.5f * (N[2] + s[c][0] * a[2] + s[c][1] * b[2]);
        v.push_back(vertex(x, y, z, N[0], N[1], N[2], (s[c][0] + 1) * 0.5f, (s[c][1] + 1) * 0.5f));
      }
      const uint32_t q[6] = { 0, 1, 2, 0, 2, 3 };
      for (uint32_t i : q) {
        idx.push_back(base + i);
      }
    }
  }

  // --alpha-cutout: a cube whose triangles sample separate regions of the cutout texture (below), so each
  // triangle's opacity class is known: on every face triangle A is opaque; triangle B is transparent on
  // faces 0-3 and mixed (checker) on faces 4-5. Per cube: 6 opaque, 4 transparent, 2 mixed triangles.
  void cutoutCube(std::vector<remixapi_HardcodedVertex>& v, std::vector<uint32_t>& idx) {
    const float n[6][3] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
    // UV regions (u0, v0, u1, v1), inset so a micro-triangle's texel footprint stays inside.
    const float opaqueRegion[4] = { 0.03f, 0.05f, 0.30f, 0.95f };
    const float transparentRegion[4] = { 0.37f, 0.05f, 0.63f, 0.95f };
    const float mixedRegion[4] = { 0.70f, 0.05f, 0.97f, 0.95f };
    for (int f = 0; f < 6; ++f) {
      const float* N = n[f];
      float a[3] = { N[1], N[2], N[0] };
      float b[3] = { N[1] * a[2] - N[2] * a[1], N[2] * a[0] - N[0] * a[2], N[0] * a[1] - N[1] * a[0] };
      const float s[4][2] = { {-1,-1}, {1,-1}, {1,1}, {-1,1} };
      const int tris[2][3] = { { 0, 1, 2 }, { 0, 2, 3 } };
      for (int t = 0; t < 2; ++t) {
        const float* r = t == 0 ? opaqueRegion : (f < 4 ? transparentRegion : mixedRegion);
        for (int k = 0; k < 3; ++k) {
          const int c = tris[t][k];
          const float x = 0.5f * (N[0] + s[c][0] * a[0] + s[c][1] * b[0]);
          const float y = 0.5f * (N[1] + s[c][0] * a[1] + s[c][1] * b[1]);
          const float z = 0.5f * (N[2] + s[c][0] * a[2] + s[c][1] * b[2]);
          const float u = r[0] + (s[c][0] + 1) * 0.5f * (r[2] - r[0]);
          const float w = r[1] + (s[c][1] + 1) * 0.5f * (r[3] - r[1]);
          idx.push_back(uint32_t(v.size()));
          v.push_back(vertex(x, y, z, N[0], N[1], N[2], u, w));
        }
      }
    }
  }

  // --alpha-cutout texture (RGBA8, 128x128, one mip), in thirds along u: alpha 255 (texels 0-42), 0 (43-85),
  // and an 8x8-texel checker of 0 / 255 (86-127). Constant along v, so a flipped v changes nothing.
  std::vector<uint8_t> cutoutTexture(uint32_t size) {
    std::vector<uint8_t> out;
    for (uint32_t y = 0; y < size; ++y) {
      for (uint32_t x = 0; x < size; ++x) {
        uint8_t alpha;
        if (x * 3 < size) {
          alpha = 255;
        } else if (x * 3 < 2 * size) {
          alpha = 0;
        } else {
          alpha = (((x / 8) + (y / 8)) & 1) ? 255 : 0;
        }
        out.push_back(80); out.push_back(170); out.push_back(60); out.push_back(alpha);
      }
    }
    return out;
  }

  // RGBA8 checker with a full mip chain, mips sequential.
  std::vector<uint8_t> checker(uint32_t size, uint32_t& mips, uint8_t r0, uint8_t g0, uint8_t b0, uint8_t r1, uint8_t g1, uint8_t b1) {
    std::vector<uint8_t> out;
    mips = 0;
    for (uint32_t s = size; s >= 1; s /= 2) {
      for (uint32_t y = 0; y < s; ++y) {
        for (uint32_t x = 0; x < s; ++x) {
          const bool odd = (((x * 8) / std::max(1u, s)) + ((y * 8) / std::max(1u, s))) & 1;
          out.push_back(odd ? r1 : r0);
          out.push_back(odd ? g1 : g0);
          out.push_back(odd ? b1 : b0);
          out.push_back(255);
        }
      }
      ++mips;
      if (s == 1) {
        break;
      }
    }
    return out;
  }

  remixapi_Transform translate(float x, float y, float z, float scale = 1.f) {
    remixapi_Transform t {};
    t.matrix[0][0] = scale; t.matrix[1][1] = scale; t.matrix[2][2] = scale;
    t.matrix[0][3] = x; t.matrix[1][3] = y; t.matrix[2][3] = z;
    return t;
  }

  std::wstring hashPath(uint64_t hash) {
    wchar_t buf[32];
    swprintf(buf, 32, L"0x%llX", static_cast<unsigned long long>(hash));
    return buf;
  }

  void check(remixapi_ErrorCode r, const char* what) {
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      std::printf("FAILED: %s (%d)\n", what, int(r));
      std::exit(1);
    }
  }

  LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    return DefWindowProcW(hwnd, msg, wParam, lParam);
  }
}

int wmain(int argc, wchar_t** argv) {
  fs::path runtime;
  int frames = 60;
  bool d3d11 = false;
  bool show = false;
  bool originShift = false;
  bool cameraEvents = false;
  bool instanced = false;
  int skinnedCount = 0;
  bool reorder = false;
  bool particles = false;
  bool usdMod = false;
  int lightUpdates = -1;
  bool assetKey = false;
  int modEditFrame = -1;
  int clustersCount = 0;
  bool distinctMeshes = false;
  int streamPerFrame = 0;
  int moversCount = 0;
  int moverPeriod = 150;
  bool alphaCutout = false;
  std::wstring d3d9Scenario;
  D3D9ScenarioArgs d3d9Args;
  bool framesGiven = false;
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    if (a == L"--runtime" && i + 1 < argc) {
      runtime = argv[++i];
    } else if (a == L"--frames" && i + 1 < argc) {
      frames = _wtoi(argv[++i]);
      framesGiven = true;
    } else if (a == L"--d3d11") {
      d3d11 = true;
    } else if (a == L"--show") {
      show = true;
    } else if (a == L"--origin-shift") {
      originShift = true;
    } else if (a == L"--camera-events") {
      cameraEvents = true;
    } else if (a == L"--instanced") {
      instanced = true;
    } else if (a == L"--skinned" && i + 1 < argc) {
      skinnedCount = _wtoi(argv[++i]);
    } else if (a == L"--reorder") {
      reorder = true;
    } else if (a == L"--particles") {
      particles = true;
    } else if (a == L"--usd-mod") {
      usdMod = true;
    } else if (a == L"--light-updates" && i + 1 < argc) {
      lightUpdates = _wtoi(argv[++i]);
    } else if (a == L"--asset-key") {
      assetKey = true;
    } else if (a == L"--mod-edit-frame" && i + 1 < argc) {
      modEditFrame = _wtoi(argv[++i]);
    } else if (a == L"--distinct-meshes") {
      distinctMeshes = true;
    } else if (a == L"--clusters" && i + 1 < argc) {
      clustersCount = _wtoi(argv[++i]);
    } else if (a == L"--stream" && i + 1 < argc) {
      streamPerFrame = _wtoi(argv[++i]);
    } else if (a == L"--alpha-cutout") {
      alphaCutout = true;
    } else if (a == L"--movers" && i + 1 < argc) {
      moversCount = _wtoi(argv[++i]);
    } else if (a == L"--mover-period" && i + 1 < argc) {
      moverPeriod = std::max(_wtoi(argv[++i]), 4);
    } else if (a == L"--d3d9" && i + 1 < argc) {
      d3d9Scenario = argv[++i];
    } else if (a == L"--d3d9-hashes") {
      printD3D9Hashes();
      return 0;
    } else if (a == L"--event-frame" && i + 1 < argc) {
      d3d9Args.eventFrame = _wtoi(argv[++i]);
    } else if (a == L"--hidden-frames" && i + 1 < argc) {
      d3d9Args.hiddenFrames = std::max(_wtoi(argv[++i]), 1);
    } else if (a == L"--overlay") {
      d3d9Args.overlay = true;
    } else if (a == L"--overlay-first") {
      d3d9Args.overlay = true;
      d3d9Args.overlayFirst = true;
    }
  }
  if (runtime.empty()) {
    std::printf("usage: remix_testhost --runtime <dir> [--frames N] [--d3d11] [--show]\n");
    return 2;
  }
  runtime = fs::absolute(runtime);
  const fs::path usdModDir = runtime / "rtx-remix" / "mods" / "testhost_usd";
  const auto writeUsdMod = [&](float intensity, float apex) {
    fs::create_directories(usdModDir);
    FILE* f = _wfopen((usdModDir / "mod.usda").wstring().c_str(), L"wb");
    if (f) {
      const std::string text = std::string(
        "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n\n"
        "def Xform \"RootNode\"\n{\n    def Xform \"meshes\"\n    {\n        def Xform \"mesh_8000000000000002\"\n        {\n"
        "            def Mesh \"pyramid\"\n            {\n"
        "                int[] faceVertexCounts = [3, 3, 3, 3, 3, 3]\n"
        "                int[] faceVertexIndices = [0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 1, 4, 1, 2, 3, 1, 3, 4]\n"
        "                point3f[] points = [(0, " + std::to_string(apex) + ", 0), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]\n"
        "                normal3f[] normals = [(0, 1, 0), (-0.577, -0.577, -0.577), (0.577, -0.577, -0.577), (0.577, -0.577, 0.577), (-0.577, -0.577, 0.577)] (\n"
        "                    interpolation = \"vertex\"\n                )\n"
        "                texCoord2f[] primvars:st = [(0.5, 1), (0, 0), (1, 0), (1, 1), (0, 1)] (\n"
        "                    interpolation = \"vertex\"\n                )\n"
        "                uniform token subdivisionScheme = \"none\"\n"
        "            }\n\n"
        "            def SphereLight \"light\"\n            {\n"
        "                color3f inputs:color = (1, 0.6, 0.2)\n"
        "                float inputs:intensity = " + std::to_string(intensity) + "\n"
        "                float inputs:radius = 0.1\n"
        "                double3 xformOp:translate = (0, 1.2, 0)\n"
        "                uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
        "            }\n        }\n    }\n}\n");
      std::fputs(text.c_str(), f);
      std::fclose(f);
    }
  };
  if (usdMod) {
    writeUsdMod(200.f, 0.8f);
  }
  SetCurrentDirectoryW(runtime.wstring().c_str());
  SetDllDirectoryW(runtime.wstring().c_str());

  const uint32_t width = 1280, height = 720;
  WNDCLASSEXW wc {};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = windowProc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.lpszClassName = L"RemixTestHost";
  RegisterClassExW(&wc);
  RECT rect { 0, 0, LONG(width), LONG(height) };
  AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
  HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"Remix Test Host", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, wc.hInstance, nullptr);
  if (show) {
    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
  }

  if (!d3d9Scenario.empty()) {
    if (framesGiven) {
      d3d9Args.frames = frames;
    }
    return runD3D9Scenario(d3d9Scenario, runtime, hwnd, width, height, d3d9Args);
  }

  // D3D11 device from the runtime's own d3d11.dll (it forwards into d3d9.dll).
  ID3D11Device* device = nullptr;
  ID3D11DeviceContext* context = nullptr;
  IDXGISwapChain* swapchain = nullptr;
  if (d3d11) {
    HMODULE d3d11Dll = LoadLibraryExW((runtime / "d3d11.dll").wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    auto create = d3d11Dll ? reinterpret_cast<PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN>(GetProcAddress(d3d11Dll, "D3D11CreateDeviceAndSwapChain")) : nullptr;
    if (!create) {
      std::printf("FAILED: no D3D11CreateDeviceAndSwapChain in %s\n", (runtime / "d3d11.dll").string().c_str());
      return 1;
    }
    DXGI_SWAP_CHAIN_DESC sd {};
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;
    sd.OutputWindow = hwnd;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    const D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    HRESULT hr = create(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &sd, &swapchain, &device, nullptr, &context);
    if (FAILED(hr)) {
      std::printf("FAILED: D3D11CreateDeviceAndSwapChain 0x%08lx\n", hr);
      return 1;
    }
  }

  HMODULE dll = LoadLibraryExW((runtime / "d3d9.dll").wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  auto initialize = dll ? reinterpret_cast<PFN_remixapi_InitializeLibrary>(GetProcAddress(dll, "remixapi_InitializeLibrary")) : nullptr;
  if (!initialize) {
    std::printf("FAILED: cannot load the runtime\n");
    return 1;
  }
  remixapi_InitializeLibraryInfo initInfo {};
  initInfo.sType = REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO;
  initInfo.version = REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR, REMIXAPI_VERSION_MINOR, REMIXAPI_VERSION_PATCH);
  check(initialize(&initInfo, &g_remix), "InitializeLibrary");

  auto proc = [&](const char* name) { return GetProcAddress(dll, name); };
  auto createRetained = reinterpret_cast<PFN_remixapi_CreateRetainedInstance>(proc("remixapi_CreateRetainedInstance"));
  auto updateRetainedMany = reinterpret_cast<PFN_remixapi_UpdateRetainedInstances>(proc("remixapi_UpdateRetainedInstances"));
  auto destroyRetained = reinterpret_cast<PFN_remixapi_DestroyRetainedInstances>(proc("remixapi_DestroyRetainedInstances"));
  auto drawRetained = reinterpret_cast<PFN_remixapi_DrawRetainedInstances>(proc("remixapi_DrawRetainedInstances"));
  auto registerD3D11 = reinterpret_cast<PFN_remixapi_RegisterD3D11Device>(proc("remixapi_RegisterD3D11Device"));
  auto renderD3D11 = reinterpret_cast<PFN_remixapi_RenderD3D11>(proc("remixapi_RenderD3D11"));
  auto prepareEx = reinterpret_cast<PFN_remixapi_PrepareD3D11TextureEx>(proc("remixapi_PrepareD3D11TextureEx"));
  auto importEx = reinterpret_cast<PFN_remixapi_ImportD3D11TextureEx>(proc("remixapi_ImportD3D11TextureEx"));
  auto release = reinterpret_cast<PFN_remixapi_ReleaseD3D11Texture>(proc("remixapi_ReleaseD3D11Texture"));
  if (!createRetained || !updateRetainedMany || !destroyRetained || !drawRetained) {
    std::printf("FAILED: retained-instance exports missing\n");
    return 1;
  }

  if (d3d11) {
    check(registerD3D11 ? registerD3D11(device) : REMIXAPI_ERROR_CODE_GET_PROC_ADDRESS_FAILURE, "RegisterD3D11Device");
  } else {
    remixapi_StartupInfo startup {};
    startup.sType = REMIXAPI_STRUCT_TYPE_STARTUP_INFO;
    startup.hwnd = hwnd;
    check(g_remix.Startup(&startup), "Startup");
  }
  g_remix.SetConfigVariable("rtx.fallbackLightMode", "0");
  g_remix.SetConfigVariable("rtx.hostImport.enableInPlaceTextureReplace", "True");

  // D3D11 import of one 2D texture with a full mip chain (data: mips sequential, tightly packed).
  auto importD3D11 = [&](DXGI_FORMAT format, uint32_t size, uint32_t blockBytes, const std::vector<uint8_t>& data, D3D11_USAGE usage,
                         uint32_t maxDimension, uint64_t hash, remixapi_TextureHandle replaceHandle) {
    const bool bc = blockBytes != 0;
    std::vector<D3D11_SUBRESOURCE_DATA> init;
    size_t offset = 0;
    uint32_t mips = 0;
    for (uint32_t s = size; ; s = std::max(1u, s / 2)) {
      const uint32_t pitch = bc ? std::max(1u, (s + 3) / 4) * blockBytes : s * 4;
      const uint32_t rows = bc ? std::max(1u, (s + 3) / 4) : s;
      init.push_back({ data.data() + offset, pitch, 0 });
      offset += size_t(pitch) * rows;
      ++mips;
      if (s == 1) {
        break;
      }
    }
    D3D11_TEXTURE2D_DESC td {};
    td.Width = size; td.Height = size; td.MipLevels = mips; td.ArraySize = 1;
    td.Format = format;
    td.SampleDesc.Count = 1;
    td.Usage = usage;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    ID3D11Texture2D* tex = nullptr;
    if (FAILED(device->CreateTexture2D(&td, init.data(), &tex))) {
      std::printf("FAILED: CreateTexture2D format %d\n", int(format));
      std::exit(1);
    }
    remixapi_D3D11TextureToken token = nullptr;
    check(prepareEx(tex, uint32_t(format), maxDimension, &token), "PrepareD3D11TextureEx");
    if (replaceHandle) {
      auto replace = reinterpret_cast<PFN_remixapi_ReplaceD3D11TextureEx>(GetProcAddress(dll, "remixapi_ReplaceD3D11TextureEx"));
      check(replace(token, replaceHandle, REMIXAPI_TEXTURE_ENCODING_NONE), "ReplaceD3D11TextureEx");
    } else {
      remixapi_TextureHandle h;
      check(importEx(token, hash, REMIXAPI_TEXTURE_ENCODING_NONE, &h), "ImportD3D11TextureEx");
    }
    release(token);
    tex->Release();
  };
  // Deterministic bytes for block-compressed textures (any bytes decode to something).
  auto blocks = [](uint32_t size, uint32_t blockBytes, uint32_t seed) {
    std::vector<uint8_t> out;
    for (uint32_t s = size; ; s = std::max(1u, s / 2)) {
      const size_t n = size_t(std::max(1u, (s + 3) / 4)) * std::max(1u, (s + 3) / 4) * blockBytes;
      for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        out.push_back(uint8_t(seed >> 24));
      }
      if (s == 1) {
        break;
      }
    }
    return out;
  };
  const uint64_t bc7TexHash = 0x6000000000000003ull, bc1TexHash = 0x6000000000000004ull;
  if (d3d11) {
    importD3D11(DXGI_FORMAT_BC7_UNORM, 128, 16, blocks(128, 16, 7), D3D11_USAGE_IMMUTABLE, 0, bc7TexHash, nullptr);
    importD3D11(DXGI_FORMAT_BC1_UNORM_SRGB, 64, 8, blocks(64, 8, 1), D3D11_USAGE_DEFAULT, 0, bc1TexHash, nullptr);
  }

  // Textures: ground via CreateTexture; cube via D3D11 import (or CreateTexture).
  const uint64_t groundTexHash = 0x6000000000000001ull, cubeTexHash = 0x6000000000000002ull;
  {
    uint32_t mips;
    std::vector<uint8_t> px = checker(256, mips, 200, 200, 200, 40, 40, 40);
    remixapi_TextureInfo ti {};
    ti.sType = REMIXAPI_STRUCT_TYPE_TEXTURE_INFO;
    ti.hash = groundTexHash;
    ti.width = 256; ti.height = 256; ti.depth = 1; ti.mipLevels = mips;
    ti.format = REMIXAPI_FORMAT_R8G8B8A8_UNORM;
    ti.data = px.data();
    ti.dataSize = px.size();
    remixapi_TextureHandle h;
    check(g_remix.CreateTexture(&ti, &h), "CreateTexture ground");
  }
  {
    uint32_t mips;
    std::vector<uint8_t> px = checker(128, mips, 220, 60, 30, 240, 200, 40);
    if (d3d11) {
      std::vector<D3D11_SUBRESOURCE_DATA> init(mips);
      size_t offset = 0;
      for (uint32_t m = 0, s = 128; m < mips; ++m, s = std::max(1u, s / 2)) {
        init[m].pSysMem = px.data() + offset;
        init[m].SysMemPitch = s * 4;
        offset += size_t(s) * s * 4;
      }
      D3D11_TEXTURE2D_DESC td {};
      td.Width = 128; td.Height = 128; td.MipLevels = mips; td.ArraySize = 1;
      td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      td.SampleDesc.Count = 1;
      td.Usage = D3D11_USAGE_IMMUTABLE;
      td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      ID3D11Texture2D* tex = nullptr;
      if (FAILED(device->CreateTexture2D(&td, init.data(), &tex))) {
        std::printf("FAILED: CreateTexture2D\n");
        return 1;
      }
      remixapi_D3D11TextureToken token = nullptr;
      check(prepareEx(tex, DXGI_FORMAT_R8G8B8A8_UNORM, 64, &token), "PrepareD3D11TextureEx");
      remixapi_TextureHandle h;
      check(importEx(token, cubeTexHash, REMIXAPI_TEXTURE_ENCODING_NONE, &h), "ImportD3D11TextureEx");
      release(token);
      tex->Release();
    } else {
      remixapi_TextureInfo ti {};
      ti.sType = REMIXAPI_STRUCT_TYPE_TEXTURE_INFO;
      ti.hash = cubeTexHash;
      ti.width = 128; ti.height = 128; ti.depth = 1; ti.mipLevels = mips;
      ti.format = REMIXAPI_FORMAT_R8G8B8A8_UNORM;
      ti.data = px.data();
      ti.dataSize = px.size();
      remixapi_TextureHandle h;
      check(g_remix.CreateTexture(&ti, &h), "CreateTexture cube");
    }
  }

  // Materials.
  const std::wstring groundPath = hashPath(groundTexHash), cubePath = hashPath(cubeTexHash);
  remixapi_MaterialHandle groundMat = nullptr, cubeMat = nullptr, blueMat = nullptr;
  {
    remixapi_MaterialInfoOpaqueEXT opaque {};
    opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
    opaque.albedoConstant = { 1, 1, 1 };
    opaque.opacityConstant = 1;
    opaque.alphaTestType = 7;  // always pass (0 = never: invisible)
    opaque.useDrawCallAlphaState = 1;
    opaque.roughnessConstant = 0.6f;
    opaque.metallicConstant = 0;
    opaque.enableDlssControlMask = 1;
    opaque.dlssControlMaskIntensity = 1;
    opaque.dlssControlMaskToneStrength = 1;
    opaque.dlssControlMaskStructuralStrength = 1;
    remixapi_MaterialInfo mi {};
    mi.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
    mi.pNext = &opaque;
    mi.hash = 0x7000000000000001ull;
    mi.albedoTexture = groundPath.c_str();
    mi.emissiveColorConstant = { 0, 0, 0 };
    mi.filterMode = 1;
    mi.wrapModeU = 1;
    mi.wrapModeV = 1;
    check(g_remix.CreateMaterial(&mi, &groundMat), "CreateMaterial ground");
    mi.hash = 0x7000000000000002ull;
    mi.albedoTexture = cubePath.c_str();
    opaque.roughnessConstant = 0.3f;
    check(g_remix.CreateMaterial(&mi, &cubeMat), "CreateMaterial cube");
    mi.hash = 0x7000000000000003ull;
    mi.albedoTexture = nullptr;
    opaque.albedoConstant = { 0.1f, 0.3f, 0.9f };
    opaque.metallicConstant = 0.8f;
    check(g_remix.CreateMaterial(&mi, &blueMat), "CreateMaterial blue");
  }
  remixapi_MaterialHandle bc7Mat = nullptr, bc1Mat = nullptr;
  const std::wstring bc7Path = hashPath(bc7TexHash), bc1Path = hashPath(bc1TexHash);
  if (d3d11) {
    remixapi_MaterialInfoOpaqueEXT opaque {};
    opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
    opaque.albedoConstant = { 1, 1, 1 };
    opaque.opacityConstant = 1;
    opaque.alphaTestType = 7;  // always pass (0 = never: invisible)
    opaque.useDrawCallAlphaState = 1;
    opaque.roughnessConstant = 0.5f;
    opaque.enableDlssControlMask = 1;
    opaque.dlssControlMaskIntensity = 1;
    opaque.dlssControlMaskToneStrength = 1;
    opaque.dlssControlMaskStructuralStrength = 1;
    remixapi_MaterialInfo mi {};
    mi.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
    mi.pNext = &opaque;
    mi.hash = 0x7000000000000004ull;
    mi.albedoTexture = bc7Path.c_str();
    mi.filterMode = 1;
    mi.wrapModeU = 1;
    mi.wrapModeV = 1;
    check(g_remix.CreateMaterial(&mi, &bc7Mat), "CreateMaterial bc7");
    mi.hash = 0x7000000000000005ull;
    mi.albedoTexture = bc1Path.c_str();
    check(g_remix.CreateMaterial(&mi, &bc1Mat), "CreateMaterial bc1");
  }

  // Meshes.
  remixapi_MeshHandle groundMesh = nullptr, cubeMesh = nullptr, skinnedMesh = nullptr;
  {
    remixapi_HardcodedVertex gv[4] = {
      vertex(-10, 0, -10, 0, 1, 0, 0, 0), vertex(10, 0, -10, 0, 1, 0, 4, 0),
      vertex(10, 0, 10, 0, 1, 0, 4, 4), vertex(-10, 0, 10, 0, 1, 0, 0, 4) };
    uint32_t gi[6] = { 0, 2, 1, 0, 3, 2 };
    remixapi_MeshInfoSurfaceTriangles s {};
    s.vertices_values = gv; s.vertices_count = 4;
    s.indices_values = gi; s.indices_count = 6;
    s.material = groundMat;
    remixapi_MeshInfo mi {};
    mi.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
    mi.hash = 0x8000000000000001ull;
    mi.surfaces_values = &s;
    mi.surfaces_count = 1;
    check(g_remix.CreateMesh(&mi, &groundMesh), "CreateMesh ground");

    std::vector<remixapi_HardcodedVertex> cv;
    std::vector<uint32_t> ci;
    cube(cv, ci);
    s.vertices_values = cv.data(); s.vertices_count = cv.size();
    s.indices_values = ci.data(); s.indices_count = ci.size();
    s.material = cubeMat;
    mi.hash = 0x8000000000000002ull;
    check(g_remix.CreateMesh(&mi, &cubeMesh), "CreateMesh cube");

    if (skinnedCount > 0) {
      // Bone 0 holds the bottom half, bone 1 the top half.
      std::vector<float> weights(cv.size(), 1.f);
      std::vector<uint32_t> bones(cv.size());
      for (size_t v = 0; v < cv.size(); ++v) {
        bones[v] = cv[v].position[1] > 0.f ? 1u : 0u;
      }
      s.skinning_hasvalue = 1;
      s.skinning_value.bonesPerVertex = 1;
      s.skinning_value.blendWeights_values = weights.data();
      s.skinning_value.blendWeights_count = uint32_t(weights.size());
      s.skinning_value.blendIndices_values = bones.data();
      s.skinning_value.blendIndices_count = uint32_t(bones.size());
      mi.hash = 0x8000000000000003ull;
      check(g_remix.CreateMesh(&mi, &skinnedMesh), "CreateMesh skinned cube");
      s.skinning_hasvalue = 0;
    }
  }

  // Lights.
  remixapi_LightHandle sphereLight = nullptr, sunLight = nullptr;
  {
    remixapi_LightInfoSphereEXT sphere {};
    sphere.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT;
    sphere.position = { 2, 4, -2 };
    sphere.radius = 0.3f;
    sphere.volumetricRadianceScale = 1;
    remixapi_LightInfo li {};
    li.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO;
    li.pNext = &sphere;
    li.hash = 0x9000000000000001ull;
    li.radiance = { 400, 350, 300 };
    check(g_remix.CreateLight(&li, &sphereLight), "CreateLight sphere");
    remixapi_LightInfoDistantEXT sun {};
    sun.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT;
    sun.direction = { -0.4f, -0.8f, 0.45f };
    sun.angularDiameterDegrees = 0.5f;
    sun.volumetricRadianceScale = 1;
    li.pNext = &sun;
    li.hash = 0x9000000000000002ull;
    li.radiance = { 4, 4, 3.6f };
    check(g_remix.CreateLight(&li, &sunLight), "CreateLight sun");
  }

  // --light-updates: a grid of persistent sphere lights.
  constexpr int kGridLights = 1000;
  std::vector<remixapi_LightHandle> gridLights;
  const auto gridLightInfo = [](int k, float wobble, remixapi_LightInfoSphereEXT& sphere, remixapi_LightInfo& li) {
    sphere = {};
    sphere.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT;
    sphere.position = { -9.5f + 0.6f * float(k % 32) + wobble, 0.3f + 0.2f * float((k / 32) % 4), -9.5f + 0.6f * float(k / 128) };
    sphere.radius = 0.03f;
    sphere.volumetricRadianceScale = 1;
    li = {};
    li.sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO;
    li.pNext = &sphere;
    li.hash = 0x9100000000000000ull + uint64_t(k);
    li.radiance = { 20.f + float(k % 7) * 5.f, 15.f, 10.f + float(k % 5) * 5.f };
  };
  if (lightUpdates >= 0) {
    gridLights.resize(kGridLights);
    for (int k = 0; k < kGridLights; ++k) {
      remixapi_LightInfoSphereEXT sphere;
      remixapi_LightInfo li;
      gridLightInfo(k, 0.f, sphere, li);
      check(g_remix.CreateLight(&li, &gridLights[k]), "CreateLight grid");
    }
  }

  if (clustersCount > 0) {
    std::vector<remixapi_HardcodedVertex> cv;
    std::vector<uint32_t> ci;
    remixapi_MaterialHandle clusterMat = cubeMat;
    if (alphaCutout) {
      // Alpha-tested cubes with known per-triangle opacity classes (rtx.alphaBake.census).
      cutoutCube(cv, ci);
      constexpr uint64_t kCutoutTexHash = 0x6000000000000010ull;
      std::vector<uint8_t> px = cutoutTexture(128);
      remixapi_TextureInfo ti {};
      ti.sType = REMIXAPI_STRUCT_TYPE_TEXTURE_INFO;
      ti.hash = kCutoutTexHash;
      ti.width = 128; ti.height = 128; ti.depth = 1; ti.mipLevels = 1;
      ti.format = REMIXAPI_FORMAT_R8G8B8A8_UNORM;
      ti.data = px.data();
      ti.dataSize = px.size();
      remixapi_TextureHandle th;
      check(g_remix.CreateTexture(&ti, &th), "CreateTexture cutout");
      const std::wstring cutoutPath = hashPath(kCutoutTexHash);
      remixapi_MaterialInfoOpaqueEXT opaque {};
      opaque.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT;
      opaque.albedoConstant = { 1, 1, 1 };
      opaque.opacityConstant = 1;
      opaque.alphaTestType = 4;          // greater
      opaque.alphaReferenceValue = 127;
      opaque.useDrawCallAlphaState = 0;
      opaque.roughnessConstant = 0.6f;
      remixapi_MaterialInfo mi {};
      mi.sType = REMIXAPI_STRUCT_TYPE_MATERIAL_INFO;
      mi.pNext = &opaque;
      mi.hash = 0x7000000000000010ull;
      mi.albedoTexture = cutoutPath.c_str();
      mi.filterMode = 0;
      mi.wrapModeU = 1;
      mi.wrapModeV = 1;
      check(g_remix.CreateMaterial(&mi, &clusterMat), "CreateMaterial cutout");
    } else {
      cube(cv, ci);
    }
    const int n = clustersCount;
    const int side = int(std::ceil(std::sqrt(double(n))));
    std::vector<remixapi_MeshHandle> meshes(n);
    std::vector<uint64_t> field(n);
    std::vector<remixapi_Transform> transforms(n);
    std::vector<uint32_t> generation(n, 0);
    // A streamed-in object is a new mesh (new hash) at the same place, as when a cell reloads.
    const auto createMesh = [&](int i) {
      remixapi_MeshInfoSurfaceTriangles s {};
      s.vertices_values = cv.data(); s.vertices_count = cv.size();
      s.indices_values = ci.data(); s.indices_count = ci.size();
      s.material = clusterMat;
      remixapi_MeshInfo mi {};
      mi.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
      mi.hash = 0x9000000000000000ull + (uint64_t(generation[i]) << 32) + uint64_t(i);
      mi.surfaces_values = &s;
      mi.surfaces_count = 1;
      check(g_remix.CreateMesh(&mi, &meshes[i]), "CreateMesh cluster cube");
    };
    for (int i = 0; i < n; ++i) {
      remixapi_MeshInfoSurfaceTriangles s {};
      s.vertices_values = cv.data(); s.vertices_count = cv.size();
      s.indices_values = ci.data(); s.indices_count = ci.size();
      s.material = clusterMat;
      remixapi_MeshInfo mi {};
      mi.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
      mi.hash = 0x9000000000000000ull + uint64_t(i);
      mi.surfaces_values = &s;
      mi.surfaces_count = 1;
      check(g_remix.CreateMesh(&mi, &meshes[i]), "CreateMesh cluster cube");
      const float x = 1.5f * float(i % side) - 0.75f * float(side);
      const float z = 1.5f * float(i / side) - 0.75f * float(side);
      transforms[i] = translate(x, 0.5f, z, 0.8f);
    }
    const auto createOne = [&](int i) {
      remixapi_InstanceInfo ii {};
      ii.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
      ii.mesh = meshes[i];
      ii.transform = transforms[i];
      check(createRetained(&ii, &field[i]), "CreateRetainedInstance cluster cube");
    };
    // --alpha-cutout cubes are drawn with DrawInstance every frame: the OMM manager only bakes for
    // instances it sees updated, and retained direct-lane instances never get an OMM.
    for (int i = 0; i < n && !alphaCutout; ++i) {
      createOne(i);
    }
    const int streamStart = 40, streamEnd = std::max(streamStart, frames - 60);
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    std::vector<double> ms(frames, 0.0);
    int next = 0;
    for (int frame = 0; frame < frames; ++frame) {
      MSG msg;
      while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
      QueryPerformanceCounter(&t0);
      if (streamPerFrame > 0 && frame >= streamStart && frame < streamEnd) {
        for (int k = 0; k < streamPerFrame; ++k) {
          const int i = next;
          next = (next + 1) % n;
          destroyRetained(&field[i], 1);
          g_remix.DestroyMesh(meshes[i]);
          ++generation[i];
          createMesh(i);
          createOne(i);
        }
      }
      if (moversCount > 0 && frame >= streamStart && frame < streamEnd) {
        // Mover m (cube m * n / M) moves on frames phase..phase+2 of each period, then rests.
        std::vector<remixapi_RetainedInstanceUpdate> ups;
        const int m = std::min(moversCount, n);
        for (int k = 0; k < m; ++k) {
          const int i = int((int64_t(k) * n) / m);
          const int phase = (k * 7) % moverPeriod;
          const int t = (frame - streamStart + moverPeriod - phase) % moverPeriod;
          if (t >= 3) {
            continue;
          }
          remixapi_RetainedInstanceUpdate up {};
          up.handle = field[i];
          up.flags = REMIXAPI_RETAINED_UPDATE_TRANSFORM;
          // Alternate bursts slide the cube up and back down, so it stays near its grid spot.
          const int burst = (frame - streamStart + moverPeriod - phase) / moverPeriod;
          const float lift = 0.05f * float((burst & 1) ? 3 - t - 1 : t + 1);
          transforms[i].matrix[1][3] = 0.5f + lift;
          up.transform = transforms[i];
          ups.push_back(up);
        }
        if (!ups.empty()) {
          check(updateRetainedMany(ups.data(), uint32_t(ups.size())), "UpdateRetainedInstances movers");
        }
      }
      remixapi_CameraInfoParameterizedEXT cam {};
      cam.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT;
      const float h = 0.9f * float(side);
      cam.position = { 0.f, h, -h };
      cam.forward = { 0.f, -0.707f, 0.707f };
      cam.right = { 1.f, 0.f, 0.f };
      cam.up = { 0.f, 0.707f, 0.707f };
      cam.fovYInDegrees = 60;
      cam.aspect = float(width) / float(height);
      cam.nearPlane = 0.1f;
      cam.farPlane = 1000;
      remixapi_CameraInfo cinfo {};
      cinfo.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
      cinfo.pNext = &cam;
      cinfo.type = REMIXAPI_CAMERA_TYPE_WORLD;
      check(g_remix.SetupCamera(&cinfo), "SetupCamera");
      remixapi_InstanceInfo ground {};
      ground.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
      ground.mesh = groundMesh;
      ground.transform = translate(0, 0, 0, 0.1f * float(side));
      ground.doubleSided = 1;
      check(g_remix.DrawInstance(&ground), "DrawInstance ground");
      if (alphaCutout) {
        for (int i = 0; i < n; ++i) {
          remixapi_InstanceInfo ii {};
          ii.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
          ii.mesh = meshes[i];
          ii.transform = transforms[i];
          check(g_remix.DrawInstance(&ii), "DrawInstance cutout cube");
        }
      }
      check(drawRetained(nullptr), "DrawRetainedInstances");
      check(g_remix.DrawLightInstance(sunLight), "DrawLightInstance sun");
      if (d3d11) {
        ID3D11Texture2D* backbuffer = nullptr;
        swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backbuffer));
        check(renderD3D11(backbuffer, 1), "RenderD3D11");
        backbuffer->Release();
        swapchain->Present(0, 0);
      } else {
        remixapi_PresentInfo pi {};
        pi.sType = REMIXAPI_STRUCT_TYPE_PRESENT_INFO;
        check(g_remix.Present(&pi), "Present");
      }
      QueryPerformanceCounter(&t1);
      ms[frame] = double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(freq.QuadPart);
    }
    const auto stats = [&](const char* name, int a, int b) {
      a = std::max(a, 10);
      b = std::min(b, frames);
      if (b <= a) {
        return;
      }
      std::vector<double> sorted(ms.begin() + a, ms.begin() + b);
      std::sort(sorted.begin(), sorted.end());
      std::printf("host frame CPU %s (frames %d-%d): p50 %.2f ms, p99 %.2f ms, max %.2f ms\n", name, a, b - 1,
                  sorted[sorted.size() / 2], sorted[std::min(sorted.size() - 1, size_t(0.99 * double(sorted.size())))],
                  sorted.back());
    };
    stats("steady", 10, streamStart);
    stats("streaming", streamStart, streamEnd);
    stats("after", streamEnd, frames);
    if (!alphaCutout) {
      destroyRetained(field.data(), uint32_t(field.size()));
    }
    g_remix.Shutdown();
    if (swapchain) {
      swapchain->Release();
    }
    if (context) {
      context->Release();
    }
    if (device) {
      device->Release();
    }
    std::printf("testhost: %d frames done (clusters %d, stream %d, movers %d / %d, %s)\n", frames, n, streamPerFrame,
                moversCount, moverPeriod, d3d11 ? "d3d11" : "d3d9");
    return 0;
  }

  // Retained instances: two cubes, one with a material override.
  uint64_t retained[2] = {};
  if (originShift) {
    // A field of static cubes around the origin-grid line x = 3 * 16384, the camera crossing it mid-run.
    constexpr float kGrid = 16384.f;
    constexpr float kLine = 3.f * kGrid;
    constexpr int kSide = 40;
    std::vector<uint64_t> field(instanced ? kSide : kSide * kSide);
    remixapi_InstanceInfo ii {};
    ii.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    ii.mesh = cubeMesh;
    if (instanced) {
      std::vector<remixapi_Transform> row(kSide);
      for (int x = 0; x < kSide; ++x) {
        row[x] = translate(3.f * float(x), 0.f, 0.f);
      }
      remixapi_InstanceInfoGpuInstancingEXT gi {};
      gi.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT;
      gi.instanceTransforms_values = row.data();
      gi.instanceTransforms_count = uint32_t(row.size());
      ii.pNext = &gi;
      for (int z = 0; z < kSide; ++z) {
        ii.transform = translate(kLine - 60.f, 0.5f, -60.f + 3.f * float(z));
        check(createRetained(&ii, &field[z]), "CreateRetainedInstance instanced row");
      }
      ii.pNext = nullptr;
    } else {
      std::vector<remixapi_HardcodedVertex> dv;
      std::vector<uint32_t> di;
      if (distinctMeshes) {
        cube(dv, di);
      }
      for (int z = 0; z < kSide; ++z) {
        for (int x = 0; x < kSide; ++x) {
          if (distinctMeshes) {
            remixapi_MeshInfoSurfaceTriangles s {};
            s.vertices_values = dv.data(); s.vertices_count = dv.size();
            s.indices_values = di.data(); s.indices_count = di.size();
            s.material = cubeMat;
            remixapi_MeshInfo mi {};
            mi.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
            mi.hash = 0xA000000000000000ull + uint64_t(z * kSide + x);
            mi.surfaces_values = &s;
            mi.surfaces_count = 1;
            remixapi_MeshHandle mesh = nullptr;
            check(g_remix.CreateMesh(&mi, &mesh), "CreateMesh field");
            ii.mesh = mesh;
          }
          ii.transform = translate(kLine - 60.f + 3.f * float(x), 0.5f, -60.f + 3.f * float(z));
          check(createRetained(&ii, &field[z * kSide + x]), "CreateRetainedInstance field");
        }
      }
    }
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    std::vector<double> ms(frames, 0.0);
    int shiftFrame = -1;
    float lastOrigin = -1.f;
    for (int frame = 0; frame < frames; ++frame) {
      MSG msg;
      while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
      QueryPerformanceCounter(&t0);
      // From 60 units before the line to 60 after, then hold the last 30 frames.
      const int t = std::min(frame, std::max(0, frames - 30));
      const float camX = kLine - 60.f + 120.f * float(t) / float(std::max(1, frames - 30));
      const float origin = std::floor(camX / kGrid) * kGrid;
      remixapi_CameraInfoFrameEventsEXT events {};
      events.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_FRAME_EVENTS_EXT;
      bool shifted = false;
      if (lastOrigin >= 0.f && origin != lastOrigin) {
        shiftFrame = frame;
        shifted = true;
        events.flags = REMIXAPI_CAMERA_FRAME_EVENT_ORIGIN_SHIFT;
        events.originShift = { lastOrigin - origin, 0.f, 0.f };
        std::printf("origin shift at frame %d: %.0f -> %.0f\n", frame, lastOrigin, origin);
      }
      lastOrigin = origin;
      remixapi_CameraInfoParameterizedEXT cam {};
      cam.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT;
      cam.position = { camX - origin, 6.f, -20.f };
      cam.forward = { 0.f, -0.3f, 0.954f };
      cam.right = { 1.f, 0.f, 0.f };
      cam.up = { 0.f, 0.954f, 0.3f };
      cam.fovYInDegrees = 60;
      cam.aspect = float(width) / float(height);
      cam.nearPlane = 0.1f;
      cam.farPlane = 1000;
      remixapi_CameraInfo ci {};
      ci.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
      ci.pNext = &cam;
      ci.type = REMIXAPI_CAMERA_TYPE_WORLD;
      if (cameraEvents && shifted) {
        cam.pNext = &events;
      }
      check(g_remix.SetupCamera(&ci), "SetupCamera");
      remixapi_InstanceInfo ground {};
      ground.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
      ground.mesh = groundMesh;
      ground.transform = translate(kLine - origin, 0, 0, 10.f);
      ground.doubleSided = 1;
      check(g_remix.DrawInstance(&ground), "DrawInstance ground");
      const float originXYZ[3] = { origin, 0.f, 0.f };
      check(drawRetained(originXYZ), "DrawRetainedInstances");
      check(g_remix.DrawLightInstance(sunLight), "DrawLightInstance sun");
      if (d3d11) {
        ID3D11Texture2D* backbuffer = nullptr;
        swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backbuffer));
        check(renderD3D11(backbuffer, 1), "RenderD3D11");
        backbuffer->Release();
        swapchain->Present(0, 0);
      } else {
        remixapi_PresentInfo pi {};
        pi.sType = REMIXAPI_STRUCT_TYPE_PRESENT_INFO;
        check(g_remix.Present(&pi), "Present");
      }
      QueryPerformanceCounter(&t1);
      ms[frame] = double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(freq.QuadPart);
    }
    std::vector<double> sorted(ms.begin() + std::min(frames, 10), ms.end());
    std::sort(sorted.begin(), sorted.end());
    if (!sorted.empty()) {
      std::printf("host frame CPU: p50 %.2f ms, p99 %.2f ms, max %.2f ms\n", sorted[sorted.size() / 2],
                  sorted[std::min(sorted.size() - 1, size_t(0.99 * double(sorted.size())))], sorted.back());
    }
    if (shiftFrame >= 0) {
      for (int f = std::max(0, shiftFrame - 3); f < std::min(frames, shiftFrame + 6); ++f) {
        std::printf("  frame %d: %.2f ms%s\n", f, ms[f], f == shiftFrame ? "  <- origin shift" : "");
      }
    }
    destroyRetained(field.data(), uint32_t(field.size()));
    g_remix.Shutdown();
    if (swapchain) {
      swapchain->Release();
    }
    if (context) {
      context->Release();
    }
    if (device) {
      device->Release();
    }
    std::printf("testhost: %d frames done (origin shift, %s)\n", frames, d3d11 ? "d3d11" : "d3d9");
    return 0;
  }
  {
    remixapi_InstanceInfo ii {};
    ii.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    ii.mesh = cubeMesh;
    ii.transform = translate(-2.5f, 0.5f, 1.5f);
    ii.doubleSided = 0;
    remixapi_InstanceInfoMaterialOverrideEXT ov {};
    ov.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_MATERIAL_OVERRIDE_EXT;
    ov.material = bc7Mat;
    ii.pNext = bc7Mat ? &ov : nullptr;
    check(createRetained(&ii, &retained[0]), "CreateRetainedInstance 0");
    ov.material = blueMat;
    ii.pNext = &ov;
    ii.transform = translate(2.5f, 0.75f, 1.5f, 1.5f);
    check(createRetained(&ii, &retained[1]), "CreateRetainedInstance 1");
  }
  // --asset-key: a second cube mesh keyed by the host, reaching the cube's mod through an alias.
  remixapi_MeshHandle keyedMesh = nullptr;
  uint64_t keyedInstance = 0;
  if (assetKey) {
    std::vector<remixapi_HardcodedVertex> kv;
    std::vector<uint32_t> ki;
    cube(kv, ki);
    remixapi_MeshInfoSurfaceTriangles ks {};
    ks.vertices_values = kv.data(); ks.vertices_count = kv.size();
    ks.indices_values = ki.data(); ks.indices_count = ki.size();
    ks.material = cubeMat;
    const uint64_t aliases[1] = { 0x8000000000000002ull };
    remixapi_AssetKeyEXT key {};
    key.sType = REMIXAPI_STRUCT_TYPE_ASSET_KEY_EXT;
    key.key = 0x0A55E70000000001ull;
    key.aliases_values = aliases;
    key.aliases_count = 1;
    key.assetName = "testhost/keyed_cube";
    remixapi_MeshInfo kmi {};
    kmi.sType = REMIXAPI_STRUCT_TYPE_MESH_INFO;
    kmi.pNext = &key;
    kmi.hash = 0x8000000000000010ull;
    kmi.surfaces_values = &ks;
    kmi.surfaces_count = 1;
    check(g_remix.CreateMesh(&kmi, &keyedMesh), "CreateMesh keyed");
    remixapi_InstanceInfo ki2 {};
    ki2.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    ki2.mesh = keyedMesh;
    ki2.transform = translate(0.f, 0.5f, -4.f);
    check(createRetained(&ki2, &keyedInstance), "CreateRetainedInstance keyed");
  }
  uint64_t serviced[3] = {};
  if (particles) {
    remixapi_InstanceInfoParticleSystemEXT ps {};
    ps.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_PARTICLE_SYSTEM_EXT;
    ps.maxNumParticles = 2000;
    ps.hideEmitter = 1;
    ps.minTimeToLive = 1.f;
    ps.maxTimeToLive = 3.f;
    ps.spawnRatePerSecond = 300.f;
    ps.initialVelocityFromNormal = 2.f;
    ps.initialVelocityConeAngleDegrees = 30.f;
    ps.gravityForce = -0.5f;
    remixapi_InstanceInfo pi {};
    pi.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    pi.pNext = &ps;
    pi.mesh = cubeMesh;
    pi.transform = translate(0.f, 0.5f, 3.f, 0.5f);
    check(createRetained(&pi, &serviced[0]), "CreateRetainedInstance particles");
    pi.pNext = nullptr;
    pi.categoryFlags = REMIXAPI_INSTANCE_CATEGORY_BIT_PARTICLE_EMITTER;
    pi.transform = translate(-3.f, 0.5f, 3.f, 0.5f);
    check(createRetained(&pi, &serviced[1]), "CreateRetainedInstance emitter category");
    pi.categoryFlags = REMIXAPI_INSTANCE_CATEGORY_BIT_VIEW_MODEL;
    pi.transform = translate(3.f, 0.5f, 3.f, 0.5f);
    check(createRetained(&pi, &serviced[2]), "CreateRetainedInstance view model");
  }

  for (int frame = 0; frame < frames; ++frame) {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    // The last 30 frames are static, so a screenshot taken on any of them shows the same scene.
    const int t = std::min(frame, std::max(0, frames - 30));
    const float angle = 0.6f + 0.01f * float(t);
    remixapi_CameraInfoParameterizedEXT cam {};
    cam.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT;
    cam.position = { 9.f * std::sin(angle), 3.5f, -9.f * std::cos(angle) };
    const float fx = -cam.position.x, fy = -cam.position.y + 0.5f, fz = -cam.position.z;
    const float fl = std::sqrt(fx * fx + fy * fy + fz * fz);
    cam.forward = { fx / fl, fy / fl, fz / fl };
    // right = up(0,1,0) x forward, normalized; up = forward x right
    float rx = cam.forward.z, rz = -cam.forward.x;
    const float rl = std::sqrt(rx * rx + rz * rz);
    rx /= rl; rz /= rl;
    cam.right = { rx, 0, rz };
    cam.up = { cam.forward.y * rz, cam.forward.z * rx - cam.forward.x * rz, -cam.forward.y * rx };
    cam.fovYInDegrees = 60;
    cam.aspect = float(width) / float(height);
    cam.nearPlane = 0.1f;
    cam.farPlane = 1000;
    remixapi_CameraInfo ci {};
    ci.sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO;
    ci.pNext = &cam;
    ci.type = REMIXAPI_CAMERA_TYPE_WORLD;
    check(g_remix.SetupCamera(&ci), "SetupCamera");

    remixapi_InstanceInfo ii {};
    ii.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
    ii.mesh = groundMesh;
    ii.transform = translate(0, 0, 0);
    ii.doubleSided = 1;
    check(g_remix.DrawInstance(&ii), "DrawInstance ground");

    remixapi_Transform row[3] = { translate(-1.2f, 0.5f, -1.5f), translate(0, 0.5f + 0.5f * std::sin(0.1f * float(t)), -1.5f), translate(1.2f, 0.5f, -1.5f) };
    remixapi_InstanceInfoGpuInstancingEXT inst {};
    inst.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT;
    inst.instanceTransforms_values = row;
    inst.instanceTransforms_count = 3;
    remixapi_InstanceInfoMaterialOverrideEXT rowMaterial {};
    rowMaterial.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_MATERIAL_OVERRIDE_EXT;
    rowMaterial.material = bc1Mat;
    inst.pNext = bc1Mat ? &rowMaterial : nullptr;
    ii.pNext = &inst;
    ii.mesh = cubeMesh;
    ii.doubleSided = 0;
    check(g_remix.DrawInstance(&ii), "DrawInstance cubes");

    remixapi_RetainedInstanceUpdate up {};
    up.handle = retained[0];
    up.flags = REMIXAPI_RETAINED_UPDATE_TRANSFORM;
    up.transform = translate(-2.5f, 0.5f, 1.5f + 0.02f * float(t));
    check(updateRetainedMany(&up, 1), "UpdateRetainedInstances");
    check(drawRetained(nullptr), "DrawRetainedInstances");

    for (int k = 0; k < skinnedCount; ++k) {
      const int s = reorder ? (k + frame) % skinnedCount : k;
      const float a = 6.2831853f * float(s) / float(skinnedCount);
      remixapi_Transform boneTransforms[2] = { translate(0, 0, 0), translate(0.3f * std::sin(0.05f * float(t) + a), 0, 0) };
      remixapi_InstanceInfoBoneTransformsEXT bonesExt {};
      bonesExt.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BONE_TRANSFORMS_EXT;
      bonesExt.boneTransforms_values = boneTransforms;
      bonesExt.boneTransforms_count = 2;
      remixapi_InstanceInfo si {};
      si.sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO;
      si.pNext = &bonesExt;
      si.mesh = skinnedMesh;
      si.transform = translate(5.f * std::cos(a), 0.5f, 5.f * std::sin(a), 0.6f);
      check(g_remix.DrawInstance(&si), "DrawInstance skinned");
    }

    check(g_remix.DrawLightInstance(sphereLight), "DrawLightInstance sphere");
    check(g_remix.DrawLightInstance(sunLight), "DrawLightInstance sun");
    if (usdMod && frame == modEditFrame) {
      writeUsdMod(800.f, 1.4f);
      std::printf("frame %d: mod rewritten\n", frame);
    }
    for (int u = 0; u < lightUpdates && !gridLights.empty(); ++u) {
      const int k = (frame * std::max(1, lightUpdates) + u) % kGridLights;
      remixapi_LightInfoSphereEXT sphere;
      remixapi_LightInfo li;
      gridLightInfo(k, 0.05f * std::sin(0.1f * float(t) + float(k)), sphere, li);
      check(g_remix.UpdateLightDefinition(gridLights[k], &li), "UpdateLightDefinition grid");
    }

    // A streamed mip upgrade: the cube texture's image is replaced in place with a green checker.
    if (d3d11 && frame == 10) {
      uint32_t mips;
      std::vector<uint8_t> px = checker(128, mips, 30, 200, 60, 240, 240, 240);
      importD3D11(DXGI_FORMAT_R8G8B8A8_UNORM, 128, 0, px, D3D11_USAGE_IMMUTABLE, 0, 0,
                  reinterpret_cast<remixapi_TextureHandle>(cubeTexHash));
    }

    if (d3d11) {
      ID3D11Texture2D* backbuffer = nullptr;
      swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&backbuffer));
      check(renderD3D11(backbuffer, 1), "RenderD3D11");
      backbuffer->Release();
      swapchain->Present(0, 0);
    } else {
      remixapi_PresentInfo pi {};
      pi.sType = REMIXAPI_STRUCT_TYPE_PRESENT_INFO;
      check(g_remix.Present(&pi), "Present");
    }
  }

  destroyRetained(retained, 2);
  if (keyedInstance != 0) {
    destroyRetained(&keyedInstance, 1);
  }
  if (particles) {
    destroyRetained(serviced, 3);
  }
  g_remix.Shutdown();
  if (usdMod) {
    std::error_code ec;
    fs::remove_all(usdModDir, ec);
  }
  if (swapchain) {
    swapchain->Release();
  }
  if (context) {
    context->Release();
  }
  if (device) {
    device->Release();
  }
  std::printf("testhost: %d frames done (%s)\n", frames, d3d11 ? "d3d11" : "d3d9");
  return 0;
}
