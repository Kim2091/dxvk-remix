// Standalone contract/lifetime smoke test. Does not launch Fallout 4.
#include <windows.h>
#include <d3d9.h>
#include <d3d11.h>
#include <remix/remix_c.h>
#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char* name) {
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
  std::fflush(stdout);
  if (!ok) std::exit(1);
}

int wmain(int argc, wchar_t** argv) {
  if (argc != 3) return 2;
  SetDllDirectoryW(argv[2]);
  HMODULE runtime = LoadLibraryW(argv[1]);
  check(runtime != nullptr, "load runtime");
  auto init = reinterpret_cast<PFN_remixapi_InitializeLibrary>(GetProcAddress(runtime, "remixapi_InitializeLibrary"));
  auto prepare = reinterpret_cast<PFN_remixapi_PrepareD3D11Texture>(GetProcAddress(runtime, "remixapi_PrepareD3D11Texture"));
  auto import = reinterpret_cast<PFN_remixapi_ImportD3D11Texture>(GetProcAddress(runtime, "remixapi_ImportD3D11Texture"));
  auto release = reinterpret_cast<PFN_remixapi_ReleaseD3D11Texture>(GetProcAddress(runtime, "remixapi_ReleaseD3D11Texture"));
  auto create11 = reinterpret_cast<decltype(&D3D11CreateDevice)>(GetProcAddress(runtime, "D3D11CreateDevice"));
  check(init && prepare && import && release && create11, "optional exports present");
  remixapi_InitializeLibraryInfo initInfo = {};
  initInfo.sType = REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO;
  initInfo.version = REMIXAPI_VERSION_MAKE(REMIXAPI_VERSION_MAJOR, REMIXAPI_VERSION_MINOR, REMIXAPI_VERSION_PATCH);
  remixapi_Interface api = {};
  check(init(&initInfo, &api) == REMIXAPI_ERROR_CODE_SUCCESS, "initialize unchanged API table");
  ID3D11Device* d11 = nullptr;
  ID3D11DeviceContext* context = nullptr;
  check(SUCCEEDED(create11(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
    D3D11_SDK_VERSION, &d11, nullptr, &context)), "create producer first");
  HWND window = CreateWindowExW(0, L"STATIC", L"Import contract test", WS_POPUP,
    0, 0, 64, 64, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
  check(window != nullptr, "hidden test window");
  const LONG_PTR originalWndProc = GetWindowLongPtrW(window, GWLP_WNDPROC);
  IDirect3D9Ex* d9 = nullptr;
  IDirect3DDevice9Ex* consumer = nullptr;
  check(api.dxvk_CreateD3D9(FALSE, &d9) == REMIXAPI_ERROR_CODE_SUCCESS, "create Remix frontend");
  D3DPRESENT_PARAMETERS pp = {};
  pp.BackBufferWidth = pp.BackBufferHeight = 64;
  pp.BackBufferFormat = D3DFMT_A8R8G8B8;
  pp.BackBufferCount = 1;
  pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  pp.hDeviceWindow = window;
  pp.Windowed = TRUE;
  pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
  check(SUCCEEDED(d9->CreateDeviceEx(0, D3DDEVTYPE_HAL, window,
    D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,
    &pp, nullptr, &consumer)), "create shared consumer");
  check(api.dxvk_RegisterD3D9Device(consumer) == REMIXAPI_ERROR_CODE_SUCCESS, "register consumer");
  remixapi_D3D11TextureToken token = reinterpret_cast<void*>(1);
  check(prepare(nullptr, 0, &token) != REMIXAPI_ERROR_CODE_SUCCESS && token == nullptr, "reject null source and clear output");
  uint32_t pixels[4] = {0xff0000ff, 0xff00ff00, 0xffff0000, 0xffffffff};
  D3D11_SUBRESOURCE_DATA data = {pixels, 8, 16};
  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = desc.Height = 2;
  desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
  desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  desc.Usage = D3D11_USAGE_DYNAMIC;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  ID3D11Texture2D* texture = nullptr;
  check(SUCCEEDED(d11->CreateTexture2D(&desc, &data, &texture)), "create mutable texture");
  check(prepare(texture, desc.Format, &token) != REMIXAPI_ERROR_CODE_SUCCESS && token == nullptr, "reject mutable resource");
  texture->Release();
  desc.Usage = D3D11_USAGE_IMMUTABLE;
  desc.CPUAccessFlags = 0;
  check(SUCCEEDED(d11->CreateTexture2D(&desc, &data, &texture)), "create immutable texture");
  check(prepare(texture, DXGI_FORMAT_R32_FLOAT, &token) != REMIXAPI_ERROR_CODE_SUCCESS && token == nullptr, "reject incompatible format");
  check(prepare(texture, desc.Format, &token) == REMIXAPI_ERROR_CODE_SUCCESS && token, "prepare immutable image");
  texture->Release();
  remixapi_TextureHandle handle = nullptr;
  check(import(token, 0, &handle) != REMIXAPI_ERROR_CODE_SUCCESS && !handle, "reject zero hash");
  check(import(token, 0x47110001, &handle) == REMIXAPI_ERROR_CODE_SUCCESS && handle, "import after source COM release");
  release(token);
  remixapi_PresentInfo present = {};
  present.sType = REMIXAPI_STRUCT_TYPE_PRESENT_INFO;
  check(api.Present(&present) == REMIXAPI_ERROR_CODE_SUCCESS, "submit retained view and barrier");
  check(api.DestroyTexture(handle) == REMIXAPI_ERROR_CODE_SUCCESS, "retire imported texture");
  check(api.Present(&present) == REMIXAPI_ERROR_CODE_SUCCESS, "submit retirement");
  context->Release();
  d11->Release();
  SetWindowLongPtrW(window, GWLP_WNDPROC, originalWndProc);
  api.Shutdown();
  // This runtime's Shutdown drains the registered D3D9 references itself.
  DestroyWindow(window);
  std::puts("PASS: import contract smoke test complete (no pixel readback assertion)");
  return 0;
}
