// remix_testhost D3D9 scenarios: remix_testhost --runtime <dir> --d3d9 <name> [--frames N] [--event-frame F]
//                                [--hidden-frames N] [--overlay]
//
// Each scenario drives the runtime's D3D9 frontend (the path D3D9 games take, not the Remix API) through a device made
// by <runtime>/d3d9.dll. Every draw is fixed function and depends only on the frame index, so two runs draw the same
// frames. Scenarios with one event (a Reset, a reclassification, a bone switch) run it at --event-frame (default 450),
// inside the second 300-frame log interval of a 720-frame run.
//
// tools/remix_replay/d3d9v2_scenarios.sh checks the counters of each scenario. --d3d9-hashes prints the texture hashes the rtx.conf tags of
// the terrain, emitter, portal and render-target scenarios need.

#include "remix_testhost_d3d9.h"

#include <d3d9.h>
#include <d3dcompiler.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define XXH_INLINE_ALL
#include "../../src/util/xxHash/xxhash.h"

namespace fs = std::filesystem;

namespace {
  constexpr float kPi = 3.14159265f;
  constexpr DWORD kFvf = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1;
  constexpr uint32_t kTexSize = 64;
  constexpr uint32_t kMonitorSize = 256;

  struct Vertex {
    float x, y, z, nx, ny, nz, u, v;
  };

  // Fixed-function indexed vertex blending: two weights (the last one is implied), four bone indices.
  struct SkinVertex {
    float x, y, z;
    float w0, w1;
    uint8_t bones[4];
    float nx, ny, nz;
    float u, v;
  };
  static_assert(sizeof(SkinVertex) == 44, "matches the vertex declaration");

  struct Vec3 {
    float x, y, z;
  };

  Vec3 sub(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
  Vec3 cross(const Vec3& a, const Vec3& b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
  float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
  Vec3 normalize(const Vec3& a) {
    const float l = std::sqrt(dot(a, a));
    return { a.x / l, a.y / l, a.z / l };
  }

  // Matrices use the D3D9 row-vector convention: v' = v * world * view * projection.
  D3DMATRIX identity() {
    D3DMATRIX m {};
    m._11 = m._22 = m._33 = m._44 = 1.f;
    return m;
  }

  D3DMATRIX mul(const D3DMATRIX& a, const D3DMATRIX& b) {
    D3DMATRIX r {};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 4; ++k) {
          r.m[i][j] += a.m[i][k] * b.m[k][j];
        }
      }
    }
    return r;
  }

  D3DMATRIX transpose(const D3DMATRIX& a) {
    D3DMATRIX r {};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        r.m[i][j] = a.m[j][i];
      }
    }
    return r;
  }

  D3DMATRIX translation(float x, float y, float z) {
    D3DMATRIX m = identity();
    m._41 = x; m._42 = y; m._43 = z;
    return m;
  }

  D3DMATRIX scaled(float s, float x, float y, float z) {
    D3DMATRIX m = translation(x, y, z);
    m._11 = m._22 = m._33 = s;
    return m;
  }

  D3DMATRIX rotationZ(float a) {
    D3DMATRIX m = identity();
    const float c = std::cos(a), s = std::sin(a);
    m._11 = c; m._12 = s;
    m._21 = -s; m._22 = c;
    return m;
  }

  D3DMATRIX lookAt(const Vec3& eye, const Vec3& at) {
    const Vec3 z = normalize(sub(at, eye));
    const Vec3 x = normalize(cross({ 0.f, 1.f, 0.f }, z));
    const Vec3 y = cross(z, x);
    D3DMATRIX m = identity();
    m._11 = x.x; m._12 = y.x; m._13 = z.x;
    m._21 = x.y; m._22 = y.y; m._23 = z.y;
    m._31 = x.z; m._32 = y.z; m._33 = z.z;
    m._41 = -dot(x, eye); m._42 = -dot(y, eye); m._43 = -dot(z, eye);
    return m;
  }

  D3DMATRIX perspective(float fovY, float aspect, float zn, float zf) {
    D3DMATRIX m {};
    const float ys = 1.f / std::tan(fovY * 0.5f);
    m._11 = ys / aspect;
    m._22 = ys;
    m._33 = zf / (zf - zn);
    m._34 = 1.f;
    m._43 = -zn * zf / (zf - zn);
    return m;
  }

  // A box of size (sx, sy, sz) centred at (cx, cy, cz), 24 vertices and 36 indices appended.
  void box(std::vector<Vertex>& v, std::vector<uint16_t>& idx, float cx, float cy, float cz, float sx, float sy, float sz) {
    const float n[6][3] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
    for (int f = 0; f < 6; ++f) {
      const float* N = n[f];
      const float a[3] = { N[1], N[2], N[0] };
      const float b[3] = { N[1] * a[2] - N[2] * a[1], N[2] * a[0] - N[0] * a[2], N[0] * a[1] - N[1] * a[0] };
      const uint16_t base = uint16_t(v.size());
      const float s[4][2] = { {-1,-1}, {1,-1}, {1,1}, {-1,1} };
      for (int c = 0; c < 4; ++c) {
        Vertex out {};
        out.x = cx + 0.5f * sx * (N[0] + s[c][0] * a[0] + s[c][1] * b[0]);
        out.y = cy + 0.5f * sy * (N[1] + s[c][0] * a[1] + s[c][1] * b[1]);
        out.z = cz + 0.5f * sz * (N[2] + s[c][0] * a[2] + s[c][1] * b[2]);
        out.nx = N[0]; out.ny = N[1]; out.nz = N[2];
        out.u = (s[c][0] + 1) * 0.5f;
        out.v = (s[c][1] + 1) * 0.5f;
        v.push_back(out);
      }
      const uint16_t q[6] = { 0, 1, 2, 0, 2, 3 };
      for (uint16_t i : q) {
        idx.push_back(uint16_t(base + i));
      }
    }
  }

  enum class Tex : uint32_t { Ground, Crate, Terrain, Emitter, PortalA, PortalB, Count };
  constexpr const char* kTexNames[uint32_t(Tex::Count)] = { "ground", "crate", "terrain", "emitter", "portalA", "portalB" };

  // A8R8G8B8 checkers, one colour pair per kind, so every kind has its own texture hash.
  std::vector<uint32_t> texturePixels(Tex kind) {
    static const uint32_t kColours[uint32_t(Tex::Count)][2] = {
      { 0xFFC8C8C8u, 0xFF282828u }, { 0xFFDC3C1Eu, 0xFFF0C828u }, { 0xFF3C8C28u, 0xFF785028u },
      { 0xFFF0F0F0u, 0xFF3060F0u }, { 0xFF2080FFu, 0xFF103060u }, { 0xFFFF8020u, 0xFF603010u } };
    std::vector<uint32_t> px(kTexSize * kTexSize);
    for (uint32_t y = 0; y < kTexSize; ++y) {
      for (uint32_t x = 0; x < kTexSize; ++x) {
        px[y * kTexSize + x] = kColours[uint32_t(kind)][((x / 8) + (y / 8)) & 1];
      }
    }
    return px;
  }

  // The runtime hashes a managed texture's mip 0 bytes with XXH3 (D3D9CommonTexture::SetupForRtxFrom). The rows are
  // tightly packed, as the runtime's CPU copy is (GetMipSize aligns a row to 4 bytes, which 64 texels already are).
  uint64_t textureHash(Tex kind) {
    const std::vector<uint32_t> px = texturePixels(kind);
    return XXH3_64bits(px.data(), px.size() * sizeof(uint32_t));
  }

  // rtx.raytracedRenderTargetTextures holds descriptor hashes: XXH3 of D3D9_COMMON_TEXTURE_DESC
  // (src/d3d9/d3d9_common_texture.h). This copies its 44-byte layout for a 1-mip A8R8G8B8 render target.
  uint64_t renderTargetHash(uint32_t width, uint32_t height) {
    struct DescCopy {
      UINT width, height, depth, arraySize, mipLevels;
      DWORD usage;
      D3DFORMAT format;
      D3DPOOL pool;
      D3DMULTISAMPLE_TYPE multiSample;
      DWORD multisampleQuality;
      bool discard, isBackBuffer, isAttachmentOnly;
      unsigned char padding;
    };
    static_assert(sizeof(DescCopy) == 44, "D3D9_COMMON_TEXTURE_DESC is 44 bytes");
    DescCopy d;
    std::memset(&d, 0, sizeof(d));
    d.width = width;
    d.height = height;
    d.depth = 1;
    d.arraySize = 1;
    d.mipLevels = 1;
    d.usage = D3DUSAGE_RENDERTARGET;
    d.format = D3DFMT_A8R8G8B8;
    d.pool = D3DPOOL_DEFAULT;
    d.multiSample = D3DMULTISAMPLE_NONE;
    return XXH3_64bits(&d, sizeof(d));
  }

  template<typename T>
  void release(T*& p) {
    if (p != nullptr) {
      p->Release();
      p = nullptr;
    }
  }

  struct Mesh {
    IDirect3DVertexBuffer9* vb = nullptr;
    IDirect3DIndexBuffer9* ib = nullptr;
    IDirect3DVertexDeclaration9* decl = nullptr;
    UINT stride = sizeof(Vertex);
    UINT vertices = 0;
    UINT primitives = 0;

    void destroy() {
      release(vb);
      release(ib);
      release(decl);
    }
  };

  void fail(const char* what, HRESULT hr) {
    std::printf("FAILED: %s (0x%08lx)\n", what, static_cast<unsigned long>(hr));
    std::exit(1);
  }

  struct Host {
    IDirect3D9* m_d3d = nullptr;
    IDirect3DDevice9* m_device = nullptr;
    D3DPRESENT_PARAMETERS m_presentParams {};
    HWND m_hwnd = nullptr;
    float m_aspect = 1.f;
    D3DMATRIX m_projection {};
    IDirect3DTexture9* m_textures[uint32_t(Tex::Count)] = {};

    ~Host() {
      for (IDirect3DTexture9*& t : m_textures) {
        release(t);
      }
      release(m_device);
      release(m_d3d);
    }

    void create(const fs::path& runtime, HWND hwnd, uint32_t width, uint32_t height) {
      using PFN_Direct3DCreate9 = IDirect3D9* (WINAPI*)(UINT);
      HMODULE dll = LoadLibraryExW((runtime / "d3d9.dll").wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
      auto create9 = dll ? reinterpret_cast<PFN_Direct3DCreate9>(GetProcAddress(dll, "Direct3DCreate9")) : nullptr;
      if (create9 == nullptr) {
        fail("no Direct3DCreate9 in the runtime's d3d9.dll", E_FAIL);
      }
      m_d3d = create9(D3D_SDK_VERSION);
      if (m_d3d == nullptr) {
        fail("Direct3DCreate9", E_FAIL);
      }
      m_hwnd = hwnd;
      m_presentParams.BackBufferWidth = width;
      m_presentParams.BackBufferHeight = height;
      m_presentParams.BackBufferFormat = D3DFMT_X8R8G8B8;
      m_presentParams.BackBufferCount = 1;
      m_presentParams.SwapEffect = D3DSWAPEFFECT_DISCARD;
      m_presentParams.hDeviceWindow = hwnd;
      m_presentParams.Windowed = TRUE;
      m_presentParams.EnableAutoDepthStencil = TRUE;
      m_presentParams.AutoDepthStencilFormat = D3DFMT_D24S8;
      m_presentParams.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
      D3DPRESENT_PARAMETERS pp = m_presentParams;
      const HRESULT hr = m_d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hwnd,
                                             D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE, &pp, &m_device);
      if (FAILED(hr)) {
        fail("CreateDevice", hr);
      }
      m_aspect = float(width) / float(height);
      m_projection = perspective(kPi / 3.f, m_aspect, 0.1f, 1000.f);
      for (uint32_t k = 0; k < uint32_t(Tex::Count); ++k) {
        m_textures[k] = makeTexture(Tex(k));
      }
      setStates();
    }

    // Fixed-function state. Reset loses it, so the Reset scenario calls this again.
    void setStates() {
      m_device->SetRenderState(D3DRS_LIGHTING, FALSE);
      m_device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
      m_device->SetRenderState(D3DRS_ZENABLE, D3DZB_TRUE);
      m_device->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);
      m_device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
      m_device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
      m_device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
      m_device->SetTextureStageState(0, D3DTSS_COLORARG2, D3DTA_DIFFUSE);
      m_device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
      m_device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
      m_device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
      m_device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
      m_device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
      m_device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
      m_device->SetTransform(D3DTS_PROJECTION, &m_projection);
    }

    IDirect3DTexture9* makeTexture(Tex kind) {
      IDirect3DTexture9* tex = nullptr;
      HRESULT hr = m_device->CreateTexture(kTexSize, kTexSize, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &tex, nullptr);
      if (FAILED(hr)) {
        fail("CreateTexture", hr);
      }
      D3DLOCKED_RECT locked;
      hr = tex->LockRect(0, &locked, nullptr, 0);
      if (FAILED(hr)) {
        fail("LockRect", hr);
      }
      const std::vector<uint32_t> px = texturePixels(kind);
      for (uint32_t y = 0; y < kTexSize; ++y) {
        std::memcpy(static_cast<uint8_t*>(locked.pBits) + size_t(y) * locked.Pitch, px.data() + size_t(y) * kTexSize,
                    kTexSize * sizeof(uint32_t));
      }
      tex->UnlockRect(0);
      return tex;
    }

    IDirect3DTexture9* texture(Tex kind) const { return m_textures[uint32_t(kind)]; }

    template<typename V>
    Mesh makeMesh(const std::vector<V>& v, const std::vector<uint16_t>& idx, D3DPOOL pool, DWORD fvf) {
      Mesh mesh;
      mesh.stride = sizeof(V);
      mesh.vertices = UINT(v.size());
      mesh.primitives = UINT(idx.size() / 3);
      const DWORD usage = pool == D3DPOOL_DEFAULT ? D3DUSAGE_WRITEONLY : 0;
      HRESULT hr = m_device->CreateVertexBuffer(UINT(v.size() * sizeof(V)), usage, fvf, pool, &mesh.vb, nullptr);
      if (FAILED(hr)) {
        fail("CreateVertexBuffer", hr);
      }
      void* p = nullptr;
      if (FAILED(hr = mesh.vb->Lock(0, 0, &p, 0))) {
        fail("Lock vertex buffer", hr);
      }
      std::memcpy(p, v.data(), v.size() * sizeof(V));
      mesh.vb->Unlock();
      hr = m_device->CreateIndexBuffer(UINT(idx.size() * sizeof(uint16_t)), usage, D3DFMT_INDEX16, pool, &mesh.ib, nullptr);
      if (FAILED(hr)) {
        fail("CreateIndexBuffer", hr);
      }
      if (FAILED(hr = mesh.ib->Lock(0, 0, &p, 0))) {
        fail("Lock index buffer", hr);
      }
      std::memcpy(p, idx.data(), idx.size() * sizeof(uint16_t));
      mesh.ib->Unlock();
      return mesh;
    }

    Mesh crate(D3DPOOL pool = D3DPOOL_MANAGED) {
      std::vector<Vertex> v;
      std::vector<uint16_t> idx;
      box(v, idx, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
      return makeMesh(v, idx, pool, kFvf);
    }

    // A 40 x 40 floor at y = 0.
    Mesh ground(D3DPOOL pool = D3DPOOL_MANAGED) {
      const std::vector<Vertex> v = {
        { -20, 0, -20, 0, 1, 0, 0, 0 }, { 20, 0, -20, 0, 1, 0, 8, 0 }, { 20, 0, 20, 0, 1, 0, 8, 8 }, { -20, 0, 20, 0, 1, 0, 0, 8 } };
      const std::vector<uint16_t> idx = { 0, 2, 1, 0, 3, 2 };
      return makeMesh(v, idx, pool, kFvf);
    }

    // A 2 x 2 quad in the XY plane facing -z (towards the cameras of the scenarios).
    Mesh quad() {
      const std::vector<Vertex> v = {
        { -1, -1, 0, 0, 0, -1, 0, 1 }, { 1, -1, 0, 0, 0, -1, 1, 1 }, { 1, 1, 0, 0, 0, -1, 1, 0 }, { -1, 1, 0, 0, 0, -1, 0, 0 } };
      const std::vector<uint16_t> idx = { 0, 1, 2, 0, 2, 3 };
      return makeMesh(v, idx, D3DPOOL_MANAGED, kFvf);
    }

    void setView(const D3DMATRIX& view) {
      m_device->SetTransform(D3DTS_VIEW, &view);
    }

    void draw(const Mesh& mesh, const D3DMATRIX& world, IDirect3DTexture9* tex) {
      m_device->SetTransform(D3DTS_WORLD, &world);
      m_device->SetTexture(0, tex);
      m_device->SetStreamSource(0, mesh.vb, 0, mesh.stride);
      m_device->SetIndices(mesh.ib);
      if (mesh.decl != nullptr) {
        m_device->SetVertexDeclaration(mesh.decl);
      } else {
        m_device->SetFVF(kFvf);
      }
      m_device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, mesh.vertices, 0, mesh.primitives);
    }

    void begin() {
      MSG msg;
      while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
      m_device->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(64, 96, 128), 1.f, 0);
      m_device->BeginScene();
    }

    void end(bool present = true) {
      m_device->EndScene();
      if (present) {
        const HRESULT hr = m_device->Present(nullptr, nullptr, nullptr, nullptr);
        if (FAILED(hr)) {
          fail("Present", hr);
        }
      }
    }
  };

  // ---- ring: repeated meshes, culled and reordered ----------------------------------------------------------------

  constexpr int kRingCrates = 24;
  constexpr float kRingRadius = 8.f;

  // How a ring frame hands its matrices to D3D9.
  // Separate puts the camera into VIEW.
  // FusedWorld puts world * view into WORLD and leaves VIEW identity.
  // FusedView puts world * view into VIEW and leaves WORLD identity.
  enum class Fuse { Separate, FusedWorld, FusedView };

  // The camera sits in the ring's centre and turns +-60 degrees with a 240-frame period.
  float ringYaw(int frame) {
    return (kPi / 3.f) * std::sin(2.f * kPi * float(frame) / 240.f);
  }

  void shuffle(std::vector<int>& v, uint32_t seed) {
    for (size_t i = v.size(); i > 1; --i) {
      seed = seed * 1664525u + 1013904223u;
      std::swap(v[i - 1], v[(seed >> 8) % i]);
    }
  }

  // The crates the host draws this frame, in a per-frame shuffled order. A crate is drawn within 12 degrees beyond
  // the horizontal half field of view, so a crate that is not drawn is clearly outside the runtime's frustum.
  std::vector<int> ringDraws(int frame, float aspect) {
    const float limit = std::atan(std::tan(kPi / 6.f) * aspect) + 12.f * kPi / 180.f;
    const float yaw = ringYaw(frame);
    std::vector<int> out;
    for (int i = 0; i < kRingCrates; ++i) {
      float d = 2.f * kPi * float(i) / float(kRingCrates) - yaw;
      d = std::atan2(std::sin(d), std::cos(d));
      if (std::fabs(d) < limit) {
        out.push_back(i);
      }
    }
    shuffle(out, uint32_t(frame) * 2654435761u + 1u);
    return out;
  }

  int runRing(Host& h, const D3D9ScenarioArgs& a, Fuse fuse) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      const float yaw = ringYaw(frame);
      const Vec3 eye = { 0.f, 2.f, 0.f };
      const D3DMATRIX view = lookAt(eye, { 10.f * std::sin(yaw), 0.5f, 10.f * std::cos(yaw) });
      const auto drawFused = [&](const Mesh& mesh, const D3DMATRIX& world, IDirect3DTexture9* tex) {
        if (fuse == Fuse::FusedWorld) {
          h.setView(identity());
          h.draw(mesh, mul(world, view), tex);
        } else if (fuse == Fuse::FusedView) {
          h.setView(mul(world, view));
          h.draw(mesh, identity(), tex);
        } else {
          h.setView(view);
          h.draw(mesh, world, tex);
        }
      };
      drawFused(ground, identity(), h.texture(Tex::Ground));
      for (const int i : ringDraws(frame, h.m_aspect)) {
        const float angle = 2.f * kPi * float(i) / float(kRingCrates);
        drawFused(crate, scaled(0.8f, kRingRadius * std::sin(angle), 0.4f, kRingRadius * std::cos(angle)), h.texture(Tex::Crate));
      }
      h.end();
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  // ---- view-space vertices with identity WORLD and VIEW -----------------------------------------------------------

  // A game without a view matrix: every crate is written in view space into its own buffer, and WORLD and VIEW stay
  // identity, so the runtime's main camera has an identity view. Four crates behind the camera are drawn until the
  // event frame and then no longer. Anti-culling must be off with such a camera, so they are collected in both modes.
  int runIdentityView(Host& h, const D3D9ScenarioArgs& a) {
    constexpr int kFront = 8;
    constexpr int kBehind = 4;
    std::vector<Mesh> crates;
    for (int i = 0; i < kFront + kBehind; ++i) {
      std::vector<Vertex> v;
      std::vector<uint16_t> idx;
      const bool behind = i >= kFront;
      const float x = behind ? -3.f + 2.f * float(i - kFront) : -3.5f + float(i);
      box(v, idx, x, -0.5f, behind ? -4.f : 6.f, 0.8f, 0.8f, 0.8f);
      crates.push_back(h.makeMesh(v, idx, D3DPOOL_MANAGED, kFvf));
    }
    bool announced = false;
    for (int frame = 0; frame < a.frames; ++frame) {
      const bool drawBehind = frame < a.eventFrame;
      if (!drawBehind && !announced) {
        std::printf("event: the %d crates behind the camera are no longer drawn from frame %d\n", kBehind, frame);
        announced = true;
      }
      h.begin();
      h.setView(identity());
      for (int i = 0; i < kFront + kBehind; ++i) {
        if (i < kFront || drawBehind) {
          h.draw(crates[i], identity(), h.texture(Tex::Crate));
        }
      }
      h.end();
    }
    for (Mesh& m : crates) {
      m.destroy();
    }
    return 0;
  }

  // ---- one mesh drawn twice in a frame ----------------------------------------------------------------------------

  int runTwice(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    const D3DMATRIX world = translation(0.f, 0.5f, 4.f);
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      h.setView(lookAt({ 0.f, 2.5f, -1.f }, { 0.f, 0.5f, 4.f }));
      h.draw(ground, identity(), h.texture(Tex::Ground));
      // With --overlay one pass is blended additively over the other, with the same buffers, world and texture.
      // --overlay-first draws the blended pass first.
      const auto blendedPass = [&](bool on) {
        h.m_device->SetRenderState(D3DRS_ALPHABLENDENABLE, on ? TRUE : FALSE);
        h.m_device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
        h.m_device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ONE);
        h.m_device->SetRenderState(D3DRS_ZWRITEENABLE, on ? FALSE : TRUE);
        h.m_device->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
      };
      for (int pass = 0; pass < 2; ++pass) {
        const bool blended = a.overlay && (pass == 0) == a.overlayFirst;
        if (blended) {
          blendedPass(true);
        }
        h.draw(crate, world, h.texture(Tex::Crate));
        if (blended) {
          blendedPass(false);
        }
      }
      h.end();
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  // ---- world-space vertices with identity world matrices ----------------------------------------------------------

  int runWorldSpace(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();
    std::vector<Vertex> cube;
    std::vector<uint16_t> cubeIdx;
    box(cube, cubeIdx, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f);
    const UINT n = UINT(cube.size());

    // One dynamic ring holds both props. Every frame it is discarded and both are written in world space, and every
    // 30 frames the two swap slots and draw order, as when a game batches props into a dynamic buffer.
    IDirect3DVertexBuffer9* vb = nullptr;
    HRESULT hr = h.m_device->CreateVertexBuffer(UINT(2 * n * sizeof(Vertex)), D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, kFvf,
                                                D3DPOOL_DEFAULT, &vb, nullptr);
    if (FAILED(hr)) {
      fail("CreateVertexBuffer dynamic", hr);
    }
    Mesh indexOnly = h.makeMesh(cube, cubeIdx, D3DPOOL_MANAGED, kFvf);
    for (int frame = 0; frame < a.frames; ++frame) {
      const int phase = (frame / 30) & 1;
      const Vec3 moving = { -1.5f + std::sin(0.04f * float(frame)), 0.5f, 4.f };
      const Vec3 still = { 2.f, 0.5f, 4.f };
      Vertex* p = nullptr;
      if (FAILED(hr = vb->Lock(0, 0, reinterpret_cast<void**>(&p), D3DLOCK_DISCARD))) {
        fail("Lock dynamic", hr);
      }
      for (UINT i = 0; i < n; ++i) {
        Vertex m = cube[i];
        Vertex s = cube[i];
        m.x += moving.x; m.y += moving.y; m.z += moving.z;
        s.x += still.x; s.y += still.y; s.z += still.z;
        p[phase * n + i] = m;
        p[(1 - phase) * n + i] = s;
      }
      vb->Unlock();

      h.begin();
      h.setView(lookAt({ 0.f, 2.5f, -1.f }, { 0.f, 0.5f, 4.f }));
      h.draw(ground, identity(), h.texture(Tex::Ground));
      const D3DMATRIX world = identity();
      h.m_device->SetTransform(D3DTS_WORLD, &world);
      h.m_device->SetTexture(0, h.texture(Tex::Crate));
      h.m_device->SetStreamSource(0, vb, 0, sizeof(Vertex));
      h.m_device->SetIndices(indexOnly.ib);
      h.m_device->SetFVF(kFvf);
      const int slots[2] = { phase, 1 - phase };
      for (const int slot : slots) {
        h.m_device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, INT(slot * n), 0, n, 0, UINT(cubeIdx.size() / 3));
      }
      h.end();
    }
    h.m_device->SetStreamSource(0, nullptr, 0, 0);
    release(vb);
    indexOnly.destroy();
    ground.destroy();
    return 0;
  }

  // ---- fixed-function skinned characters --------------------------------------------------------------------------

  int runSkinned(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();

    // A three-segment column. Segment k blends bones k, k + 1 and k + 2 (clamped to 2) with weights 0.7, 0.2, 0.1.
    std::vector<Vertex> plain;
    std::vector<uint16_t> idx;
    std::vector<SkinVertex> v;
    for (int k = 0; k < 3; ++k) {
      const size_t first = plain.size();
      box(plain, idx, 0.f, 0.35f + 0.7f * float(k), 0.f, 0.5f, 0.7f, 0.5f);
      for (size_t i = first; i < plain.size(); ++i) {
        const Vertex& p = plain[i];
        SkinVertex s {};
        s.x = p.x; s.y = p.y; s.z = p.z;
        s.w0 = 0.7f; s.w1 = 0.2f;
        s.bones[0] = uint8_t(k);
        s.bones[1] = uint8_t(std::min(k + 1, 2));
        s.bones[2] = uint8_t(std::min(k + 2, 2));
        s.bones[3] = 0;
        s.nx = p.nx; s.ny = p.ny; s.nz = p.nz;
        s.u = p.u; s.v = p.v;
        v.push_back(s);
      }
    }
    Mesh character = h.makeMesh(v, idx, D3DPOOL_MANAGED, 0);
    const D3DVERTEXELEMENT9 elements[] = {
      { 0, 0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
      { 0, 12, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0 },
      { 0, 20, D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0 },
      { 0, 24, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0 },
      { 0, 36, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
      D3DDECL_END()
    };
    const HRESULT hr = h.m_device->CreateVertexDeclaration(elements, &character.decl);
    if (FAILED(hr)) {
      fail("CreateVertexDeclaration", hr);
    }

    // Characters 0-2 spawn at one root and walk apart over the first 120 frames. Character 3 turns into a ragdoll at
    // the event frame: three bones per vertex instead of two, and other bone matrices.
    const Vec3 spawn = { 0.f, 0.f, 5.f };
    const Vec3 targets[3] = { { -2.f, 0.f, 5.f }, { 0.f, 0.f, 7.f }, { 2.f, 0.f, 5.f } };
    bool announced = false;
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      h.setView(lookAt({ 0.f, 2.5f, -2.f }, { 0.f, 1.f, 5.f }));
      h.draw(ground, identity(), h.texture(Tex::Ground));
      const float walk = std::min(1.f, float(frame) / 120.f);
      for (int k = 0; k < 4; ++k) {
        const int c = (k + frame) % 4;
        const bool ragdoll = c == 3 && frame >= a.eventFrame;
        if (ragdoll && !announced) {
          std::printf("event: character 3 switches to 3 bones per vertex at frame %d\n", frame);
          announced = true;
        }
        Vec3 root = { -3.5f, 0.f, 6.f };
        if (c < 3) {
          root = { spawn.x + walk * (targets[c].x - spawn.x), 0.f, spawn.z + walk * (targets[c].z - spawn.z) };
        }
        h.m_device->SetRenderState(D3DRS_INDEXEDVERTEXBLENDENABLE, TRUE);
        h.m_device->SetRenderState(D3DRS_VERTEXBLEND, ragdoll ? D3DVBF_2WEIGHTS : D3DVBF_1WEIGHTS);
        for (int b = 0; b < 3; ++b) {
          const float bend = ragdoll ? -0.4f * float(b) - 0.3f : 0.15f * float(b) * std::sin(0.05f * float(frame) + 1.3f * float(c));
          const D3DMATRIX bone = mul(rotationZ(bend), translation(root.x, root.y, root.z));
          h.m_device->SetTransform(D3DTS_WORLDMATRIX(b), &bone);
        }
        h.m_device->SetTexture(0, h.texture(Tex::Crate));
        h.m_device->SetStreamSource(0, character.vb, 0, character.stride);
        h.m_device->SetIndices(character.ib);
        h.m_device->SetVertexDeclaration(character.decl);
        h.m_device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, character.vertices, 0, character.primitives);
      }
      h.m_device->SetRenderState(D3DRS_VERTEXBLEND, D3DVBF_DISABLE);
      h.m_device->SetRenderState(D3DRS_INDEXEDVERTEXBLENDENABLE, FALSE);
      h.end();
    }
    character.destroy();
    ground.destroy();
    return 0;
  }

  // ---- crate grid: device Reset and an unpresented stretch --------------------------------------------------------

  void drawGrid(Host& h, const Mesh& ground, const Mesh& crate) {
    h.setView(lookAt({ 0.f, 4.f, -2.f }, { 0.f, 0.f, 8.f }));
    h.draw(ground, identity(), h.texture(Tex::Ground));
    for (int i = 0; i < 16; ++i) {
      h.draw(crate, translation(-3.f + 2.f * float(i % 4), 0.5f, 5.f + 2.f * float(i / 4)), h.texture(Tex::Crate));
    }
  }

  int runReset(Host& h, const D3D9ScenarioArgs& a) {
    // Default-pool buffers, which a Reset requires the game to release and create again.
    Mesh ground = h.ground(D3DPOOL_DEFAULT);
    Mesh crate = h.crate(D3DPOOL_DEFAULT);
    for (int frame = 0; frame < a.frames; ++frame) {
      if (frame == a.eventFrame) {
        h.m_device->SetStreamSource(0, nullptr, 0, 0);
        h.m_device->SetIndices(nullptr);
        ground.destroy();
        crate.destroy();
        D3DPRESENT_PARAMETERS pp = h.m_presentParams;
        const HRESULT hr = h.m_device->Reset(&pp);
        if (FAILED(hr)) {
          fail("Reset", hr);
        }
        ground = h.ground(D3DPOOL_DEFAULT);
        crate = h.crate(D3DPOOL_DEFAULT);
        h.setStates();
        std::printf("event: Reset at frame %d, buffers created again\n", frame);
      }
      h.begin();
      drawGrid(h, ground, crate);
      h.end();
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  int runNoPresent(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    const int last = a.eventFrame + a.hiddenFrames;
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    for (int frame = 0; frame < a.frames; ++frame) {
      // The window messages a fullscreen game gets on alt-tab, sent through the window procedure the runtime hooks.
      if (frame == a.eventFrame) {
        SendMessageW(h.m_hwnd, WM_ACTIVATE, MAKEWPARAM(WA_INACTIVE, 0), 0);
        std::printf("event: WM_ACTIVATE WA_INACTIVE at frame %d, then %d frames without Present\n", frame, a.hiddenFrames);
      } else if (frame == last) {
        SendMessageW(h.m_hwnd, WM_ACTIVATE, MAKEWPARAM(WA_ACTIVE, 0), 0);
      }
      const bool present = frame < a.eventFrame || frame >= last;
      QueryPerformanceCounter(&t0);
      h.begin();
      drawGrid(h, ground, crate);
      h.end(present);
      QueryPerformanceCounter(&t1);
      if (frame == last || frame == a.eventFrame - 1) {
        std::printf("no-present: frame %d (%s) took %.2f ms on the host\n", frame,
                    frame == last ? "first present after the stretch" : "last present before it",
                    double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(freq.QuadPart));
      }
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  // ---- a draw that becomes sky -----------------------------------------------------------------------------------

  int runSky(Host& h, const D3D9ScenarioArgs& a) {
    // rtx.skyDrawcallIdThreshold = 2 (set by the script) makes untextured draws with draw-call ID 0 or 1 sky. The
    // untextured crate is the third draw until the event frame and the first one after it, with the same inputs.
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    for (int frame = 0; frame < a.frames; ++frame) {
      if (frame == a.eventFrame) {
        std::printf("event: the untextured crate is drawn first from frame %d\n", frame);
      }
      h.begin();
      h.setView(lookAt({ 0.f, 2.5f, -1.f }, { 0.f, 0.5f, 4.f }));
      const bool skyFirst = frame >= a.eventFrame;
      if (skyFirst) {
        h.draw(crate, translation(1.5f, 0.5f, 4.f), nullptr);
      }
      h.draw(ground, identity(), h.texture(Tex::Ground));
      h.draw(crate, translation(-1.5f, 0.5f, 4.f), h.texture(Tex::Crate));
      if (!skyFirst) {
        h.draw(crate, translation(1.5f, 0.5f, 4.f), nullptr);
      }
      h.end();
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  // ---- tagged special classes ------------------------------------------------------------------------------------

  // A ground and a crate, with the textures given, for the terrain and emitter scenarios.
  int runTagged(Host& h, const D3D9ScenarioArgs& a, Tex groundTex, Tex crateTex) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      h.setView(lookAt({ 0.f, 2.5f, -1.f }, { 0.f, 0.5f, 4.f }));
      h.draw(ground, identity(), h.texture(groundTex));
      h.draw(crate, translation(0.f, 0.5f, 4.f), h.texture(crateTex));
      h.end();
    }
    ground.destroy();
    crate.destroy();
    return 0;
  }

  int runPortals(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    Mesh quad = h.quad();
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      h.setView(lookAt({ 0.f, 2.f, -1.f }, { 0.f, 1.f, 6.f }));
      h.draw(ground, identity(), h.texture(Tex::Ground));
      h.draw(quad, translation(-2.f, 1.2f, 6.f), h.texture(Tex::PortalA));
      h.draw(quad, translation(2.f, 1.2f, 6.f), h.texture(Tex::PortalB));
      h.draw(crate, translation(0.f, 0.5f, 5.f), h.texture(Tex::Crate));
      h.end();
    }
    quad.destroy();
    crate.destroy();
    ground.destroy();
    return 0;
  }

  int runRenderTarget(Host& h, const D3D9ScenarioArgs& a) {
    Mesh ground = h.ground();
    Mesh crate = h.crate();
    Mesh quad = h.quad();
    IDirect3DTexture9* monitor = nullptr;
    HRESULT hr = h.m_device->CreateTexture(kMonitorSize, kMonitorSize, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8,
                                           D3DPOOL_DEFAULT, &monitor, nullptr);
    if (FAILED(hr)) {
      fail("CreateTexture render target", hr);
    }
    IDirect3DSurface9* monitorSurface = nullptr;
    IDirect3DSurface9* backBuffer = nullptr;
    monitor->GetSurfaceLevel(0, &monitorSurface);
    h.m_device->GetRenderTarget(0, &backBuffer);
    for (int frame = 0; frame < a.frames; ++frame) {
      h.begin();
      // The monitor's view: the crate from the side, rendered into the render-target texture.
      h.m_device->SetRenderTarget(0, monitorSurface);
      h.m_device->Clear(0, nullptr, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(20, 20, 40), 1.f, 0);
      h.setView(lookAt({ 4.f, 2.f, 4.f }, { 0.f, 0.5f, 4.f }));
      h.draw(crate, translation(0.f, 0.5f, 4.f), h.texture(Tex::Crate));
      h.m_device->SetRenderTarget(0, backBuffer);
      h.m_device->Clear(0, nullptr, D3DCLEAR_ZBUFFER, 0, 1.f, 0);
      h.setView(lookAt({ 0.f, 2.f, -2.f }, { 0.f, 1.f, 4.f }));
      h.draw(ground, identity(), h.texture(Tex::Ground));
      h.draw(crate, translation(0.f, 0.5f, 4.f), h.texture(Tex::Crate));
      h.draw(quad, translation(-2.5f, 1.5f, 5.f), monitor);
      // Unbound before the next frame renders into it.
      h.m_device->SetTexture(0, nullptr);
      h.end();
    }
    release(backBuffer);
    release(monitorSurface);
    release(monitor);
    quad.destroy();
    crate.destroy();
    ground.destroy();
    return 0;
  }

  // ---- vertex texture fetch --------------------------------------------------------------------------------------

  constexpr const char kVtfVertexShader[] =
    "float4x4 wvp : register(c0);\n"
    "sampler2D heightTex : register(s0);\n"
    "struct VIn { float3 pos : POSITION; float3 nrm : NORMAL; float2 uv : TEXCOORD0; };\n"
    "struct VOut { float4 pos : POSITION; float2 uv : TEXCOORD0; };\n"
    "VOut main(VIn i) {\n"
    "  VOut o;\n"
    "  float h = tex2Dlod(heightTex, float4(0.5, 0.5, 0, 0)).r;\n"
    "  o.pos = mul(float4(i.pos + i.nrm * (0.5 * h), 1), wvp);\n"
    "  o.uv = i.uv;\n"
    "  return o;\n"
    "}\n";

  constexpr const char kVtfPixelShader[] =
    "sampler2D albedo : register(s0);\n"
    "float4 main(float2 uv : TEXCOORD0) : COLOR { return tex2D(albedo, uv); }\n";

  // Returns the bytecode, or an empty vector when d3dcompiler_47.dll is missing or compilation fails.
  std::vector<DWORD> compileShader(pD3DCompile compile, const char* source, const char* target) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(source, std::strlen(source), "vtf", nullptr, nullptr, "main", target, 0, 0, &code, &errors);
    std::vector<DWORD> out;
    if (SUCCEEDED(hr) && code != nullptr) {
      out.resize(code->GetBufferSize() / sizeof(DWORD));
      std::memcpy(out.data(), code->GetBufferPointer(), out.size() * sizeof(DWORD));
    } else if (errors != nullptr) {
      std::printf("shader %s: %s\n", target, static_cast<const char*>(errors->GetBufferPointer()));
    }
    release(code);
    release(errors);
    return out;
  }

  int runVtf(Host& h, const D3D9ScenarioArgs& a) {
    HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    auto compile = compiler ? reinterpret_cast<pD3DCompile>(GetProcAddress(compiler, "D3DCompile")) : nullptr;
    if (compile == nullptr) {
      std::printf("SKIP: vtf needs d3dcompiler_47.dll (part of Windows 10 and later)\n");
      return 3;
    }
    const std::vector<DWORD> vsCode = compileShader(compile, kVtfVertexShader, "vs_3_0");
    const std::vector<DWORD> psCode = compileShader(compile, kVtfPixelShader, "ps_3_0");
    if (vsCode.empty() || psCode.empty()) {
      fail("compiling the vtf shaders", E_FAIL);
    }
    IDirect3DVertexShader9* vs = nullptr;
    IDirect3DPixelShader9* ps = nullptr;
    HRESULT hr = h.m_device->CreateVertexShader(vsCode.data(), &vs);
    if (FAILED(hr)) {
      fail("CreateVertexShader", hr);
    }
    if (FAILED(hr = h.m_device->CreatePixelShader(psCode.data(), &ps))) {
      fail("CreatePixelShader", hr);
    }
    // The height texture is a render target the GPU fills every frame, so its pointer and hash never change.
    IDirect3DTexture9* height = nullptr;
    if (FAILED(hr = h.m_device->CreateTexture(4, 4, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &height, nullptr))) {
      fail("CreateTexture height", hr);
    }
    IDirect3DSurface9* heightSurface = nullptr;
    height->GetSurfaceLevel(0, &heightSurface);
    Mesh crate = h.crate();
    const D3DMATRIX world = translation(0.f, 0.5f, 4.f);
    const D3DMATRIX view = lookAt({ 0.f, 2.5f, -1.f }, { 0.f, 0.5f, 4.f });
    const D3DMATRIX wvp = transpose(mul(mul(world, view), h.m_projection));
    for (int frame = 0; frame < a.frames; ++frame) {
      const int level = int(127.5f + 127.5f * std::sin(0.05f * float(frame)));
      h.m_device->ColorFill(heightSurface, nullptr, D3DCOLOR_ARGB(255, level, 0, 0));
      h.begin();
      // Only the displaced crate, so the [D3D9v2] early counters are its own.
      h.setView(view);
      h.m_device->SetTransform(D3DTS_WORLD, &world);
      h.m_device->SetVertexShader(vs);
      h.m_device->SetPixelShader(ps);
      h.m_device->SetVertexShaderConstantF(0, &wvp.m[0][0], 4);
      h.m_device->SetTexture(D3DVERTEXTEXTURESAMPLER0, height);
      h.m_device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
      h.m_device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
      h.m_device->SetTexture(0, h.texture(Tex::Crate));
      h.m_device->SetStreamSource(0, crate.vb, 0, crate.stride);
      h.m_device->SetIndices(crate.ib);
      h.m_device->SetFVF(kFvf);
      h.m_device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, crate.vertices, 0, crate.primitives);
      h.m_device->SetVertexShader(nullptr);
      h.m_device->SetPixelShader(nullptr);
      h.m_device->SetTexture(D3DVERTEXTEXTURESAMPLER0, nullptr);
      h.end();
    }
    crate.destroy();
    release(heightSurface);
    release(height);
    release(ps);
    release(vs);
    return 0;
  }
}

void printD3D9Hashes() {
  for (uint32_t k = 0; k < uint32_t(Tex::Count); ++k) {
    std::printf("hash %s 0x%016llX\n", kTexNames[k], static_cast<unsigned long long>(textureHash(Tex(k))));
  }
  std::printf("hash monitor 0x%016llX\n", static_cast<unsigned long long>(renderTargetHash(kMonitorSize, kMonitorSize)));
}

int runD3D9Scenario(const std::wstring& name, const fs::path& runtime, HWND hwnd, uint32_t width, uint32_t height,
                    const D3D9ScenarioArgs& args) {
  Host host;
  host.create(runtime, hwnd, width, height);
  int rc = 0;
  if (name == L"repeat") {
    rc = runRing(host, args, Fuse::Separate);
  } else if (name == L"fused-world") {
    rc = runRing(host, args, Fuse::FusedWorld);
  } else if (name == L"fused-view") {
    rc = runRing(host, args, Fuse::FusedView);
  } else if (name == L"identity-view") {
    rc = runIdentityView(host, args);
  } else if (name == L"twice") {
    rc = runTwice(host, args);
  } else if (name == L"worldspace") {
    rc = runWorldSpace(host, args);
  } else if (name == L"skinned") {
    rc = runSkinned(host, args);
  } else if (name == L"reset") {
    rc = runReset(host, args);
  } else if (name == L"no-present") {
    rc = runNoPresent(host, args);
  } else if (name == L"sky") {
    rc = runSky(host, args);
  } else if (name == L"terrain") {
    rc = runTagged(host, args, Tex::Terrain, Tex::Crate);
  } else if (name == L"emitter") {
    rc = runTagged(host, args, Tex::Ground, Tex::Emitter);
  } else if (name == L"portals") {
    rc = runPortals(host, args);
  } else if (name == L"rendertarget") {
    rc = runRenderTarget(host, args);
  } else if (name == L"vtf") {
    rc = runVtf(host, args);
  } else {
    std::printf("unknown d3d9 scenario %ls\n", name.c_str());
    return 2;
  }
  if (rc == 0) {
    std::printf("testhost: %d frames done (d3d9 scenario %ls)\n", args.frames, name.c_str());
  }
  return rc;
}
