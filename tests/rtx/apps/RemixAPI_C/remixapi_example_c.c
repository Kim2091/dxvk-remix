#include <remix/remix_c.h>

#include <assert.h>
#include <stdio.h>

remixapi_Interface g_remix = { 0 };
HMODULE g_remix_dll = NULL;

remixapi_LightHandle g_scene_light = NULL;
remixapi_MeshHandle g_scene_mesh = NULL;
remixapi_MeshHandle g_ground_mesh = NULL;
remixapi_MeshHandle g_wall_mesh = NULL;
remixapi_MeshHandle g_emitter_mesh = NULL;
remixapi_MeshHandle g_pillar_mesh = NULL;

remixapi_HardcodedVertex makeVertex(float x, float y, float z) {
  remixapi_HardcodedVertex v = {
    .position = {x,y,z},
    .normal = {0,0,-1},
    .texcoord = {0,0},
    .color = 0xFFFFFFFF,
  };
  return v;
}

remixapi_HardcodedVertex makeVertexFull(float x, float y, float z,
                                        float nx, float ny, float nz,
                                        uint32_t color) {
  remixapi_HardcodedVertex v = {
    .position = {x,y,z},
    .normal = {nx,ny,nz},
    .texcoord = {0,0},
    .color = color,
  };
  return v;
}

remixapi_ErrorCode createQuadMesh(remixapi_HardcodedVertex v0,
                                  remixapi_HardcodedVertex v1,
                                  remixapi_HardcodedVertex v2,
                                  remixapi_HardcodedVertex v3,
                                  uint64_t hash,
                                  remixapi_MeshHandle* outMesh) {
  remixapi_HardcodedVertex verts[] = { v0, v1, v2, v3 };
  uint32_t indices[] = { 0, 1, 2, 2, 3, 0 };

  remixapi_MeshInfoSurfaceTriangles triangles = {
    .vertices_values = verts,
    .vertices_count = 4,
    .indices_values = indices,
    .indices_count = 6,
    .skinning_hasvalue = FALSE,
    .skinning_value = { 0 },
    .material = NULL,
  };

  remixapi_MeshInfo meshInfo = {
    .sType = REMIXAPI_STRUCT_TYPE_MESH_INFO,
    .pNext = NULL,
    .hash = hash,
    .surfaces_values = &triangles,
    .surfaces_count = 1,
  };

  return g_remix.CreateMesh(&meshInfo, outMesh);
}

remixapi_ErrorCode createPillarMesh(remixapi_MeshHandle* outMesh) {
  // A narrow box at x=3, z=8. 4 visible faces (front, back, left, right).
  // Dimensions: 0.5 wide (x), 10 tall (y=-5 to y=5), 0.5 deep (z)
  float x0 = 2.75f, x1 = 3.25f;
  float y0 = -5.0f, y1 = 5.0f;
  float z0 = 7.75f, z1 = 8.25f;
  uint32_t col = 0xFF606060;

  remixapi_HardcodedVertex verts[] = {
    // Front face (z=z0, normal 0,0,-1)
    makeVertexFull(x0, y0, z0, 0,0,-1, col),
    makeVertexFull(x0, y1, z0, 0,0,-1, col),
    makeVertexFull(x1, y1, z0, 0,0,-1, col),
    makeVertexFull(x1, y0, z0, 0,0,-1, col),
    // Back face (z=z1, normal 0,0,1)
    makeVertexFull(x1, y0, z1, 0,0,1, col),
    makeVertexFull(x1, y1, z1, 0,0,1, col),
    makeVertexFull(x0, y1, z1, 0,0,1, col),
    makeVertexFull(x0, y0, z1, 0,0,1, col),
    // Left face (x=x0, normal -1,0,0)
    makeVertexFull(x0, y0, z1, -1,0,0, col),
    makeVertexFull(x0, y1, z1, -1,0,0, col),
    makeVertexFull(x0, y1, z0, -1,0,0, col),
    makeVertexFull(x0, y0, z0, -1,0,0, col),
    // Right face (x=x1, normal 1,0,0)
    makeVertexFull(x1, y0, z0, 1,0,0, col),
    makeVertexFull(x1, y1, z0, 1,0,0, col),
    makeVertexFull(x1, y1, z1, 1,0,0, col),
    makeVertexFull(x1, y0, z1, 1,0,0, col),
    // Top face (y=y1, normal 0,1,0)
    makeVertexFull(x0, y1, z0, 0,1,0, col),
    makeVertexFull(x0, y1, z1, 0,1,0, col),
    makeVertexFull(x1, y1, z1, 0,1,0, col),
    makeVertexFull(x1, y1, z0, 0,1,0, col),
  };
  uint32_t indices[] = {
    0,1,2, 2,3,0,       // front
    4,5,6, 6,7,4,       // back
    8,9,10, 10,11,8,    // left
    12,13,14, 14,15,12, // right
    16,17,18, 18,19,16, // top
  };

  remixapi_MeshInfoSurfaceTriangles triangles = {
    .vertices_values = verts,
    .vertices_count = 20,
    .indices_values = indices,
    .indices_count = 30,
    .skinning_hasvalue = FALSE,
    .skinning_value = { 0 },
    .material = NULL,
  };

  remixapi_MeshInfo meshInfo = {
    .sType = REMIXAPI_STRUCT_TYPE_MESH_INFO,
    .pNext = NULL,
    .hash = 0x5,
    .surfaces_values = &triangles,
    .surfaces_count = 1,
  };

  return g_remix.CreateMesh(&meshInfo, outMesh);
}

remixapi_ErrorCode init(HWND hwnd) {
  const wchar_t* path = L"d3d9.dll";
  if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
    path = L"bin\\d3d9.dll";
    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
      printf("d3d9.dll not found.\nPlease, place it in the same folder as this .exe");
    }
  }

  {
    remixapi_ErrorCode status = remixapi_lib_loadRemixDllAndInitialize(path, &g_remix, &g_remix_dll);
    if (status != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remixapi_lib_loadRemixDllAndInitialize failed: %d", status);
      return status;
    }
  }

  {
    remixapi_StartupInfo startInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_STARTUP_INFO,
      .pNext = NULL,
      .hwnd = hwnd,
      .disableSrgbConversionForOutput = FALSE,
      .forceNoVkSwapchain = FALSE,
    };
    remixapi_ErrorCode r = g_remix.Startup(&startInfo);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::Startup() failed: %d", r);
      return r;
    }
  }

  // Dim ambient light so fire is the primary illumination
  {
    remixapi_LightInfoSphereEXT sphereLight = {
      .sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT,
      .pNext = NULL,
      .position = {0, 5, 8},
      .radius = 0.1f,
      .shaping_hasvalue = FALSE,
      .shaping_value = { 0 },
    };
    remixapi_LightInfo lightInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_LIGHT_INFO,
      .pNext = &sphereLight,
      .hash = 0x3,
      .radiance = { 15, 15, 20 },
    };

    remixapi_ErrorCode r = g_remix.CreateLight(&lightInfo, &g_scene_light);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateLight() failed: %d", r);
      return r;
    }
  }

  // Original triangle mesh (kept for GPU instancing demo)
  {
    remixapi_HardcodedVertex verts[] = {
      makeVertex( 5, -5, 10),
      makeVertex( 0, 5, 10),
      makeVertex(-5, -5, 10),
    };

    remixapi_MeshInfoSurfaceTriangles triangles = {
      .vertices_values = verts,
      .vertices_count = ARRAYSIZE(verts),
      .indices_values = NULL,
      .indices_count = 0,
      .skinning_hasvalue = FALSE,
      .skinning_value = { 0 },
      .material = NULL,
    };

    remixapi_MeshInfo meshInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_MESH_INFO,
      .pNext = NULL,
      .hash = 0x1,
      .surfaces_values = &triangles,
      .surfaces_count = 1,
    };

    remixapi_ErrorCode r = g_remix.CreateMesh(&meshInfo, &g_scene_mesh);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateMesh() failed: %d", r);
      return r;
    }
  }

  // Ground plane: large flat quad at y=-5
  {
    remixapi_ErrorCode r = createQuadMesh(
      makeVertexFull(-20, -5, -5,  0,1,0, 0xFF808080),
      makeVertexFull(-20, -5, 25,  0,1,0, 0xFF808080),
      makeVertexFull( 20, -5, 25,  0,1,0, 0xFF808080),
      makeVertexFull( 20, -5, -5,  0,1,0, 0xFF808080),
      0x2, &g_ground_mesh);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateMesh(ground) failed: %d", r);
      return r;
    }
  }

  // Back wall: vertical quad at z=15
  {
    remixapi_ErrorCode r = createQuadMesh(
      makeVertexFull(-20, -5, 15,  0,0,-1, 0xFFA0A0A0),
      makeVertexFull(-20, 10, 15,  0,0,-1, 0xFFA0A0A0),
      makeVertexFull( 20, 10, 15,  0,0,-1, 0xFFA0A0A0),
      makeVertexFull( 20, -5, 15,  0,0,-1, 0xFFA0A0A0),
      0x3, &g_wall_mesh);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateMesh(wall) failed: %d", r);
      return r;
    }
  }

  // Emitter mesh: small flat quad slightly above ground at z=8
  {
    remixapi_ErrorCode r = createQuadMesh(
      makeVertexFull(-1, -4.5f, 7,  0,1,0, 0xFFFFFFFF),
      makeVertexFull(-1, -4.5f, 9,  0,1,0, 0xFFFFFFFF),
      makeVertexFull( 1, -4.5f, 9,  0,1,0, 0xFFFFFFFF),
      makeVertexFull( 1, -4.5f, 7,  0,1,0, 0xFFFFFFFF),
      0x4, &g_emitter_mesh);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateMesh(emitter) failed: %d", r);
      return r;
    }
  }

  // Pillar: narrow box at x=3, z=8
  {
    remixapi_ErrorCode r = createPillarMesh(&g_pillar_mesh);
    if (r != REMIXAPI_ERROR_CODE_SUCCESS) {
      printf("remix::CreateMesh(pillar) failed: %d", r);
      return r;
    }
  }

  return REMIXAPI_ERROR_CODE_SUCCESS;
}

void render(uint32_t windowWidth, uint32_t windowHeight) {
  // Camera: positioned to view the campfire scene
  {
    remixapi_CameraInfoParameterizedEXT parametersForCamera = {
      .sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT,
      .position = { 0, 2, 0 },
      .forward = { 0, -0.15f, 1 },
      .up = { 0, 1, 0 },
      .right = { 1, 0, 0 },
      .fovYInDegrees = 70,
      .aspect = (float)windowWidth / (float)windowHeight,
      .nearPlane = 0.1f,
      .farPlane = 1000.0f,
    };
    remixapi_CameraInfo cameraInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_CAMERA_INFO,
      .pNext = &parametersForCamera,
    };
    g_remix.SetupCamera(&cameraInfo);
  }

  // Draw static scene geometry
  {
    remixapi_Transform identity = { {
      {1,0,0,0},
      {0,1,0,0},
      {0,0,1,0},
    } };

    // Ground plane
    remixapi_InstanceInfo groundInst = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO,
      .categoryFlags = 0,
      .mesh = g_ground_mesh,
      .transform = identity,
      .doubleSided = 1,
    };
    g_remix.DrawInstance(&groundInst);

    // Back wall
    remixapi_InstanceInfo wallInst = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO,
      .categoryFlags = 0,
      .mesh = g_wall_mesh,
      .transform = identity,
      .doubleSided = 1,
    };
    g_remix.DrawInstance(&wallInst);

    // Pillar
    remixapi_InstanceInfo pillarInst = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO,
      .categoryFlags = 0,
      .mesh = g_pillar_mesh,
      .transform = identity,
      .doubleSided = 1,
    };
    g_remix.DrawInstance(&pillarInst);
  }

  // Draw emitter mesh with volumetric particle system (campfire)
  {
    remixapi_Float4D particleMinColor[] = { { 1.f, 0.4f, 0.1f, 1.f } };
    remixapi_Float4D particleMaxColor[] = { { 1.f, 0.8f, 0.3f, 1.f } };
    remixapi_Float2D particleMinSize[] = { { 0.5f, 0.5f } };
    remixapi_Float2D particleMaxSize[] = { { 1.5f, 1.5f } };
    remixapi_Float3D particleMaxVelocity[] = { { 2.f, 5.f, 2.f } };

    remixapi_InstanceInfoParticleSystemEXT particleInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_PARTICLE_SYSTEM_EXT,
      .maxNumParticles = 5000,
      .hideEmitter = 1,
      .minColor = { particleMinColor, 1 },
      .maxColor = { particleMaxColor, 1 },
      .minSize = { particleMinSize, 1 },
      .maxSize = { particleMaxSize, 1 },
      .maxVelocity = { particleMaxVelocity, 1 },
      .minTimeToLive = 2.f,
      .maxTimeToLive = 4.f,
      .initialVelocityFromNormal = 20.f,
      .initialVelocityConeAngleDegrees = 30.f,
      .dragCoefficient = 0.1f,
      .gravityForce = -5.f,
      .spawnRatePerSecond = 200.f,
      // Volumetric smoke/fire parameters (campfire preset)
      .smokeDensity = 1.5f,
      .smokeAbsorptionCrossSection = 0.6f,
      .smokeDissipationRate = 0.08f,
      .fuelAmount = 0.8f,
      .burnTemperature = 1200.f,
      .coolingRate = 0.25f,
      .buoyancyCoefficient = 0.6f,
      .windDirection = {0.f, 0.f, 0.f},
      .vorticityConfinement = 0.3f,
      .fluidCouplingStrength = 0.9f,
      .emissionIntensityScale = 1.f,
      .volumePadding = 0.2f,
      .volumeDecayTime = 3.f,
      .pressureIterations = 30,
      .lightClusterCount = 2,
      .volumeType = 1, // Volumetric
    };

    remixapi_InstanceInfo emitterInst = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO,
      .pNext = &particleInfo,
      .categoryFlags = 0,
      .mesh = g_emitter_mesh,
      .transform = { {
        {1,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
      } },
      .doubleSided = 1,
    };
    g_remix.DrawInstance(&emitterInst);
  }

  // GPU Instancing example: moved further back (z=20+) to not interfere with campfire
  {
    remixapi_Transform gpuInstanceTransforms[5] = {
      { {{ 1,0,0,-6 }, { 0,1,0,0 }, { 0,0,1,20.0f }} },
      { {{ 1,0,0,-3 }, { 0,1,0,0 }, { 0,0,1,19.3f }} },
      { {{ 1,0,0, 0 }, { 0,1,0,2 }, { 0,0,1,20.6f }} },
      { {{ 1,0,0, 3 }, { 0,1,0,0 }, { 0,0,1,22.0f }} },
      { {{ 1,0,0, 6 }, { 0,1,0,0 }, { 0,0,1,20.3f }} },
    };
    remixapi_InstanceInfoGpuInstancingEXT gpuInstancingInfo = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT,
      .pNext = NULL,
      .instanceTransforms_values = gpuInstanceTransforms,
      .instanceTransforms_count = 5,
    };
    remixapi_InstanceInfo gpuInstancedMesh = {
      .sType = REMIXAPI_STRUCT_TYPE_INSTANCE_INFO,
      .pNext = &gpuInstancingInfo,
      .categoryFlags = 0,
      .mesh = g_scene_mesh,
      .transform = { {
        {1,0,0,0},
        {0,1,0,0},
        {0,0,1,0},
      } },
      .doubleSided = 1,
    };
    g_remix.DrawInstance(&gpuInstancedMesh);
  }

  {
    g_remix.DrawLightInstance(g_scene_light);
  }
  g_remix.Present(NULL);
}

void destroy(void) {
  if (g_remix.Shutdown) {
    remixapi_lib_shutdownAndUnloadRemixDll(&g_remix, g_remix_dll);
  }
}



#pragma region HWND boilerplate

LRESULT WINAPI MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
  case WM_DESTROY:
    PostQuitMessage(0);
    return 0;
  default:
    break;
  }
  return DefWindowProc(hwnd, msg, wParam, lParam);
}

int main(int argc, char* argv[]) {
  // Parse command-line arguments for the number of frames
  int numFrames = 0;
  if (argc >= 2) {
    numFrames = atoi(argv[1]);
  }

  WNDCLASSEX wc = {
    .cbSize = sizeof(WNDCLASSEX),
    .style = CS_CLASSDC,
    .lpfnWndProc = MsgProc,
    .cbClsExtra = 0L,
    .cbWndExtra = 0L,
    .hInstance = GetModuleHandle(NULL),
    .hIcon = NULL,
    .hCursor = NULL,
    .hbrBackground = NULL,
    .lpszMenuName = NULL,
    .lpszClassName = "Remix API Example",
    .hIconSm = NULL,
  };
  RegisterClassEx(&wc);

  DWORD dwStyle = WS_OVERLAPPEDWINDOW;
  // readjust, so the client area as specified, not the window size
  RECT clientRect = { 0, 0, 1600, 900 };
  AdjustWindowRect(&clientRect, dwStyle, FALSE);

  HWND hwnd = CreateWindow(wc.lpszClassName, "Remix API Example",
                            dwStyle,
                            CW_USEDEFAULT, CW_USEDEFAULT,
                            clientRect.right - clientRect.left,
                            clientRect.bottom - clientRect.top,
                            GetDesktopWindow(), NULL, wc.hInstance, NULL);

  int frameIdx = 0;

  if (init(hwnd) == REMIXAPI_ERROR_CODE_SUCCESS) {
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    MSG msg = { 0 };
    while (msg.message != WM_QUIT && (numFrames == 0 || frameIdx < numFrames)) {
      if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
      } else {
        RECT hwndRect = { 0 };
        GetClientRect(hwnd, &hwndRect);
        LONG w = hwndRect.right - hwndRect.left;
        LONG h = hwndRect.bottom - hwndRect.top;

        render(w > 0 ? (uint32_t)w : 0, h > 0 ? (uint32_t)h : 0);
        ++frameIdx;
      }
    }
  }

  destroy();

  UnregisterClass(wc.lpszClassName, wc.hInstance);
  return 0;
}

#pragma endregion
