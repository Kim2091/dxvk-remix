# dxvk-remix

[![Build Status](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml/badge.svg)](https://github.com/NVIDIAGameWorks/dxvk-remix/actions/workflows/build.yml)

dxvk-remix is a fork of the [DXVK](https://github.com/doitsujin/dxvk) project, which overhauls the fixed-function graphics pipeline implementation in order to remaster games with path tracing.

Thanks to all the contributors to DXVK for creating this foundational piece of software, on top of which we were able to build the RTX Remix Runtime.

While dxvk-remix is a fork of DXVK, please report bugs encountered with dxvk-remix to this repo rather than to the DXVK project.

dxvk-remix also contains a subproject in the `bridge` folder, which enables 32 bit games to communicate with the 64 bit dxvk-remix runtime.

## NGX passthrough branch (`mirrors-edge-ngx`)

This branch adds an NGX passthrough mode where Mirror's Edge's own rasterised rendering is presented unchanged (no path tracing, no scene capture) while DLSS Super Resolution / DLAA, DLSS Frame Generation, Reflex, and a subset of Remix's PostFX runs on top of it.

### NGX mode setup

**The game's MSAA *must* be off, depth cannot be resolved while it is active** - plus you want DLSS/DLAA anyway, right? Also, set `rtx.ngxPassthroughMode` at launch (rtx.conf / game profile). Depth buffers only get a shader-readable layout when the mode is active at creation. This is already pre-configured for Mirror's Edge in this branch.

How it works:

- The frame is intercepted before the game's postprocess chain. The first fullscreen composite quad that samples the scene color (in Mirror's Edge this is a fullsize FP16 scene colour resolve; the runtime tracks resolve `StretchRect`s) triggers injection. DLSS takes the linear HDR scene colour in NGX HDR mode and writes the antialiased result back; bloom, tonemapping, and dynamic contrast then run on the stabilised, unjittered image (`rtx.ngxPassthrough.prePostProcess`). Scene colour alpha stores depth so the write-back remerges the original alpha with DLSS RGB. The trigger requires a depth-test-disabled fullscreen quad so mid-scene lighting samples of the resolve do not fire early. If no post pass samples scene colour, injection falls back to the late scene-end point (post-processed LDR, before UI).
- Screen-space motion vectors come from depth reprojected through the current and previous cameras (UE3 reserved shader constants, same as path traced mode). A small velocity raster overrides that with true object motion for DLSS, Frame Generation, and Remix's PostFX motion blur (`rtx.ngxPassthrough.objectVelocities`): rigid movers (CTAB `LocalToWorld` changes), GPU-skinned meshes (current/previous bone palettes), and CPU-skinned dynamic buffers (per-frame vertex snapshots), all depth-tested against the game depth. UE3 clears depth mid-scene before the foreground DPG, so the runtime snapshots depth before the first clear and merges it at injection: world / SDPG_Intermediate from the snapshot, SDPG_Foreground from the live buffer. A phase ownership marker on the shared velocity target keeps camera-locked first person motion from leaking across intermediate/foreground movestates (depth alone cannot separate them when both DPGs draw at the same depth). Missed pixels fall back to camera reprojection.
- Sub-pixel Halton jitter is applied as a fractional viewport offset on draws to the scene render target and on fullsize screen-space passes tested against scene depth (shadow projections, distortion), so those buffers shift with the scene. UE3's `ScreenPositionScaleBias` upload is compensated by the same jitter so clip-derived UVs (shadows reading depth from scene-color alpha, translucency, distortion, fog) stay aligned. Game state is untouched; the patch stops at injection, where DLSS output is unjittered.
- Remix menu's DLSS mode selector (Full Resolution, Quality, Balanced, Performance, Ultra Performance, Auto) drives the game's internal render resolution and LOD biasing directly. Full Resolution renders natively as DLAA on the post-processed scene before the UI; the other tiers render at a reduced render resolution and DLSS upscales to the output resolution as Super Resolution.
- Frame Generation uses the synthesised depth/MVs and camera via the standard Remix DLFG presenter and interpolates the post-UI backbuffer like stock Remix. Object velocities cover first-person meshes, so they move with true motion. A HUD-less frame captured at the scene-end/UI boundary is also supplied so UI is separated from the scene without heuristics (`rtx.ngxPassthrough.dlfgHudlessInput`; stock Remix does not provide this).
- Remix post FX (`rtx.postfx.*`, Rendering -> Post FX): motion blur uses the synthesised MVs, linear view-Z, and surface flags (geometry marked static; object velocity carries real motion). First person meshes blur with the scene by default; `rtx.ngxPassthrough.motionBlurFirstPerson = False` excludes them (crisp view model). Chromatic aberration and vignette share that pass too.

Notes and limitations:

- Options under `rtx.ngxPassthrough.*`; Rendering -> General shows DLSS stats and depth/MV visualisers.
- Default DLSS model is the transformer preset (rtx.ngxPassthrough.dlssRenderPreset = 10, preset J), which keeps more detail than NGX's default CNN presets; the developer menu can switch presets live.
- Ghosting issues with particles and with textures on transparent planes (chain link fences). Will investigate solutions for this at a later date.

## Build instructions

### Requirements:
1. Windows 10 or 11
2. [Git](https://git-scm.com/download/win)
3. [Visual Studio ](https://visualstudio.microsoft.com/vs/older-downloads/)
    - VS 2019 is tested
    - VS 2022 may also work, but it is not actively tested
    - Note that our build system will always use the most recent version available on the system
4. [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/)
    - 10.0.19041.0 is tested
5. [Meson](https://mesonbuild.com/)
    - 1.8.2 has been tested
    - Follow [instructions](https://mesonbuild.com/SimpleStart.html#installing-meson) on how to install and reboot the PC before moving on (Meson will indicate as much)
6. [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows)
    - 1.4.313.2 or newer
    - You may need to uninstall previous SDK if you have an old version
7. [Python](https://www.python.org/downloads/)
    - 3.9 or newer
    - Ensure you are using python installed from the link above and not from the Microsoft Store
8. [DirectX Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=35)
    - Latest version should work.
    - This includes d3d9x*.dll which are required to run the game
    - May already be installed if you have D3D9 games installed

#### Additional notes:
- If any dependency paths change (i.e. new Vulkan library), run `meson --reconfigure` in _Compiler64 directory via a command prompt. This may revert some custom VS project settings

### Generate and build dxvk-remix Visual Studio project 
1. Clone the repository with all submodules:
	- `git clone --recursive https://github.com/NVIDIAGameWorks/dxvk-remix.git`

	If the clone was made non-recursively and the submodules are missing, clone them separately:
	- `git submodule update --init --recursive`

2. Install all the [requirements](#requirements) before proceeding further

3. Make sure PowerShell scripts are enabled
    - One-time system setup: run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` in an elevated PowerShell prompt, then close and reopen any existing PowerShell prompts
	
4. To generate and build dxvk-remix project:
    - Right Click on `dxvk-remix\build_dxvk_all_ninja.ps1` and select "Run with Powershell"
    - If that fails or has problems, run the build manually in a way you can read the errors:
        - open a windows file explorer to the `dxvk-remix` folder
        - remove artifacts from the previous attempt by deleting all folders that start with `_`, i.e. `_vs/` and `_Comp64Debug`
        - type `cmd` in the address bar to open a command line window in that folder.
        - copy and paste `powershell -command "& .\build_dxvk_all_ninja.ps1"` into the command line, then press enter
    - This will build all 3 configurations of dxvk-remix project inside subdirectories of the build tree: 
        - **_Comp64Debug** - full debug instrumentation, runtime speed may be slow
        - **_Comp64DebugOptimized** - partial debug instrumentation (i.e. asserts), runtime speed is generally comparable to that of release configuration
        - **_Comp64Release** - fastest runtime 
    - This will generate a project in the **_vs** subdirectory
    - Only x64 build targets are supported

5. Open **_vs/dxvk-remix.sln** in Visual Studio (2019+). 
    - Do not convert the solution on load if prompted when using a newer version of Visual Studio 
    - Once generated, the project can be built via Visual Studio or via powershell scripts
    - A build will copy generated DXVK DLLs to any target project as specified in **gametargets.conf** (see its [setup section](#deploy-built-binaries-to-a-game))

### Deploy built binaries to a game 
1. First time only: copy **gametargets.example.conf** to **gametargets.conf** in the project root

2. Update paths in the **gametargets.conf** for your game. Follow example in the **gametargets.example.conf**. Make sure to remove "#" from the start of all three lines

3. Open and, simply, re-save top-level **meson.build** file (i.e. via notepad) to update its time stamp, and rerun the build. This will trigger a full meson script run which will generate a project within the Visual Studio solution file and deploy built binaries into games' directories specified in **gametargets.conf**

### Profiling Remix
Remix has support for profiling using the [Tracy](https://github.com/wolfpld/tracy) tool, specifically the [v0.8 release](https://github.com/wolfpld/tracy/releases/download/v0.8/Tracy-0.8.7z)

To enable Tracy profiling:
1. Open a command line window in a build folder (i.e. `dxvk-remix/_Comp64Release/`)
2. Run `meson --reconfigure -D enable_tracy=true`
3. Rebuild dxvk-remix-nv

To profile:
1. Launch tracy.exe
2. Launch the game and reach the section you wish to profile
3. When ready, hit `Connect` in Tracy to begin profiling.
4. It's best to collect at least 500 frames worth of data, so you can average out the results.

### Remix API

If there's an intent to use the Remix Renderer in projects with *available* source code, Direct3D 9 API can be utilized, since Remix's `d3d9.dll` implements the Direct3D 9 API.
Alternatively, Remix API can be used to programmatically pass the game data to the Remix Renderer, with *or* instead of Direct3D API. [Click for more info.](/documentation/RemixSDK.md)

## Project Documentation

- [Anti-Culling System](/documentation/AntiCullingSystem.md)
- [Contributing Guide](/CONTRIBUTING.md)
- [Foliage System](/documentation/FoliageSystem.md)
- [GPU Print](/documentation/GpuPrint.md)
- [Opacity Micromap](/documentation/OpacityMicromap.md)
- [Remix API](/documentation/RemixSDK.md)
- [Rtx Options](/RtxOptions.md)
- [Terrain System](/documentation/TerrainSystem.md)
- [Unit Test](/documentation/UnitTest.md)
