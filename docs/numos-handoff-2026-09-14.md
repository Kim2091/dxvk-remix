# Numos handoff - 2026-09-14

## 2026-09-16 update

See `docs/numos-optimizations-2026-09-16.md` for the measured per-pass baseline (from the game's own
`rtx.profile.gpuStages` log) and the temporal-interleave work that followed it: the screen pass now
marches half the pixels per frame and reprojects the rest (`cloudScreenInterleaveMode`), the sun grid
bakes half its columns per frame (`cloudSunGridInterleaveMode`), and the reflection dome marches half
its rows per frame (`cloudSecondaryLutInterleaveMode`). All three fall back to a full update whenever
the cloud inputs change. Everything below this section predates that work.

## Start here

- Actual renderer repo: `C:\Users\sparkles\Projects\Fable_5_testing\wsn3g`, branch `revised-9-10`.
- Codex workspace/writable scratch: `C:\Users\sparkles\Projects\vapourkit`; scratch scripts and build logs live in `temp\` there. This is NOT the renderer source repo.
- Game: `C:\Users\sparkles\Projects\Games\Fallout New Vegas`.
- 64-bit runtime deployment: game's `.trex\d3d9.dll` and `.trex\d3d9.pdb`. Do not replace the root 32-bit bridge DLL.
- Game config: game's `rtx.conf`; mods/config layers can override defaults.
- Runtime log (only when requested/useful): game's `rtx-remix\logs\remix-dxvk.log`. User explicitly says the current performance numbers are unreliable; do not use them to override their FPS observations or read them for the accepted optimization.
- IDE tabs `skin_sss.usda` and `mpv.conf` are unrelated to this work.

## Current result and commits

- `6eec83fca`: shared ambient column integration. User tested: "quite a bit faster" and explicitly chose to keep it.
- `ec5b6cf49`: enables that accepted optimization by default and removes its experimental label.
- The cleanup commit containing this file removes the rejected world-space reflection-density cache entirely (option/UI, texture, bake, shader variant, and associated mode logging). Find it with `git log -1`. The original Fast Cloud Reflections dome remains.
- Removed experiment was `cd44b7024`; user found it same or worse with no visual improvement. Do not reintroduce it without a materially different reason. It added a density bake while retaining the full dome march; that did not establish a win.
- Last deployment was the test build from `6eec83fca`, not the subsequent default/label change or this removal. Cleanup is built/validated locally, not deployed as part of this handoff request. The game can therefore still show the removed experimental control until next deployment.
- Last deployed DLL SHA256: `AF28871EE6344E03AA0B15A6811E8B605C35A1BEB4E3749ED8FB3B80DB5266B5`.
- Last deployed PDB SHA256: `06D014BF1AE9DB0DCF1BFCEEB415291566F3122D35F62DEE08A150742DF30341`.
- Previous runtime backup suffix: `.backup-pre-ambient-scan-20260914-223000`.

## Important source files

Under `src/dxvk/rtx_render/`:
- `rtx_atmosphere.cpp`: atmosphere arguments, resource allocation, dispatch ordering, grid bake invalidation, secondary reflection dome.
- `rtx_atmosphere.h`: options/defaults, resource members, grid dimensions.
- `rtx_atmosphere_ui.cpp`: cloud settings and GUI controls.
- `rtx_context.cpp`: pass ordering and GPU timestamp logging.

Under `src/dxvk/shaders/rtx/pass/atmosphere/`:
- `cloud_ambient_density_grid.comp.slang`: accepted optimization, `CLOUD_AMBIENT_COLUMN_SCAN` compile-time variant. Option is `rtx.atmosphere.cloudAmbientColumnScan`, default true. 64 full-density samples per XZ column, groupshared suffix sum produces 32 output heights. 4.2 million density evaluations instead of nominal 12.6 million. No temporal reuse, reduced update rate, or persistent density cache. Old shader remains available via toggle.
- `cloud_sun_density_grid.comp.slang`: current sun optical-depth bake; still independently integrates per output voxel.
- `cloud_nubis3_common.slangh`: procedural density, bake controls, `sampleCloudSunOpticalDepthAtWorld`, `sampleCloudAmbientOpticalDepthAtWorld`.
- `cloud_march_common.slangh`: visible/reflection cloud marching, adaptive budget, SDF skipping, shading, moon shadows, echo second layer.
- `cloud_secondary_lut.comp.slang`: original 256x256 full-sphere reflection dome, built every frame when enabled, plus mip chain. Uses main cloud sampling settings.
- `atmosphere_sky.slangh`: dome consumer; Sky Cloud Bleed also consumes a coarse dome mip, complicating removal of dome generation.
- `atmosphere_common.slangh`: world/voxel mapping and optical-depth lookup helpers.
- `cloud_render.comp.slang`: screen cloud pass and shader variants.

Alpha/AP investigation: `src/dxvk/shaders/rtx/pass/composite/` and `src/dxvk/shaders/rtx/algorithm/geometry_resolver.slangh` / `geometry_resolver_state.slangh`.

Detailed chronological notes: `docs/numos-optimizations-2026-09-14.md`. Early sections contain historical, superseded claims (including uncommitted state and outdated test instructions); use this handoff and current code as the current status.

## Likely next optimization target

User reports 100% coverage costs 10-15 FPS even with low samples. Source confirms:
- Adaptive march limit is Max Cloud Samples, not the base Cloud Samples setting. Last observed startup max was 64; do not assume current live settings.
- Dense coverage loses cheap empty-density exits and empty-space skips.
- Lighting grids are 256x32x256, independent of view samples, and rebuild every frame with ground cloud shadows enabled. Ambient is now improved; sun remains a candidate, not a measured dominant bottleneck.
- Sun uses 8 coherent near taps plus adaptive far taps and an additional long tail, not simply 8 taps everywhere. It freezes coverage/type controls at the ray origin. A globally cached local density field is therefore NOT an exact replacement when weather spreads vary.
- Sun rays can leave the 12 km local grid, with samples out to 40 km and a further extrapolated tail. Wrapping a local cache can introduce false repeated shadowing. The procedural hex-tiled/detail field is not simply periodic at the local grid extent.
- Bake density intentionally omits camera-distance-dependent fine detail; the removed reflection cache did not. Do not blindly reuse view-density caches for lighting.
- Shared ambient sampling changes quadrature; user accepted its observed result, but no measured percentage GPU speedup is established. Sun-direction prefix integration needs its own treatment of direction, boundaries, weather controls, and sunset lighting.

## Other work / preserve these fixes

- Preserve `f8544fa82` (render-scale extent fix/removal of temporal smoothing), `a79cac79d` (silhouette reconstruction), `599ad5818` (stable reduced-scale jitter). User praised geometry edges and moving-cloud stability after these.
- Alpha-tested/transparent foliage under clouds/AP remains unresolved. `287e8323e` and `005fb0a5c` did not fix the user's examples. RR off did not resolve it. User explicitly moved on; do not resume without direction.
- Prior Terra diagnosis: foreground separation can be a no-op because it requires foreground alpha > 0 AND !performedAnyPSR(); RR off does not disable PSR, and PSR resolver does not populate that foreground resource. Debug view 880 reads primary depth and cannot identify the foliage route alone. A combined foreground-alpha/PSR diagnostic was suggested but not pursued.
- Earlier requested backlog: second-layer preset support and 50 m minimum altitude, closer parity with first-layer controls, cloud seed system, scattered preset, four named custom presets (custom1-custom4 config labels), and cloud slab-entry brightening/bottom darkening. Audit current source/history before claiming these are complete; this session did not implement or verify them.

## Working rules and build/deploy

- Commit every change locally; NEVER push. Do not spawn agents unless explicitly requested or required by applicable instructions.
- Read repo `AGENTS.md`, `.agents/skills/build-remix/SKILL.md`, `.agents/skills/deploy-and-test-game/SKILL.md`, and `docs/ShaderVariants.md` before relevant work.
- Significant optional shader paths need compile-time variants, not unconditional feature defines. Slang entry points are auto-discovered; new C++ files need Meson registration.
- Use `meson compile`, never direct ninja. Release build directory is `_Comp64Release`.
- Meson: `C:\Users\sparkles\AppData\Roaming\Python\Python314\Scripts\meson.exe`.
- Python: `C:\Python314\python.exe`.
- Validator: `C:\VulkanSDK\1.4.357.0\Bin\spirv-val.exe --target-env vulkan1.3 --scalar-block-layout <module.spv>`.
- SPIR-V: `_Comp64Release/src/dxvk/rtx_shaders/`; runtime: `_Comp64Release/src/d3d9/d3d9.dll` and matching PDB.
- Build example from repo: `& <meson path> compile -C _Comp64Release *> C:\Users\sparkles\Projects\vapourkit\temp\build.log`; propagate `$LASTEXITCODE`.
- Generated shader headers can require another Meson invocation to relink. Verify exact SPIR-V bytes occur in the final DLL with Python mmap.find before deployment. Stale unused generated files can remain in the build directory after source removal; source references and current embedded modules matter.
- Git invocation needs `-c safe.directory=C:/Users/sparkles/Projects/Fable_5_testing/wsn3g -c core.excludesFile=C:/Users/sparkles/Projects/Fable_5_testing/wsn3g/.gitignore`.
- Writes/builds/git/deploy outside Vapourkit require sandbox escalation in this environment. Keep CRLF source endings. Use script files for complex edits/quoting.
- Nonfatal vswhere diagnostic and missing external-library PDB warnings are familiar; check the actual compiler/linker exit code.
- Before deployment check both FalloutNV and NvRemixBridge. Never kill them. Back up DLL/PDB together with a unique suffix and verify hashes; recheck processes immediately before replacing; verify deployed hashes.
- Do not request repeated close/reopen tests. Build and validate first; only request closure when an actual deployment is ready and needed. Live toggles should be tested in one session. Do not auto-launch or modify the bridge.
