# Upstream review log

Remix Plus never rebases onto `upstream` (NVIDIAGameWorks/dxvk-remix `main`). Once a month, review the upstream commits since the last-reviewed commit and record the review here. Policy: `AGENTS.md`, Upstream Policy.

How to review:
1. `git fetch upstream`, then `git log <last-reviewed>..upstream/main`.
2. Give every commit one row with a verdict and a one-line reason: **take** (port it), **skip**, or **maybe** (decide later, say what would decide it).
3. Port each take on its own branch, one feature at a time, after the license check. Fill in the port commit when it lands.
4. Add a section per review: the date, the commit range, and the table. End the file with the next last-reviewed commit.

Pass 1 used **idea** for what is now **maybe**, and its takes were cherry-picked with `-x`.

## Pass 1: 2026-10-03, `b474e56b0..324aba8f8` (15 commits)

| Upstream | Date | Subject | Decision | Notes |
| --- | --- | --- | --- | --- |
| `ec8132d67` | 09-21 | Windows on Arm build instructions | skip | ARM |
| `94ec487e2` | 09-22 | RTXIO arm64 binaries | skip | ARM |
| `273376713` | 09-22 | REMIX-5768 camera turns when swapping Remix menus | take → `78a832ca1` | Bridge DirectInput hook; helps D3D9 bridge games, not API hosts. AGENTS.md conflict: kept both. |
| `84acb5227` | 09-22 | REMIX-5178 eyes in virtual instances | take → `697b56c25` | Clean. New debug view 52 (Eye Regions), no clash with fork indices. |
| `2ccdd8212` | 09-22 | Medium test image export fix | skip | Test harness |
| `6da218ee8` | 09-23 | REMIX-5137 fallback light volumetric radiance scale | take → `8526d7618` | Clean, new option with GUI. |
| `c7d3aae50` | 09-29 | Image test metrics as post-warmup averages | skip | Test harness |
| `923c61528` | 09-29 | REMIX-4964 reduced VRAM (RTXDI, FG, PSR) | take → `0b2886492` + `ccace60d0` | Include conflict in `rtx_resources.cpp`. `m_rtxdiReservoirBuffer` is now an `AliasedBuffer`: light BVH binding adapted (`ccace60d0`). With gradients off (RR + no ReSTIR GI) RTXDI uses 2 reservoir pages; light BVH select reads and writes its own pixel in page 0, which is safe. |
| `5229c042a` | 09-29 | DLSS-NR on Windows-on-Arm | skip | ARM |
| `271925bee` | 09-30 | REMIX-4964 dead RR code, GBuffer/decal refactor, NEE cleanup | idea | 391-line `geometry_resolver.slangh` rewrite collides with fork GBuffer work (split stages, decals, PSR). Revisit when the renderer split touches the resolver. |
| `9d98e3283` | 09-30 | arm64ec API shim on WoA | skip | ARM |
| `d65fdf029` | 10-01 | ARM64 / ARM64EC GitHub artifacts | skip | CI |
| `11abe8361` | 10-01 | Output installation in GitHub workflow | skip | CI |
| `e048263db` | 10-02 | Sparse rendering: obsolete options, defaults, composite/RTXDI toggle fixes | idea | Fixes are wanted but it removes options and touches RTXDI temporal/spatial reuse next to the light BVH. Port after an in-game check that sparse rendering is used at all. |
| `324aba8f8` | 10-02 | RTXIO bump for arm64 | skip | ARM |

## Between passes: SHaRC only

The SHaRC work compared upstream's `303c81336` (SHaRC 1.8.3 integration), `90d74fbe1` and `8ce1b446f` with ours and ported them by hand (`bd59552f1` .. `90da7c67f`). The rest of the range was not reviewed.

## Between passes: DLSS Frame Generation preset, 2026-10-26

Ported by hand: the DLFG Lite half of `09808cc70` (its DLSS-NR half was ported earlier), the frame generation labels of `97355c1a1` (the DLSS-NR renames were not taken; the menu keeps its own names), and `c1a891bf8` (pipeline-handle race and command-list recycle order; the duplicate acquire wait was already fixed by `616df3dc2`). The rest of the range was not reviewed.

## Between passes: incremental bindless buffer descriptors, 2026-11-02

Take: `89257d282` (merge `9dd9b9398`, REMIX-6230), ported by hand in "Bindless buffer table: updates follow the changed slots". Kept from upstream: the slot list in `RetainedBufferTable`, `PendingSlotQueues`, `BindlessBufferMirror` and the unit tests. Different from upstream: registered buffers pin their backing, because a full-buffer write would otherwise rename a table buffer behind the mirror. `rtx.trackBindlessResourcesOnChange` stays, the failed-allocation recovery stays instead of a throw, and retired buffers need no snapshot resource. Measured gain: pending. The rest of the range was not reviewed.

## Starting point, 2026-10-15

- `upstream/main` is at `029602f83` (2026-10-15).
- Last reviewed: `324aba8f8` (2026-10-02). `324aba8f8..029602f83` is 51 commits, not yet reviewed, apart from the three SHaRC commits above.

Last reviewed: `324aba8f8`.
