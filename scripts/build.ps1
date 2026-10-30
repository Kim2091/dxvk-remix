#requires -version 5.1
<#
.SYNOPSIS
  Build the dxvk-remix runtime (64-bit d3d9.dll) reliably.

.DESCRIPTION
  Replaces invoking upstream build_common.ps1::PerformBuild for fork-side
  runtime builds. Discovers Visual Studio via vswhere, calls vcvarsall.bat
  by full path, runs meson setup + compile + install in a single isolated
  cmd.exe shell so VS env stays consistent. Streams output to the
  PowerShell pipeline so progress and errors are visible. Throws on any
  non-zero exit.

  Pre-cleans the gitignored nv-private/ and tests/rtx/dxvk_rt_testing/
  directories that meson install regenerates — leaving them around can
  confuse the next install run.

  Prefers VS 18 (2026), including Build Tools, and falls back to VS 2019 / 2022.
  A build dir keeps the toolchain it was configured with. If it does not match
  the selected install, the script stops and asks for -Clean, because building
  into it fails with C1853 and can leave a stale d3d9.dll behind.

  If meson is not on PATH, the newest user-site Python Scripts dir that holds
  meson.exe (pip install --user) is prepended.

.PARAMETER Flavor
  release (default), debug, debugoptimized.

.PARAMETER EnableTracy
  Enable Tracy profiler integration (-Denable_tracy=true).

.PARAMETER Clean
  Wipe the matching _Comp64* dir before building.

.PARAMETER Jobs
  Ninja job count for meson compile (default 4). cl.exe adds its own /MP
  parallelism, so higher counts can fail a wide rebuild with C3859 / C1076
  (no virtual memory for the PCH), or slangc can exit 0xC0000005 silently.

.PARAMETER RegCheck
  After a successful build, run tools/regprobe/regcheck.py on the build's hot compute shaders (Phase 5 F2) and fail on
  REGCHECK FAIL (an occupancy tier lost, new spills, a variant removed). Needs an NVIDIA GPU and tools/regprobe/regprobe.exe
  (tools/regprobe/build.bat); minutes with a cold cache. Off by default.

.PARAMETER BuildDir
  Build directory, relative to the repo root. Defaults to _Comp64<Flavor>. Use a second dir to keep two configurations,
  for example a release build in _Comp64Rel next to the normal _Comp64Release.

.PARAMETER MesonArgs
  Extra meson options, for example '-Drelease_build=true'. They go to meson setup on a fresh dir and to meson configure
  on an existing one.

.EXAMPLE
  .\scripts\build.ps1
  .\scripts\build.ps1 -Flavor debugoptimized
  .\scripts\build.ps1 -Clean -EnableTracy
  .\scripts\build.ps1 -RegCheck
  .\scripts\build.ps1 -Jobs 6
  .\scripts\build.ps1 -BuildDir _Comp64Rel -MesonArgs '-Denable_d3d11=true','-Denable_dxgi=true','-Drelease_build=true','-Dbuild_tag=t01'
#>
[CmdletBinding()]
param(
    [ValidateSet('release','debug','debugoptimized')]
    [string]$Flavor = 'release',

    [switch]$EnableTracy,

    [switch]$Clean,

    [ValidateRange(1, 256)]
    [int]$Jobs = 4,

    [switch]$RegCheck,

    [string]$BuildDir = '',

    [string[]]$MesonArgs = @()
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$flavorTag = switch ($Flavor) {
    'release'        { 'Release' }
    'debug'          { 'Debug' }
    'debugoptimized' { 'DebugOptimized' }
}
$buildDir = if ($BuildDir) { $BuildDir } else { "_Comp64$flavorTag" }
$mesonExtra = ($MesonArgs | ForEach-Object { "`"$_`"" }) -join ' '
$tracyFlag = if ($EnableTracy) { 'true' } else { 'false' }

# --- Discover Visual Studio (no PATH dependency, no SetupVS lying) ---
$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vsWhere)) {
    throw "vswhere.exe not found at '$vsWhere'. Install the Visual Studio Installer (ships with VS2017+)."
}
# Without -products *, vswhere skips Build Tools installs.
$vcComponent = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
$vsPath = $null
foreach ($range in '[18.0,19.0)', '[16.0,18.0)') {
    $vsPath = & $vsWhere -latest -products * -requires $vcComponent -version $range -property installationPath | Select-Object -First 1
    if (-not [string]::IsNullOrWhiteSpace($vsPath)) { break }
}
if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "No Visual Studio 18, 2022 or 2019 installation with the x64 C++ tools found via vswhere."
}
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat not found at '$vcvars'. The detected VS install may be incomplete."
}

# The toolset version is 14.XX and the matching cl.exe version is 19.XX, so the minor number identifies the toolchain.
$toolsetFile = Join-Path $vsPath 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
$toolset = if (Test-Path $toolsetFile) { (Get-Content $toolsetFile -TotalCount 1).Trim() } else { '' }
$toolsetMinor = if ($toolset -match '^\d+\.(\d+)') { $Matches[1] } else { '' }

# meson and ninja installed with pip install --user live in a Scripts dir that a spawned shell may not have on PATH.
if (-not (Get-Command meson -ErrorAction SilentlyContinue)) {
    $userMeson = Get-ChildItem (Join-Path $env:APPDATA 'Python\Python3*\Scripts\meson.exe') -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Directory.Parent.Name -replace '\D', '') } -Descending | Select-Object -First 1
    if ($userMeson) {
        $env:PATH = "$($userMeson.DirectoryName);$env:PATH"
    } else {
        throw "meson not found on PATH or in a user-site Python Scripts dir. Install it with 'pip install --user meson ninja'."
    }
}

Write-Host "VS install: $vsPath"  -ForegroundColor DarkGray
Write-Host "MSVC:       $toolset" -ForegroundColor DarkGray
Write-Host "vcvars:     $vcvars"   -ForegroundColor DarkGray
Write-Host "meson:      $((Get-Command meson).Source)" -ForegroundColor DarkGray
Write-Host "Repo root:  $RepoRoot" -ForegroundColor DarkGray
Write-Host "Build dir:  $buildDir" -ForegroundColor DarkGray
Write-Host "Jobs:       $Jobs" -ForegroundColor DarkGray
Write-Host "Tracy:      $tracyFlag" -ForegroundColor DarkGray
Write-Host "Meson args: $mesonExtra" -ForegroundColor DarkGray

Set-Location $RepoRoot

# Required pre-cleanup. nv-private/ and tests/rtx/dxvk_rt_testing/ are
# gitignored, regenerated by meson install, and a stale copy can confuse
# the next install pass into copying the wrong files.
Remove-Item -Path .\nv-private -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
Remove-Item -Path .\tests\rtx\dxvk_rt_testing -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Wiping $buildDir for clean rebuild..." -ForegroundColor Yellow
    Remove-Item -Path $buildDir -Recurse -Force
}

Write-Host ""
Write-Host "=== Runtime $Flavor → $buildDir ===" -ForegroundColor Cyan

$alreadyConfigured = Test-Path (Join-Path $buildDir 'meson-private')

# A build dir configured with another toolchain fails with C1853 on its precompiled headers.
$compilersJson = Join-Path $buildDir 'meson-info\intro-compilers.json'
if ($alreadyConfigured -and $toolsetMinor -and (Test-Path $compilersJson)) {
    $dirCl = (Get-Content $compilersJson -Raw | ConvertFrom-Json).host.cpp.version
    if ($dirCl -match '^\d+\.(\d+)' -and $Matches[1] -ne $toolsetMinor) {
        throw "$buildDir was configured with cl $dirCl, but the selected VS install has MSVC $toolset ($vsPath). Run again with -Clean to reconfigure it with this toolchain."
    }
}

if ($alreadyConfigured) {
    # meson auto-reconfigures from inside the build dir if meson.build changed
    $configure = if ($MesonArgs.Count -gt 0) { "meson configure $mesonExtra $buildDir && " } else { '' }
    $chain = "call `"$vcvars`" x64 >nul 2>&1 && $configure" + "cd $buildDir && meson compile -v -j $Jobs && meson install"
} else {
    $chain = "call `"$vcvars`" x64 >nul 2>&1 && meson setup --buildtype $Flavor --backend ninja -Denable_tracy=$tracyFlag $mesonExtra $buildDir && cd $buildDir && meson compile -v -j $Jobs && meson install"
}

$buildStart = Get-Date

# Native-command stderr triggers PowerShell's NativeCommandError under
# EAP=Stop even on exit 0. Exit code is the authoritative signal here.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    cmd.exe /c $chain
    $exit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevEAP
}
if ($exit -ne 0) {
    throw "Runtime $Flavor build failed (exit $exit)."
}

# Sanity-check artifacts so the caller can trust exit 0
$artifacts = @{
    "$buildDir/src/d3d9/d3d9.dll" = 'runtime DLL (build dir)'
    '_output/d3d9.dll'            = 'runtime DLL (deploy)'
}

$missing = @()
foreach ($rel in $artifacts.Keys) {
    $abs = Join-Path $RepoRoot $rel
    if (-not (Test-Path $abs)) { $missing += "$($artifacts[$rel]) ($rel)" }
}
if ($missing.Count -gt 0) {
    throw "Build reported success but artifacts are missing: $($missing -join ', ')"
}

Write-Host ""
Write-Host "=== Runtime build OK ===" -ForegroundColor Green
foreach ($rel in $artifacts.Keys) {
    $abs = Join-Path $RepoRoot $rel
    $info = Get-Item $abs
    Write-Host ("  {0,-24} {1:yyyy-MM-dd HH:mm:ss}  {2,12:N0} bytes  {3}" -f $artifacts[$rel], $info.LastWriteTime, $info.Length, $rel) -ForegroundColor Green
}

# An incremental build with nothing to do leaves d3d9.dll alone, so this warns rather than fails.
$builtDll = Get-Item (Join-Path $RepoRoot "$buildDir/src/d3d9/d3d9.dll")
if ($builtDll.LastWriteTime -lt $buildStart) {
    Write-Host ("  WARNING: d3d9.dll was not relinked by this build (last written {0:yyyy-MM-dd HH:mm:ss})." -f $builtDll.LastWriteTime) -ForegroundColor Yellow
    Write-Host "  If you changed source or shaders, build again before deploying." -ForegroundColor Yellow
}

# --- Optional register check (tools/regprobe/regcheck.py) ---
if ($RegCheck) {
    Write-Host ""
    Write-Host "=== Register check (tools/regprobe/regcheck.py) ===" -ForegroundColor Cyan
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        python (Join-Path $RepoRoot 'tools/regprobe/regcheck.py') --build-dir (Join-Path $RepoRoot $buildDir)
        $regExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
    }
    if ($regExit -ne 0) {
        throw "Register check failed (exit $regExit): see the REGCHECK FAIL lines above."
    }
}

# --- Optional auto-deploy to user-local game install paths ---
#
# If `scripts/auto-deploy.local.ps1` exists (gitignored), dot-source it and
# copy `_output/d3d9.dll` to each path listed in `$AutoDeployTargets`. Lets
# contributors skip manual copy/paste into their test game install without
# leaking machine-specific paths into the tracked tree.
#
# Local file template:
#   $AutoDeployTargets = @(
#     'I:\path\to\game\.trex'
#   )
$autoDeployScript = Join-Path $PSScriptRoot 'auto-deploy.local.ps1'
if (Test-Path $autoDeployScript) {
    $AutoDeployTargets = @()
    . $autoDeployScript
    if ($AutoDeployTargets.Count -gt 0) {
        Write-Host ""
        Write-Host "=== Auto-deploy (from scripts/auto-deploy.local.ps1) ===" -ForegroundColor Cyan
        $sourceDll = Join-Path $RepoRoot '_output/d3d9.dll'
        foreach ($target in $AutoDeployTargets) {
            if (-not (Test-Path $target)) {
                Write-Host "  SKIP (target dir missing): $target" -ForegroundColor Yellow
                continue
            }
            $destDll = Join-Path $target 'd3d9.dll'
            try {
                Copy-Item -Path $sourceDll -Destination $destDll -Force
                $info = Get-Item $destDll
                Write-Host ("  copied -> {0:yyyy-MM-dd HH:mm}  {1,12:N0} bytes  {2}" -f $info.LastWriteTime, $info.Length, $destDll) -ForegroundColor Green
            } catch {
                Write-Host "  FAIL: $target -- $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}
