#requires -version 5.1
<#
.SYNOPSIS
  Build the dxvk-remix-mirrorsedge runtime (64-bit d3d9.dll) reliably.

.DESCRIPTION
  Adapted from the Remix Plus fork's scripts/build.ps1. Discovers Visual Studio via
  vswhere, calls vcvarsall.bat by full path, and runs meson setup + compile + install
  in a single isolated cmd.exe shell so the VS environment stays consistent. Throws on
  any non-zero exit, then verifies the artifacts actually moved.

  TWO DEVIATIONS FROM THE ORIGINAL, both machine-specific and both load-bearing:

  1. vswhere range is [16.0,19.0) -prerelease, not [16.0,18.0). This box builds with
     VS 18 BuildTools; sourcing VS 2022 Professional's vcvars alone does NOT put cl on
     PATH here, so narrowing the range produces a confusing "cl not found" failure.

  2. NoDefaultCurrentDirectoryInExePath is REMOVED from the environment, not blanked.
     This shell sets it, and with it set cmd refuses to run a command found only in the
     current directory - which is how the upstream build scripts invoke vcvarsall.bat
     and vswhere.exe. The failure is SILENT: the build exits 0 having done nothing.

.PARAMETER Flavor
  release (default), debug, debugoptimized.

.PARAMETER EnableTracy
  Enable Tracy profiler integration (-Denable_tracy=true).

.PARAMETER Clean
  Wipe the matching _Comp64* dir before building.

.EXAMPLE
  .\scripts\build.ps1
  .\scripts\build.ps1 -Clean
#>
[CmdletBinding()]
param(
    [ValidateSet('release','debug','debugoptimized')]
    [string]$Flavor = 'release',

    [switch]$EnableTracy,

    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$flavorTag = switch ($Flavor) {
    'release'        { 'Release' }
    'debug'          { 'Debug' }
    'debugoptimized' { 'DebugOptimized' }
}
$buildDir = "_Comp64$flavorTag"
$tracyFlag = if ($EnableTracy) { 'true' } else { 'false' }

# See deviation 2 above. Must be REMOVED, not set to an empty string.
if (Test-Path Env:\NoDefaultCurrentDirectoryInExePath) {
    Remove-Item Env:\NoDefaultCurrentDirectoryInExePath
}

# --- Discover Visual Studio (no PATH dependency, no SetupVS lying) ---
$vsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vsWhere)) {
    throw "vswhere.exe not found at '$vsWhere'. Install the Visual Studio Installer (ships with VS2017+)."
}
# See deviation 1 above.
$vsPath = & $vsWhere -latest -prerelease -version '[16.0,19.0)' -property installationPath
if ([string]::IsNullOrWhiteSpace($vsPath)) {
    throw "No Visual Studio 2019+ installation found via vswhere."
}
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat not found at '$vcvars'. The detected VS install may be incomplete."
}

# meson and ninja come from the conda install on this box
$env:PATH = "C:\Users\sparkles\miniconda3;C:\Users\sparkles\miniconda3\Scripts;" +
            (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer') + ";" + $env:PATH

Write-Host "VS install: $vsPath"  -ForegroundColor DarkGray
Write-Host "vcvars:     $vcvars"   -ForegroundColor DarkGray
Write-Host "Repo root:  $RepoRoot" -ForegroundColor DarkGray
Write-Host "Build dir:  $buildDir" -ForegroundColor DarkGray

Set-Location $RepoRoot

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Wiping $buildDir for clean rebuild..." -ForegroundColor Yellow
    Remove-Item -Path $buildDir -Recurse -Force
}

Write-Host ""
Write-Host "=== Runtime $Flavor -> $buildDir ===" -ForegroundColor Cyan

$alreadyConfigured = Test-Path (Join-Path $buildDir 'meson-private')
if ($alreadyConfigured) {
    $chain = "call `"$vcvars`" x64 >nul 2>&1 && cd $buildDir && meson compile -v && meson install"
} else {
    $chain = "call `"$vcvars`" x64 >nul 2>&1 && meson setup --buildtype $Flavor --backend ninja -Denable_tracy=$tracyFlag $buildDir && cd $buildDir && meson compile -v && meson install"
}

# Native-command stderr triggers PowerShell's NativeCommandError under EAP=Stop even on
# exit 0. The exit code is the authoritative signal.
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

# Sanity-check artifacts so the caller can trust exit 0. The upstream scripts exit 0 even
# when they did nothing, which is exactly what this guards against.
$artifacts = [ordered]@{
    "$buildDir/src/d3d9/d3d9.dll" = 'runtime DLL (build dir)'
    '_output/d3d9.dll'            = 'runtime DLL (deploy)'
}

$missing = @()
foreach ($rel in $artifacts.Keys) {
    if (-not (Test-Path (Join-Path $RepoRoot $rel))) { $missing += "$($artifacts[$rel]) ($rel)" }
}
if ($missing.Count -gt 0) {
    throw "Build reported success but artifacts are missing: $($missing -join ', ')"
}

Write-Host ""
Write-Host "=== Runtime build OK ===" -ForegroundColor Green
foreach ($rel in $artifacts.Keys) {
    $info = Get-Item (Join-Path $RepoRoot $rel)
    Write-Host ("  {0,-24} {1:yyyy-MM-dd HH:mm}  {2,12:N0} bytes  {3}" -f $artifacts[$rel], $info.LastWriteTime, $info.Length, $rel) -ForegroundColor Green
}
