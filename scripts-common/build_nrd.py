#!/usr/bin/env python3
"""Build NRD.dll from the unmodified submodules/nrd checkout, with the options Remix needs.

Called from the root meson.build while configuring, because the runtime's headers, shaders and install
step all need the result. On Windows it runs inside the Visual Studio environment meson was started from.
Elsewhere it cross-builds NRD.dll with the clang-cl, lld-link and llvm-rc wrappers given by --xwin-tools, while
ShaderMake, NRD's shader build tool, is built for and runs on the build machine.
NRD's CMake project downloads its pinned MathLib and ShaderMake dependencies on the first run.
The build output stays in the given build directory. CMake only writes its generated, git-ignored
Shaders/NRDConfig.hlsli into the submodule.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

CMAKE_MIN_VERSION = (3, 30)

# The normal and roughness encodings must match the ones in rtx/utility/nrd.slangh.
# Quad intrinsics need VK_KHR_compute_shader_derivatives, which the runtime does not enable.
CMAKE_OPTIONS = {
    "CMAKE_BUILD_TYPE": "Release",
    "NRD_EMBEDS_SPIRV_SHADERS": "ON",
    "NRD_EMBEDS_DXIL_SHADERS": "OFF",
    "NRD_EMBEDS_DXBC_SHADERS": "OFF",
    "NRD_NORMAL_ENCODING": "2",
    "NRD_ROUGHNESS_ENCODING": "1",
    "NRD_SUPPORTS_QUAD_INTRINSICS": "OFF",
}


def cmake_version(cmake):
    try:
        output = subprocess.run([cmake, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"cmake version (\d+)\.(\d+)", output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def find_cmake():
    candidates = [shutil.which("cmake")]
    candidates += glob.glob(os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"))
    candidates += glob.glob(os.path.expandvars(
        r"%ProgramFiles%\Microsoft Visual Studio\*\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"))
    for candidate in candidates:
        if candidate:
            version = cmake_version(candidate)
            if version and version >= CMAKE_MIN_VERSION:
                return candidate
    sys.exit("NRD needs CMake %d.%d or newer. Install it or add it to PATH." % CMAKE_MIN_VERSION)


def write_toolchain(path, tools):
    """Writes a CMake toolchain file that targets x64 Windows through the xwin wrappers."""
    tools = tools.replace("\\", "/")
    lines = [
        "set(CMAKE_SYSTEM_NAME Windows)",
        "set(CMAKE_SYSTEM_VERSION 10.0)",
        "set(CMAKE_SYSTEM_PROCESSOR AMD64)",
        'set(CMAKE_C_COMPILER "%s/clang-cl")' % tools,
        'set(CMAKE_CXX_COMPILER "%s/clang-cl")' % tools,
        'set(CMAKE_LINKER "%s/lld-link")' % tools,
        'set(CMAKE_RC_COMPILER "%s/llvm-rc")' % tools,
        "set(CMAKE_AR llvm-lib)",
        # The runtime's cross file builds with SSE4.1, and clang-cl rejects SSE4.1 intrinsics without it.
        'set(CMAKE_C_FLAGS_INIT "/clang:-msse4.1")',
        'set(CMAKE_CXX_FLAGS_INIT "/clang:-msse4.1")',
        # NRD.dll needs no manifest, so the build does not need mt.exe.
        'set(CMAKE_SHARED_LINKER_FLAGS_INIT "/MANIFEST:NO")',
        "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)",
        # xwin splats only the release CRT, and CMake's compiler checks link in the Debug configuration by default.
        "set(CMAKE_TRY_COMPILE_CONFIGURATION Release)",
    ]
    mt = shutil.which("llvm-mt")
    if mt:
        lines.append('set(CMAKE_MT "%s")' % mt)
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def link_host_shadermake(cmake_build):
    """Points the ShaderMake path that NRD calls at the build machine's ShaderMake executable.

    ShaderMake names the tool with the target's executable suffix, so a cross build looks for ShaderMake.exe,
    while the tool itself is built for the build machine without a suffix.
    """
    cache = os.path.join(cmake_build, "CMakeCache.txt")
    with open(cache, encoding="utf-8") as file:
        match = re.search(r"^SHADERMAKE_PATH:INTERNAL=(.*)$", file.read(), re.MULTILINE)
    if not match or not match.group(1).endswith(".exe"):
        return
    expected = match.group(1)
    if os.path.lexists(expected):
        return
    os.makedirs(os.path.dirname(expected), exist_ok=True)
    os.symlink(os.path.basename(expected)[:-len(".exe")], expected)


def source_revision(source):
    result = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="The NRD submodule directory.")
    parser.add_argument("--build", required=True, help="The build directory, which also holds the result.")
    parser.add_argument("--jobs", type=int, default=4, help="The number of parallel compile jobs.")
    parser.add_argument("--xwin-tools", help="The directory with the clang-cl, lld-link and llvm-rc wrappers. "
                        "Given when cross-building from a machine that is not Windows.")
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    build = os.path.abspath(args.build)
    if not os.path.exists(os.path.join(source, "CMakeLists.txt")):
        sys.exit(source + " is empty. Run: git submodule update --init submodules/nrd")

    output = os.path.join(build, "bin")
    shaders = os.path.join(build, "shaders")
    dll = os.path.join(output, "NRD.dll")
    stamp = os.path.join(build, "nrd-build.stamp")
    key = source_revision(source) + " " + repr(sorted(CMAKE_OPTIONS.items()))
    if args.xwin_tools:
        key += " xwin"

    if os.path.exists(dll) and os.path.exists(os.path.join(source, "Shaders", "NRDConfig.hlsli")) and os.path.exists(stamp):
        with open(stamp, encoding="utf-8") as file:
            if file.read() == key:
                print("NRD is up to date.")
                return

    os.makedirs(build, exist_ok=True)
    cmake = find_cmake()

    definitions = dict(CMAKE_OPTIONS)
    definitions["NRD_SHADERS_PATH"] = shaders.replace("\\", "/")
    definitions["CMAKE_RUNTIME_OUTPUT_DIRECTORY"] = output.replace("\\", "/")
    definitions["CMAKE_LIBRARY_OUTPUT_DIRECTORY"] = output.replace("\\", "/")
    if args.xwin_tools:
        toolchain = os.path.join(build, "xwin-toolchain.cmake")
        write_toolchain(toolchain, os.path.abspath(args.xwin_tools))
        definitions["CMAKE_TOOLCHAIN_FILE"] = toolchain.replace("\\", "/")
        # FXC is only needed for DXBC shaders, and finding it reads the Windows registry.
        definitions["SHADERMAKE_FIND_FXC"] = "OFF"
    defines = ["-D%s=%s" % item for item in definitions.items()]

    cmake_build = os.path.join(build, "cmake")
    subprocess.run([cmake, "-S", source, "-B", cmake_build, "-G", "Ninja"] + defines, check=True)
    if args.xwin_tools:
        link_host_shadermake(cmake_build)
    subprocess.run([cmake, "--build", cmake_build, "--target", "NRD", "--parallel", str(args.jobs)], check=True)

    if not os.path.exists(dll):
        sys.exit("The NRD build finished without producing " + dll)

    with open(stamp, "w", encoding="utf-8") as file:
        file.write(key)


if __name__ == "__main__":
    main()
