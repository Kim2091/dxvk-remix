"""Compile one dxvk-remix slang shader variant exactly like compile_shaders.py and print the driver's register statistics.

In-repo copy of the regprobe probe (Phase 5 F2); the compile flags are unchanged.
Usage: python probe.py [--tree <repo>] <shader.slang relative to src/dxvk/shaders/rtx> <tag> [DEFINE ...]
       (--tree defaults to this repository). Needs regprobe.exe (build.bat) and an NVIDIA GPU.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
INCLUDES = ['src/dxvk/shaders', 'submodules/rtxdi/rtxdi-sdk/include', 'submodules/rtxcr/shaders/include', 'submodules',
            'submodules/rtxdi/rtxdi-sdk/include/volumetrics']


def compile_variant(tree, shader, tag, defines, out_dir):
    """Compiles to out_dir/<tag>.spv. Returns (spv path or None, error text)."""
    os.makedirs(out_dir, exist_ok=True)
    spv = os.path.join(out_dir, tag + '.spv')
    cmd = [os.path.join(tree, 'external/slang/slangc.exe'), '-entry', 'main', '-target', 'spirv', '-zero-initialize',
           '-emit-spirv-directly']
    for i in INCLUDES:
        cmd += ['-I', os.path.join(tree, i)]
    cmd += [os.path.join(tree, 'src/dxvk/shaders/rtx', shader), '-D__SLANG__'] + ['-D' + d for d in defines]
    cmd += ['-matrix-layout-column-major', '-Wno-30081', '-fvk-use-scalar-layout', '-o', spv]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-3000:]
    return spv, ''


def main(argv):
    tree = REPO
    if len(argv) >= 2 and argv[0] == '--tree':
        tree, argv = argv[1], argv[2:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    shader, tag, defines = argv[0], argv[1], argv[2:]
    spv, err = compile_variant(tree, shader, tag, defines, os.path.join(HERE, 'out'))
    if spv is None:
        print(f'{tag}: COMPILE FAILED\n{err}')
        return 1
    r = subprocess.run([os.path.join(HERE, 'regprobe.exe'), spv], capture_output=True, text=True)
    line = r.stdout.strip().replace(spv, tag)
    print(line if line else f'{tag}: PROBE FAILED {r.stderr[-1000:]}')
    return 0 if line else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
