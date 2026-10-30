"""Builds and installs NUMOS's compute shaders for the runtime (submodules/numos).

  build:   numos_shaders.py build <numos dir> <slangc> <spirv out dir> <stamp file>
                            [--spirv-opt <spirv-opt> --spirv-val <spirv-val> --table <pass table out>]
  install: numos_shaders.py install <numos dir> <spirv dir> <destination dir>

build compiles every NUMOS pass with Remix's slangc through NUMOS's own tools/build_shaders.py, then
checks the result against the pass table compiled into the library (src/generated/passes.inc): the
runtime refuses any SPIR-V whose hash differs from the table, so a mismatch fails the build here
instead of leaving a black sky at run time.

With --spirv-opt (release builds), every pass then gets the same spirv-opt pass and spirv-val check
as Remix's embedded shaders, which also strips the debug names. The optimised files have new hashes,
so a pass table is generated from them into --table, and the library is compiled against it
(NUMOS_PASS_TABLE_FILE). That table must list the same passes and bindings as the committed one,
differing only in the hashes and the stripped variable names.

install copies the SPIR-V to <destination>/numos/spirv and the night sky data to
<destination>/numos/nightsky, where the runtime loads them from beside d3d9.dll.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys

# The same flags as compile_shaders.py's -spirvopt pass. Bindings and spec constants stay as declared,
# since the pass table and the host refer to them by number.
# The passes are spirv-opt's -O recipe without combine-access-chains, which drops NonUniform decorations.
SPIRV_OPT_PASSES = [
    'wrap-opkill', 'eliminate-dead-branches', 'merge-return', 'inline-entry-points-exhaustive', 'eliminate-dead-functions',
    'eliminate-dead-code-aggressive', 'private-to-local', 'eliminate-local-single-block', 'eliminate-local-single-store',
    'eliminate-dead-code-aggressive', 'scalar-replacement=0', 'convert-local-access-chains', 'eliminate-local-single-block',
    'eliminate-local-single-store', 'eliminate-dead-code-aggressive', 'ssa-rewrite', 'eliminate-dead-code-aggressive', 'ccp',
    'eliminate-dead-code-aggressive', 'loop-unroll', 'eliminate-dead-branches', 'redundancy-elimination',
    'simplify-instructions', 'scalar-replacement=0', 'convert-local-access-chains', 'eliminate-local-single-block',
    'eliminate-local-single-store', 'eliminate-dead-code-aggressive', 'ssa-rewrite', 'eliminate-dead-code-aggressive',
    'vector-dce', 'eliminate-dead-inserts', 'eliminate-dead-branches', 'simplify-instructions', 'if-conversion',
    'copy-propagate-arrays', 'reduce-load-size', 'eliminate-dead-code-aggressive', 'merge-blocks', 'redundancy-elimination',
    'eliminate-dead-branches', 'merge-blocks', 'simplify-instructions']
SPIRV_OPT_FLAGS = ['--' + p for p in SPIRV_OPT_PASSES] + ['--strip-debug', '--strip-nonsemantic', '--preserve-bindings', '--preserve-spec-constants',
                   '--target-env=vulkan1.2', '--scalar-block-layout']
SPIRV_VAL_FLAGS = ['--target-env', 'vulkan1.2', '--scalar-block-layout']


def table_layout(text):
    """Returns the pass table without the hashes and binding variable names, which an optimised build changes."""
    out = []
    for line in text.replace('\r\n', '\n').splitlines():
        if line.startswith('NUMOS_PASS('):
            out.append(re.sub(r', 0x[0-9a-f]+ull,', ', <hash>,', line))
        elif line.startswith('NUMOS_PASS_BINDING('):
            out.append(re.sub(r', "[^"]*"\)$', ')', line))
    return out


def optimize(out, spirv_opt, spirv_val):
    for spv in sorted(out.glob('*.spv')):
        tmp = spv.with_suffix('.opt.tmp')
        r = subprocess.run([str(spirv_opt), *SPIRV_OPT_FLAGS, str(spv), '-o', str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'spirv-opt failed on {spv.name}:\n{r.stdout}{r.stderr}')
        r = subprocess.run([str(spirv_val), *SPIRV_VAL_FLAGS, str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f'spirv-val failed on the optimised {spv.name}:\n{r.stdout}{r.stderr}')
        tmp.replace(spv)


def build(numos, slangc, out, stamp, spirv_opt=None, spirv_val=None, table=None):
    tools = numos / 'tools'
    committed = numos / 'src' / 'generated' / 'passes.inc'
    subprocess.run([sys.executable, str(tools / 'build_shaders.py'), '--slangc', str(slangc), '--out', str(out),
                    '--jobs', '4'], check=True)
    subprocess.run([sys.executable, str(tools / 'gen_pass_table.py'), '--spirv', str(out),
                    '--out', str(committed), '--check'], check=True)
    if spirv_opt:
        optimize(out, spirv_opt, spirv_val)
        subprocess.run([sys.executable, str(tools / 'gen_pass_table.py'), '--spirv', str(out), '--out', str(table)],
                       check=True)
        if table_layout(table.read_text(encoding='utf-8')) != table_layout(committed.read_text(encoding='utf-8')):
            sys.exit(f'spirv-opt changed the NUMOS pass layout: {table} differs from {committed} in more than '
                     'the hashes and variable names')
    pathlib.Path(stamp).write_text('\n'.join(sorted(p.name for p in out.glob('*.spv'))) + '\n')


def install(numos, spirv, dest):
    for src, sub, pattern in ((spirv, 'spirv', '*.spv'), (numos / 'assets' / 'nightsky', 'nightsky', '*')):
        dst = dest / 'numos' / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dst / f.name)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, numos = sys.argv[1], pathlib.Path(sys.argv[2])
    if mode == 'build':
        ap = argparse.ArgumentParser(prog='numos_shaders.py build')
        ap.add_argument('slangc', type=pathlib.Path)
        ap.add_argument('out', type=pathlib.Path)
        ap.add_argument('stamp')
        ap.add_argument('--spirv-opt', type=pathlib.Path)
        ap.add_argument('--spirv-val', type=pathlib.Path)
        ap.add_argument('--table', type=pathlib.Path)
        a = ap.parse_args(sys.argv[3:])
        if a.spirv_opt and not (a.spirv_val and a.table):
            sys.exit('--spirv-opt needs --spirv-val and --table')
        build(numos, a.slangc, a.out, a.stamp, a.spirv_opt, a.spirv_val, a.table)
    elif mode == 'install':
        install(numos, pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
    else:
        sys.exit(f'unknown mode {mode}')


if __name__ == '__main__':
    main()
