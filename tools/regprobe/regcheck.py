"""regcheck: register report of the build's hot compute shaders as a build check (Phase 5 F2).

Input is the build's own SPIR-V (<build dir>/src/dxvk/rtx_shaders/*.spv, exactly what ships), filtered by scope.txt; nothing
is recompiled except the `proxy:` entries of scope.txt (ray tracing bodies compiled as compute through probe.py).

  python tools/regprobe/regcheck.py                    probe (NVIDIA GPU + regprobe.exe), compare with baseline.tsv
  python tools/regprobe/regcheck.py --update-baseline  rewrite baseline.tsv from this build (commit it separately)
  python tools/regprobe/regcheck.py --commit-msg       print the rows that changed against the baseline, commit-message form
  python tools/regprobe/regcheck.py --spirv-only       no GPU: SPIR-V size, OpRayQueryInitializeKHR sites, loops, workgroup;
                                                       report only, against baseline-spirv.tsv (--update-baseline rewrites it)
Options: --build-dir <dir> (default _Comp64Release), --arch ada|blackwell (default ada), --allow-removed, --jobs N (4).
scripts/build.ps1 -RegCheck runs the first form after a successful build and fails the build on REGCHECK FAIL.

Results are cached per sha256(spv) + driver in tools/regprobe/.cache/ (gitignored), so a rerun probes only changed SPIR-V.

Fails (exit 1) with `REGCHECK FAIL <variant> 128 -> 136 regs, 16 -> 12 warps/SM` when a variant drops an occupancy tier, gains
local memory (spills) from 0 or by >= 16 B, or disappears (unless --allow-removed). Warns on more registers within the same
tier and on binary size +10 %; prints TIER UP for gains. A baseline from another driver is compared with a warning; an empty
baseline (none recorded yet) never fails.

AMD (no GPU needed, CPU only):
  python tools/regprobe/regcheck.py --amd                       RDNA2/3/4 report (default gfx1030,gfx1100,gfx1201) against
                                                                baseline-amd.tsv; --update-baseline rewrites it
  python tools/regprobe/regcheck.py --amd gfx1036,gfx1103       other targets (see AMD_TARGETS)
  Options: --amd-strict (exit 1 on AMD FAIL; default is report only), --rga-dir <dir>, --rga-rdna2-dir <dir>.
Compiles the same in-scope compute SPIR-V offline with amdllpc, the LLPC compiler of AMD's Vulkan driver, shipped with Radeon GPU
Analyzer (RGA, MIT, github.com/GPUOpen-Tools/radeon_gpu_analyzer). RGA 2.14+ dropped RDNA2 (gfx103x) from LLPC, so those targets
use an RGA 2.13 copy (default <rga dir>/2.13). VGPRs, SGPRs, LDS, scratch (bytes per lane: spills plus private arrays), wave size
(LLPC picks wave32 or wave64 per shader) and CU/WGP mode come from the PAL metadata of the compiled ELF. waves/SIMD: see
amd_occupancy(). AMD FAIL: waves/SIMD lower, scratch from 0 or >= +16 B, variant removed. Offline LLPC has no app profile, so
the real driver can choose differently (wave size above all); RADV (Linux, ACO) is a different compiler altogether. Cached per
sha256(spv) + target + RGA version.
"""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(HERE, '.cache')
REGPROBE = os.path.join(HERE, 'regprobe.exe')

# Per-SM limits: registers, resident warps, shared memory (bytes, max per SM usable by blocks), resident blocks.
ARCHES = {
    'ada': dict(regs=65536, warps=48, smem=102400, blocks=24),
    'blackwell': dict(regs=65536, warps=48, smem=102400, blocks=32),
}

BASELINE_COLUMNS = ['variant', 'registers', 'local', 'shared', 'binary', 'workgroup', 'warps']
SPIRV_COLUMNS = ['variant', 'model', 'bytes', 'rayQuerySites', 'loops', 'workgroup']
AMD_COLUMNS = ['target:variant', 'vgpr', 'sgpr', 'lds', 'scratch', 'wave', 'workgroup', 'waves']

# AMD targets: name -> (LLPC gfxip, family, 1536 physical VGPRs per SIMD, needs the RGA 2.13 LLPC).
# 1536-VGPR parts (Feature1536VGPRs in LLVM GCNFeatures.td): gfx1100, gfx1101, gfx1151 and all gfx12; the rest have 1024.
AMD_TARGETS = {
    'gfx1030': ('10.3.0', 'RDNA2', False, True),     # RX 6800/6900
    'gfx1031': ('10.3.1', 'RDNA2', False, True),     # RX 6700
    'gfx1032': ('10.3.2', 'RDNA2', False, True),     # RX 6600
    'gfx1034': ('10.3.4', 'RDNA2', False, True),     # RX 6500/6400
    'gfx1035': ('10.3.5', 'RDNA2', False, True),     # Radeon 680M
    'gfx1036': ('10.3.6', 'RDNA2', False, True),     # Radeon 610M (Ryzen 7000/9000 iGPU)
    'gfx1100': ('11.0.0', 'RDNA3', True, False),     # RX 7900
    'gfx1101': ('11.0.1', 'RDNA3', True, False),     # RX 7800/7700
    'gfx1102': ('11.0.2', 'RDNA3', False, False),    # RX 7600
    'gfx1103': ('11.0.3', 'RDNA3', False, False),    # Radeon 780M
    'gfx1150': ('11.5.0', 'RDNA3.5', False, False),  # Radeon 890M
    'gfx1151': ('11.5.1', 'RDNA3.5', True, False),   # Radeon 8060S
    'gfx1152': ('11.5.2', 'RDNA3.5', False, False),
    'gfx1153': ('11.5.3', 'RDNA3.5', False, False),
    'gfx1200': ('12.0.0', 'RDNA4', True, False),     # RX 9060
    'gfx1201': ('12.0.1', 'RDNA4', True, False),     # RX 9070
}
AMD_DEFAULT_TARGETS = 'gfx1030,gfx1100,gfx1201'
AMD_MAX_WAVES_PER_SIMD = 16  # gfx10.3, gfx11, gfx12 (FeatureMaxWavesPerEU16)


# ---- SPIR-V -----------------------------------------------------------------------------------------------------------

def spirv_info(path):
    """Execution model, workgroup size, byte size, OpRayQueryInitializeKHR count and loop count of a SPIR-V module."""
    data = open(path, 'rb').read()
    words = [int.from_bytes(data[i:i + 4], 'little') for i in range(0, len(data) - 3, 4)]
    info = dict(model=None, workgroup=(0, 0, 0), bytes=len(data), rayQuerySites=0, loops=0)
    if len(words) < 5 or words[0] != 0x07230203:
        return info
    constants = {}
    local_size_ids = None
    i = 5
    while i < len(words):
        op, length = words[i] & 0xffff, words[i] >> 16
        if length == 0:
            break
        if op == 15 and info['model'] is None:                       # OpEntryPoint
            info['model'] = words[i + 1]
        elif op == 16 and length >= 6 and words[i + 2] == 17:        # OpExecutionMode LocalSize
            info['workgroup'] = (words[i + 3], words[i + 4], words[i + 5])
        elif op == 331 and length >= 6 and words[i + 2] == 38:       # OpExecutionModeId LocalSizeId
            local_size_ids = (words[i + 3], words[i + 4], words[i + 5])
        elif op == 43 and length >= 4:                               # OpConstant
            constants[words[i + 2]] = words[i + 3]
        elif op == 4473:                                             # OpRayQueryInitializeKHR
            info['rayQuerySites'] += 1
        elif op == 246:                                              # OpLoopMerge
            info['loops'] += 1
        i += length
    if local_size_ids is not None:
        info['workgroup'] = tuple(constants.get(x, 0) for x in local_size_ids)
    return info


def is_compute(info):
    return info['model'] == 5  # GLCompute


def model_name(model):
    return {5: 'compute', 5313: 'raygen', 5314: 'intersection', 5315: 'anyhit', 5316: 'closesthit', 5317: 'miss'}.get(model, str(model))


def wg_string(wg):
    return 'x'.join(str(v) for v in wg)


# ---- scope ------------------------------------------------------------------------------------------------------------

def read_scope(path):
    globs, proxies = [], []
    for raw in open(path, encoding='utf-8'):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('proxy:'):
            parts = line[len('proxy:'):].split()
            if len(parts) >= 2:
                proxies.append((parts[0], parts[1], parts[2:]))
            continue
        globs.append(line)
    return globs, proxies


def collect(build_dir, globs):
    shader_dir = os.path.join(build_dir, 'src', 'dxvk', 'rtx_shaders')
    if not os.path.isdir(shader_dir):
        sys.exit(f'regcheck: no shader directory {shader_dir} (build first)')
    found = {}
    for name in sorted(os.listdir(shader_dir)):
        if not name.endswith('.spv'):
            continue
        variant = name[:-4]
        if any(fnmatch.fnmatchcase(variant, g) for g in globs):
            found[variant] = os.path.join(shader_dir, name)
    return found


# ---- probing ----------------------------------------------------------------------------------------------------------

STAT_RE = re.compile(r'([A-Za-z][A-Za-z ]*?)=(\S+)')


def parse_probe_line(line):
    stats = {}
    for name, value in STAT_RE.findall(line):
        stats[name.strip()] = value
    return stats


def driver_version():
    if not os.path.exists(REGPROBE):
        sys.exit(f'regcheck: {REGPROBE} not found; build it with tools/regprobe/build.bat (or use --spirv-only)')
    r = subprocess.run([REGPROBE, '--driver'], capture_output=True, text=True)
    m = re.search(r'driver=(\S+) device=(.*)', r.stdout)
    if r.returncode != 0 or m is None:
        sys.exit(f'regcheck: regprobe --driver failed (NVIDIA GPU and driver with pipeline executable properties needed): '
                 f'{(r.stdout + r.stderr).strip()[-500:]}')
    return m.group(1), m.group(2).strip()


def probe(spv, driver):
    digest = hashlib.sha256(open(spv, 'rb').read()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, f'{digest}-{driver}.txt')
    if os.path.exists(cached):
        return open(cached, encoding='utf-8').read().strip()
    r = subprocess.run([REGPROBE, spv], capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ''
    if r.returncode != 0 or not line:
        return 'PROBE FAILED ' + r.stderr.strip()[-500:]
    open(cached, 'w', encoding='utf-8').write(line + '\n')
    return line


def occupancy(regs, shared, workgroup, arch):
    a = ARCHES[arch]
    threads = max(1, workgroup[0] * workgroup[1] * workgroup[2])
    wg_warps = max(1, math.ceil(threads / 32))
    alloc = max(8, math.ceil(max(regs, 1) / 8) * 8)
    warps_by_regs = a['regs'] // (alloc * 32)
    blocks = min(warps_by_regs // wg_warps, a['warps'] // wg_warps, a['blocks'])
    if shared > 0:
        blocks = min(blocks, a['smem'] // shared)
    return blocks * wg_warps


# ---- AMD (offline LLPC from Radeon GPU Analyzer) ------------------------------------------------------------------------

AMD_STAT_RE = re.compile(r'^\s*\.([a-z_]+):\s+(\S+)\s*$')


def rga_tools(rga_dir):
    """(amdllpc.exe, llvm-readobj.exe, RGA version) of one RGA directory, or None when it is missing."""
    llpc = os.path.join(rga_dir, 'utils', 'amdllpc.exe')
    readobj = os.path.join(rga_dir, 'utils', 'lc', 'opencl', 'bin', 'llvm-readobj.exe')
    rga = os.path.join(rga_dir, 'rga.exe')
    if not (os.path.exists(llpc) and os.path.exists(readobj) and os.path.exists(rga)):
        return None
    r = subprocess.run([rga, '--version'], capture_output=True, text=True)
    m = re.search(r'Version:\s*(\S+)', r.stdout)
    return llpc, readobj, m.group(1) if m else 'unknown'


def parse_pal_cs(notes):
    """Hardware CS stage stats (vgpr_count, sgpr_count, lds_size, scratch_memory_size, wavefront_size, wgp_mode) from the
    `llvm-readobj --notes` dump of an amdpal compute pipeline ELF."""
    stats, indent = {}, None
    for line in notes.splitlines():
        if indent is None:
            if line.strip() == '.cs:':
                indent = len(line) - len(line.lstrip())
            continue
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        m = AMD_STAT_RE.match(line)
        if m:
            stats[m.group(1)] = m.group(2)
    return stats


def amd_probe(spv, target, tools):
    """Compile one SPIR-V file for one AMD target with amdllpc; returns 'vgpr=.. sgpr=.. lds=.. scratch=.. wave=.. wgp=..' or
    'PROBE FAILED ...'. The file is read once and the compile runs on a private copy, so a concurrent build cannot mix
    versions."""
    llpc, readobj, version = tools
    data = open(spv, 'rb').read()
    digest = hashlib.sha256(data).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, f'amd-{digest}-{target}-{version}.txt')
    if os.path.exists(cached):
        return open(cached, encoding='utf-8').read().strip(), digest
    with tempfile.TemporaryDirectory(prefix='regcheck-amd-') as tmp:
        src, elf = os.path.join(tmp, 'in.spv'), os.path.join(tmp, 'out.elf')
        open(src, 'wb').write(data)
        r = subprocess.run([llpc, '--auto-layout-desc', f'--gfxip={AMD_TARGETS[target][0]}', f'-o={elf}', src],
                           capture_output=True, text=True, cwd=tmp)
        if r.returncode != 0 or not os.path.exists(elf):
            return 'PROBE FAILED ' + (r.stdout + r.stderr).strip()[-500:], digest
        n = subprocess.run([readobj, '--elf-output-style=GNU', '--notes', elf], capture_output=True, text=True)
    s = parse_pal_cs(n.stdout)
    if 'vgpr_count' not in s or 'wavefront_size' not in s:
        return 'PROBE FAILED no PAL compute metadata in the ELF', digest
    line = (f'vgpr={s["vgpr_count"]} sgpr={s.get("sgpr_count", 0)} lds={s.get("lds_size", 0)} '
            f'scratch={s.get("scratch_memory_size", 0)} wave={s["wavefront_size"]} wgp={s.get("wgp_mode", "false")}')
    open(cached, 'w', encoding='utf-8').write(line + '\n')
    return line, digest


def amd_occupancy(vgpr, lds, wave, wgp, workgroup, target):
    """Waves per SIMD of a compute shader on an RDNA2/3/4 target, and the limiter ('vgpr', 'lds' or 'max').

    Same rules as LLVM AMDGPU (AMDGPUBaseInfo.cpp getNumWavesPerEUWithNumVGPRs, GCNFeatures.td) and AMD's RDNA ISA guides:
      VGPR file per SIMD     1024 wave32 VGPRs = 512 wave64 VGPRs; 1536 = 768 on gfx1100/1101/1151/gfx12
      allocation granule     16 (wave32) / 8 (wave64); 24 / 12 on the 1536-VGPR parts
      waves per SIMD         at most 16 (gfx10.3, gfx11, gfx12)
      waves(VGPR)            min(16, VGPR file / align(vgpr, granule))
      LDS                    64 KB per CU in CU mode (2 SIMDs), 128 KB per WGP in WGP mode (4 SIMDs); granule 512 B
      workgroups per CU/WGP  min(floor(waves(VGPR) * SIMDs / waves per workgroup), floor(LDS / align(lds, 512)))
      waves per SIMD         min(waves(VGPR), floor(workgroups * waves per workgroup / SIMDs))
    SGPRs never limit occupancy on RDNA (every wave gets its 106); the per-CU workgroup slot limit and barrier resources are not
    modelled."""
    full = AMD_TARGETS[target][2]
    vgpr_file = (1536 if full else 1024) * 32 // wave
    granule = (24 if full else 16) * 32 // wave
    if vgpr < granule:
        by_vgpr = AMD_MAX_WAVES_PER_SIMD
    else:
        by_vgpr = min(AMD_MAX_WAVES_PER_SIMD, max(1, vgpr_file // (math.ceil(vgpr / granule) * granule)))
    simds, lds_total = (4, 131072) if wgp else (2, 65536)
    threads = max(1, workgroup[0] * workgroup[1] * workgroup[2])
    wg_waves = max(1, math.ceil(threads / wave))
    groups = max(1, by_vgpr * simds // wg_waves)
    limiter = 'vgpr' if by_vgpr < AMD_MAX_WAVES_PER_SIMD else 'max'
    if lds > 0:
        by_lds = lds_total // (math.ceil(lds / 512) * 512)
        if by_lds < groups:
            groups, limiter = by_lds, 'lds'
    waves = min(by_vgpr, max(1, groups * wg_waves // simds)) if groups > 0 else 0
    if waves < by_vgpr and limiter != 'lds':
        limiter = 'wg'  # whole workgroups do not fill the VGPR budget
    return waves, limiter


# ---- baselines --------------------------------------------------------------------------------------------------------

def read_tsv(path):
    header, rows = {}, {}
    if not os.path.exists(path):
        return header, rows
    columns = None
    for raw in open(path, encoding='utf-8'):
        line = raw.rstrip('\r\n')
        if not line:
            continue
        if line.startswith('#'):
            for key, value in re.findall(r'(\w+)=(\S+)', line):
                header[key] = value
            continue
        fields = line.split('\t')
        if columns is None:
            columns = fields
            continue
        row = dict(zip(columns, fields))
        rows[row[columns[0]]] = row
    return header, rows


def write_tsv(path, header_line, columns, rows):
    with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(header_line + '\n')
        f.write('\t'.join(columns) + '\n')
        for variant in sorted(rows):
            f.write('\t'.join(str(rows[variant][c]) for c in columns) + '\n')


def git_commit():
    r = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() or 'unknown'


# ---- modes ------------------------------------------------------------------------------------------------------------

def run_spirv_only(args, variants):
    rows = {}
    for variant, path in variants.items():
        info = spirv_info(path)
        rows[variant] = dict(variant=variant, model=model_name(info['model']), bytes=info['bytes'],
                             rayQuerySites=info['rayQuerySites'], loops=info['loops'], workgroup=wg_string(info['workgroup']))
    baseline_path = os.path.join(HERE, 'baseline-spirv.tsv')
    if args.update_baseline:
        write_tsv(baseline_path, f'# regcheck --spirv-only baseline commit={git_commit()} variants={len(rows)}', SPIRV_COLUMNS, rows)
        print(f'regcheck: wrote {baseline_path} ({len(rows)} variants)')
        return 0
    _, base = read_tsv(baseline_path)
    changed = 0
    for variant in sorted(rows):
        row = rows[variant]
        old = base.get(variant)
        note = ''
        if old is None:
            note = 'NEW'
        else:
            diffs = [f'{c} {old[c]} -> {row[c]}' for c in ('bytes', 'rayQuerySites', 'loops', 'workgroup') if str(old[c]) != str(row[c])]
            note = ', '.join(diffs)
        if note or args.verbose:
            changed += 1 if note else 0
            print(f'{variant:70s} {row["model"]:10s} {row["bytes"]:>9} B  rq={row["rayQuerySites"]:<3} loops={row["loops"]:<4} '
                  f'wg={row["workgroup"]:<8} {note}')
    for variant in sorted(set(base) - set(rows)):
        print(f'{variant:70s} REMOVED')
    print(f'regcheck --spirv-only: {len(rows)} variants in scope, {changed} changed against baseline-spirv.tsv (report only)')
    return 0


def run_amd(args, variants, proxies):
    targets = [t.strip() for t in args.amd.split(',') if t.strip()]
    unknown = [t for t in targets if t not in AMD_TARGETS]
    if unknown:
        sys.exit(f'regcheck: unknown AMD target(s) {", ".join(unknown)}; known: {", ".join(AMD_TARGETS)}')
    tools = {}
    for legacy, rga_dir in ((False, args.rga_dir), (True, args.rga_rdna2_dir)):
        if any(AMD_TARGETS[t][3] == legacy for t in targets):
            tools[legacy] = rga_tools(rga_dir)
            if tools[legacy] is None:
                sys.exit(f'regcheck: Radeon GPU Analyzer not found in {rga_dir} (rga.exe, utils/amdllpc.exe); get the Windows zip '
                         f'from github.com/GPUOpen-Tools/radeon_gpu_analyzer/releases'
                         + (' (RDNA2 needs release 2.13, the last one whose LLPC has gfx10.3)' if legacy else ''))
    if proxies:
        sys.path.insert(0, HERE)
        import probe as probe_module
        for tag, shader, defines in proxies:
            spv, err = probe_module.compile_variant(REPO, shader, tag, defines, os.path.join(HERE, 'out'))
            if spv is None:
                print(f'AMD REGCHECK FAIL {tag} proxy compile failed: {err[-300:]}')
                return 1
            variants['proxy:' + tag] = spv

    infos = {v: spirv_info(p) for v, p in variants.items()}
    compute = [v for v in sorted(variants) if is_compute(infos[v])]
    jobs = [(v, t) for t in targets for v in compute]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for job, res in zip(jobs, ex.map(lambda j: amd_probe(variants[j[0]], j[1], tools[AMD_TARGETS[j[1]][3]]), jobs)):
            results[job] = res
    changed = sorted({v for (v, _), (_, digest) in results.items()
                      if hashlib.sha256(open(variants[v], 'rb').read()).hexdigest() != digest})
    versions = ' '.join(f'rga{"_rdna2" if legacy else ""}={tools[legacy][2]}' for legacy in sorted(tools))

    rows, failures = {}, []
    for target in targets:
        for variant in compute:
            line, _ = results[(variant, target)]
            if line.startswith('PROBE FAILED'):
                failures.append(f'AMD REGCHECK FAIL {target} {variant} {line}')
                continue
            s = dict(kv.split('=', 1) for kv in line.split())
            vgpr, lds, wave = int(s['vgpr']), int(s['lds']), int(s['wave'])
            wg = infos[variant]['workgroup']
            waves, limiter = amd_occupancy(vgpr, lds, wave, s['wgp'] == 'true', wg, target)
            key = f'{target}:{variant}'
            rows[key] = {'target:variant': key, 'vgpr': vgpr, 'sgpr': int(s['sgpr']), 'lds': lds, 'scratch': int(s['scratch']),
                         'wave': wave, 'workgroup': wg_string(wg), 'waves': waves, 'limiter': limiter}

    baseline_path = os.path.join(HERE, 'baseline-amd.tsv')
    if args.update_baseline:
        if changed:
            print(f'regcheck --amd: {len(changed)} SPIR-V file(s) changed during the run ({changed[0]} ...); not writing the '
                  f'baseline, rerun')
            return 1
        write_tsv(baseline_path, f'# regcheck --amd baseline commit={git_commit()} {versions} targets={",".join(targets)}',
                  AMD_COLUMNS, rows)
        print(f'regcheck: wrote {baseline_path} ({len(rows)} rows, {len(compute)} compute variants x {len(targets)} targets)')
        for f in failures:
            print(f)
        return 1 if failures else 0

    header, base = read_tsv(baseline_path)
    messages = []
    for key in sorted(rows):
        r = rows[key]
        old = base.get(key)
        if not base:
            continue
        if old is None:
            messages.append(('NEW', f'{key}: {r["vgpr"]} VGPRs, {r["waves"]} waves/SIMD (new)'))
            continue
        old_vgpr, old_waves, old_scratch = int(old['vgpr']), int(old['waves']), int(old['scratch'])
        tail = f'(wave{r["wave"]}' + (f', was wave{old["wave"]}' if str(old['wave']) != str(r['wave']) else '') + ')'
        if r['waves'] < old_waves:
            failures.append(f'AMD REGCHECK FAIL {key} {old_vgpr} -> {r["vgpr"]} VGPRs, {old_waves} -> {r["waves"]} waves/SIMD {tail}')
        elif r['waves'] > old_waves:
            messages.append(('TIER UP', f'AMD TIER UP {key} {old_vgpr} -> {r["vgpr"]} VGPRs, {old_waves} -> {r["waves"]} waves/SIMD {tail}'))
        elif r['vgpr'] > old_vgpr:
            messages.append(('WARN', f'AMD WARNING {key} {old_vgpr} -> {r["vgpr"]} VGPRs (same tier, {r["waves"]} waves/SIMD) {tail}'))
        elif r['vgpr'] < old_vgpr:
            messages.append(('LESS', f'AMD {key} {old_vgpr} -> {r["vgpr"]} VGPRs (same tier) {tail}'))
        if (old_scratch == 0 and r['scratch'] > 0) or r['scratch'] - old_scratch >= 16:
            failures.append(f'AMD REGCHECK FAIL {key} scratch {old_scratch} -> {r["scratch"]} B/lane (spills)')
    if base:
        compared = {k for k in base if k.split(':', 1)[0] in targets}
        for key in sorted(compared - set(rows)):
            if args.allow_removed:
                messages.append(('REMOVED', f'AMD {key}: removed (allowed)'))
            else:
                failures.append(f'AMD REGCHECK FAIL {key} removed (use --allow-removed)')

    if args.commit_msg:
        print(f'regcheck --amd ({versions}):')
        for _, text in messages:
            print('  ' + text)
        for f in failures:
            print('  ' + f)
        if not messages and not failures:
            print('  no AMD register changes')
    else:
        if not base or args.verbose:
            print(f'{"variant":64s} {"target":8s} {"VGPR":>4} {"SGPR":>4} {"LDS":>6} {"scratch":>7} {"wave":>4} {"waves/SIMD":>10}')
            for target in targets:
                for variant in compute:
                    r = rows.get(f'{target}:{variant}')
                    if r:
                        print(f'{variant:64s} {target:8s} {r["vgpr"]:>4} {r["sgpr"]:>4} {r["lds"]:>6} {r["scratch"]:>7} '
                              f'{r["wave"]:>4} {r["waves"]:>4} ({r["limiter"]})')
        for _, text in messages:
            print(text)
        for f in failures:
            print(f)
        if not base:
            print('regcheck --amd: no baseline-amd.tsv recorded yet (run with --amd --update-baseline); report only')
        print(f'regcheck --amd: {len(compute)} compute variants x {len(targets)} targets, {len(failures)} failures '
              f'({versions}; {"strict" if args.amd_strict else "report only"})')
    if changed:
        print(f'regcheck --amd WARNING: {len(changed)} SPIR-V file(s) changed during the run ({changed[0]} ...); rerun')
    return 1 if (failures and args.amd_strict) else 0


def run_probe(args, variants, proxies):
    driver, device = driver_version()
    # Proxies: ray tracing bodies compiled as compute.
    if proxies:
        sys.path.insert(0, HERE)
        import probe as probe_module
        for tag, shader, defines in proxies:
            spv, err = probe_module.compile_variant(REPO, shader, tag, defines, os.path.join(HERE, 'out'))
            if spv is None:
                print(f'REGCHECK FAIL {tag} proxy compile failed: {err[-300:]}')
                return 1
            variants['proxy:' + tag] = spv

    infos = {v: spirv_info(p) for v, p in variants.items()}
    compute = [v for v in variants if is_compute(infos[v])]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for variant, line in zip(compute, ex.map(lambda v: probe(variants[v], driver), compute)):
            results[variant] = line

    rows = {}
    failures = []
    for variant in sorted(variants):
        info = infos[variant]
        if not is_compute(info):
            print(f'{variant:70s} RT: n/a ({model_name(info["model"])})')
            continue
        line = results[variant]
        if line.startswith('PROBE FAILED'):
            failures.append(f'REGCHECK FAIL {variant} {line}')
            continue
        stats = parse_probe_line(line)
        regs = int(stats.get('Register Count', 0))
        local = int(stats.get('Local Memory Size', 0))
        shared = int(stats.get('Shared Memory Size', 0))
        binary = int(stats.get('Binary Size', 0))
        wg = info['workgroup']
        rows[variant] = dict(variant=variant, registers=regs, local=local, shared=shared, binary=binary,
                             workgroup=wg_string(wg), warps=occupancy(regs, shared, wg, args.arch))

    baseline_path = os.path.join(HERE, 'baseline.tsv')
    if args.update_baseline:
        write_tsv(baseline_path, f'# regcheck baseline gpu={device.replace(" ", "_")} driver={driver} commit={git_commit()} arch={args.arch}',
                  BASELINE_COLUMNS, rows)
        print(f'regcheck: wrote {baseline_path} ({len(rows)} compute variants, driver {driver})')
        for f in failures:
            print(f)
        return 1 if failures else 0

    header, base = read_tsv(baseline_path)
    if not base:
        for variant in sorted(rows):
            r = rows[variant]
            print(f'{variant:70s} {r["registers"]:>4} regs  {r["warps"]:>2} warps/SM  local={r["local"]} shared={r["shared"]} '
                  f'binary={r["binary"]} wg={r["workgroup"]}')
        print('regcheck: no baseline recorded yet (run with --update-baseline); report only')
        for f in failures:
            print(f)
        return 1 if failures else 0
    if header.get('driver') and header['driver'] != driver:
        print(f'regcheck WARNING: baseline driver {header["driver"]}, this driver {driver}: compared anyway, re-baseline in a '
              f'separate commit')

    messages = []
    for variant in sorted(rows):
        r = rows[variant]
        old = base.get(variant)
        if old is None:
            messages.append(('NEW', f'{variant}: {r["registers"]} regs, {r["warps"]} warps/SM (new)'))
            continue
        old_regs, old_warps = int(old['registers']), int(old['warps'])
        old_local, old_binary = int(old['local']), int(old['binary'])
        if r['warps'] < old_warps:
            failures.append(f'REGCHECK FAIL {variant} {old_regs} -> {r["registers"]} regs, {old_warps} -> {r["warps"]} warps/SM')
        elif r['warps'] > old_warps:
            messages.append(('TIER UP', f'TIER UP {variant} {old_regs} -> {r["registers"]} regs, {old_warps} -> {r["warps"]} warps/SM'))
        elif r['registers'] > old_regs:
            messages.append(('WARN', f'regcheck WARNING {variant} {old_regs} -> {r["registers"]} regs (same tier, {r["warps"]} warps/SM)'))
        elif r['registers'] < old_regs:
            messages.append(('LESS', f'{variant} {old_regs} -> {r["registers"]} regs (same tier)'))
        if (old_local == 0 and r['local'] > 0) or r['local'] - old_local >= 16:
            failures.append(f'REGCHECK FAIL {variant} local memory {old_local} -> {r["local"]} B (spills)')
        if old_binary > 0 and r['binary'] > old_binary * 1.10:
            messages.append(('WARN', f'regcheck WARNING {variant} binary {old_binary} -> {r["binary"]} B (+{100.0 * (r["binary"] / old_binary - 1):.0f} %)'))
    for variant in sorted(set(base) - set(rows)):
        if args.allow_removed:
            messages.append(('REMOVED', f'{variant}: removed (allowed)'))
        else:
            failures.append(f'REGCHECK FAIL {variant} removed (use --allow-removed)')

    if args.commit_msg:
        print('regcheck (driver ' + driver + ', ' + args.arch + '):')
        for kind, text in messages:
            if kind in ('TIER UP', 'WARN', 'LESS', 'NEW', 'REMOVED'):
                print('  ' + text)
        for f in failures:
            print('  ' + f)
        if not messages and not failures:
            print('  no register changes')
    else:
        for _, text in messages:
            print(text)
        for f in failures:
            print(f)
        print(f'regcheck: {len(rows)} compute variants, {len(failures)} failures (driver {driver}, {args.arch})')
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build-dir', default=os.path.join(REPO, '_Comp64Release'))
    parser.add_argument('--scope', default=os.path.join(HERE, 'scope.txt'))
    parser.add_argument('--arch', choices=sorted(ARCHES), default='ada')
    parser.add_argument('--update-baseline', action='store_true')
    parser.add_argument('--commit-msg', action='store_true')
    parser.add_argument('--allow-removed', action='store_true')
    parser.add_argument('--spirv-only', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--amd', nargs='?', const=AMD_DEFAULT_TARGETS, default=None, metavar='gfx1030,gfx1100,...')
    parser.add_argument('--amd-strict', action='store_true')
    rga_default = os.environ.get('REGCHECK_RGA_DIR', os.path.join(os.path.expanduser('~'), 'tools', 'rga'))
    parser.add_argument('--rga-dir', default=rga_default)
    parser.add_argument('--rga-rdna2-dir', default=None)
    args = parser.parse_args()
    if args.rga_rdna2_dir is None:
        args.rga_rdna2_dir = os.environ.get('REGCHECK_RGA_RDNA2_DIR', os.path.join(args.rga_dir, '2.13'))

    build_dir = args.build_dir if os.path.isabs(args.build_dir) else os.path.join(REPO, args.build_dir)
    globs, proxies = read_scope(args.scope)
    variants = collect(build_dir, globs)
    if args.spirv_only:
        return run_spirv_only(args, variants)
    if args.amd is not None:
        return run_amd(args, variants, proxies)
    return run_probe(args, variants, proxies)


if __name__ == '__main__':
    sys.exit(main())
