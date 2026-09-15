"""Check compiled lean SHARC isolation, shader interfaces, full-profile parity and final DLL embedding."""
import argparse
import importlib.util
import re
from pathlib import Path

# Tier suffix -> restored feature level (1: alpha-blended indirect shadows, 2: + unordered resolve).
TIERS = {'': 0, '_shadows': 1, '_particles': 2, '_particles_wboit': 2}
LEGACY_BASELINE = ['integrate_indirect_rayquery_neeCache', 'integrate_indirect_rayquery_neeCache_wboit',
                   'integrate_indirect_neeCache_material_opaque_translucent_closestHit',
                   'integrate_indirect_neeCache_material_rayportal_closestHit']
RAYGEN_PARITY = [('query_trace_lean', 'query_trace'), ('query_trace_stats_lean', 'query_trace_stats'),
                 ('query_trace_ser_lean_particles', 'query_trace_ser'),
                 ('query_trace_ser_stats_lean_particles', 'query_trace_ser_stats')]


def uniform_member_used(text, name):
    member = re.search(r'OpMemberName\s+(%\S+)\s+(\d+)\s+"' + re.escape(name) + '"', text)
    if not member:
        return False
    pointers = re.findall(r'(%\S+)\s+=\s+OpTypePointer\s+Uniform\s+' + re.escape(member[1]) + r'\b', text)
    variables = set()
    for pointer in pointers:
        variables.update(re.findall(r'(%\S+)\s+=\s+OpVariable\s+' + re.escape(pointer) + r'\s+Uniform\b', text))
    indices = set(re.findall(r'(%\S+)\s+=\s+OpConstant\s+%\S+\s+' + member[2] + r'\b', text))
    return any(base in variables and index in indices for base, index in re.findall(
        r'OpAccessChain\s+%\S+\s+(%\S+)\s+(%\S+)', text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dll', type=Path)
    parser.add_argument('--baseline-dll', type=Path, help='pre-lean DLL; legacy stages must be byte-identical to it')
    parser.add_argument('--sharc-baseline', type=Path, help='directory of full SHARC blobs that must be byte-identical')
    args = parser.parse_args()
    root = args.root
    spec = importlib.util.spec_from_file_location('contracts', root / 'scripts-common/validate_sharc_integration.py')
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    directory = root / '_Comp64Release/src/dxvk/rtx_shaders'
    dis = root / 'external/spirv_tools/spirv-dis.exe'
    val = root / 'external/spirv_tools/spirv-val.exe'
    dll = args.dll.read_bytes() if args.dll else None
    disassembled = {}

    def text_of(stem):
        if stem not in disassembled:
            disassembled[stem] = contracts.disassemble(directory / (stem + '.spv'), dis)
        return disassembled[stem]

    files = sorted(directory.glob('integrate_indirect_sharc_*_lean*.spv'))
    assert len(files) == 42, f'Expected 42 lean variants, found {len(files)}'
    signature = None
    counts = {}
    for path in files:
        contracts.validate_spirv(path, val)
        text = text_of(path.stem)
        stem = path.stem.removeprefix('integrate_indirect_sharc_')
        match = re.fullmatch(r'(.*)_lean(|_shadows|_particles|_particles_wboit)', stem)
        assert match, f'{stem}: unexpected lean variant name'
        base, tier = match.group(1), match.group(2)
        level = TIERS[tier]
        wboit = tier.endswith('_wboit')
        counts[level] = counts.get(level, 0) + 1
        bindings = set(contracts.binding_variables(text))
        assert not bindings & {10, 51}, f'{stem}: lean pipeline omits a required light/reservoir descriptor'
        full_stem = 'integrate_indirect_sharc_' + base + ('_wboit' if wboit else '')
        full = text_of(full_stem)
        assert bindings <= set(contracts.binding_variables(full)), f'{stem}: new unmanaged descriptors'
        # Level 2 restores every lean-removed feature, so query stages must compile to the full
        # no-portal stage exactly. Update stages keep the opaque/translucent material mask while the
        # full update resolves portal materials too, so they are compared by contract only.
        if level == 2 and not base.startswith('update'):
            assert path.read_bytes() == (directory / (full_stem + '.spv')).read_bytes(), f'{stem}: level 2 differs from {full_stem}'
        assert not uniform_member_used(text, 'enableRtxdiSampleStealing'), f'{stem}: still reads the stealing flag'
        # Raygen stages carry no NEE code (only the SER unordered hint at level 2); their tier is checked by identity below.
        if 'trace' not in base:
            assert uniform_member_used(text, 'enableIndirectAlphaBlendShadows') == (level >= 1), f'{stem}: alpha-blend shadow tier mismatch'
        assert uniform_member_used(text, 'enableUnorderedResolveInIndirectRays') == (level >= 2), f'{stem}: unordered tier mismatch'
        assert contracts.uses_wboit_compensation(text) == wboit, f'{stem}: WBOIT resolver mismatch'
        # Closest-hit stages carry displacement only in the POM variants; miss stages never do. The
        # raygen keeps the same runtime-gated primary POM setup as the full raygen (checked by identity below).
        if 'closesthit' in base:
            assert uniform_member_used(text, 'pomMaxIterations') == ('no_pom' not in base), f'{stem}: displacement mismatch'
        elif 'miss' in base:
            assert not uniform_member_used(text, 'pomMaxIterations'), f'{stem}: unexpected displacement code'
        if base.startswith('update'):
            # The shared contract infers the resolver from the name, so pass the full-equivalent name.
            contracts.check(base + ('_wboit' if wboit else ''), text)
            if level > 0:
                assert 'deferred' in base and 'raygen' in base, f'{stem}: restored tiers exist for ray-generation deferred updates only'
        else:
            stage = 'RayGeneration' if 'trace' in base else ('ClosestHit' if 'closesthit' in base else 'Miss')
            assert re.search(r'OpEntryPoint\s+' + stage + r'KHR\b', text), stem
            assert 231 not in bindings and 'Int64Atomics' not in text, f'{stem}: query updates cache'
            assert (235 in bindings) == ('stats' in base), f'{stem}: statistics mismatch'
            assert not bindings & {51, 170, 171, 172}, f'{stem}: foreign reservoir bindings'
            if stage == 'RayGeneration':
                contracts.check_trace(base.removeprefix('query_'), text, stage)
            elif stage == 'ClosestHit':
                assert {230, 232} <= bindings, f'{stem}: missing cache lookup'
            current = contracts.payload_signature(text)
            if signature is None:
                signature = current
            assert current == signature, f'{stem}: incompatible lean ray payload'
        if dll is not None:
            assert path.read_bytes() in dll, f'{stem}: shader absent from linked DLL'
        print(f'PASS {path.name}: ABI, tier {level}{" WBOIT" if wboit else ""}{", identical to full" if level == 2 and not base.startswith("update") else ""}, descriptors, outputs'
              + (', DLL' if dll else ''))
    assert counts == {0: 16, 1: 8, 2: 18}, f'unexpected tier distribution {counts}'

    # The lean raygen carries no feature code: it must match the full raygen byte for byte, and the
    # level-2 SER raygen (which restores the unordered coherence hint) must match the full SER raygen.
    for lean, full in RAYGEN_PARITY:
        lean_blob = (directory / f'integrate_indirect_sharc_{lean}.spv').read_bytes()
        assert lean_blob == (directory / f'integrate_indirect_sharc_{full}.spv').read_bytes(), f'{lean}: differs from {full}'
        print(f'PASS raygen parity {lean} == {full}')

    # After the stealing guard no SHARC stage reads the stealing flag or binds the reservoir / previous lights.
    # Only variants declared in the current sources count; the build directory keeps stale blobs.
    declared = set()
    for source in ('integrate_indirect.slang', 'integrate_indirect_closesthit.rchit.slang', 'integrate_indirect_miss.rmiss.slang'):
        text = (root / 'src/dxvk/shaders/rtx/pass/integrate' / source).read_text(encoding='utf-8')
        declared.update(re.findall(r'^//!variant\s+(integrate_indirect_sharc_\S+)\.\w+\s*$', text, re.MULTILINE))
    full_stems = sorted(stem for stem in declared if '_lean' not in stem)
    assert full_stems, 'no full SHARC stages found'
    assert all((directory / (stem + '.spv')).exists() for stem in full_stems), 'declared SHARC variant not compiled'
    for stem in full_stems:
        text = text_of(stem)
        assert not set(contracts.binding_variables(text)) & {10, 51}, f'{stem}: full SHARC stage still binds 10/51'
        assert not uniform_member_used(text, 'enableRtxdiSampleStealing'), f'{stem}: full SHARC stage still reads the stealing flag'
        if args.sharc_baseline:
            baseline = args.sharc_baseline / (stem + '.spv')
            assert baseline.exists(), f'{stem}: missing from SHARC baseline'
            assert baseline.read_bytes() == (directory / (stem + '.spv')).read_bytes(), f'{stem}: full SHARC stage changed'
    print(f'PASS {len(full_stems)} full SHARC stages: no stealing flag, no bindings 10/51'
          + (', byte-identical to baseline' if args.sharc_baseline else ''))
    legacy = text_of('integrate_indirect_neeCache_material_opaque_translucent_closestHit')
    assert 51 in set(contracts.binding_variables(legacy)) and uniform_member_used(legacy, 'enableRtxdiSampleStealing'), \
        'legacy TraceRay closest hit lost RTXDI sample stealing'
    print('PASS legacy TraceRay closest hit retains reservoir binding and sample stealing')

    if args.baseline_dll:
        old = args.baseline_dll.read_bytes()
        for stem in LEGACY_BASELINE:
            blob = (directory / (stem + '.spv')).read_bytes()
            assert blob in old, f'{stem}: legacy shader differs from pre-lean DLL'
            if dll is not None:
                assert blob in dll, f'{stem}: legacy shader absent from final DLL'
            print(f'PASS baseline {stem}: byte-identical SPIR-V retained')


if __name__ == '__main__':
    main()
