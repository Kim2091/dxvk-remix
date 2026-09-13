"""Check compiled lean SHARC isolation, shader interfaces and final DLL embedding."""
import argparse
import importlib.util
import re
from pathlib import Path


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
    parser.add_argument('--baseline-dll', type=Path)
    args = parser.parse_args()
    root = args.root
    spec = importlib.util.spec_from_file_location('contracts', root / 'scripts-common/validate_sharc_integration.py')
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    directory = root / '_Comp64Release/src/dxvk/rtx_shaders'
    dis = root / 'external/spirv_tools/spirv-dis.exe'
    val = root / 'external/spirv_tools/spirv-val.exe'
    files = sorted(directory.glob('integrate_indirect_sharc_*_lean.spv'))
    assert len(files) == 14, f'Expected 14 lean variants, found {len(files)}'
    dll = args.dll.read_bytes() if args.dll else None
    signature = None
    for path in files:
        contracts.validate_spirv(path, val)
        text = contracts.disassemble(path, dis)
        name = path.stem.removeprefix('integrate_indirect_sharc_').removesuffix('_lean')
        bindings = set(contracts.binding_variables(text))
        baseline = contracts.disassemble(path.with_stem(path.stem.removesuffix('_lean')), dis)
        assert bindings <= set(contracts.binding_variables(baseline)), f'{name}: new unmanaged descriptors'
        for flag in ('enableRtxdiSampleStealing', 'enableUnorderedResolveInIndirectRays', 'enableIndirectAlphaBlendShadows'):
            assert not uniform_member_used(text, flag), f'{name}: still reads {flag}'
        assert not contracts.uses_wboit_compensation(text), f'{name}: unordered WBOIT resolver remains'
        if name.startswith('update'):
            contracts.check(name, text)
        else:
            stage = 'RayGeneration' if 'trace' in name else ('ClosestHit' if 'closesthit' in name else 'Miss')
            assert re.search(r'OpEntryPoint\s+' + stage + r'KHR\b', text), name
            assert 231 not in bindings and 'Int64Atomics' not in text, f'{name}: query updates cache'
            assert (235 in bindings) == ('stats' in name), f'{name}: statistics mismatch'
            assert not bindings & {51, 170, 171, 172}, f'{name}: foreign reservoir bindings'
            if stage == 'RayGeneration':
                contracts.check_trace(name.removeprefix('query_'), text, stage)
            elif stage == 'ClosestHit':
                assert {230, 232} <= bindings, f'{name}: missing cache lookup'
            current = contracts.payload_signature(text)
            if signature is None:
                signature = current
            assert current == signature, f'{name}: incompatible lean ray payload'
        if dll is not None:
            assert path.read_bytes() in dll, f'{name}: shader absent from linked DLL'
        print(f'PASS {path.name}: ABI, stripped controls, descriptors, outputs' + (', DLL' if dll else ''))

    if args.baseline_dll:
        old = args.baseline_dll.read_bytes()
        ordinary = ['integrate_indirect_rayquery_neeCache', 'integrate_indirect_rayquery_neeCache_wboit',
                    'integrate_indirect_sharc_query_closesthit_no_portals_no_pom_wboit',
                    'integrate_indirect_sharc_update_deferred4_raygen_wboit']
        for stem in ordinary:
            blob = (directory / (stem + '.spv')).read_bytes()
            assert blob in old, f'{stem}: baseline shader differs from pre-lean DLL'
            if dll is not None:
                assert blob in dll, f'{stem}: baseline shader absent from final DLL'
            print(f'PASS baseline {stem}: byte-identical SPIR-V retained')


if __name__ == '__main__':
    main()
