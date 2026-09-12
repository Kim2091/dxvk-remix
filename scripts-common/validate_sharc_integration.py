#!/usr/bin/env python3
"""Validate the compiled SHARC integration SPIR-V contracts.

This is deliberately a static ABI check.  It does not execute shaders or
modify the build tree; ``--shader-dir`` is expected to contain the generated
SPIR-V files from a completed build.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def disassemble(path: Path, tool: Path) -> str:
    p = subprocess.run([str(tool), str(path)], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"spirv-dis failed for {path.name}: {p.stderr}")
    return p.stdout


def validate_spirv(path: Path, validator: Path) -> None:
    p = subprocess.run(
        [str(validator), "--target-env", "vulkan1.2", "--scalar-block-layout", str(path)],
        text=True, capture_output=True,
    )
    if p.returncode:
        raise RuntimeError(f"spirv-val failed for {path.name}: {p.stdout}\n{p.stderr}")


def binding_variables(text: str) -> dict[int, str]:
    return {int(binding): ident for ident, binding in re.findall(
        r"OpDecorate\s+(%\S+)\s+Binding\s+(\d+)", text)}


def strides(text: str) -> set[int]:
    return {int(x) for x in re.findall(r"ArrayStride\s+(\d+)", text)}


def check(name: str, text: str) -> None:
    bv = binding_variables(text)
    b = set(bv)
    s = strides(text)
    if name == "update":
        required = {230, 231}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if 232 in b:
            raise RuntimeError(f"{name}: query-only resolved binding 232 is present")
        if 8 not in s or 16 not in s:
            raise RuntimeError(f"{name}: expected hash/accumulation strides 8/16, got {sorted(s)}")
        if "Int64Atomics" not in text or "OpAtomic" not in text:
            raise RuntimeError(f"{name}: expected Int64Atomics capability and atomic operation")
        if "OpImageWrite" in text:
            raise RuntimeError(f"{name}: unexpected image output write")
    elif name == "query":
        required = {230, 232}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if "Int64Atomics" in text:
            raise RuntimeError(f"{name}: query must not perform atomic64 cache updates")
        if 190 not in b or "OpImageWrite" not in text:
            raise RuntimeError(f"{name}: query must write the final indirect image")
    elif name == "resolve":
        required = {230, 231, 232}
        if not required <= b:
            raise RuntimeError(f"{name}: missing SHARC bindings {sorted(required - b)}")
        if not {8, 16} <= s:
            raise RuntimeError(f"{name}: expected SHARC strides 8/16, got {sorted(s)}")
    elif name == "baseline":
        if b & {230, 231, 232}:
            raise RuntimeError(f"{name}: baseline unexpectedly declares SHARC binding")
        if "Int64Atomics" in text:
            raise RuntimeError(f"{name}: baseline unexpectedly requires Int64Atomics")

    # Updates read thread-task image 82 for the incoming PDF. Its RW image
    # declaration need not be NonWritable; absence of OpImageWrite proves isolation.
    # Query retains the ordinary indirect image and NEE feedback outputs.
    forbidden = {83, 170, 171, 172, 190}
    if name == "update" and b & forbidden:
        raise RuntimeError(f"{name}: forbidden output binding(s) present: {sorted(b & forbidden)}")
    if name == "update":
        for slot in (80, 81):
            ident = bv.get(slot)
            if ident and not re.search(rf"OpDecorate\s+{re.escape(ident)}\s+NonWritable", text):
                raise RuntimeError(f"{name}: NEE cache binding {slot} is writable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shader-dir", required=True, type=Path)
    ap.add_argument("--spirv-dis", required=True, type=Path)
    ap.add_argument("--spirv-val", required=True, type=Path)
    args = ap.parse_args()
    files = {
        "update": args.shader_dir / "integrate_indirect_sharc_update.spv",
        "query": args.shader_dir / "integrate_indirect_sharc_query.spv",
        "resolve": args.shader_dir / "sharc_resolve.spv",
        "baseline": args.shader_dir / "integrate_indirect_rayquery_neeCache.spv",
    }
    for name, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name} shader: {path}")
        validate_spirv(path, args.spirv_val)
        check(name, disassemble(path, args.spirv_dis))
        print(f"PASS {name}: {path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
