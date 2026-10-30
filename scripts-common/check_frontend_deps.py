"""Frontends rule 1: the backend never includes frontend code.

Fails (exit 1) when a source file under one of the backend directories includes a path containing "frontends/".
Run by src/frontends/meson.build at configure time. Usage: check_frontend_deps.py <source root>
"""
import os
import re
import sys

BACKEND_DIRS = ["src/dxvk", "src/util", "src/vulkan", "src/spirv", "src/lssusd"]
SOURCE_EXTS = (".h", ".hpp", ".cpp", ".c", ".inl", ".slang", ".slangh")
INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    violations = []
    for rel in BACKEND_DIRS:
        base = os.path.join(root, rel)
        for dirpath, _, filenames in os.walk(base):
            for name in filenames:
                if not name.endswith(SOURCE_EXTS):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        for lineno, line in enumerate(f, 1):
                            m = INCLUDE_RE.match(line)
                            if m and "frontends/" in m.group(1).replace("\\", "/"):
                                violations.append(f"{os.path.relpath(path, root)}:{lineno}: {m.group(1)}")
                except OSError:
                    continue
    if violations:
        print("Backend files include frontend code:")
        for v in violations:
            print("  " + v)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
