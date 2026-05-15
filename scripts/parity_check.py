#!/usr/bin/env python3
"""Verify both SDKs implement every method declared in surface.yaml.

Run from the sdk/ directory:
    python scripts/parity_check.py

Exit code 0 means parity holds. Non-zero lists every gap.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
SURFACE = ROOT / "surface.yaml"
PY_INIT = ROOT / "python" / "mathexec" / "__init__.py"
PY_CLIENT = ROOT / "python" / "mathexec" / "client.py"
JS_INDEX = ROOT / "js" / "src" / "index.ts"


def load_surface() -> dict:
    with SURFACE.open() as f:
        return yaml.safe_load(f)


def py_has_top_level(name: str, init_src: str) -> bool:
    # Must appear in __all__ AND be importable from client.
    return (
        re.search(rf'["\']{re.escape(name)}["\']', init_src) is not None
        and re.search(rf"\b{re.escape(name)}\b", init_src) is not None
    )


def py_has_member(name: str, client_src: str) -> bool:
    # Match `def name(` or `name =` or `@property\n    def name(`.
    pattern = rf"(\bdef\s+{re.escape(name)}\s*\(|\b{re.escape(name)}\s*=)"
    return re.search(pattern, client_src) is not None


def js_has_export(name: str, src: str) -> bool:
    # Match `export class X`, `export function X`, `export interface X`, `export const X`, `export async function X`.
    pattern = (
        rf"export\s+(?:async\s+)?(?:class|function|interface|const|type|enum)\s+{re.escape(name)}\b"
    )
    return re.search(pattern, src) is not None


def js_has_member(name: str, src: str) -> bool:
    # Match `name(` or `async name(` or `static async name(` or `static name(` or `get name(`.
    pattern = rf"(?:^|\n)\s+(?:static\s+)?(?:async\s+)?(?:get\s+)?{re.escape(name)}\s*[(<]"
    return re.search(pattern, src) is not None


def main() -> int:
    surface = load_surface()
    init_src = PY_INIT.read_text()
    client_src = PY_CLIENT.read_text()
    js_src = JS_INDEX.read_text()

    failures: list[str] = []

    for entry in surface.get("top_level", []):
        py_name = entry["python"]
        js_name = entry["js"]
        if not py_has_top_level(py_name, init_src):
            failures.append(
                f"[top_level/{entry['id']}] Python export `{py_name}` missing from python/mathexec/__init__.py"
            )
        if not js_has_export(js_name, js_src):
            failures.append(
                f"[top_level/{entry['id']}] JS export `{js_name}` missing from js/src/index.ts"
            )

    for entry in surface.get("model_class", []):
        py_name = entry["python"]
        js_name = entry["js"]
        if not py_has_member(py_name, client_src):
            failures.append(
                f"[model_class/{entry['id']}] Python member `{py_name}` missing from python/mathexec/client.py"
            )
        if not js_has_member(js_name, js_src):
            failures.append(
                f"[model_class/{entry['id']}] JS member `{js_name}` missing from js/src/index.ts"
            )

    for entry in surface.get("errors", []):
        expect_py = entry.get("expect_in_python")
        expect_js = entry.get("expect_in_js")
        if expect_py and expect_py not in client_src:
            failures.append(
                f"[errors/{entry['id']}] Python source missing substring {expect_py!r}"
            )
        if expect_js and expect_js not in js_src:
            failures.append(
                f"[errors/{entry['id']}] JS source missing substring {expect_js!r}"
            )

    if failures:
        print("Parity check FAILED:\n", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            f"\n{len(failures)} gap(s). Update surface.yaml or implement the missing pieces.",
            file=sys.stderr,
        )
        return 1

    counts = (
        len(surface.get("top_level", []))
        + len(surface.get("model_class", []))
        + len(surface.get("errors", []))
    )
    print(f"Parity check OK — {counts} contract entries verified across Python and JS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
