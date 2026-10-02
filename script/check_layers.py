"""Fail the build if pathquery.core reaches outside its allowance.

The core holds the evidence metrics and the rule that decides whether a
citation resolves to the text it claims. Every number this repository publishes
is computed there, so it has to be readable and testable without a retriever or
a model in the way, and that property decays the first time someone reaches for
a convenience from an outer layer. So it is checked rather than trusted.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CORE = Path("src/pathquery/core")
ALLOWED_THIRD_PARTY = {"numpy", "pydantic"}
STDLIB = sys.stdlib_module_names


def violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside core by construction.
            if node.level > 0:
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            inside_core = name == "pathquery.core" or name.startswith("pathquery.core.")
            if root in STDLIB or root in ALLOWED_THIRD_PARTY or inside_core:
                continue
            found.append(f"{path}:{node.lineno}: core imports {name}")
    return found


def main() -> int:
    failures = [line for path in sorted(CORE.rglob("*.py")) for line in violations(path)]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("layers ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
