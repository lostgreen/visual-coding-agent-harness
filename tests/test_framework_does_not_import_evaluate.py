from __future__ import annotations

import ast
from pathlib import Path


def test_framework_does_not_import_evaluation_layer() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "vcah"
    violations: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names = (node.module or "",)
            else:
                continue
            if any(name == "evaluate" or name.startswith("evaluate.") for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert violations == []
