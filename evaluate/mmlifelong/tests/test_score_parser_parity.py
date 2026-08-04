from __future__ import annotations

import ast
from pathlib import Path
import re

from evaluate.mmlifelong.evaluator import parse_score, score_mapping


def _upstream_functions():
    path = Path(__file__).resolve().parents[1] / "vendor" / "upstream" / "eval_acc.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"parse_score", "score_mapping"}
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parse_score"], namespace["score_mapping"]


def test_parser_and_score_mapping_match_upstream() -> None:
    upstream_parse, upstream_mapping = _upstream_functions()
    responses = (
        "Analysis:\nEquivalent.\n\nFinal Score:\n5",
        "Analysis: partial\nFinal Score: 3",
        "Final Score: 0",
        "The fallback value is 4 without the required label.",
        "No parseable score.",
    )
    for response in responses:
        assert parse_score(response) == upstream_parse(response)
    for raw_score in range(6):
        assert score_mapping(raw_score) == upstream_mapping(raw_score)
