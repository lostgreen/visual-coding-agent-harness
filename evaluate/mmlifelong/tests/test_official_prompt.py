from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor" / "upstream"
PROMPT = ROOT / "prompts" / "official_answer_judge.txt"


def test_vendored_files_and_prompt_match_pinned_digests() -> None:
    provenance = json.loads((UPSTREAM / "UPSTREAM.json").read_text(encoding="utf-8"))
    for name in ("eval_acc.py", "eval_ref.py"):
        digest = hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
        assert digest == provenance["files"][name]["sha256"]

    metadata = json.loads(
        (ROOT / "prompts" / "official_answer_judge.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(PROMPT.read_bytes()).hexdigest() == metadata["sha256"]


def test_official_prompt_is_the_verbatim_upstream_system_prompt_value() -> None:
    tree = ast.parse((UPSTREAM / "eval_acc.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "system_prompt" for target in node.targets)
    )
    upstream_prompt = ast.literal_eval(assignment.value)

    assert PROMPT.read_text(encoding="utf-8") == upstream_prompt
