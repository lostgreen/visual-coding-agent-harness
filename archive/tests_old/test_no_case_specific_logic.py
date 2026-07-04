from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "src" / "visual_coding_agent_harness"
BLACKLIST_PATH = Path(__file__).with_name("_anti_specialization_blacklist.txt")

SCANNED_SUFFIXES = (".py", ".md", ".txt")
CONTROL_FLOW_IDENTIFIER_NAMES = {"case_id", "video_id", "case"}
RAW_OPTION_LETTERS = {"A", "B", "C", "D"}
EQUALITY_OPS = (ast.Eq, ast.NotEq)


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    rule: str
    detail: str


def test_blacklist_file_is_reviewable_case_insensitive_regex_list() -> None:
    patterns = _load_blacklist_patterns()

    assert [pattern.pattern for pattern in patterns] == [
        "Austro",
        "Hungarian",
        "Bernini",
        "farmhouse",
        "humble background",
        "upper class",
    ]
    assert all(pattern.flags & re.IGNORECASE for pattern in patterns)


def test_runtime_sources_do_not_contain_case_specific_semantic_literals() -> None:
    patterns = _load_blacklist_patterns()
    hits: list[Hit] = []

    for path in _scanned_runtime_paths():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    hits.append(
                        Hit(
                            path=path,
                            line=line_number,
                            rule="case-specific semantic literal",
                            detail=f"/{pattern.pattern}/ matched {_short_excerpt(line)}",
                        )
                    )

    assert hits == [], _format_hits(hits)


def test_runtime_python_control_flow_does_not_branch_on_case_ids_or_raw_option_letters() -> None:
    hits: list[Hit] = []

    for path in _scanned_runtime_paths(suffixes=(".py",)):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = _ControlFlowVisitor(path)
        visitor.visit(tree)
        hits.extend(visitor.hits)

    assert hits == [], _format_hits(hits)


def test_scanned_runtime_paths_skip_macos_metadata_artifacts(tmp_path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "normal.py").write_text("print('ok')\n", encoding="utf-8")
    (runtime_root / "._normal.py").write_bytes(b"\x00\x05binary-ish metadata")

    nested = runtime_root / "pkg"
    nested.mkdir()
    (nested / "._notes.md").write_bytes(b"\x00\x05binary-ish metadata")

    hidden_fork_dir = nested / "._resource"
    hidden_fork_dir.mkdir()
    (hidden_fork_dir / "fork.txt").write_bytes(b"\x00\x05binary-ish metadata")

    apple_double_dir = nested / ".AppleDouble"
    apple_double_dir.mkdir()
    (apple_double_dir / "source.py").write_bytes(b"\x00\x05binary-ish metadata")

    macosx_dir = runtime_root / "__MACOSX"
    macosx_dir.mkdir()
    (macosx_dir / "source.py").write_bytes(b"\x00\x05binary-ish metadata")

    monkeypatch.setitem(globals(), "RUNTIME_ROOT", runtime_root)

    assert _scanned_runtime_paths() == [runtime_root / "normal.py"]


def _load_blacklist_patterns() -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for line_number, raw_line in enumerate(BLACKLIST_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            patterns.append(re.compile(line, flags=re.IGNORECASE))
        except re.error as exc:
            raise AssertionError(f"{BLACKLIST_PATH.name}:{line_number}: invalid blacklist regex {line!r}: {exc}") from exc
    return patterns


def _scanned_runtime_paths(*, suffixes: Iterable[str] = SCANNED_SUFFIXES) -> list[Path]:
    paths: set[Path] = set()
    for suffix in suffixes:
        paths.update(RUNTIME_ROOT.rglob(f"*{suffix}"))
    return sorted(path for path in paths if path.is_file() and not _is_macos_metadata_path(path))


def _is_macos_metadata_path(path: Path) -> bool:
    try:
        parts = path.relative_to(RUNTIME_ROOT).parts
    except ValueError:
        parts = path.parts
    return any(part in {"__MACOSX", ".AppleDouble"} or part.startswith("._") for part in parts)


class _ControlFlowVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hits: list[Hit] = []

    def visit_If(self, node: ast.If) -> None:
        self._inspect_test(node.test, "if condition")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._inspect_test(node.test, "while condition")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._inspect_test(node.test, "conditional expression")
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        for condition in node.ifs:
            self._inspect_test(condition, "comprehension filter")
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        for match_case in node.cases:
            if match_case.guard is not None:
                self._inspect_test(match_case.guard, "match guard")
        self.generic_visit(node)

    def _inspect_test(self, test: ast.AST, context: str) -> None:
        for node in ast.walk(test):
            if isinstance(node, ast.Compare):
                self._inspect_compare(node, context)

    def _inspect_compare(self, node: ast.Compare, context: str) -> None:
        expressions = [node.left, *node.comparators]
        for operator, left, right in zip(node.ops, expressions, expressions[1:]):
            if not isinstance(operator, EQUALITY_OPS):
                continue

            identifier_hits = sorted(
                (_identifier_names(left) | _identifier_names(right)) & CONTROL_FLOW_IDENTIFIER_NAMES
            )
            if identifier_hits:
                self.hits.append(
                    Hit(
                        path=self.path,
                        line=getattr(node, "lineno", 0),
                        rule="case/video identifier equality in control flow",
                        detail=f"{context} compares identifier(s): {', '.join(identifier_hits)}",
                    )
                )

            raw_option = _raw_option_letter(left) or _raw_option_letter(right)
            if raw_option is not None:
                self.hits.append(
                    Hit(
                        path=self.path,
                        line=getattr(node, "lineno", 0),
                        rule="raw option-letter equality in control flow",
                        detail=f"{context} compares directly to option {raw_option!r}; use membership/schema validation",
                    )
                )


def _identifier_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _raw_option_letter(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in RAW_OPTION_LETTERS:
        return node.value
    return None


def _format_hits(hits: list[Hit]) -> str:
    if not hits:
        return ""
    lines = ["Case-specific runtime guardrail violations:"]
    for hit in hits:
        path = hit.path.relative_to(REPO_ROOT)
        lines.append(f"{path}:{hit.line}: {hit.rule}: {hit.detail}")
    return "\n".join(lines)


def _short_excerpt(line: str) -> str:
    collapsed = " ".join(line.strip().split())
    if len(collapsed) <= 120:
        return repr(collapsed)
    return repr(f"{collapsed[:117]}...")
