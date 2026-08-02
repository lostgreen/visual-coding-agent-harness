from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (REPOSITORY_ROOT / "src" / "vcah", REPOSITORY_ROOT / "tools")
FORBIDDEN_MARKERS = (
    "mmlifelong-game-test-0038",
    "mmlifelong-game-test-0097",
    "mmlifelong-game-test-0117",
    "mmlifelong-game-test-0182",
    "yin tiger",
    "flaming mountains",
    "tiger's acolyte",
)


def test_runtime_contains_no_mm_lifelong_case_specialization() -> None:
    violations: list[str] = []
    for root in RUNTIME_ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8").casefold()
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    violations.append(f"{path.relative_to(REPOSITORY_ROOT)}:{marker}")

    assert not violations, "case-specific runtime markers found: " + ", ".join(violations)
