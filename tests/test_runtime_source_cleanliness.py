from __future__ import annotations

from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "src" / "visual_coding_agent_harness"
TEST_ROOT = Path(__file__).resolve().parents[1] / "tests"

FORBIDDEN_RUNTIME_TERMS = (
    "Austro-Hungarian",
    "Bernini",
    "Borghese",
    "Aeneas",
    "Anchises",
    "Ascanius",
    "Persephone",
    "humble background",
    "entered upper class",
    "born in upper class",
    "seclusion/farmhouse",
    "rise/stability/fall",
    "shows rise, stability, decline, collapse",
)


def test_runtime_source_contains_no_benchmark_semantic_constants() -> None:
    leaks: list[str] = []
    for path in sorted(RUNTIME_ROOT.rglob("*")):
        if path.suffix not in {".py", ".md"}:
            continue
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_RUNTIME_TERMS:
            if term.lower() in text.lower():
                leaks.append(f"{path.relative_to(RUNTIME_ROOT)}: {term}")

    assert leaks == []


def test_test_fixtures_may_contain_benchmark_questions() -> None:
    fixture_text = (TEST_ROOT / "fixtures" / "case_611_2" / "fixture.json").read_text(encoding="utf-8")

    assert any(term in fixture_text for term in ("Aeneas", "Persephone"))
