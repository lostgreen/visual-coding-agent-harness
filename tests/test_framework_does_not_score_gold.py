from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_multiround_never_reads_gold_or_scores_correctness() -> None:
    source = (ROOT / "src" / "vcah" / "multiround.py").read_text(encoding="utf-8")

    assert "workspace.case.gold" not in source
    assert "def _score_answer" not in source
    assert "correct=None" in source


def test_mmlifelong_runner_has_no_judge_or_accuracy_path() -> None:
    source = (ROOT / "tools" / "run_mmlifelong_interactive.py").read_text(
        encoding="utf-8"
    )

    assert "judge_free_form_answer" not in source
    assert "OpenAICompatibleJudgeClient" not in source
    assert "--judge" not in source
    assert "accuracy_score" not in source
    assert "mmlifelong_metrics.json" not in source
