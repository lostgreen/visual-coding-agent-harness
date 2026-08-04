from __future__ import annotations

from typing import Any, Mapping, Sequence

from benchmarks.mmlifelong.schema import BENCHMARK_ID, RuntimeQuestion


_EVALUATOR_ONLY_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "gold_clue_intervals",
        "reference_answer",
        "clue_intervals",
        "target_gt",
        "target_virtual_interval",
        "official_clue_intervals",
        "official_accuracy",
        "accuracy_score",
    }
)


def prediction_artifact(
    question: RuntimeQuestion,
    *,
    answer: str,
    selected_option: str,
    supporting_intervals: Sequence[Sequence[float]],
    supporting_attempt_ids: Sequence[str],
    answer_present: bool,
    candidate_answer: str = "",
    verified_answer: str = "",
    verification_status: str = "missing",
    duration_sec: float | None = None,
) -> dict[str, Any]:
    runtime_metadata = dict(question.runtime_metadata)
    if duration_sec is not None:
        runtime_metadata["duration_sec"] = float(duration_sec)
    payload = {
        "schema_version": "RuntimePredictionV1",
        "benchmark": BENCHMARK_ID,
        "case_id": question.case_id,
        "subset": question.subset,
        "split": question.split,
        "answer": str(answer),
        "selected_option": str(selected_option),
        "supporting_intervals": [
            [float(interval[0]), float(interval[1])]
            for interval in supporting_intervals
            if len(interval) == 2
        ],
        "supporting_attempt_ids": list(
            dict.fromkeys(str(item) for item in supporting_attempt_ids if str(item))
        ),
        "answer_present": bool(answer_present),
        "candidate_answer": str(candidate_answer),
        "verified_answer": str(verified_answer),
        "verification_status": str(verification_status),
        "runtime_metadata": runtime_metadata,
    }
    assert_runtime_artifact(payload)
    return payload


def assert_runtime_artifact(value: Any) -> None:
    forbidden = _find_evaluator_only_keys(value)
    if forbidden:
        raise ValueError(
            "runtime artifact contains evaluator-only keys: "
            + ", ".join(sorted(forbidden))
        )


def _find_evaluator_only_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            if key in _EVALUATOR_ONLY_KEYS:
                found.add(str(raw_key))
            found.update(_find_evaluator_only_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_evaluator_only_keys(item))
    return found
