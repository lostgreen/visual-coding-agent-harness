from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from benchmarks.mmlifelong.schema import BENCHMARK_ID, EvaluationRecord
from evaluate.common.schema import RuntimePrediction
from evaluate.mmlifelong.metrics import ref_scores


UPSTREAM_REVISION = "4244a9f1981ed2d3f3e0cb7f628b60f8b8b59918"
OFFICIAL_JUDGE_MODEL = "gpt-5"
PROMPT_PATH = Path(__file__).with_name("prompts") / "official_answer_judge.txt"
JudgeGenerate = Callable[[str, str, int], str]


@dataclass(frozen=True)
class OfficialAnswerJudgeResult:
    raw_score: int | None
    score: float | None
    judge_model: str
    prompt_sha256: str
    retry_count: int
    parse_status: str
    raw_response: str = ""
    response_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "response_metadata", dict(self.response_metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def official_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_user_prompt(
    *,
    question: str,
    reference_answer: str,
    predicted_answer: str,
) -> str:
    return (
        "Question: {}\n"
        "Groundtruth answer: {}\n"
        "Candidate answer: {}\n"
        "Your response: "
    ).format(question, reference_answer, predicted_answer)


def parse_score(text: str) -> int:
    """Match the vendored eval_acc.py parser, including its fallback."""
    match = re.search(r"Final Score:\s*([0-5])", str(text), re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([0-5])\b", str(text))
    if match:
        return int(match.group(1))
    return -1


def score_mapping(score: int) -> float:
    """Match the vendored eval_acc.py 0-5 to {0, 0.5, 1} mapping."""
    value = int(score)
    if value in (4, 5):
        return 1.0
    if value == 3:
        return 0.5
    return 0.0


def parse_official_judge_response(
    response: str,
    *,
    judge_model: str,
    retry_count: int = 0,
    response_metadata: Mapping[str, Any] | None = None,
) -> OfficialAnswerJudgeResult:
    raw = str(response or "")
    raw_score = parse_score(raw)
    parsed = 0 <= raw_score <= 5
    return OfficialAnswerJudgeResult(
        raw_score=raw_score if parsed else None,
        score=score_mapping(raw_score) if parsed else None,
        judge_model=str(judge_model),
        prompt_sha256=hashlib.sha256(official_system_prompt().encode("utf-8")).hexdigest(),
        retry_count=max(0, int(retry_count)),
        parse_status="parsed" if parsed else "failed",
        raw_response=raw,
        response_metadata=dict(response_metadata or {}),
    )


def judge_free_form_answer(
    generate: JudgeGenerate,
    *,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    judge_model: str,
    max_retries: int = 2,
    max_completion_tokens: int = 4096,
    response_metadata: Callable[[], Mapping[str, Any]] | None = None,
) -> OfficialAnswerJudgeResult:
    system_prompt = official_system_prompt()
    user_prompt = build_user_prompt(
        question=question,
        reference_answer=reference_answer,
        predicted_answer=predicted_answer,
    )
    last = parse_official_judge_response("", judge_model=judge_model)
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            raw = generate(
                system_prompt,
                user_prompt,
                max(4096, int(max_completion_tokens)),
            )
            metadata = dict(response_metadata() if response_metadata else {})
        except Exception as exc:
            raw = ""
            metadata = {"error_type": type(exc).__name__}
        last = parse_official_judge_response(
            raw,
            judge_model=judge_model,
            retry_count=attempt,
            response_metadata=metadata,
        )
        if last.parse_status == "parsed":
            return last
    return last


def evaluate_prediction(
    prediction: RuntimePrediction,
    record: EvaluationRecord,
    *,
    total_seconds: float,
    generate: JudgeGenerate,
    judge_model: str,
    judge_temperature: float | None = 0.0,
    max_retries: int = 2,
    max_completion_tokens: int = 4096,
    response_metadata: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], OfficialAnswerJudgeResult]:
    if prediction.case_id != record.case_id:
        raise ValueError(
            f"prediction case_id {prediction.case_id!r} does not match evaluation record {record.case_id!r}"
        )
    question = str(record.evaluation_metadata.get("question", "") or "")
    if not question:
        raise ValueError("MM-Lifelong evaluation record is missing the question")
    judged = judge_free_form_answer(
        generate,
        question=question,
        reference_answer=record.reference_answer,
        predicted_answer=prediction.answer,
        judge_model=judge_model,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        response_metadata=response_metadata,
    )
    model_match = str(judge_model).strip().casefold() == OFFICIAL_JUDGE_MODEL
    temperature_match = judge_temperature in {0, 0.0}
    evaluation = {
        "schema_version": "MMLifelongEvaluationV1",
        "benchmark": BENCHMARK_ID,
        "case_id": prediction.case_id,
        "answer": {
            "prediction": prediction.answer,
            "reference": record.reference_answer,
            "raw_score": judged.raw_score,
            "score": judged.score,
            "judge_model": str(judge_model),
            "official_protocol": True,
            "official_judge_model_match": model_match,
            "official_judge_config_match": model_match and temperature_match,
            "prompt_sha256": judged.prompt_sha256,
            "upstream_revision": UPSTREAM_REVISION,
            "parse_status": judged.parse_status,
            "retry_count": judged.retry_count,
            "raw_judge_response": judged.raw_response,
            "judge_response_metadata": dict(judged.response_metadata),
        },
        "reference_grounding": ref_scores(
            prediction.supporting_intervals,
            record.clue_intervals,
            total_seconds=float(total_seconds),
        ),
        "diagnostics": {
            "diagnostic_exact_match": _normalized_text(prediction.answer)
            == _normalized_text(record.reference_answer),
        },
    }
    evaluation["reference_grounding"]["reference_intervals"] = [
        list(interval) for interval in record.clue_intervals
    ]
    return evaluation, judged


def _normalized_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())
