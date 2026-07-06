from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


REFUSAL_ANSWERS = {"", "insufficient_verified_evidence", "insufficient verified evidence", "need_more_evidence"}


@dataclass(frozen=True)
class CaseSummary:
    case_id: str
    gold: str
    final_answer: str
    category: str = ""
    correct: bool | None = None
    final_verification: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CaseSummary":
        final = str(payload.get("final_answer") or payload.get("answer") or "")
        gold = str(payload.get("gold") or payload.get("target") or "")
        correct_raw = payload.get("correct")
        correct = bool(correct_raw) if correct_raw is not None else (final == gold if final and gold else None)
        return cls(
            case_id=str(payload.get("case_id") or payload.get("id") or ""),
            gold=gold,
            final_answer=final,
            category=str(payload.get("category") or _category(final, gold, correct)),
            correct=correct,
            final_verification=payload.get("final_verification") if isinstance(payload.get("final_verification"), Mapping) else {},
        )

    @property
    def refused(self) -> bool:
        return self.final_answer.strip().casefold() in REFUSAL_ANSWERS or self.category == "insufficient_verified_evidence"


@dataclass(frozen=True)
class RunSummary:
    cases: tuple[CaseSummary, ...]
    run_dir: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunSummary":
        return cls(
            cases=tuple(CaseSummary.from_mapping(item) for item in payload.get("cases", ()) if isinstance(item, Mapping)),
            run_dir=str(payload.get("run_dir") or ""),
        )

    def to_metrics(self) -> dict[str, Any]:
        total = len(self.cases)
        correct = sum(1 for case in self.cases if case.correct)
        refusals = sum(1 for case in self.cases if case.refused)
        wrong = sum(1 for case in self.cases if case.correct is False and not case.refused)
        answered = max(0, total - refusals)
        return {
            "case_count": total,
            "correct": correct,
            "wrong_answer_count": wrong,
            "refusal_count": refusals,
            "correct_rate": correct / total if total else 0.0,
            "wrong_answer_rate": wrong / total if total else 0.0,
            "refusal_rate": refusals / total if total else 0.0,
            "answered_accuracy": correct / answered if answered else 0.0,
        }


def _category(final_answer: str, gold: str, correct: bool | None) -> str:
    if str(final_answer).strip().casefold() in REFUSAL_ANSWERS:
        return "insufficient_verified_evidence"
    if correct is True:
        return "correct"
    if correct is False:
        return "wrong_answer"
    return "unknown"


def load_summary(path: Path) -> RunSummary:
    return RunSummary.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))
