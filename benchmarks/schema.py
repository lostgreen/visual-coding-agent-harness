from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_RUNTIME_FORBIDDEN_KEYS = frozenset(
    {
        "gold",
        "gold_answer",
        "gold_clue_intervals",
        "reference_answer",
        "clue_intervals",
        "target_gt",
        "target_virtual_interval",
        "official_clue_intervals",
    }
)


@dataclass(frozen=True)
class RuntimeQuestion:
    """Benchmark-neutral question data that may be exposed to the agent runtime."""

    case_id: str
    question: str
    options: Mapping[str, str] = field(default_factory=dict)
    question_type: str | None = None
    subset: str | None = None
    split: str | None = None
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "question", str(self.question))
        object.__setattr__(
            self,
            "options",
            {str(key): str(value) for key, value in dict(self.options).items()},
        )
        metadata = dict(self.runtime_metadata)
        forbidden = _find_forbidden_runtime_keys(metadata)
        if forbidden:
            raise ValueError(
                "runtime_metadata contains evaluator-only keys: "
                + ", ".join(sorted(forbidden))
            )
        object.__setattr__(self, "runtime_metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "RuntimeQuestionV1",
            **asdict(self),
            "options": dict(self.options),
            "runtime_metadata": dict(self.runtime_metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeQuestion":
        forbidden = _find_forbidden_runtime_keys(value)
        if forbidden:
            raise ValueError(
                "runtime question contains evaluator-only keys: "
                + ", ".join(sorted(forbidden))
            )
        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            options=dict(value.get("options", {})),
            question_type=(
                str(value["question_type"])
                if value.get("question_type") is not None
                else None
            ),
            subset=str(value["subset"]) if value.get("subset") is not None else None,
            split=str(value["split"]) if value.get("split") is not None else None,
            runtime_metadata=dict(
                value.get("runtime_metadata", value.get("metadata", {}))
            ),
        )


@dataclass(frozen=True)
class EvaluationRecord:
    """Evaluator-only reference data that must never be passed to the runtime."""

    case_id: str
    reference_answer: str
    clue_intervals: tuple[tuple[float, float], ...] = ()
    evaluation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "reference_answer", str(self.reference_answer))
        object.__setattr__(
            self,
            "clue_intervals",
            tuple((float(start), float(end)) for start, end in self.clue_intervals),
        )
        object.__setattr__(self, "evaluation_metadata", dict(self.evaluation_metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "EvaluationRecordV1",
            "case_id": self.case_id,
            "reference_answer": self.reference_answer,
            "clue_intervals": [list(interval) for interval in self.clue_intervals],
            "evaluation_metadata": dict(self.evaluation_metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(
            case_id=str(value["case_id"]),
            reference_answer=str(
                value.get("reference_answer", value.get("gold_answer", value.get("gold", "")))
            ),
            clue_intervals=tuple(
                tuple(interval)
                for interval in value.get(
                    "clue_intervals",
                    value.get("gold_clue_intervals", ()),
                )
            ),
            evaluation_metadata=dict(
                value.get("evaluation_metadata", value.get("metadata", {}))
            ),
        )


def _find_forbidden_runtime_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            if key in _RUNTIME_FORBIDDEN_KEYS:
                found.add(str(raw_key))
            found.update(_find_forbidden_runtime_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_find_forbidden_runtime_keys(item))
    return found
