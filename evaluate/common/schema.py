from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimePrediction:
    case_id: str
    answer: str
    selected_option: str = ""
    supporting_intervals: tuple[tuple[float, float], ...] = ()
    supporting_attempt_ids: tuple[str, ...] = ()
    answer_present: bool = False
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "answer", str(self.answer))
        object.__setattr__(self, "selected_option", str(self.selected_option))
        object.__setattr__(
            self,
            "supporting_intervals",
            tuple((float(start), float(end)) for start, end in self.supporting_intervals),
        )
        object.__setattr__(
            self,
            "supporting_attempt_ids",
            tuple(dict.fromkeys(str(item) for item in self.supporting_attempt_ids if str(item))),
        )
        object.__setattr__(self, "runtime_metadata", dict(self.runtime_metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimePrediction":
        return cls(
            case_id=str(value["case_id"]),
            answer=str(value.get("answer", "")),
            selected_option=str(value.get("selected_option", "")),
            supporting_intervals=tuple(
                tuple(interval)
                for interval in value.get("supporting_intervals", ())
            ),
            supporting_attempt_ids=tuple(value.get("supporting_attempt_ids", ())),
            answer_present=bool(value.get("answer_present", bool(value.get("answer")))),
            runtime_metadata=dict(value.get("runtime_metadata", {})),
        )
