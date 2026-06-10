"""Visual evidence and final-gate contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Sequence

VISUAL_EVIDENCE_NFRAMES: Final[int] = 128
VISUAL_EVIDENCE_NFRAMES_MIN: Final[int] = 64
VISUAL_EVIDENCE_NFRAMES_MAX: Final[int] = 256

BudgetReason = Literal[
    "default_contract",
    "user_override",
    "adaptive_downshift",
    "tool_capability_cap",
]

SamplingPolicy = Literal["uniform", "fps", "backend_default"]

GroundingQuality = Literal[
    "visually_confirmed",
    "global_impression",
    "navigation_only",
    "asr_textual",
    "ocr_textual",
    "query_global_context",
]

EvidenceStage = Literal[
    "raw",
    "distilled",
    "ledger",
    "mapped",
    "final_support",
]

CONTRACT_VERSION: Final[str] = "v1.0"

FinalRejectionReason = Literal[
    "missing_relation_binding",
    "missing_target_binding",
    "conflicting_evidence",
    "insufficient_breadth",
    "wrong_subject",
    "unsupported_modality",
    "no_per_option_coverage",
    "no_answer_grade_citation",
    "verifier_failed",
]

EvidenceSupportStatus = Literal[
    "supported",
    "partial",
    "conflicting",
    "unsupported",
    "ambiguous",
]

OptionBindingStatus = Literal[
    "supported",
    "partial",
    "unsupported",
    "ambiguous",
    "conflicting",
]

OptionRejectionReason = Literal[
    "narrower",
    "off_topic",
    "wrong_subject",
    "wrong_arc",
    "insufficient_breadth",
]

FinalGateStatus = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class EvidenceBinding:
    evidence_id: str
    target_ref: str | None
    relation_ref: str | None
    option_id: str | None
    modality: str
    source: str
    timestamp_start: float | None
    timestamp_end: float | None
    support_status: EvidenceSupportStatus
    confidence: float | None
    rationale: str = ""


@dataclass(frozen=True)
class RelationBinding:
    relation_ref: str
    ordered_target_refs: Sequence[str] = field(default_factory=tuple)
    evidence_ids: Sequence[str] = field(default_factory=tuple)
    support_status: EvidenceSupportStatus = "ambiguous"
    timestamp_order: Sequence[float] = field(default_factory=tuple)
    modality: str | None = None
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_target_refs", tuple(self.ordered_target_refs))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "timestamp_order", tuple(self.timestamp_order))


@dataclass(frozen=True)
class OptionEvaluation:
    option_id: str
    binding_status: OptionBindingStatus
    rejection_reason: OptionRejectionReason | None
    coverage_breadth: int
    supporting_evidence_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))


@dataclass(frozen=True)
class FinalGateDecision:
    proposed_option: str
    gate_status: FinalGateStatus
    reason_code: FinalRejectionReason | None = None
    supporting_evidence_ids: Sequence[str] = field(default_factory=tuple)
    missing_target_refs: Sequence[str] = field(default_factory=tuple)
    missing_relation_refs: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "missing_target_refs", tuple(self.missing_target_refs))
        object.__setattr__(self, "missing_relation_refs", tuple(self.missing_relation_refs))

    @property
    def accepted(self) -> bool:
        return self.gate_status == "accepted"


def resolve_nframes(requested: int | None, tool_cap: int | None = None) -> tuple[int, BudgetReason]:
    """Return effective frame count and the reason it was selected."""

    if requested is None:
        effective = VISUAL_EVIDENCE_NFRAMES
        reason: BudgetReason = "default_contract"
    else:
        effective = max(VISUAL_EVIDENCE_NFRAMES_MIN, min(VISUAL_EVIDENCE_NFRAMES_MAX, int(requested)))
        reason = "user_override"

    if tool_cap is not None and effective > tool_cap:
        return int(tool_cap), "tool_capability_cap"
    return effective, reason
