"""Visual evidence contracts. Single source of truth for sampling budgets."""

from __future__ import annotations

from typing import Final, Literal

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
