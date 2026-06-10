"""Review-gated constants for final answer evidence policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

SkillPolicyName = Literal[
    "main_idea",
    "visual_timeline_qa",
    "narration_timeline_qa",
    "mixed_asr_visual_qa",
    "grounded_factual_qa",
    "mutex_fact_qa",
]

AnswerGradeTranscriptPolicy = Literal["allowed", "required", "not_allowed"]

VISUAL_MODALITIES = frozenset({"visual", "vision", "frame", "frames", "qa", "caption", "ocr"})
TRANSCRIPT_MODALITIES = frozenset({"asr", "transcript", "indexed_transcript", "narration"})
MIXED_MODALITIES = frozenset({"mixed"})


@dataclass(frozen=True)
class SkillPolicy:
    skill_name: SkillPolicyName
    allowed_modalities: frozenset[str]
    min_bindings: int
    min_distinct_segments: int
    relation_required: bool
    visual_verification_mandatory: bool
    transcript_answer_grade: AnswerGradeTranscriptPolicy
    requires_per_option_coverage: bool = False
    requires_option_kind: frozenset[str] = field(default_factory=frozenset)
    requires_central_subject_overlap: bool = False
    overlap_window_seconds: float | None = None


SKILL_POLICIES: Mapping[str, SkillPolicy] = MappingProxyType(
    {
        "main_idea": SkillPolicy(
            skill_name="main_idea",
            allowed_modalities=frozenset((*VISUAL_MODALITIES, *TRANSCRIPT_MODALITIES, *MIXED_MODALITIES)),
            min_bindings=2,
            min_distinct_segments=2,
            relation_required=False,
            visual_verification_mandatory=False,
            transcript_answer_grade="allowed",
            requires_per_option_coverage=True,
            requires_option_kind=frozenset({"topic_arc", "topic_focus"}),
            requires_central_subject_overlap=True,
        ),
        "narration_timeline_qa": SkillPolicy(
            skill_name="narration_timeline_qa",
            allowed_modalities=frozenset((*TRANSCRIPT_MODALITIES,)),
            min_bindings=1,
            min_distinct_segments=1,
            relation_required=True,
            visual_verification_mandatory=False,
            transcript_answer_grade="allowed",
        ),
        "visual_timeline_qa": SkillPolicy(
            skill_name="visual_timeline_qa",
            allowed_modalities=frozenset((*VISUAL_MODALITIES,)),
            min_bindings=1,
            min_distinct_segments=1,
            relation_required=True,
            visual_verification_mandatory=True,
            transcript_answer_grade="not_allowed",
        ),
        "mixed_asr_visual_qa": SkillPolicy(
            skill_name="mixed_asr_visual_qa",
            allowed_modalities=frozenset((*VISUAL_MODALITIES, *TRANSCRIPT_MODALITIES, *MIXED_MODALITIES)),
            min_bindings=2,
            min_distinct_segments=1,
            relation_required=False,
            visual_verification_mandatory=True,
            transcript_answer_grade="required",
            overlap_window_seconds=5.0,
        ),
        "grounded_factual_qa": SkillPolicy(
            skill_name="grounded_factual_qa",
            allowed_modalities=frozenset((*VISUAL_MODALITIES, *TRANSCRIPT_MODALITIES, *MIXED_MODALITIES)),
            min_bindings=1,
            min_distinct_segments=1,
            relation_required=False,
            visual_verification_mandatory=False,
            transcript_answer_grade="allowed",
        ),
        "mutex_fact_qa": SkillPolicy(
            skill_name="mutex_fact_qa",
            allowed_modalities=frozenset((*VISUAL_MODALITIES, *TRANSCRIPT_MODALITIES, *MIXED_MODALITIES)),
            min_bindings=1,
            min_distinct_segments=1,
            relation_required=False,
            visual_verification_mandatory=False,
            transcript_answer_grade="allowed",
        ),
    }
)


def get_skill_policy(skill_name: str) -> SkillPolicy:
    try:
        return SKILL_POLICIES[skill_name]
    except KeyError as exc:
        raise KeyError(f"Unknown final-gate skill policy: {skill_name}") from exc


def modality_allowed(modality: str | None, allowed_modalities: Sequence[str] | frozenset[str]) -> bool:
    return str(modality or "").strip().lower() in allowed_modalities
