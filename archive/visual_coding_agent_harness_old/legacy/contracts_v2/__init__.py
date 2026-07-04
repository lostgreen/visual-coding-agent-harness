"""Legacy target and evidence-binding contracts."""

from .ordered_sequence import (
    OrderedTranscriptItem,
    OrderedTranscriptSequence,
    build_ordered_transcript_sequence,
    ordered_sequence_exact_option,
)
from .targets import (
    ClaimModality,
    ClaimRelation,
    EvidenceBinding,
    OptionSpec,
    RelationBinding,
    TargetRegistry,
    TargetSpec,
    TargetTextHit,
)

__all__ = [
    "ClaimModality",
    "ClaimRelation",
    "EvidenceBinding",
    "OptionSpec",
    "OrderedTranscriptItem",
    "OrderedTranscriptSequence",
    "RelationBinding",
    "TargetRegistry",
    "TargetSpec",
    "TargetTextHit",
    "build_ordered_transcript_sequence",
    "ordered_sequence_exact_option",
]
