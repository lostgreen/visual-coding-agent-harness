"""Structured, model-owned grounding plan contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

ClaimKind = Literal[
    "entity",
    "visible_event",
    "narrated_fact",
    "state",
    "state_transition",
    "attribute",
    "ordered_item",
    "topic",
    "cause",
    "consequence",
]
GroundingModality = Literal["visual", "asr", "ocr", "mixed", "unknown"]
GroundingPolarity = Literal["affirmed", "negated", "unknown"]
RelationKind = Literal["before", "after", "same_scene", "causes", "contradicts", "equivalent", "transitions_to"]

ALLOWED_CLAIM_KINDS = frozenset(ClaimKind.__args__)  # type: ignore[attr-defined]
ALLOWED_GROUNDING_MODALITIES = frozenset(GroundingModality.__args__)  # type: ignore[attr-defined]
ALLOWED_GROUNDING_POLARITIES = frozenset(GroundingPolarity.__args__)  # type: ignore[attr-defined]
ALLOWED_RELATION_KINDS = frozenset(RelationKind.__args__)  # type: ignore[attr-defined]
ALLOWED_GROUNDING_ROUTES = frozenset({"gist_global", "needle_local", "temporal_order", "mixed_asr_visual"})
ALLOWED_EVIDENCE_SOURCES = frozenset({"visual", "asr", "ocr", "mixed", "qa", "indexed_transcript", "global"})


@dataclass(frozen=True)
class GroundingSubject:
    subject_key: str
    canonical_name: str
    aliases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases if str(alias).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_key": self.subject_key,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingSubject":
        return cls(
            subject_key=str(payload.get("subject_key", "")).strip(),
            canonical_name=str(payload.get("canonical_name", "")).strip(),
            aliases=_string_tuple(payload.get("aliases", ())),
        )


@dataclass(frozen=True)
class GroundingTarget:
    target_key: str
    canonical_claim: str
    subject_key: str | None = None
    claim_kind: ClaimKind = "entity"
    claim_modality: GroundingModality = "unknown"
    aliases: Sequence[str] = field(default_factory=tuple)
    search_queries: Sequence[str] = field(default_factory=tuple)
    polarity: GroundingPolarity = "unknown"

    def __post_init__(self) -> None:
        subject_key = str(self.subject_key or "").strip() or None
        object.__setattr__(self, "subject_key", subject_key)
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases if str(alias).strip()))
        object.__setattr__(
            self,
            "search_queries",
            tuple(str(query) for query in self.search_queries if str(query).strip()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_key": self.target_key,
            "canonical_claim": self.canonical_claim,
            "subject_key": self.subject_key,
            "claim_kind": self.claim_kind,
            "claim_modality": self.claim_modality,
            "aliases": list(self.aliases),
            "search_queries": list(self.search_queries),
            "polarity": self.polarity,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingTarget":
        return cls(
            target_key=str(payload.get("target_key", "")).strip(),
            canonical_claim=str(payload.get("canonical_claim", "")).strip(),
            subject_key=str(payload.get("subject_key", "")).strip() or None,
            claim_kind=str(payload.get("claim_kind", "entity")).strip() or "entity",
            claim_modality=str(payload.get("claim_modality", "unknown")).strip() or "unknown",
            aliases=_string_tuple(payload.get("aliases", ())),
            search_queries=_string_tuple(payload.get("search_queries", ())),
            polarity=str(payload.get("polarity", "unknown")).strip() or "unknown",
        )


@dataclass(frozen=True)
class GroundingRelation:
    relation_key: str
    kind: RelationKind
    source_target_key: str
    destination_target_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_key": self.relation_key,
            "kind": self.kind,
            "source_target_key": self.source_target_key,
            "destination_target_key": self.destination_target_key,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingRelation":
        return cls(
            relation_key=str(payload.get("relation_key", "")).strip(),
            kind=str(payload.get("kind", "")).strip() or "before",
            source_target_key=str(payload.get("source_target_key", "")).strip(),
            destination_target_key=str(payload.get("destination_target_key", "")).strip(),
        )


@dataclass(frozen=True)
class GroundingOption:
    option_id: str
    required_target_keys: Sequence[str] = field(default_factory=tuple)
    ordered_target_keys: Sequence[str] = field(default_factory=tuple)
    required_relation_keys: Sequence[str] = field(default_factory=tuple)
    raw_option_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", str(self.option_id).strip().upper()[:1])
        object.__setattr__(self, "required_target_keys", _string_tuple(self.required_target_keys))
        object.__setattr__(self, "ordered_target_keys", _string_tuple(self.ordered_target_keys))
        object.__setattr__(self, "required_relation_keys", _string_tuple(self.required_relation_keys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "required_target_keys": list(self.required_target_keys),
            "ordered_target_keys": list(self.ordered_target_keys),
            "required_relation_keys": list(self.required_relation_keys),
            "raw_option_text": self.raw_option_text,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingOption":
        return cls(
            option_id=str(payload.get("option_id", "")).strip(),
            required_target_keys=_string_tuple(payload.get("required_target_keys", ())),
            ordered_target_keys=_string_tuple(payload.get("ordered_target_keys", ())),
            required_relation_keys=_string_tuple(payload.get("required_relation_keys", ())),
            raw_option_text=str(payload.get("raw_option_text", "")).strip(),
        )


@dataclass(frozen=True)
class GroundingPlan:
    route: str
    recommended_skill: str
    subjects: Sequence[GroundingSubject] = field(default_factory=tuple)
    targets: Sequence[GroundingTarget] = field(default_factory=tuple)
    relations: Sequence[GroundingRelation] = field(default_factory=tuple)
    options: Sequence[GroundingOption] = field(default_factory=tuple)
    acceptable_evidence_sources: Sequence[str] = field(default_factory=tuple)
    confidence: float = 0.0
    unresolved_ambiguities: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", tuple(self.subjects))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "acceptable_evidence_sources", _string_tuple(self.acceptable_evidence_sources))
        object.__setattr__(self, "unresolved_ambiguities", _string_tuple(self.unresolved_ambiguities))
        object.__setattr__(self, "confidence", float(self.confidence or 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "recommended_skill": self.recommended_skill,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "targets": [target.to_dict() for target in self.targets],
            "relations": [relation.to_dict() for relation in self.relations],
            "options": [option.to_dict() for option in self.options],
            "acceptable_evidence_sources": list(self.acceptable_evidence_sources),
            "confidence": self.confidence,
            "unresolved_ambiguities": list(self.unresolved_ambiguities),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingPlan":
        return cls(
            route=str(payload.get("route", "")).strip(),
            recommended_skill=str(payload.get("recommended_skill", "")).strip(),
            subjects=tuple(
                GroundingSubject.from_mapping(item)
                for item in _mapping_sequence(payload.get("subjects", ()))
            ),
            targets=tuple(
                GroundingTarget.from_mapping(item)
                for item in _mapping_sequence(payload.get("targets", ()))
            ),
            relations=tuple(
                GroundingRelation.from_mapping(item)
                for item in _mapping_sequence(payload.get("relations", ()))
            ),
            options=tuple(
                GroundingOption.from_mapping(item)
                for item in _mapping_sequence(payload.get("options", ()))
            ),
            acceptable_evidence_sources=_string_tuple(payload.get("acceptable_evidence_sources", ())),
            confidence=_float_value(payload.get("confidence", 0.0)),
            unresolved_ambiguities=_string_tuple(payload.get("unresolved_ambiguities", ())),
        )


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        values = [str(value)]
    elif isinstance(value, Sequence):
        values = [str(item) for item in value]
    else:
        values = []
    return tuple(text for text in (" ".join(item.split()).strip() for item in values) if text)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
