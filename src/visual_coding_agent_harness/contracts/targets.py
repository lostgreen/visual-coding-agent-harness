"""Target and evidence contracts for option-level verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Sequence, Tuple

_TARGET_REF_RE = re.compile(r"^T[1-9]\d*$")


class ClaimModality(str, Enum):
    NARRATED_FACT = "narrated_fact"
    VISUAL_FACT = "visual_fact"
    OCR_FACT = "ocr_fact"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    canonical_text: str
    aliases: Sequence[str] = field(default_factory=tuple)
    subject: str | None = None
    relation: str | None = None
    modality_hint: ClaimModality = ClaimModality.UNKNOWN
    source: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if not isinstance(self.modality_hint, ClaimModality):
            object.__setattr__(self, "modality_hint", ClaimModality(self.modality_hint))
        if not _TARGET_REF_RE.fullmatch(str(self.target_id)):
            raise ValueError(f"Target id must be a T<n> reference: {self.target_id}")


@dataclass(frozen=True)
class OptionSpec:
    option_id: str
    target_sequence: Sequence[str] = field(default_factory=tuple)
    required_relations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_sequence", tuple(self.target_sequence))
        object.__setattr__(self, "required_relations", tuple(self.required_relations))


@dataclass(frozen=True)
class ClaimRelation:
    relation_id: str
    kind: str
    source_target_id: str
    destination_target_id: str


@dataclass(frozen=True)
class EvidenceBinding:
    evidence_id: str
    obs_id: str
    target_id: str
    subject: str
    relation: str
    status: str
    mention_timestamp_sec: float | None
    source: str
    snippet: str
    claim_modality: ClaimModality = ClaimModality.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.claim_modality, ClaimModality):
            object.__setattr__(self, "claim_modality", ClaimModality(self.claim_modality))


@dataclass(frozen=True)
class RelationBinding:
    binding_id: str
    obs_id: str
    relation_id: str
    status: str
    source: str
    snippet: str
    mention_timestamp_sec: float | None


@dataclass(frozen=True)
class TargetRegistry:
    targets_by_id: Mapping[str, TargetSpec]
    options_by_id: Mapping[str, OptionSpec] = field(default_factory=dict)
    relations_by_id: Mapping[str, ClaimRelation] = field(default_factory=dict)
    target_to_options: Mapping[str, Tuple[OptionSpec, ...]] = field(default_factory=dict)
    canonical_to_targets: Mapping[str, Tuple[TargetSpec, ...]] = field(default_factory=dict)
    version: str = "v1"

    @classmethod
    def from_specs(
        cls,
        *,
        targets: Iterable[TargetSpec],
        options: Iterable[OptionSpec] = (),
        relations: Iterable[ClaimRelation] = (),
        version: str = "v1",
    ) -> "TargetRegistry":
        targets_by_id = _unique_by_id(targets, "target_id", "target")
        options_by_id = _unique_by_id(options, "option_id", "option")
        relations_by_id = _unique_by_id(relations, "relation_id", "relation")

        target_to_options: Dict[str, list[OptionSpec]] = {target_id: [] for target_id in targets_by_id}
        for option in options_by_id.values():
            for target_id in option.target_sequence:
                if target_id not in targets_by_id:
                    raise KeyError(f"Unknown target in option {option.option_id}: {target_id}")
                target_to_options[target_id].append(option)
            for relation_id in option.required_relations:
                if relation_id not in relations_by_id:
                    raise KeyError(f"Unknown relation in option {option.option_id}: {relation_id}")

        for relation in relations_by_id.values():
            if relation.source_target_id not in targets_by_id:
                raise KeyError(f"Unknown relation source target: {relation.source_target_id}")
            if relation.destination_target_id not in targets_by_id:
                raise KeyError(f"Unknown relation destination target: {relation.destination_target_id}")

        canonical_to_targets: Dict[str, list[TargetSpec]] = {}
        for target in targets_by_id.values():
            canonical_to_targets.setdefault(target.canonical_text, []).append(target)

        return cls(
            targets_by_id=MappingProxyType(dict(targets_by_id)),
            options_by_id=MappingProxyType(dict(options_by_id)),
            relations_by_id=MappingProxyType(dict(relations_by_id)),
            target_to_options=MappingProxyType(
                {
                    target_id: tuple(member_options)
                    for target_id, member_options in target_to_options.items()
                }
            ),
            canonical_to_targets=MappingProxyType(
                {
                    canonical_text: tuple(canonical_targets)
                    for canonical_text, canonical_targets in canonical_to_targets.items()
                }
            ),
            version=version,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets_by_id", _freeze_mapping(self.targets_by_id))
        object.__setattr__(self, "options_by_id", _freeze_mapping(self.options_by_id))
        object.__setattr__(self, "relations_by_id", _freeze_mapping(self.relations_by_id))
        object.__setattr__(
            self,
            "target_to_options",
            MappingProxyType(
                {target_id: tuple(options) for target_id, options in self.target_to_options.items()}
            ),
        )
        object.__setattr__(
            self,
            "canonical_to_targets",
            MappingProxyType(
                {
                    canonical_text: tuple(targets)
                    for canonical_text, targets in self.canonical_to_targets.items()
                }
            ),
        )

    def known_target_ref(self, target_ref: str) -> bool:
        return target_ref in self.targets_by_id

    def resolve_target_ref(self, target_ref: str) -> TargetSpec:
        if target_ref in self.targets_by_id:
            return self.targets_by_id[target_ref]

        raise KeyError(f"Unknown target reference: {target_ref}")

    def option_for(self, option_id: str) -> OptionSpec:
        try:
            return self.options_by_id[option_id]
        except KeyError as exc:
            raise KeyError(f"Unknown option: {option_id}") from exc

    def options_for_target(self, target_ref: str) -> Tuple[OptionSpec, ...]:
        target = self.resolve_target_ref(target_ref)
        return self.target_to_options.get(target.target_id, ())

    def targets_for_canonical(self, canonical_text: str) -> Tuple[TargetSpec, ...]:
        return self.canonical_to_targets.get(canonical_text, ())


def _unique_by_id(items: Iterable[object], attr: str, item_name: str) -> Dict[str, object]:
    indexed: Dict[str, object] = {}
    for item in items:
        item_id = getattr(item, attr)
        if item_id in indexed:
            raise ValueError(f"Duplicate {item_name} id: {item_id}")
        indexed[item_id] = item
    return indexed


def _freeze_mapping(mapping: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))
