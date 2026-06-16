"""Order-hypothesis contracts for same-entity timeline options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OrderedEntity:
    entity_id: str
    canonical_name: str
    aliases: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(str(alias) for alias in self.aliases if str(alias).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OrderedEntity":
        return cls(
            entity_id=str(payload.get("entity_id", "")).strip(),
            canonical_name=str(payload.get("canonical_name", "")).strip(),
            aliases=tuple(str(item) for item in payload.get("aliases", ()) if str(item).strip()),
        )


@dataclass(frozen=True)
class OptionOrderHypothesis:
    option_id: str
    ordered_entity_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", str(self.option_id).strip().upper()[:1])
        object.__setattr__(self, "ordered_entity_ids", tuple(str(item) for item in self.ordered_entity_ids if str(item).strip()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "ordered_entity_ids": list(self.ordered_entity_ids),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OptionOrderHypothesis":
        return cls(
            option_id=str(payload.get("option_id", "")).strip(),
            ordered_entity_ids=tuple(str(item) for item in payload.get("ordered_entity_ids", ()) if str(item).strip()),
        )


@dataclass(frozen=True)
class OrderedSetSpec:
    set_id: str
    entities: Sequence[OrderedEntity] = field(default_factory=tuple)
    hypotheses: Sequence[OptionOrderHypothesis] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "entities": [entity.to_dict() for entity in self.entities],
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OrderedSetSpec":
        return cls(
            set_id=str(payload.get("set_id", "")).strip(),
            entities=tuple(OrderedEntity.from_mapping(item) for item in _mapping_sequence(payload.get("entities", ()))),
            hypotheses=tuple(
                OptionOrderHypothesis.from_mapping(item)
                for item in _mapping_sequence(payload.get("hypotheses", ()))
            ),
        )


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))
