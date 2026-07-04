"""Option-level evidence relation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptionRelationKind = Literal["supports", "contradicts", "background", "distractor", "out_of_scope", "inconclusive"]


@dataclass(frozen=True)
class OptionRelation:
    option_id: str
    relation: OptionRelationKind
    rationale: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "option_id": self.option_id,
            "relation": self.relation,
            "rationale": self.rationale[:200],
        }
