"""Evidence contracts shared by agent strategies."""

from .frame_set import FrameSet
from .option_relations import OptionRelation, OptionRelationKind

__all__ = [
    "EvidenceItem",
    "FrameSet",
    "OptionRelation",
    "OptionRelationKind",
]


def __getattr__(name: str):
    if name == "EvidenceItem":
        from .item import EvidenceItem

        return EvidenceItem
    raise AttributeError(name)
