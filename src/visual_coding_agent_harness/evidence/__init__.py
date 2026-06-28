"""Evidence contracts shared by agent strategies."""

from .frame_set import FrameSet
from .option_relations import OptionRelation, OptionRelationKind

__all__ = [
    "EvidenceItem",
    "EvidenceLedger",
    "EvidenceNeed",
    "FrameSet",
    "OptionRelation",
    "OptionRelationKind",
]


def __getattr__(name: str):
    if name == "EvidenceItem":
        from .item import EvidenceItem

        return EvidenceItem
    if name == "EvidenceLedger":
        from .ledger import EvidenceLedger

        return EvidenceLedger
    if name == "EvidenceNeed":
        from .need import EvidenceNeed

        return EvidenceNeed
    raise AttributeError(name)
