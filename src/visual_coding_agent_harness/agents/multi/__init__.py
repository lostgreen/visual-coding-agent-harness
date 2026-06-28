"""Clean multi-agent architecture boundaries."""

from .driver import MultiAgentDriver
from .investigator import InvestigatorAgent
from .mutator import WorkspaceMutator
from .protocol import (
    Finding,
    FindingStatus,
    InboxMessage,
    MessageKind,
    SubGoal,
    SubGoalBudget,
    SubGoalConstraint,
    SubGoalIntent,
    SubGoalStatus,
    SubGoalSuccessCriteria,
)
from .reasoner import ReasonerAgent

__all__ = [
    "Finding",
    "FindingStatus",
    "InboxMessage",
    "InvestigatorAgent",
    "MessageKind",
    "MultiAgentDriver",
    "ReasonerAgent",
    "SubGoal",
    "SubGoalBudget",
    "SubGoalConstraint",
    "SubGoalIntent",
    "SubGoalStatus",
    "SubGoalSuccessCriteria",
    "WorkspaceMutator",
]
