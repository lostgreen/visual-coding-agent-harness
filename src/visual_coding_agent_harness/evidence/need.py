"""Evidence-facing view of multi-agent sub-goals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.multi.protocol import SubGoal, SubGoalIntent, SubGoalSuccessCriteria

EvidenceNeedPolarity = str


@dataclass(frozen=True)
class EvidenceNeed:
    """Semantic evidence request derived from a persisted SubGoal."""

    need_id: str
    option_id: str | None
    polarity: EvidenceNeedPolarity
    claim: str
    segment_id: str | None
    time_range: tuple[float, float] | None
    modality_hint: tuple[str, ...]
    budget_max_explores: int
    budget_max_verifies: int
    parent_question: str
    status: str
    rationale: str

    @classmethod
    def from_sub_goal(cls, sub_goal: SubGoal) -> "EvidenceNeed":
        return cls(
            need_id=sub_goal.sub_goal_id,
            option_id=sub_goal.constraint.option_id,
            polarity=_polarity_from_intent(sub_goal.intent),
            claim=sub_goal.constraint.claim,
            segment_id=sub_goal.constraint.segment_id,
            time_range=sub_goal.constraint.time_range,
            modality_hint=tuple(sub_goal.constraint.modality_hint),
            budget_max_explores=int(sub_goal.budget.max_explores),
            budget_max_verifies=int(sub_goal.budget.max_verifies),
            parent_question=sub_goal.parent_question,
            status=sub_goal.status,
            rationale=sub_goal.rationale,
        )

    def to_sub_goal_kwargs(
        self,
        *,
        created_by: str,
        created_round: int,
        success_criteria: SubGoalSuccessCriteria | None = None,
        max_frames: int = 256,
    ) -> dict[str, Any]:
        """Return keyword args accepted by WorkspaceMutator.create_sub_goal."""

        from ..agents.multi.protocol import SubGoalBudget, SubGoalConstraint, SubGoalSuccessCriteria

        return {
            "intent": _intent_from_polarity(self.polarity),
            "constraint": SubGoalConstraint(
                option_id=self.option_id,
                claim=self.claim,
                segment_id=self.segment_id,
                time_range=self.time_range,
                modality_hint=tuple(self.modality_hint),
            ),
            "budget": SubGoalBudget(
                max_explores=int(self.budget_max_explores),
                max_verifies=int(self.budget_max_verifies),
                max_frames=max_frames,
            ),
            "success_criteria": success_criteria or SubGoalSuccessCriteria(),
            "parent_question": self.parent_question,
            "created_by": created_by,
            "created_round": created_round,
            "rationale": self.rationale,
        }


def _polarity_from_intent(intent: SubGoalIntent) -> EvidenceNeedPolarity:
    if intent == "disprove":
        return "seek_refutation"
    if intent == "disambiguate":
        return "disambiguate"
    return "seek_support"


def _intent_from_polarity(polarity: str) -> SubGoalIntent:
    if polarity == "seek_refutation":
        return "disprove"
    if polarity == "disambiguate":
        return "disambiguate"
    return "verify"
