"""Task-type playbooks for progressive planner guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class QuestionPlaybook:
    name: str
    instructions: Sequence[str] = field(default_factory=list)
    sufficiency_rules: Sequence[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"Task playbook: {self.name}", "Playbook instructions:"]
        lines.extend(f"- {instruction}" for instruction in self.instructions)
        lines.append("Evidence sufficiency:")
        lines.extend(f"- {rule}" for rule in self.sufficiency_rules)
        return "\n".join(lines)


def select_question_playbook(question: str) -> QuestionPlaybook:
    if extract_candidate_options(question):
        return QuestionPlaybook(
            name="multiple_choice",
            instructions=[
                "Use video_ls/search_segments to localize candidates before visual inspection.",
                "Call inspect_segment with candidate_options so the inspector can attribute evidence to choices.",
                "Avoid finalizing from navigation-only evidence.",
            ],
            sufficiency_rules=[
                "At least one inspect_segment observation must support the selected option.",
                "verify option consistency against the cited observation before final.",
                "Final answer should preserve the option letter when the user provided choices.",
            ],
        )

    lowered = question.lower()
    if any(marker in lowered for marker in ["before", "after", "first", "then", "order", "sequence"]):
        return QuestionPlaybook(
            name="temporal_ordering",
            instructions=[
                "Use search/video_ls to find candidate moments, then zoom if windows are too coarse.",
                "Inspect at least the relevant earlier and later windows when order matters.",
            ],
            sufficiency_rules=[
                "Citations must include timestamped visual observations for the ordered events.",
                "Evidence must not conflict with the claimed temporal relation.",
            ],
        )

    return QuestionPlaybook(
        name="general_video_qa",
        instructions=[
            "Use query-conditioned navigation to localize likely evidence.",
            "Delegate visual reading to inspect_segment once a candidate is localized.",
        ],
        sufficiency_rules=[
            "Final answers need cited non-navigation visual evidence.",
            "State uncertainty when evidence is incomplete or ambiguous.",
        ],
    )


def extract_candidate_options(question: str) -> Sequence[str]:
    options = []
    for line in question.splitlines():
        stripped = line.strip()
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            options.append(stripped)
    return options
