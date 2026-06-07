"""Task-type playbooks for progressive planner guidance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class QuestionPlaybook:
    name: str
    route: str = "needle_local"
    instructions: Sequence[str] = field(default_factory=list)
    sufficiency_rules: Sequence[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines = [f"Task playbook: {self.name}", f"Question route: {self.route}", "Playbook instructions:"]
        lines.extend(f"- {instruction}" for instruction in self.instructions)
        lines.append("Evidence sufficiency:")
        lines.extend(f"- {rule}" for rule in self.sufficiency_rules)
        return "\n".join(lines)


def select_question_playbook(question: str) -> QuestionPlaybook:
    route = classify_question_route(question)
    if route == "gist_global":
        return QuestionPlaybook(
            name="gist_global",
            route=route,
            instructions=[
                "Start with global_gist to get a sparse whole-video topic and coverage hint.",
                "Use local inspection or indexed transcript evidence to verify the full-video coverage.",
                "Do not shred synopsis or overall-theme questions into local MCQ votes first.",
            ],
            sufficiency_rules=[
                "A global_gist observation is a topic hint, not structured option support.",
                "Prefer the option that covers the most video stages; partial ending-only coverage cannot beat a full rise/stability/fall arc.",
            ],
        )

    if route == "temporal_order":
        return QuestionPlaybook(
            name="timeline_ordering",
            route=route,
            instructions=[
                "Use coarse captions to locate candidate event segments before focused timestamp reads.",
                "Inspect at least the relevant earlier and later event windows when order matters.",
                "Use local workers for open factual descriptions; compare options only after evidence is collected.",
            ],
            sufficiency_rules=[
                "Citations must include timestamped visual observations for the ordered events.",
                "Evidence must not conflict with the claimed temporal relation.",
                "verify option consistency against the cited observation before final.",
            ],
        )

    if extract_candidate_options(question):
        return QuestionPlaybook(
            name="multiple_choice",
            route=route,
            instructions=[
                "Use video_ls/search_segments to localize candidates before visual inspection.",
                "Call inspect_segment with candidate_options only so the inspector knows what facts to look for.",
                "Local workers must report facts only; AnswerAgent maps facts to options.",
                "Avoid finalizing from navigation-only evidence.",
            ],
            sufficiency_rules=[
                "At least one cited visual observation must ground the selected option.",
                "verify option consistency against the cited observation before final.",
                "Final answer should preserve the option letter when the user provided choices.",
            ],
        )

    return QuestionPlaybook(
        name="general_video_qa",
        route=route,
        instructions=[
            "Use query-conditioned navigation to localize likely evidence.",
            "Delegate visual reading to inspect_segment once a candidate is localized.",
        ],
        sufficiency_rules=[
            "Final answers need cited non-navigation visual evidence.",
            "State uncertainty when evidence is incomplete or ambiguous.",
        ],
    )


def classify_question_route(question: str) -> str:
    """Classify whether the question needs a whole-video floor or localized search."""

    lowered = _semantic_question_text(question).lower()
    gist_markers = [
        "mainly about",
        "primarily about",
        "overall",
        "whole video",
        "entire video",
        "synopsis",
        "summary",
        "summarize",
        "theme",
        "main idea",
        "central idea",
        "main topic",
        "main subject",
        "main content",
        "general content",
        "what is the video about",
    ]
    if any(marker in lowered for marker in gist_markers):
        return "gist_global"

    temporal_markers = [
        "right after",
        "right before",
        "immediately after",
        "immediately before",
        "before",
        "after",
        "first",
        "last",
        "then",
        "order",
        "sequence",
        "temporal",
        "earlier",
        "later",
        "at the beginning",
        "at the end",
        "specific moment",
    ]
    if any(marker in lowered for marker in temporal_markers):
        return "temporal_order"
    return "needle_local"


def extract_candidate_options(question: str) -> Sequence[str]:
    options = []
    for line in question.splitlines():
        stripped = line.strip()
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            options.append(stripped)
    if options:
        return options

    normalized = re.sub(r"\bOptions\s*:\s*", " ", question, flags=re.IGNORECASE)
    matches = re.finditer(
        r"(?<![A-Za-z0-9])([A-H])([.)])\s+(.*?)(?=(?<![A-Za-z0-9])[A-H][.)]\s+|$)",
        normalized,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in matches:
        text = " ".join(str(match.group(3)).split()).strip()
        if text:
            options.append(f"{match.group(1).upper()}{match.group(2)} {text}")
    return options


def _semantic_question_text(question: str) -> str:
    match = re.search(r"\bQuestion:\s*(.*?)(?:\n\s*Options:|\Z)", question, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    lines = []
    for line in question.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[A-H][.)]\s+\S+", stripped):
            continue
        if "answer with" in stripped.lower() and "option letter" in stripped.lower():
            continue
        lines.append(stripped)
    return "\n".join(lines) or question
