"""Utilities for turning MCQ prompts into open exploratory questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..backends.base import BackendRequest, VisionLanguageBackend


_OPTION_LINE_RE = re.compile(r"^\s*[A-H][\).:-]\s+\S.*$", re.IGNORECASE)
_ANSWER_INSTRUCTION_RE = re.compile(
    r"\b(?:answer|respond|reply|select|choose|start)\b[^\n.]*\b(?:option\s+)?(?:letter|[A-H](?:\s*/\s*[A-H]){1,7}|[A-H](?:\s*,\s*[A-H]){1,7})\b[^\n.]*[.]?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplorationQuestionRewrite:
    exploration_question: str
    focus_points: tuple[str, ...] = ()
    target_entities: tuple[str, ...] = ()
    raw_text: str = ""
    used_model: bool = False
    fallback_reason: str = ""


def exploration_question(question: str, route_hint: str = "") -> str:
    """Return a model-facing open question with MCQ choices removed."""
    semantic_question = _semantic_question(question)
    parts = [semantic_question, "Do not choose an option."]
    hint = " ".join(str(route_hint or "").split())
    if hint:
        parts.append(hint)
    return " ".join(part for part in parts if part).strip()


def rewrite_exploration_question_with_model(
    backend: VisionLanguageBackend,
    *,
    question: str,
    route_hint: str = "",
) -> ExplorationQuestionRewrite:
    """Ask the text planner model to rewrite an MCQ into an option-blind exploration task."""

    fallback = _fallback_rewrite(question=question, route_hint=route_hint)
    prompt = _rewrite_prompt(question=question, route_hint=route_hint)
    try:
        response = backend.generate(
            BackendRequest(
                task="rewrite_exploration_question",
                prompt=prompt,
                max_new_tokens=512,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive fallback for non-text test backends.
        return ExplorationQuestionRewrite(
            exploration_question=fallback,
            raw_text="",
            used_model=False,
            fallback_reason=f"rewrite_backend_error:{type(exc).__name__}",
        )

    try:
        payload = json.loads(_extract_json_object(response.text))
    except (json.JSONDecodeError, ValueError) as exc:
        return ExplorationQuestionRewrite(
            exploration_question=fallback,
            raw_text=response.text,
            used_model=False,
            fallback_reason=f"rewrite_json_error:{type(exc).__name__}",
        )

    rewritten = _clean_rewritten_question(str(payload.get("exploration_question", "")))
    if not rewritten:
        return ExplorationQuestionRewrite(
            exploration_question=fallback,
            raw_text=response.text,
            used_model=False,
            fallback_reason="rewrite_empty",
        )
    if _leaks_option_surface(rewritten, question):
        return ExplorationQuestionRewrite(
            exploration_question=fallback,
            raw_text=response.text,
            used_model=False,
            fallback_reason="rewrite_option_leak",
        )
    focus_points = tuple(_clean_list(payload.get("focus_points", ())))
    target_entities = tuple(_clean_list(payload.get("target_entities", ())))
    return ExplorationQuestionRewrite(
        exploration_question=rewritten,
        focus_points=focus_points,
        target_entities=target_entities,
        raw_text=response.text,
        used_model=True,
    )


def _semantic_question(question: str) -> str:
    text = str(question or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    in_options = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(?:candidate\s+)?options?\s*:", line, flags=re.IGNORECASE):
            in_options = True
            continue
        if _OPTION_LINE_RE.match(line):
            in_options = True
            continue
        if in_options and _ANSWER_INSTRUCTION_RE.search(line):
            continue
        if in_options:
            continue
        cleaned = re.sub(r"^(?:question|query)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        cleaned = _ANSWER_INSTRUCTION_RE.sub("", cleaned).strip()
        if not cleaned:
            continue
        if re.search(r"\bmultiple-choice\b", cleaned, flags=re.IGNORECASE):
            continue
        lines.append(cleaned)
    semantic = " ".join(lines)
    semantic = _ANSWER_INSTRUCTION_RE.sub("", semantic)
    semantic = re.sub(r"\boption\s+[A-H]\b", "", semantic, flags=re.IGNORECASE)
    semantic = re.sub(r"\s+", " ", semantic).strip(" .")
    return f"{semantic}?" if semantic and not semantic.endswith("?") else semantic or "Describe the relevant visible facts."


def _rewrite_prompt(*, question: str, route_hint: str = "") -> str:
    return (
        "Rewrite a long-video multiple-choice question into an option-blind exploration question for a planner and "
        "local visual tools.\n"
        "The planner/tools must collect facts only; a separate AnswerAgent will later compare those facts with the "
        "original options.\n\n"
        "Output only JSON with this schema:\n"
        '{"exploration_question": string, "focus_points": [string], "target_entities": [string]}\n\n'
        "Rules:\n"
        "- Do not include option labels such as A, B, C, D, or words like option/choice/answer.\n"
        "- Do not copy the choices as alternatives. Extract the discriminative facts, entities, events, attributes, "
        "and temporal relations that should be observed.\n"
        "- For main-idea questions, ask for overall topic, main entity, time span, narrative arc, and major stages "
        "such as origin, growth, stability, decline, collapse, causes, or consequences when relevant.\n"
        "- For temporal-order questions, list the unique events/entities to look for without giving candidate orders, "
        "then ask for the order in which the video presents them.\n"
        "- For local factual questions, ask what visible/audio/text evidence is present in the relevant window.\n"
        "- Keep the exploration question concise but complete enough to guide search and segment inspection.\n\n"
        "Example 1 input:\n"
        "Question: What's the main idea of the video?\n"
        "Options:\n"
        "A. The fall of Rome\n"
        "B. Why the Austro-Hungarian Empire was divided\n"
        "C. A battle timeline\n"
        "D. How the Austro-Hungarian Empire rises and falls.\n"
        "Example 1 output:\n"
        '{"exploration_question":"Describe the overall topic and narrative arc of the video. Identify the main '
        'entity, time span, major stages, and whether the video covers origin, growth, stability, decline, collapse, '
        'causes, or consequences.","focus_points":["overall topic","main entity","time span","narrative stages"],'
        '"target_entities":["Austro-Hungarian Empire"]}\n\n'
        "Example 2 input:\n"
        "Question: In what order does the author present four sculptures?\n"
        "Options:\n"
        'A. "The Rape of Persephone", "Apollo and Daphne", "David", "Aeneas fleeing Troy".\n'
        'B. "David", "Aeneas fleeing Troy", "Apollo and Daphne", "The Rape of Persephone".\n'
        "Example 2 output:\n"
        '{"exploration_question":"Determine the order in which the video presents these sculptures: The Rape of '
        'Persephone, Apollo and Daphne, David, and Aeneas fleeing Troy. Record the segment or timestamp evidence for '
        'each item.","focus_points":["presentation order","timestamp evidence","artwork identification"],'
        '"target_entities":["The Rape of Persephone","Apollo and Daphne","David","Aeneas fleeing Troy"]}\n\n'
        f"Route hint: {route_hint or '(none)'}\n"
        f"Input question:\n{question}\n"
    )


def _fallback_rewrite(*, question: str, route_hint: str = "") -> str:
    semantic = _semantic_question(question)
    quoted = _quoted_targets(question)
    lowered = semantic.lower()
    if len(quoted) >= 2 and any(term in lowered for term in ["order", "sequence", "present", "shown"]):
        return (
            "Determine the order in which the video presents these items: "
            + ", ".join(quoted[:8])
            + ". Record segment or timestamp evidence for each item."
        )
    if any(term in lowered for term in ["main idea", "mainly about", "overall", "topic", "theme"]):
        return (
            "Describe the overall topic and narrative arc of the video. Identify the main entity, time span, "
            "major stages, and whether the video covers origin, growth, stability, decline, collapse, causes, "
            "or consequences."
        )
    return exploration_question(question, route_hint=route_hint).replace("Do not choose an option.", "Report facts only.")


def _clean_rewritten_question(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    cleaned = re.sub(r"\b(?:choose|select|answer)\s+(?:an?\s+)?(?:option|choice)\b", "report facts", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = " ".join(str(item).split()).strip()
        if text:
            cleaned.append(text)
    return cleaned[:12]


def _leaks_option_surface(rewritten: str, raw_question: str) -> bool:
    text = str(rewritten or "")
    if re.search(r"\b(?:option|choice|answer)\s*[A-H]\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"(?m)^\s*[A-H][\).:-]\s+\S+", text):
        return True
    raw_options = _option_texts(raw_question)
    return any(option and option in text for option in raw_options)


def _option_texts(question: str) -> list[str]:
    options = []
    for line in str(question or "").splitlines():
        match = re.match(r"^\s*[A-H][\).:-]\s*(\S.*)$", line.strip(), flags=re.IGNORECASE)
        if match:
            options.append(" ".join(match.group(1).split()).strip())
    return options


def _quoted_targets(question: str) -> list[str]:
    seen = set()
    targets = []
    for groups in re.findall(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’', str(question or "")):
        for group in groups:
            target = " ".join(group.split()).strip()
            key = target.lower()
            if target and key not in seen:
                seen.add(key)
                targets.append(target)
    return targets


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return stripped[start : end + 1]
