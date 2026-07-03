"""Utilities for turning MCQ prompts into open exploratory questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from ...backends.base import BackendRequest, VisionLanguageBackend


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


@dataclass(frozen=True)
class QuestionContext:
    raw_question: str
    options: list[str]
    planner_question: str
    answer_question: str
    navigation_question: str
    vlm_safe_question: str


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


def build_question_context(question: str) -> QuestionContext:
    raw_question = str(question or "")
    return QuestionContext(
        raw_question=raw_question,
        options=list(extract_candidate_options(raw_question)),
        planner_question=raw_question,
        answer_question=raw_question,
        navigation_question=raw_question,
        vlm_safe_question=exploration_question(raw_question),
    )


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

    focus_points = tuple(_clean_list(payload.get("focus_points", ())))
    target_entities = tuple(_clean_list(payload.get("target_entities", ())))
    normalized_target_entities = target_entities
    rewritten = _clean_rewritten_question(str(payload.get("exploration_question", "")))
    if _is_temporal_order_question(question=question, route_hint=route_hint):
        temporal_targets = _temporal_targets(target_entities, raw_question=question)
        normalized_target_entities = temporal_targets
        if len(temporal_targets) >= 2:
            rewritten = _temporal_order_exploration_question(temporal_targets)
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
    return ExplorationQuestionRewrite(
        exploration_question=rewritten,
        focus_points=focus_points,
        target_entities=normalized_target_entities or tuple(_temporal_targets((), raw_question=question)),
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
        "- Do not copy candidate answer values into exploration_question, even paraphrased as alternatives. Put useful "
        "candidate entities only in target_entities metadata.\n"
        "- exploration_question must be an open observation request, not a checklist, detector, or comparison among "
        "candidate answers. Extract the discriminative facts, entities, events, attributes, and temporal relations "
        "that should be observed.\n"
        "- For main-idea questions, ask for the overall topic, main subject, time span, narrative structure, "
        "major phases, causes, and consequences only when they are relevant to the question.\n"
        "- For temporal-order questions, extract the unique events/entities into target_entities metadata, but keep "
        "exploration_question as an open request to describe the video's actual presentation flow. Do not list those "
        "targets inside exploration_question.\n"
        "- For local factual questions, ask what visible/audio/text evidence is present in the relevant window.\n"
        "- Keep the exploration question concise but complete enough to guide search and segment inspection.\n\n"
        "Example 1 input:\n"
        "Question: What's the main idea of the video?\n"
        "Options:\n"
        "A. A single unrelated detail\n"
        "B. Why Entity X changes over time\n"
        "C. A brief side event\n"
        "D. How Entity X develops across the video.\n"
        "Example 1 output:\n"
        '{"exploration_question":"Describe the overall topic and narrative structure of the video. Identify the '
        'main subject, time span, major phases, and relevant causes or consequences.","focus_points":["overall topic",'
        '"main subject","time span","narrative phases"],"target_entities":["Entity X"]}\n\n'
        "Example 2 input:\n"
        "Question: In what order does the presenter discuss four items?\n"
        "Options:\n"
        'A. "Item Alpha", "Item Beta", "Item Gamma", "Item Delta".\n'
        'B. "Item Gamma", "Item Alpha", "Item Delta", "Item Beta".\n'
        "Example 2 output:\n"
        '{"exploration_question":"Describe the video segment by segment. Record the actual named items, '
        'onscreen text, narration, and scene transitions in the order they appear, with timestamps when possible; '
        'focus on concrete observations rather than conclusions.","focus_points":["presentation order",'
        '"timestamp evidence","item identification"],"target_entities":["Item Alpha","Item Beta","Item Gamma",'
        '"Item Delta"]}\n\n'
        f"Route hint: {route_hint or '(none)'}\n"
        f"Input question:\n{question}\n"
    )


def _fallback_rewrite(*, question: str, route_hint: str = "") -> str:
    semantic = _semantic_question(question)
    quoted = _quoted_targets(question)
    lowered = semantic.lower()
    if len(quoted) >= 2 and any(term in lowered for term in ["order", "sequence", "present", "shown"]):
        return _temporal_order_exploration_question(_sort_unique_targets(quoted[:8]))
    if any(term in lowered for term in ["main idea", "mainly about", "overall", "topic", "theme"]):
        return (
            "Describe the overall topic and narrative structure of the video. Identify the main subject, time span, "
            "major phases, and relevant causes or consequences."
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


def _is_temporal_order_question(*, question: str, route_hint: str = "") -> bool:
    lowered = f"{route_hint} {_semantic_question(question)}".lower()
    return any(term in lowered for term in ["temporal_order", "order", "sequence", "present", "shown", "before", "after"])


def _temporal_targets(target_entities: tuple[str, ...], *, raw_question: str) -> tuple[str, ...]:
    targets = list(target_entities)
    if len(targets) < 2:
        targets = _quoted_targets(raw_question)
    return tuple(_sort_unique_targets(targets))


def _sort_unique_targets(targets: list[str]) -> list[str]:
    seen = set()
    unique = []
    for target in targets:
        cleaned = " ".join(str(target or "").split()).strip(" .")
        key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
        if cleaned and key and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return sorted(unique, key=lambda item: re.sub(r"^(the|a|an)\s+", "", item.lower()))[:8]


def _temporal_order_exploration_question(targets: list[str] | tuple[str, ...]) -> str:
    return (
        "Describe the video segment by segment. Record the actual artworks, sculptures, onscreen text, narration, "
        "and scene transitions in the order they appear, with timestamps when possible; focus on concrete "
        "observations rather than conclusions."
    )


def _leaks_option_surface(rewritten: str, raw_question: str) -> bool:
    text = str(rewritten or "")
    if re.search(r"\b(?:option|choice|answer)\s*[A-H]\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"(?m)^\s*[A-H][\).:-]\s+\S+", text):
        return True
    raw_options = _option_texts(raw_question)
    normalized_text = _normalized_surface(text)
    return any(_normalized_option_leaks(option, normalized_text=normalized_text) for option in raw_options)


def _option_texts(question: str) -> list[str]:
    options = []
    for line in str(question or "").splitlines():
        match = re.match(r"^\s*[A-H][\).:-]\s*(\S.*)$", line.strip(), flags=re.IGNORECASE)
        if match:
            options.append(" ".join(match.group(1).split()).strip())
    return options


def _normalized_option_leaks(option: str, *, normalized_text: str) -> bool:
    normalized_option = _normalized_surface(option)
    if not normalized_option:
        return False
    if normalized_option in normalized_text:
        return True
    option_terms = [term for term in normalized_option.split() if len(term) >= 4]
    if len(option_terms) >= 2 and all(re.search(rf"\b{re.escape(term)}\b", normalized_text) for term in option_terms):
        return True
    return False


def _normalized_surface(text: str) -> str:
    normalized = re.sub(r"(?<!^)\b[A-H][\).:-]\s+", " ", str(text or ""), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


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
