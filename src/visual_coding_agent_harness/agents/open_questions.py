"""Utilities for turning MCQ prompts into open exploratory questions."""

from __future__ import annotations

import re


_OPTION_LINE_RE = re.compile(r"^\s*[A-H][\).:-]\s+\S.*$", re.IGNORECASE)
_ANSWER_INSTRUCTION_RE = re.compile(
    r"\b(?:answer|respond|reply|select|choose|start)\b[^\n.]*\b(?:option\s+)?(?:letter|[A-H](?:\s*/\s*[A-H]){1,7}|[A-H](?:\s*,\s*[A-H]){1,7})\b[^\n.]*[.]?",
    re.IGNORECASE,
)


def exploration_question(question: str, route_hint: str = "") -> str:
    """Return a model-facing open question with MCQ choices removed."""
    semantic_question = _semantic_question(question)
    parts = [semantic_question, "Do not choose an option."]
    hint = " ".join(str(route_hint or "").split())
    if hint:
        parts.append(hint)
    return " ".join(part for part in parts if part).strip()


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
