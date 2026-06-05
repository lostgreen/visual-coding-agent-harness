"""Output quality heuristics for answer-facing visual observations."""

from __future__ import annotations

import re
from collections import Counter


UNSUPPORTED_CONFIDENCE_SIGNAL = "unsupported"
DEGENERATE_CONFIDENCE_SIGNAL = "degenerate"

_UNSUPPORTED_PATTERNS = [
    re.compile(
        r"limitation:\s*(?:no direct evidence|not visually verifiable|not applicable|cannot confirm|no visual evidence)",
        flags=re.IGNORECASE,
    ),
    re.compile(r"limitations?:\s*the video does not", flags=re.IGNORECASE),
    re.compile(r"\bi (?:cannot|can't) (?:see|confirm|verify)\b", flags=re.IGNORECASE),
]


def confidence_signal_from_text(text: str) -> str:
    """Return a compact signal when the claim itself says it is unsupported."""

    return UNSUPPORTED_CONFIDENCE_SIGNAL if is_unsupported_claim(text) else ""


def is_unsupported_claim(text: str) -> bool:
    """Detect local-worker text that explicitly denies direct visual support."""

    normalized = str(text or "")
    return any(pattern.search(normalized) for pattern in _UNSUPPORTED_PATTERNS)


def is_degenerate(text: str, *, n: int = 8, threshold: int = 4) -> tuple[bool, str]:
    """Detect repeated n-token shingles in VLM output.

    Returns `(is_degenerate, fingerprint)` where the fingerprint is the repeated
    shingle. Shorter text is treated as non-degenerate.
    """

    tokens = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    if n <= 0 or threshold <= 1 or len(tokens) < n:
        return False, ""
    shingles = [" ".join(tokens[index : index + n]) for index in range(0, len(tokens) - n + 1)]
    if not shingles:
        return False, ""
    shingle, count = Counter(shingles).most_common(1)[0]
    if count >= threshold:
        return True, shingle
    return False, ""
