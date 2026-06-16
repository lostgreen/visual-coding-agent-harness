"""Lexical fallback discriminator derivation for grounding targets."""

from __future__ import annotations

import re
from typing import Mapping


NORMALIZATION_HINTS = {
    "rise": {"rise", "rises", "rose", "rising", "become powerful", "became powerful"},
    "fall": {"fall", "falls", "fell", "collapse", "collapsed", "broke apart", "dissolution"},
    "order": {"order", "sequence", "first", "second", "third", "fourth"},
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "with",
    }
)


def derive_discriminators_lexical(options_by_id: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    candidates_by_option: dict[str, list[str]] = {}
    candidate_sets: dict[str, set[str]] = {}
    for option_id, option_text in options_by_id.items():
        key = str(option_id or "").strip().upper()[:1]
        if not key:
            continue
        candidates = _phrase_candidates(str(option_text or ""))
        candidates_by_option[key] = candidates
        candidate_sets[key] = set(candidates)

    result: dict[str, tuple[str, ...]] = {}
    for option_id, candidates in candidates_by_option.items():
        other_candidates = set().union(*(values for key, values in candidate_sets.items() if key != option_id))
        unique = [candidate for candidate in candidates if candidate not in other_candidates]
        result[option_id] = tuple(_ranked_unique(unique)[:3])
    return result


def _phrase_candidates(text: str) -> list[str]:
    tokens = _normalized_tokens(text)
    candidates: list[str] = []
    for width in range(1, min(4, len(tokens)) + 1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[start : start + width])
            if phrase:
                candidates.append(phrase)
    return _ranked_unique(candidates)


def _normalized_tokens(text: str) -> list[str]:
    lowered = str(text or "").lower()
    for canonical, variants in NORMALIZATION_HINTS.items():
        for variant in sorted(variants, key=len, reverse=True):
            lowered = re.sub(rf"\b{re.escape(variant)}\b", canonical, lowered)
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return [token for token in normalized.split() if token and token not in _STOPWORDS]


def _ranked_unique(candidates: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return sorted(unique, key=_candidate_rank)


def _candidate_rank(candidate: str) -> tuple[int, int, str]:
    hinted = candidate in NORMALIZATION_HINTS
    width = len(candidate.split())
    return (0 if hinted else 1, width, candidate)
