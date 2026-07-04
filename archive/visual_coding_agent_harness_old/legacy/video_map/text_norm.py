"""Small text normalization helpers for indexed evidence matching."""

from __future__ import annotations

import re
from typing import Iterable


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "then",
        "this",
        "through",
        "to",
        "video",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

_IRREGULAR_STEMS = {
    "arose": "rise",
    "broke": "break",
    "broken": "break",
    "fell": "fall",
    "fallen": "fall",
    "rose": "rise",
    "risen": "rise",
}


def tokens(text: str, *, drop_stopwords: bool = True, stem: bool = True) -> list[str]:
    normalized: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", str(text or "")):
        token = raw.lower()
        if drop_stopwords and token in STOPWORDS:
            continue
        if stem:
            token = stem_token(token)
            if drop_stopwords and token in STOPWORDS:
                continue
        if token:
            normalized.append(token)
    return normalized


def unique_tokens(text: str, *, drop_stopwords: bool = True, stem: bool = True) -> set[str]:
    return set(tokens(text, drop_stopwords=drop_stopwords, stem=stem))


def stem_token(token: str) -> str:
    value = str(token or "").lower()
    if value in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[value]
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("es") and not value.endswith(("ses", "xes")):
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value


def token_spans(text: str, *, drop_stopwords: bool = True, stem: bool = True) -> Iterable[tuple[str, int, int]]:
    for match in re.finditer(r"[A-Za-z0-9]+", str(text or "")):
        token = match.group(0).lower()
        if drop_stopwords and token in STOPWORDS:
            continue
        if stem:
            token = stem_token(token)
            if drop_stopwords and token in STOPWORDS:
                continue
        if token:
            yield token, int(match.start()), int(match.end())
