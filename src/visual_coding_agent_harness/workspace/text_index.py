"""Small inverted index for cold ASR/OCR beat text."""

from __future__ import annotations

from collections import defaultdict
import json
from math import log
from pathlib import Path
import re
from typing import Literal, Sequence

from visual_coding_agent_harness.workspace.visual_index import BeatHit


TextModality = Literal["asr", "ocr"]
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class InvertedIndex:
    def __init__(self) -> None:
        self._documents: dict[str, dict[str, str]] = {}
        self._postings: dict[str, dict[str, dict[str, int]]] = {
            "asr": defaultdict(dict),
            "ocr": defaultdict(dict),
        }
        self._tokens: dict[str, dict[str, tuple[str, ...]]] = {}

    def add(self, beat_id: str, text: str, *, modality: TextModality) -> None:
        if modality not in ("asr", "ocr"):
            raise ValueError("modality must be 'asr' or 'ocr'")
        beat_id = str(beat_id)
        text = str(text or "")
        self._documents.setdefault(beat_id, {})[modality] = text
        tokens = _tokenize(text)
        self._tokens.setdefault(beat_id, {})[modality] = tokens
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token, count in counts.items():
            self._postings[modality][token][beat_id] = count

    def search(self, query: str, *, modality: Sequence[str] = ("asr", "ocr")) -> tuple[BeatHit, ...]:
        modalities = tuple(item for item in modality if item in ("asr", "ocr"))
        if not modalities:
            return ()
        phrase = _quoted_phrase(query)
        if phrase is not None:
            return self._search_phrase(phrase, modalities=modalities)
        return self._search_terms(query, modalities=modalities)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"documents": self._documents}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        documents = payload.get("documents", {}) if isinstance(payload, dict) else {}
        for beat_id, by_modality in documents.items():
            if not isinstance(by_modality, dict):
                continue
            for modality, text in by_modality.items():
                if modality in ("asr", "ocr"):
                    index.add(str(beat_id), str(text or ""), modality=modality)  # type: ignore[arg-type]
        return index

    def _search_terms(self, query: str, *, modalities: Sequence[str]) -> tuple[BeatHit, ...]:
        terms = _tokenize(query)
        if not terms:
            return ()
        beat_scores: dict[str, float] = defaultdict(float)
        total_docs = max(1, len(self._documents))
        for modality in modalities:
            postings_by_term = self._postings[modality]
            for term in terms:
                postings = postings_by_term.get(term, {})
                if not postings:
                    continue
                weight = log(1.0 + total_docs / max(1, len(postings)))
                for beat_id, tf in postings.items():
                    beat_scores[beat_id] += float(tf) * weight
        return _rank_hits(beat_scores)

    def _search_phrase(self, phrase: str, *, modalities: Sequence[str]) -> tuple[BeatHit, ...]:
        phrase_tokens = _tokenize(phrase)
        if not phrase_tokens:
            return ()
        scores: dict[str, float] = defaultdict(float)
        for beat_id, by_modality in self._tokens.items():
            for modality in modalities:
                tokens = by_modality.get(modality, ())
                if _contains_phrase(tokens, phrase_tokens):
                    scores[beat_id] += float(len(phrase_tokens))
        return _rank_hits(scores)


def _rank_hits(scores: dict[str, float]) -> tuple[BeatHit, ...]:
    return tuple(
        BeatHit(beat_id=beat_id, score=float(score), modality="text")
        for beat_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0.0
    )


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text or "")))


def _quoted_phrase(query: str) -> str | None:
    match = re.search(r'"([^"]+)"', str(query or ""))
    return match.group(1) if match else None


def _contains_phrase(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    if len(phrase_tokens) > len(tokens):
        return False
    width = len(phrase_tokens)
    return any(tuple(tokens[index : index + width]) == tuple(phrase_tokens) for index in range(0, len(tokens) - width + 1))
