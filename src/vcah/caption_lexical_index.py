from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Mapping, Sequence

from vcah.caption_schema import (
    CaptionHitV1,
    CaptionPassageV1,
    passage_in_segments,
    passage_from_dict,
    stable_digest,
)
from vcah.caption_store import resolve_caption_passages_path


TOKENIZER_VERSION = "unicode-word-cjk-bigram-v1"
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class CaptionLexicalIndex:
    def __init__(
        self,
        passages: Sequence[CaptionPassageV1],
        *,
        config_digest: str,
        k1: float = 1.5,
        b: float = 0.75,
        phrase_bonus: float = 1.5,
    ) -> None:
        self.passages = tuple(passages)
        self.config_digest = str(config_digest)
        self.k1 = float(k1)
        self.b = float(b)
        self.phrase_bonus = float(phrase_bonus)
        self._tokens = tuple(tokenize_caption_text(passage.text) for passage in self.passages)
        self._term_counts = tuple(Counter(tokens) for tokens in self._tokens)
        self._document_lengths = tuple(len(tokens) for tokens in self._tokens)
        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._document_frequency = document_frequency
        self.index_digest = stable_digest(
            {
                "config_digest": self.config_digest,
                "tokenizer_version": TOKENIZER_VERSION,
                "k1": self.k1,
                "b": self.b,
                "phrase_bonus": self.phrase_bonus,
                "passages": [asdict(passage) for passage in self.passages],
            }
        )

    @classmethod
    def from_jsonl(cls, path: Path, *, config_digest: str | None = None) -> "CaptionLexicalIndex":
        path = Path(path)
        passages = tuple(
            passage_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        digest = config_digest or _digest_from_passage_path(path)
        return cls(passages, config_digest=digest)

    @classmethod
    def from_asset_root(
        cls,
        asset_root: Path,
        *,
        config_digest: str | None = None,
    ) -> "CaptionLexicalIndex":
        path, resolved_digest = resolve_caption_passages_path(
            asset_root,
            config_digest=config_digest,
        )
        return cls.from_jsonl(path, config_digest=resolved_digest)

    def search(
        self,
        queries: Sequence[str],
        *,
        top_k: int = 12,
        time_range: tuple[float, float] | None = None,
        segment_ids: Sequence[str] = (),
        expand_neighbors: int = 0,
        per_caption_limit: int = 3,
        temporal_iou_threshold: float = 0.9,
    ) -> tuple[CaptionHitV1, ...]:
        normalized_queries = tuple(
            dict.fromkeys(normalize_caption_query(query) for query in queries if normalize_caption_query(query))
        )[:5]
        if not normalized_queries or not self.passages:
            return ()
        allowed = tuple(
            index
            for index, passage in enumerate(self.passages)
            if _in_time_range(passage, time_range)
            and passage_in_segments(passage, segment_ids)
        )
        scores: dict[int, float] = {}
        matched_queries: dict[int, list[str]] = {}
        for query in normalized_queries:
            query_tokens = tokenize_caption_text(query)
            if not query_tokens:
                continue
            for index in allowed:
                score = self._bm25_score(index, query_tokens)
                if query in normalize_caption_query(self.passages[index].text):
                    score += self.phrase_bonus
                if score <= 0.0:
                    continue
                scores[index] = scores.get(index, 0.0) + score
                matched_queries.setdefault(index, []).append(query)
        ordered = sorted(
            scores,
            key=lambda index: (
                -scores[index],
                self.passages[index].virtual_start_sec,
                self.passages[index].passage_id,
            ),
        )
        selected: list[int] = []
        caption_counts: Counter[str] = Counter()
        limit = max(1, int(top_k))
        for index in ordered:
            passage = self.passages[index]
            if caption_counts[passage.caption_id] >= max(1, int(per_caption_limit)):
                continue
            if any(
                _interval_iou(passage, self.passages[existing]) >= float(temporal_iou_threshold)
                for existing in selected
            ):
                continue
            selected.append(index)
            caption_counts[passage.caption_id] += 1
            if len(selected) >= limit:
                break

        hits = [
            self._hit(
                index,
                rank=rank,
                score=scores[index],
                metadata={"matched_queries": matched_queries.get(index, ())},
            )
            for rank, index in enumerate(selected, start=1)
        ]
        if expand_neighbors > 0:
            hits = self._with_neighbors(
                hits,
                distance=int(expand_neighbors),
                time_range=time_range,
                segment_ids=segment_ids,
            )
        return tuple(
            CaptionHitV1(**{**asdict(hit), "rank": rank})
            for rank, hit in enumerate(hits, start=1)
        )

    def query_fingerprint(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
        time_range: tuple[float, float] | None,
        expand_neighbors: int,
        segment_ids: Sequence[str] = (),
    ) -> str:
        return stable_digest(
            {
                "index_digest": self.index_digest,
                "queries": [normalize_caption_query(query) for query in queries],
                "top_k": int(top_k),
                "time_range": list(time_range) if time_range else None,
                "expand_neighbors": int(expand_neighbors),
                "segment_ids": sorted(
                    {str(item).strip() for item in segment_ids if str(item).strip()}
                ),
            }
        )

    def save_manifest(self, asset_root: Path) -> Path:
        path = Path(asset_root) / "captions" / "lexical" / f"index.{self.config_digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "index_digest": self.index_digest,
                    "config_digest": self.config_digest,
                    "tokenizer_version": TOKENIZER_VERSION,
                    "passage_count": len(self.passages),
                    "k1": self.k1,
                    "b": self.b,
                    "phrase_bonus": self.phrase_bonus,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _bm25_score(self, index: int, query_tokens: Sequence[str]) -> float:
        if not self.passages or not query_tokens:
            return 0.0
        score = 0.0
        counts = self._term_counts[index]
        document_length = self._document_lengths[index]
        average = max(self._average_length, 1.0)
        for term, query_frequency in Counter(query_tokens).items():
            frequency = counts.get(term, 0)
            if frequency <= 0:
                continue
            document_frequency = self._document_frequency.get(term, 0)
            inverse_frequency = math.log(
                1.0 + (len(self.passages) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * document_length / average
            )
            score += query_frequency * inverse_frequency * frequency * (self.k1 + 1.0) / denominator
        return score

    def _hit(
        self,
        index: int,
        *,
        rank: int,
        score: float,
        metadata: Mapping[str, Any],
    ) -> CaptionHitV1:
        passage = self.passages[index]
        return CaptionHitV1(
            passage_id=passage.passage_id,
            caption_id=passage.caption_id,
            rank=rank,
            lexical_score=score,
            dense_score=None,
            fused_score=score,
            virtual_start_sec=passage.virtual_start_sec,
            virtual_end_sec=passage.virtual_end_sec,
            wall_clock_begin=_optional_text(passage.metadata.get("wall_clock_begin")),
            wall_clock_end=_optional_text(passage.metadata.get("wall_clock_end")),
            text=passage.text,
            interval_precision=str(passage.metadata.get("interval_precision", "chunk")),
            source_pointer=f"caption://{self.config_digest}/{passage.passage_id}",
            metadata={"index_digest": self.index_digest, **dict(metadata)},
        )

    def _with_neighbors(
        self,
        hits: Sequence[CaptionHitV1],
        *,
        distance: int,
        time_range: tuple[float, float] | None,
        segment_ids: Sequence[str],
    ) -> list[CaptionHitV1]:
        passage_index = {passage.passage_id: index for index, passage in enumerate(self.passages)}
        by_caption_ordinal = {
            (passage.caption_id, passage.ordinal): index
            for index, passage in enumerate(self.passages)
        }
        expanded: list[CaptionHitV1] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.passage_id not in seen:
                expanded.append(hit)
                seen.add(hit.passage_id)
            source = self.passages[passage_index[hit.passage_id]]
            for offset in range(-distance, distance + 1):
                if offset == 0:
                    continue
                neighbor_index = by_caption_ordinal.get((source.caption_id, source.ordinal + offset))
                if neighbor_index is None:
                    continue
                neighbor = self.passages[neighbor_index]
                if neighbor.passage_id in seen:
                    continue
                if not _in_time_range(neighbor, time_range) or not passage_in_segments(
                    neighbor,
                    segment_ids,
                ):
                    continue
                expanded.append(
                    self._hit(
                        neighbor_index,
                        rank=len(expanded) + 1,
                        score=max(0.0, hit.fused_score * 0.5),
                        metadata={
                            "neighbor_of": hit.passage_id,
                            "candidate_only": True,
                        },
                    )
                )
                seen.add(neighbor.passage_id)
        return expanded


def normalize_caption_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def tokenize_caption_text(value: str) -> tuple[str, ...]:
    normalized = normalize_caption_query(value)
    words = WORD_RE.findall(normalized)
    cjk = CJK_RE.findall(normalized)
    cjk_bigrams = ["".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1))]
    return tuple(words + cjk + cjk_bigrams)


def render_caption_hits(hits: Sequence[CaptionHitV1], *, detail_limit: int = 5, char_limit: int = 4000) -> str:
    limit = max(1, int(char_limit))
    detail_count = min(len(hits), max(0, int(detail_limit)))
    legacy_lines = [f"Caption search: {len(hits)} hits"]
    for index, hit in enumerate(hits, start=1):
        legacy_lines.append(
            f"{index}. [{_clock(hit.virtual_start_sec)}-{_clock(hit.virtual_end_sec)}] "
            f"score={hit.fused_score:.4f} passage={hit.passage_id}"
        )
        if index <= detail_count:
            legacy_lines.append(f'   "{hit.text[:800]}"')
    if hits:
        legacy_lines.append("Suggested next step: inspect the highest-ranked intervals visually.")
    legacy_rendered = "\n".join(legacy_lines)
    later_count = max(0, len(hits) - detail_count)
    if later_count == 0 or len(legacy_rendered) >= limit:
        return legacy_rendered[:limit]

    snippet_limit = min(320, max(0, ((limit - len(legacy_rendered)) // later_count) - 6))
    if snippet_limit == 0:
        return legacy_rendered[:limit]

    lines = [f"Caption search: {len(hits)} hits"]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"{index}. [{_clock(hit.virtual_start_sec)}-{_clock(hit.virtual_end_sec)}] "
            f"score={hit.fused_score:.4f} passage={hit.passage_id}"
        )
        if index <= detail_count:
            lines.append(f'   "{hit.text[:800]}"')
        else:
            excerpt = _balanced_excerpt(hit.text, snippet_limit)
            if excerpt:
                lines.append(f'   "{excerpt}"')
    if hits:
        lines.append("Suggested next step: inspect the highest-ranked intervals visually.")
    return "\n".join(lines)[:limit]


def _balanced_excerpt(text: str, limit: int) -> str:
    value = str(text).strip()
    capped = max(0, int(limit))
    if not value or capped == 0:
        return ""
    if len(value) <= capped:
        return value
    separator = " ... "
    if capped <= len(separator) + 2:
        return value[:capped]
    content_limit = capped - len(separator)
    head_limit = content_limit // 2
    tail_limit = content_limit - head_limit
    return f"{value[:head_limit]}{separator}{value[-tail_limit:]}"


def _in_time_range(passage: CaptionPassageV1, time_range: tuple[float, float] | None) -> bool:
    if time_range is None:
        return True
    start, end = sorted((float(time_range[0]), float(time_range[1])))
    return passage.virtual_end_sec > start and passage.virtual_start_sec < end


def _interval_iou(left: CaptionPassageV1, right: CaptionPassageV1) -> float:
    intersection = max(
        0.0,
        min(left.virtual_end_sec, right.virtual_end_sec)
        - max(left.virtual_start_sec, right.virtual_start_sec),
    )
    union = max(left.virtual_end_sec, right.virtual_end_sec) - min(
        left.virtual_start_sec,
        right.virtual_start_sec,
    )
    return intersection / union if union > 0.0 else 0.0


def _digest_from_passage_path(path: Path) -> str:
    prefix = "passages."
    suffix = ".jsonl"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Cannot infer caption config digest from {path}")
    return name[len(prefix) : -len(suffix)]


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _clock(seconds: float) -> str:
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
