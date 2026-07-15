from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


HTML_RE = re.compile(r"<[^>]+>")
SRT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
WORD_RE = re.compile(r"[a-z0-9]+")
QUOTED_RE = re.compile(r"['\"]([^'\"]{4,120})['\"]")
CAPITALIZED_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'&.-]+|[A-Z]{2,})(?:\s+(?:of|and|the|to|in|for|"
    r"de|da|van|von|[A-Z][A-Za-z0-9'&.-]+|[A-Z]{2,})){0,5}"
)
RAW_WORD_RE = re.compile(r"[A-Za-z0-9'&.-]+")
CONNECTORS = {"of", "and", "the", "to", "in", "for", "de", "da", "van", "von"}
SENTENCE_STARTERS = {
    "according",
    "after",
    "as",
    "based",
    "before",
    "during",
    "given",
    "how",
    "in",
    "throughout",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

STOPWORDS = {
    "a",
    "about",
    "above",
    "according",
    "accordance",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "during",
    "each",
    "for",
    "from",
    "given",
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
    "mainly",
    "many",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "other",
    "shown",
    "she",
    "should",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
    "would",
    "you",
}

GENERIC_CONTENT = {
    "action",
    "actions",
    "answer",
    "background",
    "beginning",
    "camera",
    "character",
    "characters",
    "clip",
    "color",
    "colors",
    "content",
    "described",
    "displayed",
    "event",
    "events",
    "first",
    "following",
    "footage",
    "happen",
    "happened",
    "happens",
    "image",
    "last",
    "location",
    "main",
    "make",
    "man",
    "men",
    "mentioned",
    "object",
    "objects",
    "person",
    "people",
    "scene",
    "scenes",
    "screen",
    "second",
    "show",
    "showing",
    "shows",
    "statement",
    "statements",
    "thing",
    "things",
    "time",
    "times",
    "video",
    "woman",
    "women",
}

GLOBAL_CUE_RE = re.compile(
    r"\b(main|overall|primarily|theme|purpose|summariz|entire|whole video|"
    r"best describe|mainly about|following statement|not mentioned)\b",
    re.I,
)

LOCAL_CUE_RE = re.compile(
    r"\b(when|after|before|during|while|at the moment|in the scene|"
    r"demonstrating|shown|displayed|appears|mentioned|says|holding|wearing)\b",
    re.I,
)


def normalize(text: str) -> str:
    text = html.unescape(str(text))
    text = HTML_RE.sub(" ", text)
    return " ".join(WORD_RE.findall(text.lower()))


def srt_text(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.isdigit() or SRT_TIME_RE.match(line):
            continue
        lines.append(line)
    return normalize(" ".join(lines))


def content_tokens(tokens: Iterable[str]) -> list[str]:
    return [
        tok
        for tok in tokens
        if tok not in STOPWORDS and tok not in GENERIC_CONTENT and len(tok) >= 3
    ]


def question_phrases(question: str) -> set[str]:
    tokens = normalize(question).split()
    phrases: set[str] = set()
    for n in range(2, 7):
        for i in range(0, max(0, len(tokens) - n + 1)):
            gram = tokens[i : i + n]
            content = content_tokens(gram)
            if len(content) < 2:
                continue
            if len(" ".join(content)) < 8:
                continue
            if len(content) / len(gram) < 0.34:
                continue
            phrase = " ".join(gram)
            if len(phrase) >= 10:
                phrases.add(phrase)
    return phrases


def proper_phrases(question: str) -> set[str]:
    phrases: set[str] = set()
    for match in QUOTED_RE.finditer(question):
        norm = normalize(match.group(1))
        if len(content_tokens(norm.split())) >= 1 and len(norm) >= 4:
            phrases.add(norm)
    for match in CAPITALIZED_RE.finditer(question):
        raw_words = RAW_WORD_RE.findall(match.group(0).strip())
        while raw_words and raw_words[0].lower() in SENTENCE_STARTERS | CONNECTORS:
            raw_words = raw_words[1:]
        while raw_words and raw_words[-1].lower() in SENTENCE_STARTERS | CONNECTORS:
            raw_words = raw_words[:-1]
        if not raw_words:
            continue
        raw = " ".join(raw_words)
        norm = normalize(raw)
        toks = norm.split()
        if not toks:
            continue
        content = content_tokens(toks)
        if not content:
            continue
        properish = [
            word
            for word in raw_words
            if word.lower() not in CONNECTORS
            and word.lower() not in SENTENCE_STARTERS
            and (word[:1].isupper() or word.isupper())
        ]
        if len(toks) > 1 and len(properish) < 2:
            continue
        if len(toks) == 1 and (len(content[0]) < 5 or match.start() == 0):
            continue
        phrases.add(norm)
    return phrases


def ngram_hits(tokens: list[str], candidates: set[str]) -> dict[str, int]:
    hits: dict[str, int] = defaultdict(int)
    for n in range(2, 7):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in candidates:
                hits[phrase] += 1
    return hits


def row_options(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " | ".join(str(x) for x in value)
    try:
        return " | ".join(str(x) for x in list(value))  # numpy array
    except Exception:
        return str(value)


def choose_anchor(
    phrases: set[str],
    video_id: str,
    subtitle_occ: dict[str, dict[str, int]],
    question_video_count: dict[str, int],
) -> tuple[str, int, int, int]:
    best: tuple[tuple[int, int, int, int, int], str, int, int, int] | None = None
    for phrase in phrases:
        per_video = subtitle_occ.get(phrase, {})
        own_occ = int(per_video.get(video_id, 0))
        video_count = len(per_video)
        q_video_count = int(question_video_count.get(phrase, 0))
        content_len = len(content_tokens(phrase.split()))
        score = (
            1 if own_occ > 0 else 0,
            -video_count if video_count else -9999,
            -own_occ if own_occ else -9999,
            -q_video_count if q_video_count else -9999,
            content_len,
        )
        if best is None or score > best[0]:
            best = (score, phrase, own_occ, video_count, q_video_count)
    if best is None:
        return "", 0, 0, 0
    return best[1], best[2], best[3], best[4]


def choose_proper_anchor(
    phrases: set[str], question_video_count: dict[str, int]
) -> tuple[str, int]:
    if not phrases:
        return "", 0
    ranked = sorted(
        phrases,
        key=lambda phrase: (
            question_video_count.get(phrase, 9999),
            -len(content_tokens(phrase.split())),
            -len(phrase),
        ),
    )
    phrase = ranked[0]
    return phrase, int(question_video_count.get(phrase, 0))


def classify(
    question: str,
    task_type: str,
    subtitle_exists: bool,
    anchor: str,
    own_occ: int,
    subtitle_video_count: int,
    question_video_count: int,
    proper_anchor: str,
    proper_question_video_count: int,
) -> tuple[str, int, str]:
    globalish = bool(GLOBAL_CUE_RE.search(question)) or task_type == "Information Synopsis"
    localish = bool(LOCAL_CUE_RE.search(question))
    has_anchor = bool(anchor)

    if subtitle_exists and has_anchor and own_occ == 1 and subtitle_video_count == 1:
        score = 95 - (10 if globalish else 0)
        reason = "question phrase occurs exactly once in this video's subtitles and in no other subtitle"
        return "asr_unique_strong", score, reason

    if subtitle_exists and has_anchor and 1 <= own_occ <= 3 and subtitle_video_count == 1:
        score = 84 - (10 if globalish else 0)
        reason = "question phrase is subtitle-unique to this video, but repeats locally"
        return "asr_unique_video", score, reason

    if proper_anchor and proper_question_video_count == 1:
        score = 72 - (12 if globalish else 0)
        reason = "question has an explicit named or quoted anchor that is unique among VideoMME questions"
        return "proper_unique_likely", score, reason

    if has_anchor and question_video_count == 1 and localish and not globalish:
        score = 58
        reason = "question has a rare local-looking anchor, but it is not a named/quoted referent"
        return "question_unique_weak", score, reason

    if has_anchor and question_video_count <= 3 and localish and not globalish:
        return (
            "weak_unique_likely",
            55,
            "question anchor is rare and local-looking, but not uniquely supported by subtitles",
        )

    return "not_unique", 20 if globalish else 35, "generic/global question or no reliable unique anchor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = dataset_root / "videomme" / "test-00000-of-00001.parquet"
    subtitle_dir = dataset_root / "subtitle"
    video_dir = dataset_root / "video"

    df = pd.read_parquet(parquet_path)
    df["videoID"] = df["videoID"].astype(str)
    df["question_id"] = df["question_id"].astype(str)

    phrase_by_qid: dict[str, set[str]] = {}
    proper_by_qid: dict[str, set[str]] = {}
    phrase_video_sets: dict[str, set[str]] = defaultdict(set)
    proper_video_sets: dict[str, set[str]] = defaultdict(set)
    all_phrases: set[str] = set()
    for row in df.itertuples(index=False):
        phrases = question_phrases(str(row.question))
        proper = proper_phrases(str(row.question))
        phrase_by_qid[str(row.question_id)] = phrases
        proper_by_qid[str(row.question_id)] = proper
        all_phrases.update(phrases)
        for phrase in phrases:
            phrase_video_sets[phrase].add(str(row.videoID))
        for phrase in proper:
            proper_video_sets[phrase].add(str(row.videoID))

    subtitle_texts: dict[str, str] = {}
    for video_id in sorted(df["videoID"].unique()):
        path = subtitle_dir / f"{video_id}.srt"
        if path.exists():
            subtitle_texts[video_id] = srt_text(path)

    subtitle_occ: dict[str, dict[str, int]] = {}
    for video_id, text in subtitle_texts.items():
        hits = ngram_hits(text.split(), all_phrases)
        for phrase, count in hits.items():
            subtitle_occ.setdefault(phrase, {})[video_id] = count

    question_video_count = {k: len(v) for k, v in phrase_video_sets.items()}
    proper_question_video_count = {k: len(v) for k, v in proper_video_sets.items()}
    records: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        qid = str(row.question_id)
        vid = str(row.videoID)
        phrases = phrase_by_qid[qid]
        anchor, own_occ, subtitle_video_count, q_video_count = choose_anchor(
            phrases, vid, subtitle_occ, question_video_count
        )
        proper_anchor, proper_q_video_count = choose_proper_anchor(
            proper_by_qid[qid], proper_question_video_count
        )
        label, score, reason = classify(
            question=str(row.question),
            task_type=str(row.task_type),
            subtitle_exists=vid in subtitle_texts,
            anchor=anchor,
            own_occ=own_occ,
            subtitle_video_count=subtitle_video_count,
            question_video_count=q_video_count,
            proper_anchor=proper_anchor,
            proper_question_video_count=proper_q_video_count,
        )
        records.append(
            {
                "question_id": qid,
                "video_id": str(row.video_id),
                "videoID": vid,
                "duration": str(row.duration),
                "domain": str(row.domain),
                "sub_category": str(row.sub_category),
                "task_type": str(row.task_type),
                "label": label,
                "score": score,
                "anchor": anchor,
                "anchor_occurrences_in_video_subtitle": own_occ,
                "anchor_subtitle_video_count": subtitle_video_count,
                "anchor_question_video_count": q_video_count,
                "proper_anchor": proper_anchor,
                "proper_anchor_question_video_count": proper_q_video_count,
                "subtitle_exists": vid in subtitle_texts,
                "reason": reason,
                "question": str(row.question),
                "options": row_options(row.options),
                "answer": str(row.answer),
                "url": str(row.url),
            }
        )

    fieldnames = list(records[0].keys())
    csv_path = out_dir / "videomme_unique_reference_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    strong_records = [
        r
        for r in records
        if r["label"] in {"asr_unique_strong", "asr_unique_video", "proper_unique_likely"}
    ]
    strong_path = out_dir / "videomme_unique_reference_candidate_ids.txt"
    strong_path.write_text(
        "\n".join(str(r["question_id"]) for r in sorted(strong_records, key=lambda x: str(x["question_id"])))
        + "\n",
        encoding="utf-8",
    )

    video_rows: list[dict[str, object]] = []
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_video[str(record["videoID"])].append(record)
    high_labels = {"asr_unique_strong", "asr_unique_video", "proper_unique_likely"}
    asr_labels = {"asr_unique_strong", "asr_unique_video"}
    for video_id, group in sorted(by_video.items()):
        first = group[0]
        high = [r for r in group if r["label"] in high_labels]
        asr = [r for r in group if r["label"] in asr_labels]
        proper = [r for r in group if r["label"] == "proper_unique_likely"]
        weak = [r for r in group if r["label"] in {"question_unique_weak", "weak_unique_likely"}]
        video_rows.append(
            {
                "videoID": video_id,
                "video_id": first["video_id"],
                "duration": first["duration"],
                "domain": first["domain"],
                "sub_category": first["sub_category"],
                "url": first["url"],
                "mp4_path": str(video_dir / f"{video_id}.mp4"),
                "subtitle_path": str(subtitle_dir / f"{video_id}.srt"),
                "subtitle_exists": video_id in subtitle_texts,
                "candidate_count": len(high),
                "asr_candidate_count": len(asr),
                "proper_candidate_count": len(proper),
                "weak_candidate_count": len(weak),
                "candidate_question_ids": ",".join(str(r["question_id"]) for r in high),
                "asr_question_ids": ",".join(str(r["question_id"]) for r in asr),
                "proper_question_ids": ",".join(str(r["question_id"]) for r in proper),
                "weak_question_ids": ",".join(str(r["question_id"]) for r in weak),
            }
        )
    video_manifest_path = out_dir / "videomme_unique_reference_video_manifest.csv"
    with video_manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    summary = {
        "rows": len(records),
        "videos": int(df["videoID"].nunique()),
        "mp4_files": len(list(video_dir.glob("*.mp4"))),
        "subtitle_files": len(subtitle_texts),
        "label_counts": Counter(str(r["label"]) for r in records),
        "candidate_questions": len(strong_records),
        "candidate_videos": len({str(r["videoID"]) for r in strong_records}),
        "asr_candidate_questions": sum(1 for r in strong_records if r["label"] in asr_labels),
        "asr_candidate_videos": len(
            {str(r["videoID"]) for r in strong_records if r["label"] in asr_labels}
        ),
        "duration_by_label": {
            label: Counter(str(r["duration"]) for r in records if r["label"] == label)
            for label in sorted(set(str(r["label"]) for r in records))
        },
        "task_type_by_label": {
            label: Counter(str(r["task_type"]) for r in records if r["label"] == label).most_common()
            for label in sorted(set(str(r["label"]) for r in records))
        },
    }
    json_path = out_dir / "videomme_unique_reference_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def top_rows(label: str, limit: int = 12) -> list[dict[str, object]]:
        rows = [r for r in records if r["label"] == label]
        rows.sort(key=lambda r: (-int(r["score"]), str(r["duration"]), str(r["question_id"])))
        return rows[:limit]

    md_lines = [
        "# VideoMME Unique Reference Analysis",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Total questions: {len(records)}",
        f"- Unique videos: {df['videoID'].nunique()}",
        f"- MP4 files: {len(list(video_dir.glob('*.mp4')))}",
        f"- Subtitle files matched to metadata: {len(subtitle_texts)}",
        f"- CSV: `{csv_path}`",
        f"- Candidate IDs: `{strong_path}`",
        f"- Video manifest: `{video_manifest_path}`",
        f"- Summary JSON: `{json_path}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in Counter(str(r["label"]) for r in records).most_common():
        md_lines.append(f"- {label}: {count}")
    md_lines.extend(["", "## Video Selection Helpers", ""])
    top_videos = sorted(
        video_rows,
        key=lambda row: (
            -int(row["candidate_count"]),
            -int(row["asr_candidate_count"]),
            str(row["duration"]),
            str(row["videoID"]),
        ),
    )[:20]
    md_lines.append("| videoID | duration | candidates | asr | proper | question_ids |")
    md_lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for row in top_videos:
        md_lines.append(
            f"| {row['videoID']} | {row['duration']} | {row['candidate_count']} | "
            f"{row['asr_candidate_count']} | {row['proper_candidate_count']} | "
            f"{row['candidate_question_ids']} |"
        )
    md_lines.extend(["", "## Top Examples", ""])
    for label in [
        "asr_unique_strong",
        "asr_unique_video",
        "proper_unique_likely",
        "question_unique_weak",
        "not_unique",
    ]:
        md_lines.append(f"### {label}")
        rows = top_rows(label)
        if not rows:
            md_lines.append("")
            continue
        md_lines.append("| question_id | duration | task_type | anchor | question |")
        md_lines.append("| --- | --- | --- | --- | --- |")
        for r in rows:
            q = str(r["question"]).replace("|", "/")
            if len(q) > 130:
                q = q[:127] + "..."
            anchor_value = r["proper_anchor"] if label == "proper_unique_likely" else r["anchor"]
            anchor = str(anchor_value).replace("|", "/")
            md_lines.append(
                f"| {r['question_id']} | {r['duration']} | {r['task_type']} | {anchor} | {q} |"
            )
        md_lines.append("")
    md_path = out_dir / "videomme_unique_reference_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"rows={len(records)} videos={df['videoID'].nunique()} mp4={len(list(video_dir.glob('*.mp4')))} subtitles={len(subtitle_texts)}")
    print("label_counts=" + json.dumps(Counter(str(r["label"]) for r in records), sort_keys=True))
    print(f"csv={csv_path}")
    print(f"candidate_ids={strong_path}")
    print(f"video_manifest={video_manifest_path}")
    print(f"report={md_path}")


if __name__ == "__main__":
    main()
