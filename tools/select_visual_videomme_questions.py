from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path


HTML_RE = re.compile(r"<[^>]+>")
SRT_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->")
WORD_RE = re.compile(r"[a-z0-9]+")

HIGH_LABELS = {"asr_unique_strong", "asr_unique_video", "proper_unique_likely"}
VISUAL_TASKS = {
    "Action Recognition",
    "Attribute Perception",
    "Counting Problem",
    "OCR Problems",
    "Spatial Perception",
    "Spatial Reasoning",
}
VISUAL_RE = re.compile(
    r"\b(color|colour|wearing|holding|doing|how many|number|count|appears?|"
    r"shown|visible|located|position|where|text|word|written|sign|look like)\b",
    re.I,
)


def norm(text: object) -> str:
    return " ".join(WORD_RE.findall(html.unescape(str(text)).lower()))


def srt_text(path: Path) -> str:
    if not path.exists():
        return ""
    lines: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.isdigit() or SRT_TIME_RE.match(line):
            continue
        lines.append(HTML_RE.sub(" ", line))
    return norm(" ".join(lines))


def parse_options(value: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for part in value.split(" | "):
        match = re.match(r"\s*([A-D])\.\s*(.*)", part.strip())
        if match:
            parsed.append((match.group(1), match.group(2).strip()))
    return parsed


def visual_score(row: dict[str, str], answer_leak: bool, option_leak_count: int) -> int:
    question = row["question"]
    score = 0
    patterns = [
        (r"color|colour|wearing", 5),
        (r"holding|doing|action", 4),
        (r"how many|number|count", 3),
        (r"text|word|written|sign", 4),
        (r"where|located|position", 3),
        (r"shown|visible|appears?|look like", 2),
    ]
    for pattern, points in patterns:
        if re.search(pattern, question, re.I):
            score += points
    if row["task_type"] in VISUAL_TASKS:
        score += 3
    if row["label"].startswith("asr_unique"):
        score += 3
    if answer_leak:
        score -= 10
    if option_leak_count >= 2:
        score -= 4
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = analysis_dir / "videomme_unique_reference_candidates.csv"
    rows = list(csv.DictReader(candidates_path.open(encoding="utf-8")))
    subtitle_cache: dict[str, str] = {}
    scored: list[dict[str, object]] = []

    for row in rows:
        if row["label"] not in HIGH_LABELS:
            continue
        if row["task_type"] not in VISUAL_TASKS and not VISUAL_RE.search(row["question"]):
            continue

        video_id = row["videoID"]
        subtitle = subtitle_cache.setdefault(video_id, srt_text(dataset_root / "subtitle" / f"{video_id}.srt"))
        options = parse_options(row["options"])
        answer_text = next((text for letter, text in options if letter == row["answer"]), "")
        answer_norm = norm(answer_text)
        answer_leak = bool(answer_norm and len(answer_norm) >= 5 and answer_norm in subtitle)

        leaked_options: list[str] = []
        for letter, text in options:
            option_norm = norm(text)
            if len(option_norm) >= 5 and option_norm in subtitle:
                leaked_options.append(letter)

        score = visual_score(row, answer_leak, len(leaked_options))
        if score < 7 or answer_leak:
            continue

        anchor = row["anchor"] if row["label"].startswith("asr") else row["proper_anchor"]
        scored.append(
            {
                "score": score,
                "question_id": row["question_id"],
                "videoID": video_id,
                "duration": row["duration"],
                "task_type": row["task_type"],
                "label": row["label"],
                "anchor": anchor,
                "answer": row["answer"],
                "answer_text": answer_text,
                "leaked_options": ",".join(leaked_options),
                "question": row["question"],
                "options": row["options"],
                "mp4_path": str(dataset_root / "video" / f"{video_id}.mp4"),
                "subtitle_path": str(dataset_root / "subtitle" / f"{video_id}.srt"),
                "url": row["url"],
            }
        )

    scored.sort(
        key=lambda item: (
            -int(item["score"]),
            0 if str(item["label"]).startswith("asr") else 1,
            str(item["duration"]),
            str(item["question_id"]),
        )
    )

    # Keep anchors/topics diverse so they stay unambiguous when concatenated.
    selected: list[dict[str, object]] = []
    used_videos: set[str] = set()
    used_anchor_heads: set[str] = set()
    for item in scored:
        anchor_tokens = str(item["anchor"]).split()
        anchor_head = " ".join(anchor_tokens[:2]) if anchor_tokens else str(item["question_id"])
        if str(item["videoID"]) in used_videos:
            continue
        if anchor_head in used_anchor_heads:
            continue
        selected.append(item)
        used_videos.add(str(item["videoID"]))
        used_anchor_heads.add(anchor_head)
        if len(selected) >= args.limit:
            break

    csv_path = output_dir / "visual_unique_shortlist.csv"
    if selected:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)

    by_task = defaultdict(int)
    by_label = defaultdict(int)
    for item in selected:
        by_task[str(item["task_type"])] += 1
        by_label[str(item["label"])] += 1

    summary = {
        "selected": len(selected),
        "by_task": dict(sorted(by_task.items())),
        "by_label": dict(sorted(by_label.items())),
        "csv": str(csv_path),
    }
    (output_dir / "visual_unique_shortlist_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for item in selected[:30]:
        print(
            "\t".join(
                str(item[key]).replace("\n", " ")[:180]
                for key in [
                    "score",
                    "question_id",
                    "videoID",
                    "duration",
                    "task_type",
                    "label",
                    "anchor",
                    "answer",
                    "answer_text",
                    "leaked_options",
                    "question",
                ]
            )
        )
    print("summary=" + json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
