import glob
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from typing import List, Dict
from openai import OpenAI
import time
import re

# =========================
# 配置区
# =========================
base_url = ""
api_key = ""

MODEL_NAME = "gpt-5"
MAX_WORKERS = 128
RETRY_TIMES = 3
TIMEOUT_SEC = 60

BASE_DIR = ""

INPUT_JSON = f"{BASE_DIR}"
OUTPUT_JSON = f"{BASE_DIR}/eval_thinking.json"

# =========================
# Prompt
# =========================
system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct “groundtruth” answer, and the candidate answer will be provided to you.
The following is a comparison table of some proper nouns; matching any one of them is considered correct.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth
and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria (semantic similarity only):

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
Score 5 only if all required items are correct;
if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>
"""

def build_prompt(item: Dict) -> str:
    tmpl = (
        "Question: {}\n"
        "Groundtruth answer: {}\n"
        "Candidate answer: {}\n"
        "Your response: "
    )
    return tmpl.format(
        item["question"],
        item["answer"],
        item["pred"]["answer"],
    )

# =========================
# GPT 调用
# =========================
client = OpenAI(base_url=base_url, api_key=api_key)
lock = threading.Lock()

def parse_score(text: str) -> int:
    """
    从模型输出中解析 Final Score: 后面的 0–5
    """
    # 优先严格匹配 Final Score
    match = re.search(r"Final Score:\s*([0-5])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 兜底：如果模型没按格式来，再退回全文匹配
    match = re.search(r"\b([0-5])\b", text)
    if match:
        return int(match.group(1))

    return -1

def score_one(item: Dict, idx: int) -> Dict:
    prompt = build_prompt(item)
    for attempt in range(RETRY_TIMES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=TIMEOUT_SEC,
            )

            text = resp.choices[0].message.content.strip()
            score = parse_score(text)

            if 0 <= score <= 5:
                item["gpt_score"] = score
                item["gpt_raw"] = text
                return item

            raise ValueError(f"Invalid score output: {text}")

        except Exception as e:
            traceback.print_exc()
            if attempt == RETRY_TIMES - 1:
                item["gpt_score"] = -1
                item["gpt_error"] = str(e)
                return item
            time.sleep(1.5 * (attempt + 1))

    return item

from collections import Counter
def score_mapping(score):
    if score in (4,5):
        return 1.0
    elif score in (3,):
        return 0.5
    else:
        return 0
def compute_stats(data):
    """
    data: List[Dict]，每个元素包含 gpt_score 和 index
    """
    scores = []
    one_score_indices = []

    for item in data:
        if isinstance(item.get("gpt_score"), int) and item["gpt_score"] >= 0:
            mapped = score_mapping(item["gpt_score"])
            scores.append(mapped)

            if mapped == 1:
                one_score_indices.append(item["index"])

    if not scores:
        return {
            "avg_score": None,
            "count": 0,
            "distribution": {},
            "one_score_indices": []
        }

    count = len(scores)
    avg_score = sum(scores) / count

    dist = Counter(scores)
    distribution = {
        score: {
            "count": dist.get(score, 0),
            "ratio": dist.get(score, 0) / count
        }
        for score in range(6)  # 0–5
    }

    return {
        "avg_score": avg_score,
        "count": count,
        "distribution": distribution,
        "one_score_indices": one_score_indices
    }

def main():
    if not os.path.exists(OUTPUT_JSON):
        data: List[Dict] = []
        for file in glob.glob(INPUT_JSON+"/*.jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        results = [None] * len(data)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(score_one, item, idx): idx
                for idx, item in enumerate(data)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    # print(future.result())
                except Exception as e:
                    traceback.print_exc()
                    data[idx]["gpt_score"] = -1
                    data[idx]["gpt_error"] = str(e)
                    results[idx] = data[idx]

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Done. Saved to {OUTPUT_JSON}")
    else:
        results = json.load(open(OUTPUT_JSON))
    stats = compute_stats(results)

    print("===== GPT Scoring Statistics =====")
    print(f"Valid samples: {stats['count']}")
    print(f"Average score: {stats['avg_score']:.4f}")

    print("Score distribution:")
    for score, info in stats["distribution"].items():
        print(
            f"  {score}: "
            f"{info['count']} "
            f"({info['ratio'] * 100:.2f}%)"
        )
    print("Correct Index:")
    print(stats['one_score_indices'])

if __name__ == "__main__":
    main()