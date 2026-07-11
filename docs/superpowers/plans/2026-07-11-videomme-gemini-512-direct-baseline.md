# VideoMME Gemini 512-Frame Direct Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Gemini direct-answer baseline using 512 uniformly sampled frames plus complete timestamped ASR on the same VideoMME long cases as the agent harness.

**Architecture:** Put reusable sampling, ASR formatting, contact-sheet fallback, and response parsing in `src/vcah/direct_baseline.py`. Keep dataset orchestration and CLI concerns in `tools/run_videomme_direct_uniform.py`, reusing the existing Gemini-compatible client, retry policy, case-group loader, and bounded case concurrency from the interactive runner.

**Tech Stack:** Python 3.10+, ffmpeg/ffprobe, Pillow, pandas/parquet, pytest, existing OpenAI-compatible Gemini client.

---

### Task 1: Uniform Sampling And Frame Artifacts

**Files:**
- Create: `src/vcah/direct_baseline.py`
- Create: `tests/test_videomme_direct_uniform.py`

- [ ] **Step 1: Write failing midpoint and manifest tests**

```python
def test_uniform_midpoint_times_cover_all_bins():
    assert uniform_midpoint_times(100.0, 4) == (12.5, 37.5, 62.5, 87.5)

def test_write_frame_manifest_preserves_index_and_timestamp(tmp_path):
    rows = write_frame_manifest(tmp_path / "frame_manifest.jsonl", [Path("frame_0001.jpg")], [2.5])
    assert rows[0]["frame_index"] == 1
    assert rows[0]["time_sec"] == 2.5
```

- [ ] **Step 2: Run tests and confirm they fail because the module does not exist**

Run: `PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py`

- [ ] **Step 3: Implement midpoint sampling, one-command ffmpeg extraction, and manifest writing**

```python
def uniform_midpoint_times(duration_sec: float, frame_count: int) -> tuple[float, ...]:
    step = float(duration_sec) / int(frame_count)
    return tuple(round((index + 0.5) * step, 3) for index in range(frame_count))
```

The ffmpeg command starts at `step / 2`, uses `fps=frame_count/duration_sec`, scales the longest edge to the configured maximum, writes exactly `frame_count` ordered JPEG files, and raises if the output count differs.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py`

### Task 2: ASR, Response Parsing, And Contact-Sheet Fallback

**Files:**
- Modify: `src/vcah/direct_baseline.py`
- Modify: `tests/test_videomme_direct_uniform.py`

- [ ] **Step 1: Write failing tests for timestamped ASR, answer parsing, and 32-sheet ordering**

```python
def test_format_asr_keeps_source_timestamps():
    assert "[00:01.000-00:02.000] hello" in format_timestamped_asr(({"start": 1, "end": 2, "text": "hello"},))

def test_parse_direct_response_accepts_fenced_json():
    parsed = parse_direct_response('```json\n{"answer":"B","rationale":"visible"}\n```')
    assert parsed["answer"] == "B"

def test_render_contact_sheets_preserves_all_512_frames(tmp_path):
    sheets = render_contact_sheets(frame_paths, tmp_path, rows=4, cols=4)
    assert len(sheets) == 32
```

- [ ] **Step 2: Run the tests and confirm the new functions are missing**

Run: `PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py`

- [ ] **Step 3: Implement the minimal functions**

The parser returns normalized `answer`, `rationale`, and bounded `evidence`. Contact sheets use fixed 4x4 grids, ordered frame indices, 160x90 cells, and no black padding because each group contains exactly 16 frames.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py`

### Task 3: Direct Baseline CLI

**Files:**
- Create: `tools/run_videomme_direct_uniform.py`
- Modify: `tests/test_videomme_direct_uniform.py`

- [ ] **Step 1: Write failing tests for case ordering, fallback metadata, and summary metrics**

```python
def test_request_direct_answer_falls_back_only_for_request_shape_errors(tmp_path, monkeypatch):
    api = ScriptedApi((RuntimeError("HTTP 413: request too large"), '{"answer":"A"}'))
    monkeypatch.setattr(direct_runner, "render_contact_sheets", lambda paths, out_dir: (tmp_path / "sheet.jpg",))
    response, input_mode = direct_runner.request_direct_answer(
        api=api,
        prompt="question",
        frame_paths=tuple(Path(f"frame_{index:04d}.jpg") for index in range(512)),
        sheet_dir=tmp_path,
    )
    assert response["answer"] == "A"
    assert input_mode == "contact_sheets_32"

def test_summarize_results_reports_accuracy_and_latency():
    summary = summarize_results(results)
    assert summary["accuracy"] == 0.5
```

- [ ] **Step 2: Run tests and verify the runner functions do not exist**

Run: `PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py`

- [ ] **Step 3: Implement CLI orchestration**

The CLI accepts dataset root, config, output root, case group or case IDs, frame count, max image edge, workers, rebuild, and force-contact-sheets. It writes the required per-case artifacts and group `summary.json`, never logs credentials, and uses the existing `_run_case_batch` worker clamp and `OpenAICompatibleVisionClient` retry behavior.

- [ ] **Step 4: Run focused tests and the full local suite**

Run:

```bash
PYTHONPATH=src python -m pytest -q tests/test_videomme_direct_uniform.py
PYTHONPATH=src python -m pytest -q
```

- [ ] **Step 5: Commit and push the runner**

```bash
git add src/vcah/direct_baseline.py tools/run_videomme_direct_uniform.py tests/test_videomme_direct_uniform.py
git commit -m "feat: add Gemini 512-frame direct baseline"
git push
```

### Task 4: KML Smoke And Ten-Case Run

**Files:**
- Runtime artifacts only under `/m2v_intern/xuboshen/zgw/VideoAgent/videomme_direct_512_v3`

- [ ] **Step 1: Sync KML and run focused tests with the absolute Python path**

Run: `PYTHONPATH=/home/xuboshen/zgw/visual-coding-agent-harness/src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q tests/test_videomme_direct_uniform.py`

- [ ] **Step 2: Run one smoke case using 512 independent images**

Use case `672-3`, one worker, the configured Gemini YAML, and record request mode, latency, response, and request-size failure if present.

- [ ] **Step 3: If required, validate the explicit 32-sheet fallback**

Fallback is allowed only after an HTTP 400/413 or an error message indicating image-count/request-size limits. Do not silently fall back for model or parsing errors.

- [ ] **Step 4: Run all v3 cases with 10 workers**

Use `configs/eval_groups/videomme_long_hard_rotate_v3.json`, `--frames 512`, and `--workers 10`. Monitor compact process state and per-case result counts, not raw logs.

### Task 5: Attribution Report

**Files:**
- Create: `docs/debug_logs/2026-07-11-videomme-direct512-vs-agent.md`

- [ ] **Step 1: Aggregate direct accuracy and observable rationale categories**

Report answers, correctness, input mode, evidence timestamps, concise rationale category, latency, and failures. Do not report hidden chain-of-thought.

- [ ] **Step 2: Compare against the v3 agent results**

Compute direct-agent agreement, direct-only correct cases, agent-only correct cases, verifier flips, and whether agent evidence overlaps direct evidence timestamps.

- [ ] **Step 3: Select one architecture change with the largest measured error contribution**

Prioritize restoring multimodal evidence to the Reasoner if direct succeeds where the Investigator had already observed the clue, or navigation/coverage if direct succeeds where the Agent never opened the relevant interval.
