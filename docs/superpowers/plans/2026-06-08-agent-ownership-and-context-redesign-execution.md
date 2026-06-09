# Agent Ownership And Context Redesign Execution

Date: 2026-06-08
Branch: `codex/agent-ownership-context-redesign`
Code commit: `28c778a fix(video): keep answer suggestions planner-owned`
Handoff doc: pushed on top of the code commit in the same branch.
Source plan: `/Users/lostgreen/Downloads/2026-06-08-agent-ownership-and-context-redesign.md`

## Current Status

- Ticket A: completed. Text-derived ordered-list rows from `locate_targets_in_segment` now use `confidence_signal="text_inferred"` and `requires_visual_verification=true`. They are written to `timeline_candidates.md`, not `timeline.md`. AnswerAgent and skill predicates now score these rows as non-strong evidence until visual verification exists.
- Ticket B: completed for timeline heuristics and non-reserved AnswerAgent takeovers. `_timeline_temporal_decision` now records `iterative_timeline_temporal_inference` and injects `# Pending Inference`; it no longer returns final. `all_segments_inspected`, `repeated_program_guard`, `prefinal_probe`, and non-budget `evidence_table_no_growth` AnswerAgent outcomes are downgraded to planner-visible suggestions instead of direct finals.
- Ticket C: completed. Quoted ordered-list extraction rejects windows stitched across sentence boundaries, preserves compact quoted lists, and downgrades non-quoted text-position order to navigation-only inference.
- Ticket D: completed as a minimal prompt-context upgrade. `EvidenceWorkspace.recent_tool_outputs(limit=3)` returns structured recent tool payloads with field-level truncation, and planner prompts render `# Recent Tool Outputs` before the compact ledger.
- Ticket E: completed for the core verifier/gate path. `verify_segment_anchors` and `vision_read` prompt for `ORDERED_VISIBLE`; both parse/store `ordered_visible_in_window`. `verify_segment_anchors` materializes ordered timeline rows. Single-scene final gating now requires a short `<60s` `verify_segment_anchors` or `vision_read` observation covering all target items.
- Ticket F: partially deferred. Rules were not moved to a new `final_gate.py`; the new single-scene rule and answer-suggestion behavior are implemented in `iterative_agent.py`.
- Ticket G: deferred. Scene-index subwindow hints were not added in this branch.
- Ticket H: deferred. `view_observation` redesign and reflection-memory one-shot cleanup were not implemented in this branch.
- Ticket I: local and KML regression completed for synthetic three-case replay; real KML three-demo `agent_v2` eval was launched and intentionally left running per handoff.

## Verification

- Local: `PYTHONPATH=src:. pytest -q`
  - Result: `403 passed in 1.44s`.
- KML: `PYTHONPATH=src:. /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q`
  - Result: `403 passed in 5.59s`.
- KML synthetic three-case replay:
  - Command: `PYTHONPATH=src:. /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m pytest -q tests/regression/test_videomme_3case_replay.py`
  - Result: `3 passed in 0.11s`.
- `PYTHONPATH=src:. python -m visual_coding_agent_harness.cli.iterative_smoke`
  - Not run to completion because the CLI requires `--model-path`, `--media-path`, `--question`, and `--duration-sec`.

## KML Handoff 2026-06-08

Current evidence:

- Local branch `codex/agent-ownership-context-redesign` is clean and synced to GitHub. Code changes are in `28c778a`; handoff updates are doc-only commits after that.
- KML repo `/home/xuboshen/zgw/visual-coding-agent-harness` is clean and synced to code commit `28c778a`. It was not advanced to the doc-only handoff commit while the real eval was running.
- KML uses conda env `/home/xuboshen/Anaconda/envs/visual-agent-harness`; default `/usr/bin/python3` and Anaconda base do not have pytest.
- KML proxy needed for git/eval commands:
  `http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com`

Artifacts and paths:

- KML full pytest log: `/tmp/agent_ownership_pytest.log`
- KML full pytest pid path: `/tmp/agent_ownership_pytest.pid`
- KML synthetic three-case replay log: `/tmp/videomme_3case_replay.log`
- KML synthetic three-case replay pid path: `/tmp/videomme_3case_replay.pid`
- KML real three-demo eval run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ownership_context_3demo_20260608`
- KML real three-demo eval log: `/tmp/videomme_agent_ownership_context_3demo_20260608.log`
- KML real three-demo eval pid path: `/tmp/videomme_agent_ownership_context_3demo_20260608.pid`
- First real eval workspace observed before handoff: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ownership_context_3demo_20260608/workspaces/runs/605-1_xKiRmesHWIA_agent_v2`

Current running job:

- Real KML three-demo `agent_v2` eval was started with cases `605-1,611-2,612-1`, model `/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct`, dataset root `/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b`, `scene-index-mode=dual-source`, `max-rounds=20`, `max-tool-calls-per-round=4`, `default-nframes=16`, and `--export-training`.
- Last checked state before stopping monitoring: PID `12656` running, log had reached `START videomme_eval {"cases": ["605-1", "611-2", "612-1"], "strategies": ["agent_v2"]}`. The first case workspace had been created and was progressing through `605-1`.
- Per user instruction, no further monitoring was performed in this thread.

Stale evidence:

- Earlier completion review marked Ticket E as missing; local branch now has a single-scene gate and `ORDERED_VISIBLE` parsing for verifier and vision read.
- Earlier KML pytest attempt with default Python failed with return code 127 because pytest was not installed in that interpreter; ignore that result.

Suggested next actions:

1. When ready, inspect only `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_ownership_context_3demo_20260608/summary.json` if it exists, or poll `/tmp/videomme_agent_ownership_context_3demo_20260608.pid` and `/tmp/videomme_agent_ownership_context_3demo_20260608.log`.
2. Summarize the three real demo rows from `summary.json`: `question_id`, GT, selected answer, correctness, status, rounds, tool sequence, and trajectory path.
3. For `611-2`, inspect only compact trajectory facts: whether `iterative_timeline_temporal_inference` stayed hint-only, whether a short `verify_segment_anchors` or `vision_read` covered the Bernini subwindow, and whether final answer came from planner-owned JSON.

## Notes

- This branch intentionally changes old tests that expected timeline or AnswerAgent heuristics to directly final. The new contract is planner ownership: framework heuristics become visible suggestions unless the planner has explicitly finalized or the run reaches the configured final fallback.
- `timeline_candidates.md` is now the home for transcript/navigation order candidates. `timeline.md` is reserved for visual verifier rows and legacy explicit visual timestamp rows.

## Update 2026-06-09

Current goal:

- Apply the new review plan: raw MCQ remains canonical for planner, navigation/search, evidence comparison, and AnswerAgent; only local VLM worker requests receive option-free factual prompts.
- Address remote Agent interaction failure mode where indexed ASR details were visible through `read_segment_detail` but could not become answer-grade evidence, causing repeated visual exploration.

Important decisions:

- `QuestionContext` now makes question views explicit. Planner, AnswerAgent, and navigation use raw MCQ by default; `vlm_safe_question` is reserved for local VLM prompt repair.
- VideoMME runner no longer enables global MCQ rewrite by default. Legacy behavior is available with `--use-global-question-rewrite`; `--disable-mcq-rewrite` remains a compatibility no-op because rewrite is already disabled by default.
- Local VLM tools strip option labels/text from prompts and request metadata. Planner/search may still use options and option-derived atoms.
- `read_segment_detail` remains navigation-only, but it can return structured `answer_evidence_rows`; the interpreter promotes only those rows into evidence table entries as `tool="asr_cue_detail"` and `grounding_quality="indexed_transcript"`.
- Final gates now use answer-grade evidence terminology: visual evidence, indexed transcript evidence, OCR, or QA can support a final answer. Navigation-only rows and text-inferred locator rows remain insufficient.

Files changed in this update:

- Question handling and planning: `agents/open_questions.py`, `agents/question_policy.py`, `agents/iterative_agent.py`, `agents/prompt_stack.py`.
- Evidence promotion and gates: `tools/navigation.py`, `interpreter.py`, `workspace.py`, `agents/answer_agent.py`.
- Local VLM sanitization: `tools/vlm.py`, `tools/segments.py`, `tools/enrichment.py`, `tools/inspector.py`, `tools/global_view.py`.
- Runner defaults: `evals/videomme/runner.py`.
- Focused tests updated/added across question policy, open questions, VLM sanitization, navigation ASR promotion, runner config, and iterative agent prompt/final gates.

Current verification:

- Local full regression: `PYTHONPATH=src:. pytest -q`
  - Result: `410 passed in 1.48s`.
- Focused regression before full run:
  - `tests/test_question_policy.py tests/test_open_questions.py tests/test_caption_qa_tools.py tests/test_global_view.py tests/test_video_navigation.py tests/test_eval_runner.py tests/test_iterative_agent.py`
  - Result: `171 passed in 0.80s`.

Stale evidence:

- Previous local/KML pytest counts of `403 passed` are from before this 2026-06-09 rewrite/ASR-promotion patch.
- Previous KML repo state at code commit `28c778a` does not include this unpushed local patch unless synced later.

Next actions:

1. Review and commit the local patch.
2. Sync to KML via GitHub/proxy or a reviewed patch path.
3. Re-run local/KML full pytest and synthetic three-case replay after sync.
4. Re-run VideoMME cases `605-1,611-2,612-1`; for `612-1`, check whether `asr_cue_detail` rows support option B before max rounds.
