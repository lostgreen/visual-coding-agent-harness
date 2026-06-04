# Handoff: VideoMME Long Agent Loop Test

Date: 2026-06-03
Last updated: 2026-06-04

## Current Goal

Build and evaluate a Claude Code-style visual long-video Agent harness:

- Main Agent plans from text context, not full video.
- Tools inspect video/image evidence and write observations.
- EvidenceWorkspace stores trace, artifacts, and ledger.
- Agent reads the ledger and iteratively decides whether to call more tools or answer.

The original user request was to run several VideoMME long videos on the remote KML machine, inspect whether the Agent loop is useful, and summarize the process/context-management design for reporting and future optimization.

The 2026-06-04 update integrated the design feedback into the harness and ran remote KML checks:

- Query-conditioned navigation now returns structured matches and relevance reasons.
- `zoom` materializes child segments in the mutable `VideoMap`.
- `inspect_segment` acts as a Segment Inspector subagent boundary: it inspects a localized time window and returns one distilled observation to the main planner.
- `EvidenceWorkspace.compact_ledger_text()` creates a bounded planner context from the raw ledger: long-term visual evidence, navigation summary, and short-term working buffer.
- `question_policy.py` adds task-type playbooks for multiple-choice, temporal-ordering, and general video QA. MCQ options are auto-extracted and injected into `inspect_segment`.
- `verify_ledger_answer` now has citation and non-navigation visual-evidence gates; `inspect_segment` counts as visual evidence.
- `AgentBudget.free_explore(...)` is available for quality-first runs with emergency caps but without per-class cost budgeting.
- Dynamic window addressing now handles tail clamp, reusable dynamic ids, and unambiguous millisecond-valued spans.
- MCQ option propagation now overwrites letter-only `candidate_options` with full option text from the question.

Remote tests now pass on KML:

```text
PYTHONPATH=/home/xuboshen/zgw/visual-coding-agent-harness/src \
/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Result: 82 tests OK.

Current VideoMME diagnosis has shifted. The early 4-round loop mostly failed by incompletion. The latest free-explore `611-2` run reaches a final answer and retrieves conflicting evidence, including evidence supporting the ground-truth option D, but the final synthesis chooses A. The current first-order blocker is therefore AnswerAgent/Verifier arbitration, not only loop budget.

## Current Evidence

Current compact debug note:

```text
docs/debug_logs/2026-06-04-phase0-eval.md
```

Current model-interaction record:

```text
docs/debug_logs/2026-06-04-model-interaction-records.md
```

Current Round 2 arbitration note:

```text
docs/debug_logs/2026-06-04-round2-arbitration.md
```

Remote summary artifacts:

| Run | Artifact | Compact status |
| --- | --- | --- |
| Old 3-case baseline | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_loop_eval_20260603/summary.json` | `direct_full_video` 2/3; empty-index loop 2/3 but mostly incomplete; subtitle-index loop 0/3 and incomplete. |
| Phase 0 | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase0_20260604/summary.json` | Reproduced direct baseline; first agent_v2 had missing dynamic segment arguments. |
| Fixed budget4 | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase0_agentv2_fixed_20260604/summary.json` | No missing-segment crash; still incomplete at 4 rounds and 1 tool call per round. |
| Phase 1 budget-only | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_budget_20260604/summary.json` | `agent_v2` 0/3, final_rate 33.3%, incomplete_rate 66.7%. |
| Default AnswerAgent | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_answer_agent_20260604/summary.json` | `agent_v2` 1/3, final_rate 66.7%, incomplete_rate 33.3%. |
| Prefinal probe | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_prefinal_probe_20260604/summary.json` | Negative result: 0/3, final_rate 0%, incomplete_rate 100%; disabled by default. |
| Current free-explore `611-2` | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_msfix_20260604/summary.json` | final_rate 100%, incomplete_rate 0%, final A vs GT D; retrieved both A- and D-supporting evidence. |

Do not load raw logs by default. Use `summary.json`, compact workspace aggregations, selected observation ids, or filtered error fingerprints.

## Environment Decision

Use the absolute Video harness / VLMEvalKit Python path:

```text
/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python
```

This Python was verified to provide:

- Python 3.10.0
- transformers 4.57.1
- `Qwen3VLForConditionalGeneration`
- `AutoModelForImageTextToText`
- `Qwen2_5_VLForConditionalGeneration`
- `vllm`
- `qwen_vl_utils`
- `torch`, `pandas`, `decord`, `cv2`

Important: do not rely on `source ...; conda activate ...; python` for this KML workflow unless the Python path is checked. Earlier checks accidentally resolved `python` to `/usr/bin/python`, which made the environment look broken.

Recommended launch shape:

```bash
cd /home/xuboshen/zgw/visual-coding-agent-harness
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/xuboshen/zgw/visual-coding-agent-harness/src \
/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python runs/videomme_agent_loop_eval_20260603/run_eval.py
```

Proxy used in the successful job:

```bash
export http_proxy=http://10.66.37.111:11080
export https_proxy=http://10.66.37.111:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
```

## Experiment Setup

Benchmark source:

```text
/ytech_m2v5_hdd/workspace/kling_mm/Datasets/VLMEvalKit_Dataset_Cache/HFCache/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b
```

Model:

```text
/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct
```

Tested VideoMME long cases:

| Case | Task | Duration | Ground Truth |
| --- | --- | ---: | --- |
| `605-1` | Information Synopsis | 1896.0s | D |
| `611-2` | Temporal Reasoning | 1805.0s | D |
| `612-1` | Temporal Reasoning | 3070.4s | B |

Compared strategies:

- `direct_full_video`: direct full-video sparse prompting, 64 frames.
- `empty_index_loop`: Agent loop over fixed 300s windows without semantic index.
- `subtitle_index_loop`: Agent loop over fixed 300s windows enriched with VideoMME subtitle text.

Agent loop settings:

- `max_rounds = 4`
- `max_tool_calls_per_round = 1`
- `default_nframes = 8`
- `planner_receives_media = False`
- segment clip extraction enabled.

## Results

Historical 2026-06-03 baseline, before the later AnswerAgent/free-explore/dynamic-window fixes:

Compact aggregate:

| Strategy | Correct | Avg Sec | Final Runs | Budget Exhausted |
| --- | ---: | ---: | ---: | ---: |
| `direct_full_video` | 2/3 | 5.17 | 0 | 0 |
| `empty_index_loop` | 2/3 | 37.41 | 1 | 2 |
| `subtitle_index_loop` | 0/3 | 40.05 | 0 | 3 |

Case details:

| Case | Direct | Empty Index Loop | Subtitle Index Loop |
| --- | --- | --- | --- |
| `605-1`, GT D | D, correct, 6.416s | D, correct, 34.273s, exhausted | no parsed choice, wrong, 30.546s, exhausted |
| `611-2`, GT D | A, wrong, 5.033s | A, wrong, 24.55s, final | no parsed choice, wrong, 54.819s, exhausted |
| `612-1`, GT B | B, correct, 4.059s | B, correct, 53.399s, exhausted | no parsed choice, wrong, 34.78s, exhausted |

Interpretation:

- The current loop runs end to end and records useful trace/evidence.
- It does not yet outperform direct sparse full-video prompting on these three VideoMME long samples.
- `subtitle_index_loop` did change exploration behavior, for example `611-2` inspected `seg_0007`, `seg_0004`, and `seg_0001`, so the semantic map can influence navigation.
- The Answer/Verifier behavior is currently weak. The Agent often spends rounds on `video_ls` and `read_segment`, then reaches the budget without a final answer.
- This baseline is still useful as a reference, but the current free-explore evidence shows that incompletion is not the only failure mode: when the loop has enough room to explore, final synthesis can still choose the wrong option from conflicting observations.

## Observed Agent Behaviors

Current free-explore `611-2` behavior:

```text
video_ls -> inspect_segment -> search_segments -> inspect_segment -> inspect_segment
-> inspect_segment -> inspect_segment -> inspect_segment -> caption_segment
-> caption_segment -> inspect_segment -> final A
```

Compact observation fingerprint:

- The final answer was A, while ground truth is D.
- The run inspected multiple coarse and dynamic windows and did not end incomplete.
- An early observation supports D, while later observations support A or C.
- Some observations declare a supported option while their limitations say the claim is inferred, lacks explicit visual confirmation, or relies on external knowledge.
- The final step cites an A-supporting caption observation and does not arbitrate the earlier D-supporting evidence.

Observed plans:

```text
605-1 empty:
video_ls -> video_ls -> caption_segment(seg_0001) -> caption_segments

605-1 subtitle:
video_ls -> read_segment(seg_0001) -> video_ls -> read_segment(seg_0001)

611-2 empty:
video_ls -> caption_segment(seg_0001) -> final A

611-2 subtitle:
caption_segment(seg_0007) -> video_ls -> read_segment(seg_0004) -> caption_segment(seg_0001)

612-1 empty:
video_ls -> caption_segment(seg_0001) -> caption_segment(seg_0002) -> caption_segment(seg_0003)

612-1 subtitle:
read_segment(seg_0001) -> expand_window(seg_0001) -> read_segment(seg_0002) -> read_segment(seg_0001)
```

Key behavior fingerprint:

- The planner can call tools and use segment ids.
- It sometimes explores non-initial segments when a subtitle map is available.
- It overuses cheap navigation/read tools.
- It does not reliably call `qa_segment`.
- It does not reliably call `verify_ledger_answer`.
- With four rounds and one tool per round, it often has no final round left after collecting evidence.
- With free exploration, it can collect richer evidence and reach final, but it still needs a stronger AnswerAgent/Verifier to resolve conflicting option support.

## Why Budget Exhaustion Happened

The budget is not token budget. It is loop/tool budget:

```python
AgentBudget(
    max_rounds=4,
    max_tool_calls_per_round=1,
    default_nframes=8,
    high_fps_nframes=32,
    planner_receives_media=False,
)
```

This means an Agent trajectory like this already consumes the full budget:

```text
Round 1: video_ls
Round 2: read_segment
Round 3: caption_segment
Round 4: caption_segments or another read
No Round 5 remains to read the last observation and answer.
```

Budget exhaustion therefore reflects a policy/design problem, not just model failure.

2026-06-04 update: free exploration changes the diagnosis. It removes the immediate budget-exhaustion symptom on `611-2`, but exposes a deeper answer-synthesis problem. The model can retrieve useful evidence, including evidence for the ground-truth option, then still choose a later conflicting observation as the final source.

Current research policy:

```text
quality-first free exploration:
  no per-class budget
  no reserved-final pressure
  emergency caps only
  collect traces for evidence recall and verifier design
```

Efficiency budgets and Agentic RL should be reintroduced after the verifier can separate grounded trajectories from long unsupported ones.

## Current Implementation Pointers

Main loop and budget:

```text
src/visual_coding_agent_harness/agents/iterative_agent.py
```

Evidence workspace:

```text
src/visual_coding_agent_harness/workspace.py
```

Program interpreter:

```text
src/visual_coding_agent_harness/interpreter.py
```

Video navigation tools:

```text
src/visual_coding_agent_harness/tools/navigation.py
```

Segment VLM tools:

```text
src/visual_coding_agent_harness/tools/segments.py
```

Segment Inspector boundary:

```text
src/visual_coding_agent_harness/tools/inspector.py
```

Task playbook / MCQ option extraction:

```text
src/visual_coding_agent_harness/agents/question_policy.py
```

Enrichment tools:

```text
src/visual_coding_agent_harness/tools/enrichment.py
```

Verification tools:

```text
src/visual_coding_agent_harness/tools/verification.py
```

## Current Workspace Protocol

Each run writes:

```text
runs/<run_id>/
  observations.jsonl
  trace.jsonl
  ledger.md
  artifacts/
    clips/
    frames/
    crops/
    masks/
```

Tool outputs are richer than what the next Agent round sees:

- `observations.jsonl` stores full raw output, including `regions`, `candidates`, `outline`, and `raw_video_map`.
- `ledger.md` stores append-only compact lines: observation id, tool, confidence, artifacts, claim, limitations.
- The next planner now reads `EvidenceWorkspace.compact_ledger_text()`, not raw `observations.jsonl` and not the full append-only ledger.
- The compact context separates long-term visual evidence, navigation summary, and the short-term working buffer.

This addresses the previous context-management bottleneck where navigation and final visual evidence were mixed in one append-only ledger view. The full raw trace still stays on disk for debugging and training data.

## Stale Or Invalid Evidence

Treat the following as stale:

- Any conclusion that Qwen3-VL was unsupported based only on `conda activate ...; python`.
- The failed job using `visual-agent-harness` with old transformers.
- Any check where `sys.executable` was not printed.
- The old missing-`segment_id` crash as a current blocker; it has been fixed.
- The conclusion that "larger budget alone should fix the loop"; free exploration now shows answer/verifier arbitration is the next bottleneck.

Current valid evidence is the absolute Python check, the successful KML unit-test run, the remote summary artifacts listed above, and the compact model-interaction record in `docs/debug_logs/2026-06-04-model-interaction-records.md`.

Round 2 local evidence is tracked separately in `docs/debug_logs/2026-06-04-round2-arbitration.md`; latest local test status is 91 passed. The current verifier rejects both stronger uncited option conflicts and timestamped temporal-order contradictions when observations carry structured event labels.

## Main Risks

- Direct sparse full-video baseline is strong on some VideoMME long samples, so naive tool use can be slower without improving accuracy.
- 300s windows are too coarse for some temporal QA and too expensive for repeated clip extraction.
- `read_segment` over subtitle text can look informative but may not answer visual/temporal questions.
- Current partial answer extraction can accidentally parse an option letter from ledger text; this should not count as a valid final answer.
- Current verifier has rule gates for citations and visual evidence, but still lacks model-based entailment and temporal consistency checks.
- `inspect_segment` is currently a one-shot isolated VLM call, not yet a full internal multi-step subagent that can run its own navigation/zoom/QA loop.
- The final answer step can over-trust a later caption or `supported_option` field even when the observation limitations say the claim is inferred.
- MCQ temporal-order questions need explicit sequence comparison across options; single-observation citation is not enough when the ledger contains conflicting option support.

## Recommended Next Actions

1. Keep free-explore as the quality-first experiment path:
   - no per-class tool budget yet
   - emergency caps only
   - preserve full trace and artifact provenance
   - optimize for evidence recall and conflict visibility before efficiency

2. Strengthen MCQ answer policy:
   - Require final answer option letter to match cited evidence.
   - Treat conflicting high-confidence option support as a verifier failure.
   - Downweight observations whose limitations mention inference, missing explicit visual cues, or external knowledge.
   - For temporal ordering, compare the extracted evidence order against each option sequence.

3. Strengthen AnswerAgent / Verifier:
   - AnswerAgent should consume an evidence table, not just one selected ledger line.
   - Verifier should output pass/fail plus targeted follow-up tool calls.
   - If evidence conflict remains, block final instead of accepting the most recent supported option.

4. Improve Inspector from a one-shot boundary into a real internal worker:
   - allow inspector-local `zoom` / `qa_segment` / OCR when needed
   - return only one distilled observation to the main planner
   - keep inspector trace in `trace.jsonl`, not the main prompt

5. Improve long-video indexing:
   - Keep 300s coarse map for `video_ls`.
   - Automatically downshift selected candidates to 30-60s child clips for VLM QA.
   - Add subtitle/ASR lexical search and later embedding retrieval.

6. Add caching:
   - Cache extracted clips.
   - Cache sampled frames.
   - Cache VLM observations by `(tool, video, segment, nframes, question hash)`.

7. Add reporting metrics:
   - final rate
   - incomplete rate
   - conflict rate
   - option-support consistency
   - unsupported final rate
   - tool sequence
   - unique inspected segments
   - evidence citations
   - direct vs loop wall time

## Suggested Next Experiment

Run a verifier-focused free-explore experiment on `611-2` first:

- free exploration with emergency caps
- full MCQ option text in every local question
- evidence table passed to AnswerAgent
- verifier checks conflicting `supported_option` values and temporal order
- invalid if final cites an observation whose limitations say the claim is not visually grounded while conflicting evidence exists

Then rerun the same three anchor cases:

- direct full-video baseline unchanged
- `agent_v2` free-explore quality-first
- optional bounded-budget policy only as an efficiency ablation
- 300s coarse map plus 60s child windows
- planner context from `compact_ledger_text()`
- invalid if no final JSON answer

Expected goal is not immediate broad benchmark gain, but clearer evidence-grounded final behavior and fewer wrong finals under conflicting evidence.
