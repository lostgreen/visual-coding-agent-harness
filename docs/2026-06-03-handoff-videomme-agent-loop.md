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

The 2026-06-04 v4 update integrated the coding-improvement plan into the harness direction:

- Query-conditioned navigation now returns structured matches and relevance reasons.
- `zoom` materializes child segments in the mutable `VideoMap`.
- `inspect_segment` acts as a Segment Inspector subagent boundary: it inspects a localized time window and returns one distilled observation to the main planner.
- `EvidenceWorkspace.compact_ledger_text()` creates a bounded planner context from the raw ledger: long-term visual evidence, navigation summary, and short-term working buffer.
- `question_policy.py` now separates `gist_global`, `temporal_order`, and `needle_local`; wrapper text such as "answer with option letter first" no longer hijacks routing.
- `verify_ledger_answer` now has citation and non-navigation visual-evidence gates; `inspect_segment` counts as visual evidence.
- `AgentBudget.free_explore(...)` is available for quality-first runs with emergency caps but without per-class cost budgeting.
- Dynamic window addressing now handles tail clamp, reusable dynamic ids, and unambiguous millisecond-valued spans.
- MCQ option propagation now overwrites letter-only `candidate_options` with full option text from the question.
- Planner JSON parse errors caused by quoted MCQ option text now record a compact trace event and fall back to localized `inspect_segment` instead of aborting.
- `global_gist` is the direct sparse-video floor for gist/global questions and may temporarily emit `supported_option` as direct-style whole-video evidence.
- Local workers are now no-vote by default: `inspect_segment` may use MCQ options as fact-finding hints, but the prompt tells it not to choose an option or emit `supported_option`.
- `EvidenceWorkspace.evidence_table()` ignores legacy local worker vote fields and claim patterns by default, while explicit `candidate_option_relations` remain first-class option-support evidence.
- `report_metrics.py` now reports `legacy_worker_vote_rows` so old traces and regressions are visible instead of silently affecting arbitration.

Remote tests now pass on KML using the current video harness environment:

```text
PYTHONPATH=/home/xuboshen/zgw/visual-coding-agent-harness/src \
/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Result: 106 tests OK locally and on KML after the v4 no-vote sync.

Current VideoMME diagnosis has shifted. The early 4-round loop mostly failed by incompletion. Later free-explore runs showed that the agent can reach `final`, but can still pick weak or conflicting evidence. Plan v4 therefore freezes the next objective as:

```text
Hold the direct floor on gist/global questions,
route local/temporal questions into grounding + vision,
prevent local workers from voting on MCQ answers,
and make the final answer come only from a verifier-approved evidence table.
```

## Latest v4 Branch Update

Date: 2026-06-04
Branch: `codex/v4-skill-framework`

Current evidence:

- Added v4 typed schemas: `CandidateOptionRelation`, `GroundingCandidate`, `VisionFact`, `EvidenceRowV2`, `AnswerAgentDecision`, and `TemporalVerifierResult`.
- Added the first declarative skill registry/compiler skeleton with built-in `gist_qa`, `grounded_factual_qa`, and `temporal_ordering`; `select_skill()` routes from the existing question policy, and `compile_skill_program()` emits interpreter-compatible steps.
- Added the shared skill/verifier predicate library for global gist floor, structured support, weak grounding, conflict, timestamp coverage, and temporal-order checks.
- Added `EvidenceWorkspace.evidence_table_v2()` with explicit schema metadata and `legacy_worker_vote_rows`; `vision_read` rows are answer-facing evidence, while legacy local worker votes remain tracked.
- Added `vision_read` as a v4 Segment Inspector wrapper that emits typed facts with `event_label`, `time_range`, and `grounding_quality`, with no option-vote fields.
- Extended `ProgramInterpreter` with `foreach`, slot interpolation, `op` compatibility, and sufficiency-stop tracing.
- Connected the verifier to the shared predicates while preserving the existing verification result shape.
- Updated the planner prompt/tool policy so `vision_read` is exposed alongside legacy `inspect_segment`.

Last local check:

```text
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
Result: 113 tests OK.
```

Current KML status:

- New requested KML URL works when passed explicitly with `--base-url https://kml-dtmachine-23666-prod-0.kmlhb2az1l3-2.corp.kuaishou.com`.
- `pwd` on the requested KML returned `/home/xuboshen`.
- Synced the v4 foundation files to `/home/xuboshen/zgw/visual-coding-agent-harness` via `/tmp/v4-skill-framework-sync.tar` (`317440` bytes, SHA256 `daa671b2d6bcf8804c13d4bd2c723165ad759a08d8e2805448a2b75d7a523098`).
- Remote check passed:

```text
cd /home/xuboshen/zgw/visual-coding-agent-harness
PYTHONPATH=src /home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python -m unittest discover -s tests -p 'test_*.py'
Result: 113 tests OK.
```

- Local `~/.codex/kml_bridge.env` may still point to `kml-dtmachine-25361-prod-0...`; use explicit `--base-url` for `23666` until the env is intentionally updated.

Next actions:

- Add a concrete `ground_question` wrapper and wire skill execution into `IterativeVisualAgent` for `gist_qa` first, then temporal synthetic and VideoMME `611-2`/`612-1`.
- Extend `report_metrics.py` with skill-level metrics after real skill executions emit `skill_start`/`skill_end`.

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

Current Round 3 global floor note:

```text
docs/debug_logs/2026-06-04-round3-global-floor.md
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
| Round 2 free sync | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round2_free_sync_20260604/summary.json` | `agent_v2` solved `612-1`, but `605-1` and `611-2` aborted before tools due to planner JSON parse errors from quoted options. |
| Source-machine visual harness full | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_visual_harness_full_20260604/summary.json` | Current 3-case source-machine run. `agent_v2 --free-explore` reached final on all cases, no JSON/runtime error, 1/3 accuracy; direct baseline 2/3. |
| Round 3 global floor | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round3_605_global_fix_20260604/summary.json` | `605-1` routes to `global_gist`; agent D, direct D, `direct_regressions=0`, `unsupported_final=false`. |
| Round 4 v4 no-vote anchor | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round4_worker_no_vote_3case_20260604/summary.json` | Completed on KML. `agent_v2` 1/3, final_rate 100%, direct_regressions 1, unsupported_final_rate 66.7%, legacy_worker_vote_rows 1. `605-1` stayed D via `global_gist`; `611-2` final C vs GT D; `612-1` final A vs GT B while direct B. |

Do not load raw logs by default. Use `summary.json`, compact workspace aggregations, selected observation ids, or filtered error fingerprints.

## Environment Decision

Use the absolute video harness Python path for current source-machine work:

```text
/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python
```

This Python was verified to provide:

- pandas 2.3.3
- pyarrow 20.0.0
- torch 2.8.0+cu128
- transformers 4.57.0
- importable `visual_coding_agent_harness`

Important: do not rely on `source ...; conda activate ...; python` for this KML workflow unless the Python path is checked. Earlier checks accidentally resolved `python` to `/usr/bin/python`, which made the environment look broken.

Recommended launch shape:

```bash
cd /home/xuboshen/zgw/visual-coding-agent-harness
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python runs/eval_runner.py \
  --allow-any-python \
  --run-root runs/videomme_agent_visual_harness_full_20260604 \
  --strategy direct_full_video --strategy agent_v2 --free-explore
```

Proxy for current KML work:

```bash
export http_proxy=http://oversea-squid1.jp.txyun:11080
export https_proxy=http://oversea-squid1.jp.txyun:11080
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

Current source-machine Agent interaction instances:

```text
605-1, GT D, final C, wrong
tool path: video_ls -> inspect_segment -> search_segments -> inspect_segment
           -> search_segments -> inspect_segment x8
final cites: obs_0001(video_ls), obs_0003(search), obs_0005(search),
             obs_0006(inspect, supports B), obs_0007(inspect, no valid option)
fingerprint: final choice C has no structured support while many Inspector rows support B.

611-2, GT D, final A, wrong
tool path: inspect_segment x4 -> search_segments x2 -> caption_segment
final cites: obs_0007(caption_segment)
fingerprint: caption shortcut supplies A; the run reaches final but lacks a verifier check that compares option support and grounding.

612-1, GT B, final B, correct
tool path: inspect_segment x5 -> caption_segment -> zoom -> inspect/caption mix
final cites: obs_0014, obs_0018, obs_0019, obs_0021
fingerprint: succeeds after long exploration, but still contains A-supporting Inspector rows, so the verifier should report a resolved conflict rather than silently accepting recency.
```

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

Source-machine visual harness full run:

| Case | GT | Direct | Agent v2 free-explore | Current fingerprint |
| --- | --- | --- | --- | --- |
| `605-1` | D | D, correct | C, wrong, 9 rounds / 13 tools | Inspector observations mostly supported B; final cited navigation/search rows plus one B row and chose unsupported C. |
| `611-2` | D | A, wrong | A, wrong, 6 rounds / 7 tools | Final cited a caption row that answered A; structured evidence did not establish D in this run. |
| `612-1` | B | B, correct | B, correct, 20 rounds / 21 tools | Long free-explore trace eventually cited four B caption rows; still contains conflicting A-supporting Inspector rows. |

Aggregate:

| Strategy | Accuracy | Final Rate | Incomplete Rate | Avg Sec |
| --- | ---: | ---: | ---: | ---: |
| `agent_v2` | 1/3 | 100% | 0% | 304 |
| `direct_full_video` | 2/3 | 100% | 0% | 5.08 |

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

Current valid evidence is the absolute `visual-agent-harness` Python check, the successful KML unit-test run, the remote summary artifacts listed above, and the compact model-interaction record in `docs/debug_logs/2026-06-04-model-interaction-records.md`.

Round 2 local evidence is tracked separately in `docs/debug_logs/2026-06-04-round2-arbitration.md`; latest local and KML test status is 106 passed. The current verifier rejects both stronger uncited option conflicts and timestamped temporal-order contradictions when observations carry structured event labels.

## Main Risks

- Direct sparse full-video baseline is strong on some VideoMME long samples, so naive tool use can be slower without improving accuracy.
- 300s windows are too coarse for some temporal QA and too expensive for repeated clip extraction.
- `read_segment` over subtitle text can look informative but may not answer visual/temporal questions.
- Current partial answer extraction can accidentally parse an option letter from ledger text; this should not count as a valid final answer.
- Current verifier has rule gates for citations and visual evidence, but still lacks model-based entailment and temporal consistency checks.
- `inspect_segment` is currently a one-shot isolated VLM call, not yet a full internal multi-step subagent that can run its own navigation/zoom/QA loop.
- The final answer step can still over-trust a later caption or weak candidate relation if conflict resolution is incomplete.
- Legacy traces may contain local worker vote text such as `Supported option: X`; default evidence tables now ignore these rows, but reports must keep counting `legacy_worker_vote_rows`.
- MCQ temporal-order questions need explicit sequence comparison across options; single-observation citation is not enough when the ledger contains conflicting option support.

## V4 Acceptance Gates

Gate A, stop the bleeding:

```text
605-1 -> D via global_gist
direct_regressions == 0 on anchor set
unsupported_final == 0
```

Gate B, role split:

```text
No local worker observation emits supported_option
legacy_worker_vote_rows == 0 in default fresh runs
GroundingAgent outputs candidates only
VisionAgent outputs local facts only
```

Gate C, arbitration:

```text
final_with_conflict == 0 unless conflict is resolved
selected option always has structured support
611-2 -> D or need_more_evidence with targeted follow-up
612-1 -> B with temporal conflict resolved
```

Gate D, evaluation honesty:

```text
report split by route
standard VideoMME uses stateless_per_qa
stateful memory results reported separately
agent preserves direct floor on gist/global
agent targets improvement on needle/local and temporal cases
```

## Recommended Next Actions

1. Treat the Round 4 no-vote anchor as the current failure fingerprint: global floor passes, but Gate B/C fail because a fresh run still produced `legacy_worker_vote_rows=1`, two unsupported finals, and one direct regression.
2. Add report split fields from v4: `route_distribution`, `accuracy_by_route`, `walltime_by_route`, `direct_regressions_by_route`, `unsupported_final_by_route`, and `legacy_worker_vote_rows`.
3. Implement a `ground_question` wrapper or GroundingAgent that internally uses `video_ls/search_segments/zoom`, returns candidate clips only, and never emits option votes.
4. Upgrade `inspect_segment` toward `vision_read` / `inspect_segment_v2`: structured `facts`, `event_label`, `polarity`, `grounding_quality`, and no `supported_option` by default. Add a stricter failure/repair path when the model text still contains legacy option-vote phrases.
5. Upgrade AnswerAgent to map facts into `candidate_option_relations`, block unsupported finals, and return `need_more_evidence` with a targeted next action when evidence cannot distinguish options.
6. Add a temporal verifier that compares event labels and timestamp order for `611-2` / `612-1`.
7. Expand evaluation only after these gates: add 5-10 needle/local long-video cases and keep stateless VideoMME separate from any stateful memory experiment.

## Suggested Next Experiment

Current KML anchor result:

```text
/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round4_worker_no_vote_3case_20260604/summary.json
```

Compact result:

| Case | GT | Direct | Agent v2 | Current diagnosis |
| --- | --- | --- | --- | --- |
| `605-1` | D | D, correct | D via `global_gist`, correct | Gate A local sanity passes. |
| `611-2` | D | A, wrong | C, wrong, unsupported final | Needs temporal grounding/facts and AnswerAgent should prefer `need_more_evidence` over unsupported C. |
| `612-1` | B | B, correct | A, wrong, unsupported final | Direct regression; release blocker until temporal verifier / option mapping blocks unsupported A. |

Strategy-level metrics:

```text
agent_v2 accuracy 1/3
final_rate 100%
direct_regressions 1
unsupported_final_rate 66.7%
legacy_worker_vote_rows 1
```

Next rerun the same three anchor cases only after tightening worker-output validation and AnswerAgent blocking:

- direct full-video baseline unchanged
- `agent_v2` free-explore quality-first
- optional bounded-budget policy only as an efficiency ablation after quality gates pass
- 300s coarse map plus 30-60s child windows
- planner context from `compact_ledger_text()`
- invalid if selected option has no structured support or cites navigation-only rows

Expected goal is not immediate broad benchmark gain, but clearer evidence-grounded final behavior, zero direct-floor regressions, and fewer wrong finals under conflicting evidence.
