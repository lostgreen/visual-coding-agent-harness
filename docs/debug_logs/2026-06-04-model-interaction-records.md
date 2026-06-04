# Model Interaction Records: VideoMME Free-explore Pass

Date: 2026-06-04

## Purpose

This note collects the current model-interaction issues and the raw record pointers from the VideoMME agent-loop debugging pass.

It intentionally does not paste raw logs, full `trace.jsonl`, full `observations.jsonl`, or model responses. The raw records stay as artifacts on KML; this note keeps only compact summaries, failure fingerprints, and paths.

## Current Evidence Boundary

Current evidence:

- KML repo: `/home/xuboshen/zgw/visual-coding-agent-harness`
- KML Python: `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`
- Local debug note: `docs/debug_logs/2026-06-04-phase0-eval.md`
- Round 2 arbitration note: `docs/debug_logs/2026-06-04-round2-arbitration.md`
- This note: `docs/debug_logs/2026-06-04-model-interaction-records.md`

Stale or superseded evidence:

- The old missing-`segment_id` crash is fixed.
- The old conclusion "budget exhaustion is the main blocker" is incomplete. Free exploration removes incompletion on `611-2`, but answer synthesis still chooses the wrong option.
- The old conclusion that `VLMEvalKit` is the active Python for source-machine runs is superseded by the checked `visual-agent-harness` environment.
- Raw logs from earlier failed dynamic-window runs should be treated only as bug-reproduction artifacts, not as current behavior.

## Raw Record Pointers

Remote summaries:

| Run | Artifact | Compact status |
| --- | --- | --- |
| Old 3-case baseline | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_loop_eval_20260603/summary.json` | `direct_full_video` 2/3; empty-index loop 2/3 but mostly incomplete; subtitle-index loop 0/3 and incomplete. |
| Phase 0 | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase0_20260604/summary.json` | Reproduced direct baseline; first agent_v2 hit missing dynamic segment arguments. |
| Fixed budget4 | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase0_agentv2_fixed_20260604/summary.json` | No missing-segment crash; still incomplete at 4 rounds and 1 tool call per round. |
| Phase 1 budget-only | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_budget_20260604/summary.json` | `agent_v2` 0/3, final_rate 33.3%, incomplete_rate 66.7%. |
| Default AnswerAgent | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_answer_agent_20260604/summary.json` | `agent_v2` 1/3, final_rate 66.7%, incomplete_rate 33.3%. |
| Prefinal probe | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase1_prefinal_probe_20260604/summary.json` | Negative result: 0/3, final_rate 0%, incomplete_rate 100%; kept disabled by default. |
| Free-explore crash 1 | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_20260604/summary.json` | Tail window slightly exceeded float duration. |
| Free-explore tail fix | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_tailfix_20260604/summary.json` | Completed but before option propagation fix; D-supporting evidence existed, final chose A. |
| Free-explore options fix | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_optionsfix_20260604/summary.json` | Dynamic window id reuse was not resolvable. |
| Free-explore dynamic-id fix | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_dynamicfix_20260604/summary.json` | Planner supplied millisecond values in `start_sec` / `end_sec`. |
| Free-explore ms fix | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_free_explore_611_msfix_20260604/summary.json` | Current free-explore result for `611-2`: final_rate 100%, incomplete_rate 0%, answer A vs GT D. |
| Round 2 free sync | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round2_free_sync_20260604/summary.json` | `direct_full_video` 2/3; `agent_v2` solved `612-1` but `605-1` and `611-2` aborted before tools due to planner JSON parse errors from quoted option text. |
| Source-machine visual harness full | `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_visual_harness_full_20260604/summary.json` | Current 3-case run on `visual-agent-harness` Python. `agent_v2 --free-explore` final_rate 100%, incomplete_rate 0%, 1/3 accuracy; direct baseline 2/3. |

Remote workspaces for each run are under:

```text
/home/xuboshen/zgw/visual-coding-agent-harness/runs/<run_id>/workspaces/
```

Use filtered summaries, `summary.json`, compact `trace.jsonl` aggregations, and selected observation ids. Do not stream full logs by default.

## Compact Run Findings

Default AnswerAgent 3-case run:

- Accuracy: 1/3.
- final_rate: 66.7%.
- incomplete_rate: 33.3%.
- `605-1`: final C, wrong.
- `611-2`: reached max rounds.
- `612-1`: final B, correct.

Free-explore `611-2` current run:

- Ground truth: D.
- Final answer: A.
- final_rate: 100%.
- incomplete_rate: 0%.
- Runtime: about 271 seconds.
- Tool calls: `video_ls -> inspect_segment -> search_segments -> inspect_segment -> inspect_segment -> inspect_segment -> inspect_segment -> inspect_segment -> caption_segment -> caption_segment -> inspect_segment`
- Inspected spans included `seg_0004`, `seg_0002`, `seg_0001`, `window_001800000_001804957`, `seg_0007`, `window_001200000_001500000`, `seg_0003`, `seg_0005`, and `window_001500000_001800000`.

Key observation pattern from `611-2`:

- `obs_0002` supports option D with high confidence, but also says the visual frames lack explicit textual/contextual cues.
- Later observations support A or C, often with limitations such as inferred order, missing explicit visual confirmation, or reliance on external knowledge.
- Final answer cites `obs_0010`, which supports A, and ignores the earlier D-supporting observation.

Source-machine visual harness full run:

- Python: `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`
- Summary: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_visual_harness_full_20260604/summary.json`
- `direct_full_video`: 2/3; `605-1=D` correct, `611-2=A` wrong, `612-1=B` correct.
- `agent_v2 --free-explore`: 1/3; all cases reached `final` with tool use and no JSON/runtime error.
- Reporter: final_rate 100%, incomplete_rate 0%, avg 304 sec, avg vs direct 66.3.
- Parser fingerprint: real Inspector outputs put option support in claim text, e.g. `Supported option: A.`, so `evidence_table()` must parse claim text in addition to `raw_output.supported_option`.

Per-case compact status:

| Case | GT | Direct | Agent v2 | Tool / evidence fingerprint |
| --- | --- | --- | --- | --- |
| `605-1` | D | D, correct | C, wrong, 9 rounds / 13 tools | Many Inspector rows support B; final C has no structured support and cites navigation/search rows. |
| `611-2` | D | A, wrong | A, wrong, 6 rounds / 7 tools | Final cites one `caption_segment` row that directly answers A; temporal option-order verification is missing. |
| `612-1` | B | B, correct | B, correct, 20 rounds / 21 tools | Final cites four B caption rows, but ledger still includes A-supporting Inspector rows. |

## Agent Interaction Instances

These are compact, analysis-oriented examples from the current source-machine run. They are not full raw model responses.

### 605-1: unsupported final option

```text
GT: D
Agent final: C
Tool path:
video_ls -> inspect_segment -> search_segments -> inspect_segment
-> search_segments -> inspect_segment x8

Early planner rationale:
"The video discusses the collapse of the Austro-Hungarian Empire, its ethnic diversity, economic growth..."

Final citations:
obs_0001: video_ls, navigation only
obs_0003: search_segments, navigation/search only
obs_0005: search_segments, navigation/search only
obs_0006: inspect_segment, Supported option: B
obs_0007: inspect_segment, no valid option
```

Useful failure fingerprint:

- final option C is not supported by the structured evidence table.
- citations include non-answer evidence from `video_ls` / `search_segments`.
- Inspector rows repeatedly point to B, not D or C, so the next verifier should detect "selected option unsupported" separately from "ground truth mismatch".

### 611-2: caption shortcut on temporal reasoning

```text
GT: D
Agent final: A
Tool path:
inspect_segment x4 -> search_segments x2 -> caption_segment

Planner pattern:
The planner searches for Bernini's four masterpieces and then asks a low-level caption tool to answer the MCQ directly.

Final citation:
obs_0007: caption_segment, claim starts with A and lists the four sculptures in option-A order.
```

Useful failure fingerprint:

- This is not an incomplete-loop failure.
- The answer depends on a single caption-style row.
- The verifier should require explicit temporal/order extraction across the four sculptures and compare each MCQ option sequence.

### 612-1: correct final with unresolved conflict

```text
GT: B
Agent final: B
Tool path:
inspect_segment x5 -> caption_segment -> zoom -> inspect/caption mix over later windows

Final citations:
obs_0014, obs_0018, obs_0019, obs_0021

Conflicting rows:
obs_0002 supports B.
obs_0008, obs_0013, obs_0016, obs_0017 support A.
```

Useful success fingerprint:

- The final answer is correct, but correctness comes after a long trace with conflicting option support.
- This should be labeled as "resolved conflict required", not a clean single-evidence positive.
- For later SFT/RL, this trajectory should be filtered or annotated by verifier verdict before becoming a positive sample.

## Model Interaction Problems

### 1. Option Grounding Drift

The planner sometimes passed only letter options such as `["A", "B"]`, even though the task requires full MCQ option text. This makes tool-local evidence attribution unstable because the inspector cannot compare against the complete candidate sequences.

Current mitigation: full question options are propagated into `candidate_options` and appended to local tool questions.

Remaining risk: the final answer step can still cite a later observation whose `supported_option` conflicts with other high-confidence observations.

### 2. Dynamic Workspace Address Instability

The planner treated dynamic windows inconsistently:

- It reused `window_<start_ms>_<end_ms>` ids without supplying `start_sec` / `end_sec`.
- It supplied millisecond-valued `start_sec` / `end_sec`.
- It requested a tail window whose end slightly exceeded the true float duration.

Current mitigation: reusable dynamic ids resolve through the current `VideoMap`, millisecond-valued spans are normalized when unambiguous, and tail windows clamp to true duration.

Remaining risk: tool schema should make valid addressing easier than invalid addressing, especially for dynamically created windows.

### 3. Evidence Conflict Is Not Arbitrated

Free exploration proves the agent can retrieve answer-bearing evidence, including evidence that points to the ground-truth option. The failure moves to synthesis: the final step selects one later A-supporting observation while ignoring a D-supporting earlier observation.

Required verifier behavior:

- Detect multiple supported options in the ledger.
- Compare cited option support against uncited conflicting support.
- For temporal-ordering MCQ, extract event order from evidence rows and compare each option sequence explicitly.
- Block final if confidence comes mainly from observations that also declare weak visual grounding.

### 4. Caption Shortcut And External-knowledge Leakage

Some observations state a supported option while their limitations say the claim is inferred, not directly visible, or based on external knowledge. In `611-2`, this allows a caption-style observation to become the final cited source even though it is weaker than it looks.

Required AnswerAgent behavior:

- Prefer evidence with explicit visual or textual grounding.
- Downweight claims whose limitations mention inference, missing explicit cues, or external knowledge.
- Never treat `supported_option` alone as proof; it must be tied to the claim and source artifacts.

### 5. Budget Is No Longer The Only First-order Diagnosis

Earlier 4-round runs mostly failed by incompletion. The no-budget/free-explore `611-2` run reaches a final answer and inspects many relevant spans, but remains wrong.

Interpretation:

- Tool use can improve recall and finality.
- More exploration alone does not guarantee accuracy.
- The next quality bottleneck is AnswerAgent/Verifier arbitration, not simply more rounds or more tool calls.

### 6. Planner JSON Brittleness On Quoted Options

The Round 2 free sync run exposed a separate reliability failure before tool use:

- `605-1` and `611-2` `agent_v2` returned `status=error`.
- Failure fingerprint: `JSONDecodeError: Expecting ',' delimiter`.
- Likely cause: planner copied MCQ option text containing double quotes into JSON string values without escaping.
- This is a model-output contract issue, not a vision-tool failure.

Current mitigation:

- Prompt now tells the planner to pass only option letters in JSON `candidate_options`, such as `["A", "B", "C", "D"]`.
- The harness restores full candidate option text from the original question before invoking `inspect_segment`.
- If planner JSON still fails to parse, the agent writes a compact `planner_json_parse_error` trace event and falls back to localized `inspect_segment` instead of aborting the strategy.

Remaining risk: parser fallback preserves execution but does not recover the planner's intended target segment if the malformed JSON contained a non-default segment id. The next trace should verify whether fallback segment choice is sufficient for `605-1` and `611-2`.

### 7. Inspector Support Lives In Claim Text

The source-machine visual harness full run exposed a schema mismatch:

- Real Inspector observations often encode support as natural-language claim text, for example `Supported option: B.`
- `raw_output.supported_option` may be absent.
- If the harness only reads structured raw fields, AnswerAgent/Verifier see these rows as `unassigned`.

Current mitigation:

- `EvidenceWorkspace.evidence_table()` parses common claim-text forms such as `Supported option: X.`
- A regression test covers the colon form.

Remaining risk:

- Claim-text parsing recovers option letters but not rich fields such as `grounding_quality`, event order, or whether the support is visual, caption-only, inferred, or external-knowledge-dependent.
- The next Inspector prompt/schema should ask for explicit `supported_option`, `grounding_quality`, `event_label`, and `option_relation` fields, while keeping the natural-language claim for readability.

## Design Decision

For the current research phase, keep a free-exploration path:

- no per-class cost budget;
- no reserved-final pressure;
- only emergency caps such as max rounds and max tool calls per round;
- optimize for evidence recall, conflict visibility, and answer support first.

Efficiency budgets and Agentic RL should come later, after the verifier can distinguish high-quality evidence trajectories from long but unsupported ones.

## Next Actions

1. Keep collecting free-explore traces on a small anchor set, but treat them as quality diagnostics rather than efficiency results.
2. Strengthen AnswerAgent and Verifier around MCQ option consistency, evidence conflict detection, and temporal-order comparison.
3. Add metrics for conflict_rate, option_support_consistency, final_with_conflict, and unsupported_final.
4. Add a final-answer gate: if the selected option has no structured support, or citations are navigation/search-only, reject final and request targeted inspection.
5. Move Inspector self-report fields earlier: `supported_option`, `grounding_quality`, `event_label`, `option_relation`, and `limitations`.
6. Rerun the three anchor cases after final-answer gating, then rerun `611-2` after temporal-order verifier arbitration before expanding.
7. Use filtered free-explore traces later for SFT/preference/RL; do not train on unresolved conflicting traces as positive examples.
