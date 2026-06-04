# Model Interaction Records: VideoMME Free-explore Pass

Date: 2026-06-04

## Purpose

This note collects the current model-interaction issues and the raw record pointers from the VideoMME agent-loop debugging pass.

It intentionally does not paste raw logs, full `trace.jsonl`, full `observations.jsonl`, or model responses. The raw records stay as artifacts on KML; this note keeps only compact summaries, failure fingerprints, and paths.

## Current Evidence Boundary

Current evidence:

- KML repo: `/home/xuboshen/zgw/visual-coding-agent-harness`
- KML Python: `/home/xuboshen/Anaconda/envs/VLMEvalKit/bin/python`
- Local debug note: `docs/debug_logs/2026-06-04-phase0-eval.md`
- This note: `docs/debug_logs/2026-06-04-model-interaction-records.md`

Stale or superseded evidence:

- The old missing-`segment_id` crash is fixed.
- The old conclusion "budget exhaustion is the main blocker" is incomplete. Free exploration removes incompletion on `611-2`, but answer synthesis still chooses the wrong option.
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
4. Rerun `611-2` after verifier arbitration before expanding to a larger benchmark slice.
5. Use filtered free-explore traces later for SFT/preference/RL; do not train on unresolved conflicting traces as positive examples.
