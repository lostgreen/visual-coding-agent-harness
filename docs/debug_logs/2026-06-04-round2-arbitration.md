# Round 2 Arbitration Debug Note

Date: 2026-06-04

## Goal

Execute Round 2 from `videomme-agent-implementation-plan-round2.md`: make evidence conflict explicit before final answer generation, starting with Stage 0 metrics/table, Stage 1 AnswerAgent arbitration, and the first Stage 2 verifier gate.

## Current Evidence

Current valid local check:

- `PYTHONPATH=src python -m pytest tests` -> 93 passed.

Current implementation progress:

- Stage 0.1: `EvidenceWorkspace.evidence_table(question, options)` builds an option-grouped table from `observations.jsonl`, excluding navigation-only observations.
- Stage 0.2: `runs/report_metrics.py` now reports `conflict_rate`, `option_support_consistency_rate`, `final_with_conflict_rate`, and `unsupported_final_rate`.
- Stage 1.1-1.4 partial: `AnswerAgent` has deterministic table arbitration that aggregates `confidence * grounding_weight`, downweights weak/inferred/external-knowledge evidence, is order-invariant, and abstains on low-margin conflicts.
- Stage 1 integration: reserved-final / prefinal AnswerAgent calls receive the workspace evidence table; if the table has no option support, the old text AnswerAgent path remains as fallback.
- Stage 2.1-2.3 partial: `verify_ledger_answer` labels option relations as Support / Contradict / Neutral, rejects an uncited stronger well-grounded conflicting option, and rejects a temporal-order answer whose selected option contradicts timestamped event evidence.

## Latest Failure Fingerprint Addressed

Known `611-2` bug:

- Free-explore retrieved D-supporting `obs_0002`.
- Final synthesis cited later weaker A-supporting `obs_0010`.
- Previous reporter could not fingerprint this as an arbitration failure.

New local synthetic coverage:

- Workspace table groups D visual support and A inferred caption support.
- Reporter flags a final A citing weak evidence while D is top support:
  - `has_conflict = true`
  - `final_with_conflict = true`
  - `unsupported_final = true`
  - `option_support_consistency = false`
- AnswerAgent selects D over higher-confidence weak A evidence.
- Verifier rejects final A when a stronger uncited D observation exists.
- Verifier rejects a temporal-order final when the selected option sequence is the reverse of timestamped `event_label` observations.

Latest KML free-explore sync run:

- Summary: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_round2_free_sync_20260604/summary.json`
- `direct_full_video`: 2/3 (`605-1`, `612-1` correct; `611-2` wrong).
- `agent_v2`: `612-1` final B and correct, but `605-1` and `611-2` failed before tool use with planner JSON parse errors.
- Failure fingerprint: `JSONDecodeError: Expecting ',' delimiter`, caused by planner responses copying quoted MCQ option text into JSON string values.
- Mitigation added: prompt tells planner to pass option letters only in JSON `candidate_options`, and the agent now records `planner_json_parse_error` then falls back to localized `inspect_segment` instead of aborting.

## Files Changed In This Iteration

- `src/visual_coding_agent_harness/workspace.py`
- `src/visual_coding_agent_harness/agents/answer_agent.py`
- `src/visual_coding_agent_harness/agents/iterative_agent.py`
- `src/visual_coding_agent_harness/tools/verification.py`
- `runs/report_metrics.py`
- `tests/test_answer_agent.py`
- `tests/test_harness.py`
- `tests/test_iterative_agent.py`
- `tests/test_report_metrics.py`
- `tests/test_verification_tools.py`

## Current Hypothesis

The round-2 direction is locally validated: explicit table arbitration can remove recency/position bias in the known failure class. The latest KML run also shows that model-output JSON brittleness is a first-class reliability issue for MCQ questions with quoted option text. After the parser fallback is synced, the next question is whether the real failed cases produce enough structured `supported_option` / `grounding_quality` information for deterministic arbitration to fire.

## Stale Evidence

- The old "just increase budget" diagnosis remains stale.
- The old missing dynamic-window crash remains stale.
- Local synthetic tests are current, but they are not a replacement for a KML `611-2` re-baseline.

## Next Actions

1. Sync the planner JSON fallback patch to KML and run remote unit tests with compact output.
2. Rerun `agent_v2 --free-explore` on failed `605-1,611-2` cases to confirm the parser failure no longer aborts tool use.
3. Run reporter on the existing `videomme_agent_free_explore_611_msfix_20260604` workspace to confirm the real trace is flagged.
4. If real trace lacks structured `supported_option` or `event_label`, move the Stage 4 Inspector self-report earlier.
5. Then rerun the 3-case anchor set before expanding benchmark coverage.
