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

Latest source-machine visual-harness run:

- Python: `/home/xuboshen/Anaconda/envs/visual-agent-harness/bin/python`
- Summary: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_visual_harness_full_20260604/summary.json`
- `direct_full_video`: 2/3; `605-1=D` correct, `611-2=A` wrong, `612-1=B` correct.
- `agent_v2 --free-explore`: 1/3; all cases reached `final` with tool use and no JSON/runtime error.
- Case metrics: `605-1` final C after 9 rounds / 13 tools; `611-2` final A after 6 rounds / 7 tools; `612-1` final B after 20 rounds / 21 tools.
- Reporter: `agent_v2` final_rate 100%, incomplete_rate 0%, avg 304 sec, but accuracy still below direct baseline.

Latest failure fingerprint after run:

- Real inspector outputs encode option support inside claim text, e.g. `Supported option: A.`, not in `raw_output.supported_option`.
- Existing `evidence_table()` did not parse the colon form, so AnswerAgent/Verifier arbitration saw many visual observations as `unassigned`.
- Mitigation added: parse `Supported option: X.` claim text into structured table support.

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

1. Rerun reporter on `videomme_agent_visual_harness_full_20260604` after the claim-text parser fix to quantify conflict/unsupported-final tags.
2. Rerun a focused `agent_v2 --free-explore` slice if the parser fix affects verifier/AnswerAgent behavior.
3. Move Stage 4 Inspector self-report earlier if claim-text parsing is still too weak for `grounding_quality` and `event_label`.
4. Add a final-answer gate for MCQ: when final choice has no structured support but other options do, require verifier/AnswerAgent arbitration before accepting final.
5. Then rerun the 3-case anchor set before expanding benchmark coverage.
