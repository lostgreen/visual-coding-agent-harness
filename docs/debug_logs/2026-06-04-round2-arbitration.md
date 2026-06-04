# Round 2 Arbitration Debug Note

Date: 2026-06-04

## Goal

Execute Round 2 from `videomme-agent-implementation-plan-round2.md`: make evidence conflict explicit before final answer generation, starting with Stage 0 metrics/table, Stage 1 AnswerAgent arbitration, and the first Stage 2 verifier gate.

## Current Evidence

Current valid local check:

- `PYTHONPATH=src python -m pytest tests` -> 91 passed.

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

The round-2 direction is locally validated: explicit table arbitration can remove recency/position bias in the known failure class. The next question is whether the real `611-2` trace contains enough structured `supported_option` / `grounding_quality` information for this deterministic layer to fire, or whether Stage 4 Inspector self-report is needed sooner.

## Stale Evidence

- The old "just increase budget" diagnosis remains stale.
- The old missing dynamic-window crash remains stale.
- Local synthetic tests are current, but they are not a replacement for a KML `611-2` re-baseline.

## Next Actions

1. Sync this patch to KML and run remote unit tests with compact output.
2. Run reporter on the existing `videomme_agent_free_explore_611_msfix_20260604` workspace to confirm the real trace is flagged.
3. If real trace lacks structured `supported_option`, add a parser or move Stage 4 grounding self-report earlier.
4. If real temporal-order traces lack structured `event_label`, move the Stage 4 Inspector self-report earlier so observations emit event names with timestamps.
5. Then rerun `611-2` free-explore with the AnswerAgent/Verifier changes.
