# P1 Qualification State

## Goal

Prevent canonical candidates from becoming query-qualified facts until a program-evaluated requirement graph is complete; keep predicate support independent from grounding eligibility and hard forced override.

## Current Failure Fingerprint

- `445-2`: after the merge fix, canonical qualification finds 2 of the gold 3 events. The remaining miss is event discovery/prior-state recovery, not a second counting path.
- `445-3`: the sampled replay remains incomplete and forced; episode/boundary/entity blockers correctly prevent hard override.

## Current Change

- Added `src/vcah/qualification.py` with requirement graph evaluation, dependency blocking, fail-closed custom predicates, event qualification, typed option predicates, and lineage-scoped observation updates.
- Event ledger now owns canonical candidates only. `canonical_fact_snapshot()` computes `qualified_events`; `confirmed_events` is a compatibility alias.
- Option predicate verdict and grounding eligibility are independent. OptionVerdictTable reports unique support separately from hard-override eligibility and blocker telemetry.
- Query contracts compile full-video coverage, temporal max, ordinal participant, entity binding, and attribute requirements.
- Repair tasks carry requirement, candidate, episode, entity, and typed option-predicate lineage.
- Narrative facts are namespaced by episode and relation type; out-of-episode facts are excluded and co-occurrence cannot support causation.
- Same-occurrence actor attribute disagreement now produces an ambiguous participant binding while preserving one countable event; it no longer invalidates the occurrence.

## Verification

- Local: `PYTHONPATH=.:src pytest -q` -> `358 passed`.
- `py_compile` and `git diff --check` pass.
- KML implementation files uploaded to `/tmp/vcah_p0_eval` with SHA verification.
- Replay output: `/m2v_intern/xuboshen/zgw/VideoAgent/vcah_p1_qualification_replay_20260718`.
- Final `445-2`: forced `G=2` versus gold `E=3`, verified false; qualified 2, conflicted 0, hard override blocked. Before the merge fix it had qualified 0 and conflicted 9.
- Safety replay `445-3`: forced and verified false; hard override blocked by episode/boundary/entity requirements.

## Stale Evidence

- `/m2v_intern/xuboshen/zgw/VideoAgent/vcah_p05_seeded_replay_20260717/all_summary.json` is the P0.5 baseline only: false strict 0, accuracy 1/4.
- Earlier `vcah_p05_cases4*`, `vcah_p1_final_cases4*`, and raw-qid runs are not current P1 Qualification evidence.

## Remaining Work

1. Improve event discovery and prior-state recovery for the missing `445-2` occurrence as a later capability iteration.
2. Re-evaluate answer accuracy on a larger seeded set; do not weaken qualification or hard-override blockers to recover accuracy.
