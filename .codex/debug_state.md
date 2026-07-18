# Final Iteration State

## Goal

Complete the final VCAH qualification and adjudication plan: one final selector,
qualified-only canonical facts, fail-closed provenance, episode/entity binding,
and immutable multi-seed replay artifacts.

## Current State

- `final_adjudicate` is the only final-answer selector. The runner, dashboard,
  audit, and removed `_canonical_forced_answer` helper cannot mutate answers.
- Forced prompts now receive a read-only canonical finalization projection;
  the parallel option-support recommendation/dashboard has been removed.
- All-option audits and verdict tables are revision-bound to the same canonical
  snapshot, evidence digest, and query contract. Missing source revisions,
  partial option audits, and unknown competing predicates fail closed.
- Qualification, obligations, enumeration, provenance, typed state transitions,
  narrative attribution, episode binding, and sequence ledgers are implemented.
- Event qualification provenance is admissible only when a distinct witnessed
  observation corroborates the same canonical candidate. Repackaging one
  observation under multiple evidence IDs remains heuristic.
- Query obligations are compiled once per driver run. Coverage, enumeration,
  bootstrap, repair, and answer gates consume the resulting `effective_scope`.
- Soft audit correction is available only inside `final_adjudicate` for a
  fresh complete audit, an explicitly contradicted raw option, exactly one
  admissibly supported alternative, and no episode-binding conflict. It remains
  `forced_choice` with insufficient grounding.
- Replay runs are create-exclusive under `runs/<run_id>/`; each case records
  source/frame/trace checksums, content hashes, provider metadata, retry count,
  investigation ordering, and final semantic telemetry. `--seeds` emits
  per-case distributions and the targeted seed protocol status.

## Verification

- `python -m py_compile` succeeds for the changed runtime modules.
- `PYTHONPATH=.:src pytest -q` -> `384 passed`.
- `git diff --check` passes.

## Current Evidence

- Local semantic and replay tests are current.
- KML replay artifacts from P1 are stale for this final implementation and must
  not be used as release evidence.

## Next Actions

1. Commit, push, and archive the exact review-fix snapshot.
2. Pull that exact commit on KML and run the targeted immutable multi-seed
   replay for `441-2`, `441-4`, `445-2`, and `445-3`.
3. Compare the generated `summary.json` safety and semantic metrics; preserve
   any failure as an artifact rather than relaxing a gate.
