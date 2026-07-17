# P0.5 Strict-Grounding Safety State

## Goal

Eliminate false `verified_strict` answers before continuing P1 capability work.

## Latest Failure Fingerprint

- Baseline commit `faa1a95` could certify a wrong answer when the condition registry was empty, choice/candidate readiness was false, or the answer audit did not explicitly support the selected option predicate.
- The direct KML `--case-ids` invocation is invalid for constructed IDs `441-4`, `445-2`, and `445-3`; use `configs/eval_groups/vcah_p0_cases4.json`.

## Current Change

- Strict grounding now fails closed on missing critical conditions, missing choice/candidate readiness, unparseable audits, and unsupported selected-option predicates.
- Normal, audit, and forced paths share `requires_option_audit` and the canonical option verdict table.
- Repair tasks inherit gap, condition, boundary episode, and target option-predicate lineage.
- Local/global condition scope is enforced; local observations cannot close global conditions.
- Decision-time answer and entity-cluster facts are merged into the same completion status before the gate is evaluated.

## Verification

- Local: `PYTHONPATH=src:. pytest -q` -> `338 passed`.
- `py_compile` and `git diff --check` pass.
- One-case KML smoke (`441-2`) answered incorrectly but was correctly downgraded to `forced_choice`, `grounding_status=insufficient`, `verified=false`.
- Four-case seeded KML replay completed successfully at
  `/m2v_intern/xuboshen/zgw/VideoAgent/vcah_p05_seeded_replay_20260717/all_summary.json`.
- Result: 4/4 answers were `forced_choice`, `grounding_status=insufficient`, and `verified=false`.
- False strict count: 0. Accuracy: 1/4 (`441-4` correct); accuracy remains P1 work.

## Stale Evidence

- Runs under `vcah_p05_cases4_20260717`, `vcah_p05_cases4_all_20260717`, and
  `vcah_p05_casegroup_20260717` used an invalid raw-qid construction path and do not represent the four-case regression.
- Earlier `faa1a95` KML outputs are baseline evidence only.

## Next Actions

1. Commit and push P0.5 as an independent safety change.
2. Continue P1 option-accuracy work from the canonical facts and shared verdict table.
3. Focus first on `441-2`, `445-2`, and `445-3`, without weakening strict fail-closed invariants.
