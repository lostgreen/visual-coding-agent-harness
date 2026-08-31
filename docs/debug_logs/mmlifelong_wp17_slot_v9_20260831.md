# WP17-3 Slot v9 Debug State

## Goal

Repair the endpoint-blind WP17-3 state-machine contract, pass lifecycle canary,
then launch a new 121 x 3 construction root.

## Current Evidence

- v8 full root is stale, incomplete, and read-only: 12/363 results, 11 success,
  1 failure, 14 calls.
- Failure fingerprint: E1C2 attempted `WRITE` on a closed working encounter;
  repair returned malformed JSON. Both responses finished normally, so this was
  not completion-token exhaustion.
- The preceding capsule had closed encounter/participant records at version 3
  and multiple active slots with excessive accumulated provenance.
- The first v9 five-segment canary completed 15/15 results with zero terminal
  failures; independent state/history replay had zero errors. It crossed the
  original failing segment and used three state-preserving transaction
  abstentions.
- That canary's structural audit stopped only because abstain rows measured
  `model_output_json_chars` after evidence aliases were expanded in the
  persisted object, while ordinary rows measured the compact parsed response
  before expansion. This was bookkeeping, not a model-output or lifecycle
  failure; the root remains read-only diagnostic evidence.

## Implemented v9 Repairs

- Up to three sequential operations per slot and atomic handoff.
- Omitted working slots become version-stable `implicit_retain`.
- Changed-value UPDATE replaces provenance.
- Byte-preserving C1 suffix and shared 600-token maximum.
- Compact model-visible capsule; overhead share is diagnostic.
- Structured semantic/serialization repair and explicit truncation class.
- State-preserving transaction abstain with SER endpoint ineligibility.
- Per-arm failure isolation; full/canary caps 440/24.
- Zero-model reachability audit and focused runner/state/protocol tests.
- Abstain rows now retain both explicitly defined lengths: the compact parsed
  response before evidence-alias canonicalization and the compact persisted
  response after canonicalization and illegal-operation removal. The audit
  validates the accounting contract without assuming either length is larger.

## Latest Checks

- Focused WP17 tests: 27 passed.
- Full local suite: 675 passed.
- `py_compile`: passed.
- `git diff --check`: passed.

## Constraints

- Do not access Day-test140, Week outcomes, questions, gold answers, official
  intervals, or endpoint values before construction and structural audit finish.
- Do not overwrite or resume the v8 incomplete root.
- Do not persist raw model/OCR prose, logs, secrets, configs, or source paths.

## Next Actions

1. Commit and push the output-size accounting repair.
2. Run the remote full suite and a new zero-model reachability audit.
3. Prepare and freeze a new protocol manifest bound to the repair commit.
4. Rerun the exact five-segment lifecycle canary into a unique root and audit it.
5. If all structural gates pass, launch a unique 121 x 3 full root and monitor.
