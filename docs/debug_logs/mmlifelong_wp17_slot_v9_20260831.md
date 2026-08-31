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

1. Commit and push v9 code.
2. Run remote full suite and zero-model reachability audit.
3. Prepare/freeze a unique v9 protocol from the v8 structural manifest.
4. Run the 5-segment lifecycle canary and independent audit.
5. If all structural gates pass, launch a unique 121 x 3 full root and monitor.
