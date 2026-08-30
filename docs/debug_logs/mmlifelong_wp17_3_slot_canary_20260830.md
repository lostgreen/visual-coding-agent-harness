# MM-Lifelong WP17-3 Slot Canary Debug State

## Goal

Validate the question-blind 120-second `E1C0/E1C1/E1C2` construction transaction before the 121-segment run.

## Current Evidence

- Commit `864f18a` protocol-v1 canary stopped at the first `E1C0` result.
- Root: `/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-3-slot-canary3-864f18a-20260830`.
- Compact fingerprint: attempt 1 finished normally but cited a non-packet evidence ID; attempt 2 was not a complete JSON object after a length finish.
- The segment packet contained 610 overlapping OCR records but only 260 normalized surfaces; 350 rows were duplicate surface tracks.
- No endpoint values were produced or inspected. The v1 root is structural-debug evidence only.

## Repair

- Aggregate overlapping OCR records by normalized surface while preserving disjoint local time ranges, source counts, support-frame counts, and a deterministic canonical aggregate ID.
- Give the model packet-local `fNNN/oNNN/aNNN` evidence aliases and canonicalize them before persistence and slot-state mutation.
- Freeze deterministic output-size limits and explicitly prohibit exhaustive OCR transcription.
- Keep 120 seconds, 1 fps, frame preprocessing, all three arms, state semantics, model, and endpoint definitions unchanged.

## Next Checks

1. Run focused tests and the full local/remote suites.
2. Freeze protocol v2 to a new zero-call root.
3. Launch a new 3-segment canary root and require 9/9 successful independently replayable results.
