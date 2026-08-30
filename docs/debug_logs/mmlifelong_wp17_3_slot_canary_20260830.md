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

## Protocol-v2 Result

- Commit `dbebdb3` fixed evidence addressability and reduced the first canary segment from 610 OCR rows to 260 aggregates.
- `E1C0` and `E1C1` each passed on their first model call.
- `E1C2` stopped after two deterministic rejections: both responses finished normally, but the active slot capsule exceeded the unchanged 600-token hard budget.
- Root: `/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-3-slot-canary3-dbebdb3-20260830`.
- No endpoint values were produced or inspected. The v2 root is structural-debug evidence only.

## Protocol-v3 Repair

- Preserve the 600-token hard gate and all experiment inputs.
- Define the slot capsule as sparse cross-segment working memory; keep segment-local detail in the structured event record.
- Freeze a 400-token soft target and return actual capsule token/active-slot counts in deterministic retry feedback.

## Protocol-v3 Result

- Commit `87aa29c` passed first-segment `E1C0/E1C1/E1C2` in three calls; the sparse slot instruction resolved the capsule-budget blocker.
- On the second segment, `E1C1` recovered once and succeeded. `E1C2` then stopped after two schema-level rejections: an operation referenced a non-observation ID, followed by one SER field encoded as a singleton instead of a list.
- Root: `/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-3-slot-canary3-87aa29c-20260830`.
- No endpoint values were produced or inspected. The v3 root is structural-debug evidence only.

## Protocol-v4 Repair

- Explicitly bind `slot_operations.observation_ids` to IDs from the current `observations` array, never evidence IDs.
- Include unknown and valid observation IDs in deterministic retry feedback.
- Mechanically normalize SER string/object/null singletons to lists; no facts, evidence, or slot values are inferred.

## Protocol-v4 Result

- Commit `7ba13a9` passed the local and remote suites (`661/661`).
- The fresh canary stopped after four successes and one `E1C2` terminal validation failure.
- Compact fingerprint: attempt 1 exceeded the six-evidence-per-observation bound; attempt 2 reduced evidence references but produced six active slots whose capsule used 852/600 tokens.
- Endpoint-blind accounting found the first `E1C2` capsule already used 392 tokens with four slots because each slot repeated five full canonical evidence IDs. State and lifecycle ledger already preserve that complete lineage.
- Root: `/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-3-slot-canary3-7ba13a9-20260830`.
- No endpoint values were produced or inspected. The v4 root is structural-debug evidence only.

## Protocol-v5 Repair

- Preserve complete canonical provenance in state records and the append-only lifecycle ledger.
- Project model-visible working-capsule provenance to deterministic count and SHA256 digest fields; do not copy raw evidence IDs into history context.
- Keep the 600-token hard budget, 400-token soft target, slot values, lifecycle, evidence packets, model, retry cap, and endpoint definitions unchanged.
- Independently replay and compare every persisted capsule against the deterministic projection.

## Protocol-v5 Result

- Commit `ca44421` passed the local and remote suites (`662/662`).
- Zero-model replay reduced the same first-segment E1C2 capsule from 392 to 282 tokens with an identical state digest, all 20 state provenance links retained, and no raw evidence IDs in working context.
- The fresh canary passed the first segment, then stopped on second-segment `E1C1`: both attempts cited more than six valid current-packet evidence IDs for one observation.
- Root: `/m2v_intern/xuboshen/zgw/mger_runs/mmlifelong-wp17-3-slot-canary3-ca44421-20260830`.
- No endpoint values were produced or inspected. The v5 root is structural-debug evidence only.

## Protocol-v6 Repair

- Keep all current-packet evidence IDs and canonicalize them without truncation.
- Change six evidence IDs per observation from an arbitrary hard rejection to a prompt soft target; the 16-observation and 10,000-character global bounds remain hard.
- Preserve packet membership validation, complete provenance, capsule projection, model, three-arm inputs, call caps, and endpoint definitions.
