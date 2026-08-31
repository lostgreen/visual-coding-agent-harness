# MM-Lifelong WP17-3 Slot Memory v9 Repair And Execution Plan

## Objective

Repair the 120-second `E1C0 / E1C1 / E1C2` construction experiment without
using questions, answers, official intervals, Day-test140, Week outcomes, or
construction endpoint values. The intervention remains Slot Memory
Construction. This revision repairs the state-machine and baseline contracts;
it does not tune the scientific endpoint.

## Current Structural Evidence

The v8 full run is incomplete and read-only diagnostic evidence. It stopped at
12 of 363 results (11 success, 1 failure, 14 model calls).

The compact forensic reconstruction established:

- The first rejected E1C2 response ended normally and attempted to `WRITE` an
  already-working `active_encounter` slot.
- The second response also ended normally. It was malformed JSON, not a
  completion-token truncation.
- The preceding state contained closed encounter/participant slots that could
  not be replaced in one transaction under v8's one-operation-per-slot rule.
- Several changing slots had accumulated more than 100 provenance entries,
  confirming that changed-value `UPDATE` incorrectly retained stale lineage.

The blocker is therefore a deterministic contract defect, not evidence that
the model cannot maintain slots.

## v9 Contract

### Lifecycle reachability

- One slot may receive up to three ordered operations in one transaction.
- Versions are checked after every operation.
- `close -> archive -> write` is a legal atomic encounter handoff.
- A working slot omitted from `slot_operations` becomes a version-stable
  `implicit_retain` event.
- Cross-slot participant/encounter invariants are checked on the final atomic
  candidate state.

### Provenance

- `UPDATE` with a changed value replaces provenance with current evidence.
- `UPDATE` with the same value and explicit `RETAIN` may extend provenance.
- Full provenance remains in state and the append-only ledger.
- The model-visible capsule excludes provenance counts/digests and duplicate
  working-slot version maps. Inactive slot versions remain available for legal
  re-entry.

### Repair and fallback

- Semantic validation errors return a structured repair contract with a stable
  error code and safe state descriptors.
- Malformed and token-truncated responses are distinct failure classes.
- The runner allows at most three calls per result.
- If E1C2 produced a structurally valid observation/SER payload but all slot
  repairs fail, runtime may accept `transaction_abstain`:
  - slot state and ledger remain unchanged;
  - slot operations are discarded;
  - SER is retained only as diagnostic output;
  - `ser_endpoint_eligible=false` and
    `ser_trust_status=untrusted_for_endpoint` are mandatory.
- `transaction_abstain` is not sample exclusion and cannot improve an
  endpoint. Later endpoint analysis must count it as an E1C2 endpoint failure.

### Matched baseline fidelity

- E1C1 uses a token-span suffix of the original previous caption. It never
  tokenizes and rejoins the text, so Chinese and punctuation remain unchanged.
- E1C1 and E1C2 both receive the same 600-token maximum.
- Realized token counts are reported separately. Their difference is a
  diagnostic, not a structural gate.
- A broken E1C1 chain is explicit and may not silently become E1C0.

### Failure isolation and cost

- A failed arm result is persisted and the remaining independent work is
  collected; one arm no longer aborts the process immediately.
- Full-run base calls remain 363. The hard cap is 440, providing at least 20%
  retry capacity.
- The 5-segment canary has 15 base calls and a 24-call hard cap.
- Capsule overhead share is reported as a diagnostic and is never a hard gate.

## Zero-Model Gates

Before any construction call:

1. Run all focused tests and the full repository suite.
2. Run the independent reachability audit.
3. Require all six explicit operations, implicit retain, atomic handoff,
   provenance replacement, snapshot replay, and capsule budget checks to pass.
4. Freeze a unique v9 protocol manifest from the prior v8 structural metadata.
5. Verify exact frozen evidence/model/scope digests and zero endpoint access.

## Lifecycle Canary

Select a consecutive five-segment chain containing the prior structural
failure trigger. Selection is based only on the observed lifecycle failure,
not on any endpoint value.

Require:

- 15/15 successful arm results;
- independent replay with zero errors;
- exact original-text E1C1 history replay;
- exact E1C2 state/capsule replay;
- zero terminal results;
- every abstain, if any, is state-preserving and SER-ineligible;
- all capsules are within 600 tokens;
- endpoint values were not evaluated.

Illegal-operation and abstain rates are reported diagnostics, not canary gates.
Any violation of the fallback trust contract is a hard stop.

## Full Run

Only after the canary passes, launch a new 121-segment by 3-arm root. Never
resume or overwrite the v8 root. Monitor only result counts, success/failure,
model-call count, liveness, and storage. After 363/363 results, run the same
independent structural audit. Pause before endpoint analysis.

## Interpretation Boundary

Passing v9 establishes lifecycle expressibility and construction-protocol
reliability. It does not establish that Slot Memory outperforms Previous
Caption. That scientific conclusion remains the later endpoint comparison
`E1C2 - E1C1` under the frozen construction outputs.
