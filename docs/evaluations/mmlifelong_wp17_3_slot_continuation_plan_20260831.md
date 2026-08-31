# MM-Lifelong WP17-3 Slot Construction Continuation

## Objective

Complete the endpoint-blind 121-segment `E1C0/E1C1/E1C2` construction run
without overwriting its cap-exhausted parent root or silently treating failed
stateful rows as independent samples.

## Current Evidence

The parent run persisted all 363 result positions. It contains 302 successes,
54 call-cap placeholders, and 7 terminal validation failures. Safe response
metadata rules out final token truncation as the common cause: every terminal
attempt ended with `finish_reason=stop`; the only internally retried truncation
also ended with a non-truncated malformed response.

## Frozen Dependency Policy

- `E1C0`: rerun only non-success rows.
- `E1C1`: rerun from the first non-success row through the end of that local
  window because every later previous-caption digest depends on the rerun.
- `E1C2`: rerun from the first non-success row through the end of that local
  window because every later slot-state digest depends on the rerun.
- Reuse is allowed only when the parent result is successful, its file SHA256
  matches the frozen plan, and independent history/state replay succeeds.

For the current parent this yields 273 reusable rows and 90 rerun rows. The
continuation cap is `90 * 3 = 270` calls, preserving the existing maximum of
three attempts per result while preventing another aggregate-cap placeholder.

## Provenance And Audit

The continuation uses a new protocol manifest, plan, output root, and source
commit. The plan records every result action and parent SHA256. The merged root
records parent calls, continuation calls, reuse/rerun counts, and total calls.
The independent audit recomputes dependency closure, verifies copied rows are
byte-equivalent apart from continuation provenance, and replays all 363 rows.

No endpoint value, question, answer, official interval, Day-test140, or Week
outcome is consulted while preparing or executing the continuation.

## Endpoint Boundary

Only after the merged root reaches 363/363 success and the independent
structural audit passes may the pre-registered construction analysis run. Its
primary effect remains `E1C2-E1C1`; `E1C1-E1C0` is secondary. Transaction
abstentions remain endpoint failures rather than exclusions.
