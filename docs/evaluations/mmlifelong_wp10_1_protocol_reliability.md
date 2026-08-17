# MM-Lifelong WP10.1 Protocol Reliability Repair

## Objective

Repair protocol and analysis failures observed in WP10 without changing the
occurrence method. WP10.1 tests whether rule-blind supported-only evidence can
drive the existing R5 gate reliably once schema, scope, retry, and answer
parsing artifacts are removed.

## Frozen Method

- Evidence contract: `rule_blind_sparse_positive_evidence_v1`.
- Aggregation: `unique_supported_count_margin` with margin `1`.
- Candidate limit: `5`.
- Search, visual actionability, controller, budgets, backbone, judge, caption
  index, occurrence replay, and oracle arm remain unchanged from WP10.
- Signed evidence, consensus, A4.2, A5, WP7, Day-test140, and Week are out of
  scope.

## Protocol Repairs

1. Model input uses `OccurrenceEvidenceDeclarationV1`; persisted normalized
   state uses `OccurrenceEvidenceReportV1`. Report metadata is read-only, and
   report-to-declaration roundtrip must revalidate.
2. During evidence declaration, the Reasoner sees exactly one
   `evidence_scope`: active set ID, semantic target, validator-legal top-K
   candidates, and their locator passages. Out-of-scope IDs are hidden; only
   `additional_candidate_count` remains.
3. A known out-of-scope support row is dropped with a structured warning. It
   cannot fail the whole transaction. Unknown IDs remain invalid.
4. All deterministic declaration errors are returned in one retry surface.
5. `action_input.answer` is the only accepted answer alias. It is normalized
   only when top-level `answer` is empty. Conflicting values are rejected.
6. Stability analysis reports all aligned cases, evidence-valid pairs, and
   full-method-valid pairs separately. Missing events never become empty
   evidence sets.

## Canary

Use WP10 failures `0028`, `0118`, and `0017`, plus two or three normal controls.
The canary passes only when all cases meet every condition on the first launch:

- one legal model-visible evidence scope per case;
- first evidence transaction accepted without control repair;
- complete mechanical gate and resolution;
- answer-bearing prediction;
- zero terminal or contradictory occurrence failures;
- any known out-of-scope row is dropped with a non-fatal structured warning;
- schema roundtrip tests pass;
- no replacement process retry.

No frozen39 run starts before the canary passes.

Reasoner decisions use the provider's JSON-object response mode for transport
reliability. This constrains serialization only: the semantic prompt, action
schema, R5 rule, candidate scope, and budgets remain unchanged. The requested
response-format type is recorded in response metadata and participates in
matched-response request identity.

## Frozen39 Repeats

Run two independent A4 repeats with unchanged R5 and the frozen39 cohort. Each
repeat must independently have:

- 39/39 valid evidence transactions;
- 39/39 mechanical gates and resolutions;
- 39/39 answer-bearing predictions;
- zero terminal failures, contradictory gates, repair attempts, scope leaks,
  and replacement retries.

Evidence stability is measured only on evidence-valid pairs:

- supported-row Jaccard;
- strict supported-row Jaccard;
- candidate-passage Jaccard;
- support-count MAE;
- winner agreement;
- gate agreement.

Working-method thresholds remain frozen:

- winner agreement at least `0.90`;
- gate agreement at least `0.90`;
- false commit at most `0.30` in each repeat;
- commit recall at least `0.60` in each repeat;
- OSA given commit at least `0.85` in each repeat.

Passing metrics cannot compensate for a structural failure. Endpoint values
are never structural gates.

Supported-row Jaccard and its comparison with WP10 remain diagnostics. They do
not add a working-method threshold beyond winner and gate agreement.

## Post-Run Diagnostic

After both valid repeats, stratify scope size `1..5` and report false commit,
commit recall, support count, and winner margin. This is diagnostic only and
does not change R5 in WP10.1.

If every structural requirement and working-method threshold passes,
decoupled R5 becomes the working baseline. Otherwise the next experiment must
target evidence-judgment stability rather than silently retuning R5.
