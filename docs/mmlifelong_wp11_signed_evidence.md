# MM-Lifelong WP11 Signed-Evidence Shadow

## Scope

WP11 tests evidence judgment after the WP10.1 declaration/report protocol was
made structurally reliable. It does not change R5, actionability, retrieval,
budgets, or evaluation splits.

## Declaration Contract

For each question-critical constraint, the rule-blind Reasoner may sparsely
declare:

- `supported_candidates`: direct visible evidence supports the constraint.
- `contradicted_candidates`: direct visible evidence contradicts the constraint.
- omission: no direct positive or negative evidence, normalized to unknown.

Every declared row must cite a visible passage owned by the same candidate. A
candidate cannot be both supported and contradicted for one constraint. The
Reasoner does not receive the R5 rule, winner, margin, or gate outcome.

## Shadow Invariant

Contradictions are trace-only in WP11-1 and WP11-2. Runtime computes the R5
winner from `supported_candidates` exactly as before. The audit independently
reconstructs positive support counts and requires:

- `signed_evidence_shadow = true`;
- `contradiction_affects_gate = false`;
- complete positive/negative/unknown accounting;
- gate support counts equal the counts reconstructed from positive rows only.

Endpoint values never determine structural validity.

## WP11-2 Diagnostics

Two independent frozen39 repeats report:

- contradicted-row Jaccard;
- strict contradicted-row Jaccard, including description and passage IDs;
- candidate-level contradiction agreement;
- constraint-level contradiction agreement;
- gold and non-gold contradiction rates overall and by constraint type.

Signed evidence advances to a separate guard experiment only if the non-gold
minus gold contradiction direction is positive in both repeats, with special
attention to identity and event constraints. Otherwise the next candidate is
evidence-row consensus or deterministic extraction. No guard threshold is
tuned on frozen39 during the shadow phase.
