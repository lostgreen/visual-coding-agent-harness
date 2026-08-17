# MM-Lifelong WP10 Evidence-Gate Decoupling Protocol

Status: preregistered before WP10 model outcomes.

## Question

Can sparse positive-evidence declarations become stable when the Reasoner is
not told how Runtime scores those declarations?

## Single Mechanism Change

WP10 keeps the frozen candidate replay, supported-only evidence philosophy,
R5 aggregation, actionability policy, budgets, models, and judge unchanged.
Only the interface and transaction ownership change:

1. The Reasoner submits one isolated `declare_occurrence_evidence` operation.
2. Its schema contains constraints and directly supported candidate-passage
   bindings only. It contains no verdict, threshold, margin, winner, or
   resolution operation.
3. Runtime validates and persists the report, computes R5, then mechanically
   commits `select` or `no_match` in a separately traced transition.
4. Signed evidence, A4.2 constraint typing, A5 pending evidence, WP7, and R5
   calibration are excluded.

## Cohort And Repeats

- Frozen cohort: the existing 39-case complete pre-WP3 Day-val cohort.
- Matched control: the existing frozen A3 pre-treatment response fixture.
- Two independent A4 evidence-elicitation repeats are required.
- Each repeat must pass all structural, no-oracle, replay, answer, locator,
  evidence-report, gate-digest, and mechanical-resolution checks.
- This experiment measures stochastic stability, not generalization. It does
  not access Day-test140 or Week.

## Primary Stability Endpoints

- Supported-row Jaccard, keyed by set, constraint type, and candidate.
- Strict supported-row Jaccard, additionally keyed by normalized constraint
  description and evidence passage IDs.
- Candidate-passage Jaccard.
- Candidate support-count MAE.
- Winner agreement.
- Gate decision agreement.

The supported-row agreement must improve over the old coupled-R5 repeat
baseline measured with the same analyzer. Winner agreement and gate agreement
must each be at least 90% (at most three drifts among 39 cases).

## Per-Repeat Performance Guardrails

Both independent repeats must satisfy every guardrail:

- false commit rate <= 30%;
- commit recall >= 60%;
- OSA given commit >= 85%.

Taking an average across one passing and one failing repeat is prohibited.
Bound-visual clue recall, locator usage, costs, exact accuracy, verified
accuracy, and grounded correctness are reported diagnostics, not selection
gates. QA is used only to detect catastrophic collapse.

## Decision

WP10 becomes a working method only when the evidence stability criteria and
all three performance guardrails pass in both repeats. If evidence still
drifts, the next study targets support judgment stability (for example
consensus, deterministic extraction, or a separate signed-evidence ablation),
not R5 calibration or a new downstream gate.
