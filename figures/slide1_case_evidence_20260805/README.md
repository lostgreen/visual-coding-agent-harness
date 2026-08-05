# Slide 1 case evidence figures

This directory contains slide-ready evidence figures for the claim that the
MGER iterations produced informative NO-GO results rather than merely failing
to improve a score.

## Outputs

- `fig_case0031_tool_call_not_evidence.{png,pdf}`
- `fig_case0038_reference_not_grounding.{png,pdf}`
- `fig_case0146_false_precision.{png,pdf}`
- `fig_phase4_observation_collapse.{png,pdf}`
- `fig_slide1_case_evidence_overview.{png,pdf}`

The PNG files are rendered at 1920 x 1080 for presentation use. The PDF files
keep text and chart elements as vectors while embedding the source frames.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/vcah-mpl python3 figures/slide1_case_evidence_20260805/gen_slide1_case_evidence.py
```

The script reads only `evidence_data.json` and the eight source frames under
`raw/`. It does not call a model or access the network.

## Evidence provenance

- Case 0031 transcriptions come from three saved MM-Lifelong runs over the same
  visual material.
- Case 0038 compares the inspected Tiger Vanguard occurrence with the official
  Old Rattle-Drum clue occurrence.
- Case 0146 uses the exact two timestamps emitted by the sparse-window
  Investigator and the frames reached by the subsequent narrow probes.
- Phase 4 metrics are copied from `docs/MGER_PHASE4_REPORT_20260804.md`.

Exact values and run identifiers are recorded in `evidence_data.json`.
