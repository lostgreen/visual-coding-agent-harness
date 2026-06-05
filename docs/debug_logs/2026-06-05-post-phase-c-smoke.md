# Phase D KML 3-Case Smoke Result

## Scope
- Branch: `codex/visual-harness-ticket-plan`
- Commit: `ca9266c`
- Command shape: `runs/eval_runner.py --strategy agent_v2 --cases 605-1,611-2,612-1 --hard-skill-runtime --export-training --allow-any-python`
- KML run root: `/home/xuboshen/zgw/visual-coding-agent-harness/runs/videomme_agent_phase_d_hardskill_3case_negative_caption_20260605_kml`
- KML log artifact: `/tmp/videomme_phase_d_hardskill_3case_negative_caption_20260605_kml.log`

## Result
- Process result: `returncode=0`
- Summary violations: `[]`
- Training trajectory export: `true`
- Exported trajectories: `3`
- Route violations: `0`
- Unsupported citation rate: `0.0`
- Tool nframes compliance: `1.0`

## Metrics
- `accuracy`: `0.0`
- `final_rate`: `0.0`
- `timeline_completeness`: `0.0`
- `context_budget_overflow_count`: `18`
- `evidence_provenance_completeness`: `0.3333333333333333`
- `normalization_notes_per_round`: `1.4285714285714286`

## Case Status
- `605-1`: `max_rounds_reached`, no selected choice, GT `D`
- `611-2`: `max_rounds_reached`, no selected choice, GT `D`
- `612-1`: `max_rounds_reached`, no selected choice, GT `B`

## Current Interpretation
The smoke now completes cleanly at the harness/process level and exports all training trajectories. The Phase D semantic target metrics remain unmet. The most recent fix prevents negative caption prompt echoes from becoming false 0-second timeline entries; after that fix, 611-2 correctly avoids unsupported timeline reads but still needs better evidence recovery to answer.
