# Visual Coding-Agent Harness

This repository is a research prototype for a multimodal coding-agent-style harness.

The active path is the evidence-centric `multi_v3` long-video loop: a Reasoner
plans scoped evidence queries, a Driver dispatches them, and Investigators
explore low-resolution shot grids before verifying high-resolution frames.

## Run Tests

```bash
PYTHONPATH=src:. pytest
```

The legacy unittest discovery command remains supported for compatibility:

```bash
PYTHONPATH=src:. python3 -m unittest discover tests
```

## Project Layout

Package code lives under `src/visual_coding_agent_harness/`. Evaluation
runners and ablation helpers live under `src/visual_coding_agent_harness/evals/`;
command modules live under `src/visual_coding_agent_harness/cli/`.

`runs/` is reserved for generated run artifacts. The legacy Python files in
`runs/` and `scripts/` are compatibility wrappers for older commands.
Prefer the package CLIs for new usage:

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.eval_videomme --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.run_ablation --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.generate_ablation_report --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.audit_trajectory --help
```

Legacy P0 demos, traditional tools, and early VLM smoke notes live in
`docs/legacy_p0.md`.

## Evidence-Centric Long-Video Agent

The active long-video evaluation path is `--strategy multi_v3`. It uses a small
Reasoner -> Driver -> Investigator loop:

- `Reasoner` plans scoped queries or returns the final answer.
- `Driver` validates scene scope, dispatches queries in parallel, and feeds back
  compact digests.
- `Investigator` runs `explore` over low-resolution shot grids, then `verify`
  over high-resolution frames for candidate shots.

Legacy workspace_v2 tools live under `visual_coding_agent_harness.legacy`; they
are not part of the active multi_v3 tool surface.

### Active multi_v3 module boundary

Only these modules are part of the active long-video path:

```text
agents/
  driver.py
  investigator.py
  reasoner.py
tools/
  frame_cache.py
  vlm_tools.py
workspace/
  investigator_ws.py
contracts/
  query.py
  report.py
  evidence.py
video/
  index.py
  build.py
  overview.py
  pipeline.py
evals/videomme/
  runner.py
  indexing.py
  outputs.py
  metrics.py
```

Everything else under `legacy/` is compatibility-only and must not be imported
by active multi_v3 modules.

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.eval_videomme \
  --strategy multi_v3 \
  --cases 605-1 \
  --run-root /tmp/vcah-multi-v3 \
  --allow-any-python
```

Reasoner planning emits scoped evidence queries:

```json
{
  "action": "plan",
  "queries": [
    {
      "query_id": "q1",
      "goal_id": "g1",
      "natural_query": "Find whether the red car appears.",
      "scope": {"scene_ids": ["sc01"]},
      "expected_evidence": "A verified red car sighting.",
      "budget": {"max_shots_to_verify": 2, "max_frames": 32}
    }
  ],
  "rationale": "Need visual evidence before answering."
}
```

Final answers cite verified multi_v3 evidence ids:

```json
{
  "action": "answer",
  "answer": "A",
  "confidence": "high",
  "citations": ["ev_0001"]
}
```
