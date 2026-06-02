# Visual Coding-Agent Harness

This repository is a research prototype for a multimodal coding-agent-style harness.

P0 borrows the useful shape of VisProg: a text planner emits a visual program, and an interpreter executes registered visual modules. The added harness layer records tool calls, structured observations, trace events, and an evidence ledger so later versions can support long-video reasoning, subagents, verification, and tool-use training.

## Run Tests

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

## Run Demo

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from visual_coding_agent_harness.demo import run_demo

result = run_demo(Path("."), run_id="demo")
print(result)
PY
```

The demo writes:

- `runs/demo/observations.jsonl`
- `runs/demo/trace.jsonl`
- `runs/demo/ledger.md`

## P0 Scope

- No external visual models.
- No GPU requirement.
- No full long-video pipeline.
- Dummy tools preserve the schema expected from real VLM, OCR, and verifier backends.
