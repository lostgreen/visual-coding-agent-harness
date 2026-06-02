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

## Tool Protocol

Agents should call tools through `ToolRequest` and consume `ToolResult`.

```python
from visual_coding_agent_harness.protocol import ToolRequest
from visual_coding_agent_harness.tools.traditional import build_traditional_registry

registry = build_traditional_registry()
results = registry.execute_batch([
    ToolRequest(
        tool="crop_region",
        arguments={
            "image_path": "input/frame.png",
            "bbox": [0, 0, 500, 500],
            "output_path": "runs/demo/artifacts/crops/frame_crop.png",
        },
        request_id="crop_1",
        caller="spatial_worker",
    )
])
```

Traditional tools currently include:

- `crop_region`, `zoom_region`, `threshold_image`, `enhance_image`, `edge_detect`
- `sample_frames`, `extract_clip`

Image tools use Pillow. Video tools build and execute `ffmpeg` commands; they raise a clear tool error when `ffmpeg` is not installed.
