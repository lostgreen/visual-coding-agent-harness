# Legacy P0 And Traditional Tool Notes

This document preserves the pre-`multi_v3` harness notes for compatibility and
historical debugging. These APIs live under `visual_coding_agent_harness.legacy`
or compatibility wrappers; they are not part of the active long-video path.

## Original P0 Shape

P0 borrowed the useful shape of VisProg: a text planner emitted a visual program,
and an interpreter executed registered visual modules. The added harness layer
recorded tool calls, structured observations, trace events, and an evidence
ledger so later versions could support long-video reasoning, subagents,
verification, and tool-use training.

## Legacy Demo

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from visual_coding_agent_harness.legacy.demo import run_demo

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

Legacy agents call tools through `ToolRequest` and consume `ToolResult`.

```python
from visual_coding_agent_harness.legacy.core.protocol import ToolRequest
from visual_coding_agent_harness.legacy.tools.traditional import build_traditional_registry

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

Traditional tools include:

- `crop_region`, `zoom_region`, `threshold_image`, `enhance_image`, `edge_detect`
- `sample_frames`, `extract_clip`

Image tools use Pillow. Video tools build and execute `ffmpeg` commands; they
raise a clear tool error when `ffmpeg` is not installed.

## VLM Agent Smoke

The first model-backed path kept one foundation model instance shared between
the main visual agent and its VLM tools. This was the baseline for comparing
direct VLM answering against tool-use behavior without changing model size.

The old planner received a tool-use prompt with a fixed tool catalog:

- `caption_video(video_path, question, nframes=8, max_pixels=151200)`
- `qa_video(video_path, question, nframes=8, max_pixels=151200)`
- `caption_image(image_path, question)`
- `qa_image(image_path, question)`
- `caption_region(image_path, bbox, question)`
- `qa_region(image_path, bbox, question)`

Caption/QA tools used structured prompts that forced visible-evidence-only
answers, uncertainty when evidence was weak, and no invented identities/text.
Region tools first cropped the requested normalized `[0,1000]` bbox into
`runs/<run_id>/artifacts/crops/`, then sent that crop to the shared VLM backend.
