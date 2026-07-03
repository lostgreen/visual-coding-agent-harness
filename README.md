# Visual Coding-Agent Harness

This repository is a research prototype for a multimodal coding-agent-style harness.

P0 borrows the useful shape of VisProg: a text planner emits a visual program, and an interpreter executes registered visual modules. The added harness layer records tool calls, structured observations, trace events, and an evidence ledger so later versions can support long-video reasoning, subagents, verification, and tool-use training.

## Run Tests

```bash
PYTHONPATH=src:. pytest
```

The legacy unittest discovery command remains supported for compatibility:

```bash
PYTHONPATH=src:. python3 -m unittest discover tests
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

## Project Layout

Package code lives under `src/visual_coding_agent_harness/`. Evaluation
runners, metrics, schemas, and ablation helpers live under
`src/visual_coding_agent_harness/evals/`; command modules live under
`src/visual_coding_agent_harness/cli/`.

`runs/` is reserved for generated run artifacts. The legacy Python files in
`runs/` and `scripts/` are compatibility wrappers for older commands.
Prefer the package CLIs for new usage:

```bash
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.eval_videomme --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.run_ablation --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.generate_ablation_report --help
PYTHONPATH=src python3 -m visual_coding_agent_harness.cli.audit_trajectory --help
```

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

## VLM Agent Smoke

The first model-backed path keeps one foundation model instance shared between
the main visual agent and its VLM tools. This is the baseline for comparing
direct VLM answering against tool-use behavior without changing model size.

The main agent receives a tool-use prompt with a fixed tool catalog:

- `caption_video(video_path, question, nframes=8, max_pixels=151200)`
- `qa_video(video_path, question, nframes=8, max_pixels=151200)`
- `caption_image(image_path, question)`
- `qa_image(image_path, question)`
- `caption_region(image_path, bbox, question)`
- `qa_region(image_path, bbox, question)`

Caption/QA tools use structured prompts that force visible-evidence-only
answers, uncertainty when evidence is weak, and no invented identities/text.
Region tools first crop the requested normalized `[0,1000]` bbox into
`runs/<run_id>/artifacts/crops/`, then send that crop to the shared VLM backend.

Input schema:

```json
{
  "question": "What is the video mainly about?",
  "media_path": "/path/to/video.mp4",
  "media_type": "video",
  "tool_policy": "required"
}
```

The model planner is asked to emit only:

```json
{
  "answer": "...",
  "program": [
    {
      "tool": "caption_video",
      "args": {
        "video_path": "/path/to/video.mp4",
        "question": "What is the video mainly about?"
      },
      "assign": "caption"
    }
  ]
}
```

The harness normalizes media paths to the caller-provided input path, executes
the program, and returns:

```json
{
  "input": {
    "question": "What is the video mainly about?",
    "media_path": "/path/to/video.mp4",
    "media_type": "video",
    "tool_policy": "required"
  },
  "output": {
    "answer": "...",
    "program": [],
    "observation_ids": ["obs_0001"],
    "assignments": {"caption": "obs_0001"}
  },
  "debug": {
    "planner_text": "raw planner response"
  }
}
```

```bash
python -m visual_coding_agent_harness.cli.vlm_smoke \
  --model-path /m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct \
  --media-path /path/to/video.mp4 \
  --question "What happens in the video?" \
  --base-dir . \
  --run-id qwen3_vl_4b_smoke
```

The run writes the same harness artifacts under `runs/<run_id>/`:

- `observations.jsonl`
- `trace.jsonl`
- `ledger.md`

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
