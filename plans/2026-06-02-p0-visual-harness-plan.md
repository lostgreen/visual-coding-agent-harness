# P0 Visual Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, testable Visual Coding-Agent Harness that borrows VisProg's program/module idea while adding coding-agent-style workspace, trace, evidence ledger, and tool dispatch.

**Architecture:** The P0 runtime is a small Python package. A text-only planner can emit a visual program as ordered tool calls; `ProgramInterpreter` executes those calls through `ToolRegistry`; `EvidenceWorkspace` persists observations, trace events, artifacts, and answer-facing ledger entries. Built-in dummy visual tools make the harness runnable without GPUs or external model dependencies.

**Tech Stack:** Python 3.9+, standard library only for P0, `unittest` for tests, JSONL for traces/observations, Markdown for the evidence ledger.

---

## Current State

The directory `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness` has been initialized as a standalone git repository. The current uncommitted skeleton includes:

- `.gitignore`, which ignores `docs/`, `artifacts/`, `experiments/`, `.pytest_cache/`, `__pycache__/`, `runs/`, and build outputs.
- `pyproject.toml`, defining a minimal Python package.
- `src/visual_coding_agent_harness/__init__.py`.
- `src/visual_coding_agent_harness/registry.py`.
- `src/visual_coding_agent_harness/workspace.py`.
- `src/visual_coding_agent_harness/interpreter.py`.
- `tests/test_harness.py`.

The latest completed check was:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected current result after the existing skeleton:

```text
Ran 3 tests
OK
```

There is also one intentionally added but not-yet-implemented test expectation for `visual_coding_agent_harness.tools.dummy.build_dummy_registry`. Task 2 below resolves it.

---

## Reference Project Choice

Use **VisProg** as the first reference to modify from conceptually, not by directly copying its full repository. The useful transferable unit is:

- a program is a sequence of module/tool calls;
- a registry maps module names to handlers;
- an interpreter executes the program and maintains state.

The difference in our implementation is:

- every tool call writes a trace event;
- every visual result becomes a structured observation;
- every observation can be rendered into an answer-facing evidence ledger;
- the runtime is designed for future replacement of dummy tools with VideoSEAL-style long-video tools, Agentic-MME-style evaluation tools, and ParaVT-style parallel workers.

---

## File Structure

Create or modify these files only for P0:

```text
/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/
  .gitignore
  pyproject.toml
  plans/
    2026-06-02-p0-visual-harness-plan.md
  src/
    visual_coding_agent_harness/
      __init__.py
      interpreter.py
      registry.py
      workspace.py
      demo.py
      tools/
        __init__.py
        dummy.py
  tests/
    test_harness.py
    test_demo.py
```

Responsibilities:

- `registry.py`: tool metadata, registration, argument validation, and execution.
- `workspace.py`: run directory creation, observation IDs, `observations.jsonl`, `trace.jsonl`, and `ledger.md`.
- `interpreter.py`: sequential visual-program execution, assignment tracking, and trace/ledger integration.
- `tools/dummy.py`: deterministic seed tools for smoke tests and demos.
- `demo.py`: CLI-like runnable demo that creates a run and executes a tiny visual program.
- `tests/test_harness.py`: unit tests for registry, workspace, interpreter, and dummy tools.
- `tests/test_demo.py`: integration test for the demo runner.

---

### Task 1: Stabilize Repository Skeleton

**Files:**
- Modify: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/.gitignore`
- Modify: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/pyproject.toml`
- Modify: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/src/visual_coding_agent_harness/__init__.py`
- Test: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/tests/test_harness.py`

- [ ] **Step 1: Verify git repository exists**

Run:

```bash
git rev-parse --show-toplevel
```

Expected output:

```text
/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness
```

- [ ] **Step 2: Verify ignored design folders**

Run:

```bash
git status --ignored --short
```

Expected facts:

```text
?? .gitignore
?? pyproject.toml
?? plans/
?? src/
?? tests/
!! artifacts/
!! docs/
!! experiments/
```

- [ ] **Step 3: Ensure package metadata is minimal**

`pyproject.toml` should contain:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "visual-coding-agent-harness"
version = "0.1.0"
description = "A lightweight visual coding-agent harness for multimodal tool-use research."
requires-python = ">=3.9"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Run current tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected current result before Task 2:

```text
FAILED
ModuleNotFoundError: No module named 'visual_coding_agent_harness.tools'
```

If this test does not fail with the missing `tools` package, inspect `tests/test_harness.py` and confirm the dummy tool test is present.

---

### Task 2: Add Dummy Visual Tools

**Files:**
- Create: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/src/visual_coding_agent_harness/tools/__init__.py`
- Create: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/src/visual_coding_agent_harness/tools/dummy.py`
- Test: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/tests/test_harness.py`

- [ ] **Step 1: Confirm failing dummy-tool test**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected:

```text
ModuleNotFoundError: No module named 'visual_coding_agent_harness.tools'
```

- [ ] **Step 2: Create tools package marker**

Create `src/visual_coding_agent_harness/tools/__init__.py`:

```python
"""Built-in visual tools for local demos and smoke tests."""
```

- [ ] **Step 3: Implement deterministic dummy tools**

Create `src/visual_coding_agent_harness/tools/dummy.py`:

```python
"""Deterministic seed tools for the P0 harness.

These are placeholders for real VLM/OCR/verifier backends. They preserve the
same return schema expected from production visual tools.
"""

from __future__ import annotations

from typing import Mapping

from visual_coding_agent_harness.registry import ToolRegistry, tool


@tool(name="caption_image", description="Return a deterministic caption for an image artifact.")
def caption_image(image_path: str) -> Mapping[str, object]:
    return {
        "claim": f"{image_path} contains a red cup on a table.",
        "confidence": 0.75,
        "input_artifacts": [image_path],
        "limitations": "Dummy tool; replace with a VLM captioner for real runs.",
    }


@tool(name="ocr_region", description="Return deterministic OCR text for a cropped image artifact.")
def ocr_region(image_path: str) -> Mapping[str, object]:
    return {
        "claim": "The visible text reads EXIT.",
        "confidence": 0.9,
        "input_artifacts": [image_path],
        "limitations": "Dummy tool; replace with OCR backend for real runs.",
    }


@tool(name="verify_answer", description="Check whether an answer string is supported by ledger text.")
def verify_answer(answer: str, ledger_text: str) -> Mapping[str, object]:
    answer_terms = {token.strip(".,:;!?").lower() for token in answer.split() if len(token) > 2}
    ledger_lower = ledger_text.lower()
    overlap = sum(1 for token in answer_terms if token in ledger_lower)
    confidence = min(1.0, overlap / max(1, len(answer_terms)))
    return {
        "claim": f"Answer support score is {confidence:.2f}.",
        "confidence": confidence,
        "input_artifacts": [],
        "limitations": "Lexical dummy verifier; replace with evidence-aware verifier.",
    }


def build_dummy_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(caption_image)
    registry.register(ocr_region)
    registry.register(verify_answer)
    return registry
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add .gitignore pyproject.toml plans src tests
git commit -m "feat: add p0 visual harness skeleton"
```

Expected:

```text
[main ...] feat: add p0 visual harness skeleton
```

---

### Task 3: Add Program Validation

**Files:**
- Modify: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/src/visual_coding_agent_harness/interpreter.py`
- Modify: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/tests/test_harness.py`

- [ ] **Step 1: Add failing test for malformed program steps**

Append this test method to `HarnessTest` in `tests/test_harness.py`:

```python
    def test_interpreter_rejects_step_without_tool_name(self):
        registry = ToolRegistry()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = EvidenceWorkspace.create(Path(tmp), run_id="case_003")
            interpreter = ProgramInterpreter(registry=registry, workspace=workspace)

            with self.assertRaises(ValueError) as context:
                interpreter.run([{"args": {}}])

            self.assertIn("missing required 'tool'", str(context.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected:

```text
FAILED
KeyError: 'tool'
```

- [ ] **Step 3: Implement explicit validation**

Modify the start of the `for` loop in `ProgramInterpreter.run`:

```python
        for index, step in enumerate(program, start=1):
            if "tool" not in step:
                raise ValueError(f"Program step {index} is missing required 'tool'")
            tool_name = str(step["tool"])
            arguments = dict(step.get("args", {}))
```

- [ ] **Step 4: Run tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_harness.py
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/visual_coding_agent_harness/interpreter.py tests/test_harness.py
git commit -m "test: validate malformed visual programs"
```

Expected:

```text
[main ...] test: validate malformed visual programs
```

---

### Task 4: Add Demo Runner

**Files:**
- Create: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/src/visual_coding_agent_harness/demo.py`
- Create: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/tests/test_demo.py`

- [ ] **Step 1: Write failing demo integration test**

Create `tests/test_demo.py`:

```python
import tempfile
import unittest
from pathlib import Path

from visual_coding_agent_harness.demo import run_demo


class DemoTest(unittest.TestCase):
    def test_run_demo_creates_trace_observations_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_demo(base_dir=Path(tmp), run_id="demo_case")
            run_root = Path(tmp) / "runs" / "demo_case"

            self.assertEqual(result.observation_ids, ["obs_0001", "obs_0002"])
            self.assertTrue((run_root / "observations.jsonl").exists())
            self.assertTrue((run_root / "trace.jsonl").exists())
            self.assertIn("red cup", (run_root / "ledger.md").read_text())
            self.assertIn("EXIT", (run_root / "ledger.md").read_text())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_demo.py
```

Expected:

```text
ModuleNotFoundError: No module named 'visual_coding_agent_harness.demo'
```

- [ ] **Step 3: Implement demo runner**

Create `src/visual_coding_agent_harness/demo.py`:

```python
"""Runnable P0 demo for the visual coding-agent harness."""

from __future__ import annotations

from pathlib import Path

from visual_coding_agent_harness.interpreter import ProgramResult, ProgramInterpreter
from visual_coding_agent_harness.tools.dummy import build_dummy_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


def run_demo(base_dir: Path, run_id: str = "demo") -> ProgramResult:
    workspace = EvidenceWorkspace.create(base_dir=base_dir, run_id=run_id)
    interpreter = ProgramInterpreter(
        registry=build_dummy_registry(),
        workspace=workspace,
    )
    return interpreter.run(
        [
            {
                "tool": "caption_image",
                "args": {"image_path": "input/frame_001.jpg"},
                "assign": "global_caption",
            },
            {
                "tool": "ocr_region",
                "args": {"image_path": "artifacts/crops/sign_crop.jpg"},
                "assign": "sign_text",
            },
        ]
    )
```

- [ ] **Step 4: Run demo tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_demo.py
```

Expected:

```text
Ran 1 test
OK
```

- [ ] **Step 5: Run full tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover tests
```

Expected:

```text
Ran 6 tests
OK
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/visual_coding_agent_harness/demo.py tests/test_demo.py
git commit -m "feat: add runnable p0 harness demo"
```

Expected:

```text
[main ...] feat: add runnable p0 harness demo
```

---

### Task 5: Add Minimal README Outside Ignored Docs

**Files:**
- Create: `/Users/lostgreen/Desktop/AIM3Lab/AgenticMLLM/visual-coding-agent-harness/README.md`
- Test: manual command check

- [ ] **Step 1: Create README**

Create `README.md`:

```markdown
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
```

- [ ] **Step 2: Run README demo command**

Run:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from visual_coding_agent_harness.demo import run_demo

result = run_demo(Path("."), run_id="demo")
print(result)
PY
```

Expected output contains:

```text
ProgramResult
obs_0001
obs_0002
```

Expected files:

```text
runs/demo/observations.jsonl
runs/demo/trace.jsonl
runs/demo/ledger.md
```

- [ ] **Step 3: Confirm generated run is ignored**

Run:

```bash
git status --ignored --short
```

Expected fact:

```text
!! runs/
```

- [ ] **Step 4: Commit**

Run:

```bash
git add README.md
git commit -m "docs: add p0 harness usage notes"
```

Expected:

```text
[main ...] docs: add p0 harness usage notes
```

---

## P1 Follow-Up Plan

After P0 is complete, write a separate plan for P1. Do not mix these into P0.

P1 should add:

- real visual backend adapter interfaces;
- `sample_frames`, `crop_region`, `ocr_region`, `caption_image`, and `verify_answer` as non-dummy tools;
- `timeline_index.json` for long-video indexing;
- Agentic-MME-style evaluation cases;
- VideoSEAL-style planner/inspector separation;
- trace export for SFT/preference/RL data.

---

## Self-Review

Spec coverage:

- Multi-tool visual-oriented harness: covered by registry, dummy tools, interpreter, and workspace.
- Collaborative workflow: deferred to P1 except for trace/ledger foundations; P0 intentionally avoids subagents.
- Training-free observation: covered by JSONL traces and observations; training export is deferred to P1.
- Reference-code reuse: VisProg is used as the conceptual reference for program/module/interpreter design.

Placeholder scan:

- No `TBD` or `TODO` placeholders.
- P1 items are explicitly scoped as follow-up and not required for P0 completion.

Type consistency:

- `ProgramInterpreter.run(...)` returns `ProgramResult`.
- `ProgramResult.observation_ids` and `ProgramResult.assignments` are used consistently in tests and demo.
- `build_dummy_registry()` returns `ToolRegistry`.
