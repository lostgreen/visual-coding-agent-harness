from pathlib import Path

import pytest

from visual_coding_agent_harness.agent_contracts import CONTRACT_VERSION
from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.registry import ToolRegistry
from visual_coding_agent_harness.tools.global_view import build_global_view_registry
from visual_coding_agent_harness.tools.inspector import build_segment_inspector_registry
from visual_coding_agent_harness.tools.segments import build_segment_vlm_registry
from visual_coding_agent_harness.workspace import EvidenceWorkspace


class RecordingBackend(VisionLanguageBackend):
    def __init__(self):
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=f"{request.task} observation", raw={"task": request.task})


def test_global_gist_creates_and_links_contract_manifest(tmp_path: Path):
    backend = RecordingBackend()
    workspace = EvidenceWorkspace.create(tmp_path, "global_manifest")
    registry = build_global_view_registry(backend)

    result = ProgramInterpreter(registry=registry, workspace=workspace).run(
        [
            {
                "tool": "global_gist",
                "args": {
                    "video_path": "/videos/long.mp4",
                    "question": "What is the video mainly about?",
                    "duration_sec": 300.0,
                },
            }
        ]
    )

    observation = workspace.get_observation(result.observation_ids[0])
    assert observation is not None
    assert observation.frame_set_id is not None
    manifest = workspace.get_manifest(observation.frame_set_id)
    assert manifest is not None
    assert manifest.created_by_tool == "global_gist"
    assert manifest.observation_id == observation.observation_id
    assert manifest.video_path == "/videos/long.mp4"
    assert manifest.segment_id is None
    assert manifest.start_sec == 0.0
    assert manifest.end_sec == 300.0
    assert manifest.nframes == 128
    assert manifest.target_nframes == 128
    assert manifest.budget_reason == "default_contract"
    assert manifest.sampling_policy == "uniform"
    assert manifest.frame_times_approximate is True
    assert len(manifest.frame_times_sec) == 128
    assert manifest.contract_version == CONTRACT_VERSION
    assert backend.requests[0].metadata["nframes"] == 128


@pytest.mark.parametrize("tool_name,question_arg", [("vision_read", "ask_for"), ("inspect_segment", "question")])
def test_local_inspection_tools_create_and_link_contract_manifest(
    tmp_path: Path,
    tool_name: str,
    question_arg: str,
):
    backend = RecordingBackend()
    workspace = EvidenceWorkspace.create(tmp_path, f"{tool_name}_manifest")
    registry = build_segment_inspector_registry(backend)

    result = ProgramInterpreter(registry=registry, workspace=workspace).run(
        [
            {
                "tool": tool_name,
                "args": {
                    "video_path": "/videos/long.mp4",
                    "segment_id": "seg_0004",
                    "start_sec": 30.0,
                    "end_sec": 42.0,
                    question_arg: "What happens?",
                },
            }
        ]
    )

    observation = workspace.get_observation(result.observation_ids[0])
    assert observation is not None
    assert observation.frame_set_id is not None
    manifest = workspace.get_manifest(observation.frame_set_id)
    assert manifest is not None
    assert manifest.created_by_tool == tool_name
    assert manifest.segment_id == "seg_0004"
    assert manifest.start_sec == 30.0
    assert manifest.end_sec == 42.0
    assert manifest.nframes == 128
    assert manifest.target_nframes == 128
    assert manifest.budget_reason == "default_contract"
    assert len(manifest.frame_times_sec) == 128


@pytest.mark.parametrize("tool_name", ["caption_segment", "qa_segment"])
def test_segment_caption_and_qa_tools_create_and_link_contract_manifest(tmp_path: Path, tool_name: str):
    backend = RecordingBackend()
    workspace = EvidenceWorkspace.create(tmp_path, f"{tool_name}_manifest")
    registry = build_segment_vlm_registry(backend)

    result = ProgramInterpreter(registry=registry, workspace=workspace).run(
        [
            {
                "tool": tool_name,
                "args": {
                    "video_path": "/videos/long.mp4",
                    "segment_id": "seg_0005",
                    "start_sec": 50.0,
                    "end_sec": 70.0,
                    "question": "Describe the segment.",
                },
            }
        ]
    )

    observation = workspace.get_observation(result.observation_ids[0])
    assert observation is not None
    assert observation.frame_set_id is not None
    manifest = workspace.get_manifest(observation.frame_set_id)
    assert manifest is not None
    assert manifest.created_by_tool == tool_name
    assert manifest.segment_id == "seg_0005"
    assert manifest.start_sec == 50.0
    assert manifest.end_sec == 70.0
    assert manifest.nframes == 128
    assert manifest.target_nframes == 128
    assert manifest.budget_reason == "default_contract"


def test_manifest_records_user_override_after_contract_clamp(tmp_path: Path):
    backend = RecordingBackend()
    workspace = EvidenceWorkspace.create(tmp_path, "override_manifest")
    registry = ToolRegistry()
    registry.extend(build_segment_inspector_registry(backend))

    result = ProgramInterpreter(registry=registry, workspace=workspace).run(
        [
            {
                "tool": "vision_read",
                "args": {
                    "video_path": "/videos/long.mp4",
                    "segment_id": "seg_0004",
                    "start_sec": 30.0,
                    "end_sec": 42.0,
                    "ask_for": "What happens?",
                    "nframes": 20,
                },
            }
        ]
    )

    observation = workspace.get_observation(result.observation_ids[0])
    assert observation is not None
    assert observation.frame_set_id is not None
    manifest = workspace.get_manifest(observation.frame_set_id)
    assert manifest is not None
    assert manifest.nframes == 64
    assert manifest.target_nframes == 64
    assert manifest.budget_reason == "user_override"
    assert backend.requests[0].metadata["nframes"] == 64
