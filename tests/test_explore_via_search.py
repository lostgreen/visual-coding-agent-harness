from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse
from visual_coding_agent_harness.contracts.query import QueryBudget, QueryScope, ScopedQuery
from visual_coding_agent_harness.tools.vlm_tools import explore_via_search
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


class RecordingBackend:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=json.dumps({"picks": [{"shot_id": "bt00002", "score": 0.91, "reason": "name match"}]}))


class EmptyEmbeddingBackend:
    embedding_dim = 1

    def encode_images(self, paths):
        raise AssertionError("visual index should not need image encoding in this test")

    def encode_text(self, queries):
        raise AssertionError("visual index is empty, so search should not encode text")


def _image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)
    return str(path)


def test_explore_via_search_uses_text_hits_to_narrow_vlm_batches(tmp_path: Path) -> None:
    beats = (
        Beat("bt00001", "ch01", 0.0, 2.0, _image(tmp_path / "one.jpg"), "nothing relevant", (), ("sc01_sh001",)),
        Beat("bt00002", "ch01", 2.0, 4.0, _image(tmp_path / "two.jpg"), "Whitehead speaks here", (), ("sc01_sh002",)),
    )
    text_index = InvertedIndex()
    for beat in beats:
        text_index.add(beat.beat_id, beat.asr_verbatim, modality="asr")
    workspace = VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=4.0,
        chapters=(Chapter("ch01", 0.0, 4.0, ("bt00001", "bt00002"), beats[0].keyframe_path),),
        beats=beats,
        text_index=text_index,
        visual_index=VisualIndex(EmptyEmbeddingBackend()),
    )
    query = ScopedQuery(
        query_id="q1",
        goal_id="g1",
        natural_query="Where is Whitehead?",
        scope=QueryScope(scene_ids=("ch01",)),
        expected_evidence="Whitehead",
        budget=QueryBudget(max_shots_to_verify=2, max_frames=4),
    )
    backend = RecordingBackend()

    result = explore_via_search(workspace=workspace, query=query, backend=backend, top_k=5)

    assert [candidate.shot_id for candidate in result] == ["sc01_sh002"]
    assert result.batch_count == 1
    assert "bt00002" in backend.requests[0].prompt
    assert "bt00001" not in backend.requests[0].prompt
