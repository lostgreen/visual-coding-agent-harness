import tempfile
from pathlib import Path

from visual_coding_agent_harness.backends.base import BackendRequest, BackendResponse, VisionLanguageBackend
from visual_coding_agent_harness.legacy.contracts_v2 import ClaimModality, TargetRegistry, TargetSpec
from visual_coding_agent_harness.legacy.interpreter import ProgramInterpreter
from visual_coding_agent_harness.legacy.tools.asr_binding import build_asr_binding_registry
from visual_coding_agent_harness.video._map import VideoMap, VideoMapSegment, VideoMapStore
from visual_coding_agent_harness.legacy.workspace_v2 import EvidenceWorkspace


class AsrBindingBackend(VisionLanguageBackend):
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        return BackendResponse(text=self.text)


class SequenceAsrBindingBackend(VisionLanguageBackend):
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.requests: list[BackendRequest] = []

    def generate(self, request: BackendRequest) -> BackendResponse:
        self.requests.append(request)
        if not self.texts:
            return BackendResponse(text="{}")
        return BackendResponse(text=self.texts.pop(0))


def _video_map(*, cue_id: str | None = None) -> VideoMap:
    sentence = {
        "start_sec": 3.0,
        "end_sec": 7.0,
        "text": "The narrator says Goya came from a humble background.",
    }
    if cue_id is not None:
        sentence["cue_id"] = cue_id
    return VideoMap(
        video_path="/videos/asr.mp4",
        duration_sec=30.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0001",
                start_sec=0.0,
                end_sec=30.0,
                asr_sentences=[sentence],
            )
        ],
    )


def _workspace(root: Path) -> EvidenceWorkspace:
    workspace = EvidenceWorkspace.create(root, "asr_binding")
    workspace.target_registry = TargetRegistry.from_specs(
        targets=[
            TargetSpec(
                "T1",
                "Goya came from a humble background",
                subject="Goya",
                modality_hint=ClaimModality.NARRATED_FACT,
            )
        ]
    )
    return workspace


def _registry(*, video_map: VideoMap, backend: VisionLanguageBackend, workspace: EvidenceWorkspace):
    return build_asr_binding_registry(video_map_store=VideoMapStore(video_map), backend=backend, workspace=workspace)


def test_bind_asr_claim_promotes_supported_legal_cue_to_answer_evidence_row() -> None:
    backend = AsrBindingBackend('{"T1": {"verdict": "supports", "cue_ids": ["cue_0001"], "quote": "Goya came from a humble background"}}')
    with tempfile.TemporaryDirectory() as tmp:
        workspace = _workspace(Path(tmp))
        registry = _registry(video_map=_video_map(), backend=backend, workspace=workspace)

        result = ProgramInterpreter(registry=registry, workspace=workspace).run(
            [{"tool": "bind_asr_claim", "args": {"segment_id": "seg_0001", "target_refs": ["T1"]}}]
        )
        observation = workspace.get_observation(result.observation_ids[0])
        rows = workspace.read_evidence_table_v3(question="What is said?", options=[]).get("rows", [])

    assert backend.requests[0].task == "asr_claim_binding"
    assert backend.requests[0].max_new_tokens == 800
    assert backend.requests[0].temperature == 0.0
    assert "cue_0001" in backend.requests[0].prompt
    assert observation is not None
    raw_row = observation.raw_output["answer_evidence_rows"][0]
    assert raw_row["target_id"] == "T1"
    assert raw_row["snippet"] == "Goya came from a humble background"
    assert raw_row["evidence_binding"]["status"] == "supported"
    assert observation.raw_output["evidence_bindings"][0]["target_id"] == "T1"
    assert rows[0]["segment_id"] == "seg_0001"
    assert rows[0]["grounding_quality"] == "indexed_transcript"
    assert rows[0]["evidence_binding"]["target_id"] == "T1"
    assert rows[0]["evidence_binding"]["status"] == "supported"


def test_bind_asr_claim_rejects_illegal_cue_without_supported_row() -> None:
    backend = AsrBindingBackend('{"T1": {"verdict": "supported", "cue_ids": ["cue_9999"], "quote": "not indexed"}}')
    with tempfile.TemporaryDirectory() as tmp:
        workspace = _workspace(Path(tmp))
        registry = _registry(video_map=_video_map(), backend=backend, workspace=workspace)

        output = registry.execute("bind_asr_claim", {"segment_id": "seg_0001", "target_refs": ["T1"]})

    assert output["answer_evidence_rows"] == []
    assert "illegal cue_ids" in output["limitations"]


def test_bind_asr_claim_normalizes_cue_prefixed_numeric_ids() -> None:
    backend = AsrBindingBackend(
        '{"T1": {"verdict": "supported", "cue_ids": ["cue_142"], "quote": "Goya came from a humble background"}}'
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = _workspace(Path(tmp))
        registry = _registry(video_map=_video_map(cue_id="142"), backend=backend, workspace=workspace)

        output = registry.execute("bind_asr_claim", {"segment_id": "seg_0001", "target_refs": ["T1"]})

    assert output["answer_evidence_rows"][0]["raw_asr_ref"]["cue_ids"] == ["142"]
    assert output["evidence_bindings"][0]["cue_ids"] == ["142"]
    assert "normalized_cue_ids" in output["limitations"]


def test_bind_asr_claim_prompt_example_uses_real_cue_ids() -> None:
    backend = AsrBindingBackend(
        '{"T1": {"verdict": "supported", "cue_ids": ["142"], "quote": "Goya came from a humble background"}}'
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = _workspace(Path(tmp))
        registry = _registry(video_map=_video_map(cue_id="142"), backend=backend, workspace=workspace)

        registry.execute("bind_asr_claim", {"segment_id": "seg_0001", "target_refs": ["T1"]})

    assert '"cue_ids": ["142"]' in backend.requests[0].prompt
    assert '"cue_ids": ["cue_0001"]' not in backend.requests[0].prompt


def test_bind_asr_claim_retries_once_after_still_illegal_cue_ids() -> None:
    backend = SequenceAsrBindingBackend(
        [
            '{"T1": {"verdict": "supported", "cue_ids": ["cue_9999"], "quote": "bad"}}',
            '{"T1": {"verdict": "supported", "cue_ids": ["142"], "quote": "Goya came from a humble background"}}',
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = _workspace(Path(tmp))
        registry = _registry(video_map=_video_map(cue_id="142"), backend=backend, workspace=workspace)

        output = registry.execute("bind_asr_claim", {"segment_id": "seg_0001", "target_refs": ["T1"]})

    assert len(backend.requests) == 2
    assert "Previous response used illegal cue_ids" in backend.requests[1].prompt
    assert output["answer_evidence_rows"][0]["raw_asr_ref"]["cue_ids"] == ["142"]
