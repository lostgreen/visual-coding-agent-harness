from __future__ import annotations

from visual_coding_agent_harness.agents.driver import counter_check_mcq
from visual_coding_agent_harness.contracts.evidence import EvidenceRecord
from visual_coding_agent_harness.workspace.text_index import InvertedIndex
from visual_coding_agent_harness.workspace.video_workspace import Beat, Chapter, VideoWorkspace
from visual_coding_agent_harness.workspace.visual_index import VisualIndex


class EmptyEmbeddingBackend:
    embedding_dim = 1

    def encode_images(self, paths):
        return []

    def encode_text(self, queries):
        return []


def test_counter_check_mcq_scans_non_answer_option_keywords() -> None:
    beat = Beat("bt00001", "ch01", 0.0, 3.0, "", "The blue jacket appears later.", (), ("sh001",))
    text_index = InvertedIndex()
    text_index.add(beat.beat_id, beat.asr_verbatim, modality="asr")
    workspace = VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=3.0,
        chapters=(Chapter("ch01", 0.0, 3.0, (beat.beat_id,), ""),),
        beats=(beat,),
        text_index=text_index,
        visual_index=VisualIndex(EmptyEmbeddingBackend()),
    )

    hits = counter_check_mcq(
        workspace=workspace,
        question="Which jacket is shown?",
        options={"A": "red jacket", "C": "blue jacket"},
        proposed_answer="A",
    )

    assert hits[0].option_id == "C"
    assert hits[0].beat_id == "bt00001"
    assert "blue jacket" in hits[0].verbatim.lower()


def test_counter_check_does_not_block_when_existing_evidence_refutes_hit() -> None:
    beat = Beat("bt00001", "ch01", 0.0, 3.0, "", "The blue jacket appears later.", (), ("sh001",))
    text_index = InvertedIndex()
    text_index.add(beat.beat_id, beat.asr_verbatim, modality="asr")
    workspace = VideoWorkspace(
        video_path="/videos/demo.mp4",
        duration_sec=3.0,
        chapters=(Chapter("ch01", 0.0, 3.0, (beat.beat_id,), ""),),
        beats=(beat,),
        text_index=text_index,
        visual_index=VisualIndex(EmptyEmbeddingBackend()),
    )
    refutation = EvidenceRecord(
        evidence_id="er_refute_blue",
        claim="Option C blue jacket is not the answer.",
        stance="refutes",
        modality="asr",
        time_sec=1.0,
        pointer="bt00001",
        verbatim="The blue jacket mention is unrelated to the shown answer.",
        query_id="q_refute",
        beat_id="bt00001",
    )

    result = counter_check_mcq(
        workspace=workspace,
        question="Which jacket is shown?",
        options={"A": "red jacket", "C": "blue jacket"},
        proposed_answer="A",
        existing_evidence=(refutation,),
    )

    assert result.status == "refuted_by_existing_evidence"
    assert result.hits == ()
