from __future__ import annotations

import tempfile
from pathlib import Path

from visual_coding_agent_harness.agents.grounding.compiler import compile_fallback_plan, compile_grounding_plan
from visual_coding_agent_harness.evidence.order_extraction import (
    extract_observed_order_from_text,
    match_observed_order_to_hypotheses,
)
from visual_coding_agent_harness.interpreter import ProgramInterpreter
from visual_coding_agent_harness.tools.navigation import build_video_navigation_registry
from visual_coding_agent_harness.video_map import VideoMap, VideoMapSegment
from visual_coding_agent_harness.workspace import EvidenceWorkspace


OPTIONS = [
    'A. "The rape of Persephone", "Apollo and Daphne", "David", "Aeneas"',
    'B. "David", "Aeneas", "Apollo and Daphne", "The rape of Persephone"',
    'C. "Apollo and Daphne", "Aeneas", "David", "The rape of Persephone"',
    'D. "Aeneas", "David", "The rape of Persephone", "Apollo and Daphne"',
]


def _compiled_order_plan():
    plan = compile_fallback_plan("Which order are the artworks presented in?", OPTIONS, route_hint="temporal_order")
    return compile_grounding_plan(plan, raw_options={option[0]: option[3:] for option in OPTIONS})


def test_extract_observed_order_from_text_matches_ordered_entities() -> None:
    ordered_set = _compiled_order_plan().ordered_sets[0]

    observed = extract_observed_order_from_text(
        "The narration lists Aeneas, David, The rape of Persephone, and Apollo and Daphne.",
        ordered_set,
    )

    assert observed is not None
    assert observed.entity_order == ("E1", "E2", "E3", "E4")
    assert observed.source == "indexed_asr"


def test_match_observed_order_to_hypotheses_exact_option() -> None:
    ordered_set = _compiled_order_plan().ordered_sets[0]
    observed = extract_observed_order_from_text(
        "The narration lists Aeneas, David, The rape of Persephone, and Apollo and Daphne.",
        ordered_set,
    )

    match = match_observed_order_to_hypotheses(observed, ordered_set.hypotheses)  # type: ignore[arg-type]

    assert match.status == "full_match"
    assert match.option_id == "D"


def test_read_segment_detail_promotes_ordered_list_evidence_for_option_d() -> None:
    compiled = _compiled_order_plan()
    video_map = VideoMap(
        video_path="/videos/bernini.mp4",
        duration_sec=60.0,
        segments=[
            VideoMapSegment(
                segment_id="seg_0002",
                start_sec=20.0,
                end_sec=40.0,
                asr_text="The narration lists Aeneas, David, The rape of Persephone, and Apollo and Daphne.",
            )
        ],
    )
    with tempfile.TemporaryDirectory() as tmp:
        workspace = EvidenceWorkspace.create(Path(tmp), run_id="ordered_list_promotion")
        workspace.target_registry = compiled.registry
        workspace.ordered_sets = compiled.ordered_sets
        registry = build_video_navigation_registry(video_map, workspace=workspace)

        result = ProgramInterpreter(registry=registry, workspace=workspace).run(
            [{"tool": "read_segment_detail", "args": {"segment_id": "seg_0002", "promote_answer_evidence": True}}]
        )
        observation = workspace.get_observation(result.observation_ids[0])
        table = workspace.read_evidence_table_v3(
            question="Which order?",
            options=OPTIONS,
        )
        trace = (workspace.root / "trace.jsonl").read_text(encoding="utf-8")

    ordered_rows = [row for row in table["rows"] if row.get("evidence_type") == "ordered_list"]
    assert ordered_rows
    assert ordered_rows[0]["supported_option"] == "D"
    assert ordered_rows[0]["source_observation_id"] == observation.observation_id
    ordered_raw_rows = [
        row for row in observation.raw_output["answer_evidence_rows"] if row.get("evidence_type") == "ordered_list"
    ]
    assert ordered_raw_rows
    assert all("evidence_id" not in row for row in ordered_raw_rows)
    assert "post_observation_ordered_list_promoted" in trace
