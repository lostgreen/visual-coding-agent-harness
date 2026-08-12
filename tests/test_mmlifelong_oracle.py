from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from benchmarks.mmlifelong.oracle import (
    CaptionPacketIntervention,
    OracleIntervention,
    bootstrap_tasks,
)
from vcah.caption_schema import (
    CaptionHitV1,
    CaptionPassageV1,
    passage_to_dict,
)
from vcah.investigator import InvestigationReport, ObservationAttempt
from vcah.multiround import (
    InvestigationTask,
    ReasonerDecision,
    VirtualVideoMultiRoundDriver,
)
from vcah.virtual_video import (
    VirtualVideoCase,
    VirtualVideoManifest,
    VirtualVideoSegment,
    VirtualVideoWorkspace,
)
from vcah.workspace import stable_attempt_id


def _load_prepare_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "prepare_mmlifelong_oracle_day.py"
    spec = importlib.util.spec_from_file_location("prepare_mmlifelong_oracle_day", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = _load_prepare_module()


def _workspace(tmp_path: Path) -> VirtualVideoWorkspace:
    segment = VirtualVideoSegment(
        "seg_0001",
        "video_1",
        "video.mp4",
        0.0,
        300.0,
        0.0,
        300.0,
    )
    return VirtualVideoWorkspace.create(
        tmp_path / "workspace",
        manifest=VirtualVideoManifest(
            workspace_id="mmlifelong-game-test-0000",
            segments=(segment,),
        ),
        case=VirtualVideoCase(
            case_id="mmlifelong-game-test-0000",
            question="What happens near the end?",
            subset="game",
            split="test",
            question_type="Event Tracking",
        ),
    )


def _passage(
    passage_id: str,
    start_sec: float,
    end_sec: float,
    text: str,
) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id=f"caption_{passage_id}",
        text=text,
        virtual_start_sec=start_sec,
        virtual_end_sec=end_sec,
        anchor_virtual_sec=start_sec,
        ordinal=0,
        metadata={"source_segments": ["seg_0001"]},
    )


def _hit(passage: CaptionPassageV1, rank: int) -> CaptionHitV1:
    return CaptionHitV1(
        passage_id=passage.passage_id,
        caption_id=passage.caption_id,
        rank=rank,
        lexical_score=1.0 / rank,
        dense_score=1.0 / rank,
        fused_score=1.0 / rank,
        virtual_start_sec=passage.virtual_start_sec,
        virtual_end_sec=passage.virtual_end_sec,
        wall_clock_begin=None,
        wall_clock_end=None,
        text=passage.text,
        interval_precision="passage",
        source_pointer=f"caption://digest/{passage.passage_id}",
        metadata={
            "source_segments": ["seg_0001"],
            "source_video_ids": ["video_1"],
        },
    )


def _packet(hits: list[CaptionHitV1]) -> dict[str, Any]:
    return {
        "queries": ["question"],
        "config_digest": "caption-digest",
        "index_digest": "index-digest",
        "query_fingerprint": "query-fingerprint",
        "hits": [hit.__dict__ for hit in hits],
        "occurrence_set": {},
        "rendered": "natural",
    }


def _intervention() -> OracleIntervention:
    return OracleIntervention(
        case_id="mmlifelong-game-test-0000",
        normalized_clue_intervals=((205.0, 206.0),),
        experiment_seed=20260811,
        caption_config_digest="caption-digest",
    )


def _write_passages(
    workspace: VirtualVideoWorkspace,
    passages: list[CaptionPassageV1],
) -> None:
    path = workspace.asset_root / "captions" / "passages.caption-digest.jsonl"
    path.parent.mkdir()
    path.write_text(
        "".join(
            json.dumps(passage_to_dict(passage), sort_keys=True) + "\n"
            for passage in passages
        ),
        encoding="utf-8",
    )


def test_o1_replaces_a_distractor_without_changing_pool_size(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p1", 100.0, 120.0, "middle"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    transform = CaptionPacketIntervention(
        arm="o1",
        intervention=_intervention(),
        workspace=workspace,
        audit_path=workspace.root_dir / "audit.json",
    )

    transformed = transform(_packet([_hit(passages[0], 1), _hit(passages[1], 2)]))

    assert len(transformed["hits"]) == 2
    assert transform.audit["natural_clue_recall"] == 0.0
    assert transform.audit["final_clue_recall"] == 1.0
    assert transform.audit["injected_candidate_count"] == 1
    visible = json.dumps(
        {
            "hits": transformed["hits"],
            "occurrence_set": transformed["occurrence_set"],
            "rendered": transformed["rendered"],
        }
    ).casefold()
    assert "oracle" not in visible
    assert "gold" not in visible


def test_o15_and_o175_preserve_o1_pool_and_hide_exact_boundaries(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p1", 100.0, 120.0, "middle"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    packet = _packet([_hit(passages[0], 1), _hit(passages[1], 2)])

    transformed = {}
    audits = {}
    for arm in ("o1", "o1.5", "o1.75"):
        intervention = CaptionPacketIntervention(
            arm=arm,
            intervention=_intervention(),
            workspace=workspace,
            audit_path=workspace.root_dir / f"audit-{arm}.json",
        )
        transformed[arm] = intervention(packet)
        audits[arm] = intervention.audit

    assert transformed["o1"]["hits"] == transformed["o1.5"]["hits"]
    assert transformed["o1"]["hits"] == transformed["o1.75"]["hits"]
    assert audits["o1"]["candidate_passage_ids"] == audits["o1.5"][
        "candidate_passage_ids"
    ]
    assert audits["o1.5"]["candidate_passage_ids"] == audits["o1.75"][
        "candidate_passage_ids"
    ]

    guidance_15 = transformed["o1.5"]["oracle_guidance"]
    guidance_175 = transformed["o1.75"]["oracle_guidance"]
    assert guidance_15["guidance_type"] == "selected_coarse_candidates"
    assert guidance_15["selected_candidate_guarantee"] == (
        "overlaps_annotated_occurrence"
    )
    assert guidance_15["boundary_visibility"] == "hidden"
    assert guidance_15["selected_candidates"][0]["inspection_range"] == [
        200.0,
        220.0,
    ]
    assert "anchor_timestamps_sec" not in guidance_15
    assert guidance_175["selected_candidates"] == guidance_15[
        "selected_candidates"
    ]
    assert guidance_175["anchor_timestamps_sec"] == [205.5]
    assert guidance_175["point_anchors"] == [
        {
            "anchor_timestamp_sec": 205.5,
            "selected_candidate_rank": guidance_15["selected_candidates"][0][
                "candidate_rank"
            ],
            "selected_candidate_passage_id": "p2",
        }
    ]
    assert 205.0 not in _numeric_leaves(guidance_175)
    assert 206.0 not in _numeric_leaves(guidance_175)
    assert audits["o1.5"]["selected_candidate_clue_recall"] == 1.0
    assert audits["o1.75"]["selected_candidate_clue_recall"] == 1.0
    assert audits["o1.5"]["anchor_count"] == 0
    assert audits["o1.75"]["anchor_count"] == 1


def test_o175_point_anchor_remains_exact_when_coarse_candidate_is_partial(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("partial", 200.0, 205.25, "start of decisive event"),
    ]
    _write_passages(workspace, passages)
    transform = CaptionPacketIntervention(
        arm="o1.75",
        intervention=_intervention(),
        workspace=workspace,
        audit_path=workspace.root_dir / "audit.json",
    )

    transformed = transform(_packet([_hit(passages[0], 1)]))
    guidance = transformed["oracle_guidance"]

    assert guidance["selected_candidates"][0]["inspection_range"] == [
        200.0,
        205.25,
    ]
    assert guidance["point_anchors"][0]["anchor_timestamp_sec"] == 205.5
    assert guidance["point_anchors"][0]["selected_candidate_passage_id"] == (
        "partial"
    )


def test_o175_forced_preserves_o175_visible_packet(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    packet = _packet([_hit(passages[0], 1)])
    transformed = {}
    audits = {}
    for arm in ("o1.75", "o1.75-forced"):
        intervention = CaptionPacketIntervention(
            arm=arm,
            intervention=_intervention(),
            workspace=workspace,
            audit_path=workspace.root_dir / f"audit-{arm}.json",
        )
        transformed[arm] = intervention(packet)
        audits[arm] = intervention.audit

    assert transformed["o1.75-forced"] == transformed["o1.75"]
    assert audits["o1.75-forced"]["arm"] == "o1.75-forced"
    assert audits["o1.75"]["anchor_execution_policy"] == "agent_controlled"
    assert (
        audits["o1.75-forced"]["anchor_execution_policy"]
        == "force_if_requested"
    )


def test_o2_uses_exact_clue_ranges_without_answer_fields(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    transform = CaptionPacketIntervention(
        arm="o2",
        intervention=_intervention(),
        workspace=workspace,
        audit_path=workspace.root_dir / "audit.json",
    )

    transformed = transform(_packet([_hit(passages[0], 1)]))

    assert len(transformed["hits"]) == 1
    assert transformed["hits"][0]["virtual_start_sec"] == 205.0
    assert transformed["hits"][0]["virtual_end_sec"] == 206.0
    assert transform.audit["exact_locator_count"] == 1
    assert "answer" not in json.dumps(transformed).casefold()


def test_o2_center_adds_center_guidance_to_exact_locators(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    transform = CaptionPacketIntervention(
        arm="o2-center",
        intervention=_intervention(),
        workspace=workspace,
        audit_path=workspace.root_dir / "audit.json",
    )

    transformed = transform(_packet([_hit(passages[0], 1)]))
    guidance = transformed["oracle_guidance"]

    assert transformed["hits"][0]["virtual_start_sec"] == 205.0
    assert transformed["hits"][0]["virtual_end_sec"] == 206.0
    assert guidance["guidance_type"] == "exact_locators_with_point_anchors"
    assert guidance["boundary_visibility"] == "exact"
    assert guidance["anchor_timestamps_sec"] == [205.5]
    assert guidance["selected_candidates"][0]["inspection_range"] == [205.0, 206.0]
    assert transform.audit["exact_boundaries_visible"] is True
    assert transform.audit["anchor_count"] == 1


def test_o2_guided_matches_center_guidance_without_anchor_fields(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    passages = [
        _passage("p0", 0.0, 20.0, "early"),
        _passage("p2", 200.0, 220.0, "decisive event"),
    ]
    _write_passages(workspace, passages)
    packet = _packet([_hit(passages[0], 1)])
    transformed = {}
    audits = {}
    for arm in ("o2", "o2-guided", "o2-center"):
        intervention = CaptionPacketIntervention(
            arm=arm,
            intervention=_intervention(),
            workspace=workspace,
            audit_path=workspace.root_dir / f"audit-{arm}.json",
        )
        transformed[arm] = intervention(packet)
        audits[arm] = intervention.audit

    assert transformed["o2"]["hits"] == transformed["o2-guided"]["hits"]
    assert transformed["o2-guided"]["hits"] == transformed["o2-center"]["hits"]
    guided = transformed["o2-guided"]["oracle_guidance"]
    center = transformed["o2-center"]["oracle_guidance"]
    assert guided["guidance_type"] == "exact_locators"
    assert guided["selected_candidates"] == center["selected_candidates"]
    assert guided["selected_candidate_guarantee"] == "exact_annotated_occurrence"
    assert guided["boundary_visibility"] == "exact"
    assert "anchor_timestamps_sec" not in guided
    assert "point_anchors" not in guided
    assert audits["o2-guided"]["anchor_count"] == 0
    assert audits["o2-guided"]["point_anchor_candidate_ranks"] == []


def test_intervention_manifest_rejects_answer_bearing_keys() -> None:
    with pytest.raises(ValueError, match="answer-bearing"):
        OracleIntervention.from_mapping(
            {
                "case_id": "case",
                "normalized_clue_intervals": [[1, 2]],
                "caption_config_digest": "digest",
                "reference_answer": "leaked",
            }
        )


def test_only_intervention_arms_receive_bootstrap_candidates() -> None:
    assert bootstrap_tasks(arm="o0", question="q", index_mode="hybrid") == ()
    for arm in (
        "c0",
        "o1",
        "o1.5",
        "o1.75",
        "o1.75-forced",
        "o2",
        "o2-guided",
        "o2-center",
    ):
        tasks = bootstrap_tasks(arm=arm, question="q", index_mode="hybrid")
        assert len(tasks) == 1
        assert tasks[0].inspection_mode == "search_caption"
        assert tasks[0].top_k == 12


def test_prepare_splits_legacy_gold_from_runtime_case(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    segment = VirtualVideoSegment(
        "seg_0001",
        "video_1",
        "video.mp4",
        0.0,
        30.0,
        0.0,
        30.0,
    )
    VirtualVideoWorkspace.create(
        legacy_root / "mmlifelong-game-test-0000",
        manifest=VirtualVideoManifest(
            workspace_id="legacy",
            segments=(segment,),
        ),
        case=VirtualVideoCase(
            case_id="mmlifelong-game-test-0000",
            question="What happens?",
            gold="an event",
            gold_clue_intervals=((9.0, 11.0),),
            subset="game",
            split="test",
            question_type="Event Tracking",
            metadata={"source_index": 0, "source_subset": "day"},
        ),
    )
    dataset_root = tmp_path / "dataset"
    (dataset_root / "day").mkdir(parents=True)
    (dataset_root / "day" / "test.json").write_text(
        json.dumps(
            [
                {
                    "index": 0,
                    "question": "What happens?",
                    "answer": "an event",
                    "question_type": "Event Tracking",
                    "clue_intervals": [[9.0, 11.0]],
                }
            ]
        ),
        encoding="utf-8",
    )
    out_root = tmp_path / "prepared"

    summary = PREPARE.prepare_oracle_day(
        legacy_case_root=legacy_root,
        dataset_root=dataset_root,
        out_root=out_root,
        caption_config_digest="caption-digest",
        experiment_seed=7,
    )

    runtime = json.loads(
        (
            out_root
            / "cases"
            / "mmlifelong-game-test-0000"
            / "case.json"
        ).read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (
            out_root
            / "cases"
            / "mmlifelong-game-test-0000"
            / "evaluation_case.json"
        ).read_text(encoding="utf-8")
    )
    intervention = json.loads(
        (
            out_root
            / "interventions"
            / "cases"
            / "mmlifelong-game-test-0000.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["runtime_gold_separation"] == "passed"
    assert len(intervention["source_manifest_digest"]) == 64
    assert "gold" not in json.dumps(runtime).casefold()
    assert evaluation["reference_answer"] == "an event"
    assert "answer" not in json.dumps(intervention).casefold()


class _BootstrapInvestigator:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset_run_state(self) -> None:
        self.events.append("reset")

    def mechanical_status(self) -> dict[str, Any]:
        return {}

    def run_batch(self, tasks: tuple[InvestigationTask, ...]) -> tuple[InvestigationReport, ...]:
        self.events.append("bootstrap")
        attempt = ObservationAttempt(
            attempt_id=stable_attempt_id(
                frame_refs=("caption://digest/p0",),
                modality="caption_search",
            ),
            task_id=tasks[0].query_id,
            sampling_config={
                "mode": "search_caption",
                "modality": "caption_search",
                "hits": [
                    {
                        "passage_id": "p0",
                        "range": [1.0, 2.0],
                        "caption_excerpt": "candidate",
                    }
                ],
            },
            frame_refs=("caption://digest/p0",),
            modality="caption_search",
            evidence_role="candidate",
        )
        return (
            InvestigationReport(
                query_id=tasks[0].query_id,
                status="completed",
                attempts=(attempt,),
                cost={"consumes_budget": False},
            ),
        )


class _ImmediateReasoner:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def decide(self, **_: Any) -> ReasonerDecision:
        self.events.append("decide")
        return ReasonerDecision(action="answer", answer="candidate")


def test_driver_records_bootstrap_before_first_reasoner_decision(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    events: list[str] = []
    driver = VirtualVideoMultiRoundDriver(
        reasoner=_ImmediateReasoner(events),
        investigator=_BootstrapInvestigator(events),
        max_rounds=1,
        answer_policy="benchmark_best_effort",
        evidence_control_mode="shadow",
        controller_mode="frozen_baseline",
        bootstrap_tasks=bootstrap_tasks(
            arm="c0",
            question=workspace.case.question,
            index_mode="hybrid",
        ),
    )

    result = driver.run(workspace)

    assert events[:3] == ["reset", "bootstrap", "decide"]
    assert result.trace[0]["type"] == "bootstrap_observation_batch"
    rows = [
        json.loads(line)
        for line in (workspace.root_dir / "observation_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["round_id"] == "bootstrap"


def _numeric_leaves(value: Any) -> tuple[float, ...]:
    if isinstance(value, dict):
        return tuple(number for item in value.values() for number in _numeric_leaves(item))
    if isinstance(value, (list, tuple)):
        return tuple(number for item in value for number in _numeric_leaves(item))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    return ()
