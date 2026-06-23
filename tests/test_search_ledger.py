from pathlib import Path

from visual_coding_agent_harness.workspace import EvidenceWorkspace


def test_explore_candidate_discovery_updates_search_ledger(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_candidates")

    workspace.write_observation(
        tool_name="explore",
        claim="Found candidate windows for Great Attractor.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "what is the great attractor according to the video",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "candidate_id": "cand_0001",
                    "segment_id": "seg_0004",
                    "time_range": [610.0, 630.0],
                    "matched_targets": ["target_1"],
                    "source_modalities": ["asr", "visual"],
                }
            ],
        },
    )

    snapshot = workspace.search_ledger_snapshot()

    assert snapshot["records"][0]["query_norm"] == "what is the great attractor according to the video"
    assert snapshot["candidates"][0]["candidate_key"] == "obs_0001:cand_0001"
    assert snapshot["candidates"][0]["status"] == "pending"
    rendered = workspace.render_search_ledger_view()
    assert "## Exploration Ledger" in rendered
    assert "obs_0001:cand_0001" in rendered
    assert "verify_window" in rendered


def test_verify_window_updates_candidate_status_and_trace(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_verify")
    workspace.write_observation(
        tool_name="explore",
        claim="Found candidate windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "ruler used in eclipse viewer",
            "candidate_windows": [
                {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [0.0, 10.0]}
            ],
        },
    )

    workspace.write_observation(
        tool_name="verify_window",
        claim="ruler not found locally",
        confidence=0.8,
        raw_output={
            "mode": "verify_window",
            "candidate_key": "obs_0001:cand_0001",
            "verification_results": [
                {
                    "target_id": "target_1",
                    "claim": "A ruler is used.",
                    "verdict": "not_found_in_window",
                    "scope": {"segment_id": "seg_0001", "time_range": [0.0, 10.0]},
                }
            ],
        },
    )

    snapshot = workspace.search_ledger_snapshot()

    assert snapshot["candidates"][0]["status"] == "verified_negative"
    assert snapshot["candidates"][0]["checked_claims"] == ["A ruler is used."]
    assert snapshot["candidates"][0]["last_verification_observation_id"] == "obs_0002"
    events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "exploration_ledger_update" for event in events)


def test_negative_only_candidate_status_recommends_time_range_sweep(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_negative_sweep")
    workspace.write_observation(
        tool_name="explore",
        claim="Found candidate windows.",
        confidence=0.6,
        raw_output={
            "mode": "candidate_discovery",
            "support_status": "candidate_only",
            "query": "berries on the Christmas tree",
            "candidate_windows": [
                {
                    "candidate_key": "obs_0001:cand_0001",
                    "segment_id": "seg_0001",
                    "time_range": [0.0, 8.88],
                    "segment_start_sec": 0.0,
                    "segment_end_sec": 74.3,
                }
            ],
        },
    )
    workspace.write_observation(
        tool_name="verify_window",
        claim="berries not found locally",
        confidence=0.8,
        raw_output={
            "mode": "verify_window",
            "candidate_key": "obs_0001:cand_0001",
            "segment_id": "seg_0001",
            "time_range": [0.0, 8.88],
            "verification_results": [
                {
                    "target_id": "target_1",
                    "claim": "Berries are visible.",
                    "verdict": "not_found_in_window",
                    "scope": {"segment_id": "seg_0001", "time_range": [0.0, 8.88]},
                }
            ],
        },
    )

    rendered = workspace.render_search_ledger_view()

    assert "verify_window segment=seg_0001 time=[8.9-17.8]" in rendered
    assert "extend visual coverage" in rendered


def test_wrong_scope_caption_updates_option_coverage(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_options")

    workspace.write_observation(
        tool_name="explore",
        claim="Balkans slavery evidence is related but wrong scope.",
        confidence=0.7,
        raw_output={
            "mode": "caption_fact",
            "support_status": "caption_supported",
            "query": "option B Balkans slavery",
            "condition_match": {"matches_original_question": False, "match_level": "related_but_wrong_scope"},
            "answer_mapping": {"related_option": "B", "option_relation": "wrong_scope"},
            "answer_options": {
                "A": "They settled peacefully",
                "B": "They became enslaved in the Balkans",
                "C": "They fought with Selic or Seljuk Turks",
                "D": "They returned to India",
            },
        },
    )

    snapshot = workspace.search_ledger_snapshot()

    assert snapshot["options"]["B"]["status"] == "wrong_scope"
    assert snapshot["options"]["B"]["related_observation_ids"] == ["obs_0001"]
    rendered = workspace.render_search_ledger_view()
    assert "B: wrong_scope" in rendered
    assert "Untested: A, C, D" in rendered
    assert "test untested options" in rendered


def test_repeated_explore_with_new_candidate_keeps_raw_output_and_warns(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_repeat")
    raw_output = {
        "mode": "candidate_discovery",
        "support_status": "candidate_only",
        "query": "how many timeouts does hun take",
        "candidate_windows": [
            {"candidate_key": "obs_0001:cand_0001", "segment_id": "seg_0001", "time_range": [10.0, 20.0]}
        ],
    }
    workspace.write_observation(tool_name="explore", claim="Timeout candidates.", confidence=0.5, raw_output=raw_output)
    workspace.write_observation(tool_name="explore", claim="Timeout candidates.", confidence=0.5, raw_output=raw_output)
    third = workspace.write_observation(
        tool_name="explore",
        claim="Timeout candidates.",
        confidence=0.5,
        raw_output={
            **raw_output,
            "candidate_windows": [
                {"candidate_key": "obs_0003:cand_0002", "segment_id": "seg_0002", "time_range": [20.0, 30.0]}
            ],
        },
    )

    assert third.raw_output["mode"] == "candidate_discovery"
    assert third.raw_output["candidate_windows"][0]["candidate_key"] == "obs_0003:cand_0002"
    snapshot = workspace.search_ledger_snapshot()
    assert any(candidate["candidate_key"] == "obs_0003:cand_0002" for candidate in snapshot["candidates"])
    events = workspace._read_jsonl_dicts("trace.jsonl")
    assert any(event["type"] == "repeated_explore_detected" for event in events)
    assert not any(event["type"] == "planner_recovery_hint_emitted" for event in events)


def test_event_candidate_discovery_updates_counting_ledger(tmp_path: Path) -> None:
    workspace = EvidenceWorkspace.create(tmp_path, "search_ledger_counting")

    workspace.write_observation(
        tool_name="explore",
        claim="Found timeout candidates.",
        confidence=0.6,
        raw_output={
            "mode": "event_candidate_discovery",
            "support_status": "candidate_only",
            "query": "HUN timeout consumed",
            "event_candidates": [
                {
                    "event_id": "ev_001",
                    "event_type": "timeout_consumed",
                    "team": "HUN",
                    "time_range": [100.0, 120.0],
                    "source": "ocr",
                    "raw_excerpt": "HUN timeout",
                    "status": "pending_verification",
                }
            ],
            "needs_visual_verify": True,
            "cannot_final_cite": True,
        },
    )

    snapshot = workspace.search_ledger_snapshot()

    assert snapshot["candidates"][0]["event_id"] == "ev_001"
    assert snapshot["candidates"][0]["event_type"] == "timeout_consumed"
    assert "## Counting Ledger" in workspace.render_search_ledger_view()
