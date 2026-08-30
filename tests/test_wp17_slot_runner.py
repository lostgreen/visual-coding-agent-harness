from __future__ import annotations

from vcah.wp17_slot_runner import (
    WP17_OCR_AGGREGATION_CONTRACT,
    alias_current_evidence,
    build_asr_packet,
    build_ocr_packet,
    construction_prompt,
    frame_evidence_ids,
    packet_digest,
)


def test_packets_are_segment_scoped_compact_and_path_free() -> None:
    ocr = build_ocr_packet(
        (
            {
                "evidence_id": "ocr:1",
                "start_sec": 110.0,
                "end_sec": 130.0,
                "surface": "Yin Tiger",
                "surfaces": [{"surface": "Yin Tiger"}],
                "entity_types": ["screen_text"],
                "ui_regions": ["top"],
                "support_frame_ids": ["f1", "f2"],
                "max_confidence": "high",
                "normalized_surface": "yin tiger",
                "source_path": "/must/not/persist.mp4",
            },
            {
                "evidence_id": "ocr:2",
                "start_sec": 140.0,
                "end_sec": 141.0,
                "surface": "Yin Tiger",
                "normalized_surface": "yin tiger",
                "support_frame_ids": ["f3"],
            },
            {"evidence_id": "ocr:outside", "start_sec": 250.0, "end_sec": 251.0},
        ),
        segment_id="segment-1",
        start_sec=100.0,
        end_sec=220.0,
    )
    asr = build_asr_packet(
        (
            {"start": 105.0, "end": 108.0, "text": "challenge begins"},
            {"start": 250.0, "end": 251.0, "text": "outside"},
        ),
        segment_id="segment-1",
        start_sec=100.0,
        end_sec=220.0,
    )

    assert len(ocr) == 1 and ocr[0]["local_time_range_sec"] == [10.0, 41.0]
    assert ocr[0]["occurrence_ranges_sec"] == [[10.0, 30.0], [40.0, 41.0]]
    assert ocr[0]["source_evidence_count"] == 2
    assert ocr[0]["contract"] == WP17_OCR_AGGREGATION_CONTRACT
    assert "source_path" not in str(ocr)
    assert len(asr) == 1 and asr[0]["evidence_id"] == "asr:segment-1:0000"
    assert packet_digest(ocr) == packet_digest(ocr)


def test_packet_local_aliases_are_short_unique_and_reversible() -> None:
    frame_ids = frame_evidence_ids("segment-1", 2)
    ocr = (
        {
            "evidence_id": "ocragg:segment-1:abc",
            "surfaces": ["Yin Tiger"],
        },
    )
    asr = ({"evidence_id": "asr:segment-1:0000", "text": "fight"},)

    frames, aliased_ocr, aliased_asr, mapping = alias_current_evidence(
        frame_ids,
        ocr,
        asr,
    )

    assert frames == ("f001", "f002")
    assert aliased_ocr[0]["evidence_id"] == "o001"
    assert aliased_asr[0]["evidence_id"] == "a001"
    assert mapping["o001"] == "ocragg:segment-1:abc"
    assert len(mapping) == 4


def test_prompt_hides_evaluation_fields_and_freezes_arm_contract() -> None:
    prompt = construction_prompt(
        arm="e1c2",
        segment_duration_sec=120.0,
        frame_ids=frame_evidence_ids("segment-1", 2),
        ocr_packet=(),
        asr_packet=(),
        history_context='{"slots":[],"versions":{}}',
        history_token_count=9,
        history_token_limit=9,
    )

    assert "active_participants" in prompt
    assert "expected_version" in prompt
    assert "do not instantiate every allowed slot" in prompt
    assert "never a frame/OCR/ASR evidence ID" in prompt
    assert "target at most 400 tokens" in prompt
    assert "target at most 6 directly useful evidence IDs" in prompt
    assert "official interval" in prompt
    assert "slot_capsule" in prompt
