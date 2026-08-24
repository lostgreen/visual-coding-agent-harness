from __future__ import annotations

import pytest

from vcah.caption_schema import CaptionPassageV1
from vcah.occurrence_field_index import (
    OCCURRENCE_FIELDS,
    augment_oracle_occurrence_passage,
    normalize_occurrence_fields,
    occurrence_field_queries,
    oracle_field_passage,
    passage_rank,
    reciprocal_rank_fusion,
    select_oracle_occurrence_passage,
)


def _passage(passage_id: str, start: float, end: float, text: str) -> CaptionPassageV1:
    return CaptionPassageV1(
        passage_id=passage_id,
        caption_id=f"caption-{passage_id}",
        text=text,
        virtual_start_sec=start,
        virtual_end_sec=end,
        anchor_virtual_sec=start,
        ordinal=0,
    )


def _fields() -> dict[str, dict[str, list[str]]]:
    return {
        "entity": {
            "query_terms": ["Red Loong"],
            "document_terms": ["Red Loong", "赤髯龙"],
        },
        "event": {
            "query_terms": ["defeat boss"],
            "document_terms": ["boss defeat", "victory"],
        },
        "state": {
            "query_terms": ["combat end"],
            "document_terms": ["combat_end"],
        },
    }


def test_select_and_augment_only_oracle_passage() -> None:
    passages = (
        _passage("p0", 0.0, 10.0, "walks through a cave"),
        _passage("p1", 10.0, 30.0, "beast collapses"),
        _passage("p2", 30.0, 40.0, "opens a chest"),
    )
    selected = select_oracle_occurrence_passage(passages, ((18.0, 28.0),))
    assert selected is not None
    assert selected.passage_id == "p1"

    augmented = augment_oracle_occurrence_passage(
        passages,
        oracle_passage_id="p1",
        fields=_fields(),
        selected_fields=("entity", "state"),
    )
    assert augmented[0] == passages[0]
    assert augmented[2] == passages[2]
    assert "entity: Red Loong ; 赤髯龙" in augmented[1].text
    assert "state: combat_end" in augmented[1].text
    assert "event:" not in augmented[1].text
    assert augmented[1].metadata["oracle_fields"] == ["entity", "state"]


def test_field_only_document_and_queries_are_separated() -> None:
    passage = _passage("p1", 10.0, 30.0, "generic victory")
    field_passage = oracle_field_passage(
        passage,
        field_name="event",
        fields=_fields(),
    )
    assert field_passage.text == "event: boss defeat ; victory"
    assert "generic victory" not in field_passage.text
    assert occurrence_field_queries(_fields()) == {
        "entity": ("Red Loong",),
        "event": ("defeat boss",),
        "state": ("combat end",),
    }


def test_normalize_requires_all_nonempty_fields() -> None:
    assert tuple(normalize_occurrence_fields(_fields())) == OCCURRENCE_FIELDS
    broken = _fields()
    broken.pop("state")
    with pytest.raises(ValueError, match="missing occurrence field"):
        normalize_occurrence_fields(broken)


def test_rrf_is_deterministic_and_accumulates_channels() -> None:
    fused = reciprocal_rank_fusion(
        {
            "caption": ("distractor", "gold"),
            "entity": ("gold",),
            "event": ("gold",),
        },
        rrf_k0=60,
    )
    assert fused == ("gold", "distractor")
    assert passage_rank(fused, "gold") == 1
    assert passage_rank(fused, "missing") is None
    with pytest.raises(ValueError, match="cannot be negative"):
        reciprocal_rank_fusion(
            {"caption": ("gold",)},
            channel_weights={"caption": -1.0},
        )


def test_oracle_passage_requires_positive_overlap() -> None:
    passages = (_passage("p0", 0.0, 10.0, "walks"),)
    assert select_oracle_occurrence_passage(passages, ((10.0, 20.0),)) is None
    with pytest.raises(ValueError, match="absent"):
        augment_oracle_occurrence_passage(
            passages,
            oracle_passage_id="missing",
            fields=_fields(),
            selected_fields=("entity",),
        )
