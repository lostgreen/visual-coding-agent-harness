from __future__ import annotations

from argparse import Namespace

from tools.run_mmlifelong_global_entity_ocr import _selected_passages
from vcah.caption_schema import CaptionPassageV1


def _passages(count: int) -> tuple[CaptionPassageV1, ...]:
    return tuple(
        CaptionPassageV1(
            passage_id=f"p{index:04d}",
            caption_id="c",
            text="caption",
            virtual_start_sec=float(index),
            virtual_end_sec=float(index + 1),
            anchor_virtual_sec=float(index),
            ordinal=index,
            metadata={},
        )
        for index in range(count)
    )


def test_hash20_selection_uses_frozen_seed_and_exact_count() -> None:
    protocol = {
        "sampling": {
            "canary_selection": {
                "seed": "frozen",
                "exact_passage_count": 2,
            }
        }
    }
    selected = _selected_passages(
        _passages(10),
        args=Namespace(selection_mode="hash20"),
        protocol=protocol,
    )
    assert len(selected) == 2
    assert selected == _selected_passages(
        tuple(reversed(_passages(10))),
        args=Namespace(selection_mode="hash20"),
        protocol=protocol,
    )


def test_smoke_selection_is_a_prefix_of_the_hash20_selection() -> None:
    protocol = {
        "sampling": {
            "preflight_selection": {
                "seed": "frozen",
                "exact_passage_count": 2,
            },
            "canary_selection": {
                "seed": "frozen",
                "exact_passage_count": 5,
            },
        }
    }
    smoke = _selected_passages(
        _passages(10),
        args=Namespace(selection_mode="smoke"),
        protocol=protocol,
    )
    canary = _selected_passages(
        _passages(10),
        args=Namespace(selection_mode="hash20"),
        protocol=protocol,
    )
    assert smoke == canary[:2]


def test_full_selection_preserves_caption_order() -> None:
    passages = _passages(3)
    selected = _selected_passages(
        passages,
        args=Namespace(selection_mode="full"),
        protocol={},
    )
    assert selected == passages
