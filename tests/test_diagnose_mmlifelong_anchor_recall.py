from __future__ import annotations

from tools.diagnose_mmlifelong_anchor_recall import (
    _best_frozen_rank,
    _best_hit_rank,
)


def test_best_hit_rank_uses_strict_positive_overlap() -> None:
    hits = (
        {"rank": 1, "range": [0.0, 10.0]},
        {"rank": 2, "range": [10.0, 20.0]},
        {"rank": 3, "virtual_start_sec": 15.0, "virtual_end_sec": 25.0},
    )

    assert _best_hit_rank(hits, ((20.0, 30.0),)) == 3
    assert _best_hit_rank(hits[:2], ((20.0, 30.0),)) is None


def test_best_frozen_rank_selects_best_across_packets() -> None:
    packets = (
        {"hits": [{"rank": 4, "range": [40.0, 50.0]}]},
        {"hits": [{"rank": 2, "range": [45.0, 55.0]}]},
    )

    assert _best_frozen_rank(packets, ((46.0, 47.0),)) == 2
