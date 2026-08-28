from __future__ import annotations

import importlib.util
from pathlib import Path
import threading


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "prepare_mmlifelong_change_triggered_entity_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("change_sampling", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)


def test_ordered_parallel_map_runs_concurrently_and_preserves_input_order() -> None:
    barrier = threading.Barrier(2, timeout=2)

    def work(value: int) -> int:
        barrier.wait()
        return value * 10

    assert sampling._ordered_parallel_map((2, 1), work, workers=2) == (20, 10)


def test_ordered_parallel_map_clamps_worker_count() -> None:
    assert sampling._ordered_parallel_map((1, 2), lambda value: value, workers=0) == (
        1,
        2,
    )
