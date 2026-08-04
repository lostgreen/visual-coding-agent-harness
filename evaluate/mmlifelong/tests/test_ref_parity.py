from __future__ import annotations

import importlib.util
from pathlib import Path

from evaluate.mmlifelong.metrics import ref_score


def _upstream_ref_n():
    path = Path(__file__).resolve().parents[1] / "vendor" / "upstream" / "eval_ref.py"
    spec = importlib.util.spec_from_file_location("mmlifelong_upstream_eval_ref", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Ref_N


def test_ref_n_wrapper_matches_vendored_upstream() -> None:
    upstream = _upstream_ref_n()
    cases = (
        ([(100.0, 400.0), (600.0, 900.0)], [(200.0, 500.0), (800.0, 1200.0)]),
        ([(-20.0, 60.0)], [(0.0, 60.0)]),
        ([(60.0, 120.0)], [(0.0, 60.0)]),
        ([(10.0, 10.0)], []),
    )
    for predicted, reference in cases:
        for bucket in (60, 300, 600):
            expected = upstream(predicted, reference, 1500.0, bucket)
            actual = ref_score(
                predicted,
                reference,
                total_seconds=1500.0,
                bucket_size=bucket,
            )
            assert actual == expected
