"""Small-sample statistics for occurrence-agent mechanism evaluations."""

from __future__ import annotations

from collections import Counter
import math
from typing import Hashable, Sequence


_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int) -> list[float | None]:
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + _Z_95 * _Z_95 / total
    center = (proportion + _Z_95 * _Z_95 / (2 * total)) / denominator
    spread = (
        _Z_95
        * math.sqrt(
            proportion * (1 - proportion) / total
            + _Z_95 * _Z_95 / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - spread), min(1.0, center + spread)]


def newcombe_difference_interval(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
) -> list[float | None]:
    """Newcombe's score interval for the difference of independent proportions."""

    if total_a <= 0 or total_b <= 0:
        return [None, None]
    rate_a = successes_a / total_a
    rate_b = successes_b / total_b
    lower_a, upper_a = wilson_interval(successes_a, total_a)
    lower_b, upper_b = wilson_interval(successes_b, total_b)
    assert None not in (lower_a, upper_a, lower_b, upper_b)
    difference = rate_a - rate_b
    lower = difference - math.sqrt(
        (rate_a - float(lower_a)) ** 2 + (float(upper_b) - rate_b) ** 2
    )
    upper = difference + math.sqrt(
        (float(upper_a) - rate_a) ** 2 + (rate_b - float(lower_b)) ** 2
    )
    return [max(-1.0, lower), min(1.0, upper)]


def fisher_exact_two_sided(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
) -> float | None:
    """Two-sided Fisher exact p-value using fixed table margins."""

    if total_a <= 0 or total_b <= 0:
        return None
    if not 0 <= successes_a <= total_a or not 0 <= successes_b <= total_b:
        raise ValueError("success counts must be within their totals")
    total = total_a + total_b
    positive_total = successes_a + successes_b
    denominator = math.comb(total, total_a)

    def probability(successes_in_a: int) -> float:
        return (
            math.comb(positive_total, successes_in_a)
            * math.comb(total - positive_total, total_a - successes_in_a)
            / denominator
        )

    lower = max(0, total_a - (total - positive_total))
    upper = min(total_a, positive_total)
    observed = probability(successes_a)
    tolerance = max(1e-15, observed * 1e-12)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(lower, upper + 1)
            if probability(value) <= observed + tolerance
        ),
    )


def cohen_kappa(
    pairs: Sequence[tuple[Hashable, Hashable]],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float | None:
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(left == right for left, right in pairs) / total
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    support = tuple(labels or sorted(set(left_counts) | set(right_counts)))
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in support
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)
