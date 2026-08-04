from typing import List, Tuple, Set

Interval = Tuple[float, float]

def Ref_N(
    intervals_a: List[Interval],
    intervals_b: List[Interval],
    total_seconds: float,
    bucket_size: float = 300.0,
) -> float:
    def intervals_to_buckets(intervals: List[Interval]) -> Set[int]:
        buckets: Set[int] = set()
        for s, e in intervals:
            # clamp
            s = max(0.0, s)
            e = min(total_seconds, e)
            if s >= e:
                continue

            start = int(s // bucket_size)
            end = int((e - 1e-9) // bucket_size)
            buckets.update(range(start, end + 1))
        return buckets

    buckets_a = intervals_to_buckets(intervals_a)
    buckets_b = intervals_to_buckets(intervals_b)

    if not buckets_a and not buckets_b:
        return 0.0

    return len(buckets_a & buckets_b) / len(buckets_a | buckets_b)


if __name__ == "__main__":
    intervals_a = [(100.0, 400.0), (600.0, 900.0)]
    intervals_b = [(200.0, 500.0), (800.0, 1200.0)]
    total_seconds = 1500.0 # Video Duration
    bucket_size = 300.0

    score = Ref_N(intervals_a, intervals_b, total_seconds, bucket_size)
    print(f"Ref_N score: {score:.3f}")