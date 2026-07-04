from runs.summary_schema import RunSummary, validate


def test_with_defaults_has_zero_unsupported():
    summary = RunSummary.with_defaults("run_001", ["case_001"])

    assert summary.unsupported_final_rate == 0.0


def test_validate_rejects_nonzero_unsupported():
    summary = RunSummary.with_defaults("run_001", ["case_001"])
    summary.unsupported_final_rate = 0.1

    assert validate(summary)


def test_to_dict_roundtrip():
    summary = RunSummary.with_defaults("run_001", ["case_001"])

    assert RunSummary.from_dict(summary.to_dict()) == summary


def test_validate_rejects_legacy_votes():
    summary = RunSummary.with_defaults("run_001", ["case_001"])
    summary.legacy_worker_vote_rows = 1

    assert validate(summary)
