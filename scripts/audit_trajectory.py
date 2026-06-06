#!/usr/bin/env python3
"""Compatibility wrapper for packaged trajectory auditing."""

from __future__ import annotations

import sys

from visual_coding_agent_harness.evals.ablation import audit as _impl


if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
