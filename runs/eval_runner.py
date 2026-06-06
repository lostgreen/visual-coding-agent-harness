#!/usr/bin/env python3
"""Compatibility wrapper for the packaged VideoMME eval runner."""

from __future__ import annotations

import sys

from visual_coding_agent_harness.evals.videomme import runner as _impl


if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
