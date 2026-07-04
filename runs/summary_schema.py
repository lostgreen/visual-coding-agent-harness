"""Compatibility wrapper for packaged VideoMME summary schemas."""

import sys

from visual_coding_agent_harness.evals.videomme import outputs as _impl

sys.modules[__name__] = _impl
