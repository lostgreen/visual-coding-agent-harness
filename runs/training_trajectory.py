"""Compatibility wrapper for packaged training trajectory schemas."""

import sys

from visual_coding_agent_harness.legacy.evals.videomme import training_trajectory as _impl

sys.modules[__name__] = _impl
