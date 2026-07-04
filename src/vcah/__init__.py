"""VideoARM-style slim core for the visual coding-agent harness."""

from vcah.agent import VideoAgent
from vcah.index import ColdIndex, build_cold_index
from vcah.types import Answer

__all__ = ["Answer", "ColdIndex", "VideoAgent", "build_cold_index"]
