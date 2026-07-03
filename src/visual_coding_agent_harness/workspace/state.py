"""Compatibility facade for durable workspace state exports."""

from __future__ import annotations

from . import workspace_state as _impl
from .workspace_state import *  # noqa: F401,F403

__all__ = [name for name in dir(_impl) if not name.startswith("_")]


def __getattr__(name: str) -> object:
    return getattr(_impl, name)
