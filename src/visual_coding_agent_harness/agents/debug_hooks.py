"""Opt-in debugpy breakpoints for agent flow debugging."""

from __future__ import annotations

import os
from typing import Any


_STOPS_ENV = "VCAH_MULTI_AGENT_DEBUG_STOPS"
_WAIT_ENV = "VCAH_MULTI_AGENT_DEBUG_WAIT"


def maybe_break(label: str, **context: Any) -> None:
    """Stop in debugpy when the requested multi-agent debug label is enabled."""

    configured = os.getenv(_STOPS_ENV, "").strip()
    if not configured:
        return
    labels = {item.strip() for item in configured.replace(";", ",").split(",") if item.strip()}
    if "*" not in labels and "all" not in labels and label not in labels:
        return

    context.setdefault("debug_label", label)
    try:
        import debugpy  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - debugging fallback only
        breakpoint()
        return

    try:
        connected = bool(debugpy.is_client_connected())
    except Exception:  # noqa: BLE001 - debugger best-effort only
        connected = False
    if not connected:
        wait = os.getenv(_WAIT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        if not wait:
            return
        try:
            debugpy.wait_for_client()
        except Exception:  # noqa: BLE001 - debugger best-effort only
            return

    debugpy.breakpoint()
