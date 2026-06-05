"""Agent loops built on top of visual tool registries."""

__all__ = ["AgentRunResult", "VisualAgent"]


def __getattr__(name: str):
    if name in __all__:
        from .vlm_agent import AgentRunResult, VisualAgent

        return {"AgentRunResult": AgentRunResult, "VisualAgent": VisualAgent}[name]
    raise AttributeError(name)
