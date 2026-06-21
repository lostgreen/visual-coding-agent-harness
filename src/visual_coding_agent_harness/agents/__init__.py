"""MVP workspace agent loop."""

__all__ = ["WorkspaceRunResult", "WorkspaceVisualAgent"]


def __getattr__(name: str):
    if name in __all__:
        from .workspace_agent import WorkspaceRunResult, WorkspaceVisualAgent

        return {
            "WorkspaceRunResult": WorkspaceRunResult,
            "WorkspaceVisualAgent": WorkspaceVisualAgent,
        }[name]
    raise AttributeError(name)
