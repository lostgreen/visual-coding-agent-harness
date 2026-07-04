"""Legacy tool protocol and registry primitives."""

from .protocol import ToolRequest, ToolResult
from .registry import ToolError, ToolRegistry, ToolRuntimeSpec, tool

__all__ = ["ToolError", "ToolRegistry", "ToolRequest", "ToolResult", "ToolRuntimeSpec", "tool"]
