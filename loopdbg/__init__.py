"""Autonomous debugging agent: an Anthropic tool-use loop with a Kubernetes sandbox."""

from __future__ import annotations

from .agent import DebugAgent, DebugReport, Phase, StopReason
from .config import AgentConfig, ConfigError, ModelConfig, SandboxConfig
from .executors import ExecResult, Executor, KubernetesExecutor, LocalExecutor, SandboxError
from .tools import TOOL_SCHEMAS, ToolBox, ToolOutcome

__all__ = [
    "TOOL_SCHEMAS",
    "AgentConfig",
    "ConfigError",
    "DebugAgent",
    "DebugReport",
    "ExecResult",
    "Executor",
    "KubernetesExecutor",
    "LocalExecutor",
    "ModelConfig",
    "Phase",
    "SandboxConfig",
    "SandboxError",
    "StopReason",
    "ToolBox",
    "ToolOutcome",
]

__version__ = "1.0.0"
