"""Typed, environment-driven configuration for the autonomous debugging agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

ExecutorKind = Literal["kubernetes", "local"]

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-5"
DEFAULT_SANDBOX_IMAGE: Final[str] = "python:3.11-slim"
DEFAULT_NAMESPACE: Final[str] = "debug-agent"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or internally inconsistent."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Settings for the ephemeral execution sandbox."""

    kind: ExecutorKind = "kubernetes"
    namespace: str = DEFAULT_NAMESPACE
    image: str = DEFAULT_SANDBOX_IMAGE
    service_account: str = "debug-sandbox"
    workdir: str = "/workspace"
    # Images the model is allowed to request. Prefixes, matched with startswith.
    allowed_image_prefixes: tuple[str, ...] = (
        "python:",
        "ghcr.io/astral-sh/uv:",
        "docker.io/library/python:",
    )
    pod_ready_timeout_s: int = 120
    exec_timeout_s: int = 300
    cpu_request: str = "250m"
    cpu_limit: str = "1"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    # Pods are labelled so an operator can garbage-collect orphans by selector.
    labels: dict[str, str] = field(
        default_factory=lambda: {"app.kubernetes.io/managed-by": "autonomous-debug-agent"}
    )

    def validate_image(self, image: str) -> str:
        """Reject model-supplied images outside the allowlist.

        The image name arrives as a tool argument, so an unchecked value would
        let the loop pull and run an arbitrary container in the cluster.
        """
        if not self.allowed_image_prefixes:
            return image
        if any(image.startswith(prefix) for prefix in self.allowed_image_prefixes):
            return image
        raise ConfigError(
            f"image {image!r} is not allowed; permitted prefixes: " + ", ".join(self.allowed_image_prefixes)
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings for the Anthropic model driving the loop."""

    model: str = DEFAULT_MODEL
    max_tokens: int = 8192
    temperature: float = 0.0
    max_steps: int = 30
    # Turns kept verbatim before older tool traffic is summarised away.
    history_window_turns: int = 12
    # Tool results longer than this are middle-truncated before entering history.
    max_tool_result_chars: int = 12_000


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Top-level configuration object."""

    repo_root: Path
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api_key: str | None = None
    # Guard rails: the agent may never touch files outside repo_root.
    max_read_bytes: int = 200_000
    max_search_matches: int = 200
    dry_run: bool = False

    def __post_init__(self) -> None:
        # Normalise first: resolve_in_repo compares resolved paths, so a relative
        # or symlinked root would make every file tool report a false traversal.
        object.__setattr__(self, "repo_root", self.repo_root.expanduser().resolve())
        if not self.repo_root.is_dir():
            raise ConfigError(f"repo_root does not exist or is not a directory: {self.repo_root}")

    @classmethod
    def from_env(cls, repo_root: Path | str | None = None) -> AgentConfig:
        root = Path(repo_root or _env_str("DEBUG_AGENT_REPO", str(Path.cwd()))).resolve()
        kind_raw = _env_str("DEBUG_AGENT_EXECUTOR", "kubernetes")
        if kind_raw not in ("kubernetes", "local"):
            raise ConfigError(f"DEBUG_AGENT_EXECUTOR must be 'kubernetes' or 'local', got {kind_raw!r}")
        kind: ExecutorKind = "kubernetes" if kind_raw == "kubernetes" else "local"
        return cls(
            repo_root=root,
            sandbox=SandboxConfig(
                kind=kind,
                namespace=_env_str("DEBUG_AGENT_NAMESPACE", DEFAULT_NAMESPACE),
                image=_env_str("DEBUG_AGENT_IMAGE", DEFAULT_SANDBOX_IMAGE),
                service_account=_env_str("DEBUG_AGENT_SANDBOX_SA", "debug-sandbox"),
                pod_ready_timeout_s=_env_int("DEBUG_AGENT_POD_TIMEOUT", 120),
                exec_timeout_s=_env_int("DEBUG_AGENT_EXEC_TIMEOUT", 300),
            ),
            model=ModelConfig(
                model=_env_str("DEBUG_AGENT_MODEL", DEFAULT_MODEL),
                max_tokens=_env_int("DEBUG_AGENT_MAX_TOKENS", 8192),
                max_steps=_env_int("DEBUG_AGENT_MAX_STEPS", 30),
            ),
            api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            dry_run=_env_bool("DEBUG_AGENT_DRY_RUN", False),
        )

    def resolve_in_repo(self, filepath: str) -> Path:
        """Resolve a caller-supplied path, refusing anything outside the repo root."""
        raw = Path(filepath)
        candidate = raw.resolve() if raw.is_absolute() else (self.repo_root / raw).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise PermissionError(f"path escapes repo root: {filepath}") from exc
        return candidate
