"""Tool schemas and implementations for the Anthropic tool-use loop.

``TOOL_SCHEMAS`` is passed verbatim to ``messages.create(tools=...)``.
:class:`ToolBox` owns the mutable session state (the live sandbox) and
dispatches ``tool_use`` blocks to typed Python implementations.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from .config import AgentConfig, ConfigError
from .executors import ExecResult, Executor, SandboxError, build_executor

log = logging.getLogger(__name__)

SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


class ToolError(RuntimeError):
    """Recoverable failure: reported back to the model as an error tool_result."""


@dataclass(slots=True)
class ToolOutcome:
    """Result of one tool invocation."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# JSON schemas
# --------------------------------------------------------------------------- #

TOOL_SCHEMAS: Final[list[dict[str, Any]]] = [
    {
        "name": "create_ephemeral_debug_pod",
        "description": (
            "Create an isolated sandbox (a Kubernetes pod in the debug namespace) and copy the "
            "project into it. Call this once, before running any command or test. Returns the "
            "pod name and namespace to use for subsequent calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": (
                        "Container image for the sandbox, e.g. 'python:3.11-slim'. "
                        "Omit to use the configured default."
                    ),
                },
                "code_volume_map": {
                    "type": "object",
                    "description": (
                        "Map of local source path -> destination path inside the sandbox, e.g. "
                        '{".": "/workspace"}. Omit to mount the whole repository at the '
                        "configured workdir."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_command_in_pod",
        "description": (
            "Run a command inside the debug pod and return exit code, stdout and stderr. "
            "Use for reproducing crashes, running linters, installing dependencies, or "
            "inspecting the environment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argv list, e.g. ['python', '-m', 'pytest', '-x'].",
                },
                "timeout_s": {"type": "integer", "description": "Per-command timeout in seconds."},
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "run_test_suite",
        "description": (
            "Run pytest inside the debug pod against a target (file, directory, or node id) and "
            "return a parsed pass/fail summary alongside the raw output. Re-syncs local edits into "
            "the pod first, so patches are always exercised."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "test_target": {
                    "type": "string",
                    "description": (
                        "pytest target such as 'tests', 'tests/test_calc.py' or "
                        "'tests/test_calc.py::test_median'."
                    ),
                },
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional pytest flags, e.g. ['-x', '-q'].",
                },
            },
            "required": ["test_target"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a source file, optionally a line range. Output is 1-indexed and line-numbered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path relative to the repository root."},
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-indexed, inclusive).",
                },
                "end_line": {"type": "integer", "description": "Last line to return (inclusive)."},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "search_codebase",
        "description": (
            "Search the repository with a Python regular expression and return matching "
            "file:line:text triples. Use it to locate error signatures, definitions and call sites."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "regex_pattern": {"type": "string", "description": "Python re syntax."},
                "file_pattern": {
                    "type": "string",
                    "description": "Glob filter on file names, default '*.py'.",
                },
                "max_results": {"type": "integer", "description": "Cap on returned matches."},
            },
            "required": ["regex_pattern"],
        },
    },
    {
        "name": "apply_unified_diff_or_patch",
        "description": (
            "Replace an exact snippet in a file with a new snippet. The original snippet must appear "
            "exactly once. Python files are re-parsed after the edit and the change is rolled back if "
            "it introduces a syntax error. Keep edits minimal and atomic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path relative to the repository root."},
                "original_snippet": {
                    "type": "string",
                    "description": (
                        "Exact text to replace, including indentation. Must be unique in the file."
                    ),
                },
                "replacement_snippet": {"type": "string", "description": "Replacement text."},
            },
            "required": ["filepath", "original_snippet", "replacement_snippet"],
        },
    },
    {
        "name": "cleanup_pod",
        "description": (
            "Delete the debug pod created by create_ephemeral_debug_pod. "
            "Call this once the investigation is finished."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "report_resolution",
        "description": (
            "Terminate the debugging loop. Call this only after run_test_suite reports zero failures, "
            "or when you are certain no further progress is possible."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "resolved": {"type": "boolean", "description": "True if the failing tests now pass."},
                "root_cause": {"type": "string", "description": "One or two sentences naming the defect."},
                "fix_summary": {"type": "string", "description": "What was changed, file by file."},
                "verification": {
                    "type": "string",
                    "description": "The command run and its result.",
                },
                "files_changed": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["resolved", "root_cause", "fix_summary", "verification"],
        },
    },
]

TERMINAL_TOOL: Final[str] = "report_resolution"


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PatchRecord:
    """A single applied edit, retained so the agent can be audited or rolled back."""

    filepath: str
    original: str
    replacement: str


class ToolBox:
    """Stateful dispatcher backing the Anthropic tool-use loop."""

    def __init__(self, config: AgentConfig, executor: Executor | None = None) -> None:
        self._config = config
        self._executor: Executor | None = executor
        # An injected executor is not a *started* one: create_ephemeral_debug_pod
        # still has to run so the workspace gets seeded.
        self._executor_started = False
        self.patches: list[PatchRecord] = []
        self.backups: dict[str, str] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolOutcome]] = {
            "create_ephemeral_debug_pod": self._create_pod,
            "execute_command_in_pod": self._exec_in_pod,
            "run_test_suite": self._run_tests,
            "read_file": self._read_file,
            "search_codebase": self._search,
            "apply_unified_diff_or_patch": self._patch,
            "cleanup_pod": self._cleanup_pod,
        }

    # ------------------------------------------------------------- dispatch --
    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._handlers) | {TERMINAL_TOOL}

    def dispatch(self, name: str, payload: dict[str, Any]) -> ToolOutcome:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolOutcome(f"unknown tool: {name}", is_error=True)
        try:
            return handler(payload)
        except (ToolError, SandboxError, PermissionError) as exc:
            log.warning("tool %s failed: %s", name, exc)
            return ToolOutcome(f"{type(exc).__name__}: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 - never kill the loop on a tool bug
            log.exception("unhandled error in tool %s", name)
            return ToolOutcome(f"internal tool error: {type(exc).__name__}: {exc}", is_error=True)

    # ------------------------------------------------------------- sandbox ---
    @property
    def executor(self) -> Executor:
        if self._executor is None or not self._executor_started:
            raise ToolError("no debug pod exists yet - call create_ephemeral_debug_pod first")
        return self._executor

    def _default_volume_map(self) -> dict[str, str]:
        return {str(self._config.repo_root): self._config.sandbox.workdir}

    def _create_pod(self, payload: dict[str, Any]) -> ToolOutcome:
        if self._executor_started:
            return ToolOutcome(
                f"pod already running: {self._pod_name()} in namespace {self._config.sandbox.namespace}"
            )
        raw_map = payload.get("code_volume_map") or {}
        volume_map = {str(k): str(v) for k, v in raw_map.items()} or self._default_volume_map()
        volume_map = {
            str(self._config.resolve_in_repo(k)) if k != "." else str(self._config.repo_root): v
            for k, v in volume_map.items()
        }
        sandbox_cfg = self._config.sandbox
        requested = str(payload.get("image") or sandbox_cfg.image)
        try:
            image = sandbox_cfg.validate_image(requested)
        except ConfigError as exc:
            raise ToolError(str(exc)) from exc
        if self._executor is None:
            self._executor = build_executor(replace(sandbox_cfg, image=image))
        pod = self._executor.start(volume_map)
        self._executor_started = True
        mounted = ", ".join(f"{k} -> {v}" for k, v in volume_map.items())
        return ToolOutcome(
            f"sandbox ready\npod_name={pod}\nnamespace={self._config.sandbox.namespace}\n"
            f"backend={self._executor.name}\nimage={image}\nmounted: {mounted}",
            metadata={"pod_name": pod},
        )

    def _pod_name(self) -> str:
        return str(getattr(self._executor, "pod_name", "") or "")

    def _exec_in_pod(self, payload: dict[str, Any]) -> ToolOutcome:
        cmd_raw = payload.get("cmd")
        if not isinstance(cmd_raw, list) or not cmd_raw:
            raise ToolError("cmd must be a non-empty array of strings")
        cmd = [str(part) for part in cmd_raw]
        raw_timeout = payload.get("timeout_s")
        timeout: int | None = None
        if raw_timeout is not None:
            timeout = int(raw_timeout)
            if timeout <= 0:
                raise ToolError("timeout_s must be a positive number of seconds")
        result = self.executor.exec(cmd, timeout)
        return ToolOutcome(result.render(), is_error=False, metadata={"exit_code": result.exit_code})

    def _run_tests(self, payload: dict[str, Any]) -> ToolOutcome:
        target = str(payload.get("test_target") or ".")
        extra = [str(a) for a in payload.get("extra_args") or []]
        self.executor.sync(self._default_volume_map())
        cmd = ["python", "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider", *extra]
        result = self.executor.exec(cmd)
        summary = _summarise_pytest(result)
        return ToolOutcome(
            f"{summary}\n\n{result.render()}",
            metadata={"exit_code": result.exit_code, "passed": result.exit_code == 0},
        )

    def _cleanup_pod(self, _payload: dict[str, Any]) -> ToolOutcome:
        if self._executor is None or not self._executor_started:
            return ToolOutcome("no pod to clean up")
        name = self._pod_name()
        self._executor.cleanup()
        self._executor_started = False
        return ToolOutcome(f"pod {name} deleted")

    def close(self) -> None:
        """Idempotent teardown, safe to call from a finally block."""
        if self._executor is not None and self._executor_started:
            try:
                self._executor.cleanup()
            except Exception:  # noqa: BLE001 - teardown must not raise
                log.exception("sandbox cleanup failed")
            finally:
                self._executor_started = False

    # ---------------------------------------------------------------- files --
    def _read_file(self, payload: dict[str, Any]) -> ToolOutcome:
        path = self._config.resolve_in_repo(str(payload["filepath"]))
        if not path.is_file():
            raise ToolError(f"not a file: {payload['filepath']}")
        if path.stat().st_size > self._config.max_read_bytes:
            raise ToolError(f"file exceeds {self._config.max_read_bytes} bytes; request a line range instead")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(payload.get("start_line") or 1))
        end = min(len(lines), int(payload.get("end_line") or len(lines)))
        if start > len(lines):
            raise ToolError(f"start_line {start} is past end of file ({len(lines)} lines)")
        if start > end:
            raise ToolError(f"start_line {start} is after end_line {end}")
        width = len(str(end))
        body = "\n".join(f"{i:>{width}} | {lines[i - 1]}" for i in range(start, end + 1))
        rel = path.relative_to(self._config.repo_root)
        return ToolOutcome(f"{rel} (lines {start}-{end} of {len(lines)})\n{body}")

    def _iter_files(self, file_pattern: str):  # type: ignore[no-untyped-def]
        for path in sorted(self._config.repo_root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if fnmatch.fnmatch(path.name, file_pattern):
                yield path

    def _search(self, payload: dict[str, Any]) -> ToolOutcome:
        pattern_src = str(payload["regex_pattern"])
        try:
            pattern = re.compile(pattern_src)
        except re.error as exc:
            raise ToolError(f"invalid regex {pattern_src!r}: {exc}") from exc
        file_pattern = str(payload.get("file_pattern") or "*.py")
        limit = self._config.max_search_matches
        cap = min(int(payload.get("max_results") or limit), limit)
        hits: list[str] = []
        for path in self._iter_files(file_pattern):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    rel = path.relative_to(self._config.repo_root)
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= cap:
                        break
            if len(hits) >= cap:
                break
        if not hits:
            return ToolOutcome(f"no matches for {pattern_src!r} in files matching {file_pattern!r}")
        return ToolOutcome(f"{len(hits)} match(es):\n" + "\n".join(hits))

    def _patch(self, payload: dict[str, Any]) -> ToolOutcome:
        rel_path = str(payload["filepath"])
        path = self._config.resolve_in_repo(rel_path)
        if not path.is_file():
            raise ToolError(f"not a file: {rel_path}")
        original = str(payload["original_snippet"])
        replacement = str(payload["replacement_snippet"])
        if original == replacement:
            raise ToolError("original_snippet and replacement_snippet are identical")
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(original)
        if occurrences == 0:
            raise ToolError(
                "original_snippet not found verbatim. Re-read the file and copy the exact text, "
                "including leading whitespace."
            )
        if occurrences > 1:
            raise ToolError(
                f"original_snippet appears {occurrences} times; extend it with surrounding "
                "context to make it unique"
            )
        self.backups.setdefault(str(path), text)
        patched = text.replace(original, replacement, 1)
        if path.suffix == ".py":
            try:
                ast.parse(patched, filename=str(path))
            except SyntaxError as exc:
                raise ToolError(
                    f"patch rejected: it would introduce a syntax error at line {exc.lineno}: {exc.msg}"
                ) from exc
        path.write_text(patched, encoding="utf-8")
        self.patches.append(PatchRecord(rel_path, original, replacement))
        removed = len(original.splitlines())
        added = len(replacement.splitlines())
        return ToolOutcome(
            f"patched {rel_path}: -{removed}/+{added} lines. Re-run run_test_suite to verify.",
            metadata={"filepath": rel_path},
        )

    def revert_all(self) -> list[str]:
        """Restore every touched file to its pre-run contents."""
        restored: list[str] = []
        for path_str, text in self.backups.items():
            Path(path_str).write_text(text, encoding="utf-8")
            restored.append(path_str)
        self.backups.clear()
        self.patches.clear()
        return restored

    def diff(self) -> str:
        """Unified diff of every change made during the run."""
        import difflib

        chunks: list[str] = []
        for path_str, before in self.backups.items():
            after = Path(path_str).read_text(encoding="utf-8")
            if before == after:
                continue
            rel = Path(path_str).relative_to(self._config.repo_root)
            chunks.extend(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
        return "".join(chunks)


_PYTEST_SUMMARY = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


def _summarise_pytest(result: ExecResult) -> str:
    """Turn pytest's tail into a one-line verdict the model can act on."""
    blob = f"{result.stdout}\n{result.stderr}"
    # pytest prints counts twice on a collection failure - once in the
    # "Interrupted: N errors during collection" banner and once in the final
    # summary line. Only the last matching line is authoritative.
    summary_lines = [line for line in blob.splitlines() if _PYTEST_SUMMARY.search(line)]
    tail = summary_lines[-1] if summary_lines else ""
    counts: dict[str, int] = {}
    for count, label in _PYTEST_SUMMARY.findall(tail):
        key = label.rstrip("s")
        counts[key] = counts.get(key, 0) + int(count)
    if result.timed_out:
        return "VERDICT: TIMEOUT - the suite did not finish; suspect an infinite loop or a blocking call."
    if result.exit_code == 0:
        return f"VERDICT: PASS - {counts.get('passed', 0)} passed, {counts.get('skipped', 0)} skipped."
    if result.exit_code == 5:
        return "VERDICT: NO TESTS COLLECTED - check the target path."
    failed = counts.get("failed", 0) + counts.get("error", 0)
    return f"VERDICT: FAIL - {failed} failing, {counts.get('passed', 0)} passing (exit {result.exit_code})."


def tool_schema_names() -> list[str]:
    return [str(schema["name"]) for schema in TOOL_SCHEMAS]


def dump_schemas(indent: int = 2) -> str:
    return json.dumps(TOOL_SCHEMAS, indent=indent)
