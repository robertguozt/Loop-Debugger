"""Tools.

Two rules shape everything here:

1. Errors are written for the agent, not for a log file. "path escapes the
   workspace root" tells the model what to do differently; "PermissionError"
   does not.
2. Writes are idempotent and confined. The agent gets a workspace root and
   cannot address anything outside it, so a bad path is a rejected tool call
   rather than a modified file somewhere on your disk.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .state import signature

MAX_OUTPUT_CHARS = 6000


class ToolError(Exception):
    """Raised with a message the model is expected to read and act on."""


@dataclass
class TestResult:
    passed: int
    failed: int
    ok: bool
    output: str
    sig: str
    timed_out: bool = False

    @property
    def summary(self) -> str:
        if self.timed_out:
            return "test run timed out"
        return f"{self.passed} passed, {self.failed} failed"


class Workspace:
    """A rooted view of the project the agent is allowed to touch."""

    def __init__(self, root: str | Path, test_paths: list[str] | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"workspace root does not exist: {self.root}")
        self.test_paths = test_paths or []
        self._protected = {self._resolve(p) for p in self.test_paths}

    def _resolve(self, rel: str) -> Path:
        candidate = (self.root / rel).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError(
                f"path escapes the workspace root: {rel!r}. "
                f"Use a path relative to the project, e.g. 'src/module.py'."
            )
        return candidate

    # ---- reads -------------------------------------------------------

    def list_files(self, pattern: str = "**/*.py") -> list[str]:
        skip = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
        out = []
        for p in sorted(self.root.glob(pattern)):
            if any(part in skip for part in p.parts):
                continue
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return out

    def read(self, rel: str) -> str:
        path = self._resolve(rel)
        if not path.is_file():
            available = ", ".join(self.list_files()[:20]) or "(none)"
            raise ToolError(
                f"no such file: {rel!r}. Files in this workspace: {available}"
            )
        return path.read_text(encoding="utf-8", errors="replace")

    # ---- writes ------------------------------------------------------

    def write(self, rel: str, content: str) -> str:
        """Overwrite a source file. Refuses to touch the test files.

        Letting an agent edit its own grader is the single most common way a
        debugging loop 'succeeds' without fixing anything.
        """
        path = self._resolve(rel)
        if path in self._protected:
            raise ToolError(
                f"{rel!r} is a test file and is read-only. Fix the source "
                f"code so the existing tests pass — do not change the tests."
            )
        if path.suffix != ".py":
            raise ToolError(f"only .py files may be written, got {rel!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return rel

    def snapshot(self, rels: list[str]) -> dict[str, str]:
        return {r: self.read(r) for r in rels}

    def restore(self, snap: dict[str, str]) -> None:
        for rel, content in snap.items():
            self._resolve(rel).write_text(content, encoding="utf-8")

    # ---- the feedback signal ----------------------------------------

    def run_tests(self, timeout: int = 60) -> TestResult:
        """Run pytest and reduce it to a comparable signal.

        The loop cares about three things: did it pass, how many failed, and
        is this the same failure as last time. The signature strips the noise
        that changes between runs (durations, temp paths, addresses) so
        identical failures hash identically.
        """
        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
        cmd += self.test_paths or ["."]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return TestResult(0, 1, False, "test run timed out", "timeout", True)
        except FileNotFoundError:
            raise ToolError("pytest is not installed — run: pip install pytest")

        out = (proc.stdout + proc.stderr)[-MAX_OUTPUT_CHARS:]
        passed, failed = _parse_counts(out)
        ok = proc.returncode == 0 and failed == 0
        return TestResult(passed, failed, ok, out, signature(_normalise(out)))


_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors)")
_NOISE = [
    (re.compile(r"in \d+\.\d+s"), "in Xs"),
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"/tmp/[^\s:]+"), "/tmp/PATH"),
    (re.compile(r"\r"), ""),
]


def _normalise(out: str) -> str:
    for pattern, repl in _NOISE:
        out = pattern.sub(repl, out)
    return out.strip()


def _parse_counts(out: str) -> tuple[int, int]:
    passed = failed = 0
    for n, label in _COUNT.findall(out):
        if label == "passed":
            passed += int(n)
        else:
            failed += int(n)
    if failed == 0 and passed == 0 and "no tests ran" not in out.lower():
        # Collection error: pytest never got far enough to count anything.
        if "error" in out.lower() or "Traceback" in out:
            failed = 1
    return passed, failed
