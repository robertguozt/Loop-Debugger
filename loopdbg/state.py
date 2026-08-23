"""Loop state.

The harness needs to answer three questions on every turn:
  - how much have we spent?
  - are we actually getting closer?
  - have we seen this exact situation before?

Everything needed to answer those lives here, so brakes and the trace
writer read one object instead of reaching into the loop's locals.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    RUNNING = "running"
    FIXED = "fixed"
    BRAKED = "braked"
    ERROR = "error"


def signature(text: str) -> str:
    """Stable short hash. Used to detect 'we have been here before'."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class Attempt:
    """One trip around the loop."""

    n: int
    hypothesis: str = ""
    patched_files: list[str] = field(default_factory=list)
    patch_sig: str = ""
    test_sig: str = ""
    failures: int = 0
    passed: int = 0
    tests_pass: bool = False
    verifier_verdict: str = ""
    verifier_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    stdout_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "hypothesis": self.hypothesis,
            "patched_files": self.patched_files,
            "patch_sig": self.patch_sig,
            "test_sig": self.test_sig,
            "failures": self.failures,
            "passed": self.passed,
            "tests_pass": self.tests_pass,
            "verifier_verdict": self.verifier_verdict,
            "verifier_reason": self.verifier_reason,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "seconds": round(self.seconds, 2),
            "stdout_tail": self.stdout_tail,
        }


@dataclass
class LoopState:
    goal: str = ""
    status: Status = Status.RUNNING
    iteration: int = 0
    started_at: float = field(default_factory=time.monotonic)
    attempts: list[Attempt] = field(default_factory=list)
    stop_reason: str = ""

    # Rolling counters the brakes read.
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def best_failure_count(self) -> int | None:
        """Fewest failures seen so far — the loop's only progress signal."""
        counts = [a.failures for a in self.attempts if a.test_sig]
        return min(counts) if counts else None

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        self.tokens_in += attempt.tokens_in
        self.tokens_out += attempt.tokens_out

    def seen_test_signatures(self) -> list[str]:
        return [a.test_sig for a in self.attempts if a.test_sig]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "iterations": self.iteration,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "elapsed": round(self.elapsed, 2),
            "attempts": [a.to_dict() for a in self.attempts],
        }
