"""Brakes.

An agent loop without brakes has exactly one failure mode and it is
expensive: it decides it is making progress, and it is wrong, and it keeps
going until your budget is gone.

Every brake answers the same question — should this loop stop right now? —
and returns a human-readable reason when the answer is yes. They are checked
in order and the first one to trip wins, so cheap checks go first.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .state import LoopState


class Brake(Protocol):
    name: str

    def check(self, state: LoopState) -> str | None:
        """Return a stop reason, or None to let the loop continue."""
        ...


@dataclass
class StepCap:
    """Hard ceiling on iterations. The backstop when every other signal lies."""

    max_steps: int = 6
    name: str = "step-cap"

    def check(self, state: LoopState) -> str | None:
        if state.iteration >= self.max_steps:
            return f"hit the step cap of {self.max_steps} iterations"
        return None


@dataclass
class TokenBudget:
    """Ceiling on total tokens. Checked after each turn, not before."""

    max_tokens: int = 120_000
    name: str = "token-budget"

    def check(self, state: LoopState) -> str | None:
        if state.total_tokens >= self.max_tokens:
            return (
                f"spent {state.total_tokens:,} tokens against a budget of "
                f"{self.max_tokens:,}"
            )
        return None


@dataclass
class WallClock:
    max_seconds: float = 300.0
    name: str = "wall-clock"

    def check(self, state: LoopState) -> str | None:
        if state.elapsed >= self.max_seconds:
            return f"ran for {state.elapsed:.0f}s against a limit of {self.max_seconds:.0f}s"
        return None


@dataclass
class NoProgress:
    """Stop when the failure count has not improved for N consecutive turns.

    Progress is measured against the best result seen so far, not the previous
    one, so an agent cannot buy extra turns by breaking things and then
    un-breaking them.
    """

    patience: int = 2
    name: str = "no-progress"

    def check(self, state: LoopState) -> str | None:
        scored = [a for a in state.attempts if a.test_sig]
        if len(scored) <= self.patience:
            return None
        best = min(a.failures for a in scored)
        recent = scored[-self.patience :]
        if all(a.failures > best for a in recent) or all(
            a.failures == best for a in scored[-(self.patience + 1) :]
        ):
            return (
                f"no improvement on {best} failing test(s) across "
                f"{self.patience + 1} consecutive attempts"
            )
        return None


@dataclass
class Oscillation:
    """Stop when the same test output has been produced too many times.

    Identical failure output after a patch means the patch changed nothing
    that mattered. Seeing it three times means the agent is circling.
    """

    max_repeats: int = 3
    name: str = "oscillation"

    def check(self, state: LoopState) -> str | None:
        sigs = state.seen_test_signatures()
        if not sigs:
            return None
        sig, count = Counter(sigs).most_common(1)[0]
        if count >= self.max_repeats:
            return f"produced identical test output {count} times — the loop is circling"
        return None


def default_brakes(
    max_steps: int = 6,
    max_tokens: int = 120_000,
    max_seconds: float = 300.0,
) -> list[Brake]:
    return [
        StepCap(max_steps=max_steps),
        TokenBudget(max_tokens=max_tokens),
        WallClock(max_seconds=max_seconds),
        Oscillation(),
        NoProgress(),
    ]


def first_tripped(brakes: list[Brake], state: LoopState) -> tuple[str, str] | None:
    for brake in brakes:
        reason = brake.check(state)
        if reason:
            return brake.name, reason
    return None
