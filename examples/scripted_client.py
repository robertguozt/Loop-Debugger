"""A deterministic stand-in for ``anthropic.Anthropic`` used by the offline demo.

It replays a fixed sequence of tool calls that mirrors what the real model does
on ``examples/broken_project``: reproduce, localise, patch, verify, report. The
point is to exercise the loop, the tool implementations, the sandbox, and the
history compaction without spending an API call.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

ORIGINAL_MEDIAN = """    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return ordered[midpoint]"""

FIXED_MEDIAN = """    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0"""


@dataclass(slots=True)
class _Usage:
    input_tokens: int = 1200
    output_tokens: int = 180


@dataclass(slots=True)
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Response:
    content: list[_Block]
    stop_reason: str = "tool_use"
    usage: _Usage = field(default_factory=_Usage)


def _turn(step: int, text: str, tool: str, payload: dict[str, Any]) -> _Response:
    return _Response(
        content=[
            _Block(type="text", text=text),
            _Block(type="tool_use", id=f"toolu_{step:03d}", name=tool, input=payload),
        ]
    )


def _script() -> Iterator[_Response]:
    yield _turn(
        1,
        "PHASE 1 - REPRODUCE. Creating the sandbox and mounting the project.",
        "create_ephemeral_debug_pod",
        {"image": "python:3.11-slim"},
    )
    yield _turn(
        2,
        "PHASE 1 - REPRODUCE. Running the suite to capture the real failure.",
        "run_test_suite",
        {"test_target": "tests", "extra_args": ["-x"]},
    )
    yield _turn(
        3,
        "PHASE 2 - LOCALIZE. Two median assertions fail, both on even-sized samples. Finding the definition.",
        "search_codebase",
        {"regex_pattern": r"def median", "file_pattern": "*.py"},
    )
    yield _turn(
        4,
        "PHASE 2 - LOCALIZE. Reading the function body.",
        "read_file",
        {"filepath": "calc/stats.py", "start_line": 20, "end_line": 34},
    )
    yield _turn(
        5,
        "PHASE 3 - PATCH. Root cause: median returns ordered[midpoint] unconditionally, which is "
        "the upper of the two central values when the sample size is even. The docstring and the "
        "tests both require the mean of the central pair. Applying the minimal fix.",
        "apply_unified_diff_or_patch",
        {
            "filepath": "calc/stats.py",
            "original_snippet": ORIGINAL_MEDIAN,
            "replacement_snippet": FIXED_MEDIAN,
        },
    )
    yield _turn(
        6,
        "PHASE 4 - VERIFY. Re-running the full suite for regressions.",
        "run_test_suite",
        {"test_target": "tests"},
    )
    yield _turn(7, "PHASE 4 - VERIFY. Suite is green. Tearing down the sandbox.", "cleanup_pod", {})
    yield _turn(
        8,
        "Reporting.",
        "report_resolution",
        {
            "resolved": True,
            "root_cause": (
                "calc.stats.median returned ordered[len(ordered) // 2] for every sample size, so an "
                "even-sized sample yielded the upper central value instead of the mean of the two "
                "central values."
            ),
            "fix_summary": (
                "calc/stats.py: median now branches on parity, returning the central element for odd "
                "samples and the mean of ordered[midpoint - 1] and ordered[midpoint] for even ones."
            ),
            "verification": "python -m pytest tests -q -> 8 passed, exit code 0.",
            "files_changed": ["calc/stats.py"],
        },
    )


class ScriptedMessages:
    def __init__(self, script: Iterator[_Response]) -> None:
        self._script = script
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        try:
            return next(self._script)
        except StopIteration:  # pragma: no cover - script exhausted
            return _Response(content=[_Block(type="text", text="No further action.")], stop_reason="end_turn")


class ScriptedClient:
    """Duck-types the one attribute the agent touches: ``.messages.create``."""

    def __init__(self) -> None:
        self.messages = ScriptedMessages(_script())
