"""Tests for the control loop itself, driven by the scripted client."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from examples.scripted_client import ScriptedClient, _Block, _Response
from loopdbg import AgentConfig, DebugAgent, ModelConfig, SandboxConfig, StopReason, ToolBox

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "broken_project"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "broken_project"
    shutil.copytree(EXAMPLE, target)
    return target


def _config(repo: Path, **model_kwargs: Any) -> AgentConfig:
    return AgentConfig(
        repo_root=repo,
        sandbox=SandboxConfig(kind="local"),
        model=ModelConfig(**model_kwargs),
    )


def test_scripted_run_resolves_the_defect(repo: Path) -> None:
    config = _config(repo, max_steps=15)
    agent = DebugAgent(config, toolbox=ToolBox(config), client=ScriptedClient())
    report = agent.run_debug_cycle("tests")

    assert report.stop_reason is StopReason.RESOLVED
    assert report.resolved is True
    assert report.files_changed == ["calc/stats.py"]
    assert "median" in report.diff
    assert report.steps_used == 8  # 8 model turns, the last one calling report_resolution
    assert len(report.transcript) == 7  # 7 executed tools; report_resolution is terminal
    assert report.input_tokens > 0
    # The patched file really does pass now.
    assert (
        "return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0" in (repo / "calc" / "stats.py").read_text()
    )


def test_max_steps_stops_the_loop(repo: Path) -> None:
    class Looper:
        class _M:
            def create(self, **_: Any) -> _Response:
                return _Response(
                    content=[
                        _Block(
                            type="tool_use",
                            id="t1",
                            name="search_codebase",
                            input={"regex_pattern": "def"},
                        )
                    ]
                )

        messages = _M()

    config = _config(repo, max_steps=4)
    agent = DebugAgent(config, toolbox=ToolBox(config), client=Looper())
    report = agent.run_debug_cycle("tests")

    assert report.stop_reason is StopReason.MAX_STEPS
    assert report.resolved is False
    assert report.steps_used == 4


def test_prose_only_replies_are_nudged_then_abandoned(repo: Path) -> None:
    class Chatty:
        class _M:
            def create(self, **_: Any) -> _Response:
                return _Response(content=[_Block(type="text", text="I think the bug is somewhere.")])

        messages = _M()

    config = _config(repo, max_steps=10)
    agent = DebugAgent(config, toolbox=ToolBox(config), client=Chatty())
    report = agent.run_debug_cycle("tests")

    assert report.stop_reason is StopReason.NO_TOOL_CALL


def test_api_errors_become_a_report_not_a_crash(repo: Path) -> None:
    class Broken:
        class _M:
            def create(self, **_: Any) -> _Response:
                raise RuntimeError("overloaded_error")

        messages = _M()

    config = _config(repo, max_steps=3)
    agent = DebugAgent(config, toolbox=ToolBox(config), client=Broken())
    report = agent.run_debug_cycle("tests")

    assert report.stop_reason is StopReason.API_ERROR
    assert "overloaded_error" in report.error


def test_history_compaction_elides_old_tool_output(repo: Path) -> None:
    config = _config(repo, max_steps=15, history_window_turns=2)
    agent = DebugAgent(config, toolbox=ToolBox(config), client=ScriptedClient())
    agent.run_debug_cycle("tests")

    elided = [
        block
        for message in agent.messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and str(block.get("content", "")).startswith("[elided:")
    ]
    assert elided, "expected old tool results to be collapsed"


def test_sandbox_is_torn_down_even_on_failure(repo: Path) -> None:
    class Broken:
        class _M:
            def create(self, **_: Any) -> _Response:
                raise RuntimeError("boom")

        messages = _M()

    config = _config(repo, max_steps=2)
    toolbox = ToolBox(config)
    agent = DebugAgent(config, toolbox=toolbox, client=Broken())
    agent.run_debug_cycle("tests")
    assert toolbox._executor_started is False
