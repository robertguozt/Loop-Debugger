"""Regression tests for defects found in review."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from examples.scripted_client import _Block, _Response
from loopdbg import AgentConfig, DebugAgent, ModelConfig, SandboxConfig, StopReason, ToolBox
from loopdbg.executors import ERROR_CHANNEL, ExecResult, _exit_code_from
from loopdbg.tools import _summarise_pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "broken_project"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "broken_project"
    shutil.copytree(EXAMPLE, target)
    return target


def _box(repo: Path) -> ToolBox:
    return ToolBox(AgentConfig(repo_root=repo, sandbox=SandboxConfig(kind="local")))


# --- collection errors must not be double counted -------------------------- #


def _result(stdout: str, exit_code: int) -> ExecResult:
    return ExecResult(command=["pytest"], exit_code=exit_code, stdout=stdout, stderr="", duration_s=0.1)


def test_collection_error_counted_once() -> None:
    stdout = "!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!\n1 error in 0.14s\n"
    assert "1 failing" in _summarise_pytest(_result(stdout, 2))


def test_mixed_summary_is_read_from_the_final_line() -> None:
    stdout = "collected 8 items\n\n2 failed, 6 passed in 0.04s\n"
    summary = _summarise_pytest(_result(stdout, 1))
    assert "2 failing" in summary
    assert "6 passing" in summary


def test_pass_verdict() -> None:
    assert "VERDICT: PASS" in _summarise_pytest(_result("8 passed in 0.04s\n", 0))


# --- exec exit-code parsing must never raise ------------------------------- #


class _FakeWS:
    def __init__(self, status: str) -> None:
        self._channels = {ERROR_CHANNEL: status}

    def read_channel(self, channel: int) -> str:
        return str(self._channels.pop(channel, ""))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ('{"status":"Success"}', 0),
        ('{"status":"Failure","details":{"causes":[{"reason":"ExitCode","message":"7"}]}}', 7),
        ('{"status":"Failure","message":"error dialing backend","reason":"InternalError"}', 1),
        ("", 1),
        ("not json at all", 1),
    ],
)
def test_exit_code_parsing_is_total(status: str, expected: int) -> None:
    assert _exit_code_from(_FakeWS(status)) == expected


# --- config hardening ------------------------------------------------------ #


def test_relative_repo_root_is_resolved(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo.parent)
    config = AgentConfig(repo_root=Path("broken_project"))
    assert config.repo_root.is_absolute()
    assert config.resolve_in_repo("calc/stats.py").is_file()


def test_disallowed_image_is_refused(repo: Path) -> None:
    box = _box(repo)
    out = box.dispatch("create_ephemeral_debug_pod", {"image": "evil.example.com/miner:latest"})
    assert out.is_error
    assert "not allowed" in out.content


def test_allowed_image_does_not_mutate_shared_config(repo: Path) -> None:
    config = AgentConfig(repo_root=repo, sandbox=SandboxConfig(kind="local"))
    box = ToolBox(config)
    try:
        assert not box.dispatch("create_ephemeral_debug_pod", {"image": "python:3.12-slim"}).is_error
        assert config.sandbox.image == "python:3.11-slim"
    finally:
        box.close()


def test_read_file_rejects_an_inverted_range(repo: Path) -> None:
    box = _box(repo)
    out = box.dispatch("read_file", {"filepath": "calc/stats.py", "start_line": 10, "end_line": 3})
    assert out.is_error


def test_zero_timeout_is_rejected(repo: Path) -> None:
    box = _box(repo)
    try:
        box.dispatch("create_ephemeral_debug_pod", {})
        out = box.dispatch("execute_command_in_pod", {"cmd": ["true"], "timeout_s": 0})
        assert out.is_error
    finally:
        box.close()


# --- loop hardening -------------------------------------------------------- #


def test_empty_model_content_does_not_poison_the_conversation(repo: Path) -> None:
    class Empty:
        class _M:
            def create(self, **_: Any) -> _Response:
                return _Response(content=[])

        messages = _M()

    config = AgentConfig(repo_root=repo, sandbox=SandboxConfig(kind="local"), model=ModelConfig(max_steps=5))
    agent = DebugAgent(config, toolbox=ToolBox(config), client=Empty())
    report = agent.run_debug_cycle("tests")

    assert report.stop_reason is StopReason.NO_TOOL_CALL
    for message in agent.messages:
        content = message.get("content")
        assert content, "no message may carry an empty content list"


def test_cache_tokens_are_counted(repo: Path) -> None:
    class Cached:
        class _Usage:
            input_tokens = 10
            cache_creation_input_tokens = 100
            cache_read_input_tokens = 1000
            output_tokens = 5

        class _M:
            def create(self, **_: Any) -> _Response:
                response = _Response(content=[_Block(type="text", text="thinking")])
                response.usage = Cached._Usage()  # type: ignore[assignment]
                return response

        messages = _M()

    config = AgentConfig(repo_root=repo, sandbox=SandboxConfig(kind="local"), model=ModelConfig(max_steps=1))
    agent = DebugAgent(config, toolbox=ToolBox(config), client=Cached())
    report = agent.run_debug_cycle("tests")
    assert report.input_tokens == 1110
