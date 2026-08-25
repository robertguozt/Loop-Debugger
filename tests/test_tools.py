"""Unit tests for the tool implementations and the sandbox contract."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from loopdbg import AgentConfig, LocalExecutor, SandboxConfig, ToolBox

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "broken_project"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "broken_project"
    shutil.copytree(EXAMPLE, target)
    return target


@pytest.fixture()
def toolbox(repo: Path) -> ToolBox:
    config = AgentConfig(repo_root=repo, sandbox=SandboxConfig(kind="local"))
    box = ToolBox(config, executor=LocalExecutor(config.sandbox))
    box._executor_started = False  # force the agent to call create_ephemeral_debug_pod
    yield box
    box.close()


def test_read_file_returns_numbered_lines(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("read_file", {"filepath": "calc/stats.py", "start_line": 1, "end_line": 3})
    assert not out.is_error
    assert "1 | " in out.content


def test_read_file_refuses_paths_outside_the_repo(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("read_file", {"filepath": "../../etc/passwd"})
    assert out.is_error
    assert "escapes repo root" in out.content


def test_search_finds_the_definition(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("search_codebase", {"regex_pattern": r"def median"})
    assert not out.is_error
    assert "calc/stats.py" in out.content


def test_search_rejects_a_bad_regex(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("search_codebase", {"regex_pattern": "("})
    assert out.is_error


def test_patch_requires_a_unique_snippet(toolbox: ToolBox) -> None:
    out = toolbox.dispatch(
        "apply_unified_diff_or_patch",
        {
            "filepath": "calc/stats.py",
            "original_snippet": "ordered = sorted(values)",
            "replacement_snippet": "x = 1",
        },
    )
    assert out.is_error
    assert "appears" in out.content


def test_patch_rolls_back_a_syntax_error(toolbox: ToolBox, repo: Path) -> None:
    before = (repo / "calc" / "stats.py").read_text()
    out = toolbox.dispatch(
        "apply_unified_diff_or_patch",
        {
            "filepath": "calc/stats.py",
            "original_snippet": "    return ordered[midpoint]",
            "replacement_snippet": "    return ordered[midpoint",
        },
    )
    assert out.is_error
    assert "syntax error" in out.content
    assert (repo / "calc" / "stats.py").read_text() == before


def test_tools_require_a_sandbox_first(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("run_test_suite", {"test_target": "tests"})
    assert out.is_error
    assert "create_ephemeral_debug_pod" in out.content


def test_full_cycle_fails_then_passes(toolbox: ToolBox) -> None:
    created = toolbox.dispatch("create_ephemeral_debug_pod", {})
    assert not created.is_error

    failing = toolbox.dispatch("run_test_suite", {"test_target": "tests"})
    assert "VERDICT: FAIL" in failing.content
    assert failing.metadata["passed"] is False

    patched = toolbox.dispatch(
        "apply_unified_diff_or_patch",
        {
            "filepath": "calc/stats.py",
            "original_snippet": "    midpoint = len(ordered) // 2\n    return ordered[midpoint]",
            "replacement_snippet": (
                "    midpoint = len(ordered) // 2\n"
                "    if len(ordered) % 2 == 1:\n"
                "        return float(ordered[midpoint])\n"
                "    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0"
            ),
        },
    )
    assert not patched.is_error

    passing = toolbox.dispatch("run_test_suite", {"test_target": "tests"})
    assert "VERDICT: PASS" in passing.content
    assert passing.metadata["passed"] is True
    assert "median" in toolbox.diff()


def test_revert_restores_the_original_file(toolbox: ToolBox, repo: Path) -> None:
    before = (repo / "calc" / "stats.py").read_text()
    toolbox.dispatch(
        "apply_unified_diff_or_patch",
        {
            "filepath": "calc/stats.py",
            "original_snippet": "    return ordered[midpoint]",
            "replacement_snippet": "    return float(ordered[midpoint])",
        },
    )
    assert (repo / "calc" / "stats.py").read_text() != before
    toolbox.revert_all()
    assert (repo / "calc" / "stats.py").read_text() == before


def test_unknown_tool_is_reported_not_raised(toolbox: ToolBox) -> None:
    out = toolbox.dispatch("teleport", {})
    assert out.is_error
