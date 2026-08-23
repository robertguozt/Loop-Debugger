"""Tests for the harness.

These drive the whole loop with a scripted model and no network. That is the
point: brake behaviour is the part of an agent you cannot check by eyeballing
a live run, because the interesting cases are the ones where the model
misbehaves and you cannot make it misbehave on demand.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from loopdbg import loop
from loopdbg.brakes import NoProgress, Oscillation, StepCap, TokenBudget, first_tripped
from loopdbg.model import ScriptedModel, parse_json
from loopdbg.state import Attempt, LoopState, Status
from loopdbg.tools import ToolError, Workspace
from loopdbg.verifier import mechanical_check

BROKEN = """def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return low
    return value
"""

FIXED = """def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value
"""

CHEAT = """def clamp(value, low, high):
    if value == 10 and low == 0 and high == 5:
        return 5
    if value < low:
        return low
    if value > high:
        return low
    return value
"""

TESTS = """from stats import clamp

def test_clamp_high():
    assert clamp(10, 0, 5) == 5

def test_clamp_inside():
    assert clamp(3, 0, 5) == 3
"""


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    (tmp_path / "stats.py").write_text(BROKEN)
    (tmp_path / "test_stats.py").write_text(TESTS)
    return Workspace(tmp_path, test_paths=["test_stats.py"])


def patch(files: dict[str, str], hypothesis: str = "h") -> str:
    return json.dumps({"hypothesis": hypothesis, "files": files})


def verdict(v: str, reason: str = "r") -> str:
    return json.dumps({"verdict": v, "reason": reason})


# --- the sandbox ------------------------------------------------------


def test_test_files_are_read_only(ws: Workspace):
    with pytest.raises(ToolError, match="read-only"):
        ws.write("test_stats.py", "def test_x(): assert True")


def test_paths_cannot_escape_the_workspace(ws: Workspace):
    with pytest.raises(ToolError, match="escapes the workspace"):
        ws.write("../evil.py", "x = 1")


def test_only_python_files_are_writable(ws: Workspace):
    with pytest.raises(ToolError, match="only .py"):
        ws.write("notes.txt", "hello")


def test_missing_file_error_lists_what_exists(ws: Workspace):
    with pytest.raises(ToolError, match="stats.py"):
        ws.read("nope.py")


# --- the feedback signal ---------------------------------------------


def test_identical_failures_hash_identically(ws: Workspace):
    assert ws.run_tests().sig == ws.run_tests().sig


def test_signature_changes_when_the_failure_changes(ws: Workspace):
    before = ws.run_tests().sig
    ws.write("stats.py", FIXED)
    assert ws.run_tests().sig != before


# --- the brakes -------------------------------------------------------


def test_step_cap_trips_at_the_limit():
    state = LoopState()
    state.iteration = 4
    assert StepCap(max_steps=4).check(state)
    state.iteration = 3
    assert StepCap(max_steps=4).check(state) is None


def test_token_budget_counts_both_directions():
    state = LoopState()
    state.record(Attempt(n=1, tokens_in=600, tokens_out=500))
    assert TokenBudget(max_tokens=1000).check(state)


def test_oscillation_trips_on_repeated_output():
    state = LoopState()
    for n in range(3):
        state.record(Attempt(n=n, test_sig="same", failures=2))
    assert Oscillation(max_repeats=3).check(state)


def test_no_progress_tolerates_early_turns():
    state = LoopState()
    state.record(Attempt(n=1, test_sig="a", failures=3))
    assert NoProgress(patience=2).check(state) is None


def test_no_progress_trips_when_stuck_at_the_same_best():
    state = LoopState()
    for n, sig in enumerate(["a", "b", "c"], 1):
        state.record(Attempt(n=n, test_sig=sig, failures=2))
    assert NoProgress(patience=2).check(state)


def test_first_tripped_returns_the_cheapest_brake_first():
    state = LoopState()
    state.iteration = 99
    state.record(Attempt(n=1, tokens_in=10**9))
    name, _ = first_tripped([StepCap(max_steps=1), TokenBudget(max_tokens=1)], state)
    assert name == "step-cap"


# --- the checker ------------------------------------------------------


def test_mechanical_check_catches_a_skip_marker():
    v = mechanical_check({"a.py": ""}, {"a.py": "@pytest.mark.skip\ndef f(): pass"})
    assert v and not v.ok


def test_mechanical_check_catches_a_no_op_patch():
    v = mechanical_check({"a.py": "x = 1"}, {"a.py": "x = 1"})
    assert v and not v.ok


def test_mechanical_check_allows_an_honest_fix():
    assert mechanical_check({"stats.py": BROKEN}, {"stats.py": FIXED}) is None


# --- reply parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure, here you go:\n{"a": 1}\nHope that helps.',
        '{"a": 1}\n\ntrailing prose with } braces {',
    ],
)
def test_parse_json_survives_the_ways_models_wrap_json(raw):
    assert parse_json(raw)["a"] == 1


def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json("no object here")


# --- the loop end to end ----------------------------------------------


def test_clean_fix_is_accepted(ws: Workspace):
    model = ScriptedModel([patch({"stats.py": FIXED}), verdict("pass")])
    out = loop.run(ws, model)
    assert out.fixed
    assert out.state.iteration == 1
    assert out.state.status is Status.FIXED


def test_a_rejected_cheat_is_rolled_back(ws: Workspace):
    model = ScriptedModel([patch({"stats.py": CHEAT}), verdict("fail", "special-cases")] * 4)
    out = loop.run(ws, model)
    assert not out.fixed
    # The cheat must not survive on disk.
    assert "value == 10" not in ws.read("stats.py")
    # And it must not be scored as progress.
    assert all(a.failures > 0 for a in out.state.attempts)


def test_writes_to_test_files_are_reported_back_to_the_model(ws: Workspace):
    model = ScriptedModel([patch({"test_stats.py": "def test_x(): assert True"})] * 8)
    out = loop.run(ws, model, brakes=[StepCap(max_steps=2)])
    assert not out.fixed
    assert "read-only" in out.state.attempts[0].verifier_reason
    assert "assert True" not in ws.read("test_stats.py")


def test_unparseable_replies_do_not_crash_the_loop(ws: Workspace):
    out = loop.run(ws, ScriptedModel(["not json"] * 6), brakes=[StepCap(max_steps=3)])
    assert not out.fixed
    assert out.state.status is Status.BRAKED


def test_a_passing_suite_short_circuits(tmp_path: Path):
    (tmp_path / "stats.py").write_text(FIXED)
    (tmp_path / "test_stats.py").write_text(TESTS)
    w = Workspace(tmp_path, test_paths=["test_stats.py"])
    out = loop.run(w, ScriptedModel([]))
    assert out.fixed
    assert out.state.iteration == 0


def test_events_are_emitted_in_order(ws: Workspace):
    seen: list[str] = []
    model = ScriptedModel([patch({"stats.py": FIXED}), verdict("pass")])
    loop.run(ws, model, on_event=lambda k, d: seen.append(k))
    assert seen[0] == "baseline"
    assert "hypothesis" in seen and "verdict" in seen
    assert seen[-1] == "done"
