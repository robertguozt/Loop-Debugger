"""The harness.

The loop itself is about thirty lines and always has been. Everything that
determines whether it works in practice — what goes into context each turn,
when it stops, who checks the work, what happens to a rejected patch — lives
around it. That is the whole argument for treating the loop as the thing you
engineer.

    observe  -> run the tests, reduce them to a comparable signal
    orient   -> build context from the failure and the current source only
    decide   -> ask the model for a hypothesis and a full-file patch
    act      -> apply the patch inside the workspace sandbox
    verify   -> mechanical checks, then an independent reviewer
    brake    -> check every stop condition before going round again
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import verifier
from .brakes import Brake, default_brakes, first_tripped
from .model import Model, parse_json
from .state import Attempt, LoopState, Status, signature
from .tools import ToolError, Workspace

SYSTEM = """You are fixing a bug in a Python project. Tests are already \
written and they are correct — your job is to change the source so they pass.

Each turn you see the current source and the current test output. Form one \
specific hypothesis about the defect, then rewrite the file that contains it.

Rules:
- Return the COMPLETE new contents of each file you change, not a diff.
- Never edit test files. They are read-only and writes to them are rejected.
- Fix the underlying logic. Do not special-case the values the tests use.
- If your last attempt did not work, say what it ruled out before trying again.

Reply with JSON only, no prose and no code fences:
{"hypothesis": "<one sentence on what is broken and why>",
 "files": {"path/to/file.py": "<complete file contents>"}}"""


@dataclass
class Outcome:
    state: LoopState
    fixed: bool

    @property
    def summary(self) -> str:
        s = self.state
        if s.status is Status.FIXED:
            return f"Fixed in {s.iteration} iteration(s), {s.total_tokens:,} tokens."
        return f"Stopped after {s.iteration} iteration(s): {s.stop_reason}"


def run(
    workspace: Workspace,
    model: Model,
    brakes: list[Brake] | None = None,
    goal: str = "make the failing tests pass",
    on_event: Callable[[str, dict], None] | None = None,
    verify: bool = True,
) -> Outcome:
    brakes = brakes if brakes is not None else default_brakes()
    state = LoopState(goal=goal)
    emit = on_event or (lambda kind, data: None)

    # --- observe once before entering the loop -------------------------
    baseline = workspace.run_tests()
    emit("baseline", {"summary": baseline.summary, "output": baseline.output})
    if baseline.ok:
        state.status = Status.FIXED
        state.stop_reason = "tests already pass — nothing to fix"
        return Outcome(state, True)

    original_failure = baseline.output
    current = baseline
    transcript: list[dict] = []

    while True:
        state.iteration += 1
        attempt = Attempt(n=state.iteration)
        started = time.monotonic()
        emit("iteration_start", {"n": state.iteration})

        # --- orient ----------------------------------------------------
        sources = [f for f in workspace.list_files() if f not in workspace.test_paths]
        try:
            source_blob = "\n\n".join(
                f"### {p}\n{workspace.read(p)}" for p in sources[:12]
            )
        except ToolError as exc:
            state.status = Status.ERROR
            state.stop_reason = str(exc)
            return Outcome(state, False)

        user = (
            f"Goal: {goal}\n\n"
            f"Current source:\n{source_blob}\n\n"
            f"Current test output ({current.summary}):\n{current.output[-3000:]}"
        )
        if transcript:
            user += "\n\nWhat you already tried:\n" + "\n".join(
                f"- attempt {t['n']}: {t['hypothesis']} -> {t['result']}"
                for t in transcript
            )

        # --- decide ----------------------------------------------------
        reply = model.complete(SYSTEM, [{"role": "user", "content": user}])
        attempt.tokens_in += reply.tokens_in
        attempt.tokens_out += reply.tokens_out
        try:
            plan = parse_json(reply.text)
            files = plan.get("files") or {}
            attempt.hypothesis = str(plan.get("hypothesis", "")).strip()
            if not isinstance(files, dict) or not files:
                raise ValueError("reply contained no files to write")
        except ValueError as exc:
            attempt.verifier_verdict = verifier.VERDICT_FAIL
            attempt.verifier_reason = f"unusable reply: {exc}"
            state.record(attempt)
            emit("attempt", attempt.to_dict())
            if (tripped := first_tripped(brakes, state)) :
                return _stop(state, tripped)
            continue

        emit("hypothesis", {"n": state.iteration, "text": attempt.hypothesis})

        # --- act -------------------------------------------------------
        targets = [p for p in files if p not in workspace.test_paths]
        before = {}
        for p in targets:
            try:
                before[p] = workspace.read(p)
            except ToolError:
                before[p] = ""
        rollback = dict(before)

        written, rejected = [], []
        for path, content in files.items():
            try:
                written.append(workspace.write(path, str(content)))
            except ToolError as exc:
                rejected.append(str(exc))
        attempt.patched_files = written
        attempt.patch_sig = signature("".join(sorted(written)) + str(sorted(files.values())))

        if not written:
            attempt.verifier_reason = "; ".join(rejected) or "no writable files in patch"
            state.record(attempt)
            emit("attempt", attempt.to_dict())
            transcript.append({"n": attempt.n, "hypothesis": attempt.hypothesis,
                               "result": attempt.verifier_reason})
            if (tripped := first_tripped(brakes, state)):
                return _stop(state, tripped)
            continue

        # --- observe ---------------------------------------------------
        current = workspace.run_tests()
        attempt.passed, attempt.failures = current.passed, current.failed
        attempt.tests_pass = current.ok
        attempt.test_sig = current.sig
        attempt.stdout_tail = current.output[-1200:]
        emit("tests", {"n": state.iteration, "summary": current.summary, "ok": current.ok})

        # --- verify ----------------------------------------------------
        if current.ok and verify:
            after = {p: workspace.read(p) for p in written}
            mech = verifier.mechanical_check(rollback, after)
            if mech is not None:
                verdict = mech
            else:
                verdict, tin, tout = verifier.review(model, original_failure, rollback, after)
                attempt.tokens_in += tin
                attempt.tokens_out += tout
            attempt.verifier_verdict = verdict.verdict
            attempt.verifier_reason = verdict.reason
            emit("verdict", {"n": state.iteration, **verdict.__dict__})

            if not verdict.ok:
                # Green tests earned dishonestly are worse than red ones:
                # roll back so the next turn cannot build on the cheat.
                workspace.restore(rollback)
                current = workspace.run_tests()
                # Re-score against the restored tree. Without this the attempt
                # keeps the cheat's failure count of 0 and the no-progress
                # brake reads a rejected patch as the best result so far.
                attempt.tests_pass = False
                attempt.passed, attempt.failures = current.passed, current.failed
                attempt.test_sig = current.sig
        elif current.ok:
            attempt.verifier_verdict = verifier.VERDICT_SKIPPED

        attempt.seconds = time.monotonic() - started
        state.record(attempt)
        emit("attempt", attempt.to_dict())

        if attempt.tests_pass:
            state.status = Status.FIXED
            state.stop_reason = "tests pass and the change was verified"
            emit("done", {"status": "fixed", "n": state.iteration})
            return Outcome(state, True)

        transcript.append({
            "n": attempt.n,
            "hypothesis": attempt.hypothesis,
            "result": attempt.verifier_reason or current.summary,
        })

        # --- brake -----------------------------------------------------
        if (tripped := first_tripped(brakes, state)):
            return _stop(state, tripped)


def _stop(state: LoopState, tripped: tuple[str, str]) -> Outcome:
    name, reason = tripped
    state.status = Status.BRAKED
    state.stop_reason = f"{name}: {reason}"
    return Outcome(state, False)
