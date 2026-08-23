"""The checker half of maker/checker.

Green tests are necessary but not sufficient. A debugging agent has several
ways to turn the bar green without fixing anything: special-case the input
the test happens to use, weaken an assertion it can reach, catch and swallow
the exception, or skip the test outright.

So verification runs in two stages. Mechanical checks run first because they
are free and cannot be talked out of a verdict. The LLM review runs second,
sees only the diff and the original failure, and is told nothing about how
many attempts it took — a checker that knows the maker is on its last try
starts finding reasons to approve.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Model, parse_json

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_SKIPPED = "skipped"

# Patterns that make tests pass without making code correct.
_SMELLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"@?pytest\.mark\.(skip|xfail)"), "adds a skip/xfail marker"),
    (re.compile(r"pytest\.skip\s*\("), "calls pytest.skip()"),
    (re.compile(r"except\s+\w*(Error|Exception)[^\n]*:\s*\n\s*(pass|return\s+\w+)\s*$", re.M),
     "swallows an exception instead of handling it"),
    (re.compile(r"^\s*sys\.exit\(0\)", re.M), "forces a clean exit"),
]


@dataclass
class Verdict:
    verdict: str
    reason: str

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_PASS


def mechanical_check(before: dict[str, str], after: dict[str, str]) -> Verdict | None:
    """Free checks that run before spending a token. None means 'no objection'."""
    for path, new in after.items():
        old = before.get(path, "")
        added = _added_lines(old, new)
        if not added:
            continue
        blob = "\n".join(added)
        for pattern, description in _SMELLS:
            if pattern.search(blob):
                return Verdict(
                    VERDICT_FAIL,
                    f"the patch to {path} {description}, which suppresses the "
                    f"failure rather than fixing it",
                )
    if all(before.get(p, "") == c for p, c in after.items()):
        return Verdict(VERDICT_FAIL, "the patch changed nothing")
    return None


def _added_lines(old: str, new: str) -> list[str]:
    old_lines = set(old.splitlines())
    return [ln for ln in new.splitlines() if ln.strip() and ln not in old_lines]


SYSTEM = """You review a proposed bug fix. You did not write it and you have \
no stake in it being correct.

Decide one thing: does this change fix the underlying defect, or does it only \
make the failing test stop failing?

Reject the change if it special-cases the exact values the test uses, weakens \
or deletes an assertion, catches an exception to hide it, or hard-codes an \
expected result. Accept it if the logic is now correct for inputs beyond the \
ones the test exercises.

Reply with JSON only, no prose and no code fences:
{"verdict": "pass" | "fail", "reason": "<one sentence>"}"""


def review(
    model: Model,
    original_failure: str,
    before: dict[str, str],
    after: dict[str, str],
) -> tuple[Verdict, int, int]:
    diff_blocks = []
    for path, new in after.items():
        diff_blocks.append(
            f"--- {path} (before)\n{before.get(path, '(new file)')}\n"
            f"+++ {path} (after)\n{new}"
        )
    prompt = (
        f"Original test failure:\n{original_failure[-2500:]}\n\n"
        f"Proposed change:\n" + "\n\n".join(diff_blocks)
    )
    reply = model.complete(SYSTEM, [{"role": "user", "content": prompt}])
    try:
        data = parse_json(reply.text)
        verdict = Verdict(
            str(data.get("verdict", VERDICT_FAIL)).lower(),
            str(data.get("reason", "")) or "no reason given",
        )
    except ValueError:
        verdict = Verdict(VERDICT_FAIL, "checker reply was not valid JSON")
    return verdict, reply.tokens_in, reply.tokens_out
