"""Command line interface.

Every brake is exposed as a flag. That is deliberate: the defaults are a
starting point, not a recommendation, and the right values depend entirely on
how expensive your test suite is and how much you trust the model on this
codebase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .brakes import NoProgress, Oscillation, StepCap, TokenBudget, WallClock
from .loop import run
from .model import DEFAULT_MODEL, AnthropicModel
from .state import Status
from .tools import ToolError, Workspace

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


def _c(text: str, colour: str, enabled: bool) -> str:
    return f"{colour}{text}{RESET}" if enabled else text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loopdbg",
        description="Run an autonomous debugging loop against a failing pytest suite.",
    )
    p.add_argument("workspace", help="project directory the agent may edit")
    p.add_argument(
        "-t", "--test", action="append", default=[],
        help="test file, relative to the workspace. Repeatable. These are "
             "read-only to the agent. Defaults to every test_*.py found.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None, help="defaults to $ANTHROPIC_API_KEY")

    brakes = p.add_argument_group("brakes")
    brakes.add_argument("--max-steps", type=int, default=6)
    brakes.add_argument("--max-tokens", type=int, default=120_000)
    brakes.add_argument("--max-seconds", type=float, default=300.0)
    brakes.add_argument("--patience", type=int, default=2,
                        help="turns without improvement before stopping")
    brakes.add_argument("--no-verify", action="store_true",
                        help="skip the checker. Faster, and you get what you pay for.")

    p.add_argument("--trace", type=Path, default=None, help="write a JSON trace here")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-colour", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    colour = sys.stdout.isatty() and not args.no_colour

    root = Path(args.workspace)
    tests = args.test or [
        str(p.relative_to(root))
        for p in sorted(root.glob("**/test_*.py"))
        if "__pycache__" not in p.parts
    ]
    if not tests:
        print("No test files found. Point at them with --test.", file=sys.stderr)
        return 2

    try:
        ws = Workspace(root, test_paths=tests)
    except ToolError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    brakes = [
        StepCap(max_steps=args.max_steps),
        TokenBudget(max_tokens=args.max_tokens),
        WallClock(max_seconds=args.max_seconds),
        Oscillation(),
        NoProgress(patience=args.patience),
    ]

    def on_event(kind: str, data: dict) -> None:
        if args.quiet:
            return
        if kind == "baseline":
            print(_c(f"  baseline   {data['summary']}", DIM, colour))
        elif kind == "iteration_start":
            print(_c(f"\n[{data['n']}] ", BOLD, colour) + _c("observe -> decide -> act -> verify", DIM, colour))
        elif kind == "hypothesis":
            print(f"  hypothesis {data['text']}")
        elif kind == "tests":
            tint = GREEN if data["ok"] else RED
            print(f"  tests      {_c(data['summary'], tint, colour)}")
        elif kind == "verdict":
            tint = GREEN if data["verdict"] == "pass" else YELLOW
            print(f"  checker    {_c(data['verdict'], tint, colour)} — {data['reason']}")

    print(_c(f"loopdbg  {root}", BOLD, colour))
    print(_c(f"  tests      {', '.join(tests)} (read-only)", DIM, colour))
    print(_c(f"  brakes     {args.max_steps} steps · {args.max_tokens:,} tokens · "
             f"{args.max_seconds:.0f}s · patience {args.patience}", DIM, colour))

    model = AnthropicModel(model=args.model, api_key=args.api_key)
    outcome = run(ws, model, brakes=brakes, on_event=on_event, verify=not args.no_verify)

    tint = GREEN if outcome.fixed else YELLOW
    print("\n" + _c(outcome.summary, tint, colour))

    if args.trace:
        args.trace.write_text(json.dumps(outcome.state.to_dict(), indent=2))
        print(_c(f"trace written to {args.trace}", DIM, colour))

    return 0 if outcome.state.status is Status.FIXED else 1


if __name__ == "__main__":
    raise SystemExit(main())
