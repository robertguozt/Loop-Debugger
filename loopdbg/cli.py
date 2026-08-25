"""Command line entry point: ``python -m loopdbg``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .agent import DebugAgent, StopReason
from .config import AgentConfig, ConfigError, ModelConfig, SandboxConfig
from .tools import ToolBox


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loopdbg",
        description="Autonomously reproduce, localise, patch and verify a failing test suite.",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root to debug.")
    parser.add_argument("--target", required=True, help="pytest target, e.g. tests/test_calc.py")
    parser.add_argument("--hint", default=None, help="Optional description of the reported symptom.")
    parser.add_argument(
        "--executor",
        choices=("kubernetes", "local"),
        default="kubernetes",
        help="Sandbox backend. 'local' runs in a temp directory and needs no cluster.",
    )
    parser.add_argument("--namespace", default="debug-agent")
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--model", default=ModelConfig().model)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="Print the run report as JSON.")
    parser.add_argument("--revert", action="store_true", help="Undo the agent's edits before exiting.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        config = AgentConfig(
            repo_root=args.repo.resolve(),
            sandbox=SandboxConfig(kind=args.executor, namespace=args.namespace, image=args.image),
            model=ModelConfig(model=args.model, max_steps=args.max_steps),
            api_key=AgentConfig.from_env(args.repo.resolve()).api_key,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    toolbox = ToolBox(config)
    agent = DebugAgent(config, toolbox=toolbox)
    report = agent.run_debug_cycle(args.target, args.hint)

    if args.revert:
        reverted = toolbox.revert_all()
        print(f"reverted {len(reverted)} file(s)", file=sys.stderr)

    if args.json:
        print(report.to_json())
    else:
        print(_render(report))
    return 0 if report.stop_reason is StopReason.RESOLVED else 1


def _render(report) -> str:  # type: ignore[no-untyped-def]
    lines = [
        "=" * 72,
        f"stop reason : {report.stop_reason.value}",
        f"resolved    : {report.resolved}",
        f"steps       : {report.steps_used}",
        f"tokens      : {report.input_tokens} in / {report.output_tokens} out",
        f"wall time   : {report.wall_time_s:.1f}s",
        "=" * 72,
    ]
    if report.root_cause:
        lines += ["", "ROOT CAUSE", report.root_cause]
    if report.fix_summary:
        lines += ["", "FIX", report.fix_summary]
    if report.verification:
        lines += ["", "VERIFICATION", report.verification]
    if report.diff:
        lines += ["", "DIFF", report.diff.rstrip()]
    if report.error:
        lines += ["", "ERROR", report.error]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
