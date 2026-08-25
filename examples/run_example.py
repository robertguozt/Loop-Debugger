#!/usr/bin/env python3
"""End-to-end scenario: a broken project goes in, a green suite comes out.

    python examples/run_example.py              # offline, scripted model
    python examples/run_example.py --live       # real API call, local sandbox
    python examples/run_example.py --live --k8s # real API call, Kubernetes sandbox

The project under test is copied to a scratch directory first, so the checked-in
example stays broken and the demo is repeatable.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.scripted_client import ScriptedClient  # noqa: E402
from loopdbg import AgentConfig, DebugAgent, ModelConfig, SandboxConfig, ToolBox  # noqa: E402

SOURCE = Path(__file__).resolve().parent / "broken_project"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the real Anthropic API.")
    parser.add_argument("--k8s", action="store_true", help="Use a Kubernetes sandbox instead of a local one.")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch copy for inspection.")
    parser.add_argument("--model", default=ModelConfig().model)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    workdir = Path(mkdtemp(prefix="debug-agent-demo-"))
    repo = workdir / "broken_project"
    shutil.copytree(SOURCE, repo)
    print(f"scratch copy: {repo}\n")

    config = AgentConfig(
        repo_root=repo,
        sandbox=SandboxConfig(kind="kubernetes" if args.k8s else "local", workdir="/workspace"),
        model=ModelConfig(model=args.model, max_steps=25),
    )
    client = None if args.live else ScriptedClient()
    toolbox = ToolBox(config)
    agent = DebugAgent(config, toolbox=toolbox, client=client)

    report = agent.run_debug_cycle(
        "tests",
        failure_hint="test_median_of_even_sample_averages_the_middle_pair asserts 3 == 2.5",
    )

    print("\n" + "=" * 72)
    for record in report.transcript:
        flag = "ERR" if record.is_error else " ok"
        print(f"  {record.index:>2} [{record.phase:<9}] {flag} {record.tool}  {record.input_preview[:70]}")
    print("=" * 72)
    print(f"stop reason : {report.stop_reason.value}")
    print(f"resolved    : {report.resolved}")
    print(f"steps       : {report.steps_used}")
    print(f"tokens      : {report.input_tokens} in / {report.output_tokens} out")
    print(f"\nROOT CAUSE\n{report.root_cause}")
    print(f"\nFIX\n{report.fix_summary}")
    print(f"\nVERIFICATION\n{report.verification}")
    if report.diff:
        print(f"\nDIFF\n{report.diff}")

    if args.keep:
        print(f"\nscratch copy kept at {repo}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0 if report.resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
