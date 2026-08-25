# loopdbg

An autonomous debugging agent built on the native Anthropic Python SDK. Point it at a
failing pytest target. It reproduces the failure inside a Kubernetes pod, reads the source
around the traceback, applies a minimal patch, and re-runs the suite until the failure is
gone or a stop condition fires.

This is an exercise in **loop engineering**. The reason, act, observe cycle is about forty
lines and always has been. What decides whether it works in practice is the harness around
it: what enters context each turn, what gets elided, when the loop stops, what a tool is
allowed to touch, and what happens to a patch that would break the parser.

Docs: **https://robertguozt.github.io/Loop-Debugger/**

```
 1 [reproduce]  ok create_ephemeral_debug_pod  {"image": "python:3.11-slim"}
 2 [localize ]  ok run_test_suite              VERDICT: FAIL - 2 failing, 6 passing
 3 [localize ]  ok search_codebase             {"regex_pattern": "def median"}
 4 [localize ]  ok read_file                   calc/stats.py:20-34
 5 [verify   ]  ok apply_unified_diff_or_patch calc/stats.py -1/+3
 6 [verify   ]  ok run_test_suite              VERDICT: PASS - 8 passed
 7 [verify   ]  ok cleanup_pod
 8            report_resolution  resolved=true
```

## What the loop does

`run_debug_cycle` sends the transcript to the Messages API, pulls `tool_use` blocks out of
the response, runs the matching Python function, and returns `tool_result` blocks in the
next user turn. It repeats until the model calls `report_resolution` or `max_steps` is hit.
Between turns it collapses superseded tool output into stubs, so a twenty-step run does not
carry twenty pytest transcripts.

The system prompt drives four phases. Reproduce, then localize, then patch, then verify.
The agent has to see a failure in tool output before it may name a root cause, and it has
to see a passing run before it may claim success.

## Layout

| Path | What lives there |
|---|---|
| `loopdbg/agent.py` | The control loop, phase machine, history compaction, step budget |
| `loopdbg/tools.py` | Tool JSON schemas and their Python implementations |
| `loopdbg/executors.py` | Kubernetes pod lifecycle and exec streaming; a local backend for tests |
| `loopdbg/config.py` | Typed settings, path guard, image allowlist |
| `loopdbg/cli.py` | `loopdbg --repo … --target …` |
| `k8s/` | Namespace, RBAC, NetworkPolicy, quota, runner Job |
| `examples/` | A broken library, a scripted model client, the demo runner |

## Try it without a cluster

```bash
git clone https://github.com/robertguozt/Loop-Debugger
cd Loop-Debugger
pip install -e ".[dev]"
python examples/run_example.py
```

That run uses a scripted stand-in for the model, so it needs no API key and no cluster. It
still exercises the real loop, the real tools, and a real sandbox: a temp directory seeded
from `examples/broken_project`, where `calc.stats.median` returns the upper middle element
of an even-sized sample instead of the mean of the central pair. Two of eight tests fail
going in. All eight pass coming out.

Swap in the real model with `--live` (needs `ANTHROPIC_API_KEY`), and the real cluster with
`--live --k8s`.

## Run it against your own repository

```bash
export ANTHROPIC_API_KEY=sk-ant-...
kubectl apply -f k8s/
loopdbg --repo /path/to/project --target tests/test_thing.py --namespace debug-agent -v
```

Add `--executor local` to skip Kubernetes. Add `--revert` to undo the agent's edits before
it exits, which is what you want the first few times you point it at something you care about.

## The tools the model sees

| Tool | Signature |
|---|---|
| `create_ephemeral_debug_pod` | `(image, code_volume_map)` |
| `execute_command_in_pod` | `(cmd, timeout_s)` |
| `run_test_suite` | `(test_target, extra_args)` |
| `read_file` | `(filepath, start_line, end_line)` |
| `search_codebase` | `(regex_pattern, file_pattern, max_results)` |
| `apply_unified_diff_or_patch` | `(filepath, original_snippet, replacement_snippet)` |
| `cleanup_pod` | `()` |
| `report_resolution` | `(resolved, root_cause, fix_summary, verification, files_changed)` |

`report_resolution` is the terminal tool. Nothing else ends the loop.

## Guard rails

Every file path resolves against the repository root and anything outside it raises before
the read or write happens. Patches must match exactly once, and a patch that would break
`ast.parse` on a Python file is rejected instead of written. The image name arrives from the
model, so it checks against a prefix allowlist. Sandbox pods carry no service account token,
drop all capabilities, run as UID 1000 under `RuntimeDefault` seccomp, and a NetworkPolicy
denies their egress except DNS. The runner's Role grants `get`/`create`/`delete` on pods and
`get`/`create` on `pods/exec` in one namespace, and nothing else.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 31 tests
mypy               # strict, on loopdbg
ruff check .
```

## License

MIT
