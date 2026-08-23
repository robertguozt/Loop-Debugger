# loopdbg

An autonomous debugging agent: point it at a project with a failing pytest
suite and it iterates until the tests pass or a brake stops it.

It exists as an exercise in **loop engineering** — designing the control system
around a model rather than prompting the model turn by turn. The reason → act →
observe loop is about thirty lines and always has been. Everything that decides
whether it works in practice lives in the harness around it: what goes into
context each turn, when it stops, who checks the work, and what happens to a
patch that games the tests.

**Live demo:** https://robertguozt.github.io/Loop-Debugger/ — runs real Python
in your browser via Pyodide. Works with no API key (replays a recorded run), or
paste your own key to drive it live.

## The loop

```
observe  run the tests, reduce them to a comparable signal
orient   build context from the failure and current source only
decide   ask for one hypothesis and a full-file patch
act      apply it inside the workspace sandbox
verify   mechanical checks, then an independent reviewer
brake    check every stop condition before going round again
```

## Install

```bash
git clone https://github.com/robertguozt/Loop-Debugger
cd Loop-Debugger && pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Use

```bash
loopdbg examples/buggy_project --trace trace.json

# every brake is a flag; the defaults are a starting point, not a recommendation
loopdbg ./myproject -t tests/test_api.py --max-steps 10 --max-tokens 200000 --patience 3
```

Exit code is `0` only when the tests pass **and** the checker approved the
change. `--trace` writes the whole run as JSON: every hypothesis, patch
signature, token count, and verdict.

## The brakes

Five stop conditions run before each turn, cheapest first; the first to trip
wins. An agent loop without them has one failure mode and it is expensive: it
decides it is making progress, it is wrong, and it keeps going until the budget
is gone.

| brake | stops when |
| --- | --- |
| `step-cap` | iteration ceiling reached — the backstop for when every other signal lies |
| `token-budget` | total tokens spent, counted both directions, checked after each turn |
| `wall-clock` | real time exceeded, which matters when the suite is the slow part |
| `oscillation` | the same test output appeared 3× — a patch that changes nothing that matters |
| `no-progress` | no improvement on the *best* result for N turns |

`no-progress` measures against the best result so far rather than the previous
one. Otherwise an agent buys extra turns by breaking something and un-breaking
it.

## Why there is a checker

Green tests are necessary, not sufficient. A debugging agent has cheaper routes
to green than fixing the defect: special-case the input the test uses, weaken an
assertion, swallow the exception, skip the test.

Two defences. Test files are **read-only at the tool layer** — a write to one is
rejected with an error written for the model to act on, not for a log file. And
every green result goes to a checker before it counts:

1. **Mechanical checks** run first because they are free and cannot be argued
   with — skip markers, swallowed exceptions, no-op patches.
2. **An independent review** then sees only the diff and the original failure. It
   is told nothing about how many attempts it took; a checker that knows the
   maker is on its last try starts finding reasons to approve.

A rejected patch is rolled back and re-scored against the restored tree. Green
tests earned dishonestly are worse than red ones, because the next turn would
build on the cheat.

## Tests

```bash
pip install -e . && pytest
```

26 tests drive the whole loop with a scripted model and no network. That is the
point: brake behaviour is the part you cannot check by watching a live run,
because the interesting cases are the ones where the model misbehaves and you
cannot make it misbehave on demand. Covered: the sandbox rejects escapes and
test-file writes, failure signatures are stable, each brake trips at its
boundary, the checker catches cheats, JSON parsing survives the ways models wrap
it, and a rolled-back cheat is not scored as progress.

## Honest limits

- **The browser demo is not the CLI.** Pyodide has no pytest, so the page runs a
  minimal assert-based runner over three fixed defects. The CLI runs real pytest
  against your actual project. The harness, brakes and checker logic are the
  same; the test runner is not.
- **Full-file patches, not diffs.** Simpler and more reliable to apply, but it
  puts every edited file through the model each turn. On large files that gets
  expensive quickly.
- **Whole-source context.** Up to 12 source files go into each prompt with no
  retrieval step. Fine for a module, wrong for a monorepo.
- **The checker is a model too.** It catches the obvious cheats and it is not
  infallible. Read the diff before you merge it.
- **It runs the code it writes.** Use it on a repo you can `git checkout .` and
  ideally in a container. There is no approval step before a patch is applied.

## Layout

```
loopdbg/
  loop.py      the harness — the ~30 lines everything else exists to support
  brakes.py    five stop conditions behind one interface
  tools.py     rooted workspace, read-only tests, pytest runner
  verifier.py  mechanical checks + independent review
  model.py     pluggable model access (ScriptedModel is what makes tests possible)
  state.py     what the brakes read
  cli.py       every brake exposed as a flag
index.html     the live demo — single file, no build step
tests/         26 tests, no network
```

MIT.
