# broken_project

A deliberately defective sample library. Two of the eight tests in
`tests/test_stats.py` fail because `calc.stats.median` returns the upper middle
element of an even-sized sample instead of the mean of the two central values.

Run it yourself to see the failure:

```bash
cd examples/broken_project && python -m pytest -q
```

The agent's job is to find that line, fix it, and prove the suite passes.
