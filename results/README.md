Benchmark output lands here.

```
python -m mlc.bench --models bert-base gpt2-small --batches 1 8 32 --out results/bench.md
```

writes `bench.md` (tables) and `bench.json` (raw records). Both are
gitignored except this file: numbers are machine-specific and belong in a
commit only once, with the device named alongside them.
