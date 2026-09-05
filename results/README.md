Benchmark output.

- `RESULTS.md` is the assembled report: summary, every variant, per-pass
  attribution.
- `inductor-comparison.md` counts what each compiler generates for the same
  model.
- `*.json` are the raw per-configuration records the report is built from,
  one file per (model, dtype, batch).

Reproduce with:

```
bash scripts/gpu_check.sh     # correctness first
bash scripts/gpu_bench.sh     # the sweep
python scripts/assemble_results.py --dir results --out results/RESULTS.md
```

Numbers are machine-specific. The ones checked in were measured on a rented
A40; the device and library versions are recorded at the top of `RESULTS.md`.
