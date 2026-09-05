"""Merge the per-configuration benchmark JSON files into one report.

The sweep is run one configuration at a time so a crash cannot lose
everything, which leaves a directory of fragments. This puts them back
together and adds the two summaries that matter: what the best variant does
against each baseline, and what each pass contributed.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

LADDER = ["mlc/no-passes", "mlc/elementwise", "mlc/+reduction", "mlc/+memory",
          "mlc/+cuda-graphs"]
BASELINES = ["eager", "torch.compile", "torch.compile/reduce-overhead"]


def load(directory: pathlib.Path) -> list[dict]:
    """Read every fragment, filling in dtype from the file name when the run
    that produced it predates the field."""
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            batch = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}", file=sys.stderr)
            continue
        guess = "float16" if "float16" in path.name else (
            "float32" if "float32" in path.name else "")
        for r in batch:
            r.setdefault("dtype", guess)
            if not r.get("dtype"):
                r["dtype"] = guess
        rows.extend(batch)
    return rows


def key(r: dict) -> tuple:
    return (r["model"], r.get("dtype", ""), r["batch"], r["seq"])


def fmt(x, digits=3) -> str:
    return "" if x is None or x != x else f"{x:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("results"))
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    rows = load(a.dir)
    if not rows:
        print("no results found", file=sys.stderr)
        return 1

    groups: dict[tuple, list[dict]] = collections.OrderedDict()
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    out: list[str] = ["# Benchmark results", ""]
    if a.device:
        out += [f"Device: {a.device}", ""]

    # -- headline ---------------------------------------------------------
    out += ["## Summary", "",
            "Best mlc variant against each baseline. Latency is the median of "
            "50 timed calls after warmup; memory is peak allocated.", "",
            "| model | dtype | batch | eager ms | torch.compile ms | "
            "t.c reduce-overhead ms | mlc ms | vs eager | vs best t.c | "
            "mlc peak MB | t.c peak MB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (model, dtype, batch, seq), rs in groups.items():
        by = {r["variant"]: r for r in rs if r.get("ok")}
        base = by.get("eager")
        mlc_rows = [by[v] for v in LADDER if v in by]
        best = min(mlc_rows, key=lambda r: r["latency_ms"]) if mlc_rows else None
        tc = [by[v] for v in ("torch.compile", "torch.compile/reduce-overhead") if v in by]
        best_tc = min(tc, key=lambda r: r["latency_ms"]) if tc else None
        if not base or not best:
            continue
        cells = [
            model, dtype, str(batch),
            fmt(base["latency_ms"]),
            fmt(by.get("torch.compile", {}).get("latency_ms")),
            fmt(by.get("torch.compile/reduce-overhead", {}).get("latency_ms")),
            fmt(best["latency_ms"]),
            f"{base['latency_ms'] / best['latency_ms']:.2f}x",
            f"{best_tc['latency_ms'] / best['latency_ms']:.2f}x" if best_tc else "",
            fmt(best.get("peak_mb"), 1),
            fmt(best_tc.get("peak_mb"), 1) if best_tc else "",
        ]
        out.append("| " + " | ".join(cells) + " |")

    # -- per configuration -------------------------------------------------
    out += ["", "## Every variant", ""]
    for (model, dtype, batch, seq), rs in groups.items():
        out += [f"### {model}  dtype={dtype}  batch={batch}  seq={seq}", "",
                "| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        base = next((r for r in rs if r["variant"] == "eager" and r.get("ok")), None)
        for r in rs:
            if not r.get("ok"):
                out.append(f"| {r['variant']} | | | | | | {r.get('note', 'failed')} |")
                continue
            speed = (f"{base['latency_ms'] / r['latency_ms']:.2f}x"
                     if base and r["latency_ms"] else "")
            out.append(
                f"| {r['variant']} | {r['kernels'] or ''} | {fmt(r['latency_ms'])} "
                f"| {fmt(r.get('latency_p10'))} | {speed} | {fmt(r.get('peak_mb'), 1)} "
                f"| {r['max_err']:.1e} |"
            )
        out.append("")

    # -- attribution -------------------------------------------------------
    out += ["## Per-pass attribution", "",
            "Each row is the gain over the row above, so the difference "
            "between adjacent rows is one pass.", ""]
    for (model, dtype, batch, seq), rs in groups.items():
        by = {r["variant"]: r for r in rs if r.get("ok")}
        present = [v for v in LADDER if v in by]
        if len(present) < 2:
            continue
        out += [f"### {model}  dtype={dtype}  batch={batch}", "",
                "| pass enabled | kernels | latency ms | gain | cumulative |",
                "|---|---:|---:|---:|---:|"]
        first = by[present[0]]
        prev = None
        for v in present:
            r = by[v]
            gain = f"{prev['latency_ms'] / r['latency_ms']:.2f}x" if prev else "baseline"
            cum = f"{first['latency_ms'] / r['latency_ms']:.2f}x"
            out.append(f"| {v} | {r['kernels']} | {fmt(r['latency_ms'])} | {gain} | {cum} |")
            prev = r
        out.append("")

    text = "\n".join(out) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
