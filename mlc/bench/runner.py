"""Benchmark harness: latency and peak memory, against eager and torch.compile.

Every variant is timed the same way and checked against eager for correctness
before it is timed at all, so a fast wrong answer cannot show up in a table.

The variant list is a ladder rather than a single "ours": each rung turns on
one more pass, so the difference between two adjacent rows is that pass's
contribution. Reduction fusion is reported separately from elementwise for
exactly this reason.

``torch.compile`` appears twice. Its default mode is the like-for-like
comparison against our fused kernels; ``reduce-overhead`` adds CUDA graphs, so
it is the honest comparison for our graph-captured variant. Reporting only the
first would flatter us at batch size 1, which is the configuration where the
claim matters most.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import torch

import mlc
from mlc.config import Config

#: Pass settings per variant, as overrides applied to the device's profile.
#: The device supplies flops_per_byte and the launch-overhead constants; the
#: variant supplies which passes run. Keeping them separate means the whole
#: ladder is priced for the card it is running on.
VARIANT_PASSES: dict[str, dict] = {
    "eager": {},
    "torch.compile": {},
    "torch.compile/reduce-overhead": {},
    "mlc/no-passes": dict(elementwise_fusion=False, recompute=False,
                          reduction_fusion=False, memory_planning=False,
                          cuda_graphs=False),
    "mlc/elementwise": dict(reduction_fusion=False, memory_planning=False,
                            cuda_graphs=False),
    "mlc/+reduction": dict(memory_planning=False, cuda_graphs=False),
    "mlc/+memory": dict(cuda_graphs=False),
    "mlc/+cuda-graphs": dict(),
}

VARIANTS = VARIANT_PASSES  # name kept for the CLI's default list

#: Rows whose difference from the previous rung isolates one pass.
LADDER = ["mlc/no-passes", "mlc/elementwise", "mlc/+reduction", "mlc/+memory",
          "mlc/+cuda-graphs"]


@dataclass
class Result:
    model: str
    batch: int
    seq: int
    variant: str
    dtype: str = "float32"
    latency_ms: float = float("nan")
    latency_p10: float = float("nan")
    peak_mb: float = float("nan")
    kernels: int = 0
    compile_s: float = 0.0
    max_err: float = float("nan")
    ok: bool = True
    note: str = ""

    def key(self) -> tuple:
        return (self.model, self.batch, self.seq, self.dtype)


# --------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_callable(fn: Callable, args, device: torch.device,
                  warmup: int = 20, iters: int = 100) -> tuple[float, float]:
    """Median and 10th-percentile latency in milliseconds.

    The median is the headline; the p10 is reported alongside because it is
    much less sensitive to whatever else is on the machine, and a large gap
    between them means the number should not be trusted.
    """
    for _ in range(warmup):
        fn(*args)
    _sync(device)

    samples: list[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn(*args)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args)
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return statistics.median(samples), samples[max(0, len(samples) // 10)]


def peak_memory_mb(fn: Callable, args, device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _max_err(got, want) -> float:
    gs = got if isinstance(got, (list, tuple)) else [got]
    ws = want if isinstance(want, (list, tuple)) else [want]
    worst = 0.0
    for g, w in zip(gs, ws):
        worst = max(worst, (g.float() - w.float()).abs().max().item())
    return worst


def _build_variant(name: str, model, args, device, base: Config):
    """Returns (callable, kernel count, note) or raises."""
    if name == "eager":
        with torch.no_grad():
            return (lambda *a: _no_grad_call(model, a)), 0, ""
    if name.startswith("torch.compile"):
        mode = "reduce-overhead" if "reduce-overhead" in name else None
        compiled = torch.compile(model, mode=mode, fullgraph=False, dynamic=False)
        return (lambda *a: _no_grad_call(compiled, a)), 0, mode or "default"
    cfg = base.replace(**VARIANT_PASSES[name])
    compiled = mlc.compile(model, args, cfg, device=device)
    return compiled, len(compiled.schedule), compiled.backend


def _no_grad_call(model, args):
    with torch.no_grad():
        return model(*args)


def run_one(model_name: str, batch: int, seq: int, device: torch.device,
            variants: Sequence[str], iters: int = 100,
            tolerance: float = 2e-3, base: Config | None = None,
            dtype: torch.dtype = torch.float32) -> list[Result]:
    from ..devices import profile_for
    from .models import build

    base = base if base is not None else profile_for(device).to_config()
    model, args = build(model_name, batch, seq, device, dtype)
    if dtype == torch.float16:
        tolerance = max(tolerance, 5e-2)
    with torch.no_grad():
        reference = model(*args)

    out: list[Result] = []
    for name in variants:
        r = Result(model_name, batch, seq, name, str(dtype).replace("torch.", ""))
        try:
            t0 = time.perf_counter()
            fn, kernels, note = _build_variant(name, model, args, device, base)
            got = fn(*args)
            r.compile_s = time.perf_counter() - t0
            r.kernels = kernels
            r.note = note
            r.max_err = _max_err(got, reference)
            r.ok = r.max_err <= tolerance
            if not r.ok:
                r.note = f"WRONG (max err {r.max_err:.2e})"
                out.append(r)
                continue
            r.latency_ms, r.latency_p10 = time_callable(fn, args, device, iters=iters)
            r.peak_mb = peak_memory_mb(fn, args, device)
        except Exception as exc:  # a variant failing must not stop the sweep
            r.ok = False
            r.note = f"{type(exc).__name__}: {exc}"[:120]
        out.append(r)
        del fn
        gc.collect()
    return out


def run_suite(models: Sequence[str], batches: Sequence[int], seq: int,
              device: torch.device, variants: Sequence[str] = tuple(VARIANT_PASSES),
              iters: int = 100, base: Config | None = None,
              dtype: torch.dtype = torch.float32) -> list[Result]:
    results: list[Result] = []
    for m in models:
        for b in batches:
            results.extend(run_one(m, b, seq, device, variants, iters=iters,
                                   base=base, dtype=dtype))
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def format_table(results: list[Result]) -> str:
    """One table per (model, batch), with speedups relative to eager."""
    groups: dict[tuple, list[Result]] = {}
    for r in results:
        groups.setdefault(r.key(), []).append(r)

    out: list[str] = []
    for key, rows in groups.items():
        model, batch, seq = key
        base = next((r for r in rows if r.variant == "eager" and r.ok), None)
        out.append(f"\n### {model}  batch={batch}  seq={seq}\n")
        out.append("| variant | kernels | latency ms | vs eager | peak MB | max err |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            if not r.ok:
                out.append(f"| {r.variant} | | | | | {r.note} |")
                continue
            speed = (f"{base.latency_ms / r.latency_ms:.2f}x"
                     if base and r.latency_ms > 0 else "")
            kern = str(r.kernels) if r.kernels else ""
            peak = "" if r.peak_mb != r.peak_mb else f"{r.peak_mb:.1f}"
            out.append(f"| {r.variant} | {kern} | {r.latency_ms:.3f} | {speed} | "
                       f"{peak} | {r.max_err:.1e} |")
        out.append("")
        out.append(_attribution(rows))
    return "\n".join(out)


def _attribution(rows: list[Result]) -> str:
    """What each pass contributed, as the gap between adjacent ladder rungs."""
    by_name = {r.variant: r for r in rows if r.ok and r.latency_ms == r.latency_ms}
    lines = ["Per-pass contribution (each row is the gain over the row above):", ""]
    lines.append("| pass enabled | latency ms | gain |")
    lines.append("|---|---:|---:|")
    prev = None
    for name in LADDER:
        r = by_name.get(name)
        if r is None:
            continue
        gain = f"{prev.latency_ms / r.latency_ms:.2f}x" if prev else "baseline"
        lines.append(f"| {name} | {r.latency_ms:.3f} | {gain} |")
        prev = r
    return "\n".join(lines) + "\n"


def to_json(results: list[Result]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
