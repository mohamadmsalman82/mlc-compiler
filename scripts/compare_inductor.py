"""Compare our schedule against Inductor's, kernel for kernel.

Both compilers emit Triton, so the comparison can be structural rather than
inferential: count what each one generates, and see which ops each fuses that
the other does not. This is what turns "torch.compile is faster here" into a
statement about a specific missing fusion.

Inductor's generated source is recovered from its on-disk cache, which is
where it writes the module it compiles.
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import re
import shutil
import sys
import tempfile
import warnings

import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mlc  # noqa: E402
from mlc.api import build_pipeline  # noqa: E402
from mlc.bench.models import build  # noqa: E402
from mlc.devices import profile_for  # noqa: E402
from mlc.ir.capture import capture  # noqa: E402
from mlc.kernels import ExternKernel, PointwiseKernel, ReductionKernel  # noqa: E402


def inductor_source(model, args) -> str:
    """Compile with Inductor and return every Triton module it wrote."""
    cache = tempfile.mkdtemp(prefix="inductor_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)
    with torch.no_grad():
        compiled(*args)
    parts = []
    for path in sorted(pathlib.Path(cache).rglob("*.py")):
        text = path.read_text()
        if "@triton" in text or "def call(" in text:
            parts.append(text)
    shutil.rmtree(cache, ignore_errors=True)
    return "\n".join(parts)


def inductor_stats(source: str) -> dict:
    """Kernel counts from Inductor's generated code.

    Its naming is load-bearing and stable: triton_poi_ is pointwise,
    triton_red_ and triton_per_ are reductions (looped and persistent), and
    extern_kernels.* are the calls it hands to cuBLAS.
    """
    # Inductor does not write `def triton_poi_...`: it binds the name to an
    # async_compile.triton(...) call whose body defines a function called
    # `triton_`. Match the binding name wherever it appears and dedupe.
    kernels = sorted(set(re.findall(r"\btriton_(?:poi|red|per)_fused_\w+", source)))
    kinds = collections.Counter(k.split("_")[1] for k in kernels)
    return {
        "pointwise": kinds.get("poi", 0),
        "reduction": kinds.get("red", 0) + kinds.get("per", 0),
        "persistent_reduction": kinds.get("per", 0),
        "extern": len(re.findall(r"extern_kernels\.\w+", source)),
        "total_generated": len(kernels),
    }


def mlc_stats(model, args, cfg) -> tuple[dict, object]:
    sched = build_pipeline(capture(model, args), cfg)
    counts = sched.counts()
    return {
        "pointwise": counts.get("pointwise", 0),
        "reduction": counts.get("reduction", 0),
        "persistent_reduction": sum(
            1 for k in sched.kernels
            if isinstance(k, ReductionKernel) and not k.two_pass
        ),
        "extern": counts.get("extern", 0),
        "total_generated": counts.get("pointwise", 0) + counts.get("reduction", 0),
    }, sched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base", "gpt2-small"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()

    dtype = getattr(torch, a.dtype)
    cfg = profile_for(a.device).to_config()
    lines = ["# mlc vs Inductor: what each one generates", "",
             f"device `{a.device}`, dtype `{a.dtype}`, seq {a.seq}", "",
             "| model | batch | compiler | generated kernels | pointwise | reduction "
             "| of which persistent | extern calls |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]

    for name in a.models:
        for batch in a.batches:
            model, args = build(name, batch, a.seq, a.device, dtype)
            mine, sched = mlc_stats(model, args, cfg)
            try:
                theirs = inductor_stats(inductor_source(model, args))
            except Exception as exc:
                print(f"inductor failed on {name} b{batch}: {exc}", file=sys.stderr)
                theirs = None
            for label, st in (("mlc", mine), ("inductor", theirs)):
                if st is None:
                    lines.append(f"| {name} | {batch} | {label} | failed | | | | |")
                    continue
                lines.append(
                    f"| {name} | {batch} | {label} | {st['total_generated']} "
                    f"| {st['pointwise']} | {st['reduction']} "
                    f"| {st['persistent_reduction']} | {st['extern']} |"
                )
            del model
            torch.cuda.empty_cache() if a.device == "cuda" else None

    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
