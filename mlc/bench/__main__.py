"""``python -m mlc.bench`` -- run the benchmark sweep."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

from .models import SUITE
from .runner import VARIANTS, format_table, run_suite, to_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mlc.bench", description=__doc__)
    ap.add_argument("--models", nargs="+", default=["bert-base", "gpt2-small"],
                    choices=sorted(SUITE))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write the markdown table here; JSON goes alongside it")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("warning: not running on CUDA. Latency and peak memory numbers "
              "here do not mean anything, and Triton is unavailable so mlc "
              "falls back to its reference backend.\n", file=sys.stderr)

    results = run_suite(args.models, args.batches, args.seq, device,
                        args.variants, iters=args.iters)
    table = format_table(results)
    header = (f"# Benchmarks\n\ndevice: `{device}`"
              + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
              + f"\ntorch: `{torch.__version__}`\n")
    print(header + table)
    if args.out:
        args.out.write_text(header + table)
        args.out.with_suffix(".json").write_text(to_json(results))
        print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
