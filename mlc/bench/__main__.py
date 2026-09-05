"""``python -m mlc.bench`` -- run the benchmark sweep."""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

from ..devices import profile_for
from .models import SUITE
from .runner import VARIANT_PASSES, format_table, run_suite, to_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mlc.bench", description=__doc__)
    ap.add_argument("--models", nargs="+", default=["bert-base", "gpt2-small"],
                    choices=sorted(SUITE))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_PASSES))
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the cost model constants on this device "
                         "instead of taking them from the spec table")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="fp16 is the realistic configuration on a consumer card")
    ap.add_argument("--quick-calibrate", action="store_true",
                    help="same, with shorter measurements")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write the markdown table here; JSON goes alongside it")
    args = ap.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("warning: not running on CUDA. Latency and peak memory numbers "
              "here do not mean anything, and Triton is unavailable so mlc "
              "falls back to its reference backend.\n", file=sys.stderr)

    profile = profile_for(device)
    if (device.type == "cuda" and "no table entry" in profile.name
            and not (args.calibrate or args.quick_calibrate)):
        print(f"warning: no cost-model entry for {profile.name}. Falling back to "
              "A100 constants, which will be wrong for this card. Re-run with "
              "--calibrate to measure them.\n", file=sys.stderr)
    if (args.calibrate or args.quick_calibrate) and device.type == "cuda":
        from .calibrate import calibrate, report

        print("calibrating...", file=sys.stderr)
        measured = calibrate(device, quick=args.quick_calibrate)
        print(report(measured, profile), file=sys.stderr)
        profile = measured
    base = profile.to_config()

    results = run_suite(args.models, args.batches, args.seq, device,
                        args.variants, iters=args.iters, base=base,
                        dtype=getattr(torch, args.dtype))
    table = format_table(results)
    header = (f"# Benchmarks\n\ndevice: `{device}`"
              + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
              + f"\ntorch: `{torch.__version__}`  dtype: `{args.dtype}`\n\n"
              + "cost model:\n```\n" + profile.summary() + "\n```\n")
    if profile.l2_bytes:
        header += _l2_note(results, profile)
    print(header + table)
    if args.out:
        args.out.write_text(header + table)
        args.out.with_suffix(".json").write_text(to_json(results))
        print(f"\nwrote {args.out} and {args.out.with_suffix('.json')}", file=sys.stderr)
    return 0


def _l2_note(results, profile) -> str:
    """Flag configurations whose whole working set fits in L2.

    Ada cards carry a large L2, and a planned arena is small. When the two
    cross, fusion stops saving HBM traffic (there was none to save) and only
    the launch-overhead term is left. Any speedup measured there is a
    statement about launches, not bandwidth, and the tables should say so.
    """
    small = sorted({(r.model, r.batch) for r in results if r.variant == "mlc/+memory"})
    if not small:
        return ""
    return (f"\nNote: L2 on this device is {profile.l2_bytes / 1024 ** 2:.0f} MiB. "
            "Where the planned arena is smaller than that, the working set is "
            "L2-resident and fusion's benefit is launch overhead rather than "
            "bandwidth. Compare the arena size in `python -m mlc show`.\n")


if __name__ == "__main__":
    raise SystemExit(main())
