"""``python -m mlc`` -- inspect what the compiler does to a model.

Three views of the same compilation, because the interesting claims in this
project are claims about intermediate artefacts:

    python -m mlc graph  gpt-small          the graph IR after capture
    python -m mlc show   bert-base          the kernel schedule and memory plan
    python -m mlc source gpt-small --kernel k7   the generated Triton
"""

from __future__ import annotations

import argparse
import sys

import torch

from .api import build_pipeline
from .config import Config
from .ir.capture import capture
from .ir.printer import format_graph
from .kernels import ReductionKernel


def _load(name: str, batch: int, seq: int, device: str):
    from .bench.models import SUITE, build

    if name in SUITE:
        return build(name, batch, seq, device)
    if ":" in name:
        import importlib

        mod_name, attr = name.split(":", 1)
        mod = importlib.import_module(mod_name)
        model, args = getattr(mod, attr)()
        return model.eval(), args
    raise SystemExit(
        f"unknown model {name!r}. Choose one of {', '.join(sorted(SUITE))}, "
        "or pass module:factory where factory() returns (model, example_inputs)."
    )


def _config(args) -> Config:
    return Config(
        elementwise_fusion=not args.no_elementwise,
        recompute=not args.no_recompute,
        reduction_fusion=not args.no_reduction,
        memory_planning=not args.no_memory,
        cuda_graphs=False,
        max_persistent_row=args.max_row,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mlc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["graph", "show", "source", "passes"])
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kernel", default=None, help="source: show only this kernel")
    ap.add_argument("--max-row", type=int, default=Config().max_persistent_row,
                    help="rows longer than this stream instead of staying resident")
    ap.add_argument("--no-elementwise", action="store_true")
    ap.add_argument("--no-recompute", action="store_true")
    ap.add_argument("--no-reduction", action="store_true")
    ap.add_argument("--no-memory", action="store_true")
    args = ap.parse_args(argv)

    model, inputs = _load(args.model, args.batch, args.seq, args.device)
    graph = capture(model, inputs)

    if args.command == "passes":
        return _passes(model, inputs, graph)

    if args.command == "graph":
        print(format_graph(graph, show_buffers=True))
        return 0

    cfg = _config(args)
    schedule = build_pipeline(graph, cfg)

    if args.command == "show":
        print(f"{graph.name}: {len(graph.nodes)} graph nodes -> {len(schedule)} kernels")
        print(schedule.format())
        streamed = [k for k in schedule.kernels
                    if isinstance(k, ReductionKernel) and k.two_pass]
        if streamed:
            print(f"\n{len(streamed)} reduction kernels stream their row "
                  f"(longer than {cfg.max_persistent_row})")
        return 0

    from .codegen.triton_backend import generate_module

    src = generate_module(schedule, cfg)
    if args.kernel:
        blocks = src.split("@triton.jit")
        hit = [b for b in blocks if f"def {args.kernel}(" in b]
        if not hit:
            named = next((k for k in schedule.kernels if k.name == args.kernel), None)
            if named is not None:
                print(f"{args.kernel} is an extern kernel and has no generated "
                      f"source: {named.summary()}", file=sys.stderr)
            else:
                print(f"no kernel named {args.kernel!r}", file=sys.stderr)
            return 1
        print("@triton.jit" + hit[0].split("\n\n\n")[0].rstrip())
    else:
        print(src)
    return 0


def _passes(model, inputs, graph) -> int:
    """Kernel count and memory after each pass, for attribution.

    Each configuration re-captures rather than reusing one graph, because a
    pipeline run annotates the graph's buffers with arena offsets.
    """
    ladder = [
        ("no passes", Config(elementwise_fusion=False, recompute=False,
                             reduction_fusion=False, memory_planning=False,
                             cuda_graphs=False)),
        ("elementwise", Config(reduction_fusion=False, memory_planning=False,
                               cuda_graphs=False)),
        ("+ reduction", Config(memory_planning=False, cuda_graphs=False)),
        ("+ memory", Config(cuda_graphs=False)),
    ]
    print(f"{graph.name}: {len(graph.nodes)} graph nodes\n")
    print(f"{'pass':14s} {'kernels':>8s} {'pointwise':>10s} {'reduction':>10s} "
          f"{'extern':>7s} {'intermediates':>14s}")
    for label, cfg in ladder:
        s = build_pipeline(capture(model, inputs), cfg)
        c = s.counts()
        live = (s.plan.arena_bytes if s.plan.mode == "arena" else s.plan.total_bytes)
        print(f"{label:14s} {len(s):8d} {c.get('pointwise', 0):10d} "
              f"{c.get('reduction', 0):10d} {c.get('extern', 0):7d} "
              f"{live / 1024:11.0f} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
