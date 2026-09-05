"""The cost model fusion decisions are made against.

Inference on these models is memory bound, so the model is denominated in
bytes of HBM traffic. Two other effects are converted into that currency:

  * **arithmetic**, at ``flops_per_byte`` -- the ops a device retires while
    moving one byte. Recomputing a value is worth it exactly when the
    arithmetic it costs is cheaper than the round trip it avoids.
  * **launch overhead**, at ``launch_overhead_bytes`` -- what one kernel
    launch costs expressed as forgone bandwidth. This term is why fusion pays
    so much more at batch size 1: the traffic terms shrink with batch size but
    the launch term does not.

Both constants are device properties and are meant to be set per GPU. The
defaults describe an A100 in fp32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from ..config import Config
from ..ir.graph import Node, Value
from ..ir.scalar import Expr, size as expr_size


def itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def value_bytes(v: Value) -> int:
    """Bytes actually touched when reading or writing ``v``.

    Uses the view's element count, not the buffer's: a kernel reading one
    slice of a fused qkv projection touches a third of the buffer.
    """
    return v.numel * itemsize(v.dtype)


def traffic(reads: Iterable[Value], writes: Iterable[Value]) -> int:
    """HBM bytes for one kernel, counting each distinct buffer read once.

    Reading the same buffer twice in one kernel is nearly free in practice --
    the second read hits L2 -- so double counting it would bias the model
    toward splitting kernels.
    """
    seen: set[int] = set()
    total = 0
    for v in reads:
        if id(v.buffer) not in seen:
            seen.add(id(v.buffer))
            total += value_bytes(v)
    for v in writes:
        total += value_bytes(v)
    return total


@dataclass
class MergeDecision:
    merge: bool
    benefit: float
    cost: float
    reason: str

    def __bool__(self) -> bool:
        return self.merge


def evaluate_merge(
    cfg: Config,
    *,
    saved_stores: Sequence[Value],
    saved_loads: Sequence[Value],
    extra_loads: Sequence[Value],
    extra_evaluations: int,
    expr_ops: int,
    saved_launches: int = 1,
) -> MergeDecision:
    """Should two candidate groups share a kernel?

    ``saved_stores`` / ``saved_loads`` are the values that stop round-tripping
    through memory. ``extra_loads`` are operands the receiving kernel did not
    already read. ``extra_evaluations`` is how many additional times the moved
    work runs, which happens when it is pulled into a larger iteration space.
    """
    benefit = float(sum(value_bytes(v) for v in saved_stores))
    benefit += float(sum(value_bytes(v) for v in saved_loads))
    benefit += saved_launches * cfg.effective_launch_bytes

    cost = float(sum(value_bytes(v) for v in extra_loads))
    if extra_evaluations > 0 and expr_ops > 0:
        cost += extra_evaluations * expr_ops / max(cfg.flops_per_byte, 1e-6)

    if expr_ops > cfg.max_recompute_ops and extra_evaluations > 0:
        return MergeDecision(False, benefit, cost, f"expression too large ({expr_ops} ops)")
    ok = benefit > cost
    return MergeDecision(ok, benefit, cost, "profitable" if ok else "not profitable")


def body_ops(body: Sequence[tuple[str, Expr]]) -> int:
    """Scalar operations in a kernel body, for the arithmetic term."""
    return sum(expr_size(e) for _, e in body)
