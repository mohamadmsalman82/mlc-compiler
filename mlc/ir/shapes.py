"""Shape and dtype propagation.

The compiler derives every intermediate type itself, from the input specs
forward, rather than reading the FakeTensor metadata the exporter attaches.
That metadata is kept as an oracle: :func:`verify_against_export` cross-checks
the two and is run by the test suite over every model in the benchmark set. A
mismatch is a bug in a shape rule, and the test says exactly which op.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .graph import Node, Value
from .ops import OpClass, lookup, op_class
from .types import Layout


class ShapeError(Exception):
    pass


# --------------------------------------------------------------------------
# dtype
# --------------------------------------------------------------------------

def promote(dtypes: Sequence[torch.dtype]) -> torch.dtype:
    """Type promotion over tensor operands only.

    Python scalars are deliberately excluded by the caller: torch treats them
    as weakly typed, so ``half_tensor * 2.0`` stays half. Dropping them here
    reproduces that without modelling the full weak-type lattice.
    """
    if not dtypes:
        return torch.float32
    out = dtypes[0]
    for d in dtypes[1:]:
        out = torch.promote_types(out, d)
    return out


# --------------------------------------------------------------------------
# broadcasting
# --------------------------------------------------------------------------

def broadcast_shapes(shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    rank = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(rank):
        size = 1
        for s in shapes:
            # right-align
            j = i - (rank - len(s))
            d = s[j] if j >= 0 else 1
            if d == 1:
                continue
            if size not in (1, d):
                raise ShapeError(f"cannot broadcast {list(shapes)} at dim {i}: {size} vs {d}")
            size = d
        out.append(size)
    return tuple(out)


# --------------------------------------------------------------------------
# per-class rules
# --------------------------------------------------------------------------

def infer_pointwise(op: str, args: list, kwargs: dict) -> tuple[tuple[int, ...], torch.dtype]:
    tensors = [a for a in args if isinstance(a, Value)]
    if not tensors:
        raise ShapeError(f"{op} has no tensor operand")
    shape = broadcast_shapes([t.shape for t in tensors])
    spec = lookup(op)
    if spec and spec.dtype_rule is not None:
        dt = spec.dtype_rule(args, kwargs)
        if dt is not None:
            return shape, dt
    dtype = promote([t.dtype for t in tensors])
    # A float scalar operand forces a float result even on integer tensors,
    # matching torch's weak-scalar rule for the one case that actually shows up.
    if dtype in (torch.int32, torch.int64, torch.bool) and any(
        isinstance(a, float) for a in args
    ):
        dtype = torch.float32
    return shape, dtype


def infer_reduction(op: str, args: list, kwargs: dict) -> list[tuple[tuple[int, ...], torch.dtype]]:
    spec = lookup(op)
    assert spec is not None and spec.reduce is not None
    src = args[0]
    kind, dims, keepdim = spec.reduce(args, kwargs)
    shape = []
    for i, d in enumerate(src.shape):
        if i in dims:
            if keepdim:
                shape.append(1)
        else:
            shape.append(d)
    out_dtype = kwargs.get("dtype") or (
        torch.float32 if src.dtype in (torch.bool, torch.int32, torch.int64) and kind in ("mean", "var", "var_mean")
        else src.dtype
    )
    n = spec.n_outputs
    return [(tuple(shape), out_dtype)] * n


def infer_matmul(op: str, args: list, kwargs: dict) -> tuple[tuple[int, ...], torch.dtype]:
    if op == "aten.mm.default":
        a, b = args[0], args[1]
        _check(a.shape[1] == b.shape[0], f"mm: {a.shape} x {b.shape}")
        return (a.shape[0], b.shape[1]), promote([a.dtype, b.dtype])
    if op == "aten.bmm.default":
        a, b = args[0], args[1]
        _check(a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1], f"bmm: {a.shape} x {b.shape}")
        return (a.shape[0], a.shape[1], b.shape[2]), promote([a.dtype, b.dtype])
    if op == "aten.addmm.default":
        bias, a, b = args[0], args[1], args[2]
        _check(a.shape[1] == b.shape[0], f"addmm: {a.shape} x {b.shape}")
        return (a.shape[0], b.shape[1]), promote([bias.dtype, a.dtype, b.dtype])
    if op == "aten.baddbmm.default":
        bias, a, b = args[0], args[1], args[2]
        return (a.shape[0], a.shape[1], b.shape[2]), promote([bias.dtype, a.dtype, b.dtype])
    if op == "aten.matmul.default":
        a, b = args[0], args[1]
        if a.rank == 2 and b.rank == 2:
            return (a.shape[0], b.shape[1]), promote([a.dtype, b.dtype])
        batch = broadcast_shapes([a.shape[:-2], b.shape[:-2]])
        return batch + (a.shape[-2], b.shape[-1]), promote([a.dtype, b.dtype])
    raise ShapeError(f"no matmul rule for {op}")


def infer_view(op: str, args: list, kwargs: dict) -> Layout:
    spec = lookup(op)
    assert spec is not None and spec.view is not None
    return spec.view(args[0].layout, args, kwargs)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ShapeError(msg)


def infer(op: str, args: list, kwargs: dict) -> list[tuple[tuple[int, ...], torch.dtype]]:
    """Output (shape, dtype) for every result of ``op``. Views are handled
    separately by :func:`infer_view` because they return a Layout, not a
    fresh allocation."""
    cls = op_class(op)
    if cls is OpClass.POINTWISE:
        return [infer_pointwise(op, args, kwargs)]
    if cls is OpClass.REDUCTION:
        return infer_reduction(op, args, kwargs)
    if cls is OpClass.MATMUL:
        return [infer_matmul(op, args, kwargs)]
    raise ShapeError(f"no shape rule for {op} (class {cls.value})")


# --------------------------------------------------------------------------
# verification against the exporter's own metadata
# --------------------------------------------------------------------------

def verify_against_export(graph) -> list[str]:
    """Compare every propagated type with the FakeTensor the exporter recorded.

    Returns a list of human-readable mismatches; empty means the propagation
    agrees with torch everywhere. Opaque ops are skipped, since their types
    come from that metadata in the first place.
    """
    problems: list[str] = []
    for n in graph.nodes:
        ref = n.meta.get("export_val")
        if ref is None or n.meta.get("types_from_export"):
            continue
        refs = ref if isinstance(ref, (list, tuple)) else [ref]
        if len(refs) != len(n.outputs):
            problems.append(f"{n.op}: {len(n.outputs)} outputs, export says {len(refs)}")
            continue
        for got, want in zip(n.outputs, refs):
            if want is None:
                continue
            if tuple(want.shape) != tuple(got.shape):
                problems.append(f"{n.op} %{got.name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
            if want.dtype != got.dtype:
                problems.append(f"{n.op} %{got.name}: dtype {got.dtype} != {want.dtype}")
    return problems
