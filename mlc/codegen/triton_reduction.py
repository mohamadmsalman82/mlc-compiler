"""Triton source for reduction kernels.

One program per row. Two shapes of kernel come out of here.

**Persistent.** The row fits the register budget, so it is loaded once and
every step runs over it in order. Two reductions cost two cross-lane reduces
over registers and no extra memory traffic at all, which is why softmax and
layer norm each collapse to a single kernel with a single read of their input.

**Streamed.** The row does not fit, so each reduction gets its own loop over
the row and the element-space stores get one more. Accumulators are kept as
vectors across loop iterations and reduced across lanes once at the end, which
avoids loop-carried scalars and is the shape Triton generates the best code
for.

The streamed path is where online softmax earns its place: a max followed by a
sum of ``exp(x - max)`` would otherwise be two full passes over memory, and
the recurrence turns it into one.
"""

from __future__ import annotations

import math

from ..config import Config
from ..ir.scalar import Expr, loads as expr_loads
from ..kernels import Bind, Reduce, ReductionKernel
from .reduction_schedule import plan_passes, scalar_binds, varies_by_column
from .triton_backend import (LaunchSpec, _load_suffix, next_pow2, render_expr,
                             store_value)

#: Identity of each reduction, used to neutralise masked lanes.
IDENTITY = {
    "sum": "0.0",
    "max": '-float("inf")',
    "min": 'float("inf")',
    "prod": "1.0",
}

#: Cross-lane reduce. ``prod`` goes through tl.reduce with a helper because
#: tl.prod is not present in every Triton version we care about.
REDUCE_FN = {
    "sum": "tl.sum({0}, axis=0)",
    "max": "tl.max({0}, axis=0)",
    "min": "tl.min({0}, axis=0)",
    "prod": "tl.reduce({0}, 0, _prod_combine)",
}

#: How a vector accumulator absorbs one chunk.
ACCUMULATE = {
    "sum": "{acc} + {x}",
    "max": "tl.maximum({acc}, {x})",
    "min": "tl.minimum({acc}, {x})",
    "prod": "{acc} * {x}",
}

INIT = {
    "sum": "tl.zeros([BLOCK_COL], dtype=tl.float32)",
    "max": 'tl.full([BLOCK_COL], -float("inf"), tl.float32)',
    "min": 'tl.full([BLOCK_COL], float("inf"), tl.float32)',
    "prod": "tl.full([BLOCK_COL], 1.0, tl.float32)",
}


def warps_for_reduction(block: int) -> int:
    if block <= 256:
        return 4
    if block <= 1024:
        return 8
    return 16


def emit_reduction(k: ReductionKernel, cfg: Config) -> tuple[str, LaunchSpec]:
    return (_emit_streamed(k, cfg) if k.two_pass else _emit_persistent(k, cfg))


def _signature(k: ReductionKernel) -> list[str]:
    params = [f"in_ptr{i}" for i in range(len(k.inputs))]
    params += [f"out_ptr{i}" for i in range(len(k.outputs))]
    params.append("BLOCK_COL: tl.constexpr")
    return params


def _header(k: ReductionKernel, mode: str) -> list[str]:
    return [
        "@triton.jit",
        f"def {k.name}({', '.join(_signature(k))}):",
        f"    # {mode}: {k.n_rows} rows of {k.reduce_numel}, "
        f"row space {list(k.row_space)}, reduced {list(k.reduce_space)}",
        "    row = tl.program_id(0)",
    ]


def _emit_loads(k, names, indent, masked, only=None):
    """Load operands. Anything that does not vary along the column axis is a
    scalar load with no mask, which is how a per-row statistic or a bias costs
    one word instead of a vector."""
    lines = []
    for i, arg in enumerate(k.inputs):
        if only is not None and i not in only:
            continue
        name = names[i]
        if arg.index.varies_along("col"):
            guard = ", mask=cmask, other=0.0" if masked else ""
            expr = arg.index.render()
        else:
            guard = ""
            expr = arg.index.render_without("col")
        lines.append(f"{indent}{name} = tl.load(in_ptr{i} + ({expr}){guard})"
                     f"{_load_suffix(arg.dtype)}  # {arg.value.name}")
    return lines


def _emit_stores(k, names, indent, masked, only=None):
    lines = []
    for i, arg in enumerate(k.outputs):
        nm = arg.value.name
        if only is not None and nm not in only:
            continue
        expr_ir = k.out_expr[nm]
        value = render_expr(expr_ir, names)
        if nm in k.row_outputs:
            expr = arg.index.render_without("col")
            guard = ""
        else:
            expr = arg.index.render()
            guard = ", mask=cmask" if masked else ""
        lines.append(f"{indent}tl.store(out_ptr{i} + ({expr}), "
                     f"{store_value(expr_ir, value, arg.dtype, 'BLOCK_COL')}{guard})"
                     f"  # {nm}")
    return lines


def _masked(expr_src: str, kind: str, masked: bool) -> str:
    """Neutralise lanes outside the row before they reach a reduction.

    ``other=0.0`` on the load is not enough: zero is not the identity of a max,
    so a row of negative values would come back as zero.
    """
    if not masked:
        return expr_src
    return f"tl.where(cmask, {expr_src}, {IDENTITY[kind]})"


# --------------------------------------------------------------------------

def _emit_persistent(k: ReductionKernel, cfg: Config) -> tuple[str, LaunchSpec]:
    n = k.reduce_numel
    block = max(next_pow2(n), 16)
    masked = block != n
    names = [f"v{i}" for i in range(len(k.inputs))]

    lines = _header(k, "persistent")
    lines.append("    col = tl.arange(0, BLOCK_COL)")
    if masked:
        lines.append(f"    cmask = col < {n}")
    else:
        lines.append(f"    # {n} is exactly BLOCK_COL: no column mask needed")
    lines += _emit_loads(k, names, "    ", masked)

    for step in k.steps:
        if isinstance(step, Reduce):
            src = _masked(render_expr(step.expr, names), step.kind, masked)
            lines.append(f"    {step.name} = {REDUCE_FN[step.kind].format(src)}")
        else:
            lines.append(f"    {step.name} = {render_expr(step.expr, names)}")

    lines += _emit_stores(k, names, "    ", masked)
    spec = LaunchSpec(grid=k.n_rows, constants={"BLOCK_COL": block},
                      num_warps=warps_for_reduction(block))
    return "\n".join(lines), spec


# --------------------------------------------------------------------------

def _emit_streamed(k: ReductionKernel, cfg: Config) -> tuple[str, LaunchSpec]:
    n = k.reduce_numel
    block = min(next_pow2(min(cfg.max_persistent_row, 1024)), next_pow2(n))
    block = max(block, 16)
    masked = n % block != 0
    names = [f"v{i}" for i in range(len(k.inputs))]

    passes = plan_passes(k)
    by_reduce = {r.name: p for p in passes for r in p.reduces}
    scalar_names = {b.name for b in scalar_binds(k)}
    column_slots = {i for i, a in enumerate(k.inputs) if a.index.varies_along("col")}
    scalar_slots = set(range(len(k.inputs))) - column_slots

    lines = _header(k, "streamed")
    lines.append(f"    # row does not fit registers ({n} > {cfg.max_persistent_row}), "
                 f"so each reduction streams it")
    lines += _emit_loads(k, names, "    ", False, only=scalar_slots)

    emitted: set[int] = set()

    def pass_slots(p):
        """Operands this pass actually reads. Emitting the rest would move
        bytes the pass never looks at, once per chunk."""
        used: set[int] = set()
        for b in p.binds:
            used |= expr_loads(b.expr)
        for r in p.reduces:
            used |= expr_loads(r.expr)
        if p.online_softmax is not None:
            used |= expr_loads(p.online_softmax.value)
        for nm in p.stores:
            used |= expr_loads(k.out_expr[nm])
        return used & column_slots

    def emit_pass(p, index):
        out = []
        online = p.online_softmax
        if online is not None:
            out.append(f"    # online softmax: max and sum(exp(x - max)) in one pass")
            out.append(f"    _m = {INIT['max']}")
            out.append(f"    _l = {INIT['sum']}")
        else:
            for r in p.reduces:
                out.append(f"    _acc_{r.name} = {INIT[r.kind]}")
        out.append(f"    for _base in range(0, {n}, BLOCK_COL):")
        out.append("        col = _base + tl.arange(0, BLOCK_COL)")
        if masked:
            out.append(f"        cmask = col < {n}")
        out += _emit_loads(k, names, "        ", masked, only=pass_slots(p))
        for b in p.binds:
            out.append(f"        {b.name} = {render_expr(b.expr, names)}")
        if online is not None:
            src = _masked(render_expr(online.value, names), "max", masked)
            out.append(f"        _x = {src}")
            out.append("        _m_new = tl.maximum(_m, _x)")
            out.append('        _alpha = tl.where(_m == -float("inf"), 0.0, tl.exp(_m - _m_new))')
            rescaled = "tl.exp(_x - _m_new)"
            if masked:
                rescaled = f"tl.where(cmask, {rescaled}, 0.0)"
            out.append(f"        _l = _l * _alpha + {rescaled}")
            out.append("        _m = _m_new")
        else:
            for r in p.reduces:
                src = _masked(render_expr(r.expr, names), r.kind, masked)
                out.append(f"        _acc_{r.name} = "
                           + ACCUMULATE[r.kind].format(acc=f"_acc_{r.name}", x=src))
        for nm in p.stores:
            out += _emit_stores(k, names, "        ", masked, only={nm})
        if online is not None:
            out.append(f"    {online.max_step.name} = tl.max(_m, axis=0)")
            out.append(f"    _w = tl.where(_m == -float(\"inf\"), 0.0, "
                       f"tl.exp(_m - {online.max_step.name}))")
            out.append(f"    {online.sum_step.name} = tl.sum(_l * _w, axis=0)")
        else:
            for r in p.reduces:
                out.append(f"    {r.name} = {REDUCE_FN[r.kind].format(f'_acc_{r.name}')}")
        return out

    for step in k.steps:
        if isinstance(step, Reduce):
            p = by_reduce[step.name]
            if id(p) in emitted:
                continue
            emitted.add(id(p))
            lines += emit_pass(p, len(emitted))
        elif step.name in scalar_names:
            lines.append(f"    {step.name} = {render_expr(step.expr, names)}")

    for p in passes:
        if p.stores and not p.reduces:
            lines += emit_pass(p, len(emitted))

    row_stores = [a.value.name for a in k.outputs if a.value.name in k.row_outputs]
    if row_stores:
        lines += _emit_stores(k, names, "    ", False, only=set(row_stores))

    spec = LaunchSpec(grid=k.n_rows, constants={"BLOCK_COL": block},
                      num_warps=warps_for_reduction(block))
    return "\n".join(lines), spec
