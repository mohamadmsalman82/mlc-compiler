"""Build a ReductionKernel from a fused group.

The frame is a (row, column) split of the element space: kept dimensions
become the row axis, reduced dimensions the column axis. Every operand is
permuted into that order and turned into an index map over ``row`` and
``col``, at which point a value that does not vary along ``col`` -- a
reduction result, a per-row statistic -- simply has no column term, and the
kernel treats it as a scalar without needing to know why.

Composite reductions expand here rather than in the frontend, because their
expansion depends on the frame. ``mean`` becomes a sum and a divide.
``var`` becomes two reductions over the same resident row: the mean, then the
sum of squared deviations from it. That is the textbook two-pass formula,
which is numerically exact, and it costs nothing extra because the second
pass is over registers, not memory. The one-pass alternative -- accumulating
sum and sum-of-squares and subtracting -- catastrophically cancels when the
mean is large relative to the variance, which is exactly the regime a
normalised activation lives in.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from ..codegen.index import IndexMap, build_index
from ..config import Config
from ..ir.graph import Graph, Node, Value
from ..ir.ops import OpClass, _S, lookup, op_class
from ..ir.scalar import Const, Expr, Load, Ref, call
from ..ir.types import Layout
from ..kernels import Bind, KernelArg, Reduce, ReductionKernel
from ..passes.reduction_fusion import (ELEMENT, ROW, Frame, _group_frame, classify,
                                       frame_axes, storable)
from ..passes.scheduler import source_value


class ReductionLoweringError(Exception):
    pass


def frame_layout(v: Value, target: tuple[int, ...], frame: Frame) -> tuple[list, Layout]:
    """Axes and layout to read ``v`` by, given the shape it broadcasts to.

    ``target`` is the output shape of the node reading ``v``. Three cases:
    the target lives in the element shape and is permuted into (row, col)
    order; it lives in the row shape and gets the reduced axes appended with
    stride 0; or it is a row-major reshape of one of those, in which case its
    own leading dimensions become the row axis.
    """
    target = tuple(target)
    frame_split = [("row", frame.row), ("col", frame.col)]

    if _broadcasts(target, frame.element):
        # The ordinary case. Always indexed by the frame's own split, even
        # when the target would also match the reshaped pattern: the layout is
        # in element order here, and the two splits have to agree.
        lay = v.layout if tuple(v.shape) == frame.element else v.layout.expand(frame.element)
        return frame_split, lay.permute(frame.order)

    axes = frame_axes(target, frame)
    row_shape = axes[0][1]
    if row_shape != frame.row:
        # Reshaped element shape: index it in its own frame, which covers the
        # same elements in the same order.
        lay = v.layout if tuple(v.shape) == target else v.layout.expand(target)
        return axes, lay

    # Row-shaped target: the reduced dimensions are gone, so append them back
    # with stride 0 so the value is constant along the column axis.
    lay = v.layout if tuple(v.shape) == frame.row else v.layout.expand(frame.row)
    return frame_split, Layout(frame.row + frame.col,
                               lay.strides + (0,) * len(frame.col), lay.offset)


def _broadcasts(shape, to) -> bool:
    from ..ir.shapes import ShapeError, broadcast_shapes

    try:
        return broadcast_shapes([tuple(shape), tuple(to)]) == tuple(to)
    except ShapeError:
        return False


class _Builder:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.inputs: list[KernelArg] = []
        self._slot: dict[tuple, int] = {}
        self.steps: list[Any] = []
        self.env: dict[int, tuple[Expr, IndexMap]] = {}
        self._n = 0
        #: (kind, expr) -> Ref, so a reduction computed twice is emitted once
        self._reduce_cache: dict[tuple, Expr] = {}
        #: expr -> Ref, common subexpression elimination over the whole body
        self._bind_cache: dict[Expr, Expr] = {}

    def fresh(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n - 1}"

    def axes(self):
        return [("row", self.frame.row), ("col", self.frame.col)]

    def load(self, v: Value, imap: IndexMap) -> Expr:
        key = (id(v.buffer), v.dtype, imap)
        slot = self._slot.get(key)
        if slot is None:
            slot = len(self.inputs)
            self._slot[key] = slot
            self.inputs.append(KernelArg(v, imap))
        return Load(slot)

    def operand(self, v: Value, target: tuple[int, ...]) -> Expr:
        axes, lay = frame_layout(v, target, self.frame)
        imap = build_index(axes, lay)
        src = source_value(v)
        live = self.env.get(id(src))
        if live is not None and live[1] == imap:
            return live[0]
        return self.load(v, imap)

    def bind(self, expr: Expr, prefix: str = "t") -> Expr:
        """Bind an expression to a temp, reusing an identical earlier one.

        CSE matters more here than in a pointwise kernel. Expanding ``var``
        produces its own mean and deviation, and the decomposed graph computes
        the same mean again through a separate sum; without this the layer norm
        kernel would carry two copies of both.
        """
        if isinstance(expr, (Load, Ref, Const)):
            return expr
        cached = self._bind_cache.get(expr)
        if cached is not None:
            return cached
        name = self.fresh(prefix)
        self.steps.append(Bind(name, expr))
        ref = Ref(name)
        self._bind_cache[expr] = ref
        return ref

    def reduce(self, kind: str, expr: Expr) -> Expr:
        key = (kind, expr)
        cached = self._reduce_cache.get(key)
        if cached is not None:
            return cached
        name = self.fresh("r")
        self.steps.append(Reduce(name, kind, expr))
        ref = Ref(name)
        self._reduce_cache[key] = ref
        return ref


def _correction(node: Node) -> float:
    kw = dict(node.kwargs)
    if "correction" in kw:
        return float(kw["correction"] or 0)
    for a in node.args[1:]:
        if isinstance(a, (int, float)) and not isinstance(a, bool):
            return float(a)
    return 0.0


def _expand_reduction(b: _Builder, node: Node, kind: str, src: Expr) -> list[Expr]:
    """Lower one reduction node to steps, returning one expression per output."""
    n = float(b.frame.n_cols)
    if kind in ("sum", "max", "min", "prod"):
        return [b.reduce(kind, src)]
    if kind == "mean":
        total = b.reduce("sum", src)
        return [b.bind(call("div", total, Const(n)))]
    if kind in ("var", "var_mean"):
        total = b.reduce("sum", src)
        mean = b.bind(call("div", total, Const(n)))
        dev = b.bind(call("sub", src, mean))
        sq = b.bind(call("mul", dev, dev))
        ss = b.reduce("sum", sq)
        denom = n - _correction(node)
        if denom <= 0:
            raise ReductionLoweringError(f"{node.op}: correction {_correction(node)} >= n={n}")
        var = b.bind(call("div", ss, Const(denom)))
        return [var, mean] if kind == "var_mean" else [var]
    raise ReductionLoweringError(f"no lowering for reduction kind {kind!r}")


def build_reduction(graph: Graph, group, stores: set[int], name: str,
                    cfg: Config) -> ReductionKernel:
    frame = _group_frame(group.nodes)
    if frame is None:
        raise ReductionLoweringError(
            "reduction group has no single frame: "
            + ", ".join(n.op for n in group.nodes)
        )
    b = _Builder(frame)

    for node in group.nodes:
        cls = op_class(node.op)
        out0 = node.outputs[0]
        if cls is OpClass.REDUCTION:
            spec = lookup(node.op)
            kind, _, _ = spec.reduce(list(node.args), dict(node.kwargs))
            src_val = node.args[0]
            src = b.operand(src_val, tuple(src_val.shape))
            results = _expand_reduction(b, node, kind, src)
            if len(results) < len(node.outputs):
                raise ReductionLoweringError(
                    f"{node.op} has {len(node.outputs)} outputs, lowering gave {len(results)}"
                )
            for o, expr in zip(node.outputs, results):
                b.env[id(o)] = (expr, _out_index(o, frame, b))
            continue

        if cls is not OpClass.POINTWISE:
            raise ReductionLoweringError(f"{node.op} cannot live in a reduction kernel")

        target = tuple(out0.shape)
        arg_exprs: list[Expr] = []
        for a in node.args:
            if isinstance(a, Value):
                arg_exprs.append(b.operand(a, target))
            elif isinstance(a, (bool, int, float)):
                arg_exprs.append(_S(a))
            else:
                arg_exprs.append(a)
        spec = lookup(node.op)
        if spec is None or spec.lower is None:
            raise ReductionLoweringError(f"no lowering rule for {node.op}")
        expr = b.bind(spec.lower(list(node.args), arg_exprs, dict(node.kwargs)))
        b.env[id(out0)] = (expr, _out_index(out0, frame, b))

    outputs: list[KernelArg] = []
    out_expr: dict[str, Expr] = {}
    row_outputs: set[str] = set()
    for node in group.nodes:
        if id(node) not in group.owned:
            continue
        for o in node.outputs:
            if id(o) not in stores:
                continue
            if not storable(o.shape, frame):
                raise ReductionLoweringError(
                    f"{node.op} result {o.name}{list(o.shape)} is broadcast inside its "
                    "kernel but escapes it; the fusion pass should not have grouped it"
                )
            where = classify(o.shape, frame)
            if where is ROW:
                imap = build_index([("row", frame.row)], Layout.contiguous(frame.row))
                row_outputs.add(o.name)
            else:
                axes, lay = frame_layout(o, tuple(o.shape), frame)
                imap = build_index(axes, lay)
            outputs.append(KernelArg(o, imap))
            out_expr[o.name] = b.env[id(o)][0]

    inputs, steps = _prune(b.inputs, b.steps, out_expr)
    two_pass = frame.n_cols > cfg.max_persistent_row
    return ReductionKernel(
        name=name,
        row_space=frame.row,
        reduce_space=frame.col,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        out_expr=out_expr,
        row_outputs=row_outputs,
        nodes=list(group.nodes),
        two_pass=two_pass,
    )


def _out_index(v: Value, frame: Frame, b: _Builder) -> IndexMap:
    """Index map a value produced inside the kernel is live at.

    Used to decide whether a later read can reuse the register instead of
    loading, so it is built exactly the way an operand read would be.
    """
    axes, lay = frame_layout(v, tuple(v.shape), frame)
    if classify(v.shape, frame) is ROW and not _broadcasts(v.shape, frame.element):
        # keepdim=False result: constant along the column axis
        base = v.layout if tuple(v.shape) == frame.row else v.layout.expand(frame.row)
        lay = Layout(frame.row + frame.col,
                     base.strides + (0,) * len(frame.col), base.offset)
        axes = [("row", frame.row), ("col", frame.col)]
    return build_index(axes, lay)


def _prune(inputs: list[KernelArg], steps: list, out_expr: dict[str, Expr]):
    """Drop operands nothing reads, and renumber the rest."""
    from ..ir.scalar import loads as expr_loads
    from ..ir.scalar import substitute

    used: set[int] = set()
    for s in steps:
        used |= expr_loads(s.expr)
    for e in out_expr.values():
        used |= expr_loads(e)
    if len(used) == len(inputs):
        return inputs, steps
    keep = sorted(used)
    mapping = {Load(old): Load(new) for new, old in enumerate(keep)}
    new_steps = []
    for s in steps:
        s.expr = substitute(s.expr, mapping)
        new_steps.append(s)
    for k in list(out_expr):
        out_expr[k] = substitute(out_expr[k], mapping)
    return [inputs[i] for i in keep], new_steps
