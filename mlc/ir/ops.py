"""Operator registry.

Every op the compiler understands has an :class:`OpSpec` giving

  * its **class** (pointwise / reduction / view / matmul / opaque), which is
    what the fusion passes key off,
  * a **shape and dtype rule**, so the compiler propagates types itself
    rather than trusting the exporter, and
  * for pointwise ops a **lowering rule** producing a scalar expression, and
    for view ops a **layout rule** producing a new view over the same buffer.

Ops with no entry are legal: they become opaque nodes that run as a plain
torch call and act as fusion barriers. Coverage is a performance question,
not a correctness one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import torch

from .graph import Value
from .scalar import Cast, Const, Expr, call
from .types import Layout


class OpClass(str, Enum):
    POINTWISE = "pointwise"
    REDUCTION = "reduction"
    VIEW = "view"
    MATMUL = "matmul"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class ReduceSpec:
    """How a reduction accumulates.

    ``kind`` names the combining function. ``associative`` records whether
    partial results may be combined in any order -- the legality condition for
    splitting a reduction across thread blocks. Every reduction we support is
    associative in exact arithmetic; float addition is not associative in the
    strict sense, but reassociating it is standard practice and we bound the
    resulting error by testing against torch at a fixed tolerance.
    """

    kind: str  # "sum" | "max" | "min" | "prod"
    associative: bool = True
    identity: float = 0.0


REDUCE_KINDS = {
    "sum": ReduceSpec("sum", True, 0.0),
    "max": ReduceSpec("max", True, -math.inf),
    "min": ReduceSpec("min", True, math.inf),
    "prod": ReduceSpec("prod", True, 1.0),
}


@dataclass
class OpSpec:
    name: str
    cls: OpClass
    #: pointwise: (args, exprs, kwargs) -> Expr
    lower: Callable[[list, list[Expr], dict], Expr] | None = None
    #: view: (in_layout, args, kwargs) -> Layout
    view: Callable[[Layout, list, dict], Layout] | None = None
    #: reduction: (args, kwargs) -> (kind, dims, keepdim)
    reduce: Callable[[list, dict], tuple[str, tuple[int, ...], bool]] | None = None
    #: overrides the default dtype rule for this op
    dtype_rule: Callable[[list, dict], torch.dtype] | None = None
    #: number of results; >1 for ops like var_mean / max.dim / split
    n_outputs: int = 1


REGISTRY: dict[str, OpSpec] = {}


def _reg(spec: OpSpec) -> OpSpec:
    REGISTRY[spec.name] = spec
    return spec


def lookup(op: str) -> OpSpec | None:
    return REGISTRY.get(op)


def op_class(op: str) -> OpClass:
    spec = REGISTRY.get(op)
    return spec.cls if spec else OpClass.OPAQUE


# ==========================================================================
# Pointwise
# ==========================================================================

def _S(x: Any) -> Expr:
    """Coerce a raw python scalar into a Const expression."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return Const(x, torch.bool)
    if isinstance(x, int):
        return Const(x, torch.int64)
    if isinstance(x, float):
        return Const(float(x), None)
    raise TypeError(f"cannot use {type(x).__name__} as a scalar operand")


def pointwise(*names: str, arity: int | None = None):
    """Register ``fn`` as the lowering rule for each of ``names``."""

    def deco(fn):
        for n in names:
            _reg(OpSpec(n, OpClass.POINTWISE, lower=fn))
        return fn

    return deco


def simple(fn_name: str, *names: str):
    """Register ops that map one-to-one onto a scalar function."""

    def build(args, exprs, kwargs, _f=fn_name):
        return call(_f, *exprs[: _ARITY[_f]])

    for n in names:
        _reg(OpSpec(n, OpClass.POINTWISE, lower=build))


from .scalar import FUNCTIONS as _FUNCS  # noqa: E402

_ARITY = {k: v.arity for k, v in _FUNCS.items()}

# -- arithmetic ------------------------------------------------------------

@pointwise("aten.add.Tensor", "aten.add.Scalar", "prims.add.default")
def _add(args, exprs, kwargs):
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
    rhs = exprs[1] if alpha == 1 else call("mul", exprs[1], _S(alpha))
    return call("add", exprs[0], rhs)


@pointwise("aten.sub.Tensor", "aten.sub.Scalar", "prims.sub.default")
def _sub(args, exprs, kwargs):
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
    rhs = exprs[1] if alpha == 1 else call("mul", exprs[1], _S(alpha))
    return call("sub", exprs[0], rhs)


@pointwise("aten.rsub.Scalar", "aten.rsub.Tensor")
def _rsub(args, exprs, kwargs):
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
    lhs = exprs[0] if alpha == 1 else call("mul", exprs[0], _S(alpha))
    return call("sub", exprs[1], lhs)


simple("mul", "aten.mul.Tensor", "aten.mul.Scalar", "prims.mul.default")
simple("div", "aten.div.Tensor", "aten.div.Scalar", "prims.div.default",
       "aten.true_divide.Tensor")
simple("neg", "aten.neg.default", "prims.neg.default")
simple("abs", "aten.abs.default", "prims.abs.default")
simple("reciprocal", "aten.reciprocal.default", "prims.reciprocal.default")
simple("exp", "aten.exp.default", "prims.exp.default")
simple("log", "aten.log.default", "prims.log.default")
simple("sqrt", "aten.sqrt.default", "prims.sqrt.default")
simple("rsqrt", "aten.rsqrt.default", "prims.rsqrt.default")
simple("sin", "aten.sin.default", "prims.sin.default")
simple("cos", "aten.cos.default", "prims.cos.default")
simple("tanh", "aten.tanh.default", "prims.tanh.default")
simple("erf", "aten.erf.default", "prims.erf.default")
simple("sigmoid", "aten.sigmoid.default")
simple("floor", "aten.floor.default", "prims.floor.default")
simple("maximum", "aten.maximum.default", "prims.maximum.default")
simple("minimum", "aten.minimum.default", "prims.minimum.default")
simple("pow", "aten.pow.Tensor_Tensor", "prims.pow.default")
simple("where", "aten.where.self", "prims.where.default")

_CMP = {
    "aten.gt.Tensor": "gt", "aten.gt.Scalar": "gt", "prims.gt.default": "gt",
    "aten.lt.Tensor": "lt", "aten.lt.Scalar": "lt", "prims.lt.default": "lt",
    "aten.ge.Tensor": "ge", "aten.ge.Scalar": "ge", "prims.ge.default": "ge",
    "aten.le.Tensor": "le", "aten.le.Scalar": "le", "prims.le.default": "le",
    "aten.eq.Tensor": "eq", "aten.eq.Scalar": "eq", "prims.eq.default": "eq",
    "aten.ne.Tensor": "ne", "aten.ne.Scalar": "ne", "prims.ne.default": "ne",
}
for _name, _fn in _CMP.items():
    _reg(OpSpec(_name, OpClass.POINTWISE,
                lower=(lambda a, e, k, _f=_fn: call(_f, e[0], e[1])),
                dtype_rule=lambda a, k: torch.bool))

_BOOL = {
    "aten.logical_and.default": "logical_and",
    "aten.logical_or.default": "logical_or",
    "aten.logical_not.default": "logical_not",
    "aten.bitwise_not.default": "logical_not",
}
for _name, _fn in _BOOL.items():
    _reg(OpSpec(_name, OpClass.POINTWISE,
                lower=(lambda a, e, k, _f=_fn: call(_f, *e[: _ARITY[_f]])),
                dtype_rule=lambda a, k: torch.bool))


@pointwise("aten.pow.Tensor_Scalar")
def _pow_scalar(args, exprs, kwargs):
    exponent = args[1]
    # Small integer powers become multiplies: cheaper and exact, and it is the
    # form the tanh-gelu decomposition produces (x ** 3).
    if isinstance(exponent, int) and 0 <= exponent <= 4:
        if exponent == 0:
            return Const(1.0)
        acc = exprs[0]
        for _ in range(exponent - 1):
            acc = call("mul", acc, exprs[0])
        return acc
    if exponent == 0.5:
        return call("sqrt", exprs[0])
    if exponent == -0.5:
        return call("rsqrt", exprs[0])
    if exponent == -1:
        return call("reciprocal", exprs[0])
    return call("pow", exprs[0], _S(exponent))


@pointwise("aten.relu.default")
def _relu(args, exprs, kwargs):
    return call("maximum", exprs[0], Const(0.0))


@pointwise("aten.gelu.default")
def _gelu(args, exprs, kwargs):
    """Kept for graphs where gelu survives decomposition."""
    approximate = kwargs.get("approximate", args[1] if len(args) > 1 else "none")
    x = exprs[0]
    if approximate == "tanh":
        # 0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        x3 = call("mul", call("mul", x, x), x)
        inner = call("mul", Const(0.7978845608028654),
                     call("add", x, call("mul", Const(0.044715), x3)))
        return call("mul", call("mul", Const(0.5), x),
                    call("add", Const(1.0), call("tanh", inner)))
    inner = call("mul", x, Const(0.7071067811865476))
    return call("mul", call("mul", Const(0.5), x),
                call("add", Const(1.0), call("erf", inner)))


@pointwise("aten.silu.default")
def _silu(args, exprs, kwargs):
    return call("mul", exprs[0], call("sigmoid", exprs[0]))


@pointwise("aten.clamp.default", "aten.clamp.Tensor")
def _clamp(args, exprs, kwargs):
    lo = kwargs.get("min", args[1] if len(args) > 1 else None)
    hi = kwargs.get("max", args[2] if len(args) > 2 else None)
    out = exprs[0]
    if lo is not None:
        out = call("maximum", out, exprs[1] if isinstance(lo, Expr) else _S(lo))
    if hi is not None:
        out = call("minimum", out, _S(hi))
    return out


@pointwise("aten.clamp_min.default")
def _clamp_min(args, exprs, kwargs):
    return call("maximum", exprs[0], _S(args[1]))


@pointwise("aten.clamp_max.default")
def _clamp_max(args, exprs, kwargs):
    return call("minimum", exprs[0], _S(args[1]))


@pointwise("aten.masked_fill.Scalar", "aten.masked_fill.Tensor")
def _masked_fill(args, exprs, kwargs):
    # where(mask, value, self)
    return call("where", exprs[1], _S(args[2]) if not isinstance(exprs[2], Expr) else exprs[2], exprs[0])


def _fill_rule(default: float):
    """Constant fills. Pointwise with no data dependence: the tensor operand
    supplies the shape only, and the unused load is pruned during lowering."""

    def build(args, exprs, kwargs, _d=default):
        fill = kwargs.get("fill_value")
        if fill is None:
            fill = next((a for a in args[1:] if isinstance(a, (int, float, bool))), None)
        return _S(_d if fill is None else fill)

    return build


for _n, _d in (("aten.full_like.default", 0.0), ("aten.zeros_like.default", 0.0),
               ("aten.ones_like.default", 1.0), ("aten.new_full.default", 0.0),
               ("aten.new_zeros.default", 0.0), ("aten.new_ones.default", 1.0)):
    _reg(OpSpec(_n, OpClass.POINTWISE, lower=_fill_rule(_d)))


# Copies. A clone is a pointwise identity; when the source layout already
# matches the destination the copy pass deletes it, and when it does not
# (the usual case: contiguify after a permute) it becomes a real kernel that
# fuses with whatever produced the source.
@pointwise("aten.clone.default", "prims.clone.default", "aten.contiguous.default",
           "aten.detach.default", "aten.alias.default", "aten.lift_fresh_copy.default")
def _identity(args, exprs, kwargs):
    return exprs[0]


#: Float types the backends keep in fp32 registers regardless of how they are
#: stored, so a cast between them and fp32 is a no-op in the kernel body.
_PROMOTED = (torch.float16, torch.bfloat16, torch.float32)


def _convert(args, exprs, kwargs):
    """Dtype conversion, dropped when it cannot change anything.

    Both backends load narrow floats into fp32 and cast back on store, so a
    cast to fp32 of a value that is already there is dead. Half-precision
    graphs are full of these: layer norm decomposes into an fp32 computation
    bracketed by conversions, and every one of them would otherwise reach the
    generated kernel.
    """
    dt = kwargs.get("dtype", args[1] if len(args) > 1 else None)
    if dt is None:
        return exprs[0]
    src = args[0].dtype if isinstance(args[0], Value) else None
    if src == dt:
        return exprs[0]
    if dt == torch.float32 and src in _PROMOTED:
        return exprs[0]
    return Cast(exprs[0], dt)


for _n in ("aten._to_copy.default", "prims.convert_element_type.default", "aten.to.dtype"):
    _reg(OpSpec(_n, OpClass.POINTWISE, lower=_convert,
                dtype_rule=lambda a, k: k.get("dtype", a[1] if len(a) > 1 else None)))


# ==========================================================================
# Views
# ==========================================================================

def view_op(*names: str):
    def deco(fn):
        for n in names:
            _reg(OpSpec(n, OpClass.VIEW, view=fn))
        return fn

    return deco


@view_op("aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
         "aten.view.dtype", "prims.view_of.default")
def _view(lay: Layout, args, kwargs):
    shape = list(args[1])
    # -1 resolves against the element count.
    if -1 in shape:
        known = math.prod(d for d in shape if d != -1)
        shape[shape.index(-1)] = lay.numel // max(known, 1)
    out = lay.reshape(tuple(shape))
    if out is None:
        raise NonViewable(f"view{tuple(shape)} needs a copy of a non-contiguous layout {lay}")
    return out


class NonViewable(Exception):
    """Raised when a nominal view op cannot be expressed as a restride.

    The capture pass turns these into an explicit materialising copy followed
    by the view, which keeps the IR's "views are free" invariant true.
    """


@view_op("aten.permute.default", "prims.transpose.default")
def _permute(lay: Layout, args, kwargs):
    return lay.permute(args[1])


@view_op("aten.transpose.int")
def _transpose(lay: Layout, args, kwargs):
    dims = list(range(lay.rank))
    a, b = args[1] % lay.rank, args[2] % lay.rank
    dims[a], dims[b] = dims[b], dims[a]
    return lay.permute(dims)


@view_op("aten.t.default")
def _t(lay: Layout, args, kwargs):
    return lay if lay.rank < 2 else lay.permute([1, 0])


@view_op("aten.expand.default", "aten.broadcast_to.default")
def _expand(lay: Layout, args, kwargs):
    return lay.expand(args[1])


@view_op("prims.broadcast_in_dim.default")
def _broadcast_in_dim(lay: Layout, args, kwargs):
    """prims form: place each input dim at position ``bcast_dims[i]`` of the
    output, filling the rest with size-1 (stride-0) dims."""
    shape, bdims = list(args[1]), list(args[2])
    strides = [0] * len(shape)
    for i, d in enumerate(bdims):
        strides[d] = lay.strides[i] if lay.shape[i] == shape[d] else 0
    return Layout(tuple(shape), tuple(strides), lay.offset)


@view_op("aten.unsqueeze.default")
def _unsqueeze(lay: Layout, args, kwargs):
    return lay.unsqueeze(args[1])


@view_op("aten.squeeze.dim", "aten.squeeze.dims", "aten.squeeze.default")
def _squeeze(lay: Layout, args, kwargs):
    if len(args) < 2:
        return lay.squeeze(None)
    dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
    return lay.squeeze(dims)


@view_op("aten.slice.Tensor", "prims.slice.default")
def _slice(lay: Layout, args, kwargs):
    dim = args[1] if len(args) > 1 else 0
    start = args[2] if len(args) > 2 and args[2] is not None else 0
    end = args[3] if len(args) > 3 and args[3] is not None else lay.shape[dim % lay.rank]
    step = args[4] if len(args) > 4 else 1
    end = min(end, lay.shape[dim % lay.rank])
    return lay.slice(dim, start, end, step)


@view_op("aten.select.int")
def _select(lay: Layout, args, kwargs):
    dim, idx = args[1] % lay.rank, args[2]
    if idx < 0:
        idx += lay.shape[dim]
    return lay.slice(dim, idx, idx + 1).squeeze([dim])


@view_op("aten.as_strided.default")
def _as_strided(lay: Layout, args, kwargs):
    shape, strides = tuple(args[1]), tuple(args[2])
    offset = args[3] if len(args) > 3 and args[3] is not None else lay.offset
    return Layout(shape, strides, offset)


@view_op("aten.split_with_sizes.default", "aten.split.Tensor", "aten.unbind.int")
def _split_unsupported(lay: Layout, args, kwargs):  # pragma: no cover
    raise NonViewable("multi-output splits are expanded during capture")


# ==========================================================================
# Reductions
# ==========================================================================

def _norm_dims(dims, rank) -> tuple[int, ...]:
    if dims is None:
        return tuple(range(rank))
    if isinstance(dims, int):
        dims = [dims]
    if len(dims) == 0:
        return tuple(range(rank))
    return tuple(sorted(d % rank for d in dims))


def reduction(name: str, kind: str, dim_arg: int = 1, keepdim_arg: int | None = 2,
              dtype_rule=None) -> None:
    def rule(args, kwargs, _k=kind, _d=dim_arg, _kd=keepdim_arg):
        rank = args[0].rank
        dims = kwargs.get("dim", args[_d] if len(args) > _d else None)
        keep = False
        if _kd is not None:
            keep = kwargs.get("keepdim", args[_kd] if len(args) > _kd else False)
        return _k, _norm_dims(dims, rank), bool(keep)

    _reg(OpSpec(name, OpClass.REDUCTION, reduce=rule, dtype_rule=dtype_rule))


reduction("aten.sum.dim_IntList", "sum")
reduction("aten.sum.default", "sum")
reduction("prims.sum.default", "sum", keepdim_arg=None)
reduction("aten.amax.default", "max")
reduction("prims.amax.default", "max", keepdim_arg=None)
reduction("aten.amin.default", "min")
reduction("prims.amin.default", "min", keepdim_arg=None)
reduction("aten.prod.dim_int", "prod")

#: Reductions that need more than a single accumulator. They get their own
#: multi-stage lowering in the reduction fusion pass.
COMPOSITE_REDUCTIONS = {
    "aten.mean.dim": "mean",
    "aten.mean.default": "mean",
    "prims.var.default": "var",
    "aten.var.correction": "var",
    "aten.var_mean.correction": "var_mean",
}


def _composite_rule(args, kwargs, name):
    rank = args[0].rank
    dims = kwargs.get("dim", args[1] if len(args) > 1 else None)
    keep = kwargs.get("keepdim", args[2] if len(args) > 2 and isinstance(args[2], bool) else False)
    return COMPOSITE_REDUCTIONS[name], _norm_dims(dims, rank), bool(keep)


for _n in COMPOSITE_REDUCTIONS:
    _reg(OpSpec(_n, OpClass.REDUCTION,
                reduce=(lambda a, k, _n=_n: _composite_rule(a, k, _n)),
                n_outputs=2 if _n.startswith("aten.var_mean") else 1))


# ==========================================================================
# Matmuls -- opaque to fusion, dispatched to cuBLAS at runtime
# ==========================================================================

for _n in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default",
           "aten.baddbmm.default", "aten.matmul.default", "aten._unsafe_index.Tensor"):
    _reg(OpSpec(_n, OpClass.MATMUL))


def is_pointwise(op: str) -> bool:
    return op_class(op) is OpClass.POINTWISE


def is_reduction(op: str) -> bool:
    return op_class(op) is OpClass.REDUCTION


def is_view(op: str) -> bool:
    return op_class(op) is OpClass.VIEW


# A materialising copy the compiler inserts itself, when a nominal view turns
# out to need real data movement. Pointwise, so it still fuses with neighbours.
_reg(OpSpec("mlc.copy", OpClass.POINTWISE, lower=lambda a, e, k: e[0]))
