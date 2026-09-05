"""Scalar expression IR.

Fused kernels are described as a DAG of scalar expressions over loaded values.
Two backends consume the same DAG:

  * ``mlc.codegen.triton_backend`` renders it to Triton source.
  * ``mlc.codegen.torch_backend`` evaluates it with broadcasting torch ops.

Having both means the operator lowering tables (which operand goes where, how
``alpha`` is folded, what the output dtype is) are testable on any machine,
including one without a GPU. Only the Triton mechanics -- masking, block sizes,
program ids -- need real hardware to exercise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


class Expr:
    """Base class for scalar expressions. Instances are hashable and immutable
    so they can be used as CSE keys."""

    __slots__ = ()


@dataclass(frozen=True)
class Load(Expr):
    """Read of the kernel input in slot ``slot``."""

    slot: int


@dataclass(frozen=True)
class Ref(Expr):
    """Reference to a named temporary already bound in the kernel body.

    Used for reduction results (``Ref("sum_0")``) and for values that the
    scheduler decided to materialise once rather than recompute.
    """

    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: Any
    dtype: torch.dtype | None = None


@dataclass(frozen=True)
class Call(Expr):
    fn: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Cast(Expr):
    x: Expr
    dtype: torch.dtype


def cast(x: Expr, dtype: torch.dtype) -> Expr:
    """Cast, folded when the operand is a literal.

    A backend cannot always emit a conversion on a bare constant: in Triton a
    python literal is a ``constexpr``, which has no ``.to``. Folding the dtype
    into the constant avoids ever generating one.
    """
    if isinstance(x, Const):
        return Const(x.value, dtype)
    return Cast(x, dtype)


# --------------------------------------------------------------------------
# Function table
# --------------------------------------------------------------------------
# ``arity`` is checked at build time so a bad lowering rule fails loudly in the
# frontend rather than producing Triton source that does not compile.
# ``bool_result`` marks comparisons, whose output dtype is torch.bool
# regardless of operand dtype.

@dataclass(frozen=True)
class FnSpec:
    arity: int
    bool_result: bool = False
    #: True when the function is only defined for floating point operands.
    float_only: bool = False


FUNCTIONS: dict[str, FnSpec] = {
    # unary
    "neg": FnSpec(1),
    "abs": FnSpec(1),
    "reciprocal": FnSpec(1, float_only=True),
    "exp": FnSpec(1, float_only=True),
    "log": FnSpec(1, float_only=True),
    "sqrt": FnSpec(1, float_only=True),
    "rsqrt": FnSpec(1, float_only=True),
    "sin": FnSpec(1, float_only=True),
    "cos": FnSpec(1, float_only=True),
    "tanh": FnSpec(1, float_only=True),
    "erf": FnSpec(1, float_only=True),
    "sigmoid": FnSpec(1, float_only=True),
    "floor": FnSpec(1),
    "logical_not": FnSpec(1, bool_result=True),
    # binary
    "add": FnSpec(2),
    "sub": FnSpec(2),
    "mul": FnSpec(2),
    "div": FnSpec(2),
    "pow": FnSpec(2),
    "maximum": FnSpec(2),
    "minimum": FnSpec(2),
    "gt": FnSpec(2, bool_result=True),
    "lt": FnSpec(2, bool_result=True),
    "ge": FnSpec(2, bool_result=True),
    "le": FnSpec(2, bool_result=True),
    "eq": FnSpec(2, bool_result=True),
    "ne": FnSpec(2, bool_result=True),
    "logical_and": FnSpec(2, bool_result=True),
    "logical_or": FnSpec(2, bool_result=True),
    # ternary
    "where": FnSpec(3),
}


def call(fn: str, *args: Expr) -> Expr:
    spec = FUNCTIONS.get(fn)
    if spec is None:
        raise KeyError(f"unknown scalar function {fn!r}")
    if len(args) != spec.arity:
        raise ValueError(f"{fn} takes {spec.arity} args, got {len(args)}")
    for a in args:
        if not isinstance(a, Expr):
            raise TypeError(f"{fn} arg is {type(a).__name__}, expected Expr")
    return _fold(fn, args) or Call(fn, tuple(args))


def _is(e: Expr, v) -> bool:
    return isinstance(e, Const) and not isinstance(e.value, bool) and e.value == v


def _fold(fn: str, args: tuple[Expr, ...]) -> Expr | None:
    """Exact algebraic identities, applied as expressions are built.

    Only the ones that hold for every float, including NaN and infinity: x*1,
    x+0, x-0, x/1. Not x*0, which is NaN for NaN x. Small, but they matter --
    decomposing addmm produces ``mm * 1 + bias * 1`` and every one of those
    multiplies would otherwise reach the generated kernel.
    """
    if fn in ("mul", "div") and _is(args[1], 1):
        return args[0]
    if fn == "mul" and _is(args[0], 1):
        return args[1]
    if fn in ("add", "sub") and _is(args[1], 0):
        return args[0]
    if fn == "add" and _is(args[0], 0):
        return args[1]
    # constant folding, so a chain of scalar-only arithmetic collapses
    if all(isinstance(a, Const) for a in args):
        folded = _eval_const(fn, [a.value for a in args])
        if folded is not None:
            return Const(folded)
    return None


def _eval_const(fn: str, vals: list):
    import math as _m

    try:
        if fn == "add":
            return vals[0] + vals[1]
        if fn == "sub":
            return vals[0] - vals[1]
        if fn == "mul":
            return vals[0] * vals[1]
        if fn == "div":
            return vals[0] / vals[1]
        if fn == "neg":
            return -vals[0]
        if fn == "sqrt":
            return _m.sqrt(vals[0])
        if fn == "exp":
            return _m.exp(vals[0])
    except (ZeroDivisionError, ValueError, OverflowError):
        return None
    return None


def walk(e: Expr):
    """Yield every subexpression of ``e``, children before parents."""
    if isinstance(e, Call):
        for a in e.args:
            yield from walk(a)
    elif isinstance(e, Cast):
        yield from walk(e.x)
    yield e


def loads(e: Expr) -> set[int]:
    return {sub.slot for sub in walk(e) if isinstance(sub, Load)}


def refs(e: Expr) -> set[str]:
    return {sub.name for sub in walk(e) if isinstance(sub, Ref)}


def substitute(e: Expr, mapping: dict[Expr, Expr]) -> Expr:
    """Rewrite ``e``, replacing any subexpression found in ``mapping``."""
    if e in mapping:
        return mapping[e]
    if isinstance(e, Call):
        return Call(e.fn, tuple(substitute(a, mapping) for a in e.args))
    if isinstance(e, Cast):
        return Cast(substitute(e.x, mapping), e.dtype)
    return e


def size(e: Expr) -> int:
    """Number of nodes in the expression tree, counting shared subtrees once."""
    return len(set(walk(e)))
