"""How a reduction kernel's steps map onto passes over the row.

A persistent kernel has one pass: load the row, run every step, store. When
the row is too large to hold, each reduction needs its own streaming pass over
memory, plus a final pass to compute and store the element-space outputs. This
module works out that partition, and both backends consume it, so the Triton
kernel and the reference implementation stream the row the same way.

It also spots the one pattern where two streaming passes collapse into one:
online softmax.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ir.scalar import Call, Cast, Expr, Load, Ref
from ..kernels import Bind, Reduce, ReductionKernel


@dataclass
class Pass:
    """One traversal of the row.

    ``binds`` are the steps that have to be recomputed inside this pass
    because they depend on the column index. ``reduces`` are the accumulators
    it updates; there are two only for a fused online-softmax pass.
    ``stores`` are the element-space outputs written during it.
    """

    binds: list[Bind] = field(default_factory=list)
    reduces: list[Reduce] = field(default_factory=list)
    stores: list[str] = field(default_factory=list)
    #: set for the fused max-then-sum-of-exp pass
    online_softmax: "OnlineSoftmax | None" = None


@dataclass
class OnlineSoftmax:
    """A max reduction and a sum of ``exp(x - max)`` over the same row.

    Streaming them separately costs two passes over memory. The running
    recurrence

        m' = max(m, max(chunk))
        l' = l * exp(m - m') + sum(exp(chunk - m'))

    computes both in one, and is exactly as stable as the two-pass form: every
    exponential still has a non-positive argument, so nothing overflows. This
    is the identity flash attention is built on. It only pays when the row is
    streamed; a persistent kernel already holds the row and reduces twice over
    registers for free.
    """

    max_step: Reduce
    sum_step: Reduce
    #: the expression being maxed, which the sum exponentiates a shift of
    value: Expr


def _refs(e: Expr) -> set[str]:
    from ..ir.scalar import refs

    return refs(e)


def dependencies(steps: list) -> dict[str, set[str]]:
    """name -> the steps it depends on, stopping at reductions.

    A reduction's result is a scalar that survives across passes, so nothing
    behind it needs recomputing. Recursing through one would drag the whole
    prologue into every later pass: the layer norm epilogue would recompute
    the squared deviations it already summed, in a loop, for nothing.
    """
    by_name = {s.name: s for s in steps}
    memo: dict[str, set[str]] = {}

    def go(name: str) -> set[str]:
        if name in memo:
            return memo[name]
        memo[name] = set()
        out: set[str] = set()
        step = by_name.get(name)
        if step is not None:
            for r in _refs(step.expr):
                out.add(r)
                if not isinstance(by_name.get(r), Reduce):
                    out |= go(r)
        memo[name] = out
        return out

    return {s.name: go(s.name) for s in steps}


def varies_by_column(kernel: ReductionKernel) -> set[str]:
    """Steps whose value is a vector over the column axis.

    A reduction result is a scalar, and anything derived only from scalars
    stays one. Those can be hoisted out of a streaming loop; the rest cannot.
    """
    column_slots = {i for i, a in enumerate(kernel.inputs) if a.index.varies_along("col")}
    vector: set[str] = set()
    for s in kernel.steps:
        if isinstance(s, Reduce):
            continue  # reduces collapse the column axis
        loads = {sub.slot for sub in _walk(s.expr) if isinstance(sub, Load)}
        refs = _refs(s.expr)
        if loads & column_slots or refs & vector:
            vector.add(s.name)
    return vector


def _walk(e: Expr):
    from ..ir.scalar import walk

    return walk(e)


def detect_online_softmax(kernel: ReductionKernel) -> OnlineSoftmax | None:
    """Find a max reduce whose only consumer chain is ``sum(exp(x - max))``."""
    reduces = kernel.reduces
    if len(reduces) != 2:
        return None
    first, second = reduces
    if first.kind != "max" or second.kind != "sum":
        return None
    by_name = {s.name: s for s in kernel.steps}

    def resolve(e: Expr) -> Expr:
        if isinstance(e, Ref):
            step = by_name.get(e.name)
            if isinstance(step, Bind):
                return resolve(step.expr)
        return e

    body = resolve(second.expr)
    if not (isinstance(body, Call) and body.fn == "exp"):
        return None
    inner = resolve(body.args[0])
    if not (isinstance(inner, Call) and inner.fn == "sub"):
        return None
    shifted, subtracted = resolve(inner.args[0]), inner.args[1]
    if not (isinstance(subtracted, Ref) and subtracted.name == first.name):
        return None
    if shifted != resolve(first.expr):
        return None
    return OnlineSoftmax(first, second, first.expr)


def plan_passes(kernel: ReductionKernel) -> list[Pass]:
    """Partition the steps into passes over the row.

    A persistent kernel gets exactly one pass. A streamed kernel gets one per
    reduction (or one for a fused online-softmax pair) plus a final pass for
    the element-space stores.
    """
    if not kernel.two_pass:
        p = Pass(
            binds=[s for s in kernel.steps if isinstance(s, Bind)],
            reduces=kernel.reduces,
            stores=[a.value.name for a in kernel.outputs
                    if a.value.name not in kernel.row_outputs],
        )
        return [p]

    deps = dependencies(kernel.steps)
    vector = varies_by_column(kernel)
    by_name = {s.name: s for s in kernel.steps}
    online = detect_online_softmax(kernel)

    passes: list[Pass] = []
    handled: set[str] = set()
    reduces = kernel.reduces
    i = 0
    while i < len(reduces):
        r = reduces[i]
        if online is not None and r is online.max_step:
            # Only what the maxed expression needs. The steps between the two
            # reductions -- the subtract and the exponential -- are subsumed
            # by the recurrence, and evaluating them here would reference the
            # max before it exists.
            needed = deps[online.max_step.name]
            binds = [by_name[n] for n in _ordered(kernel.steps, needed)
                     if isinstance(by_name[n], Bind) and n in vector]
            passes.append(Pass(binds=binds,
                               reduces=[online.max_step, online.sum_step],
                               online_softmax=online))
            handled |= {online.max_step.name, online.sum_step.name}
            i += 2
            continue
        needed = deps[r.name] | {r.name}
        binds = [by_name[n] for n in _ordered(kernel.steps, needed)
                 if isinstance(by_name[n], Bind) and n in vector]
        passes.append(Pass(binds=binds, reduces=[r]))
        handled.add(r.name)
        i += 1

    element_stores = [a.value.name for a in kernel.outputs
                      if a.value.name not in kernel.row_outputs]
    if element_stores:
        needed: set[str] = set()
        for name in element_stores:
            for ref in _refs(kernel.out_expr[name]):
                needed.add(ref)
                needed |= deps.get(ref, set())
        binds = [by_name[n] for n in _ordered(kernel.steps, needed)
                 if isinstance(by_name[n], Bind) and n in vector]
        passes.append(Pass(binds=binds, stores=element_stores))
    return passes


def scalar_binds(kernel: ReductionKernel) -> list[Bind]:
    """Steps that are scalars and so live outside any streaming loop."""
    vector = varies_by_column(kernel)
    return [s for s in kernel.steps if isinstance(s, Bind) and s.name not in vector]


def _ordered(steps: list, names: set[str]) -> list[str]:
    return [s.name for s in steps if s.name in names]
