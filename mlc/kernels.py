"""Kernel IR: what the fusion passes produce and the backends consume.

A :class:`Schedule` is an ordered list of kernels. Each kernel names the
values it reads and writes, so the memory planner can compute live ranges
without knowing anything about the kernel's internals, and each backend can
lower it without knowing anything about the graph it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from .codegen.index import IndexMap
from .ir.graph import Node, Value
from .ir.scalar import Expr
from .ir.types import Layout


@dataclass
class KernelArg:
    """One tensor operand, with the index map the kernel reads it by.

    The map is expressed in the kernel's own iteration variables, so
    broadcasting has already been folded away: a bias vector read across a
    batch carries a map that mentions only the feature dimension. Kernels
    receive the *buffer* base pointer; any view offset lives in the map's
    ``base``.
    """

    value: Value
    index: IndexMap

    @property
    def name(self) -> str:
        return self.value.name

    @property
    def buffer(self):
        return self.value.buffer

    @property
    def dtype(self) -> torch.dtype:
        return self.value.dtype


@dataclass
class Kernel:
    name: str

    def reads(self) -> list[Value]:
        raise NotImplementedError

    def writes(self) -> list[Value]:
        raise NotImplementedError

    def summary(self) -> str:
        return self.name


@dataclass
class PointwiseKernel(Kernel):
    """One kernel over a flat iteration space.

    ``body`` is an ordered list of ``(temp, expr)``. Each fused graph node
    contributes exactly one temp, and later exprs refer to earlier ones with
    ``Ref``. Nothing in ``body`` touches memory: that is the whole point.
    """

    space: tuple[int, ...]
    inputs: list[KernelArg]
    outputs: list[KernelArg]
    body: list[tuple[str, Expr]]
    #: output value name -> the expression to store, usually a Ref into body
    out_expr: dict[str, Expr]
    nodes: list[Node] = field(default_factory=list)

    @property
    def numel(self) -> int:
        return math.prod(self.space) if self.space else 1

    def reads(self) -> list[Value]:
        return [a.value for a in self.inputs]

    def writes(self) -> list[Value]:
        return [a.value for a in self.outputs]

    def summary(self) -> str:
        ops = ", ".join(n.op.split(".")[-2] for n in self.nodes)
        return (f"{self.name}: pointwise{list(self.space)} "
                f"{len(self.inputs)}in {len(self.outputs)}out [{ops}]")


@dataclass
class Bind:
    """One pointwise value computed inside a reduction kernel."""

    name: str
    expr: Expr


@dataclass
class Reduce:
    """One block-level reduction over the column axis."""

    name: str
    kind: str  # "sum" | "max" | "min" | "prod"
    expr: Expr


Step = Bind | Reduce


@dataclass
class ReductionKernel(Kernel):
    """A persistent kernel: one row per thread block.

    ``steps`` is a single ordered list rather than a prologue/reduce/epilogue
    split, because real reductions interleave. Softmax is
    ``bind, reduce(max), bind, bind, reduce(sum), bind``: the subtract and the
    exponential sit *between* the two reductions, and each reduce may read any
    earlier result. Layer norm is the same shape with two reduces sharing one
    load of the row.

    Everything a step computes lives in registers for the life of the kernel.
    A value that does not vary along the column axis -- a reduction result, a
    per-row bias -- is a scalar there, and Triton broadcasts it against the
    column vector for free. That is what makes both fusion directions work:
    producers feed the reduce without a store, and consumers read the result
    without a reload.
    """

    #: shape of the kept dimensions; one program per point
    row_space: tuple[int, ...]
    #: shape of the reduced dimensions, flattened into the column axis
    reduce_space: tuple[int, ...]
    inputs: list[KernelArg]
    outputs: list[KernelArg]
    steps: list[Step]
    out_expr: dict[str, Expr]
    #: names of outputs that live in row space, stored once per program
    row_outputs: set[str] = field(default_factory=set)
    nodes: list[Node] = field(default_factory=list)
    #: True when the row does not fit the register budget, so each reduce
    #: streams the row in chunks instead of holding it
    two_pass: bool = False

    @property
    def reduce_numel(self) -> int:
        return math.prod(self.reduce_space) if self.reduce_space else 1

    @property
    def n_rows(self) -> int:
        return math.prod(self.row_space) if self.row_space else 1

    @property
    def reduces(self) -> list["Reduce"]:
        return [s for s in self.steps if isinstance(s, Reduce)]

    def reads(self) -> list[Value]:
        return [a.value for a in self.inputs]

    def writes(self) -> list[Value]:
        return [a.value for a in self.outputs]

    def summary(self) -> str:
        ops = ", ".join(n.op.split(".")[-2] for n in self.nodes)
        kinds = "+".join(r.kind for r in self.reduces) or "none"
        mode = "streamed" if self.two_pass else "persistent"
        return (f"{self.name}: reduction {self.n_rows}x{self.reduce_numel} "
                f"{kinds} ({mode}) {len(self.inputs)}in {len(self.outputs)}out [{ops}]")


@dataclass
class ExternKernel(Kernel):
    """An op we do not generate code for: matmuls and anything opaque.

    Dispatched to the torch operator at runtime. These are fusion barriers,
    but they still participate in memory planning like any other kernel.
    """

    node: Node
    inputs: list[Value]
    outputs: list[Value]

    def reads(self) -> list[Value]:
        return list(self.inputs)

    def writes(self) -> list[Value]:
        return list(self.outputs)

    def summary(self) -> str:
        shapes = "x".join(str(list(v.shape)) for v in self.inputs)
        return f"{self.name}: extern {self.node.op} {shapes}"


@dataclass
class Schedule:
    """The compiled program: kernels in execution order over a graph."""

    graph: object
    kernels: list[Kernel] = field(default_factory=list)
    #: filled in by the memory planner
    arena_bytes: int = 0
    plan: object | None = None

    def __iter__(self):
        return iter(self.kernels)

    def __len__(self) -> int:
        return len(self.kernels)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in self.kernels:
            key = type(k).__name__.replace("Kernel", "").lower()
            out[key] = out.get(key, 0) + 1
        return out

    def format(self) -> str:
        lines = [f"schedule({len(self.kernels)} kernels, {self.counts()})"]
        for k in self.kernels:
            lines.append("  " + k.summary())
        if self.plan is not None:
            lines.append("  " + self.plan.summary())
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return self.format()
