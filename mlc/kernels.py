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
class ReductionKernel(Kernel):
    """A persistent kernel: one row per thread block.

    The kernel loads a whole row of ``reduce_numel`` elements, runs a sequence
    of ``stages`` over it -- each stage being a pointwise prologue followed by
    a block-level reduce -- and then evaluates an epilogue that may reference
    any stage result. Keeping the row resident is what lets producers and
    consumers fuse in without a second trip through HBM.
    """

    #: shape of the kept (non-reduced) dimensions; one program per point
    row_space: tuple[int, ...]
    #: shape of the reduced dimensions, flattened by the backend
    reduce_space: tuple[int, ...]
    inputs: list[KernelArg]
    outputs: list[KernelArg]
    #: prologue evaluated once per (row, column) element
    prologue: list[tuple[str, Expr]]
    #: ordered reduce stages: (result_name, kind, expr_over_row)
    stages: list["ReduceStage"]
    #: epilogue evaluated per element again, may use stage results via Ref
    epilogue: list[tuple[str, Expr]]
    out_expr: dict[str, Expr]
    #: outputs whose shape is row_space rather than the full element space
    row_outputs: set[str] = field(default_factory=set)
    nodes: list[Node] = field(default_factory=list)
    #: True when the row does not fit the register/SRAM budget and the kernel
    #: has to stream it twice instead of holding it.
    two_pass: bool = False

    @property
    def reduce_numel(self) -> int:
        return math.prod(self.reduce_space) if self.reduce_space else 1

    @property
    def n_rows(self) -> int:
        return math.prod(self.row_space) if self.row_space else 1

    def reads(self) -> list[Value]:
        return [a.value for a in self.inputs]

    def writes(self) -> list[Value]:
        return [a.value for a in self.outputs]

    def summary(self) -> str:
        ops = ", ".join(n.op.split(".")[-2] for n in self.nodes)
        kinds = "+".join(s.kind for s in self.stages)
        mode = "two-pass" if self.two_pass else "persistent"
        return (f"{self.name}: reduction {self.n_rows}x{self.reduce_numel} "
                f"{kinds} ({mode}) [{ops}]")


@dataclass
class ReduceStage:
    """One block-level reduction over the loaded row."""

    name: str
    kind: str  # "sum" | "max" | "min" | "prod"
    expr: Expr
    #: True when the stage may reference earlier stage results
    depends: tuple[str, ...] = ()


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
