"""Graph IR: SSA values over buffers, in topological order."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch

from .types import Buffer, Layout


@dataclass(eq=False)
class Value:
    """One SSA value: a named view (``layout``) into a ``buffer``."""

    name: str
    buffer: Buffer
    layout: Layout
    producer: "Node | None" = None
    index: int = 0

    @property
    def dtype(self) -> torch.dtype:
        return self.buffer.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.layout.shape

    @property
    def rank(self) -> int:
        return self.layout.rank

    @property
    def numel(self) -> int:
        return self.layout.numel

    def is_view_of(self, other: "Value") -> bool:
        return self.buffer is other.buffer

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        d = str(self.dtype).replace("torch.", "")
        return f"%{self.name}:{list(self.shape)}:{d}"

    __hash__ = object.__hash__


@dataclass(eq=False)
class Node:
    """One operation. ``op`` is the qualified aten/prims name, or one of the
    pseudo-ops ``placeholder`` / ``output``."""

    op: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    outputs: list[Value] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def out(self) -> Value:
        if len(self.outputs) != 1:
            raise ValueError(f"{self.op} has {len(self.outputs)} outputs")
        return self.outputs[0]

    def tensor_args(self) -> list[Value]:
        """The Value operands, in argument order, duplicates included."""
        found: list[Value] = []
        for a in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(a, Value):
                found.append(a)
            elif isinstance(a, (list, tuple)):
                found.extend(x for x in a if isinstance(x, Value))
        return found

    def unique_tensor_args(self) -> list[Value]:
        seen, out = set(), []
        for v in self.tensor_args():
            if id(v) not in seen:
                seen.add(id(v))
                out.append(v)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        outs = ", ".join(str(o) for o in self.outputs)
        args = ", ".join(_fmt_arg(a) for a in self.args)
        kw = ", ".join(f"{k}={_fmt_arg(v)}" for k, v in self.kwargs.items())
        joined = ", ".join(x for x in (args, kw) if x)
        return f"{outs} = {self.op}({joined})"

    __hash__ = object.__hash__


def _fmt_arg(a: Any) -> str:
    if isinstance(a, Value):
        return f"%{a.name}"
    if isinstance(a, (list, tuple)):
        return "[" + ", ".join(_fmt_arg(x) for x in a) + "]"
    return repr(a)


class Graph:
    """A whole program: placeholders, a topologically ordered node list, and
    a list of outputs.

    Placeholders are split into ``inputs`` (activations the caller passes each
    call) and ``params`` (weights, resident for the life of the program). The
    split matters for memory planning: params are never packed into the arena.
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Value] = []
        self.params: list[Value] = []
        self.outputs: list[Value] = []
        #: value name -> the actual weight tensor
        self.param_tensors: dict[str, torch.Tensor] = {}
        #: value name -> state-dict key it came from, for diagnostics
        self.param_origin: dict[str, str] = {}
        self._counter = itertools.count()
        self.meta: dict = {}

    # -- construction ------------------------------------------------------
    def fresh_name(self, prefix: str = "t") -> str:
        used = {v.name for v in self.values()}
        while True:
            name = f"{prefix}{next(self._counter)}"
            if name not in used:
                return name

    def new_buffer(self, shape: Sequence[int], dtype: torch.dtype, kind: str = "intermediate",
                   name: str | None = None) -> Buffer:
        return Buffer(name or self.fresh_name("buf"), tuple(shape), dtype, kind)

    def add_node(self, node: Node, before: Node | None = None) -> Node:
        if before is None:
            self.nodes.append(node)
        else:
            self.nodes.insert(self.nodes.index(before), node)
        for i, o in enumerate(node.outputs):
            o.producer = node
            o.index = i
        return node

    # -- queries -----------------------------------------------------------
    def placeholders(self) -> list[Value]:
        """All placeholders in the order the compiled callable expects them."""
        return list(self.params) + list(self.inputs)

    def values(self) -> Iterator[Value]:
        yield from self.params
        yield from self.inputs
        for n in self.nodes:
            yield from n.outputs

    def buffers(self) -> list[Buffer]:
        seen: dict[int, Buffer] = {}
        for v in self.values():
            seen.setdefault(id(v.buffer), v.buffer)
        return list(seen.values())

    def use_map(self) -> dict[int, list[Node]]:
        """value id -> nodes consuming it, in program order.

        Recomputed on demand rather than cached, so a pass that rewrites the
        node list cannot leave a stale use list behind.
        """
        uses: dict[int, list[Node]] = {}
        for n in self.nodes:
            for v in n.unique_tensor_args():
                uses.setdefault(id(v), []).append(n)
        return uses

    def users_of(self, v: Value) -> list[Node]:
        return self.use_map().get(id(v), [])

    def index_of(self, node: Node) -> int:
        return self.nodes.index(node)

    def topo_check(self) -> None:
        """Raise if any node reads a value that is not defined before it."""
        defined = {id(v) for v in self.placeholders()}
        for n in self.nodes:
            for v in n.unique_tensor_args():
                if id(v) not in defined:
                    raise ValueError(f"{n} reads undefined value %{v.name}")
            for o in n.outputs:
                if id(o) in defined:
                    raise ValueError(f"{n} redefines %{o.name}")
                defined.add(id(o))
        for v in self.outputs:
            if id(v) not in defined:
                raise ValueError(f"graph output %{v.name} is undefined")

    def dead_code_eliminate(self) -> int:
        """Drop nodes whose outputs nobody reads. Returns the count removed."""
        live = {id(v) for v in self.outputs}
        keep: list[Node] = []
        for n in reversed(self.nodes):
            if n.meta.get("pin") or any(id(o) in live for o in n.outputs):
                keep.append(n)
                for v in n.unique_tensor_args():
                    live.add(id(v))
        removed = len(self.nodes) - len(keep)
        self.nodes = list(reversed(keep))
        return removed

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        from .printer import format_graph

        return format_graph(self)
