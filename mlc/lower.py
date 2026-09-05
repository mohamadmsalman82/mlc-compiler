"""Turn fusion groups into kernels.

This is where the abstract grouping decision becomes something a backend can
emit: an iteration space, a list of operands with their index maps, a body of
scalar expressions, and a list of stores.

The one interesting decision here is when a read becomes a register reference
instead of a load. A value produced inside the kernel is already in a
register; reading it back is only valid if the read touches the same address
the producer wrote for the same iteration. Index maps make that a structural
equality test, and because the maps are canonical it sees through reshapes:
[B*T, D] and [B, T, D] views of the same contiguous buffer produce the same
map, so a linear layer's output feeds the next elementwise op without ever
reaching memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .codegen.index import IndexMap, build_index, flat_index
from .config import Config
from .ir.graph import Graph, Node, Value
from .ir.ops import OpClass, lookup, op_class
from .ir.scalar import Const, Expr, Load, Ref, substitute
from .ir.ops import _S
from .ir.types import Layout
from .kernels import ExternKernel, Kernel, KernelArg, PointwiseKernel, Schedule
from .passes.fusion import KernelGroup, _fits_space, group_space
from .passes.scheduler import UseInfo, source_value


class LoweringError(Exception):
    pass


def build_schedule(graph: Graph, groups: list[KernelGroup], cfg: Config) -> Schedule:
    uses = UseInfo(graph)
    stores = _store_set(graph, groups, uses)
    kernels: list[Kernel] = []
    for i, group in enumerate(groups):
        kind = group.kind
        if kind == "pointwise":
            kernels.append(build_pointwise(graph, group, stores, f"k{i}"))
        elif kind == "reduction" and cfg.reduction_fusion:
            from .codegen.reduction_lowering import build_reduction

            kernels.append(build_reduction(graph, group, stores, f"k{i}", cfg))
        else:
            # Reductions with the pass disabled, matmuls, and anything opaque:
            # dispatched to the torch operator and treated as a fusion barrier.
            if len(group.nodes) != 1:
                raise LoweringError(
                    f"group of {len(group.nodes)} nodes has no kernel form: "
                    f"{[n.op for n in group.nodes]}"
                )
            node = group.nodes[0]
            kernels.append(ExternKernel(f"k{i}", node, node.unique_tensor_args(), list(node.outputs)))
    return Schedule(graph, kernels)


def _store_set(graph: Graph, groups: list[KernelGroup], uses: UseInfo) -> set[int]:
    """Values that have to be written to memory.

    A value needs a store when it is a graph output, or when some consumer
    lives in a kernel that does not itself compute it. After the recompute
    pass a node can appear in several kernels, so this is decided globally
    rather than per kernel.
    """
    contains: dict[int, set[int]] = {}
    for gi, g in enumerate(groups):
        for n in g.nodes:
            contains.setdefault(id(n), set()).add(gi)

    need: set[int] = set()
    for g in groups:
        for n in g.nodes:
            for o in n.outputs:
                if uses.escapes(o):
                    need.add(id(o))
                    continue
                producer_kernels = contains.get(id(n), set())
                for cid in uses.consumers(o):
                    if not contains.get(cid, set()) <= producer_kernels:
                        need.add(id(o))
                        break
    return need


def _prune_inputs(inputs: list[KernelArg], body, out_expr) -> list[KernelArg]:
    """Drop operands nothing in the body reads.

    Some lowerings ignore an operand they were handed: ``full_like`` takes a
    tensor only to borrow its shape, and folding ``x * 1`` can leave a load
    with no user. Pruning here keeps the kernel from moving bytes it does not
    need, which the cost model would otherwise be charged for.
    """
    from .ir.scalar import loads as expr_loads

    used: set[int] = set()
    for _, e in body:
        used |= expr_loads(e)
    for e in out_expr.values():
        used |= expr_loads(e)
    if len(used) == len(inputs):
        return inputs
    keep = sorted(used)
    remap = {old: new for new, old in enumerate(keep)}
    mapping = {Load(old): Load(new) for old, new in remap.items()}
    for i, (nm, e) in enumerate(body):
        body[i] = (nm, substitute(e, mapping))
    for k in list(out_expr):
        out_expr[k] = substitute(out_expr[k], mapping)
    return [inputs[i] for i in keep]


# --------------------------------------------------------------------------
# Pointwise
# --------------------------------------------------------------------------

class _BodyBuilder:
    """Shared machinery for building a kernel body from a node list."""

    def __init__(self, graph: Graph, nodes: list[Node]) -> None:
        self.graph = graph
        self.nodes = nodes
        self.member_ids = {id(n) for n in nodes}
        self.inputs: list[KernelArg] = []
        self._slot: dict[tuple, int] = {}
        #: value id -> (expression holding it, index map it was computed at)
        self.env: dict[int, tuple[Expr, IndexMap]] = {}
        self.body: list[tuple[str, Expr]] = []
        self._temps = 0

    def fresh(self) -> str:
        self._temps += 1
        return f"t{self._temps - 1}"

    def load(self, v: Value, imap: IndexMap) -> Expr:
        key = (id(v.buffer), v.dtype, imap)
        slot = self._slot.get(key)
        if slot is None:
            slot = len(self.inputs)
            self._slot[key] = slot
            self.inputs.append(KernelArg(v, imap))
        return Load(slot)

    def operand(self, v: Value, imap: IndexMap) -> Expr:
        """A register reference when the value is already live here at the
        same address, otherwise a load."""
        src = source_value(v)
        live = self.env.get(id(src))
        if live is not None and live[1] == imap:
            return live[0]
        return self.load(v, imap)

    def emit(self, node: Node, arg_exprs: list[Expr]) -> Expr:
        """Lower one node and bind its result.

        A node whose lowering is a bare load or reference -- ``clone``, a
        no-op cast, a multiply by one that folded away -- gets no line of its
        own. Later uses refer straight to the original, so the generated
        kernel has one statement per real operation and nothing else.
        """
        spec = lookup(node.op)
        if spec is None or spec.lower is None:
            raise LoweringError(f"no lowering rule for {node.op}")
        expr = spec.lower(list(node.args), arg_exprs, dict(node.kwargs))
        if isinstance(expr, (Load, Ref, Const)):
            return expr
        name = self.fresh()
        self.body.append((name, expr))
        return Ref(name)


def build_pointwise(graph: Graph, group: KernelGroup, stores: set[int],
                    name: str) -> PointwiseKernel:
    nodes = group.nodes
    space = group_space(nodes)
    b = _BodyBuilder(graph, nodes)

    for node in nodes:
        out = node.out
        fit = _fits_space(out.shape, space)
        if fit is None:
            raise LoweringError(f"{node.op} shape {out.shape} does not fit space {space}")
        frame = out.shape if fit == "full" else space

        arg_exprs: list[Expr] = []
        for a in node.args:
            if isinstance(a, Value):
                lay = a.layout if a.shape == frame else a.layout.expand(frame)
                if fit == "broadcast" and frame is space:
                    lay = a.layout.expand(out.shape).expand(space)
                arg_exprs.append(b.operand(a, flat_index(frame if fit == "full" else space, lay)))
            elif isinstance(a, (bool, int, float)):
                arg_exprs.append(_S(a))
            else:
                # dims, dtypes, memory formats: the lowering rule reads these
                # off node.args directly, so pass them through untouched.
                arg_exprs.append(a)

        value_expr = b.emit(node, arg_exprs)
        out_index = flat_index(out.shape if fit == "full" else space,
                               out.layout if fit == "full" else out.layout.expand(space))
        b.env[id(out)] = (value_expr, out_index)

    outputs: list[KernelArg] = []
    out_expr: dict[str, Expr] = {}
    for node in nodes:
        if id(node) not in group.owned:
            continue
        out = node.out
        if id(out) not in stores:
            continue
        fit = _fits_space(out.shape, space)
        if fit != "full":
            raise LoweringError(
                f"{node.op} is broadcast inside its kernel but its result escapes; "
                "the fusion pass should not have grouped it"
            )
        outputs.append(KernelArg(out, flat_index(out.shape, out.layout)))
        out_expr[out.name] = b.env[id(out)][0]

    inputs = _prune_inputs(b.inputs, b.body, out_expr)
    return PointwiseKernel(
        name=name,
        space=tuple(space),
        inputs=inputs,
        outputs=outputs,
        body=b.body,
        out_expr=out_expr,
        nodes=list(nodes),
    )
