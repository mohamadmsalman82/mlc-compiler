"""Fusion planning: decide which nodes share a kernel.

The elementwise pass finds maximal chains of ops over one iteration space.
Two rules do most of the work.

**Space compatibility.** A node joins a group when its output either
broadcasts to the group's iteration space, or has the same element count. The
second case is what lets fusion cross a reshape: a linear layer produces
[B*T, D] and the next op reads a [B, T, D] view of it. Both flatten to the
same row-major order, so the flat index means the same element in either
frame and no data has to move. Without that rule every reshape in a
transformer would end a kernel.

**Escape.** A node whose output is smaller than the group's iteration space
is recomputed at every point of that space. That is fine as long as nobody
outside needs the value, because a broadcast write would have several
iterations storing to the same address. So such a node may only join if all
its consumers join with it.

Everything else is the cost model in :mod:`mlc.passes.cost`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import Config
from ..ir.graph import Graph, Node, Value
from ..ir.ops import OpClass, is_pointwise, op_class
from ..ir.shapes import ShapeError, broadcast_shapes
from . import cost as costmod
from .scheduler import GroupGraph, UseInfo, defining_node


@dataclass
class KernelGroup:
    """A set of nodes destined for one kernel.

    ``owned`` are the nodes this kernel is responsible for storing. A node
    duplicated into a group by the recompute pass appears in ``nodes`` but not
    in ``owned``: it is evaluated for its value, not for its side effect.
    """

    nodes: list[Node]
    owned: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.owned:
            self.owned = {id(n) for n in self.nodes}

    @property
    def kind(self) -> str:
        classes = {op_class(n.op) for n in self.nodes}
        if OpClass.REDUCTION in classes:
            return "reduction"
        if classes == {OpClass.POINTWISE}:
            return "pointwise"
        return "extern"

    def space(self) -> tuple[int, ...]:
        return group_space(self.nodes)


def group_space(nodes) -> tuple[int, ...]:
    """The iteration space of a set of pointwise nodes: the output shape with
    the most elements. Ties keep the first in program order, which keeps the
    generated index arithmetic closest to the natural one."""
    best = ()
    best_n = -1
    for n in nodes:
        for o in n.outputs:
            n_elem = o.numel
            if n_elem > best_n:
                best, best_n = o.shape, n_elem
    return best


def _fits_space(shape: tuple[int, ...], space: tuple[int, ...]) -> str | None:
    """How ``shape`` relates to ``space``: 'full', 'broadcast', or None."""
    n_shape = math.prod(shape) if shape else 1
    n_space = math.prod(space) if space else 1
    if n_shape == n_space:
        # Both operands are contiguous allocations, so equal element counts
        # means one is a row-major reshape of the other and the flat index
        # addresses the same element in both frames.
        return "full"
    if n_shape > n_space:
        return None
    try:
        if broadcast_shapes([shape, space]) == space:
            return "broadcast"
    except ShapeError:
        return None
    return None


def _merged_space(nodes) -> tuple[int, ...] | None:
    space = group_space(nodes)
    for n in nodes:
        for o in n.outputs:
            if _fits_space(o.shape, space) is None:
                return None
    return space


def _escape_ok(nodes, space, uses: UseInfo, member_ids: set[int]) -> bool:
    """Broadcast members may not be visible outside the kernel."""
    for n in nodes:
        for o in n.outputs:
            if _fits_space(o.shape, space) != "broadcast":
                continue
            if uses.escapes(o):
                return False
            if not uses.consumers(o) <= member_ids:
                return False
    return True


def _estimate_ops(node: Node) -> int:
    """Rough scalar-op count for a node, for the arithmetic term of the cost
    model. Transcendentals are charged more because they are."""
    expensive = ("exp", "log", "erf", "tanh", "sigmoid", "pow", "sqrt", "rsqrt", "sin", "cos")
    tail = node.op.split(".")[-2] if "." in node.op else node.op
    return 12 if tail in expensive else 2


def fuse_elementwise(graph: Graph, gg: GroupGraph, cfg: Config, uses: UseInfo) -> None:
    """Greedy maximal fusion of pointwise chains, in program order."""
    if not cfg.elementwise_fusion:
        return

    for node in gg.nodes:
        if not is_pointwise(node.op):
            continue
        gi = gg.group_of(node)
        assert gi is not None
        for v in node.unique_tensor_args():
            producer = defining_node(v)
            if producer is None or not is_pointwise(producer.op):
                continue
            gj = gg.group_of(producer)
            if gj is None or gg.find(gj) == gg.find(gi):
                continue
            if not gg.can_merge(gi, gj):
                continue
            merged = gg.group_members(gg.find(gi)) + gg.group_members(gg.find(gj))
            space = _merged_space(merged)
            if space is None:
                continue
            member_ids = {id(n) for n in merged}
            if not _escape_ok(merged, space, uses, member_ids):
                continue
            if not _worth_merging(gg, gi, gj, space, uses, member_ids, cfg):
                continue
            gi = gg.merge(gi, gj)


def _worth_merging(gg, gi, gj, space, uses, member_ids, cfg: Config) -> bool:
    """Cost check for one merge.

    Merging never adds loads: the fused kernel reads the union of what the two
    read. It can add arithmetic, when a group is pulled into a larger
    iteration space and its work is repeated per point. The saving is the
    intermediates that stop round-tripping, plus one kernel launch.
    """
    a = gg.group_members(gg.find(gi))
    b = gg.group_members(gg.find(gj))
    n_space = math.prod(space) if space else 1

    saved_stores: list[Value] = []
    saved_loads: list[Value] = []
    for group, other in ((a, b), (b, a)):
        other_ids = {id(n) for n in other}
        for n in group:
            for o in n.outputs:
                if not (uses.consumers(o) & other_ids):
                    continue
                saved_loads.append(o)
                if not uses.escapes(o) and uses.consumers(o) <= member_ids:
                    saved_stores.append(o)

    extra_evals = 0
    expr_ops = 0
    for group in (a, b):
        gs = group_space(group)
        n_group = math.prod(gs) if gs else 1
        if n_group < n_space:
            ops = sum(_estimate_ops(n) for n in group)
            extra_evals += n_space - n_group
            expr_ops = max(expr_ops, ops)

    decision = costmod.evaluate_merge(
        cfg,
        saved_stores=saved_stores,
        saved_loads=saved_loads,
        extra_loads=[],
        extra_evaluations=extra_evals,
        expr_ops=expr_ops,
    )
    return bool(decision)


# --------------------------------------------------------------------------
# Recompute: duplicate a producer instead of storing it
# --------------------------------------------------------------------------

def apply_recompute(graph: Graph, groups: list[KernelGroup], cfg: Config,
                    uses: UseInfo) -> list[KernelGroup]:
    """Inline a small producer group into every group that reads it.

    Union-find fusion can only attach a producer to *one* consumer; a producer
    with several consumers still writes its result for the rest. When the
    producer is cheap, evaluating it again in each consumer beats the store
    plus the reloads. This is the pass the spec calls out: a multi-consumer
    producer fuses only when recomputation costs less than the round trip it
    saves.
    """
    if not cfg.recompute:
        return groups

    changed = True
    while changed:
        changed = False
        index = {id(n): gi for gi, g in enumerate(groups) for n in g.nodes if id(n) in g.owned}
        for gi, g in enumerate(groups):
            if g.kind != "pointwise":
                continue
            produced = [o for n in g.nodes if id(n) in g.owned for o in n.outputs]
            if any(uses.escapes(o) for o in produced):
                continue
            consumer_groups: set[int] = set()
            for o in produced:
                for cid in uses.consumers(o):
                    cg = index.get(cid)
                    if cg is None or cg == gi:
                        continue
                    consumer_groups.add(cg)
            if len(consumer_groups) < 2:
                continue  # a single consumer is already handled by merging
            if not _recompute_legal(g, [groups[cg] for cg in consumer_groups], uses):
                continue
            if not _recompute_profitable(graph, groups, gi, consumer_groups, produced, cfg):
                continue
            for cg in consumer_groups:
                dup = [n for n in g.nodes if id(n) not in {id(x) for x in groups[cg].nodes}]
                groups[cg].nodes = _sorted_nodes(graph, groups[cg].nodes + dup)
            groups[gi] = None  # type: ignore[call-overload]
            changed = True
            break
        if changed:
            groups = [g for g in groups if g is not None]
    return groups


def _recompute_legal(producer: KernelGroup, targets: list[KernelGroup],
                     uses: UseInfo) -> bool:
    """May ``producer``'s nodes be evaluated inside every one of ``targets``?

    The same two rules as ordinary merging, with one difference that matters.
    The escape check has to be made against the union of all the targets, not
    against each one separately: after duplication every consumer holds its
    own copy, so a value read by two consumers has not escaped anything.
    Checking one target at a time would refuse precisely the multi-consumer
    case this pass exists for, and did.

    The targets must stay pointwise. Inlining a producer into a reduction is a
    different transform with its own legality, and it belongs to the reduction
    pass.
    """
    if producer.kind != "pointwise" or any(t.kind != "pointwise" for t in targets):
        return False
    everyone = {id(n) for t in targets for n in t.nodes} | {id(n) for n in producer.nodes}
    for target in targets:
        merged = target.nodes + [n for n in producer.nodes if n not in target.nodes]
        space = _merged_space(merged)
        if space is None:
            return False
        if not _escape_ok(merged, space, uses, everyone):
            return False
    return True


def _recompute_profitable(graph, groups, gi, consumer_groups, produced, cfg) -> bool:
    g = groups[gi]
    ops = sum(_estimate_ops(n) for n in g.nodes)
    own_space = g.space()
    n_own = math.prod(own_space) if own_space else 1

    extra_evals = 0
    for cg in consumer_groups:
        cs = groups[cg].space()
        extra_evals += math.prod(cs) if cs else 1
    extra_evals -= n_own  # the original evaluation is not extra

    # Producer inputs the consumers do not already read have to be loaded.
    consumer_reads = set()
    for cg in consumer_groups:
        for n in groups[cg].nodes:
            for v in n.unique_tensor_args():
                consumer_reads.add(id(v.buffer))
    extra_loads = [v for n in g.nodes for v in n.unique_tensor_args()
                   if id(v.buffer) not in consumer_reads]

    decision = costmod.evaluate_merge(
        cfg,
        saved_stores=produced,
        saved_loads=produced * len(consumer_groups),
        extra_loads=extra_loads,
        extra_evaluations=extra_evals,
        expr_ops=ops,
        saved_launches=1,
    )
    return bool(decision)


def _sorted_nodes(graph: Graph, nodes) -> list[Node]:
    order = {id(n): i for i, n in enumerate(graph.nodes)}
    seen, out = set(), []
    for n in sorted(nodes, key=lambda n: order[id(n)]):
        if id(n) not in seen:
            seen.add(id(n))
            out.append(n)
    return out


def plan(graph: Graph, cfg: Config) -> list[KernelGroup]:
    """Run the fusion passes and return the final grouping, in order.

    Elementwise first, so reduction fusion sees whole pointwise chains rather
    than individual ops, then reduction fusion over the same group graph so
    the acyclicity check covers both. Recompute runs last, on the materialised
    groups, because duplicating a node puts it in two groups at once and
    union-find cannot represent that.
    """
    from .reduction_fusion import fuse_reductions

    uses = UseInfo(graph)
    gg = GroupGraph(graph)
    fuse_elementwise(graph, gg, cfg, uses)
    fuse_reductions(graph, gg, cfg, uses)
    groups = [KernelGroup(_sorted_nodes(graph, gg.group_members(r)))
              for r in gg.roots_in_order()]
    groups = apply_recompute(graph, groups, cfg, uses)
    _validate(groups)
    return groups


def _validate(groups: list[KernelGroup]) -> None:
    """Every pointwise group must have an iteration space each member fits.

    Cheap, and it turns a subtle miscompile into an assertion at the pass
    boundary that names the offending group.
    """
    for g in groups:
        if g.kind != "pointwise":
            continue
        if _merged_space(g.nodes) is None:
            raise AssertionError(
                "fusion produced an incoherent group: "
                + ", ".join(f"{n.op}{list(n.out.shape)}" for n in g.nodes)
            )
