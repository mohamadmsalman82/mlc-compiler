"""Reduction fusion.

Two directions, both aimed at the same thing: keep a row of data resident and
do everything to it while it is there.

**Producers into the reduction.** The ops feeding a reduce -- the scale and
the mask before a softmax -- are evaluated inside the kernel on data already
loaded, so their results never reach memory.

**Reductions into consumers.** The ops reading a reduce's result broadcast it
back across the axis that was reduced. Layer norm's subtract-and-scale, and
the affine transform after it, are all of this shape. Since the row is still
in registers, the consumer runs without re-reading anything.

Sibling reductions over the same axis merge too, which is what collapses layer
norm's mean and variance into one pass over the row instead of two.

Legality, all enforced below and each with a named check:

  * **One frame.** Every reduction in a group must reduce the same axes of the
    same element shape. Different frames cannot share an iteration space.
  * **Shape.** Every member's output must broadcast to the element shape or to
    the row shape. Unlike the elementwise pass, reshapes that change rank are
    refused: the row/column split makes flat order frame-dependent, so the
    same-element-count shortcut is not sound here.
  * **Escape.** A value that is broadcast inside the kernel, or that lives in
    row space when its consumers want element space, may not be visible
    outside it.
  * **Associativity.** Splitting a reduce across blocks requires the combining
    operation to be associative. Every kind we emit is; the check exists so
    that adding a non-associative one fails loudly rather than silently
    reordering.
  * **Capacity.** The reduced axis has to fit the register budget for a
    persistent kernel. Beyond that the kernel streams the row instead, which
    is correct but costs one HBM pass per reduce.

The multi-consumer producer rule is the same cost comparison as the
elementwise pass: fuse a producer with several consumers only when
recomputing it beats the round trip.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..config import Config
from ..ir.graph import Graph, Node, Value
from ..ir.ops import OpClass, REDUCE_KINDS, lookup, op_class
from ..ir.shapes import ShapeError, broadcast_shapes
from . import cost as costmod
from .scheduler import GroupGraph, UseInfo, defining_node


class Frame:
    """The (row, column) iteration frame a reduction group runs in."""

    __slots__ = ("element", "reduce_dims", "row", "col", "order")

    def __init__(self, element: tuple[int, ...], reduce_dims: tuple[int, ...]) -> None:
        self.element = element
        self.reduce_dims = tuple(sorted(d % len(element) for d in reduce_dims))
        kept = [i for i in range(len(element)) if i not in self.reduce_dims]
        self.row = tuple(element[i] for i in kept)
        self.col = tuple(element[i] for i in self.reduce_dims)
        #: permutation putting kept dims first and reduced dims last
        self.order = tuple(kept) + self.reduce_dims

    @property
    def n_rows(self) -> int:
        return math.prod(self.row) if self.row else 1

    @property
    def n_cols(self) -> int:
        return math.prod(self.col) if self.col else 1

    def __eq__(self, other) -> bool:
        return (isinstance(other, Frame) and self.element == other.element
                and self.reduce_dims == other.reduce_dims)

    def __hash__(self) -> int:
        return hash((self.element, self.reduce_dims))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Frame({list(self.element)} reduce {list(self.reduce_dims)})"


def frame_of(node: Node) -> Frame | None:
    """The frame a reduction node defines, or None if it is not a reduction."""
    spec = lookup(node.op)
    if spec is None or spec.reduce is None:
        return None
    src = node.args[0]
    if not isinstance(src, Value):
        return None
    _, dims, _ = spec.reduce(list(node.args), dict(node.kwargs))
    return Frame(src.shape, dims)


def reduce_kind(node: Node) -> str:
    spec = lookup(node.op)
    kind, _, _ = spec.reduce(list(node.args), dict(node.kwargs))
    return kind


#: Composite reductions expand into several primitive stages during lowering.
COMPOSITE = {"mean", "var", "var_mean"}


def is_associative(kind: str) -> bool:
    if kind in COMPOSITE:
        return True
    spec = REDUCE_KINDS.get(kind)
    return spec is not None and spec.associative


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------

ELEMENT, ROW, ILLEGAL = "element", "row", None


def classify(shape: tuple[int, ...], frame: Frame) -> str | None:
    """Where a value of ``shape`` lives relative to ``frame``.

    ``row`` means one value per program: it does not vary along any reduced
    axis, so inside the kernel it is a scalar. ``element`` means it does vary
    and is a vector over the column axis.

    The distinction is not simply "smaller shape". A reduction with
    ``keepdim=True`` has shape [B, T, 1] which broadcasts to the element shape
    [B, T, D] perfectly well, yet it is a row value; a weight vector of shape
    [D] also broadcasts to [B, T, D] and is an element value. What separates
    them is whether the reduced positions are size 1 after right-alignment,
    which is exactly the test below.
    """
    shape = tuple(shape)
    rank = len(frame.element)
    try:
        if broadcast_shapes([shape, frame.element]) == frame.element:
            aligned = (1,) * (rank - len(shape)) + shape
            if all(aligned[d] == 1 for d in frame.reduce_dims):
                return ROW
            return ELEMENT
    except ShapeError:
        pass
    # keepdim=False reduction outputs have the reduced dims removed entirely,
    # so they do not right-align against the element shape at all.
    if shape == frame.row:
        return ROW
    if frame.row:
        try:
            if broadcast_shapes([shape, frame.row]) == frame.row:
                return ROW
        except ShapeError:
            pass
    if (math.prod(shape) if shape else 1) == 1:
        return ROW
    return ILLEGAL


def storable(shape: tuple[int, ...], frame: Frame) -> bool:
    """Can a value of this shape be written out by the kernel?

    Only if every point of its own shape is visited exactly once. A value
    broadcast inside the kernel is not: several iterations would store to the
    same address, so it has to stay internal.
    """
    n = math.prod(shape) if shape else 1
    where = classify(shape, frame)
    if where is ROW:
        return n == frame.n_rows
    if where is ELEMENT:
        return n == frame.n_rows * frame.n_cols
    return False


def _group_frame(nodes: Iterable[Node]) -> Frame | None:
    """The single frame every reduction in ``nodes`` agrees on, or None."""
    frames = {frame_of(n) for n in nodes if op_class(n.op) is OpClass.REDUCTION}
    frames.discard(None)
    if len(frames) != 1:
        return None
    return next(iter(frames))


def _members_fit(nodes: Iterable[Node], frame: Frame) -> bool:
    """Shape and associativity legality for a candidate member set."""
    for n in nodes:
        cls = op_class(n.op)
        if cls is OpClass.REDUCTION:
            if frame_of(n) != frame:
                return False  # one frame per kernel
            if not is_associative(reduce_kind(n)):
                return False  # cannot reorder across blocks
            if any(classify(o.shape, frame) is not ROW for o in n.outputs):
                return False
            continue
        if cls is not OpClass.POINTWISE:
            return False
        if any(classify(o.shape, frame) is ILLEGAL for o in n.outputs):
            return False
        # Operands are checked against the node's own output shape, which
        # pointwise typing already guarantees they broadcast to; what matters
        # here is that the output can be placed in the frame at all.
    return True


def _escape_ok(nodes, frame: Frame, uses: UseInfo, member_ids: set[int]) -> bool:
    """A value that is broadcast inside the kernel must not be needed outside.

    Element-space nodes whose own shape is smaller than the element shape are
    evaluated once per column and never stored, so a reader outside the kernel
    would find nothing. Row-space nodes are fine to store -- they are written
    once per program -- so only the broadcast case is restricted.
    """
    for n in nodes:
        for o in n.outputs:
            if storable(o.shape, frame):
                continue
            if uses.escapes(o) or not uses.consumers(o) <= member_ids:
                return False
    return True


def _profitable(gg: GroupGraph, a: int, b: int, frame: Frame, uses: UseInfo,
                member_ids: set[int], cfg: Config) -> bool:
    """Same currency as the elementwise pass: bytes that stop moving."""
    left = gg.group_members(gg.find(a))
    right = gg.group_members(gg.find(b))
    saved_stores: list[Value] = []
    saved_loads: list[Value] = []
    for group, other in ((left, right), (right, left)):
        other_ids = {id(n) for n in other}
        for n in group:
            for o in n.outputs:
                if not (uses.consumers(o) & other_ids):
                    continue
                saved_loads.append(o)
                if not uses.escapes(o) and uses.consumers(o) <= member_ids:
                    saved_stores.append(o)
    if not saved_loads:
        # Sibling reductions over the same row share the *load* rather than an
        # intermediate: the saving is one full read of the row, plus a launch.
        reduction_nodes = [n for n in left + right if op_class(n.op) is OpClass.REDUCTION]
        if len(reduction_nodes) < 2:
            return False
        shared = {id(v.buffer): v for n in left for v in n.unique_tensor_args()}
        common = [v for n in right for v in n.unique_tensor_args() if id(v.buffer) in shared]
        if not common:
            return False
        saved_loads = common[:1]

    decision = costmod.evaluate_merge(
        cfg,
        saved_stores=saved_stores,
        saved_loads=saved_loads,
        extra_loads=[],
        extra_evaluations=0,
        expr_ops=0,
    )
    return bool(decision)


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------

def fuse_reductions(graph: Graph, gg: GroupGraph, cfg: Config, uses: UseInfo) -> None:
    """Grow each reduction group over its producers, siblings and consumers.

    Runs to a fixed point: absorbing a consumer can expose a further consumer
    of *its* result, which is how layer norm picks up the affine transform
    after the normalisation it was already fused with.
    """
    if not cfg.reduction_fusion:
        return

    seeds = [n for n in gg.nodes if op_class(n.op) is OpClass.REDUCTION]
    if not seeds:
        return

    changed = True
    while changed:
        changed = False
        for seed in seeds:
            gi = gg.group_of(seed)
            if gi is None:
                continue
            frame = _group_frame(gg.group_members(gg.find(gi)))
            if frame is None:
                continue
            for gj in _neighbours(gg, gi):
                if gg.find(gj) == gg.find(gi):
                    continue
                if not gg.can_merge(gi, gj):
                    continue
                merged = gg.group_members(gg.find(gi)) + gg.group_members(gg.find(gj))
                if _group_frame(merged) != frame:
                    continue
                if not _members_fit(merged, frame):
                    continue
                member_ids = {id(n) for n in merged}
                if not _escape_ok(merged, frame, uses, member_ids):
                    continue
                if not _profitable(gg, gi, gj, frame, uses, member_ids, cfg):
                    continue
                gi = gg.merge(gi, gj)
                changed = True


def _neighbours(gg: GroupGraph, gi: int) -> list[int]:
    """Groups directly adjacent to ``gi``, plus groups reading the same
    buffers -- the latter is how sibling reductions over one row find each
    other, since neither reads the other's output."""
    root = gg.find(gi)
    out: list[int] = []
    seen: set[int] = set()
    members = gg.group_members(root)
    for i, node in enumerate(gg.nodes):
        if gg.find(i) == root:
            for j in gg.preds[i] | gg.succs[i]:
                r = gg.find(j)
                if r != root and r not in seen:
                    seen.add(r)
                    out.append(r)

    read_buffers = {id(v.buffer) for n in members for v in n.unique_tensor_args()}
    for i, node in enumerate(gg.nodes):
        r = gg.find(i)
        if r == root or r in seen or op_class(node.op) is not OpClass.REDUCTION:
            continue
        if any(id(v.buffer) in read_buffers for v in node.unique_tensor_args()):
            seen.add(r)
            out.append(r)
    return out
