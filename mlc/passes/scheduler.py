"""Grouping machinery shared by the fusion passes.

Fusion is a partition problem: decide which graph nodes share a kernel. The
constraint that makes it non-trivial is acyclicity. If group A feeds group C
which feeds group B, then A and B cannot be merged -- the merged kernel would
have to run both before and after C.

:class:`GroupGraph` is a union-find over nodes that refuses any merge which
would create such a cycle. It maintains, per group, a bitset of transitive
ancestors and descendants, so the check is two bitwise ANDs.
"""

from __future__ import annotations

from typing import Iterable

from ..ir.graph import Graph, Node, Value
from ..ir.ops import OpClass, op_class


def defining_node(v: Value) -> Node | None:
    """The node that actually computes ``v``, looking through views.

    Views are free restrides, so for scheduling purposes a value read through
    a chain of permutes and reshapes depends on whatever produced the buffer.
    """
    n = v.producer
    while n is not None and op_class(n.op) is OpClass.VIEW:
        src = n.args[0]
        if not isinstance(src, Value):
            return n
        n = src.producer
    return n


def source_value(v: Value) -> Value:
    """Walk back through views to the value that owns the buffer."""
    n = v.producer
    while n is not None and op_class(n.op) is OpClass.VIEW:
        src = n.args[0]
        if not isinstance(src, Value):
            break
        v, n = src, src.producer
    return v


class GroupGraph:
    """Union-find over the schedulable nodes of a graph, cycle-safe."""

    def __init__(self, graph: Graph, nodes: Iterable[Node] | None = None) -> None:
        self.graph = graph
        # Views never become kernels; they are folded into whoever reads them.
        self.nodes = [n for n in (nodes if nodes is not None else graph.nodes)
                      if op_class(n.op) is not OpClass.VIEW]
        self.index = {id(n): i for i, n in enumerate(self.nodes)}
        self.parent = list(range(len(self.nodes)))
        self.members: list[list[int]] = [[i] for i in range(len(self.nodes))]

        n = len(self.nodes)
        # direct edges, in group-index space
        self.preds: list[set[int]] = [set() for _ in range(n)]
        self.succs: list[set[int]] = [set() for _ in range(n)]
        for i, node in enumerate(self.nodes):
            for v in node.unique_tensor_args():
                d = defining_node(v)
                if d is not None and id(d) in self.index:
                    j = self.index[id(d)]
                    if j != i:
                        self.preds[i].add(j)
                        self.succs[j].add(i)

        self.anc: list[int] = [0] * n  # bitset of transitive predecessors
        self.desc: list[int] = [0] * n  # bitset of transitive successors
        for i in range(n):
            bits = 0
            for j in self.preds[i]:
                bits |= (1 << j) | self.anc[j]
            self.anc[i] = bits
        for i in range(n - 1, -1, -1):
            bits = 0
            for j in self.succs[i]:
                bits |= (1 << j) | self.desc[j]
            self.desc[i] = bits

    # -- union-find --------------------------------------------------------
    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def group_of(self, node: Node) -> int | None:
        i = self.index.get(id(node))
        return None if i is None else self.find(i)

    def group_members(self, root: int) -> list[Node]:
        return [self.nodes[i] for i in sorted(self.members[root])]

    # -- merging -----------------------------------------------------------
    def can_merge(self, a: int, b: int) -> bool:
        """True when groups ``a`` and ``b`` can share a kernel.

        Illegal exactly when some third group sits on a path between them: it
        would have to run both after the merged kernel (because it depends on
        one half) and before it (because the other half depends on it).
        """
        a, b = self.find(a), self.find(b)
        if a == b:
            return True
        a_bit, b_bit = 1 << a, 1 << b
        # groups strictly between a and b, in either direction
        between = (self.desc[a] & self.anc[b]) | (self.desc[b] & self.anc[a])
        between &= ~(a_bit | b_bit)
        return between == 0

    def merge(self, a: int, b: int) -> int:
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if not self.can_merge(a, b):
            raise ValueError("merge would create a cycle")
        # keep the earlier root so group ids stay roughly in program order
        keep, gone = (a, b) if a < b else (b, a)
        self.parent[gone] = keep
        self.members[keep].extend(self.members[gone])
        self.members[gone] = []

        merged_anc = (self.anc[a] | self.anc[b]) & ~((1 << a) | (1 << b))
        merged_desc = (self.desc[a] | self.desc[b]) & ~((1 << a) | (1 << b))
        gone_bit, keep_bit = 1 << gone, 1 << keep
        for i in range(len(self.nodes)):
            if self.anc[i] & gone_bit:
                self.anc[i] = (self.anc[i] & ~gone_bit) | keep_bit
            if self.desc[i] & gone_bit:
                self.desc[i] = (self.desc[i] & ~gone_bit) | keep_bit
        self.anc[keep] = merged_anc
        self.desc[keep] = merged_desc
        self.anc[gone] = self.desc[gone] = 0
        # Anyone downstream of the merged group inherits its ancestors, and
        # vice versa, or a later can_merge could miss a path.
        for i in range(len(self.nodes)):
            if self.find(i) != i:
                continue
            if self.desc[keep] & (1 << i):
                self.anc[i] |= merged_anc | keep_bit
            if self.anc[keep] & (1 << i):
                self.desc[i] |= merged_desc | keep_bit
        return keep

    def roots_in_order(self) -> list[int]:
        """Group roots, ordered so every group follows its dependencies.

        Program order of the *earliest* member is not always a valid schedule
        after merging, so this is a real topological sort over the group DAG.
        """
        roots = [i for i in range(len(self.nodes)) if self.find(i) == i and self.members[i]]
        rank = {r: min(self.members[r]) for r in roots}
        indeg = {r: 0 for r in roots}
        edges: dict[int, set[int]] = {r: set() for r in roots}
        for i, node in enumerate(self.nodes):
            gi = self.find(i)
            for j in self.preds[i]:
                gj = self.find(j)
                if gi != gj and gi not in edges[gj]:
                    edges[gj].add(gi)
                    indeg[gi] += 1
        import heapq

        ready = [(rank[r], r) for r in roots if indeg[r] == 0]
        heapq.heapify(ready)
        out: list[int] = []
        while ready:
            _, r = heapq.heappop(ready)
            out.append(r)
            for s in edges[r]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    heapq.heappush(ready, (rank[s], s))
        if len(out) != len(roots):
            raise RuntimeError("group graph has a cycle; a merge check is wrong")
        return out


# --------------------------------------------------------------------------
# Use analysis
# --------------------------------------------------------------------------

class UseInfo:
    """Who reads what, with view chains collapsed.

    Views are not kernels, so "the consumers of a value" means the consumers
    of every view reachable from it. Every fusion decision needs this, and
    getting it wrong in the optimistic direction drops a store that something
    still reads.
    """

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._users = graph.use_map()
        self._memo: dict[int, frozenset[int]] = {}
        self.escaping_buffers = {id(v.buffer) for v in graph.outputs}

    def consumers(self, v: Value) -> frozenset[int]:
        """ids of the non-view nodes that read ``v``, directly or via views."""
        key = id(v)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        self._memo[key] = frozenset()  # guard against cycles; the IR has none
        out: set[int] = set()
        for u in self._users.get(key, []):
            if op_class(u.op) is OpClass.VIEW:
                for o in u.outputs:
                    out |= self.consumers(o)
            else:
                out.add(id(u))
        result = frozenset(out)
        self._memo[key] = result
        return result

    def escapes(self, v: Value) -> bool:
        """True when ``v``'s buffer is a graph output and must be written."""
        return id(v.buffer) in self.escaping_buffers

    def node_consumers(self, node: Node) -> frozenset[int]:
        out: set[int] = set()
        for o in node.outputs:
            out |= self.consumers(o)
        return frozenset(out)
