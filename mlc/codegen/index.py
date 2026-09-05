"""Index arithmetic for generated kernels.

A kernel iterates over one or more variables -- a flat ``idx`` for pointwise
kernels, a ``(row, col)`` pair for reductions. Reading an operand means
turning those variables into a buffer offset:

    offset = base + sum over axes of sum over dims of
             ((var // out_stride) % size) * in_stride

With static shapes every coefficient is a compile-time constant, so this
module's job is to shrink that sum before it reaches the generated source.
Three simplifications, per axis, in order:

  * drop size-1 dimensions,
  * merge adjacent dimensions contiguous in *both* the iteration space and
    the operand, which collapses a contiguous operand to ``offset = idx``,
  * drop stride-0 dimensions, which is how broadcasting disappears -- a bias
    vector read across a batch reduces to indexing by the feature dim alone,
    and a row statistic read back across its row drops out of the column
    index entirely.

The canonical form doubles as an equality test. Two operands with the same
:class:`IndexMap` touch the same address at the same iteration, which is what
lets fusion replace a load with a register reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..ir.types import Layout, contiguous_strides


@dataclass(frozen=True)
class Axis:
    """The contribution of one iteration variable to an offset."""

    var: str
    numel: int
    #: (size, out_stride, in_stride), outermost first
    terms: tuple[tuple[int, int, int], ...] = ()

    @property
    def is_constant(self) -> bool:
        """The operand does not vary along this axis at all."""
        return not self.terms

    @property
    def is_identity(self) -> bool:
        return self.terms == ((self.numel, 1, 1),)

    def render(self) -> list[str]:
        parts: list[str] = []
        for size, out_stride, in_stride in self.terms:
            t = self.var
            if out_stride != 1:
                t = f"({t} // {out_stride})"
            if size * out_stride != self.numel:
                t = f"({t} % {size})"
            if in_stride != 1:
                t = f"{t} * {in_stride}"
            parts.append(t)
        return parts


@dataclass(frozen=True)
class IndexMap:
    axes: tuple[Axis, ...]
    base: int = 0

    def axis(self, var: str) -> Axis | None:
        for a in self.axes:
            if a.var == var:
                return a
        return None

    def varies_along(self, var: str) -> bool:
        a = self.axis(var)
        return a is not None and not a.is_constant

    def is_identity(self) -> bool:
        return self.base == 0 and len(self.axes) == 1 and self.axes[0].is_identity

    def is_scalar(self) -> bool:
        """Every iteration reads the same element."""
        return all(a.is_constant for a in self.axes)

    def render(self) -> str:
        parts: list[str] = []
        for a in self.axes:
            parts.extend(a.render())
        if self.base:
            parts.append(str(self.base))
        if not parts:
            return "0"
        return parts[0] if len(parts) == 1 else "(" + " + ".join(parts) + ")"

    def render_without(self, var: str) -> str:
        """Render the part of the offset that does not depend on ``var``.

        Used by reduction kernels to hoist the row offset out of the column
        loop, so a persistent kernel computes a row base once and then indexes
        it with the column vector.
        """
        parts: list[str] = []
        for a in self.axes:
            if a.var != var:
                parts.extend(a.render())
        if self.base:
            parts.append(str(self.base))
        if not parts:
            return "0"
        return parts[0] if len(parts) == 1 else "(" + " + ".join(parts) + ")"


def _simplify(space: Sequence[int], strides: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    out_strides = contiguous_strides(space)
    terms = [(space[d], out_strides[d], strides[d]) for d in range(len(space))]
    terms = [t for t in terms if t[0] != 1]

    changed = True
    while changed and len(terms) > 1:
        changed = False
        for d in range(len(terms) - 1):
            (n0, o0, i0), (n1, o1, i1) = terms[d], terms[d + 1]
            if o0 == o1 * n1 and i0 == i1 * n1:
                terms[d : d + 2] = [(n0 * n1, o1, i1)]
                changed = True
                break

    return tuple(t for t in terms if t[2] != 0)


def build_index(groups: Sequence[tuple[str, Sequence[int]]], layout: Layout) -> IndexMap:
    """Index map for ``layout`` under an iteration split into named groups.

    The concatenation of the group shapes must equal the layout's shape, which
    the caller arranges by permuting reduced dimensions to the end (a free
    restride) and expanding broadcasts.
    """
    full: list[int] = []
    for _, shape in groups:
        full.extend(int(d) for d in shape)
    full_shape = tuple(full)
    if layout.shape != full_shape:
        layout = layout.expand(full_shape)

    axes: list[Axis] = []
    pos = 0
    for var, shape in groups:
        shape = tuple(int(d) for d in shape)
        k = len(shape)
        strides = layout.strides[pos : pos + k]
        axes.append(Axis(var, math.prod(shape) if shape else 1, _simplify(shape, strides)))
        pos += k
    return IndexMap(tuple(axes), layout.offset)


def flat_index(space: Sequence[int], layout: Layout, var: str = "idx") -> IndexMap:
    """Single-axis index map: the pointwise case."""
    return build_index([(var, space)], layout)
