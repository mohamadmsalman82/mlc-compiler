"""Buffers and layouts.

The compiler separates *storage* from *view*:

  * A :class:`Buffer` is a contiguous block of memory. It is the unit the
    memory planner allocates and the unit a kernel writes to.
  * A :class:`Layout` is a (shape, strides, offset) triple describing how to
    read a buffer. View ops -- ``view``, ``permute``, ``expand``,
    ``as_strided``, ``slice`` -- produce a new Layout over an *existing*
    Buffer and cost nothing at runtime.

This split is what lets fusion see through a transpose. If views were
materialised as copies, an attention block would break into a dozen kernels
with a copy between each one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

Shape = tuple[int, ...]
Strides = tuple[int, ...]


def contiguous_strides(shape: Sequence[int]) -> Strides:
    strides: list[int] = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


@dataclass(frozen=True)
class Layout:
    shape: Shape
    strides: Strides
    offset: int = 0

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.strides):
            raise ValueError(f"rank mismatch: shape={self.shape} strides={self.strides}")
        if any(d < 0 for d in self.shape):
            raise ValueError(f"negative dim in {self.shape}")

    # -- construction ------------------------------------------------------
    @staticmethod
    def contiguous(shape: Sequence[int], offset: int = 0) -> "Layout":
        shape = tuple(int(d) for d in shape)
        return Layout(shape, contiguous_strides(shape), offset)

    # -- queries -----------------------------------------------------------
    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    def is_contiguous(self) -> bool:
        return self.offset == 0 and self.strides == contiguous_strides(self.shape)

    def is_dense_from_zero(self) -> bool:
        """True when the layout covers ``[0, numel)`` of its buffer exactly."""
        return self.is_contiguous()

    def max_offset(self) -> int:
        """One past the largest element offset this layout can address."""
        end = self.offset
        for d, s in zip(self.shape, self.strides):
            if d > 0:
                end += (d - 1) * s
        return end + 1

    # -- view algebra ------------------------------------------------------
    def permute(self, dims: Sequence[int]) -> "Layout":
        dims = [d % self.rank for d in dims]
        if sorted(dims) != list(range(self.rank)):
            raise ValueError(f"permute({dims}) is not a permutation of rank {self.rank}")
        return Layout(
            tuple(self.shape[d] for d in dims),
            tuple(self.strides[d] for d in dims),
            self.offset,
        )

    def expand(self, shape: Sequence[int]) -> "Layout":
        """Broadcast to ``shape``. ``-1`` keeps the existing size."""
        shape = list(shape)
        if len(shape) < self.rank:
            raise ValueError(f"cannot expand rank {self.rank} to {len(shape)} dims")
        pad = len(shape) - self.rank
        old_shape = (1,) * pad + self.shape
        old_strides = (0,) * pad + self.strides
        out_shape, out_strides = [], []
        for i, want in enumerate(shape):
            have = old_shape[i]
            if want == -1:
                want = have
            if have == want:
                out_shape.append(want)
                out_strides.append(old_strides[i])
            elif have == 1:
                # Broadcast: stride 0 means every index maps to the same element.
                out_shape.append(want)
                out_strides.append(0)
            else:
                raise ValueError(f"cannot expand dim {i} from {have} to {want}")
        return Layout(tuple(out_shape), tuple(out_strides), self.offset)

    def unsqueeze(self, dim: int) -> "Layout":
        dim = dim if dim >= 0 else dim + self.rank + 1
        stride = self.strides[dim] * self.shape[dim] if dim < self.rank else 1
        return Layout(
            self.shape[:dim] + (1,) + self.shape[dim:],
            self.strides[:dim] + (stride,) + self.strides[dim:],
            self.offset,
        )

    def squeeze(self, dims: Iterable[int] | None = None) -> "Layout":
        if dims is None:
            keep = [i for i, d in enumerate(self.shape) if d != 1]
        else:
            drop = {d % self.rank for d in dims}
            keep = [i for i in range(self.rank) if not (i in drop and self.shape[i] == 1)]
        return Layout(
            tuple(self.shape[i] for i in keep),
            tuple(self.strides[i] for i in keep),
            self.offset,
        )

    def slice(self, dim: int, start: int, end: int, step: int = 1) -> "Layout":
        dim %= self.rank
        n = self.shape[dim]
        start = max(0, min(n, start if start >= 0 else start + n))
        end = max(start, min(n, end if end >= 0 else end + n))
        length = (end - start + step - 1) // step
        shape = list(self.shape)
        strides = list(self.strides)
        shape[dim] = length
        strides[dim] = self.strides[dim] * step
        return Layout(tuple(shape), tuple(strides), self.offset + start * self.strides[dim])

    def reshape(self, shape: Sequence[int]) -> "Layout | None":
        """Reshape without copying, or ``None`` if that is impossible.

        A reshape is expressible as a restride whenever the new shape only
        splits and merges dimensions that are already contiguous *runs* in the
        old layout. That is more general than requiring the whole tensor be
        contiguous, and the difference matters: the qkv projection in an
        attention block slices a [B, T, 3D] buffer and then reshapes each
        slice to [B, T, H, D/H]. The slice is strided, but the reshape only
        splits its last dim, so it is still free. Falling back to a copy there
        would cost three extra kernels and three extra buffers per block.

        This is the same run-detection algorithm torch uses to decide whether
        ``Tensor.view`` can succeed.
        """
        shape = tuple(int(d) for d in shape)
        if math.prod(shape) != self.numel:
            raise ValueError(f"reshape {self.shape} -> {shape} changes element count")
        if self.numel == 0:
            return Layout(shape, contiguous_strides(shape), self.offset)
        strides = _restride(self.shape, self.strides, shape)
        return None if strides is None else Layout(shape, strides, self.offset)

    def as_torch(self, storage: torch.Tensor) -> torch.Tensor:
        """Materialise this layout as a view of a flat 1-D ``storage`` tensor.

        ``as_strided`` takes an offset into the *storage*, not into the tensor
        it is called on, so the tensor's own storage offset has to be added.
        It is zero for a standalone allocation and non-zero for a slice of the
        memory arena, which is exactly the case that would otherwise read the
        wrong bytes with no error.
        """
        return storage.as_strided(self.shape, self.strides,
                                  self.offset + storage.storage_offset())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        tag = "" if self.is_contiguous() else f" strides={self.strides} off={self.offset}"
        return f"Layout{self.shape}{tag}"


def _restride(old_shape: Shape, old_strides: Strides, new_shape: Shape) -> Strides | None:
    """Strides for ``new_shape`` viewing the same storage, or None.

    Walks the old shape from the fastest-moving dim outward, cutting it into
    maximal contiguous runs. Each run must be consumed exactly by a whole
    number of new dims; if a new dim straddles a discontinuity, no restride
    exists and the caller has to copy.
    """
    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    chunk_base = old_strides[-1] if old_shape else 1
    tensor_numel = 1
    view_numel = 1
    for d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[d]
        run_ends = d == 0 or (
            old_shape[d - 1] != 1 and old_strides[d - 1] != tensor_numel * chunk_base
        )
        if not run_ends:
            continue
        while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
            new_strides[view_d] = view_numel * chunk_base
            view_numel *= new_shape[view_d]
            view_d -= 1
        if view_numel != tensor_numel:
            return None
        if d > 0:
            chunk_base = old_strides[d - 1]
            tensor_numel = 1
            view_numel = 1
    return tuple(new_strides) if view_d == -1 else None


# --------------------------------------------------------------------------
# Buffers
# --------------------------------------------------------------------------

#: Buffers the planner may pack into the arena. Everything else is either
#: supplied by the caller or lives for the whole program.
PLANNABLE = ("intermediate",)

BUFFER_KINDS = ("input", "param", "intermediate", "output", "constant")


@dataclass(eq=False)
class Buffer:
    name: str
    shape: Shape
    dtype: torch.dtype
    kind: str = "intermediate"
    #: Set by the memory planner: byte offset into the shared arena.
    arena_offset: int | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        self.shape = tuple(int(d) for d in self.shape)
        if self.kind not in BUFFER_KINDS:
            raise ValueError(f"bad buffer kind {self.kind!r}")

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def itemsize(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def plannable(self) -> bool:
        return self.kind in PLANNABLE

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Buffer({self.name}, {self.shape}, {str(self.dtype).replace('torch.','')}, {self.kind})"
