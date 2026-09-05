"""Layout algebra is checked differentially against torch itself.

Every view op in the IR has a torch equivalent whose stride behaviour is the
specification. If the two ever disagree the compiler will silently read the
wrong memory, so these tests enumerate rather than sample.
"""

import itertools
import math

import pytest
import torch

from mlc.ir.types import Layout, contiguous_strides


def _tensors():
    """A spread of base tensors: contiguous, permuted, sliced, broadcast."""
    out = []
    for shape in [(4, 6, 8), (2, 3, 4, 5), (24, 8), (7,), (4, 1, 8)]:
        t = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        out.append(t)
        for perm in itertools.permutations(range(len(shape))):
            out.append(t.permute(perm))
        if len(shape) >= 2:
            out.append(t[..., : shape[-1] // 2])
            out.append(t[..., ::2])
    return out


def _layout_of(t: torch.Tensor) -> Layout:
    return Layout(tuple(t.shape), tuple(t.stride()), t.storage_offset())


@pytest.mark.parametrize("t", _tensors(), ids=lambda t: f"{tuple(t.shape)}{t.stride()}")
def test_reshape_matches_torch_view(t):
    """Our restride must accept exactly the reshapes torch.view accepts, and
    produce the same strides when it does."""
    lay = _layout_of(t)
    n = t.numel()
    candidates = {(n,), (1, n), (n, 1)}
    for d in range(1, n + 1):
        if n % d == 0:
            candidates.add((d, n // d))
    candidates.add(tuple(t.shape))
    for new in sorted(candidates):
        try:
            want = t.view(new)
            torch_ok = True
        except RuntimeError:
            torch_ok = False
        got = lay.reshape(new)
        assert torch_ok == (got is not None), f"{tuple(t.shape)}{t.stride()} -> {new}"
        if torch_ok:
            assert got.shape == tuple(want.shape)
            assert got.strides == tuple(want.stride())
            assert got.offset == want.storage_offset()


@pytest.mark.parametrize("t", _tensors()[:12], ids=lambda t: f"{tuple(t.shape)}{t.stride()}")
def test_view_ops_match_torch(t):
    storage = torch.arange(t.untyped_storage().nbytes() // 4, dtype=torch.float32)
    lay = _layout_of(t)

    for perm in itertools.permutations(range(t.dim())):
        assert torch.equal(lay.permute(perm).as_torch(storage), t.permute(perm))

    for dim in range(t.dim()):
        n = t.shape[dim]
        for start, end, step in [(0, n, 1), (0, max(n // 2, 1), 1), (1, n, 2)]:
            if start >= n:
                continue
            idx = [slice(None)] * t.dim()
            idx[dim] = slice(start, end, step)
            assert torch.equal(lay.slice(dim, start, end, step).as_torch(storage), t[tuple(idx)])

    for dim in range(t.dim() + 1):
        assert torch.equal(lay.unsqueeze(dim).as_torch(storage), t.unsqueeze(dim))


def test_expand_uses_zero_strides():
    lay = Layout.contiguous((1, 4))
    out = lay.expand([3, 4])
    assert out.shape == (3, 4)
    assert out.strides[0] == 0, "broadcast must not consume memory"
    storage = torch.arange(4.0)
    assert torch.equal(out.as_torch(storage), torch.arange(4.0).view(1, 4).expand(3, 4))


def test_expand_rejects_incompatible():
    with pytest.raises(ValueError):
        Layout.contiguous((3, 4)).expand([5, 4])


def test_contiguous_strides():
    assert contiguous_strides((2, 3, 4)) == (12, 4, 1)
    assert contiguous_strides(()) == ()
    assert Layout.contiguous((2, 3)).is_contiguous()
    assert not Layout.contiguous((2, 3)).permute([1, 0]).is_contiguous()


def test_max_offset_bounds_the_buffer():
    lay = Layout.contiguous((2, 8, 192)).slice(2, 128, 192)
    assert lay.max_offset() == 2 * 8 * 192
    assert Layout.contiguous((4, 4)).max_offset() == 16
