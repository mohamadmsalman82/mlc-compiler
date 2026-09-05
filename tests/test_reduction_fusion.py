"""Reduction fusion: both directions, the legality rules, and the two kernel
shapes it can produce.

The streamed path is exercised by lowering ``max_persistent_row`` so that even
a 32-element row does not fit. That runs the same pass partition and the same
online-softmax recurrence the GPU would, on a CPU, which is the only way to
test it here.
"""

import ast

import pytest
import torch
import torch.nn as nn

import mlc
from mlc.api import build_pipeline
from mlc.codegen.reduction_schedule import (detect_online_softmax, plan_passes,
                                            scalar_binds, varies_by_column)
from mlc.codegen.triton_backend import generate_module
from mlc.config import Config
from mlc.ir.capture import capture
from mlc.kernels import Bind, PointwiseKernel, Reduce, ReductionKernel
from mlc.passes.reduction_fusion import (ELEMENT, ROW, Frame, classify, frame_of,
                                         storable)

from models import all_models

FULL = Config(cuda_graphs=False)
NO_REDUCTION = Config(reduction_fusion=False, cuda_graphs=False)
STREAMED = Config(cuda_graphs=False, max_persistent_row=4)


def _sched(name, cfg):
    _, model, args = next(t for t in all_models() if t[0] == name)
    g = capture(model, args)
    return g, build_pipeline(g, cfg)


def _reductions(sched):
    return [k for k in sched.kernels if isinstance(k, ReductionKernel)]


# -- the two fusion patterns the pass exists for ---------------------------

def test_softmax_becomes_one_kernel():
    """Producers into the reduction: the scale and the mask before the
    softmax, plus both reductions and the final divide."""
    _, sched = _sched("softmax", FULL)
    assert len(sched) == 1
    k = sched.kernels[0]
    assert isinstance(k, ReductionKernel)
    assert [r.kind for r in k.reduces] == ["max", "sum"]
    ops = [n.op.split(".")[-2] for n in k.nodes]
    assert ops == ["mul", "add", "amax", "sub", "exp", "sum", "div"]
    assert len(k.outputs) == 1, "only the final result should reach memory"


def test_layernorm_fuses_with_its_affine_transform():
    """Reduction into consumers: the normalisation and both affine scalings
    read the row that is already resident."""
    _, sched = _sched("layernorm_scale", FULL)
    assert len(sched) == 1
    k = sched.kernels[0]
    assert isinstance(k, ReductionKernel)
    assert [r.kind for r in k.reduces] == ["sum", "sum"], "mean and variance, one row"
    assert len(k.outputs) == 1


def test_sibling_reductions_share_one_pass():
    """Layer norm's mean and variance both reduce the same axis of the same
    tensor. Fused, the row is read once instead of twice."""
    _, sched = _sched("layernorm_scale", FULL)
    k = _reductions(sched)[0]
    row_reads = [a for a in k.inputs if a.index.varies_along("col")
                 and a.index.varies_along("row")]
    assert len(row_reads) == 1, "the row should be loaded exactly once"


def test_reduction_fusion_reduces_kernel_count():
    for name, _, _ in all_models():
        _, without = _sched(name, NO_REDUCTION)
        _, with_it = _sched(name, FULL)
        assert len(with_it) <= len(without), name


def test_block_kernel_count_drops():
    _, without = _sched("block", NO_REDUCTION)
    _, with_it = _sched("block", FULL)
    assert len(with_it) < len(without)


# -- legality --------------------------------------------------------------

def test_frame_classifies_keepdim_and_broadcast_apart():
    """A keepdim reduction result [B, T, 1] and a weight vector [D] both
    broadcast to [B, T, D], but one is a row value and the other is not."""
    frame = Frame((2, 4, 8), (2,))
    assert frame.row == (2, 4) and frame.col == (8,)
    assert classify((2, 4, 8), frame) is ELEMENT
    assert classify((8,), frame) is ELEMENT
    assert classify((2, 4, 1), frame) is ROW
    assert classify((2, 4), frame) is ROW
    assert classify((), frame) is ROW
    assert classify((3, 3, 3), frame) is None


def test_frame_handles_a_non_trailing_reduced_axis():
    frame = Frame((2, 4, 8), (1,))
    assert frame.row == (2, 8) and frame.col == (4,)
    assert frame.order == (0, 2, 1)
    assert classify((2, 1, 8), frame) is ROW
    assert classify((2, 4, 8), frame) is ELEMENT


def test_storable_rejects_broadcast_values():
    frame = Frame((2, 4, 8), (2,))
    assert storable((2, 4, 8), frame)
    assert storable((2, 4), frame)
    assert storable((2, 4, 1), frame)
    assert not storable((8,), frame), "broadcast along the row: several writers per address"


def test_reductions_over_different_axes_do_not_fuse():
    class TwoAxes(nn.Module):
        def forward(self, x):
            return x.sum(-1).unsqueeze(-1) + x.sum(-2).unsqueeze(-2)

    g = capture(TwoAxes(), (torch.randn(4, 6, 8),))
    sched = build_pipeline(g, FULL)
    reds = _reductions(sched)
    assert len(reds) == 2, "different frames cannot share an iteration space"
    frames = {(k.row_space, k.reduce_space) for k in reds}
    assert len(frames) == 2


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_reduction_kernels_are_internally_consistent(name):
    _, sched = _sched(name, FULL)
    for k in _reductions(sched):
        assert k.n_rows * k.reduce_numel > 0
        for a in k.outputs:
            if a.value.name in k.row_outputs:
                assert not a.index.varies_along("col")
                assert a.value.numel == k.n_rows
            else:
                assert a.value.numel == k.n_rows * k.reduce_numel
        bound = set()
        for s in k.steps:
            from mlc.ir.scalar import refs
            assert refs(s.expr) <= bound, f"{s.name} reads an unbound temp"
            bound.add(s.name)
        for e in k.out_expr.values():
            from mlc.ir.scalar import refs
            assert refs(e) <= bound


# -- numerics --------------------------------------------------------------

@pytest.mark.parametrize("name", [m[0] for m in all_models()])
@pytest.mark.parametrize("cfg_name", ["full", "streamed"])
def test_matches_eager(name, cfg_name):
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == name)
    model.eval()
    with torch.no_grad():
        want = model(*args)
    cfg = FULL if cfg_name == "full" else STREAMED
    torch.testing.assert_close(mlc.compile(model, args, cfg)(*args), want,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_streaming_agrees_with_persistent(name):
    """The streamed kernel is a different program computing the same thing.
    Online softmax in particular reassociates the sum, so this bounds the
    drift that introduces."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == name)
    persistent = mlc.compile(model, args, FULL)(*args)
    streamed = mlc.compile(model, args, STREAMED)(*args)
    torch.testing.assert_close(streamed, persistent, rtol=1e-5, atol=1e-6)


def test_softmax_is_stable_on_large_logits():
    """Subtracting the row max is what keeps this finite. A softmax that
    exponentiated first would overflow to nan here."""

    class M(nn.Module):
        def forward(self, x):
            return x.softmax(dim=-1)

    x = torch.randn(4, 64) * 100
    for cfg in (FULL, STREAMED):
        got = mlc.compile(M(), (x,), cfg)(x)
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, x.softmax(dim=-1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(got.sum(-1), torch.ones(4), rtol=1e-5, atol=1e-6)


def test_variance_uses_the_two_pass_formula_not_the_cancelling_one():
    """With a mean of 1e4 and a variance of 1, ``E[x^2] - E[x]^2`` subtracts
    two numbers near 1e8 to get 1. In fp32 the ulp at 1e8 is about 8, so the
    answer is noise. The two-pass form -- mean first, then the sum of squared
    deviations from it -- keeps full precision, and this checks that against a
    float64 reference rather than against torch, whose own summation order is
    not the thing under test.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 128) + 1e4
    exact = x.double().var(dim=-1, unbiased=False)

    naive = (x.pow(2).mean(-1) - x.mean(-1).pow(2)).double()
    naive_err = (naive - exact).abs().max().item()

    class Var(nn.Module):
        def forward(self, z):
            return z.var(dim=-1, unbiased=False)

    for cfg in (FULL, STREAMED):
        got = mlc.compile(Var(), (x,), cfg)(x).double()
        err = (got - exact).abs().max().item()
        assert err < 1e-3, f"two-pass variance drifted by {err}"
        assert err < naive_err / 100, (
            f"two-pass error {err} is not decisively better than the "
            f"cancelling form's {naive_err}"
        )


def test_layernorm_on_a_large_mean_stays_accurate():
    """The same input through a full layer norm, against a float64 reference."""
    torch.manual_seed(0)
    x = torch.randn(8, 128) + 1e4
    model = nn.LayerNorm(128).eval()
    with torch.no_grad():
        exact = torch.nn.functional.layer_norm(
            x.double(), (128,), model.weight.double(), model.bias.double(), model.eps
        )
    for cfg in (FULL, STREAMED):
        got = mlc.compile(model, (x,), cfg)(x).double()
        assert torch.isfinite(got).all()
        assert (got - exact).abs().max().item() < 5e-3


# -- online softmax --------------------------------------------------------

def test_online_softmax_is_detected_only_for_the_real_pattern():
    _, sched = _sched("softmax", STREAMED)
    k = _reductions(sched)[0]
    assert detect_online_softmax(k) is not None
    _, ln = _sched("layernorm_scale", STREAMED)
    assert detect_online_softmax(_reductions(ln)[0]) is None


def test_online_softmax_saves_a_pass():
    _, sched = _sched("softmax", STREAMED)
    k = _reductions(sched)[0]
    passes = plan_passes(k)
    assert len(passes) == 2, "one fused reduce pass plus the store pass"
    assert passes[0].online_softmax is not None
    assert len(passes[0].reduces) == 2


def test_persistent_kernel_is_a_single_pass():
    _, sched = _sched("softmax", FULL)
    k = _reductions(sched)[0]
    assert not k.two_pass
    assert len(plan_passes(k)) == 1


def test_scalar_and_vector_steps_are_separated():
    _, sched = _sched("layernorm_scale", STREAMED)
    k = _reductions(sched)[0]
    vector = varies_by_column(k)
    scalars = {b.name for b in scalar_binds(k)}
    assert vector and scalars
    assert not (vector & scalars)


# -- generated source ------------------------------------------------------

@pytest.mark.parametrize("name", [m[0] for m in all_models()])
@pytest.mark.parametrize("cfg_name", ["full", "streamed"])
def test_generated_reduction_source_parses(name, cfg_name):
    cfg = FULL if cfg_name == "full" else STREAMED
    _, sched = _sched(name, cfg)
    ast.parse(generate_module(sched, cfg))


def test_persistent_source_has_no_loop():
    _, sched = _sched("softmax", FULL)
    src = generate_module(sched, FULL)
    assert "for _base" not in src
    assert "tl.max(" in src and "tl.sum(" in src


def test_streamed_source_loops_and_rescales():
    _, sched = _sched("softmax", STREAMED)
    src = generate_module(sched, STREAMED)
    assert "for _base" in src
    assert "_m_new = tl.maximum(_m, _x)" in src
    assert "_alpha" in src, "the online rescale must be there"


def test_masked_reduce_uses_the_right_identity():
    """A max over a partially masked block must see -inf in the dead lanes,
    not the zero a load's `other` would supply."""
    _, sched = _sched("softmax", FULL)
    src = generate_module(sched, FULL)
    assert 'tl.where(cmask, t1, -float("inf"))' in src


def test_row_statistics_load_as_scalars():
    """A value that does not vary along the column axis is one word, not a
    vector, and needs no mask."""

    class M(nn.Module):
        def forward(self, x, rowbias):
            return (x - x.amax(-1, keepdim=True)).exp().sum(-1) * rowbias

    g = capture(M(), (torch.randn(6, 32), torch.randn(6)))
    sched = build_pipeline(g, FULL)
    src = generate_module(sched, FULL)
    scalar_loads = [l for l in src.splitlines()
                    if "tl.load" in l and "mask=" not in l and "col" not in l]
    assert scalar_loads, src
