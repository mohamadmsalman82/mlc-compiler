"""Half precision, which is what these models actually run in on a GPU.

The convention both backends follow is: narrow floats are loaded into fp32,
computed in fp32, and cast back on store. Reductions in particular are
unusable in fp16. These tests pin that convention and check the generated
source reflects it, because getting it wrong shows up as slow drift rather
than as a failure.
"""

import re

import pytest
import torch
import torch.nn as nn

import mlc
from mlc.api import build_pipeline
from mlc.codegen.triton_backend import generate_module
from mlc.config import Config
from mlc.ir.capture import capture

from models import all_models

CFG = Config(cuda_graphs=False)
HALF_MODELS = ["pointwise", "broadcasting", "mlp", "softmax", "layernorm_scale",
               "attention", "block", "residual_chain"]


def _halve(model, args):
    model = model.half().eval()
    args = tuple(a.half() if a.is_floating_point() else a for a in args)
    return model, args


@pytest.mark.parametrize("name", HALF_MODELS)
def test_half_precision_matches_eager(name):
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == name)
    model, args = _halve(model, args)
    with torch.no_grad():
        want = model(*args)
    got = mlc.compile(model, args, CFG)(*args)
    assert got.dtype == want.dtype
    torch.testing.assert_close(got, want, rtol=1e-2, atol=5e-2)


@pytest.mark.parametrize("name", HALF_MODELS)
def test_half_precision_is_close_to_the_float32_answer(name):
    """A kernel that accumulated a reduction in fp16 would pass the test above
    against eager and still be much less accurate than it should be, so this
    checks against the fp32 result too."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == name)
    with torch.no_grad():
        exact = model.float().eval()(*args)
    half_model, half_args = _halve(model, args)
    got = mlc.compile(half_model, half_args, CFG)(*half_args).float()
    torch.testing.assert_close(got, exact, rtol=2e-2, atol=5e-2)


def test_narrow_loads_promote_and_stores_cast_back():
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "layernorm_scale")
    model, args = _halve(model, args)
    src = generate_module(build_pipeline(capture(model, args), CFG), CFG)
    loads = [l for l in src.splitlines() if "tl.load" in l]
    assert loads and all(".to(tl.float32)" in l for l in loads), loads
    stores = [l for l in src.splitlines() if "tl.store" in l]
    assert stores and all(".to(tl.float16)" in l for l in stores), stores


def test_reductions_accumulate_in_fp32():
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "softmax")
    model, args = _halve(model, args)
    src = generate_module(build_pipeline(capture(model, args), CFG), CFG)
    for line in src.splitlines():
        if "tl.sum(" in line or "tl.max(" in line:
            assert "float16" not in line, f"reduction in half precision: {line}"


def test_no_op_casts_do_not_reach_the_source():
    """Half-precision layer norm decomposes into an fp32 computation bracketed
    by conversions. Casting a value already in fp32 registers to fp32 is dead
    and must be folded away."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "layernorm_scale")
    model, args = _halve(model, args)
    src = generate_module(build_pipeline(capture(model, args), CFG), CFG)
    body = [l for l in src.splitlines() if re.match(r"\s+t\d+ = \(v\d+\)\.to\(tl\.float32\)", l)]
    assert not body, body


def test_float64_still_works():
    """Not a target, but nothing should special-case fp32 in a way that
    breaks a wider type."""

    class M(nn.Module):
        def forward(self, x):
            return (x * 2.0).softmax(dim=-1)

    x = torch.randn(4, 8, dtype=torch.float64)
    with torch.no_grad():
        want = M()(x)
    torch.testing.assert_close(mlc.compile(M(), (x,), CFG)(x), want)


def test_bool_masks_survive_the_pipeline():
    class M(nn.Module):
        def forward(self, x, m):
            return torch.where(m > 0, x, torch.zeros_like(x)).sum(-1)

    x, m = torch.randn(4, 8), torch.rand(4, 8)
    with torch.no_grad():
        want = M()(x, m)
    torch.testing.assert_close(mlc.compile(M(), (x, m), CFG)(x, m), want,
                               rtol=1e-5, atol=1e-6)


def test_half_precision_matmuls_are_not_upcast():
    """Regression, and the one that would have quietly ruined a benchmark.

    Torch's addmm decomposition is wrapped in a cast-for-opmath: in half
    precision it upcasts both operands to fp32, runs the matmul there, and
    casts back. Forcing addmm apart to expose its bias add therefore turned
    every fp16 GEMM into an fp32 GEMM plus two conversion passes over the
    weights, giving up the tensor cores. The compiler splits addmm itself now,
    so the dtypes survive.
    """
    from mlc.bench.models import build

    for name in ("gpt-small", "bert-small"):
        model, args = build(name, 1, 32, "cpu", torch.float16)
        g = capture(model, args)
        matmuls = [n for n in g.nodes
                   if n.op in ("aten.mm.default", "aten.bmm.default",
                               "aten.addmm.default", "aten.baddbmm.default")]
        assert matmuls
        wrong = [n for n in matmuls if n.out.dtype != torch.float16]
        assert not wrong, f"{name}: {len(wrong)}/{len(matmuls)} matmuls left fp32"

        upcast_weights = [
            n for n in g.nodes
            if "_to_copy" in n.op
            and n.args[0].dtype == torch.float16 and n.out.dtype == torch.float32
            and any(c.op.endswith("mm.default")
                    for c in g.nodes if id(c) in _consumers(g, n.out))
        ]
        assert not upcast_weights, f"{name}: {len(upcast_weights)} operands upcast for a matmul"


def _consumers(g, value):
    from mlc.passes.scheduler import UseInfo

    return UseInfo(g).consumers(value)


def test_half_precision_does_not_cost_extra_kernels():
    """fp16 and fp32 should compile to the same schedule shape. A gap means
    conversions are failing to fuse somewhere."""
    from mlc.api import build_pipeline
    from mlc.bench.models import build

    counts = {}
    for dt in (torch.float32, torch.float16):
        model, args = build("gpt-small", 1, 32, "cpu", dt)
        counts[dt] = len(build_pipeline(capture(model, args), CFG))
    assert counts[torch.float16] <= counts[torch.float32] + 2, counts


def test_addmm_still_exposes_its_bias_add_to_fusion():
    """Splitting addmm by hand has to keep the reason for splitting it."""
    from mlc.api import build_pipeline
    from mlc.kernels import PointwiseKernel, ReductionKernel

    model = nn.Sequential(nn.Linear(16, 32), nn.GELU()).eval()
    sched = build_pipeline(capture(model, (torch.randn(8, 16),)), CFG)
    fused = [k for k in sched.kernels
             if isinstance(k, (PointwiseKernel, ReductionKernel))]
    ops = {n.op.split(".")[-2] for k in fused for n in k.nodes}
    assert "add" in ops and "erf" in ops, (
        "the bias add should share a kernel with the activation"
    )
    assert len(fused) == 1
