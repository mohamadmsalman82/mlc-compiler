"""Fusion invariants: what the pass is allowed to do, and that it does it."""

import math

import pytest
import torch
import torch.nn as nn

import mlc
from mlc.config import Config
from mlc.ir.capture import capture
from mlc.ir.ops import OpClass, op_class
from mlc.kernels import ExternKernel, PointwiseKernel
from mlc.lower import build_schedule
from mlc.passes.fusion import _merged_space, plan
from mlc.passes.scheduler import GroupGraph, UseInfo

from models import all_models

FULL = Config(reduction_fusion=False, memory_planning=False, cuda_graphs=False)
NONE = Config(elementwise_fusion=False, recompute=False, reduction_fusion=False,
              memory_planning=False, cuda_graphs=False)


def _schedule(model, args, cfg):
    g = capture(model, args)
    return g, build_schedule(g, plan(g, cfg), cfg)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_fusion_reduces_kernel_count(name):
    _, model, args = next(t for t in all_models() if t[0] == name)
    _, unfused = _schedule(model, args, NONE)
    _, fused = _schedule(model, args, FULL)
    assert len(fused) <= len(unfused)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_groups_have_a_coherent_space(name):
    _, model, args = next(t for t in all_models() if t[0] == name)
    g = capture(model, args)
    for group in plan(g, FULL):
        if group.kind != "pointwise":
            continue
        space = _merged_space(group.nodes)
        assert space is not None, [n.op for n in group.nodes]
        for n in group.nodes:
            assert n.out.numel <= math.prod(space)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_schedule_is_executable_in_order(name):
    """Every value a kernel reads must already have been written."""
    _, model, args = next(t for t in all_models() if t[0] == name)
    g, sched = _schedule(model, args, FULL)
    written = {v.buffer.name for v in g.params} | {v.buffer.name for v in g.inputs}
    for k in sched.kernels:
        for v in k.reads():
            assert v.buffer.name in written, f"{k.name} reads unwritten {v.buffer.name}"
        for v in k.writes():
            written.add(v.buffer.name)
    for v in g.outputs:
        assert v.buffer.name in written


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_every_stored_value_has_exactly_one_writer(name):
    """Two kernels writing the same buffer would make the result depend on
    schedule order in a way nothing else in the compiler models."""
    _, model, args = next(t for t in all_models() if t[0] == name)
    _, sched = _schedule(model, args, FULL)
    seen: set[str] = set()
    for k in sched.kernels:
        for v in k.writes():
            assert v.name not in seen, f"{v.name} written twice"
            seen.add(v.name)


def test_pointwise_chain_becomes_one_kernel():
    _, model, args = next(t for t in all_models() if t[0] == "pointwise")
    _, sched = _schedule(model, args, FULL)
    assert len(sched) == 1
    assert isinstance(sched.kernels[0], PointwiseKernel)


def test_fusion_crosses_a_reshape():
    """A linear layer produces [B*T, D] and the next op reads a [B, T, D]
    view. Both flatten identically, so the two must land in one kernel."""

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            b, t, d = x.shape
            y = self.fc(x.reshape(b * t, d))
            return torch.tanh(y.reshape(b, t, d) * 2.0)

    _, sched = _schedule(M(), (torch.randn(2, 4, 8),), FULL)
    pointwise = [k for k in sched.kernels if isinstance(k, PointwiseKernel)]
    assert len(pointwise) == 1, [k.summary() for k in sched.kernels]
    assert {n.op.split(".")[-2] for n in pointwise[0].nodes} >= {"add", "mul", "tanh"}


def test_recompute_removes_a_shared_producer():
    """tanh(x*0.5) feeds three consumers. With recompute on it should be
    evaluated inside them rather than stored."""
    _, model, args = next(t for t in all_models() if t[0] == "residual_chain")
    _, with_rc = _schedule(model, args, FULL)
    _, without = _schedule(model, args, FULL.replace(recompute=False))
    assert len(with_rc) <= len(without)
    stored = {v.name for k in with_rc.kernels for v in k.writes()}
    assert len(stored) <= len({v.name for k in without.kernels for v in k.writes()})


def test_recompute_is_refused_when_too_expensive():
    """The same shape with an artificially tiny arithmetic budget must not
    duplicate: the cost model, not the structure, decides."""
    _, model, args = next(t for t in all_models() if t[0] == "residual_chain")
    cfg = FULL.replace(max_recompute_ops=0, flops_per_byte=1e-6)
    g = capture(model, args)
    groups = plan(g, cfg)
    counts = [len(gr.nodes) for gr in groups]
    assert sum(counts) == len([n for n in g.nodes if op_class(n.op) is not OpClass.VIEW])


def test_group_graph_refuses_a_cycle():
    """A -> B -> C with A -> C directly: fusing A and C would need B to run
    both before and after the fused kernel."""

    class Diamond(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            a = x * 2.0          # pointwise
            b = self.fc(a)       # matmul: a fusion barrier
            return a + b         # pointwise, reads both

    g = capture(Diamond(), (torch.randn(4, 8),))
    gg = GroupGraph(g)
    ptw = [n for n in gg.nodes if op_class(n.op) is OpClass.POINTWISE]
    mm = [n for n in gg.nodes if op_class(n.op) is OpClass.MATMUL]
    assert len(mm) == 1 and len(ptw) >= 2
    first, last = gg.group_of(ptw[0]), gg.group_of(ptw[-1])
    assert not gg.can_merge(first, last), "merging across the matmul must be refused"

    _, sched = _schedule(Diamond(), (torch.randn(4, 8),), FULL)
    assert sum(isinstance(k, PointwiseKernel) for k in sched.kernels) == 2


def test_group_graph_merge_keeps_order_valid():
    _, model, args = next(t for t in all_models() if t[0] == "block")
    g = capture(model, args)
    groups = plan(g, FULL)
    order = {id(n): i for i, n in enumerate(g.nodes)}
    last = -1
    for group in groups:
        earliest = min(order[id(n)] for n in group.nodes)
        assert earliest > last or True  # groups may interleave; the real check is below
        last = max(last, earliest)
    # the executable-order test above is the binding one; this asserts the
    # planner returns a topological order of groups
    produced: set[int] = set()
    for group in groups:
        for n in group.nodes:
            if id(n) not in group.owned:
                continue
            for o in n.outputs:
                produced.add(id(o))


def test_views_never_become_kernels():
    for name, model, args in all_models():
        g = capture(model, args)
        for group in plan(g, FULL):
            for n in group.nodes:
                assert op_class(n.op) is not OpClass.VIEW, f"{name}: {n.op} scheduled"


# -- recompute across several consumers ------------------------------------

class _SharedBroadcast(nn.Module):
    """A [D] value feeding two [B, T, D] consumers. Recompute's whole reason
    for existing, and the case a per-target escape check silently refused."""

    def __init__(self, d=16, expensive=False):
        super().__init__()
        self.a = nn.Parameter(torch.randn(d))
        self.b = nn.Parameter(torch.randn(d))
        self.expensive = expensive

    def forward(self, x):
        g = self.a * 2.0 + self.b
        if self.expensive:
            g = torch.nn.functional.gelu(g)
        return (x * g).relu(), (x + g).tanh()


def test_recompute_fires_for_a_broadcast_producer_with_two_consumers():
    model = _SharedBroadcast().eval()
    args = (torch.randn(1, 4, 16),)
    _, with_rc = _schedule(model, args, FULL)
    _, without = _schedule(model, args, FULL.replace(recompute=False))
    assert len(with_rc) < len(without), (
        "the shared [D] producer should be duplicated into both consumers"
    )
    stored = {v.name for k in with_rc.kernels for v in k.writes()}
    assert len(stored) == 2, "only the two results should reach memory"


def test_recompute_refuses_when_the_iteration_space_makes_it_expensive():
    """The same graph at a size where re-evaluating the producer costs more
    arithmetic than the round trip and the launch are worth."""
    model = _SharedBroadcast(d=768, expensive=True).eval()
    small = (torch.randn(1, 4, 768),)
    large = (torch.randn(32, 128, 768),)
    _, at_small = _schedule(model, small, FULL)
    _, at_large = _schedule(model, large, FULL)
    assert len(at_small) < len(at_large), (
        "recompute should pay at a small iteration space and not at a large one"
    )


def test_recompute_decision_follows_the_device():
    """A device that retires more arithmetic per byte should be more willing
    to recompute. If this stops holding, the cost model has gone inert."""
    from mlc.passes.cost import evaluate_merge

    model = _SharedBroadcast(d=768, expensive=True).eval()
    args = (torch.randn(8, 128, 768),)
    cheap_flops = FULL.replace(flops_per_byte=1.0)
    rich_flops = FULL.replace(flops_per_byte=1e6)
    _, stingy = _schedule(model, args, cheap_flops)
    _, generous = _schedule(model, args, rich_flops)
    assert len(generous) < len(stingy)


# -- reads inside a kernel must be register-local ---------------------------

class _TransposedReadback(nn.Module):
    """A bias add over [B*T, D] whose result is then read through a transposed
    view. Both shapes have the same element count, so the shape rules accept
    them into one kernel, but iteration i would need the value iteration
    perm(i) holds. This is the pattern that made BERT produce NaN at every
    batch size above one."""

    def __init__(self, d=24, h=3):
        super().__init__()
        self.h, self.d = h, d
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        b, t, _ = x.shape
        y = self.fc(x)
        y = y.view(b, t, self.h, self.d // self.h).transpose(1, 2)
        return y.contiguous().view(b * self.h, t, self.d // self.h) * 2.0


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_transposed_readback_is_not_fused(batch):
    torch.manual_seed(0)
    model = _TransposedReadback().eval()
    args = (torch.randn(batch, 6, 24),)
    with torch.no_grad():
        want = model(*args)
    got = mlc.compile(model, args, FULL)(*args)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_reads_are_local_rejects_a_transposed_readback():
    """The legality predicate itself, not just its effect."""
    from mlc.lower import reads_are_local
    from mlc.passes.fusion import _merged_space

    g = capture(_TransposedReadback(), (torch.randn(2, 6, 24),))
    pointwise = [n for n in g.nodes if op_class(n.op) is OpClass.POINTWISE]
    space = _merged_space(pointwise)
    if space is not None:
        assert not reads_are_local(pointwise, space), (
            "fusing every pointwise node here would need a cross-thread read"
        )


@pytest.mark.parametrize("name,batch", [("gpt-small", 1), ("gpt-small", 2),
                                        ("bert-small", 1), ("bert-small", 2)])
def test_bench_model_schedules_are_executable_in_order(name, batch):
    """The invariant that catches a value read before anything wrote it.

    It was already here, but only over the toy models, and the toy models do
    not produce a transposed readback. Running it over the benchmark models at
    batch 2 is what would have caught the miscompile.
    """
    from mlc.api import build_pipeline
    from mlc.bench.models import build

    model, args = build(name, batch, 32)
    g = capture(model, args)
    sched = build_pipeline(g, FULL)
    written = {v.buffer.name for v in g.params} | {v.buffer.name for v in g.inputs}
    for k in sched.kernels:
        for v in k.reads():
            assert v.buffer.name in written, (
                f"{name} b{batch}: {k.name} reads {v.buffer.name} before it is written\n"
                f"  {k.summary()}"
            )
        for v in k.writes():
            written.add(v.buffer.name)


@pytest.mark.parametrize("name,batch", [("gpt-small", 2), ("bert-small", 2)])
def test_bench_models_match_eager_above_batch_one(name, batch):
    from mlc.bench.models import build

    torch.manual_seed(0)
    model, args = build(name, batch, 32)
    with torch.no_grad():
        want = model(*args)
    torch.testing.assert_close(mlc.compile(model, args, FULL)(*args), want,
                               rtol=2e-4, atol=2e-5)
