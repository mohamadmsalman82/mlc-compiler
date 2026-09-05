"""Memory planning: the packing must be tight, and it must never alias.

Overlapping two buffers that are alive at the same time is a silent
miscompile -- the numbers just come out wrong -- so the overlap check is run
directly on every model and on randomised interval sets, not inferred from
the algorithm looking correct.
"""

import random

import pytest
import torch

import mlc
from mlc.config import Config
from mlc.ir.capture import capture
from mlc.ir.types import Buffer, Layout
from mlc.passes.memory_planning import (ALIGNMENT, LiveRange, compute_live_ranges,
                                        pack, peak_live_bytes, plan_memory, verify)

from models import all_models

PLANNED = Config(reduction_fusion=False, cuda_graphs=False)
UNPLANNED = Config(reduction_fusion=False, cuda_graphs=False, memory_planning=False)


def _compiled(name, cfg):
    _, model, args = next(t for t in all_models() if t[0] == name)
    return mlc.compile(model, args, cfg), model, args


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_no_two_live_buffers_share_bytes(name):
    compiled, _, _ = _compiled(name, PLANNED)
    plan = compiled.schedule.plan
    ranges = list(plan.live_ranges.values())
    problems = verify(ranges, plan.offsets)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_arena_is_between_the_bound_and_the_naive_total(name):
    compiled, _, _ = _compiled(name, PLANNED)
    plan = compiled.schedule.plan
    assert plan.arena_bytes >= plan.peak_bytes, "arena smaller than the lower bound"
    assert plan.arena_bytes <= plan.total_bytes, "planning made it worse than no planning"


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_live_ranges_are_well_formed(name):
    compiled, _, _ = _compiled(name, PLANNED)
    sched = compiled.schedule
    for name_, r in sched.plan.live_ranges.items():
        assert r.start <= r.end
        assert r.buffer.plannable
        assert r.size >= r.buffer.nbytes
        assert r.size % ALIGNMENT == 0
        writers = [i for i, k in enumerate(sched.kernels)
                   if any(v.buffer.name == name_ for v in k.writes())]
        assert writers and min(writers) == r.start


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_planning_does_not_change_results(name):
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == name)
    model.eval()
    with torch.no_grad():
        want = model(*args)
    planned = mlc.compile(model, args, PLANNED)(*args)
    unplanned = mlc.compile(model, args, UNPLANNED)(*args)
    torch.testing.assert_close(planned, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(planned, unplanned, rtol=0, atol=0)


def test_planning_actually_saves_memory():
    compiled, _, _ = _compiled("block", PLANNED)
    plan = compiled.schedule.plan
    assert plan.arena_bytes < plan.total_bytes * 0.6, plan.summary()


def test_layout_respects_a_non_zero_storage_offset():
    """Regression: as_strided takes an offset into the storage, not into the
    tensor it is called on. Every arena slice after the first has a non-zero
    storage offset, and getting this wrong reads the wrong bytes silently."""
    arena = torch.arange(64, dtype=torch.float32)
    piece = arena[16:32]
    lay = Layout.contiguous((4, 4))
    assert torch.equal(lay.as_torch(piece), arena[16:32].reshape(4, 4))
    strided = Layout((2, 4), (8, 1), 0)
    assert torch.equal(strided.as_torch(piece), arena[16:32].reshape(4, 4)[::2])


def _random_ranges(n, rng, horizon=40):
    out = []
    for i in range(n):
        start = rng.randrange(horizon)
        end = min(horizon, start + rng.randrange(1, 8))
        size = rng.choice([256, 512, 1024, 4096, 16384])
        out.append(LiveRange(Buffer(f"b{i}", (size // 4,), torch.float32), start, end, size))
    return out


@pytest.mark.parametrize("seed", range(25))
def test_packing_never_aliases_on_random_intervals(seed):
    rng = random.Random(seed)
    ranges = _random_ranges(rng.randrange(1, 40), rng)
    offsets, total = pack(ranges)
    assert not verify(ranges, offsets)
    assert total >= peak_live_bytes(ranges, 40)
    assert total <= sum(r.size for r in ranges)
    for r in ranges:
        assert offsets[r.buffer.name] % ALIGNMENT == 0
        assert offsets[r.buffer.name] + r.size <= total


def test_packing_is_optimal_on_a_chain():
    """Buffers that never coexist must all land at offset zero."""
    ranges = [LiveRange(Buffer(f"b{i}", (256,), torch.float32), i, i, 1024) for i in range(8)]
    offsets, total = pack(ranges)
    assert total == 1024
    assert set(offsets.values()) == {0}


def test_packing_stacks_simultaneous_buffers():
    ranges = [LiveRange(Buffer(f"b{i}", (256,), torch.float32), 0, 5, 1024) for i in range(4)]
    offsets, total = pack(ranges)
    assert total == 4096
    assert sorted(offsets.values()) == [0, 1024, 2048, 3072]


def test_graph_outputs_are_not_packed():
    """An output buffer has to survive the last kernel, so it must never be
    handed a slice something else reuses."""
    compiled, _, _ = _compiled("block", PLANNED)
    out_buffers = {v.buffer.name for v in compiled.graph.outputs}
    assert out_buffers
    assert not (out_buffers & set(compiled.schedule.plan.offsets))


def test_unplanned_mode_frees_as_it_goes():
    compiled, _, _ = _compiled("block", UNPLANNED)
    plan = compiled.schedule.plan
    assert plan.mode == "eager"
    assert plan.alloc_at and plan.free_at
    freed = {n for names in plan.free_at.values() for n in names}
    allocated = {b.name for bs in plan.alloc_at.values() for b in bs}
    assert freed <= allocated
    assert len(freed) >= len(allocated) - len(compiled.graph.outputs)
