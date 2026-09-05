"""The benchmark models must compile and be correct, or the numbers mean
nothing. Only the small variants run here; the full-size ones are covered by
a shape and structure check that does not need a forward pass.
"""

import ast

import pytest
import torch

import mlc
from mlc.api import build_pipeline
from mlc.bench.models import SUITE, build
from mlc.codegen.triton_backend import generate_module
from mlc.config import Config
from mlc.ir.capture import capture
from mlc.kernels import ReductionKernel

SMALL = ["gpt-small", "bert-small"]
CFG = Config(cuda_graphs=False)


@pytest.mark.parametrize("name", SMALL)
def test_small_models_match_eager(name):
    model, args = build(name, batch=1, seq=32)
    with torch.no_grad():
        want = model(*args)
    got = mlc.compile(model, args, CFG)(*args)
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("name", sorted(SUITE))
def test_every_model_compiles_and_generates_valid_source(name):
    model, args = build(name, batch=1, seq=32)
    g = capture(model, args)
    sched = build_pipeline(g, CFG)
    assert len(sched) < len(g.nodes)
    ast.parse(generate_module(sched, CFG))


@pytest.mark.parametrize("name", sorted(SUITE))
def test_only_embeddings_stay_opaque(name):
    """Everything else must reach a generated kernel. An op quietly falling
    into the opaque bucket is how a compiler stops compiling."""
    model, args = build(name, batch=1, seq=32)
    g = capture(model, args)
    opaque = {n.op for n in g.nodes if n.meta.get("opaque")}
    assert opaque <= {"aten.embedding.default"}, opaque


@pytest.mark.parametrize("name", SMALL)
def test_softmax_and_layernorm_each_become_one_kernel(name):
    """Structural check on the real models, not just the toy ones."""
    model, args = build(name, batch=1, seq=32)
    sched = build_pipeline(capture(model, args), CFG)
    reductions = [k for k in sched.kernels if isinstance(k, ReductionKernel)]
    assert reductions
    kinds = [tuple(r.kind for r in k.reduces) for k in reductions]
    assert ("max", "sum") in kinds, "softmax should be one two-stage kernel"
    assert ("sum", "sum") in kinds, "layer norm should be one two-stage kernel"


@pytest.mark.parametrize("name", SMALL)
def test_memory_planning_beats_the_unplanned_total_by_a_lot(name):
    model, args = build(name, batch=1, seq=32)
    sched = build_pipeline(capture(model, args), CFG)
    plan = sched.plan
    assert plan.arena_bytes < plan.total_bytes / 4, plan.summary()
    assert plan.arena_bytes >= plan.peak_bytes


@pytest.mark.parametrize("batch", [1, 2])
def test_batch_size_changes_the_plan_but_not_the_answer(batch):
    model, args = build("gpt-small", batch=batch, seq=32)
    with torch.no_grad():
        want = model(*args)
    torch.testing.assert_close(mlc.compile(model, args, CFG)(*args), want,
                               rtol=2e-4, atol=2e-5)


def test_cli_run_verifies_against_eager():
    """The command the first GPU session will use. Exercised on CPU so a
    regression in it does not wait for hardware to surface."""
    from mlc.__main__ import main

    assert main(["run", "gpt-small", "--batch", "1", "--seq", "16"]) == 0


def test_cli_run_reports_failure_with_an_impossible_tolerance():
    from mlc.__main__ import main

    assert main(["run", "gpt-small", "--batch", "1", "--seq", "16",
                 "--tol", "0"]) == 1


def test_cli_views_all_work():
    from mlc.__main__ import main

    for cmd in (["graph", "gpt-small"], ["show", "gpt-small"],
                ["passes", "gpt-small"], ["source", "gpt-small"]):
        assert main(cmd + ["--batch", "1", "--seq", "16"]) == 0
