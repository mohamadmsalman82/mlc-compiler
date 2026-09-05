"""End-to-end: the compiled model must match eager, under every pass setting.

The configurations matter as much as the models. Each pass is toggled
independently so a regression points at a pass rather than at "the compiler",
and so the benchmark's per-pass attribution rests on configurations that are
known to be correct.
"""

import pytest
import torch

import mlc
from mlc.config import Config

from models import all_models

TOL = dict(rtol=1e-5, atol=1e-6)

CONFIGS = {
    "no_fusion": Config(elementwise_fusion=False, recompute=False,
                        reduction_fusion=False, memory_planning=False, cuda_graphs=False),
    "elementwise": Config(reduction_fusion=False, recompute=False,
                          memory_planning=False, cuda_graphs=False),
    "elementwise_recompute": Config(reduction_fusion=False, memory_planning=False,
                                    cuda_graphs=False),
    "planned": Config(reduction_fusion=False, cuda_graphs=False),
    "all_passes": Config(cuda_graphs=False),
    "streamed_reductions": Config(cuda_graphs=False, max_persistent_row=4),
}

CASES = [(name, cfg_name) for name, _, _ in all_models() for cfg_name in CONFIGS]


def _reference(model, args):
    model.eval()
    with torch.no_grad():
        return model(*args)


@pytest.mark.parametrize("model_name,cfg_name", CASES,
                         ids=[f"{m}-{c}" for m, c in CASES])
def test_matches_eager(model_name, cfg_name):
    torch.manual_seed(0)
    name, model, args = next(t for t in all_models() if t[0] == model_name)
    want = _reference(model, args)
    compiled = mlc.compile(model, args, CONFIGS[cfg_name])
    got = compiled(*args)
    torch.testing.assert_close(got, want, **TOL)


#: Configurations that must be bit-identical to each other. None of them
#: changes the order of any floating-point operation: they only change which
#: kernel a value is computed in, and where it lives while it waits.
EXACT = ["no_fusion", "elementwise", "elementwise_recompute", "planned"]

#: Configurations that reassociate reductions and so may differ in the last
#: bits. Fusing a reduction replaces torch's kernel with ours and computes the
#: variance in two passes rather than however torch does it; streaming
#: reassociates it again. That is the pass doing its job, so the bound here is
#: on drift, not on equality.
APPROXIMATE = ["all_passes", "streamed_reductions"]


@pytest.mark.parametrize("model_name", [m[0] for m in all_models()])
def test_reordering_free_passes_are_bit_identical(model_name):
    torch.manual_seed(0)
    name, model, args = next(t for t in all_models() if t[0] == model_name)
    base = mlc.compile(model, args, CONFIGS["no_fusion"])(*args)
    for cfg_name in EXACT[1:]:
        got = mlc.compile(model, args, CONFIGS[cfg_name])(*args)
        torch.testing.assert_close(
            got, base, rtol=0, atol=0,
            msg=lambda m, c=cfg_name: f"{c} is not bit-identical to no_fusion:\n{m}",
        )


@pytest.mark.parametrize("model_name", [m[0] for m in all_models()])
def test_reduction_fusion_stays_within_tolerance(model_name):
    torch.manual_seed(0)
    name, model, args = next(t for t in all_models() if t[0] == model_name)
    base = mlc.compile(model, args, CONFIGS["no_fusion"])(*args)
    for cfg_name in APPROXIMATE:
        got = mlc.compile(model, args, CONFIGS[cfg_name])(*args)
        torch.testing.assert_close(
            got, base, rtol=1e-5, atol=1e-6,
            msg=lambda m, c=cfg_name: f"{c} drifted from no_fusion:\n{m}",
        )


def test_repeated_calls_are_stable():
    """Buffers are reused between calls; a kernel that reads a stale
    intermediate would only show up on the second call."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "block")
    compiled = mlc.compile(model, args, CONFIGS["elementwise_recompute"])
    first = compiled(*args).clone()
    for _ in range(3):
        torch.testing.assert_close(compiled(*args), first, rtol=0, atol=0)


def test_rejects_wrong_input_shape():
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "mlp")
    compiled = mlc.compile(model, args, CONFIGS["elementwise"])
    with pytest.raises(Exception, match="compiled for"):
        compiled(torch.randn(7, 32))


def test_different_inputs_give_different_answers():
    """Guards against a compiled model that accidentally captured its example
    inputs as constants."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "mlp")
    compiled = mlc.compile(model, args, CONFIGS["elementwise"])
    other = (torch.randn_like(args[0]),)
    torch.testing.assert_close(compiled(*other), _reference(model, other), **TOL)
    assert not torch.allclose(compiled(*other), compiled(*args))


# -- output aliasing -------------------------------------------------------

PLANNED = Config(cuda_graphs=False)


def test_outputs_alias_buffers_the_next_call_overwrites():
    """Once the buffer table is resident, results are views onto storage the
    next call reuses. This is the same contract torch.compile's
    reduce-overhead mode has, and it is a footgun worth pinning: a test that
    compares two calls' results without cloning compares a tensor with
    itself."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "mlp")
    compiled = mlc.compile(model, args, PLANNED)
    other = (torch.randn_like(args[0]),)

    first = compiled(*args)
    kept = first.clone()
    second = compiled(*other)
    assert first.data_ptr() == second.data_ptr(), "expected the same storage"
    assert not torch.allclose(kept, second), "the two inputs should differ"
    torch.testing.assert_close(second, _reference(model, other), **TOL)
    torch.testing.assert_close(compiled(*args), kept, **TOL)


def test_planned_model_is_correct_across_alternating_inputs():
    """Alternating inputs through a resident buffer table, which is where a
    stale cached pointer or a missed input copy would show up."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "block")
    compiled = mlc.compile(model, args, PLANNED)
    a = args
    b = (torch.randn_like(args[0]),)
    want_a = _reference(model, a)
    want_b = _reference(model, b)
    for _ in range(3):
        torch.testing.assert_close(compiled(*a), want_a, **TOL)
        torch.testing.assert_close(compiled(*b), want_b, **TOL)


def test_input_is_copied_not_aliased():
    """Mutating the caller's tensor after a call must not change the result,
    and must not corrupt the next one."""
    torch.manual_seed(0)
    _, model, args = next(t for t in all_models() if t[0] == "mlp")
    compiled = mlc.compile(model, args, PLANNED)
    live = args[0].clone()
    want = _reference(model, (live.clone(),))
    got = compiled(live).clone()
    torch.testing.assert_close(got, want, **TOL)
    live.zero_()
    torch.testing.assert_close(compiled(args[0]), _reference(model, args), **TOL)
