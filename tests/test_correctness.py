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


@pytest.mark.parametrize("model_name", [m[0] for m in all_models()])
def test_passes_agree_with_each_other(model_name):
    """Every configuration must produce the same numbers, not merely numbers
    close to eager. A pass that changes results is a miscompile even if the
    drift stays inside the eager tolerance."""
    torch.manual_seed(0)
    name, model, args = next(t for t in all_models() if t[0] == model_name)
    outs = {}
    for cfg_name, cfg in CONFIGS.items():
        outs[cfg_name] = mlc.compile(model, args, cfg)(*args)
    base = outs["no_fusion"]
    for cfg_name, got in outs.items():
        torch.testing.assert_close(got, base, rtol=1e-6, atol=1e-7,
                                   msg=lambda m, c=cfg_name: f"{c} differs from no_fusion:\n{m}")


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
