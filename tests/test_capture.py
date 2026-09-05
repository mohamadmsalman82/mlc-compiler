"""The frontend must produce a well-formed graph whose types agree with torch."""

import pytest
import torch

from mlc.ir import shapes
from mlc.ir.capture import capture
from mlc.ir.ops import OpClass, op_class

from models import all_models

CASES = all_models()
IDS = [c[0] for c in CASES]


@pytest.fixture(scope="module", params=CASES, ids=IDS)
def captured(request):
    name, model, args = request.param
    return name, capture(model, args), model, args


def test_graph_is_well_formed(captured):
    _, g, _, _ = captured
    g.topo_check()
    assert g.outputs, "graph produced no outputs"
    assert g.nodes


def test_propagated_types_match_export(captured):
    """Our own shape and dtype rules, checked against the exporter's
    FakeTensor metadata on every non-opaque node."""
    _, g, _, _ = captured
    problems = shapes.verify_against_export(g)
    assert not problems, "\n".join(problems)


def test_views_alias_their_source(captured):
    """A view node must never allocate. If one does, memory planning will
    double-count and codegen will emit a pointless copy."""
    _, g, _, _ = captured
    for n in g.nodes:
        if op_class(n.op) is OpClass.VIEW:
            src = n.args[0]
            assert n.out.buffer is src.buffer, f"{n.op} allocated a new buffer"
            assert n.out.layout.max_offset() <= src.buffer.numel


def test_every_param_has_a_tensor(captured):
    _, g, _, _ = captured
    for p in g.params:
        assert p.name in g.param_tensors
        assert tuple(g.param_tensors[p.name].shape) == p.shape


def test_no_dynamic_shapes(captured):
    _, g, _, _ = captured
    for v in g.values():
        assert all(isinstance(d, int) for d in v.shape)


def test_opaque_ops_are_the_exception(captured):
    """Coverage guard: if a change to the registry starts dropping ops into
    the opaque bucket, the fusion passes go quiet and this catches it."""
    name, g, _, _ = captured
    opaque = [n.op for n in g.nodes if n.meta.get("opaque")]
    assert not opaque, f"{name} has unlowered ops: {sorted(set(opaque))}"
