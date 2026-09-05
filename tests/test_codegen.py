"""The generated Triton source is an artefact, so it gets tested like one.

None of this runs a kernel -- that needs a GPU. What it checks is everything
up to the launch: that the module is valid Python, that its launch table
covers every kernel, that the constant folding actually reached the source,
and that the index expressions in it agree with the ones the reference
backend executes.
"""

import ast
import re

import pytest
import torch

from mlc.api import build_pipeline
from mlc.codegen.index import build_index, flat_index
from mlc.codegen.triton_backend import (choose_block, generate_module,
                                        next_pow2, render_const)
from mlc.config import Config
from mlc.ir.capture import capture
from mlc.ir.types import Layout
from mlc.kernels import ExternKernel, PointwiseKernel

from models import all_models

CFG = Config(reduction_fusion=False, memory_planning=False, cuda_graphs=False)


def _module(name):
    _, model, args = next(t for t in all_models() if t[0] == name)
    g = capture(model, args)
    sched = build_pipeline(g, CFG)
    return sched, generate_module(sched, CFG)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_generated_module_is_valid_python(name):
    _, src = _module(name)
    ast.parse(src)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_launch_table_covers_every_kernel(name):
    sched, src = _module(name)
    tree = ast.parse(src)
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    expected = {k.name for k in sched.kernels if not isinstance(k, ExternKernel)}
    assert expected <= defined
    for kname in expected:
        assert re.search(rf"'{kname}': dict\(grid=", src), f"{kname} missing from LAUNCH"


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_kernel_signature_matches_operand_count(name):
    sched, src = _module(name)
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for k in sched.kernels:
        if isinstance(k, ExternKernel):
            continue
        params = [a.arg for a in fns[k.name].args.args]
        assert sum(p.startswith("in_ptr") for p in params) == len(k.inputs)
        assert sum(p.startswith("out_ptr") for p in params) == len(k.outputs)


@pytest.mark.parametrize("name", [m[0] for m in all_models()])
def test_generation_is_deterministic(name):
    _, a = _module(name)
    _, b = _module(name)
    assert a == b


def test_multiply_by_one_never_reaches_the_source():
    """addmm decomposes to ``mm * 1 + bias * 1``. Both must fold away."""
    _, src = _module("mlp")
    body = [l for l in src.splitlines() if re.match(r"\s+t\d+ = ", l)]
    assert body
    assert not any(re.search(r"\* 1\)", l) for l in body), body


def test_mask_is_omitted_when_the_block_divides():
    _, src = _module("mlp")
    assert "no bounds mask needed" in src
    for fn in src.split("@triton.jit")[1:]:
        if "no bounds mask needed" in fn:
            assert "mask=" not in fn


def test_block_choice():
    assert choose_block(128, 1024) == (128, False)
    assert choose_block(100, 1024)[1] is True
    block, masked = choose_block(4096, 1024)
    assert block == 1024 and masked is False
    block, masked = choose_block(1000000, 1024)
    assert 1000000 % block == 0 or masked
    assert next_pow2(1) == 1 and next_pow2(33) == 64


def test_const_rendering():
    assert render_const(1.5, None) == "1.5"
    assert render_const(float("inf"), None) == 'float("inf")'
    assert render_const(-float("inf"), None) == '-float("inf")'
    assert render_const(True, torch.bool) == "1"


def test_index_rendering_shapes():
    assert flat_index((2, 8, 64), Layout.contiguous((2, 8, 64))).render() == "idx"
    assert flat_index((2, 8, 64), Layout.contiguous((64,))).render() == "(idx % 64)"
    assert flat_index((2, 8, 64), Layout.contiguous((2, 8, 1))).render() == "(idx // 64)"
    m = build_index([("row", (6,)), ("col", (32,))], Layout.contiguous((6, 32)))
    assert m.render() == "(row * 32 + col)"
    assert m.render_without("col") == "row * 32"


def test_index_map_equality_sees_through_reshape():
    """The property fusion relies on: two views of one contiguous buffer with
    the same element count index identically."""
    a = flat_index((16, 192), Layout.contiguous((16, 192)))
    b = flat_index((2, 8, 192), Layout.contiguous((2, 8, 192)))
    assert a == b
    c = flat_index((16, 192), Layout.contiguous((192, 16)).permute([1, 0]))
    assert a != c, "a transposed view must not compare equal"


def test_broadcast_drops_out_of_the_column_axis():
    """A row statistic read back across its row must not appear in the column
    index; that is what makes the reduction epilogue free."""
    stat = Layout.contiguous((6, 1)).expand((6, 32))
    m = build_index([("row", (6,)), ("col", (32,))], stat)
    assert not m.varies_along("col")
    assert m.render() == "row"
