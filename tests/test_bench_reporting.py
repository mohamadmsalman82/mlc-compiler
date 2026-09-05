"""The reporting path, which had no coverage and shipped two bugs because of it.

Both were the same shape: a field added to Result, and a formatter that still
assumed the old one. Neither showed up until a GPU sweep had already spent
minutes measuring, and then threw the measurements away.
"""

import json

import pytest

from mlc.bench.runner import Result, format_table, to_json


def _rows(model="bert-base", dtype="float16", batch=1):
    spec = [("eager", 6.0, 0), ("torch.compile", 3.4, 0),
            ("torch.compile/reduce-overhead", 3.5, 0),
            ("mlc/no-passes", 34.0, 682), ("mlc/elementwise", 24.4, 322),
            ("mlc/+reduction", 11.4, 200), ("mlc/+memory", 7.4, 200),
            ("mlc/+cuda-graphs", 1.08, 200)]
    return [Result(model, batch, 128, v, dtype, latency_ms=ms, latency_p10=ms * 0.98,
                   peak_mb=234.0, kernels=k, max_err=2.9e-3, ok=True)
            for v, ms, k in spec]


def test_format_table_renders_every_row():
    text = format_table(_rows())
    assert "bert-base" in text and "float16" in text
    for variant in ("eager", "torch.compile", "mlc/+cuda-graphs"):
        assert variant in text
    assert "5.56x" in text, "speedup against eager should be reported"


def test_format_table_survives_a_failed_variant():
    rows = _rows()
    rows[1].ok = False
    rows[1].note = "OutOfMemoryError"
    text = format_table(rows)
    assert "OutOfMemoryError" in text
    assert "mlc/+cuda-graphs" in text


def test_format_table_survives_a_missing_baseline():
    rows = [r for r in _rows() if r.variant != "eager"]
    text = format_table(rows)
    assert "mlc/+cuda-graphs" in text


def test_format_table_groups_by_dtype_and_batch():
    rows = _rows(batch=1) + _rows(batch=8) + _rows(dtype="float32")
    text = format_table(rows)
    assert text.count("### bert-base") == 3


def test_attribution_ladder_is_present():
    text = format_table(_rows())
    assert "Per-pass contribution" in text
    assert "mlc/elementwise" in text


def test_result_key_matches_what_the_formatter_unpacks():
    """The exact mismatch that broke two sweeps."""
    r = _rows()[0]
    assert len(r.key()) == 4


def test_json_round_trips():
    rows = _rows()
    parsed = json.loads(to_json(rows))
    assert len(parsed) == len(rows)
    assert parsed[0]["dtype"] == "float16"
    assert {"model", "batch", "seq", "variant", "latency_ms", "peak_mb"} <= set(parsed[0])


# -- extern binding must not execute ---------------------------------------

def test_binding_an_extern_call_does_not_execute_it():
    """Resolving the launch plan happens before any kernel has run, so every
    intermediate buffer still holds uninitialised memory. A binder that tried
    the call to see whether it worked ran an embedding lookup against garbage
    indices and tripped a device-side assert."""
    import torch

    import mlc
    from mlc.config import Config
    from mlc.runtime.executor import CompiledModel, bind_extern
    from mlc.ir.capture import capture
    from mlc.api import build_pipeline
    from mlc.kernels import ExternKernel

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(16, 8)

        def forward(self, ids):
            return self.emb(ids % 16) * 2.0

    args = (torch.zeros(4, dtype=torch.long),)
    cfg = Config(cuda_graphs=False)
    g = capture(M().eval(), args)
    sched = build_pipeline(g, cfg)
    c = CompiledModel(g, sched, cfg, torch.device("cpu"))
    buffers = c._resident_buffers(args)
    # Poison every intermediate so any execution during binding misbehaves.
    for name, t in buffers.items():
        if t.dtype == torch.long and name not in {v.buffer.name for v in g.inputs}:
            t.fill_(10_000)
    for k in sched.kernels:
        if isinstance(k, ExternKernel):
            bind_extern(k, buffers)  # must not raise, must not run the op
