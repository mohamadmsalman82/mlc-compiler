"""Runtime: allocate buffers, run kernels, hand back outputs.

Every buffer is held as a flat 1-D tensor. Shapes and strides live in the IR,
not in the storage, so a view costs an ``as_strided`` at the boundary and
nothing at all inside a generated kernel. It is also what makes the arena
possible: once the planner assigns byte offsets, every intermediate is a slice
of one allocation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ..codegen import torch_backend
from ..config import Config
from ..ir.graph import Graph, Value
from ..ir.types import Buffer
from ..kernels import ExternKernel, PointwiseKernel, ReductionKernel, Schedule


class ExecutionError(Exception):
    pass


def select_backend(cfg: Config, device: torch.device) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    if device.type == "cuda":
        try:
            import triton  # noqa: F401

            return "triton"
        except ImportError:
            return "torch"
    return "torch"


class CompiledModel:
    """A compiled graph, callable like the module it came from."""

    def __init__(self, graph: Graph, schedule: Schedule, cfg: Config,
                 device: torch.device) -> None:
        self.graph = graph
        self.schedule = schedule
        self.cfg = cfg
        self.device = torch.device(device)
        self.backend = select_backend(cfg, self.device)

        # Parameters are resident: uploaded once, flattened once.
        self.params: dict[str, torch.Tensor] = {}
        for v in graph.params:
            t = graph.param_tensors[v.name]
            self.params[v.buffer.name] = t.to(self.device).contiguous().reshape(-1)

        self._arena: torch.Tensor | None = None
        self._module = None
        if self.backend == "triton":
            from .triton_runtime import TritonRuntime

            self._module = TritonRuntime(schedule, cfg, self.device)

        self._graph_replay = None
        self._static_inputs: list[torch.Tensor] | None = None
        self._static_outputs: Any = None

    # -- allocation --------------------------------------------------------
    def _bind_resident(self, inputs: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Buffers that exist for the whole call: params, inputs, outputs."""
        buffers = dict(self.params)
        if len(inputs) != len(self.graph.inputs):
            raise ExecutionError(
                f"expected {len(self.graph.inputs)} inputs, got {len(inputs)}"
            )
        for v, t in zip(self.graph.inputs, inputs):
            if tuple(t.shape) != v.shape:
                raise ExecutionError(
                    f"input {v.name}: compiled for {v.shape}, called with {tuple(t.shape)}"
                )
            buffers[v.buffer.name] = t.to(self.device).contiguous().reshape(-1)
        for b in self.graph.buffers():
            if b.name not in buffers and not b.plannable:
                buffers[b.name] = torch.empty(b.numel, dtype=b.dtype, device=self.device)
        return buffers

    def _bind_arena(self, buffers: dict[str, torch.Tensor]) -> None:
        """Point every planned buffer at its slice of the one allocation."""
        plan = self.schedule.plan
        arena = self._get_arena()
        for b in self.graph.buffers():
            if b.name in buffers:
                continue
            offset = plan.offsets.get(b.name) if plan is not None else None
            if offset is None:
                buffers[b.name] = torch.empty(b.numel, dtype=b.dtype, device=self.device)
            else:
                buffers[b.name] = arena[offset : offset + b.nbytes].view(b.dtype)

    def _get_arena(self) -> torch.Tensor:
        need = max(self.schedule.arena_bytes, 1)
        if self._arena is None or self._arena.numel() < need:
            self._arena = torch.empty(need, dtype=torch.uint8, device=self.device)
        return self._arena

    # -- execution ---------------------------------------------------------
    def run_kernels(self, buffers: dict[str, torch.Tensor]) -> None:
        for k in self.schedule.kernels:
            self.run_one(k, buffers)

    def run_one(self, k, buffers: dict[str, torch.Tensor]) -> None:
        if isinstance(k, ExternKernel):
            run_extern(k, buffers)
        elif self._module is not None:
            self._module.launch(k, buffers)
        elif isinstance(k, PointwiseKernel):
            torch_backend.run_pointwise(k, buffers)
        elif isinstance(k, ReductionKernel):
            torch_backend.run_reduction(k, buffers)
        else:
            raise ExecutionError(f"cannot execute {type(k).__name__}")

    def _run_eager(self, buffers: dict[str, torch.Tensor], plan) -> None:
        """Allocate each intermediate when it is written, drop it when it is
        last read. This is the honest unplanned baseline: it leans on torch's
        caching allocator exactly the way a compiler without a planner would,
        rather than reserving everything up front."""
        for i, k in enumerate(self.schedule.kernels):
            for b in plan.alloc_at.get(i, ()):
                buffers[b.name] = torch.empty(b.numel, dtype=b.dtype, device=self.device)
            self.run_one(k, buffers)
            for name in plan.free_at.get(i, ()):
                buffers.pop(name, None)

    def forward(self, *inputs: torch.Tensor):
        buffers = self._bind_resident(inputs)
        plan = self.schedule.plan
        if plan is not None and plan.mode == "eager":
            self._run_eager(buffers, plan)
        else:
            self._bind_arena(buffers)
            self.run_kernels(buffers)
        return self._collect(buffers)

    def _collect(self, buffers: dict[str, torch.Tensor]):
        outs = [v.layout.as_torch(buffers[v.buffer.name]) for v in self.graph.outputs]
        spec = self.graph.meta.get("out_spec")
        if spec is not None:
            from torch.utils._pytree import tree_unflatten

            try:
                return tree_unflatten(outs, spec)
            except Exception:
                pass
        return outs[0] if len(outs) == 1 else tuple(outs)

    def __call__(self, *inputs: torch.Tensor):
        if self.cfg.cuda_graphs and self.device.type == "cuda":
            return self._call_captured(inputs)
        return self.forward(*inputs)

    # -- CUDA graphs -------------------------------------------------------
    def _call_captured(self, inputs):
        """Replay the whole schedule as one CUDA graph.

        Static shapes plus a planned arena mean every pointer and every launch
        parameter is fixed after the first call, which is exactly the
        precondition for graph capture. The payoff is that per-kernel launch
        overhead stops scaling with kernel count, which is what dominates at
        batch size 1.
        """
        if self._graph_replay is None:
            self._capture(inputs)
        assert self._static_inputs is not None
        for dst, src in zip(self._static_inputs, inputs):
            dst.copy_(src.to(self.device, non_blocking=True).contiguous().reshape(-1))
        self._graph_replay.replay()
        return self._static_outputs

    def _capture(self, inputs) -> None:
        buffers = self._bind_resident(inputs)
        static = [buffers[v.buffer.name] for v in self.graph.inputs]
        # Capture needs every pointer fixed for the life of the graph, so the
        # eager alloc/free plan is not usable here: bind everything up front.
        self._bind_arena(buffers)

        # Warm up on a side stream. Kernels have to be compiled and cuBLAS
        # workspaces allocated before capture, or capture records the
        # allocation instead of the work.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.run_kernels(buffers)
        torch.cuda.current_stream().wait_stream(side)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.run_kernels(buffers)
        self._graph_replay = g
        self._static_inputs = static
        self._static_outputs = self._collect(buffers)

    # -- introspection -----------------------------------------------------
    def source(self) -> str:
        if self._module is None:
            return "# torch reference backend: no generated source"
        return self._module.source

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"CompiledModel({self.graph.name}, {len(self.schedule)} kernels, "
                f"backend={self.backend}, device={self.device})")


def run_extern(k: ExternKernel, buffers: dict[str, torch.Tensor]) -> None:
    """Dispatch one op to torch, reading and writing through the flat buffers."""
    node = k.node
    target = node.meta.get("target")
    if target is None:
        raise ExecutionError(f"{node.op} has no dispatch target")

    def materialise(a):
        if isinstance(a, Value):
            return a.layout.as_torch(buffers[a.buffer.name])
        if isinstance(a, (list, tuple)):
            return type(a)(materialise(x) for x in a)
        return a

    args = [materialise(a) for a in node.args]
    kwargs = {k2: materialise(v) for k2, v in node.kwargs.items()}
    result = target(*args, **kwargs)
    results = result if isinstance(result, (list, tuple)) else [result]
    for out, r in zip(node.outputs, results):
        dest = buffers[out.buffer.name]
        dest.copy_(r.contiguous().reshape(-1))
