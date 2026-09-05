"""Public entry point."""

from __future__ import annotations

from typing import Sequence

import torch

from .config import Config
from .ir.capture import capture
from .kernels import Schedule
from .lower import build_schedule
from .passes.fusion import plan
from .runtime.executor import CompiledModel


def compile(
    model: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    config: Config | None = None,
    device: str | torch.device | None = None,
) -> CompiledModel:
    """Compile ``model`` for exactly the shapes of ``example_inputs``.

    The result is callable with tensors of those shapes and no others: that
    restriction is what buys whole-graph memory planning and CUDA graph
    capture.
    """
    cfg = config or Config()
    device = torch.device(device) if device is not None else _infer_device(example_inputs)
    graph = capture(model, tuple(example_inputs))
    schedule = build_pipeline(graph, cfg)
    return CompiledModel(graph, schedule, cfg, device)


def build_pipeline(graph, cfg: Config) -> Schedule:
    """Run every pass, in order, over an already captured graph."""
    groups = plan(graph, cfg)
    schedule = build_schedule(graph, groups, cfg)
    if cfg.memory_planning:
        from .passes.memory_planning import plan_memory

        plan_memory(graph, schedule, cfg)
    return schedule


def _infer_device(inputs) -> torch.device:
    for t in inputs:
        if torch.is_tensor(t):
            return t.device
    return torch.device("cpu")
