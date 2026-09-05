"""Compile and launch the generated Triton module."""

from __future__ import annotations

import types

import torch

from ..codegen.triton_backend import generate_module
from ..config import Config
from ..kernels import ExternKernel, Kernel, Schedule


class TritonRuntime:
    def __init__(self, schedule: Schedule, cfg: Config, device: torch.device) -> None:
        self.source = generate_module(schedule, cfg)
        self.module = types.ModuleType("mlc_generated")
        self.module.__dict__["__file__"] = "<mlc-generated>"
        exec(compile(self.source, "<mlc-generated>", "exec"), self.module.__dict__)
        self.launch_table = self.module.LAUNCH

    def launch(self, k: Kernel, buffers: dict[str, torch.Tensor]) -> None:
        if isinstance(k, ExternKernel):
            raise TypeError("extern kernels do not go through the Triton runtime")
        fn = getattr(self.module, k.name)
        spec = self.launch_table[k.name]
        ptrs = [buffers[a.value.buffer.name] for a in k.inputs]
        ptrs += [buffers[a.value.buffer.name] for a in k.outputs]
        fn[spec["grid"]](
            *ptrs,
            **spec["constants"],
            num_warps=spec["num_warps"],
            num_stages=spec["num_stages"],
        )
