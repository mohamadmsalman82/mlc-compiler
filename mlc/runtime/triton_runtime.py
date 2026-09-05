"""Compile and launch the generated Triton module.

The module is written to a real file rather than ``exec``-ed from a string.
Triton's ``@triton.jit`` calls ``inspect.getsource`` on every kernel it
compiles, and that fails outright on a function whose module has no file
behind it. Writing it out also means the generated code is on disk to read,
which is where most of the debugging happens.

Files are named by a hash of their contents, so recompiling the same schedule
reuses the file and Triton's own on-disk cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import sys
import types

import torch

from ..codegen.triton_backend import generate_module
from ..config import Config
from ..kernels import ExternKernel, Kernel, Schedule


def cache_dir() -> pathlib.Path:
    root = os.environ.get("MLC_CACHE_DIR")
    path = pathlib.Path(root) if root else pathlib.Path.home() / ".cache" / "mlc"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_module(source: str) -> pathlib.Path:
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    path = cache_dir() / f"mlc_kernels_{digest}.py"
    if not path.exists() or path.read_text() != source:
        # Write then rename, so a second process never imports a partial file.
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(source)
        tmp.replace(path)
    return path


def load_module(path: pathlib.Path) -> types.ModuleType:
    name = f"mlc_generated.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: Triton resolves the kernel's module by name
    # when it goes looking for the source.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TritonRuntime:
    def __init__(self, schedule: Schedule, cfg: Config, device: torch.device) -> None:
        self.source = generate_module(schedule, cfg)
        self.path = write_module(self.source)
        self.module = load_module(self.path)
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
