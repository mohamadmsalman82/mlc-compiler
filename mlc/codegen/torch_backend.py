"""Reference backend: execute kernel IR with torch tensor ops.

This exists so the parts of the compiler that are easy to get subtly wrong --
operator lowering rules, index arithmetic, which reads become register
references -- can be tested anywhere, including on a machine with no GPU.

It is deliberately literal. Index expressions are evaluated from the *same
rendered source string* the Triton backend emits, against an ``arange``, so a
bug in the rendering shows up here rather than at runtime on the GPU. That
makes it slow, which is fine: it is a correctness oracle, not a fast path.
"""

from __future__ import annotations

import torch

from ..ir.scalar import Call, Cast, Const, Expr, Load, Ref
from ..kernels import ExternKernel, PointwiseKernel, ReductionKernel

TORCH_FN = {
    "neg": lambda a: -a,
    "abs": torch.abs,
    "reciprocal": torch.reciprocal,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "erf": torch.erf,
    "sigmoid": torch.sigmoid,
    "floor": torch.floor,
    "logical_not": torch.logical_not,
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "pow": torch.pow,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "gt": torch.gt,
    "lt": torch.lt,
    "ge": torch.ge,
    "le": torch.le,
    "eq": torch.eq,
    "ne": torch.ne,
    "logical_and": torch.logical_and,
    "logical_or": torch.logical_or,
    "where": torch.where,
}

#: Compute precision. Triton accumulates in fp32 for narrow float inputs and
#: this backend matches it, so the two agree bit-for-bit far more often than
#: they would if this one used the input dtype.
COMPUTE_DTYPE = torch.float32


def _promote(t: torch.Tensor) -> torch.Tensor:
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.to(COMPUTE_DTYPE)
    return t


def eval_expr(e: Expr, loads: list[torch.Tensor], refs: dict[str, torch.Tensor],
              like: torch.Tensor) -> torch.Tensor:
    if isinstance(e, Load):
        return loads[e.slot]
    if isinstance(e, Ref):
        return refs[e.name]
    if isinstance(e, Const):
        if isinstance(e.value, bool):
            return torch.full_like(like, float(e.value), dtype=torch.bool)
        return torch.as_tensor(e.value, dtype=e.dtype, device=like.device)
    if isinstance(e, Cast):
        return eval_expr(e.x, loads, refs, like).to(e.dtype)
    if isinstance(e, Call):
        fn = TORCH_FN.get(e.fn)
        if fn is None:
            raise KeyError(f"no torch equivalent for scalar function {e.fn!r}")
        args = [eval_expr(a, loads, refs, like) for a in e.args]
        return fn(*args)
    raise TypeError(f"cannot evaluate {type(e).__name__}")


def _gather(buf: torch.Tensor, index_src: str, env: dict) -> torch.Tensor:
    """Evaluate a rendered index expression and gather with it.

    ``index_src`` is the exact string the Triton backend puts in the kernel.
    Running it here means the rendering itself is under test.
    """
    if index_src == "idx" and buf.numel() == env["idx"].numel():
        return buf
    offsets = eval(index_src, {"__builtins__": {}}, env)  # noqa: S307 - generated source
    if not torch.is_tensor(offsets):
        offsets = torch.full_like(env["idx"], int(offsets))
    return buf[offsets]


def run_pointwise(k: PointwiseKernel, buffers: dict[str, torch.Tensor]) -> None:
    device = next(iter(buffers.values())).device if buffers else torch.device("cpu")
    idx = torch.arange(k.numel, device=device, dtype=torch.int64)
    env = {"idx": idx}

    loads = [_promote(_gather(buffers[a.value.buffer.name], a.index.render(), env))
             for a in k.inputs]
    like = loads[0] if loads else idx
    refs: dict[str, torch.Tensor] = {}
    for name, expr in k.body:
        refs[name] = eval_expr(expr, loads, refs, like)

    for a in k.outputs:
        val = eval_expr(k.out_expr[a.value.name], loads, refs, like)
        dest = buffers[a.value.buffer.name]
        src = index_target(a.index.render(), env, k.numel)
        if not torch.is_tensor(val) or val.numel() == 1:
            val = torch.as_tensor(val, device=device).expand(k.numel)
        dest[src] = val.reshape(-1).to(dest.dtype)


def index_target(index_src: str, env: dict, numel: int) -> torch.Tensor:
    offsets = eval(index_src, {"__builtins__": {}}, env)  # noqa: S307 - generated source
    if not torch.is_tensor(offsets):
        offsets = torch.full_like(env["idx"], int(offsets))
    return offsets


def run_reduction(k: ReductionKernel, buffers: dict[str, torch.Tensor]) -> None:
    """Evaluate a persistent reduction kernel row by row, vectorised over the
    row dimension exactly as the Triton kernel is over the column dimension."""
    device = next(iter(buffers.values())).device if buffers else torch.device("cpu")
    n_rows, n_cols = k.n_rows, k.reduce_numel
    row = torch.arange(n_rows, device=device, dtype=torch.int64).unsqueeze(1)
    col = torch.arange(n_cols, device=device, dtype=torch.int64).unsqueeze(0)
    env = {"row": row, "col": col}

    loads = []
    for a in k.inputs:
        src = a.index.render()
        buf = buffers[a.value.buffer.name]
        offsets = eval(src, {"__builtins__": {}}, env)  # noqa: S307 - generated source
        if not torch.is_tensor(offsets):
            offsets = torch.zeros_like(row) + int(offsets)
        loads.append(_promote(buf[offsets.expand(n_rows, n_cols)]))

    like = loads[0] if loads else row.expand(n_rows, n_cols)
    refs: dict[str, torch.Tensor] = {}
    for name, expr in k.prologue:
        refs[name] = eval_expr(expr, loads, refs, like)
    for stage in k.stages:
        v = eval_expr(stage.expr, loads, refs, like)
        v = v.expand(n_rows, n_cols) if v.dim() < 2 else v
        if stage.kind == "sum":
            acc = v.sum(dim=1, keepdim=True)
        elif stage.kind == "max":
            acc = v.amax(dim=1, keepdim=True)
        elif stage.kind == "min":
            acc = v.amin(dim=1, keepdim=True)
        elif stage.kind == "prod":
            acc = v.prod(dim=1, keepdim=True)
        else:
            raise ValueError(f"unknown reduce kind {stage.kind}")
        refs[stage.name] = acc
    for name, expr in k.epilogue:
        refs[name] = eval_expr(expr, loads, refs, like)

    for a in k.outputs:
        val = eval_expr(k.out_expr[a.value.name], loads, refs, like)
        dest = buffers[a.value.buffer.name]
        is_row = a.value.name in k.row_outputs
        src = a.index.render()
        if is_row:
            offsets = eval(src, {"__builtins__": {}}, {"row": row, "col": torch.zeros_like(row)})
            dest[offsets.reshape(-1)] = val.reshape(-1).to(dest.dtype)
        else:
            offsets = eval(src, {"__builtins__": {}}, env)
            val = val.expand(n_rows, n_cols) if val.dim() < 2 or val.shape[1] == 1 else val
            dest[offsets.expand(n_rows, n_cols).reshape(-1)] = val.reshape(-1).to(dest.dtype)
