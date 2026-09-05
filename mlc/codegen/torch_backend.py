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
from ..kernels import Bind, ExternKernel, PointwiseKernel, Reduce, ReductionKernel

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


def run_reduction(k: ReductionKernel, buffers: dict[str, torch.Tensor],
                  chunk: int = 1024) -> None:
    """Evaluate a reduction kernel, vectorised over rows.

    Two paths, mirroring the two the Triton backend emits. The persistent path
    holds the whole row and walks the steps in order, which is the only thing
    that works when a later reduce reads an earlier one. The streamed path
    follows the pass partition from :mod:`mlc.codegen.reduction_schedule`,
    including its online-softmax fusion, so lowering ``max_persistent_row`` in
    a test exercises the streamed code on a CPU.
    """
    device = next(iter(buffers.values())).device if buffers else torch.device("cpu")
    n_rows, n_cols = k.n_rows, k.reduce_numel
    row = torch.arange(n_rows, device=device, dtype=torch.int64).unsqueeze(1)
    zero_col = torch.zeros((1, 1), device=device, dtype=torch.int64)

    def gather(arg, col):
        buf = buffers[arg.value.buffer.name]
        offsets = eval(arg.index.render(), {"__builtins__": {}},  # noqa: S307
                       {"row": row, "col": col})
        if not torch.is_tensor(offsets):
            offsets = torch.zeros_like(row) + int(offsets)
        return _promote(buf[offsets.expand(n_rows, col.shape[1])])

    def load_all(col):
        return [gather(a, col) for a in k.inputs]

    def reduce_over(expr, loads, refs, width):
        v = eval_expr(expr, loads, refs, loads[0] if loads else row)
        if not torch.is_tensor(v):
            v = torch.as_tensor(v, device=device)
        return v.expand(n_rows, width) if v.dim() < 2 or v.shape[1] == 1 else v

    def store(name, refs, loads, col):
        arg = next(a for a in k.outputs if a.value.name == name)
        val = eval_expr(k.out_expr[name], loads, refs, loads[0] if loads else row)
        dest = buffers[arg.value.buffer.name]
        offsets = eval(arg.index.render(), {"__builtins__": {}},  # noqa: S307
                       {"row": row, "col": col})
        shape = (n_rows, col.shape[1])
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, device=dest.device)
        dest[offsets.expand(shape).reshape(-1)] = val.expand(shape).reshape(-1).to(dest.dtype)

    element_stores = [a.value.name for a in k.outputs if a.value.name not in k.row_outputs]
    refs: dict[str, torch.Tensor] = {}

    if not k.two_pass:
        col = torch.arange(n_cols, device=device, dtype=torch.int64).unsqueeze(0)
        loads = load_all(col)
        for step in k.steps:
            if isinstance(step, Reduce):
                refs[step.name] = _REDUCE[step.kind](reduce_over(step.expr, loads, refs, n_cols))
            else:
                refs[step.name] = eval_expr(step.expr, loads, refs,
                                            loads[0] if loads else row)
        for name in element_stores:
            store(name, refs, loads, col)
        scalar_loads = loads
    else:
        from .reduction_schedule import plan_passes, scalar_binds

        passes = plan_passes(k)
        by_reduce = {r.name: p for p in passes for r in p.reduces}
        scalar_names = {b.name for b in scalar_binds(k)}
        scalar_loads = load_all(zero_col)
        width = max(1, min(n_cols, chunk))

        def run_pass(p):
            online = p.online_softmax
            accs: dict[str, torch.Tensor] = {}
            m = torch.full((n_rows, 1), -float("inf"), device=device)
            acc_l = torch.zeros((n_rows, 1), device=device)
            for c0 in range(0, n_cols, width):
                c1 = min(c0 + width, n_cols)
                col = torch.arange(c0, c1, device=device, dtype=torch.int64).unsqueeze(0)
                loads = load_all(col)
                local = dict(refs)
                for b in p.binds:
                    local[b.name] = eval_expr(b.expr, loads, local,
                                              loads[0] if loads else row)
                if online is not None:
                    x = reduce_over(online.value, loads, local, c1 - c0)
                    m_new = torch.maximum(m, x.amax(dim=1, keepdim=True))
                    acc_l = (acc_l * torch.exp(m - m_new)
                             + torch.exp(x - m_new).sum(dim=1, keepdim=True))
                    m = m_new
                else:
                    for r in p.reduces:
                        part = _REDUCE[r.kind](reduce_over(r.expr, loads, local, c1 - c0))
                        accs[r.name] = (part if r.name not in accs
                                        else _COMBINE[r.kind](accs[r.name], part))
                for name in p.stores:
                    store(name, local, loads, col)
            if online is not None:
                refs[online.max_step.name] = m
                refs[online.sum_step.name] = acc_l
            refs.update(accs)

        for step in k.steps:
            if isinstance(step, Reduce):
                if step.name not in refs:
                    run_pass(by_reduce[step.name])
            elif step.name in scalar_names:
                refs[step.name] = eval_expr(step.expr, scalar_loads, refs,
                                            scalar_loads[0] if scalar_loads else row)
        for p in passes:
            if p.stores and not p.reduces:
                run_pass(p)

    for a in k.outputs:
        if a.value.name not in k.row_outputs:
            continue
        val = eval_expr(k.out_expr[a.value.name], scalar_loads, refs,
                        scalar_loads[0] if scalar_loads else row)
        dest = buffers[a.value.buffer.name]
        offsets = eval(a.index.render(), {"__builtins__": {}},  # noqa: S307
                       {"row": row, "col": zero_col})
        if not torch.is_tensor(val):
            val = torch.as_tensor(val, device=device).expand(n_rows, 1)
        dest[offsets.reshape(-1)] = val.reshape(-1).to(dest.dtype)


_REDUCE = {
    "sum": lambda v: v.sum(dim=1, keepdim=True),
    "max": lambda v: v.amax(dim=1, keepdim=True),
    "min": lambda v: v.amin(dim=1, keepdim=True),
    "prod": lambda v: v.prod(dim=1, keepdim=True),
}

_COMBINE = {
    "sum": lambda a, b: a + b,
    "max": torch.maximum,
    "min": torch.minimum,
    "prod": lambda a, b: a * b,
}


def _store_element(k, name, buffers, refs, loads, row, col) -> None:
    arg = next(a for a in k.outputs if a.value.name == name)
    val = eval_expr(k.out_expr[name], loads, refs, loads[0] if loads else row)
    dest = buffers[arg.value.buffer.name]
    offsets = eval(arg.index.render(), {"__builtins__": {}},  # noqa: S307
                   {"row": row, "col": col})
    shape = (row.shape[0], col.shape[1])
    if not torch.is_tensor(val):
        val = torch.as_tensor(val, device=dest.device)
    dest[offsets.expand(shape).reshape(-1)] = val.expand(shape).reshape(-1).to(dest.dtype)
