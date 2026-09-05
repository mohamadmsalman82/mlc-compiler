"""Frontend: PyTorch module -> mlc graph IR.

Capture goes through ``torch.export``, which gives a static-shape ATen graph.
We then run a decomposition table chosen so the interesting structure is
*visible*: without it, ``softmax`` and ``layer_norm`` arrive as single opaque
ATen ops and there is nothing for a fusion pass to do. With it, softmax
becomes amax/sub/exp/sum/div and layer norm becomes var/sum/rsqrt/mul/add,
which is exactly the shape the reduction fusion pass looks for.

Types are then propagated by :mod:`mlc.ir.shapes` from the placeholders
forward. The exporter's FakeTensor metadata is carried along in
``node.meta['export_val']`` purely so the test suite can check us against it.
"""

from __future__ import annotations

import operator
import warnings
from typing import Any, Sequence

import torch
from torch.export import export

from . import shapes
from .graph import Graph, Node, Value
from .ops import NonViewable, OpClass, lookup, op_class
from .types import Buffer, Layout

#: Ops we force apart even though they are "core ATen". Each one hides a
#: reduction or a pointwise chain that the fusion passes want to see.
#: ``addmm`` is deliberately absent below and split by hand instead. Torch's
#: decomposition of it is wrapped in a cast-for-opmath, so in half precision
#: it upcasts both operands to fp32, runs the matmul there, and casts back.
#: That silently turns every fp16 GEMM into an fp32 one plus two conversion
#: passes over the weights, which on a consumer card gives up the tensor
#: cores entirely. Splitting it ourselves keeps the dtypes exactly as they
#: were and still exposes the bias add to fusion.
FORCE_DECOMPOSE = [
    "_softmax",
    "_log_softmax",
    "native_layer_norm",
    "gelu",
    "silu",
    "var_mean.correction",
    "std_mean.correction",
    "split_with_sizes",
    "native_batch_norm",
    "_native_batch_norm_legit_no_training",
]


def _decomp_table(extra: Sequence[str]):
    from torch._decomp import core_aten_decompositions, get_decompositions

    ops = []
    for name in extra:
        parts = name.split(".")
        packet = getattr(torch.ops.aten, parts[0], None)
        if packet is None:
            continue
        overload = getattr(packet, parts[1]) if len(parts) > 1 else getattr(packet, "default", None)
        if overload is not None:
            ops.append(overload)
    table = dict(core_aten_decompositions())
    table.update(get_decompositions(ops))
    return table


class CaptureError(Exception):
    pass


def capture(
    model: torch.nn.Module,
    example_inputs: tuple,
    *,
    name: str | None = None,
    extra_decompositions: Sequence[str] = FORCE_DECOMPOSE,
) -> Graph:
    """Trace ``model`` at the shapes of ``example_inputs`` into an mlc graph."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = export(model, tuple(example_inputs))
            ep = ep.run_decompositions(_decomp_table(extra_decompositions))
    finally:
        if was_training:
            model.train()

    g = Graph(name or type(model).__name__)
    g.meta["in_spec"] = ep.call_spec.in_spec if hasattr(ep, "call_spec") else None
    g.meta["out_spec"] = getattr(ep, "_out_spec", None)
    _build(g, ep)
    g.topo_check()
    return g


# --------------------------------------------------------------------------

def _fake(node) -> Any:
    return node.meta.get("val")


def _as_type(fake) -> tuple[tuple[int, ...], torch.dtype]:
    shape = tuple(int(d) for d in fake.shape)
    return shape, fake.dtype


def _build(g: Graph, ep) -> None:
    gm = ep.graph_module
    sig = ep.graph_signature
    state = dict(ep.state_dict)
    consts = dict(getattr(ep, "constants", {}) or {})

    env: dict[Any, Any] = {}
    spec_iter = iter(sig.input_specs)

    for fx_node in gm.graph.nodes:
        if fx_node.op == "placeholder":
            _placeholder(g, fx_node, next(spec_iter), state, consts, env)
        elif fx_node.op == "call_function":
            _call(g, fx_node, env)
        elif fx_node.op == "output":
            _output(g, fx_node, sig, env)
        elif fx_node.op in ("get_attr",):
            raise CaptureError(f"unexpected get_attr {fx_node.target}; export should have lifted it")
        else:
            raise CaptureError(f"unsupported fx node kind {fx_node.op}")


def _placeholder(g: Graph, fx_node, spec, state, consts, env) -> None:
    from torch.export.graph_signature import InputKind

    fake = _fake(fx_node)
    if fake is None:
        raise CaptureError(f"placeholder {fx_node.name} has no shape metadata")
    shape, dtype = _as_type(fake)
    if any(not isinstance(d, int) for d in shape):
        raise CaptureError(f"dynamic shape in {fx_node.name}: {shape}; mlc requires static shapes")

    is_input = spec.kind == InputKind.USER_INPUT
    kind = "input" if is_input else "param"
    buf = Buffer(fx_node.name, shape, dtype, kind)
    val = Value(fx_node.name, buf, Layout.contiguous(shape))
    if is_input:
        g.inputs.append(val)
    else:
        g.params.append(val)
        fqn = spec.target
        tensor = state.get(fqn, consts.get(fqn))
        if tensor is None:
            raise CaptureError(f"no tensor for lifted input {fqn}")
        g.param_tensors[val.name] = tensor.detach()
        g.param_origin[val.name] = fqn
    env[fx_node] = val


def _resolve(a: Any, env: dict):
    if isinstance(a, torch.fx.Node):
        v = env[a]
        if isinstance(v, list):
            raise CaptureError(f"multi-output value {a.name} used without getitem")
        return v
    if isinstance(a, (list, tuple)):
        return type(a)(_resolve(x, env) for x in a)
    if isinstance(a, torch.SymInt):
        raise CaptureError("symbolic int in args; mlc requires static shapes")
    return a


def _call(g: Graph, fx_node, env) -> None:
    target = fx_node.target
    if target is operator.getitem:
        src = env[fx_node.args[0]]
        env[fx_node] = src[fx_node.args[1]]
        return

    op = str(target)
    if op in DROP_OPS:
        return
    args = [_resolve(a, env) for a in fx_node.args]
    kwargs = {k: _resolve(v, env) for k, v in fx_node.kwargs.items()}
    cls = op_class(op)

    if op in _SPLIT_OPS:
        env[fx_node] = _expand_split(g, op, args, kwargs, fx_node)
        return

    if op == "aten.addmm.default" and _plain_addmm(args, kwargs):
        env[fx_node] = _split_addmm(g, args, kwargs, fx_node)
        return

    if cls is OpClass.VIEW:
        env[fx_node] = _make_view(g, op, args, kwargs, fx_node)
        return

    if cls is OpClass.OPAQUE:
        env[fx_node] = _make_opaque(g, op, args, kwargs, fx_node, target)
        return

    types = shapes.infer(op, args, kwargs)
    node = Node(op, tuple(args), kwargs)
    node.meta["export_val"] = _fake(fx_node)
    node.meta["target"] = target
    outs = []
    for i, (shape, dtype) in enumerate(types):
        nm = fx_node.name if len(types) == 1 else f"{fx_node.name}_{i}"
        buf = Buffer(nm, shape, dtype, "intermediate")
        outs.append(Value(nm, buf, Layout.contiguous(shape)))
    node.outputs = outs
    g.add_node(node)
    env[fx_node] = outs[0] if len(outs) == 1 else outs


def _make_view(g: Graph, op: str, args, kwargs, fx_node) -> Value:
    src: Value = args[0]
    try:
        layout = shapes.infer_view(op, args, kwargs)
    except NonViewable:
        # The nominal view needs real data movement (a reshape across a
        # transposed layout, say). Materialise a contiguous copy first, then
        # the view becomes a pure restride of that copy.
        src = _materialise(g, src)
        args = [src] + list(args[1:])
        layout = shapes.infer_view(op, args, kwargs)
    if layout.max_offset() > src.buffer.numel:
        raise CaptureError(f"{op} produces {layout} outside buffer {src.buffer}")
    out = Value(fx_node.name, src.buffer, layout)
    node = Node(op, tuple(args), kwargs, outputs=[out])
    node.meta["export_val"] = _fake(fx_node)
    node.meta["target"] = getattr(fx_node, "target", None)
    node.meta["is_view"] = True
    g.add_node(node)
    return out


def _materialise(g: Graph, src: Value) -> Value:
    """Emit a copy of ``src`` into a fresh contiguous buffer."""
    nm = g.fresh_name("copy")
    buf = Buffer(nm, src.shape, src.dtype, "intermediate")
    out = Value(nm, buf, Layout.contiguous(src.shape))
    node = Node("mlc.copy", (src,), {}, outputs=[out])
    node.meta["target"] = None
    g.add_node(node)
    return out


def _make_opaque(g: Graph, op: str, args, kwargs, fx_node, target):
    """Ops with no registered rule. They run as a plain torch call and act as
    fusion barriers; their types come from the exporter."""
    fake = _fake(fx_node)
    fakes = fake if isinstance(fake, (list, tuple)) else [fake]
    node = Node(op, tuple(args), kwargs)
    node.meta["export_val"] = fake
    node.meta["target"] = target
    node.meta["types_from_export"] = True
    node.meta["opaque"] = True
    outs = []
    for i, f in enumerate(fakes):
        if f is None or not hasattr(f, "shape"):
            raise CaptureError(f"opaque op {op} has a non-tensor result mlc cannot represent")
        shape, dtype = _as_type(f)
        nm = fx_node.name if len(fakes) == 1 else f"{fx_node.name}_{i}"
        outs.append(Value(nm, Buffer(nm, shape, dtype, "intermediate"), Layout.contiguous(shape)))
    node.outputs = outs
    g.add_node(node)
    return outs[0] if len(outs) == 1 else outs


#: Nodes torch.export inserts to record assumptions it already checked at
#: trace time. They produce no tensor and have no meaning at run time for a
#: static-shape graph, so they are dropped rather than made opaque.
DROP_OPS = {
    "aten._assert_tensor_metadata.default",
    "aten._assert_async.msg",
    "aten._assert_async.default",
    "aten._assert_scalar.default",
    "aten.sym_constrain_range.default",
    "aten.sym_constrain_range_for_size.default",
    "aten._functional_assert_scalar.default",
}


def _plain_addmm(args, kwargs) -> bool:
    """True for addmm with the default scalings, which is every one a Linear
    produces. Anything else stays a single extern call."""
    beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
    alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
    return beta == 1 and alpha == 1


def _split_addmm(g: Graph, args, kwargs, fx_node) -> Value:
    """``addmm(bias, a, b)`` -> ``mm(a, b)`` then ``add(., bias)``.

    Done here rather than by a decomposition table so the dtypes are untouched
    (see the note on FORCE_DECOMPOSE). The point of splitting at all is that
    the bias add is pointwise and fuses into whatever follows it, which for a
    transformer is the next layer norm or activation.
    """
    bias, mat1, mat2 = args[0], args[1], args[2]
    shape, dtype = shapes.infer_matmul("aten.mm.default", [mat1, mat2], {})
    nm = f"{fx_node.name}_mm"
    mm_out = Value(nm, Buffer(nm, shape, dtype, "intermediate"), Layout.contiguous(shape))
    mm = Node("aten.mm.default", (mat1, mat2), {}, outputs=[mm_out])
    mm.meta["target"] = torch.ops.aten.mm.default
    g.add_node(mm)

    out_shape, out_dtype = shapes.infer_pointwise(
        "aten.add.Tensor", [mm_out, bias], {}
    )
    out = Value(fx_node.name, Buffer(fx_node.name, out_shape, out_dtype, "intermediate"),
                Layout.contiguous(out_shape))
    add = Node("aten.add.Tensor", (mm_out, bias), {}, outputs=[out])
    add.meta["export_val"] = _fake(fx_node)
    add.meta["target"] = torch.ops.aten.add.Tensor
    g.add_node(add)
    return out


_SPLIT_OPS = {
    "aten.split_with_sizes.default",
    "aten.split.Tensor",
    "aten.unbind.int",
}


def _expand_split(g: Graph, op: str, args, kwargs, fx_node) -> list[Value]:
    """Rewrite a split into independent slice views.

    Splitting is free -- each piece is a strided window on the same buffer --
    but only if it is expressed that way. Left as one multi-output op it would
    force every consumer to treat the whole thing as opaque.
    """
    src: Value = args[0]
    if op == "aten.unbind.int":
        dim = (args[1] if len(args) > 1 else 0) % src.rank
        sizes = [1] * src.shape[dim]
        squeeze = True
    else:
        squeeze = False
        if op == "aten.split.Tensor":
            chunk = args[1]
            dim = (args[2] if len(args) > 2 else 0) % src.rank
            n = src.shape[dim]
            sizes = [min(chunk, n - i) for i in range(0, n, chunk)]
        else:
            sizes = list(args[1])
            dim = (args[2] if len(args) > 2 else 0) % src.rank

    outs: list[Value] = []
    start = 0
    for i, size in enumerate(sizes):
        nm = f"{fx_node.name}_{i}"
        layout = src.layout.slice(dim, start, start + size)
        node_args = (src, dim, start, start + size, 1)
        if squeeze:
            layout = layout.squeeze([dim])
        out = Value(nm, src.buffer, layout)
        node = Node("aten.slice.Tensor", node_args, {}, outputs=[out])
        node.meta["is_view"] = True
        node.meta["from_split"] = True
        g.add_node(node)
        outs.append(out)
        start += size
    return outs


def _output(g: Graph, fx_node, sig, env) -> None:
    from torch.export.graph_signature import OutputKind

    flat = fx_node.args[0]
    flat = list(flat) if isinstance(flat, (list, tuple)) else [flat]
    specs = list(sig.output_specs)
    for i, a in enumerate(flat):
        kind = specs[i].kind if i < len(specs) else OutputKind.USER_OUTPUT
        if kind != OutputKind.USER_OUTPUT:
            raise CaptureError(
                f"output {i} is a {kind}; mlc compiles pure inference graphs only"
            )
        if a is None:
            raise CaptureError("None outputs are not supported")
        v = _resolve(a, env)
        g.outputs.append(v)
        if v.buffer.kind == "intermediate":
            v.buffer.kind = "output"
