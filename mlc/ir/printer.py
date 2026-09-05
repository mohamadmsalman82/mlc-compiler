"""Human-readable graph dumps. Used by tests and by ``--dump`` on the CLI."""

from __future__ import annotations

from .graph import Graph


def format_graph(g: Graph, show_buffers: bool = False) -> str:
    lines = [f"graph {g.name}("]
    for v in g.params:
        lines.append(f"    param {v!r},")
    for v in g.inputs:
        lines.append(f"    input {v!r},")
    lines.append(") {")
    for n in g.nodes:
        tag = ""
        if n.meta.get("group") is not None:
            tag = f"    # group {n.meta['group']}"
        lines.append(f"    {n!r}{tag}")
    lines.append(f"    return {', '.join(repr(v) for v in g.outputs)}")
    lines.append("}")
    if show_buffers:
        lines.append("buffers:")
        for b in g.buffers():
            off = "" if b.arena_offset is None else f" @{b.arena_offset}"
            lines.append(f"    {b!r} {b.nbytes}B{off}")
    return "\n".join(lines)
