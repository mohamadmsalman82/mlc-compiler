"""Compiler configuration.

Every flag that changes what the compiler does lives here, so a benchmark can
turn one pass off and attribute the difference to it. That is a requirement of
the project rather than a convenience: the headline number is meaningless
without knowing which pass produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # -- passes ------------------------------------------------------------
    #: fuse chains of elementwise ops sharing an iteration space
    elementwise_fusion: bool = True
    #: duplicate a small producer into every consumer instead of storing it
    recompute: bool = True
    #: fuse producers into reductions and reductions into consumers
    reduction_fusion: bool = True
    #: pack intermediates into one arena by interval-graph colouring
    memory_planning: bool = True
    #: capture the final schedule as a CUDA graph
    cuda_graphs: bool = True

    # -- codegen -----------------------------------------------------------
    #: elements per program for pointwise kernels
    pointwise_block: int = 1024
    #: largest row a persistent reduction kernel will hold; above this the
    #: kernel streams the row twice instead
    max_persistent_row: int = 16384
    #: accumulate reductions in fp32 even for fp16/bf16 inputs
    fp32_accumulate: bool = True

    # -- cost model --------------------------------------------------------
    #: arithmetic the device retires in the time it moves one byte. Roughly
    #: peak_flops / peak_bandwidth: ~12 for an A100 in fp32, ~40 for a 4090.
    #: Raising it makes the compiler more willing to recompute.
    flops_per_byte: float = 20.0
    #: cost of one kernel launch, expressed as the bytes the device could
    #: have moved instead. ~3us at ~1.5 TB/s. Under CUDA graphs this is much
    #: smaller, which is why cuda_graphs and fusion partly substitute.
    launch_overhead_bytes: float = 4.5e6
    #: refuse to recompute an expression bigger than this many scalar ops
    max_recompute_ops: int = 32

    # -- misc --------------------------------------------------------------
    backend: str = "auto"  # "auto" | "triton" | "torch"
    debug: bool = False

    def replace(self, **kw) -> "Config":
        from dataclasses import replace as _r

        return _r(self, **kw)


#: Everything off: the baseline the per-pass attribution measures against.
NO_FUSION = Config(
    elementwise_fusion=False,
    recompute=False,
    reduction_fusion=False,
    memory_planning=False,
    cuda_graphs=False,
)

#: Elementwise only, so the reduction pass's contribution is isolable.
ELEMENTWISE_ONLY = Config(reduction_fusion=False)
