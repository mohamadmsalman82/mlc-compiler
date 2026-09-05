# mlc

An ahead-of-time compiler for static-shape PyTorch inference. It captures a
model with `torch.export`, lowers it to its own graph IR, fuses elementwise
and reduction chains into single kernels, packs every intermediate into one
memory arena, generates Triton, and captures the result as a CUDA graph.

The question it exists to answer: **what does specializing to static-shape
inference buy, and where does it stop paying.** `torch.compile` has to stay
general. Fixing shapes and dropping training lets a compiler plan memory
across the whole graph and freeze the entire schedule into a replayable
graph. The benchmarks report where that wins and where it does not.

## Pipeline

```
torch.export ──▶ graph IR ──▶ elementwise ──▶ reduction ──▶ memory ──▶ Triton ──▶ CUDA
  + decomp       + shapes       fusion         fusion       planning    codegen    graph
```

| stage | module | what it does |
|---|---|---|
| capture | `mlc/ir/capture.py` | `torch.export` plus a decomposition table that forces softmax, layer norm and gelu apart so their structure is visible |
| types | `mlc/ir/shapes.py` | shape, dtype and layout propagation, cross-checked against the exporter in tests |
| elementwise fusion | `mlc/passes/fusion.py` | maximal chains over one iteration space; recompute for multi-consumer producers |
| reduction fusion | `mlc/passes/reduction_fusion.py` | persistent kernels, one row per block |
| memory planning | `mlc/passes/memory_planning.py` | live ranges packed into one arena by interval-graph colouring |
| codegen | `mlc/codegen/triton_backend.py` | one `@triton.jit` per kernel plus a launch table |
| runtime | `mlc/runtime/executor.py` | flat buffers, extern dispatch, CUDA graph capture |

## Use

```python
import torch, mlc

model = MyModel().eval()
example = (torch.randn(1, 128, 768, device="cuda"),)

compiled = mlc.compile(model, example)      # valid for these shapes only
out = compiled(*example)

print(compiled.schedule.format())           # the kernel list
print(compiled.source())                    # the generated Triton
```

Every pass has a flag, so its contribution can be measured on its own:

```python
from mlc import Config
mlc.compile(model, example, Config(reduction_fusion=False))
```

## Design notes

**Views are free.** A `Buffer` is storage; a `Layout` is a (shape, strides,
offset) view of it. `view`, `permute`, `expand`, `slice` and `as_strided`
produce a new layout over the same buffer and generate no code. `reshape` uses
torch's contiguous-run algorithm rather than requiring full contiguity, so the
qkv split in an attention block stays free instead of forcing three copies.

**Index maps are canonical.** Each operand carries an affine map from the
kernel's iteration variables to a buffer offset, simplified by dropping
size-1 dimensions, merging contiguous ones, and dropping stride-0 ones. Two
operands with equal maps touch the same address at the same iteration, which
is how the compiler decides a read can be a register reference instead of a
load. Because the form is canonical it sees through reshapes: `[B*T, D]` and
`[B, T, D]` views of a contiguous buffer compare equal, so fusion crosses
every reshape a transformer contains.

**Fusion legality** is documented alongside the code in
`mlc/passes/fusion.py` and `mlc/passes/reduction_fusion.py`, and the cost
model it appeals to is in `mlc/passes/cost.py`. The model is denominated in
HBM bytes, with arithmetic and kernel-launch overhead converted into that
currency at device-specific rates.

## Scope

**In:** static shapes, inference, elementwise and reduction fusion, memory
planning, Triton codegen, CUDA graphs.

**Out:** training, dynamic shapes, autotuning, MLIR, LLVM, custom GEMM,
convolution. Matmuls dispatch to cuBLAS and act as fusion barriers.

## Testing without a GPU

Triton needs CUDA, but most of what can go wrong here does not. The reference
backend in `mlc/codegen/torch_backend.py` executes the same kernel IR with
torch ops, evaluating index expressions from the *same rendered source string*
the Triton backend emits. So operator lowering, index arithmetic, fusion
legality, memory planning and scheduling are all under test on any machine;
only the Triton mechanics -- masking, block sizes, program ids -- need real
hardware.

```
pytest tests/          # runs anywhere
```

## Status

- [x] graph capture, shape and dtype propagation
- [x] elementwise fusion, recompute, Triton codegen, runtime
- [x] memory planning
- [ ] reduction fusion
- [ ] benchmarks against eager and `torch.compile`
