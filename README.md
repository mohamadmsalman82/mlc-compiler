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

**Fusion legality** is written up in full in
[docs/fusion-legality.md](docs/fusion-legality.md), and the benchmark protocol
in [docs/benchmarks.md](docs/benchmarks.md): acyclicity, iteration
spaces, index maps, escape, associativity and numerical stability, and the
cost model that decides profit. Each rule says what breaks without it and
names the test that covers it.

## What it compiles to

`python -m mlc passes bert-base --batch 1 --seq 128`:

| pass enabled | kernels | pointwise | reduction | extern | intermediates |
|---|---:|---:|---:|---:|---:|
| no passes | 730 | 556 | 0 | 174 | 331 MiB |
| elementwise | 323 | 149 | 0 | 174 | 140 MiB |
| + reduction | 200 | 63 | 37 | 100 | 112 MiB |
| + memory planning | 200 | 63 | 37 | 100 | **3.4 MiB** |

1229 graph nodes become 200 kernels, 100 of which are the matmuls and
embeddings that dispatch to cuBLAS. GPT-2 small is 1111 nodes to 148 kernels.

Softmax compiles to one kernel -- the scale, the mask, both reductions and the
divide -- with the causal mask folded into the index arithmetic:

```python
@triton.jit
def k6(in_ptr0, in_ptr1, out_ptr0, BLOCK_COL: tl.constexpr):
    # persistent: 512 rows of 64, row space [1, 8, 64], reduced [64]
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_COL)
    # 64 is exactly BLOCK_COL: no column mask needed
    v0 = tl.load(in_ptr0 + ((row * 64 + col)))              # scores
    v1 = tl.load(in_ptr1 + (((row % 64) * 256 + col)))      # causal mask
    t0 = (v0 / 8.0)
    t1 = (t0 + v1)
    r2 = tl.max(t1, axis=0)
    t3 = (t1 - r2)
    t4 = tl.exp(t3)
    r5 = tl.sum(t4, axis=0)
    t6 = (t4 / r5)
    tl.store(out_ptr0 + ((row * 64 + col)), t6)             # probs
```

Layer norm is the same shape, and picks up the residual add and the bias
before it as producers and the affine transform after it as an epilogue.

## First run on a GPU

```
pip install torch pytest numpy      # Triton comes with torch on Linux
bash scripts/gpu_check.sh           # tests, then every model through Triton
bash scripts/gpu_bench.sh           # the sweep, once the above passes
```

`gpu_check.sh` logs to `gpu_check.log`: versions, the test suite, each model
compiled and executed through Triton and checked against eager, the CUDA graph
path at batch 32, the streamed reduction path, and a calibration of the cost
model constants. It is the file to send if anything fails.

On a consumer card fp32 is the weak path, so `--dtype float16` is the
realistic configuration:

```
python -m mlc.bench --calibrate --dtype float16 \
    --models bert-base gpt2-small --batches 1 8 32 --out results/bench.md
```

`run` compiles the model, executes it through the Triton backend, and checks
the result against eager. It reports the backend actually used, so a silent
fallback to the reference backend cannot be mistaken for a passing GPU run.

## Benchmarks

```
python -m mlc.bench --models bert-base gpt2-small --batches 1 8 32 --out results/bench.md
```

The cost model's constants are device properties: `flops_per_byte` is about
12 on an A100 and about 56 on a 4060, and launch overhead is 4.7 MB of forgone
bandwidth on the first and under 1 MB on the second. `mlc/devices.py` has a
table, and `--calibrate` measures them on the actual card instead, including
the per-kernel cost under CUDA graph replay. These are not cosmetic: on
BERT-base at batch 32 with capture on, the 4060 profile produces a different
schedule than the A100 one.

Variants run as a ladder -- no passes, elementwise, `+ reduction`,
`+ memory`, `+ cuda-graphs` -- so the difference between adjacent rows is one
pass's contribution. `torch.compile` appears twice, in its default mode and in
`reduce-overhead`, because the second uses CUDA graphs and is the honest
comparison for the graph-captured variant. Every variant is checked against
eager before it is timed.

Results are in [results/RESULTS.md](results/RESULTS.md), and what they do and
do not say is in [docs/benchmarks.md](docs/benchmarks.md).

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
pytest tests/          # 472 tests, runs anywhere
```

The streamed reduction path, including its online-softmax recurrence, is
covered by lowering `max_persistent_row` so that even a 32-element row does
not fit. Both backends share the pass partition, so that runs the same program
structure the GPU would.

## Status

- [x] graph capture, shape and dtype propagation
- [x] elementwise fusion, recompute, Triton codegen, runtime
- [x] memory planning
- [x] reduction fusion
- [x] benchmark harness, per-pass attribution, `torch.compile` baselines
- [x] fusion legality writeup
- [x] fp16 support, with fp32 accumulation in reductions
- [ ] latency and memory numbers (needs a CUDA device)
- [ ] analysis of a case where `torch.compile` wins
