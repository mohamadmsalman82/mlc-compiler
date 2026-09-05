# Benchmark methodology

What is measured, how, and what the numbers do not say. Written before the
results so the protocol is not chosen to flatter them.

## Models

BERT-base and GPT-2 small, both written out in `mlc/bench/models.py` rather
than pulled from a model hub, so the graph the compiler sees is exactly what
is in the repository. Real configurations: 12 layers, 768 hidden, 12 heads,
3072 intermediate; sequence length 128.

The two differ deliberately. BERT is post-norm, so its layer norms sit after
the residual add. GPT is pre-norm, so they sit on the residual branch before
the projection. Reduction fusion reaches a different set of ops in each.

## Variants

A ladder, not a single number. Each rung turns on one more pass, so the
difference between adjacent rows is that pass's contribution:

| variant | passes on |
|---|---|
| `mlc/no-passes` | none: one kernel per op, per-op allocation |
| `mlc/elementwise` | elementwise fusion and recompute |
| `mlc/+reduction` | and reduction fusion |
| `mlc/+memory` | and arena memory planning |
| `mlc/+cuda-graphs` | and CUDA graph capture |

Against three baselines:

- **eager**, the thing being beaten.
- **`torch.compile`** in its default mode, the like-for-like comparison
  against fused Triton kernels.
- **`torch.compile` with `mode="reduce-overhead"`**, which adds CUDA graphs.
  This is the honest comparison for `mlc/+cuda-graphs`. Reporting only the
  default mode would flatter us at batch size 1, which is exactly where the
  claim matters most.

## Protocol

Every variant is **checked against eager before it is timed**. A variant whose
maximum absolute error exceeds tolerance is reported as wrong and not timed at
all, so a fast wrong answer cannot appear in a table.

Latency is the **median of 50 timed calls after 20 warmup calls**, measured
with CUDA events on the device rather than wall clock on the host. The 10th
percentile is reported alongside: a large gap between the two means the
machine was busy and the number should not be trusted.

Peak memory is `torch.cuda.max_memory_allocated` around a single call, after
`empty_cache` and a stats reset, so it includes the weights and whatever the
variant needs on top of them.

Tolerance is 2e-3 absolute in fp32 and 5e-2 in fp16.

## Cost model constants

The compiler's fusion decisions depend on two device properties, and they are
**measured on the machine the benchmark runs on**, not taken from a spec
sheet (`--calibrate`):

- achieved bandwidth, from a large streaming copy sized past L2
- fp32 throughput, from a dependent FMA chain in a Triton kernel
- kernel launch overhead
- kernel launch overhead under CUDA graph replay

Vendor peak fp32 is not reachable by pointwise code and launch overhead is as
much a property of the host CPU as the GPU, so measuring matters. The
measured figures come out at roughly 80 to 90 percent of published bandwidth
and 74 to 83 percent of published fp32.

## What these numbers do not say

**One sequence length.** Everything is at 128 tokens. Attention is quadratic
in sequence length and the pointwise work is linear, so the balance between
matmul time and fused-kernel time moves with it. A longer sequence would
shift the results toward the matmuls, which we do not generate.

**Matmuls are not ours.** Every GEMM goes to cuBLAS, in both our schedule and
Inductor's. A large part of the runtime at batch 32 is therefore identical
between the two, which compresses the apparent difference.

**L2 residency.** A planned arena is small. Where it fits in the device's L2,
fusion has no HBM traffic left to save and its benefit is launch overhead
alone. The benchmark prints the L2 size next to the arena size so the reader
can tell which regime a row is in. This matters more on cards with a large L2
(Ada) than on ones without.

**Peak memory includes the weights**, which dominate at batch 1 and are
identical across variants. The interesting quantity is the difference between
variants, not the absolute number.

**A rented GPU is a shared machine.** Neighbours affect timing. The p10 column
is the check on this, and the whole sweep is run in one session so that all
variants see the same conditions.

## Reproducing

```
bash scripts/gpu_check.sh     # correctness on every model, dtype and path
bash scripts/gpu_bench.sh     # the sweep
python scripts/assemble_results.py --dir results --out results/RESULTS.md
```
