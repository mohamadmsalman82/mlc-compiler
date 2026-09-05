# Where torch.compile wins, and why

At batch size 1 this compiler is 3.05x faster than `torch.compile` on
BERT-base and 2.19x on GPT-2 small. At batch 8 it is behind at 0.92x and
0.89x, and at batch 32 at 0.85x and 0.71x. This is that crossover, measured
rather than guessed at.

All numbers are BERT-base, fp16, sequence 128, on an A40.

## First, the baseline is not handicapped

Before explaining a loss it is worth ruling out having rigged the comparison.

- **`torch.compile` gets the whole model in one graph.** `torch._dynamo.explain`
  reports **0 graph breaks**. Nothing is falling back to eager.
- **`mode="reduce-overhead"` really does use CUDA graphs.** Inductor's
  `cudagraph_trees` logger shows `Recording function 0 of graph recording id 0`
  and `recording cudagraph tree for graph without symints`. Its timing matched
  the default mode, which looked suspicious, but capture is genuinely on: at
  these sizes it simply does not help Inductor much, because it has far fewer
  launches to save.

## The crossover

| batch | eager | torch.compile | mlc | mlc vs eager | mlc vs t.c |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.914 | 3.297 | 1.080 | 5.48x | **3.05x** |
| 8 | 5.102 | 3.436 | 3.735 | 1.37x | 0.92x |
| 32 | 16.004 | 11.744 | 13.833 | 1.16x | **0.85x** |

Note eager at batch 8 (5.10 ms) is *faster* than eager at batch 1 (5.91 ms).
Eight times the work in less time means batch 1 is not doing work at all; it
is paying for launches. That is the regime the whole design targets, and it
is also why the advantage evaporates as soon as there is real work to do.

## Where our time goes

`python -m mlc profile bert-base --batch 32 --dtype float16`:

| kind | kernels | ms | share |
|---|---:|---:|---:|
| extern (cuBLAS) | 100 | 9.05 | 63.5% |
| pointwise | 51 | 2.05 | 14.4% |
| reduction | 37 | 1.93 | 13.5% |
| **layout copies** | **48** | **1.22** | **8.6%** |

At batch 32 the matmuls are 63% of the runtime, and they are the *same cuBLAS
calls* in both compilers. So the entire 2.3 ms gap has to come out of the
36% we generate ourselves.

## The cause: we choose layouts, badly

Counting what each compiler emits for one BERT-base forward:

| | generated launches | distinct kernels | extern calls |
|---|---:|---:|---:|
| mlc | 136 | 136 | 100 |
| Inductor | 41 | 6 | 73 |

Inductor emits **six** distinct kernels and launches them 41 times, reusing
one layer-norm kernel across all 24 sites. We emit one function per schedule
position. That difference is mostly cosmetic for runtime, but the launch count
is not, and neither is this:

**Inductor emits no copy kernels. We emit 48.**

They come from attention. `q @ k.transpose(-2, -1)` needs its operands in a
particular layout, and `torch.export` renders that as `expand → clone → view`.
Our IR treats views as free restrides, so the `expand` and the `view` cost
nothing, but the `clone` is real data movement: this compiler always allocates
a kernel's output contiguous in that operation's natural shape, so when the
next consumer wants a different layout, something has to copy.

Inductor does not have that constraint. It picks the output layout of each
kernel it generates, so the qkv projection writes its result already in the
layout the batched matmul wants, and the copy never exists.

The cost is checkable two ways and they agree:

- **measured**: 48 kernels, 1.22 ms of 14.26 ms
- **from first principles**: each copy moves a `[32, 12, 128, 64]` fp16
  tensor in and out, 12 MB per copy, 576 MB per forward. At this device's
  measured 561 GB/s that is **1.08 ms**.

So roughly **half the gap to `torch.compile` at batch 32 is data movement that
exists only because we fix output layouts.**

## What it is not

The obvious explanation is that Inductor autotunes block sizes and warp counts
and we use fixed defaults. Autotuning is explicitly out of scope here, so this
was the comfortable answer. It is wrong:

| pointwise block | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| ms | 14.21 | 14.26 | 14.09 | 14.11 | 14.28 |

A 16x range of block sizes moves the total by 1.3%. The schedule is not
block-size bound, and autotuning would not close this gap.

The remaining ~1.2 ms is the launch count (136 generated launches against 41)
and Inductor's use of `addmm`, which lets cuBLAS apply the bias in the GEMM
epilogue where we run a separate fused kernel.

## What would fix it

Layout selection: let a kernel choose the layout of its output based on what
consumes it, rather than always writing the operation's natural contiguous
form. That is a real pass, and a well-understood one -- it is what Inductor's
layout planning does. It interacts with memory planning (a strided output
still needs a dense allocation) and with the index maps, which already handle
arbitrary strides on the *read* side and would need the same on the write
side. The machinery is mostly there; the pass is not.

That is the honest answer to "where does specializing to static-shape
inference stop paying." It stops paying as soon as there is enough work per
kernel that launch overhead stops dominating, and at that point what matters
is the ordinary compiler quality that generality does not actually cost you.
