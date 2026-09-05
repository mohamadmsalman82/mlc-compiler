# Building an ahead-of-time compiler for static-shape PyTorch inference

A complete account of the project: what it set out to answer, how it was
built, every design decision that mattered, every bug that mattered, what it
measured, and what the numbers actually mean.

---

## Contents

1. [The premise](#1-the-premise)
2. [Scope](#2-scope)
3. [Build order, and why](#3-build-order-and-why)
4. [The IR](#4-the-ir)
5. [The frontend](#5-the-frontend)
6. [Elementwise fusion](#6-elementwise-fusion)
7. [Reduction fusion](#7-reduction-fusion)
8. [Memory planning](#8-memory-planning)
9. [Code generation](#9-code-generation)
10. [The runtime](#10-the-runtime)
11. [The cost model](#11-the-cost-model)
12. [Testing without a GPU](#12-testing-without-a-gpu)
13. [Taking it to a GPU](#13-taking-it-to-a-gpu)
14. [Results](#14-results)
15. [Where torch.compile wins](#15-where-torchcompile-wins)
16. [What this cost](#16-what-this-cost)
17. [What I would do next](#17-what-i-would-do-next)
18. [Appendix](#18-appendix)

---

## 1. The premise

PyTorch eager executes one operator at a time. Every intermediate tensor is
written to HBM and read back by the next op, and every op pays a kernel
launch. For a transformer at small batch size this is almost all of the
runtime: the arithmetic is trivial and the machine spends its time moving
bytes and starting kernels.

`torch.compile` addresses this by fusing operators into generated Triton
kernels. But Inductor has to stay general. It handles dynamic shapes,
training, mutation, and graph breaks, and that generality has a price.

The question this project was built to answer:

> **What does specializing all the way to static-shape inference actually buy,
> and where does it stop paying?**

Two things become possible when shapes are fixed and there is no autograd:

- **Whole-graph memory planning.** Every tensor's live range is known at
  compile time, so every intermediate can be packed into one arena with no
  runtime allocation at all.
- **Capturing the entire schedule as a CUDA graph.** Once memory is planned,
  every pointer and every launch parameter is fixed, which is exactly the
  precondition for graph capture.

The project set out to build that compiler, measure it honestly against both
eager and `torch.compile`, and explain the result rather than only the wins.

The answer turned out to be clean: **a great deal at batch size 1, and nothing
by batch 8.** Section 15 explains why in detail.

## 2. Scope

**In:** static shapes, inference only, elementwise fusion, reduction fusion,
memory planning, Triton code generation, CUDA graph capture.

**Out:** training, dynamic shapes, autotuning, MLIR, LLVM, custom GEMM
kernels, convolution. Matmuls dispatch to cuBLAS and act as fusion barriers.

Those exclusions are load-bearing for the analysis in section 15. Not writing
a GEMM means the matmuls are literally the same cuBLAS calls in both this
compiler and Inductor, which is what makes the comparison at large batch sizes
interpretable: any difference has to come from the kernels we generate
ourselves.

## 3. Build order, and why

The pipeline was built in this order, one commit per stage:

```
capture + shapes  ->  elementwise fusion + codegen + runtime  ->
memory planning   ->  reduction fusion
```

The reasoning: get one simple kernel all the way through to a measured number
before adding passes. A compiler that produces the right answer through a
short pipeline is debuggable. A compiler with four passes and no working
backend is not, because a wrong number could come from anywhere.

That paid off. When the first miscompile appeared (section 13), bisecting it
by pass took one command, because every pass already had a flag and every
configuration was already known to be correct on its own.

The full commit sequence:

| # | commit | what landed |
|---:|---|---|
| 1 | skeleton | gitignore, requirements |
| 2 | graph IR | buffers, layouts, `torch.export` frontend, shape propagation |
| 3 | elementwise fusion | fusion pass, kernel IR, Triton codegen, runtime, end to end |
| 4 | memory planning | live ranges, arena packing |
| 5 | reduction fusion | persistent kernels, both directions |
| 6 | benchmarks | harness, CLI, fusion-legality writeup |
| 7 | `mlc run` | compile-and-verify command |
| 8 | device profiles | cost-model constants per GPU, calibration |
| 9 | fp16 | half precision as a first-class path |
| 10 | GPU scripts | run scripts, rental cards |
| 11 | **miscompile fix** | register-locality legality rule |
| 12-19 | GPU fixes | seven more bugs, all found by running |
| 20 | results | measurements and the crossover analysis |
| 21 | verification log | evidence |

## 4. The IR

Three ideas do most of the work.

### 4.1 Buffers are separate from layouts

A `Buffer` is a contiguous block of memory. A `Layout` is a
`(shape, strides, offset)` triple describing how to read one.

View operations (`view`, `permute`, `expand`, `slice`, `as_strided`) produce a
new `Layout` over an **existing** `Buffer` and generate no code whatsoever.
They are not scheduled, they do not allocate, and they are skipped entirely
when kernels are emitted.

This is not a micro-optimization. If views were materialized as copies, an
attention block would break into a dozen kernels with a copy between each one,
and nothing downstream would fuse.

`Layout.reshape` was originally conservative: it only succeeded on fully
contiguous layouts. That forced three copies per attention block, because the
qkv projection slices a `[B, T, 3D]` buffer and then reshapes each slice to
`[B, T, H, D/H]`. The slice is strided, so the reshape "failed" and a copy was
inserted. Replacing it with torch's contiguous-run algorithm (the same one
`Tensor.view` uses to decide whether it can succeed) removed those copies.
It is differentially tested against `torch.Tensor.view` across every layout
the test suite can enumerate: the two must accept exactly the same reshapes
and produce exactly the same strides.

### 4.2 Index maps are canonical

Each operand of a kernel carries an affine map from the kernel's iteration
variables to a buffer offset:

```
offset = base + sum over axes, dims of ((var // out_stride) % size) * in_stride
```

With static shapes every coefficient is a compile-time constant, so the map is
simplified before it ever reaches generated source:

1. drop size-1 dimensions
2. merge adjacent dimensions contiguous in **both** the iteration space and
   the operand
3. drop stride-0 dimensions

Step 2 is what collapses a contiguous operand to the single term `idx`. Step 3
is how broadcasting disappears: a bias vector read across a batch becomes
`idx % D`, and a row statistic read back across its row loses its column term
entirely, which is why a reduction epilogue costs nothing.

```
contiguous       -> idx
bias broadcast   -> (idx % 64)
row statistic    -> (idx // 64)
transposed       -> ((idx // 4) + (idx % 4) * 8)
qkv slice        -> ((idx // 64) * 192 + (idx % 64) + 64)
softmax row      -> (row * 8 + col),  row base: row * 8
```

Because the form is canonical, **equality of two index maps is a decidable
test for "these touch the same address at the same iteration."** That single
property does three jobs:

- It decides whether a read inside a kernel can be a register reference
  instead of a load.
- It lets fusion cross reshapes: `[16, 192]` and `[2, 8, 192]` contiguous
  views of one buffer both reduce to `idx`, so they compare equal and the
  value never reaches memory. Every linear layer in a transformer is followed
  by such a reshape.
- It correctly refuses transposed views, which reduce to a different map.

### 4.3 The kernel IR

Fusion produces kernels, not code. A `PointwiseKernel` has an iteration space,
a list of operands with their index maps, an ordered body of scalar
expressions, and a list of stores. A `ReductionKernel` has a `(row, column)`
split and an ordered list of steps that interleave binds and reduces.

Two backends consume that IR: the Triton emitter and a torch reference
backend. That split is what made the whole project testable without a GPU
(section 12).

## 5. The frontend

Capture goes through `torch.export`, which gives a static-shape ATen graph.

The important detail is the **decomposition table**. Without forcing it,
`softmax` and `layer_norm` arrive as single opaque ATen ops and there is
nothing for a fusion pass to do. The compiler merges `core_aten_decompositions`
with explicit entries for `_softmax`, `native_layer_norm`, `gelu`, `silu`,
`var_mean` and `split_with_sizes`, which turns softmax into
`amax / sub / exp / sum / div` and layer norm into
`var / sum / rsqrt / sub / mul / add`. Those are exactly the shapes the
reduction pass looks for.

**`addmm` is deliberately excluded from that table and split by hand instead.**
Torch's decomposition of it is wrapped in a cast-for-opmath, so in half
precision it upcasts both operands to fp32, runs the matmul there, and casts
back. Forcing it apart to expose the bias add therefore turned every fp16 GEMM
into an fp32 GEMM plus two conversion passes over the weights, giving up the
tensor cores entirely. Splitting it in the compiler keeps the dtypes exactly
as they were and still exposes the bias add to fusion. This was worth 16
kernels and, on a consumer card, most of the matmul throughput.

Shapes, dtypes and layouts are then propagated by the compiler itself from the
placeholders forward. The exporter's FakeTensor metadata is kept only as a
**test oracle**: `verify_against_export` compares the two on every non-opaque
node, and the test suite runs it over every model. A mismatch names the
offending op.

Ops with no registered rule become opaque nodes that run as a plain torch call
and act as fusion barriers. Coverage is a performance question, not a
correctness one. On BERT-base and GPT-2 the only opaque op is
`aten.embedding`, which is a gather and not fusible in this scheme anyway.

## 6. Elementwise fusion

Fusion is a partition problem: decide which nodes share a kernel. The
constraint that makes it non-trivial is acyclicity.

`GroupGraph` is a union-find over nodes that refuses any merge which would
create a cycle. Each group carries a bitset of transitive ancestors and
another of descendants, so the check is two bitwise ANDs:

```python
between = (desc[A] & anc[B]) | (desc[B] & anc[A])
legal   = (between & ~(A | B)) == 0
```

Both fusion passes run over the same group graph, so the check covers merges
between them as well as within them.

Four legality rules, each with a test that names it:

**Iteration space.** A node joins when its output either broadcasts to the
group's space or has the same element count. The second rule is what lets
fusion cross a reshape, and it is sound only because both operands are
contiguous allocations.

**Escape.** A node whose output is smaller than the space is evaluated at
every point of that space, so several iterations compute the same value. Fine
as long as nobody outside needs it, and not fine if it has to be stored,
because a broadcast write would have several iterations storing to the same
address. So a value that is broadcast inside its kernel may not be visible
outside it.

**Register locality.** A pointwise kernel gives each iteration its own
registers, so one iteration cannot see a value another computed. If a member
reads another member's result at a *different* index, that read has to come
from memory, and the value it wants was never stored. This rule was missing
originally, and section 13 covers the miscompile it caused.

**Profit.** Section 11.

### Recompute

Union-find can only attach a producer to one consumer. A producer with several
consumers still writes its result for the rest. When the producer is cheap,
evaluating it again inside each consumer beats the store plus the reloads.
That pass runs after grouping, because duplicating a node puts it in two
kernels at once and union-find cannot represent that.

Its escape check has to be made against the **union** of all consumer groups,
not each one separately. After duplication every consumer holds its own copy,
so a value read by two of them has not escaped anything. Checking one at a
time refuses precisely the multi-consumer case the pass exists for, which is
what it did until it was found by instrumenting the decisions rather than
reading the code.

## 7. Reduction fusion

The harder pass. Two directions, both aimed at keeping a row of data resident
and doing everything to it while it is there.

**Producers into the reduction.** The scale and the mask before a softmax are
evaluated inside the kernel on data already loaded, so their results never
reach memory.

**Reductions into consumers.** The ops reading a reduce's result broadcast it
back across the axis that was reduced. Layer norm's subtract-and-scale, and
the affine transform after it, are all of this shape. The row is still in
registers, so the consumer runs without re-reading anything.

**Sibling reductions merge too**, which collapses layer norm's mean and
variance into one pass over the row instead of two.

The result: **softmax compiles to one kernel** (scale, mask, both reductions,
divide) and **layer norm compiles to one kernel** including the residual add
and bias before it and the affine transform after it.

### The frame

A reduction group runs in a `(row, column)` frame: kept dimensions on the row
axis, reduced dimensions on the column axis, with the operand layout permuted
into that order. A value that does not vary along the column axis is a scalar
in the kernel, and Triton broadcasts it against the column vector for free.

Classifying a value as row or element is subtler than "smaller shape". A
reduction with `keepdim=True` has shape `[B, T, 1]` and a layer norm weight
has shape `[D]`. Both broadcast to `[B, T, D]` perfectly well, but the first
is one value per program and the second is a vector. What separates them is
whether the reduced positions are size 1 after right-alignment. A
`keepdim=False` result has the reduced dimensions removed entirely and does
not right-align at all, so it is checked separately.

### Numerical stability is a correctness question here

**Variance.** The one-pass form `E[x^2] - E[x]^2` subtracts two large numbers
to get a small one. For a batch with mean 1e4 and variance 1, both terms are
near 1e8, where the fp32 ulp is about 8: the answer is noise. The compiler
expands variance into **two reductions over the same resident row** (the mean,
then the sum of squared deviations from it), which is exact and costs nothing
extra because the second pass is over registers rather than memory. The test
checks the result against a float64 reference and requires it to beat the
cancelling form by two orders of magnitude.

**Softmax.** Subtracting the row maximum before exponentiating is what keeps
`exp` from overflowing. In a persistent kernel this is free: the row is
already in registers, so the max and the sum are two cross-lane reduces over
the same data and the kernel still reads memory once.

When the row does not fit registers the kernel streams it, and then max and
sum would be two separate passes over memory. The compiler recognizes the
pattern (a `max` followed by a `sum` of `exp(x - max)`) and fuses them into
the online recurrence:

```
m' = max(m, max(chunk))
l' = l * exp(m - m') + sum(exp(chunk - m'))
```

which computes both in one pass and is exactly as stable, since every exponent
stays non-positive. This is the identity flash attention is built on. It is
applied only on the streamed path, because a persistent kernel has nothing to
gain from it.

Both backends share the pass partition, so lowering `max_persistent_row` in a
test exercises the streamed path and the recurrence on a CPU.

## 8. Memory planning

With static shapes and no autograd the whole allocation pattern is known at
compile time. Each intermediate is written by exactly one kernel and read by a
known set of later ones, so its live range is an interval on the schedule, and
two buffers whose intervals do not overlap can share memory.

That is interval-graph colouring, but *packing* is harder than colouring:
contiguous byte ranges are needed, not abstract colours, which makes it 2-D
strip packing. The heuristic is the standard one, largest first at the lowest
offset that clears every conflicting buffer already placed.

The lower bound (peak simultaneously-live bytes) is computed alongside, so the
gap is reported rather than assumed. On nine of ten test models the arena
lands **exactly on the lower bound**. On BERT-base at batch 1 it takes
intermediates from 331 MB to 3.4 MB.

Live ranges are computed per **buffer**, not per value. Several values can be
views of one buffer, and the buffer stays live until the last of them is read.
Getting that wrong at the value level would let the planner reuse memory a
later view still points into.

The overlap check is run directly, on every model and on randomised interval
sets, rather than inferred from the algorithm looking correct. Two buffers
alive at the same time sharing a byte is a silent miscompile, not a crash.

**The unplanned baseline is honest.** It gets its own alloc/free schedule and
leans on torch's caching allocator the way a compiler without a planner would,
so the comparison is planner-versus-allocator rather than
planner-versus-allocating-everything-up-front.

## 9. Code generation

The Triton emitter produces one `@triton.jit` function per kernel plus a
`LAUNCH` table the runtime reads, so the generated module is a standalone
artefact you can open and read. Most of the interesting claims in this project
are claims about generated code, and those are easier to check against a file
than against a debugger.

Everything the kernel needs is a compile-time constant. Element counts are
baked in, index arithmetic is constant-folded before it arrives, and block
sizes are chosen to divide the iteration space where possible so the bounds
mask disappears entirely.

Small things that matter in the output:

- **Copy propagation.** A node whose lowering is a bare load or reference
  (`clone`, a no-op cast, a multiply by one that folded away) gets no line of
  its own, so the kernel has one statement per real operation.
- **Algebraic folding at build time.** Only identities that hold for every
  float including NaN and infinity: `x*1`, `x+0`, `x-0`, `x/1`. Not `x*0`.
  Small, but decomposing `addmm` produces `mm * 1 + bias * 1` and every one of
  those would otherwise reach the kernel.
- **Masked reductions use the right identity.** `other=0.0` on the load is not
  enough, because zero is not the identity of a max, so a row of negative
  values would come back as zero.
- **Narrow floats are loaded into fp32 and cast back on store.** Reductions
  are unusable in fp16, and the reference backend makes the same choice so the
  two agree.

## 10. The runtime

Every buffer is a flat 1-D tensor. Shapes and strides live in the IR, so a
view costs an `as_strided` at the boundary and nothing inside a kernel.

The runtime went through three rounds of work, all driven by measurement on
the GPU (section 13). The final design:

- **Only live buffers are allocated.** Fusion leaves most values in registers,
  so their buffers are never touched by anything. On BERT-base that is 482 of
  681 buffers.
- **The buffer table is built once.** Inputs are copied into fixed buffers
  rather than aliased, so every pointer is stable for the life of the model.
- **The launch sequence is resolved once** into `(function, grid, arguments,
  constants)`. What remains per call is the launch itself, which is exactly
  what CUDA graph capture then removes as well.
- **Extern ops dispatch through the `.out` overload** where the schema allows,
  so cuBLAS writes into the arena directly instead of into a tensor torch
  allocated that then gets copied.

Compiled outputs are views onto storage the next call overwrites. That is the
same contract `torch.compile(mode="reduce-overhead")` has, and it is the
reason a captured graph can return anything at all. It is documented and
pinned by a test, because a test comparing two calls' results without cloning
compares a tensor with itself and passes for the wrong reason.

## 11. The cost model

Inference on these models is memory bound, so the model is denominated in
bytes of HBM traffic. Two other effects are converted into that currency:

- **arithmetic**, at `flops_per_byte`, the operations the device retires while
  moving one byte
- **launch overhead**, at `launch_overhead_bytes`, one kernel launch expressed
  as forgone bandwidth

Both are device properties and differ by more than an order of magnitude
across cards. An A100 retires about 12 fp32 ops per byte; a 4060 about 56.
There is a table of published specifications, and `--calibrate` measures the
same quantities on the actual card, which is preferable because vendor peak
fp32 is not reachable by pointwise code and launch overhead depends as much on
the host CPU as the GPU.

Measured on the A40 the benchmarks ran on:

```
561 GB/s achieved  (81% of the published 696)
28.0 TFLOP/s fp32  (75% of the published 37.4)
6.9 us per launch, 1.0 us under CUDA graph replay
```

**Capture changes the launch term.** Once the schedule is replayed as a graph,
most of the per-launch cost is gone, so a merge worth taking purely to
eliminate a launch may not be worth taking any more. The model uses a separate
constant when capture is on. This is not a refinement: on BERT-base at batch
32 it changes the schedule.

### What the model actually decides

Worth stating plainly, because it is easy to assume a cost model is doing more
than it is. **On these models, at every batch size and device profile tested,
every legal elementwise and reduction merge is also profitable, and the model
accepts all of them.** Fusing a pointwise chain is close to unconditionally
good, and the model saying so is the correct answer rather than an idle one.

Where it discriminates is recompute. A `[D]` value broadcast into a
`[32, 128, D]` space is re-evaluated four million times; when its expression
contains a transcendental, that costs more arithmetic than the round trip and
the launch are worth, and the model refuses. The same graph at `[1, 4, D]` is
duplicated happily. The decision also moves with the device.

## 12. Testing without a GPU

The whole compiler was developed on a Mac with no CUDA and no Triton. That
constraint produced the single most useful piece of infrastructure in the
project.

**The reference backend executes the same kernel IR with torch tensor ops, and
evaluates index expressions from the same rendered source string the Triton
backend emits.** Not an equivalent expression: the literal string, through
`eval`, against an `arange`. So a bug in index rendering surfaces on a laptop
rather than at runtime on a GPU.

That puts operator lowering, index arithmetic, fusion legality, memory
planning, scheduling, dtype handling and the reduction pass structure all
under test on any machine. Only the Triton mechanics (masking, block sizes,
program ids) need real hardware.

The streamed reduction path is covered by lowering `max_persistent_row` so
that even a 32-element row does not fit, which runs the same pass partition
and the same online-softmax recurrence the GPU would.

Other testing choices that earned their place:

- **Differential testing against torch.** Layout algebra is checked against
  `torch.Tensor.view` and torch's own striding, enumerated rather than
  sampled. Shape propagation is checked against the exporter on every node.
- **Cross-configuration agreement, split by whether it should be exact.**
  Passes that only change *where* a value is computed must be bit-identical to
  the unfused baseline. Reduction fusion reassociates floating-point sums by
  design, so it is bounded by tolerance instead. Conflating those two would
  have hidden either real miscompiles or expected drift.
- **Invariants, not just outputs.** Every stored value has exactly one writer;
  every kernel's reads are written before it runs; views never allocate; no
  group has an incoherent iteration space. The executable-order invariant is
  the one that catches a value read before anything wrote it.

578 tests, about 90 seconds, CPU only. They run in CI on every push.

## 13. Taking it to a GPU

The compiler was correct on CPU across 574 tests and ten models before it ever
saw CUDA. Running it on a GPU found **eight more bugs in a few hours.** They
are worth listing individually, because the pattern in them is the point.

### 13.1 Triton needs the generated code in a real file

`@triton.jit` calls `inspect.getsource` on every kernel it compiles, and that
fails outright on a module created by `exec` from a string. The generated
module is now written to a content-hashed file and imported properly, which
also means it is on disk to read, which is where most of the debugging
happened afterwards.

### 13.2 A literal is a `constexpr` and has no `.to`

`torch.zeros_like(input_ids)` in BERT's embedding produces a store whose
expression reads nothing at all. Emitting `(0).to(tl.int64)` is valid Python
and invalid Triton. A pure-constant store is now materialized with
`tl.full([BLOCK], value, dtype)`, and a cast wrapping a bare constant folds
into the constant so one can never be generated.

### 13.3 The miscompile

**BERT returned NaN at every batch size above one.** Both backends agreed on
the wrong answer, which immediately said it was a lowering bug rather than
codegen, and made it reproducible on the laptop.

Bisecting by pass took one command and pointed at elementwise fusion. Diffing
the intermediate buffers against the unfused schedule found the first
divergence at a single kernel:

```
k7: pointwise[64, 768] 3in 1out [add, clone]
   in2: expand [2, 12, 32, 64] idx=((idx // 24576) * 24576 + ...)
```

A bias add over `[B*T, D]` had been fused with a `clone` of a **transposed
view of its own result**. Both have the same element count, so the shape rules
accepted them into one kernel. But a pointwise kernel gives each iteration its
own registers: iteration `i` computed `add[i]` and needed the value iteration
`perm(i)` held. The read correctly fell back to a load, and the value it
wanted had never been stored, because the only consumer was inside the same
kernel.

The fix is the missing legality rule from section 6: a merge is refused when
any member would read another member's result at a different index map. The
check shares its index computation with the emitter so the two cannot drift.

**Batch size 1 hid it**, because `torch.export` emits far fewer clones there
and the pattern never arose. The invariant that catches this (every kernel's
reads are written before it runs) was already in the suite, but only over the
toy models, which do not produce a transposed readback. It now runs over the
benchmark models at batch 2 as well.

### 13.4 Memory planning was making things worse

The first real measurement showed `mlc/+memory` at 13.9 ms against 11.4
without it, and 454 MB peak against 234. Both directions wrong, on the two
axes the pass exists to improve. Three separate causes:

**482 of 681 buffers were being allocated for nothing.** Fusion leaves most
values in registers, so their buffers are never read or written and the
planner correctly gives them no arena slot. The executor then allocated each
one anyway: 218 MB of memory for data nothing looks at.

**The buffer table was rebuilt every call.** 884 tensor views per forward, 4 ms
of Python, several times the kernel time.

**Every extern op copied its result.** Each matmul wrote into a tensor torch
allocated and was then copied into ours, about a hundred extra full-tensor
copies per BERT forward.

After: 3.62 ms and 234 MB.

### 13.5 The interpreter was the bottleneck

Even after that, 200 kernels took 6.6 ms, which is about 30 us of Python per
launch against 7 us of actual launch overhead. The runtime was resolving every
kernel's arguments on every forward.

With static shapes there is nothing to resolve. Inputs are copied into fixed
buffers, and the launch sequence is precomputed once. That took the planned
non-captured variant from 6.6 ms to 3.6 ms, against 4.9 ms eager.

This mattered for honesty as much as for speed: the previous numbers credited
CUDA graphs with interpreter overhead the other passes were paying, which
would have made the per-pass attribution meaningless.

### 13.6 Binding must not execute

Making the extern path use `.out` overloads introduced a bug of its own: the
binder decided whether an overload was usable by *trying the call*. That
executed an embedding lookup while resolving the launch plan, before any
kernel had filled its index buffer, and tripped a device-side assert on
garbage indices. Applicability is now read off the schema. There is a test
that poisons every intermediate buffer and binds anyway.

### 13.7 Two sweeps lost to an untested report path

Adding a `dtype` field made the result group key a 4-tuple while the formatter
still unpacked three. The exception fired *after* the timing was done, so
minutes of GPU time went in the bin, twice.

Two fixes: the runner writes the JSON before it formats anything, so a bug in
the report can never discard measurements; and the reporting path has tests
now, which is why it had shipped two bugs.

### 13.8 Triton resolves names from module globals

The calibration probe defined its kernel inside the measurement function, with
`import triton.language as tl` in local scope. Triton walks the AST and
resolves names from the function's module globals, so `tl` was invisible.

### The pattern

Every one of these is a case where CPU testing could not have helped, and
every one was cheap to find once something ran. Two of the eight were in the
compiler proper; six were in the runtime, the tooling, or the measurement
harness. That ratio is itself a result: the passes were the part that got
designed carefully and tested hard, and the plumbing was the part that had
never been exercised.

## 14. Results

Measured on a rented **NVIDIA A40** (46 GB, Ampere), torch 2.4.1+cu124, Triton
3.0.0, driver 570.169. Sequence length 128. Median of 50 timed calls after 20
warmup calls, timed with CUDA events. **Every variant is checked against eager
before it is timed**, so a fast wrong answer cannot appear in a table.

### Latency

| model | dtype | batch | eager | `torch.compile` | reduce-overhead | **mlc** | vs eager | vs t.c |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bert-base | fp16 | 1 | 5.914 | 3.300 | 3.297 | **1.080** | 5.48x | **3.05x** |
| bert-base | fp16 | 8 | 5.102 | 3.436 | 3.452 | 3.735 | 1.37x | 0.92x |
| bert-base | fp16 | 32 | 16.004 | 11.744 | 11.747 | 13.833 | 1.16x | 0.85x |
| bert-base | fp32 | 1 | 5.843 | 3.321 | 3.138 | **2.903** | 2.01x | **1.08x** |
| bert-base | fp32 | 8 | 15.014 | 13.654 | 13.650 | 14.216 | 1.06x | 0.96x |
| gpt2-small | fp16 | 1 | 5.395 | 2.506 | 2.494 | **1.141** | 4.73x | **2.19x** |
| gpt2-small | fp16 | 8 | 8.159 | 4.746 | 4.743 | 5.340 | 1.53x | 0.89x |
| gpt2-small | fp16 | 32 | 21.521 | 14.395 | 14.332 | 20.303 | 1.06x | 0.71x |

### Peak memory

Peak allocated, including weights. The mlc column is `+memory`, the variant
the pass is about; `+cuda-graphs` costs about 10 MB more for its static
buffers.

| model | batch | eager | `torch.compile` | **mlc** |
|---|---:|---:|---:|---:|
| bert-base | 32 | 292 MB | 346 MB | **292 MB** |
| gpt2-small | 8 | 629 MB | 627 MB | **545 MB** |
| gpt2-small | 32 | 1522 MB | 1593 MB | **1183 MB** |

Memory is the one advantage that improves with batch size, because it does not
depend on launch overhead.

### Per-pass attribution

Each row is the gain over the row above.

**BERT-base fp16, batch 1:**

| pass enabled | kernels | ms | gain | cumulative |
|---|---:|---:|---:|---:|
| no passes | 682 | 39.968 | baseline | 1.00x |
| elementwise | 322 | 19.014 | 2.10x | 2.10x |
| + reduction | 200 | 7.938 | 2.40x | 5.04x |
| + memory | 200 | 3.784 | 2.10x | 10.56x |
| + cuda graphs | 200 | 1.080 | 3.50x | **37.01x** |

**BERT-base fp16, batch 32:**

| pass enabled | kernels | ms | gain | cumulative |
|---|---:|---:|---:|---:|
| no passes | 718 | 44.165 | baseline | 1.00x |
| elementwise | 359 | 20.137 | 2.19x | 2.19x |
| + reduction | 236 | 14.560 | 1.38x | 3.03x |
| + memory | 236 | 14.127 | 1.03x | 3.13x |
| + cuda graphs | 236 | 13.833 | 1.02x | **3.19x** |

The contrast is the whole finding. At batch 1 every pass contributes and CUDA
graphs contribute most. At batch 32 elementwise fusion still pays and
everything after it is noise, because there is enough work per kernel that
launch overhead and interpreter overhead have stopped mattering.

Note that memory planning's latency contribution at batch 1 is mostly that a
fixed arena is the **precondition for a resolved launch plan**, not that it
saves bandwidth. Both are real consequences of planning, and the results
should say which is which.

### What it compiles to

| pass enabled | kernels | pointwise | reduction | extern | intermediates |
|---|---:|---:|---:|---:|---:|
| no passes | 730 | 556 | 0 | 174 | 331 MB |
| elementwise | 323 | 149 | 0 | 174 | 140 MB |
| + reduction | 200 | 63 | 37 | 100 | 112 MB |
| + memory | 200 | 63 | 37 | 100 | **3.4 MB** |

1229 graph nodes become 200 kernels, 100 of which are matmuls and embeddings
dispatched to cuBLAS.

## 15. Where torch.compile wins

At batch 32 `torch.compile` is 15% ahead on BERT and 41% ahead on GPT-2. This
is the deliverable the project promised: explain the loss rather than hide it.

### The baseline is not handicapped

Worth ruling out first.

- **Zero graph breaks.** `torch._dynamo.explain` reports 0. Nothing falls back
  to eager.
- **`reduce-overhead` really uses CUDA graphs.** Inductor's `cudagraph_trees`
  logger shows `Recording function 0 of graph recording id 0`. Its timing
  matched the default mode, which looked suspicious, but capture is genuinely
  on: at these sizes it simply has far fewer launches to save.

### The matmuls are shared

At batch 32, `python -m mlc profile` gives:

| kind | kernels | ms | share |
|---|---:|---:|---:|
| extern (cuBLAS) | 100 | 9.05 | 63.5% |
| pointwise | 51 | 2.05 | 14.4% |
| reduction | 37 | 1.93 | 13.5% |
| **layout copies** | **48** | **1.22** | **8.6%** |

63% of the runtime is cuBLAS calls that are identical in both compilers. So
the entire 2.3 ms gap has to come out of the third we generate.

### The cause: we fix output layouts

| | generated launches | distinct kernels | extern calls |
|---|---:|---:|---:|
| mlc | 136 | 136 | 100 |
| Inductor | 41 | 6 | 73 |

Inductor emits **six** distinct kernels and launches them 41 times, reusing
one layer-norm kernel across all 24 sites. And critically:

**Inductor emits no copy kernels. We emit 48.**

They come from attention. `q @ k.transpose(-2, -1)` needs its operands in a
particular layout, and `torch.export` renders that as `expand -> clone ->
view`. The `expand` and `view` are free restrides in this IR, but the `clone`
is real data movement, because this compiler always allocates a kernel's
output contiguous in that operation's natural shape. When the consumer wants a
different layout, something has to copy.

Inductor does not have that constraint. It picks the output layout of each
kernel it generates, so the qkv projection writes its result already in the
layout the batched matmul wants, and the copy never exists.

The cost checks out two independent ways:

- **measured**: 48 kernels, 1.22 ms of 14.26 ms
- **from first principles**: each copy moves a `[32, 12, 128, 64]` fp16 tensor
  in and out, 12 MB per copy, 576 MB per forward, which at the device's
  measured 561 GB/s is **1.08 ms**

Roughly half the gap is data movement that exists only because we fix output
layouts.

### It is not autotuning

The comfortable answer was that Inductor autotunes block sizes and we use
fixed defaults, which is true and out of scope. It is also wrong:

| pointwise block | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| ms | 14.21 | 14.26 | **14.09** | 14.11 | 14.28 |

A 16x range moves the total by 1.3%. The schedule is not block-size bound.

The remaining ~1.2 ms is the launch count (136 against 41) and Inductor's use
of `addmm`, which lets cuBLAS apply the bias in the GEMM epilogue where we run
a separate fused kernel.

### The answer to the original question

**Specializing to static-shape inference stops paying as soon as there is
enough work per kernel that launch overhead stops dominating.** At that point
what matters is ordinary compiler quality: layout selection, kernel reuse,
epilogue fusion. None of those things are made harder by being general, which
is why `torch.compile` gets them and this does not.

The corollary is that the win at batch 1 is real and not an artifact. Batch 1
is exactly the regime where launch overhead is the whole cost, and it is also
the regime that matters for interactive and latency-sensitive serving.

## 16. What this cost

The compiler was written on a Mac with no GPU. The measurement phase used a
rented GPU on RunPod.

| | |
|---|---|
| GPU time | about 4 hours, across two pods |
| Total spend | **$1.96** |
| First pod | RTX 4090, died mid-sweep and would not restart |
| Second pod | A40 (Ampere, 46 GB), Secure Cloud, $0.49/hr |

A detour worth recording: the cheapest option was a Community Cloud pod at
$0.22/hr, but Community pods route SSH through a proxy that rejected both
registered keys, and a pod created without `PUBLIC_KEY` in its environment
never starts an SSH daemon at all. Two pods and about twenty minutes went into
that before switching to Secure Cloud, where the first pod had worked
immediately. The lesson is the ordinary one: when a cheaper path has already
cost more than the difference, stop.

Both pods were terminated. Spend rate is zero.

## 17. What I would do next

In rough order of value:

**Layout selection.** The single change that would close half the batch-32
gap. Let a kernel choose the layout of its output based on what consumes it,
rather than always writing the operation's natural contiguous form. The
machinery is mostly there: index maps already handle arbitrary strides on the
read side and would need the same on the write side. It interacts with memory
planning, because a strided output still needs a dense allocation.

**Kernel deduplication.** Twelve identical layers produce twelve identical
kernels. Inductor emits one and launches it twelve times. This costs compile
time and code size more than runtime, but it is nearly free to fix: hash the
generated body and reuse.

**Epilogue fusion into cuBLAS.** Using `addmm` rather than `mm` plus a
pointwise add would let cuBLAS apply the bias for free. Narrow, and it undoes
part of a decision made deliberately in section 5, so it needs measuring.

**Row tiling in reduction kernels.** One row per program is wasteful for short
rows. Attention softmax at batch 32 launches 49152 programs of 128 elements
each.

**More sequence lengths.** Everything here is at 128 tokens. Attention is
quadratic in sequence length and pointwise work is linear, so the balance
between matmul time and generated-kernel time moves with it, and the crossover
point probably moves too.

## 18. Appendix

### Environment

```
Development:  macOS, Python 3.14, torch 2.12.1, no CUDA, no Triton
Measurement:  NVIDIA A40 (Ampere, 46 GB), CUDA 12.4, driver 570.169
              Python 3.11.10, torch 2.4.1+cu124, Triton 3.0.0
CI:           ubuntu-latest, Python 3.11 and 3.12, CPU torch
```

### Size

```
7106 lines  compiler       (mlc/)
1867 lines  tests          (tests/)
 578 tests  ~90 seconds, CPU only
  21 commits
```

Largest modules: the operator registry (538), the runtime (474), the frontend
(387), elementwise fusion (381), reduction fusion (379).

### Reproducing

```bash
pytest tests/                                    # anywhere
bash scripts/gpu_check.sh                        # correctness, on a GPU
bash scripts/gpu_bench.sh                        # the sweep
python scripts/assemble_results.py --dir results --out results/RESULTS.md
python scripts/compare_inductor.py --dtype float16
```

### Related documents

| document | what it covers |
|---|---|
| [README.md](README.md) | overview and headline numbers |
| [docs/fusion-legality.md](docs/fusion-legality.md) | every legality rule and the cost model |
| [docs/benchmarks.md](docs/benchmarks.md) | benchmark protocol and caveats |
| [docs/where-torch-compile-wins.md](docs/where-torch-compile-wins.md) | the crossover analysis |
| [results/RESULTS.md](results/RESULTS.md) | every measurement |
| [results/gpu_check.log](results/gpu_check.log) | verification evidence |
