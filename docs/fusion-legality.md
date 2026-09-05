# Fusion legality

Fusion is the decision to compute several graph nodes in one kernel. It is
only worth doing if the result is the same, and "the same" turns out to
require more care than it sounds. This is the full set of conditions the
compiler checks, why each one exists, and what breaks without it.

Two things are separate throughout and worth keeping separate while reading:
**legality** is whether a fusion produces the right answer, and **profit** is
whether it is faster. A legal fusion that loses is a missed optimisation; an
illegal one is a wrong answer. Everything in sections 1 to 5 is legality.
Section 6 is profit.

---

## 1. Acyclicity

Fusing groups A and B means running them as one kernel. If some third group C
reads A and feeds B, the merged kernel would have to run both before C
(because C depends on A) and after it (because B depends on C). No schedule
exists.

The check is over the *group* graph, not the node graph, and it has to be
re-checked after every merge, because merging changes reachability. Each
group carries a bitset of its transitive ancestors and another of its
descendants, so the test is

```
between = (desc[A] & anc[B]) | (desc[B] & anc[A])
legal   = (between & ~(A | B)) == 0
```

Both fusion passes run over the same group graph so this check covers merges
between them as well as within them.

*Without it:* the scheduler produces a kernel order that cannot be topologically
sorted, and the compiler either crashes in `roots_in_order` or silently reads
an uninitialised buffer. The test `test_group_graph_refuses_a_cycle` builds
the minimal case: `a = x * 2`, `b = linear(a)`, `return a + b`, where the two
pointwise ops must not merge across the matmul.

---

## 2. Iteration space

Every node in a kernel is evaluated at every point of one iteration space.
A node can only join if its output can be indexed there.

### Elementwise kernels

The space is the member output shape with the most elements. A node fits if
either

- its output **broadcasts** to that space, or
- its output has the **same element count**.

The second rule is what lets fusion cross a reshape. A linear layer produces
`[B*T, D]`; the next op reads a `[B, T, D]` view of the same buffer. Both are
contiguous, so both flatten to the same row-major order, and a flat index
means the same element in either frame. Refusing this would end a kernel at
every reshape, and a transformer has one after every projection.

That rule is sound only because both operands are contiguous allocations. A
*permuted* view with the same element count is a different order entirely, and
the compiler must not treat it as interchangeable. It does not: the check is
not on element count alone but on the canonical index map (section 3).

### Reduction kernels

The space is a `(row, column)` pair: kept dimensions on the row axis, reduced
dimensions on the column axis. A member's output is classified as

- **row** if it does not vary along any reduced axis, or
- **element** if it does.

The classification is subtler than "smaller shape". A reduction with
`keepdim=True` has shape `[B, T, 1]`, and a layer norm weight has shape `[D]`.
Both broadcast to the element shape `[B, T, D]` perfectly well, but the first
is one value per program and the second is a vector. What separates them is
whether the reduced positions are size 1 after right-alignment. A
`keepdim=False` result has the reduced dimensions removed entirely and does
not right-align against the element shape at all, so it is checked separately.

Unlike the elementwise pass, **reduction kernels do not cross reshapes**. The
same-element-count shortcut is not sound here: flat order depends on which
frame you flatten in, and the row/column split makes those differ whenever the
reduced axis is not trailing. Refusing is a missed optimisation, not a wrong
answer, and the alternative is a class of bug that would only appear on
reductions over interior axes.

---

## 3. Index maps and register reuse

Each operand carries an affine map from the kernel's iteration variables to a
buffer offset:

```
offset = base + sum over axes, dims of ((var // out_stride) % size) * in_stride
```

It is canonicalised: size-1 dimensions dropped, adjacent dimensions merged
when contiguous in *both* the iteration space and the operand, stride-0
dimensions dropped. Every coefficient is a compile-time constant because
shapes are static.

The canonical form is what makes register reuse decidable. A value produced
inside the kernel is already in a register; reading it back is legal exactly
when the read touches the same address the producer wrote at the same
iteration. With canonical maps that is structural equality:

```
if index_map(read) == index_map(producer_output): use the register
else: emit a load
```

This is the mechanism behind the reshape rule above. `[16, 192]` and
`[2, 8, 192]` contiguous views of one buffer both canonicalise to the single
term `idx`, so they compare equal and the value never reaches memory. A
transposed view of the same buffer canonicalises to
`(idx // 4) + (idx % 4) * 8`, does not compare equal, and correctly falls back
to a load.

It is also how broadcasting disappears. A bias vector read across a batch
reduces to `idx % D`; a row statistic read back across its row loses its
column term entirely, which is why the reduction epilogue costs nothing.

---

## 4. Escape

A node whose output is smaller than the kernel's iteration space is evaluated
at every point of that space, which means several iterations compute the same
value. That is fine as long as nobody outside the kernel needs it. It is not
fine if the value has to be stored: several iterations would write to the same
address.

So: **a value that is broadcast inside its kernel may not be visible outside
it.** Concretely, a node may only join a group with a larger iteration space
if every consumer of its result joins too, and its buffer is not a graph
output.

In reduction kernels the same rule applies in both spaces. A value is
storable only when every point of its own shape is visited exactly once:
element values must have the full element count, row values the full row
count. Anything else stays internal.

*Without it:* the compiler emits a kernel with a broadcast store. On a GPU
that is a race between threads writing the same address. They happen to write
the same value, so it usually works, which is the worst kind of bug.

---

## 5. Associativity and numerical stability

Splitting a reduction across thread blocks, or across chunks of a streamed
row, reorders the combining operation. That requires associativity. Every kind
the compiler emits (`sum`, `max`, `min`, `prod`) is associative in exact
arithmetic, and the check exists so that adding one that is not fails loudly
rather than silently reordering.

Floating-point addition is not associative in the strict sense. Reassociating
it is standard and the compiler does it deliberately; what it does not do is
pretend the results are bit-identical. The test suite is split accordingly:
passes that only change *where* a value is computed must be bit-identical to
the unfused baseline, while reduction fusion is bounded by tolerance.

Two places where the choice of formula is a correctness question, not a
performance one:

**Variance.** The one-pass form `E[x^2] - E[x]^2` subtracts two large numbers
to get a small one. For a batch with mean 1e4 and variance 1, both terms are
near 1e8, where the fp32 ulp is about 8: the answer is noise. The compiler
expands variance into two reductions over the same resident row instead --
the mean, then the sum of squared deviations from it -- which is exact and
costs nothing extra, because the second pass is over registers rather than
memory. `test_variance_uses_the_two_pass_formula_not_the_cancelling_one`
checks the result against a float64 reference and requires it to beat the
cancelling form by two orders of magnitude.

**Softmax.** Subtracting the row maximum before exponentiating is what keeps
`exp` from overflowing. In a persistent kernel this is free: the row is
already in registers, so the max and the sum are two cross-lane reduces over
the same data and the kernel still reads memory once.

When the row does not fit registers the kernel streams it, and then the max
and the sum would be two separate passes over memory. The compiler recognises
the pattern -- a `max` reduction followed by a `sum` of `exp(x - max)` -- and
fuses them into the online recurrence

```
m' = max(m, max(chunk))
l' = l * exp(m - m') + sum(exp(chunk - m'))
```

which computes both in one pass and is exactly as stable, since every
exponent stays non-positive. This is the identity flash attention is built
on. It is applied only on the streamed path; a persistent kernel has nothing
to gain from it.

---

## 6. Profit

Everything above is about correctness. Whether a legal fusion is worth doing
is a separate question, and the answer is denominated in bytes of HBM traffic,
because inference on these models is memory bound.

Two other effects are converted into that currency:

- **arithmetic**, at `flops_per_byte` -- the operations the device retires
  while moving one byte. About 12 for an A100 in fp32, about 40 for a 4090.
- **launch overhead**, at `launch_overhead_bytes` -- what one kernel launch
  costs expressed as forgone bandwidth. Roughly 3 microseconds, which at
  1.5 TB/s is several megabytes.

Both are device properties and are meant to be set per GPU.

A merge is taken when

```
saved_stores + saved_loads + saved_launches * launch_cost
    >  extra_loads + extra_evaluations * ops / flops_per_byte
```

Merging two kernels never adds loads: the fused kernel reads the union of what
the two read. It can add arithmetic, when a group is pulled into a larger
iteration space and its work repeats per point.

The launch term is the reason fusion pays so much more at batch size 1. The
traffic terms shrink with batch size; the launch term does not. It is also why
CUDA graph capture and fusion partly substitute for each other: capture makes
launches cheap, which lowers the value of merging kernels that were only worth
merging for the launch saving.

### Recomputation

Union-find fusion attaches a producer to exactly one consumer. A producer with
several consumers still writes its result for the rest. When the producer is
cheap, evaluating it again inside each consumer beats the store plus the
reloads:

```
benefit = one store + one load per consumer + one launch
cost    = (total consumer iterations - own iterations) * ops / flops_per_byte
          + operands the consumers do not already read
```

subject to a hard cap on expression size, so a long chain cannot be duplicated
into many consumers on the strength of a favourable ratio.

The pass runs after grouping rather than during it, because duplicating a node
puts it in two kernels at once and union-find cannot represent that.

### Capacity

The reduced axis has to fit the register budget for a persistent kernel. Above
`max_persistent_row` the kernel streams the row instead: one pass per
reduction, plus one for the element-space stores. That is correct but costs
real bandwidth, which is what makes the online-softmax fusion worth
implementing.

---

## What is deliberately not done

- **Fusing into or through a matmul.** Matmuls dispatch to cuBLAS and act as
  barriers. An epilogue fusion (bias and activation folded into the GEMM)
  needs a GEMM of one's own, which is explicitly out of scope.
- **Reshape-crossing inside reduction kernels**, for the reason in section 2.
- **Fusing reductions over different axes of the same tensor.** They have
  different frames and cannot share an iteration space. Two reductions over
  the *same* axis do fuse, which is what collapses layer norm's mean and
  variance into one pass over the row.
- **Recomputing across a reduction.** A value already reduced to a scalar is
  carried between passes; recomputing what produced it would drag the whole
  prologue into every later pass.
