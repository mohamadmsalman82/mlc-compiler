# Benchmark results

Measured on a rented **NVIDIA A40** (46 GB, Ampere), torch 2.4.1+cu124,
Triton 3.0.0, driver 570.169. Sequence length 128 throughout. Protocol and
caveats are in [../docs/benchmarks.md](../docs/benchmarks.md); the analysis of
where `torch.compile` wins is in
[../docs/where-torch-compile-wins.md](../docs/where-torch-compile-wins.md).

Cost-model constants were measured on this device rather than taken from the
spec sheet:

```
NVIDIA A40 (measured): 561 GB/s, 28.0 TFLOP/s fp32,
                       6.9 us/launch (1.0 us captured), L2 6 MiB
  -> flops_per_byte=49.9, launch_overhead_bytes=3857 KB (575 KB captured)
  spec sheet says 696 GB/s and 37.4 TFLOP/s; measured is 81% and 75% of that
```

The 6.9x gap between an uncaptured launch and a captured one is the whole
reason the cost model prices a saved launch differently when capture is on.

## The short version

**Batch size 1 is where specializing pays.** BERT-base is 5.48x faster than
eager and 3.05x faster than `torch.compile`; GPT-2 small is 4.73x and 2.19x.

**By batch 8 it has stopped paying, and `torch.compile` is ahead** at 0.92x
(BERT) and 0.89x (GPT-2), falling to 0.85x and 0.71x at batch 32. That is the
honest result and the reason for the analysis document: the cause is a
specific structural choice, not a missing tuning knob.

**Memory planning wins everywhere and wins more with batch size**, because it
is the one advantage that does not depend on launch overhead: 292 MB against
346 for BERT at batch 32, and 1194 MB against 1593 for GPT-2, a 25% reduction.


## Summary

Best mlc variant against each baseline. Latency is the median of 50 timed calls after warmup; memory is peak allocated.

| model | dtype | batch | eager ms | torch.compile ms | t.c reduce-overhead ms | mlc ms | vs eager | vs best t.c | mlc peak MB | t.c peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert-base | float16 | 1 | 5.914 | 3.300 | 3.297 | 1.080 | 5.48x | 3.05x | 250.5 | 235.2 |
| bert-base | float16 | 32 | 16.004 | 11.744 | 11.747 | 13.833 | 1.16x | 0.85x | 302.4 | 346.1 |
| bert-base | float16 | 8 | 5.102 | 3.436 | 3.452 | 3.735 | 1.37x | 0.92x | 261.8 | 262.5 |
| bert-base | float32 | 1 | 5.843 | 3.321 | 3.138 | 2.903 | 2.01x | 1.08x | 446.5 | 433.7 |
| bert-base | float32 | 8 | 15.014 | 13.654 | 13.650 | 14.216 | 1.06x | 0.96x | 471.2 | 484.5 |
| gpt2-small | float16 | 1 | 5.395 | 2.506 | 2.494 | 1.141 | 4.73x | 2.19x | 375.3 | 369.1 |
| gpt2-small | float16 | 32 | 21.521 | 14.395 | 14.332 | 20.303 | 1.06x | 0.71x | 1193.7 | 1593.2 |
| gpt2-small | float16 | 8 | 8.159 | 4.746 | 4.743 | 5.340 | 1.53x | 0.89x | 559.1 | 627.4 |

## Every variant

### bert-base  dtype=float16  batch=1  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 5.914 | 5.777 | 1.00x | 234.3 | 0.0e+00 |
| torch.compile |  | 3.300 | 3.227 | 1.79x | 235.2 | 3.2e-03 |
| torch.compile/reduce-overhead |  | 3.297 | 3.238 | 1.79x | 235.2 | 3.2e-03 |
| mlc/no-passes | 682 | 39.968 | 39.520 | 0.15x | 237.8 | 2.6e-03 |
| mlc/elementwise | 322 | 19.014 | 18.703 | 0.31x | 233.9 | 2.2e-03 |
| mlc/+reduction | 200 | 7.938 | 7.734 | 0.74x | 233.7 | 2.9e-03 |
| mlc/+memory | 200 | 3.784 | 3.731 | 1.56x | 234.4 | 2.9e-03 |
| mlc/+cuda-graphs | 200 | 1.080 | 1.077 | 5.48x | 250.5 | 2.9e-03 |

### bert-base  dtype=float16  batch=32  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 16.004 | 15.956 | 1.00x | 292.1 | 0.0e+00 |
| torch.compile |  | 11.744 | 11.728 | 1.36x | 346.1 | 4.2e-03 |
| torch.compile/reduce-overhead |  | 11.747 | 11.735 | 1.36x | 346.1 | 4.2e-03 |
| mlc/no-passes | 718 | 44.165 | 44.098 | 0.36x | 382.2 | 3.2e-03 |
| mlc/elementwise | 359 | 20.137 | 20.029 | 0.79x | 292.4 | 2.9e-03 |
| mlc/+reduction | 236 | 14.560 | 14.522 | 1.10x | 286.2 | 3.7e-03 |
| mlc/+memory | 236 | 14.127 | 14.113 | 1.13x | 292.2 | 3.7e-03 |
| mlc/+cuda-graphs | 236 | 13.833 | 13.812 | 1.16x | 302.4 | 3.7e-03 |

### bert-base  dtype=float16  batch=8  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 5.102 | 5.043 | 1.00x | 249.0 | 0.0e+00 |
| torch.compile |  | 3.436 | 3.430 | 1.48x | 262.5 | 3.1e-03 |
| torch.compile/reduce-overhead |  | 3.452 | 3.438 | 1.48x | 262.5 | 3.1e-03 |
| mlc/no-passes | 718 | 38.114 | 26.746 | 0.13x | 270.3 | 2.7e-03 |
| mlc/elementwise | 358 | 18.406 | 18.188 | 0.28x | 247.1 | 2.7e-03 |
| mlc/+reduction | 236 | 8.436 | 8.140 | 0.60x | 245.9 | 3.5e-03 |
| mlc/+memory | 236 | 4.142 | 4.105 | 1.23x | 247.8 | 3.5e-03 |
| mlc/+cuda-graphs | 236 | 3.735 | 3.730 | 1.37x | 261.8 | 3.5e-03 |

### bert-base  dtype=float32  batch=1  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 5.843 | 5.765 | 1.00x | 431.6 | 0.0e+00 |
| torch.compile |  | 3.321 | 3.288 | 1.76x | 433.7 | 1.4e-06 |
| torch.compile/reduce-overhead |  | 3.138 | 3.115 | 1.86x | 433.7 | 1.4e-06 |
| mlc/no-passes | 584 | 34.489 | 33.910 | 0.17x | 433.2 | 7.4e-07 |
| mlc/elementwise | 322 | 21.014 | 20.521 | 0.28x | 431.2 | 7.3e-07 |
| mlc/+reduction | 200 | 8.629 | 8.286 | 0.68x | 431.2 | 8.0e-07 |
| mlc/+memory | 200 | 3.363 | 3.284 | 1.74x | 430.6 | 8.0e-07 |
| mlc/+cuda-graphs | 200 | 2.903 | 2.894 | 2.01x | 446.5 | 8.0e-07 |

### bert-base  dtype=float32  batch=8  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 15.014 | 14.990 | 1.00x | 457.5 | 0.0e+00 |
| torch.compile |  | 13.654 | 13.624 | 1.10x | 484.5 | 1.9e-06 |
| torch.compile/reduce-overhead |  | 13.650 | 13.640 | 1.10x | 484.5 | 1.9e-06 |
| mlc/no-passes | 620 | 34.724 | 24.254 | 0.43x | 465.9 | 1.3e-06 |
| mlc/elementwise | 358 | 21.264 | 18.158 | 0.71x | 453.9 | 1.1e-06 |
| mlc/+reduction | 236 | 15.000 | 14.827 | 1.00x | 454.6 | 1.6e-06 |
| mlc/+memory | 236 | 14.471 | 14.390 | 1.04x | 458.5 | 1.6e-06 |
| mlc/+cuda-graphs | 236 | 14.216 | 14.206 | 1.06x | 471.2 | 1.6e-06 |

### gpt2-small  dtype=float16  batch=1  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 5.395 | 5.360 | 1.00x | 369.3 | 0.0e+00 |
| torch.compile |  | 2.506 | 2.485 | 2.15x | 369.1 | 4.9e-03 |
| torch.compile/reduce-overhead |  | 2.494 | 2.481 | 2.16x | 369.1 | 4.9e-03 |
| mlc/no-passes | 627 | 23.781 | 23.542 | 0.23x | 374.5 | 3.4e-03 |
| mlc/elementwise | 271 | 15.742 | 15.614 | 0.34x | 371.2 | 3.4e-03 |
| mlc/+reduction | 148 | 5.859 | 5.741 | 0.92x | 370.6 | 3.7e-03 |
| mlc/+memory | 148 | 2.939 | 2.879 | 1.84x | 359.2 | 3.7e-03 |
| mlc/+cuda-graphs | 148 | 1.141 | 1.139 | 4.73x | 375.3 | 3.7e-03 |

### gpt2-small  dtype=float16  batch=32  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 21.521 | 21.417 | 1.00x | 1522.0 | 0.0e+00 |
| torch.compile |  | 14.395 | 14.308 | 1.49x | 1593.2 | 4.8e-03 |
| torch.compile/reduce-overhead |  | 14.332 | 14.289 | 1.50x | 1593.2 | 4.8e-03 |
| mlc/no-passes | 663 | 49.756 | 49.675 | 0.43x | 1660.0 | 3.9e-03 |
| mlc/elementwise | 307 | 25.935 | 25.700 | 0.83x | 1582.2 | 3.9e-03 |
| mlc/+reduction | 184 | 20.780 | 20.593 | 1.04x | 1564.0 | 3.9e-03 |
| mlc/+memory | 184 | 20.462 | 20.357 | 1.05x | 1183.4 | 3.9e-03 |
| mlc/+cuda-graphs | 184 | 20.303 | 20.130 | 1.06x | 1193.7 | 3.9e-03 |

### gpt2-small  dtype=float16  batch=8  seq=128

| variant | kernels | latency ms | p10 ms | vs eager | peak MB | max err |
|---|---:|---:|---:|---:|---:|---:|
| eager |  | 8.159 | 8.131 | 1.00x | 628.8 | 0.0e+00 |
| torch.compile |  | 4.746 | 4.738 | 1.72x | 627.4 | 4.4e-03 |
| torch.compile/reduce-overhead |  | 4.743 | 4.734 | 1.72x | 627.4 | 4.4e-03 |
| mlc/no-passes | 663 | 31.951 | 31.652 | 0.26x | 663.2 | 3.9e-03 |
| mlc/elementwise | 307 | 18.430 | 18.197 | 0.44x | 644.1 | 3.9e-03 |
| mlc/+reduction | 184 | 8.086 | 7.924 | 1.01x | 639.3 | 3.9e-03 |
| mlc/+memory | 184 | 5.635 | 5.627 | 1.45x | 544.7 | 3.9e-03 |
| mlc/+cuda-graphs | 184 | 5.340 | 5.336 | 1.53x | 559.1 | 3.9e-03 |

## Per-pass attribution

Each row is the gain over the row above, so the difference between adjacent rows is one pass.

### bert-base  dtype=float16  batch=1

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 682 | 39.968 | baseline | 1.00x |
| mlc/elementwise | 322 | 19.014 | 2.10x | 2.10x |
| mlc/+reduction | 200 | 7.938 | 2.40x | 5.04x |
| mlc/+memory | 200 | 3.784 | 2.10x | 10.56x |
| mlc/+cuda-graphs | 200 | 1.080 | 3.50x | 37.01x |

### bert-base  dtype=float16  batch=32

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 718 | 44.165 | baseline | 1.00x |
| mlc/elementwise | 359 | 20.137 | 2.19x | 2.19x |
| mlc/+reduction | 236 | 14.560 | 1.38x | 3.03x |
| mlc/+memory | 236 | 14.127 | 1.03x | 3.13x |
| mlc/+cuda-graphs | 236 | 13.833 | 1.02x | 3.19x |

### bert-base  dtype=float16  batch=8

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 718 | 38.114 | baseline | 1.00x |
| mlc/elementwise | 358 | 18.406 | 2.07x | 2.07x |
| mlc/+reduction | 236 | 8.436 | 2.18x | 4.52x |
| mlc/+memory | 236 | 4.142 | 2.04x | 9.20x |
| mlc/+cuda-graphs | 236 | 3.735 | 1.11x | 10.21x |

### bert-base  dtype=float32  batch=1

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 584 | 34.489 | baseline | 1.00x |
| mlc/elementwise | 322 | 21.014 | 1.64x | 1.64x |
| mlc/+reduction | 200 | 8.629 | 2.44x | 4.00x |
| mlc/+memory | 200 | 3.363 | 2.57x | 10.26x |
| mlc/+cuda-graphs | 200 | 2.903 | 1.16x | 11.88x |

### bert-base  dtype=float32  batch=8

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 620 | 34.724 | baseline | 1.00x |
| mlc/elementwise | 358 | 21.264 | 1.63x | 1.63x |
| mlc/+reduction | 236 | 15.000 | 1.42x | 2.31x |
| mlc/+memory | 236 | 14.471 | 1.04x | 2.40x |
| mlc/+cuda-graphs | 236 | 14.216 | 1.02x | 2.44x |

### gpt2-small  dtype=float16  batch=1

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 627 | 23.781 | baseline | 1.00x |
| mlc/elementwise | 271 | 15.742 | 1.51x | 1.51x |
| mlc/+reduction | 148 | 5.859 | 2.69x | 4.06x |
| mlc/+memory | 148 | 2.939 | 1.99x | 8.09x |
| mlc/+cuda-graphs | 148 | 1.141 | 2.58x | 20.84x |

### gpt2-small  dtype=float16  batch=32

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 663 | 49.756 | baseline | 1.00x |
| mlc/elementwise | 307 | 25.935 | 1.92x | 1.92x |
| mlc/+reduction | 184 | 20.780 | 1.25x | 2.39x |
| mlc/+memory | 184 | 20.462 | 1.02x | 2.43x |
| mlc/+cuda-graphs | 184 | 20.303 | 1.01x | 2.45x |

### gpt2-small  dtype=float16  batch=8

| pass enabled | kernels | latency ms | gain | cumulative |
|---|---:|---:|---:|---:|
| mlc/no-passes | 663 | 31.951 | baseline | 1.00x |
| mlc/elementwise | 307 | 18.430 | 1.73x | 1.73x |
| mlc/+reduction | 184 | 8.086 | 2.28x | 3.95x |
| mlc/+memory | 184 | 5.635 | 1.44x | 5.67x |
| mlc/+cuda-graphs | 184 | 5.340 | 1.06x | 5.98x |

