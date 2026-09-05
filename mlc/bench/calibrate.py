"""Measure the cost model's constants on the actual device.

Published specifications are the wrong input for two reasons. Vendor peak
fp32 is not reachable by pointwise code, which is bound by load/store issue
long before it is bound by arithmetic. And launch overhead is as much a
property of the host CPU and the driver as of the GPU, so it cannot be looked
up at all.

Four measurements:

  * **bandwidth**, from a large streaming copy sized well past L2.
  * **fp32 throughput**, from a pointwise kernel doing many fused multiply-adds
    per element on data that fits in L2, so the arithmetic is what is timed.
  * **launch overhead**, from a long run of empty kernels.
  * **launch overhead under CUDA graph replay**, which is the number that
    applies once the schedule is captured, and is much smaller.

The last one matters for fusion decisions. Merging two kernels purely to
eliminate a launch is worth far less when launches are already nearly free,
so the compiler should be less willing to do it when capture is on.
"""

from __future__ import annotations

import statistics
import time

import torch

from ..devices import MB, DeviceProfile, profile_for


def _time_cuda(fn, iters: int, warmup: int = 5) -> float:
    """Seconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(7):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1e3 / iters)
    return statistics.median(samples)


def measure_bandwidth(device, nbytes: int = 256 * MB) -> float:
    """Bytes per second for a streaming copy, read plus write."""
    n = nbytes // 4
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty(n, dtype=torch.float32, device=device)
    a.normal_()
    per_call = _time_cuda(lambda: b.copy_(a), iters=20)
    return 2 * nbytes / per_call


def measure_launch_overhead(device, iters: int = 2000) -> float:
    """Seconds a single kernel launch costs when nothing is captured.

    Uses the smallest real kernel available so the measurement is a launch and
    not an empty submission the driver can elide.
    """
    x = torch.zeros(1, device=device)
    return _time_cuda(lambda: x.add_(0.0), iters=iters, warmup=50)


def measure_graph_launch_overhead(device, nodes: int = 512) -> float:
    """Seconds per kernel when the schedule is replayed as a CUDA graph."""
    x = torch.zeros(1, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for _ in range(nodes):
                x.add_(0.0)
    torch.cuda.current_stream().wait_stream(side)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(nodes):
            x.add_(0.0)
    per_replay = _time_cuda(g.replay, iters=20)
    return per_replay / nodes


def measure_fp32_throughput(device, n: int = 1 << 20, chain: int = 512) -> float:
    """Operations per second for pointwise fp32 arithmetic.

    Needs a Triton kernel: a chain of fused multiply-adds in torch would be a
    chain of kernel launches, and would measure bandwidth instead. Returns 0
    when Triton is unavailable, and the caller falls back to the table.
    """
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return 0.0

    @triton.jit
    def _fma_chain(x_ptr, out_ptr, N: tl.constexpr, CHAIN: tl.constexpr,
                   BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + idx, mask=idx < N, other=0.0)
        acc = v
        for _ in tl.static_range(CHAIN):
            acc = acc * 1.0000001 + 0.0000001
        tl.store(out_ptr + idx, acc, mask=idx < N)

    x = torch.randn(n, device=device)
    out = torch.empty_like(x)
    block = 1024
    grid = (triton.cdiv(n, block),)
    per_call = _time_cuda(
        lambda: _fma_chain[grid](x, out, N=n, CHAIN=chain, BLOCK=block, num_warps=4),
        iters=20,
    )
    # Two operations per FMA, so the count is not flattered by calling it one.
    return (n * chain * 2) / per_call


def calibrate(device: torch.device | str = "cuda", quick: bool = False) -> DeviceProfile:
    """Measure everything and return a profile. Falls back per-measurement."""
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("calibration needs a CUDA device")

    table = profile_for(device)
    props = torch.cuda.get_device_properties(device)

    bandwidth = measure_bandwidth(device, 64 * MB if quick else 256 * MB)
    launch = measure_launch_overhead(device, 500 if quick else 2000)
    try:
        graph_launch = measure_graph_launch_overhead(device, 128 if quick else 512)
    except Exception:
        graph_launch = launch / 4
    flops = measure_fp32_throughput(device, chain=128 if quick else 512)

    profile = DeviceProfile(
        name=props.name,
        bandwidth=bandwidth,
        fp32_flops=flops or table.fp32_flops,
        launch_overhead=launch,
        graph_launch_overhead=graph_launch,
        l2_bytes=getattr(props, "L2_cache_size", 0) or table.l2_bytes,
        total_memory=props.total_memory,
        source="measured" if flops else "measured (fp32 from table)",
    )
    return profile


def report(profile: DeviceProfile, table: DeviceProfile | None = None) -> str:
    lines = [profile.summary()]
    if table is not None and table.source != "default":
        lines.append(
            f"  spec sheet says {table.bandwidth / 1e9:.0f} GB/s and "
            f"{table.fp32_flops / 1e12:.1f} TFLOP/s; measured is "
            f"{100 * profile.bandwidth / table.bandwidth:.0f}% and "
            f"{100 * profile.fp32_flops / table.fp32_flops:.0f}% of that"
        )
    return "\n".join(lines)
