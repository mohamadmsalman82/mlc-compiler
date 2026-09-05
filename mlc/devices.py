"""Device profiles: the constants the cost model is denominated in.

Two numbers decide almost every fusion decision, and both are properties of
the card rather than of the model:

  * ``flops_per_byte`` -- operations retired while one byte moves. It sets how
    willing the compiler is to recompute a value instead of storing it.
  * ``launch_overhead_bytes`` -- one kernel launch expressed as forgone
    bandwidth. It sets how much a merge is worth purely for eliminating a
    launch, which is the term that dominates at batch size 1.

These differ by more than an order of magnitude across cards. An A100 moves
1.5 TB/s and retires about 12 fp32 operations per byte; a 4060 moves 272 GB/s
and retires about 55. Using A100 constants on a 4060 makes the compiler four
times too eager to fuse for launch savings and three times too reluctant to
recompute, so the profile is not a detail.

The table below is from published specifications. Prefer
:func:`mlc.bench.calibrate.calibrate`, which measures the same numbers on the
actual device: vendor peak flops are rarely reachable by pointwise code, and
launch overhead depends as much on the host CPU and driver as on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config

MB = 1024 ** 2
GB = 1024 ** 3


@dataclass
class DeviceProfile:
    name: str
    #: achieved HBM bandwidth, bytes per second
    bandwidth: float
    #: fp32 throughput for non-tensor-core work, operations per second
    fp32_flops: float
    #: wall time one kernel launch adds, seconds
    launch_overhead: float
    #: wall time per kernel when the schedule is replayed as a CUDA graph.
    #: Much smaller, which is why capture and fusion partly substitute.
    graph_launch_overhead: float = 0.0
    l2_bytes: int = 0
    total_memory: int = 0
    #: "measured", "table", or "default"
    source: str = "table"

    @property
    def flops_per_byte(self) -> float:
        return self.fp32_flops / self.bandwidth

    @property
    def launch_overhead_bytes(self) -> float:
        return self.launch_overhead * self.bandwidth

    @property
    def graph_launch_overhead_bytes(self) -> float:
        """Per-kernel cost once the schedule is captured.

        Defaults to a quarter of the uncaptured cost when not measured, which
        is conservative: replay typically does better than that.
        """
        overhead = self.graph_launch_overhead or self.launch_overhead / 4
        return overhead * self.bandwidth

    def to_config(self, base: Config | None = None) -> Config:
        return (base or Config()).replace(
            flops_per_byte=self.flops_per_byte,
            launch_overhead_bytes=self.launch_overhead_bytes,
            graph_launch_overhead_bytes=self.graph_launch_overhead_bytes,
        )

    def summary(self) -> str:
        return (
            f"{self.name} ({self.source}): "
            f"{self.bandwidth / 1e9:.0f} GB/s, "
            f"{self.fp32_flops / 1e12:.1f} TFLOP/s fp32, "
            f"{self.launch_overhead * 1e6:.1f} us/launch "
            f"({self.graph_launch_overhead * 1e6:.1f} us captured), "
            f"L2 {self.l2_bytes / MB:.0f} MiB\n"
            f"  -> flops_per_byte={self.flops_per_byte:.1f}, "
            f"launch_overhead_bytes={self.launch_overhead_bytes / 1e3:.0f} KB "
            f"({self.graph_launch_overhead_bytes / 1e3:.0f} KB captured)"
        )


#: Published specifications, keyed by a substring of the device name. Ordered
#: most specific first, since "4060 Ti" also contains "4060". Keyword
#: arguments throughout: these were positional once, and inserting a field
#: silently shifted every entry by one.
def _spec(name, gb_s, tflops, launch_us, l2_mb, mem_gb) -> DeviceProfile:
    return DeviceProfile(
        name=name,
        bandwidth=gb_s * 1e9,
        fp32_flops=tflops * 1e12,
        launch_overhead=launch_us * 1e-6,
        l2_bytes=l2_mb * MB,
        total_memory=mem_gb * GB,
    )


TABLE: list[tuple[str, DeviceProfile]] = [
    # Most specific first. A key that is a substring of a later one shadows
    # it: "A40" sits inside "RTX A4000", so the workstation cards have to be
    # listed before the bare datacenter names. test_table_is_ordered_most_
    # specific_first enforces this, and caught exactly that pair.
    ("4060 Ti",      _spec("RTX 4060 Ti",      288,  22.06, 3.5, 32,  8)),
    ("4060 Laptop",  _spec("RTX 4060 Laptop",  256,  11.61, 4.0, 24,  8)),
    ("4060",         _spec("RTX 4060",         272,  15.11, 3.5, 24,  8)),
    ("4070",         _spec("RTX 4070",         504,  29.15, 3.5, 36, 12)),
    ("4080",         _spec("RTX 4080",         717,  48.74, 3.0, 64, 16)),
    ("4090",         _spec("RTX 4090",        1008,  82.58, 3.0, 72, 24)),
    ("3090",         _spec("RTX 3090",         936,  35.58, 3.5,  6, 24)),
    ("3080",         _spec("RTX 3080",         760,  29.77, 3.5,  5, 10)),
    ("RTX 6000 Ada", _spec("RTX 6000 Ada",     960,  91.06, 3.0, 96, 48)),
    ("RTX 5000 Ada", _spec("RTX 5000 Ada",     576,  65.28, 3.0, 64, 32)),
    ("RTX 4000 Ada", _spec("RTX 4000 Ada",     360,  26.73, 3.5, 48, 20)),
    ("RTX A6000",    _spec("RTX A6000",        768,  38.71, 3.0,  6, 48)),
    ("RTX A5000",    _spec("RTX A5000",        768,  27.77, 3.5,  6, 24)),
    ("RTX A4500",    _spec("RTX A4500",        640,  23.65, 3.5,  6, 20)),
    ("RTX A4000",    _spec("RTX A4000",        448,  19.17, 3.5,  4, 16)),
    ("A100",         _spec("A100",            1555,  19.49, 3.0, 40, 40)),
    ("H100",         _spec("H100",            3350,  66.91, 2.5, 50, 80)),
    ("A40",          _spec("A40",              696,  37.42, 3.0,  6, 48)),
    ("A10G",         _spec("A10G",             600,  31.52, 3.5,  6, 24)),
    ("A10",          _spec("A10",              600,  31.24, 3.5,  6, 24)),
    ("L40S",         _spec("L40S",             864,  91.61, 3.0, 96, 48)),
    ("L40",          _spec("L40",              864,  90.52, 3.0, 96, 48)),
    ("L4",           _spec("L4",               300,  30.29, 3.5, 48, 24)),
    ("T4",           _spec("T4",               320,   8.14, 4.0,  4, 16)),
    ("V100",         _spec("V100",             900,  15.67, 3.5,  6, 16)),
]

#: What Config() uses when nothing is known. Deliberately an A100, because
#: that is what the docstrings in the cost model describe.
DEFAULT = _spec("unknown", 1555, 19.49, 3.0, 40, 0)
DEFAULT.source = "default"


def profile_for(device: torch.device | str | None = None) -> DeviceProfile:
    """Look up a profile by device name, falling back to the default."""
    device = torch.device(device) if device is not None else None
    if device is not None and device.type != "cuda":
        return DEFAULT
    if not torch.cuda.is_available():
        return DEFAULT
    props = torch.cuda.get_device_properties(device)
    name = props.name
    for key, profile in TABLE:
        if key.lower() in name.lower():
            found = DeviceProfile(**{**profile.__dict__})
            found.name = name
            found.total_memory = props.total_memory
            found.l2_bytes = getattr(props, "L2_cache_size", 0) or profile.l2_bytes
            return found
    unknown = DeviceProfile(**{**DEFAULT.__dict__})
    unknown.name = f"{name} (no table entry)"
    unknown.total_memory = props.total_memory
    unknown.l2_bytes = getattr(props, "L2_cache_size", 0) or DEFAULT.l2_bytes
    return unknown


def config_for(device: torch.device | str | None = None,
               base: Config | None = None) -> Config:
    return profile_for(device).to_config(base)
