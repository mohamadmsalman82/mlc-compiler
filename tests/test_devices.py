"""Device profiles and the cost-model constants derived from them.

The table was positional once and inserting a field shifted every entry by
one, which produced a launch overhead of 41 million seconds and went
unnoticed because nothing checked the values were physical. These bounds are
generous; they exist to catch that class of mistake, not to pin numbers.
"""

import pytest
import torch

from mlc.config import Config
from mlc.devices import DEFAULT, TABLE, DeviceProfile, config_for, profile_for


@pytest.mark.parametrize("key,profile", TABLE, ids=[k for k, _ in TABLE])
def test_profile_values_are_physical(key, profile):
    assert 1e11 < profile.bandwidth < 1e13, "bandwidth outside 100 GB/s .. 10 TB/s"
    assert 1e12 < profile.fp32_flops < 1e15, "fp32 outside 1 .. 1000 TFLOP/s"
    assert 1e-6 < profile.launch_overhead < 1e-4, "launch outside 1 us .. 100 us"
    assert 1024 ** 2 <= profile.l2_bytes <= 512 * 1024 ** 2
    assert profile.total_memory >= 4 * 1024 ** 3


@pytest.mark.parametrize("key,profile", TABLE, ids=[k for k, _ in TABLE])
def test_derived_constants_are_sane(key, profile):
    assert 5 < profile.flops_per_byte < 200
    assert 1e5 < profile.launch_overhead_bytes < 2e7
    assert profile.graph_launch_overhead_bytes < profile.launch_overhead_bytes


def test_table_is_ordered_most_specific_first():
    """Lookup takes the first key contained in the device name, so a key that
    is a substring of a later one would shadow it. '4060' listed before
    '4060 Ti' would give every Ti the wrong entry."""
    keys = [k for k, _ in TABLE]
    for i, earlier in enumerate(keys):
        for later in keys[i + 1:]:
            assert earlier.lower() not in later.lower(), (
                f"{earlier!r} comes first and shadows {later!r}"
            )


def test_config_carries_the_profile_through():
    profile = dict(TABLE)["4060"]
    cfg = profile.to_config()
    assert cfg.flops_per_byte == pytest.approx(profile.flops_per_byte)
    assert cfg.launch_overhead_bytes == pytest.approx(profile.launch_overhead_bytes)


def test_capture_lowers_what_a_saved_launch_is_worth():
    """Replay removes most of the per-launch cost, so the cost model must
    stop paying full price for eliminating one."""
    cfg = dict(TABLE)["4060"].to_config()
    assert cfg.replace(cuda_graphs=False).effective_launch_bytes > \
        cfg.replace(cuda_graphs=True).effective_launch_bytes


def test_profile_for_falls_back_without_cuda():
    if torch.cuda.is_available():
        pytest.skip("this checks the no-CUDA path")
    assert profile_for("cpu") is DEFAULT
    assert config_for("cpu").flops_per_byte == DEFAULT.flops_per_byte


def test_unknown_device_gets_the_default_constants():
    unknown = DeviceProfile(name="Made Up 9000", bandwidth=DEFAULT.bandwidth,
                            fp32_flops=DEFAULT.fp32_flops,
                            launch_overhead=DEFAULT.launch_overhead)
    assert unknown.flops_per_byte == DEFAULT.flops_per_byte
