"""Tests for hardware-aware TimesFM runtime profiles."""

import pytest

from ml.timesfm_profiles import HardwareCapabilities, select_runtime_profile
from ml.timesfm_runtime import TimesFMRuntimeConfig


@pytest.mark.parametrize(
    ("free_vram", "tier", "batch"),
    [
        (31.0, "enthusiast", 256),
        (15.0, "high", 128),
        (11.0, "performance", 64),
        (7.5, "standard", 64),
        (5.0, "constrained", 16),
    ],
)
def test_balanced_profile_uses_free_vram_tier(free_vram, tier, batch):
    hardware = HardwareCapabilities(
        "cuda", "Test GPU", total_vram_gb=32, free_vram_gb=free_vram
    )
    profile = select_runtime_profile(hardware, "watchlist", "balanced")
    assert profile.memory_tier == tier
    assert profile.batch_size == batch
    assert profile.chunk_size <= profile.batch_size * 2


def test_thorough_profile_expands_context_but_reduces_batch():
    hardware = HardwareCapabilities(
        "cuda", "RTX 5090", total_vram_gb=32, free_vram_gb=30
    )
    balanced = select_runtime_profile(hardware, "research", "balanced")
    thorough = select_runtime_profile(hardware, "research", "thorough")
    assert thorough.forecast_context > balanced.forecast_context
    assert thorough.batch_size < balanced.batch_size
    assert thorough.max_context <= 16384


def test_cpu_profile_scales_down_on_low_available_ram():
    hardware = HardwareCapabilities("cpu", available_ram_gb=8)
    profile = select_runtime_profile(hardware, "research", "auto")
    assert profile.batch_size == 8
    assert profile.forecast_context == 512
    assert profile.device == "cpu"


def test_profile_converts_to_runtime_config():
    hardware = HardwareCapabilities(
        "cuda", "RTX 5070", total_vram_gb=12, free_vram_gb=11
    )
    profile = select_runtime_profile(hardware, "interactive", "balanced")
    config = TimesFMRuntimeConfig.from_profile(profile)
    assert config.profile_name == profile.name
    assert config.effective_context == profile.forecast_context
    assert config.chunk_size == 1
    assert config.batch_size == 32


def test_rejects_unknown_profile_inputs():
    hardware = HardwareCapabilities("cpu", available_ram_gb=16)
    with pytest.raises(ValueError, match="use case"):
        select_runtime_profile(hardware, "unknown", "auto")
    with pytest.raises(ValueError, match="profile"):
        select_runtime_profile(hardware, "interactive", "turbo")
