"""Hardware-aware TimesFM runtime profiles.

Profiles control resource use and throughput. They deliberately do not select
trading thresholds, target representations, or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


PROFILE_PREFERENCES = ("auto", "fast", "balanced", "thorough")
USE_CASES = ("interactive", "watchlist", "backtest", "research")


@dataclass(frozen=True)
class HardwareCapabilities:
    device: str
    gpu_name: str | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    available_ram_gb: float | None = None

    @property
    def memory_tier(self) -> str:
        if self.device != "cuda":
            return "cpu"
        usable = self.free_vram_gb or self.total_vram_gb or 0.0
        if usable >= 24:
            return "enthusiast"
        if usable >= 14:
            return "high"
        if usable >= 10:
            return "performance"
        if usable >= 7:
            return "standard"
        return "constrained"


@dataclass(frozen=True)
class TimesFMProfile:
    name: str
    preference: str
    use_case: str
    device: str
    memory_tier: str
    max_context: int
    forecast_context: int
    max_horizon: int
    batch_size: int
    chunk_size: int
    memory_target_fraction: float
    rationale: str


def _available_ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def detect_hardware(torch_module: Any | None = None) -> HardwareCapabilities:
    """Detect available TimesFM hardware without loading model weights."""
    available_ram = _available_ram_gb()
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return HardwareCapabilities("cpu", available_ram_gb=available_ram)

    try:
        if not torch_module.cuda.is_available():
            return HardwareCapabilities("cpu", available_ram_gb=available_ram)
        properties = torch_module.cuda.get_device_properties(0)
        total = float(properties.total_memory) / 1024**3
        try:
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(0)
            free = float(free_bytes) / 1024**3
            total = float(total_bytes) / 1024**3
        except (AttributeError, RuntimeError, TypeError):
            free = total
        return HardwareCapabilities(
            device="cuda",
            gpu_name=torch_module.cuda.get_device_name(0),
            total_vram_gb=total,
            free_vram_gb=free,
            available_ram_gb=available_ram,
        )
    except (AttributeError, RuntimeError):
        return HardwareCapabilities("cpu", available_ram_gb=available_ram)


_TIER_BATCH = {
    "enthusiast": 256,
    "high": 128,
    "performance": 64,
    "standard": 64,
    "constrained": 16,
    "cpu": 32,
}

_USE_CASE = {
    "interactive": {"context": 1024, "cap": 2048, "chunk": 1},
    "watchlist": {"context": 1024, "cap": 2048, "chunk": 128},
    "backtest": {"context": 1024, "cap": 2048, "chunk": 64},
    "research": {"context": 2048, "cap": 4096, "chunk": 128},
}


def select_runtime_profile(
    hardware: HardwareCapabilities,
    use_case: str = "interactive",
    preference: str = "auto",
) -> TimesFMProfile:
    """Select a conservative profile from free memory and workload shape."""
    if use_case not in USE_CASES:
        raise ValueError(f"Unsupported TimesFM use case: {use_case}")
    if preference not in PROFILE_PREFERENCES:
        raise ValueError(f"Unsupported TimesFM profile: {preference}")

    resolved_preference = "balanced" if preference == "auto" else preference
    tier = hardware.memory_tier
    workload = _USE_CASE[use_case]
    context = int(workload["context"])
    context_cap = int(workload["cap"])
    batch = _TIER_BATCH[tier]
    chunk = min(int(workload["chunk"]), batch * 2)

    # Large compiled batches help broad sweeps but can waste memory and latency
    # when the caller needs only one forecast.
    if use_case == "interactive":
        batch = min(batch, 32)
    elif use_case == "backtest":
        batch = min(batch, 128)

    if tier == "constrained":
        context = min(context, 512)
        context_cap = min(context_cap, 1024)
    elif tier == "cpu":
        if (hardware.available_ram_gb or 0) < 12:
            batch = 8
            context = min(context, 512)
            context_cap = min(context_cap, 1024)
        chunk = min(chunk, batch)

    if resolved_preference == "fast":
        context = max(32, context // 2)
        context_cap = max(context, context_cap // 2)
        batch = min(batch, 64)
        chunk = min(chunk, batch)
        memory_fraction = 0.65
    elif resolved_preference == "thorough":
        # Thorough expands the feasible research envelope. The selected context
        # remains visible and must still be validated for predictive quality.
        context = min(context * 2, 8192)
        context_cap = min(max(context_cap * 2, context), 16384)
        batch = max(8, batch // 2)
        chunk = min(chunk, batch)
        memory_fraction = 0.82
    else:
        memory_fraction = 0.72

    label = hardware.gpu_name or "CPU"
    rationale = (
        f"{resolved_preference.title()} {use_case} profile for {label}; "
        f"selected from {tier} memory tier using currently available memory. "
        "Batch and chunk size affect throughput, not forecast methodology."
    )
    return TimesFMProfile(
        name=f"{resolved_preference}-{use_case}-{tier}",
        preference=resolved_preference,
        use_case=use_case,
        device=hardware.device,
        memory_tier=tier,
        max_context=context_cap,
        forecast_context=context,
        max_horizon=256,
        batch_size=batch,
        chunk_size=max(1, chunk),
        memory_target_fraction=memory_fraction,
        rationale=rationale,
    )
