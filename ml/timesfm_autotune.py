"""Opt-in empirical workload probe for a configured TimesFM runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from ml.timesfm_runtime import TimesFMRuntime


@dataclass(frozen=True)
class AutotuneTrial:
    series_per_call: int
    elapsed_seconds: float
    series_per_second: float
    peak_vram_gb: float | None


@dataclass(frozen=True)
class AutotuneResult:
    profile_name: str
    context: int
    horizon: int
    recommended_chunk_size: int
    trials: tuple[AutotuneTrial, ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _candidate_sizes(limit: int) -> list[int]:
    values = [1]
    while values[-1] * 2 <= limit:
        values.append(values[-1] * 2)
    if values[-1] != limit:
        values.append(limit)
    return values


def autotune_runtime(
    runtime: TimesFMRuntime,
    horizon: int = 5,
    max_series_per_call: int | None = None,
) -> AutotuneResult:
    """Measure representative calls and recommend a workload chunk size.

    This is intentionally opt-in because it loads model weights and exercises
    the accelerator. It tunes throughput only; it does not tune forecast quality.
    """
    limit = max_series_per_call or runtime.config.chunk_size
    limit = max(1, min(limit, runtime.config.chunk_size))
    context = runtime.config.effective_context
    base = np.linspace(100.0, 110.0, context, dtype=np.float32)
    trials: list[AutotuneTrial] = []
    failures: list[str] = []

    runtime.forecast([base], horizon=horizon)  # load and warm up
    for size in _candidate_sizes(limit):
        peak_vram = None
        try:
            try:
                import torch

                if runtime.status.resolved_device == "cuda":
                    torch.cuda.reset_peak_memory_stats(0)
            except (ImportError, RuntimeError):
                torch = None

            started = time.perf_counter()
            runtime.forecast([base] * size, horizon=horizon)
            elapsed = max(time.perf_counter() - started, 1e-9)
            if torch is not None and runtime.status.resolved_device == "cuda":
                peak_vram = torch.cuda.max_memory_allocated(0) / 1024**3
            trials.append(
                AutotuneTrial(
                    series_per_call=size,
                    elapsed_seconds=elapsed,
                    series_per_second=size / elapsed,
                    peak_vram_gb=peak_vram,
                )
            )
        except RuntimeError as error:
            failures.append(f"{size} series: {error}")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            break

    if not trials:
        raise RuntimeError("TimesFM autotune did not complete any successful trials")
    total_vram = runtime.status.gpu_memory_gb
    eligible = trials
    if total_vram:
        memory_limit = total_vram * runtime.config.memory_target_fraction
        eligible = [
            trial
            for trial in trials
            if trial.peak_vram_gb is None or trial.peak_vram_gb <= memory_limit
        ] or trials[:1]
    recommended = max(eligible, key=lambda trial: trial.series_per_second)
    return AutotuneResult(
        profile_name=runtime.config.profile_name,
        context=context,
        horizon=horizon,
        recommended_chunk_size=recommended.series_per_call,
        trials=tuple(trials),
        failures=tuple(failures),
    )
