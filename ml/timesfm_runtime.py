"""TimesFM 2.5 runtime, preflight checks, and probabilistic forecast types."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
import os
import shutil
import sys
from typing import Callable

import numpy as np

from ml.timesfm_profiles import TimesFMProfile


DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
QUANTILE_LEVELS = tuple(round(level / 10, 1) for level in range(1, 10))


@dataclass(frozen=True)
class TimesFMRuntimeConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    max_context: int = 1024
    forecast_context: int | None = None
    max_horizon: int = 256
    batch_size: int = 32
    chunk_size: int = 32
    memory_target_fraction: float = 0.72
    profile_name: str = "custom"
    use_case: str = "interactive"

    @property
    def effective_context(self) -> int:
        return min(self.forecast_context or self.max_context, self.max_context)

    def __post_init__(self) -> None:
        if self.max_context < 32:
            raise ValueError("TimesFM max_context must be at least 32")
        if self.effective_context < 32:
            raise ValueError("TimesFM forecast_context must be at least 32")
        if self.max_horizon < 1:
            raise ValueError("TimesFM max_horizon must be positive")
        if self.batch_size < 1 or self.chunk_size < 1:
            raise ValueError("TimesFM batch and chunk sizes must be positive")
        if not 0.1 <= self.memory_target_fraction <= 0.95:
            raise ValueError("TimesFM memory target fraction must be between 0.1 and 0.95")

    @classmethod
    def from_environment(cls) -> "TimesFMRuntimeConfig":
        """Build runtime settings from optional CAPITALISMAN_TIMESFM_* variables."""
        return cls(
            model_id=os.getenv("CAPITALISMAN_TIMESFM_MODEL_ID", DEFAULT_MODEL_ID),
            device=os.getenv("CAPITALISMAN_TIMESFM_DEVICE", "auto"),
            max_context=int(os.getenv("CAPITALISMAN_TIMESFM_MAX_CONTEXT", "1024")),
            forecast_context=int(
                os.getenv("CAPITALISMAN_TIMESFM_FORECAST_CONTEXT", "1024")
            ),
            max_horizon=int(os.getenv("CAPITALISMAN_TIMESFM_MAX_HORIZON", "256")),
            batch_size=int(os.getenv("CAPITALISMAN_TIMESFM_BATCH_SIZE", "32")),
            chunk_size=int(os.getenv("CAPITALISMAN_TIMESFM_CHUNK_SIZE", "32")),
            memory_target_fraction=float(
                os.getenv("CAPITALISMAN_TIMESFM_MEMORY_TARGET", "0.72")
            ),
            profile_name=os.getenv("CAPITALISMAN_TIMESFM_PROFILE", "custom"),
            use_case=os.getenv("CAPITALISMAN_TIMESFM_USE_CASE", "interactive"),
        )

    @classmethod
    def from_profile(cls, profile: TimesFMProfile) -> "TimesFMRuntimeConfig":
        return cls(
            model_id=os.getenv("CAPITALISMAN_TIMESFM_MODEL_ID", DEFAULT_MODEL_ID),
            device=profile.device,
            max_context=profile.max_context,
            forecast_context=profile.forecast_context,
            max_horizon=profile.max_horizon,
            batch_size=profile.batch_size,
            chunk_size=profile.chunk_size,
            memory_target_fraction=profile.memory_target_fraction,
            profile_name=profile.name,
            use_case=profile.use_case,
        )


@dataclass
class TimesFMRuntimeStatus:
    state: str
    message: str
    requested_device: str
    resolved_device: str = "unavailable"
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    torch_version: str | None = None
    timesfm_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    gpu_free_memory_gb: float | None = None
    available_ram_gb: float | None = None
    free_disk_gb: float | None = None


@dataclass(frozen=True)
class TimesFMForecast:
    horizon: int
    point: np.ndarray
    quantiles: dict[float, np.ndarray]
    model_id: str
    device: str

    @property
    def terminal_point(self) -> float:
        return float(self.point[-1])

    def terminal_quantile(self, level: float) -> float:
        return float(self.quantiles[level][-1])


def probability_above(forecast: TimesFMForecast, threshold: float) -> float:
    """Approximate P(value > threshold) from forecast quantiles."""
    values = np.array(
        [forecast.terminal_quantile(level) for level in QUANTILE_LEVELS],
        dtype=float,
    )
    levels = np.array(QUANTILE_LEVELS, dtype=float)
    values = np.maximum.accumulate(values)
    cdf = float(np.interp(threshold, values, levels, left=0.0, right=1.0))
    return max(0.0, min(1.0, 1.0 - cdf))


class TimesFMRuntime:
    """Lazy TimesFM 2.5 runtime with explicit status and dependency checks."""

    def __init__(
        self,
        config: TimesFMRuntimeConfig | None = None,
        model_factory: Callable[[], object] | None = None,
    ) -> None:
        self.config = config or TimesFMRuntimeConfig.from_environment()
        self._model_factory = model_factory
        self._model = None
        self._status: TimesFMRuntimeStatus | None = None

    @property
    def status(self) -> TimesFMRuntimeStatus:
        return self._status or self.preflight()

    def preflight(self) -> TimesFMRuntimeStatus:
        requested = self.config.device.lower()
        if requested not in {"auto", "cuda", "cpu"}:
            self._status = TimesFMRuntimeStatus(
                "failed", f"Unsupported device '{requested}'", requested
            )
            return self._status

        if requested == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        status = TimesFMRuntimeStatus(
            state="checking",
            message="Checking TimesFM runtime dependencies",
            requested_device=requested,
        )
        status.available_ram_gb = self._available_ram_gb()
        status.free_disk_gb = shutil.disk_usage(Path.home()).free / 1024**3

        try:
            import torch
        except ImportError:
            status.state = "unavailable"
            status.message = "PyTorch is not installed in this environment"
            self._status = status
            return status

        status.torch_version = torch.__version__
        status.cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            status.gpu_name = torch.cuda.get_device_name(0)
            status.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                status.gpu_free_memory_gb = free_bytes / 1024**3
            except (AttributeError, RuntimeError, TypeError):
                pass

        if requested == "cuda" and not cuda_available:
            status.state = "unavailable"
            status.message = "CUDA was requested but is not available to PyTorch"
            self._status = status
            return status

        status.resolved_device = "cuda" if requested != "cpu" and cuda_available else "cpu"

        try:
            status.timesfm_version = metadata.version("timesfm")
            import timesfm
        except (ImportError, metadata.PackageNotFoundError):
            status.state = "unavailable"
            status.message = "TimesFM 2.5 is not installed in this environment"
            self._status = status
            return status

        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            status.state = "incompatible"
            status.message = "Installed TimesFM package does not provide the 2.5 PyTorch API"
            self._status = status
            return status

        if status.available_ram_gb is not None and status.available_ram_gb < 2:
            status.state = "unavailable"
            status.message = "Less than 2 GB of system RAM is available"
        elif status.free_disk_gb is not None and status.free_disk_gb < 2:
            status.state = "unavailable"
            status.message = "Less than 2 GB of free disk space is available"
        elif status.resolved_device == "cuda" and (status.gpu_memory_gb or 0) < 2:
            status.state = "unavailable"
            status.message = "TimesFM requires at least 2 GB of GPU memory"
        else:
            status.state = "ready"
            status.message = "Runtime dependencies are ready; model is not loaded"

        self._status = status
        return status

    def load(self) -> TimesFMRuntimeStatus:
        if self._model is not None:
            return self.status

        if self._model_factory is not None:
            status = TimesFMRuntimeStatus(
                "ready", "Injected model is ready", self.config.device, "test"
            )
            self._status = status
        else:
            status = self.preflight()
            if status.state != "ready":
                raise RuntimeError(status.message)

        try:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                import torch
                import timesfm

                torch.set_float32_matmul_precision("high")
                self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.config.model_id
                )
                self._model.compile(
                    timesfm.ForecastConfig(
                        max_context=self.config.max_context,
                        max_horizon=self.config.max_horizon,
                        normalize_inputs=True,
                        per_core_batch_size=self.config.batch_size,
                        use_continuous_quantile_head=True,
                        force_flip_invariance=True,
                        infer_is_positive=True,
                        fix_quantile_crossing=True,
                    )
                )
            self._status.state = "loaded"
            self._status.message = "TimesFM model loaded successfully"
        except Exception as error:
            self._model = None
            status.state = "failed"
            status.message = f"TimesFM model load failed: {error}"
            self._status = status
            raise RuntimeError(status.message) from error

        return self._status

    def forecast(self, inputs: list[np.ndarray], horizon: int) -> list[TimesFMForecast]:
        if horizon < 1 or horizon > self.config.max_horizon:
            raise ValueError(
                f"Horizon must be between 1 and {self.config.max_horizon}, got {horizon}"
            )
        if not inputs:
            return []

        prepared = [
            np.asarray(values[-self.config.effective_context :], dtype=np.float32)
            for values in inputs
        ]
        for index, values in enumerate(prepared):
            if values.ndim != 1 or len(values) < 32:
                raise ValueError(
                    f"Input {index} must be a one-dimensional series with at least 32 values"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Input {index} contains NaN or infinite values")
        self.load()
        input_count = len(prepared)
        point_parts = []
        quantile_parts = []
        for start in range(0, input_count, self.config.chunk_size):
            chunk = list(prepared[start : start + self.config.chunk_size])
            chunk_point, chunk_quantiles = self._model.forecast(
                horizon=horizon, inputs=chunk
            )
            point_parts.append(np.asarray(chunk_point, dtype=float))
            quantile_parts.append(np.asarray(chunk_quantiles, dtype=float))
        point = np.concatenate(point_parts, axis=0)
        raw_quantiles = np.concatenate(quantile_parts, axis=0)
        if point.shape != (input_count, horizon):
            raise RuntimeError(f"Unexpected point forecast shape: {point.shape}")
        if (
            raw_quantiles.ndim != 3
            or raw_quantiles.shape[:2] != (input_count, horizon)
            or raw_quantiles.shape[2] < 10
        ):
            raise RuntimeError(f"Unexpected quantile forecast shape: {raw_quantiles.shape}")
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(raw_quantiles)):
            raise RuntimeError("TimesFM returned NaN or infinite forecast values")
        if np.any(np.diff(raw_quantiles[:, :, 1:10], axis=2) < -1e-6):
            raise RuntimeError("TimesFM returned crossing forecast quantiles")

        forecasts = []
        for batch_idx in range(input_count):
            quantiles = {
                level: raw_quantiles[batch_idx, :, quantile_idx]
                for level, quantile_idx in zip(QUANTILE_LEVELS, range(1, 10))
            }
            forecasts.append(
                TimesFMForecast(
                    horizon=horizon,
                    point=point[batch_idx],
                    quantiles=quantiles,
                    model_id=self.config.model_id,
                    device=self.status.resolved_device,
                )
            )
        return forecasts

    @staticmethod
    def _available_ram_gb() -> float | None:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / 1024**3
        except (ValueError, OSError, AttributeError):
            return None


_runtime: TimesFMRuntime | None = None


def get_timesfm_runtime(config: TimesFMRuntimeConfig | None = None) -> TimesFMRuntime:
    global _runtime
    if _runtime is None or (config is not None and _runtime.config != config):
        _runtime = TimesFMRuntime(config)
    return _runtime


def reset_timesfm_runtime() -> None:
    global _runtime
    _runtime = None
