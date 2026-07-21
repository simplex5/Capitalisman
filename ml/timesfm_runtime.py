"""TimesFM 2.5 runtime, preflight checks, and probabilistic forecast types."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
import os
import shutil
import sys
import threading
from typing import Callable

import numpy as np


DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
QUANTILE_LEVELS = tuple(round(level / 10, 1) for level in range(1, 10))


@dataclass(frozen=True)
class TimesFMRuntimeConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    max_context: int = 1024
    max_horizon: int = 256
    batch_size: int = 32

    @classmethod
    def from_environment(cls) -> "TimesFMRuntimeConfig":
        """Build runtime settings from optional CAPITALISMAN_TIMESFM_* variables."""
        return cls(
            model_id=os.getenv("CAPITALISMAN_TIMESFM_MODEL_ID", DEFAULT_MODEL_ID),
            device=os.getenv("CAPITALISMAN_TIMESFM_DEVICE", "auto"),
            max_context=int(os.getenv("CAPITALISMAN_TIMESFM_MAX_CONTEXT", "1024")),
            max_horizon=int(os.getenv("CAPITALISMAN_TIMESFM_MAX_HORIZON", "256")),
            batch_size=int(os.getenv("CAPITALISMAN_TIMESFM_BATCH_SIZE", "32")),
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
        # Streamlit serves sessions from multiple threads; without a lock two
        # sessions can double-load the multi-GB model or race on _status.
        self._load_lock = threading.Lock()

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
        with self._load_lock:
            if self._model is not None:  # lost the race — another thread loaded
                return self.status
            return self._load_locked()

    def _load_locked(self) -> TimesFMRuntimeStatus:
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
                # Place the model on the preflight-resolved device explicitly
                # rather than trusting library defaults; the wrapper may manage
                # its own placement, so failures are tolerated and the real
                # device is read back below for truthful reporting.
                target_device = status.resolved_device
                mover = getattr(self._model, "to", None)
                if callable(mover) and target_device:
                    try:
                        mover(target_device)
                    except Exception:
                        pass
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
            actual_device = self._model_device()
            if actual_device is not None:
                self._status.resolved_device = actual_device
        except Exception as error:
            self._model = None
            status.state = "failed"
            status.message = f"TimesFM model load failed: {error}"
            self._status = status
            raise RuntimeError(status.message) from error

        return self._status

    def _model_device(self) -> str | None:
        """Read the device the loaded model actually lives on."""
        params = getattr(self._model, "parameters", None)
        if not callable(params):
            return None
        try:
            first = next(iter(params()), None)
            return first.device.type if first is not None else None
        except Exception:
            return None

    def unload(self) -> None:
        """Release the model reference and any GPU memory it holds."""
        with self._load_lock:
            if self._model is None:
                return
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if self._status is not None and self._status.state == "loaded":
                self._status.state = "ready"
                self._status.message = "Model unloaded; dependencies remain ready"

    def forecast(self, inputs: list[np.ndarray], horizon: int) -> list[TimesFMForecast]:
        if horizon < 1 or horizon > self.config.max_horizon:
            raise ValueError(
                f"Horizon must be between 1 and {self.config.max_horizon}, got {horizon}"
            )
        if not inputs:
            return []

        self.load()
        prepared = [
            np.asarray(values[-self.config.max_context :], dtype=np.float32)
            for values in inputs
        ]
        input_count = len(prepared)
        point, raw_quantiles = self._model.forecast(
            horizon=horizon, inputs=list(prepared)
        )
        point = np.asarray(point, dtype=float)
        raw_quantiles = np.asarray(raw_quantiles, dtype=float)
        if point.shape != (input_count, horizon):
            raise RuntimeError(f"Unexpected point forecast shape: {point.shape}")
        if raw_quantiles.shape[:2] != (input_count, horizon) or raw_quantiles.shape[2] < 10:
            raise RuntimeError(f"Unexpected quantile forecast shape: {raw_quantiles.shape}")

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
_runtime_lock = threading.Lock()


def get_timesfm_runtime(config: TimesFMRuntimeConfig | None = None) -> TimesFMRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None or (config is not None and _runtime.config != config):
            if _runtime is not None:
                _runtime.unload()  # don't orphan a loaded model in VRAM
            _runtime = TimesFMRuntime(config)
        return _runtime


def reset_timesfm_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.unload()
        _runtime = None
