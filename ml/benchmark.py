"""Point-in-time evaluation utilities for probabilistic TimesFM forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from ml.timesfm_runtime import TimesFMRuntime, probability_above


@dataclass(frozen=True)
class ForecastEvaluation:
    origin: str
    target: str
    current_price: float
    actual_price: float
    median_price: float
    lower_price: float
    upper_price: float
    probability_up: float


@dataclass(frozen=True)
class BenchmarkResult:
    generated_at: str
    horizon: int
    observations: int
    model_id: str
    device: str
    median_return_mae: float
    naive_return_mae: float
    directional_accuracy: float
    interval_80_coverage: float
    interval_80_mean_width: float
    probability_up_brier: float
    pinball_q10: float
    pinball_q50: float
    pinball_q90: float
    evaluations: tuple[ForecastEvaluation, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _pinball_loss(actual: np.ndarray, predicted: np.ndarray, level: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def benchmark_close_series(
    close: pd.Series,
    runtime: TimesFMRuntime,
    horizon: int = 10,
    min_context: int = 64,
    step: int | None = None,
    batch_size: int = 16,
) -> BenchmarkResult:
    """Evaluate TimesFM at strictly historical origins with no future leakage."""
    clean = close.dropna().astype(float)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if min_context < 32:
        raise ValueError("min_context must be at least 32 for TimesFM")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(clean) < min_context + horizon:
        raise ValueError(
            f"Need at least {min_context + horizon} prices, got {len(clean)}"
        )

    origin_step = step or horizon
    if origin_step < 1:
        raise ValueError("step must be positive")
    origins = list(range(min_context - 1, len(clean) - horizon, origin_step))
    evaluations: list[ForecastEvaluation] = []

    for origin_batch in _chunks(origins, batch_size):
        inputs = [
            clean.iloc[: origin + 1].to_numpy(dtype=np.float32)
            for origin in origin_batch
        ]
        forecasts = runtime.forecast(inputs, horizon=horizon)
        for origin, forecast in zip(origin_batch, forecasts):
            current = float(clean.iloc[origin])
            actual = float(clean.iloc[origin + horizon])
            evaluations.append(
                ForecastEvaluation(
                    origin=str(clean.index[origin]),
                    target=str(clean.index[origin + horizon]),
                    current_price=current,
                    actual_price=actual,
                    median_price=forecast.terminal_quantile(0.5),
                    lower_price=forecast.terminal_quantile(0.1),
                    upper_price=forecast.terminal_quantile(0.9),
                    probability_up=probability_above(forecast, current),
                )
            )

    current = np.array([item.current_price for item in evaluations])
    actual_return = np.array([item.actual_price for item in evaluations]) / current - 1.0
    median_return = np.array([item.median_price for item in evaluations]) / current - 1.0
    lower_return = np.array([item.lower_price for item in evaluations]) / current - 1.0
    upper_return = np.array([item.upper_price for item in evaluations]) / current - 1.0
    probability_up = np.array([item.probability_up for item in evaluations])
    actual_up = (actual_return > 0).astype(float)
    status = runtime.status

    return BenchmarkResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        observations=len(evaluations),
        model_id=runtime.config.model_id,
        device=status.resolved_device,
        median_return_mae=float(np.mean(np.abs(actual_return - median_return))),
        naive_return_mae=float(np.mean(np.abs(actual_return))),
        directional_accuracy=float(np.mean((median_return > 0) == (actual_return > 0))),
        interval_80_coverage=float(
            np.mean((actual_return >= lower_return) & (actual_return <= upper_return))
        ),
        interval_80_mean_width=float(np.mean(upper_return - lower_return)),
        probability_up_brier=float(np.mean((probability_up - actual_up) ** 2)),
        pinball_q10=_pinball_loss(actual_return, lower_return, 0.1),
        pinball_q50=_pinball_loss(actual_return, median_return, 0.5),
        pinball_q90=_pinball_loss(actual_return, upper_return, 0.9),
        evaluations=tuple(evaluations),
    )
