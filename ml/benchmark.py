"""Point-in-time evaluation utilities for probabilistic TimesFM forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable
from numbers import Integral

import numpy as np
import pandas as pd

from ml.timesfm_runtime import TimesFMRuntime, probability_above
from research.uncertainty import paired_mae_interval


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
    baseline_prices: dict[str, float] = field(default_factory=dict)


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
    baseline_metrics: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _pinball_loss(actual: np.ndarray, predicted: np.ndarray, level: float) -> float:
    error = actual - predicted
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def baseline_prices(history: pd.Series, horizon: int) -> dict[str, float]:
    """Fixed, causal baselines; windows are specified before test evaluation."""
    recent = history.iloc[-60:]
    last = float(recent.iloc[-1])
    drift = (last - float(recent.iloc[0])) / (len(recent) - 1)
    return {
        "last_price": last,
        "drift_60": max(0.0, last + horizon * drift),
        "moving_average_20": float(history.iloc[-20:].mean()),
        "exponential_smoothing_20": float(history.ewm(span=20, adjust=False).mean().iloc[-1]),
    }


def benchmark_close_series(
    close: pd.Series,
    runtime: TimesFMRuntime,
    horizon: int = 10,
    min_context: int = 64,
    step: int | None = None,
    batch_size: int = 16,
    evaluation_start: pd.Timestamp | None = None,
) -> BenchmarkResult:
    """Evaluate TimesFM at strictly historical origins with no future leakage."""
    clean = close.astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex) or clean.index.hasnans or not clean.index.is_unique or not clean.index.is_monotonic_increasing:
        raise ValueError("Prices require valid, sorted, unique datetime indices")
    if not np.isfinite(clean).all() or (clean <= 0).any():
        raise ValueError("Prices must be finite and positive; missing bars must be resolved explicitly")
    for name, value in (("horizon", horizon), ("min_context", min_context), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
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

    origin_step = horizon if step is None else step
    if isinstance(origin_step, bool) or not isinstance(origin_step, Integral) or origin_step < 1:
        raise ValueError("step must be positive")
    start_position = min_context - 1
    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start)
        if pd.isna(start) or (start.tzinfo is None) != (clean.index.tz is None):
            raise ValueError("Evaluation start must be valid and match the data's timezone awareness")
        start_position = int(clean.index.searchsorted(start))
        if start_position < min_context - 1:
            raise ValueError("Test start must leave the requested context before the first origin")
    origins = list(range(start_position, len(clean) - horizon, origin_step))
    if not origins:
        raise ValueError("No complete forecast targets in the test period")
    evaluations: list[ForecastEvaluation] = []

    for origin_batch in _chunks(origins, batch_size):
        inputs = [
            clean.iloc[max(0, origin + 1 - runtime.config.effective_context):origin + 1].to_numpy(dtype=np.float32)
            for origin in origin_batch
        ]
        forecasts = runtime.forecast(inputs, horizon=horizon)
        if len(forecasts) != len(origin_batch):
            raise ValueError("Model returned a different number of forecasts than origins")
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
                    baseline_prices=baseline_prices(clean.iloc[:origin + 1], horizon),
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
    model_mae = float(np.mean(np.abs(actual_return - median_return)))
    comparisons = {}
    for name in evaluations[0].baseline_prices:
        predicted = np.array([item.baseline_prices[name] for item in evaluations]) / current - 1
        mae = float(np.mean(np.abs(actual_return - predicted)))
        comparisons[name] = {
            "return_mae": mae,
            "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual_return))),
            "model_mae_improvement": mae - model_mae,
            "mae_improvement_interval": paired_mae_interval(
                np.abs(actual_return - median_return), np.abs(actual_return - predicted),
                horizon=horizon, step=origin_step,
            ),
        }

    return BenchmarkResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizon=horizon,
        observations=len(evaluations),
        model_id=runtime.config.model_id,
        device=status.resolved_device,
        median_return_mae=model_mae,
        naive_return_mae=float(np.mean(np.abs(actual_return))),
        directional_accuracy=float(np.mean(np.sign(median_return) == np.sign(actual_return))),
        interval_80_coverage=float(
            np.mean((actual_return >= lower_return) & (actual_return <= upper_return))
        ),
        interval_80_mean_width=float(np.mean(upper_return - lower_return)),
        probability_up_brier=float(np.mean((probability_up - actual_up) ** 2)),
        pinball_q10=_pinball_loss(actual_return, lower_return, 0.1),
        pinball_q50=_pinball_loss(actual_return, median_return, 0.5),
        pinball_q90=_pinball_loss(actual_return, upper_return, 0.9),
        evaluations=tuple(evaluations),
        baseline_metrics=comparisons,
        configuration={
            "horizon": horizon, "min_context": min_context, "step": origin_step,
            "batch_size": batch_size, "evaluation_start": str(clean.index[start_position]),
            "evaluation_end": str(clean.index[-1]), "runtime": asdict(runtime.config),
            "observed_model_device": status.model_device,
            "overlapping_targets": origin_step < horizon,
            "protocol": "Fixed baselines; no fitting or selection on test outcomes; exploratory paired block intervals, no multiplicity adjustment",
        },
    )
