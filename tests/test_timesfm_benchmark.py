"""Tests for point-in-time TimesFM benchmark metrics."""

import numpy as np
import pandas as pd
import pytest

from ml.benchmark import benchmark_close_series
from ml.timesfm_runtime import TimesFMRuntime


class TrendModel:
    def forecast(self, horizon, inputs):
        point = np.stack(
            [np.linspace(values[-1], values[-1] * 1.02, horizon) for values in inputs]
        )
        quantiles = np.zeros((len(inputs), horizon, 10))
        quantiles[:, :, 0] = point
        for quantile_idx in range(1, 10):
            quantiles[:, :, quantile_idx] = point * (
                0.94 + quantile_idx * 0.012
            )
        return point, quantiles


def test_benchmark_computes_probabilistic_metrics():
    close = pd.Series(
        np.linspace(100, 140, 90),
        index=pd.date_range("2025-01-01", periods=90, freq="D"),
    )
    runtime = TimesFMRuntime(model_factory=TrendModel)
    result = benchmark_close_series(
        close, runtime, horizon=5, min_context=32, step=10, batch_size=2
    )

    assert result.observations == 6
    assert result.directional_accuracy == 1.0
    assert 0.0 <= result.interval_80_coverage <= 1.0
    assert 0.0 <= result.probability_up_brier <= 1.0
    assert result.pinball_q10 >= 0.0
    assert result.evaluations[0].origin == str(close.index[31])
    assert result.evaluations[0].target == str(close.index[36])


def test_benchmark_rejects_insufficient_history():
    runtime = TimesFMRuntime(model_factory=TrendModel)
    with pytest.raises(ValueError, match="Need at least"):
        benchmark_close_series(pd.Series(np.arange(20) + 1, index=pd.date_range("2025-01-01", periods=20)), runtime, horizon=5, min_context=32)


def test_holdout_baselines_share_origins_and_ignore_future_values():
    close = pd.Series(np.arange(100) + 100., index=pd.date_range("2025-01-01", periods=100))
    runtime = TimesFMRuntime(model_factory=TrendModel)
    first = benchmark_close_series(close, runtime, horizon=5, min_context=32, evaluation_start=close.index[60])
    changed = close.copy()
    changed.iloc[61:] *= 2
    second = benchmark_close_series(changed, runtime, horizon=5, min_context=32, evaluation_start=close.index[60])
    assert first.evaluations[0].origin == str(close.index[60])
    assert first.evaluations[0].target == str(close.index[65])
    assert first.evaluations[0].baseline_prices == second.evaluations[0].baseline_prices
    assert first.evaluations[0].median_price == second.evaluations[0].median_price
    assert first.baseline_metrics["drift_60"]["return_mae"] == pytest.approx(0)
    assert set(first.baseline_metrics) == {"last_price", "drift_60", "moving_average_20", "exponential_smoothing_20"}
    assert first.configuration["overlapping_targets"] is False


@pytest.mark.parametrize("problem", ["nan", "zero", "duplicate", "unsorted", "zero_step"])
def test_benchmark_rejects_invalid_inputs_instead_of_changing_observation_spacing(problem):
    close = pd.Series(np.arange(100) + 100., index=pd.date_range("2025-01-01", periods=100))
    if problem == "nan":
        close.iloc[50] = np.nan
    if problem == "zero":
        close.iloc[50] = 0
    if problem == "duplicate":
        close.index = [close.index[0]] * len(close)
    if problem == "unsorted":
        close = close.iloc[::-1]
    with pytest.raises(ValueError):
        benchmark_close_series(close, TimesFMRuntime(model_factory=TrendModel),
                               min_context=32, step=0 if problem == "zero_step" else None)
