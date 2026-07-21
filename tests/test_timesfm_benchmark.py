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
        benchmark_close_series(pd.Series(np.arange(20)), runtime, horizon=5, min_context=32)
