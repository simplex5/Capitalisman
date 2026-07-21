"""Tests for the TimesFM runtime adapter without loading model weights."""

import numpy as np
import pytest

from ml.timesfm_runtime import (
    TimesFMRuntime,
    TimesFMRuntimeConfig,
    probability_above,
)


class FakeTimesFMModel:
    def forecast(self, horizon, inputs):
        point = np.stack([
            np.linspace(values[-1], values[-1] * 1.05, horizon) for values in inputs
        ])
        quantiles = np.zeros((len(inputs), horizon, 10))
        quantiles[:, :, 0] = point
        for quantile_idx in range(1, 10):
            quantiles[:, :, quantile_idx] = point * (
                0.90 + quantile_idx * 0.02
            )
        return point, quantiles


def test_forecast_parses_point_and_quantiles():
    runtime = TimesFMRuntime(model_factory=FakeTimesFMModel)
    result = runtime.forecast([np.arange(1, 101, dtype=float)], horizon=5)[0]
    assert result.point.shape == (5,)
    assert result.quantiles[0.1].shape == (5,)
    assert result.terminal_quantile(0.1) < result.terminal_quantile(0.9)
    assert result.horizon == 5


def test_probability_above_uses_terminal_distribution():
    runtime = TimesFMRuntime(model_factory=FakeTimesFMModel)
    result = runtime.forecast([np.arange(1, 101, dtype=float)], horizon=5)[0]
    low_threshold = result.terminal_quantile(0.2)
    high_threshold = result.terminal_quantile(0.8)
    assert probability_above(result, low_threshold) > probability_above(
        result, high_threshold
    )


def test_rejects_horizon_above_compiled_maximum():
    runtime = TimesFMRuntime(
        TimesFMRuntimeConfig(max_horizon=4), model_factory=FakeTimesFMModel
    )
    with pytest.raises(ValueError, match="Horizon must be"):
        runtime.forecast([np.arange(20)], horizon=5)


def test_rejects_unexpected_model_shapes():
    class BadModel:
        def forecast(self, horizon, inputs):
            return np.zeros((1, 1)), np.zeros((1, 1, 1))

    runtime = TimesFMRuntime(model_factory=BadModel)
    with pytest.raises(RuntimeError, match="Unexpected point forecast shape"):
        runtime.forecast([np.arange(20)], horizon=5)


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("CAPITALISMAN_TIMESFM_DEVICE", "cuda")
    monkeypatch.setenv("CAPITALISMAN_TIMESFM_BATCH_SIZE", "8")
    config = TimesFMRuntimeConfig.from_environment()
    assert config.device == "cuda"
    assert config.batch_size == 8


def test_model_cannot_mutate_adapter_batch_size():
    class PaddingModel(FakeTimesFMModel):
        def forecast(self, horizon, inputs):
            original = super().forecast(horizon, inputs)
            inputs.extend([np.zeros(3)] * 4)
            return original

    runtime = TimesFMRuntime(model_factory=PaddingModel)
    results = runtime.forecast([np.arange(20)], horizon=3)
    assert len(results) == 1


def test_timesfm_indicator_is_excluded_from_backtests():
    """Pretraining overlap + sparse-origin columns make historical TimesFM
    values non-point-in-time; the engine must exclude it (CMAN-D1-002)."""
    from indicators.forecast import TimesFMForecast

    indicator = TimesFMForecast(runtime=TimesFMRuntime(model_factory=FakeTimesFMModel))
    assert indicator.backtest_safe is False
