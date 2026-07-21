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
        runtime.forecast([np.arange(40)], horizon=5)


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
    results = runtime.forecast([np.arange(40)], horizon=3)
    assert len(results) == 1


def test_forecast_uses_selected_context_length():
    seen_lengths = []

    class RecordingModel(FakeTimesFMModel):
        def forecast(self, horizon, inputs):
            seen_lengths.extend(len(values) for values in inputs)
            return super().forecast(horizon, inputs)

    runtime = TimesFMRuntime(
        TimesFMRuntimeConfig(
            max_context=256, forecast_context=64, chunk_size=2
        ),
        model_factory=RecordingModel,
    )
    runtime.forecast([np.arange(200)] * 3, horizon=3)
    assert seen_lengths == [64, 64, 64]


def test_forecast_rejects_non_finite_inputs():
    values = np.arange(40, dtype=float)
    values[-1] = np.nan
    runtime = TimesFMRuntime(model_factory=FakeTimesFMModel)
    with pytest.raises(ValueError, match="NaN or infinite"):
        runtime.forecast([values], horizon=3)


def test_forecast_chunks_large_calls():
    calls = []

    class RecordingModel(FakeTimesFMModel):
        def forecast(self, horizon, inputs):
            calls.append(len(inputs))
            return super().forecast(horizon, inputs)

    runtime = TimesFMRuntime(
        TimesFMRuntimeConfig(chunk_size=2), model_factory=RecordingModel
    )
    results = runtime.forecast([np.arange(40)] * 5, horizon=3)
    assert len(results) == 5
    assert calls == [2, 2, 1]
