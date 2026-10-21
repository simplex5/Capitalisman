"""Forecast targets, cost policy and bounded probability interpretation."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import weakref

from backtesting.engine import run_backtest
from indicators.forecast import TimesFMForecast
from ml.timesfm_runtime import (
    TimesFMRuntime, TimesFMRuntimeConfig, TimesFMForecast as Forecast, probability_above,
)
from signals.base import SignalDirection
from tests.test_backtest import AlwaysBuyIndicator
from tests.test_backtest_accounting import bars


class FixedQuantileModel:
    def __init__(self, terminal_quantiles):
        self.values = np.asarray(terminal_quantiles)
        self.calls = 0

    def forecast(self, horizon, inputs):
        self.calls += 1
        point = np.full((len(inputs), horizon), self.values[4])
        output = np.empty((len(inputs), horizon, 10))
        output[:, :, 0] = point
        output[:, :, 1:] = self.values
        return point, output


def indicator(values):
    model = FixedQuantileModel(values)
    return TimesFMForecast(TimesFMRuntime(model_factory=lambda: model)), model


def test_tail_estimates_never_extrapolate_to_certainty():
    forecast = Forecast(1, np.array([100.]),
                        {i / 10: np.array([90. + 2.5 * (i - 1)]) for i in range(1, 10)}, "test", "test")
    assert probability_above(forecast, 89) == pytest.approx(0.9)
    assert probability_above(forecast, 111) == pytest.approx(0.1)
    assert probability_above(forecast, 100) == pytest.approx(0.5)


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_backtest_uses_historical_origins_and_matching_target_even_with_interactive_runtime(horizon):
    forecast, model = indicator(np.linspace(101, 109, 9))
    df = bars(120)
    with patch.object(forecast.runtime, "forecast", wraps=forecast.runtime.forecast) as inference:
        report = run_backtest(df, {forecast.name: forecast}, "TEST", "1y", horizon_days=horizon)
    origin_count = len(range(31, len(df), horizon))
    chunk_size = forecast.runtime.config.chunk_size
    assert inference.call_count == (origin_count + chunk_size - 1) // chunk_size
    assert report.trades
    for trade in report.trades:
        origin = df.index.get_loc(trade.signal_date)
        assert (origin - 31) % horizon == 0
        assert trade.exit_date == trade.target_date == df.index[origin + horizon]
    assert forecast.runtime.config.use_case == "interactive"


@pytest.mark.parametrize("values", [np.linspace(99, 103, 9), np.linspace(97, 101, 9)])
def test_configured_cost_changes_trade_eligibility_on_both_sides(values):
    forecast, _ = indicator(values)
    df = bars(120)
    low = run_backtest(df, {forecast.name: forecast}, "TEST", "1y", cost_per_trade_pct=0)
    high = run_backtest(df, {forecast.name: forecast}, "TEST", "1y", cost_per_trade_pct=5)
    assert low.trades
    assert not high.trades


def test_short_requires_probability_of_clearing_costs_not_just_any_decline():
    forecast, _ = indicator([99.7, 99.75, 99.8, 99.85, 99.89, 99.99, 100.01, 100.02, 100.03])
    computed = forecast.compute_for_backtest(bars(122), 5, 0.1)
    assert forecast.latest_analysis.probability_down > 0.6
    assert forecast.latest_analysis.probability_short_profit < 0.6
    assert forecast.get_signal_for_horizon(computed, 5).direction == SignalDirection.HOLD


def test_profit_hurdles_match_two_fill_accounting():
    forecast, _ = indicator(np.linspace(95, 105, 9))
    forecast.compute_for_backtest(bars(122), 5, 2)
    analysis = forecast.latest_analysis
    assert analysis.cost_per_trade_pct == 2
    assert analysis.probability_profit == probability_above(analysis.forecast, 100 * 1.01 / 0.99)
    assert analysis.probability_short_profit == 1 - probability_above(analysis.forecast, 100 * 0.99 / 1.01)


@pytest.mark.parametrize("length", [117, 120, 122])
def test_historical_forecast_availability_does_not_depend_on_future_bars(length):
    forecast, _ = indicator(np.linspace(101, 109, 9))
    df = bars(125)
    full = forecast.compute_for_backtest(df, 5, 0.1)
    prefix = forecast.compute_for_backtest(df.iloc[:length], 5, 0.1)
    columns = [name for name in full if name.startswith("TFM_")]
    pd.testing.assert_frame_equal(full[columns].iloc[:length], prefix[columns])


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
@pytest.mark.parametrize("mixed_strategy", [False, True])
def test_failed_historical_forecast_aborts_backtest_without_retry(error_type, mixed_strategy):
    forecast, model = indicator(np.linspace(101, 109, 9))
    def fail(horizon, inputs):
        model.calls += 1
        raise error_type("Unavailable model")
    model.forecast = fail
    selected = {forecast.name: forecast}
    if mixed_strategy:
        healthy = AlwaysBuyIndicator()
        selected[healthy.name] = healthy
    with pytest.raises(error_type, match="Unavailable model"):
        run_backtest(bars(120), selected, "TEST", "1y")
    assert model.calls == 1


def test_historical_model_load_failure_aborts_backtest():
    def fail_load():
        raise RuntimeError("Weights unavailable")
    forecast = TimesFMForecast(TimesFMRuntime(model_factory=fail_load))
    with pytest.raises(RuntimeError, match="Weights unavailable"):
        run_backtest(bars(120), {forecast.name: forecast}, "TEST", "1y")


def test_interactive_failure_preserves_error_detail_without_retry():
    forecast, _ = indicator(np.linspace(101, 109, 9))
    with patch.object(forecast.runtime, "forecast", side_effect=RuntimeError("Unavailable model")) as inference:
        computed = forecast.compute_for_horizon(bars(120), 5)
        signal = forecast.get_signal_for_horizon(computed, 5)
    assert inference.call_count == 1
    assert forecast.last_error == signal.detail == "Unavailable model"
    assert signal.direction == SignalDirection.HOLD
    assert signal.confidence == 0


@pytest.mark.parametrize("max_context,forecast_context", [(128, 64), (64, 128), (64, None)])
def test_historical_contexts_are_capped_and_prepared_in_bounded_batches(
    max_context, forecast_context, monkeypatch,
):
    config = TimesFMRuntimeConfig(
        max_context=max_context, forecast_context=forecast_context, chunk_size=32,
    )
    model = FixedQuantileModel(np.linspace(101, 109, 9))
    runtime = TimesFMRuntime(config, model_factory=lambda: model)
    forecast = TimesFMForecast(runtime)
    df = bars(2000)
    df["Close"] = np.linspace(100, 300, len(df))
    infer = runtime.forecast
    next_origin = 31
    batch_bytes = []
    previous_inputs = []

    def inspect_inputs(inputs, horizon):
        nonlocal next_origin
        assert all(reference() is None for reference in previous_inputs)
        previous_inputs[:] = [weakref.ref(values) for values in inputs]
        assert 0 < len(inputs) <= config.chunk_size
        batch_bytes.append(sum(values.nbytes for values in inputs))
        for values in inputs:
            assert values.dtype == np.float32
            assert len(values) <= config.effective_context
            expected = df["Close"].iloc[max(0, next_origin - 63):next_origin + 1]
            np.testing.assert_array_equal(values, expected.to_numpy(dtype=np.float32))
            next_origin += 1
        return infer(inputs, horizon)

    monkeypatch.setattr(runtime, "forecast", inspect_inputs)
    computed = forecast.compute_for_backtest(df, 1, 0.1)
    assert next_origin == len(df)
    assert max(batch_bytes) == 32 * 64 * 4
    assert sum(batch_bytes) == 501952
    assert computed["TFM_point"].iloc[:31].isna().all()
    assert computed["TFM_point"].iloc[31:].eq(105).all()
    assert forecast.latest_analysis.origin == df.index[-1]


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_bounded_preparation_matches_full_prefix_inference(horizon):
    class ContextModel:
        def forecast(self, horizon, inputs):
            means = np.array([values.mean() for values in inputs])
            point = np.repeat(means[:, None], horizon, axis=1)
            quantiles = point[:, :, None] * np.linspace(0.9, 1.1, 9)
            return point, np.concatenate([point[:, :, None], quantiles], axis=2)

    runtime = TimesFMRuntime(
        TimesFMRuntimeConfig(max_context=64, chunk_size=3), model_factory=ContextModel,
    )
    forecast = TimesFMForecast(runtime)
    df = bars(147)
    df["Close"] = 100 + np.arange(len(df)) ** 1.5
    origins = list(range(31, len(df), horizon))
    reference = runtime.forecast(
        [df["Close"].iloc[:origin + 1].to_numpy(dtype=np.float32) for origin in origins],
        horizon,
    )
    computed = forecast.compute_for_backtest(df, horizon, 0.1)
    np.testing.assert_array_equal(
        computed["TFM_point"].iloc[origins], [item.point[-1] for item in reference],
    )
    for quantile in (0.1, 0.5, 0.9):
        np.testing.assert_array_equal(
            computed[f"TFM_q{int(100 * quantile)}"].iloc[origins],
            [item.quantiles[quantile][-1] for item in reference],
        )
    prefix = forecast.compute_for_backtest(df.iloc[:132], horizon, 0.1)
    columns = [name for name in computed if name.startswith("TFM_")]
    pd.testing.assert_frame_equal(computed[columns].iloc[:132], prefix[columns])


@pytest.mark.parametrize("historical", [True, False])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_later_batch_failure_does_not_return_partial_forecasts(historical, error_type):
    model = FixedQuantileModel(np.linspace(101, 109, 9))
    runtime = TimesFMRuntime(
        TimesFMRuntimeConfig(chunk_size=2, use_case="backtest"),
        model_factory=lambda: model,
    )
    forecast = TimesFMForecast(runtime)
    infer = runtime.forecast
    calls = 0

    def fail_second_batch(inputs, horizon):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("Later batch failed")
        return infer(inputs, horizon)

    with patch.object(runtime, "forecast", side_effect=fail_second_batch):
        if historical:
            with pytest.raises(error_type, match="Later batch failed"):
                forecast.compute_for_backtest(bars(120), 5, 0.1)
        else:
            computed = forecast.compute_for_horizon(bars(120), 5)
            columns = [c for c in computed if c.startswith("TFM_") and c != "TFM_cost_pct"]
            assert computed[columns].isna().all().all()
            assert computed["TFM_cost_pct"].eq(0.1).all()
            signal = forecast.get_signal_for_horizon(computed, 5)
            assert signal.direction == SignalDirection.HOLD
            assert signal.detail == "Later batch failed"
    assert calls == 2
    assert forecast.last_error == "Later batch failed"
    assert forecast.latest_analysis is None
