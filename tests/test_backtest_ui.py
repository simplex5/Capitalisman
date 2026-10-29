"""Backtest exposes a complete research archive, including no-trade results."""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.conftest import make_ohlcv


def test_backtest_can_render_research_archive_download():
    frame = make_ohlcv(360, trend="volatile")
    app = AppTest.from_file("app.py").run(timeout=20)
    app.radio[0].set_value("Backtest").run(timeout=20)
    with patch("ui.page_backtest.fetch_ohlcv", return_value=frame), patch("ui.page_backtest.render_price_chart"):
        app.button(key="backtest_run").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    downloads = app.get("download_button")
    assert any(button.proto.label == "Download research archive" for button in downloads)


def test_timesfm_failure_shows_error_without_success_metrics_or_archive():
    frame = make_ohlcv(360, trend="volatile")
    app = AppTest.from_file("app.py").run(timeout=20)
    app.radio[0].set_value("Backtest").run(timeout=20)
    app.multiselect(key="backtest_indicators").set_value(["TimesFM Forecast", "RSI"]).run(timeout=20)
    with patch("ui.page_backtest.fetch_ohlcv", return_value=frame), patch(
        "ml.timesfm_runtime.TimesFMRuntime.forecast", side_effect=RuntimeError("TimesFM inference failed")
    ):
        app.button(key="backtest_run").click().run(timeout=20)
    assert not app.exception
    assert any("TimesFM inference failed" in error.value for error in app.error)
    assert not app.metric
    assert not app.get("download_button")
    assert not any("No trades generated" in warning.value for warning in app.warning)


def test_experimental_model_selection_updates_runtime_without_loading_weights():
    from ml.timesfm_runtime import get_timesfm_runtime
    app = AppTest.from_file("app.py").run(timeout=20)
    app.radio[0].set_value("Backtest").run(timeout=20)
    app.multiselect(key="backtest_indicators").set_value(["TimesFM Forecast"]).run(timeout=20)
    assert get_timesfm_runtime().config.backend == "2.5"
    app.selectbox(key="backtest_indicators_timesfm_backend").set_value("TimesFM 3.0 (experimental)").run(timeout=20)
    assert not app.exception
    assert get_timesfm_runtime().config.backend == "3.0"
    assert get_timesfm_runtime()._model is None
    app.selectbox(key="backtest_indicators_timesfm_backend").set_value("TimesFM 2.5").run(timeout=20)
    assert get_timesfm_runtime().config.backend == "2.5"
