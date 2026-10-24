"""Explicit history controls, request invalidation and actual input coverage."""

import numpy as np
from streamlit.testing.v1 import AppTest

from ml.timesfm_profiles import HardwareCapabilities
from ml.timesfm_runtime import TimesFMForecast, TimesFMRuntime, TimesFMRuntimeStatus, get_timesfm_runtime
from tests.conftest import make_ohlcv


def test_extended_context_reaches_forecast_and_invalidates_old_result(monkeypatch):
    import ui.components as components
    import ui.page_predict as predict
    import ml.timesfm_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_runtime", None)
    monkeypatch.setenv("CAPITALISMAN_TIMESFM_DEVICE", "cpu")
    original_spec = components.importlib.util.find_spec
    monkeypatch.setattr(components.importlib.util, "find_spec",
                        lambda name, *a, **kw: object() if name in ("timesfm", "timesfm3")
                        else original_spec(name, *a, **kw))
    monkeypatch.setattr(components, "detect_hardware", lambda: HardwareCapabilities(
        "cpu", available_ram_gb=16))
    monkeypatch.setattr(TimesFMRuntime, "preflight", lambda self: TimesFMRuntimeStatus(
        "ready", "offline fixture", "cpu", resolved_device="cpu", backend=self.config.backend))
    # More than 8192 input rows proves the cap is applied before inference.
    frame = make_ohlcv(8300)
    monkeypatch.setattr(predict, "fetch_with_warmup", lambda *a, **kw: (frame, frame))
    monkeypatch.setattr(predict, "render_price_chart", lambda *a, **kw: None)
    calls = []

    def forecast(self, inputs, horizon, **kwargs):
        calls.append((len(inputs[0]), horizon))
        point = np.full(horizon, inputs[0][-1])
        return [TimesFMForecast(horizon, point,
                {i / 10: point * (.9 + i / 50) for i in range(1, 10)},
                self.config.model_id, "cpu")]

    monkeypatch.setattr(TimesFMRuntime, "forecast", forecast)
    app = AppTest.from_file("app.py").run(timeout=20)
    app.multiselect(key="predict_indicators").set_value(["TimesFM Forecast"]).run()
    app.selectbox(key="predict_indicators_timesfm_backend").set_value("TimesFM 3.0 (experimental)").run()
    profile_config = get_timesfm_runtime().config
    app.slider(key="predict_horizon").set_value(5).run()
    app.selectbox(key="predict_indicators_timesfm_context_v3").set_value(8192).run()
    config = get_timesfm_runtime().config
    assert config.max_context == config.effective_context == 8192
    assert config.device == profile_config.device == "cpu"
    assert config.batch_size == profile_config.batch_size
    assert config.chunk_size == profile_config.chunk_size
    assert config.memory_target_fraction == profile_config.memory_target_fraction
    assert not config.torch_compile
    app.button(key="predict_analyze").click().run(timeout=20)
    assert not app.exception
    assert calls[0] == (8192, 5)
    assert all(length == 8192 for length, _ in calls)
    previous_calls = list(calls)
    assert any("History supplied to TimesFM: 8,192 daily bars" in c.value for c in app.caption)
    assert any("google/timesfm-3.0-pytorch" in m.value for m in app.markdown)
    app.selectbox(key="predict_indicators_timesfm_context_v3").set_value(4096).run()
    assert calls == previous_calls  # Must explicitly run again after configuration changes.
    assert not any("History supplied to TimesFM" in c.value for c in app.caption)
    # Short histories are reported honestly rather than implying padding is real data.
    frame = frame.iloc[-600:]
    app.button(key="predict_analyze").click().run(timeout=20)
    assert calls[len(previous_calls)] == (600, 5)
    assert any("History supplied to TimesFM: 600 daily bars" in c.value for c in app.caption)
    app.selectbox(key="predict_indicators_timesfm_context_v3").set_value(0).run()
    assert get_timesfm_runtime().config == profile_config
    app.selectbox(key="predict_indicators_timesfm_backend").set_value("TimesFM 2.5").run()
    assert not app.exception
    assert not any(w.label == "TimesFM Historical Context" for w in app.selectbox)
    assert get_timesfm_runtime().config.effective_context == 1024
