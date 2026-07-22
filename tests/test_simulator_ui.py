"""Small in-process smoke test for Simulator navigation and first step."""

from datetime import date
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_simulator_can_create_and_step_a_replay():
    index = pd.bdate_range("2024-01-01", "2026-12-31")
    prices = pd.Series(range(100, 100 + len(index)), index=index, dtype=float)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": 1_000.0,
        },
        index=index,
    )
    app = AppTest.from_file("app.py").run(timeout=20)

    app.radio[0].set_value("Simulator").run(timeout=20)
    assert not app.exception
    assert app.header[0].value == "Simulator"
    app.date_input[0].set_value(date(2025, 7, 21)).run(timeout=20)

    with patch("ui.page_simulator.fetch_ohlcv", return_value=df):
        app.button[0].click().run(timeout=20)

    assert not app.exception
    assert app.button[0].label == "Step one bar"
    app.button[0].click().run(timeout=20)

    assert not app.exception
    assert app.metric[0].label == "Portfolio"
    assert any("**Status:** Paused" in block.value for block in app.markdown)
