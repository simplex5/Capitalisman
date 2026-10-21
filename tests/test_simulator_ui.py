"""Small in-process smoke test for Simulator navigation and first step."""

from datetime import date
from functools import partial
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from research.data import completed_daily_bars
from simulation.strategies import prepare_strategy


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
        app.button(key="sim_create").click().run(timeout=20)

    assert not app.exception
    assert app.button[0].label == "Step one bar"
    app.button(key="sim_step").click().run(timeout=20)

    assert not app.exception
    assert app.metric[0].label == "Portfolio"
    assert any("**Status:** Paused" in block.value for block in app.markdown)
    engine = app.session_state["historical_simulator_engine"]
    assert len(engine.state.snapshots) == 1

    app.button(key="sim_resume").click().run(timeout=20)
    app.button(key="sim_advance").click().run(timeout=20)
    app.button(key="sim_pause").click().run(timeout=20)
    assert not app.exception
    assert len(engine.state.snapshots) == 6
    assert any("**Status:** Paused" in block.value for block in app.markdown)

    app.button(key="sim_reset").click().run(timeout=20)
    assert not app.exception
    assert "historical_simulator_engine" not in app.session_state


@pytest.mark.parametrize("tz,last", [(None, "2026-09-26"), ("America/New_York", "2026-09-25")])
def test_simulator_prepares_and_replays_only_completed_bars(tz, last):
    frame = pd.DataFrame(
        {"Open": 100., "High": 100., "Low": 100., "Close": 100., "Volume": 1000.},
        index=pd.date_range("2026-09-01", "2026-09-28", tz=tz),
    )
    cutoff = partial(completed_daily_bars, as_of="2026-09-27T01:00:00Z")
    expected = cutoff(frame)
    app = AppTest.from_file("app.py").run(timeout=20)
    app.radio[0].set_value("Simulator").run(timeout=20)
    app.date_input(key="sim_start_date").set_value(date(2026, 9, 24)).run(timeout=20)
    with patch("ui.page_simulator.fetch_ohlcv", return_value=frame), patch(
        "ui.page_simulator.completed_daily_bars", wraps=cutoff
    ), patch("ui.page_simulator.prepare_strategy", wraps=prepare_strategy) as prepare:
        app.button(key="sim_create").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    pd.testing.assert_frame_equal(prepare.call_args.args[1], expected)
    engine = app.session_state["historical_simulator_engine"]
    pd.testing.assert_frame_equal(engine.df, expected)
    pd.testing.assert_frame_equal(app.session_state["historical_simulator_data"], expected)
    assert engine.active_strategy.matches_data(expected)
    assert str(engine.resolved_end_date.date()) == last
    engine.resume()
    engine.advance(100)
    assert engine.state.terminal
    assert engine.state.snapshots[-1].timestamp == expected.index[-1]
    assert len(frame) == 28  # the provider's frame remains intact


def test_simulator_reports_no_completed_bars_before_preparing_strategy():
    frame = pd.DataFrame(
        {"Open": 100., "High": 100., "Low": 100., "Close": 100., "Volume": 1000.},
        index=pd.date_range("2026-09-27", periods=2),
    )
    app = AppTest.from_file("app.py").run(timeout=20)
    app.radio[0].set_value("Simulator").run(timeout=20)
    app.date_input(key="sim_start_date").set_value(date(2026, 9, 27)).run(timeout=20)
    with patch("ui.page_simulator.fetch_ohlcv", return_value=frame), patch(
        "ui.page_simulator.completed_daily_bars",
        wraps=partial(completed_daily_bars, as_of="2026-09-27T12:00:00Z"),
    ), patch("ui.page_simulator.prepare_strategy", wraps=prepare_strategy) as prepare:
        app.button(key="sim_create").click().run(timeout=20)
    assert not app.exception
    assert any("No completed daily bars" in error.value for error in app.error)
    prepare.assert_not_called()
    assert "historical_simulator_engine" not in app.session_state
