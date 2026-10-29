"""Verify the chart boundary without loading a browser or optional renderer."""

from unittest.mock import Mock

import pandas as pd
import pytest

from charts import tradingview


@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_intraday_candles_and_named_overlays_preserve_timestamps(monkeypatch, timezone):
    index = pd.date_range("2024-01-02 09:30", periods=3, freq="min", tz=timezone)
    df = pd.DataFrame(
        {"Open": 100., "High": 102., "Low": 99., "Close": 101., "SMA": [None, 100., 101.]},
        index=index,
    )
    chart = Mock()
    monkeypatch.setattr(tradingview, "TV_AVAILABLE", True)
    monkeypatch.setattr(tradingview, "StreamlitChart", Mock(return_value=chart), raising=False)

    tradingview.create_tv_chart(df, overlays=[{"columns": ["SMA"]}])

    candles = chart.set.call_args.args[0]
    assert candles["time"].is_unique
    assert pd.DatetimeIndex(candles["time"]).equals(index)
    line = chart.create_line.return_value.set.call_args.args[0]
    assert list(line.columns) == ["time", "SMA"]
    assert pd.DatetimeIndex(line["time"]).equals(index[1:])
