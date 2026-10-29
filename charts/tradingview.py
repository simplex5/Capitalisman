"""TradingView Lightweight Charts rendering for Streamlit."""

from typing import Any

import pandas as pd

try:
    from lightweight_charts.widgets import StreamlitChart

    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False


def is_available() -> bool:
    return TV_AVAILABLE


def create_tv_chart(
    df: pd.DataFrame,
    title: str = "",
    overlays: list[dict[str, Any]] | None = None,
    subplots: list[dict[str, Any]] | None = None,
    height: int = 600,
    width: int | None = None,
) -> Any | None:
    """Create a TradingView Lightweight Chart in Streamlit.

    Returns the chart object, or None if lightweight-charts is unavailable.
    """
    if not TV_AVAILABLE:
        return None

    overlays = overlays or []
    subplots = subplots or []

    chart = StreamlitChart(height=height, width=width)
    chart.layout(background_color="#1E1E1E", text_color="#DDD")
    chart.candle_style(
        up_color="#26A69A",
        down_color="#EF5350",
        wick_up_color="#26A69A",
        wick_down_color="#EF5350",
    )

    # Prepare data for lightweight-charts (needs 'time', 'open', 'high', 'low', 'close')
    chart_df = df[["Open", "High", "Low", "Close"]].copy()
    chart_df.columns = ["open", "high", "low", "close"]
    if isinstance(df.index, pd.DatetimeIndex):
        # Pass real timestamps so minute/hour bars retain distinct times.
        chart_df["time"] = df.index.as_unit("ns")
    elif hasattr(df.index, "strftime"):
        chart_df["time"] = df.index.strftime("%Y-%m-%d")
    else:
        chart_df["time"] = df.index.astype(str)
    chart_df = chart_df[["time", "open", "high", "low", "close"]]

    chart.set(chart_df)

    # Add overlays as line series
    for overlay in overlays:
        for col_name in overlay.get("columns", []):
            if col_name in df.columns:
                color = overlay.get("colors", {}).get(col_name, "#888888")
                line = chart.create_line(name=col_name, color=color, width=1)
                line_df = pd.DataFrame({
                    "time": chart_df["time"],
                    col_name: df[col_name].values,
                }).dropna()
                line.set(line_df)

    return chart
