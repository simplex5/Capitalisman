"""Compare page: side-by-side analysis of two tickers."""

from html import escape

import numpy as np
import streamlit as st

from charts.plotly_fallback import create_comparison_chart
from data.fetcher import (
    compute_buy_and_hold,
    fetch_with_warmup,
    get_asset_info,
    staleness_note,
)
from indicators.registry import get_all_indicators
from signals.base import SignalDirection
from signals.combiner import combine_signals
from ui.components import (
    analysis_settings_signature,
    horizon_input,
    indicator_picker,
    period_select,
)


def _signal_color(direction: SignalDirection) -> str:
    if direction == SignalDirection.BUY:
        return "#4CAF50"
    elif direction == SignalDirection.SELL:
        return "#F44336"
    return "#FFC107"


def _signal_emoji(direction: SignalDirection) -> str:
    if direction == SignalDirection.BUY:
        return "▲"
    elif direction == SignalDirection.SELL:
        return "▼"
    return "●"


def render():
    st.header("Compare")

    # Sidebar controls
    with st.sidebar:
        ticker_a = st.text_input(
            "Ticker A", value="AAPL", key="compare_ticker_a",
            help="First ticker to compare",
        ).upper().strip()
        ticker_b = st.text_input(
            "Ticker B", value="MSFT", key="compare_ticker_b",
            help="Second ticker to compare",
        ).upper().strip()
        period = period_select(key="compare_period")
        horizon = horizon_input(key="compare_horizon")
        selected_indicators = indicator_picker(key="compare_indicators")
        compare_clicked = st.button(
            "Compare",
            type="primary",
            use_container_width=True,
            key="compare_run",
        )

    if not ticker_a or not ticker_b:
        st.info("Enter two ticker symbols in the sidebar to compare.")
        return

    if ticker_a == ticker_b:
        st.warning("Enter two different tickers to compare.")
        return

    if not selected_indicators:
        st.warning("Select at least one indicator.")
        return

    request_signature = (
        ticker_a,
        ticker_b,
        period,
        horizon,
        tuple(selected_indicators),
        analysis_settings_signature(),
    )
    if compare_clicked:
        st.session_state["compare_request_signature"] = request_signature
    if st.session_state.get("compare_request_signature") != request_signature:
        st.info("Review the sidebar settings, then click **Compare**.")
        return

    # Fetch data for both tickers with warmup
    try:
        with st.spinner(f"Fetching {ticker_a}..."):
            full_a, display_a = fetch_with_warmup(ticker_a, period=period)
    except ValueError as e:
        st.error(f"{ticker_a}: {e}")
        return
    except Exception as e:
        st.error(f"{ticker_a}: unexpected error fetching data: {e}")
        return

    try:
        with st.spinner(f"Fetching {ticker_b}..."):
            full_b, display_b = fetch_with_warmup(ticker_b, period=period)
    except ValueError as e:
        st.error(f"{ticker_b}: {e}")
        return
    except Exception as e:
        st.error(f"{ticker_b}: unexpected error fetching data: {e}")
        return

    for label, frame in ((ticker_a, display_a), (ticker_b, display_b)):
        stale = staleness_note(frame)
        if stale:
            st.warning(f"{label}: {stale}")

    # Asset info
    info_a = get_asset_info(ticker_a)
    info_b = get_asset_info(ticker_b)

    # Metrics row (use display data for user-facing values)
    st.subheader("Overview")
    col1, col2 = st.columns(2)

    ret_a = compute_buy_and_hold(display_a)
    ret_b = compute_buy_and_hold(display_b)

    with col1:
        name_a = info_a["name"] if info_a else ticker_a
        sector_a = info_a["sector"] if info_a else "N/A"
        st.metric(
            label=f"{ticker_a} — {name_a}",
            value=f"${display_a['Close'].iloc[-1]:,.2f}",
            delta=f"{ret_a:+.1%} period return" if ret_a is not None else None,
        )
        st.caption(f"Sector: {sector_a}")

    with col2:
        name_b = info_b["name"] if info_b else ticker_b
        sector_b = info_b["sector"] if info_b else "N/A"
        st.metric(
            label=f"{ticker_b} — {name_b}",
            value=f"${display_b['Close'].iloc[-1]:,.2f}",
            delta=f"{ret_b:+.1%} period return" if ret_b is not None else None,
        )
        st.caption(f"Sector: {sector_b}")

    # Normalized comparison chart (display range only)
    st.subheader("Normalized Price Comparison")
    fig = create_comparison_chart(display_a, display_b, label_a=ticker_a, label_b=ticker_b)
    st.plotly_chart(fig, use_container_width=True)

    # Signal comparison
    st.subheader("Signal Comparison")

    all_indicators = get_all_indicators()
    chosen = {n: all_indicators[n] for n in selected_indicators if n in all_indicators}

    try:
        # Compute indicators and signals on full (warmup) data
        computed_a = full_a.copy()
        for indicator in chosen.values():
            computed_a = indicator.compute_for_horizon(computed_a, horizon)
        signal_a = combine_signals(chosen, computed_a, horizon_days=horizon, precomputed=True)

        computed_b = full_b.copy()
        for indicator in chosen.values():
            computed_b = indicator.compute_for_horizon(computed_b, horizon)
        signal_b = combine_signals(chosen, computed_b, horizon_days=horizon, precomputed=True)
    except Exception as e:
        st.error(f"Error computing signals: {e}")
        return

    col1, col2 = st.columns(2)

    for col, ticker, signal in [(col1, ticker_a, signal_a), (col2, ticker_b, signal_b)]:
        color = _signal_color(signal.direction)
        arrow = _signal_emoji(signal.direction)
        with col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}22, {color}11);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 16px;
                ">
                    <h2 style="color: {color}; margin: 0;">
                        {arrow} {escape(ticker)}: {signal.direction.value}
                    </h2>
                    <p style="font-size: 1.1em; opacity: 0.9; margin: 6px 0 0 0;">
                        Directional agreement: {signal.directional_agreement:.0%} &bull; Evidence strength: {signal.evidence_strength:.0%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if signal.reasoning:
                st.caption(signal.reasoning)

    # Correlation
    st.subheader("Correlation")

    returns_a = display_a["Close"].pct_change().dropna()
    returns_b = display_b["Close"].pct_change().dropna()

    # Inner join on dates to handle different trading calendars
    combined = returns_a.to_frame("A").join(returns_b.to_frame("B"), how="inner").dropna()

    if len(combined) > 1:
        corr = np.corrcoef(combined["A"], combined["B"])[0, 1]

        if abs(corr) >= 0.7:
            corr_label = "Strong"
        elif abs(corr) >= 0.4:
            corr_label = "Moderate"
        else:
            corr_label = "Weak"

        if corr >= 0:
            corr_desc = f"{corr_label} positive correlation"
        else:
            corr_desc = f"{corr_label} negative correlation"

        st.metric("Daily Return Correlation", f"{corr:.3f}")
        st.caption(
            f"{corr_desc} over {len(combined)} overlapping trading days. "
            f"A correlation near 1.0 means they tend to move together; "
            f"near -1.0 means they move oppositely."
        )
    else:
        st.info("Insufficient overlapping data to compute correlation.")
