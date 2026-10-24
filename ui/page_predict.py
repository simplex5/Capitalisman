"""Predict page: signal generation and display."""

from html import escape

import plotly.graph_objects as go
import streamlit as st

from charts.factory import render_price_chart
from data.fetcher import fetch_with_warmup
from indicators.registry import get_all_indicators
from signals.base import SignalDirection
from signals.combiner import combine_signals
from ui.components import (
    advanced_settings,
    analysis_settings_signature,
    check_data_sufficiency,
    horizon_input,
    indicator_picker,
    period_select,
    record_recent_ticker,
    render_recent_tickers,
    ticker_input,
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


def _render_timesfm_panel(ticker, horizon, analysis, model_signal, error):
    """Render a distinct model-only result or an explicit failure state."""
    st.subheader("TimesFM Model Forecast")
    st.caption(
        "Standalone local-ML result. This is separate from the combined indicator "
        "signal shown below."
    )

    if analysis is None:
        st.error(
            "TimesFM was selected but did not produce a forecast. "
            f"{error or 'Review the runtime status in the sidebar and try again.'}"
        )
        return

    direction = model_signal.direction if model_signal else SignalDirection.HOLD
    color = _signal_color(direction)
    arrow = _signal_emoji(direction)
    safe_ticker = escape(str(ticker))
    safe_origin = escape(str(analysis.origin))
    safe_model = escape(str(analysis.forecast.model_id))
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #6f42c122, #00bcd411);
            border: 2px solid #9c6ade;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <p style="font-size: 0.8em; letter-spacing: 0.12em; opacity: 0.75; margin: 0;">
                {safe_model} &bull; {analysis.forecast.device.upper()} &bull; MODEL-ONLY
            </p>
            <h2 style="color: {color}; margin: 6px 0;">
                {arrow} {direction.value} FORECAST
            </h2>
            <p style="margin: 0; opacity: 0.85;">
                {safe_ticker} &bull; {horizon}-bar horizon &bull; origin {safe_origin}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upper_return = analysis.upper_price / analysis.current_price - 1.0
    return_paths = {
        "q10": analysis.forecast.quantiles[0.1] / analysis.current_price - 1.0,
        "median": analysis.forecast.quantiles[0.5] / analysis.current_price - 1.0,
        "q90": analysis.forecast.quantiles[0.9] / analysis.current_price - 1.0,
    }

    return_cols = st.columns(4)
    return_cols[0].metric("Median Return", f"{analysis.expected_return:+.1%}")
    return_cols[1].metric("q10 Downside", f"{analysis.downside_return:+.1%}")
    return_cols[2].metric("q90 Upside", f"{upper_return:+.1%}")
    return_cols[3].metric("q10–q90 Width", f"{analysis.interval_width:.1%}")

    probability_cols = st.columns(3)
    def probability_label(value):
        if value <= 0.1000001:
            return "≤10%"
        if value >= 0.8999999:
            return "≥90%"
        return f"{value:.0%}"

    probability_cols[0].metric("Model Up Estimate", probability_label(analysis.probability_up))
    probability_cols[1].metric(
        "Long Profit Estimate", probability_label(analysis.probability_profit)
    )
    probability_cols[2].metric(
        "Short Profit Estimate", probability_label(analysis.probability_short_profit)
    )
    st.caption(
        "Uncalibrated model estimates from quantiles. Values beyond q10–q90 are "
        "shown as tail bounds. Profit estimates use the origin close and "
        f"{analysis.cost_per_trade_pct:g}% quoted round-trip cost, split across fills; "
        "the next opening price is unknown."
    )

    future_bars = list(range(1, analysis.horizon + 1))
    forecast_chart = go.Figure()
    forecast_chart.add_trace(
        go.Scatter(
            x=future_bars,
            y=return_paths["q10"],
            mode="lines",
            name="q10 downside",
            line=dict(color="#38BDF8", width=1.5),
            hovertemplate="Future bar %{x}<br>q10: %{y:.2%}<extra></extra>",
        )
    )
    forecast_chart.add_trace(
        go.Scatter(
            x=future_bars,
            y=return_paths["q90"],
            mode="lines",
            name="q90 upside",
            line=dict(color="#A78BFA", width=1.5),
            fill="tonexty",
            fillcolor="rgba(139, 92, 246, 0.18)",
            hovertemplate="Future bar %{x}<br>q90: %{y:.2%}<extra></extra>",
        )
    )
    forecast_chart.add_trace(
        go.Scatter(
            x=future_bars,
            y=return_paths["median"],
            mode="lines+markers",
            name="Median forecast",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=5),
            hovertemplate="Future bar %{x}<br>Median: %{y:.2%}<extra></extra>",
        )
    )
    forecast_chart.add_hline(
        y=0,
        line_dash="dot",
        line_color="rgba(148, 163, 184, 0.7)",
        annotation_text="Current price",
    )
    forecast_chart.update_layout(
        title="Forecast Return Path and 80% Quantile Band",
        xaxis_title="Future bar",
        yaxis_title="Return from current price",
        yaxis_tickformat="+.1%",
        hovermode="x unified",
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(forecast_chart, use_container_width=True)
    st.caption(
        "The purple area is the model's q10–q90 interval. A wider band means "
        "greater forecast uncertainty; the orange line is the median path."
    )
    if model_signal:
        st.markdown(f"**Model interpretation:** {model_signal.detail}")
    st.caption(
        f"Model: {analysis.forecast.model_id} · Device: "
        f"{analysis.forecast.device.upper()} · q10–q90 is an estimated 80% interval, "
        "not a guarantee."
    )


def render():
    st.header("Predict")

    # Sidebar controls
    with st.sidebar:
        ticker = ticker_input(key="predict_ticker")
        render_recent_tickers("predict_ticker")
        period = period_select(key="predict_period")
        horizon = horizon_input(key="predict_horizon")
        selected_indicators = indicator_picker(key="predict_indicators")
        advanced_settings(key_prefix="predict_adv")
        timesfm_selected = "TimesFM Forecast" in selected_indicators
        analyze_clicked = st.button(
            "Run TimesFM Forecast" if timesfm_selected else "Analyze",
            type="primary",
            use_container_width=True,
            key="predict_analyze",
        )

    if not ticker:
        st.info("Enter a ticker symbol in the sidebar to get started.")
        return

    if not selected_indicators:
        st.warning("Select at least one indicator.")
        return

    request_signature = (
        ticker,
        period,
        horizon,
        tuple(selected_indicators),
        analysis_settings_signature(),
    )
    if analyze_clicked:
        st.session_state["predict_request_signature"] = request_signature
    if st.session_state.get("predict_request_signature") != request_signature:
        st.info("Review the sidebar settings, then click **Analyze**.")
        return

    record_recent_ticker(ticker)

    # Fetch data with warmup for short periods
    try:
        with st.spinner("Fetching data..."):
            full_df, display_df = fetch_with_warmup(ticker, period=period)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Unexpected error fetching data: {e}")
        return

    try:
        # Check data sufficiency against full (warmup) data
        data_warnings = check_data_sufficiency(len(full_df), selected_indicators)
        if data_warnings:
            st.warning(
                f"Insufficient data for {len(data_warnings)} indicator(s) "
                f"({len(full_df)} bars available):\n\n" + "\n".join(f"- {w}" for w in data_warnings)
                + "\n\nThese will report HOLD. Use a longer period for full analysis."
            )

        # Build selected indicators
        all_indicators = get_all_indicators()
        chosen = {n: all_indicators[n] for n in selected_indicators if n in all_indicators}

        # Compute indicators on full data (includes warmup)
        computed_df = full_df.copy()
        for name, indicator in chosen.items():
            computed_df = indicator.compute_for_horizon(computed_df, horizon)

        # Generate combined signal from full data
        signal = combine_signals(chosen, computed_df, horizon_days=horizon, precomputed=True)
        timesfm_indicator = chosen.get("TimesFM Forecast")
        timesfm_analysis = getattr(timesfm_indicator, "latest_analysis", None)
        timesfm_error = getattr(timesfm_indicator, "last_error", None)
        timesfm_signal = next(
            (
                item
                for item in signal.individual_signals
                if item.indicator_name == "TimesFM Forecast"
            ),
            None,
        )

        # Trim computed data to display range for charting
        computed_display = computed_df.iloc[-len(display_df):]
    except Exception as e:
        st.error(f"Error computing signals: {e}")
        return

    if timesfm_selected:
        if timesfm_analysis is not None:
            limit = timesfm_indicator.runtime.config.effective_context
            used = min(len(full_df), limit)
            st.caption(
                f"History supplied to TimesFM: {used:,} daily bars "
                f"({full_df.index[-used].date()} through {full_df.index[-1].date()}); "
                f"configured limit: {limit:,}."
            )
        _render_timesfm_panel(
            ticker,
            horizon,
            timesfm_analysis,
            timesfm_signal,
            timesfm_error,
        )
        st.divider()
        st.subheader("Combined Indicator Context")
        st.caption(
            f"Ensemble result from {len(selected_indicators)} selected indicators, "
            "including TimesFM."
        )

    # Multi-timeframe signal cards (1d, 5d, 20d)
    _MTF_HORIZONS = [("1d", 1), ("5d", 5), ("20d", 20)]
    mtf_cols = st.columns(len(_MTF_HORIZONS))
    for col, (label, h) in zip(mtf_cols, _MTF_HORIZONS):
        mtf_signal = combine_signals(chosen, computed_df, horizon_days=h, precomputed=True)
        mtf_color = _signal_color(mtf_signal.direction)
        mtf_arrow = _signal_emoji(mtf_signal.direction)
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid {mtf_color};
                    border-radius: 8px;
                    padding: 12px;
                    text-align: center;
                    margin-bottom: 12px;
                ">
                    <p style="font-size: 0.85em; opacity: 0.7; margin: 0;">{label}</p>
                    <p style="color: {mtf_color}; font-weight: bold; font-size: 1.2em; margin: 4px 0;">
                        {mtf_arrow} {mtf_signal.direction.value}
                    </p>
                    <p style="font-size: 0.85em; opacity: 0.85; margin: 0;">
                        Agreement: {mtf_signal.directional_agreement:.0%} &bull; Evidence: {mtf_signal.evidence_strength:.0%}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Primary signal card
    color = _signal_color(signal.direction)
    arrow = _signal_emoji(signal.direction)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 2px solid {color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <h1 style="color: {color}; margin: 0;">
                {arrow} {signal.direction.value}
            </h1>
            <p style="font-size: 1.3em; opacity: 0.9; margin: 8px 0 0 0;">
                Directional agreement: {signal.directional_agreement:.0%} &bull; Evidence strength: {signal.evidence_strength:.0%}
            </p>
            <p style="font-size: 0.9em; opacity: 0.7; margin: 4px 0 0 0;">
                {horizon}-bar horizon &bull; {len(selected_indicators)} indicators
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Reasoning narrative
    if signal.reasoning:
        st.markdown(f"**Analysis:** {signal.reasoning}")
    st.caption(
        "Directional agreement measures the share of actionable weighted votes "
        "supporting the leading direction. Evidence strength divides actionable weighted "
        "confidence by all selected indicator weight, including neutral indicators. "
        "Neither is a probability of success. The default 15% evidence floor is a heuristic."
    )

    # Score breakdown
    col1, col2 = st.columns(2)
    col1.metric("BUY Score", f"{signal.scores.get('BUY', 0):.2f}")
    col2.metric("SELL Score", f"{signal.scores.get('SELL', 0):.2f}")

    # Chart
    overlays = []
    subplot_configs = []
    for name, indicator in chosen.items():
        chart_cfg = indicator.get_chart_config()
        if chart_cfg.get("overlay"):
            overlays.append(chart_cfg)
        else:
            subplot_configs.append(chart_cfg)

    render_price_chart(
        computed_display,
        title=f"{ticker} — Signal: {signal.direction.value}",
        overlays=overlays,
        subplots=subplot_configs,
        height=700,
    )

    # Indicator breakdown table
    st.subheader("Indicator Breakdown")
    rows = []
    for sig in signal.individual_signals:
        rows.append({
            "Indicator": sig.indicator_name,
            "Direction": sig.direction.value,
            "Confidence": f"{sig.confidence:.0%}",
            "Detail": sig.detail,
        })

    st.table(rows)
