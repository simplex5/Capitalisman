"""Backtest page: historical prediction accuracy evaluation."""

import pandas as pd
import streamlit as st

from backtesting.engine import run_backtest
from charts.factory import render_equity_curve, render_price_chart
from config.settings import BACKTEST_FETCH_PERIOD
from data.fetcher import (
    compute_open_to_close_return,
    fetch_ohlcv,
    slice_date_range,
    trim_to_period,
)
from indicators.registry import get_all_indicators
from ui.components import (
    advanced_settings,
    analysis_settings_signature,
    capital_input,
    check_data_sufficiency,
    cost_input,
    horizon_input,
    indicator_picker,
    period_select,
    record_recent_ticker,
    render_recent_tickers,
    ticker_input,
)

_DEFAULT_BENCHMARK = "SPY"


def render():
    st.header("Backtest")

    # Sidebar controls
    with st.sidebar:
        ticker = ticker_input(key="backtest_ticker")
        render_recent_tickers("backtest_ticker")
        period = period_select(key="backtest_period", default="1y")
        horizon = horizon_input(key="backtest_horizon")
        selected_indicators = indicator_picker(
            key="backtest_indicators", timesfm_use_case="backtest"
        )
        initial_capital = capital_input(key="backtest_capital")
        cost_pct = cost_input(key="backtest_cost")
        advanced_settings(key_prefix="backtest_adv")
        run_clicked = st.button(
            "Run Backtest",
            type="primary",
            use_container_width=True,
            key="backtest_run",
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
        initial_capital,
        cost_pct,
        analysis_settings_signature(),
    )
    if run_clicked:
        st.session_state["backtest_request_signature"] = request_signature
    if st.session_state.get("backtest_request_signature") != request_signature:
        st.info("Review the sidebar settings, then click **Run Backtest**.")
        return

    record_recent_ticker(ticker)

    # Fetch data — for short periods, fetch extra history so indicators
    # and the backtest engine have enough warmup bars.
    fetch_period = BACKTEST_FETCH_PERIOD.get(period, period)
    try:
        with st.spinner("Fetching data..."):
            df = fetch_ohlcv(ticker, period=fetch_period)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Unexpected error fetching data: {e}")
        return

    # Check data sufficiency
    data_warnings = check_data_sufficiency(len(df), selected_indicators)
    if data_warnings:
        st.warning(
            f"Insufficient data for {len(data_warnings)} indicator(s) "
            f"({len(df)} bars available):\n\n" + "\n".join(f"- {w}" for w in data_warnings)
            + "\n\nThese will report HOLD until enough warmup data is available."
        )

    # Build selected indicators
    all_indicators = get_all_indicators()
    chosen = {n: all_indicators[n] for n in selected_indicators if n in all_indicators}
    evaluation_df = trim_to_period(df, period)
    evaluation_start = evaluation_df.index[0] if not evaluation_df.empty else None

    # Run backtest
    try:
        with st.spinner("Running backtest..."):
            report = run_backtest(
                df, chosen, ticker=ticker, period=period,
                horizon_days=horizon, initial_capital=initial_capital,
                cost_per_trade_pct=cost_pct,
                evaluation_start=evaluation_start,
            )
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        return

    if report.excluded_indicators:
        st.warning(
            "Excluded because the indicator is not historically compatible with this setup "
            "(point-in-time data or horizon constraint): "
            + ", ".join(report.excluded_indicators)
            + ". These indicators remain available for current predictions."
        )

    if report.total_trades == 0:
        st.warning(
            "No trades generated in the requested evaluation window. "
            "Try a longer period or indicators with shorter lookbacks."
        )
        return

    first_entry = report.trades[0].entry_date
    last_exit = report.trades[-1].exit_date
    st.caption(
        f"Evaluation: {first_entry:%Y-%m-%d} to {last_exit:%Y-%m-%d}. "
        f"Signals use closing data and enter at the next bar's open; exits use the close "
        f"after {horizon} bars. Short positions are modeled without borrow costs."
    )

    # Compute benchmark/buy-and-hold returns for context
    ticker_eval = slice_date_range(df, first_entry, last_exit)
    ticker_bh = compute_open_to_close_return(ticker_eval)

    benchmark_bh = None
    if ticker.upper() != _DEFAULT_BENCHMARK:
        try:
            bench_df = fetch_ohlcv(_DEFAULT_BENCHMARK, period=fetch_period)
            bench_eval = slice_date_range(bench_df, first_entry, last_exit)
            benchmark_bh = compute_open_to_close_return(bench_eval)
        except Exception:
            pass  # benchmark fetch failed — not critical

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Trades", report.total_trades)
    m2.metric("Win Rate", f"{report.win_rate:.1%}",
              help="Percentage of trades with positive P&L (after costs)")
    m3.metric("Prediction Accuracy", f"{report.prediction_accuracy:.1%}",
              help="Percentage of trades where predicted direction matched actual price movement")
    m4.metric("Return", f"{report.cumulative_return:+.1%}")
    m5.metric("Max Drawdown", f"{report.max_drawdown:.1%}")

    m6, m7, m8 = st.columns(3)
    pf_display = "No losses" if report.profit_factor == float("inf") else f"{report.profit_factor:.2f}"
    m6.metric("Sharpe Ratio", f"{report.sharpe_ratio:.2f}")
    m7.metric("Profit Factor", pf_display)
    m8.metric(
        "Time in Market",
        f"{report.exposure_pct:.1%}",
        help="Share of the evaluated calendar span with an open modeled position",
    )

    # Benchmark context
    if ticker_bh is not None or benchmark_bh is not None:
        st.markdown("---")
        st.subheader("vs Buy & Hold")
        bh_cols = st.columns(3)
        bh_cols[0].metric(
            "Strategy Return",
            f"{report.cumulative_return:+.1%}",
        )
        if ticker_bh is not None:
            alpha = report.cumulative_return - ticker_bh
            bh_cols[1].metric(
                f"{ticker} Buy & Hold",
                f"{ticker_bh:+.1%}",
                delta=f"{alpha:+.1%} strategy alpha" if alpha != 0 else None,
                delta_color="normal",
            )
        if benchmark_bh is not None:
            alpha_vs_bench = report.cumulative_return - benchmark_bh
            bh_cols[2].metric(
                f"{_DEFAULT_BENCHMARK} Buy & Hold",
                f"{benchmark_bh:+.1%}",
                delta=f"{alpha_vs_bench:+.1%} vs market" if alpha_vs_bench != 0 else None,
                delta_color="normal",
            )

    # Price chart with prediction markers
    computed_df = df.copy()
    for name, indicator in chosen.items():
        computed_df = indicator.compute_for_horizon(computed_df, horizon)

    correct_trades = [t for t in report.trades if t.correct]
    incorrect_trades = [t for t in report.trades if not t.correct]

    markers = []
    if correct_trades:
        markers.append({
            "dates": [t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date)
                      for t in correct_trades],
            "prices": [t.entry_price for t in correct_trades],
            "color": "#4CAF50",
            "symbol": "triangle-up",
            "name": "Correct",
            "size": 8,
        })
    if incorrect_trades:
        markers.append({
            "dates": [t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date)
                      for t in incorrect_trades],
            "prices": [t.entry_price for t in incorrect_trades],
            "color": "#F44336",
            "symbol": "triangle-down",
            "name": "Incorrect",
            "size": 8,
        })

    computed_display = computed_df.loc[computed_df.index >= evaluation_start]
    render_price_chart(
        computed_display,
        title=f"{ticker} — Backtest Results",
        markers=markers,
        height=500,
    )

    # Equity curve
    if not report.equity_curve.empty:
        render_equity_curve(report.equity_curve, title="Equity Curve")

    # Trade log
    with st.expander("Trade Log", expanded=False):
        trade_data = []
        for t in report.trades:
            trade_data.append({
                "Entry Date": t.entry_date.strftime("%Y-%m-%d") if hasattr(t.entry_date, "strftime") else str(t.entry_date),
                "Exit Date": t.exit_date.strftime("%Y-%m-%d") if hasattr(t.exit_date, "strftime") else str(t.exit_date),
                "Direction": t.direction,
                "Entry $": f"{t.entry_price:.2f}",
                "Exit $": f"{t.exit_price:.2f}",
                "P&L": f"{t.pnl_pct:+.2%}",
                "Correct": "Yes" if t.correct else "No",
            })
        trade_df = pd.DataFrame(trade_data)
        st.dataframe(trade_df, use_container_width=True)

        csv = trade_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"backtest_{ticker}_{period}_{horizon}d.csv",
            mime="text/csv",
        )
