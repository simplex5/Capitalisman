"""Interactive, replayable historical portfolio simulator."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from charts.factory import render_equity_curve, render_price_chart
from config.settings import DEFAULT_INITIAL_CAPITAL, DEFAULT_SIMULATION_COST_PER_FILL_PCT
from data.fetcher import fetch_ohlcv
from simulation.engine import HistoricalSimulationEngine
from simulation.models import (
    CompletionReason,
    GoalDefinition,
    GoalKind,
    SimulationConfig,
    SimulationStatus,
)
from simulation.strategies import (
    CORE_SIMULATION_INDICATORS,
    build_custom_strategy,
    get_strategy_presets,
    prepare_strategy,
)
from ui.components import record_recent_ticker, render_recent_tickers, ticker_input


_ENGINE_KEY = "historical_simulator_engine"
_DATA_KEY = "historical_simulator_data"


def _strategy_controls():
    presets = get_strategy_presets()
    choice = st.selectbox(
        "Strategy",
        [*presets, "Custom"],
        key="sim_strategy_choice",
    )
    if choice != "Custom":
        definition = presets[choice]
        if definition.buy_and_hold:
            st.caption("Invest at the first eligible open and remain long.")
        else:
            st.caption(
                f"{', '.join(definition.indicator_names)} · "
                f"{definition.horizon_days}-bar weighting horizon"
            )
        return definition

    selected = st.multiselect(
        "Indicators",
        CORE_SIMULATION_INDICATORS,
        default=["SMA Crossover", "MACD", "RSI"],
        key="sim_custom_indicators",
        help="Only causal, single-asset technical indicators are available in the MVP.",
    )
    horizon = st.slider(
        "Weighting horizon (bars)", 1, 30, 5, key="sim_custom_horizon"
    )
    ambiguity = st.slider(
        "Ambiguity threshold",
        0.0,
        0.5,
        0.10,
        0.01,
        key="sim_custom_ambiguity",
        help="Return HOLD when bullish and bearish weighted scores are this close.",
    )
    weights = {}
    with st.expander("Custom indicator weights"):
        for indicator in selected:
            weights[indicator] = st.number_input(
                indicator,
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key=f"sim_weight_{indicator}",
            )
    if not selected:
        return None
    return build_custom_strategy(
        selected,
        horizon_days=horizon,
        ambiguity_threshold=ambiguity,
        weights=weights,
    )


def _goal_controls() -> GoalDefinition:
    choice = st.selectbox(
        "Success goal",
        ["No profit goal", "Profit ($)", "Return (%)"],
        key="sim_goal_kind",
    )
    if choice == "No profit goal":
        return GoalDefinition()
    if choice == "Profit ($)":
        target = st.number_input(
            "Target profit ($)", min_value=1.0, value=5_000.0, step=100.0,
            key="sim_goal_dollars",
        )
        kind = GoalKind.PROFIT_DOLLARS
    else:
        target = st.number_input(
            "Target return (%)", min_value=0.1, value=5.0, step=0.5,
            key="sim_goal_percent",
        ) / 100.0
        kind = GoalKind.RETURN_PERCENT
    stop = st.checkbox(
        "Stop when goal is reached", value=True, key="sim_goal_stop"
    )
    return GoalDefinition(kind, float(target), stop_when_reached=stop)


def _requested_end(start_date: date) -> pd.Timestamp | None:
    mode = st.selectbox(
        "Run until",
        ["Latest available bar", "Specific date", "Duration"],
        key="sim_endpoint_mode",
    )
    if mode == "Latest available bar":
        return None
    if mode == "Specific date":
        selected = st.date_input(
            "End date",
            value=date.today(),
            min_value=start_date,
            key="sim_end_date",
        )
        return pd.Timestamp(selected) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    duration = st.number_input(
        "Calendar days", min_value=1, max_value=10_000, value=365, step=30,
        key="sim_duration_days",
    )
    return pd.Timestamp(start_date) + pd.Timedelta(days=int(duration))


def _details(event) -> dict[str, str]:
    return dict(event.details)


def _render_market_chart(engine: HistoricalSimulationEngine) -> None:
    # Show recent context only through the last revealed close. Before the first
    # step this means history ending immediately before the drop-in bar.
    end = engine.state.current_position
    start = max(0, engine.start_position - 90)
    display = engine.df.iloc[start : end + 1]
    if display.empty:
        return
    markers = []
    for action, color, symbol in (
        ("BUY", "#4CAF50", "triangle-up"),
        ("SELL", "#F44336", "triangle-down"),
    ):
        fills = [
            event
            for event in engine.state.events
            if event.event_type.value == "order_filled" and _details(event).get("action") == action
        ]
        if fills:
            markers.append(
                {
                    "dates": [event.timestamp.strftime("%Y-%m-%d") for event in fills],
                    "prices": [float(_details(event)["price"]) for event in fills],
                    "color": color,
                    "symbol": symbol,
                    "name": action.title(),
                    "size": 9,
                }
            )
    render_price_chart(
        display,
        title=f"{engine.config.ticker} — Historical Replay",
        markers=markers,
        height=480,
    )


def _render_run(engine: HistoricalSimulationEngine) -> None:
    state = engine.state
    portfolio = state.portfolio
    equity = portfolio.equity
    profit = equity - engine.config.starting_capital
    return_pct = profit / engine.config.starting_capital

    st.subheader(f"{engine.config.ticker} replay")
    st.caption(
        f"Resolved range: {engine.resolved_start_date:%Y-%m-%d} to "
        f"{engine.resolved_end_date:%Y-%m-%d}. Decisions use the close and fill at "
        "the next open. Long or cash only; each fill incurs the configured cost."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Portfolio", f"${equity:,.2f}", f"{return_pct:+.2%}")
    m2.metric("Profit / Loss", f"${profit:+,.2f}")
    m3.metric("Cash", f"${portfolio.cash:,.2f}")
    m4.metric("Position", f"{portfolio.quantity:,.6f}")
    m5.metric("Fees", f"${portfolio.total_fees:,.2f}")

    date_label = state.current_date.strftime("%Y-%m-%d") if state.current_date is not None else "Not started"
    pending = state.pending_order.action.value if state.pending_order else "None"
    st.write(
        f"**Status:** {state.status.value.title()} · **Date:** {date_label} · "
        f"**Strategy:** {engine.active_strategy.definition.name} · "
        f"**Pending next-open order:** {pending}"
    )
    total_bars = engine.end_position - engine.start_position + 1
    revealed_bars = max(0, state.current_position - engine.start_position + 1)
    st.progress(min(1.0, revealed_bars / total_bars))
    st.caption(f"Revealed {revealed_bars:,} of {total_bars:,} market bars.")
    if engine.config.goal.kind != GoalKind.NONE and engine.config.goal.target is not None:
        if engine.config.goal.kind == GoalKind.PROFIT_DOLLARS:
            progress = profit / engine.config.goal.target
            goal_label = f"${engine.config.goal.target:,.2f} profit"
        else:
            progress = return_pct / engine.config.goal.target
            goal_label = f"{engine.config.goal.target:.1%} return"
        st.progress(max(0.0, min(1.0, progress)))
        st.caption(f"Goal progress toward {goal_label}: {max(0.0, progress):.1%}")
    if state.terminal:
        reason = state.completion_reason.value.replace("_", " ").title()
        if state.completion_reason == CompletionReason.CAPITAL_DEPLETED:
            st.error(f"Replay complete: {reason}.")
        elif engine.config.goal.kind != GoalKind.NONE and not state.goal_reached:
            st.warning(f"Replay complete: {reason}. The selected profit goal was not reached.")
        elif state.goal_reached:
            st.success(f"Replay complete: {reason}. The selected profit goal was reached.")
        else:
            st.info(f"Replay complete: {reason}.")

    speed = st.select_slider(
        "Batch size",
        options=[1, 2, 5, 10, 25, 50, 100],
        value=5,
        key="sim_speed",
        help="Each unit advances one daily market bar. Continuous timed autoplay is deferred.",
    )
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Step one bar", disabled=state.terminal, use_container_width=True):
        engine.step()
        st.rerun()
    if c2.button(
        "Resume batch mode",
        disabled=state.terminal or state.status == SimulationStatus.RUNNING,
        use_container_width=True,
    ):
        engine.resume()
        st.rerun()
    if c3.button(
        f"Advance {speed} bar{'s' if speed != 1 else ''}",
        disabled=state.terminal or state.status != SimulationStatus.RUNNING,
        use_container_width=True,
    ):
        engine.advance(int(speed))
        st.rerun()
    if c4.button(
        "Pause",
        disabled=state.terminal or state.status != SimulationStatus.RUNNING,
        use_container_width=True,
    ):
        engine.pause()
        st.rerun()

    st.caption(
        "Batch mode is deterministic: 100 bars produces the same result as 100 single steps. "
        "Pause before changing strategy."
    )

    switch_col, reset_col = st.columns([3, 1])
    with switch_col:
        if st.button(
            "Apply selected sidebar strategy",
            disabled=state.terminal or state.status == SimulationStatus.RUNNING,
            help="Keeps the current holding, cancels any pending order, and applies the new strategy prospectively.",
        ):
            definition = st.session_state.get("simulator_selected_definition")
            if definition is not None:
                try:
                    engine.change_strategy(prepare_strategy(definition, engine.df))
                    st.rerun()
                except (ValueError, RuntimeError) as error:
                    st.error(str(error))
    with reset_col:
        if st.button("Reset replay", use_container_width=True):
            st.session_state.pop(_ENGINE_KEY, None)
            st.session_state.pop(_DATA_KEY, None)
            st.rerun()

    _render_market_chart(engine)
    equity_series = engine.equity_series()
    if len(equity_series) >= 2:
        render_equity_curve(equity_series, title="Marked-to-Market Portfolio Equity")

    with st.expander("Simulation event ledger", expanded=False):
        rows = []
        for event in state.events:
            row = {
                "#": event.sequence,
                "Date": event.timestamp.strftime("%Y-%m-%d"),
                "Event": event.event_type.value.replace("_", " ").title(),
                "Description": event.message,
            }
            row.update(_details(event))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render() -> None:
    st.header("Simulator")
    st.write(
        "Drop into a historical date and replay a long-or-cash strategy one daily bar at a time. "
        "This MVP prioritizes causal execution, marked-to-market accounting, and reproducibility."
    )

    with st.sidebar:
        st.subheader("Simulation setup")
        ticker = ticker_input(key="sim_ticker")
        render_recent_tickers("sim_ticker")
        default_start = date.today() - timedelta(days=365)
        start = st.date_input("Drop-in date", value=default_start, key="sim_start_date")
        initial_capital = st.number_input(
            "Starting capital ($)",
            min_value=100.0,
            max_value=100_000_000.0,
            value=DEFAULT_INITIAL_CAPITAL,
            step=1_000.0,
            key="sim_capital",
        )
        transaction_cost = st.number_input(
            "Cost per fill (%)",
            min_value=0.0,
            max_value=5.0,
            value=DEFAULT_SIMULATION_COST_PER_FILL_PCT,
            step=0.05,
            key="sim_cost",
            help="Charged once when buying and once when selling.",
        )
        definition = _strategy_controls()
        st.session_state["simulator_selected_definition"] = definition
        goal = _goal_controls()
        end = _requested_end(start)
        create_clicked = st.button(
            "Create replay",
            type="primary",
            use_container_width=True,
            disabled=not ticker or definition is None,
            key="sim_create",
        )

    if create_clicked:
        try:
            with st.spinner("Fetching daily history and preparing causal signals..."):
                df = fetch_ohlcv(ticker, period="max", interval="1d")
                prepared = prepare_strategy(definition, df)
                config = SimulationConfig(
                    ticker=ticker,
                    start_date=pd.Timestamp(start),
                    end_date=end,
                    starting_capital=float(initial_capital),
                    transaction_cost_pct=float(transaction_cost),
                    goal=goal,
                )
                engine = HistoricalSimulationEngine(df, config, prepared)
            st.session_state[_ENGINE_KEY] = engine
            st.session_state[_DATA_KEY] = df
            record_recent_ticker(ticker)
            st.rerun()
        except (ValueError, KeyError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Could not create simulation: {error}")

    engine = st.session_state.get(_ENGINE_KEY)
    if engine is None:
        st.info("Choose a ticker, date, strategy, endpoint, and optional goal, then create a replay.")
        st.markdown(
            "**MVP boundaries:** one ticker, daily bars, one all-in long position or cash, "
            "no shorting or margin, and technical indicators only. TimesFM and external-data "
            "signals remain out until their historical inputs can be reproduced safely."
        )
        return
    _render_run(engine)
