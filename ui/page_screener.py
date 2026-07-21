"""Screener page: scan multiple tickers and rank by signal strength."""

import streamlit as st

from config.settings import WATCHLIST_PRESETS
from data.fetcher import fetch_ohlcv, get_asset_info
from data.watchlists import delete_watchlist, load_watchlists, save_watchlist
from indicators.registry import get_all_indicators
from signals.base import SignalDirection
from signals.combiner import combine_signals
from ui.components import horizon_input, indicator_picker


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
    st.header("Screener")

    # Sidebar controls
    with st.sidebar:
        # Merge built-in presets with user-saved watchlists
        builtin_names = list(WATCHLIST_PRESETS.keys())
        user_watchlists = load_watchlists()
        user_names = [f"{n} (saved)" for n in user_watchlists]
        all_options = builtin_names + user_names

        preset_name = st.selectbox(
            "Watchlist",
            options=all_options,
            key="screener_preset",
        )

        is_user_watchlist = preset_name.endswith(" (saved)")
        raw_name = preset_name.removesuffix(" (saved)") if is_user_watchlist else preset_name

        if preset_name == "Custom":
            custom_input = st.text_input(
                "Tickers (comma-separated)",
                placeholder="AAPL, MSFT, GOOGL",
                key="screener_custom",
            )
            tickers = list(dict.fromkeys(
                t.strip().upper() for t in custom_input.split(",") if t.strip()
            ))

            # Save watchlist controls
            if tickers:
                save_col, _ = st.columns([2, 1])
                with save_col:
                    wl_name = st.text_input("Watchlist name", key="screener_save_name")
                if wl_name and st.button("Save Watchlist"):
                    save_watchlist(wl_name, tickers)
                    st.success(f"Saved '{wl_name}' with {len(tickers)} tickers.")
                    st.rerun()
        elif is_user_watchlist:
            tickers = user_watchlists.get(raw_name, [])
            if st.button("Delete Watchlist"):
                delete_watchlist(raw_name)
                st.success(f"Deleted '{raw_name}'.")
                st.rerun()
        else:
            tickers = WATCHLIST_PRESETS[preset_name]

        if tickers:
            st.caption(f"{len(tickers)} tickers: {', '.join(tickers)}")

        horizon = horizon_input(key="screener_horizon")
        selected_indicators = indicator_picker(
            key="screener_indicators", timesfm_use_case="watchlist"
        )

        scan_clicked = st.button("Scan Watchlist", type="primary", use_container_width=True)

    if not tickers:
        st.info("Select a watchlist or enter custom tickers in the sidebar.")
        return

    if not selected_indicators:
        st.warning("Select at least one indicator.")
        return

    # Run scan
    if scan_clicked:
        all_indicators = get_all_indicators()
        chosen = {n: all_indicators[n] for n in selected_indicators if n in all_indicators}

        results = []
        failures = []
        progress = st.progress(0, text="Scanning...")

        for i, ticker in enumerate(tickers):
            progress.progress((i + 1) / len(tickers), text=f"Scanning {ticker}...")
            try:
                df = fetch_ohlcv(ticker, period="6mo")
                computed = df.copy()
                for indicator in chosen.values():
                    computed = indicator.compute_for_horizon(computed, horizon)
                signal = combine_signals(chosen, computed, horizon_days=horizon, precomputed=True)

                info = get_asset_info(ticker)
                name = info["name"] if info else ticker
                price = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2] if len(df) > 1 else price
                change_pct = (price - prev_close) / prev_close * 100

                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": price,
                    "change_pct": change_pct,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                    "scores": signal.scores,
                })
            except Exception as error:
                failures.append(f"{ticker}: {error}")

        progress.empty()

        if not results:
            st.warning("No results. All tickers failed to fetch or compute.")
            return

        # Sort by confidence descending
        results.sort(key=lambda r: r["confidence"], reverse=True)
        st.session_state["screener_results"] = results
        st.session_state["screener_failures"] = failures
        st.session_state["screener_params"] = {
            "tickers": tickers,
            "horizon": horizon,
            "indicators": selected_indicators,
        }

    # Display results
    current_params = {
        "tickers": tickers,
        "horizon": horizon,
        "indicators": selected_indicators,
    }
    results = st.session_state.get("screener_results")
    if st.session_state.get("screener_params") != current_params:
        results = None
    if not results:
        st.info("Click **Scan Watchlist** to analyze tickers.")
        return

    failures = st.session_state.get("screener_failures", [])
    if failures:
        with st.expander(f"{len(failures)} ticker(s) could not be scanned"):
            for failure in failures:
                st.write(failure)

    st.markdown(f"**{len(results)} results** — sorted by directional agreement")

    header_cols = st.columns([2, 2, 1.5, 1, 1])
    for column, label in zip(
        header_cols,
        ["Ticker", "Price / Change", "Signal", "Agreement", "Action"],
    ):
        column.caption(label)

    for i, r in enumerate(results):
        color = _signal_color(r["direction"])
        emoji = _signal_emoji(r["direction"])
        change_color = "#4CAF50" if r["change_pct"] >= 0 else "#F44336"

        col_ticker, col_price, col_signal, col_conf, col_action = st.columns(
            [2, 2, 1.5, 1, 1]
        )

        with col_ticker:
            st.markdown(f"**{r['ticker']}**  \n{r['name']}")

        with col_price:
            st.markdown(
                f"${r['price']:,.2f}  \n"
                f"<span style='color:{change_color}'>{r['change_pct']:+.2f}%</span>",
                unsafe_allow_html=True,
            )

        with col_signal:
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>"
                f"{emoji} {r['direction'].value}</span>",
                unsafe_allow_html=True,
            )

        with col_conf:
            st.markdown(f"{r['confidence']:.0%}")

        with col_action:

            def _go_predict(ticker=r["ticker"]):
                st.session_state["predict_ticker"] = ticker
                st.session_state["nav_page"] = "Predict"

            st.button(
                "View",
                key=f"screener_view_{i}",
                on_click=_go_predict,
            )

        with st.expander(f"Reasoning — {r['ticker']}", expanded=False):
            st.write(r["reasoning"])
            scores = r["scores"]
            sc1, sc2 = st.columns(2)
            sc1.metric("BUY", f"{scores.get('BUY', 0):.2f}")
            sc2.metric("SELL", f"{scores.get('SELL', 0):.2f}")

        if i < len(results) - 1:
            st.divider()

    # CSV export for screener results
    import pandas as pd

    csv_rows = []
    for r in results:
        csv_rows.append({
            "Ticker": r["ticker"],
            "Name": r["name"],
            "Price": r["price"],
            "Change %": r["change_pct"],
            "Signal": r["direction"].value,
            "Directional Agreement": r["confidence"],
            "BUY Score": r["scores"].get("BUY", 0),
            "SELL Score": r["scores"].get("SELL", 0),
            "Reasoning": r["reasoning"],
        })
    csv_string = pd.DataFrame(csv_rows).to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv_string,
        file_name=f"screener_{preset_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
