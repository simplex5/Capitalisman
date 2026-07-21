"""Shared UI widgets for sidebar controls."""

import importlib.util
import os

import streamlit as st

from config.settings import (
    DEFAULT_BACKTEST_PERIOD,
    DEFAULT_COST_PER_TRADE_PCT,
    DEFAULT_INDICATORS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_PERIOD,
    DEFAULT_PREDICTION_HORIZON,
    INDICATOR_WEIGHTS,
    INTRADAY_WARNINGS,
    INTERVALS,
    PERIOD_LABELS,
    PERIODS,
    TUNABLE_THRESHOLDS,
)
from config.overrides import clear_overrides, get_setting, set_override
from indicators.registry import get_all_indicators
from ml.timesfm_profiles import (
    HardwareCapabilities,
    PROFILE_PREFERENCES,
    detect_hardware,
    select_runtime_profile,
)
from ml.timesfm_runtime import TimesFMRuntimeConfig, get_timesfm_runtime


def ticker_input(key: str = "ticker") -> str:
    """Render ticker input and return the value."""
    return st.text_input(
        "Ticker Symbol",
        value="AAPL",
        key=key,
        help="Enter a stock (AAPL) or crypto (BTC-USD) ticker",
    ).upper().strip()


def period_select(key: str = "period", default: str = DEFAULT_PERIOD) -> str:
    """Render period selector."""
    idx = PERIODS.index(default) if default in PERIODS else 3
    return st.selectbox(
        "Period",
        PERIODS,
        index=idx,
        key=key,
        format_func=lambda p: PERIOD_LABELS.get(p, p),
    )


def interval_select(key: str = "interval") -> str:
    """Render interval selector."""
    return st.selectbox("Interval", INTERVALS, index=4, key=key)  # default "1d"


def horizon_input(key: str = "horizon") -> int:
    """Render prediction horizon slider."""
    return st.slider(
        "Signal Horizon (daily bars)",
        min_value=1,
        max_value=30,
        value=DEFAULT_PREDICTION_HORIZON,
        key=key,
        help="Adjusts indicator weights by horizon. In backtests, stocks skip non-trading days.",
    )


def indicator_picker(
    key: str = "indicators", timesfm_use_case: str = "interactive"
) -> list[str]:
    """Render indicator multi-select and return chosen names."""
    all_indicators = get_all_indicators()
    names = sorted(all_indicators.keys())
    defaults = [name for name in DEFAULT_INDICATORS if name in all_indicators]
    selected = st.multiselect(
        "Indicators",
        options=names,
        default=defaults,
        key=key,
        help="Core technical indicators are selected by default. Add experimental or optional signals as needed.",
    )
    optional_status = []
    if "FRED Macro" in selected and not os.environ.get("FRED_API_KEY"):
        optional_status.append("FRED Macro needs FRED_API_KEY")
    if "TimesFM Forecast" in selected:
        if importlib.util.find_spec("timesfm") is None:
            optional_status.append("TimesFM Forecast needs TimesFM 2.5 + PyTorch")
        else:
            profile_choice = st.selectbox(
                "TimesFM Runtime Profile",
                PROFILE_PREFERENCES,
                index=0,
                key=f"{key}_timesfm_profile",
                format_func=str.title,
                help=(
                    "Controls hardware utilization and workload size. It does not "
                    "change trading thresholds or claim a more accurate forecast."
                ),
            )
            hardware = detect_hardware()
            if os.getenv("CAPITALISMAN_TIMESFM_DEVICE", "auto").lower() == "cpu":
                hardware = HardwareCapabilities(
                    "cpu", available_ram_gb=hardware.available_ram_gb
                )
            profile = select_runtime_profile(
                hardware, use_case=timesfm_use_case, preference=profile_choice
            )
            runtime = get_timesfm_runtime(TimesFMRuntimeConfig.from_profile(profile))
            runtime_status = runtime.status
            if runtime_status.state not in {"ready", "loaded"}:
                optional_status.append(f"TimesFM: {runtime_status.message}")
            else:
                st.info(
                    f"**TimesFM {profile.preference.title()} profile** — "
                    f"{runtime_status.resolved_device.upper()}, "
                    f"{profile.forecast_context:,}-bar context, batch "
                    f"{profile.batch_size}, chunk {profile.chunk_size}. "
                    f"Use case: {profile.use_case}."
                )
    if optional_status:
        st.caption("Unavailable: " + "; ".join(optional_status))
    return selected


def check_data_sufficiency(
    df_len: int,
    selected_indicators: list[str],
) -> list[str]:
    """Check which selected indicators lack sufficient data.

    Returns a list of warning strings for indicators that need more bars
    than the dataframe provides. Returns empty list if all are fine.
    """
    all_indicators = get_all_indicators()
    warnings = []
    for name in selected_indicators:
        if name not in all_indicators:
            continue
        indicator = all_indicators[name]
        if df_len < indicator.lookback:
            warnings.append(
                f"**{name}** needs {indicator.lookback} bars, got {df_len}"
            )
    return warnings


def analysis_settings_signature() -> tuple:
    """Return a stable signature for settings that affect signal output."""
    weights = tuple(sorted(get_setting("INDICATOR_WEIGHTS").items()))
    thresholds = tuple(
        (name, get_setting(name)) for name in sorted(TUNABLE_THRESHOLDS)
    )
    runtime = get_timesfm_runtime()
    timesfm = (
        runtime.config.profile_name,
        runtime.config.use_case,
        runtime.config.max_context,
        runtime.config.effective_context,
        runtime.config.batch_size,
        runtime.config.chunk_size,
        runtime.config.memory_target_fraction,
    )
    return weights, thresholds, timesfm


def capital_input(key: str = "capital") -> float:
    """Render initial capital input."""
    return st.number_input(
        "Initial Capital ($)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=DEFAULT_INITIAL_CAPITAL,
        step=1000.0,
        key=key,
    )


def cost_input(key: str = "cost") -> float:
    """Render transaction cost input (round-trip %)."""
    return st.number_input(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_COST_PER_TRADE_PCT,
        step=0.05,
        key=key,
        help="Round-trip cost per trade (slippage + commission) as a percentage",
    )


def show_interval_warning(interval: str) -> None:
    """Show a warning if the selected interval has data limits."""
    warning = INTRADAY_WARNINGS.get(interval)
    if warning:
        st.caption(f"⚠ {warning}")


_MAX_RECENT = 8


def record_recent_ticker(ticker: str) -> None:
    """Add a ticker to the recent tickers list in session state."""
    if not ticker:
        return
    if "recent_tickers" not in st.session_state:
        st.session_state["recent_tickers"] = []
    recents = st.session_state["recent_tickers"]
    # Move to front if already present, otherwise prepend
    if ticker in recents:
        recents.remove(ticker)
    recents.insert(0, ticker)
    st.session_state["recent_tickers"] = recents[:_MAX_RECENT]


def render_recent_tickers(input_key: str) -> None:
    """Render quick-select buttons for recent tickers.

    When clicked, sets the value of the ticker input identified by input_key.
    """
    recents = st.session_state.get("recent_tickers", [])
    # Don't show the ticker that's currently in the input
    current = st.session_state.get(input_key, "")
    visible = [t for t in recents if t != current]
    if not visible:
        return
    st.caption("Recent:")
    cols = st.columns(min(len(visible), 4))
    for i, ticker in enumerate(visible[:4]):

        def _select(t=ticker):
            st.session_state[input_key] = t

        cols[i].button(ticker, key=f"recent_{input_key}_{ticker}", on_click=_select)


def advanced_settings(key_prefix: str = "adv") -> None:
    """Render an Advanced Settings expander with weight and threshold sliders.

    Overrides are stored in session state via config.overrides so that
    indicators and the combiner pick them up automatically.
    """
    with st.expander("Advanced Settings"):
        # Indicator weight sliders
        st.markdown("**Indicator Weights**")
        for ind_name, default_weight in INDICATOR_WEIGHTS.items():
            slider_key = f"{key_prefix}_weight_{ind_name}"
            val = st.slider(
                ind_name,
                min_value=0.0,
                max_value=2.0,
                value=float(get_setting("INDICATOR_WEIGHTS").get(ind_name, default_weight)),
                step=0.1,
                key=slider_key,
            )
            # Store per-indicator weight overrides in a dict
            weights = dict(get_setting("INDICATOR_WEIGHTS"))
            weights[ind_name] = val
            set_override("INDICATOR_WEIGHTS", weights)

        st.markdown("**Thresholds**")
        for setting_name, meta in TUNABLE_THRESHOLDS.items():
            slider_key = f"{key_prefix}_thresh_{setting_name}"
            current = get_setting(setting_name)
            val = st.slider(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=type(meta["min"])(current),
                step=meta["step"],
                key=slider_key,
            )
            set_override(setting_name, val)

        if st.button("Reset to Defaults", key=f"{key_prefix}_reset"):
            clear_overrides()
            widget_prefixes = (
                f"{key_prefix}_weight_",
                f"{key_prefix}_thresh_",
            )
            for state_key in list(st.session_state):
                if state_key.startswith(widget_prefixes):
                    del st.session_state[state_key]
            st.rerun()
