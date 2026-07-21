"""Capitalisman: Price Prediction & Backtesting App."""

import streamlit as st

st.set_page_config(
    page_title="Capitalisman",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Importing indicators triggers auto-registration via indicators/__init__.py
import indicators  # noqa: F401, E402

from ui import page_predict, page_backtest, page_simulator, page_explore  # noqa: E402
from ui import page_search, page_compare, page_screener  # noqa: E402

# Sidebar navigation
st.sidebar.title("Capitalisman")
page = st.sidebar.radio(
    "Navigation",
    ["Predict", "Backtest", "Simulator", "Explore", "Search", "Compare", "Screener"],
    key="nav_page",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

if page == "Predict":
    page_predict.render()
elif page == "Backtest":
    page_backtest.render()
elif page == "Simulator":
    page_simulator.render()
elif page == "Explore":
    page_explore.render()
elif page == "Search":
    page_search.render()
elif page == "Compare":
    page_compare.render()
elif page == "Screener":
    page_screener.render()
