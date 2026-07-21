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

from ui import page_predict, page_backtest, page_explore  # noqa: E402
from ui import page_search, page_compare, page_screener  # noqa: E402

# Sidebar navigation
st.sidebar.title("Capitalisman")

# Cross-page "View"/"Analyze" buttons request navigation via nav_request; the
# widget key itself may only be written before the radio is instantiated
# (writing it from a callback raises StreamlitAPIException on newer versions).
if "nav_request" in st.session_state:
    st.session_state["nav_page"] = st.session_state.pop("nav_request")

page = st.sidebar.radio(
    "Navigation",
    ["Predict", "Backtest", "Explore", "Search", "Compare", "Screener"],
    key="nav_page",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

if page == "Predict":
    page_predict.render()
elif page == "Backtest":
    page_backtest.render()
elif page == "Explore":
    page_explore.render()
elif page == "Search":
    page_search.render()
elif page == "Compare":
    page_compare.render()
elif page == "Screener":
    page_screener.render()
