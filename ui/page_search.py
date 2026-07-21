"""Search page: discover tickers via Yahoo Finance search."""

import streamlit as st

from data.fetcher import search_tickers


def render():
    st.header("Search")
    st.caption("Find ticker symbols by company name, keyword, or symbol.")

    query = st.text_input(
        "Search tickers",
        placeholder="e.g. Apple, Tesla, Bitcoin...",
        key="search_query",
    )
    search_clicked = st.button("Search", type="primary", key="search_submit")

    if not query:
        # Show recent searches if available
        recents = st.session_state.get("recent_searches", [])
        if recents:
            st.markdown("**Recent searches:**")
            cols = st.columns(min(len(recents), 4))
            for i, q in enumerate(recents[:4]):

                def _set_query(val=q):
                    st.session_state["search_query"] = val

                cols[i].button(q, key=f"recent_search_{i}", on_click=_set_query)
        else:
            st.info("Type a company name or symbol above to search.")
        return

    if search_clicked:
        st.session_state["submitted_search_query"] = query
    if st.session_state.get("submitted_search_query") != query:
        st.info("Click **Search** to find matching tickers.")
        return

    # Track recent searches
    recents = st.session_state.get("recent_searches", [])
    if query not in recents:
        recents.insert(0, query)
        st.session_state["recent_searches"] = recents[:8]

    with st.spinner("Searching..."):
        results = search_tickers(query)

    if results is None:
        st.error(
            "Search is temporarily unavailable (network or rate limit). "
            "Try again in a minute."
        )
        return
    if not results:
        st.warning(f"No results found for '{query}'.")
        return

    st.markdown(f"**{len(results)} result(s)**")

    for i, r in enumerate(results):
        symbol = r["symbol"]
        name = r["name"] or symbol
        exchange = r["exchange"]
        qtype = r["type"]

        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{symbol}** — {name}"
                + (f"  \n`{exchange}` | {qtype}" if exchange or qtype else "")
            )
        with col_btn:

            def _analyze(sym=symbol):
                st.session_state["predict_ticker"] = sym
                st.session_state["nav_request"] = "Predict"

            st.button("Analyze", key=f"analyze_{i}_{symbol}", on_click=_analyze)

        if i < len(results) - 1:
            st.divider()
