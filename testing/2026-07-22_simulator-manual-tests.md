# Manual Tests — Ludofinancial Simulator page (CMAN-D1-020)

Feature merged from upstream in CMAN-D1-019 (`9ac7421` + merge `fe61b3e`).
Covers the new **Simulator** nav page: setup sidebar, replay controls, goals,
strategy switching, charts, and event ledger.

## Prerequisites

- [ ] App launches (`./run-timesfm.sh` or `venv/bin/streamlit run app.py`)
- [ ] Internet access available (simulator fetches full daily history via yfinance)
- [ ] A liquid ticker for most tests (e.g. `AAPL`), a delisted/invalid one for error tests

## Setup sidebar

- [ ] "Simulator" appears in the sidebar nav between Backtest and Explore; selecting it shows the setup sidebar and the "Choose a ticker…" info box with MVP boundaries text
- [ ] Create replay button is disabled when the ticker field is empty
- [ ] Recent-tickers row appears under the ticker input and clicking one fills it in
- [ ] Drop-in date defaults to one year ago
- [ ] Starting capital rejects values below $100 (widget floor) and defaults to $10,000
- [ ] Cost per fill defaults to 0.05% and is capped at 5% in the widget
- [ ] Strategy dropdown lists: Buy & Hold, Balanced Technical, Trend Following, Mean Reversion, Custom
- [ ] Selecting a preset shows its caption (Buy & Hold: "Invest at the first eligible open…"; others: indicator list + horizon)
- [ ] Selecting Custom shows: indicator multiselect (9 causal indicators only — no TimesFM/FRED/cross-asset), horizon slider 1–30, ambiguity slider 0–0.5, and a weights expander with one number input (0.1–5.0) per selected indicator
- [ ] Deselecting all Custom indicators disables Create replay
- [ ] "Run until" offers: Latest available bar / Specific date / Duration; Specific date cannot be set before the drop-in date; Duration accepts 1–10,000 calendar days

## Creating a replay

- [ ] Happy path: AAPL, defaults, Buy & Hold, Create replay → spinner, then replay view renders with status Ready and "Not started" date
- [ ] Invalid ticker → readable error message, no replay created, no stack trace
- [ ] Drop-in date in the future (after last available bar) → error "Start date is after the available market data.", no replay created
- [ ] Drop-in date before the ticker's first bar (e.g. 1950) → replay clamps to earliest usable data or errors readably — no crash
- [ ] Creating a second replay while one exists replaces it cleanly

## Replay view and metrics

- [ ] Five metrics render: Portfolio (with % delta), Profit/Loss, Cash, Position, Fees
- [ ] Before first step: Portfolio = starting capital, P/L = $0.00, Position = 0, Fees = $0.00
- [ ] Status line shows Status / Date / Strategy / Pending next-open order
- [ ] Resolved date range caption matches the requested drop-in date and endpoint
- [ ] Bar-reveal progress bar and "Revealed N of M market bars" count advance with steps
- [ ] Market chart shows ~90 bars of pre-drop-in context and only revealed bars — scroll forward and confirm no future bars are visible (anti-lookahead)
- [ ] Equity curve appears once at least 2 equity points exist

## Playback controls

- [ ] Step one bar advances exactly one bar (revealed count +1)
- [ ] Batch size slider offers 1/2/5/10/25/50/100, default 5
- [ ] Advance and Pause are disabled until Resume batch mode is clicked; after Resume, Advance N advances exactly N bars and Pause re-disables it
- [ ] Determinism: two fresh identical replays, one advanced 100 bars in one batch, one stepped 100× → identical Portfolio, P/L, Fees, and event ledger
- [ ] At the endpoint the replay goes terminal: all playback buttons disable and a completion banner shows (info for latest-data/end-date)

## Trading semantics and calculations

- [ ] Buy & Hold: one BUY fills at the *second* bar's open (decision at close, fill at next open), then no further orders
- [ ] BUY/SELL markers (green up / red down triangles) appear on the chart at fill dates and prices
- [ ] Each fill charges the configured cost once — after one buy + one sell, Fees ≈ 2 × cost% × traded value; with cost 0%, Fees stays $0.00
- [ ] Spot-check accounting after first BUY: Cash + Position×Close = Portfolio metric (marked-to-market)
- [ ] Long-or-cash only: Position never goes negative, no margin (Cash never negative)
- [ ] Event ledger lists Created, Status Changed, Order Queued, Order Filled events in sequence with dates and details; fills match chart markers

## Goals

- [ ] "No profit goal" → no goal progress bar
- [ ] Profit ($) goal with "Stop when reached" checked: replay completes with success banner "Goal Reached" when profit crosses the target (pick a small target, e.g. $100 on a bull run)
- [ ] Goal progress bar and percentage update as equity rises, clamped at 0–100%
- [ ] With "Stop when reached" unchecked: goal-reached event logs in the ledger but the replay continues to the endpoint
- [ ] Return (%) goal behaves the same, using return instead of dollars
- [ ] Replay ending without reaching the goal → warning banner "The selected profit goal was not reached."

## Strategy switching

- [ ] "Apply selected sidebar strategy" is disabled while status is Running and after the replay is terminal
- [ ] While paused: select a different strategy in the sidebar, apply → status line shows the new strategy name, Strategy Changed event in the ledger, current holding is kept, pending order (if any) is cancelled
- [ ] Continuing after the switch produces signals from the new strategy (e.g. Buy & Hold → Mean Reversion starts emitting sells)

## Persistence and navigation

- [ ] Navigate to another page (e.g. Predict) and back to Simulator → replay state (bar position, portfolio, ledger) is intact
- [ ] Cross-page "View"/"Analyze" navigation from other pages still works after the merge (nav_request routing, CMAN-D1-006)
- [ ] Reset replay clears the run and returns to the setup info box
- [ ] Browser refresh clears the replay (expected: session-state only, no disk persistence in the MVP)

## Edge cases

- [ ] Duration endpoint of 1 day → tiny replay completes immediately or after one step without errors
- [ ] Drop-in date on a weekend/holiday → start aligns to the next trading day without error
- [ ] Very volatile/cheap ticker with 5% cost per fill and an active strategy (Mean Reversion) → fees accumulate; if equity is wiped out, capital-depleted error banner shows and replay is terminal
- [ ] Custom strategy with a single indicator and weight 5.0 → creates and runs without error
- [ ] Ambiguity threshold at max (0.5) → strategy mostly HOLDs; no orders is a valid outcome, no errors
- [ ] Ticker with a stock split inside the replay window (e.g. AAPL 2020) → no discontinuity spikes in the equity curve at the split date

## Notes

- Continuous timed autoplay is intentionally deferred upstream; batch advance is the supported fast-forward.
- TimesFM, FRED, and cross-asset indicators are deliberately excluded from the simulator (non-reproducible historical inputs) — their absence from the Custom picker is correct behavior, not a bug.
- Record failures as CMAN tickets referencing the checklist item.
