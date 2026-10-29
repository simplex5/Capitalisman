# How Capitalisman works

A plain-language tour of what the app does, how it reaches a BUY / SELL / HOLD call, and what
TimesFM (the AI forecaster) is. Every result is a research call; the app never places a trade.

## 1. Prices in

For any stock or crypto ticker, the app downloads daily price and volume history from Yahoo
Finance (the `yfinance` library). Some indicators also read other assets (copper, gold, VIX) or
US economic data (FRED).

## 2. Sixteen indicators each cast a vote

Each indicator looks at the chart in its own way and votes BUY, SELL or HOLD, with a strength
from 0 to 1.

| Group | Indicators | What it asks |
|---|---|---|
| Trend | SMA Crossover, EMA Crossover, MACD, ADX | Is the price heading somewhere? |
| Momentum | RSI, Stochastic | Is the move speeding up or running out of steam? |
| Volume | OBV, VWAP | Is real money behind the move? |
| Volatility and structure | Bollinger Bands, VPIN, Bubble Risk | Is the price stretched, or is trading turning one-sided? |
| The wider market | FRED Macro, Copper-Gold Ratio, VIX Term Structure, Market Correlation | What do the economy and other assets say? |
| AI forecast | TimesFM Forecast | What does a pretrained forecasting model expect next? |

## 3. The combiner counts the votes

`signals/combiner.py` weighs every vote and reports two numbers:

- **Agreement:** how much the indicators that voted agree on one direction.
- **Evidence:** how much of the total possible weight actually voted.

When evidence is under 15%, or BUY and SELL are too close, the answer is HOLD. One weak vote
among many silent indicators can never become a confident call. Both numbers are rules of
thumb, not probabilities of making money.

## 4. The pages

- **Predict:** today's call for one ticker, with each indicator's vote.
- **Backtest:** replays the strategy over past years. A signal on day `t`'s close enters at day
  `t+1`'s open and exits N days later; fees are counted. It reports return, win rate, Sharpe
  ratio and profit factor, all from the account's value day by day.
- **Simulator:** steps through history one day at a time, so you can pause and switch strategy.
- **Screener:** runs the call over a whole watchlist.
- **Explore / Search / Compare:** look at charts, find tickers, compare assets side by side.

## 5. The rule that keeps backtests honest

A backtest may only use data that existed on each day. An indicator must prove this before it
is allowed into backtests (`historical_safe`). FRED Macro, Copper-Gold Ratio, VIX Term Structure
and Market Correlation have not proved it yet, so backtests leave them out.

## 6. What TimesFM is

TimesFM is Google Research's pretrained forecasting model for any series of numbers over time.
Think of a language model, but instead of predicting the next word it predicts the next values
of a series. It was trained on a large mix of real time series, so it forecasts prices without
being trained on your data ("zero-shot").

### How the app turns its forecast into a vote

1. TimesFM reads the ticker's past closing prices (up to the context length, e.g. 8,192 days)
   and forecasts a *range* for N days ahead: from a low case (10th percentile, "q10") through
   the middle to a high case (90th percentile, "q90").
2. From that range the app works out the expected return and a **profit score**: roughly, how
   much of the forecast range ends above the entry price after fees.
3. BUY when the profit score is at least 60% and the expected return beats the round-trip fees;
   SELL in the mirror case; otherwise HOLD. That is one vote among the sixteen.

### Caveats

- The profit score is not a checked probability: nobody has measured that "60%" comes true 60%
  of the time (ticket CMAN-D1-016).
- Whether TimesFM beats a naive guess on stocks is unproven (CMAN-D1-014, the benchmark sweep);
  fine-tuning (CMAN-D1-017) waits on that answer.

## 7. TimesFM 2.5 and TimesFM 3

| | TimesFM 2.5 (default) | TimesFM 3.0 (experimental) |
|---|---|---|
| Model | `google/timesfm-2.5-200m-pytorch` | `google/timesfm-3.0-pytorch` |
| Parameters | 231 million | 331 million (about 43% more) |
| Weights on disk | 0.93 GB | 1.32 GB |
| Longest context | set by the runtime profile | capped at 15,360 bars |
| License | Apache-2.0 | non-commercial, non-production use only |

Both fit easily on this machine's RTX 3090 (24 GB). In the app, both read closing prices only.

## 8. Running it on this machine

| Environment | Python | What it runs |
|---|---|---|
| `venv/` (in the repo) | 3.12 | The app without the AI vote; the test suite |
| `~/.venvs/capitalisman-timesfm` | 3.12 | The app with TimesFM 2.5 (timesfm 2.0.2) |
| `~/.venvs/capitalisman-timesfm3` | 3.11 | The app with TimesFM 2.5 **and** 3.0 (timesfm 3.0.2) |

All three use the GPU through PyTorch 2.11 built for CUDA 12.8.

- With TimesFM 2.5:
  ```bash
  cd ~/projects/Capitalisman && ./run-timesfm.sh
  ```
- With TimesFM 2.5 and 3.0 to choose from:
  ```bash
  cd ~/projects/Capitalisman && TIMESFM_ENV=~/.venvs/capitalisman-timesfm3 ./run-timesfm.sh
  ```
  On the Predict page, tick **TimesFM Forecast**, then pick **TimesFM 3.0 (experimental)** in
  the **TimesFM Model** box. The **TimesFM Historical Context** box then sets how much history
  the model reads.
- Without the AI vote:
  ```bash
  cd ~/projects/Capitalisman && venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
  ```

The TimesFM 3 environment keeps Streamlit at 1.59.2: Streamlit 1.66 changes how the UI tests
find `app.py`, so 8 of them fail there (the app itself runs on either version).
