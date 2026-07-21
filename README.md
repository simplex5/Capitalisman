# Capitalisman

<p align="center">
  <img
    src="Capitalisman.png"
    alt="Capitalisman pixel-art logo: a financial divination wheel surrounded by lunar phases"
    title="Capitalisman"
    width="400"
  >
</p>

A personal stock and crypto research tool that analyzes price data using classical technical indicators **and novel cross-asset, structural, and microstructure signals**, combines them into a BUY/SELL/HOLD research signal with a directional-agreement score, and lets you evaluate those signals on historical data through a web dashboard.

No trading experience required to get started. The core tool works with no API keys, but optional integrations (FRED economic data, TimesFM ML forecasting) unlock additional indicators.

## What It Does

### Predict — Get a Signal for Any Stock or Crypto

Pick a ticker (like `AAPL` for Apple, or `BTC-USD` for Bitcoin) and a time horizon (how many days ahead you want to predict). Nine core technical indicators are selected by default, while experimental cross-asset indicators and optional FRED and TimesFM integrations can be enabled as needed.

- **BUY** — indicators suggest the price is likely to go up
- **SELL** — indicators suggest the price is likely to go down
- **HOLD** — signals are mixed or too close to call

Each prediction comes with a **directional-agreement score** (0–100%) showing how much of the actionable weighted vote supports the winning direction. It is not a calibrated probability that the prediction will be correct.

**Multi-Timeframe Signals** — Above the primary signal card, three compact cards show the signal for 1-day, 5-day, and 20-day horizons simultaneously, so you can see whether short-term and long-term outlooks agree at a glance.

### Backtest — Test How Well Predictions Would Have Worked

Before trusting any strategy, you want to know: "How accurate would this have been in the past?" The backtest page answers that question by running predictions across historical data and measuring the results.

You'll see:

- **Win Rate** — what percentage of modeled trades were profitable after costs
- **Total Return** — how much money you would have made or lost
- **Max Drawdown** — the worst peak-to-valley decline (how much pain you'd have endured)
- **Sharpe Ratio** — return relative to risk (higher is better; above 1.0 is generally good)
- **Profit Factor** — total gains divided by total losses (above 1.0 means profitable)
- **Equity Curve** — a chart showing your portfolio value over time
- **Trade Log** — every individual trade with entry/exit prices and profit/loss
- **CSV Export** — download the full trade log as a CSV file for further analysis in Excel or Google Sheets

Backtests include **configurable transaction costs** (slippage and commissions, default 0.1% per trade). Signals are evaluated at a bar's close, entered at the next bar's open, and exited at the close after the selected number of bars. Revised FRED series are excluded because the integration does not provide point-in-time vintages. Short positions are modeled without borrow costs.

### Search — Find Any Ticker

Don't know the exact Yahoo Finance symbol? The Search page lets you type a company name, keyword, or partial symbol and browse matching results. Each result shows the symbol, full name, exchange, and asset type. Click **Analyze** on any result to jump straight to the Predict page with that ticker loaded.

### Compare — Side-by-Side Analysis

Pick two tickers and compare them head-to-head. The Compare page shows:

- **Overview metrics** — name, sector, current price, and period return for each ticker
- **Normalized price chart** — both tickers rebased to 100 so you can see relative performance regardless of absolute price
- **Signal comparison** — BUY/SELL/HOLD signal cards for both tickers side by side
- **Correlation** — daily return correlation to see how closely the two assets move together

### Screener — Scan Multiple Tickers at Once

Select a preset watchlist (Tech Giants, S&P 500 Top 10, Major Crypto, Indices) or enter your own comma-separated list of tickers. Hit **Scan Watchlist** and the tool runs a full signal analysis on every ticker, then ranks the results by directional agreement. Each result shows price, daily change, signal direction, agreement, and expandable reasoning. Click **View** to jump to the Predict page for any ticker.

**Persistent Watchlists** — When using a custom ticker list, you can save it as a named watchlist. Saved watchlists appear in the dropdown with a "(saved)" suffix and persist across sessions (stored in `~/.capitalisman/watchlists.json`). You can delete user-created watchlists at any time; built-in presets cannot be deleted.

**CSV Export** — Download the full screener results (ticker, name, price, change, signal, agreement, scores, reasoning) as a CSV file.

### Explore — Browse Charts and Data

An interactive charting page where you can look at any stock or crypto with professional-style candlestick charts. Toggle different indicator overlays on and off, view company information, and inspect the raw price data.

## Getting Started

### Installation

You'll need Python 3.10 or newer installed on your computer.

```bash
git clone <repository-url>
cd Capitalisman
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

This opens the dashboard in your web browser at `http://localhost:8501`. Use the sidebar on the left to switch between pages and adjust settings.

### Optional: Enable FRED Economic Data

The FRED Macro indicator pulls yield curve, jobless claims, and fed funds rate data from the Federal Reserve. It requires a free API key:

1. Sign up at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Set the key as an environment variable before running the app:

```bash
export FRED_API_KEY="your_key_here"
streamlit run app.py
```

Without the key, the FRED Macro indicator gracefully returns HOLD with zero confidence — all other indicators work normally.

Install the optional integration first with `pip install -r requirements-optional.txt`.

### Optional: Enable TimesFM ML Forecasting

The TimesFM Forecast indicator uses Google's TimesFM 2.5 model for zero-shot,
probabilistic time-series prediction. Keep its PyTorch/CUDA dependencies isolated
from your base Python installation:

```bash
conda env create -f environment-timesfm.yml
conda run -n capitalisman-timesfm python -m pip install \
  torch --index-url https://download.pytorch.org/whl/cu128
conda run -n capitalisman-timesfm python -m pip install -r requirements-timesfm.txt
```

Use the appropriate PyTorch CUDA wheel for your GPU and driver. On first use,
model weights are downloaded from Hugging Face and cached locally. Without a
compatible runtime, the indicator returns HOLD and displays the precise preflight
error; all other indicators continue to work. See [`docs/TIMESFM.md`](docs/TIMESFM.md)
for health checks, configuration, interpretation, and point-in-time benchmarking.

#### Current TimesFM Status

- Uses the official TimesFM 2.5 PyTorch API with lazy model loading.
- Detects CUDA availability, package compatibility, memory, and disk space before
  attempting inference.
- Selects hardware-aware Fast, Balanced, and Thorough runtime profiles from free
  VRAM and workload type, while keeping performance settings separate from
  predictive methodology.
- Includes an opt-in empirical throughput probe for choosing a safe workload
  chunk size on the installed GPU and software stack.
- Produces horizon-aware point and quantile forecasts rather than a single fixed
  10-day estimate.
- Displays median return, q10/q90 range, probability of an upward move, and
  probability of clearing configured transaction costs.
- Includes a rolling point-in-time benchmark against a naïve last-price forecast,
  with directional, calibration, Brier, MAE, and pinball metrics.
- Has been smoke-tested with CUDA 12.8 on an NVIDIA RTX 5090 from the isolated
  `capitalisman-timesfm` environment.

TimesFM remains an experimental research input. A successful forecast does not
establish predictive edge, and the model should not receive greater signal weight
until it demonstrates consistent out-of-sample improvement across assets, market
regimes, and horizons. Planned calibration, baseline, regime, and ensemble work
is tracked in [`ROADMAP.md`](ROADMAP.md).

To see an unambiguous TimesFM result in the app:

1. Activate `capitalisman-timesfm` and launch Streamlit from that environment.
2. Open **Predict** and add **TimesFM Forecast** to the Indicators selection.
3. Confirm the sidebar says **TimesFM mode selected** and identifies CUDA or CPU.
4. Choose a ticker and horizon, then click **Run TimesFM Forecast**.
5. Read the dedicated **TimesFM Model Forecast** panel at the top of the results.
   Its direction, probabilities, interval, model, and device are model-only. The
   separately labeled **Combined Indicator Context** below is the ensemble result.

If the model cannot run, the dedicated panel remains visible and reports the
failure instead of disappearing or presenting the ensemble HOLD as a model result.

### Quick Start

1. **Try a prediction** — The app starts on the Predict page. Type `AAPL`, set the horizon to 5 days, and click **Analyze**. You'll see a BUY/SELL/HOLD signal with directional agreement and a detailed chart.

2. **Run a backtest** — Switch to Backtest, type `AAPL`, pick "1y", keep the 5-bar horizon, and click **Run Backtest**. Warmup history is fetched separately so the reported evaluation remains bounded to the selected period.

3. **Search for a ticker** — Switch to the Search page. Type "Tesla" and you'll see matching results. Click **Analyze** on `TSLA` to jump to the Predict page with it loaded.

4. **Compare two tickers** — Switch to the Compare page. The defaults are AAPL and MSFT. You'll see a normalized price chart, side-by-side signals, and their correlation.

5. **Screen a watchlist** — Switch to the Screener page. Select "Tech Giants" from the dropdown and click **Scan Watchlist**. You'll get a ranked table of all 7 tickers sorted by directional agreement.

6. **Explore a chart** — Switch to the Explore page. Try `BTC-USD` (Bitcoin) and toggle different indicators on and off to see how they overlay on the price chart.

### What Tickers Can I Use?

Anything available on Yahoo Finance:

- **Stocks** — `AAPL` (Apple), `GOOGL` (Google), `MSFT` (Microsoft), `TSLA` (Tesla), etc.
- **ETFs** — `SPY` (S&P 500 ETF), `QQQ` (Nasdaq 100 ETF), `VTI` (Total Market ETF), etc.
- **Crypto** — `BTC-USD` (Bitcoin), `ETH-USD` (Ethereum), `SOL-USD` (Solana), etc.
- **Indices** — `^GSPC` (S&P 500), `^DJI` (Dow Jones), etc.

## Technical Indicators

The tool uses 16 indicators across 9 categories. The first 9 are classical technical indicators that analyze the ticker's own price and volume data. The remaining 7 are novel signals that draw on cross-asset data, FRED economic data, ML-based forecasting, statistical physics, market microstructure, and systemic risk research.

### Trend Indicators — "Which direction is the price moving?"

| Indicator | What It Does |
|---|---|
| **SMA Crossover** (20/50) | Compares a short-term average (20 days) to a long-term average (50 days). When the short average crosses above the long one ("golden cross"), it's a BUY signal. The reverse ("death cross") is a SELL signal. |
| **EMA Crossover** (12/26) | Similar to SMA but uses exponential moving averages, which react faster to recent price changes. |
| **MACD** (12, 26, 9) | Measures the momentum of trend changes. A BUY signal occurs when the MACD line crosses above its signal line, suggesting upward momentum is building. |
| **ADX** (14) | Measures how strong the current trend is (regardless of direction). When the trend is strong, it looks at directional indicators (+DI/-DI) to determine whether it's bullish or bearish. |

### Momentum Indicators — "Is the price moving too fast?"

| Indicator | What It Does |
|---|---|
| **RSI** (14) | Measures whether an asset is "oversold" (below 30 — potentially undervalued, BUY signal) or "overbought" (above 70 — potentially overvalued, SELL signal). |
| **Stochastic** (14, 3, 3) | Similar to RSI but compares the closing price to its recent range. Signals occur when the fast line (%K) crosses the slow line (%D) in extreme zones (below 20 or above 80). |

### Volatility Indicators — "How wild are the price swings?"

| Indicator | What It Does |
|---|---|
| **Bollinger Bands** (20, 2) | Creates upper and lower bands around the price based on volatility. When the price touches the lower band, it may be oversold (BUY). When it touches the upper band, it may be overbought (SELL). |

### Volume Indicators — "Is the trading volume confirming the price movement?"

| Indicator | What It Does |
|---|---|
| **VWAP** (20-day rolling) | Volume-Weighted Average Price — the average price weighted by how much was traded at each level over a rolling 20-day window. Price above VWAP suggests bullish sentiment; below suggests bearish. |
| **OBV** | On-Balance Volume — tracks cumulative volume flow. If the price is falling but volume is rising (positive divergence), it may signal an upcoming reversal (BUY). |

### Macro Regime Indicators — "What is the broader economic environment signaling?"

These indicators fetch cross-asset data automatically to gauge market-wide conditions. Their signals apply to all equities.

| Indicator | What It Does |
|---|---|
| **Copper-Gold Ratio** | Divides the copper futures price (industrial demand proxy) by the gold futures price (safe-haven proxy). A rising ratio signals economic expansion (BUY); a falling ratio signals contraction and flight-to-safety (SELL). The ratio is compared against its 50-day and 200-day moving averages to determine trend direction. Research shows this ratio is a leading indicator for 3–12 month equity returns. |
| **VIX Term Structure** | Compares the near-term VIX (^VIX) to the 3-month VIX (^VIX3M). When near-term VIX exceeds the 3-month VIX (backwardation), markets are in acute stress — historically a precursor to further equity downside (SELL). Normal contango (VIX < VIX3M) signals relative calm (mild BUY). Extreme complacency (very low VIX with deep contango) is flagged as a potential reversal risk. |
| **FRED Macro** | Combines three leading economic signals from the Federal Reserve Economic Data (FRED) API into a composite macro score. **(1) Yield Curve** (T10Y2Y) — the 10-Year minus 2-Year Treasury spread, the most reliable recession predictor: an inverted (negative) curve has preceded every US recession since the 1960s. **(2) Initial Jobless Claims** (ICSA) — a 4-week vs 13-week moving average crossover detects early labor market deterioration before it shows in payrolls. **(3) Fed Funds Rate** (DFF) — the 3-month rate-of-change indicates monetary policy stance: tightening cycles precede slowdowns (SELL), easing supports risk assets (BUY). Sub-signals are weighted 50/30/20 respectively. Requires a free FRED API key (see Setup below). |

### Structural Indicators — "Is the price in a bubble?"

| Indicator | What It Does |
|---|---|
| **Bubble Risk** | Detects super-exponential growth patterns characteristic of speculative bubbles. Combines two measures: the **Hurst exponent** (persistence of returns via Rescaled Range analysis — values above 0.5 indicate trending, above 0.7 indicate potential bubble behavior) and **log-price acceleration** (positive curvature in log-price space signals faster-than-exponential growth). The composite bubble score ranges 0–1; scores above 0.35 warrant caution and above 0.6 indicate elevated bubble risk (SELL). Inspired by LPPLS (Log-Periodic Power Law Singularity) research. |

### Microstructure Indicators — "What is the smart money doing?"

| Indicator | What It Does |
|---|---|
| **VPIN** (Flow Toxicity) | Volume-Synchronized Probability of Informed Trading — measures order-flow toxicity using Bulk Volume Classification. Each bar's volume is partitioned into buy-initiated and sell-initiated components using the normalized price change within the bar. The rolling absolute imbalance between buy and sell volume is then z-scored against its own recent history. High VPIN (>2σ above mean) signals extreme informed-trading activity and reliably predicts imminent volatility spikes (SELL). Low VPIN signals calm, uninformed flow (mild BUY). Based on Easley, López de Prado & O'Hara (2012). |

### Forecast Indicators — "What does machine learning predict?"

| Indicator | What It Does |
|---|---|
| **TimesFM Forecast** | Uses TimesFM 2.5 to forecast the selected horizon from up to 1,024 historical Close prices. It exposes a median path, q10 downside, q90 upside, interval width, probability of a positive return, and probability of clearing transaction costs. BUY/SELL requires at least 60% estimated directional probability. Forecasts are computed only at valid point-in-time origins and are never forward-filled as fresh observations. See [`docs/TIMESFM.md`](docs/TIMESFM.md). |

### Systemic Risk Indicators — "Is the market structurally fragile?"

| Indicator | What It Does |
|---|---|
| **Market Correlation** (Absorption Ratio) | Tracks the interconnectedness of 11 S&P 500 sector ETFs using eigenvalue analysis of the rolling 60-day correlation matrix. The absorption ratio is the fraction of total market variance captured by the first eigenvalue. When it's high (>0.5), cross-sector correlations are elevated, diversification is breaking down, and the probability of a correlated selloff is elevated (SELL). Low absorption ratios indicate healthy diversification (mild BUY). Based on Kritzman, Li, Page & Rigobon (2011). |

## How Predictions Work

The tool doesn't rely on any single indicator. Instead, it combines all selected indicators using a weighted voting system:

1. **Each indicator votes independently** — it produces a direction (BUY, SELL, or HOLD) and a signal-strength value based on how far the current reading is from key thresholds.

2. **Votes are weighted** — some indicators carry more weight than others. For example, MACD (weight 1.2) has slightly more influence than OBV (weight 0.7).

3. **Weights adapt to your time horizon** — if you're predicting 1–3 days ahead, momentum indicators, VPIN, and ML forecasts get boosted because they're better at short-term signals. For predictions beyond 10 days, trend indicators, macro regime signals, and bubble risk get boosted instead. Each of the 9 indicator categories has its own timescale profile.

4. **Only BUY and SELL compete** — indicators that vote HOLD are recorded but don't influence the directional outcome. The direction with the highest weighted score wins.

5. **Ambiguous signals become HOLD** — if BUY and SELL scores are within 10% of each other, the result is HOLD rather than making a low-confidence call.

## Backtest Settings

When running a backtest, you can configure:

- **Ticker** — which stock or crypto to test
- **Period** — how far back to test (1 month to max available history)
- **Signal Horizon** — a 1–30 daily-bar horizon used to adjust indicator weights and measure backtest exits; stock bars exclude weekends and market holidays
- **Indicators** — which indicators to include in the signal
- **Initial Capital** — starting portfolio value (default $10,000)
- **Transaction Cost** — round-trip cost per trade as a percentage (default 0.1%), covering slippage and commissions

The displayed period is the evaluation window, not the warmup window. The app fetches earlier observations for rolling indicators, then restricts trades and benchmarks to matching dates.

## Configuration

### In-App Advanced Settings

The Predict and Backtest pages both include an **Advanced Settings** expander in the sidebar where you can tune the signal engine without editing any code:

- **Indicator Weights** (0.0–2.0 per indicator) — increase or decrease each indicator's influence on the combined signal. Setting a weight to 0 effectively disables that indicator.
- **RSI Oversold / Overbought** — adjust the thresholds that trigger RSI buy and sell signals (defaults: 30 / 70).
- **Stochastic Oversold / Overbought** — adjust the zones for Stochastic crossover signals (defaults: 20 / 80).
- **Ambiguity Threshold** — how close BUY and SELL scores need to be before the result becomes HOLD (default 0.10).
- **Reset to Defaults** — one-click button to restore all settings to their original values.

Overrides are session-scoped — they apply immediately to signal generation and backtests but reset when you close the browser tab. For permanent changes, edit `config/settings.py` directly.

### File-Based Configuration

Advanced users can adjust defaults in `config/settings.py`:

- **Indicator parameters** — periods, thresholds (RSI overbought/oversold levels, Stochastic zones, Bollinger Band width, etc.)
- **Indicator weights** — how much influence each indicator has in the combined signal
- **Timescale adjustments** — how weights shift for short, medium, and long prediction horizons
- **Ambiguity threshold** — how close BUY and SELL scores need to be before the result becomes HOLD (default 10%)
- **Transaction cost** — default round-trip cost percentage for backtests
- **Watchlist presets** — predefined ticker lists for the Screener (Tech Giants, S&P 500 Top 10, Major Crypto, Indices)

## Project Structure

```
Capitalisman/
├── app.py                      # App entry point
├── Capitalisman.png            # Project logo and README artwork
├── ROADMAP.md                  # Reserved plans for future development
├── environment-timesfm.yml     # Isolated TimesFM Conda environment
├── requirements-timesfm.txt    # TimesFM environment packages
├── docs/
│   └── TIMESFM.md              # Runtime, CUDA, interpretation, and benchmarks
├── requirements.txt            # Core Python dependencies
├── requirements-optional.txt   # Optional integrations and charting dependencies
├── requirements-dev.txt        # Development and test dependencies
├── config/
│   ├── settings.py             # All configurable defaults & watchlist presets
│   └── overrides.py            # Session-scoped settings override system
├── data/
│   ├── fetcher.py              # Market data fetching, search, and caching
│   └── watchlists.py           # Persistent watchlist storage (~/.capitalisman/)
├── indicators/
│   ├── base.py                 # Indicator interface
│   ├── registry.py             # Auto-registration system
│   ├── _utils.py               # Cross-asset data fetching & date alignment utilities
│   ├── trend.py                # SMA, EMA, MACD, ADX
│   ├── momentum.py             # RSI, Stochastic
│   ├── volatility.py           # Bollinger Bands
│   ├── volume.py               # VWAP, OBV
│   ├── macro.py                # Copper-Gold Ratio, VIX Term Structure
│   ├── structural.py           # Bubble Risk (Hurst + log-price acceleration)
│   ├── microstructure.py       # VPIN (flow toxicity via Bulk Volume Classification)
│   ├── systemic.py             # Market Correlation (absorption ratio via eigenvalue analysis)
│   ├── fred.py                 # FRED Macro (yield curve, jobless claims, fed funds rate)
│   └── forecast.py             # TimesFM Forecast (zero-shot ML price prediction)
├── ml/
│   ├── timesfm_runtime.py      # Lazy model loading and CUDA preflight
│   └── benchmark.py            # Point-in-time probabilistic evaluation
├── scripts/
│   ├── timesfm_check.py        # Runtime health and inference smoke test
│   └── benchmark_timesfm.py    # Rolling ticker benchmark CLI
├── signals/
│   ├── base.py                 # Signal data types
│   └── combiner.py             # Weighted voting combiner
├── backtesting/
│   ├── engine.py               # Walk-forward backtest engine
│   ├── metrics.py              # Performance metric calculations
│   └── report.py               # Backtest results structure
├── charts/
│   ├── tradingview.py          # TradingView chart rendering
│   ├── plotly_fallback.py      # Plotly chart rendering (candlestick, comparison, equity)
│   └── factory.py              # Auto-selects best chart renderer
├── ui/
│   ├── components.py           # Shared sidebar controls & advanced settings
│   ├── page_predict.py         # Predict page (with multi-timeframe signals)
│   ├── page_backtest.py        # Backtest page (with CSV export)
│   ├── page_explore.py         # Explore page
│   ├── page_search.py          # Search page
│   ├── page_compare.py         # Compare page
│   └── page_screener.py        # Screener page (with CSV export & persistent watchlists)
└── tests/
    ├── conftest.py             # Test fixtures & synthetic OHLCV data factory
    ├── test_indicators.py      # Indicator computation & signal tests
    ├── test_combiner.py        # Signal combination logic tests
    ├── test_backtest.py        # Backtest engine & metrics tests
    └── test_fetcher.py         # Data fetcher utility tests
```

## Dependencies

Installed automatically via `pip install -r requirements.txt`:

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `yfinance` | Free market data from Yahoo Finance (price data, cross-asset reference data) |
| `ta` | Technical indicator calculations |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data processing |
| `scipy` | Scientific computing (used by novel indicators for statistical functions) |
| `fredapi` | Optional FRED economic data access (in `requirements-optional.txt`) |
| `lightweight-charts` | Optional TradingView-style charts (in `requirements-optional.txt`) |
| `pytest` | Development testing (in `requirements-dev.txt`) |
| `timesfm` + `torch` | Google TimesFM 2.5 forecasting (isolated optional environment) |

## Running Tests

The project includes a test suite covering indicators, signal combination, backtesting, and data utilities:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The suite currently contains 196 tests and uses synthetic OHLCV data for core
behavior. Cross-asset indicators (Copper-Gold Ratio, VIX Term Structure, Market
Correlation) may attempt to fetch live reference data; without network access,
they gracefully fall back to HOLD with zero confidence. FRED Macro behaves the
same way when `FRED_API_KEY` is unavailable. TimesFM unit tests use an injected
model and never download weights; real CUDA inference is an explicit smoke test
documented in [`docs/TIMESFM.md`](docs/TIMESFM.md).

## Adding Your Own Indicators

The indicator system uses a plugin architecture. To add a new indicator:

1. Create a class that extends `BaseIndicator` (in `indicators/base.py`)
2. Implement `name`, `category`, `lookback`, `compute()`, `get_signal()`, and `get_chart_config()`
3. Decorate it with `@register` from `indicators/registry.py`
4. Import the module in `indicators/__init__.py`
5. Add entries to `INDICATOR_WEIGHTS`, `INDICATOR_CATEGORIES`, and `TIMESCALE_ADJUSTMENTS` in `config/settings.py`

The indicator is then automatically available in all pages (Predict, Backtest, Explore, Screener) with no further wiring needed.

For indicators that need data from other tickers (like the macro and systemic indicators), use the helpers in `indicators/_utils.py` — `fetch_reference_close()` provides cached fetching, and `align_to_index()` handles timezone-safe date alignment.

## Disclaimer

This tool is for **educational and personal research purposes only**. It is not financial advice. Technical indicators are backward-looking tools that analyze past price patterns — they cannot predict the future with certainty. The novel indicators (macro regime, bubble risk, VPIN, systemic correlation) incorporate cross-asset and statistical signals that go beyond simple price pattern analysis, but they are still fallible and should not be used as the sole basis for trading decisions. Past performance of backtested strategies does not guarantee future results. Always do your own research before making investment decisions.
