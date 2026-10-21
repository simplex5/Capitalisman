# Capitalisman

<p align="center">
  <img src="Capitalisman.png" alt="Capitalisman financial divination wheel logo" width="240">
</p>

Stock and crypto research dashboard with technical signals, backtesting,
historical simulation, reproducible benchmarks, and optional TimesFM forecasts.
Built with Python and Streamlit, using market data from Yahoo Finance.

## Features

- **Predict:** combine nine default technical indicators into BUY/SELL/HOLD
  research signals, with separate directional agreement and evidence strength.
- **Backtest:** evaluate historical signals with next-open entries, transaction
  costs, daily equity, trade exports, and checksummed research archives.
- **Simulator:** step through historical daily bars with a single-asset long/cash
  strategy, configurable costs, and an event ledger.
- **Explore, Search, Compare, and Screener:** browse charts, find symbols,
  compare assets, and scan saved watchlists.
- **Optional indicators:** FRED macro data, experimental cross-asset and
  structural signals, and TimesFM point and quantile forecasts with explicit
  CPU/GPU selection. TimesFM 2.5 is the default; 3.0 is an experimental option
  for non-commercial, non-production research.
- **Research automation:** compare fixed strategies and forecast baselines;
  run bounded asset/horizon suites with saved inputs, time limits, failure
  records, and exploratory forecast uncertainty intervals. Compare TimesFM
  2.5 and 3.0 on frozen inputs with a local HTML report and interactive charts.

Signals and model scores are uncalibrated research outputs. Historical results
and bootstrap intervals do not establish a trading edge. There is no brokerage
execution integration.

## Quick start

The reproducible dependency lock targets **Python 3.11 on Linux x86_64**:

```bash
git clone https://github.com/NicholasCStanley/Capitalisman.git
cd Capitalisman
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://localhost:8501**, enter a symbol such as `AAPL`, `SPY`, or `BTC-USD`,
and choose a page from the sidebar. Core features need internet access for
market data but no API key.

On other platforms, install `requirements.txt` in a virtual environment;
that installation does not reproduce the verified Linux lock. Development
packages are in `requirements-dev.txt`.

## Optional integrations

- **FRED:** install `requirements-optional.txt` in the app environment and set
  `FRED_API_KEY`. This file also includes optional TradingView-style charting.
- **TimesFM:** use the separate model environment described in the
  [TimesFM guide](docs/TIMESFM.md). It covers installation, cached weights,
  device selection, runtime checks, and forecasting limits.

## Reproducible research

Backtests use a signal at close `t`, enter at open `t+1`, and exit at close
`t+horizon`. Backtest fees are quoted round trip; simulator fees are per fill.
Historical tools exclude today's daily bar. FRED and live cross-asset reference
indicators remain excluded from backtests until their historical availability
can be established.

The [research workflow](docs/RESEARCH_WORKFLOW.md) explains archive inspection
and execution replay, explicit test windows, baseline comparisons, and the
JSON protocol for bounded suites. Inspect the CLI options with your selected
Python environment:

```bash
python -m scripts.compare_strategies --help
python -m scripts.benchmark_timesfm --help
python -m scripts.benchmark_suite --help
```

## Documentation and agent use

- [Agent instructions](AGENTS.md) and [operation guide](docs/AGENT_OPERATIONS.md):
  supported commands, research guardrails, and failure handling.
  [CLAUDE.md](CLAUDE.md) imports the shared policy for Claude Code.
- [Research workflow](docs/RESEARCH_WORKFLOW.md): accounting assumptions,
  artifacts, benchmarks, uncertainty, and dependency verification.
- [TimesFM guide](docs/TIMESFM.md): optional model setup and operation.
- [October 20 change record](docs/CHANGES_2026-10-20.md): implemented fixes,
  experimental TimesFM 3.0, future testing instructions, and the AAPL comparison.
- [Roadmap](ROADMAP.md): completed work and planned improvements.
- [Correctness review status](CRITICAL_FIXES.md): fixes applied to the original review.

## Development

From the verified environment, or after installing `requirements-dev.txt`:

```bash
python -m pip check
python -m ruff check .
python -m pytest -q
git diff --check
```

Tests run offline with synthetic data and injected feeds/models. Real model
checks are separate. New indicators must explicitly opt into historical use
and pass causality tests; external feeds also need a historical availability
contract. See [AGENTS.md](AGENTS.md) before making changes.
