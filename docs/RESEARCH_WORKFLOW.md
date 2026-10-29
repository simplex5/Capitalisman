# Reproducible research workflow

## Signal interpretation

Directional agreement is the larger BUY/SELL weighted score divided by their
sum. Evidence strength is that sum divided by **all selected indicator weight**,
including indicators returning HOLD or unavailable data. Actionable coverage is
the selected weight supplying a nonzero directional vote divided by all weight.
These quantities are not calibrated probabilities.

For example, one BUY at confidence 0.02 and one equally weighted HOLD produce
100% agreement, 1% evidence and 50% coverage. The result is HOLD because evidence
is below the default 15% floor. This floor is a configurable heuristic, not an
empirically fitted threshold. Advanced Settings and custom simulator strategies
expose it; run configurations capture it. Existing results can change under this
new policy. The screener ranks by evidence first, then agreement.

## Historical admission

Indicators default to `historical_safe = False`. Opt-in indicators are discovered
from the registry by the causality test, which compares computed history and
signals against truncated inputs, including positive and negative bar indices.
TimesFM uses an injected model in tests; that validates integration, not weights
or GPU kernels. Additional core tests use several random seeds.

Copper-Gold Ratio, VIX Term Structure and Market Correlation pass algebraic
prefix tests on fixed reference fixtures. They remain excluded from backtests:
their live feeds do not establish publication times, cross-market close
availability, or immutable historical vintages. FRED remains excluded too.
These indicators remain available for current predictions. A publication lag
alone does not solve revised-data leakage. Primary adjusted price histories can
also change; archives preserve what the run used, not proof of historical truth.

## Saved runs

The Backtest page's **Download research archive** button produces a versioned ZIP
containing JSON files for input OHLCV, computed indicator columns, effective
configuration, trades, daily portfolio snapshots and equity, metrics, package
versions, and a source-code fingerprint. Each payload has a SHA-256 checksum.
The format uses no pickle. Checksums detect corruption; they are not signatures.
Non-finite report metrics use null or the strings `Infinity` / `-Infinity`.

Load or replay an archive from Python:

```python
from pathlib import Path
from research.artifacts import load_archive, replay_backtest_archive

data = Path("saved-backtest.zip").read_bytes()
saved = load_archive(data)
market = saved["frames"]["market"]
report = replay_backtest_archive(data)
```

Replay uses stored computations and settings without fetching data or running
TimesFM again. It requires matching source code. The archive records source
identity but does not bundle source or model weights: preserve the matching
checkout and dependency environment. Downloaded backtest archives support
execution replay; comparison archives contain multiple results and can supply
their market frame to a new comparison.

## Held-out strategy comparisons

Choose a test date **before inspecting its outcomes**. Earlier bars provide
warmup; no strategy is fitted or selected on test outcomes. The fixed ensemble
is compared with SMA Crossover, one continuous buy-and-hold position, and cash.
All use the same initial capital, execution window and fee rates. Technical
strategies use the requested holding horizon; buy-and-hold exits at the end.
Insufficient warmup is an error rather than silently moving the start date.
Insolvency ends a run early, and the summary exposes its actual end and reason;
Sharpe then covers that shorter observed duration.

```bash
python -m scripts.compare_strategies prices.csv --ticker AAPL \
  --test-start 2025-01-02 --horizon 5 --cost-pct 0.1 --output comparison.zip
```

The CSV needs `Date,Open,High,Low,Close,Volume`, including pre-test warmup. An
existing research ZIP can replace the CSV. Output paths must be new, preserving
older artifacts. `research.evaluation.compare_strategies` also accepts explicit
historically safe indicators, including an independently configured TimesFM
instance. The CLI defaults to the nine core technical indicators.

Historical research commands and the Backtest and Simulator pages exclude today's
dated bar and future bars in the provider's index timezone (UTC for date-only indices).
The Simulator applies this cutoff before preparing strategy signals and uses the
same filtered history for replay and subsequent strategy changes. If no completed
bars remain, it reports an error without preparing a strategy or creating a replay.
This conservatively omits even a stock bar after today's exchange close. An
exchange calendar and provider finality contract would be needed to admit it.
The lower-level evaluation APIs operate on the supplied historical frame;
callers can use `research.data.completed_daily_bars` with an explicit cutoff.

## Held-out forecast comparisons

In the TimesFM environment:

```bash
python -m scripts.benchmark_timesfm AAPL --period 5y --horizon 10 \
  --min-context 128 --test-start 2025-01-02 --device cuda \
  --archive forecast.zip --output forecast.json
```

The test start is the first eligible **forecast origin**; all targets follow it.
Each model forecast and baseline sees only history through that origin. Fixed
baselines are last price, 60-bar arithmetic drift (clamped at zero), 20-bar moving
average and exponential smoothing with span 20. Reports include each baseline's
return MAE and directional accuracy on identical origins; positive
`model_mae_improvement` means the model has lower MAE. Directional accuracy
compares signs, treating a flat forecast as neutral. By default origins are
spaced by the horizon; smaller steps are recorded as overlapping targets.

Inputs, runtime settings and per-origin outputs are archived. Each baseline also
has a paired circular-block bootstrap interval for the mean MAE improvement
(`baseline absolute error - model absolute error`, in return units). Positive
values favor the model. Pairing keeps both forecasts on the same origins; blocks
retain local temporal dependence. The fixed seed is 1729 with 2,000 resamples.
Block length is the larger of the cube root of sample size (rounded up) and
`ceil(horizon / step)`, covering overlapping targets. Fewer than ten origins or
three blocks produces `insufficient_observations` and null interval endpoints.

These exploratory 95% percentile intervals assume sufficiently stable temporal
dependence; block-length sensitivity, regime changes and multiple comparisons
remain unresolved. They do not calibrate probabilities or prove profitable
execution. No across-asset pooling or winner selection is performed. Strategy
returns/Sharpe remain descriptive without bootstrap intervals. Repeat the same
predeclared protocol across symbols and horizons before drawing conclusions.
Synthetic tests validate machinery, not forecasting value on market data.

## Bounded suites from local data

For a TimesFM 2.5 versus experimental 3.0 comparison, use `mode: "models"`.
This creates two sequential jobs per asset/horizon, using the same frozen
input snapshot and runtime sizes. Each job runs the existing forecast benchmark
and a TimesFM-only backtest with the fixed SMA, buy-and-hold and cash baselines.
The suite writes `comparison.html` with charts and a table, preserving failed
jobs and partial attempts. See [the setup and protocol](TIMESFM.md#experimental-timesfm-30-comparison).
Both checkpoints must already be cached in a model environment with
`timesfm==3.0.2`; the original 2.5 environment can remain unchanged.

Create a protocol JSON such as `protocol.json` before examining outcomes:

```json
{
  "assets": [{"ticker": "AAPL", "input": "prices.csv"}],
  "horizons": [5, 10],
  "mode": "both",
  "test_start": "2025-01-02",
  "as_of": "2026-09-27T00:00:00Z",
  "cost_pct": 0.1,
  "min_context": 128,
  "max_jobs": 4,
  "job_timeout_seconds": 120,
  "total_timeout_seconds": 480,
  "runtime": {
    "device": "cpu", "max_context": 128, "max_horizon": 30,
    "batch_size": 1, "chunk_size": 1, "torch_compile": false
  }
}
```

```bash
python -m scripts.benchmark_suite protocol.json --output-dir research_runs/suite-001
```

Input paths are relative to the protocol file. Each input is a local OHLCV CSV
or research ZIP, with sufficient warmup. `as_of` is an explicit timezone-aware
completed-bar cutoff. All market snapshots and job settings are saved before
any evaluation. Modes are `strategies` (default), `forecast`, `both`, or `models`;
`models` runs two jobs per asset/horizon. Strategy jobs compare the four existing
fixed strategies, with cost quoted round trip. Forecast jobs compare TimesFM
with the four fixed price baselines, using horizon-spaced origins. The same
`test_start` denotes first eligible execution for strategies and first eligible
origin for forecasts; these are distinct evaluations, not pooled returns.

Jobs run sequentially in separate processes using the launching interpreter.
Forecast jobs require the model environment and cached weights; the runner sets
`HF_HUB_OFFLINE=1` and never requests market data. Defaults permit at most 12
jobs, 600 seconds per worker and 600 seconds for the worker phase overall.
Protocols may raise these up to 100 jobs and 86,400 seconds. Timed-out process
groups are killed on Linux; Ctrl-C cancels the active job and records remaining
jobs as cancelled. There are no automatic retries or changes to settings.
Input preparation precedes the worker clock. Memory, disk use and arbitrary
third-party network behavior require host limits if stronger isolation is needed.

The output directory must be new and its parent must exist. It contains the
resolved protocol/source fingerprint, checksummed input snapshots, per-job
settings, stdout/stderr, result ZIPs/JSON, status records and consolidated
`summary.json`. Failed and timed-out jobs stay in the summary; budget-exhausted
jobs are marked `skipped_budget`. The CLI exits nonzero unless every job
completed. A preparation failure preserves its protocol and failure summary.
Use a new directory when correcting failed input; never overwrite an old run.

## Verified dependency environment and CI

The hashed lock targets **CPython 3.11, Linux x86_64** and contains core/development
packages, including PyArrow 25.0.1. Optional integrations and the CUDA stack have
separate requirements. Keep them in the documented separate environment.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
```

CI runs these checks on pushes and pull requests. Tests replace reference feeds
and forbid real model loading, so they need no credentials, internet or GPU.
Ruff currently enforces syntax/undefined-name correctness rules; broad style
cleanup and static type checking remain separate work.

To update the lock, edit `constraints-tested.txt`, then resolve in a **new empty**
Python 3.11 Linux environment (the install report must contain every dependency):

```bash
python3.11 -m venv /tmp/capitalisman-new-lock
/tmp/capitalisman-new-lock/bin/python -m pip --isolated install \
  -r requirements-dev.txt -c constraints-tested.txt --report /tmp/resolution.json
python -m scripts.lock_requirements /tmp/resolution.json
```

Verify installation from the resulting lock in another clean environment and
run CI checks before accepting the update. Hashes follow pip's
[repeatable installation guidance](https://pip.pypa.io/en/stable/topics/repeatable-installs/).
The PyArrow minimum excludes the version implicated in the earlier warning;
see the [Apache Arrow security advisories](https://arrow.apache.org/security/).
