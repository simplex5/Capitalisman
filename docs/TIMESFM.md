# TimesFM Research Runtime

Capitalisman treats TimesFM as an optional research component, not a guaranteed
trading oracle. The integration uses TimesFM 2.5's PyTorch API to produce a
point forecast and quantiles, converts those into cost-aware probabilities, and
keeps the model isolated from the core application environment.

## Safe Isolated Installation

Do not install or upgrade PyTorch in the base Conda environment. Create the
reserved environment first:

```bash
conda env create -f environment-timesfm.yml
conda run -n capitalisman-timesfm python -m pip install --upgrade pip
```

The environment sets `PYTHONNOUSERSITE=1` so packages installed in the user's
global Python directory cannot leak into this runtime.

This workstation's RTX 5090 and driver support a CUDA 12.8 PyTorch wheel. Install
that wheel inside the new environment, then install the app and TimesFM packages:

```bash
conda run -n capitalisman-timesfm python -m pip install \
  torch --index-url https://download.pytorch.org/whl/cu128
conda run -n capitalisman-timesfm python -m pip install -r requirements-timesfm.txt
```

On a different machine, use the command produced by the official PyTorch
installer selector instead of assuming `cu128`. CUDA toolkit packages installed
system-wide are not modified by these commands; PyTorch's runtime stays inside
the Conda environment.

## Runtime Check

First run the dependency and hardware preflight, which does not load weights:

```bash
conda run -n capitalisman-timesfm python -m scripts.timesfm_check --device cuda
```

Then run an explicit smoke test:

```bash
conda run -n capitalisman-timesfm python -m scripts.timesfm_check \
  --device cuda --smoke
```

The smoke test downloads the model on first use and caches it in the normal
Hugging Face cache. It prints point, 10th-, 50th-, and 90th-percentile paths.
The preflight reports the selected device, package versions, GPU, GPU memory,
available RAM, and free disk. Errors remain visible in the indicator rather than
silently masquerading as a valid neutral prediction.

Optional runtime settings:

| Variable | Default | Purpose |
|---|---:|---|
| `CAPITALISMAN_TIMESFM_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `CAPITALISMAN_TIMESFM_MODEL_ID` | `google/timesfm-2.5-200m-pytorch` | Model source |
| `CAPITALISMAN_TIMESFM_MAX_CONTEXT` | `1024` | Maximum input bars |
| `CAPITALISMAN_TIMESFM_FORECAST_CONTEXT` | `1024` | Selected input bars, bounded by maximum context |
| `CAPITALISMAN_TIMESFM_MAX_HORIZON` | `256` | Maximum output bars |
| `CAPITALISMAN_TIMESFM_BATCH_SIZE` | `32` | Compiled inference batch size |
| `CAPITALISMAN_TIMESFM_CHUNK_SIZE` | `32` | Maximum series sent to the model per call |
| `CAPITALISMAN_TIMESFM_MEMORY_TARGET` | `0.72` | Fraction of total VRAM allowed by empirical recommendations |
| `CAPITALISMAN_TIMESFM_PROFILE` | `custom` | Profile label recorded with results |
| `CAPITALISMAN_TIMESFM_USE_CASE` | `interactive` | Workload label recorded with results |

## Hardware and Workload Profiles

Capitalisman separates hardware utilization from predictive methodology. Runtime
profiles control context capacity, selected context, batch size, and workload
chunking. They do not change signal thresholds, indicator weights, target
representation, or probability calibration.

The application offers `Auto`, `Fast`, `Balanced`, and `Thorough` profiles when
TimesFM is selected. Selection uses currently free VRAM rather than relying only
on the GPU product name, then adapts to the workload:

| Use case | Behavior |
|---|---|
| `interactive` | Latest forecast with a single-series chunk |
| `watchlist` | Batched latest forecasts across assets |
| `backtest` | Bounded historical-origin batches |
| `research` | Larger context and batch envelope for benchmark sweeps |

The starting memory tiers are conservative: at least 24 GB free VRAM uses the
enthusiast tier, 14 GB the high tier, 10 GB the performance tier, and 7 GB the
standard tier. Lower-memory GPUs and CPUs receive reduced contexts and batches.
Because other applications may occupy VRAM, an RTX 5090 can intentionally select
a lower tier when little memory is free.

Balanced starting points for common dedicated GPUs are:

| Typical hardware | Free-VRAM tier | Interactive | Watchlist | Research |
|---|---|---|---|---|
| RTX 5090, about 32 GB free | Enthusiast | context 1024, batch 32, chunk 1 | context 1024, batch 256, chunk 128 | context 2048, batch 256, chunk 128 |
| RTX 5070 Ti, about 16 GB free | High | context 1024, batch 32, chunk 1 | context 1024, batch 128, chunk 128 | context 2048, batch 128, chunk 128 |
| RTX 5070, about 12 GB free | Performance | context 1024, batch 32, chunk 1 | context 1024, batch 64, chunk 128 | context 2048, batch 64, chunk 128 |
| 8 GB GPU | Standard | context 1024, batch 32, chunk 1 | context 1024, batch 64, chunk 128 | context 2048, batch 64, chunk 128 |

These are starting limits, not claims of superior predictive accuracy. `Fast`
halves the selected context and caps batches at 64. `Thorough` doubles the
selected context where feasible, reduces batch size, and raises the memory target
from 72% to 82%. The empirical probe should be used before sustained large runs.

Inspect the recommendation without loading weights:

```bash
conda run -n capitalisman-timesfm python -m scripts.timesfm_check \
  --profile balanced --use-case watchlist
```

Run the opt-in empirical probe, which loads the model and compares representative
series-per-call sizes:

```bash
conda run -n capitalisman-timesfm python -m scripts.timesfm_check \
  --profile thorough --use-case research --autotune
```

The probe recommends a throughput chunk size for that installed GPU, driver,
PyTorch, TimesFM, context, and horizon combination. It does not tune forecast
accuracy. Context length and other analytic choices must still be selected using
chronological out-of-sample evaluation.

To require acceleration when launching the app:

```bash
CAPITALISMAN_TIMESFM_DEVICE=cuda \
  conda run -n capitalisman-timesfm streamlit run app.py
```

## What the Indicator Computes

For the selected analysis horizon, `TimesFM Forecast`:

1. sends only Close prices available at the forecast origin;
2. obtains the point path and nine forecast quantiles;
3. reports median expected return, q10 downside, q90 upside, and interval width;
4. estimates `P(up)` and `P(return > transaction costs)` from the quantiles; and
5. requires at least 60% directional probability before emitting BUY or SELL.

Forecast values are written only at actual forecast origins. They are not
forward-filled across bars, which avoids presenting stale forecasts as if they
were newly generated. Signal generation is horizon-aware throughout prediction,
screening, comparison, and backtesting.

## Point-in-Time Benchmark

Use a rolling evaluation before assigning material weight to the model:

```bash
conda run -n capitalisman-timesfm python -m scripts.benchmark_timesfm AAPL \
  --period 2y --horizon 10 --step 10 --device cuda \
  --output /tmp/aapl-timesfm.json
```

Each origin sees only the history that existed at that origin. The report stores
every origin/target pair and summarizes:

- median return mean absolute error;
- naïve last-price return mean absolute error;
- directional accuracy;
- q10-q90 empirical coverage and mean width;
- probability-of-up Brier score; and
- q10, q50, and q90 pinball losses.

The model adds forecasting value only if it improves out-of-sample metrics over
simple baselines consistently across symbols, regimes, and horizons. An 80%
interval whose observed coverage is far from 80% is poorly calibrated even when
its point forecast looks accurate.

## Current Limits

- TimesFM is pretrained on general time series, not specifically optimized for
  tradeable risk-adjusted returns.
- Close-only input ignores volume, volatility, fundamentals, and market regime.
- Quantile interpolation is an approximation and clips probability outside the
  provided q10-q90 range.
- Backtests still need realistic slippage, latency, survivorship-bias controls,
  and broader naïve/statistical baselines before supporting investment claims.
- Model downloads require network access on first use; inference can be local
  after weights are cached.

The next development stage should add benchmark result views, multi-asset and
multi-regime evaluation, probability calibration on a held-out window, and a
learned ensemble that can down-weight TimesFM when it fails to beat baselines.
