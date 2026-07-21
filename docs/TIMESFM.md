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
| `CAPITALISMAN_TIMESFM_MAX_HORIZON` | `256` | Maximum output bars |
| `CAPITALISMAN_TIMESFM_BATCH_SIZE` | `32` | Compiled inference batch size |

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
screening, and comparison.

The indicator is excluded from walk-forward backtests (`backtest_safe = False`):
TimesFM's pretraining corpus overlaps any realistic backtest window, so its
historical forecasts are not point-in-time, and forecast columns exist only at
sparse origins while the backtest engine requires prefix-invariant values at
every bar. Use the rolling benchmark below for historical evaluation; the
indicator remains fully available for current predictions.

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
