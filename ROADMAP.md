# Capitalisman Development Roadmap

This document reserves and organizes ideas for future development. Items here
are proposals, not commitments, and are not implemented unless explicitly
marked complete.

## Product Direction

Evolve Capitalisman from an indicator dashboard into a repeatable research
workflow that explains:

- What the current signal suggests.
- How strongly the underlying evidence agrees.
- How similar signals performed historically.
- Which market conditions support or weaken the result.
- What uncertainty and downside remain.

## Near-Term Priorities

### TimesFM Foundation — Completed

- [x] Isolate PyTorch, CUDA, and TimesFM dependencies in a dedicated Conda
      environment that ignores user-site packages.
- [x] Replace the legacy TimesFM wrapper with the official TimesFM 2.5 PyTorch
      runtime, lazy model loading, hardware preflight, and actionable failures.
- [x] Make forecasts honor the user-selected horizon throughout prediction,
      comparison, screening, and backtesting.
- [x] Expose point, q10, q50, and q90 paths plus cost-aware directional
      probabilities in the Predict interface.
- [x] Add leakage-resistant rolling evaluation with naïve-baseline MAE,
      directional accuracy, interval coverage, Brier score, and pinball loss.
- [x] Validate the test suite and real inference on an NVIDIA CUDA device.

### TimesFM Runtime Profiles — Completed

- [x] Detect currently free VRAM and available system RAM instead of selecting
      settings from GPU product names alone.
- [x] Add Auto, Fast, Balanced, and Thorough runtime profiles for interactive,
      watchlist, backtest, and research workloads.
- [x] Keep compiled context capacity separate from the selected forecast context
      so hardware capacity does not silently become a modeling decision.
- [x] Bound large inference requests with profile-specific chunks and validate
      input length, finite values, output shapes, finite forecasts, and quantile
      ordering.
- [x] Add an opt-in empirical throughput probe that reports latency, throughput,
      peak allocated VRAM, failures, and a recommended workload chunk size.
- [x] Make interactive and watchlist profiles forecast only the latest origin,
      while retaining scheduled historical origins for backtest and research.
- [x] Prevent sparse historical rows from triggering redundant full-model
      inference.
- [x] Document starting profiles for RTX 5090, RTX 5070 Ti, RTX 5070, 8 GB GPU,
      and CPU systems without presenting throughput settings as accuracy tuning.

### TimesFM Forecast Development

#### Phase 0 — Correctness and Runtime Hardening

- [ ] Replace hard 0%/100% probability extrapolation outside q10-q90 with honest
      tail semantics, then label uncalibrated values as model-derived scores
      until held-out reliability calibration exists.
- [ ] Make transaction-cost treatment symmetric by calculating the short-side
      probability of finishing below `current_price * (1 - cost)`.
- [ ] Remove process-global `CUDA_VISIBLE_DEVICES` mutation and support explicit,
      reversible Auto, CUDA, and CPU selection with actual loaded-device checks.
- [ ] Record actual model/tensor device, peak allocated VRAM, compilation time,
      inference time, and realized series throughput in runtime status.
- [ ] Persist empirical profile results using a fingerprint of GPU UUID, driver,
      CUDA, PyTorch, TimesFM, model, context, horizon, and workload; invalidate
      stale results automatically.
- [ ] Run and record real profile smoke tests and empirical probes on the target
      RTX 5090 environment, plus at least one lower-VRAM CUDA configuration.
- [ ] Add explicit cancellation and safe out-of-memory recovery for long research
      and backtest jobs.

#### Phase 1 — Establish Measurable Value

- [ ] Add drift, moving-average, exponential-smoothing, and simple autoregressive
      baselines alongside the existing last-price baseline.
- [ ] Run reproducible multi-asset benchmark suites across equities, indices,
      rates, commodities, and crypto.
- [ ] Evaluate multiple forecast horizons and minimum-context lengths without
      selecting settings on the final test period.
- [ ] Store benchmark inputs, model/package versions, configuration, metrics, and
      per-origin results as versioned research artifacts.
- [ ] Add a benchmark results page that clearly shows whether TimesFM beats each
      baseline and where it fails.
- [ ] Define minimum acceptance criteria before increasing TimesFM's voting weight
      or presenting it as more than experimental.
- [ ] Add block-bootstrap confidence intervals and appropriate forecast-comparison
      tests so small apparent improvements are not treated as established edge.
- [ ] Account for overlapping forecast targets when selecting origin spacing,
      estimating uncertainty, and splitting train/calibration/test periods.

#### Phase 2 — Calibration and Regime Awareness

- [ ] Add chronological train/calibration/test partitions and nested rolling
      evaluation for any learned thresholds.
- [ ] Measure reliability curves for `P(up)` and empirical coverage for every
      forecast interval.
- [ ] Calibrate probabilities and quantiles on held-out data using isotonic,
      conformal, or other appropriate post-processing methods.
- [ ] Break down accuracy and calibration by volatility, trend, liquidity, asset
      class, and macro regime.
- [ ] Detect distribution shift and warn when the current context is unlike the
      model's validated operating conditions.
- [ ] Make signal confidence depend on calibrated historical reliability rather
      than raw model probability alone.

#### Phase 3 — Stronger Local ML

- [ ] Compare raw price, log-price, return, volatility-scaled return, and
      detrended input representations using identical evaluation windows.
- [ ] Add covariate experiments for volume, realized volatility, market returns,
      rates, and regime features without introducing future leakage.
- [ ] Require every dynamic XReg experiment to declare how future-horizon
      covariates are known or forecast; reject realized future market inputs.
- [ ] Build a local PyTorch ensemble that learns when to use TimesFM versus
      statistical and technical baselines.
- [ ] Investigate parameter-efficient fine-tuning only after the zero-shot
      benchmark and data-splitting framework are stable.
- [ ] Validate mixed-precision inference against float32 forecasts and calibration
      before enabling it; add reusable compiled-model caching across compatible
      profiles.
- [ ] Track model lineage, data lineage, random seeds, and reproducibility metadata
      for every trained or calibrated artifact.

#### Phase 4 — Forecast User Experience

- [ ] Visualize historical forecast cones against realized prices so users can
      see calibration rather than only the latest path.
- [ ] Explain in plain language when TimesFM agrees with, contradicts, or adds no
      value beyond simpler indicators.
- [ ] Show baseline comparisons, sample size, regime, data freshness, and model
      status beside every forecast.
- [x] Let users select Auto, Fast, Balanced, and Thorough workload profiles and
      inspect selected context, batch, chunk, device, and use case.
- [ ] Let users select Auto, CUDA, or CPU and inspect model/cache information from
      the interface.
- [ ] Provide cancellable progress, estimated workload, and bounded benchmark
      presets for expensive local analysis.
- [ ] Export a self-contained forecast research report with assumptions,
      uncertainty, metrics, and configuration.

### Signal Interpretation

- [ ] Add a signed contribution chart for every indicator.
- [ ] Separate directional agreement, historical reliability, and expected move.
- [ ] Prevent a single weak actionable vote from appearing as 100% directional
      agreement merely because every other indicator returned HOLD.
- [ ] Summarize conflicting indicator groups in plain language.
- [ ] Add approachable explanations for backtest metrics and assumptions.
- [ ] Display data freshness, missing data, and unavailable indicators.
- [ ] Expose the exact weights, thresholds, and configuration behind each result.

### Strategy Workflows

- [x] Introduce an immutable strategy configuration model for causal,
      single-asset historical simulation.
- [ ] Expand the initial Buy & Hold, Balanced Technical, Trend Following, and
      Mean Reversion simulator presets with Conservative Trend, Short-Term
      Momentum, Macro Risk, and Crypto 24/7 after their execution assumptions
      are defined.
- [ ] Allow named strategies to be saved, duplicated, edited, and restored.
- [ ] Keep ticker, period, horizon, and strategy synchronized between pages.
- [ ] Add a guided first-run workflow for less experienced users.
- [ ] Export complete research reports in addition to raw CSV files.

### Historical Replay Simulator

#### MVP Foundation — Completed

- [x] Add a Streamlit-independent daily-bar replay engine with deterministic
      `step` and bounded `advance` operations.
- [x] Model one all-in long position or cash, mark open holdings to every close,
      and charge an explicit percentage cost on each fill.
- [x] Enforce close-to-next-open execution so decisions cannot use an unseen
      bar's open or close.
- [x] Support drop-in and end-date normalization, latest-data and duration
      endpoints, fixed-dollar and percentage-return goals, and capital-depletion
      termination.
- [x] Support pausing and prospective strategy changes while preserving the
      current holding and cancelling stale pending orders.
- [x] Record an append-only event ledger plus immutable portfolio snapshots for
      audit and deterministic replay tests.
- [x] Add a Simulator page with preset/custom setup, manual stepping, bounded
      1/2/5/10/25/50/100-bar advances, charts, metrics, and event inspection.

#### Simulator Expansion

- [ ] Add wall-clock autoplay with responsive pause controls and configurable
      animation cadence; keep speed separate from the number of market bars.
- [ ] Persist, resume, export, and import versioned simulation runs and strategy
      definitions.
- [ ] Add end-of-run comparisons against buy-and-hold and other presets on the
      identical resolved date range.
- [ ] Add custom success-condition composition, drawdown/risk goals, scorecards,
      achievements, and richer explanations without changing core accounting.
- [ ] Add partial allocation, position sizing, shorting, margin, dividends,
      corporate actions, stops, multi-asset portfolios, and intraday execution
      only as separately validated execution-model layers.
- [ ] Admit TimesFM, FRED, and cross-asset strategies only when every historical
      input is point-in-time reproducible and unavailable observations fail
      explicitly rather than silently becoming neutral signals.

## Research Quality

### Historical Calibration

- [ ] Measure hit rate, median return, and downside for comparable past signals.
- [ ] Show expected-return distributions and confidence intervals.
- [ ] Compare results against buy-and-hold, simple technical baselines, and
      random-direction baselines.
- [ ] Report whether observed performance is statistically meaningful.

### Out-of-Sample Validation

- [ ] Add explicit training, validation, and test date ranges.
- [x] Add causal rolling-origin TimesFM evaluation and next-bar-entry strategy
      backtesting without using future rows in signal construction.
- [ ] Prevent thresholds or weights selected on one period from being evaluated
      as though they were chosen independently on that same period.
- [ ] Add parameter-sensitivity charts to reveal fragile configurations.
- [ ] Store reproducible configuration identifiers with every backtest.

### Regime Analysis

- [ ] Classify trend, volatility, liquidity, and macro regimes.
- [ ] Break down performance by regime.
- [ ] Warn when a strategy is being used outside its historically favorable
      conditions.
- [ ] Support regime-dependent weights or strategy selection.

## Decision Support

- [ ] Add volatility and ATR context to each signal.
- [ ] Add research-oriented invalidation levels and risk ranges.
- [ ] Add volatility-based position-sizing examples with prominent caveats.
- [ ] Support user-selected benchmarks instead of always using SPY.
- [ ] Add alerts for signal changes, agreement thresholds, volatility spikes,
      and indicator crossovers.
- [ ] Add a watchlist dashboard showing changes since the previous scan.

## Portfolio Research

- [ ] Backtest multiple assets and simultaneous positions.
- [ ] Model cash allocation, position limits, and maximum portfolio exposure.
- [ ] Add configurable rebalancing and position-sizing rules.
- [ ] Model stop-loss, take-profit, and trailing-exit rules.
- [ ] Account for short borrow costs and asset-specific execution assumptions.
- [ ] Report portfolio-level volatility, drawdown, concentration, and turnover.

## Backtest and Market-Data Integrity

- [ ] Replace trade-return Sharpe approximation with a daily marked-to-market
      equity series that represents cash and open long/short positions.
- [ ] Model financing, borrow availability and fees, dividends, splits, delisting
      returns, trading halts, and asset-specific transaction costs where relevant.
- [ ] Store immutable, timestamped market-data snapshots so revised adjusted
      history cannot silently change a previously reported result.
- [ ] Add survivorship-bias-aware historical universes for multi-asset research
      instead of evaluating only today's constituents and watchlists.
- [ ] Validate chronological ordering, duplicate timestamps, interval regularity,
      freshness, timezone, OHLC relationships, volume, and corporate-action
      treatment at the data-provider boundary.
- [ ] Distinguish data-provider failure, missing optional data, insufficient
      history, and genuine neutral signals instead of broadly swallowing errors.
- [ ] Add reproducible execution assumptions and configuration identifiers to
      every exported backtest report.

## Technical Foundation

- [ ] Separate the research engine from Streamlit-specific caching and state.
- [ ] Support CLI, scheduled-job, and API entry points.
- [ ] Add a market-data provider abstraction rather than relying on one source.
- [ ] Add formal data validation, freshness checks, and provider diagnostics.
- [ ] Persist strategies and research results in SQLite or Parquet.
- [ ] Add vectorized or batch backtesting for screeners and parameter grids.
- [ ] Add UI integration tests for navigation, saved strategies, reset behavior,
      and stale-state prevention.
- [ ] Add performance benchmarks for indicator computation and watchlist scans.
- [ ] Add continuous integration with formatting, linting, type checking, unit
      tests, and optional TimesFM API-contract tests that do not download weights.
- [ ] Add an opt-in CUDA integration test matrix for supported runtime profiles,
      including output equivalence across chunk and batch sizes.

## Suggested Delivery Order

1. Finish TimesFM probability, cost-symmetry, device-selection, and profile
   persistence correctness work.
2. Expand TimesFM baselines and multi-asset benchmark coverage.
3. Add benchmark persistence and an interpretable results view.
4. Add chronological calibration, statistical uncertainty, and regime breakdowns.
5. Correct signal-confidence semantics and add contribution visualization.
6. Establish immutable market-data artifacts and marked-to-market backtesting.
7. Add strategy presets plus saved and versioned strategy configurations.
8. Evaluate safe covariates, local ensembles, and parameter-efficient fine-tuning.
9. Add portfolio backtesting, research reports, and alerts.

## Guiding Principle

Capitalisman should not merely provide a signal. It should show when similar
signals worked, how robust the evidence is, what could invalidate the result,
and how much uncertainty remains.
