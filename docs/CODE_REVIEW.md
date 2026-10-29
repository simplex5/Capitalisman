# Capitalisman — Code Review & Product Improvement Plan

Reviewed at commit `cdf85aa` (Ludofinancial Simulation). ~8,950 lines of Python
across 9 packages, 238 tests.

This is the original review, not current implementation status. Several findings
have since been addressed; see [the corrective-pass status](../CRITICAL_FIXES.md)
and [the research workflow](RESEARCH_WORKFLOW.md). In particular, a fixed release
lag alone does **not** make revised FRED data valid for historical backtests.

---

## 1. Executive summary

The architecture is sound: an indicator ABC with auto-registration, typed signal
results, a walk-forward backtester, and an event-sourced simulation engine. The
README and ROADMAP are notably honest about what the app does and does not claim.

The problems are not code quality in the small. They are:

1. **Three divergent copies of the core scoring logic** (combiner, simulation,
   and implicitly the backtester's precompute path). This is the single largest
   source of debt.
2. **A per-signal architecture that cannot express the product's actual claim.**
   `confidence` is vote agreement, not probability. Every downstream metric
   inherits this, and it caps how sophisticated the product can become.
3. **O(n·window) Python loops in the indicator layer** that make the Screener
   unusable with the interesting indicators enabled.
4. **No CI, no linting, no type checking**, and the only end-to-end UI test is
   red on `main`.

---

## 2. Correctness issues

### 2.1 The only end-to-end UI test is failing on `main`

```
tests/test_simulator_ui.py::test_simulator_can_create_and_step_a_replay
ValueError: '1y' is not in list
```

`ui/components.py:period_select` passes `format_func=lambda p: PERIOD_LABELS...`
to `st.selectbox`. Streamlit's `AppTest` resolves `Selectbox.index` by looking up
the *raw* value in the *formatted* options list, so any `AppTest` run that touches
a page with a period selector now throws. The friendly-labels feature silently
broke the smoke test.

**Fix:** either drop `format_func` and store `("1y", "1 Year")` tuples, or have the
test drive `set_value` through the formatted label. The former is cleaner and
removes the `PERIODS` / `PERIOD_LABELS` / `PERIOD_CALENDAR_DAYS` /
`WARMUP_FETCH_PERIOD` / `BACKTEST_FETCH_PERIOD` five-way parallel-dict problem
(see §4.3).

### 2.2 The multi-timeframe cards trigger three redundant TimesFM inferences

`ui/page_predict.py:277-303` renders 1d/5d/20d cards by calling:

```python
mtf_signal = combine_signals(chosen, computed_df, horizon_days=h, precomputed=True)
```

`precomputed=True` is honored by every indicator *except* TimesFM. Its
`get_signal_for_horizon` checks `_has_forecast(df, h, idx)` — the frame's
`TFM_horizon` column holds the user's chosen horizon, not `h` — then
`_has_computed_horizon(df, h)`, also false. So it falls through to a **full
`compute_for_horizon` call, running the model again**, three times, inside a
display loop.

Worse, each of those calls reassigns `self.latest_analysis` and
`self.last_error`. The panel is rendered before the loop so it displays correctly
today, but this is order-dependent by accident.

**Fix:** two parts. (a) Make the MTF cards reuse the already-computed frame or
skip forecast-category indicators. (b) Remove mutable state from the indicator —
see §3.2.

### 2.3 Negative-index wraparound in `get_signal` is latent look-ahead

The pattern `prev_idx = idx - 1 if idx != -1 else -2` appears in every crossover
indicator. At `idx == 0` this yields `prev_idx == -1`, which reads **the last bar
of the frame** — i.e. the future.

`indicators/volume.py:114` is worse:

```python
lookback = min(10, len(df) - 1)          # uses full frame length, not idx
start_idx = idx - lookback if idx != -1 else -(lookback + 1)
```

For any `idx < 10`, `start_idx` is negative and wraps to the end of the series.

Neither is reachable today (the backtester starts at `warmup ≥ 50`,
`prepare_strategy` starts at `required_lookback`), but they are silent
correctness landmines guarded only by a convention that lives in a different
module.

**Fix:** normalize the index once at the top of `get_signal` —
`i = idx if idx >= 0 else len(df) + idx` — and return HOLD when `i < lookback`.
Better: hoist this into a `BaseIndicator` helper so no subclass can get it wrong.

### 2.4 `advanced_settings` writes overrides on every rerun

`ui/components.py:249-262` loops over every indicator weight and calls
`set_override("INDICATOR_WEIGHTS", weights)` inside the loop, re-reading and
copying the whole dict each iteration. It runs unconditionally on every script
rerun, whether or not the expander was ever opened, so the session always carries
an "override" identical to the defaults. `analysis_settings_signature()` then
hashes that dict, so the signature is stable — but the coupling is invisible and
the O(n²) dict copying is pure waste.

**Fix:** build the dict once after the loop, and only write when it differs from
the current effective value.

### 2.5 Adjusted prices in a point-in-time backtest

`yf.Ticker.history()` returns split- and dividend-adjusted prices. Adjustments
are applied retroactively, so a backtest of a 2019 trade uses a price series that
encodes corporate actions announced in 2023. This is a real (if modest) leak that
no amount of `backtest_safe` flagging catches, because it is in the data, not
the indicator.

**Fix:** fetch with `auto_adjust=False`, keep the raw OHLC plus the adjustment
factor, and apply adjustments only up to the evaluation bar. At minimum,
document it in the backtest caption alongside the existing borrow-cost note.

---

## 3. Technical debt

### 3.1 The scoring logic is forked (highest priority)

`signals/combiner.py:combine_signals` and
`simulation/strategies.py:_combine_at` implement the same algorithm — weighted
directional voting with a timescale adjustment and an ambiguity threshold —
independently. `_get_timescale` and `_timescale` are byte-for-byte equivalent
functions in different modules.

They have **already diverged**:

| | `combine_signals` | `_combine_at` |
|---|---|---|
| Weight source | `get_setting("INDICATOR_WEIGHTS")` — respects UI overrides | `INDICATOR_WEIGHTS` — ignores UI overrides |
| Ambiguity threshold | `get_setting("AMBIGUITY_THRESHOLD")` — session-scoped | `definition.ambiguity_threshold` — strategy-scoped |
| Category source | `INDICATOR_CATEGORIES` only | `INDICATOR_CATEGORIES` falling back to `indicator.category` |
| Reasoning | rich `_build_reasoning` narrative | one-line driver list |

So the Predict page and the Simulator can display **different signals for the
same indicators on the same bar**, and only one of them honors the Advanced
Settings sliders. That is a product bug wearing a refactoring costume.

**Fix:** extract one scorer.

```python
@dataclass(frozen=True)
class ScoringPolicy:
    weights: Mapping[str, float]
    ambiguity_threshold: float
    timescale_adjustments: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_session(cls) -> "ScoringPolicy": ...
    @classmethod
    def from_strategy(cls, d: StrategyDefinition) -> "ScoringPolicy": ...

def score(signals: Sequence[SignalResult], policy: ScoringPolicy,
          horizon_days: int) -> CombinedSignal: ...
```

`score` takes already-produced `SignalResult`s — it does not know about
DataFrames or indicators. That makes it trivially unit-testable and forces the
two callers back onto one code path. This is a ~150-line change that deletes
~100 lines and closes a whole class of divergence bugs.

### 3.2 Indicators are stateful, violating their own contract

`BaseIndicator.compute` is documented as "Must not modify the input DataFrame" —
a purity claim. But `TimesFMForecast.__init__` holds `self.runtime`,
`self.latest_analysis`, and `self.last_error`, and `compute_for_horizon` mutates
the latter two. Meanwhile `registry.get_all_indicators()` constructs a **fresh
instance on every call**, so state set by one call site is invisible to another.
`ui/page_predict.py` only works because it happens to hold the same instance it
computed with.

The registry compounds this: `register` instantiates the class at import time
purely to read `.name`, and `get_indicators_by_category` instantiates every
registered class on every call.

**Fix:** make `name`/`category`/`lookback` class-level attributes rather than
instance properties, so the registry never needs to instantiate to introspect.
Return the analysis as data — have `compute_for_horizon` return
`(df, ForecastArtifacts)` or stash artifacts in a `dict` keyed by indicator name
that the caller owns. An indicator should be a pure function of
`(df, horizon)`.

### 3.3 `backtest_safe` is an unverified honor-system flag

The backtester's core safety argument is:

> Their value at bar t must be identical whether computed on `data[:t+1]` or on
> the full series.

This is asserted in a comment and enforced by a boolean the indicator sets about
itself. Nothing checks it. A future indicator using `.rolling(center=True)`,
`.bfill()`, `.interpolate()`, or a full-series `.mean()` would silently
manufacture alpha.

**Fix:** add a property-based conformance test that every registered
`backtest_safe` indicator must pass:

```python
@pytest.mark.parametrize("name", list_indicator_names())
def test_declared_causal_indicators_are_actually_causal(name, synthetic_ohlcv):
    ind = get_indicator(name)
    if not ind.backtest_safe:
        pytest.skip("not declared causal")
    full = ind.compute(synthetic_ohlcv)
    for t in (60, 90, 120):
        truncated = ind.compute(synthetic_ohlcv.iloc[:t + 1])
        assert_series_equal(full.iloc[t], truncated.iloc[t], ...)
```

This converts the project's most important invariant from prose into CI. It is
maybe 30 lines and it is the highest-leverage test in the repo.

### 3.4 No CI, linting, type checking, or dependency pinning

- No `.github/workflows/`, no `pyproject.toml`, no `setup.cfg`, no lockfile.
- Every requirement is `>=`, unpinned. `ta`, `yfinance`, and `streamlit` all
  break APIs regularly — `yfinance` especially.
- `numpy`/`pandas` type hints exist but nothing runs `mypy`.
- 19 bare `except Exception` handlers outside tests, several of which
  (`data/fetcher.py:168`, `:188`, `indicators/_utils.py:32`,
  `indicators/fred.py:59`) swallow the error entirely and return `None`, so a
  network failure and an invalid ticker are indistinguishable to the caller.
- `__pycache__/` and `.pytest_cache/` are committed to git despite `.gitignore`
  covering `__pycache__/` — they were added before the ignore rule.

**Fix:** a `pyproject.toml` with ruff + mypy, one GitHub Actions workflow running
`pytest`, and `git rm -r --cached` the stale caches. A day of work that pays for
itself the first time `yfinance` ships a breaking release.

### 3.5 Caching is inconsistent and partly invisible

Three caching mechanisms coexist:

- `@st.cache_data(ttl=300)` in `data/fetcher.py` — TTL'd, keyed, Streamlit-aware.
- `_reference_cache` in `indicators/_utils.py` — plain module dict, **never
  expires**, unbounded, process-lifetime.
- `_fred_cache` in `indicators/fred.py` — same pattern, separately implemented.

The module-level caches mean a long-running Streamlit server serves 2019 VIX data
forever, and there is no way to invalidate them from the UI. They also make tests
order-dependent (hence the `clear_reference_cache()` / `clear_fred_cache()`
escape hatches, which exist only for tests).

**Fix:** one `@cached(ttl=...)` decorator in a `data/cache.py` that works with or
without Streamlit, used by all three. Delete the two hand-rolled dicts.

---

## 4. What's awkward

### 4.1 Presentation logic copy-pasted across pages

`_signal_color` and `_signal_emoji` are defined identically in
`ui/page_predict.py`, `ui/page_screener.py`, and `ui/page_compare.py`. The signal
card HTML block is pasted four times with minor variations, each with
`unsafe_allow_html=True` (6 sites total). `page_predict` escapes its
interpolations; `page_compare` escapes the ticker; `page_screener` interpolates
`r['name']` — which comes from `yfinance`'s `shortName` — into markdown without
escaping.

**Fix:** one `ui/signal_card.py` with `render_signal_card(signal, *, size, subtitle)`
and a `DIRECTION_STYLE: dict[SignalDirection, Style]` table. Escape once, there.

### 4.2 `app.py`'s dispatch chain

Seven `elif page == ...` branches mapping strings to `render()` calls, with the
same strings repeated in the `st.sidebar.radio` options list. Adding a page means
editing three places (import, options list, dispatch).

**Fix:**

```python
PAGES = {
    "Predict": page_predict.render,
    "Backtest": page_backtest.render,
    ...
}
page = st.sidebar.radio("Navigation", list(PAGES), key="nav_page", ...)
PAGES[page]()
```

Or adopt Streamlit's native `st.navigation`/`st.Page` API, which also gives real
URLs per page — a genuine product win for sharing links.

### 4.3 Period configuration is five parallel dictionaries

`PERIODS`, `PERIOD_LABELS`, `PERIOD_CALENDAR_DAYS`, `WARMUP_FETCH_PERIOD`, and
`BACKTEST_FETCH_PERIOD` all key off the same period strings, with different and
partially overlapping coverage. `PERIOD_CALENDAR_DAYS` is missing `"max"`;
`WARMUP_FETCH_PERIOD` covers 6 of 11; `BACKTEST_FETCH_PERIOD` covers 10 of 11.
Adding a period means remembering all five, and the failure mode for forgetting
one is a silent `.get(period, period)` fallback.

**Fix:**

```python
@dataclass(frozen=True)
class Period:
    code: str
    label: str
    calendar_days: int | None
    warmup_fetch: str
    backtest_fetch: str

PERIODS: tuple[Period, ...] = (...)
```

One table, impossible to half-populate.

### 4.4 The `compute` / `get_signal` contract is ambiguous

Every `get_signal` starts with `if "SMA_short" not in df.columns: df = self.compute(df)`.
So `get_signal` sometimes computes and sometimes doesn't, `combine_signals` has a
`precomputed: bool` flag to control this from outside, and the backtester
pre-computes separately. Three layers each partially responsible for the same
decision, coordinated by a column-name existence check.

The `compute` / `compute_for_horizon` and `get_signal` / `get_signal_for_horizon`
pairs double the surface again — the non-horizon variants exist only as
delegating shims, and `TimesFMForecast.compute` delegates to
`compute_for_horizon(df, DEFAULT_FORECAST_HORIZON)`, quietly ignoring whatever
horizon the caller cared about.

**Fix:** collapse to a single `compute(df, horizon) -> pd.DataFrame` and
`signal(df, horizon, i) -> SignalResult`, where `signal` **requires** computed
columns and raises if absent. Delete the `precomputed` flag; make the caller's
responsibility explicit. This removes four methods from the ABC.

### 4.5 `MIN_DIRECTIONAL_PROBABILITY` and friends are scattered magic numbers

Thresholds live in three places: `config/settings.py` (RSI, Stochastic,
ambiguity), module constants (`forecast.py:MIN_DIRECTIONAL_PROBABILITY = 0.60`,
`structural.py` score cutoffs of 0.6/0.35, `microstructure.py` z-cutoffs of
2.0/1.0/-1.0, `systemic.py` AR cutoffs of 0.55/0.45/0.30), and inline literals in
signal bodies (`min(0.8, roc_conf)`, `spread * 20`, `norm_hist * 200`).

Those multipliers — `* 20`, `* 25`, `* 200`, `* 5000` — are the actual model.
They are unfitted, undocumented, and untestable, and they determine every
confidence value the product reports.

**Fix:** at minimum, move them to named class constants with a comment on their
scale rationale. Properly, see §5.2.

---

## 5. What could be done more elegantly / more sophisticatedly

### 5.1 Vectorize the O(n·window) indicators

`Bubble Risk` takes **3.8 seconds** for 1,250 bars (measured). It recomputes a
full Hurst R/S analysis and a quadratic polyfit from scratch at every bar, in
nested Python loops. `Market Correlation` does a 60×11 correlation matrix plus
`eigvalsh` per bar. In the Screener with an 8-ticker watchlist, that is ~30
seconds of pure CPU before any I/O.

Three levels of fix, in order of effort:

1. **Stride the expensive ones.** Compute every 5th bar and forward-fill.
   Bubble risk is a weeks-to-months signal; per-bar resolution is meaningless.
   ~5× for free.
2. **Vectorize the R/S computation.** The inner chunk loop is
   `np.cumsum` over a reshaped array — replaceable with a single strided
   `sliding_window_view` + `cumsum` along an axis. ~50× realistic.
3. **Incremental rolling eigendecomposition.** The correlation matrix at bar
   `t+1` differs from bar `t` by one rank-1 update; `scipy.linalg.eigh` with a
   warm start, or simply tracking the leading eigenvector via power iteration
   seeded from the previous bar, converges in 2–3 iterations instead of a full
   decomposition.

`scipy` is already a declared dependency and is currently unused — `_norm_cdf`
in `microstructure.py` hand-rolls an Abramowitz & Stegun approximation
specifically to "avoid a scipy dependency" that the project already has.

### 5.2 Replace vote-agreement with calibrated probability

This is the biggest product opportunity in the codebase.

Right now, `confidence` means "share of weighted directional votes." The UI is
scrupulously honest about this — *"Directional agreement measures the share of
actionable weighted votes supporting the result; it is not a probability of
success"* — which is exactly right and exactly the problem. **The product's core
number is admittedly not the number users want.**

Everything needed to fix it already exists:

- `backtesting/` produces `(predicted_direction, actual_direction)` pairs.
- `ml/benchmark.py` already computes Brier score, pinball loss, and interval
  coverage. Someone on this project understands calibration.
- The indicator weights (`INDICATOR_WEIGHTS`) are hand-set constants that a
  logistic regression would fit in one line.

**The change:** treat each indicator's `SignalResult` as a *feature*, not a vote.
Fit a logistic regression (or isotonic calibration on the existing weighted
score) on walk-forward folds, predicting `P(return > cost | features)`. Report
that. Then:

- "Directional Agreement: 67%" becomes "Probability of profit: 58% (±6%)."
- The weights become *learned* rather than guessed, and the Advanced Settings
  sliders become an expert override with a visible baseline.
- The Backtest page can show a **reliability diagram** — the single most
  persuasive artifact a signal product can display.
- `ml/benchmark.py`'s Brier score generalizes from a TimesFM-only metric to the
  headline quality metric for the whole app.

This is the difference between "a dashboard that combines indicators" and "a
research tool that tells you how much to trust it." The ROADMAP's stated product
direction — *"What uncertainty and downside remain"* — is unreachable without it.

### 5.3 Give the backtester a position-sizing and portfolio abstraction

`backtesting/engine.py` hard-codes: 100% of capital per trade, one position at a
time, fixed `horizon_days` holding period, no stops, no shorting costs, sequential
non-overlapping trades. `compute_metrics` then annualizes Sharpe from observed
trade frequency while `exposure_pct` shows time in market is often well under
100% — so the Sharpe is being scaled by a frequency that doesn't reflect capital
at risk. It is not comparable to any published Sharpe.

Meanwhile `simulation/engine.py` — the newest code — already has a proper
`PortfolioState` with cash, quantity, and fees, and a correct next-open fill
model.

**Fix:** the simulation engine's portfolio model is the good one. Lift
`PortfolioState`, `PendingOrder`, and the fill logic into a shared
`portfolio/` package and have the backtester drive it, with a pluggable
`PositionSizer` (fixed-fraction, volatility-targeted, Kelly-capped) and
`ExitRule` (fixed-horizon, trailing stop, signal-reversal). The two engines
converge instead of diverging, and the backtest becomes comparable to the
simulator — which today it is not.

### 5.4 Make the simulation engine's design the house style

`simulation/` is the strongest code in the repo and should be the template:

- Frozen dataclasses with `__post_init__` validation.
- An append-only `SimulationEvent` log with sequence numbers — genuinely
  auditable.
- `_fingerprint_market_data` via `hash_pandas_object` to guarantee a prepared
  strategy matches its data.
- An explicit, documented causal invariant (signal after close → fill at next
  open).
- Precomputed signals for deterministic replay.

Compare `backtesting/report.py`: a mutable dataclass with 12 default-zero fields
that `compute_metrics` fills in by mutation. Same domain, much weaker design.

**Fix:** port the fingerprint + event-log + frozen-result pattern to
`BacktestReport`. A backtest run should be as replayable and auditable as a
simulation run. That is also what makes results *shareable*, which is a product
feature.

### 5.5 Reference-data alignment deserves a first-class abstraction

`indicators/_utils.align_to_index` does timezone stripping, normalization to
midnight, reindex-with-ffill, and index restoration — correct, and used by three
indicator families. But it conflates *observation date* with *availability date*.

For FRED this matters enormously: initial jobless claims for week ending Jan 6
are published Jan 11. Aligning the value to Jan 6 is a five-day look-ahead. The
project correctly marks `FREDMacroRegime.backtest_safe = False` and excludes it
— but that means an entire indicator category is simply unavailable for
backtesting, rather than being correctly lagged.

**Fix:** add a release-lag parameter:

```python
def align_to_index(series, target_index, *, release_lag: pd.Timedelta = pd.Timedelta(0)):
    series.index = series.index + release_lag
    ...
```

FRED's API exposes real-time vintages (`ALFRED`) if you want to be exact, but
even a conservative constant lag per series turns `FRED Macro` from
backtest-excluded into backtest-*valid*. Same for any future fundamental data.
This unlocks a whole indicator category.

---

## 6. Prioritized plan

### Phase 1 — Stop the bleeding (~1 week)

| # | Item | § |
|---|---|---|
| 1 | Fix the failing `AppTest` smoke test | 2.1 |
| 2 | Add `pyproject.toml` (ruff + mypy) and a GitHub Actions workflow | 3.4 |
| 3 | Pin dependencies; `git rm -r --cached` committed caches | 3.4 |
| 4 | Add the causality conformance test for every `backtest_safe` indicator | 3.3 |
| 5 | Fix the redundant TimesFM inferences in the MTF cards | 2.2 |

### Phase 2 — Consolidate (~2 weeks)

| # | Item | § |
|---|---|---|
| 6 | Extract one `ScoringPolicy` + `score()`; delete `_combine_at` | 3.1 |
| 7 | Normalize index handling in `BaseIndicator`; kill the `-1` wraparound | 2.3 |
| 8 | Collapse the `*_for_horizon` method pairs; drop `precomputed` | 4.4 |
| 9 | Make indicators stateless; class-level `name`/`category`/`lookback` | 3.2 |
| 10 | One `Period` dataclass; one cache decorator; one signal card component | 4.1–4.3, 3.5 |

### Phase 3 — Performance (~1 week)

| # | Item | § |
|---|---|---|
| 11 | Stride + vectorize Bubble Risk and Market Correlation | 5.1 |
| 12 | Use the already-declared `scipy`; delete the hand-rolled normal CDF | 5.1 |
| 13 | Parallelize the Screener across tickers | 5.1 |

### Phase 4 — Product (~1 month)

| # | Item | § |
|---|---|---|
| 14 | Calibrated probability output + reliability diagram | 5.2 |
| 15 | Shared `portfolio/` package; position sizing + exit rules | 5.3 |
| 16 | Port simulation's audit/fingerprint pattern to the backtester | 5.4 |
| 17 | Release-lag alignment; make FRED backtest-valid | 5.5 |
| 18 | `st.navigation` for per-page URLs and shareable results | 4.2 |

---

## 7. What not to change

- The indicator ABC + registry decorator pattern. It works, it is well
  documented in the README's "Adding Your Own Indicators" section, and it is the
  right extensibility seam.
- The `simulation/` package's design. Extend it; don't rewrite it.
- The honesty of the UI copy. *"not a probability of success"*, *"Short positions
  are modeled without borrow costs"*, the excluded-indicator warnings — this
  builds trust and is rare in this product category. Keep it as the numbers get
  better.
- The ROADMAP's practice of marking items complete only when actually done.
