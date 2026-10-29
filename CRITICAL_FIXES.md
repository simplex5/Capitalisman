# Capitalisman Critical Correctness and Hardening Review

## Implementation status

The findings below describe the original review. The first corrective pass has
implemented findings **1** (OBV future-data slice), **3** (non-finite confidence),
**5** (profit factor from compounded dollar P&L), and **12** (period selector and
Simulator UI smoke test). Core technical indicators now have historical-prefix
signal regression tests. Simulator controls use callbacks so a button press
performs its action once, and the UI test exercises step, batch playback, pause,
and reset.

The second pass implements findings **4** (stop at short liquidation and retain
gap debt), **8** (shared pure scorer), **9** (backtest input validation), and
**11** (effective TimesFM costs). Backtesting now records daily portfolio
snapshots and dollar trade P&L, derives drawdown/Sharpe from daily equity, and
shares fill accounting with simulation. Forecast and trade targets agree at
`signal_origin + horizon`. Strategy definitions capture parameters, weights and
horizon adjustments; run/event records include configuration identities.
TimesFM uses symmetric fee hurdles and bounded, explicitly uncalibrated estimates.

The third pass adds a captured minimum evidence threshold, separate agreement
and evidence metrics, registry-wide causal-conformance tests, and explicit
historical opt-in. Cross-asset live feeds are excluded until their historical
availability contract is established. Offline CI, a hashed dependency lock,
checksummed research archives with execution replay, and explicit held-out
strategy/forecast baseline comparisons are now implemented. See
[`docs/RESEARCH_WORKFLOW.md`](docs/RESEARCH_WORKFLOW.md).

The fourth pass adds explicit verified CPU/CUDA loading without process CUDA
environment changes, model release on OOM, cooperative cancellation and bounded
benchmark worker processes. Reference/FRED caches expire and retry failures;
forecast JSON outputs cannot overwrite existing files. Local-input suites retain
failures and add exploratory paired block-bootstrap MAE intervals.

TimesFM calibration, historical external-data availability,
statistical validation on market data, and production execution realism remain open. The short model
uses a daily-bar zero-equity boundary, not a broker-specific margin model.

## Executive summary

Capitalisman has a thoughtful overall structure, particularly in its simulation
package, but its backtest results should not currently be treated as reliable.
Several implementation defects can leak future information, manufacture maximum
confidence from invalid or negligible evidence, and produce financially
impossible performance reports.

The highest-priority work is:

1. Remove the OBV look-ahead leak.
2. Add automated causal-conformance tests for every backtest-safe indicator.
3. Reject non-finite signal values instead of silently promoting them.
4. Require meaningful absolute evidence before emitting BUY or SELL.
5. Replace percentage-only backtest accounting with explicit portfolio
   accounting and insolvency rules.

---

## Critical correctness findings

### 1. OBV leaks future data into historical signals

Location: `indicators/volume.py`, particularly lines 110–138.

The OBV signal derives its lookback from the length of the full DataFrame and
calculates its normalization range with:

```python
obv_slice = df["OBV"].iloc[start_idx:]
```

When the backtester evaluates an earlier bar, this slice continues through every
later bar in the full DataFrame. Future volume therefore changes the confidence
assigned to a historical OBV signal. Because confidence is included in the
weighted scorer, the leak can change whether a trade is generated and can alter
the combined direction.

A causal-equivalence probe reproduced the problem. At the same historical bar,
OBV returned:

- Confidence using the complete series: `0.0017959`
- Confidence using data truncated at the evaluated bar: `0.0086943`

This affects both backtesting and simulation because both precompute indicators
over the complete market-data frame.

#### Proposed fix

Normalize `idx` to a non-negative position, derive the lookback relative to that
position, and end every historical slice at `actual_idx + 1`.

For example:

```python
actual_idx = idx if idx >= 0 else len(df) + idx
lookback = min(10, actual_idx)
start = actual_idx - lookback
obv_slice = df["OBV"].iloc[start : actual_idx + 1]
```

Return HOLD if the normalized position does not contain enough history.

This fix must be accompanied by a regression test that compares the signal
calculated on a full frame at `idx=t` with the signal calculated at `-1` on the
same frame truncated through `t`.

---

### 2. Infinitesimal evidence can produce a 100% agreement signal

Location: `signals/combiner.py`, lines 111–143.

The combiner normalizes confidence only across actionable BUY and SELL weight.
It has no minimum absolute score, minimum evidence threshold, or indicator
quorum.

As a result, this input:

```text
BUY score:  0.000000000001
SELL score: 0
```

produces:

```text
Direction: BUY
Directional agreement: 100%
```

The value is technically the share of actionable weighted votes, but the
categorical BUY or SELL result is not justified by the amount of evidence.
Numerical dust or one extremely weak indicator can initiate a modeled trade.

#### Proposed fix

Separate two concepts:

- **Directional agreement:** how one-sided the actionable evidence is.
- **Evidence strength:** how much credible absolute evidence exists.

Emit HOLD unless the signal satisfies both a directional-separation threshold
and an absolute evidence or quorum threshold. An explicit scoring policy might
contain:

```python
@dataclass(frozen=True)
class ScoringPolicy:
    weights: Mapping[str, float]
    ambiguity_threshold: float
    minimum_directional_score: float
    minimum_actionable_indicators: int
```

The UI should display evidence strength alongside agreement. Longer term, the
headline output should be an empirically calibrated probability rather than a
hand-weighted vote share.

---

### 3. NaN confidence silently becomes maximum confidence

Location: `signals/base.py`, lines 22–23.

`SignalResult.__post_init__` clamps confidence with:

```python
self.confidence = max(0.0, min(1.0, self.confidence))
```

In Python, this converts `float("nan")` to `1.0` in this expression. A broken
indicator can therefore fail open as a maximum-confidence BUY or SELL.
Positive infinity also becomes `1.0`.

#### Proposed fix

Validate before clamping:

```python
value = float(self.confidence)
if not math.isfinite(value):
    raise ValueError("Signal confidence must be finite")
self.confidence = max(0.0, min(1.0, value))
```

For a user-facing analysis pipeline, indicator execution should be isolated so
that one invalid indicator becomes an explicit zero-confidence HOLD/error result
without aborting unrelated indicators. Invalid values must never be silently
converted to strong evidence.

Apply equivalent finite-value validation to:

- Combined confidence and direction scores
- OHLCV prices and volume
- Transaction costs and initial capital
- Forecast outputs
- Trade returns and portfolio equity

---

### 4. The backtester continues trading after insolvency

Locations:

- `backtesting/engine.py`, lines 113–120
- `backtesting/metrics.py`, lines 26–36

A modeled short can lose more than 100%. The resulting return is mechanically
multiplied into equity, making equity negative. The backtester nevertheless
continues processing and compounding subsequent trades.

For example:

```text
Starting equity:  $10,000
First return:        -200%
Equity:            -$10,000
Second return:        -50%
Equity:             -$5,000
Reported return:      -150%
Max drawdown:          200%
```

The second losing trade appears to improve an already insolvent account. This is
not a valid portfolio model.

#### Proposed fix

Use explicit portfolio accounting:

- Cash
- Position quantity
- Entry notional
- Mark-to-market value
- Realized and unrealized P&L
- Fees
- Collateral or margin requirement
- Liquidation state

For an initial safe implementation, either:

1. Disable shorts entirely in the backtester, matching the simulator; or
2. Use a fully specified collateral model and liquidate the run when equity
   reaches zero or the maintenance-margin boundary.

Do not record or process new trades after insolvency. Cap economically realized
long-only loss at available capital.

The simulation engine already has stronger portfolio state, fill handling, and
input validation. Its accounting primitives should become shared infrastructure
used by both simulation and backtesting.

---

### 5. Profit factor is incorrect for a compounded portfolio

Location: `backtesting/metrics.py`, lines 58–61.

Profit factor is currently calculated from summed percentage returns:

```python
gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
```

Standard profit factor uses gross profit dollars divided by gross loss dollars.
Percentage returns cannot be summed when the capital base changes.

For a full-capital sequence of `+100%` followed by `-50%`:

```text
$10,000 -> $20,000 -> $10,000
```

The true dollar profit factor is `10,000 / 10,000 = 1.0`. The current
implementation reports `1.0 / 0.5 = 2.0`.

#### Proposed fix

Record dollar P&L on every trade based on actual portfolio equity and position
size. Calculate:

```python
gross_profit = sum(max(trade.pnl_dollars, 0) for trade in trades)
gross_loss = abs(sum(min(trade.pnl_dollars, 0) for trade in trades))
```

Metrics should be derived from the actual equity ledger, not reconstructed from
isolated percentages.

---

## Safety architecture problems

### 6. Backtest safety fails open

Location: `indicators/base.py`, lines 29–36.

Every new indicator inherits:

```python
def backtest_safe(self) -> bool:
    return True
```

This makes historical safety an unverified honor-system flag and defaults new
implementations to trusted. The OBV defect demonstrates that the declaration is
not sufficient.

#### Proposed fix

- Default `backtest_safe` to `False`.
- Require each causal indicator to opt in explicitly.
- Add a causal-conformance test for every opted-in indicator.
- Exclude an indicator automatically if the conformance test fails.

A representative invariant is:

```python
full = indicator.compute(data)
truncated = indicator.compute(data.iloc[: t + 1])

assert signal(full, idx=t) == signal(truncated, idx=-1)
```

Compare direction, confidence, and every computed indicator value at `t`.
Exercise multiple positions, randomized OHLCV datasets, constant series,
missing-volume cases, extreme gaps, split-like jumps, and zero-volume windows.

Also prohibit common future-looking operations through review and tests,
including centered rolling windows, negative shifts, backward fill, and
full-series statistics used at historical indices.

---

### 7. Historical index handling is unsafe and duplicated

Several crossover indicators use:

```python
prev_idx = idx - 1 if idx != -1 else -2
```

At `idx == 0`, `prev_idx` becomes `-1`, wrapping to the last and therefore future
bar. Warmup conventions currently make this difficult to reach, but correctness
depends on callers in other modules always enforcing those conventions.

#### Proposed fix

Centralize index normalization in `BaseIndicator`:

```python
def normalize_position(df: pd.DataFrame, idx: int) -> int:
    position = idx if idx >= 0 else len(df) + idx
    if position < 0 or position >= len(df):
        raise IndexError(...)
    return position
```

Every indicator should return HOLD when `position < required_lookback` and must
never rely on negative-index wraparound for historical access.

---

### 8. Scoring logic is independently implemented twice

Locations:

- `signals/combiner.py`
- `simulation/strategies.py`, `_combine_at`

Prediction/backtesting and simulation duplicate weighted direction scoring,
timescale adjustment, ambiguity handling, and confidence normalization. Their
policies differ in legitimate ways, but duplicating the mechanism invites
semantic drift.

#### Proposed fix

Extract one pure scorer:

```python
def score_signals(
    signals: Sequence[SignalResult],
    policy: ScoringPolicy,
    horizon_days: int,
) -> CombinedSignal:
    ...
```

Prediction can construct a policy from session settings; simulation can
construct a policy from `StrategyDefinition`. The implementation of score
validation, evidence thresholds, ambiguity, and confidence should exist only
once.

---

### 9. The backtesting engine does not validate its inputs

Unlike the simulation engine, `run_backtest` does not validate:

- Required columns
- Datetime index type
- Sorted and unique timestamps
- Finite values
- Positive entry prices
- Valid high/low relationships
- Initial capital
- Horizon
- Transaction-cost range
- Evaluation bounds

Malformed direct input can produce division by zero, nonsensical returns, or
misordered trades.

#### Proposed fix

Create one shared immutable `MarketData` validation boundary used by prediction,
simulation, and backtesting. Validate immediately after provider ingestion and
again at public engine boundaries.

At minimum:

```text
Open, High, Low, Close: finite
Volume: finite and non-negative
High >= max(Open, Close, Low)
Low <= min(Open, Close, High)
Index: DatetimeIndex, unique, strictly increasing
Initial capital: finite and positive
Horizon: positive integer
Costs: finite and within a declared supported range
```

Whether negative market prices are supported must be an explicit product
decision. If unsupported, reject them with a clear message.

---

## Model and UI correctness

### 10. TimesFM multi-timeframe cards can repeat full inference

Locations:

- `ui/page_predict.py`, lines 276–280
- `indicators/forecast.py`, lines 115–136

The primary horizon is precomputed, but the 1-, 5-, and 20-bar cards call the
combiner again. When a card horizon does not match the stored TimesFM horizon,
`get_signal_for_horizon` runs another complete forecast. Depending on the
selected primary horizon, this can cause multiple expensive inferences during
one display pass.

The indicator also stores mutable `latest_analysis` and `last_error` attributes,
so display behavior depends on call order.

#### Proposed fix

- Make forecast artifacts return values owned by the caller.
- Keep indicators stateless.
- Cache forecasts by market-data fingerprint, model/runtime configuration,
  origin, and horizon.
- Compute all requested horizons explicitly once.
- Allow multi-timeframe cards to exclude indicators that have no cached result
  for that horizon.

---

### 11. TimesFM profitability uses the wrong transaction cost

Location: `indicators/forecast.py`, line 136.

TimesFM determines whether a forecast clears costs using
`DEFAULT_COST_PER_TRADE_PCT`, even when the backtest uses a user-selected cost.
The strategy can therefore authorize a trade using a 0.1% hurdle and then apply
a much larger configured cost to its P&L.

#### Proposed fix

Make trading cost part of the explicit scoring/forecast policy. The same
effective cost must be used for:

- Probability-above-cost calculations
- BUY/SELL decision thresholds
- Backtest execution
- Displayed model explanation

---

### 12. The current end-to-end UI test fails

The test run produced:

```text
237 passed, 1 failed
```

The failing test is:

```text
tests/test_simulator_ui.py::test_simulator_can_create_and_step_a_replay
ValueError: '1y' is not in list
```

Location: `ui/components.py`, lines 43–52.

The selectbox stores raw period values such as `"1y"` while AppTest observes
formatted options such as `"1 Year"`. Navigation to the simulator fails before
the simulator test can execute.

#### Proposed fix

Use a period value object or `(code, label)` representation with an explicit
mapping at the UI boundary. Avoid maintaining parallel period lists and
dictionaries.

The fixed smoke test should run in CI on every change.

---

## Reliability and operational hardening

### 13. One indicator failure can abort an entire analysis

Several pages compute selected indicators inside one broad `try` block. A single
optional, experimental, or malformed indicator can prevent all otherwise valid
results from being displayed.

#### Proposed fix

Execute each indicator through an isolation boundary that returns either:

```python
IndicatorOutcome(result=..., error=None)
```

or:

```python
IndicatorOutcome(
    result=SignalResult(name, HOLD, 0.0, "Explicit failure message"),
    error=structured_error,
)
```

Show partial results with clear exclusions. Log the underlying exception without
silently treating infrastructure failure as a market HOLD.

---

### 14. Caching can permanently preserve transient failures

The cross-asset and FRED caches are process-global dictionaries with no expiry.
They also cache `None`. One temporary provider failure can disable an indicator
until the process restarts.

#### Proposed fix

Use one cache implementation with:

- Bounded size
- Success TTL
- Much shorter failure TTL
- Explicit invalidation
- Provider/configuration-aware keys
- Observable hit, miss, and failure metrics

Do not describe network failure and genuinely neutral market conditions with the
same state.

---

### 15. Watchlist persistence is not safe for multi-user deployment

Location: `data/watchlists.py`.

All sessions share one file:

```text
~/.capitalisman/watchlists.json
```

The implementation performs an unlocked, non-atomic read-modify-write. In a
multi-user deployment:

- Users can see or overwrite shared watchlists.
- Concurrent saves can lose updates.
- Interrupted writes can corrupt the file.
- Invalid element types inside ticker lists are accepted.

#### Proposed fix

For a local-only application, write through a temporary file followed by an
atomic replace and use a lock around updates. Validate names, ticker values, and
size limits.

For a deployed multi-user application, store watchlists under an authenticated
user identity in a transactional database.

---

### 16. Dependency and CI controls are missing

`requirements.txt` contains only lower bounds. There is no lockfile, CI workflow,
lint configuration, or type-checking configuration. The local test run also
reported that installed PyArrow 11.0.0 has a known security issue.

#### Proposed fix

- Add a reproducible lock or constraints file.
- Test the minimum and locked dependency sets.
- Add GitHub Actions or equivalent CI.
- Run pytest, Ruff, a type checker, and dependency/security auditing.
- Add scheduled dependency-update testing.
- Keep TimesFM/CUDA dependencies isolated from the base environment.

---

## Product-level limitations

### Confidence is not calibrated

Indicator confidence multipliers and weights are hand-set. The displayed
agreement explicitly is not a probability of success, which is honest, but it
also means the tool cannot tell a user how much to trust the result.

The existing backtesting and TimesFM benchmark code already contains most of the
data needed to improve this.

#### Proposed direction

Treat indicator outputs as features. Fit and evaluate a probability model using
strict walk-forward folds:

```text
P(net return > transaction cost | point-in-time indicator features)
```

Report:

- Probability of profit
- Calibration/reliability diagram
- Brier score
- Out-of-sample sample count
- Confidence interval
- Performance by asset class and market regime
- Comparison with simple baselines

Do not fit weights on the same interval used to report performance.

---

## Recommended implementation order

### Phase 1: Stop invalid results

1. Fix the OBV future slice and historical index normalization.
2. Add causal-conformance tests for every backtest-safe indicator.
3. Make historical safety opt-in rather than inherited by default.
4. Reject NaN and infinite values at every data-model boundary.
5. Add an evidence floor and quorum to the signal combiner.
6. Disable or correctly collateralize shorts.
7. Stop backtesting immediately at insolvency.
8. Calculate profit factor and returns from dollar portfolio accounting.

### Phase 2: Consolidate correctness

1. Extract one pure scoring implementation with explicit policies.
2. Reuse the simulator's portfolio, fill, fee, and event-ledger design in the
   backtester.
3. Add a shared validated market-data abstraction.
4. Make indicator failures isolated and observable.
5. Make TimesFM artifacts stateless and cost-aware.

### Phase 3: Establish a safety net

1. Fix the failing Streamlit smoke test.
2. Add CI for tests, linting, types, and dependency auditing.
3. Pin a reproducible dependency set.
4. Add property-based tests for causal behavior and accounting invariants.
5. Add golden replay tests proving deterministic simulation and backtest output.

### Phase 4: Improve usefulness

1. Replace raw vote agreement with calibrated out-of-sample probabilities.
2. Add reliability diagrams and baseline comparisons.
3. Add realistic position sizing, margin, borrow costs, and exit policies.
4. Make results reproducible with data, configuration, strategy, and software
   fingerprints.
5. Make caches and persistence safe for the intended deployment model.

---

## Required regression invariants

The following invariants should become executable tests:

1. Adding future bars cannot change an indicator value or signal at an earlier
   bar for any indicator declared backtest-safe.
2. No non-finite value can enter a combined score, trade, metric, or portfolio
   snapshot.
3. Negligible evidence cannot produce an actionable BUY or SELL.
4. Portfolio equity cannot become negative unless a documented margin model
   explicitly permits and handles it.
5. No trade can be opened after liquidation or completion.
6. Every fill occurs strictly after the signal information used to create it was
   available.
7. Profit factor reconciles to the dollar P&L ledger.
8. Batch simulation and repeated single-step simulation produce identical state
   and events.
9. Prediction, backtest, and simulation scoring use the same scorer when given
   the same signals and policy.
10. Changing configured transaction costs changes both trade eligibility and
    realized P&L consistently.

---

## Architectural recommendation

The simulation package is the strongest implementation in the repository. It
already uses validated configuration, explicit state, next-open fills,
market-data fingerprints, and an event ledger. It should become the foundation
for shared execution and accounting.

The desired architecture is:

```text
Validated market data
        |
        v
Causal indicator computation
        |
        v
Finite, validated SignalResult values
        |
        v
One pure scorer + explicit ScoringPolicy
        |
        v
Shared portfolio/fill/accounting engine
        |
        +--> Historical backtest report
        |
        +--> Interactive simulation replay
```

This removes the current semantic drift between prediction, backtesting, and
simulation and provides one place to enforce causality, execution timing,
transaction costs, liquidation, and accounting invariants.
