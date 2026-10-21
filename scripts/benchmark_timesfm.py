"""Run a rolling, point-in-time TimesFM benchmark for a market symbol."""

import argparse
import json
from pathlib import Path
import pandas as pd

from data.fetcher import fetch_ohlcv
from ml.benchmark import benchmark_close_series
from ml.timesfm_runtime import BACKEND_MODEL_IDS, TimesFMRuntime, TimesFMRuntimeConfig
from research.artifacts import build_archive
from research.data import completed_daily_bars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="Yahoo Finance symbol, for example AAPL")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--step", type=int)
    parser.add_argument("--min-context", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--backend", choices=BACKEND_MODEL_IDS, default="2.5")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-start", required=True, help="First held-out forecast origin (ISO date)")
    parser.add_argument("--archive", type=Path, required=True, help="New .zip research archive")
    args = parser.parse_args()

    destinations = [args.archive] + ([args.output] if args.output is not None else [])
    if len({path.resolve() for path in destinations}) != len(destinations):
        parser.error("Archive and JSON output must be different paths")
    for path in destinations:
        if path.exists() or path.is_symlink():
            parser.error(f"Output already exists: {path}; choose a new path")
        if not path.parent.is_dir():
            parser.error(f"Output directory does not exist: {path.parent}")
    frame = completed_daily_bars(fetch_ohlcv(args.ticker, period=args.period))
    prices = frame["Close"]
    test_start = pd.Timestamp(args.test_start)
    if prices.index.tz is not None and test_start.tzinfo is None:
        test_start = test_start.tz_localize(prices.index.tz)
    runtime = TimesFMRuntime(TimesFMRuntimeConfig(device=args.device, backend=args.backend))
    result = benchmark_close_series(
        prices,
        runtime,
        horizon=args.horizon,
        min_context=args.min_context,
        step=args.step,
        batch_size=args.batch_size,
        evaluation_start=test_start,
    )
    with args.archive.open("xb") as output:
        output.write(build_archive("forecast_benchmark", {"market": frame},
                                  {"ticker": args.ticker, "period": args.period, **result.configuration}, result))
    payload = json.dumps(result.to_dict(), indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
