"""Compare fixed strategies on held-out OHLCV data and save a research archive."""

import argparse
from pathlib import Path

import pandas as pd

from research.artifacts import load_archive
from research.data import completed_daily_bars
from research.evaluation import compare_strategies, build_comparison_archive, strategy_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OHLCV CSV with Date column, or an existing research ZIP")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--test-start", required=True, help="First test execution date; earlier rows supply warmup only")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--cost-pct", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True, help="New archive path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve the original run")
    if args.input.suffix.lower() == ".zip":
        frame = load_archive(args.input.read_bytes())["frames"]["market"]
    else:
        frame = pd.read_csv(args.input, index_col="Date", parse_dates=["Date"], float_precision="round_trip")
    frame = completed_daily_bars(frame)
    start = pd.Timestamp(args.test_start)
    if frame.index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(frame.index.tz)
    reports = compare_strategies(frame, ticker=args.ticker, test_start=start,
                                horizon=args.horizon, cost_pct=args.cost_pct)
    with args.output.open("xb") as output:
        output.write(build_comparison_archive(frame, reports))
    print(strategy_summary(reports).to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
