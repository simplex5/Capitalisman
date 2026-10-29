"""Auditable TimesFM comparisons using the existing forecast and backtest engines."""

from dataclasses import fields
from html import escape

import pandas as pd
import plotly.graph_objects as go

from research.artifacts import build_archive, load_archive
from research.evaluation import strategy_summary


def build_model_archive(frame, forecast, reports):
    frames = {"market": frame}
    results = {}
    for name, report in reports.items():
        frames[f"computed_{name}"] = report.computed_data
        results[name] = {field.name: getattr(report, field.name) for field in fields(report)
                         if field.name != "computed_data"}
    return build_archive("model_comparison", frames, {
        "forecast": forecast.configuration,
        "strategies": {name: report.configuration for name, report in reports.items()},
        "protocol": "Exploratory univariate model comparison; fixed settings, no tuning on outcomes",
    }, {"forecast": forecast.to_dict(), "reports": results,
        "summary": strategy_summary(reports).to_dict("records")})


def write_comparison_html(root, summary):
    """Render local, self-contained charts; failed jobs remain visible in the table."""
    rows = []
    forecasts = go.Figure()
    equity = go.Figure()
    for job in summary["jobs"]:
        row = {"Asset": job["ticker"], "Horizon": job["horizon"],
               "TimesFM": job["backend"], "Status": job["status"],
               "Seconds (including load)": job.get("elapsed_seconds")}
        if job["status"] == "completed":
            saved = load_archive((root / job["job"] / "result.zip").read_bytes())
            forecast = saved["result"]["forecast"]
            strategy = next(s for s in saved["result"]["summary"] if s["strategy"] == "combined")
            row.update({"Origins": forecast["observations"], "Return MAE": forecast["median_return_mae"],
                        "Last-price MAE": forecast["naive_return_mae"],
                        "Direction accuracy": forecast["directional_accuracy"],
                        "80% interval coverage": forecast["interval_80_coverage"],
                        "Backtest return": strategy["return"], "Max drawdown": strategy["max_drawdown"],
                        "Trades": strategy["trades"], "Start": strategy["start"], "End": strategy["end"],
                        "Completion": strategy["completion"]})
            label = f"{job['ticker']} / {job['horizon']} bars / TimesFM {job['backend']}"
            evaluations = forecast["evaluations"]
            forecasts.add_scatter(x=[r["target"] for r in evaluations],
                                  y=[r["median_price"] for r in evaluations], name=label)
            forecasts.add_scatter(x=[r["target"] for r in evaluations],
                                  y=[r["actual_price"] for r in evaluations],
                                  name=label + " actual", line={"dash": "dot"})
            for name in ("combined", "buy_hold", "cash"):
                curve = saved["result"]["reports"][name]["equity_curve"]
                equity.add_scatter(x=curve["index"], y=[value[0] for value in curve["values"]],
                                   name=label + " / " + name)
        rows.append(row)
    forecasts.update_layout(title="Forecast median versus observed target close", yaxis_title="Price")
    equity.update_layout(title="Daily equity after fill costs", yaxis_title="Portfolio equity")
    table = pd.DataFrame(rows).to_html(index=False, escape=True, float_format=lambda x: f"{x:.5f}")
    body = f"""<!doctype html><meta charset="utf-8"><title>TimesFM model comparison</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;font-size:.85rem}}
td,th{{padding:.5rem;border:1px solid #ccc}}.table{{overflow:auto}}</style>
<h1>TimesFM 2.5 vs 3.0 — exploratory comparison</h1>
<p>Status: {escape(summary['status'])}. Both models use the same frozen completed daily bars,
context, horizons and round-trip fill costs. Settings were not tuned on these outcomes.</p>
<p>Forecast test start is an origin date; backtest test start is an execution date.
Read each archive for resolved dates, quantile metrics, inputs, settings and dependencies.
Fractions such as 0.01 represent 1%. Seconds include loading and both evaluations, not pure inference.
Failed or timed-out models have no result; they are never represented as cash-only success.</p>
<p>Small differences are descriptive, not proof of a trading edge. Quantile-derived scores are
uncalibrated. Backtests omit margin/borrow costs; insolvency may shorten a strategy's window.
TimesFM 3.0 downloaded weights are restricted to non-commercial, non-production research.</p>
<div class="table">{table}</div>
{forecasts.to_html(full_html=False, include_plotlyjs=True)}
{equity.to_html(full_html=False, include_plotlyjs=False)}
"""
    with (root / "comparison.html").open("x", encoding="utf-8") as output:
        output.write(body)
