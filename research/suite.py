"""Sequential, bounded research jobs over frozen local inputs.

Each worker has its own process, artifacts and logs. Forecast jobs only use
cached weights; the suite never fetches market data or selects winning settings.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pandas as pd

from research.artifacts import _json_value, build_archive, load_archive, source_fingerprint
from research.data import completed_daily_bars


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(_json_value(value), output, indent=2, allow_nan=False)
        output.write("\n")


def validate_protocol(protocol):
    allowed = {"assets", "horizons", "mode", "test_start", "as_of", "cost_pct",
               "max_jobs", "job_timeout_seconds", "total_timeout_seconds", "runtime", "min_context"}
    if not isinstance(protocol, dict) or set(protocol) - allowed:
        raise ValueError("Protocol must be an object with only documented fields")
    p = {"mode": "strategies", "cost_pct": 0.1, "max_jobs": 12,
         "job_timeout_seconds": 600, "total_timeout_seconds": 600,
         "runtime": {"device": "cpu", "max_context": 128, "max_horizon": 30,
                     "batch_size": 1, "chunk_size": 1, "torch_compile": False},
         "min_context": 128, **protocol}
    if p["mode"] not in {"strategies", "forecast", "both", "models"}:
        raise ValueError("mode must be strategies, forecast, both or models")
    for key in ("test_start", "as_of"):
        if not isinstance(p.get(key), str) or pd.isna(pd.Timestamp(p[key])):
            raise ValueError(f"{key} must be an explicit ISO timestamp")
    if pd.Timestamp(p["as_of"]).tzinfo is None:
        raise ValueError("as_of must include a timezone")
    for key in ("max_jobs", "min_context"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if p["max_jobs"] > 100 or p["min_context"] < 32:
        raise ValueError("At most 100 jobs; min_context must be at least 32")
    for key in ("job_timeout_seconds", "total_timeout_seconds"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or not 0 < p[key] <= 86400:
            raise ValueError(f"{key} must be positive and at most 86400")
    if type(p["cost_pct"]) not in (int, float) or not math.isfinite(p["cost_pct"]) or not 0 <= p["cost_pct"] < 100:
        raise ValueError("cost_pct must be finite and between 0 and 100")
    if not isinstance(p.get("assets"), list) or not p["assets"]:
        raise ValueError("assets must be a nonempty list of ticker/input objects")
    for asset in p["assets"]:
        if not isinstance(asset, dict) or set(asset) != {"ticker", "input"}:
            raise ValueError("Each asset requires exactly ticker and input")
        if not all(isinstance(value, str) and value.strip() for value in asset.values()):
            raise ValueError("Asset ticker and input must be nonempty strings")
        if not Path(asset["input"]).is_file():
            raise ValueError(f"Input does not exist: {asset['input']}")
    if not isinstance(p.get("horizons"), list) or not p["horizons"] or any(
        type(h) is not int or h < 1 for h in p["horizons"]
    ) or len(set(p["horizons"])) != len(p["horizons"]):
        raise ValueError("horizons must contain distinct positive integers")
    jobs = len(p["assets"]) * len(p["horizons"]) * (2 if p["mode"] in {"both", "models"} else 1)
    if jobs > p["max_jobs"]:
        raise ValueError(f"Protocol requests {jobs} jobs, exceeding max_jobs={p['max_jobs']}")
    from ml.timesfm_runtime import TimesFMRuntimeConfig
    runtime = TimesFMRuntimeConfig(**p["runtime"])
    if runtime.device not in {"auto", "cpu", "cuda"} or type(runtime.torch_compile) is not bool:
        raise ValueError("Invalid runtime device or torch_compile")
    if p["mode"] != "strategies" and (max(p["horizons"]) > runtime.max_horizon or p["min_context"] > runtime.effective_context):
        raise ValueError("Runtime context/horizon must cover the requested forecast workload")
    p["runtime"] = asdict(runtime)
    if p["mode"] == "models":
        # Validate both backends before creating files or running either model.
        for backend in ("2.5", "3.0"):
            replace(runtime, backend=backend, model_id=None)
    return p


def run_worker(job_path):
    """Execute one recorded job. Exceptions yield a nonzero CLI exit and log."""
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    frame = load_archive(Path(job["input"]).read_bytes())["frames"]["market"]
    start = pd.Timestamp(job["test_start"])
    if frame.index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(frame.index.tz)
    if job["kind"] == "strategies":
        from research.evaluation import compare_strategies, build_comparison_archive, strategy_summary
        reports = compare_strategies(frame, ticker=job["ticker"], test_start=start,
                                     horizon=job["horizon"], cost_pct=job["cost_pct"])
        archive = build_comparison_archive(frame, reports)
        result = {"strategies": strategy_summary(reports).to_dict("records")}
    elif job["kind"] in {"forecast", "model"}:
        from ml.benchmark import benchmark_close_series
        from ml.timesfm_runtime import TimesFMRuntime, TimesFMRuntimeConfig
        runtime = TimesFMRuntime(TimesFMRuntimeConfig(**job["runtime"]))
        try:
            report = benchmark_close_series(frame["Close"], runtime, horizon=job["horizon"],
                                            min_context=job["min_context"], batch_size=runtime.config.chunk_size,
                                            evaluation_start=start)
            archive = build_archive("forecast_benchmark", {"market": frame},
                                    {"ticker": job["ticker"], **report.configuration}, report)
            result = report.to_dict()
            result.pop("evaluations")  # full per-origin observations remain in the ZIP
            if job["kind"] == "model":
                from indicators.forecast import TimesFMForecast
                from research.evaluation import compare_strategies, strategy_summary
                from research.timesfm_comparison import build_model_archive
                indicator = TimesFMForecast(runtime)
                reports = compare_strategies(
                    frame, ticker=job["ticker"], test_start=start, horizon=job["horizon"],
                    cost_pct=job["cost_pct"], indicators={indicator.name: indicator},
                )
                archive = build_model_archive(frame, report, reports)
                result["strategies"] = strategy_summary(reports).to_dict("records")
        finally:
            runtime.unload()
    else:
        raise ValueError("Unknown job kind")
    with (job_path.parent / "result.zip").open("xb") as output:
        output.write(archive)
    write_json(job_path.parent / "result.json", result)


def _stop_process(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass  # worker exited between the timeout/interrupt and the kill
    process.wait()


def _execute(job_path, timeout):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    with (job_path.parent / "stdout.log").open("x") as stdout, (job_path.parent / "stderr.log").open("x") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "scripts.benchmark_suite", "--worker", str(job_path)],
            cwd=Path(__file__).resolve().parents[1], env=env,
            stdout=stdout, stderr=stderr, start_new_session=os.name == "posix",
        )
        try:
            code = process.wait(timeout=timeout)
            return {"status": "completed" if code == 0 else "failed", "exit_code": code}
        except subprocess.TimeoutExpired:
            _stop_process(process)
            return {"status": "timed_out"}
        except BaseException:
            _stop_process(process)
            raise


def _prepare_jobs(p, root):
    jobs = []
    for index, asset in enumerate(p["assets"]):
        path = Path(asset["input"])
        raw = path.read_bytes()
        if path.suffix.lower() == ".zip":
            frame = load_archive(raw)["frames"]["market"]
        else:
            from io import BytesIO
            frame = pd.read_csv(BytesIO(raw), index_col="Date", parse_dates=["Date"], float_precision="round_trip")
        frame = completed_daily_bars(frame, as_of=p["as_of"])
        snapshot = root / f"input-{index:03d}.zip"
        with snapshot.open("xb") as output:
            output.write(build_archive("suite_input", {"market": frame},
                                       {**asset, "original_sha256": hashlib.sha256(raw).hexdigest(), "as_of": p["as_of"]}, {}))
        kinds = ("strategies", "forecast") if p["mode"] == "both" else (p["mode"],)
        if p["mode"] == "models":
            kinds = ("2.5", "3.0")
        for horizon in p["horizons"]:
            for kind in kinds:
                directory = root / f"job-{len(jobs):03d}"
                directory.mkdir()
                job = {key: p[key] for key in ("test_start", "cost_pct", "runtime", "min_context")}
                job.update(kind=kind, ticker=asset["ticker"], input=str(snapshot), horizon=horizon)
                if p["mode"] == "models":
                    from ml.timesfm_runtime import TimesFMRuntimeConfig
                    job["kind"] = "model"
                    job["runtime"] = asdict(replace(
                        TimesFMRuntimeConfig(**p["runtime"]), backend=kind, model_id=None,
                    ))
                write_json(directory / "job.json", job)
                jobs.append({"job": directory.name, "ticker": asset["ticker"], "kind": job["kind"],
                             "backend": job["runtime"]["backend"] if job["kind"] in {"model", "forecast"} else None,
                             "horizon": horizon, "status": "pending"})
    return jobs


def run_suite(protocol, output_dir):
    p = validate_protocol(protocol)
    root = Path(output_dir).resolve()
    root.mkdir()  # exclusive: never reuse a run directory
    write_json(root / "protocol.json", {**p, "source_fingerprint": source_fingerprint(),
                                       "python": sys.executable, "network": "cached-model-only"})
    try:
        jobs = _prepare_jobs(p, root)
    except (Exception, KeyboardInterrupt) as error:
        summary = {"status": "incomplete", "stage": "input_preparation", "jobs": [],
                   "error": f"{type(error).__name__}: {error}"}
        write_json(root / "summary.json", summary)
        return summary
    # All inputs and settings are captured before any outcome is examined.
    deadline = time.monotonic() + p["total_timeout_seconds"]
    interrupted = False
    for record in jobs:
        directory = root / record["job"]
        remaining = deadline - time.monotonic()
        if interrupted:
            record["status"] = "cancelled"
        elif remaining <= 0:
            record["status"] = "skipped_budget"
        else:
            started = time.monotonic()
            try:
                record.update(_execute(directory / "job.json", min(remaining, p["job_timeout_seconds"])))
                if record["status"] == "completed":
                    # Verify the archive before admitting its summary.
                    load_archive((directory / "result.zip").read_bytes())
                    record["result"] = json.loads((directory / "result.json").read_text())
            except KeyboardInterrupt:
                interrupted = True
                record["status"] = "cancelled"
            except Exception as error:
                record.update(status="failed", error=str(error))
            record["elapsed_seconds"] = time.monotonic() - started
        write_json(directory / "status.json", record)
    summary = {"status": "completed" if all(j["status"] == "completed" for j in jobs) else "incomplete",
               "jobs": jobs, "notes": "No tuning or pooled ranking. Forecast intervals are exploratory and unadjusted; strategy metrics are descriptive."}
    write_json(root / "summary.json", summary)
    if p["mode"] == "models":
        from research.timesfm_comparison import write_comparison_html
        write_comparison_html(root, summary)
    return summary
