"""Portable, checksummed research archives. No pickle or executable payloads."""

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib import metadata
from io import BytesIO
import json
import math
from pathlib import Path
import platform
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1


def _json_value(value):
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, pd.Series):
        return _frame_payload(value.to_frame(name=value.name or "value"))
    if isinstance(value, pd.DataFrame):
        return _frame_payload(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _frame_payload(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique or frame.index.hasnans:
        raise ValueError("Archive frames require unique valid datetime indices")
    if not frame.columns.is_unique or not all(isinstance(name, str) for name in frame.columns):
        raise ValueError("Archive frames require unique string columns")
    return {
        "index": [stamp.isoformat() for stamp in frame.index],
        "timezone": str(frame.index.tz) if frame.index.tz is not None else None,
        "index_name": frame.index.name,
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "values": _json_value(list(frame.itertuples(index=False, name=None))),
    }


def _read_frame(payload):
    tz = payload["timezone"]
    index = pd.to_datetime(payload["index"], utc=bool(tz))
    if tz:
        index = index.tz_convert(tz)
    index.name = payload["index_name"]
    frame = pd.DataFrame(payload["values"], index=index, columns=payload["columns"])
    return frame.astype(dict(zip(payload["columns"], payload["dtypes"])))


def _json_bytes(value):
    return json.dumps(_json_value(value), sort_keys=True, allow_nan=False, indent=2).encode()


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "app.py"]
    for directory in ("backtesting", "charts", "config", "data", "indicators", "ml",
                      "portfolio", "research", "signals", "simulation", "ui", "scripts"):
        paths.extend((root / directory).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def build_archive(kind: str, frames: dict[str, pd.DataFrame], configuration: dict, result) -> bytes:
    """Store exact numeric inputs/results, package versions, source identity and hashes."""
    members = {f"frames/{name}.json": _json_bytes(_frame_payload(frame)) for name, frame in frames.items()}
    if any(not name.replace("_", "").isalnum() for name in frames):
        raise ValueError("Frame names must be alphanumeric with optional underscores")
    members["configuration.json"] = _json_bytes(configuration)
    members["result.json"] = _json_bytes(result)
    versions = {}
    for name in ("numpy", "pandas", "ta", "scipy", "streamlit", "pyarrow", "yfinance", "timesfm", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema_version": SCHEMA_VERSION, "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "packages": versions,
        "source_fingerprint": source_fingerprint(),
        "sha256": {name: hashlib.sha256(value).hexdigest() for name, value in members.items()},
    }
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
        archive.writestr("manifest.json", _json_bytes(manifest))
    return output.getvalue()


def load_archive(data: bytes) -> dict:
    """Verify all member hashes before reading; never extract files to disk."""
    with ZipFile(BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        if sum(info.file_size for info in archive.infolist()) > 256 * 1024 * 1024:
            raise ValueError("Research archive exceeds the 256 MiB read limit")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported research archive schema")
        if set(names) != {"manifest.json", *manifest["sha256"]}:
            raise ValueError("Archive members do not match the manifest")
        members = {}
        for name, expected in manifest["sha256"].items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            members[name] = json.loads(raw)
    return {
        "manifest": manifest,
        "configuration": members["configuration.json"],
        "result": members["result.json"],
        "frames": {name[7:-5]: _read_frame(value) for name, value in members.items() if name.startswith("frames/")},
    }


def build_backtest_archive(report, raw: pd.DataFrame) -> bytes:
    from backtesting.engine import _validate_data
    data = _validate_data(raw)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()
    if fingerprint != report.data_fingerprint:
        raise ValueError("Market data does not match the completed backtest")
    result = {field.name: getattr(report, field.name) for field in fields(report)
              if field.name not in {"computed_data", "configuration"}}
    frames = {"market": data}
    if not report.computed_data.empty:
        frames["computed"] = report.computed_data
    return build_archive("backtest", frames, report.configuration, result)


def replay_backtest_archive(data: bytes):
    """Replay fills from archived computations without network or model inference."""
    from backtesting.engine import run_backtest
    from indicators.registry import get_indicator
    from ml.timesfm_runtime import TimesFMRuntime, TimesFMRuntimeConfig
    from signals.combiner import ScoringPolicy
    saved = load_archive(data)
    if saved["manifest"]["kind"] != "backtest":
        raise ValueError("Expected a backtest archive")
    if saved["manifest"]["source_fingerprint"] != source_fingerprint():
        raise ValueError("Source code differs from the archived run; use the original code to replay")
    config = saved["configuration"]
    indicators = {name: get_indicator(name) for name, _ in config["scoring"]["weights"]}
    for name, runtime_config in config["models"].items():
        indicators[name].runtime = TimesFMRuntime(TimesFMRuntimeConfig(**runtime_config))
    computed = saved["frames"]["computed"]
    if "TimesFM Forecast" in indicators:
        computed.attrs["timesfm_horizon"] = config["horizon_bars"]
    return run_backtest(
        saved["frames"]["market"], indicators, ticker=config["ticker"], period=config["period"],
        horizon_days=config["horizon_bars"], initial_capital=config["initial_capital"],
        cost_per_trade_pct=config["round_trip_cost_pct"], evaluation_start=pd.Timestamp(config["evaluation_start"]),
        indicator_parameters=config["indicator_parameters"],
        scoring_policy=ScoringPolicy(**config["scoring"]), precomputed_data=computed,
    )
