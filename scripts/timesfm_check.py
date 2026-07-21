"""Inspect the TimesFM runtime and optionally perform a model smoke test."""

import argparse
from dataclasses import asdict
import json

import numpy as np

from ml.timesfm_runtime import TimesFMRuntime, TimesFMRuntimeConfig
from ml.timesfm_autotune import autotune_runtime
from ml.timesfm_profiles import (
    PROFILE_PREFERENCES,
    USE_CASES,
    detect_hardware,
    select_runtime_profile,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--profile", choices=PROFILE_PREFERENCES, default="auto")
    parser.add_argument("--use-case", choices=USE_CASES, default="interactive")
    parser.add_argument("--smoke", action="store_true", help="Load weights and forecast")
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Load weights and empirically probe safe workload chunk throughput",
    )
    args = parser.parse_args()

    hardware = detect_hardware()
    if args.device == "cpu":
        hardware = type(hardware)("cpu", available_ram_gb=hardware.available_ram_gb)
    elif args.device == "cuda" and hardware.device != "cuda":
        print(json.dumps({"error": "CUDA requested but no CUDA device was detected"}, indent=2))
        return 1
    profile = select_runtime_profile(hardware, args.use_case, args.profile)
    runtime = TimesFMRuntime(TimesFMRuntimeConfig.from_profile(profile))
    status = runtime.preflight()
    print(
        json.dumps(
            {
                "hardware": asdict(hardware),
                "profile": asdict(profile),
                "status": asdict(status),
            },
            indent=2,
        )
    )
    if status.state != "ready":
        return 1
    if args.smoke:
        forecast = runtime.forecast(
            [np.linspace(100.0, 110.0, 128, dtype=np.float32)], horizon=5
        )[0]
        print(
            json.dumps(
                {
                    "point": forecast.point.tolist(),
                    "q10": forecast.quantiles[0.1].tolist(),
                    "q50": forecast.quantiles[0.5].tolist(),
                    "q90": forecast.quantiles[0.9].tolist(),
                },
                indent=2,
            )
        )
    if args.autotune:
        result = autotune_runtime(runtime)
        print(json.dumps({"autotune": result.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
