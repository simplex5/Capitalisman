"""Run a bounded strategy/forecast suite from a frozen JSON protocol and local data."""

import argparse
import json
from pathlib import Path

from research.suite import run_suite, run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path, nargs="?", help="JSON protocol; see docs/RESEARCH_WORKFLOW.md")
    parser.add_argument("--output-dir", type=Path, help="New directory (its parent must exist)")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
        return 0
    if args.protocol is None or args.output_dir is None:
        parser.error("protocol and --output-dir are required")
    protocol = json.loads(args.protocol.read_text())
    # Relative inputs resolve against the protocol, regardless of launch cwd.
    for asset in protocol.get("assets", []):
        if isinstance(asset, dict) and isinstance(asset.get("input"), str):
            asset["input"] = str((args.protocol.resolve().parent / asset["input"]).resolve())
    result = run_suite(protocol, args.output_dir)
    paths = {"status": result["status"], "summary": str(args.output_dir / "summary.json")}
    if (args.output_dir / "comparison.html").exists():
        paths["comparison"] = str(args.output_dir / "comparison.html")
    print(json.dumps(paths, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
