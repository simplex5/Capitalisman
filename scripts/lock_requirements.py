"""Convert a clean pip --report resolution into a platform-specific hashed lock."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("requirements-lock.txt"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    lines = [
        "# Generated from a clean Python 3.11 Linux x86_64 pip resolution.",
        "# Install: python -m pip install --require-hashes -r requirements-lock.txt",
        "# Regeneration instructions: docs/RESEARCH_WORKFLOW.md",
    ]
    for item in sorted(report["install"], key=lambda item: item["metadata"]["name"].lower()):
        info = item["metadata"]
        digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
        lines.append(f"{info['name']}=={info['version']} --hash=sha256:{digest}")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
