"""Run every recommended case serially, using a new process per measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from benchmarks.common import PREPARATIONS, SPECIAL_CONSTRAINT_CASES

# (operation, instance, formulation, special_constraints, preparation, size)
Case = tuple[str, str, str, str, str, int]


def benchmark_cases() -> list[Case]:
    cases: list[Case] = []
    for operation in ("instance-to-request", "response-to-solution"):
        for name, sizes, formulations in (
            ("knapsack", (100, 400, 900), ("regular",)),
            ("assignment", (10, 20, 30), ("regular", "one-hot")),
            ("tsp", (10, 20, 30), ("regular", "one-hot")),
        ):
            for formulation in formulations:
                for size in sizes:
                    cases.append((operation, name, formulation, "none", "none", size))
        for special in SPECIAL_CONSTRAINT_CASES:
            preparations = ("none",) if special == "none" else PREPARATIONS
            for preparation in preparations:
                for size in (10, 20, 30):
                    cases.append(
                        (
                            operation,
                            "one-hot-preparation",
                            "one-hot",
                            special,
                            preparation,
                            size,
                        )
                    )
    if "recommended" in PREPARATIONS:
        for special in SPECIAL_CONSTRAINT_CASES:
            if special != "none":
                for size in (10, 20, 30):
                    cases.append(
                        (
                            "prepare",
                            "one-hot-preparation",
                            "one-hot",
                            special,
                            "recommended",
                            size,
                        )
                    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric", choices=("timing", "memory", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if args.sample_count < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("sample-count/repeat must be positive; warmup must be nonnegative")

    root = Path(__file__).resolve().parents[1]
    cases = benchmark_cases()
    metrics = ("timing", "memory") if args.metric == "both" else (args.metric,)
    # An existing run is never overwritten, including a partial failed run.
    args.output.mkdir(parents=True, exist_ok=False)

    def git(*arguments: str) -> str:
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--porcelain", "--untracked-files=no"),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in distributions()
        },
        "seed": args.seed,
        "sample_count": args.sample_count,
        "unique_solutions": 1,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "metrics": metrics,
        "cases_per_metric": len(cases),
        "cases": cases,
        "completed_measurements": 0,
        "status": "running",
    }
    metadata_path = args.output / "metadata.json"

    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    try:
        for metric in metrics:
            with (args.output / f"{metric}.csv").open("w", newline="") as output:
                writer = None
                for index, case in enumerate(cases, start=1):
                    operation, name, formulation, special, preparation, size = case
                    command = [
                        sys.executable,
                        "-m",
                        f"benchmarks.{metric}",
                        operation,
                        "--instance",
                        name,
                        "--formulation",
                        formulation,
                        "--special-constraints",
                        special,
                        "--preparation",
                        preparation,
                        "--size",
                        str(size),
                        "--seed",
                        str(args.seed),
                        "--sample-count",
                        str(args.sample_count),
                    ]
                    if metric == "timing":
                        command += [
                            "--warmup",
                            str(args.warmup),
                            "--repeat",
                            str(args.repeat),
                        ]
                    result = subprocess.run(
                        command, cwd=root, check=True, capture_output=True, text=True
                    )
                    reader = csv.DictReader(io.StringIO(result.stdout))
                    rows = list(reader)
                    if len(rows) != 1 or reader.fieldnames is None:
                        raise ValueError(
                            f"Expected one CSV result for {case}: {result.stdout}"
                        )
                    if writer is None:
                        writer = csv.DictWriter(
                            output, fieldnames=reader.fieldnames, lineterminator="\n"
                        )
                        writer.writeheader()
                    writer.writerow(rows[0])
                    output.flush()
                    metadata["completed_measurements"] += 1
                    save_metadata()
                    print(f"{metric} {index}/{len(cases)}: {case}", flush=True)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            metadata["stderr"] = error.stderr
        raise
    else:
        metadata["status"] = "complete"
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_metadata()


if __name__ == "__main__":
    main()
