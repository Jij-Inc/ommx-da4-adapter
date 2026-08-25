from __future__ import annotations

import argparse
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.common import (
    FORMULATIONS,
    INSTANCE_NAMES,
    PACKAGE_VERSIONS,
    SPECIAL_CONSTRAINT_CASES,
    build_instance,
    make_benchmark_operation,
    preparation_name,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("prepare", "instance-to-request", "response-to-solution"),
    )
    parser.add_argument("--instance", choices=INSTANCE_NAMES, default="tsp")
    parser.add_argument("--formulation", choices=FORMULATIONS, default="regular")
    parser.add_argument(
        "--special-constraints", choices=SPECIAL_CONSTRAINT_CASES, default="none"
    )
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=16)
    args = parser.parse_args()

    try:
        import memray  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise SystemExit("Run with `uv run --frozen --with memray`.") from error

    try:
        instance = build_instance(
            args.instance,
            args.size,
            args.seed,
            args.formulation,
            args.special_constraints,
        )
        benchmark = make_benchmark_operation(
            args.operation,
            instance,
            args.instance,
            args.size,
            args.sample_count,
            args.special_constraints,
        )
    except ValueError as error:
        parser.error(str(error))

    gc.collect()
    with TemporaryDirectory() as directory:
        first_capture = Path(directory) / "first.bin"
        first_context = benchmark.setup()
        first_result: Any = None
        with memray.Tracker(first_capture):
            first_result = benchmark.run(first_context)
        del first_result
        del first_context
        first_reader = memray.FileReader(first_capture)
        first_peak_memory_bytes = first_reader.metadata.peak_memory
        first_reader.close()

        gc.collect()
        warmed_capture = Path(directory) / "warmed.bin"
        context = benchmark.setup()
        result: Any = None
        with memray.Tracker(warmed_capture):
            result = benchmark.run(context)
        del result
        del context
        reader = memray.FileReader(warmed_capture)
        peak_memory_bytes = reader.metadata.peak_memory
        reader.close()

    print(
        "operation,instance,formulation,special_constraints,preparation,size,"
        "sample_count,first_peak_memory_bytes,peak_memory_bytes,ommx_version,"
        "pydantic_version,adapter_version"
    )
    print(
        args.operation,
        args.instance,
        args.formulation,
        args.special_constraints,
        preparation_name(args.special_constraints),
        args.size,
        args.sample_count,
        first_peak_memory_bytes,
        peak_memory_bytes,
        *PACKAGE_VERSIONS,
        sep=",",
    )


if __name__ == "__main__":
    main()
