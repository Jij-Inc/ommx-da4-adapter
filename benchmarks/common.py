from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from ommx.v1 import Instance

from benchmarks.instance import (
    build_assignment_instance,
    build_clique_instance,
    build_feasible_entries,
    build_knapsack_instance,
    build_one_hot_preparation_instance,
    build_tsp_instance,
)
from ommx_da4_adapter import OMMXDA4Adapter
from ommx_da4_adapter.models import (
    QuboResponse,
    QuboSolution,
    QuboSolutionList,
    SolverTiming,
)

INSTANCE_BUILDERS = {
    "knapsack": build_knapsack_instance,
    "assignment": build_assignment_instance,
    "tsp": build_tsp_instance,
    "clique": build_clique_instance,
    "one-hot-preparation": build_one_hot_preparation_instance,
}
INSTANCE_NAMES = tuple(INSTANCE_BUILDERS)
FORMULATIONS = ("regular", "one-hot")
SPECIAL_CONSTRAINT_CASES = ("none", "indicator", "sos1", "indicator-sos1")
PACKAGE_VERSIONS = (
    version("ommx"),
    version("pydantic"),
    version("ommx_da4_adapter"),
)


@dataclass(frozen=True)
class BenchmarkOperation:
    """Separate per-sample setup from the operation being measured."""

    setup: Callable[[], Any]
    run: Callable[[Any], Any]


def build_instance(
    name: str,
    size: int,
    seed: int,
    formulation: str,
    special_constraints: str = "none",
) -> Instance:
    """Select and build a benchmark Instance."""
    if special_constraints != "none":
        raise ValueError("OMMX v2 does not support Indicator or SOS1 constraints")
    return INSTANCE_BUILDERS[name](size, seed, formulation)


def build_response(
    adapter: OMMXDA4Adapter,
    name: str,
    size: int,
    sample_count: int,
) -> QuboResponse:
    """Build a deterministic feasible DA4 response outside the measured call."""
    if sample_count < 1:
        raise ValueError("sample-count must be at least 1")
    entries = build_feasible_entries(name, size)
    configuration = {
        str(adapter._variable_map[variable_id]): bool(value)
        for variable_id, value in entries.items()
    }
    return QuboResponse(
        qubo_solution=QuboSolutionList(
            progress=[],
            result_status=True,
            solutions=[
                QuboSolution(
                    energy=0,
                    penalty_energy=0,
                    frequency=sample_count,
                    configuration=configuration,
                )
            ],
            timing=SolverTiming(solve_time="0", total_elapsed_time="0"),
        ),
        status="Done",
    )


def make_benchmark_operation(
    operation: str,
    instance: Instance,
    instance_name: str,
    size: int,
    sample_count: int,
) -> BenchmarkOperation:
    """Prepare everything outside the measured operation."""
    if operation == "instance-to-request":
        return BenchmarkOperation(
            setup=lambda: instance,
            run=lambda target: OMMXDA4Adapter(target).sampler_input,
        )

    adapter = OMMXDA4Adapter(instance)
    response = build_response(adapter, instance_name, size, sample_count)
    return BenchmarkOperation(
        setup=lambda: response,
        run=lambda target: adapter.decode(target),
    )


def preparation_name(special_constraints: str) -> str:
    return "none"
