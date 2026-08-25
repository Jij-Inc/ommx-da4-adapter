from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from ommx import Instance

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
}
PREPARATION_INSTANCE_NAME = "one-hot-preparation"
INSTANCE_NAMES = (*INSTANCE_BUILDERS, PREPARATION_INSTANCE_NAME)
FORMULATIONS = ("regular", "one-hot")
SPECIAL_CONSTRAINT_CASES = ("none", "indicator", "sos1", "indicator-sos1")
PREPARATIONS = ("none", "recommended")
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
    preparation: str = "none",
) -> Instance:
    """Select and build a benchmark Instance."""
    if name == PREPARATION_INSTANCE_NAME:
        return build_one_hot_preparation_instance(
            size,
            seed,
            formulation,
            special_constraints,
            preparation,
        )
    if special_constraints != "none":
        raise ValueError(
            "Special constraints are available only for one-hot-preparation"
        )
    if preparation != "none":
        raise ValueError(
            "Preparation is available only for one-hot-preparation special constraints"
        )
    return INSTANCE_BUILDERS[name](size, seed, formulation)


def prepare_instance(instance: Instance) -> Instance:
    """Copy and prepare an Instance without mutating the benchmark source."""
    input_class = OMMXDA4Adapter.INPUT_CLASS
    if input_class is None:
        raise RuntimeError("The adapter does not declare INPUT_CLASS")
    prepared = copy.copy(instance)
    prepared.prepare(
        input_class,
        OMMXDA4Adapter.recommended_preparation_policy(),
    )
    return prepared


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
    special_constraints: str,
    preparation: str,
) -> BenchmarkOperation:
    """Prepare everything outside the measured operation."""
    if operation == "prepare":
        if special_constraints == "none" or preparation != "recommended":
            raise ValueError(
                "prepare requires Indicator and/or SOS1 constraints with "
                "recommended preparation"
            )
        input_class = OMMXDA4Adapter.INPUT_CLASS
        if input_class is None:
            raise RuntimeError("The adapter does not declare INPUT_CLASS")

        def setup_preparation() -> tuple[Instance, Any]:
            return (
                copy.copy(instance),
                OMMXDA4Adapter.recommended_preparation_policy(),
            )

        def run_preparation(context: tuple[Instance, Any]) -> Instance:
            prepared, policy = context
            prepared.prepare(input_class, policy)
            return prepared

        return BenchmarkOperation(setup=setup_preparation, run=run_preparation)

    adapter_instance = (
        prepare_instance(instance) if preparation == "recommended" else instance
    )
    if operation == "instance-to-request":
        return BenchmarkOperation(
            setup=lambda: adapter_instance,
            run=lambda target: OMMXDA4Adapter(target).sampler_input,
        )

    adapter = OMMXDA4Adapter(adapter_instance)
    response = build_response(adapter, instance_name, size, sample_count)
    return BenchmarkOperation(
        setup=lambda: response,
        run=lambda target: adapter.decode(target),
    )
