from __future__ import annotations

import itertools

import pytest

from benchmarks.common import build_response
from benchmarks.instance import (
    build_assignment_instance,
    build_clique_instance,
    build_knapsack_instance,
    build_one_hot_preparation_instance,
    build_tsp_instance,
)
from ommx_da4_adapter import OMMXDA4Adapter


@pytest.mark.parametrize(
    ("builder", "size", "formulation"),
    [
        (build_knapsack_instance, 4, "regular"),
        (build_assignment_instance, 3, "regular"),
        (build_assignment_instance, 3, "one-hot"),
        (build_tsp_instance, 3, "regular"),
        (build_tsp_instance, 3, "one-hot"),
        (build_clique_instance, 4, "regular"),
        (build_one_hot_preparation_instance, 3, "one-hot"),
    ],
)
def test_benchmark_instances_are_accepted(builder, size, formulation):
    instance = builder(size, formulation=formulation)

    assert OMMXDA4Adapter(instance).sampler_input is not None


def test_overlapping_one_hot_benchmark_uses_native_and_penalty_paths():
    size = 3
    adapter = OMMXDA4Adapter(build_assignment_instance(size, formulation="one-hot"))
    request = adapter.sampler_input

    assert request.fujitsuDA3.one_way_one_hot_groups == {"numbers": [size] * size}
    assert request.penalty_binary_polynomial is not None


def test_one_hot_preparation_baseline():
    size = 3
    instance = build_one_hot_preparation_instance(size)

    assert len(instance.decision_variables) == size**2
    assert len(instance.constraints) == size
    assert instance.constraint_hints is not None
    assert len(instance.constraint_hints.one_hot_constraints) == size


@pytest.mark.parametrize(
    ("special_constraints", "indicator_count", "sos1_count"),
    [
        ("indicator", 4, 0),
        ("sos1", 0, 4),
        ("indicator-sos1", 2, 2),
    ],
)
def test_one_hot_preparation_special_workload_is_lowered_to_regular_constraints(
    special_constraints: str,
    indicator_count: int,
    sos1_count: int,
):
    size = 4
    instance = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
    )
    request = OMMXDA4Adapter(instance).sampler_input

    assert len(instance.decision_variables) == size**2
    assert len(instance.constraints) == size * 2
    assert [constraint.name for constraint in instance.constraints].count(
        "lowered-indicator"
    ) == indicator_count
    assert [constraint.name for constraint in instance.constraints].count(
        "lowered-sos1"
    ) == sos1_count
    assert instance.constraint_hints is not None
    assert len(instance.constraint_hints.one_hot_constraints) == size
    assert request.fujitsuDA3.one_way_one_hot_groups == {"numbers": [size] * size}
    assert request.inequalities is not None
    assert len(request.inequalities) == size


@pytest.mark.parametrize(
    "special_constraints",
    ["indicator", "sos1", "indicator-sos1"],
)
def test_one_hot_preparation_special_workload_preserves_feasibility_and_objective(
    special_constraints: str,
):
    size = 2
    baseline = build_one_hot_preparation_instance(size)
    workload = build_one_hot_preparation_instance(
        size,
        special_constraints=special_constraints,
    )

    for values in itertools.product((0, 1), repeat=size**2):
        state = dict(enumerate(values))
        baseline_solution = baseline.evaluate(state)
        workload_solution = workload.evaluate(state)

        assert workload_solution.feasible == baseline_solution.feasible
        assert workload_solution.objective == pytest.approx(baseline_solution.objective)


def test_synthetic_response_decodes_to_a_feasible_solution():
    size = 3
    adapter = OMMXDA4Adapter(build_tsp_instance(size, formulation="one-hot"))
    response = build_response(adapter, "tsp", size, sample_count=4)

    solution = adapter.decode(response)

    assert solution.feasible
    assert len(solution.decision_variables) == size**2
