from __future__ import annotations

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


def test_synthetic_response_decodes_to_a_feasible_solution():
    size = 3
    adapter = OMMXDA4Adapter(build_tsp_instance(size, formulation="one-hot"))
    response = build_response(adapter, "tsp", size, sample_count=4)

    solution = adapter.decode(response)

    assert solution.feasible
    assert len(solution.decision_variables) == size**2
