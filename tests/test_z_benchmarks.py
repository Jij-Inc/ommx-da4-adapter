import itertools

import pytest
from ommx import ProvenanceKind, SpecialConstraintKind, State
from ommx.adapter import AdapterNotApplicableError

from benchmarks.common import build_response, prepare_instance
from benchmarks.instance import (
    build_assignment_instance,
    build_clique_instance,
    build_feasible_entries,
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
    ],
)
def test_benchmark_instances_are_accepted(builder, size, formulation) -> None:
    instance = builder(size=size, seed=0, formulation=formulation)

    assert OMMXDA4Adapter.check_applicability(instance).is_applicable
    OMMXDA4Adapter(instance)


def test_overlapping_one_hot_benchmark_exercises_both_da4_paths() -> None:
    instance = build_assignment_instance(size=3, formulation="one-hot")
    before = instance.to_v2_bytes()

    adapter = OMMXDA4Adapter(instance)
    request = adapter.sampler_input

    assert instance.to_v2_bytes() == before
    assert adapter._one_hot_dict == {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
    }
    assert adapter._penalty_one_hot_dict == {
        3: [0, 3, 6],
        4: [1, 4, 7],
        5: [2, 5, 8],
    }
    assert request.fujitsuDA3.one_way_one_hot_groups == {"numbers": [3, 3, 3]}
    assert request.penalty_binary_polynomial is not None


@pytest.mark.parametrize(
    ("special_constraints", "expected_provenance"),
    [
        ("indicator", {ProvenanceKind.IndicatorConstraint}),
        ("sos1", {ProvenanceKind.Sos1Constraint}),
        (
            "indicator-sos1",
            {
                ProvenanceKind.IndicatorConstraint,
                ProvenanceKind.Sos1Constraint,
            },
        ),
    ],
)
def test_preparation_benchmark_lowers_only_requested_special_constraints(
    special_constraints,
    expected_provenance,
) -> None:
    instance = build_one_hot_preparation_instance(
        size=3,
        formulation="one-hot",
        special_constraints=special_constraints,
    )
    before = instance.to_v2_bytes()

    assert not OMMXDA4Adapter.check_applicability(instance).is_applicable
    with pytest.raises(AdapterNotApplicableError):
        OMMXDA4Adapter(instance)

    prepared = prepare_instance(instance)
    provenance_kinds = {
        constraint.provenance[-1].kind
        for constraint in prepared.constraints.values()
        if constraint.provenance
    }

    assert instance.to_v2_bytes() == before
    assert prepared.indicator_constraints == {}
    assert prepared.sos1_constraints == {}
    assert set(prepared.one_hot_constraints) == {0, 1, 2}
    assert prepared.active_special_constraint_kinds == {SpecialConstraintKind.OneHot}
    assert provenance_kinds == expected_provenance
    assert OMMXDA4Adapter.check_applicability(prepared).is_applicable
    OMMXDA4Adapter(prepared)


@pytest.mark.parametrize(
    "special_constraints",
    ["none", "indicator", "sos1", "indicator-sos1"],
)
def test_preparation_benchmark_feasible_region_is_consistent(
    special_constraints,
) -> None:
    size = 2
    instance = build_one_hot_preparation_instance(
        size=size,
        formulation="one-hot",
        special_constraints=special_constraints,
    )

    feasible_states = []
    for values in itertools.product((0, 1), repeat=size * size):
        solution = instance.evaluate(
            State({variable_id: value for variable_id, value in enumerate(values)})
        )
        if solution.feasible:
            feasible_states.append(values)

    assert feasible_states == [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]


def test_synthetic_response_decodes_to_a_feasible_solution() -> None:
    size = 3
    instance = build_tsp_instance(size=size, formulation="one-hot")
    adapter = OMMXDA4Adapter(instance)
    response = build_response(adapter, "tsp", size, sample_count=16)

    solution = adapter.decode(response)

    assert solution.feasible
    assert solution.state.entries == {
        variable_id: float(value)
        for variable_id, value in build_feasible_entries("tsp", size).items()
    }
