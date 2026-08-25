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
    ("special_constraints", "indicator_count", "sos1_count"),
    [
        ("indicator", 4, 0),
        ("sos1", 0, 4),
        ("indicator-sos1", 2, 2),
    ],
)
def test_direct_and_prepared_cases_have_aligned_active_constraints(
    special_constraints: str,
    indicator_count: int,
    sos1_count: int,
) -> None:
    size = 4
    direct = build_one_hot_preparation_instance(
        size=size,
        formulation="one-hot",
        special_constraints=special_constraints,
        preparation="none",
    )
    source = build_one_hot_preparation_instance(
        size=size,
        formulation="one-hot",
        special_constraints=special_constraints,
        preparation="recommended",
    )
    before = source.to_v2_bytes()

    assert len(direct.constraints) == size
    assert len(direct.one_hot_constraints) == size
    assert direct.indicator_constraints == {}
    assert direct.sos1_constraints == {}
    assert len(source.constraints) == 0
    assert len(source.one_hot_constraints) == size
    assert len(source.indicator_constraints) == indicator_count
    assert len(source.sos1_constraints) == sos1_count
    assert not OMMXDA4Adapter.check_applicability(source).is_applicable
    with pytest.raises(AdapterNotApplicableError):
        OMMXDA4Adapter(source)

    prepared = prepare_instance(source)
    provenance_kinds = {
        constraint.provenance[-1].kind
        for constraint in prepared.constraints.values()
        if constraint.provenance
    }
    expected_provenance = (
        {ProvenanceKind.IndicatorConstraint, ProvenanceKind.Sos1Constraint}
        if indicator_count and sos1_count
        else {
            ProvenanceKind.IndicatorConstraint
            if indicator_count
            else ProvenanceKind.Sos1Constraint
        }
    )

    assert source.to_v2_bytes() == before
    assert len(prepared.constraints) == size
    assert prepared.indicator_constraints == {}
    assert prepared.sos1_constraints == {}
    assert len(prepared.one_hot_constraints) == size
    assert len(prepared.removed_indicator_constraints) == indicator_count
    assert len(prepared.removed_sos1_constraints) == sos1_count
    assert prepared.active_special_constraint_kinds == {SpecialConstraintKind.OneHot}
    assert provenance_kinds == expected_provenance
    assert OMMXDA4Adapter.check_applicability(direct).is_applicable
    assert OMMXDA4Adapter.check_applicability(prepared).is_applicable
    assert (
        OMMXDA4Adapter(direct).sampler_input == OMMXDA4Adapter(prepared).sampler_input
    )


@pytest.mark.parametrize(
    "special_constraints",
    ["indicator", "sos1", "indicator-sos1"],
)
def test_direct_source_and_prepared_cases_have_identical_feasible_states(
    special_constraints: str,
) -> None:
    size = 2
    baseline = build_one_hot_preparation_instance(size=size, formulation="one-hot")
    direct = build_one_hot_preparation_instance(
        size=size,
        formulation="one-hot",
        special_constraints=special_constraints,
        preparation="none",
    )
    source = build_one_hot_preparation_instance(
        size=size,
        formulation="one-hot",
        special_constraints=special_constraints,
        preparation="recommended",
    )
    prepared = prepare_instance(source)

    for values in itertools.product((0, 1), repeat=size * size):
        state = State({variable_id: value for variable_id, value in enumerate(values)})
        expected = baseline.evaluate(state)

        for instance in (direct, source, prepared):
            evaluation = instance.evaluate(state)
            assert evaluation.feasible == expected.feasible
            assert evaluation.objective == expected.objective


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
