import itertools

import pytest

from benchmarks.common import (
    PREPARATIONS,
    build_instance,
    build_response,
    make_benchmark_operation,
)
from benchmarks.instance import build_assignment_instance, build_tsp_instance
from benchmarks.run import benchmark_cases
from ommx_da4_adapter import DA4Client, OMMXDA4Adapter


@pytest.fixture(autouse=True)
def forbid_da4_connection(monkeypatch):
    def unexpected_client(*args, **kwargs):
        pytest.fail("Conversion benchmarks must not construct a DA4 client")

    monkeypatch.setattr(DA4Client, "__init__", unexpected_client)


# Each mathematical workload once; large sizes are exercised by the full run.
WORKLOADS = sorted(
    {case[1:5] for case in benchmark_cases() if case[0] == "response-to-solution"}
)


@pytest.mark.parametrize(("name", "formulation", "special", "preparation"), WORKLOADS)
def test_decode_workloads_are_feasible_without_a_client(
    name, formulation, special, preparation
):
    instance = build_instance(name, 3, 7, formulation, special, preparation)
    preparation_arguments = (
        {"special_constraints": special, "preparation": preparation}
        if "recommended" in PREPARATIONS
        else {}
    )
    benchmark = make_benchmark_operation(
        "response-to-solution", instance, name, 16, **preparation_arguments
    )
    response = benchmark.setup()
    solution = benchmark.run(response)

    assert solution.feasible
    assert len(response.qubo_solution.solutions) == 1
    assert response.qubo_solution.solutions[0].frequency == 16
    assert instance.evaluate(solution.state).objective == pytest.approx(
        solution.objective
    )


def test_response_frequency_is_preserved_in_sample_ids():
    adapter = OMMXDA4Adapter(build_tsp_instance(3, formulation="one-hot"))
    response = build_response(adapter, "tsp", sample_count=16)
    sampleset = adapter.decode_to_sampleset(response)

    assert sampleset.sample_ids == list(range(16))
    assert sampleset.best_feasible.feasible


@pytest.mark.parametrize("builder", [build_assignment_instance, build_tsp_instance])
def test_regular_and_one_hot_formulations_have_the_same_mathematical_problem(builder):
    regular = builder(2, seed=7, formulation="regular")
    one_hot = builder(2, seed=7, formulation="one-hot")
    for values in itertools.product((0, 1), repeat=4):
        state = dict(enumerate(values))
        expected = regular.evaluate(state)
        actual = one_hot.evaluate(state)
        assert actual.feasible == expected.feasible
        assert actual.objective == pytest.approx(expected.objective)

    regular_request = OMMXDA4Adapter(regular).sampler_input
    one_hot_request = OMMXDA4Adapter(one_hot).sampler_input
    assert regular_request.fujitsuDA3.one_way_one_hot_groups is None
    assert one_hot_request.fujitsuDA3.one_way_one_hot_groups == {"numbers": [2, 2]}


def test_matrix_covers_each_comparison_once():
    cases = benchmark_cases()
    assert len(cases) == len(set(cases))
    if "recommended" in PREPARATIONS:
        assert len(cases) == 87
        assert sum(case[0] == "prepare" for case in cases) == 9
    else:
        assert len(cases) == 60
        assert all(case[0] != "prepare" for case in cases)
    for case in cases:
        operation, name, formulation, special, preparation, size = case
        if preparation == "recommended" and operation != "prepare":
            assert (operation, name, formulation, special, "none", size) in cases
