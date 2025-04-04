import random
from .models import (
    Progress,
    QuboRequest,
    QuboResponse,
    QuboSolutionList,
    QuboSolution,
    SolverTiming,
)


def get_variable_number_keys(qubo_request: QuboRequest) -> list[str]:
    variable_number_set = set()

    if (
        "numbers" in qubo_request.fujitsuDA3.one_way_one_hot_groups
        and "numbers" in qubo_request.fujitsuDA3.two_way_one_hot_groups
    ):
        raise ValueError(
            "one_way_one_hot_groups and two_way_one_hot_groups cannot be specified at the same time."
        )

    if "numbers" in qubo_request.fujitsuDA3.one_way_one_hot_groups:
        num_one_way_one_hot_variables = sum(
            qubo_request.fujitsuDA3.one_way_one_hot_groups["numbers"]
        )
        variable_number_set.union(range(num_one_way_one_hot_variables))

    if "numbers" in qubo_request.fujitsuDA3.two_way_one_hot_groups:
        num_two_way_one_hot_variables = sum(
            qubo_request.fujitsuDA3.two_way_one_hot_groups["numbers"]
        )
        variable_number_set.union(range(num_two_way_one_hot_variables))

    if qubo_request.binary_polynomial is not None:
        for term in qubo_request.binary_polynomial.terms:
            for id in term.p:
                variable_number_set.add(id)

    if qubo_request.penalty_binary_polynomial is not None:
        for term in qubo_request.penalty_binary_polynomial.terms:
            for id in term.p:
                variable_number_set.add(id)

    if qubo_request.inequalities is not None:
        for inequality in qubo_request.inequalities:
            for term in inequality.terms:
                for id in term.p:
                    variable_number_set.add(id)

    return sorted([str(variable_number) for variable_number in variable_number_set])


def random_qubo_solution(variable_number_keys: list[str]):
    return QuboSolution(
        energy=-48.0,
        penalty_energy=0.0,
        frequency=1,
        configuration={
            variable_number_key: random.choice([True, False])
            for variable_number_key in variable_number_keys
        },
    )


def random_qubo_response(qubo_request: QuboRequest) -> QuboResponse:
    num_configurations = (
        qubo_request.fujitsuDA3.num_output_solution * qubo_request.fujitsuDA3.num_group
    )

    variable_number_keys = get_variable_number_keys(qubo_request)

    dummy_progress = [
        Progress(energy=0.0, penalty_energy=7.0, time=0.161),
        Progress(energy=-42.0, penalty_energy=0.0, time=0.355),
        Progress(energy=-45.0, penalty_energy=0.0, time=0.697),
        Progress(energy=-46.0, penalty_energy=0.0, time=1.109),
        Progress(energy=-47.0, penalty_energy=0.0, time=1.331),
        Progress(energy=-48.0, penalty_energy=0.0, time=1.629),
    ]
    dummy_result_status = True
    dummy_timing = SolverTiming(solve_time="11027", total_elapsed_time="11062")
    dummy_status = "Done"
    qubo_solutions = [
        random_qubo_solution(variable_number_keys) for _ in range(num_configurations)
    ]

    return QuboResponse(
        qubo_solution=QuboSolutionList(
            progress=dummy_progress,
            result_status=dummy_result_status,
            solutions=qubo_solutions,
            timing=dummy_timing,
        ),
        status=dummy_status,
    )
