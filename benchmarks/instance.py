from __future__ import annotations

import math
import random

from ommx import (
    Constraint,
    DecisionVariable,
    Equality,
    IndicatorConstraint,
    Instance,
    Linear,
    OneHotConstraint,
    Quadratic,
    Sense,
    Sos1Constraint,
)


def _check_size(size: int, minimum: int = 1) -> None:
    if size < minimum:
        raise ValueError(f"Size must be at least {minimum}")


def _require_regular(formulation: str) -> None:
    if formulation != "regular":
        raise ValueError("This Instance supports only the regular formulation")


def _build_one_hot_constraints(
    specs: list[tuple[str, list[int], list[int]]],
    formulation: str,
    variables_by_id: dict[int, DecisionVariable],
) -> tuple[dict[int, Constraint], dict[int, OneHotConstraint]]:
    if formulation not in ("regular", "one-hot"):
        raise ValueError(f"Unknown formulation: {formulation}")

    if formulation == "regular":
        return (
            {
                constraint_id: Constraint(
                    function=Linear(
                        terms={variable_id: 1 for variable_id in variable_ids},
                        constant=-1,
                    ),
                    equality=Equality.EqualToZero,
                    name=name,
                    subscripts=subscripts,
                )
                for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
            },
            {},
        )

    return {}, {
        constraint_id: OneHotConstraint(
            variables=[variables_by_id[variable_id] for variable_id in variable_ids],
            name=name,
            subscripts=subscripts,
        )
        for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
    }


def build_knapsack_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a binary linear knapsack problem."""
    _check_size(size)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    weights = [random_generator.randint(1, 10) for _ in range(size)]
    values = [random_generator.randint(1, 20) for _ in range(size)]
    variables = [
        DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(size)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=dict(enumerate(values))),
        constraints={
            0: Constraint(
                function=Linear(
                    terms=dict(enumerate(weights)),
                    constant=-(sum(weights) // 2),
                ),
                equality=Equality.LessThanOrEqualToZero,
                name="capacity",
            )
        },
        sense=Sense.Maximize,
    )


def build_assignment_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build an assignment problem whose row and column OneHot groups overlap."""
    _check_size(size)
    random_generator = random.Random(seed)

    def variable_id(worker: int, task: int) -> int:
        return worker * size + task

    variables = [
        DecisionVariable.binary(
            variable_id(worker, task), name="x", subscripts=[worker, task]
        )
        for worker in range(size)
        for task in range(size)
    ]
    variables_by_id = {variable.id: variable for variable in variables}
    costs = {
        variable_id(worker, task): random_generator.uniform(0.5, 1.5)
        for worker in range(size)
        for task in range(size)
    }
    specs = [
        (
            "one-task",
            [worker],
            [variable_id(worker, task) for task in range(size)],
        )
        for worker in range(size)
    ] + [
        (
            "one-worker",
            [task],
            [variable_id(worker, task) for worker in range(size)],
        )
        for task in range(size)
    ]
    constraints, one_hot_constraints = _build_one_hot_constraints(
        specs, formulation, variables_by_id
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=costs),
        constraints=constraints,
        one_hot_constraints=one_hot_constraints,
        sense=Sense.Minimize,
    )


def build_tsp_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a quadratic TSP with overlapping city and time OneHot groups."""
    _check_size(size, minimum=2)
    random_generator = random.Random(seed)
    coordinates = [
        (random_generator.random(), random_generator.random()) for _ in range(size)
    ]
    distances = [
        [math.dist(coordinates[i], coordinates[j]) for j in range(size)]
        for i in range(size)
    ]
    max_distance = max(max(row) for row in distances)
    distances = [[distance / max_distance for distance in row] for row in distances]

    def variable_id(city: int, time: int) -> int:
        return city * size + time

    variables = [
        DecisionVariable.binary(
            variable_id(city, time), name="x", subscripts=[city, time]
        )
        for city in range(size)
        for time in range(size)
    ]
    variables_by_id = {variable.id: variable for variable in variables}
    columns: list[int] = []
    rows: list[int] = []
    values: list[float] = []
    for time in range(size):
        next_time = (time + 1) % size
        for city_i in range(size):
            for city_j in range(size):
                distance = distances[city_i][city_j]
                if distance == 0:
                    continue
                columns.append(variable_id(city_i, time))
                rows.append(variable_id(city_j, next_time))
                values.append(distance)

    specs = [
        (
            "one-city",
            [time],
            [variable_id(city, time) for city in range(size)],
        )
        for time in range(size)
    ] + [
        (
            "one-time",
            [city],
            [variable_id(city, time) for time in range(size)],
        )
        for city in range(size)
    ]
    constraints, one_hot_constraints = _build_one_hot_constraints(
        specs, formulation, variables_by_id
    )

    return Instance.from_components(
        decision_variables=variables,
        objective=Quadratic(columns=columns, rows=rows, values=values),
        constraints=constraints,
        one_hot_constraints=one_hot_constraints,
        sense=Sense.Minimize,
    )


def build_clique_instance(
    size: int, seed: int = 0, formulation: str = "regular"
) -> Instance:
    """Build a planted-clique problem with a quadratic equality constraint."""
    _check_size(size, minimum=2)
    _require_regular(formulation)
    random_generator = random.Random(seed)
    clique_size = (size + 1) // 2
    random_vertex_count = size - clique_size
    clique_vertices = range(random_vertex_count, size)
    edges = [
        (u, v)
        for u in range(random_vertex_count)
        for v in range(u + 1, random_vertex_count)
        if random_generator.random() < 0.2
    ]
    edges.extend((u, v) for u in clique_vertices for v in range(u + 1, size))
    edges.extend((u, random_vertex_count) for u in range(random_vertex_count))
    variables = [
        DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(size)
    ]

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms={}),
        constraints={
            0: Constraint(
                function=Linear(
                    terms={i: 1 for i in range(size)}, constant=-clique_size
                ),
                equality=Equality.EqualToZero,
                name="clique-size",
            ),
            1: Constraint(
                function=Quadratic(
                    columns=[u for u, _ in edges],
                    rows=[v for _, v in edges],
                    values=[1 for _ in edges],
                    linear=Linear(
                        terms={},
                        constant=-(clique_size * (clique_size - 1) // 2),
                    ),
                ),
                equality=Equality.EqualToZero,
                name="complete-subgraph",
            ),
        },
        sense=Sense.Minimize,
    )


def build_one_hot_preparation_instance(
    size: int,
    seed: int = 0,
    formulation: str = "one-hot",
    special_constraints: str = "none",
) -> Instance:
    """Build a grouped binary problem for measuring Instance preparation."""
    _check_size(size, minimum=2)
    if formulation != "one-hot":
        raise ValueError(
            "The one-hot-preparation Instance supports only one-hot formulation"
        )
    if special_constraints not in (
        "none",
        "indicator",
        "sos1",
        "indicator-sos1",
    ):
        raise ValueError(f"Unknown special constraints: {special_constraints}")
    random_generator = random.Random(seed)

    def variable_id(group: int, choice: int) -> int:
        return group * size + choice

    variables = [
        DecisionVariable.binary(
            variable_id(group, choice), name="x", subscripts=[group, choice]
        )
        for group in range(size)
        for choice in range(size)
    ]
    variables_by_id = {variable.id: variable for variable in variables}
    groups = [
        [variable_id(group, choice) for choice in range(size)] for group in range(size)
    ]
    one_hot_constraints = {
        group: OneHotConstraint(
            variables=[variables_by_id[variable_id] for variable_id in variable_ids],
            name="one-choice",
            subscripts=[group],
        )
        for group, variable_ids in enumerate(groups)
    }

    indicator_constraints: dict[int, IndicatorConstraint] = {}
    if special_constraints in ("indicator", "indicator-sos1"):
        indicator_constraints = {
            group: IndicatorConstraint(
                indicator_variable=variables_by_id[variable_ids[0]],
                function=Linear(
                    terms={variable_id: 1 for variable_id in variable_ids[1:]}
                ),
                equality=Equality.LessThanOrEqualToZero,
                name="remaining-choices-disabled",
                subscripts=[group],
            )
            for group, variable_ids in enumerate(groups)
        }

    sos1_constraints: dict[int, Sos1Constraint] = {}
    if special_constraints in ("sos1", "indicator-sos1"):
        sos1_constraints = {
            group: Sos1Constraint(
                variables=[
                    variables_by_id[variable_id] for variable_id in variable_ids
                ],
                name="at-most-one-choice",
                subscripts=[group],
            )
            for group, variable_ids in enumerate(groups)
        }

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(
            terms={
                variable.id: random_generator.uniform(0.5, 1.5)
                for variable in variables
            }
        ),
        constraints={},
        indicator_constraints=indicator_constraints,
        one_hot_constraints=one_hot_constraints,
        sos1_constraints=sos1_constraints,
        sense=Sense.Minimize,
    )


def build_feasible_entries(name: str, size: int) -> dict[int, int]:
    """Return one deterministic feasible state for a benchmark Instance."""
    if name == "knapsack":
        return {i: 0 for i in range(size)}
    if name in ("assignment", "tsp"):
        return {
            row * size + column: int(row == column)
            for row in range(size)
            for column in range(size)
        }
    if name == "clique":
        clique_size = (size + 1) // 2
        first_clique_vertex = size - clique_size
        return {i: int(i >= first_clique_vertex) for i in range(size)}
    if name == "one-hot-preparation":
        return {
            group * size + choice: int(choice == 0)
            for group in range(size)
            for choice in range(size)
        }
    raise ValueError(f"Unknown Instance: {name}")
