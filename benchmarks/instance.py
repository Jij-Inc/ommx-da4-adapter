from __future__ import annotations

import math
import random

from ommx.v1 import (
    Constraint,
    ConstraintHints,
    DecisionVariable,
    Instance,
    Linear,
    OneHot,
    Quadratic,
)


def _check_size(size: int, minimum: int = 1) -> None:
    if size < minimum:
        raise ValueError(f"Size must be at least {minimum}")


def _require_regular(formulation: str) -> None:
    if formulation != "regular":
        raise ValueError("This Instance supports only the regular formulation")


def _build_one_hot_constraints(
    specs: list[tuple[str, list[int], list[int]]], formulation: str
) -> tuple[list[Constraint], ConstraintHints | None]:
    if formulation not in ("regular", "one-hot"):
        raise ValueError(f"Unknown formulation: {formulation}")

    constraints = [
        Constraint(
            id=constraint_id,
            function=Linear(
                terms={variable_id: 1 for variable_id in variable_ids},
                constant=-1,
            ),
            equality=Constraint.EQUAL_TO_ZERO,
            name=name,
            subscripts=subscripts,
        )
        for constraint_id, (name, subscripts, variable_ids) in enumerate(specs)
    ]
    if formulation == "regular":
        return constraints, None

    return constraints, ConstraintHints(
        one_hot_constraints=[
            OneHot(id=constraint_id, variables=variable_ids)
            for constraint_id, (_, _, variable_ids) in enumerate(specs)
        ]
    )


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
        constraints=[
            Constraint(
                id=0,
                function=Linear(
                    terms=dict(enumerate(weights)),
                    constant=-(sum(weights) // 2),
                ),
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
                name="capacity",
            )
        ],
        sense=Instance.MAXIMIZE,
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
    constraints, constraint_hints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(terms=costs),
        constraints=constraints,
        constraint_hints=constraint_hints,
        sense=Instance.MINIMIZE,
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
    constraints, constraint_hints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=Quadratic(columns=columns, rows=rows, values=values),
        constraints=constraints,
        constraint_hints=constraint_hints,
        sense=Instance.MINIMIZE,
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
        constraints=[
            Constraint(
                id=0,
                function=Linear(
                    terms={i: 1 for i in range(size)}, constant=-clique_size
                ),
                equality=Constraint.EQUAL_TO_ZERO,
                name="clique-size",
            ),
            Constraint(
                id=1,
                function=Quadratic(
                    columns=[u for u, _ in edges],
                    rows=[v for _, v in edges],
                    values=[1 for _ in edges],
                    linear=Linear(
                        terms={},
                        constant=-(clique_size * (clique_size - 1) // 2),
                    ),
                ),
                equality=Constraint.EQUAL_TO_ZERO,
                name="complete-subgraph",
            ),
        ],
        sense=Instance.MINIMIZE,
    )


def build_one_hot_preparation_instance(
    size: int, seed: int = 0, formulation: str = "one-hot"
) -> Instance:
    """Build the OMMX v2 baseline for the v3 preparation workload."""
    _check_size(size, minimum=2)
    if formulation != "one-hot":
        raise ValueError(
            "The one-hot-preparation Instance supports only one-hot formulation"
        )
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
    specs = [
        (
            "one-choice",
            [group],
            [variable_id(group, choice) for choice in range(size)],
        )
        for group in range(size)
    ]
    constraints, constraint_hints = _build_one_hot_constraints(specs, formulation)

    return Instance.from_components(
        decision_variables=variables,
        objective=Linear(
            terms={
                variable.id: random_generator.uniform(0.5, 1.5)
                for variable in variables
            }
        ),
        constraints=constraints,
        constraint_hints=constraint_hints,
        sense=Instance.MINIMIZE,
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
