import pytest

from ommx.v1 import Constraint, DecisionVariable, Instance, ConstraintHints, OneHot

from ommx_da4_adapter import OMMXDA4Adapter
from ommx_da4_adapter.exception import OMMXDA4AdapterError
from ommx_da4_adapter.models import BinaryPolynomialTerm, QuboResponse


# Function to sort based on the length of binary polynomials
# Used to align the order (because the order may differ depending on the environment)
def sort_terms(terms: list[BinaryPolynomialTerm]) -> list[BinaryPolynomialTerm]:
    return sorted(terms, key=lambda term: term.p)


@pytest.fixture
def instance_for_validation():
    x_1 = DecisionVariable.binary(id=0, name="x_1")
    x_2 = DecisionVariable.binary(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 == 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )

    return instance


def test_time_limit_sec_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`time_limit_sec` must be between 1 and 3600"):
        OMMXDA4Adapter(instance_for_validation, time_limit_sec=0)
    with pytest.raises(ValueError, match="`time_limit_sec` must be between 1 and 3600"):
        OMMXDA4Adapter(instance_for_validation, time_limit_sec=3601)


def test_target_energy_validate(instance_for_validation):
    with pytest.raises(
        ValueError, match=r"target_energy.*must be between \- 2\^126 and 2\^126"
    ):
        OMMXDA4Adapter(instance_for_validation, target_energy=-pow(2, 127))

    with pytest.raises(
        ValueError, match=r"target_energy.*must be between \- 2\^126 and 2\^126"
    ):
        OMMXDA4Adapter(instance_for_validation, target_energy=pow(2, 127))


def test_num_run_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`num_run` must be between 1 and 1024"):
        OMMXDA4Adapter(instance_for_validation, num_run=0)
    with pytest.raises(ValueError, match="`num_run` must be between 1 and 1024"):
        OMMXDA4Adapter(instance_for_validation, num_run=1025)


def test_num_group_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`num_group` must be between 1 and 16"):
        OMMXDA4Adapter(instance_for_validation, num_group=0)
    with pytest.raises(ValueError, match="`num_group` must be between 1 and 16"):
        OMMXDA4Adapter(instance_for_validation, num_group=17)


def test_num_output_solution_validate(instance_for_validation):
    with pytest.raises(
        ValueError, match="`num_output_solution` must be between 1 and 1024"
    ):
        OMMXDA4Adapter(instance_for_validation, num_output_solution=0)
    with pytest.raises(
        ValueError, match="`num_output_solution` must be between 1 and 1024"
    ):
        OMMXDA4Adapter(instance_for_validation, num_output_solution=1025)


def test_gs_level_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`gs_level` must be between 0 and 100"):
        OMMXDA4Adapter(instance_for_validation, gs_level=-1)
    with pytest.raises(ValueError, match="`gs_level` must be between 0 and 100"):
        OMMXDA4Adapter(instance_for_validation, gs_level=101)


def test_gs_cutoff_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`gs_cutoff` must be between 0 and 1000000"):
        OMMXDA4Adapter(instance_for_validation, gs_cutoff=-1)
    with pytest.raises(ValueError, match="`gs_cutoff` must be between 0 and 1000000"):
        OMMXDA4Adapter(instance_for_validation, gs_cutoff=1000001)


def test_one_hot_level_validate(instance_for_validation):
    with pytest.raises(ValueError, match="`one_hot_level` must be between 0 and 100"):
        OMMXDA4Adapter(instance_for_validation, one_hot_level=-1)
    with pytest.raises(ValueError, match="`one_hot_level` must be between 0 and 100"):
        OMMXDA4Adapter(instance_for_validation, one_hot_level=101)


def test_one_hot_cutoff_validate(instance_for_validation):
    with pytest.raises(
        ValueError, match="`one_hot_cutoff` must be between 0 and 1000000"
    ):
        OMMXDA4Adapter(instance_for_validation, one_hot_cutoff=-1)
    with pytest.raises(
        ValueError, match="`one_hot_cutoff` must be between 0 and 1000000"
    ):
        OMMXDA4Adapter(instance_for_validation, one_hot_cutoff=1000001)


def test_penalty_auto_mode(instance_for_validation):
    with pytest.raises(
        ValueError, match="`penalty_auto_mode` must be between 0 and 10000"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_auto_mode=-1)
    with pytest.raises(
        ValueError, match="`penalty_auto_mode` must be between 0 and 10000"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_auto_mode=10001)


def test_penalty_coef(instance_for_validation):
    with pytest.raises(
        ValueError, match="`penalty_coef` must be between 1 and 9223372036854775807"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_coef=0)
    with pytest.raises(
        ValueError, match="`penalty_coef` must be between 1 and 9223372036854775807"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_coef=9223372036854775808)


def test_penalty_inc_rate(instance_for_validation):
    with pytest.raises(
        ValueError, match="`penalty_inc_rate` must be between 100 and 200"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_inc_rate=99)
    with pytest.raises(
        ValueError, match="`penalty_inc_rate` must be between 100 and 200"
    ):
        OMMXDA4Adapter(instance_for_validation, penalty_inc_rate=201)


def test_max_penalty_coef(instance_for_validation):
    with pytest.raises(
        ValueError, match="`max_penalty_coef` must be between 0 and 9223372036854775807"
    ):
        OMMXDA4Adapter(instance_for_validation, max_penalty_coef=-1)
    with pytest.raises(
        ValueError, match="`max_penalty_coef` must be between 0 and 9223372036854775807"
    ):
        OMMXDA4Adapter(instance_for_validation, max_penalty_coef=9223372036854775808)


@pytest.fixture
def instance_for_integer_variable():
    x_1 = DecisionVariable.integer(id=0, name="x_1")
    x_2 = DecisionVariable.integer(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 == 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MAXIMIZE,
    )

    return instance


def test_check_integer_variable(instance_for_integer_variable):
    with pytest.raises(
        OMMXDA4AdapterError,
        match=r"The decision variable must be binary: id .*",
    ):
        OMMXDA4Adapter(instance_for_integer_variable)


@pytest.fixture
def instance_for_continuous_variable():
    x_1 = DecisionVariable.continuous(id=0, name="x_1")
    x_2 = DecisionVariable.continuous(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 == 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MAXIMIZE,
    )

    return instance


def test_check_continuous_variable(instance_for_continuous_variable):
    with pytest.raises(
        OMMXDA4AdapterError,
        match=r"The decision variable must be binary: id .*",
    ):
        OMMXDA4Adapter(instance_for_continuous_variable)


@pytest.fixture
def instance():
    x_1 = DecisionVariable.binary(id=0, name="x_1")
    x_2 = DecisionVariable.binary(id=1, name="x_2")
    x_3 = DecisionVariable.binary(id=2, name="x_3")

    # objective: 2x₁x₂x₃ + 3x₁x₂ + 4x₁ + 5x₂ + 6x₃ + 6
    # constraint_equality_1: x₁x₂x₃ == 0
    # constraint_equality_2: 3x₁x₂x₃ + 2x₁ + 5 == 0
    # constraint_inequality_1: 4x₁x₂x₃ ≤ 0
    # constraint_inequality_2: x₁x₂x₃ + 2x₃ + 3 ≤ 0

    objective = 2 * x_1 * x_2 * x_3 + 3 * x_1 * x_2 + 4 * x_1 + 5 * x_2 + 6 * x_3 + 6
    constraint_equality_1 = x_1 * x_2 * x_3 == 0
    constraint_equality_2 = 3 * x_1 * x_2 * x_3 + 2 * x_1 + 5 == 0
    constraint_inequality_1 = 4 * x_1 * x_2 * x_3 <= 0
    constraint_inequality_2 = x_1 * x_2 * x_3 + 2 * x_3 + 3 <= 0

    assert isinstance(constraint_inequality_1, Constraint)
    assert isinstance(constraint_inequality_2, Constraint)
    constraint_inequality_1.set_id(2)
    constraint_inequality_2.set_id(3)

    instance = Instance.from_components(
        decision_variables=[x_1, x_2, x_3],
        objective=objective,
        constraints=[
            constraint_equality_1,
            constraint_equality_2,
            constraint_inequality_1,
            constraint_inequality_2,
        ],
        sense=Instance.MINIMIZE,
    )

    return instance


def test_adapter_default_value(instance):
    adapter = OMMXDA4Adapter(instance)
    qubo_request = adapter.sampler_input

    # Expected default values for fujitsuDA3 parameters
    expected_values = {
        "time_limit_sec": 10,
        "target_energy": None,
        "num_run": 16,
        "num_group": 1,
        "num_output_solution": 5,
        "gs_level": 5,
        "gs_cutoff": 8000,
        "one_hot_level": 3,
        "one_hot_cutoff": 100,
        "penalty_auto_mode": 1,
        "penalty_coef": 1,
        "penalty_inc_rate": 150,
        "max_penalty_coef": 0,
        "guidance_config": None,
        "fixed_config": None,
    }

    for param, expected in expected_values.items():
        assert getattr(qubo_request.fujitsuDA3, param) == expected

    assert adapter._inequalities_lambda is None

    assert qubo_request.bucket_name is None
    assert qubo_request.binary_polynomial_object_name is None
    assert qubo_request.penalty_binary_polynomial_object_name is None
    assert qubo_request.inequalities_object_name is None


def test_fujitsuDA3(instance):
    adapter = OMMXDA4Adapter(
        instance,
        time_limit_sec=10,
        target_energy=1,
        num_run=16,
        num_group=1,
        num_output_solution=5,
        gs_level=5,
        gs_cutoff=8000,
        one_hot_level=3,
        one_hot_cutoff=100,
        penalty_auto_mode=1,
        penalty_coef=1,
        penalty_inc_rate=150,
        max_penalty_coef=0,
        guidance_config={"description": True},
        fixed_config={"description": True},
    )
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.time_limit_sec == 10
    assert qubo_request.fujitsuDA3.target_energy == 1
    assert qubo_request.fujitsuDA3.num_run == 16
    assert qubo_request.fujitsuDA3.num_group == 1
    assert qubo_request.fujitsuDA3.num_output_solution == 5
    assert qubo_request.fujitsuDA3.gs_level == 5
    assert qubo_request.fujitsuDA3.gs_cutoff == 8000
    assert qubo_request.fujitsuDA3.one_hot_level == 3
    assert qubo_request.fujitsuDA3.one_hot_cutoff == 100
    assert qubo_request.fujitsuDA3.internal_penalty == 0
    assert qubo_request.fujitsuDA3.penalty_auto_mode == 1
    assert qubo_request.fujitsuDA3.penalty_coef == 1
    assert qubo_request.fujitsuDA3.penalty_inc_rate == 150
    assert qubo_request.fujitsuDA3.max_penalty_coef == 0


def test_binary_polynomial(instance):
    adapter = OMMXDA4Adapter(instance)
    qubo_request = adapter.sampler_input

    # assert with manually calculated values
    # 2x₁x₂x₃ + 3x₁x₂ + 4x₁ + 5x₂ + 6x₃ + 6
    assert qubo_request.binary_polynomial is not None and sort_terms(
        qubo_request.binary_polynomial.terms
    ) == sort_terms(
        [
            BinaryPolynomialTerm(c=6.0, p=[]),
            BinaryPolynomialTerm(c=4.0, p=[0]),
            BinaryPolynomialTerm(c=3.0, p=[0, 1]),
            BinaryPolynomialTerm(c=2.0, p=[0, 1, 2]),
            BinaryPolynomialTerm(c=5.0, p=[1]),
            BinaryPolynomialTerm(c=6.0, p=[2]),
        ]
    )


def test_penalty_binary_polynomial(instance):
    adapter = OMMXDA4Adapter(instance)
    qubo_request = adapter.sampler_input

    # assert with manually calculated values
    # 52y₁y₂y₃ + 24y₁ + 25
    assert qubo_request.penalty_binary_polynomial is not None and sort_terms(
        qubo_request.penalty_binary_polynomial.terms
    ) == sort_terms(
        [
            BinaryPolynomialTerm(c=52.0, p=[0, 1, 2]),
            BinaryPolynomialTerm(c=24.0, p=[0]),
            BinaryPolynomialTerm(c=25.0, p=[]),
        ]
    )


def test_inequalities_lambda_default(instance):
    adapter = OMMXDA4Adapter(instance, inequalities_lambda={})
    qubo_request = adapter.sampler_input

    # if inequalities_lambda is None, failure
    assert qubo_request.inequalities is not None, "inequalities is None"

    for inequality in qubo_request.inequalities:
        assert inequality.lambda_ == 1


def test_inequalities_lambda(instance):
    adapter = OMMXDA4Adapter(instance, inequalities_lambda={2: 6, 3: 2})
    qubo_request = adapter.sampler_input

    # if inequalities_lambda is None, failure
    assert qubo_request.inequalities is not None, "inequalities is None"
    for inequality in qubo_request.inequalities:
        print(inequality)

    for inequality in qubo_request.inequalities:
        print(inequality.lambda_)
        if inequality.lambda_ == 2:
            assert inequality.lambda_ == 2
        elif inequality.lambda_ == 6:
            assert inequality.lambda_ == 6
        else:
            assert False


def test_inequalities(instance):
    adapter = OMMXDA4Adapter(instance)
    qubo_request = adapter.sampler_input

    # assert with manually calculated values
    # 4x₁x₂x₃ ≤ 0
    # x₁x₂x₃ + 2x₃ + 3 ≤ 0
    assert qubo_request.inequalities is not None
    assert sort_terms(qubo_request.inequalities[0].terms) == sort_terms(
        [
            BinaryPolynomialTerm(c=4.0, p=[0, 1, 2]),
        ]
    )
    assert sort_terms(qubo_request.inequalities[1].terms) == sort_terms(
        [
            BinaryPolynomialTerm(c=3.0, p=[]),
            BinaryPolynomialTerm(c=1.0, p=[0, 1, 2]),
            BinaryPolynomialTerm(c=2.0, p=[2]),
        ]
    )


@pytest.fixture
def instance_for_MAXIMIZE():
    x_1 = DecisionVariable.binary(id=0, name="x_1")
    x_2 = DecisionVariable.binary(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 == 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MAXIMIZE,
    )

    return instance


def test_binary_polynomial_for_MAXIMIZE(instance_for_MAXIMIZE):
    adapter = OMMXDA4Adapter(instance_for_MAXIMIZE)
    qubo_request = adapter.sampler_input

    # assert with manually calculated values
    # -y₁ - y₂
    assert qubo_request.binary_polynomial is not None and sort_terms(
        qubo_request.binary_polynomial.terms
    ) == sort_terms(
        [
            BinaryPolynomialTerm(c=-1.0, p=[0]),
            BinaryPolynomialTerm(c=-1.0, p=[1]),
        ]
    )


@pytest.fixture
def instance_for_no_penalty_binary_polynomial():
    x_1 = DecisionVariable.binary(id=0, name="x_1")
    x_2 = DecisionVariable.binary(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 <= 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )

    return instance


def test_no_penalty_binary_polynomial(instance_for_no_penalty_binary_polynomial):
    adapter = OMMXDA4Adapter(instance_for_no_penalty_binary_polynomial)
    qubo_request = adapter.sampler_input

    assert qubo_request.penalty_binary_polynomial is None


@pytest.fixture
def instance_for_no_inequalities():
    x_1 = DecisionVariable.binary(id=0, name="x_1")
    x_2 = DecisionVariable.binary(id=1, name="x_2")

    objective = x_1 + x_2
    constraints = [x_1 * x_2 == 0]

    instance = Instance.from_components(
        decision_variables=[x_1, x_2],
        objective=objective,
        constraints=constraints,
        sense=Instance.MINIMIZE,
    )

    return instance


def test_no_inequalities(instance_for_no_inequalities):
    adapter = OMMXDA4Adapter(instance_for_no_inequalities)
    qubo_request = adapter.sampler_input

    assert qubo_request.inequalities is None


@pytest.fixture
def instance_with_a_one_hot_constraint():
    x = [DecisionVariable.binary(id=i, name="x", subscripts=[i]) for i in range(5)]
    objective = sum(x[i] for i in range(5))
    constraint = x[1] + x[2] + x[3] == 1

    assert isinstance(constraint, Constraint)
    constraint.set_id(0)

    hints = ConstraintHints(one_hot_constraints=[OneHot(id=0, variables=[1, 2, 3])])

    ommx_instance = Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=[constraint],
        sense=Instance.MAXIMIZE,
        constraint_hints=hints,
    )

    return ommx_instance


def test_variable_map_with_a_one_hot_constraint(instance_with_a_one_hot_constraint):
    adapter = OMMXDA4Adapter(instance_with_a_one_hot_constraint)
    """
    for hint in hints.one_hot_constraints:
        print(hint.decision_variables)

    [1, 2, 3]
    """
    assert adapter._variable_map == {
        1: 0,
        2: 1,
        3: 2,
        0: 3,
        4: 4,
    }


def test_penalty_binary_polynomial_with_a_one_hot_constraint(
    instance_with_a_one_hot_constraint,
):
    adapter = OMMXDA4Adapter(instance_with_a_one_hot_constraint)
    qubo_request = adapter.sampler_input

    assert qubo_request.penalty_binary_polynomial is None


def test_one_way_one_hot_groups_with_a_one_hot_constraint(
    instance_with_a_one_hot_constraint,
):
    adapter = OMMXDA4Adapter(instance_with_a_one_hot_constraint)
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups is not None, (
        "one_way_one_hot_groups is None"
    )

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups["numbers"] == [3]


def test_internal_penalty_with_a_one_hot_constraint(
    instance_with_a_one_hot_constraint,
):
    adapter = OMMXDA4Adapter(instance_with_a_one_hot_constraint)
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.internal_penalty == 1


@pytest.fixture
def instance_with_various_constraints():
    x = [DecisionVariable.binary(id=i, name="x", subscripts=[i]) for i in range(10)]

    objective = sum(x[i] for i in range(10))
    constraint_1 = sum(x[i] for i in range(3)) == 1
    constraint_2 = sum(x[i] for i in range(3, 8)) == 1
    constraint_3 = sum(x[i] for i in range(8, 10)) <= 1

    assert isinstance(constraint_1, Constraint)
    assert isinstance(constraint_2, Constraint)
    assert isinstance(constraint_3, Constraint)
    constraint_1.set_id(0)
    constraint_2.set_id(1)
    constraint_3.set_id(2)

    hints = ConstraintHints(
        one_hot_constraints=[
            OneHot(id=0, variables=[0, 1, 2]),
            OneHot(id=1, variables=[3, 4, 5, 6, 7]),
        ]
    )

    ommx_instance = Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=[constraint_1, constraint_2, constraint_3],
        sense=Instance.MAXIMIZE,
        constraint_hints=hints,
    )

    return ommx_instance


def test_variable_map_with_various_constraints(instance_with_various_constraints):
    adapter = OMMXDA4Adapter(instance_with_various_constraints)
    """
    for hint in hints.one_hot_constraints:
        print(hint.decision_variables)

    [0, 1, 2]
    [3, 4, 5, 6, 7]
    """
    assert adapter._variable_map == {
        3: 0,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        0: 5,
        1: 6,
        2: 7,
        8: 8,
        9: 9,
    }


def test_one_way_one_hot_groups_with_various_constraints(
    instance_with_various_constraints,
):
    adapter = OMMXDA4Adapter(instance_with_various_constraints)
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups is not None, (
        "one_way_one_hot_groups is None"
    )

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups["numbers"] == [5, 3]


@pytest.fixture
def instance_with_no_one_hot_constraint():
    x = [DecisionVariable.binary(id=i, name="x", subscripts=[i]) for i in range(5)]
    objective = sum(x[i] for i in range(5))
    constraint = [x[1] + x[2] + x[3] == 2]
    ommx_instance = Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=constraint,
        sense=Instance.MAXIMIZE,
    )

    return ommx_instance


def test_variable_map_with_no_one_hot_constraint(instance_with_no_one_hot_constraint):
    adapter = OMMXDA4Adapter(instance_with_no_one_hot_constraint)
    """
    for hint in hints.one_hot_constraints:
        print(hint.decision_variables)

    []
    """
    assert adapter._variable_map == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    }


def test_one_way_one_hot_groups_with_no_one_hot_constraint(
    instance_with_no_one_hot_constraint,
):
    adapter = OMMXDA4Adapter(instance_with_no_one_hot_constraint)
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups is None


@pytest.fixture
def instance_with_duplicates():
    x = [DecisionVariable.binary(id=i, name="x", subscripts=[i]) for i in range(5)]
    objective = sum(x[i] for i in range(5))
    constraint_1 = sum(x[i] for i in range(2)) == 1
    constraint_2 = sum(x[i] for i in range(1, 4)) == 1

    assert isinstance(constraint_1, Constraint)
    assert isinstance(constraint_2, Constraint)
    constraint_1.set_id(0)
    constraint_2.set_id(1)

    hints = ConstraintHints(
        one_hot_constraints=[
            OneHot(id=0, variables=[0, 1]),
            OneHot(id=1, variables=[1, 2, 3]),
        ]
    )

    ommx_instance = Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=[constraint_1, constraint_2],
        sense=Instance.MAXIMIZE,
        constraint_hints=hints,
    )

    return ommx_instance


def test_variable_map_with_duplicates(instance_with_duplicates):
    adapter = OMMXDA4Adapter(instance_with_duplicates)
    """
    for hint in hints.one_hot_constraints:
        print(hint.decision_variables)

    [0, 1]
    [1, 2, 3]
    """
    # one hot constraints is only [1, 2, 3]
    assert adapter._variable_map == {
        1: 0,
        2: 1,
        3: 2,
        0: 3,
        4: 4,
    }


def test_one_way_one_hot_groups_with_duplicates(instance_with_duplicates):
    adapter = OMMXDA4Adapter(instance_with_duplicates)
    qubo_request = adapter.sampler_input

    assert qubo_request.fujitsuDA3.one_way_one_hot_groups is not None, (
        "one_way_one_hot_groups is None"
    )
    assert qubo_request.fujitsuDA3.one_way_one_hot_groups["numbers"] == [3]


def test_penalty_binary_polynomial_with_duplicates(instance_with_duplicates):
    adapter = OMMXDA4Adapter(instance_with_duplicates)
    qubo_request = adapter.sampler_input

    # assert with manually calculated values
    # (x_0 + x_1 - 1)^2
    assert qubo_request.penalty_binary_polynomial is not None and sort_terms(
        qubo_request.penalty_binary_polynomial.terms
    ) == sort_terms(
        [
            BinaryPolynomialTerm(c=-1.0, p=[3]),
            BinaryPolynomialTerm(c=2.0, p=[3, 0]),
            BinaryPolynomialTerm(c=-1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[]),
        ]
    )


@pytest.fixture
def instance_knapsack_problem():
    # Knapsack Problem
    v = [10, 13, 18, 31, 7, 15]
    w = [11, 25, 20, 35, 10, 33]
    W = 47
    N = len(v)

    x = [
        DecisionVariable.binary(
            id=i,
            name="x",
            subscripts=[i],
        )
        for i in range(N)
    ]

    objective = sum(v[i] * x[i] for i in range(N))

    constraint = sum(w[i] * x[i] for i in range(N)) - W <= 0
    assert isinstance(constraint, Constraint)

    instance = Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=[constraint],
        sense=Instance.MAXIMIZE,
    )

    return instance


def test_decode_to_sampleset(instance_knapsack_problem):
    adapter = OMMXDA4Adapter(instance_knapsack_problem)

    # qubo_response from mock
    qubo_response = {
        "qubo_solution": {
            "progress": [
                {"energy": 0.0, "penalty_energy": 7.0, "time": 0.161},
                {"energy": -42.0, "penalty_energy": 0.0, "time": 0.355},
                {"energy": -45.0, "penalty_energy": 0.0, "time": 0.697},
                {"energy": -46.0, "penalty_energy": 0.0, "time": 1.109},
                {"energy": -47.0, "penalty_energy": 0.0, "time": 1.331},
                {"energy": -48.0, "penalty_energy": 0.0, "time": 1.629},
            ],
            "result_status": True,
            "solutions": [
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": True,
                        "2": False,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": True,
                        "2": True,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": False,
                        "2": False,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": True,
                        "1": False,
                        "2": True,
                        "3": False,
                        "4": False,
                        "5": True,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": False,
                        "2": False,
                        "3": False,
                        "4": False,
                        "5": True,
                    },
                },
            ],
            "timing": {"solve_time": "11027", "total_elapsed_time": "11062"},
        },
        "status": "Done",
    }

    sampleset = adapter.decode_to_sampleset(QuboResponse(**qubo_response))

    assert sampleset is not None
    assert len(sampleset.decision_variables) == 6
    assert len(sampleset.sample_ids) == 5


def test_sampele_without_token(instance_knapsack_problem):
    with pytest.raises(
        OMMXDA4AdapterError,
        match="token is required. Please set the token to use the DA4 API.",
    ):
        OMMXDA4Adapter.sample(instance_knapsack_problem)


def test_solve_adapter_default_value(instance_knapsack_problem):
    adapter = OMMXDA4Adapter(instance_knapsack_problem)
    qubo_request = adapter.solver_input

    # Expected default values for fujitsuDA3 parameters
    expected_values = {
        "time_limit_sec": 10,
        "target_energy": None,
        "num_run": 16,
        "num_group": 1,
        "num_output_solution": 5,
        "gs_level": 5,
        "gs_cutoff": 8000,
        "one_hot_level": 3,
        "one_hot_cutoff": 100,
        "penalty_auto_mode": 1,
        "penalty_coef": 1,
        "penalty_inc_rate": 150,
        "max_penalty_coef": 0,
        "guidance_config": None,
        "fixed_config": None,
    }

    for param, expected in expected_values.items():
        assert getattr(qubo_request.fujitsuDA3, param) == expected

    assert adapter._inequalities_lambda is None

    assert qubo_request.bucket_name is None
    assert qubo_request.binary_polynomial_object_name is None
    assert qubo_request.penalty_binary_polynomial_object_name is None
    assert qubo_request.inequalities_object_name is None


def test_decode_to_sample(instance_knapsack_problem):
    adapter = OMMXDA4Adapter(instance_knapsack_problem)

    # qubo_response from mock
    qubo_response = {
        "qubo_solution": {
            "progress": [
                {"energy": 0.0, "penalty_energy": 7.0, "time": 0.161},
                {"energy": -42.0, "penalty_energy": 0.0, "time": 0.355},
                {"energy": -45.0, "penalty_energy": 0.0, "time": 0.697},
                {"energy": -46.0, "penalty_energy": 0.0, "time": 1.109},
                {"energy": -47.0, "penalty_energy": 0.0, "time": 1.331},
                {"energy": -48.0, "penalty_energy": 0.0, "time": 1.629},
            ],
            "result_status": True,
            "solutions": [
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": True,
                        "2": False,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": True,
                        "2": True,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": False,
                        "2": False,
                        "3": True,
                        "4": True,
                        "5": False,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": True,
                        "1": False,
                        "2": True,
                        "3": False,
                        "4": False,
                        "5": True,
                    },
                },
                {
                    "energy": -48.0,
                    "penalty_energy": 0.0,
                    "frequency": 1,
                    "configuration": {
                        "0": False,
                        "1": False,
                        "2": False,
                        "3": False,
                        "4": False,
                        "5": True,
                    },
                },
            ],
            "timing": {"solve_time": "11027", "total_elapsed_time": "11062"},
        },
        "status": "Done",
    }

    solution = adapter.decode(QuboResponse(**qubo_response))

    assert solution is not None
    assert len(solution.decision_variables) == 6


def validate_qubo_request(qubo_request, expected_terms, expected_inequality_terms):
    """Helper function to validate QUBO request with expected binary polynomial and inequality terms."""
    assert qubo_request.binary_polynomial is not None
    expected_terms_sorted = sort_terms(expected_terms)
    actual_terms = sort_terms(qubo_request.binary_polynomial.terms)
    assert actual_terms == expected_terms_sorted
    
    assert qubo_request.inequalities is not None
    expected_inequality_terms_sorted = sort_terms(expected_inequality_terms)
    actual_inequality_terms = sort_terms(qubo_request.inequalities[0].terms)
    assert actual_inequality_terms == expected_inequality_terms_sorted


def test_partial_evaluate():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + x[1] + x[2],
        constraints=[(x[0] + x[1] + x[2] <= 1).set_id(0)],
        sense=Instance.MINIMIZE,
    )
    assert instance.used_decision_variables == x

    partial = instance.partial_evaluate({0: 1})
    # x[0] is no longer present in the problem
    assert partial.used_decision_variables == x[1:]

    adapter = OMMXDA4Adapter(partial)
    qubo_request = adapter.sampler_input

    validate_qubo_request(
        qubo_request,
        expected_terms=[
            BinaryPolynomialTerm(c=1.0, p=[]),
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
        expected_inequality_terms=[
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
    )

    partial = instance.partial_evaluate({1: 1})
    assert partial.used_decision_variables == [x[0], x[2]]

    adapter = OMMXDA4Adapter(partial)
    qubo_request = adapter.sampler_input

    validate_qubo_request(
        qubo_request,
        expected_terms=[
            BinaryPolynomialTerm(c=1.0, p=[]),
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
        expected_inequality_terms=[
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
    )

    partial = instance.partial_evaluate({2: 1})
    assert partial.used_decision_variables == x[0:2]

    adapter = OMMXDA4Adapter(partial)
    qubo_request = adapter.sampler_input

    validate_qubo_request(
        qubo_request,
        expected_terms=[
            BinaryPolynomialTerm(c=1.0, p=[]),
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
        expected_inequality_terms=[
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ],
    )


def test_relax_constraint():
    x = [DecisionVariable.binary(i, name="x", subscripts=[i]) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] + x[1],
        constraints=[(x[0] + 2 * x[1] <= 1).set_id(0), (x[1] + x[2] <= 1).set_id(1)],
        sense=Instance.MINIMIZE,
    )

    assert instance.used_decision_variables == x
    instance.relax_constraint(1, "relax")
    # id for x[2] is listed as irrelevant
    assert instance.decision_variable_analysis().irrelevant() == {x[2].id}

    adapter = OMMXDA4Adapter(instance)
    qubo_request = adapter.sampler_input

    # After relaxing constraint 1, x[2] should not appear in binary_polynomial
    # since it's only used in the relaxed constraint
    assert qubo_request.binary_polynomial is not None

    # Expected terms: x[0] + x[1] (objective only, constraint relaxed)
    expected_terms = sort_terms(
        [
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=1.0, p=[1]),
        ]
    )
    actual_terms = sort_terms(qubo_request.binary_polynomial.terms)
    assert actual_terms == expected_terms

    # After relaxing constraint 1, only constraint 0 remains: x[0] + 2*x[1] <= 1
    assert qubo_request.inequalities is not None
    assert len(qubo_request.inequalities) == 1
    expected_inequality_terms = sort_terms(
        [
            BinaryPolynomialTerm(c=1.0, p=[0]),
            BinaryPolynomialTerm(c=2.0, p=[1]),
            BinaryPolynomialTerm(c=-1.0, p=[]),
        ]
    )
    actual_inequality_terms = sort_terms(qubo_request.inequalities[0].terms)
    assert actual_inequality_terms == expected_inequality_terms
