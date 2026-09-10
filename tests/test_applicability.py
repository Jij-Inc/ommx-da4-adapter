import pytest
from ommx import (
    DecisionVariable,
    Equality,
    Function,
    Instance,
    InstanceClassMismatch,
    Kind,
    OneHotConstraint,
    Sense,
    Sos1Constraint,
)
from ommx.adapter import AdapterNotApplicableError

from ommx_da4_adapter import OMMXDA4Adapter


@pytest.fixture
def instance_with_unsupported_special_constraints() -> Instance:
    x = DecisionVariable.binary(0)
    y = DecisionVariable.binary(1)
    return Instance.from_components(
        decision_variables=[x, y],
        objective=x + y,
        constraints={},
        indicator_constraints={10: (y <= 0).with_indicator(x)},
        sos1_constraints={30: Sos1Constraint(variables=[x, y])},
        sense=Sense.Minimize,
    )


@pytest.mark.parametrize("sense", [Sense.Minimize, Sense.Maximize])
def test_input_class_accepts_binary_quadratic_boundary(sense):
    x = [DecisionVariable.binary(i) for i in range(4)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] * x[1] + x[2] * x[3],
        constraints={
            0: x[0] * x[1] == 0,
            1: x[1] + x[2] <= 1,
        },
        one_hot_constraints={10: OneHotConstraint(variables=[x[0], x[3]])},
        sense=sense,
    )
    before = instance.to_v2_bytes()

    report = OMMXDA4Adapter.check_applicability(instance)
    OMMXDA4Adapter(instance)

    assert report.is_member
    assert report.matching_clauses == [(0, "da4-binary-polynomial-with-one-hot")]
    assert instance.to_v2_bytes() == before


def test_rejects_cubic_objective_without_mutating_input():
    x = [DecisionVariable.binary(i) for i in range(3)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] * x[1] * x[2],
        constraints={},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXDA4Adapter(instance)

    [mismatch] = error.value.report.clause_reports[0].mismatches
    assert isinstance(mismatch, InstanceClassMismatch.ObjectiveDegreeExceedsBound)
    assert mismatch.actual_degree == 3
    assert mismatch.bound.maximum_degree == 2
    assert instance.to_v2_bytes() == before


@pytest.mark.parametrize(
    ("relation", "maximum_degree"),
    [(Equality.EqualToZero, 2), (Equality.LessThanOrEqualToZero, 1)],
)
def test_rejects_constraint_above_degree_bound_without_mutating_input(
    relation, maximum_degree
):
    x = [DecisionVariable.binary(i) for i in range(3)]
    constraint = (
        x[0] * x[1] * x[2] == 0
        if relation == Equality.EqualToZero
        else x[0] * x[1] <= 0
    )
    instance = Instance.from_components(
        decision_variables=x,
        objective=sum(x),
        constraints={0: constraint},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXDA4Adapter(instance)

    [mismatch] = error.value.report.clause_reports[0].mismatches
    assert isinstance(
        mismatch, InstanceClassMismatch.RegularConstraintDegreeExceedsBound
    )
    assert mismatch.relation == relation
    assert mismatch.actual_degrees == {0: maximum_degree + 1}
    assert mismatch.bound.maximum_degree == maximum_degree
    assert instance.to_v2_bytes() == before


def test_accepts_unused_unsupported_variable_kind_without_mutating_input():
    used = DecisionVariable.binary(0)
    unused = DecisionVariable.integer(1)
    instance = Instance.from_components(
        decision_variables=[used, unused],
        objective=used,
        constraints={},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    report = OMMXDA4Adapter.check_applicability(instance)
    OMMXDA4Adapter(instance)

    assert report.is_member
    assert report.matching_clauses == [(0, "da4-binary-polynomial-with-one-hot")]
    assert instance.to_v2_bytes() == before


@pytest.mark.parametrize(
    ("variable", "kind"),
    [
        (DecisionVariable.integer(0), Kind.Integer),
        (DecisionVariable.continuous(0), Kind.Continuous),
        (DecisionVariable.semi_integer(0, lower=1, upper=3), Kind.SemiInteger),
        (
            DecisionVariable.semi_continuous(0, lower=1, upper=3),
            Kind.SemiContinuous,
        ),
    ],
)
def test_rejects_used_unsupported_variable_kinds(variable, kind):
    instance = Instance.from_components(
        decision_variables=[variable],
        objective=variable,
        constraints={},
        sense=Sense.Minimize,
    )
    before = instance.to_v2_bytes()

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXDA4Adapter(instance)

    [mismatch] = error.value.report.clause_reports[0].mismatches
    assert isinstance(mismatch, InstanceClassMismatch.VariableKindNotAllowed)
    assert mismatch.kind == kind
    assert mismatch.variable_ids == {0}
    assert mismatch.allowed_kinds == {Kind.Binary}
    assert instance.to_v2_bytes() == before


def test_asserts_if_unsupported_variable_kind_reaches_conversion(
    monkeypatch: pytest.MonkeyPatch,
):
    x = DecisionVariable.integer(0)
    instance = Instance.from_components(
        decision_variables=[x],
        objective=x,
        constraints={},
        sense=Sense.Minimize,
    )
    monkeypatch.setattr(
        OMMXDA4Adapter,
        "require_applicable",
        lambda self, instance: None,
    )

    with pytest.raises(
        AssertionError,
        match="Unsupported decision variable kind reached after applicability validation",
    ):
        OMMXDA4Adapter(instance)


def test_asserts_if_non_polynomial_objective_reaches_conversion(
    monkeypatch: pytest.MonkeyPatch,
):
    x = DecisionVariable.binary(0)
    instance = Instance.from_components(
        decision_variables=[x],
        objective=abs(Function(x - 0.5)),
        constraints={},
        sense=Sense.Minimize,
    )
    monkeypatch.setattr(
        OMMXDA4Adapter,
        "require_applicable",
        lambda self, instance: None,
    )

    with pytest.raises(
        AssertionError,
        match="Non-polynomial objective reached after applicability validation",
    ):
        OMMXDA4Adapter(instance)


def test_asserts_if_non_polynomial_constraint_reaches_conversion(
    monkeypatch: pytest.MonkeyPatch,
):
    x = DecisionVariable.binary(0)
    instance = Instance.from_components(
        decision_variables=[x],
        objective=0,
        constraints={7: abs(Function(x - 0.5)) == 0},
        sense=Sense.Minimize,
    )
    monkeypatch.setattr(
        OMMXDA4Adapter,
        "require_applicable",
        lambda self, instance: None,
    )

    with pytest.raises(
        AssertionError,
        match=(
            "Non-polynomial constraint reached after applicability validation: "
            "constraint 7"
        ),
    ):
        OMMXDA4Adapter(instance)


def test_rejects_unsupported_special_constraints_without_mutating_input(
    instance_with_unsupported_special_constraints: Instance,
) -> None:
    instance = instance_with_unsupported_special_constraints
    before = instance.to_v2_bytes()

    report = OMMXDA4Adapter.check_applicability(instance)
    assert not report.is_member
    assert instance.to_v2_bytes() == before

    with pytest.raises(AdapterNotApplicableError) as error:
        OMMXDA4Adapter(instance)

    mismatches = error.value.report.clause_reports[0].mismatches
    by_type = {type(mismatch): mismatch for mismatch in mismatches}
    assert set(by_type) == {
        InstanceClassMismatch.IndicatorConstraintsNotAllowed,
        InstanceClassMismatch.Sos1ConstraintsNotAllowed,
    }

    indicator = by_type[InstanceClassMismatch.IndicatorConstraintsNotAllowed]
    assert isinstance(indicator, InstanceClassMismatch.IndicatorConstraintsNotAllowed)
    assert indicator.constraint_ids == {10}

    sos1 = by_type[InstanceClassMismatch.Sos1ConstraintsNotAllowed]
    assert isinstance(sos1, InstanceClassMismatch.Sos1ConstraintsNotAllowed)
    assert sos1.constraint_ids == {30}
    assert instance.to_v2_bytes() == before


@pytest.mark.parametrize(
    "method_name",
    ["sample_without_preparation", "solve_without_preparation"],
)
def test_preparation_free_apis_reject_unprepared_input_without_mutation(
    method_name: str,
    instance_with_unsupported_special_constraints: Instance,
) -> None:
    instance = instance_with_unsupported_special_constraints
    before = instance.to_v2_bytes()

    method = getattr(OMMXDA4Adapter, method_name)
    with pytest.raises(AdapterNotApplicableError) as error:
        method(instance, token="test-token")

    mismatches = error.value.report.clause_reports[0].mismatches
    mismatch_types = {type(mismatch) for mismatch in mismatches}
    assert InstanceClassMismatch.IndicatorConstraintsNotAllowed in mismatch_types
    assert InstanceClassMismatch.Sos1ConstraintsNotAllowed in mismatch_types
    assert instance.to_v2_bytes() == before
