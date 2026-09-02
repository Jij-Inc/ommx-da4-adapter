import pytest
from ommx import (
    DecisionVariable,
    DegreeBound,
    Equality,
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


def test_declares_binary_polynomial_input_class() -> None:
    [clause] = OMMXDA4Adapter.INPUT_CLASS.clauses

    assert clause.label == "da4-binary-polynomial-with-one-hot"
    assert clause.allowed_variable_kinds == {Kind.Binary}
    assert clause.objective_degree_bound == DegreeBound.unbounded()
    assert clause.regular_constraint_degree_bounds == {
        Equality.EqualToZero: DegreeBound.unbounded(),
        Equality.LessThanOrEqualToZero: DegreeBound.unbounded(),
    }
    assert clause.indicator_constraint_degree_bounds == {}
    assert clause.allows_one_hot
    assert not clause.allows_sos1
    assert clause.allowed_senses == {Sense.Minimize, Sense.Maximize}


@pytest.mark.parametrize("sense", [Sense.Minimize, Sense.Maximize])
def test_input_class_accepts_complete_binary_polynomial_boundary(sense):
    x = [DecisionVariable.binary(i) for i in range(4)]
    instance = Instance.from_components(
        decision_variables=x,
        objective=x[0] * x[1] * x[2] * x[3],
        constraints={
            0: x[0] * x[1] * x[2] == 0,
            1: x[1] * x[2] <= 1,
        },
        one_hot_constraints={10: OneHotConstraint(variables=[x[0], x[3]])},
        sense=sense,
    )

    report = OMMXDA4Adapter.check_applicability(instance)

    assert report.is_member
    assert report.matching_clauses == [(0, "da4-binary-polynomial-with-one-hot")]


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
