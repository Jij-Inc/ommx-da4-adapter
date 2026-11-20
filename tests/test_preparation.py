import copy

import pytest
from ommx import (
    ConstraintKind,
    DecisionVariable,
    IndicatorConstraintID,
    Instance,
    OneHotConstraint,
    OneHotConstraintID,
    Sense,
    Sos1Constraint,
    Sos1ConstraintID,
)
from ommx.adapter import AdapterNotApplicableError

from ommx_da4_adapter import OMMXDA4Adapter


@pytest.fixture
def instance_requiring_special_constraint_lowering() -> Instance:
    indicator = DecisionVariable.binary(0)
    one_hot_variables = [DecisionVariable.binary(i) for i in range(1, 3)]
    value = DecisionVariable.binary(3)
    return Instance.from_components(
        decision_variables=[indicator, *one_hot_variables, value],
        objective=value,
        active_constraints={},
        active_indicator_constraints={30: (value <= 0).with_indicator(indicator)},
        active_one_hot_constraints={
            10: OneHotConstraint(variables=one_hot_variables),
        },
        active_sos1_constraints={20: Sos1Constraint(variables=one_hot_variables)},
        sense=Sense.Maximize,
    )


def test_recommended_preparation_policies_are_independent() -> None:
    first = OMMXDA4Adapter.recommended_preparation_policy()
    second = OMMXDA4Adapter.recommended_preparation_policy()

    assert first is not second
    first.special_constraints = None
    assert second.special_constraints is not None


def test_recommended_preparation_lowers_only_indicator_and_sos1(
    instance_requiring_special_constraint_lowering: Instance,
) -> None:
    instance = instance_requiring_special_constraint_lowering
    before = instance.to_v2_bytes()
    input_class = OMMXDA4Adapter.INPUT_CLASS

    assert not OMMXDA4Adapter.check_applicability(instance).is_member
    with pytest.raises(AdapterNotApplicableError):
        OMMXDA4Adapter(instance)
    assert instance.to_v2_bytes() == before

    prepared = copy.copy(instance)
    prepared.prepare(
        input_class,
        OMMXDA4Adapter.recommended_preparation_policy(),
    )

    assert set(instance.active_constraints) == {
        IndicatorConstraintID(30),
        OneHotConstraintID(10),
        Sos1ConstraintID(20),
    }
    assert not prepared.active_constraints.of_kind(ConstraintKind.Indicator)
    assert set(prepared.active_constraints.of_kind(ConstraintKind.OneHot)) == {
        OneHotConstraintID(10)
    }
    assert not prepared.active_constraints.of_kind(ConstraintKind.Sos1)
    assert prepared.active_special_constraint_kinds == {
        ConstraintKind.OneHot,
    }
    assert input_class.contains(prepared)
    assert OMMXDA4Adapter.check_applicability(prepared).is_member

    adapter = OMMXDA4Adapter(prepared)
    assert adapter._ommx_instance is prepared


@pytest.mark.parametrize(
    ("method_name", "preparation_free_method_name"),
    [
        ("sample", "sample_without_preparation"),
        ("solve", "solve_without_preparation"),
    ],
)
def test_easy_apis_prepare_an_isolated_copy(
    method_name: str,
    preparation_free_method_name: str,
    instance_requiring_special_constraint_lowering: Instance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = instance_requiring_special_constraint_lowering
    before = instance.to_v2_bytes()
    expected = object()
    captured = {}

    def preparation_free_method(
        cls,
        prepared,
        *,
        token=None,
        url=None,
        version=None,
        diagnostics=None,
    ):
        captured.update(
            prepared=prepared,
            token=token,
            url=url,
            version=version,
            diagnostics=diagnostics,
        )
        return expected

    monkeypatch.setattr(
        OMMXDA4Adapter,
        preparation_free_method_name,
        classmethod(preparation_free_method),
    )

    method = getattr(OMMXDA4Adapter, method_name)
    actual = method(
        instance,
        token="test-token",
        url="https://example.com/da4",
        version="v3c",
    )

    prepared = captured["prepared"]
    assert actual is expected
    assert prepared is not instance
    assert OMMXDA4Adapter.check_applicability(prepared).is_member
    assert not prepared.active_constraints.of_kind(ConstraintKind.Indicator)
    assert set(prepared.active_constraints.of_kind(ConstraintKind.OneHot)) == {
        OneHotConstraintID(10)
    }
    assert not prepared.active_constraints.of_kind(ConstraintKind.Sos1)
    assert captured == {
        "prepared": prepared,
        "token": "test-token",
        "url": "https://example.com/da4",
        "version": "v3c",
        "diagnostics": None,
    }
    assert instance.to_v2_bytes() == before
