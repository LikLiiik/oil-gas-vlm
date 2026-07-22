import numpy as np
import pytest

from geo_adapter.semantics.unit_converter import convert_curve_unit


@pytest.mark.parametrize(
    ("unit", "input_value", "expected"),
    [("us/ft", 100.0, 328.0839895), ("us/m", 300.0, 300.0)],
)
def test_ac_units(unit, input_value, expected) -> None:
    values, target, _, warnings = convert_curve_unit("AC", np.array([input_value]), unit)
    assert values[0] == pytest.approx(expected)
    assert target == "us/m"
    assert not warnings


@pytest.mark.parametrize(
    ("unit", "input_values", "expected"),
    [("fraction", [0.2, 0.3], [0.2, 0.3]), ("%", [20.0, 30.0], [0.2, 0.3])],
)
def test_nphi_decimal_and_percent(unit, input_values, expected) -> None:
    values, target, _, _ = convert_curve_unit("CNL", np.array(input_values), unit)
    assert target == "fraction"
    assert np.allclose(values, expected)


def test_nphi_missing_unit_is_not_silently_scaled() -> None:
    values, _, _, warnings = convert_curve_unit("CNL", np.array([20.0, 30.0]), None)
    assert np.allclose(values, [20.0, 30.0])
    assert warnings

