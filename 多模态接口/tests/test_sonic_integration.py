import numpy as np
import pandas as pd
import pytest

from geo_adapter.time_depth.control_point_calibration import calibrate_with_control_points
from geo_adapter.time_depth.sonic_integrator import integrate_sonic


def test_constant_us_m_integrates_correct_twt() -> None:
    table, warnings, limitations = integrate_sonic(
        np.array([0.0, 10.0, 20.0]), np.array([500.0, 500.0, 500.0]), depth_axis="TVDSS", t0_ms=100.0
    )
    assert np.allclose(table["twt_ms"], [100.0, 110.0, 120.0])
    assert not warnings
    assert not limitations


def test_long_sonic_gap_is_not_bridged() -> None:
    table, warnings, limitations = integrate_sonic(
        np.array([0.0, 10.0, 20.0, 30.0]), np.array([500.0, np.nan, np.nan, 500.0]), depth_axis="TVDSS", t0_ms=0.0
    )
    assert table["twt_ms"].isna().iloc[1:].all()
    assert warnings and limitations


def test_uncalibrated_without_t0_is_relative_low_basis() -> None:
    table, warnings, limitations = integrate_sonic(
        np.array([100.0, 110.0]), np.array([500.0, 500.0]), depth_axis="TVDSS"
    )
    assert table["twt_ms"].iloc[0] == 0.0
    assert warnings
    assert "absolute_twt_origin_unknown" in limitations


def test_control_point_calibration_reports_error_metrics() -> None:
    sonic = pd.DataFrame({"depth": [0.0, 10.0, 20.0], "twt_ms": [0.0, 10.0, 20.0]})
    points = pd.DataFrame({"depth": [0.0, 10.0, 20.0], "twt_ms": [100.0, 120.0, 140.0]})
    calibrated, metrics = calibrate_with_control_points(sonic, points)
    assert np.allclose(calibrated["twt_ms"], [100.0, 120.0, 140.0])
    assert metrics["control_point_count"] == 3
    assert metrics["rmse_ms"] == pytest.approx(0.0, abs=1e-10)

