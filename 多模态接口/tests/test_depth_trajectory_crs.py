import numpy as np
import pytest

from geo_adapter.coordinates.crs import crs_info
from geo_adapter.depth.minimum_curvature import minimum_curvature
from geo_adapter.depth.references import compute_tvdss


def test_minimum_curvature_vertical_well() -> None:
    result = minimum_curvature(np.array([0.0, 100.0, 200.0]), np.zeros(3), np.zeros(3))
    assert np.allclose(result["tvd"], [0.0, 100.0, 200.0])
    assert np.allclose(result[["x_offset", "y_offset"]], 0.0)


def test_minimum_curvature_deviated_has_offsets() -> None:
    result = minimum_curvature(np.array([0.0, 100.0, 200.0]), np.array([0.0, 10.0, 20.0]), np.array([0.0, 90.0, 90.0]))
    assert result["x_offset"].iloc[-1] > 0
    assert result["tvd"].iloc[-1] < 200


def test_tvdss_requires_explicit_references() -> None:
    with pytest.raises(ValueError):
        compute_tvdss(np.array([100.0]), 50.0, tvd_reference_surface="", elevation_datum="MSL")
    result, record = compute_tvdss(np.array([100.0, 200.0]), 50.0, tvd_reference_surface="KB", elevation_datum="MSL")
    assert np.allclose(result, [50.0, 150.0])
    assert record["reference_surface"] == "KB"


def test_crs_explicit_and_ambiguous() -> None:
    assert crs_info("EPSG:4326", "test")["confidence"] == "explicit"
    assert crs_info("WGS84", "test")["confidence"] == "ambiguous"
    assert crs_info(None, "test")["confidence"] == "unknown"

