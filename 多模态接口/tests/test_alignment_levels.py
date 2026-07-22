import pytest

from geo_adapter.alignment.classifier import classify_alignment
from geo_adapter.models import TimeDepthData, TrajectoryData


def _location(explicit=True):
    return {
        "available": True,
        "x": 1.0,
        "y": 2.0,
        "source_crs": {"confidence": "explicit" if explicit else "unknown", "epsg": 32631, "name": "EPSG:32631"},
    }


SEISMIC_CRS = {"confidence": "explicit", "epsg": 32631, "name": "EPSG:32631"}


@pytest.mark.parametrize(
    ("seismic", "well", "location", "trajectory", "time_depth", "mode"),
    [
        (True, False, None, None, None, "seismic_only"),
        (False, True, None, None, None, "well_log_only"),
        (True, True, None, None, None, "multimodal_unaligned"),
        (True, True, _location(), None, None, "multimodal_location_aligned"),
        (True, True, _location(), TrajectoryData(True, quality="complete", subsurface_xy_available=True), TimeDepthData(True, source="sonic_integrated", confidence="low"), "multimodal_approximate_aligned"),
        (True, True, _location(), TrajectoryData(True, quality="complete", subsurface_xy_available=True), TimeDepthData(True, source="checkshot", measured=True, calibrated=True, confidence="high"), "multimodal_precise_aligned"),
    ],
)
def test_run_modes(seismic, well, location, trajectory, time_depth, mode) -> None:
    result = classify_alignment(
        seismic_available=seismic,
        well_logs_available=well,
        well_location=location,
        trajectory=trajectory,
        time_depth=time_depth,
        seismic_crs=SEISMIC_CRS,
        depth_reference_explicit=True,
    )
    assert result["run_mode"] == mode


def test_unknown_crs_never_reaches_h3() -> None:
    result = classify_alignment(
        seismic_available=True,
        well_logs_available=True,
        well_location=_location(explicit=False),
        trajectory=TrajectoryData(True, quality="complete", subsurface_xy_available=True),
        time_depth=TimeDepthData(True, source="sonic_integrated"),
        seismic_crs=SEISMIC_CRS,
        depth_reference_explicit=True,
    )
    assert result["horizontal_level"] == "trajectory_available"
    assert result["fusion_permission"] != "approximate_vertical_mapping"


def test_uncalibrated_sonic_never_gets_high_permission() -> None:
    result = classify_alignment(
        seismic_available=True,
        well_logs_available=True,
        well_location=_location(),
        trajectory=TrajectoryData(True, quality="complete", subsurface_xy_available=True),
        time_depth=TimeDepthData(True, source="sonic_integrated", calibrated=False),
        seismic_crs=SEISMIC_CRS,
        depth_reference_explicit=True,
    )
    assert result["vertical_level"] == "sonic_uncalibrated"
    assert result["fusion_permission"] == "approximate_vertical_mapping"
