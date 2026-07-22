from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geo_adapter.readers.time_depth import read_time_depth_table
from geo_adapter.readers.trajectory import read_trajectory
from geo_adapter.readers.well_location import read_well_location
from geo_adapter.schemas.config import AdapterConfig


def _config(tmp_path: Path, *, trajectory: Path | None = None, location: Path | None = None, project_crs=None) -> AdapterConfig:
    return AdapterConfig.model_validate(
        {
            "sample_id": "test",
            "inputs": {
                "trajectory": {"path": trajectory},
                "well_location": {"path": location},
            },
            "coordinate_system": {"project_crs": project_crs},
            "output": {"directory": tmp_path / "run"},
        }
    )


def test_complete_trajectory(tmp_path) -> None:
    path = tmp_path / "trajectory.csv"
    pd.DataFrame({"MD": [0, 100], "TVD": [0, 99], "X": [500000, 500010], "Y": [6000000, 6000005]}).to_csv(path, index=False)
    result = read_trajectory(_config(tmp_path, trajectory=path))
    assert result.quality == "complete"
    assert result.subsurface_xy_available


def test_md_inc_azi_computes_minimum_curvature(tmp_path) -> None:
    path = tmp_path / "survey.csv"
    pd.DataFrame({"MD": [0, 100, 200], "INC": [0, 10, 20], "AZI": [0, 45, 45]}).to_csv(path, index=False)
    result = read_trajectory(_config(tmp_path, trajectory=path))
    assert result.quality == "computed"
    assert result.computation_method == "minimum_curvature"
    assert {"tvd", "x_offset", "y_offset"}.issubset(result.frame.columns)


def test_md_tvd_is_vertical_only(tmp_path) -> None:
    path = tmp_path / "vertical.csv"
    pd.DataFrame({"MD": [0, 100], "TVD": [0, 98]}).to_csv(path, index=False)
    result = read_trajectory(_config(tmp_path, trajectory=path))
    assert result.quality == "vertical_only"
    assert not result.subsurface_xy_available


def test_wgs84_longitude_latitude_is_explicit(tmp_path) -> None:
    path = tmp_path / "location.json"
    path.write_text(json.dumps({"井名": "W1", "经度": 120.0, "纬度": 30.0}, ensure_ascii=False), encoding="utf-8")
    result = read_well_location(_config(tmp_path, location=path))
    assert result["available"]
    assert result["source_crs"]["epsg"] == 4326
    assert result["source_crs"]["confidence"] == "explicit"


def test_projected_and_unknown_crs_are_distinguished(tmp_path) -> None:
    explicit = tmp_path / "explicit.csv"
    unknown = tmp_path / "unknown.csv"
    pd.DataFrame([{"X": 500000, "Y": 6000000, "CRS": "EPSG:32631"}]).to_csv(explicit, index=False)
    pd.DataFrame([{"X": 500000, "Y": 6000000}]).to_csv(unknown, index=False)
    assert read_well_location(_config(tmp_path, location=explicit))["source_crs"]["confidence"] == "explicit"
    assert read_well_location(_config(tmp_path, location=unknown))["source_crs"]["confidence"] == "unknown"


def test_checkshot_table_owt_converts_to_twt(tmp_path) -> None:
    path = tmp_path / "checkshot.csv"
    pd.DataFrame({"TVDSS": [100, 200, 300], "OWT_MS": [50, 100, 150]}).to_csv(path, index=False)
    table = read_time_depth_table(path)
    assert table["twt_ms"].tolist() == [100, 200, 300]

