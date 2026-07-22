"""Optional read-only smoke tests against the user's local competition dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from geo_adapter.errors import OptionalDependencyError
from geo_adapter.readers.seismic import read_seismic
from geo_adapter.readers.well_log import read_well_log
from geo_adapter.schemas.config import AdapterConfig


def _dataset_root(project_root: Path) -> Path:
    return project_root.parent / "数据集-预处理"


def test_real_teapot_las_mapping_read_only(project_root, tmp_path) -> None:
    las_path = _dataset_root(project_root) / "Teapot Dome/Teapot_ML/well_logs/deeper/490252280700/75X10.LAS"
    if not las_path.is_file():
        pytest.skip("本机未提供 Teapot Dome 抽样 LAS")
    config = AdapterConfig.model_validate(
        {
            "sample_id": "real_las_smoke",
            "inputs": {"well_log": {"path": las_path, "optional": False}},
            "output": {"directory": tmp_path / "unused"},
        }
    )
    data = read_well_log(config, project_root / "configs/curve_aliases.yaml")
    assert len(data.depth) == 11301
    # These are source mnemonics observed in this LAS, not fabricated replacements.
    assert data.curves["GR"].selected_curve == "GR"
    assert data.curves["SP"].selected_curve == "SP"
    assert data.curves["CAL"].selected_curve == "CAL"
    assert data.curves["RES_DEEP"].selected_curve == "RD"
    assert data.curves["RES_MEDIUM_SHALLOW"].selected_curve == "RS"
    assert data.curves["DEN"].selected_curve == "ZDEN"
    assert data.curves["CNL"].selected_curve == "CNC"
    assert not data.curves["AC"].available


def test_real_segy_uses_optional_dependency_or_lazy_slice(project_root, tmp_path) -> None:
    segy_path = _dataset_root(project_root) / "Teapot Dome/rmotc/DataSets/Seismic/CD files/3D_Seismic/filt_mig.sgy"
    if not segy_path.is_file():
        pytest.skip("本机未提供 Teapot Dome 抽样 SEG-Y")
    config = AdapterConfig.model_validate(
        {
            "sample_id": "real_segy_smoke",
            "inputs": {"seismic": {"path": segy_path, "domain": "time", "optional": False}},
            "processing": {"seismic": {"views": ["inline"]}},
            "output": {"directory": tmp_path / "unused"},
        }
    )
    if importlib.util.find_spec("segyio") is None:
        with pytest.raises(OptionalDependencyError, match=r"pip install -e .*\[segy\]"):
            read_seismic(config)
    else:
        data = read_seismic(config)
        assert data.source_format == "segy"
        assert "inline" in data.views
        assert data.qc["metadata"]["lazy_slice_policy"] is True

