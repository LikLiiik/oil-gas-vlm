from pathlib import Path

import numpy as np

from geo_adapter.readers.seismic import read_seismic
from geo_adapter.schemas.config import AdapterConfig


def _config(tmp_path: Path, path: Path, views) -> AdapterConfig:
    return AdapterConfig.model_validate(
        {
            "sample_id": "seismic",
            "inputs": {"seismic": {"path": path, "domain": "time", "crs": "EPSG:32631"}},
            "processing": {"seismic": {"views": views}},
            "output": {"directory": tmp_path / "run"},
        }
    )


def test_3d_views_are_independent_not_rgb(tmp_path) -> None:
    path = tmp_path / "cube.npy"
    np.save(path, np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10))
    data = read_seismic(_config(tmp_path, path, ["inline", "crossline", "slice", "local_patch"]))
    assert set(data.views) == {"inline", "crossline", "slice", "local_patch"}
    assert all(view.raw.ndim == 2 for view in data.views.values())
    assert data.views["inline"].physical_view != data.views["crossline"].physical_view


def test_2d_input_is_labeled_user_patch(tmp_path) -> None:
    path = tmp_path / "patch.npz"
    np.savez(path, amplitude=np.ones((10, 12), dtype=np.float32))
    data = read_seismic(_config(tmp_path, path, ["inline"]))
    assert list(data.views) == ["patch"]
    assert data.views["patch"].physical_view == "user_provided_2d_patch"

