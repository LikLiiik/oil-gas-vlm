from __future__ import annotations

from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def generated_sample_data(project_root: Path) -> None:
    namespace = runpy.run_path(str(project_root / "examples/generate_sample_data.py"))
    namespace["main"]()


@pytest.fixture(scope="session")
def prepared_run(project_root: Path, generated_sample_data: None) -> Path:
    from geo_adapter import prepare_geo_sample

    result = prepare_geo_sample(project_root / "examples/sample_config.yaml")
    assert result.success, result.errors
    assert result.output_directory is not None
    return result.output_directory
