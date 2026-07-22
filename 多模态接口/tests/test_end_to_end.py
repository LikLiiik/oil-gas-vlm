from __future__ import annotations

import json
import os
import subprocess
import sys

import jsonschema
import numpy as np

from geo_adapter import inspect_geo_sample, validate_run
from geo_adapter.schemas.manifest import Manifest


def test_inspect_is_read_only_and_reports_mappings(project_root) -> None:
    result = inspect_geo_sample(project_root / "examples/sample_config.yaml")
    assert result.success, result.errors
    assert result.inputs["well_logs"]["curve_mapping"]["RES_DEEP"]["selected"] == "ILD"
    assert result.inputs["time_depth"]["calibrated"]


def test_end_to_end_output_and_manifest_schema(prepared_run) -> None:
    result = validate_run(prepared_run)
    assert result.success, result.errors
    manifest_raw = json.loads((prepared_run / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((prepared_run / "schemas/manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(manifest_raw, schema)
    manifest = Manifest.model_validate(manifest_raw)
    assert manifest.run_mode == "multimodal_precise_aligned"
    assert manifest.alignment.horizontal_level == "seismic_crs_aligned"
    assert manifest.alignment.vertical_level == "sonic_calibrated"
    assert manifest.alignment.fusion_permission == "calibrated_joint_analysis"
    assert len(manifest.well_logs.curves) == 9


def test_masks_and_missing_are_explicit(prepared_run) -> None:
    values = np.load(prepared_run / "arrays/well_values.npy")
    valid = np.load(prepared_run / "arrays/well_valid_mask.npy")
    interpolated = np.load(prepared_run / "arrays/well_interpolated_mask.npy")
    available = np.load(prepared_run / "arrays/curve_available.npy")
    assert values.shape[1] == valid.shape[1] == interpolated.shape[1] == 9
    assert available.shape == (9,)
    assert interpolated[:, 6].sum() == 2  # AC short gap
    assert not (valid & interpolated).any()


def test_request_references_only_existing_assets(prepared_run) -> None:
    request = json.loads((prepared_run / "request.json").read_text(encoding="utf-8"))
    image_views = []
    for message in request["messages"]:
        for item in message["content"]:
            reference = item.get("path") or item.get("text_path")
            assert reference and (prepared_run / reference).is_file()
            if item["type"] == "image":
                image_views.append(item["physical_view"])
                assert (prepared_run / item["analysis_path"]).is_file()
                if item["physical_view"] != "well_log_panel":
                    assert len(item["native_shape"]) == 2
                    assert len(item["axis_labels"]) == 2
    assert len(image_views) == len(set(image_views))  # every physical view remains independent
    assert "inline" in image_views and "crossline" in image_views


def test_structured_well_summary_is_authoritative(prepared_run) -> None:
    manifest = json.loads((prepared_run / "manifest.json").read_text(encoding="utf-8"))
    rel = manifest["well_logs"]["numeric_summary_path"]
    summary = json.loads((prepared_run / rel).read_text(encoding="utf-8"))
    assert summary["source"] == "structured_well_log_table"
    assert "PNG is trend-only" in summary["policy"]
    assert summary["curve_stats"]["GR"]["count"] > 0
    assert len(summary["representative_samples"]) == 9

    request = json.loads((prepared_run / "request.json").read_text(encoding="utf-8"))
    json_items = [
        item
        for message in request["messages"]
        for item in message["content"]
        if item["type"] == "json"
    ]
    assert any(item.get("name") == "well_numeric_summary" for item in json_items)


def test_cli_validate_and_show_manifest(prepared_run, project_root) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    validate = subprocess.run(
        [sys.executable, "-m", "geo_adapter.cli", "validate", "--run-dir", str(prepared_run)],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stdout + validate.stderr
    show = subprocess.run(
        [sys.executable, "-m", "geo_adapter.cli", "show-manifest", "--run-dir", str(prepared_run)],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert show.returncode == 0, show.stdout + show.stderr
