from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from pydantic import ValidationError

from geo_adapter.schemas.manifest import Manifest
from geo_adapter.schemas.request import ModelRequest
from geo_adapter.schemas.results import ValidationResult


def _safe_reference(run_dir: Path, relative: str) -> Path:
    target = (run_dir / relative).resolve()
    if not target.is_relative_to(run_dir.resolve()):
        raise ValueError(f"路径引用越出运行目录: {relative}")
    return target


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run_directory(run_dir: str | Path) -> ValidationResult:
    """Validate schemas, referenced files, masks, images, and alignment claims."""
    root = Path(run_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: set[Path] = set()
    if not root.is_dir():
        return ValidationResult(success=False, errors=[f"运行目录不存在: {root}"])
    required = [
        "input_config.yaml",
        "manifest.json",
        "request.json",
        "prompts/system_prompt.txt",
        "prompts/user_prompt.txt",
        "qc/quality_report.json",
        "qc/processing_log.json",
        "schemas/expected_model_output.schema.json",
        "schemas/manifest.schema.json",
    ]
    for relative in required:
        path = root / relative
        if not path.is_file():
            errors.append(f"缺少必需文件: {relative}")
        else:
            checked.add(path)
    if errors:
        return ValidationResult(success=False, errors=errors, checked_files=len(checked))

    try:
        raw_manifest = _load_json(root / "manifest.json")
        manifest = Manifest.model_validate(raw_manifest)
        manifest_schema = _load_json(root / "schemas/manifest.schema.json")
        jsonschema.Draft202012Validator.check_schema(manifest_schema)
        jsonschema.validate(raw_manifest, manifest_schema)
    except (OSError, ValueError, ValidationError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        errors.append(f"manifest 校验失败: {exc}")
        manifest = None

    try:
        raw_request = _load_json(root / "request.json")
        request = ModelRequest.model_validate(raw_request)
        expected_schema = _load_json(root / request.expected_output_schema)
        jsonschema.Draft202012Validator.check_schema(expected_schema)
        checked.add(root / request.expected_output_schema)
        image_names: set[str] = set()
        for message in request.messages:
            for item in message.content:
                reference = item.path or item.text_path
                if not reference:
                    errors.append(f"request 的 {item.type} 项缺少路径")
                    continue
                try:
                    path = _safe_reference(root, reference)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if not path.is_file():
                    errors.append(f"request 引用不存在: {reference}")
                    continue
                checked.add(path)
                if item.type == "image":
                    if not item.name or not item.physical_view:
                        errors.append(f"图像项缺少 name/physical_view: {reference}")
                    if item.name in image_names:
                        errors.append(f"图像名称重复: {item.name}")
                    image_names.add(item.name or "")
                    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                        errors.append(f"图像不是有效 PNG 签名: {reference}")
    except (OSError, ValueError, ValidationError, jsonschema.SchemaError) as exc:
        errors.append(f"request/schema 校验失败: {exc}")
        request = None

    if manifest:
        # Every manifest asset/table/array path must exist.
        references: list[str] = []
        for view in (manifest.seismic.get("views") or {}).values():
            references.extend(
                value for key, value in view.items() if key.endswith("_path") and isinstance(value, str)
            )
        for curve in manifest.well_logs.curves.values():
            references.extend(
                value
                for value in (curve.values_path, curve.valid_mask_path, curve.interpolated_mask_path)
                if value
            )
        references.extend(
            value
            for value in (
                manifest.trajectory.get("table_path"),
                manifest.time_depth_relation.table_path,
            )
            if value
        )
        for reference in references:
            try:
                path = _safe_reference(root, reference)
                if not path.is_file():
                    errors.append(f"manifest 引用不存在: {reference}")
                else:
                    checked.add(path)
            except ValueError as exc:
                errors.append(str(exc))

        arrays_dir = root / "arrays"
        well_values = arrays_dir / "well_values.npy"
        if manifest.availability.well_logs:
            expected = [well_values, arrays_dir / "well_valid_mask.npy", arrays_dir / "well_interpolated_mask.npy", arrays_dir / "curve_available.npy"]
            if not all(path.is_file() for path in expected):
                errors.append("测井可用但数组/Mask/curve_available 不完整")
            else:
                values = np.load(expected[0], allow_pickle=False)
                valid = np.load(expected[1], allow_pickle=False)
                interpolated = np.load(expected[2], allow_pickle=False)
                available = np.load(expected[3], allow_pickle=False)
                checked.update(expected)
                if values.shape != valid.shape or values.shape != interpolated.shape:
                    errors.append(f"测井值与 Mask 维度不一致: {values.shape}/{valid.shape}/{interpolated.shape}")
                if values.ndim != 2 or values.shape[1] != 9:
                    errors.append(f"测井数组必须为 (N,9)，实际 {values.shape}")
                if available.shape != (9,):
                    errors.append(f"curve_available 必须为 (9,)，实际 {available.shape}")
                if np.any(valid & interpolated):
                    errors.append("valid_mask 与 interpolated_mask 不应在同一样本同时为真")

        horizontal = manifest.alignment.horizontal_level
        vertical = manifest.alignment.vertical_level
        fusion = manifest.alignment.fusion_permission
        if horizontal == "seismic_crs_aligned":
            seismic_crs = manifest.seismic.get("crs") or {}
            location_crs = manifest.well_location.get("project_crs") or manifest.well_location.get("source_crs") or {}
            same = (
                seismic_crs.get("confidence") == "explicit"
                and location_crs.get("confidence") == "explicit"
                and (
                    seismic_crs.get("epsg") == location_crs.get("epsg")
                    or (seismic_crs.get("name") and seismic_crs.get("name") == location_crs.get("name"))
                )
            )
            if not same or not manifest.trajectory.get("subsurface_xy_available"):
                errors.append("H3/seismic_crs_aligned 声明与 CRS/地下轨迹状态不一致")
        td = manifest.time_depth_relation
        if vertical == "sonic_uncalibrated" and (td.source != "sonic_integrated" or td.calibration.status != "uncalibrated"):
            errors.append("V2 声明与声波未标定状态不一致")
        if vertical == "sonic_calibrated" and td.calibration.status != "calibrated":
            errors.append("V3 声明与标定状态不一致")
        if vertical == "measured_time_depth" and td.source not in {"checkshot", "vsp", "provided_time_depth_table"}:
            errors.append("V4 声明缺少实测/外部时深来源")
        allowed_fusions = {
            "precise_joint_analysis": horizontal == "seismic_crs_aligned" and vertical == "measured_time_depth",
            "calibrated_joint_analysis": horizontal == "seismic_crs_aligned" and vertical == "sonic_calibrated",
            "approximate_vertical_mapping": horizontal == "seismic_crs_aligned" and vertical == "sonic_uncalibrated",
            "location_level_association": horizontal != "none",
            "separate_analysis_only": True,
        }
        if not allowed_fusions.get(fusion, False):
            errors.append(f"融合权限 {fusion} 高于当前 H/V 状态允许范围")
        if manifest.run_mode == "multimodal_precise_aligned" and fusion not in {"precise_joint_analysis", "calibrated_joint_analysis"}:
            errors.append("运行模式与融合权限不一致")

    return ValidationResult(
        success=not errors,
        warnings=warnings,
        errors=errors,
        checked_files=len(checked),
    )

