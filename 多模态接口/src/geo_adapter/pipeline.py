from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import yaml

from geo_adapter import __version__
from geo_adapter.alignment.classifier import classify_alignment
from geo_adapter.errors import GeoAdapterError, InputDataError
from geo_adapter.models import SeismicData, TimeDepthData, TrajectoryData, WellLogData
from geo_adapter.packaging.files import sha256_file, write_json
from geo_adapter.packaging.prompt_builder import build_prompts
from geo_adapter.packaging.request_builder import build_request
from geo_adapter.qc.validators import validate_run_directory
from geo_adapter.readers.seismic import read_seismic
from geo_adapter.readers.time_depth import build_time_depth
from geo_adapter.readers.trajectory import read_trajectory
from geo_adapter.readers.well_location import read_well_location
from geo_adapter.readers.well_log import read_well_log
from geo_adapter.schemas.config import AdapterConfig, InputSpec, load_config
from geo_adapter.schemas.manifest import (
    AlignmentInfo,
    Availability,
    CalibrationInfo,
    CRSInfo,
    CurveInfo,
    DepthAxis,
    InputFileRecord,
    Manifest,
    Provenance,
    QualityInfo,
    TaskInfo,
    TimeDepthInfo,
    WellLogsInfo,
)
from geo_adapter.schemas.model_output import ExpectedModelOutput
from geo_adapter.schemas.results import InspectionResult, PrepareResult, ValidationResult
from geo_adapter.semantics.curve_mapper import CANONICAL_SLOTS, PHYSICAL_QUANTITIES
from geo_adapter.visualization.seismic import save_seismic_images
from geo_adapter.visualization.well_log import save_well_log_images


LOGGER = logging.getLogger("geo_adapter")
T = TypeVar("T")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_resource(config: AdapterConfig, explicit: Path | None, name: str) -> Path:
    if explicit is not None:
        return explicit
    development = _project_root() / "configs" / name
    if development.is_file():
        return development
    packaged = Path(__file__).resolve().parent / "resources" / name
    if packaged.is_file():
        return packaged
    raise InputDataError(f"找不到内置配置资源: {name}")


def _read_optional(
    role: str,
    spec: InputSpec,
    reader: Callable[[], T],
    warnings: list[str],
    errors: list[str],
) -> T | None:
    if spec.path is None:
        if not spec.optional:
            errors.append(f"必需输入 {role} 未配置")
        else:
            warnings.append(f"未提供 {role}")
        return None
    if not spec.path.is_file():
        message = f"{role} 文件不存在: {spec.path}"
        (warnings if spec.optional else errors).append(message)
        return None
    try:
        return reader()
    except (GeoAdapterError, ValueError, OSError) as exc:
        errors.append(f"{role} 处理失败: {exc}")
        return None


def _collect_inputs(config: AdapterConfig) -> tuple[
    SeismicData | None,
    WellLogData | None,
    dict[str, Any] | None,
    TrajectoryData | None,
    TimeDepthData,
    dict[str, Any],
    list[str],
    list[str],
]:
    warnings: list[str] = []
    errors: list[str] = []
    alias_path = _config_resource(config, config.curve_aliases_path, "curve_aliases.yaml")
    seismic = _read_optional("seismic", config.inputs.seismic, lambda: read_seismic(config), warnings, errors)
    well = _read_optional("well_log", config.inputs.well_log, lambda: read_well_log(config, alias_path), warnings, errors)
    location = _read_optional("well_location", config.inputs.well_location, lambda: read_well_location(config), warnings, errors)
    trajectory = _read_optional(
        "trajectory",
        config.inputs.trajectory,
        lambda: read_trajectory(config, location),
        warnings,
        errors,
    )
    try:
        time_depth = build_time_depth(config, well, trajectory)
    except (GeoAdapterError, ValueError, OSError) as exc:
        if config.inputs.time_depth.optional:
            warnings.append(f"time_depth 不可用: {exc}")
            time_depth = TimeDepthData(False, warnings=[str(exc)], limitations=["vertical_alignment_unavailable"])
        else:
            errors.append(f"time_depth 处理失败: {exc}")
            time_depth = TimeDepthData(False, limitations=["vertical_alignment_unavailable"])

    if seismic:
        warnings.extend(seismic.warnings)
    if well:
        warnings.extend(well.warnings)
    if location:
        warnings.extend(location.get("warnings", []))
    if trajectory:
        warnings.extend(trajectory.warnings)
    warnings.extend(time_depth.warnings)
    alignment = classify_alignment(
        seismic_available=seismic is not None,
        well_logs_available=well is not None,
        well_location=location,
        trajectory=trajectory,
        time_depth=time_depth,
        seismic_crs=seismic.crs if seismic else None,
        depth_reference_explicit=bool(
            config.depth_reference.well_log_axis
            and config.depth_reference.reference_surface
            and config.depth_reference.positive_direction != "unknown"
        ),
    )
    if seismic is None and well is None:
        errors.append("地震和测井均不可用，无法准备样本")
    return seismic, well, location, trajectory, time_depth, alignment, list(dict.fromkeys(warnings)), errors


def inspect_geo_sample(config_path: str | Path) -> InspectionResult:
    """Inspect configured inputs without writing a run package."""
    try:
        config = load_config(config_path)
        seismic, well, location, trajectory, time_depth, alignment, warnings, errors = _collect_inputs(config)
    except (GeoAdapterError, ValueError, OSError) as exc:
        return InspectionResult(success=False, errors=[str(exc)])
    inputs: dict[str, Any] = {
        "sample_id": config.sample_id,
        "run_mode": alignment["run_mode"],
        "alignment": alignment,
        "seismic": None
        if seismic is None
        else {"path": str(seismic.source_path), "shape": seismic.shape, "domain": seismic.domain, "views": list(seismic.views), "crs": seismic.crs, "qc": seismic.qc},
        "well_logs": None
        if well is None
        else {
            "path": str(well.source_path),
            "well_id": well.well_id,
            "depth_axis": well.depth_name,
            "curve_mapping": {
                name: {
                    "available": curve.available,
                    "selected": curve.selected_curve,
                    "alternatives": curve.alternative_curves,
                    "unit": [curve.original_unit, curve.canonical_unit],
                    "missing_ratio": curve.missing_ratio,
                    "measurement_family": curve.measurement_family,
                }
                for name, curve in well.curves.items()
            },
            "qc": well.qc,
        },
        "well_location": location,
        "trajectory": None
        if trajectory is None
        else {"available": trajectory.available, "quality": trajectory.quality, "qc": trajectory.qc},
        "time_depth": {
            "available": time_depth.available,
            "source": time_depth.source,
            "calibrated": time_depth.calibrated,
            "confidence": time_depth.confidence,
            "limitations": time_depth.limitations,
        },
    }
    return InspectionResult(success=not errors, inputs=inputs, warnings=warnings, errors=errors)


def _prepare_output(directory: Path, overwrite: bool) -> None:
    target = directory.resolve()
    if target == Path(target.anchor) or len(target.parts) < 3:
        raise InputDataError(f"拒绝使用过宽的输出目录: {target}")
    if target.exists():
        if not overwrite:
            raise InputDataError(f"输出目录已存在且 overwrite=false: {target}")
        if not target.is_dir():
            raise InputDataError(f"输出路径不是目录: {target}")
        shutil.rmtree(target)
    for relative in ("assets/seismic", "assets/well_logs", "arrays", "tables", "prompts", "qc", "schemas"):
        (target / relative).mkdir(parents=True, exist_ok=True)


def _save_seismic_arrays(data: SeismicData, run_dir: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for index, (name, view) in enumerate(data.views.items()):
        raw_name = "seismic_raw.npy" if index == 0 else f"seismic_{name}_raw.npy"
        processed_name = "seismic_processed.npy" if index == 0 else f"seismic_{name}_processed.npy"
        raw_path = run_dir / "arrays" / raw_name
        processed_path = run_dir / "arrays" / processed_name
        np.save(raw_path, np.asarray(view.raw), allow_pickle=False)
        np.save(processed_path, view.processed, allow_pickle=False)
        outputs[name] = {
            "physical_view": view.physical_view,
            "raw_array_path": raw_path.relative_to(run_dir).as_posix(),
            "processed_array_path": processed_path.relative_to(run_dir).as_posix(),
            "array_shape": list(view.raw.shape),
            "axis_labels": list(view.axis_labels),
            "source_indices": view.source_indices,
            "pixel_to_physical_mapping": {
                "type": "index_mapping",
                "x_axis": view.axis_labels[0],
                "y_axis": view.axis_labels[1],
                "limitations": ["physical coordinate arrays were not provided"] if data.source_format in {"npy", "npz"} else [],
            },
            "normalization": view.normalization,
        }
    return outputs


def _save_well_arrays(data: WellLogData, run_dir: Path) -> None:
    rows = len(data.depth)
    values = np.zeros((rows, len(CANONICAL_SLOTS)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    interpolated = np.zeros_like(values, dtype=bool)
    available = np.zeros(len(CANONICAL_SLOTS), dtype=bool)
    clean = pd.DataFrame({data.depth_name: data.depth})
    mapping_rows: list[dict[str, Any]] = []
    for column, slot in enumerate(CANONICAL_SLOTS):
        curve = data.curves[slot]
        if curve.available and curve.values is not None and curve.valid_mask is not None and curve.interpolated_mask is not None:
            usable = curve.valid_mask | curve.interpolated_mask
            values[usable, column] = curve.values[usable]
            valid[:, column] = curve.valid_mask
            interpolated[:, column] = curve.interpolated_mask
            available[column] = True
            clean[slot] = np.where(usable, curve.values, np.nan)
            if slot.startswith("RES_"):
                clean[f"{slot}_LOG10"] = np.where(usable & (curve.values > 0), np.log10(curve.values), np.nan)
        else:
            clean[slot] = np.nan
        mapping_rows.append(
            {
                "canonical_name": slot,
                "physical_quantity": curve.physical_quantity,
                "available": curve.available,
                "selected_curve": curve.selected_curve,
                "alternative_curves": "|".join(curve.alternative_curves),
                "alternative_curve_details": yaml.safe_dump(curve.alternative_curve_details, allow_unicode=True, default_flow_style=True).strip(),
                "original_unit": curve.original_unit,
                "canonical_unit": curve.canonical_unit,
                "mapping_confidence": curve.mapping_confidence,
                "selection_reason": curve.selection_reason,
                "investigation_depth": curve.investigation_depth,
                "measurement_family": curve.measurement_family,
                "missing_ratio": curve.missing_ratio,
                "array_column": column,
            }
        )
    np.save(run_dir / "arrays/well_values.npy", values, allow_pickle=False)
    np.save(run_dir / "arrays/well_valid_mask.npy", valid, allow_pickle=False)
    np.save(run_dir / "arrays/well_interpolated_mask.npy", interpolated, allow_pickle=False)
    np.save(run_dir / "arrays/curve_available.npy", available, allow_pickle=False)
    clean.to_csv(run_dir / "tables/well_logs_clean.csv", index=False, encoding="utf-8")
    data.raw_frame.to_csv(run_dir / "tables/well_logs_raw.csv", index=False, encoding="utf-8")
    pd.DataFrame(mapping_rows).to_csv(run_dir / "tables/curve_mapping.csv", index=False, encoding="utf-8")


def _curve_manifest(data: WellLogData | None) -> dict[str, CurveInfo]:
    output: dict[str, CurveInfo] = {}
    for index, slot in enumerate(CANONICAL_SLOTS):
        if data is None:
            output[slot] = CurveInfo(
                canonical_name=slot,
                physical_quantity=PHYSICAL_QUANTITIES[slot],
                available=False,
                limitations=["well_log_modality_unavailable"],
            )
            continue
        curve = data.curves[slot]
        preprocessing = [*curve.preprocessing, {"array_column": index}]
        output[slot] = CurveInfo(
            canonical_name=slot,
            physical_quantity=curve.physical_quantity,
            available=curve.available,
            selected_curve=curve.selected_curve,
            alternative_curves=curve.alternative_curves,
            alternative_curve_details=curve.alternative_curve_details,
            original_mnemonic=curve.selected_curve,
            original_unit=curve.original_unit,
            canonical_unit=curve.canonical_unit,
            mapping_confidence=curve.mapping_confidence,
            selection_reason=curve.selection_reason,
            missing_ratio=curve.missing_ratio,
            values_path="arrays/well_values.npy" if curve.available else None,
            valid_mask_path="arrays/well_valid_mask.npy" if curve.available else None,
            interpolated_mask_path="arrays/well_interpolated_mask.npy" if curve.available else None,
            investigation_depth=curve.investigation_depth,
            measurement_family=curve.measurement_family,
            preprocessing=preprocessing,
            warnings=curve.warnings,
            limitations=curve.limitations,
        )
    return output


def _input_records(config: AdapterConfig) -> list[InputFileRecord]:
    records: list[InputFileRecord] = []
    for role in ("seismic", "well_log", "well_location", "trajectory", "time_depth"):
        path = getattr(config.inputs, role).path
        if path is not None and path.is_file():
            records.append(InputFileRecord(role=role, path=str(path), size_bytes=path.stat().st_size, sha256=sha256_file(path)))
    control = config.processing.time_depth.calibration.control_points_path
    if control is not None and control.is_file():
        records.append(InputFileRecord(role="time_depth_control_points", path=str(control), size_bytes=control.stat().st_size, sha256=sha256_file(control)))
    return records


def _time_depth_manifest(data: TimeDepthData, table_path: str | None) -> TimeDepthInfo:
    finite = pd.DataFrame() if data.frame is None else data.frame.dropna(subset=["depth", "twt_ms"])
    if data.measured:
        status = "measured"
    elif data.calibrated:
        status = "calibrated"
    elif data.available:
        status = "uncalibrated"
    else:
        status = "unavailable"
    return TimeDepthInfo(
        available=data.available,
        source=data.source,
        table_path=table_path,
        integration_depth_axis=data.integration_depth_axis,
        depth_range=[float(finite["depth"].min()), float(finite["depth"].max())] if not finite.empty else None,
        twt_range_ms=[float(finite["twt_ms"].min()), float(finite["twt_ms"].max())] if not finite.empty else None,
        calibration=CalibrationInfo(
            status=status,
            method=("measured_or_provided_table" if data.measured else "affine_control_points" if data.calibrated else "uncalibrated" if data.available else None),
            t0_ms=data.t0_ms,
            replacement_velocity_m_s=data.replacement_velocity_m_s,
            control_point_count=data.control_point_count,
            rmse_ms=data.rmse_ms,
            correlation=data.correlation,
        ),
        confidence=data.confidence,
        warnings=data.warnings,
        limitations=data.limitations,
    )


def prepare_geo_sample(config_path: str | Path) -> PrepareResult:
    """Execute the complete adapter pipeline and return paths/status."""
    config_file = Path(config_path).expanduser().resolve()
    try:
        config = load_config(config_file)
        _prepare_output(config.output.directory, config.output.overwrite)
    except (GeoAdapterError, ValueError, OSError) as exc:
        return PrepareResult(success=False, errors=[str(exc)])
    run_dir = config.output.directory
    LOGGER.info("Preparing sample %s in %s", config.sample_id, run_dir)
    resolved_config = config.model_dump(mode="json")
    (run_dir / "input_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    seismic, well, location, trajectory, time_depth, alignment, warnings, errors = _collect_inputs(config)
    if errors:
        write_json(run_dir / "qc/quality_report.json", {"status": "invalid", "warnings": warnings, "errors": errors})
        write_json(run_dir / "qc/processing_log.json", {"sample_id": config.sample_id, "steps": [], "errors": errors})
        (run_dir / "qc/warnings.txt").write_text("\n".join([*warnings, *errors]) + "\n", encoding="utf-8")
        return PrepareResult(success=False, output_directory=run_dir, warnings=warnings, errors=errors, run_mode=alignment["run_mode"])

    seismic_images: dict[str, dict[str, str]] = {}
    well_images: dict[str, str] = {}
    seismic_views: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    if seismic:
        seismic_views = _save_seismic_arrays(seismic, run_dir)
        seismic_images = save_seismic_images(seismic, run_dir / "assets/seismic")
        for name in seismic_views:
            seismic_views[name]["model_image_path"] = Path(seismic_images[name]["model"]).relative_to(run_dir).as_posix()
            seismic_views[name]["qc_image_path"] = Path(seismic_images[name]["qc"]).relative_to(run_dir).as_posix()
        steps.append({"step": "seismic_read_extract_normalize", "views": list(seismic.views), "source_loaded_as_full_cube": seismic.source_format != "segy"})
    if well:
        _save_well_arrays(well, run_dir)
        well_images = save_well_log_images(well, run_dir / "assets/well_logs")
        steps.append({"step": "well_log_map_convert_mask", "slot_order": CANONICAL_SLOTS})
    if location:
        write_json(run_dir / "tables/well_location_normalized.json", location)
        steps.append({"step": "well_location_field_and_crs_check", "transformed": location.get("coordinates_transformed", False)})
    trajectory_table_path = None
    if trajectory and trajectory.frame is not None:
        trajectory.frame.to_csv(run_dir / "tables/trajectory_normalized.csv", index=False, encoding="utf-8")
        trajectory_table_path = "tables/trajectory_normalized.csv"
        steps.append({"step": "trajectory_normalization", "quality": trajectory.quality, "method": trajectory.computation_method})
    time_depth_table_path = None
    if time_depth.available and time_depth.frame is not None:
        time_depth.frame.to_csv(run_dir / "tables/time_depth.csv", index=False, encoding="utf-8")
        time_depth_table_path = "tables/time_depth.csv"
        steps.append({"step": "time_depth", "source": time_depth.source, "calibrated": time_depth.calibrated})

    limitations = alignment["limitations"]
    all_warnings = list(dict.fromkeys(warnings))
    quality_status = "usable_with_limitations" if limitations else "usable_with_warnings" if all_warnings else "valid"
    seismic_manifest: dict[str, Any] = {
        "available": bool(seismic),
        "source_path": str(seismic.source_path) if seismic else None,
        "source_format": seismic.source_format if seismic else None,
        "shape": list(seismic.shape) if seismic else None,
        "domain": seismic.domain if seismic else "unknown",
        "crs": seismic.crs if seismic else CRSInfo(source="missing").model_dump(),
        "views": seismic_views,
        "qc": seismic.qc if seismic else {},
        "warnings": seismic.warnings if seismic else ["seismic_modality_unavailable"],
    }
    trajectory_manifest: dict[str, Any] = {
        "available": bool(trajectory and trajectory.available),
        "quality": trajectory.quality if trajectory else "missing",
        "computation_method": trajectory.computation_method if trajectory else None,
        "subsurface_xy_available": bool(trajectory and trajectory.subsurface_xy_available),
        "table_path": trajectory_table_path,
        "qc": trajectory.qc if trajectory else {},
        "warnings": trajectory.warnings if trajectory else ["trajectory_unavailable"],
    }
    manifest = Manifest(
        sample_id=config.sample_id,
        task=TaskInfo(type=config.task.type, target_classes=config.task.target_classes),
        run_mode=alignment["run_mode"],
        availability=Availability(
            seismic=seismic is not None,
            well_logs=well is not None,
            well_location=bool(location and location.get("available")),
            trajectory=bool(trajectory and trajectory.available),
            time_depth=time_depth.available,
        ),
        seismic=seismic_manifest,
        well_logs=WellLogsInfo(
            available=well is not None,
            well_id=well.well_id if well else None,
            source_path=str(well.source_path) if well else None,
            depth_axis=DepthAxis(
                type=config.depth_reference.well_log_axis,
                unit=config.depth_reference.unit,
                reference_surface=config.depth_reference.reference_surface,
                positive_direction=config.depth_reference.positive_direction,
            )
            if well
            else None,
            depth_range=[float(well.depth.min()), float(well.depth.max())] if well else None,
            curve_order=CANONICAL_SLOTS,
            curves=_curve_manifest(well),
        ),
        well_location=location or {"available": False, "warnings": ["well_location_unavailable"]},
        trajectory=trajectory_manifest,
        time_depth_relation=_time_depth_manifest(time_depth, time_depth_table_path),
        alignment=AlignmentInfo(
            horizontal_level=alignment["horizontal_level"],
            vertical_level=alignment["vertical_level"],
            fusion_permission=alignment["fusion_permission"],
            limitations=limitations,
        ),
        quality=QualityInfo(status=quality_status, warnings=all_warnings, errors=[]),
        provenance=Provenance(
            created_at=datetime.now(timezone.utc),
            software_version=__version__,
            config_hash=sha256_file(config_file),
            input_files=_input_records(config),
            processing_steps=steps,
        ),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
    write_json(run_dir / "schemas/manifest.schema.json", Manifest.model_json_schema())
    write_json(run_dir / "schemas/expected_model_output.schema.json", ExpectedModelOutput.model_json_schema())

    templates = _config_resource(config, config.prompt_templates_path, "prompt_templates.yaml")
    system_prompt, user_prompt = build_prompts(manifest, templates)
    (run_dir / "prompts/system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "prompts/user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    request = build_request(sample_id=config.sample_id, run_dir=run_dir, seismic_images=seismic_images, well_images=well_images)
    write_json(run_dir / "request.json", request.model_dump(mode="json", exclude_none=True))

    quality_report = {
        "schema_version": "1.0",
        "sample_id": config.sample_id,
        "status": quality_status,
        "seismic": seismic.qc if seismic else {"available": False},
        "well_logs": well.qc if well else {"available": False},
        "well_location": None if location is None else {"available": location.get("available"), "crs": location.get("source_crs"), "warnings": location.get("warnings", [])},
        "trajectory": None if trajectory is None else trajectory.qc,
        "time_depth": {
            "available": time_depth.available,
            "source": time_depth.source,
            "monotonic": bool(time_depth.frame is not None and time_depth.frame.dropna(subset=["twt_ms"])["twt_ms"].is_monotonic_increasing),
            "control_point_count": time_depth.control_point_count,
            "t0_determined": time_depth.t0_ms is not None,
            "replacement_velocity_determined": time_depth.replacement_velocity_m_s is not None,
            "calibrated": time_depth.calibrated,
            "rmse_ms": time_depth.rmse_ms,
            "confidence": time_depth.confidence,
            "limitations": time_depth.limitations,
        },
        "alignment": alignment,
        "warnings": all_warnings,
        "errors": [],
    }
    write_json(run_dir / "qc/quality_report.json", quality_report)
    write_json(
        run_dir / "qc/processing_log.json",
        {
            "schema_version": "1.0",
            "sample_id": config.sample_id,
            "software_version": __version__,
            "config_hash": manifest.provenance.config_hash,
            "steps": steps,
            "input_files": [item.model_dump() for item in manifest.provenance.input_files],
        },
    )
    (run_dir / "qc/warnings.txt").write_text(("\n".join(all_warnings) if all_warnings else "无警告") + "\n", encoding="utf-8")

    validation = validate_run_directory(run_dir)
    if not validation.success:
        manifest.quality.status = "invalid"
        manifest.quality.errors = validation.errors
        write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
        quality_report["status"] = "invalid"
        quality_report["errors"] = validation.errors
        write_json(run_dir / "qc/quality_report.json", quality_report)
    return PrepareResult(
        success=validation.success,
        output_directory=run_dir,
        manifest_path=run_dir / "manifest.json",
        request_path=run_dir / "request.json",
        warnings=[*all_warnings, *validation.warnings],
        errors=validation.errors,
        run_mode=alignment["run_mode"],
        horizontal_alignment=alignment["horizontal_level"],
        vertical_alignment=alignment["vertical_level"],
        fusion_permission=alignment["fusion_permission"],
    )


def validate_run(run_dir: str | Path) -> ValidationResult:
    """Public validation API."""
    return validate_run_directory(run_dir)
