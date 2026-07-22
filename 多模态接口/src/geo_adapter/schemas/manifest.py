from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    type: str
    target_classes: list[str] = Field(default_factory=list)


class Availability(BaseModel):
    seismic: bool = False
    well_logs: bool = False
    well_location: bool = False
    trajectory: bool = False
    time_depth: bool = False


class DepthAxis(BaseModel):
    type: Literal["MD", "TVD", "TVDSS"]
    unit: str
    reference_surface: str | None = None
    positive_direction: str


class CurveInfo(BaseModel):
    canonical_name: str
    physical_quantity: str
    available: bool
    selected_curve: str | None = None
    alternative_curves: list[str] = Field(default_factory=list)
    alternative_curve_details: list[dict[str, Any]] = Field(default_factory=list)
    original_mnemonic: str | None = None
    original_unit: str | None = None
    canonical_unit: str | None = None
    mapping_confidence: str = "none"
    selection_reason: str | None = None
    missing_ratio: float = 1.0
    values_path: str | None = None
    valid_mask_path: str | None = None
    interpolated_mask_path: str | None = None
    investigation_depth: str | None = None
    measurement_family: str | None = None
    preprocessing: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class WellLogsInfo(BaseModel):
    available: bool
    well_id: str | None = None
    source_path: str | None = None
    depth_axis: DepthAxis | None = None
    depth_range: list[float] | None = None
    curve_order: list[str] = Field(default_factory=list)
    curves: dict[str, CurveInfo] = Field(default_factory=dict)
    numeric_summary_path: str | None = None


class CRSInfo(BaseModel):
    name: str | None = None
    datum: str | None = None
    projection: str | None = None
    epsg: int | None = None
    zone: str | None = None
    central_meridian: float | None = None
    unit: str | None = None
    axis_order: str | None = None
    source: str | None = None
    confidence: str = "unknown"


class CalibrationInfo(BaseModel):
    status: str = "unavailable"
    method: str | None = None
    t0_ms: float | None = None
    replacement_velocity_m_s: float | None = None
    control_point_count: int = 0
    rmse_ms: float | None = None
    correlation: float | None = None


class TimeDepthInfo(BaseModel):
    available: bool = False
    source: str = "none"
    table_path: str | None = None
    integration_depth_axis: str | None = None
    depth_range: list[float] | None = None
    twt_range_ms: list[float] | None = None
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    confidence: str = "none"
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AlignmentInfo(BaseModel):
    horizontal_level: str
    vertical_level: str
    fusion_permission: str
    limitations: list[str] = Field(default_factory=list)


class QualityInfo(BaseModel):
    status: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class InputFileRecord(BaseModel):
    role: str
    path: str
    size_bytes: int
    sha256: str


class Provenance(BaseModel):
    created_at: datetime
    software_version: str
    config_hash: str
    input_files: list[InputFileRecord] = Field(default_factory=list)
    processing_steps: list[dict[str, Any]] = Field(default_factory=list)


class Manifest(BaseModel):
    schema_version: str = "1.0"
    sample_id: str
    task: TaskInfo
    run_mode: str
    availability: Availability
    seismic: dict[str, Any] = Field(default_factory=dict)
    well_logs: WellLogsInfo
    well_location: dict[str, Any] = Field(default_factory=dict)
    trajectory: dict[str, Any] = Field(default_factory=dict)
    time_depth_relation: TimeDepthInfo
    alignment: AlignmentInfo
    quality: QualityInfo
    provenance: Provenance
