from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geo_adapter.errors import ConfigurationError


class TaskConfig(BaseModel):
    type: str = "geological_target_detection"
    target_classes: list[str] = Field(default_factory=list)


class InputSpec(BaseModel):
    path: Path | None = None
    format: str = "auto"
    optional: bool = True
    well_id: str | None = None
    domain: Literal["auto", "time", "depth", "unknown"] = "auto"
    crs: Any | None = None
    array_key: str | None = None


class InputsConfig(BaseModel):
    seismic: InputSpec = Field(default_factory=InputSpec)
    well_log: InputSpec = Field(default_factory=InputSpec)
    well_location: InputSpec = Field(default_factory=InputSpec)
    trajectory: InputSpec = Field(default_factory=InputSpec)
    time_depth: InputSpec = Field(default_factory=InputSpec)


class CoordinateSystemConfig(BaseModel):
    project_crs: str | int | None = None
    seismic_crs: str | int | None = None
    well_crs: str | int | None = None
    allow_unknown_crs: bool = True
    require_explicit_crs_for_precise_alignment: bool = True


class DepthReferenceConfig(BaseModel):
    well_log_axis: Literal["MD", "TVD", "TVDSS"] = "MD"
    unit: str = "m"
    reference_surface: str | None = "KB"
    positive_direction: Literal["down", "up", "unknown"] = "down"
    vertical_datum: str | None = "MSL"
    tvdss_sign_convention: str = "positive_below_sea_level"


class ShortGapConfig(BaseModel):
    enabled: bool = True
    max_gap_samples: int = Field(default=3, ge=0)
    method: Literal["linear"] = "linear"


class WellLogProcessingConfig(BaseModel):
    resample_step: float | None = Field(default=None, gt=0)
    short_gap_interpolation: ShortGapConfig = Field(default_factory=ShortGapConfig)
    preferred_curves: dict[str, str] = Field(default_factory=dict)
    curve_units: dict[str, str] = Field(default_factory=dict)
    curve_descriptions: dict[str, str] = Field(default_factory=dict)
    resistivity_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)


class PercentileClipConfig(BaseModel):
    lower: float = Field(default=1.0, ge=0, le=100)
    upper: float = Field(default=99.0, ge=0, le=100)

    @model_validator(mode="after")
    def check_order(self) -> "PercentileClipConfig":
        if self.lower >= self.upper:
            raise ValueError("percentile lower must be less than upper")
        return self


class SeismicProcessingConfig(BaseModel):
    views: list[Literal["inline", "crossline", "slice", "local_patch", "patch"]] = Field(
        default_factory=lambda: ["inline", "crossline", "local_patch"]
    )
    percentile_clip: PercentileClipConfig = Field(default_factory=PercentileClipConfig)
    normalization: Literal["symmetric", "minmax", "none"] = "symmetric"
    inline_index: int | None = None
    crossline_index: int | None = None
    sample_index: int | None = None
    local_patch_radius: int = Field(default=16, ge=1)


class SonicIntegrationConfig(BaseModel):
    enabled: bool = True
    preferred_depth_axis: Literal["MD", "TVD", "TVDSS"] = "TVDSS"
    require_trajectory_for_deviated_well: bool = True


class ScalarSourceConfig(BaseModel):
    policy: str = "per_well"
    value_ms: float | None = None
    value_m_s: float | None = Field(default=None, gt=0)
    source: str | None = None


class CalibrationConfig(BaseModel):
    required_for_joint_analysis: bool = True
    method: str | None = None
    control_points_path: Path | None = None


class TimeDepthProcessingConfig(BaseModel):
    preferred_sources: list[str] = Field(
        default_factory=lambda: ["checkshot", "vsp", "provided_table", "sonic_integrated"]
    )
    sonic_integration: SonicIntegrationConfig = Field(default_factory=SonicIntegrationConfig)
    t0: ScalarSourceConfig = Field(default_factory=ScalarSourceConfig)
    replacement_velocity: ScalarSourceConfig = Field(default_factory=ScalarSourceConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class ProcessingConfig(BaseModel):
    seismic: SeismicProcessingConfig = Field(default_factory=SeismicProcessingConfig)
    well_logs: WellLogProcessingConfig = Field(default_factory=WellLogProcessingConfig)
    time_depth: TimeDepthProcessingConfig = Field(default_factory=TimeDepthProcessingConfig)


class OutputConfig(BaseModel):
    directory: Path
    overwrite: bool = False


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    sample_id: str
    task: TaskConfig = Field(default_factory=TaskConfig)
    inputs: InputsConfig
    field_mapping: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    coordinate_system: CoordinateSystemConfig = Field(default_factory=CoordinateSystemConfig)
    depth_reference: DepthReferenceConfig = Field(default_factory=DepthReferenceConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig
    curve_aliases_path: Path | None = None
    field_aliases_path: Path | None = None
    prompt_templates_path: Path | None = None


_PATH_FIELDS = ("curve_aliases_path", "field_aliases_path", "prompt_templates_path")


def load_config(path: str | Path) -> AdapterConfig:
    """Load YAML and resolve every relative path from the project/config context."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"配置文件不存在: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = AdapterConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ConfigurationError(f"配置解析失败: {exc}") from exc

    # Prefer paths relative to the current working directory when they exist;
    # otherwise resolve relative to the config file. This supports both README
    # commands run from project root and relocatable config directories.
    cwd = Path.cwd()
    base = config_path.parent

    def resolve(candidate: Path | None) -> Path | None:
        if candidate is None or candidate.is_absolute():
            return candidate
        cwd_candidate = (cwd / candidate).resolve()
        return cwd_candidate if cwd_candidate.exists() else (base / candidate).resolve()

    for spec in type(config.inputs).model_fields:
        item = getattr(config.inputs, spec)
        item.path = resolve(item.path)
    if not config.output.directory.is_absolute():
        # Output paths in documented CLI usage are rooted at the invocation
        # directory so examples/sample_config.yaml writes to project/runs.
        config.output.directory = (cwd / config.output.directory).resolve()
    config.processing.time_depth.calibration.control_points_path = resolve(
        config.processing.time_depth.calibration.control_points_path
    )
    for name in _PATH_FIELDS:
        setattr(config, name, resolve(getattr(config, name)))
    return config
