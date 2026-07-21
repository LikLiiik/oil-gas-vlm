from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProcessedCurve:
    canonical_name: str
    physical_quantity: str
    available: bool
    selected_curve: str | None = None
    alternative_curves: list[str] = field(default_factory=list)
    alternative_curve_details: list[dict[str, Any]] = field(default_factory=list)
    original_unit: str | None = None
    canonical_unit: str | None = None
    mapping_confidence: str = "none"
    selection_reason: str | None = None
    investigation_depth: str | None = None
    measurement_family: str | None = None
    raw_values: np.ndarray | None = None
    values: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    interpolated_mask: np.ndarray | None = None
    missing_ratio: float = 1.0
    max_gap_samples: int = 0
    preprocessing: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class WellLogData:
    source_path: Path
    well_id: str | None
    depth: np.ndarray
    depth_name: str
    depth_unit: str
    raw_frame: pd.DataFrame
    curves: dict[str, ProcessedCurve]
    warnings: list[str] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeismicView:
    name: str
    physical_view: str
    raw: np.ndarray
    processed: np.ndarray
    axis_labels: tuple[str, str]
    source_indices: dict[str, int | list[int]]
    normalization: dict[str, Any]


@dataclass
class SeismicData:
    source_path: Path
    shape: tuple[int, ...]
    domain: str
    views: dict[str, SeismicView]
    crs: dict[str, Any]
    source_format: str
    warnings: list[str] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryData:
    available: bool
    frame: pd.DataFrame | None = None
    quality: str = "missing"
    computation_method: str | None = None
    subsurface_xy_available: bool = False
    warnings: list[str] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeDepthData:
    available: bool
    source: str = "none"
    frame: pd.DataFrame | None = None
    integration_depth_axis: str | None = None
    calibrated: bool = False
    measured: bool = False
    t0_ms: float | None = None
    replacement_velocity_m_s: float | None = None
    control_point_count: int = 0
    rmse_ms: float | None = None
    correlation: float | None = None
    confidence: str = "none"
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
