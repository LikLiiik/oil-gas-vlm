from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from geo_adapter.depth.minimum_curvature import minimum_curvature
from geo_adapter.depth.references import compute_tvdss
from geo_adapter.errors import InputDataError
from geo_adapter.models import TrajectoryData
from geo_adapter.readers.structured import read_structured_table
from geo_adapter.schemas.config import AdapterConfig
from geo_adapter.semantics.field_mapper import map_fields


DEFAULT_ALIASES = {
    "well_id": ["WELL", "WELL_NAME", "WELL_ID", "井名"],
    "md": ["MD", "MEASURED_DEPTH", "测量井深"],
    "tvd": ["TVD", "TRUE_VERTICAL_DEPTH", "垂深"],
    "tvdss": ["TVDSS"],
    "inclination": ["INC", "INCLINATION", "井斜角"],
    "azimuth": ["AZI", "AZIMUTH", "方位角"],
    "x_offset": ["DX", "X_OFFSET", "X偏移量"],
    "y_offset": ["DY", "Y_OFFSET", "Y偏移量"],
    "x_absolute": ["X", "EASTING"],
    "y_absolute": ["Y", "NORTHING"],
}


def _numeric(frame: pd.DataFrame, names: list[str]) -> None:
    for name in names:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")


def read_trajectory(config: AdapterConfig, well_location: dict[str, Any] | None = None) -> TrajectoryData:
    spec = config.inputs.trajectory
    if spec.path is None or not spec.path.is_file():
        raise InputDataError(f"井轨迹文件不存在: {spec.path}")
    aliases = {**DEFAULT_ALIASES, **config.field_mapping.get("trajectory", {})}
    frame, mapping = map_fields(read_structured_table(spec.path), aliases)
    _numeric(frame, ["md", "tvd", "tvdss", "inclination", "azimuth", "x_offset", "y_offset", "x_absolute", "y_absolute"])
    if "md" not in frame or frame.empty or frame["md"].isna().any():
        raise InputDataError("轨迹必须包含完整 MD")
    if (np.diff(frame["md"].to_numpy(dtype=float)) < 0).any():
        raise InputDataError("轨迹 MD 必须单调非递减")

    warnings: list[str] = []
    computation_method = None
    has_tvd = "tvd" in frame and frame["tvd"].notna().all()
    has_angles = all(name in frame and frame[name].notna().all() for name in ("inclination", "azimuth"))
    has_offsets = all(name in frame and frame[name].notna().all() for name in ("x_offset", "y_offset"))
    has_absolute = all(name in frame and frame[name].notna().all() for name in ("x_absolute", "y_absolute"))

    if not has_tvd and has_angles:
        calculated = minimum_curvature(frame["md"].to_numpy(), frame["inclination"].to_numpy(), frame["azimuth"].to_numpy())
        for name in ("tvd", "x_offset", "y_offset"):
            frame[name] = calculated[name]
        has_tvd = has_offsets = True
        computation_method = "minimum_curvature"
        quality = "computed"
    elif has_tvd and (has_offsets or has_absolute):
        quality = "complete"
    elif has_tvd:
        quality = "vertical_only"
    else:
        quality = "missing"
        warnings.append("轨迹字段不足：既无 TVD，也无完整 INC/AZI")

    well_x = None if not well_location else well_location.get("x")
    well_y = None if not well_location else well_location.get("y")
    if has_offsets and well_x is not None and well_y is not None:
        computed_x = float(well_x) + frame["x_offset"]
        computed_y = float(well_y) + frame["y_offset"]
        if has_absolute:
            errors = np.hypot(frame["x_absolute"] - computed_x, frame["y_absolute"] - computed_y)
            max_error = float(errors.max())
            if max_error > 1.0:
                warnings.append(f"绝对坐标与井口+偏移最大不一致 {max_error:.3f}（坐标单位）")
        else:
            frame["x_absolute"] = computed_x
            frame["y_absolute"] = computed_y
            has_absolute = True

    conversion_record = None
    if has_tvd and "tvdss" not in frame and well_location and well_location.get("kb_elevation") is not None:
        try:
            frame["tvdss"], conversion_record = compute_tvdss(
                frame["tvd"].to_numpy(),
                float(well_location["kb_elevation"]),
                tvd_reference_surface=config.depth_reference.reference_surface or "",
                elevation_datum=config.depth_reference.vertical_datum or "",
                sign_convention=config.depth_reference.tvdss_sign_convention,
            )
        except ValueError as exc:
            warnings.append(str(exc))

    tvd_reasonable = None
    if has_tvd:
        tvd = frame["tvd"].to_numpy(dtype=float)
        tvd_reasonable = bool(np.all(np.diff(tvd) >= -1e-8) and np.all(tvd <= frame["md"].to_numpy() + 1e-6))
        if not tvd_reasonable:
            warnings.append("TVD 与 MD 的关系异常，请核实单位、参考面或轨迹")
    return TrajectoryData(
        available=quality != "missing",
        frame=frame,
        quality=quality,
        computation_method=computation_method,
        subsurface_xy_available=has_absolute,
        warnings=warnings,
        qc={
            "row_count": len(frame),
            "md_monotonic": True,
            "tvd_reasonable": tvd_reasonable,
            "absolute_xy_available": has_absolute,
            "offset_xy_available": has_offsets,
            "field_mapping": mapping,
            "tvdss_conversion": conversion_record,
        },
    )

