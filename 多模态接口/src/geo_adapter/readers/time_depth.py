from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from geo_adapter.errors import InputDataError
from geo_adapter.models import TimeDepthData, TrajectoryData, WellLogData
from geo_adapter.readers.structured import read_structured_table
from geo_adapter.schemas.config import AdapterConfig
from geo_adapter.semantics.field_mapper import map_fields
from geo_adapter.time_depth.control_point_calibration import calibrate_with_control_points
from geo_adapter.time_depth.sonic_integrator import integrate_sonic


TIME_DEPTH_ALIASES = {
    "depth": ["DEPTH", "MD", "TVD", "TVDSS", "深度"],
    "twt_ms": ["TWT_MS", "TWT", "TWO_WAY_TIME_MS", "双程时间", "双程时间毫秒"],
    "owt_ms": ["OWT_MS", "OWT", "ONE_WAY_TIME_MS", "单程时间"],
}


def read_time_depth_table(path: Path, aliases: dict[str, list[str]] | None = None) -> pd.DataFrame:
    frame, _ = map_fields(read_structured_table(path), {**TIME_DEPTH_ALIASES, **(aliases or {})})
    if "depth" not in frame:
        raise InputDataError("时深表缺少深度列")
    if "twt_ms" not in frame and "owt_ms" in frame:
        frame["twt_ms"] = 2.0 * pd.to_numeric(frame["owt_ms"], errors="coerce")
    if "twt_ms" not in frame:
        raise InputDataError("时深表缺少 TWT/OWT 列")
    result = frame[["depth", "twt_ms"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(result) < 2:
        raise InputDataError("时深表至少需要两个有效点")
    if (np.diff(result["depth"]) <= 0).any() or (np.diff(result["twt_ms"]) <= 0).any():
        raise InputDataError("时深表的深度和 TWT 必须严格递增")
    return result.reset_index(drop=True)


def _sonic_depth_axis(
    config: AdapterConfig, well: WellLogData, trajectory: TrajectoryData | None
) -> tuple[np.ndarray, str, list[str]]:
    preferred = config.processing.time_depth.sonic_integration.preferred_depth_axis
    warnings: list[str] = []
    if preferred == well.depth_name:
        return well.depth, preferred, warnings
    if trajectory and trajectory.available and trajectory.frame is not None and "md" in trajectory.frame and preferred.lower() in trajectory.frame:
        source = trajectory.frame.dropna(subset=["md", preferred.lower()]).sort_values("md")
        if len(source) >= 2:
            converted = np.interp(well.depth, source["md"], source[preferred.lower()], left=np.nan, right=np.nan)
            warnings.append(f"声波积分轴由 {well.depth_name} 通过轨迹插值为 {preferred}")
            return converted, preferred, warnings
    if config.processing.time_depth.sonic_integration.require_trajectory_for_deviated_well and preferred != well.depth_name:
        raise InputDataError(f"要求使用 {preferred} 积分，但轨迹无法提供该深度轴")
    warnings.append(f"无法获得首选 {preferred}，退回 {well.depth_name}；不可据此假设斜井垂向时间")
    return well.depth, well.depth_name, warnings


def build_time_depth(
    config: AdapterConfig, well: WellLogData | None, trajectory: TrajectoryData | None
) -> TimeDepthData:
    spec = config.inputs.time_depth
    if spec.path is not None and spec.path.is_file():
        frame = read_time_depth_table(spec.path, config.field_mapping.get("time_depth"))
        source = spec.format if spec.format in {"checkshot", "vsp", "provided_table"} else "provided_time_depth_table"
        return TimeDepthData(
            available=True,
            source=source,
            frame=frame,
            integration_depth_axis=config.depth_reference.well_log_axis,
            calibrated=True,
            measured=True,
            control_point_count=len(frame),
            confidence="high" if source in {"checkshot", "vsp"} else "medium",
            limitations=[] if source in {"checkshot", "vsp"} else ["provided_table_quality_not_independently_verified"],
        )
    if not config.processing.time_depth.sonic_integration.enabled or well is None:
        return TimeDepthData(False, warnings=["无可靠时深表，且未执行 AC/DT 积分"], limitations=["vertical_alignment_unavailable"])
    ac = well.curves.get("AC")
    if ac is None or not ac.available or ac.values is None:
        return TimeDepthData(False, warnings=["AC/DT 整条缺失，未生成伪时深表"], limitations=["vertical_alignment_unavailable"])
    if ac.canonical_unit != "us/m" or any("单位未确认" in item for item in ac.warnings):
        return TimeDepthData(False, warnings=["AC/DT 单位不明确，禁止时深积分"], limitations=["sonic_unit_ambiguous"])
    depth, axis, axis_warnings = _sonic_depth_axis(config, well, trajectory)
    finite_depth = np.isfinite(depth)
    if finite_depth.sum() < 2:
        return TimeDepthData(False, warnings=["可用于声波积分的垂向深度样本不足"], limitations=["vertical_axis_missing"])
    # Keep curve and depth aligned, but reject non-finite/outside-trajectory depth samples.
    frame, warnings, limitations = integrate_sonic(
        depth[finite_depth],
        ac.values[finite_depth],
        depth_axis=axis,
        t0_ms=config.processing.time_depth.t0.value_ms,
        replacement_velocity_m_s=config.processing.time_depth.replacement_velocity.value_m_s,
    )
    calibration = config.processing.time_depth.calibration
    calibrated = False
    metrics: dict[str, float | int | str | None] = {}
    if calibration.control_points_path is not None:
        if not calibration.control_points_path.is_file():
            raise InputDataError(f"控制点文件不存在: {calibration.control_points_path}")
        points = read_time_depth_table(calibration.control_points_path, config.field_mapping.get("time_depth"))
        frame, metrics = calibrate_with_control_points(frame, points)
        calibrated = True
        warnings = [item for item in warnings if "t0 与井级替换速度" not in item]
        limitations = [item for item in limitations if item != "absolute_twt_origin_unknown"]
        limitations.append("control_point_calibration_valid_within_control_coverage")
    t0_value = config.processing.time_depth.t0.value_ms
    limitations = limitations + ([] if calibrated else ["sonic_integration_uncalibrated"])
    return TimeDepthData(
        available=bool(frame["twt_ms"].notna().sum() >= 2),
        source="sonic_integrated",
        frame=frame,
        integration_depth_axis=axis,
        calibrated=calibrated,
        measured=False,
        t0_ms=float(metrics.get("intercept_ms")) if calibrated else t0_value,
        replacement_velocity_m_s=config.processing.time_depth.replacement_velocity.value_m_s,
        control_point_count=int(metrics.get("control_point_count", 0)),
        rmse_ms=float(metrics["rmse_ms"]) if metrics.get("rmse_ms") is not None else None,
        correlation=float(metrics["correlation"]) if metrics.get("correlation") is not None else None,
        confidence="medium" if calibrated else "low",
        warnings=axis_warnings + warnings,
        limitations=limitations,
    )
