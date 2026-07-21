from __future__ import annotations

import numpy as np
import pandas as pd


def integrate_sonic(
    depth_m: np.ndarray,
    slowness_us_m: np.ndarray,
    *,
    depth_axis: str,
    t0_ms: float | None = None,
    replacement_velocity_m_s: float | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Integrate sonic slowness into TWT while retaining unknown shallow time.

    A relative curve is returned when neither ``t0_ms`` nor a per-well
    replacement velocity is available. Long gaps are not bridged.
    """
    depth = np.asarray(depth_m, dtype=float)
    dt = np.asarray(slowness_us_m, dtype=float)
    if len(depth) != len(dt) or len(depth) < 2:
        raise ValueError("声波积分至少需要两个等长的深度/慢度样本")
    if not np.all(np.isfinite(depth)) or np.any(np.diff(depth) <= 0):
        raise ValueError("声波积分深度轴必须有限且严格递增")
    warnings: list[str] = []
    limitations: list[str] = []
    if depth_axis == "MD":
        limitations.append("integration_uses_md; valid_vertical_time_requires_confirmed_vertical_well")
    base = 0.0
    time_reference = "relative_from_first_sonic_sample"
    if t0_ms is not None:
        base = float(t0_ms)
        time_reference = "per_well_t0"
    elif replacement_velocity_m_s is not None:
        base = 2.0 * max(float(depth[0]), 0.0) / float(replacement_velocity_m_s) * 1000.0
        time_reference = "per_well_replacement_velocity"
    else:
        warnings.append("t0 与井级替换速度均未确定，TWT 仅相对声波起测点")
        limitations.append("absolute_twt_origin_unknown")

    twt = np.full(len(depth), np.nan, dtype=float)
    if np.isfinite(dt[0]) and dt[0] > 0:
        twt[0] = base
    invalid = ~np.isfinite(dt) | (dt <= 0)
    if invalid.any():
        warnings.append(f"声波曲线含 {int(invalid.sum())} 个不可积分样本；未跨越长缺口")
        limitations.append("sonic_gaps_limit_coverage")
    for index in range(1, len(depth)):
        if invalid[index - 1] or invalid[index] or not np.isfinite(twt[index - 1]):
            continue
        delta_depth = depth[index] - depth[index - 1]
        one_way_us = 0.5 * (dt[index - 1] + dt[index]) * delta_depth
        twt[index] = twt[index - 1] + 2.0 * one_way_us / 1000.0
    frame = pd.DataFrame(
        {
            "depth": depth,
            "depth_axis": depth_axis,
            "twt_ms": twt,
            "valid": np.isfinite(twt),
            "time_reference": time_reference,
        }
    )
    return frame, warnings, limitations

