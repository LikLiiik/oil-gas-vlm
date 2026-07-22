from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_with_control_points(
    sonic_table: pd.DataFrame, control_points: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    """Fit an affine correction from sonic TWT to measured/control TWT."""
    required = {"depth", "twt_ms"}
    if not required.issubset(sonic_table.columns) or not required.issubset(control_points.columns):
        raise ValueError("积分表和控制点都必须包含 depth、twt_ms")
    source = sonic_table.dropna(subset=["depth", "twt_ms"]).sort_values("depth")
    points = control_points.dropna(subset=["depth", "twt_ms"]).sort_values("depth")
    if len(source) < 2 or len(points) < 2:
        raise ValueError("控制点标定至少需要两个有效控制点和两个积分样本")
    in_range = points[(points["depth"] >= source["depth"].min()) & (points["depth"] <= source["depth"].max())]
    if len(in_range) < 2:
        raise ValueError("至少两个控制点必须位于声波积分覆盖范围内")
    sonic_at_points = np.interp(in_range["depth"], source["depth"], source["twt_ms"])
    measured = in_range["twt_ms"].to_numpy(dtype=float)
    design = np.column_stack([sonic_at_points, np.ones(len(sonic_at_points))])
    scale, intercept = np.linalg.lstsq(design, measured, rcond=None)[0]
    fitted = scale * sonic_at_points + intercept
    residual = measured - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    correlation = float(np.corrcoef(fitted, measured)[0, 1]) if len(measured) > 1 else None
    calibrated = sonic_table.copy()
    calibrated["twt_ms_uncalibrated"] = calibrated["twt_ms"]
    calibrated["twt_ms"] = scale * calibrated["twt_ms"] + intercept
    calibrated["calibrated"] = calibrated["twt_ms"].notna()
    return calibrated, {
        "method": "affine_control_points",
        "control_point_count": int(len(in_range)),
        "scale": float(scale),
        "intercept_ms": float(intercept),
        "rmse_ms": rmse,
        "correlation": correlation,
    }

