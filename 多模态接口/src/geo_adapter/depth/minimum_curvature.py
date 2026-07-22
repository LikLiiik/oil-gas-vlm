from __future__ import annotations

import numpy as np
import pandas as pd


def minimum_curvature(md: np.ndarray, inclination_deg: np.ndarray, azimuth_deg: np.ndarray) -> pd.DataFrame:
    """Compute TVD, east and north offsets using the minimum-curvature method.

    Azimuth is interpreted clockwise from north; angles are in degrees and MD
    increments retain the input length unit.
    """
    md = np.asarray(md, dtype=float)
    inc = np.deg2rad(np.asarray(inclination_deg, dtype=float))
    azi = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    if not (len(md) == len(inc) == len(azi)) or len(md) == 0:
        raise ValueError("MD、INC、AZI 长度必须相同且非空")
    if not np.all(np.isfinite(md)) or not np.all(np.isfinite(inc)) or not np.all(np.isfinite(azi)):
        raise ValueError("MD、INC、AZI 不能包含缺失值")
    if np.any(np.diff(md) < 0):
        raise ValueError("MD 必须单调非递减")

    tvd = np.zeros(len(md), dtype=float)
    east = np.zeros(len(md), dtype=float)
    north = np.zeros(len(md), dtype=float)
    tvd[0] = md[0] * np.cos(inc[0])
    east[0] = md[0] * np.sin(inc[0]) * np.sin(azi[0])
    north[0] = md[0] * np.sin(inc[0]) * np.cos(azi[0])
    for index in range(1, len(md)):
        delta_md = md[index] - md[index - 1]
        cosine = (
            np.cos(inc[index - 1]) * np.cos(inc[index])
            + np.sin(inc[index - 1]) * np.sin(inc[index]) * np.cos(azi[index] - azi[index - 1])
        )
        dogleg = np.arccos(np.clip(cosine, -1.0, 1.0))
        ratio = 1.0 if abs(dogleg) < 1e-12 else 2.0 * np.tan(dogleg / 2.0) / dogleg
        tvd[index] = tvd[index - 1] + 0.5 * delta_md * (
            np.cos(inc[index - 1]) + np.cos(inc[index])
        ) * ratio
        east[index] = east[index - 1] + 0.5 * delta_md * (
            np.sin(inc[index - 1]) * np.sin(azi[index - 1])
            + np.sin(inc[index]) * np.sin(azi[index])
        ) * ratio
        north[index] = north[index - 1] + 0.5 * delta_md * (
            np.sin(inc[index - 1]) * np.cos(azi[index - 1])
            + np.sin(inc[index]) * np.cos(azi[index])
        ) * ratio
    return pd.DataFrame({"md": md, "tvd": tvd, "x_offset": east, "y_offset": north})

