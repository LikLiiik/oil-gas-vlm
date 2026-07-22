from __future__ import annotations

import numpy as np


def compute_tvdss(
    tvd: np.ndarray,
    reference_elevation: float,
    *,
    tvd_reference_surface: str,
    elevation_datum: str,
    sign_convention: str = "positive_below_sea_level",
) -> tuple[np.ndarray, dict[str, object]]:
    """Convert TVD from a confirmed elevated reference to TVDSS."""
    if not tvd_reference_surface or not elevation_datum:
        raise ValueError("计算 TVDSS 前必须明确 TVD 参考面与高程基准")
    result = np.asarray(tvd, dtype=float) - float(reference_elevation)
    if sign_convention != "positive_below_sea_level":
        result = -result
    return result, {
        "operation": "tvd_to_tvdss",
        "formula": "TVDSS=TVD-reference_elevation",
        "reference_surface": tvd_reference_surface,
        "reference_elevation": float(reference_elevation),
        "elevation_datum": elevation_datum,
        "sign_convention": sign_convention,
    }

