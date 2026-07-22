from __future__ import annotations

import numpy as np


def build_synthetic_seismogram(
    slowness_us_m: np.ndarray,
    density_g_cm3: np.ndarray,
    sample_interval_ms: float,
    dominant_frequency_hz: float = 25.0,
) -> dict[str, np.ndarray | float]:
    """Build a transparent 1-D normal-incidence synthetic trace.

    The function performs no automatic seismic tie and makes no calibration
    claim. Inputs must already share a sampling grid.
    """
    dt = np.asarray(slowness_us_m, dtype=float)
    density = np.asarray(density_g_cm3, dtype=float)
    if dt.shape != density.shape or dt.ndim != 1 or len(dt) < 3:
        raise ValueError("AC 与 DEN 必须是一维、等长且至少三个样本")
    valid = np.isfinite(dt) & np.isfinite(density) & (dt > 0) & (density > 0)
    impedance = np.full_like(dt, np.nan)
    impedance[valid] = (1_000_000.0 / dt[valid]) * (density[valid] * 1000.0)
    reflectivity = np.zeros_like(dt)
    pair_valid = valid[1:] & valid[:-1]
    denominator = impedance[1:] + impedance[:-1]
    reflectivity[1:][pair_valid] = (
        (impedance[1:] - impedance[:-1])[pair_valid] / denominator[pair_valid]
    )
    half_width_s = max(0.064, 3.0 / dominant_frequency_hz)
    time_s = np.arange(-half_width_s, half_width_s + sample_interval_ms / 1000.0, sample_interval_ms / 1000.0)
    a = (np.pi * dominant_frequency_hz * time_s) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    synthetic = np.convolve(reflectivity, wavelet, mode="same")
    return {
        "acoustic_impedance": impedance,
        "reflectivity": reflectivity,
        "wavelet": wavelet,
        "synthetic_trace": synthetic,
        "dominant_frequency_hz": dominant_frequency_hz,
        "sample_interval_ms": sample_interval_ms,
    }

