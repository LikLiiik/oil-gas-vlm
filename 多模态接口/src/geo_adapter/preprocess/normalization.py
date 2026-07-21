from __future__ import annotations

from typing import Any

import numpy as np


def normalize_seismic(
    values: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    method: str = "symmetric",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip finite amplitudes and normalize without modifying the source."""
    source = np.asarray(values, dtype=np.float32)
    finite = source[np.isfinite(source)]
    if finite.size == 0:
        raise ValueError("地震视图不含有限振幅值")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    clipped = np.clip(np.nan_to_num(source, nan=0.0, posinf=high, neginf=low), low, high)
    params: dict[str, Any] = {
        "method": method,
        "percentile_clip": [lower_percentile, upper_percentile],
        "clip_values": [float(low), float(high)],
        "before": {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
        },
    }
    if method == "symmetric":
        scale = max(abs(float(low)), abs(float(high)))
        processed = clipped / scale if scale > 0 else np.zeros_like(clipped)
        params["scale"] = scale
        params["output_range"] = [-1.0, 1.0]
    elif method == "minmax":
        span = float(high - low)
        processed = (clipped - low) / span if span > 0 else np.zeros_like(clipped)
        params["output_range"] = [0.0, 1.0]
    elif method == "none":
        processed = clipped
        params["output_range"] = [float(low), float(high)]
    else:
        raise ValueError(f"不支持的地震归一化方法: {method}")
    params["after"] = {
        "min": float(processed.min()),
        "max": float(processed.max()),
        "mean": float(processed.mean()),
        "std": float(processed.std()),
    }
    return processed.astype(np.float32), params

