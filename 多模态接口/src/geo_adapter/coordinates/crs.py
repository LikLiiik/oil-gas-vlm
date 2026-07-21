from __future__ import annotations

from typing import Any

import numpy as np

from geo_adapter.errors import OptionalDependencyError


def crs_info(value: str | int | dict[str, Any] | None, source: str) -> dict[str, Any]:
    """Build conservative CRS metadata; parse EPSG only when explicit."""
    info: dict[str, Any] = {
        "name": None,
        "datum": None,
        "projection": None,
        "epsg": None,
        "zone": None,
        "central_meridian": None,
        "unit": None,
        "axis_order": None,
        "source": source,
        "confidence": "unknown",
    }
    if value is None:
        return info
    if isinstance(value, dict):
        info.update({key: val for key, val in value.items() if key in info})
        complete = bool(info.get("epsg") or (info.get("datum") and info.get("projection")))
        info["confidence"] = value.get("confidence", "explicit" if complete else "ambiguous")
        return info
    text = str(value).strip()
    info["name"] = text
    if text.isdigit():
        info["epsg"] = int(text)
        info["confidence"] = "explicit"
    elif text.upper().startswith("EPSG:") and text.split(":", 1)[1].isdigit():
        info["epsg"] = int(text.split(":", 1)[1])
        info["confidence"] = "explicit"
    else:
        info["confidence"] = "ambiguous"
    return info


def transform_xy(
    x: np.ndarray, y: np.ndarray, source_crs: str | int, target_crs: str | int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise OptionalDependencyError("坐标转换需要可选依赖: pip install -e .[crs]") from exc
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)

