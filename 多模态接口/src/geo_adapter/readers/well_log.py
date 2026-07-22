from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geo_adapter.errors import InputDataError, OptionalDependencyError
from geo_adapter.models import WellLogData
from geo_adapter.schemas.config import AdapterConfig
from geo_adapter.semantics.curve_mapper import load_curve_aliases, map_and_process_curves


DEPTH_ALIASES = {"DEPT", "DEPTH", "MD", "MEASURED_DEPTH", "井深", "测量井深"}


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).strip().upper() if ch not in " _-./\\()[]")


def _read_las(path: Path) -> tuple[pd.DataFrame, dict[str, str | None], dict[str, str], str | None]:
    try:
        import lasio
    except ImportError as exc:
        raise OptionalDependencyError("读取 LAS 需要可选依赖: pip install -e .[las]") from exc
    try:
        las = lasio.read(path, encoding="utf-8", ignore_header_errors=True)
    except Exception as exc:  # lasio exposes format-specific exceptions inconsistently
        raise InputDataError(f"LAS 读取失败: {path}: {exc}") from exc
    frame = las.df().reset_index()
    units: dict[str, str | None] = {}
    descriptions: dict[str, str] = {}
    for curve in las.curves:
        units[str(curve.mnemonic)] = str(curve.unit).strip() or None
        descriptions[str(curve.mnemonic)] = str(curve.descr).strip()
    well_id = None
    for key in ("WELL", "UWI", "API"):
        try:
            value = str(las.well[key].value).strip()
            if value:
                well_id = value
                break
        except (KeyError, AttributeError):
            continue
    return frame, units, descriptions, well_id


def read_well_log(config: AdapterConfig, alias_path: Path) -> WellLogData:
    """Read LAS/CSV and map curves to the nine independent semantic slots."""
    spec = config.inputs.well_log
    if spec.path is None or not spec.path.is_file():
        raise InputDataError(f"测井文件不存在: {spec.path}")
    suffix = spec.path.suffix.lower()
    if suffix == ".las":
        frame, units, descriptions, source_well_id = _read_las(spec.path)
    elif suffix == ".csv":
        try:
            frame = pd.read_csv(spec.path)
        except (OSError, ValueError) as exc:
            raise InputDataError(f"CSV 测井读取失败: {spec.path}: {exc}") from exc
        units = {str(key): value for key, value in config.processing.well_logs.curve_units.items()}
        descriptions = config.processing.well_logs.curve_descriptions.copy()
        source_well_id = None
    else:
        raise InputDataError(f"第一版仅支持 LAS/CSV 测井，不支持: {suffix}")
    if frame.empty:
        raise InputDataError("测井表为空")

    depth_column = next((str(column) for column in frame.columns if _norm(column) in {_norm(x) for x in DEPTH_ALIASES}), None)
    if depth_column is None:
        raise InputDataError(f"未找到深度列，支持别名: {sorted(DEPTH_ALIASES)}")
    depth = pd.to_numeric(frame[depth_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(depth).all():
        raise InputDataError("测井深度列含缺失或非数值，无法建立可靠深度轴")
    aliases = load_curve_aliases(alias_path)
    curve_frame = frame.drop(columns=[depth_column])
    curves = map_and_process_curves(curve_frame, units, descriptions, aliases, config.processing.well_logs)

    delta = np.diff(depth)
    duplicate_count = int(pd.Series(depth).duplicated().sum())
    warnings: list[str] = []
    if np.any(delta <= 0):
        warnings.append("测井深度不是严格单调递增，未静默重排")
    if duplicate_count:
        warnings.append(f"测井深度含 {duplicate_count} 个重复值")
    available_count = sum(curve.available for curve in curves.values())
    qc: dict[str, Any] = {
        "readable": True,
        "row_count": len(frame),
        "depth_monotonic_increasing": bool(np.all(delta > 0)),
        "duplicate_depth_count": duplicate_count,
        "depth_range": [float(depth.min()), float(depth.max())],
        "depth_unit": config.depth_reference.unit,
        "available_curve_count": available_count,
        "available_curves": [name for name, curve in curves.items() if curve.available],
        "missing_curves": [name for name, curve in curves.items() if not curve.available],
        "curves": {
            name: {
                "available": curve.available,
                "missing_ratio": curve.missing_ratio,
                "max_consecutive_gap_samples": curve.max_gap_samples,
                "nonpositive_resistivity_invalidated": next(
                    (
                        step.get("count", 0)
                        for step in curve.preprocessing
                        if step.get("operation") == "invalidate_nonpositive_resistivity"
                    ),
                    0,
                ),
                "unit_known": bool(curve.original_unit),
                "warnings": curve.warnings,
            }
            for name, curve in curves.items()
        },
    }
    return WellLogData(
        source_path=spec.path,
        well_id=spec.well_id or source_well_id,
        depth=depth,
        depth_name=config.depth_reference.well_log_axis,
        depth_unit=config.depth_reference.unit,
        raw_frame=frame,
        curves=curves,
        warnings=warnings + [warning for curve in curves.values() for warning in curve.warnings],
        qc=qc,
    )
