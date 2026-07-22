from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geo_adapter.coordinates.crs import crs_info
from geo_adapter.errors import InputDataError, OptionalDependencyError
from geo_adapter.models import SeismicData, SeismicView
from geo_adapter.preprocess.normalization import normalize_seismic
from geo_adapter.schemas.config import AdapterConfig


def _clamped(value: int | None, length: int) -> int:
    return length // 2 if value is None else max(0, min(int(value), length - 1))


def _view(
    name: str,
    raw: np.ndarray,
    physical_view: str,
    axes: tuple[str, str],
    indices: dict[str, int | list[int]],
    config: AdapterConfig,
) -> SeismicView:
    clip = config.processing.seismic.percentile_clip
    processed, normalization = normalize_seismic(
        raw, clip.lower, clip.upper, config.processing.seismic.normalization
    )
    return SeismicView(name, physical_view, np.asarray(raw), processed, axes, indices, normalization)


def _extract_numpy_views(array: np.ndarray, config: AdapterConfig) -> dict[str, SeismicView]:
    if array.ndim not in {2, 3}:
        raise InputDataError(f"地震数组必须为 2D 或 3D，实际为 {array.ndim}D")
    requested = config.processing.seismic.views
    views: dict[str, SeismicView] = {}
    if array.ndim == 2:
        # 2-D input is a physical patch/profile; it is not silently relabeled as inline.
        views["patch"] = _view("patch", array, "user_provided_2d_patch", ("trace", "sample"), {}, config)
        return views
    ni, nx, ns = array.shape
    ii = _clamped(config.processing.seismic.inline_index, ni)
    xi = _clamped(config.processing.seismic.crossline_index, nx)
    si = _clamped(config.processing.seismic.sample_index, ns)
    if "inline" in requested:
        views["inline"] = _view("inline", array[ii, :, :], "inline", ("crossline_index", "sample_index"), {"inline_index": ii}, config)
    if "crossline" in requested:
        views["crossline"] = _view("crossline", array[:, xi, :], "crossline", ("inline_index", "sample_index"), {"crossline_index": xi}, config)
    if "slice" in requested:
        views["slice"] = _view("slice", array[:, :, si], "time_or_depth_slice", ("inline_index", "crossline_index"), {"sample_index": si}, config)
    if "local_patch" in requested:
        radius = config.processing.seismic.local_patch_radius
        i0, i1 = max(0, ii - radius), min(ni, ii + radius + 1)
        x0, x1 = max(0, xi - radius), min(nx, xi + radius + 1)
        # A local horizontal patch at the selected sample is physically distinct from an inline.
        views["local_patch"] = _view(
            "local_patch",
            array[i0:i1, x0:x1, si],
            "local_horizontal_patch",
            ("inline_index", "crossline_index"),
            {"inline_range": [i0, i1 - 1], "crossline_range": [x0, x1 - 1], "sample_index": si},
            config,
        )
    return views


def _read_numpy(path: Path, array_key: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        if path.suffix.lower() == ".npy":
            return np.load(path, mmap_mode="r", allow_pickle=False), {}
        archive = np.load(path, allow_pickle=False)
        keys = list(archive.files)
        selected = array_key or ("amplitude" if "amplitude" in keys else next((k for k in keys if np.asarray(archive[k]).ndim >= 2), None))
        if selected is None or selected not in keys:
            raise InputDataError(f"NPZ 中未找到地震数组；可用键: {keys}")
        metadata: dict[str, Any] = {"array_key": selected, "available_keys": keys}
        if "domain" in keys:
            metadata["domain"] = str(np.asarray(archive["domain"]).item())
        return np.asarray(archive[selected]), metadata
    except (OSError, ValueError) as exc:
        raise InputDataError(f"NumPy 地震读取失败: {path}: {exc}") from exc


def _read_segy_views(path: Path, config: AdapterConfig) -> tuple[tuple[int, ...], dict[str, SeismicView], dict[str, Any]]:
    try:
        import segyio
    except ImportError as exc:
        raise OptionalDependencyError("读取 SEG-Y 需要可选依赖: pip install -e .[segy]") from exc
    try:
        with segyio.open(str(path), "r", strict=False, ignore_geometry=False) as handle:
            handle.mmap()
            samples = len(handle.samples)
            ilines = list(handle.ilines)
            xlines = list(handle.xlines)
            if not ilines or not xlines:
                raise InputDataError("SEG-Y 缺少可建立 inline/crossline 几何的头字段")
            ii = _clamped(config.processing.seismic.inline_index, len(ilines))
            xi = _clamped(config.processing.seismic.crossline_index, len(xlines))
            views: dict[str, SeismicView] = {}
            if "inline" in config.processing.seismic.views:
                views["inline"] = _view(
                    "inline", np.asarray(handle.iline[ilines[ii]]), "inline",
                    ("crossline", "sample"),
                    {"inline_index": ii, "inline_number": int(ilines[ii])}, config,
                )
            if "crossline" in config.processing.seismic.views:
                views["crossline"] = _view(
                    "crossline", np.asarray(handle.xline[xlines[xi]]), "crossline",
                    ("inline", "sample"),
                    {"crossline_index": xi, "crossline_number": int(xlines[xi])}, config,
                )
            # No unconditional cube read. Horizontal slice/local-patch extraction is deferred
            # until a geometry-aware implementation can bound the trace access safely.
            sample_values = np.asarray(handle.samples, dtype=np.float64)
            sample_interval_ms = (
                float(np.median(np.diff(sample_values)))
                if sample_values.size > 1 else None
            )
            metadata = {
                "inline_count": len(ilines),
                "crossline_count": len(xlines),
                "sample_count": samples,
                "sample_interval_ms": sample_interval_ms,
                "sample_start_ms": float(sample_values[0]) if sample_values.size else None,
                "lazy_slice_policy": True,
            }
            return (len(ilines), len(xlines), samples), views, metadata
    except OptionalDependencyError:
        raise
    except Exception as exc:
        if isinstance(exc, InputDataError):
            raise
        raise InputDataError(f"SEG-Y 扫描/切片失败: {path}: {exc}") from exc


def read_seismic(config: AdapterConfig) -> SeismicData:
    """Read a seismic source and extract independent physical views."""
    spec = config.inputs.seismic
    if spec.path is None or not spec.path.is_file():
        raise InputDataError(f"地震文件不存在: {spec.path}")
    suffix = spec.path.suffix.lower()
    warnings: list[str] = []
    if suffix in {".npy", ".npz"}:
        array, metadata = _read_numpy(spec.path, spec.array_key)
        views = _extract_numpy_views(array, config)
        shape = tuple(int(x) for x in array.shape)
        source_format = suffix.lstrip(".")
        finite = np.asarray(array)[np.isfinite(array)]
        nan_inf_ratio = float(1.0 - finite.size / array.size) if array.size else 1.0
        amplitude = {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
        }
    elif suffix in {".sgy", ".segy"}:
        shape, views, metadata = _read_segy_views(spec.path, config)
        source_format = "segy"
        nan_inf_ratio = None
        amplitude = {}
    else:
        raise InputDataError(f"不支持的地震格式: {suffix}")
    domain = spec.domain
    if domain == "auto":
        domain = str(metadata.get("domain", "unknown")).lower()
    if domain not in {"time", "depth"}:
        domain = "unknown"
        warnings.append("地震时间域/深度域未明确，图轴不会伪标为 TWT 或深度")
    crs_value = spec.crs or config.coordinate_system.seismic_crs
    crs = crs_info(crs_value, "seismic_config")
    if crs["confidence"] != "explicit":
        warnings.append("地震 CRS 未明确，不能声明 H3 空间配准")
    qc = {
        "readable": True,
        "shape": list(shape),
        "dimension": len(shape),
        "nan_inf_ratio": nan_inf_ratio,
        "amplitude": amplitude,
        "near_constant": bool(amplitude.get("std") is not None and amplitude["std"] < 1e-12),
        "inline_crossline_available": len(shape) == 3 and bool(views),
        "coordinate_headers_complete": True if source_format != "segy" else bool(metadata.get("inline_count") and metadata.get("crossline_count")),
        "crs_explicit": crs["confidence"] == "explicit",
        "domain": domain,
        "metadata": metadata,
    }
    return SeismicData(spec.path, shape, domain, views, crs, source_format, warnings, qc)
