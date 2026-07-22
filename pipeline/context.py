"""Build the inference-only data contract consumed by downstream models."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _run_path(run_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def canonicalize_seismic_view(
    array: np.ndarray,
    axis_labels: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, str]:
    """Return a 2D view in the same orientation as the model image.

    Inline/crossline arrays are stored as (trace, sample) by geo_adapter but
    rendered transposed. Downstream algorithms use (sample, trace). Horizontal
    slices are already stored and rendered as (inline, crossline).
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        return arr, "native"
    labels = [str(label).lower() for label in (axis_labels or [])]
    if len(labels) == 2 and "sample" in labels[1] and "sample" not in labels[0]:
        return np.ascontiguousarray(arr.T), "sample_trace"
    return np.ascontiguousarray(arr), "map_yx"


def _read_first_csv_column(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return None
        values: list[float] = []
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[0]))
            except (TypeError, ValueError):
                values.append(float("nan"))
    return np.asarray(values, dtype=np.float32)


def _fill_internal_gaps(values: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Fill only for numerical inference; the original masks remain in context."""
    result = np.asarray(values, dtype=np.float32).copy()
    valid_idx = np.flatnonzero(usable & np.isfinite(result))
    if valid_idx.size == 0:
        return result
    missing_idx = np.flatnonzero(~usable | ~np.isfinite(result))
    if missing_idx.size:
        result[missing_idx] = np.interp(missing_idx, valid_idx, result[valid_idx])
    return result


def _load_well_context(pkg) -> dict[str, Any]:
    run_dir = Path(pkg.run_dir)
    values_path = run_dir / "arrays" / "well_values.npy"
    valid_path = run_dir / "arrays" / "well_valid_mask.npy"
    interp_path = run_dir / "arrays" / "well_interpolated_mask.npy"
    available_path = run_dir / "arrays" / "curve_available.npy"
    required = (values_path, valid_path, interp_path, available_path)
    if not all(path.is_file() for path in required):
        return {}

    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    interpolated = np.load(interp_path, mmap_mode="r", allow_pickle=False)
    available = np.load(available_path, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2 or valid.shape != values.shape or interpolated.shape != values.shape:
        return {}

    well_meta = pkg.manifest.get("well_logs") or {}
    order = list(well_meta.get("curve_order") or [])
    if len(order) != values.shape[1]:
        order = [
            "SP", "GR", "CAL", "RES_DEEP", "RES_MEDIUM_SHALLOW",
            "RES_MICRO", "AC", "DEN", "CNL",
        ][:values.shape[1]]

    depth = _read_first_csv_column(run_dir / "tables" / "well_logs_clean.csv")
    if depth is None or len(depth) != values.shape[0]:
        depth_range = well_meta.get("depth_range") or [0.0, float(values.shape[0] - 1)]
        depth = np.linspace(float(depth_range[0]), float(depth_range[-1]), values.shape[0], dtype=np.float32)

    curves: dict[str, np.ndarray] = {"depth": depth}
    curve_valid_mask: dict[str, np.ndarray] = {}
    curve_interpolated_mask: dict[str, np.ndarray] = {}
    curve_availability: dict[str, bool] = {}
    for column, name in enumerate(order):
        is_available = bool(available[column]) if column < len(available) else False
        curve_availability[name] = is_available
        if not is_available:
            continue
        usable = np.asarray(valid[:, column] | interpolated[:, column], dtype=bool)
        curves[name] = _fill_internal_gaps(values[:, column], usable)
        curve_valid_mask[name] = np.asarray(valid[:, column], dtype=bool)
        curve_interpolated_mask[name] = np.asarray(interpolated[:, column], dtype=bool)

    # Existing inference rules use RT as the canonical deep-resistivity alias.
    if "RES_DEEP" in curves:
        curves["RT"] = curves["RES_DEEP"]
        curve_valid_mask["RT"] = curve_valid_mask["RES_DEEP"]
        curve_interpolated_mask["RT"] = curve_interpolated_mask["RES_DEEP"]
        curve_availability["RT"] = True

    return {
        "curves": curves,
        "curve_valid_mask": curve_valid_mask,
        "curve_interpolated_mask": curve_interpolated_mask,
        "curve_availability": curve_availability,
    }


def _load_time_depth_context(pkg) -> dict[str, Any]:
    relation = pkg.manifest.get("time_depth_relation") or {}
    path = _run_path(Path(pkg.run_dir), relation.get("table_path"))
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    names = {name.lower(): name for name in rows[0]}
    depth_key = names.get("depth") or names.get("tvd") or names.get("md")
    twt_key = names.get("twt_ms") or names.get("twt") or names.get("time_ms")
    if depth_key is None or twt_key is None:
        return {}
    pairs = []
    for row in rows:
        try:
            pairs.append((float(row[depth_key]), float(row[twt_key])))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return {}
    return {
        "time_depth_pairs": np.asarray(pairs, dtype=np.float32),
        "time_depth_confidence": relation.get("confidence", "none"),
        "fusion_permission": (pkg.manifest.get("alignment") or {}).get("fusion_permission"),
    }


def _load_formation_tops_context(pkg) -> dict[str, Any]:
    """Load optional formation tops without changing the geo_adapter schema.

    Real-data preparation keeps the table beside the normalized well-log CSV.
    A packaged copy under ``tables`` is also accepted.  Tops are advisory
    geological context; they never overwrite curve-derived measurements.
    """
    run_dir = Path(pkg.run_dir)
    candidates = [
        run_dir / "tables" / "formation_tops_m.csv",
        run_dir / "formation_tops_m.csv",
    ]
    well_meta = pkg.manifest.get("well_logs") or {}
    source_path = _run_path(run_dir, well_meta.get("source_path"))
    if source_path is not None:
        candidates.append(source_path.parent / "formation_tops_m.csv")

    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tops: list[dict[str, Any]] = []
    for row in rows:
        normalized = {str(key).strip().upper(): value for key, value in row.items()}
        depth_value = normalized.get("MD_M") or normalized.get("DEPTH_M")
        name = normalized.get("FORMATION") or normalized.get("TOP_NAME")
        try:
            depth_m = float(depth_value)
        except (TypeError, ValueError):
            continue
        if not name or not np.isfinite(depth_m):
            continue
        tops.append({"formation": str(name).strip(), "depth_m": depth_m})
    if not tops:
        return {}
    tops.sort(key=lambda item: item["depth_m"])
    return {
        "formation_tops": tops,
        "formation_tops_source": str(path),
    }


def build_downstream_context(image, pkg) -> dict[str, Any] | None:
    """Load arrays, curves, masks and alignment metadata for one inference step."""
    context: dict[str, Any] = {}
    view_meta = pkg.view_meta(image.physical_view) or {}
    if view_meta:
        context["view_meta"] = view_meta
        raw_path = _run_path(
            Path(pkg.run_dir),
            view_meta.get("raw_array_path")
            or view_meta.get("array_path")
            or view_meta.get("source_array"),
        )
        if raw_path is not None and raw_path.is_file():
            raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
            array, layout = canonicalize_seismic_view(raw, view_meta.get("axis_labels"))
            context["array"] = array
            context["coordinate_shape"] = list(array.shape)
            context["array_layout"] = layout

        seismic_meta = pkg.manifest.get("seismic") or {}
        qc_meta = (seismic_meta.get("qc") or {}).get("metadata") or {}
        interval_ms = qc_meta.get("sample_interval_ms")
        if interval_ms is not None and context.get("array_layout") == "sample_trace":
            interval_ms = float(interval_ms)
            context["sample_interval_ms"] = interval_ms
            context["time_axis_ms"] = np.arange(context["array"].shape[0], dtype=np.float32) * interval_ms

    context.update(_load_well_context(pkg))
    context.update(_load_time_depth_context(pkg))
    context.update(_load_formation_tops_context(pkg))
    return context or None
