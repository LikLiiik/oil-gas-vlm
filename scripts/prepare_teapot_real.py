"""Prepare the uploaded Teapot/RMOTC real sample for geo_adapter.

The public ``filt_mig.sgy`` stores crossline/inline in non-standard trace
header bytes 13/17.  This script creates a byte-for-byte work copy and only
fills the standard SEG-Y rev1 bytes 189/193.  Samples, textual/binary headers,
coordinates, and all other trace headers are retained.

No unrelated checkshot table is attached to 56-TpX-10.  The generated config
therefore uses the well's DT curve for an explicitly uncalibrated approximate
time-depth relation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


FT_TO_M = 0.3048
TARGET_API = "49025106100000"
TARGET_WELL = "56-TpX-10"
SOURCE_CRS = "EPSG:32056"  # NAD27 / Wyoming East Central (US survey foot)
PROJECT_CRS = "EPSG:32613"  # WGS84 / UTM zone 13N (metre)


def _digits(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def api_key(value: Any) -> str:
    """Return the 12-digit key shared by LAS 14-digit and Excel IDs."""
    digits = _digits(value)
    return digits[:12] if len(digits) >= 12 else digits


def _norm_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _find_file(root: Path, candidates: list[str]) -> Path:
    for relative in candidates:
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"None of the expected files exists under {root}: {candidates}"
    )


def read_well_header(path: Path, target_api: str) -> dict[str, Any]:
    target = api_key(target_api)
    for sheet in pd.ExcelFile(path).sheet_names:
        # Row 1 is the actual header; row 0 documents the coordinate system.
        frame = pd.read_excel(path, sheet_name=sheet, header=1)
        if frame.empty:
            continue
        api_column = next(
            (column for column in frame.columns if "API" in str(column).upper()),
            None,
        )
        if api_column is None:
            continue
        match = frame[api_column].map(api_key) == target
        if match.any():
            row = frame.loc[match].iloc[0]
            return {str(column).strip(): row[column] for column in frame.columns}
    raise ValueError(f"API {target_api} was not found in {path}")


def _value(record: dict[str, Any], *names: str) -> Any:
    lookup = {_norm_name(key): value for key, value in record.items()}
    for name in names:
        key = _norm_name(name)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"None of {names} was present; columns={list(record)}")


def extract_directional_survey(
    path: Path, target_api: str, target_well: str
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    target_key = api_key(target_api)
    name_key = _norm_name(target_well)
    starts: list[int] = []
    for index, row in raw.iterrows():
        first = str(row.iloc[0]).strip().upper() if len(row) else ""
        second = _norm_name(row.iloc[1]) if len(row) > 1 else ""
        if first == "WELL:" and second == name_key:
            starts.append(int(index))
    if not starts:
        raise ValueError(f"Well {target_well} was not found in {path}")

    blocks: list[pd.DataFrame] = []
    for start in starts:
        rows: list[list[Any]] = []
        for index in range(start + 2, len(raw)):
            row = raw.iloc[index]
            if row.isna().all() or str(row.iloc[0]).strip().upper() == "WELL:":
                break
            if api_key(row.iloc[0]) == target_key:
                rows.append([row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]])
        if rows:
            blocks.append(pd.DataFrame(rows, columns=["API", "MD_FT", "INC", "AZI"]))
    if not blocks:
        raise ValueError(f"No survey stations matched API {target_api} in {path}")

    survey = pd.concat(blocks, ignore_index=True)
    for column in ("MD_FT", "INC", "AZI"):
        survey[column] = pd.to_numeric(survey[column], errors="coerce")
    survey = survey.dropna(subset=["MD_FT", "INC", "AZI"])
    survey = survey.sort_values("MD_FT").drop_duplicates("MD_FT", keep="last")
    # The public survey starts near 3000 ft.  Anchor minimum-curvature
    # integration at the surface instead of extrapolating the first angle.
    if survey.iloc[0]["MD_FT"] > 0:
        surface = pd.DataFrame(
            [{"API": target_key, "MD_FT": 0.0, "INC": 0.0, "AZI": 0.0}]
        )
        survey = pd.concat([surface, survey], ignore_index=True)
    return pd.DataFrame(
        {
            "WELL_ID": TARGET_API,
            "MD": survey["MD_FT"].to_numpy(float) * FT_TO_M,
            "INCLINATION": survey["INC"].to_numpy(float),
            "AZIMUTH": survey["AZI"].to_numpy(float),
        }
    )


def extract_formation_tops(path: Path, target_api: str) -> pd.DataFrame:
    target = api_key(target_api)
    matches: list[pd.DataFrame] = []
    for sheet in pd.ExcelFile(path).sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        if raw.shape[1] < 4:
            continue
        keep = raw.iloc[:, 0].map(api_key) == target
        if keep.any():
            part = raw.loc[keep, raw.columns[:4]].copy()
            part.columns = ["API", "WELL_ID", "FORMATION", "MD_FT"]
            matches.append(part)
    if not matches:
        return pd.DataFrame(columns=["API", "WELL_ID", "FORMATION", "MD_M"])
    tops = pd.concat(matches, ignore_index=True)
    tops["MD_FT"] = pd.to_numeric(tops["MD_FT"], errors="coerce")
    tops = tops.dropna(subset=["MD_FT"])
    tops["API"] = TARGET_API
    tops["MD_M"] = tops.pop("MD_FT") * FT_TO_M
    return tops


def normalize_las_to_csv(las_path: Path, output_path: Path) -> dict[str, Any]:
    try:
        import lasio
    except ImportError as exc:
        raise RuntimeError("Install lasio in the project .venv before running") from exc
    las = lasio.read(str(las_path), ignore_header_errors=True)
    frame = las.df().reset_index()
    depth_column = str(frame.columns[0])
    depth_unit = str(las.curves[0].unit).strip().upper()
    depth = pd.to_numeric(frame[depth_column], errors="coerce")
    if depth_unit in {"F", "FT", "FEET"}:
        depth = depth * FT_TO_M
    elif depth_unit not in {"M", "METRE", "METER"}:
        raise ValueError(f"Unsupported LAS depth unit: {depth_unit!r}")
    frame[depth_column] = depth
    frame = frame.rename(columns={depth_column: "DEPT"})
    frame.to_csv(output_path, index=False)
    return {
        "rows": int(len(frame)),
        "depth_range_m": [float(depth.min()), float(depth.max())],
        "curves": [str(column) for column in frame.columns],
    }


def standardize_rmotc_geometry(source: Path, destination: Path) -> dict[str, Any]:
    """Copy SEG-Y and mirror custom IL/XL fields into standard rev1 fields."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        shutil.copyfile(source, destination)

    with destination.open("r+b") as handle:
        handle.seek(3200)
        binary = handle.read(400)
        sample_count = struct.unpack(">H", binary[20:22])[0]
        sample_format = struct.unpack(">H", binary[24:26])[0]
        if sample_format not in {1, 2, 3, 5, 8}:
            raise ValueError(f"Unsupported SEG-Y sample format code {sample_format}")
        sample_bytes = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}[sample_format]
        trace_size = 240 + sample_count * sample_bytes
        payload = destination.stat().st_size - 3600
        if payload <= 0 or payload % trace_size:
            raise ValueError("SEG-Y size is inconsistent with its binary header")
        trace_count = payload // trace_size
        inlines: set[int] = set()
        xlines: set[int] = set()
        xy_ilxl: list[tuple[float, float, int, int]] = []
        for trace_index in range(trace_count):
            offset = 3600 + trace_index * trace_size
            handle.seek(offset)
            header = bytearray(handle.read(240))
            xline = struct.unpack(">i", header[12:16])[0]
            inline = struct.unpack(">i", header[16:20])[0]
            scalar = struct.unpack(">h", header[70:72])[0]
            raw_x = struct.unpack(">i", header[72:76])[0]
            raw_y = struct.unpack(">i", header[76:80])[0]
            scale = 1.0 / abs(scalar) if scalar < 0 else float(scalar or 1)
            struct.pack_into(">i", header, 188, inline)
            struct.pack_into(">i", header, 192, xline)
            handle.seek(offset)
            handle.write(header)
            inlines.add(inline)
            xlines.add(xline)
            xy_ilxl.append((raw_x * scale, raw_y * scale, inline, xline))

    if len(inlines) * len(xlines) != trace_count:
        raise ValueError(
            "RMOTC geometry is not a complete rectangular grid: "
            f"{len(inlines)}*{len(xlines)} != {trace_count}"
        )
    return {
        "trace_count": int(trace_count),
        "sample_count": int(sample_count),
        "sample_format": int(sample_format),
        "inline_values": sorted(inlines),
        "crossline_values": sorted(xlines),
        "xy_ilxl": xy_ilxl,
    }


def nearest_well_indices(
    geometry: dict[str, Any], easting_ft: float, northing_ft: float
) -> dict[str, Any]:
    coordinates = np.asarray(geometry["xy_ilxl"], dtype=np.float64)
    design = np.column_stack(
        [coordinates[:, 0], coordinates[:, 1], np.ones(len(coordinates))]
    )
    il_coeff = np.linalg.lstsq(design, coordinates[:, 2], rcond=None)[0]
    xl_coeff = np.linalg.lstsq(design, coordinates[:, 3], rcond=None)[0]
    point = np.asarray([easting_ft, northing_ft, 1.0])
    predicted_il = float(point @ il_coeff)
    predicted_xl = float(point @ xl_coeff)
    ilines = np.asarray(geometry["inline_values"], dtype=int)
    xlines = np.asarray(geometry["crossline_values"], dtype=int)
    il_index = int(np.argmin(np.abs(ilines - predicted_il)))
    xl_index = int(np.argmin(np.abs(xlines - predicted_xl)))
    residual_il = design @ il_coeff - coordinates[:, 2]
    residual_xl = design @ xl_coeff - coordinates[:, 3]
    return {
        "predicted_inline": predicted_il,
        "predicted_crossline": predicted_xl,
        "inline_index": il_index,
        "crossline_index": xl_index,
        "inline_number": int(ilines[il_index]),
        "crossline_number": int(xlines[xl_index]),
        "geometry_rmse_bins": float(
            np.sqrt(np.mean(residual_il**2 + residual_xl**2))
        ),
    }


def build_config(
    project_root: Path,
    prepared: Path,
    run_dir: Path,
    indices: dict[str, Any],
) -> dict[str, Any]:
    rel = lambda path: path.resolve().relative_to(project_root.resolve()).as_posix()
    return {
        "schema_version": "1.0",
        "sample_id": "real_teapot_49025106100000",
        "task": {
            "type": "geological_target_detection",
            "target_classes": ["fault", "horizon", "seismic_facies", "reservoir_candidate"],
        },
        "inputs": {
            "seismic": {
                "path": rel(prepared / "filt_mig_standard_headers.sgy"),
                "format": "segy",
                "domain": "time",
                "crs": SOURCE_CRS,
                "optional": False,
            },
            "well_log": {
                "path": rel(prepared / "well_log_m.csv"),
                "format": "csv",
                "well_id": TARGET_API,
                "optional": False,
            },
            "well_location": {
                "path": rel(prepared / "well_location.csv"),
                "crs": SOURCE_CRS,
                "optional": False,
            },
            "trajectory": {
                "path": rel(prepared / "trajectory_m.csv"),
                "optional": False,
            },
            "time_depth": {"path": None, "optional": True},
        },
        "field_mapping": {
            "well_location": {
                "well_id": ["WELL_ID"], "x": ["EASTING_FT"],
                "y": ["NORTHING_FT"], "kb": ["KB_M"],
                "ground_elevation": ["GROUND_ELEVATION_M"],
                "total_depth": ["TOTAL_DEPTH_M"],
            },
            "trajectory": {
                "well_id": ["WELL_ID"], "md": ["MD"],
                "inclination": ["INCLINATION"], "azimuth": ["AZIMUTH"],
            },
            "time_depth": {
                "depth": ["DEPTH_M", "TVD_M", "MD_M"],
                "twt_ms": ["TWT_MS"],
            },
        },
        "coordinate_system": {
            "project_crs": PROJECT_CRS,
            "seismic_crs": SOURCE_CRS,
            "well_crs": SOURCE_CRS,
            "allow_unknown_crs": False,
            "require_explicit_crs_for_precise_alignment": True,
        },
        "depth_reference": {
            "well_log_axis": "MD", "unit": "m", "reference_surface": "KB",
            "positive_direction": "down", "vertical_datum": "MSL",
            "tvdss_sign_convention": "positive_below_sea_level",
        },
        "processing": {
            "seismic": {
                "views": ["inline", "crossline"],
                "percentile_clip": {"lower": 1.0, "upper": 99.0},
                "normalization": "symmetric",
                "inline_index": indices["inline_index"],
                "crossline_index": indices["crossline_index"],
                "sample_index": 750,
                "local_patch_radius": 16,
            },
            "well_logs": {
                "resample_step": None,
                "short_gap_interpolation": {
                    "enabled": True, "max_gap_samples": 3, "method": "linear"
                },
                "preferred_curves": {
                    "GR": "GRD", "SP": "SPR", "CAL": "CALD",
                    "RES_DEEP": "ILD", "RES_MEDIUM_SHALLOW": "ILM",
                    "RES_MICRO": "SFL", "AC": "DT", "DEN": "RHOB", "CNL": "NPHI",
                },
                "curve_units": {
                    "GRD": "GAPI", "SPR": "mV", "CALD": "in",
                    "ILD": "ohm_m", "ILM": "ohm_m", "SFL": "ohm_m",
                    "DT": "us/ft", "RHOB": "g/cm3", "NPHI": "DEC",
                },
                "curve_descriptions": {
                    "GRD": "gamma ray", "SPR": "spontaneous potential",
                    "CALD": "caliper", "ILD": "deep induction resistivity",
                    "ILM": "medium induction resistivity",
                    "SFL": "shallow focused resistivity", "DT": "sonic slowness",
                    "RHOB": "bulk density", "NPHI": "neutron porosity",
                },
                "resistivity_overrides": {
                    "SFL": {"investigation_depth": "micro", "measurement_family": "focused"}
                },
            },
            "time_depth": {
                "preferred_sources": ["sonic_integrated"],
                "sonic_integration": {
                    "enabled": True, "preferred_depth_axis": "TVD",
                    "require_trajectory_for_deviated_well": True,
                },
                "t0": {"policy": "unknown", "value_ms": None, "source": None},
                "replacement_velocity": {
                    "policy": "unknown", "value_m_s": None, "source": None
                },
                "calibration": {
                    "required_for_joint_analysis": True,
                    "method": None, "control_points_path": None,
                },
            },
        },
        "curve_aliases_path": "多模态接口/configs/curve_aliases.yaml",
        "field_aliases_path": "多模态接口/configs/field_aliases.yaml",
        "prompt_templates_path": "多模态接口/configs/prompt_templates.yaml",
        "output": {"directory": rel(run_dir), "overwrite": True},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/real/teapot"))
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/real_teapot_49025106100000")
    )
    parser.add_argument(
        "--config-out", type=Path, default=Path("configs/real_teapot_49025106100000.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    data_root = args.data_root.resolve()
    prepared = (args.prepared_dir or (data_root / "prepared")).resolve()
    run_dir = args.run_dir.resolve()
    config_out = args.config_out.resolve()
    prepared.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)

    segy_path = _find_file(data_root, ["seismic/filt_mig.sgy", "seismic/filt_mig.segy"])
    las_path = _find_file(
        data_root,
        [
            "wells/49025106100000.LAS",
            "wells/49025106100000.las",
            "well_logs/deeper/49025106100000_13345_00010H306588.LAS",
        ],
    )
    headers_path = _find_file(
        data_root,
        [
            "metadata/well_headers.xlsx",
            "well_information/01_well_locations_and_headers/TeapotDomeWellHeaders02-09-10.xlsx",
        ],
    )
    surveys_path = _find_file(
        data_root,
        [
            "metadata/directional_surveys.xlsx",
            "well_information/02_directional_surveys/DirectionalSurveys_020910.xlsx",
        ],
    )
    tops_path = _find_file(
        data_root,
        [
            "metadata/formation_tops.xls",
            "well_information/03_formation_and_geology/TeapotDomeFormationLogTops.xls",
        ],
    )

    header = read_well_header(headers_path, TARGET_API)
    easting_ft = float(_value(header, "Easting"))
    northing_ft = float(_value(header, "Northing"))
    kb_m = float(_value(header, "Datum Elevation")) * FT_TO_M
    ground_m = float(_value(header, "Ground Elevation")) * FT_TO_M
    total_depth_m = float(_value(header, "Total Depth")) * FT_TO_M
    pd.DataFrame(
        [{
            "WELL_ID": TARGET_API, "WELL_NAME": TARGET_WELL,
            "EASTING_FT": easting_ft, "NORTHING_FT": northing_ft,
            "KB_M": kb_m, "GROUND_ELEVATION_M": ground_m,
            "TOTAL_DEPTH_M": total_depth_m, "SOURCE_CRS": SOURCE_CRS,
        }]
    ).to_csv(prepared / "well_location.csv", index=False)

    trajectory = extract_directional_survey(surveys_path, TARGET_API, TARGET_WELL)
    trajectory.to_csv(prepared / "trajectory_m.csv", index=False)
    tops = extract_formation_tops(tops_path, TARGET_API)
    tops.to_csv(prepared / "formation_tops_m.csv", index=False)
    las_qc = normalize_las_to_csv(las_path, prepared / "well_log_m.csv")

    geometry = standardize_rmotc_geometry(
        segy_path, prepared / "filt_mig_standard_headers.sgy"
    )
    indices = nearest_well_indices(geometry, easting_ft, northing_ft)
    config = build_config(project_root, prepared, run_dir, indices)
    config_out.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    qc = {
        "sample_id": config["sample_id"],
        "well": {
            "api": TARGET_API, "name": TARGET_WELL,
            "easting_ft": easting_ft, "northing_ft": northing_ft,
            "kb_m": kb_m, "ground_elevation_m": ground_m,
            "total_depth_m": total_depth_m,
        },
        "trajectory_rows": int(len(trajectory)),
        "formation_top_rows": int(len(tops)),
        "las": las_qc,
        "segy": {
            "trace_count": geometry["trace_count"],
            "sample_count": geometry["sample_count"],
            "sample_format": geometry["sample_format"],
            "inline_count": len(geometry["inline_values"]),
            "inline_range": [
                geometry["inline_values"][0], geometry["inline_values"][-1]
            ],
            "crossline_count": len(geometry["crossline_values"]),
            "crossline_range": [
                geometry["crossline_values"][0], geometry["crossline_values"][-1]
            ],
        },
        "well_to_seismic": indices,
        "time_depth": {
            "provided_table": False,
            "reason": "No uploaded checkshot/VSP sheet belongs to 56-TpX-10",
            "fallback": "DT sonic integration; uncalibrated and not valid for precise tie",
        },
        "config": str(config_out),
        "run_dir": str(run_dir),
    }
    (prepared / "preparation_qc.json").write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
