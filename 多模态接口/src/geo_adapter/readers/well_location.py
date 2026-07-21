from __future__ import annotations

from typing import Any

from geo_adapter.coordinates.crs import crs_info, transform_xy
from geo_adapter.errors import InputDataError
from geo_adapter.readers.structured import first_record_as_dict, read_structured_table
from geo_adapter.schemas.config import AdapterConfig
from geo_adapter.semantics.field_mapper import map_fields


DEFAULT_ALIASES = {
    "well_id": ["WELL", "WELL_NAME", "WELL_ID", "井名"],
    "x": ["X", "EASTING", "WELL_X", "X坐标"],
    "y": ["Y", "NORTHING", "WELL_Y", "Y坐标"],
    "longitude": ["LON", "LONGITUDE", "经度"],
    "latitude": ["LAT", "LATITUDE", "纬度"],
    "kb": ["KB", "KB_ELEV", "补心海拔"],
    "ground_elevation": ["GL", "GROUND_ELEVATION", "地面海拔"],
    "total_depth": ["TD", "TOTAL_DEPTH", "完钻井深"],
    "crs": ["CRS", "EPSG", "坐标系"],
}


def read_well_location(config: AdapterConfig) -> dict[str, Any]:
    spec = config.inputs.well_location
    if spec.path is None or not spec.path.is_file():
        raise InputDataError(f"井位文件不存在: {spec.path}")
    aliases = {**DEFAULT_ALIASES, **config.field_mapping.get("well_location", {})}
    frame, mapping = map_fields(read_structured_table(spec.path), aliases)
    record = first_record_as_dict(frame)
    explicit_crs = record.get("crs") or config.coordinate_system.well_crs or spec.crs
    source_info = crs_info(explicit_crs, "well_location")
    x, y = record.get("x"), record.get("y")
    lon, lat = record.get("longitude"), record.get("latitude")
    if (x is None or y is None) and lon is not None and lat is not None:
        x, y = float(lon), float(lat)
        if explicit_crs is None:
            source_info = crs_info("EPSG:4326", "explicit_longitude_latitude_fields")
    transformed = False
    target_info = source_info.copy()
    project = config.coordinate_system.project_crs
    if x is not None and y is not None and project is not None and source_info.get("confidence") == "explicit":
        source_ref = source_info.get("epsg") or source_info.get("name")
        if str(source_ref).upper() != str(project).upper().replace("EPSG:", ""):
            tx, ty = transform_xy([float(x)], [float(y)], source_ref, project)
            x, y = float(tx[0]), float(ty[0])
            transformed = True
            target_info = crs_info(project, "project_crs")
    available = x is not None and y is not None
    warnings: list[str] = []
    if not available:
        warnings.append("井位缺少完整 X/Y 或经纬度")
    if source_info["confidence"] != "explicit":
        warnings.append("井位 CRS 未明确，不能声明可靠空间配准")
    return {
        "available": available,
        "well_id": record.get("well_id") or spec.well_id,
        "x": x,
        "y": y,
        "longitude": lon,
        "latitude": lat,
        "kb_elevation": record.get("kb"),
        "ground_elevation": record.get("ground_elevation"),
        "total_depth": record.get("total_depth"),
        "total_depth_type": "unknown" if record.get("total_depth") is not None else None,
        "depth_reference": config.depth_reference.reference_surface,
        "source_crs": source_info,
        "project_crs": target_info if project is not None else None,
        "coordinates_transformed": transformed,
        "field_mapping": mapping,
        "source_path": str(spec.path),
        "warnings": warnings,
    }

