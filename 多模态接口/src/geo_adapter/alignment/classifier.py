from __future__ import annotations

from typing import Any

from geo_adapter.models import TimeDepthData, TrajectoryData


def classify_alignment(
    *,
    seismic_available: bool,
    well_logs_available: bool,
    well_location: dict[str, Any] | None,
    trajectory: TrajectoryData | None,
    time_depth: TimeDepthData | None,
    seismic_crs: dict[str, Any] | None,
    depth_reference_explicit: bool,
) -> dict[str, Any]:
    """Classify independent horizontal/vertical levels and fusion permission."""
    limitations: list[str] = []
    location_available = bool(well_location and well_location.get("available"))
    trajectory_available = bool(trajectory and trajectory.available)
    subsurface_xy = bool(trajectory and trajectory.subsurface_xy_available)
    well_crs = (well_location or {}).get("project_crs") or (well_location or {}).get("source_crs") or {}
    crs_match = bool(
        seismic_crs
        and seismic_crs.get("confidence") == "explicit"
        and well_crs.get("confidence") == "explicit"
        and (
            seismic_crs.get("epsg") == well_crs.get("epsg")
            or (seismic_crs.get("name") and seismic_crs.get("name") == well_crs.get("name"))
        )
    )
    if not location_available:
        horizontal = "none"
        limitations.append("well_location_unavailable")
    elif trajectory_available and subsurface_xy and crs_match:
        horizontal = "seismic_crs_aligned"
    elif trajectory_available and subsurface_xy:
        horizontal = "trajectory_available"
        limitations.append("well_and_seismic_crs_not_confirmed_equal")
    else:
        horizontal = "wellhead_only"
        if not trajectory_available:
            limitations.append("subsurface_trajectory_unavailable")

    if time_depth and time_depth.available:
        if time_depth.measured:
            vertical = "measured_time_depth"
        elif time_depth.source == "sonic_integrated" and time_depth.calibrated:
            vertical = "sonic_calibrated"
        elif time_depth.source == "sonic_integrated":
            vertical = "sonic_uncalibrated"
        else:
            vertical = "measured_time_depth" if time_depth.confidence == "high" else "sonic_calibrated"
        limitations.extend(time_depth.limitations)
    elif depth_reference_explicit and well_logs_available:
        vertical = "depth_reference_only"
        limitations.append("time_depth_relation_unavailable")
    else:
        vertical = "none"
        limitations.append("vertical_reference_unavailable")

    if horizontal == "seismic_crs_aligned" and vertical == "measured_time_depth":
        fusion = "precise_joint_analysis"
    elif horizontal == "seismic_crs_aligned" and vertical == "sonic_calibrated":
        fusion = "calibrated_joint_analysis"
    elif horizontal == "seismic_crs_aligned" and vertical == "sonic_uncalibrated":
        fusion = "approximate_vertical_mapping"
    elif horizontal in {"wellhead_only", "trajectory_available", "seismic_crs_aligned"}:
        fusion = "location_level_association"
    else:
        fusion = "separate_analysis_only"

    if seismic_available and not well_logs_available:
        run_mode = "seismic_only"
    elif well_logs_available and not seismic_available:
        run_mode = "well_log_only"
    elif not seismic_available and not well_logs_available:
        run_mode = "invalid"
    elif fusion in {"precise_joint_analysis", "calibrated_joint_analysis"}:
        run_mode = "multimodal_precise_aligned"
    elif fusion == "approximate_vertical_mapping":
        run_mode = "multimodal_approximate_aligned"
    elif fusion == "location_level_association":
        run_mode = "multimodal_location_aligned"
    else:
        run_mode = "multimodal_unaligned"
    return {
        "horizontal_level": horizontal,
        "vertical_level": vertical,
        "fusion_permission": fusion,
        "run_mode": run_mode,
        "limitations": list(dict.fromkeys(limitations)),
    }

