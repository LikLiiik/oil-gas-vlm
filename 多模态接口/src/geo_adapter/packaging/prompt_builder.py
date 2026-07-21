from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from geo_adapter.schemas.manifest import Manifest


def build_prompts(manifest: Manifest, templates_path: Path) -> tuple[str, str]:
    """Render prompts from manifest state; no alignment claim is hard-coded."""
    templates: dict[str, str] = yaml.safe_load(templates_path.read_text(encoding="utf-8"))
    system_parts = [templates["system_base"]]
    vertical = manifest.alignment.vertical_level
    if vertical in {"none", "depth_reference_only"}:
        system_parts.append(templates["no_time_depth"])
    elif vertical == "sonic_uncalibrated":
        system_parts.append(templates["sonic_uncalibrated"])
    elif vertical == "sonic_calibrated":
        system_parts.append(templates["sonic_calibrated"])
    elif vertical == "measured_time_depth":
        td = manifest.time_depth_relation
        system_parts.append(templates["measured_time_depth"])
        system_parts.append(
            "时深来源={source}；控制点数={count}；覆盖深度={depth}；TWT范围={twt}；RMSE={rmse}。".format(
                source=td.source,
                count=td.calibration.control_point_count,
                depth=td.depth_range,
                twt=td.twt_range_ms,
                rmse=td.calibration.rmse_ms,
            )
        )
    if manifest.alignment.fusion_permission in {"separate_analysis_only", "location_level_association"}:
        system_parts.append("cross_modal_analysis.allowed 必须为 false；不得生成具体井震纵向对应结论。")
    else:
        system_parts.append("cross_modal_analysis.allowed 可为 true，但结论必须受 manifest 限制项约束。")

    curves = manifest.well_logs.curves
    available = [name for name, curve in curves.items() if curve.available]
    missing = [name for name, curve in curves.items() if not curve.available]
    families = {
        name: curve.measurement_family
        for name, curve in curves.items()
        if name.startswith("RES_") and curve.available
    }
    seismic_views = list((manifest.seismic.get("views") or {}).keys())
    values: dict[str, Any] = {
        "sample_id": manifest.sample_id,
        "task_type": manifest.task.type,
        "target_classes": ", ".join(manifest.task.target_classes) or "未指定",
        "run_mode": manifest.run_mode,
        "horizontal_level": manifest.alignment.horizontal_level,
        "vertical_level": vertical,
        "fusion_permission": manifest.alignment.fusion_permission,
        "seismic_views": ", ".join(seismic_views) or "无",
        "seismic_domain": manifest.seismic.get("domain", "unknown"),
        "available_curves": ", ".join(available) or "无",
        "missing_curves": ", ".join(missing) or "无",
        "depth_range": manifest.well_logs.depth_range,
        "resistivity_families": families or "无",
        "well_location_status": "available" if manifest.availability.well_location else "missing",
        "trajectory_status": manifest.trajectory.get("quality", "missing"),
        "crs_aligned": manifest.alignment.horizontal_level == "seismic_crs_aligned",
        "time_depth_source": manifest.time_depth_relation.source,
        "t0_status": manifest.time_depth_relation.calibration.t0_ms,
        "calibration_error": manifest.time_depth_relation.calibration.rmse_ms,
    }
    user_prompt = templates["user_base"].format(**values)
    if manifest.alignment.limitations:
        user_prompt += "\n当前限制：" + "；".join(manifest.alignment.limitations) + "。"
    return "\n\n".join(system_parts).strip() + "\n", user_prompt.strip() + "\n"

