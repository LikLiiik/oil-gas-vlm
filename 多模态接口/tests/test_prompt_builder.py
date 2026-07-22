from geo_adapter.packaging.prompt_builder import build_prompts
from geo_adapter.schemas.manifest import Manifest


def test_calibrated_prompt_matches_alignment(prepared_run, project_root) -> None:
    manifest = Manifest.model_validate_json((prepared_run / "manifest.json").read_text(encoding="utf-8"))
    system, user = build_prompts(manifest, project_root / "configs/prompt_templates.yaml")
    assert "完成井级控制点标定" in system
    assert "不得虚构" in system
    assert manifest.alignment.fusion_permission in user


def test_no_time_depth_prompt_forbids_mapping(prepared_run, project_root) -> None:
    manifest = Manifest.model_validate_json((prepared_run / "manifest.json").read_text(encoding="utf-8")).model_copy(deep=True)
    manifest.alignment.vertical_level = "none"
    manifest.alignment.fusion_permission = "separate_analysis_only"
    manifest.time_depth_relation.available = False
    manifest.time_depth_relation.source = "none"
    system, _ = build_prompts(manifest, project_root / "configs/prompt_templates.yaml")
    assert "不得将测井深度区间映射到地震时间轴" in system
    assert "cross_modal_analysis.allowed 必须为 false" in system


def test_uncalibrated_prompt_marks_high_uncertainty(prepared_run, project_root) -> None:
    manifest = Manifest.model_validate_json((prepared_run / "manifest.json").read_text(encoding="utf-8")).model_copy(deep=True)
    manifest.alignment.vertical_level = "sonic_uncalibrated"
    manifest.alignment.fusion_permission = "approximate_vertical_mapping"
    system, _ = build_prompts(manifest, project_root / "configs/prompt_templates.yaml")
    assert "只能作为粗略参考" in system
    assert "高不确定性" in system

