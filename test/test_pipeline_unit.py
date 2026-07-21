"""Pipeline 内部单元测试（不加载 VLM，秒级跑完）。

    python test/test_pipeline_unit.py
    # 或 pytest test/test_pipeline_unit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline import downstream
from pipeline.downstream._shared import normalize_depth_ranges as _normalize_ranges
from pipeline.vlm import extract_json
from schemas import (
    WORKFLOW_PLAN_SCHEMA, WORKFLOW_VERIFICATION_SCHEMA, validate_output,
)


# ---- extract_json --------------------------------------------------------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}

def test_extract_json_embedded_in_text():
    text = 'here is my answer:\n```json\n{"x": [1,2], "y":"z"}\n```\nDone.'
    assert extract_json(text) == {"x": [1, 2], "y": "z"}

def test_extract_json_with_braces_in_strings():
    """字符串里的 { } 不应该干扰括号计数。"""
    text = '{"pattern": "regex {abc}", "n": 3}'
    assert extract_json(text) == {"pattern": "regex {abc}", "n": 3}

def test_extract_json_none_on_bad_input():
    assert extract_json("no JSON here at all") is None


# ---- _normalize_ranges: VLM 格式健壮性 -----------------------------------

def test_normalize_ranges_dict_form():
    r = _normalize_ranges([
        {"top_m": 1200, "bottom_m": 1255},
        {"top_m": 1550, "bottom_m": 1625},
    ])
    assert r == [(1200.0, 1255.0), (1550.0, 1625.0)]

def test_normalize_ranges_nested_list():
    r = _normalize_ranges([[1200, 1255], [1550, 1625]])
    assert r == [(1200.0, 1255.0), (1550.0, 1625.0)]

def test_normalize_ranges_flat_single_range():
    """VLM 实测输出过 [1500, 1700] 这种扁平写法。"""
    assert _normalize_ranges([1500, 1700]) == [(1500.0, 1700.0)]

def test_normalize_ranges_string_values():
    """VLM 实测输出过 ['1600', '1800'] 字符串。"""
    assert _normalize_ranges(["1600", "1800"]) == [(1600.0, 1800.0)]

def test_normalize_ranges_empty():
    assert _normalize_ranges([]) == []
    assert _normalize_ranges(None) == []

def test_normalize_ranges_alt_field_names():
    r = _normalize_ranges([{"top": 100, "bottom": 200}])
    assert r == [(100.0, 200.0)]


# ---- 下游注册表 ----------------------------------------------------------

def test_registry_bootstrapped():
    names = downstream.available_names()
    assert "sam" in names
    assert "traditional_code" in names
    assert "seismic_domain_model" in names
    assert "horizon_tracker" in names
    assert "facies_classifier" in names
    assert "well_log_analyzer" in names
    assert "attribute_extractor" in names
    assert len(names) >= 9  # 8 core + cig models

def test_registry_can_override():
    """业务方可以覆盖某个下游实现。"""
    class Stub:
        name = "sam"
        description = "stub"
        required_fields = []
        output_shape = ""
        def detect(self, instruction, image=None, context=None):
            return [{"stub": True}]
    downstream.register(Stub())
    assert downstream.get("sam").detect({}) == [{"stub": True}]
    # 恢复默认
    from pipeline.downstream.sam import Sam
    downstream.register(Sam())

def test_traditional_code_handles_vlm_flat_ranges():
    """曾在真实 smoke test 里出现的 case：VLM 给了扁平 range 导致 0 结果。"""
    np.random.seed(0)
    tc = downstream.get("traditional_code")
    out = tc.detect({"rules": [
        {"class_name": "sand", "rule": "GR<50",
         "expected_depth_ranges": [1500, 1700]},
    ]})
    assert len(out) >= 1
    assert out[0]["class_name"] == "sand"


# ---- Schema：per-model instruction 强约束 --------------------------------

_VALID_SDM_STEP = {
    "step": 1, "model": "seismic_domain_model",
    "instruction": {
        "task": "fault_detection",
        "attribute": "gradient",
        "confidence_threshold": 0.3,
        "min_region_area_pixels": 50,
    },
}
_VALID_TC_STEP = {
    "step": 1, "model": "traditional_code",
    "instruction": {"rules": [
        {"class_name": "sand", "rule": "GR<50",
         "expected_depth_ranges": [{"top_m": 1200, "bottom_m": 1260}]},
    ]},
}

def _plan(*steps):
    return {"scene_understanding": "test", "workflow_steps": list(steps)}

def test_plan_schema_accepts_valid():
    ok, _ = validate_output(WORKFLOW_PLAN_SCHEMA, _plan(_VALID_SDM_STEP, _VALID_TC_STEP))
    assert ok

def test_plan_schema_rejects_seismic_domain_without_task():
    """seismic_domain_model 必须含 task 字段。"""
    bad = {"step": 1, "model": "seismic_domain_model",
           "instruction": {"attribute": "gradient"}}
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan(bad))
    assert not ok
    assert "task" in " ".join(errs).lower()

def test_plan_schema_rejects_traditional_code_without_rules():
    bad = {"step": 1, "model": "traditional_code",
           "instruction": {"some_field": "x"}}
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan(bad))
    assert not ok
    assert "rules" in " ".join(errs).lower()

def test_plan_schema_rejects_unknown_model():
    bad = {"step": 1, "model": "wizardry", "instruction": {}}
    ok, _ = validate_output(WORKFLOW_PLAN_SCHEMA, _plan(bad))
    assert not ok


# ---- Verification schema -------------------------------------------------

def test_verification_schema_minimal_valid():
    ok, _ = validate_output(WORKFLOW_VERIFICATION_SCHEMA,
                            {"verified": [], "need_retry": False})
    assert ok

def test_verification_schema_missing_need_retry():
    ok, errs = validate_output(WORKFLOW_VERIFICATION_SCHEMA,
                               {"verified": [{"is_real": True}]})
    assert not ok
    assert "need_retry" in " ".join(errs).lower()


# ---- 手动 runner（不装 pytest 也能跑） -----------------------------------

if __name__ == "__main__":
    import inspect
    fns = [(n, f) for n, f in globals().items()
           if n.startswith("test_") and callable(f)
           and inspect.signature(f).parameters == {}]
    passed = 0
    failed = []
    for n, f in fns:
        try:
            f()
            passed += 1
            print(f"  ✓ {n}")
        except AssertionError as e:
            failed.append((n, str(e) or "assertion failed"))
            print(f"  ✗ {n}: {e}")
        except Exception as e:
            failed.append((n, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {n}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if not failed else 1)
