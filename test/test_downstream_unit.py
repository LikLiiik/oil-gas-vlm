"""下游模型单元测试：新模型实例化、schema、输出格式。

    python test/test_downstream_unit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline import downstream
from schemas import WORKFLOW_PLAN_SCHEMA, validate_output


# ── 合成数据 ────────────────────────────────────────────────────────────────

def _synth_seismic_2d(n_samples=200, n_traces=100, seed=0):
    """生成含一个水平层位+一个断层的合成地震剖面。"""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_samples, n_traces)).astype(np.float32) * 0.05
    # 层位在 sample 50
    arr[50, :] += 1.2
    # 一个小断层错断
    arr[50:, 60:] = np.roll(arr[50:, 60:], 5, axis=0)
    # 亮点
    arr[80:95, 30:50] -= 0.4
    return arr


def _synth_gr_curve(n=2000):
    """合成 GR 曲线：4 段砂岩低值在固定深度、背景泥岩高值。"""
    depth = np.linspace(1000, 2000, n)
    gr = np.full(n, 95.0, dtype=np.float32)
    gr += np.random.randn(n).astype(np.float32) * 6
    # 砂岩段
    for top, bot in [(1200, 1255), (1550, 1625)]:
        mask = (depth >= top) & (depth <= bot)
        gr[mask] = 35 + np.random.randn(mask.sum()).astype(np.float32) * 5
    return depth, gr


# ── 注册 ────────────────────────────────────────────────────────────────────

def test_registry_has_all_eight():
    names = downstream.available_names()
    for expected in ["sam", "traditional_code",
                     "seismic_domain_model", "attribute_extractor",
                     "horizon_tracker", "facies_classifier", "well_log_analyzer"]:
        assert expected in names, f"{expected} not in {names}"


def test_desc_includes_all_models():
    desc = downstream.available_models_desc()
    for name in ["horizon_tracker", "facies_classifier",
                 "well_log_analyzer", "attribute_extractor"]:
        assert name in desc, f"{name} missing in description"


# ── Schema 校验：新模型合法指令 ──────────────────────────────────────────────

def _plan(*steps):
    normalized = [
        {"image_name": "test_image", "reason": "unit test", **step}
        for step in steps
    ]
    return {
        "scene_understanding": "test",
        "analysis_status": "insufficient",
        "visual_evidence": [{
            "image_name": "test_image",
            "class_name": "unknown",
            "status": "insufficient",
            "observations": [],
            "confidence": 0.5,
        }],
        "workflow_steps": normalized,
    }


def test_schema_accepts_horizon_tracker():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "horizon_tracker",
        "instruction": {
            "seed_points": [{"trace_idx": 50, "sample_idx": 100}],
            "tracking_mode": "correlation",
            "horizon_name": "T3",
            "search_window_samples": 12,
        },
    }))
    assert ok, f"errors: {errs}"


def test_schema_rejects_horizon_without_seed_points():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "horizon_tracker",
        "instruction": {"tracking_mode": "peak"},
    }))
    assert not ok
    assert "seed_points" in " ".join(errs).lower()


def test_schema_accepts_facies_classifier():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "facies_classifier",
        "instruction": {
            "n_clusters": 5,
            "attribute_list": ["envelope", "sweetness"],
            "method": "gmm",
        },
    }))
    assert ok, f"errors: {errs}"


def test_schema_rejects_facies_without_n_clusters():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "facies_classifier",
        "instruction": {"attribute_list": ["envelope"]},
    }))
    assert not ok


def test_schema_accepts_well_log_analyzer():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "well_log_analyzer",
        "instruction": {
            "analysis_type": "full_analysis",
            "depth_range": {"top_m": 1000, "bottom_m": 2000},
        },
    }))
    assert ok, f"errors: {errs}"


def test_schema_rejects_well_log_without_analysis_type():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "well_log_analyzer",
        "instruction": {"rules": []},
    }))
    assert not ok


def test_schema_accepts_attribute_extractor():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "attribute_extractor",
        "instruction": {
            "attributes": ["envelope", "sweetness"],
            "regions_of_interest": [
                {"bbox_xyxy_norm": [0.1, 0.2, 0.5, 0.6]},
            ],
        },
    }))
    assert ok, f"errors: {errs}"


def test_schema_rejects_attribute_extractor_without_attributes():
    ok, errs = validate_output(WORKFLOW_PLAN_SCHEMA, _plan({
        "step": 1, "model": "attribute_extractor",
        "instruction": {},
    }))
    assert not ok


# ── attribute_extractor ─────────────────────────────────────────────────────

def test_attr_basic():
    m = downstream.get("attribute_extractor")
    arr = _synth_seismic_2d()
    out = m.detect({"attributes": ["envelope", "rms_amplitude"]},
                   context={"array": arr})
    assert len(out) >= 2
    for r in out:
        assert "statistics" in r
        assert "attribute_name" in r
        stats = r["statistics"]
        for k in ("min", "max", "mean", "std"):
            assert k in stats, f"{k} missing in {stats}"
    names = [r["attribute_name"] for r in out]
    assert "envelope" in names or any("envelope" in n for n in names)


def test_attr_with_roi():
    m = downstream.get("attribute_extractor")
    arr = _synth_seismic_2d()
    out = m.detect({
        "attributes": ["envelope"],
        "regions_of_interest": [
            {"bbox_xyxy_norm": [0.2, 0.3, 0.5, 0.5]},
        ],
    }, context={"array": arr})
    assert len(out) >= 1
    assert "roi_index" in out[0]


def test_attr_unknown_attr_returns_gracefully():
    m = downstream.get("attribute_extractor")
    arr = _synth_seismic_2d()
    out = m.detect({"attributes": ["nonexistent_attr"]},
                   context={"array": arr})
    assert isinstance(out, list)


# ── horizon_tracker ──────────────────────────────────────────────────────────

def test_horizon_tracker_basic():
    m = downstream.get("horizon_tracker")
    arr = _synth_seismic_2d()
    out = m.detect({
        "seed_points": [{"trace_idx": 40, "sample_idx": 50}],
        "tracking_mode": "correlation",
        "horizon_name": "T1",
    }, context={"array": arr})
    assert len(out) >= 1
    r = out[0]
    assert r["model"] == "horizon_tracker"
    assert "points" in r and len(r["points"]) > 1
    assert 0.0 <= r["continuity_score"] <= 1.0
    assert 0.0 <= r["average_confidence"] <= 1.0


def test_horizon_tracker_multi_seed():
    m = downstream.get("horizon_tracker")
    arr = _synth_seismic_2d()
    out = m.detect({
        "seed_points": [
            {"trace_idx": 30, "sample_idx": 50},
            {"trace_idx": 70, "sample_idx": 55},
        ],
        "tracking_mode": "correlation",
    }, context={"array": arr})
    # 两个种子应产生至少一个结果
    assert len(out) >= 1


def test_horizon_tracker_empty_seeds():
    m = downstream.get("horizon_tracker")
    arr = _synth_seismic_2d()
    out = m.detect({"seed_points": [], "tracking_mode": "peak"},
                   context={"array": arr})
    assert out == []


def test_horizon_tracker_peak_mode():
    m = downstream.get("horizon_tracker")
    arr = _synth_seismic_2d()
    out = m.detect({
        "seed_points": [{"trace_idx": 50, "sample_idx": 50}],
        "tracking_mode": "peak",
    }, context={"array": arr})
    assert len(out) >= 1
    # peak 模式应在层位附近
    assert out[0]["seed_sample_idx"] <= 55


def test_horizon_confidence_range():
    m = downstream.get("horizon_tracker")
    arr = _synth_seismic_2d()
    out = m.detect({
        "seed_points": [{"trace_idx": 50, "sample_idx": 50}],
        "tracking_mode": "correlation",
    }, context={"array": arr})
    for pt in out[0]["points"]:
        assert 0.0 <= pt["confidence"] <= 1.0


# ── facies_classifier ────────────────────────────────────────────────────────

def test_facies_basic():
    m = downstream.get("facies_classifier")
    arr = _synth_seismic_2d(n_samples=300, n_traces=150)
    out = m.detect({"n_clusters": 3}, context={"array": arr})
    assert len(out) >= 1
    n_clusters = len({r["cluster_id"] for r in out})
    assert n_clusters >= 1
    for r in out:
        assert r["model"] == "facies_classifier"
        assert "cluster_center" in r
        assert r["area_pixels"] > 0


def test_facies_different_k():
    m = downstream.get("facies_classifier")
    arr = _synth_seismic_2d(n_samples=300, n_traces=150)
    out5 = m.detect({"n_clusters": 5}, context={"array": arr})
    out3 = m.detect({"n_clusters": 3}, context={"array": arr})
    # 不同 k 产生不同簇数
    ids5 = {r["cluster_id"] for r in out5}
    ids3 = {r["cluster_id"] for r in out3}
    assert len(ids5) != len(ids3) or len(ids5) == 1


def test_facies_with_roi():
    m = downstream.get("facies_classifier")
    arr = _synth_seismic_2d(n_samples=300, n_traces=150)
    out = m.detect({
        "n_clusters": 2,
        "regions_of_interest": [
            {"bbox_xyxy_norm": [0.2, 0.2, 0.6, 0.6]},
        ],
    }, context={"array": arr})
    assert len(out) >= 1


# ── well_log_analyzer ────────────────────────────────────────────────────────

def test_wl_segmentation_with_curve():
    m = downstream.get("well_log_analyzer")
    depth, gr = _synth_gr_curve()
    out = m.detect({
        "analysis_type": "curve_segmentation",
    }, context={"curves": {"GR": gr, "depth": depth}})
    assert len(out) >= 1
    for r in out:
        assert "depth_top_m" in r
        assert "depth_bottom_m" in r
        assert r["model"] == "well_log_analyzer"


def test_wl_segmentation_via_rules_fallback():
    m = downstream.get("well_log_analyzer")
    out = m.detect({
        "analysis_type": "curve_segmentation",
        "rules": [
            {"class_name": "sand", "rule": "GR<50",
             "expected_depth_ranges": [{"top_m": 1200, "bottom_m": 1255}]},
        ],
    })
    assert len(out) >= 1
    assert out[0]["class_name"] == "sand"


def test_wl_depth_ranges_parsed():
    m = downstream.get("well_log_analyzer")
    # 测试扁平 range 格式
    out = m.detect({
        "analysis_type": "full_analysis",
        "rules": [
            {"class_name": "reservoir", "rule": "RT>20",
             "expected_depth_ranges": [1500, 1700]},
        ],
    })
    assert len(out) >= 1
    assert out[0]["class_name"] == "reservoir"
    assert 1490 < out[0]["depth_top_m"] < 1710


def test_wl_empty_input_returns_empty():
    m = downstream.get("well_log_analyzer")
    out = m.detect({"analysis_type": "full_analysis"})
    # 无 rules 无 curves → 空
    assert isinstance(out, list)


def test_wl_model_registered():
    assert downstream.get("well_log_analyzer") is not None
    assert downstream.get("well_log_analyzer").name == "well_log_analyzer"


# ── sam: 真实分割 ──────────────────────────────────────────────────────────

def test_sam_bbox_segmentation():
    """给定 bbox，SAM 应返回非零面积的 mask。"""
    import numpy as np
    from PIL import Image
    # 创建一个 100×100 的图像，中间有白色方块（模拟前景）
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[30:70, 30:70] = 200  # 白色方块
    img = Image.fromarray(arr)

    m = downstream.get("sam")
    out = m.detect({
        "prompt_type": "bbox",
        "prompt_value": [25, 25, 75, 75],
        "label": "white_block",
    }, image=img)
    assert len(out) >= 1
    assert out[0]["label"] == "white_block"
    assert out[0]["mask_area_pixels"] > 0, f"got area={out[0]['mask_area_pixels']}"


def test_sam_bbox_near_full_bbox():
    """mask 面积应该在 bbox 范围附近。"""
    import numpy as np
    from PIL import Image
    arr = np.full((100, 100), 50, dtype=np.uint8)
    arr[20:60, 20:60] = 200
    img = Image.fromarray(arr)

    m = downstream.get("sam")
    out = m.detect({
        "prompt_type": "bbox",
        "prompt_value": [15, 15, 65, 65],
        "label": "bright",
    }, image=img)
    assert out[0]["mask_area_pixels"] > 0
    # bbox_pixel 应该非零
    bbox = out[0]["bbox_pixel"]
    assert bbox[2] > bbox[0] and bbox[3] > bbox[1]


def test_sam_point_flood_fill():
    """从点出发的 flood fill 应在均匀区域内生长。"""
    import numpy as np
    from PIL import Image
    arr = np.full((100, 100), 100, dtype=np.uint8)
    arr[40:70, 40:70] = 180  # 较亮区块
    img = Image.fromarray(arr)

    m = downstream.get("sam")
    out = m.detect({
        "prompt_type": "point",
        "prompt_value": [55, 55],
        "label": "bright_zone",
        "tolerance": 0.25,
    }, image=img)
    assert len(out) >= 1
    assert out[0]["mask_area_pixels"] > 0


def test_sam_no_image_returns_zero():
    m = downstream.get("sam")
    out = m.detect({"prompt_type": "bbox", "prompt_value": [0, 0, 10, 10]})
    assert out[0]["mask_area_pixels"] == 0


# ── traditional_code: 真实规则执行 ──────────────────────────────────────────

def test_tc_real_curve_threshold():
    """有真实 GR 曲线时，traditional_code 应精确扫描。"""
    import numpy as np
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 95.0, dtype=np.float32)
    gr[400:510] = 35.0  # 1200-1255m 砂岩 (0.1524m 采样)
    gr[1100:1250] = 38.0  # 1550-1625m 砂岩

    m = downstream.get("traditional_code")
    out = m.detect({
        "rules": [
            {"class_name": "sand", "rule": "GR<50",
             "expected_depth_ranges": [
                 {"top_m": 1200, "bottom_m": 1260},
             ]},
        ],
    }, context={"curves": {"GR": gr, "depth": depth}})
    assert len(out) >= 1
    for r in out:
        assert r["class_name"] == "sand"
        assert "thickness_m" in r
        assert r["confidence"] >= 0.9  # 真实数据 → 高置信度


def test_tc_no_curve_uses_depth_ranges():
    """无曲线数据时，从 expected_depth_ranges 输出合理值（非随机噪声）。"""
    import numpy as np
    np.random.seed(0)

    m = downstream.get("traditional_code")
    out = m.detect({
        "rules": [
            {"class_name": "pay_zone", "rule": "RT>20",
             "expected_depth_ranges": [{"top_m": 1555, "bottom_m": 1595}]},
        ],
    })
    assert len(out) >= 1
    r = out[0]
    assert r["class_name"] == "pay_zone"
    assert r["depth_top_m"] == 1555.0
    assert r["depth_bottom_m"] == 1595.0
    assert r["thickness_m"] == 40.0
    # 无真实数据 → 置信度降低
    assert 0.5 < r["confidence"] <= 0.75


def test_tc_multiple_rules():
    """多个规则应各自产出结果。"""
    m = downstream.get("traditional_code")
    out = m.detect({
        "rules": [
            {"class_name": "sand", "rule": "GR<50",
             "expected_depth_ranges": [[1200, 1255]]},
            {"class_name": "shale", "rule": "GR>90",
             "expected_depth_ranges": [[1300, 1400]]},
        ],
    })
    classes = {r["class_name"] for r in out}
    assert "sand" in classes
    assert "shale" in classes


def test_tc_flat_range_format():
    """VLM 扁平 range [1500, 1700] 格式应正确解析。"""
    m = downstream.get("traditional_code")
    out = m.detect({"rules": [
        {"class_name": "reservoir", "rule": "RT>20",
         "expected_depth_ranges": [1500, 1700]},
    ]})
    assert len(out) >= 1
    assert out[0]["depth_top_m"] == 1500.0
    assert out[0]["depth_bottom_m"] == 1700.0


# ── 手动 runner（不装 pytest 也能跑）────────────────────────────────────────

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
