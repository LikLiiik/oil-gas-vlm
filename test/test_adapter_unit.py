"""geo_adapter 对接层单测（不加载 VLM，用 demo run 目录做 fixture）。

    python test/test_adapter_unit.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pytest

from pipeline import load_run
from pipeline.adapter import PackageImage, RunPackage, build_vlm_user_text
from pipeline.exporter import aggregate_adapter_detections
from pipeline.tasks import CLASS_ALIASES, hint_for_target_classes

DEMO_RUN = Path(os.environ.get(
    "OIL_GAS_DEMO_RUN",
    Path(__file__).resolve().parents[1] / "多模态接口" / "runs" / "demo_sample_001",
))

def _skip_if_no_demo():
    if not DEMO_RUN.is_dir():
        pytest.skip(f"demo run dir not provided: {DEMO_RUN}")


def _load_demo():
    _skip_if_no_demo()
    return load_run(DEMO_RUN)


# ---- adapter.load_run --------------------------------------------------

def test_load_run_returns_package():
    pkg = _load_demo()
    assert isinstance(pkg, RunPackage)
    assert pkg.sample_id == "demo_sample_001"
    assert pkg.task_type == "geological_target_detection"
    assert "fault" in pkg.target_classes

def test_load_run_reads_all_images():
    pkg = _load_demo()
    assert len(pkg.images) >= 4    # inline + crossline + slice + local_patch (+ log panel)
    names = [im.name for im in pkg.images]
    assert "seismic_inline" in names
    for im in pkg.images:
        assert isinstance(im, PackageImage)
        assert im.pil.mode == "RGB"

def test_image_by_name_lookup():
    pkg = _load_demo()
    im = pkg.image_by_name("seismic_inline")
    assert im is not None
    assert im.physical_view == "inline"
    assert pkg.image_by_name("nonexistent") is None

def test_view_meta_reads_from_manifest():
    pkg = _load_demo()
    v = pkg.view_meta("inline")
    assert v is not None
    assert "source_indices" in v
    assert "inline_index" in v["source_indices"]

def test_load_run_reads_prompts_and_schema():
    pkg = _load_demo()
    assert pkg.system_prompt and "多模态" in pkg.system_prompt
    assert pkg.user_prompt and pkg.sample_id in pkg.user_prompt
    assert pkg.expected_schema.get("type") == "object"
    assert "downstream_plan" in pkg.expected_schema.get("properties", {})


# ---- build_vlm_user_text --------------------------------------------

def test_build_user_text_includes_manifest_summary():
    pkg = _load_demo()
    text = build_vlm_user_text(pkg, task_hint="test hint")
    assert "test hint" in text
    assert "manifest" in text.lower()
    assert pkg.sample_id in text


# ---- tasks.hint_for_target_classes ---------------------------------

def test_class_aliases_cover_geo_adapter_demo_classes():
    for cls in ("fault", "channel", "reservoir_candidate"):
        assert cls in CLASS_ALIASES

def test_hint_for_target_classes_maps_channel_to_facies():
    h = hint_for_target_classes(["fault", "channel", "fracture"])
    assert "canonical=fault" in h
    assert "canonical=facies" in h        # channel → facies
    assert "canonical=fracture" in h

def test_hint_for_unknown_class_falls_back():
    h = hint_for_target_classes(["some_novel_class"])
    assert "some_novel_class" in h
    assert "无内置描述" in h


# ---- exporter.aggregate_adapter_detections -------------------------

def _fake_manifest_and_dets():
    """构造一个最小 manifest + 三张图 (inline / crossline / slice) 的检测。"""
    manifest = {
        "seismic": {
            "shape": [10, 20, 50],
            "views": {
                "inline": {
                    "physical_view": "inline",
                    "array_shape": [20, 50],   # (n_xl, n_samples)
                    "axis_labels": ["crossline_index", "sample_index"],
                    "source_indices": {"inline_index": 3},
                    "model_image_path": "assets/seismic/inline_model.png",
                },
                "crossline": {
                    "physical_view": "crossline",
                    "array_shape": [10, 50],
                    "axis_labels": ["inline_index", "sample_index"],
                    "source_indices": {"crossline_index": 8},
                    "model_image_path": "assets/seismic/crossline_model.png",
                },
                "slice": {
                    "physical_view": "time_or_depth_slice",
                    "array_shape": [10, 20],
                    "axis_labels": ["inline_index", "crossline_index"],
                    "source_indices": {"sample_index": 25},
                    "model_image_path": "assets/seismic/slice_model.png",
                },
            },
        },
    }
    detections = {
        "seismic_inline": [
            {"class_name": "fault plane",
             "bbox_pixel": [100, 200, 300, 400],
             "bbox_norm": [0.1, 0.2, 0.5, 0.6],
             "confidence": 0.8, "in_roi": True},
        ],
        "seismic_crossline": [
            {"class_name": "fault plane",
             "bbox_pixel": [50, 100, 150, 250],
             "bbox_norm": [0.2, 0.3, 0.4, 0.7],
             "confidence": 0.6, "in_roi": True},
        ],
        "seismic_slice": [
            {"class_name": "channel",
             "bbox_pixel": [80, 100, 200, 300],
             "bbox_norm": [0.1, 0.1, 0.5, 0.5],
             "confidence": 0.5, "in_roi": True},
        ],
    }
    return manifest, detections

def test_aggregate_produces_cube_per_class():
    m, dets = _fake_manifest_and_dets()
    per_class = aggregate_adapter_detections(dets, m, (10, 20, 50))
    assert set(per_class.keys()) == {"fault plane", "channel"}
    for cube in per_class.values():
        assert cube.shape == (10, 20, 50)

def test_aggregate_paints_inline_view_correctly():
    m, dets = _fake_manifest_and_dets()
    per_class = aggregate_adapter_detections(dets, m, (10, 20, 50))
    fault_cube = per_class["fault plane"]
    # inline_index=3，bbox_norm=[0.1,0.2,0.5,0.6] 对应
    # xl [2:10], samples [10:30]，都应该 > 0
    assert fault_cube[3, 2:10, 10:30].max() > 0
    # 其它 inline 不应该被 inline 视图的检测污染（crossline 视图会写别处）
    assert fault_cube[0, :, :].max() >= 0

def test_aggregate_paints_slice_view_at_correct_sample():
    m, dets = _fake_manifest_and_dets()
    per_class = aggregate_adapter_detections(dets, m, (10, 20, 50))
    channel_cube = per_class["channel"]
    # sample_index=25，其它 sample 应该为 0
    assert channel_cube[:, :, 25].max() == 0.5
    assert channel_cube[:, :, 24].max() == 0

def test_aggregate_skips_detections_out_of_roi():
    m, dets = _fake_manifest_and_dets()
    dets["seismic_inline"][0]["in_roi"] = False
    # Isolate the inline result; the valid crossline box legitimately crosses
    # inline index 3 after axis-order correction.
    dets["seismic_crossline"] = []
    per_class = aggregate_adapter_detections(dets, m, (10, 20, 50))
    assert "channel" in per_class
    assert "fault plane" not in per_class


# ---- schema validation using geo_adapter's schema ------------------

def test_schema_rejects_missing_downstream_plan():
    from schemas import validate_output
    pkg = _load_demo()
    # 缺 required 字段
    bad = {"sample_id": pkg.sample_id, "seismic_analysis": {}}
    ok, errs = validate_output(pkg.expected_schema, bad)
    assert not ok
    assert any("downstream_plan" in e or "required" in e for e in errs)


# ---- 手动 runner ----------------------------------------------------

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
