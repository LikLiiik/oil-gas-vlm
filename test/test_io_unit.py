"""I/O + geometry + exporter 单测（不加载 VLM）。

    python test/test_io_unit.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from pipeline.agents import AgentResult
from pipeline.exporter import (
    build_slice_mask, export_annotated_png, export_json,
    export_volume_attribute,
)
from pipeline.io import (
    SegyVolume, SliceGeometry, data_to_pixel, extract_inline_slice,
    extract_time_slice, extract_xline_slice, pixel_to_data,
    read_segy, render_slice, synthetic_volume, write_attribute_segy,
)
from pipeline.tasks import TASKS, get as get_task


# ---- geometry --------------------------------------------------------

def _geom(width=1000, height=500):
    return SliceGeometry(
        axis_x_name="CDP", axis_y_name="time_ms",
        x_min=0, x_max=300, y_top=0, y_bottom=2500,
        pixel_width=width, pixel_height=height,
    )

def test_pixel_to_data_roundtrip():
    g = _geom()
    px, py = data_to_pixel(150.0, 1250.0, g)
    d = pixel_to_data([px, py, px, py], g)
    assert abs(d["CDP_min"] - 150) < 0.5
    assert abs(d["time_ms_top"] - 1250) < 0.5

def test_pixel_to_data_bbox():
    g = _geom()
    # bbox 覆盖左上四分之一（0..500 px, 0..250 px）
    d = pixel_to_data([0, 0, 500, 250], g)
    assert d["CDP_min"] == 0
    assert d["CDP_max"] == 150
    assert d["time_ms_top"] == 0
    assert d["time_ms_bottom"] == 1250


# ---- SEG-Y read/write roundtrip -------------------------------------

def test_synthetic_volume_shape():
    v = synthetic_volume(n_il=10, n_xl=20, n_samples=100)
    assert v.cube.shape == (10, 20, 100)
    assert v.inlines.shape == (10,)
    assert v.xlines.shape == (20,)
    assert v.sample_interval_ms == 4.0
    assert v.time_axis_ms.shape == (100,)

def test_segy_roundtrip_via_synthetic():
    """构造 volume → 写 SEG-Y → 读回 → 比较。"""
    v0 = synthetic_volume(n_il=8, n_xl=12, n_samples=50)
    attr = np.random.rand(*v0.cube.shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "attr.sgy"
        write_attribute_segy(v0, attr, str(out_path))
        assert out_path.exists() and out_path.stat().st_size > 0
        v1 = read_segy(str(out_path), strict=False)
        assert v1.cube.shape == v0.cube.shape
        # SEG-Y 精度 IEEE float32，允许极小误差
        assert np.allclose(v1.cube, attr, atol=1e-3)

def test_slice_extractors():
    v = synthetic_volume(n_il=6, n_xl=10, n_samples=30)
    il = extract_inline_slice(v, 3)
    xl = extract_xline_slice(v, 5)
    ts = extract_time_slice(v, 15)
    assert il.shape == (30, 10)     # (n_samples, n_xl)
    assert xl.shape == (30, 6)      # (n_samples, n_il)
    assert ts.shape == (6, 10)      # (n_il, n_xl)


# ---- render_slice ---------------------------------------------------

def test_render_slice_returns_pil_and_geom():
    v = synthetic_volume(n_il=6, n_xl=10, n_samples=30)
    arr = extract_inline_slice(v, 3)
    img, geom = render_slice(
        arr, x_min=200, x_max=209, y_top=0, y_bottom=120,
        axis_x_name="crossline", axis_y_name="time_ms",
        slice_kind="inline", slice_index=103,
    )
    assert img.mode == "RGB"
    assert geom.axis_x_name == "crossline"
    assert geom.slice_index == 103
    assert geom.pixel_width == img.width


# ---- tasks registry ------------------------------------------------

def test_tasks_registry_has_four():
    assert set(TASKS.keys()) == {"fault", "horizon", "facies", "fracture"}

def test_task_prompt_hint_contains_classes():
    t = get_task("fault")
    hint = t.prompt_hint()
    assert "fault plane" in hint


# ---- exporter: mask build + annotated PNG + attribute SEG-Y ---------

def _fake_result_with_bboxes():
    r = AgentResult(agent="unit_test")
    r.ok = True
    r.results = [
        {"id": "yolo_fault_0", "class_name": "fault plane",
         "bbox_pixel": [100, 50, 200, 150],
         "confidence": 0.75, "coordinate_system": "pixel"},
        {"id": "yolo_fault_1", "class_name": "fault plane",
         "bbox_pixel": [600, 200, 800, 400],
         "confidence": 0.55, "coordinate_system": "pixel"},
    ]
    return r

def test_build_slice_mask_shape_and_values():
    r = _fake_result_with_bboxes()
    g = _geom(width=1000, height=500)
    mask = build_slice_mask(r, g, shape=(50, 100), task=get_task("fault"))
    assert mask.shape == (50, 100)
    assert mask.max() > 0    # 至少有一个检测落到 mask 上
    # 默认背景是 0
    assert (mask == 0).any()

def test_export_annotated_png_writes_file():
    r = _fake_result_with_bboxes()
    g = _geom()
    img = Image.new("RGB", (g.pixel_width, g.pixel_height), "black")
    with tempfile.TemporaryDirectory() as td:
        p = export_annotated_png(r, img, g, get_task("fault"), td)
        assert p.exists() and p.stat().st_size > 0
        assert p.suffix == ".png"

def test_export_json_captures_data_coords():
    r = _fake_result_with_bboxes()
    g = _geom()
    with tempfile.TemporaryDirectory() as td:
        p = export_json(r, g, "fault", td)
        import json as _json
        d = _json.loads(p.read_text())
        assert d["task"] == "fault"
        assert len(d["detections_data_coords"]) == 2
        assert "CDP_min" in d["detections_data_coords"][0]

def test_export_volume_attribute_writes_segy():
    v = synthetic_volume(n_il=6, n_xl=10, n_samples=30)
    # 每个 inline 一个 mask
    per_slice = {i: np.random.rand(30, 10).astype(np.float32) for i in range(6)}
    with tempfile.TemporaryDirectory() as td:
        p = export_volume_attribute(v, per_slice, get_task("fault"), td)
        assert p.exists() and p.stat().st_size > 0
        # 读回验证 shape
        v2 = read_segy(str(p), strict=False)
        assert v2.cube.shape == v.cube.shape


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
