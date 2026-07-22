from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from downstream.scripts.evaluate_segmentation_f1 import binary_metrics, threshold_sweep
from pipeline import downstream
from pipeline.orchestrator import Pipeline
from pipeline.context import build_downstream_context
from pipeline.downstream.cig_models import _get_volume
from pipeline.exporter import aggregate_adapter_detections, normalize_detection_format
from pipeline.io.segy import (
    read_segy,
    synthetic_volume,
    write_attribute_segy,
    write_attribute_segy_like,
)


def test_context_loads_canonical_seismic_and_masked_curves(tmp_path):
    (tmp_path / "arrays").mkdir()
    (tmp_path / "tables").mkdir()
    raw = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.save(tmp_path / "arrays" / "inline_raw.npy", raw)

    values = np.zeros((5, 9), dtype=np.float32)
    values[:, 1] = [80, 70, 60, 50, 40]
    values[:, 3] = [2, 3, 4, 5, 6]
    valid = np.zeros_like(values, dtype=bool)
    valid[:, [1, 3]] = True
    interpolated = np.zeros_like(values, dtype=bool)
    available = np.zeros(9, dtype=bool)
    available[[1, 3]] = True
    np.save(tmp_path / "arrays" / "well_values.npy", values)
    np.save(tmp_path / "arrays" / "well_valid_mask.npy", valid)
    np.save(tmp_path / "arrays" / "well_interpolated_mask.npy", interpolated)
    np.save(tmp_path / "arrays" / "curve_available.npy", available)
    (tmp_path / "tables" / "well_logs_clean.csv").write_text(
        "DEPTH,GR\n1000,80\n1001,70\n1002,60\n1003,50\n1004,40\n",
        encoding="utf-8",
    )
    (tmp_path / "tables" / "time_depth.csv").write_text(
        "depth,twt_ms\n1000,800\n1004,808\n", encoding="utf-8",
    )
    (tmp_path / "tables" / "formation_tops_m.csv").write_text(
        "FORMATION,MD_M\nTop_A,1001.5\n", encoding="utf-8",
    )

    view = {
        "physical_view": "inline",
        "raw_array_path": "arrays/inline_raw.npy",
        "array_shape": [4, 6],
        "axis_labels": ["crossline", "sample"],
        "source_indices": {"inline_index": 2},
    }
    manifest = {
        "seismic": {
            "views": {"inline": view},
            "qc": {"metadata": {"sample_interval_ms": 2.0}},
        },
        "well_logs": {
            "curve_order": [
                "SP", "GR", "CAL", "RES_DEEP", "RES_MEDIUM_SHALLOW",
                "RES_MICRO", "AC", "DEN", "CNL",
            ],
            "depth_range": [1000, 1004],
        },
        "time_depth_relation": {
            "table_path": "tables/time_depth.csv", "confidence": "high",
        },
        "alignment": {"fusion_permission": "allowed"},
    }
    pkg = SimpleNamespace(
        run_dir=tmp_path,
        manifest=manifest,
        view_meta=lambda name: view if name == "inline" else None,
    )
    context = build_downstream_context(
        SimpleNamespace(physical_view="inline"), pkg,
    )

    assert context["array"].shape == (6, 4)
    assert np.array_equal(context["array"], raw.T)
    assert np.array_equal(context["curves"]["RT"], values[:, 3])
    assert "CNL" not in context["curves"]
    assert context["curve_availability"]["CNL"] is False
    assert context["time_axis_ms"].tolist() == [0, 2, 4, 6, 8, 10]
    assert context["time_depth_pairs"].shape == (2, 2)
    assert context["formation_tops"] == [
        {"formation": "Top_A", "depth_m": 1001.5}
    ]


def test_mask_is_aggregated_instead_of_its_bounding_box():
    manifest = {
        "seismic": {
            "views": {
                "inline": {
                    "physical_view": "inline",
                    "array_shape": [4, 6],
                    "axis_labels": ["crossline", "sample"],
                    "source_indices": {"inline_index": 1},
                    "model_image_path": "assets/seismic/inline_model.png",
                }
            }
        }
    }
    mask = np.zeros((6, 4), dtype=bool)
    mask[2, 1] = True
    raw = {
        "seismic_inline": [{
            "id": "one_pixel",
            "class_name": "fault",
            "confidence": 0.9,
            "bbox_pixel": [1, 2, 1, 2],
            "coordinate_space": "array",
            "coordinate_shape": [6, 4],
            "_mask_array": mask,
        }]
    }
    image = SimpleNamespace(pil=SimpleNamespace(size=(400, 600)))
    normalized = normalize_detection_format(
        raw, manifest, {"seismic_inline": image},
    )
    cube = aggregate_adapter_detections(normalized, manifest, (3, 4, 6))["fault"]
    assert np.count_nonzero(cube) == 1
    assert np.isclose(cube[1, 1, 2], 0.9)


def test_binary_f1_and_threshold_sweep():
    prediction = np.array([0.9, 0.8, 0.4, 0.1], dtype=np.float32)
    target = np.array([1, 1, 0, 0], dtype=np.uint8)
    metrics = binary_metrics(prediction, target, threshold=0.5)
    assert metrics["f1"] == 1.0
    sweep = threshold_sweep(prediction, target, 0.3, 0.7, 0.2)
    assert sweep["best"]["f1"] == 1.0


def test_default_registry_contains_no_runtime_trained_or_placeholder_models():
    names = downstream.available_names()
    assert "well_log_ml" not in names
    assert "seismic_foundation" not in names


def test_vlm_rejection_is_advisory_for_dense_domain_probability_maps():
    dense = {
        "id": "dense_fault_1",
        "class_name": "fault",
        "model": "seismic_domain_model",
        "_probability_map": np.ones((4, 5), dtype=np.float32),
    }
    bbox = {
        "id": "bbox_fault_1",
        "model": "yolo_world",
        "bbox_pixel": [0, 0, 2, 2],
    }
    filtered = Pipeline._filter_competition_detections(
        {"seismic_inline": [dense, bbox]},
        [
            {"result_id": "dense_fault_1", "is_real": False},
            {"result_id": "bbox_fault_1", "is_real": False},
        ],
    )
    assert len(filtered["seismic_inline"]) == 1
    retained = filtered["seismic_inline"][0]
    assert retained["class_name"] == "fault_candidate"
    assert retained["original_class_name"] == "fault"
    assert retained["verification_status"] == "rejected_by_vlm"
    assert retained["_probability_map"] is dense["_probability_map"]


def test_plan_sanitizer_adds_crossline_fault_and_reports_seed_clamp():
    inline_view = {
        "array_shape": [188, 1501],
        "axis_labels": ["crossline", "sample"],
    }
    crossline_view = {
        "array_shape": [345, 1501],
        "axis_labels": ["inline", "sample"],
    }
    images = [
        SimpleNamespace(name="seismic_inline", physical_view="inline"),
        SimpleNamespace(name="seismic_crossline", physical_view="crossline"),
    ]
    pkg = SimpleNamespace(
        images=images,
        manifest={"seismic": {}, "well_logs": {}},
        view_meta=lambda name: {
            "inline": inline_view,
            "crossline": crossline_view,
        }.get(name),
    )
    steps = [
        {
            "step": 1,
            "model": "seismic_domain_model",
            "image_name": "seismic_inline",
            "instruction": {
                "task": "fault_detection",
                "attribute": "gradient",
                "regions_of_interest": [{"bbox_xyxy_norm": [0, 0, 1, 1]}],
            },
        },
        {
            "step": 2,
            "model": "horizon_tracker",
            "image_name": "seismic_inline",
            "instruction": {
                "seed_points": [{"trace_idx": 200, "sample_idx": 1600}],
            },
        },
    ]
    sanitized, adjustments = Pipeline._sanitize_competition_steps(pkg, steps)
    fault_steps = [s for s in sanitized if s["model"] == "seismic_domain_model"]
    assert {s["image_name"] for s in fault_steps} == {
        "seismic_inline", "seismic_crossline"
    }
    mirrored = next(s for s in fault_steps if s["image_name"] == "seismic_crossline")
    assert "regions_of_interest" not in mirrored["instruction"]
    horizon = next(s for s in sanitized if s["model"] == "horizon_tracker")
    assert horizon["instruction"]["seed_points"][0] == {
        "trace_idx": 187, "sample_idx": 1500
    }
    assert any("seed 0 adjusted" in item for item in adjustments)
    assert any("inline/crossline fault consistency" in item for item in adjustments)


def test_verifier_summary_keeps_both_directions():
    detections = {
        "seismic_inline": [
            {"id": f"i{index}", "model": "seismic_domain_model", "confidence": index}
            for index in range(20)
        ],
        "seismic_crossline": [
            {"id": f"x{index}", "model": "seismic_domain_model", "confidence": index}
            for index in range(20)
        ],
    }
    payload = Pipeline._summarize_for_verification(
        detections, max_per_image_model=3
    )
    assert [item["id"] for item in payload["seismic_inline"]] == ["i19", "i18", "i17"]
    assert [item["id"] for item in payload["seismic_crossline"]] == ["x19", "x18", "x17"]


def test_fp_dominant_fault_retry_cannot_loosen_either_direction():
    steps = [
        {
            "step": 3,
            "model": "seismic_domain_model",
            "image_name": "seismic_inline",
            "instruction": {
                "task": "fault_detection",
                "confidence_threshold": 0.5,
                "min_region_area_pixels": 1000,
            },
        },
        {
            "step": 7,
            "model": "seismic_domain_model",
            "image_name": "seismic_crossline",
            "instruction": {
                "task": "fault_detection",
                "confidence_threshold": 0.5,
                "min_region_area_pixels": 1000,
            },
        },
    ]
    verification = {
        "verified": [
            {"result_id": "a", "is_real": False},
            {"result_id": "b", "is_real": False},
            {"result_id": "c", "is_real": True},
        ],
        "retry_instructions": {
            "step": 7,
            "adjusted_params": {
                "confidence_threshold": 0.3,
                "min_region_area_pixels": 80,
            },
        },
    }
    adjustments = Pipeline._apply_competition_retry(steps, verification)
    assert len(adjustments) == 2
    for step in steps:
        assert step["instruction"]["confidence_threshold"] >= 0.55
        assert step["instruction"]["min_region_area_pixels"] >= 1000


def test_fault_candidate_limit_and_verification_coverage():
    detections = {
        "seismic_inline": [
            {
                "id": f"f{index}",
                "class_name": "fault",
                "model": "seismic_domain_model",
                "confidence": index / 10,
                "area_pixels": index,
            }
            for index in range(12)
        ] + [{"id": "facies", "model": "facies_classifier"}],
    }
    limited, stats = Pipeline._limit_fault_candidates(detections, max_per_image=3)
    assert stats == {
        "limit_per_image": 3,
        "raw": 12,
        "kept": 3,
        "dropped": 9,
    }
    assert {item["id"] for item in limited["seismic_inline"]} == {
        "f9", "f10", "f11", "facies",
    }

    limited["seismic_inline"][1]["verification_status"] = "verified"
    limited["seismic_inline"][2]["verification_status"] = "rejected_by_vlm"
    coverage = Pipeline._fault_verification_coverage(limited)
    assert coverage == {
        "total_candidates": 3,
        "reviewed": 2,
        "verified": 1,
        "rejected": 1,
        "unreviewed": 1,
        "review_fraction": 0.666667,
    }


def test_cig_rejects_pseudo_3d_single_slice():
    assert _get_volume({"array": np.zeros((8, 8), dtype=np.float32)}) is None
    volume = np.zeros((4, 8, 8), dtype=np.float32)
    assert _get_volume({"volume": volume}).shape == volume.shape


def test_write_attribute_segy_like_preserves_geometry(tmp_path):
    reference_volume = synthetic_volume(n_il=3, n_xl=4, n_samples=8)
    reference_volume.inlines = np.array([101, 103, 107], dtype=np.int32)
    reference_volume.xlines = np.array([201, 205, 209, 213], dtype=np.int32)
    reference_path = tmp_path / "reference.sgy"
    output_path = tmp_path / "attribute.sgy"
    write_attribute_segy(
        reference_volume, np.zeros_like(reference_volume.cube), str(reference_path),
    )
    attribute = np.full(reference_volume.cube.shape, 0.75, dtype=np.float32)
    write_attribute_segy_like(str(reference_path), attribute, str(output_path))
    result = read_segy(str(output_path))
    assert result.inlines.tolist() == [101, 103, 107]
    assert result.xlines.tolist() == [201, 205, 209, 213]
    assert np.allclose(result.cube, attribute)


def test_demo_run_executes_downstream_without_training_or_gpu(tmp_path):
    demo = Path(__file__).resolve().parents[1] / "多模态接口" / "runs" / "demo_sample_001"
    if not demo.is_dir():
        import pytest
        pytest.skip("demo run is not available")

    plan = {
        "scene_understanding": "synthetic smoke test",
        "workflow_steps": [{
            "step": 1,
            "model": "seismic_domain_model",
            "image_name": "seismic_inline",
            "reason": "CPU inference contract smoke test",
            "instruction": {
                "task": "fault_detection",
                "attribute": "gradient",
                "confidence_threshold": 0.4,
                "min_region_area_pixels": 20,
            },
        }],
        "max_iterations": 1,
    }

    class FakeVLM:
        def call_json(self, *args, **kwargs):
            return SimpleNamespace(
                data=plan,
                schema_valid=True,
                schema_errors=[],
                text=json.dumps(plan),
                elapsed_s=0.0,
                attempts=1,
            )

    report = Pipeline(vlm=FakeVLM(), verbose=False).run_from_adapter(
        demo, out_dir=tmp_path / "out", verify=False, max_iterations=1,
    )
    assert report["ok"] is True
    saved = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    serialized = json.dumps(saved, ensure_ascii=False)
    assert "probability_map_summary" in serialized
    assert '"_probability_map"' not in serialized
