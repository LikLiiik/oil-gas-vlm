"""Run prepared synthetic packages while reusing one loaded VLM instance."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="synthetic_benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fault_metrics(report: dict, label_dir: Path, run_dir: Path) -> dict | None:
    fault_steps = [
        step for step in ((report.get("vlm_plan") or {}).get("workflow_steps") or [])
        if step.get("model") in {"seismic_domain_model", "cig_fault"}
        and (step.get("instruction") or {}).get("task", "fault_detection") == "fault_detection"
    ]
    if not fault_steps:
        return None

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return {"error": f"manifest not found: {manifest_path}"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = (manifest.get("seismic") or {}).get("views") or {}
    observed_planes: list[tuple[str, int]] = []
    for step in fault_steps:
        image_name = str(step.get("image_name", ""))
        view_name = image_name.removeprefix("seismic_")
        view = views.get(view_name) or {}
        indices = view.get("source_indices") or {}
        if view_name == "inline" and "inline_index" in indices:
            observed_planes.append(("inline", int(indices["inline_index"])))
        elif view_name == "crossline" and "crossline_index" in indices:
            observed_planes.append(("crossline", int(indices["crossline_index"])))

    paths = (report.get("outputs") or {}).get("attribute_sgy") or {}
    candidate = next((value for key, value in paths.items() if "fault" in key.lower()), None)
    target_path = label_dir / "fault_mask.npy"
    if not target_path.is_file():
        return {"error": f"target not found: {target_path}"}
    target = np.load(target_path, allow_pickle=False).astype(bool)
    prediction = np.zeros(target.shape, dtype=np.float32)
    prediction_path = Path(candidate) if candidate else None
    try:
        if prediction_path is not None and prediction_path.is_file():
            import segyio
            with segyio.open(str(prediction_path), "r", strict=False, ignore_geometry=False) as handle:
                prediction = segyio.tools.cube(handle).astype(np.float32)
    except Exception as exc:
        return {"error": str(exc)}
    if prediction.shape != target.shape:
        return {"error": f"shape mismatch: {prediction.shape} != {target.shape}"}
    visible = np.zeros(target.shape, dtype=bool)
    for kind, index in observed_planes:
        if kind == "inline" and 0 <= index < target.shape[0]:
            visible[index, :, :] = True
        elif kind == "crossline" and 0 <= index < target.shape[1]:
            visible[:, index, :] = True
    if not visible.any():
        return {"error": "fault step has no evaluable inline/crossline source index"}
    finite = np.isfinite(prediction) & visible
    best = None
    for threshold in np.arange(0.1, 0.91, 0.05):
        pred = prediction[finite] >= float(threshold)
        truth = target[finite]
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        item = {"threshold": round(float(threshold), 3), "precision": precision, "recall": recall, "f1": f1}
        if best is None or (item["f1"], item["recall"]) > (best["f1"], best["recall"]):
            best = item
    best["evaluation_scope"] = "observed_fault_slices"
    best["observed_planes"] = [list(item) for item in observed_planes]
    best["prediction_available"] = bool(candidate)
    best["evaluated_voxels"] = int(np.count_nonzero(finite))
    return best


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    workspace = (project / args.workspace).resolve()
    index_path = workspace / "dataset_index.json"
    if not index_path.is_file():
        raise SystemExit(f"dataset index not found: {index_path}")
    dataset = [item for item in json.loads(index_path.read_text(encoding="utf-8")) if item.get("status") in {"ok", "exists"}]
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    if not dataset:
        raise SystemExit("no prepared samples")

    from pipeline import Pipeline
    runner = Pipeline(verbose=True)
    output_root = workspace / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    model_counts: Counter[str] = Counter()

    for position, item in enumerate(dataset, 1):
        sample_id = item["sample_id"]
        output_dir = output_root / sample_id
        report_path = output_dir / "report.json"
        print(f"\n===== [{position}/{len(dataset)}] {sample_id} =====")
        started = time.time()
        try:
            if args.resume and report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                status = "resumed"
            else:
                report = runner.run_from_adapter(
                    item["run_dir"], output_dir,
                    verify=not args.no_verify,
                    max_iterations=args.max_iter,
                )
                status = "ok" if report.get("ok") else "failed"
            elapsed = time.time() - started
            models = (report.get("downstream") or {}).get("models_used") or []
            model_counts.update(models)
            metrics = fault_metrics(
                report, Path(item["label_dir"]), Path(item["run_dir"]),
            )
            row = {
                "sample_id": sample_id, "seed": item["seed"], "status": status,
                "elapsed_s": round(elapsed, 3), "models_used": models,
                "n_detections": (report.get("downstream") or {}).get("n_detections", 0),
                "warnings": report.get("warnings") or [], "fault_metrics": metrics,
            }
        except Exception as exc:
            row = {
                "sample_id": sample_id, "seed": item["seed"], "status": "exception",
                "elapsed_s": round(time.time() - started, 3), "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        (workspace / "benchmark_progress.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    successful = [row for row in results if row["status"] in {"ok", "resumed"}]
    f1_values = [row["fault_metrics"]["f1"] for row in successful if isinstance(row.get("fault_metrics"), dict) and "f1" in row["fault_metrics"]]
    summary = {
        "samples": len(results),
        "successful": len(successful),
        "success_rate": len(successful) / len(results),
        "mean_elapsed_s": float(np.mean([row["elapsed_s"] for row in results])),
        "model_usage": dict(model_counts),
        "fault_f1_available_samples": len(f1_values),
        "mean_best_fault_f1": float(np.mean(f1_values)) if f1_values else None,
        "note": "Synthetic best-threshold F1 is a regression diagnostic, not a field-data score.",
        "results": results,
    }
    summary_path = workspace / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}")
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
