"""Evaluate binary 2D/3D geological segmentation without training a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Prediction .npy/.npz/.sgy/.segy/image.")
    parser.add_argument("--target", required=True, help="Binary ground-truth file.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--prediction-key", default=None, help="Array key for prediction NPZ.")
    parser.add_argument("--target-key", default=None, help="Array key for target NPZ.")
    parser.add_argument(
        "--sweep",
        default=None,
        help="Optional start,stop,step threshold sweep, e.g. 0.1,0.9,0.05.",
    )
    parser.add_argument("--save", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def load_array(path: str | Path, key: str | None = None) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(source, allow_pickle=False))
    if suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        selected = key or (archive.files[0] if archive.files else None)
        if selected is None or selected not in archive.files:
            raise ValueError(f"NPZ key not found; available keys: {archive.files}")
        return np.asarray(archive[selected])
    if suffix in {".sgy", ".segy"}:
        import segyio
        with segyio.open(str(source), "r", strict=False, ignore_geometry=False) as handle:
            return segyio.tools.cube(handle).astype(np.float32)
    from PIL import Image
    return np.asarray(Image.open(source).convert("L"), dtype=np.float32)


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != target shape {target.shape}"
        )
    finite = np.isfinite(prediction) & np.isfinite(target)
    pred = prediction[finite] >= float(threshold)
    truth = target[finite] > 0
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "threshold": float(threshold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "dice": round(f1, 6),
        "iou": round(iou, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "evaluated_elements": int(np.count_nonzero(finite)),
    }


def threshold_sweep(
    prediction: np.ndarray,
    target: np.ndarray,
    start: float,
    stop: float,
    step: float,
) -> dict[str, Any]:
    if step <= 0 or stop < start:
        raise ValueError("sweep requires step > 0 and stop >= start")
    thresholds = np.arange(start, stop + step * 0.5, step)
    results = [binary_metrics(prediction, target, float(value)) for value in thresholds]
    best = max(results, key=lambda item: (item["f1"], item["recall"], -item["threshold"]))
    return {"best": best, "results": results}


def main() -> int:
    args = parse_args()
    prediction = load_array(args.prediction, args.prediction_key)
    target = load_array(args.target, args.target_key)
    payload: dict[str, Any] = {
        "prediction": str(Path(args.prediction).expanduser().resolve()),
        "target": str(Path(args.target).expanduser().resolve()),
        "shape": list(prediction.shape),
    }
    if args.sweep:
        parts = [float(value.strip()) for value in args.sweep.split(",")]
        if len(parts) != 3:
            raise SystemExit("--sweep must be start,stop,step")
        payload["sweep"] = threshold_sweep(prediction, target, *parts)
    else:
        payload["metrics"] = binary_metrics(prediction, target, args.threshold)

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.save:
        output = Path(args.save).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
