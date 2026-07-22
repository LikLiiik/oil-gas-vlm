"""Evaluate YOLO-World adapter outputs.

This is a lightweight smoke-check, not a labeled benchmark.
It measures whether the detected boxes:

1. cover the prompted classes
2. fall inside the prompted ROI when ROI metadata is available
3. carry reasonable confidence scores
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="YOLO-World output JSON.")
    parser.add_argument("--save", default=None, help="Optional path to save the summary JSON.")
    return parser.parse_args()


def _center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _in_roi(center: tuple[float, float], roi: list[int] | None) -> bool:
    if not roi:
        return True
    x, y = center
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    payload = _load(Path(args.input).expanduser().resolve())

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise SystemExit("Invalid output: records must be a list.")

    sample_summaries = []
    total_prompt_classes = 0
    total_detected_prompt_classes = 0
    total_detections = 0
    total_roi_hits = 0
    all_scores: list[float] = []

    for record in records:
        class_prompts = record.get("class_prompts", []) or []
        detections = record.get("detections", []) or []
        total_prompt_classes += len(class_prompts)
        total_detections += len(detections)

        prompt_names = [item.get("class_name") for item in class_prompts if isinstance(item, dict)]
        detected_names = {det.get("class_name") for det in detections if isinstance(det, dict)}
        detected_prompt_classes = [name for name in prompt_names if name in detected_names]
        total_detected_prompt_classes += len(detected_prompt_classes)

        roi_hits = 0
        valid_boxes = 0
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("bbox_xyxy")
            if not (isinstance(box, list) and len(box) == 4):
                continue
            valid_boxes += 1
            score = det.get("score")
            if isinstance(score, (int, float)):
                all_scores.append(float(score))
            if _in_roi(_center([int(v) for v in box]), det.get("roi_xyxy")):
                roi_hits += 1
        total_roi_hits += roi_hits

        sample_summaries.append(
            {
                "sample_id": record.get("sample_id"),
                "prompt_classes": len(class_prompts),
                "detections": len(detections),
                "covered_prompt_classes": len(detected_prompt_classes),
                "prompt_coverage": round(len(detected_prompt_classes) / max(len(prompt_names), 1), 4),
                "roi_hit_rate": round(roi_hits / max(valid_boxes, 1), 4),
                "avg_score": round(statistics.fmean([float(det["score"]) for det in detections if isinstance(det, dict) and isinstance(det.get("score"), (int, float))]), 4)
                if detections
                else None,
                "detected_classes": sorted(detected_names),
            }
        )

    summary = {
        "records": len(records),
        "total_prompt_classes": total_prompt_classes,
        "total_detected_prompt_classes": total_detected_prompt_classes,
        "class_coverage": round(total_detected_prompt_classes / max(total_prompt_classes, 1), 4),
        "total_detections": total_detections,
        "roi_hit_rate": round(total_roi_hits / max(total_detections, 1), 4),
        "mean_score": round(statistics.fmean(all_scores), 4) if all_scores else None,
        "samples": sample_summaries,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.save:
        save_path = Path(args.save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
