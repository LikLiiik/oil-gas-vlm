"""Run downstream detection and segmentation in one command.

Pipeline:

VLM output JSON / direct YOLO request
  -> YOLO-World adapter
  -> SAM-family adapter
  -> summary JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.sam_adapter import process_payload as process_sam_payload
from adapters.sam_adapter import write_output as write_sam_output
from adapters.yolo_world_adapter import process_payload as process_yolo_payload
from adapters.yolo_world_adapter import read_json
from adapters.yolo_world_adapter import write_output as write_yolo_output
from schemas.sam_schema import validate_sam_output
from schemas.yolo_world_schema import validate_detection_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="VLM output JSON or direct YOLO request.")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory.")
    parser.add_argument(
        "--yolo-backend",
        default="mock",
        choices=["mock", "ultralytics-yolo-world"],
        help="YOLO backend.",
    )
    parser.add_argument(
        "--sam-backend",
        default="mock",
        choices=["mock", "sam3-placeholder"],
        help="SAM-family backend. Only mock is implemented now.",
    )
    parser.add_argument("--yolo-model-path", default=None, help="YOLO-World weight path.")
    parser.add_argument("--sam-model-path", default=None, help="Future SAM/SAM3 checkpoint path.")
    parser.add_argument("--device", default="cpu", help="Device for real backends, e.g. cuda:0.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--write-overlays", action="store_true", help="Write SVG overlays for YOLO and SAM.")
    parser.add_argument("--validate", action="store_true", help="Validate intermediate outputs.")
    parser.add_argument("--conf-override", type=float, default=None, help="Debug option for YOLO confidence.")
    parser.add_argument("--disable-roi-filter", action="store_true", help="Debug option for YOLO ROI filtering.")
    parser.add_argument("--skip-sam", action="store_true", help="Run only YOLO detection.")
    return parser.parse_args()


def _write_summary(
    output_dir: Path,
    yolo_path: Path,
    sam_path: Path | None,
    yolo_result: dict,
    sam_result: dict | None,
    validation: dict,
) -> Path:
    yolo_records = yolo_result.get("records", [])
    yolo_detections = sum(len(record.get("detections", []) or []) for record in yolo_records)
    sam_records = sam_result.get("records", []) if sam_result else []
    sam_masks = sum(len(record.get("masks", []) or []) for record in sam_records)

    summary = {
        "schema_version": "oil-gas.downstream-pipeline.v1",
        "outputs": {
            "yolo_detections": str(yolo_path),
            "sam_masks": str(sam_path) if sam_path else None,
        },
        "counts": {
            "samples": len(yolo_records),
            "detections": yolo_detections,
            "masks": sam_masks,
        },
        "backends": {
            "yolo": yolo_result.get("backend"),
            "sam": sam_result.get("backend") if sam_result else None,
        },
        "validation": validation,
        "warnings": {
            "yolo": yolo_result.get("warnings", []),
            "sam": sam_result.get("warnings", []) if sam_result else [],
        },
    }
    path = output_dir / "pipeline_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    yolo_dir = output_dir / "yolo"
    sam_dir = output_dir / "sam"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = read_json(input_path)
    yolo_result = process_yolo_payload(
        payload,
        backend=args.yolo_backend,
        model_path=args.yolo_model_path,
        device=args.device,
        imgsz=args.imgsz,
        output_dir=yolo_dir,
        write_overlays=args.write_overlays,
        conf_override=args.conf_override,
        disable_roi_filter=args.disable_roi_filter,
    )
    yolo_path = write_yolo_output(yolo_result, yolo_dir, "yolo_world_detections.json")

    validation = {}
    if args.validate:
        yolo_ok, yolo_errors = validate_detection_output(yolo_result)
        validation["yolo"] = {"ok": yolo_ok, "errors": yolo_errors}

    sam_result = None
    sam_path = None
    if not args.skip_sam:
        sam_result = process_sam_payload(
            yolo_result,
            backend=args.sam_backend,
            model_path=args.sam_model_path,
            device=args.device,
            output_dir=sam_dir,
            write_overlays=args.write_overlays,
        )
        sam_path = write_sam_output(sam_result, sam_dir, "sam_masks.json")
        if args.validate:
            sam_ok, sam_errors = validate_sam_output(sam_result)
            validation["sam"] = {"ok": sam_ok, "errors": sam_errors}

    summary_path = _write_summary(output_dir, yolo_path, sam_path, yolo_result, sam_result, validation)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "yolo_output": str(yolo_path),
                "sam_output": str(sam_path) if sam_path else None,
                "detections": sum(len(record.get("detections", []) or []) for record in yolo_result.get("records", [])),
                "masks": sum(len(record.get("masks", []) or []) for record in sam_result.get("records", [])) if sam_result else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
