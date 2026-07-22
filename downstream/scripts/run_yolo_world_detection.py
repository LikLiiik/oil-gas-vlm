"""Run the YOLO-World downstream detection adapter.

This script consumes either:

1. A direct request:
   {
     "image_path": "...",
     "classes": ["fault plane", "channel"]
   }

2. A VLM output record from the upstream analysis module:
   {
     "sample_id": "...",
     "image_path": "...",
     "coordinate_system": {...},
     "downstream_prompts": {
       "yolo_world": {
         "categories": [...]
       }
     }
   }

The output is a fixed JSON structure with normalized YOLO-World classes and
detections. A mock backend is available for interface testing when the real
model is not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.yolo_world_adapter import process_payload, read_json, write_output
from schemas.yolo_world_schema import validate_detection_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file with a direct request or VLM output.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the normalized detection JSON and overlay SVGs will be written.",
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "ultralytics-yolo-world"],
        help="Mock backend or real Ultralytics YOLO-World backend.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="YOLO-World weight path for the real backend, e.g. yolov8s-world.pt.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the real backend, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for the real backend.",
    )
    parser.add_argument(
        "--save-json",
        default="yolo_world_detections.json",
        help="Name of the output JSON file inside --output-dir.",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Also write SVG overlays for each sample.",
    )
    parser.add_argument(
        "--conf-override",
        type=float,
        default=None,
        help="Debug option: override all per-class confidence thresholds, e.g. 0.01.",
    )
    parser.add_argument(
        "--disable-roi-filter",
        action="store_true",
        help="Debug option: keep detections outside VLM expected ranges.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the output JSON against the local schema if jsonschema is installed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = read_json(input_path)

    result = process_payload(
        payload,
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        imgsz=args.imgsz,
        output_dir=output_dir,
        write_overlays=args.write_overlays,
        conf_override=args.conf_override,
        disable_roi_filter=args.disable_roi_filter,
    )

    if args.validate:
        ok, errors = validate_detection_output(result)
        if not ok:
            print(json.dumps({"validation": "failed", "errors": errors}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"validation": "passed"}, ensure_ascii=False, indent=2))

    output_path = write_output(result, output_dir, args.save_json)
    print(json.dumps({"output": str(output_path), "records": len(result.get("records", []))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
