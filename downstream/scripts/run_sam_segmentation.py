"""Run the SAM-family downstream segmentation adapter.

Current scope:

1. Consume YOLO-World detection JSON and turn detections into bbox prompts.
2. Consume direct SAM prompts from upstream VLM output.
3. Produce a stable mask JSON.

The real SAM/SAM3 backend is intentionally left as an adapter hook. The mock
backend is enough to test data contracts before real seismic slices are ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.sam_adapter import process_payload, read_json, write_output
from schemas.sam_schema import validate_sam_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="YOLO detection JSON, VLM output JSON, or direct SAM request.")
    parser.add_argument("--output-dir", required=True, help="Directory for mask JSON and overlays.")
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "sam3-placeholder"],
        help="Only mock is implemented now. sam3-placeholder reserves the future interface.",
    )
    parser.add_argument("--model-path", default=None, help="Future SAM/SAM3 checkpoint path.")
    parser.add_argument("--device", default="cpu", help="Future backend device, e.g. cuda:0.")
    parser.add_argument("--save-json", default="sam_masks.json")
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_json(Path(args.input).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()

    result = process_payload(
        payload,
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        output_dir=output_dir,
        write_overlays=args.write_overlays,
    )

    if args.validate:
        ok, errors = validate_sam_output(result)
        print(json.dumps({"validation": "passed" if ok else "failed", "errors": errors}, ensure_ascii=False, indent=2))

    output_path = write_output(result, output_dir, args.save_json)
    print(json.dumps({"output": str(output_path), "records": len(result.get("records", []))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
