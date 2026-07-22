"""JSON schema and lightweight validation for the YOLO-World adapter."""

from __future__ import annotations

from typing import Any

YOLO_WORLD_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "backend", "records"],
    "properties": {
        "schema_version": {"type": "string"},
        "backend": {
            "type": "object",
            "required": ["name", "device"],
            "properties": {
                "name": {"type": "string"},
                "model_path": {"type": ["string", "null"]},
                "device": {"type": "string"},
                "imgsz": {"type": "integer"},
            },
        },
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sample_id", "image_path", "image_size", "class_prompts", "detections"],
                "properties": {
                    "sample_id": {"type": "string"},
                    "image_path": {"type": "string"},
                    "image_size": {
                        "type": "object",
                        "required": ["width", "height"],
                        "properties": {
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                        },
                    },
                    "prompt_source": {"type": "string"},
                    "coordinate_system": {"type": ["object", "null"]},
                    "class_prompts": {"type": "array"},
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["class_name", "score", "bbox_xyxy"],
                            "properties": {
                                "class_name": {"type": "string"},
                                "score": {"type": "number"},
                                "bbox_xyxy": {
                                    "type": "array",
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "items": {"type": "integer"},
                                },
                                "class_index": {"type": "integer"},
                                "roi_xyxy": {"type": ["array", "null"]},
                            },
                        },
                    },
                    "overlay_svg": {"type": ["string", "null"]},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "validation_errors": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_detection_output(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the output schema.

    Uses jsonschema if available; otherwise falls back to a small structural check.
    """

    try:  # pragma: no cover - optional dependency
        import jsonschema

        jsonschema.validate(data, YOLO_WORLD_OUTPUT_SCHEMA)
        return True, []
    except Exception:
        pass

    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["Output is not a dict."]
    for key in ("schema_version", "backend", "records"):
        if key not in data:
            errors.append(f"Missing required field: {key}")
    if not isinstance(data.get("backend"), dict):
        errors.append("backend must be an object.")
    if not isinstance(data.get("records"), list):
        errors.append("records must be an array.")
        return False, errors
    for idx, record in enumerate(data.get("records", [])):
        if not isinstance(record, dict):
            errors.append(f"records[{idx}] must be an object.")
            continue
        for key in ("sample_id", "image_path", "image_size", "class_prompts", "detections"):
            if key not in record:
                errors.append(f"records[{idx}] missing {key}.")
        image_size = record.get("image_size")
        if not isinstance(image_size, dict) or "width" not in image_size or "height" not in image_size:
            errors.append(f"records[{idx}].image_size must contain width/height.")
    return len(errors) == 0, errors
