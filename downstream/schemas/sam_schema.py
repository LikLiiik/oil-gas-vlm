"""JSON schema and lightweight validation for SAM-family mask outputs."""

from __future__ import annotations

from typing import Any


SAM_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "backend", "records"],
    "properties": {
        "schema_version": {"type": "string"},
        "backend": {"type": "object"},
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sample_id", "image_path", "image_size", "prompts", "masks"],
                "properties": {
                    "sample_id": {"type": "string"},
                    "image_path": {"type": "string"},
                    "image_size": {"type": "object"},
                    "prompt_source": {"type": "string"},
                    "prompts": {"type": "array"},
                    "masks": {"type": "array"},
                    "overlay_svg": {"type": ["string", "null"]},
                    "warnings": {"type": "array"},
                },
            },
        },
        "warnings": {"type": "array"},
    },
}


def validate_sam_output(data: dict[str, Any]) -> tuple[bool, list[str]]:
    try:  # pragma: no cover - optional dependency
        import jsonschema

        jsonschema.validate(data, SAM_OUTPUT_SCHEMA)
        return True, []
    except Exception:
        pass

    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["Output is not a dict."]
    for key in ("schema_version", "backend", "records"):
        if key not in data:
            errors.append(f"Missing required field: {key}")
    if not isinstance(data.get("records"), list):
        errors.append("records must be a list.")
        return False, errors
    for idx, record in enumerate(data.get("records", [])):
        if not isinstance(record, dict):
            errors.append(f"records[{idx}] must be a dict.")
            continue
        for key in ("sample_id", "image_path", "image_size", "prompts", "masks"):
            if key not in record:
                errors.append(f"records[{idx}] missing {key}.")
    return len(errors) == 0, errors
