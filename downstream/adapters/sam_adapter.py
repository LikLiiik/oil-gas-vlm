"""SAM-family downstream segmentation adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from adapters.yolo_world_adapter import infer_image_size
from schemas.sam_schema import validate_sam_output


DEFAULT_IMAGE_SIZE = {"width": 512, "height": 384}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_output(payload: dict[str, Any], output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_bbox(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(_is_number(v) for v in value):
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return [x1, y1, x2, y2]
    return None


def _coerce_point(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value):
        return [int(round(float(value[0]))), int(round(float(value[1])))]
    return None


def _sample_image_size(sample: dict[str, Any]) -> dict[str, int]:
    size = sample.get("image_size")
    if isinstance(size, dict) and _is_number(size.get("width")) and _is_number(size.get("height")):
        return {"width": int(size["width"]), "height": int(size["height"])}
    if _is_number(sample.get("width")) and _is_number(sample.get("height")):
        return {"width": int(sample["width"]), "height": int(sample["height"])}
    inferred = infer_image_size(sample.get("image_path"))
    if inferred:
        return {"width": inferred[0], "height": inferred[1]}
    return dict(DEFAULT_IMAGE_SIZE)


def _records_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") == "oil-gas.yolo-world-detection.v1":
        return [record for record in payload.get("records", []) if isinstance(record, dict)]
    if isinstance(payload.get("records"), list):
        return [record for record in payload["records"] if isinstance(record, dict)]
    if isinstance(payload.get("samples"), list):
        return [sample for sample in payload["samples"] if isinstance(sample, dict)]
    return [payload]


def _prompts_from_record(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    direct = record.get("sam_prompts")
    if isinstance(direct, list):
        return [_normalize_prompt(item, idx, "direct_sam_prompts") for idx, item in enumerate(direct)], "direct_sam_prompts"

    downstream = record.get("downstream_prompts")
    if isinstance(downstream, dict):
        sam_section = downstream.get("sam") or downstream.get("sam3")
        if isinstance(sam_section, dict) and isinstance(sam_section.get("prompts"), list):
            return [
                _normalize_prompt(item, idx, "vlm_downstream_prompts")
                for idx, item in enumerate(sam_section["prompts"])
            ], "vlm_downstream_prompts"

    detections = record.get("detections")
    if isinstance(detections, list):
        prompts = []
        for idx, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            bbox = _coerce_bbox(detection.get("bbox_xyxy"))
            if bbox is None:
                continue
            prompts.append(
                {
                    "type": "bbox",
                    "label": str(detection.get("class_name") or detection.get("label") or f"det_{idx}"),
                    "bbox_xyxy": bbox,
                    "source": "yolo_detection",
                    "source_score": detection.get("score"),
                    "source_index": idx,
                }
            )
        return prompts, "yolo_detections"

    return [], "none"


def _normalize_prompt(item: Any, index: int, source: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"type": "unknown", "label": f"prompt_{index}", "source": source, "source_index": index}
    prompt_type = str(item.get("type", "bbox")).lower()
    label = str(item.get("label") or item.get("class_name") or item.get("name") or f"prompt_{index}")
    bbox = _coerce_bbox(item.get("bbox_xyxy") or item.get("bbox"))
    point = _coerce_point(item.get("point") or item.get("point_xy"))
    normalized = {
        "type": prompt_type,
        "label": label,
        "bbox_xyxy": bbox,
        "point_xy": point,
        "description": item.get("description"),
        "source": source,
        "source_index": index,
    }
    return normalized


def _bbox_to_polygon(bbox: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _point_to_polygon(point: list[int], image_size: dict[str, int], radius: int = 18) -> list[list[int]]:
    x, y = point
    width = image_size["width"]
    height = image_size["height"]
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width - 1, x + radius)
    y2 = min(height - 1, y + radius)
    return _bbox_to_polygon([x1, y1, x2, y2])


def _mock_masks(prompts: list[dict[str, Any]], image_size: dict[str, int]) -> list[dict[str, Any]]:
    masks = []
    for idx, prompt in enumerate(prompts):
        polygon = None
        bbox = prompt.get("bbox_xyxy")
        point = prompt.get("point_xy")
        if bbox:
            polygon = _bbox_to_polygon(bbox)
        elif point:
            polygon = _point_to_polygon(point, image_size)
        if polygon is None:
            continue
        masks.append(
            {
                "mask_id": f"M{idx + 1}",
                "label": prompt.get("label", f"mask_{idx}"),
                "score": 1.0,
                "source_prompt_type": prompt.get("type"),
                "polygon_xy": polygon,
                "bbox_xyxy": bbox,
                "source": "mock_sam_mask",
            }
        )
    return masks


def _render_overlay(record: dict[str, Any], masks: list[dict[str, Any]], output_path: Path, image_size: dict[str, int]) -> Path:
    width = image_size["width"]
    height = image_size["height"]
    title = escape(str(record.get("sample_id", "sample")))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#20242a"/>',
        f'<text x="12" y="24" fill="#fff" font-size="16" font-family="Arial">{title}</text>',
    ]
    for mask in masks:
        polygon = mask.get("polygon_xy", [])
        if not polygon:
            continue
        points = " ".join(f"{p[0]},{p[1]}" for p in polygon)
        label = escape(str(mask.get("label", "mask")))
        x = polygon[0][0]
        y = max(16, polygon[0][1] - 6)
        lines.append(f'<polygon points="{points}" fill="#2dd4bf44" stroke="#2dd4bf" stroke-width="3"/>')
        lines.append(f'<text x="{x}" y="{y}" fill="#ccfbf1" font-size="13" font-family="Arial">{label}</text>')
    if not masks:
        lines.append('<text x="12" y="52" fill="#ffec99" font-size="14" font-family="Arial">No masks.</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def process_payload(
    payload: dict[str, Any],
    backend: str = "mock",
    model_path: str | None = None,
    device: str = "cpu",
    output_dir: Path | None = None,
    write_overlays: bool = True,
) -> dict[str, Any]:
    if backend != "mock":
        raise NotImplementedError("Real SAM/SAM3 backend is reserved; use --backend mock for now.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays" if output_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    records = []
    warnings: list[str] = []
    for idx, record in enumerate(_records_from_payload(payload)):
        sample_id = str(record.get("sample_id") or record.get("id") or f"sample_{idx}")
        image_size = _sample_image_size(record)
        prompts, prompt_source = _prompts_from_record(record)
        if not prompts:
            warnings.append(f"{sample_id}: no SAM prompts found.")
        masks = _mock_masks(prompts, image_size)
        overlay_svg = None
        if write_overlays and overlay_dir:
            overlay_svg = str(_render_overlay(record, masks, overlay_dir / f"{sample_id}_sam_overlay.svg", image_size))
        records.append(
            {
                "sample_id": sample_id,
                "image_path": str(record.get("image_path", "")),
                "image_size": image_size,
                "prompt_source": prompt_source,
                "prompts": prompts,
                "masks": masks,
                "overlay_svg": overlay_svg,
                "warnings": [],
            }
        )

    result = {
        "schema_version": "oil-gas.sam-segmentation.v1",
        "backend": {"name": backend, "model_path": model_path, "device": device},
        "records": records,
        "warnings": warnings,
    }
    ok, errors = validate_sam_output(result)
    if not ok:
        result["validation_errors"] = errors
    return result
