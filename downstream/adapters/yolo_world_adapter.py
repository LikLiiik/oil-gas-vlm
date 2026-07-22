"""YOLO-World downstream adapter.

The adapter keeps the interface stable between the VLM analysis module and the
specialized detection backend. It accepts the prompt output produced by the
upstream VLM and turns it into YOLO-World text prompts, normalized categories,
and a fixed JSON detection result.
"""

from __future__ import annotations

import hashlib
import html
import json
import struct
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from schemas.yolo_world_schema import YOLO_WORLD_OUTPUT_SCHEMA, validate_detection_output

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


def _coerce_pair(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2 and _is_number(value[0]) and _is_number(value[1]):
        return [float(value[0]), float(value[1])]
    return None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def infer_image_size(image_path: str | None) -> tuple[int, int] | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.exists():
        return None

    with path.open("rb") as fp:
        header = fp.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header[:2] == b"\xff\xd8":
            fp.seek(2)
            while True:
                byte = fp.read(1)
                if not byte:
                    break
                if byte != b"\xff":
                    continue
                marker = fp.read(1)
                while marker == b"\xff":
                    marker = fp.read(1)
                if not marker:
                    break
                marker_code = marker[0]
                if marker_code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB}:
                    length = struct.unpack(">H", fp.read(2))[0]
                    _precision = fp.read(1)
                    height = struct.unpack(">H", fp.read(2))[0]
                    width = struct.unpack(">H", fp.read(2))[0]
                    return int(width), int(height)
                length_bytes = fp.read(2)
                if len(length_bytes) != 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                fp.seek(length - 2, 1)
    return None


def _axis_range(coord: dict[str, Any], keys: tuple[str, ...]) -> list[float] | None:
    for key in keys:
        value = coord.get(key)
        pair = _coerce_pair(value)
        if pair is not None:
            return pair
    return None


def _map_value(
    value: float,
    source_min: float,
    source_max: float,
    target_size: int,
    direction: str = "increasing",
) -> int:
    if source_max == source_min:
        return 0
    ratio = (value - source_min) / (source_max - source_min)
    ratio = max(0.0, min(1.0, ratio))
    if direction in {"decreasing", "bottom_to_top", "right_to_left"}:
        ratio = 1.0 - ratio
    return int(round(ratio * max(target_size - 1, 0)))


def _make_roi_xyxy(
    category: dict[str, Any],
    coordinate_system: dict[str, Any],
    image_size: dict[str, int],
) -> list[int] | None:
    x_range = _axis_range(
        coordinate_system,
        ("x_range", "cdp_range", "inline_range", "crossline_range", "expected_x_range"),
    )
    y_range = _axis_range(
        coordinate_system,
        ("y_range", "time_range_ms", "depth_range_m", "expected_y_range"),
    )
    expected_x = _axis_range(
        category,
        ("expected_cdp_range", "expected_inline_range", "expected_crossline_range", "expected_x_range"),
    )
    expected_y = _axis_range(
        category,
        ("expected_time_range_ms", "expected_depth_range", "expected_y_range"),
    )
    if not x_range or not y_range or not expected_x or not expected_y:
        return None

    width = int(image_size["width"])
    height = int(image_size["height"])
    x_direction = coordinate_system.get("x_direction", "increasing")
    y_direction = coordinate_system.get("y_direction", "top_to_bottom")

    x1 = _map_value(min(expected_x), x_range[0], x_range[1], width, x_direction)
    x2 = _map_value(max(expected_x), x_range[0], x_range[1], width, x_direction)
    y1 = _map_value(min(expected_y), y_range[0], y_range[1], height, y_direction)
    y2 = _map_value(max(expected_y), y_range[0], y_range[1], height, y_direction)
    x_lo, x_hi = sorted([x1, x2])
    y_lo, y_hi = sorted([y1, y2])
    return [x_lo, y_lo, x_hi, y_hi]


def _normalize_category(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        raw = {"class_name": item}
    elif isinstance(item, dict):
        raw = item
    else:
        raw = {}

    class_name = str(
        _first_non_empty(
            raw.get("class_name"),
            raw.get("label"),
            raw.get("name"),
            raw.get("class"),
            f"class_{index}",
        )
    ).strip()
    description = str(raw.get("description", "")).strip()
    max_detections = _coerce_int(raw.get("max_detections", 1), 1)
    if max_detections < 1:
        max_detections = 1
    confidence_threshold = _coerce_float(raw.get("confidence_threshold", 0.25), 0.25)
    if confidence_threshold < 0:
        confidence_threshold = 0.0
    if confidence_threshold > 1:
        confidence_threshold = 1.0

    normalized = {
        "class_name": class_name,
        "description": description,
        "text_prompt": class_name,
        "confidence_threshold": confidence_threshold,
        "max_detections": max_detections,
        "expected_cdp_range": _coerce_pair(
            _first_non_empty(
                raw.get("expected_cdp_range"),
                raw.get("expected_inline_range"),
                raw.get("expected_crossline_range"),
                raw.get("expected_x_range"),
            )
        ),
        "expected_time_range_ms": _coerce_pair(
            _first_non_empty(
                raw.get("expected_time_range_ms"),
                raw.get("expected_depth_range"),
                raw.get("expected_y_range"),
            )
        ),
        "source_index": index,
    }
    if description:
        normalized["description"] = description
    if "evidence" in raw:
        normalized["evidence"] = raw["evidence"]
    return normalized


def _extract_request_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("samples"), list):
        return [sample for sample in payload["samples"] if isinstance(sample, dict)]
    return [payload]


def _extract_yolo_config(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    yolo_section = {}
    downstream = sample.get("downstream_prompts")
    if isinstance(downstream, dict):
        maybe = downstream.get("yolo_world")
        if isinstance(maybe, dict):
            yolo_section = maybe

    raw_categories = None
    prompt_source = "direct_request"
    if yolo_section:
        raw_categories = yolo_section.get("categories")
        prompt_source = "vlm_downstream_prompts"

    if raw_categories is None:
        raw_categories = sample.get("categories")
    if raw_categories is None:
        raw_categories = sample.get("classes")
    if raw_categories is None:
        raw_categories = []

    categories = [_normalize_category(item, idx) for idx, item in enumerate(raw_categories)]
    if not categories:
        raise ValueError(
            "No YOLO-World classes were found. Expected either sample.classes or "
            "sample.downstream_prompts.yolo_world.categories."
        )

    coordinate_system = {}
    if isinstance(sample.get("coordinate_system"), dict):
        coordinate_system = sample["coordinate_system"]
    elif isinstance(sample.get("image_coordinate_system"), dict):
        coordinate_system = sample["image_coordinate_system"]

    return categories, coordinate_system, prompt_source


def _sample_image_size(sample: dict[str, Any]) -> tuple[int, int]:
    image_size = sample.get("image_size")
    if isinstance(image_size, dict):
        width = image_size.get("width")
        height = image_size.get("height")
        if _is_number(width) and _is_number(height):
            return int(width), int(height)

    width = sample.get("width")
    height = sample.get("height")
    if _is_number(width) and _is_number(height):
        return int(width), int(height)

    inferred = infer_image_size(sample.get("image_path"))
    if inferred is not None:
        return inferred
    return DEFAULT_IMAGE_SIZE["width"], DEFAULT_IMAGE_SIZE["height"]


def _deterministic_boxes(
    sample_id: str,
    image_path: str,
    category: dict[str, Any],
    image_size: dict[str, int],
    coordinate_system: dict[str, Any],
) -> list[dict[str, Any]]:
    width = int(image_size["width"])
    height = int(image_size["height"])
    roi = _make_roi_xyxy(category, coordinate_system, image_size)
    if roi is None:
        roi = [0, 0, max(width - 1, 0), max(height - 1, 0)]

    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_w = max(roi_x2 - roi_x1, 1)
    roi_h = max(roi_y2 - roi_y1, 1)
    class_name = category["class_name"]
    seed = hashlib.sha1(f"{sample_id}|{image_path}|{class_name}".encode("utf-8")).digest()
    max_detections = category["max_detections"]
    count = 1 + seed[0] % max(1, min(max_detections, 3))
    threshold = category["confidence_threshold"]

    detections = []
    for idx in range(count):
        offset = idx * 4
        rx = seed[(1 + offset) % len(seed)] / 255.0
        ry = seed[(2 + offset) % len(seed)] / 255.0
        rw = seed[(3 + offset) % len(seed)] / 255.0
        rh = seed[(4 + offset) % len(seed)] / 255.0
        box_w = max(16, int(roi_w * (0.22 + 0.26 * rw)))
        box_h = max(16, int(roi_h * (0.22 + 0.34 * rh)))
        x1 = roi_x1 + int(max(roi_w - box_w, 0) * rx)
        y1 = roi_y1 + int(max(roi_h - box_h, 0) * ry)
        x2 = min(width - 1, x1 + box_w)
        y2 = min(height - 1, y1 + box_h)
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        score = threshold + 0.15 + (seed[(5 + offset) % len(seed)] / 255.0) * (1.0 - threshold - 0.15)
        detections.append(
            {
                "class_name": class_name,
                "score": round(min(score, 0.99), 4),
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "class_index": category["source_index"],
                "roi_xyxy": roi,
            }
        )
    return detections


def _center_in_roi(box: list[int], roi: list[int] | None) -> bool:
    if roi is None:
        return True
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]


def _run_mock_backend(
    sample: dict[str, Any],
    categories: list[dict[str, Any]],
    image_size: dict[str, int],
    coordinate_system: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    detections: list[dict[str, Any]] = []
    warnings: list[str] = []
    sample_id = str(sample.get("sample_id", Path(str(sample.get("image_path", "sample"))).stem))
    image_path = str(sample.get("image_path", ""))

    for category in categories:
        category_detections = _deterministic_boxes(sample_id, image_path, category, image_size, coordinate_system)
        roi = category_detections[0]["roi_xyxy"] if category_detections else None
        filtered = [det for det in category_detections if _center_in_roi(det["bbox_xyxy"], roi)]
        if len(filtered) < len(category_detections):
            warnings.append(f"{category['class_name']}: some mock boxes were filtered by ROI.")
        detections.extend(filtered[: category["max_detections"]])

    detections.sort(key=lambda item: item["score"], reverse=True)
    return detections, warnings


def _run_ultralytics_backend(
    sample: dict[str, Any],
    categories: list[dict[str, Any]],
    image_size: dict[str, int],
    coordinate_system: dict[str, Any],
    model_path: str | None,
    device: str,
    imgsz: int,
    conf_override: float | None = None,
    disable_roi_filter: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not model_path:
        raise ValueError(
            "Real YOLO-World backend requires --model-path, e.g. yolov8s-world.pt or a local weight path."
        )
    image_path = sample.get("image_path")
    if not image_path:
        raise ValueError("Real YOLO-World backend requires sample.image_path.")
    source_path = Path(str(image_path)).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {source_path}. "
            "For the demo input, run: python scripts/create_demo_seismic_image.py"
        )

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Ultralytics is not installed. Install it first, or use --backend mock."
        ) from exc

    model = YOLO(model_path)
    class_names = [category["class_name"] for category in categories]
    if hasattr(model, "set_classes"):
        model.set_classes(class_names)
    elif hasattr(model, "model") and hasattr(model.model, "set_classes"):
        model.model.set_classes(class_names)

    threshold_by_name = {
        category["class_name"]: (
            max(0.0, min(float(conf_override), 1.0))
            if conf_override is not None
            else category["confidence_threshold"]
        )
        for category in categories
    }
    min_conf = min(threshold_by_name.values())
    max_det = max(sum(category["max_detections"] for category in categories), 1)
    result = model.predict(
        source=str(source_path),
        conf=min_conf,
        device=device,
        imgsz=imgsz,
        verbose=False,
        max_det=max_det,
    )[0]

    names = result.names
    detections: list[dict[str, Any]] = []
    warnings: list[str] = []
    roi_by_name = {category["class_name"]: _make_roi_xyxy(category, coordinate_system, image_size) for category in categories}
    max_det_by_name = {category["class_name"]: category["max_detections"] for category in categories}
    per_class_counts: dict[str, int] = {}

    if getattr(result, "boxes", None) is None:
        return [], ["YOLO backend returned no boxes."]

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    raw_box_count = len(xyxy)
    unknown_label_count = 0
    low_conf_count = 0
    roi_filtered_count = 0
    max_det_filtered_count = 0
    for box, score, cls_idx in zip(xyxy, confs, classes):
        label = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else str(names[int(cls_idx)])
        if label not in threshold_by_name:
            unknown_label_count += 1
            continue
        if float(score) < threshold_by_name[label]:
            low_conf_count += 1
            continue
        roi = roi_by_name.get(label)
        if (
            not disable_roi_filter
            and roi is not None
            and not _center_in_roi([int(box[0]), int(box[1]), int(box[2]), int(box[3])], roi)
        ):
            roi_filtered_count += 1
            continue
        count = per_class_counts.get(label, 0)
        if count >= max_det_by_name[label]:
            max_det_filtered_count += 1
            continue
        per_class_counts[label] = count + 1
        detections.append(
            {
                "class_name": label,
                "score": round(float(score), 4),
                "bbox_xyxy": [int(round(v)) for v in box],
                "class_index": int(cls_idx),
                "roi_xyxy": roi,
            }
        )

    detections.sort(key=lambda item: item["score"], reverse=True)
    warnings.append(
        "YOLO raw boxes: "
        f"{raw_box_count}, kept: {len(detections)}, "
        f"unknown_label: {unknown_label_count}, low_conf: {low_conf_count}, "
        f"roi_filtered: {roi_filtered_count}, max_det_filtered: {max_det_filtered_count}, "
        f"conf_used: {min_conf}"
    )
    return detections, warnings


def _render_overlay_svg(
    sample: dict[str, Any],
    detections: list[dict[str, Any]],
    output_path: Path,
    image_size: dict[str, int],
    categories: list[dict[str, Any]],
) -> Path:
    width = int(image_size["width"])
    height = int(image_size["height"])
    title = escape(str(sample.get("sample_id", "sample")))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="bg" patternUnits="userSpaceOnUse" width="40" height="18">',
        '<rect width="40" height="18" fill="#20242a"/>',
        '<path d="M0 9 C10 0 30 18 40 9" stroke="#8fa7b5" stroke-width="1.2" fill="none" opacity="0.75"/>',
        "</pattern>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>',
        f'<text x="12" y="24" fill="#ffffff" font-size="16" font-family="Arial">{title}</text>',
    ]
    if categories:
        prompt_text = ", ".join(escape(item["class_name"]) for item in categories[:5])
        lines.append(
            f'<text x="12" y="44" fill="#d6e4ff" font-size="12" font-family="Arial">classes: {prompt_text}</text>'
        )
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        label = escape(str(detection["class_name"]))
        score = detection["score"]
        lines.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="#ffcc0033" stroke="#ffcc00" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x1}" y="{max(16, y1 - 6)}" fill="#ffec99" font-size="13" font-family="Arial">{label} {score}</text>'
        )
    if not detections:
        lines.append(
            '<text x="12" y="62" fill="#ffec99" font-size="14" font-family="Arial">No detections.</text>'
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def process_payload(
    payload: dict[str, Any],
    backend: str = "mock",
    model_path: str | None = None,
    device: str = "cpu",
    imgsz: int = 640,
    output_dir: Path | None = None,
    write_overlays: bool = True,
    conf_override: float | None = None,
    disable_roi_filter: bool = False,
) -> dict[str, Any]:
    records = []
    warnings: list[str] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays" if output_dir is not None else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(_extract_request_records(payload)):
        sample_id = str(
            _first_non_empty(sample.get("sample_id"), sample.get("id"), Path(str(sample.get("image_path", f"sample_{index}"))).stem)
        )
        image_path = str(sample.get("image_path", ""))
        image_width, image_height = _sample_image_size(sample)
        image_size = {"width": image_width, "height": image_height}
        categories, coordinate_system, prompt_source = _extract_yolo_config(sample)

        if backend == "mock":
            detections, backend_warnings = _run_mock_backend(sample, categories, image_size, coordinate_system)
        elif backend == "ultralytics-yolo-world":
            detections, backend_warnings = _run_ultralytics_backend(
                sample,
                categories,
                image_size,
                coordinate_system,
                model_path=model_path,
                device=device,
                imgsz=imgsz,
                conf_override=conf_override,
                disable_roi_filter=disable_roi_filter,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        warnings.extend(backend_warnings)
        overlay_svg = None
        if write_overlays and overlay_dir is not None:
            overlay_path = overlay_dir / f"{sample_id}_overlay.svg"
            overlay_svg = str(_render_overlay_svg(sample, detections, overlay_path, image_size, categories))

        records.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "image_size": image_size,
                "prompt_source": prompt_source,
                "coordinate_system": coordinate_system or None,
                "class_prompts": categories,
                "detections": detections,
                "overlay_svg": overlay_svg,
                "warnings": backend_warnings,
            }
        )

    result = {
        "schema_version": "oil-gas.yolo-world-detection.v1",
        "backend": {
            "name": backend,
            "model_path": model_path,
            "device": device,
            "imgsz": imgsz,
        },
        "records": records,
        "warnings": warnings,
    }
    ok, errors = validate_detection_output(result)
    if not ok:
        result["validation_errors"] = errors
    return result
