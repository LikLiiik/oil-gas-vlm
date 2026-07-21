"""AgentResult → 可视化 PNG / JSON / 属性 SEG-Y。

单切片: export_slice(result, image, geom, task, out_dir)
  → out_dir/<task>_<slice_kind>_<idx>.{png,json}
体级: export_volume_attribute(volume, per_slice_masks, task, out_dir)
  → out_dir/<task>_attribute.sgy
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .agents import AgentResult
from .io.geometry import SliceGeometry, pixel_to_data
from .io.segy import SegyVolume, write_attribute_segy
from .tasks import GeologicalTask


def _slice_label(geom: SliceGeometry) -> str:
    idx = geom.slice_index if geom.slice_index is not None else "na"
    return f"{geom.slice_kind}_{idx}"


def export_json(result: AgentResult, geom: SliceGeometry | None,
                task_name: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task_name,
        "geometry": geom.to_dict() if geom else None,
        "agent": result.to_dict(),
        "detections_data_coords": _convert_bboxes_to_data(result, geom),
    }
    slab = _slice_label(geom) if geom else "single"
    path = out_dir / f"{task_name}_{slab}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _convert_bboxes_to_data(result: AgentResult,
                             geom: SliceGeometry | None) -> list[dict]:
    """把 pixel bbox 转成数据坐标，方便后续可视化/落盘。"""
    if geom is None:
        return []
    out = []
    for r in result.results:
        bbox = r.get("bbox_pixel") or r.get("bbox")
        if not bbox or r.get("coordinate_system") != "pixel":
            continue
        try:
            data = pixel_to_data(bbox, geom)
        except Exception:
            continue
        out.append({
            "id": r.get("id"),
            "class_name": r.get("class_name"),
            "confidence": r.get("confidence"),
            "bbox_pixel": bbox,
            **data,
        })
    return out


def export_annotated_png(result: AgentResult, image: Image.Image,
                          geom: SliceGeometry | None,
                          task: GeologicalTask,
                          out_dir: str | Path) -> Path:
    """把检测 bbox 叠在原图上导出 PNG。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(np.asarray(image))
    for r in result.results:
        bbox = r.get("bbox_pixel") or r.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.8, edgecolor=task.overlay_color, facecolor="none",
        )
        ax.add_patch(rect)
        label = f"{r.get('class_name','?')} {r.get('confidence','')}"
        ax.text(x1, max(y1 - 4, 0), label, fontsize=7,
                color=task.overlay_color,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.7))
    if geom is not None:
        ax.set_title(
            f"{task.name} — {geom.slice_kind}"
            + (f"[{geom.slice_index}]" if geom.slice_index is not None else "")
        )
    ax.set_xlabel("pixel_x")
    ax.set_ylabel("pixel_y")
    plt.tight_layout()

    slab = _slice_label(geom) if geom else "single"
    path = out_dir / f"{task.name}_{slab}.png"
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def build_slice_mask(result: AgentResult, geom: SliceGeometry,
                     shape: tuple[int, int], task: GeologicalTask) -> np.ndarray:
    """把 bbox 检测转成 (n_y, n_x) mask，用于聚合到 3D 体。

    shape: (n_samples, n_traces) — 与原始数据 2D 切片同 shape，不是像素。
    """
    ny, nx = shape
    mask = np.full(shape, task.attribute_default, dtype=np.float32)
    for r in result.results:
        bbox = r.get("bbox_pixel") or r.get("bbox")
        if not bbox:
            continue
        try:
            data = pixel_to_data(bbox, geom)
        except Exception:
            continue
        x_key = f"{geom.axis_x_name}_min"
        # 数据 x 范围 → trace index
        x_min = data.get(f"{geom.axis_x_name}_min")
        x_max = data.get(f"{geom.axis_x_name}_max")
        y_top = data.get(f"{geom.axis_y_name}_top")
        y_bot = data.get(f"{geom.axis_y_name}_bottom")
        if None in (x_min, x_max, y_top, y_bot):
            continue
        # 归一到 trace/sample index
        dx = (geom.x_max - geom.x_min) or 1
        dy = (geom.y_bottom - geom.y_top) or 1
        i_x1 = int(np.clip((x_min - geom.x_min) / dx * nx, 0, nx - 1))
        i_x2 = int(np.clip((x_max - geom.x_min) / dx * nx, 0, nx - 1))
        i_y1 = int(np.clip((y_top - geom.y_top) / dy * ny, 0, ny - 1))
        i_y2 = int(np.clip((y_bot - geom.y_top) / dy * ny, 0, ny - 1))
        conf = float(r.get("confidence", 1.0))
        mask[i_y1:i_y2 + 1, min(i_x1, i_x2):max(i_x1, i_x2) + 1] = max(
            conf, task.attribute_default)
    return mask


def export_volume_attribute(volume: SegyVolume,
                              per_slice_masks: dict[int, np.ndarray],
                              task: GeologicalTask,
                              out_dir: str | Path,
                              slice_axis: str = "inline") -> Path:
    """把逐切片的 mask 聚合成 3D 属性体并写 SEG-Y。

    per_slice_masks[il_idx] -> (n_samples, n_xl) （inline 切片方向），
    未提供的 inline 用 task.attribute_default 填充。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube = np.full_like(volume.cube, task.attribute_default, dtype=np.float32)
    for il_idx, m in per_slice_masks.items():
        if slice_axis != "inline":
            raise NotImplementedError("目前只支持 inline 方向聚合")
        # slice shape 是 (n_samples, n_xl)，cube[il] 是 (n_xl, n_samples)
        cube[il_idx, :, :] = m.T
    out_path = out_dir / f"{task.name}_attribute.sgy"
    write_attribute_segy(volume, cube, str(out_path))
    return out_path


def normalize_detection_format(
    detections_by_image: dict[str, list[dict]],
    manifest: dict,
    image_by_name: dict,
) -> dict[str, list[dict]]:
    """把 8 个下游模型的各异输出格式统一为 {bbox_norm, class_name, confidence}。

    各模型原始输出:
      - yolo_world: 已有 bbox_norm → 直通
      - seismic_domain_model: bbox_pixel → /image_size → bbox_norm
      - sam: bbox_pixel → /image_size → bbox_norm
      - horizon_tracker: points[{trace_idx, sample_idx}] → min/max → bbox_norm
      - facies_classifier: centroid_xy + area_pixels → 估计 bbox → bbox_norm
      - well_log_analyzer / traditional_code: depth区间 → 跳过（非空间）
      - attribute_extractor: 仅统计 → 跳过
    """
    views = (manifest.get("seismic") or {}).get("views") or {}

    def _view_for_image(img_name: str) -> dict | None:
        stem = img_name.replace("seismic_", "")
        for vn, vm in views.items():
            if vn == stem or f"seismic_{vn}" == img_name:
                return {**vm, "view_name": vn}
        return None

    normalized: dict[str, list[dict]] = {}

    for img_name, dets in detections_by_image.items():
        vm = _view_for_image(img_name)
        arr_shape = vm.get("array_shape") if vm else None
        # 图像像素尺寸
        im = image_by_name.get(img_name)
        img_w, img_h = (im.pil.size[0], im.pil.size[1]) if im else (1, 1)

        converted: list[dict] = []
        for d in dets:
            cname = (d.get("class_name")
                     or d.get("label")
                     or d.get("horizon_name")
                     or (f"cluster_{d.get('cluster_id')}" if d.get('cluster_id') is not None else None)
                     or "unknown")
            conf = float(d.get("confidence", 0.5))
            bn = None

            # 1) 已有 bbox_norm → 直通
            if d.get("bbox_norm"):
                bn = d["bbox_norm"]

            # 2) bbox_pixel → bbox_norm
            elif d.get("bbox_pixel") and img_w > 1 and img_h > 1:
                x1, y1, x2, y2 = d["bbox_pixel"]
                bn = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

            # 3) points (horizon_tracker) → bbox_norm (用图像像素, 保证round-trip一致)
            elif d.get("points"):
                pts = d["points"]
                traces = [p.get("trace_idx", 0) for p in pts]
                samples = [p.get("sample_idx", 0) for p in pts]
                bn = [
                    min(traces) / max(img_w, 1),
                    min(samples) / max(img_h, 1),
                    max(traces) / max(img_w, 1),
                    max(samples) / max(img_h, 1),
                ]

            # 4) centroid_xy + area_pixels (facies_classifier) → 估计 bbox
            elif d.get("centroid_xy"):
                cx, cy = d["centroid_xy"]
                area = d.get("area_pixels", 100)
                half_side = max(float(np.sqrt(float(area))) / 2, 2.0)
                bx1 = max(0, cx - half_side) / max(img_w, 1)
                by1 = max(0, cy - half_side) / max(img_h, 1)
                bx2 = min(img_w, cx + half_side) / max(img_w, 1)
                by2 = min(img_h, cy + half_side) / max(img_h, 1)
                bn = [bx1, by1, bx2, by2]

            if bn is None:
                continue

            converted.append({
                "class_name": cname,
                "confidence": conf,
                "bbox_norm": [float(v) for v in bn],
                "model": d.get("model", "unknown"),
                "source_image": img_name,
            })

        if converted:
            normalized[img_name] = converted

    return normalized


def aggregate_adapter_detections(
    detections_by_image: dict[str, list[dict]],
    manifest: dict,
    volume_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """把下游模型检测聚合成 {class_name: 3D 属性体}。

    输入已通过 normalize_detection_format() 归一化，每条都有 bbox_norm。
    """
    n_il, n_xl, n_samples = volume_shape
    views = (manifest.get("seismic") or {}).get("views") or {}
    # image_name → view_meta：通过 model_image_path 的文件名匹配
    view_by_image: dict[str, dict] = {}
    for vn, vm in views.items():
        model_img = vm.get("model_image_path")
        if not model_img:
            continue
        # 允许 image_name 是 "seismic_inline" 也允许是 model_img 本身
        stem = Path(model_img).stem  # e.g. "inline_model"
        view_by_image[stem] = {"view_name": vn, **vm}
        view_by_image[f"seismic_{vn}"] = {"view_name": vn, **vm}
        view_by_image[vn] = {"view_name": vn, **vm}

    per_class: dict[str, np.ndarray] = {}

    for image_name, dets in detections_by_image.items():
        vm = view_by_image.get(image_name)
        if vm is None:
            continue
        vn = vm["view_name"]
        src_idx = vm.get("source_indices") or {}
        array_shape = vm.get("array_shape") or []
        if len(array_shape) != 2:
            continue
        arr_h, arr_w = array_shape  # (y_axis, x_axis) 从 axis_labels 语义

        for d in dets:
            if d.get("in_roi") is False:
                continue
            bn = d.get("bbox_norm")
            if not bn:
                continue
            conf = float(d.get("confidence", 1.0))
            cname = d.get("class_name") or "unknown"
            # 归一化 bbox → 数组下标（跟 array_shape 对齐）
            ax1 = int(np.clip(bn[0] * arr_w, 0, arr_w - 1))
            ay1 = int(np.clip(bn[1] * arr_h, 0, arr_h - 1))
            ax2 = int(np.clip(bn[2] * arr_w, 0, arr_w - 1))
            ay2 = int(np.clip(bn[3] * arr_h, 0, arr_h - 1))
            if cname not in per_class:
                per_class[cname] = np.zeros(volume_shape, dtype=np.float32)
            cube = per_class[cname]

            if vn == "inline":
                # array is (n_xl, n_samples)，切片位置 = source_indices.inline_index
                il = int(src_idx.get("inline_index", 0))
                if 0 <= il < n_il:
                    xl_lo, xl_hi = sorted((ax1, ax2))
                    s_lo, s_hi = sorted((ay1, ay2))
                    xl_lo = min(xl_lo, n_xl - 1); xl_hi = min(xl_hi, n_xl - 1)
                    s_lo = min(s_lo, n_samples - 1); s_hi = min(s_hi, n_samples - 1)
                    cube[il, xl_lo:xl_hi + 1, s_lo:s_hi + 1] = np.maximum(
                        cube[il, xl_lo:xl_hi + 1, s_lo:s_hi + 1], conf)
            elif vn == "crossline":
                # array is (n_il, n_samples)
                xl = int(src_idx.get("crossline_index", 0))
                if 0 <= xl < n_xl:
                    il_lo, il_hi = sorted((ax1, ax2))
                    s_lo, s_hi = sorted((ay1, ay2))
                    il_lo = min(il_lo, n_il - 1); il_hi = min(il_hi, n_il - 1)
                    s_lo = min(s_lo, n_samples - 1); s_hi = min(s_hi, n_samples - 1)
                    cube[il_lo:il_hi + 1, xl, s_lo:s_hi + 1] = np.maximum(
                        cube[il_lo:il_hi + 1, xl, s_lo:s_hi + 1], conf)
            elif vn == "slice":
                # 时间切片：array is (n_il, n_xl)
                s = int(src_idx.get("sample_index", 0))
                if 0 <= s < n_samples:
                    il_lo, il_hi = sorted((ay1, ay2))
                    xl_lo, xl_hi = sorted((ax1, ax2))
                    il_lo = min(il_lo, n_il - 1); il_hi = min(il_hi, n_il - 1)
                    xl_lo = min(xl_lo, n_xl - 1); xl_hi = min(xl_hi, n_xl - 1)
                    cube[il_lo:il_hi + 1, xl_lo:xl_hi + 1, s] = np.maximum(
                        cube[il_lo:il_hi + 1, xl_lo:xl_hi + 1, s], conf)
            # local_patch 不聚合到体（是局部放大图，位置模糊）

    return per_class


def summary_report(results: dict, out_dir: str | Path) -> Path:
    """把整个任务运行的元信息汇总成一份 report.json，方便后续检索。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
