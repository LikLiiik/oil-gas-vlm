"""SeismicDomainDetector — 领域地震属性检测器。

基于相干体(coherence) + 结构张量(structural tensor)等经典地震属性做断层/裂缝检测。
同时保留预训练 DL 模型接口——当 finetuned checkpoint 可用时自动切换。

注册名: "seismic_domain_model"
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._shared import image_to_array as _get_array


# ── DL 模型路径（预训练 checkpoint，暂无 finetuned head；将来接）───────
_DOMAIN_PROJECT = Path(
    os.environ.get("OIL_GAS_MODEL_PROJECT",
                   "/data/yxjiang/oil-gas-multimodal-model")
)
_DEFAULT_CKPT = os.environ.get(
    "OIL_GAS_MODEL_CKPT",
    str(_DOMAIN_PROJECT / "checkpoints/pretrain_multi/best_stage1.pt"),
)


# ── 辅助函数 ──────────────────────────────────────────────────────

def _non_max_suppression(detections: list[dict],
                         iou_thr: float = 0.3) -> list[dict]:
    """对 bbox 列表做 NMS，按 confidence 降序保留。"""
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: -d["confidence"])
    keep = []
    for d in dets:
        suppressed = False
        for k in keep:
            bx1, by1, bx2, by2 = d["bbox_pixel"]
            kx1, ky1, kx2, ky2 = k["bbox_pixel"]
            xo = max(0, min(bx2, kx2) - max(bx1, kx1))
            yo = max(0, min(by2, ky2) - max(by1, ky1))
            inter = xo * yo
            area_d = (bx2 - bx1) * (by2 - by1)
            area_k = (kx2 - kx1) * (ky2 - ky1)
            iou = inter / (area_d + area_k - inter + 1e-8)
            if iou > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
    return keep


# ── 相干体计算 ──────────────────────────────────────────────────────

def coherence_map(arr: np.ndarray, win: int = 9) -> np.ndarray:
    """2D semblance-based coherence (trace-to-trace similarity)。

    arr: (n_samples, n_traces) float32
    win: 滑动窗口半宽

    返回 (n_samples, n_traces) coherence [0, 1]，低值→断层/不连续。
    """
    from scipy.ndimage import uniform_filter1d
    ns, nt = arr.shape
    # 沿 trace 方向平滑（抑制随机噪声）
    smoothed = uniform_filter1d(arr, size=3, axis=0, mode="reflect")
    coh = np.ones((ns, nt), dtype=np.float32)
    for t in range(nt - 1):
        # 两相邻道在局部窗口内的归一化互相关
        a = smoothed[:, t]
        b = smoothed[:, t + 1]
        if win >= ns:
            win_use = ns // 2
        else:
            win_use = win
        a2 = uniform_filter1d(a * a, size=win_use, mode="reflect")
        b2 = uniform_filter1d(b * b, size=win_use, mode="reflect")
        ab = uniform_filter1d(a * b, size=win_use, mode="reflect")
        denom = np.sqrt(a2 * b2) + 1e-8
        sim = ab / denom          # running correlation
        # 取两道之间更差的那个方向
        coh[:, t] = np.minimum(coh[:, t], np.clip(sim, 0, 1))
        coh[:, t + 1] = np.minimum(coh[:, t + 1], np.clip(sim, 0, 1))
    return coh


def gradient_fault_prob(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """垂直方向梯度 → 高斯平滑 → 寻找近垂直不连续面（断层）。

    只取 gy（垂直梯度），抑制 gx（水平梯度 = 层位边界）。
    断层 = 同相轴在纵向上突然终止 → 垂直梯度大。
    """
    from scipy.ndimage import gaussian_filter, sobel
    gy = sobel(arr, axis=0, mode="reflect")   # 垂直方向：检测同相轴终止
    gx = sobel(arr, axis=1, mode="reflect")   # 水平方向：检测层位边界（抑制）
    # 断层特征：垂直梯度 >> 水平梯度
    faultiness = np.abs(gy) - 0.5 * np.abs(gx)
    faultiness = np.clip(faultiness, 0, None)
    # 高斯平滑 → 连接断层的分段边缘
    faultiness = gaussian_filter(faultiness, sigma, mode="reflect")
    faultiness /= (faultiness.max() + 1e-8)
    # 高值 → 可能断层
    return faultiness


def local_variance(arr: np.ndarray, win: int = 15) -> np.ndarray:
    """局部方差——对 AGC 数据有效的断层属性。
    不连续面附近振幅模式变化剧烈，方差局部偏高。
    """
    from scipy.ndimage import uniform_filter
    mu = uniform_filter(arr, size=win, mode="reflect")
    mu2 = uniform_filter(arr * arr, size=win, mode="reflect")
    var = np.clip(mu2 - mu * mu, 0, None)
    var /= (var.max() + 1e-8)
    return var


def structure_tensor_edge(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """2D 梯度结构张量的边缘强度。

    arr: (n_samples, n_traces)
    返回 (n_samples, n_traces) edge [0, 1]，高值→断层/不连续边界。
    """
    from scipy.ndimage import gaussian_filter, sobel
    gx = sobel(arr, axis=1, mode="reflect")
    gy = sobel(arr, axis=0, mode="reflect")
    # 结构张量分量，高斯平滑
    Jxx = gaussian_filter(gx * gx, sigma, mode="reflect")
    Jyy = gaussian_filter(gy * gy, sigma, mode="reflect")
    Jxy = gaussian_filter(gx * gy, sigma, mode="reflect")
    # 特征值 λ1 ≥ λ2 ≥ 0
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
    lambda1 = (trace + disc) / 2
    lambda2 = (trace - disc) / 2
    # 边缘强度：λ1 - λ2（等效于 coherence 越低越可能是断层）
    edge = np.clip(lambda1 - lambda2, 0, None)
    edge /= (edge.max() + 1e-8)
    return 1 - edge  # 低值→断层，和 coherence 统一


# ── 下游模型 ────────────────────────────────────────────────────────

class SeismicDomainDetector:
    name = "seismic_domain_model"
    description = (
        "领域地震属性检测（相干体+结构张量）。适合断层/裂缝/不连续面检测。"
        "未来可接 finetuned 3D CNN 做高精度分割。"
    )
    required_fields = [
        "task (fault_detection|facies_classification)",
        "confidence_threshold",
        "min_region_area_pixels",
        "attribute (可选: coherence|structure_tensor|gradient|variance|both,"
        " 默认 both=gradient+structure_tensor，AGC数据推荐 gradient)",
    ]
    output_shape = (
        "list[{id, class_name, bbox_pixel:[x1,y1,x2,y2], confidence, "
        "area_pixels, attribute}]"
    )

    def __init__(self, ckpt_path: str | None = None):
        self.ckpt_path = ckpt_path or _DEFAULT_CKPT
        self._dl_loaded = False
        self._dl_model = None

    def detect(self, instruction: dict, image=None,
               context: dict | None = None) -> list[dict]:
        """在 2D 切片上计算相干/结构张量，阈值化后提取不连续区域作为断层候选。

        instruction 可含 regions_of_interest: [{bbox_norm: [x1,y1,x2,y2]}, ...]
        → 仅在 ROI 内精细检测（使用更低阈值）。VLM 先看图像圈 ROI，
        领域模型再在 ROI 内精细属性分析——混合工作流。"""
        task = instruction.get("task", "fault_detection")
        conf_thr = float(instruction.get("confidence_threshold", 0.3))
        min_area = int(instruction.get("min_region_area_pixels", 100))
        attr_name = instruction.get("attribute", "gradient")
        rois = instruction.get("regions_of_interest") or []

        # 获取 raw array
        arr = _get_array(image, context)
        if arr is None:
            return []

        # 归一化到 [-1, 1]
        vmax = np.percentile(np.abs(arr), 99) or 1.0
        arr_norm = np.clip(arr / vmax, -1, 1)
        ny, nx = arr_norm.shape

        # ── 有 ROI：仅在 ROI 内精细检测 ──
        if rois:
            all_results = []
            for roi in rois:
                bn = roi.get("bbox_norm") or roi.get("bbox_xyxy_norm")
                if not bn or len(bn) != 4:
                    continue
                # 归一化 → 像素
                x1 = int(np.clip(bn[0] * nx, 0, nx - 1))
                y1 = int(np.clip(bn[1] * ny, 0, ny - 1))
                x2 = int(np.clip(bn[2] * nx, 0, nx - 1))
                y2 = int(np.clip(bn[3] * ny, 0, ny - 1))
                if x2 - x1 < 5 or y2 - y1 < 10:
                    continue
                sub_arr = arr_norm[y1:y2 + 1, x1:x2 + 1]
                # ROI 内使用更敏感的阈值
                sub_results = self._detect_on_array(
                    sub_arr, task, conf_thr * 0.7, min_area // 2,
                    attr_name, offset_x=x1, offset_y=y1,
                )
                all_results.extend(sub_results)
            return _non_max_suppression(all_results)

        # ── 无 ROI：全局检测 ──
        return self._detect_on_array(
            arr_norm, task, conf_thr, min_area, attr_name,
        )

    def _detect_on_array(self, arr: np.ndarray, task: str,
                         conf_thr: float, min_area: int,
                         attr_name: str,
                         offset_x: int = 0, offset_y: int = 0) -> list[dict]:
        """在 arr 上做属性计算 + 阈值化 + 连通域提取。"""
        # ── 属性融合 ──
        prob = None
        if attr_name in ("coherence", "both", "coherence+gradient"):
            coh = coherence_map(arr, win=max(3, min(9, arr.shape[0] // 4)))
            fault_prob = 1 - np.clip(coh, 0, 1)
            prob = fault_prob if prob is None else np.minimum(prob, fault_prob)
        if attr_name in ("structure_tensor", "both"):
            st = structure_tensor_edge(arr)
            prob = st if prob is None else np.minimum(prob, st)
        if attr_name in ("gradient", "both"):
            gf = gradient_fault_prob(arr)
            prob = gf if prob is None else np.minimum(prob, gf)
        if attr_name == "variance":
            prob = local_variance(arr)

        if prob is None:
            return []

        # ── 自适应阈值化 ──
        # conf_thr 在 [0, 1) 时：取 prob 的上 conf_thr 分位作为绝对阈值
        # conf_thr >= 1 时：取 prob > top_{conf_thr} 百分位
        try:
            from scipy import ndimage
        except ImportError:
            return []
        if conf_thr <= 0:
            # 0 或负数 → 全返回（不做阈值过滤）
            binary = (prob >= 0).astype(np.uint8)
        elif conf_thr < 1:
            abs_thr = conf_thr
            binary = (prob >= abs_thr).astype(np.uint8)
        else:
            # conf_thr >= 1 → 作为百分比，例 95 → top 5%
            perc = max(1, min(99.5, float(conf_thr)))
            abs_thr = np.percentile(prob, perc)
            binary = (prob >= abs_thr).astype(np.uint8)
            if binary.sum() == 0:
                return []
        labeled, n_regions = ndimage.label(binary)
        class_prefix = {"fault_detection": "fault",
                        "facies_classification": "facies",
                        "fracture": "fracture"}.get(task, task)
        # 统计所有区域的高度/宽度
        regions = []
        for rid in range(1, n_regions + 1):
            ys, xs = np.where(labeled == rid)
            if len(ys) < min_area:
                continue
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            h = y2 - y1 + 1
            w = x2 - x1 + 1
            aspect = max(h, w) / (min(h, w) + 1)
            # 断层特征：高度>>宽度（垂直延伸的线性特征）
            if h > 3 * w or w > 3 * h:
                shape_score = 1.0
            elif h > 2 * w or w > 2 * h:
                shape_score = 0.8
            else:
                shape_score = 0.5  # 团块，不太像断层
            conf = float(prob[ys, xs].mean()) * shape_score
            regions.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "h": h, "w": w, "aspect": aspect,
                "area": int(len(ys)), "conf": conf,
                "rid": rid, "ys": ys, "xs": xs,
            })

        # 非极大值抑制：重叠的 bbox 只保留置信度最高的
        regions.sort(key=lambda r: -r["conf"])
        out = []
        for ri, r in enumerate(regions):
            suppressed = False
            for rj in regions[:ri]:
                x_overlap = min(r["x2"], rj["x2"]) - max(r["x1"], rj["x1"])
                y_overlap = min(r["y2"], rj["y2"]) - max(r["y1"], rj["y1"])
                if x_overlap > 0 and y_overlap > 0:
                    iou = x_overlap * y_overlap / (
                        (r["x2"]-r["x1"])*(r["y2"]-r["y1"]) +
                        (rj["x2"]-rj["x1"])*(rj["y2"]-rj["y1"]) -
                        x_overlap * y_overlap + 1e-8)
                    if iou > 0.3:
                        suppressed = True
                        break
            if not suppressed:
                out.append({
                    "id": f"seisattr_{class_prefix}_{r['rid']}",
                    "class_name": class_prefix,
                    "bbox_pixel": [float(r["x1"] + offset_x),
                                   float(r["y1"] + offset_y),
                                   float(r["x2"] + offset_x),
                                   float(r["y2"] + offset_y)],
                    "confidence": round(r["conf"], 3),
                    "area_pixels": r["area"],
                    "aspect_ratio": round(r["aspect"], 1),
                    "attribute": attr_name,
                    "model": self.name,
                })
        return out
