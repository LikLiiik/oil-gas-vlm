"""下游模型共享工具：图像→数组转换、深度区间解析等。"""
from __future__ import annotations

import numpy as np


def image_to_array(image, context: dict | None = None) -> np.ndarray | None:
    """从 context["array"] 或 PIL image 获取 2D 浮点数组。

    有 context["array"] → 直接返回；否则从 PIL 灰度图提取并归一化到 [-1, 1]。
    所有下游模型统一使用此函数。
    """
    if context and "array" in context:
        return np.asarray(context["array"], dtype=np.float32)
    if image is not None:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        mu, sigma = arr.mean(), arr.std() or 1.0
        arr = np.clip(arr, mu - 2 * sigma, mu + 2 * sigma)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 2 - 1
        return arr
    return None


def normalize_depth_ranges(raw) -> list[tuple[float, float]]:
    """统一深度区间格式为 [(top, bot), ...] 数值对。

    支持:
      [{top_m, bottom_m}], [[top, bot]], [top, bot], 字符串值,
      [{top, bottom}] (无 _m 后缀)
    """
    if not raw:
        return []

    def _num(x) -> float | None:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    # 扁平 [top, bot]: 整个列表都是标量数字或字符串
    if all(not isinstance(x, (dict, list, tuple)) for x in raw):
        if len(raw) >= 2:
            t, b = _num(raw[0]), _num(raw[1])
            if t is not None and b is not None:
                return [(t, b)]
        return []

    out: list[tuple[float, float]] = []
    for r in raw:
        if isinstance(r, dict):
            t = _num(r.get("top_m", r.get("top")))
            b = _num(r.get("bottom_m", r.get("bottom")))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            t, b = _num(r[0]), _num(r[1])
        else:
            continue
        if t is not None and b is not None:
            out.append((t, b))
    return out


def extract_depth_ranges(rule: dict) -> list[tuple[float, float]]:
    """从 rule dict 里提取深度区间，兼容多种字段命名。"""
    for k in ("expected_depth_ranges", "expected_range", "target_range",
              "depth_range", "depth_ranges"):
        v = rule.get(k)
        if v:
            return normalize_depth_ranges(v)
    return []
