"""切片渲染：2D 数组 → PIL 图像 + SliceGeometry（供坐标反变换）。"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .geometry import SliceGeometry


def render_slice(array2d: np.ndarray,
                 x_min: float, x_max: float,
                 y_top: float, y_bottom: float,
                 axis_x_name: str = "CDP",
                 axis_y_name: str = "time_ms",
                 slice_kind: str = "inline",
                 slice_index: int | float | None = None,
                 cmap: str = "seismic", dpi: int = 100,
                 title: str | None = None,
                 clip_percentile: float = 98.0) -> tuple[Image.Image, SliceGeometry]:
    """
    渲染 2D 切片到 PIL 图像 + 返回 SliceGeometry。

    array2d.shape == (n_y, n_x)，如 (n_samples, n_traces)
    """
    if array2d.ndim != 2:
        raise ValueError(f"expect 2D array, got shape {array2d.shape}")

    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = np.percentile(np.abs(array2d), clip_percentile) or 1.0
    ax.imshow(array2d, cmap=cmap, aspect="auto",
              vmin=-vmax, vmax=vmax,
              extent=[x_min, x_max, y_bottom, y_top])
    ax.set_xlabel(axis_x_name)
    ax.set_ylabel(axis_y_name)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    geom = SliceGeometry(
        axis_x_name=axis_x_name, axis_y_name=axis_y_name,
        x_min=float(x_min), x_max=float(x_max),
        y_top=float(y_top), y_bottom=float(y_bottom),
        pixel_width=img.width, pixel_height=img.height,
        slice_kind=slice_kind, slice_index=slice_index,
    )
    return img, geom


def render_synthetic_seismic(seed: int = 42) -> tuple[Image.Image, SliceGeometry]:
    """test/data.py 的合成剖面，附带 geometry。方便冒烟测试。"""
    from test.data import generate_seismic  # 循环导入 OK：函数内 import
    img = generate_seismic(seed=seed)
    # test.data 的绘图用 extent=[1, 300, 2500, 0]
    geom = SliceGeometry(
        axis_x_name="CDP", axis_y_name="time_ms",
        x_min=1.0, x_max=300.0, y_top=0.0, y_bottom=2500.0,
        pixel_width=img.width, pixel_height=img.height,
        slice_kind="inline", slice_index=None,
    )
    return img, geom
