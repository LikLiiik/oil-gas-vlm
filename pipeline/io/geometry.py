"""2D 切片坐标几何：像素 ↔ 数据（CDP/time 或 inline/depth）双向映射。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SliceGeometry:
    """一张 2D 切片的坐标信息。

    - 图像像素坐标：原点在左上，x 向右、y 向下
    - 数据坐标：axis_x 从 x_min 到 x_max（含 inline/CDP 反向都允许），
                axis_y 从 y_top 到 y_bottom（时间/深度是"往下增大"）
    """
    axis_x_name: str        # "CDP" | "inline" | "crossline" | "distance"
    axis_y_name: str        # "time_ms" | "depth_m"
    x_min: float
    x_max: float
    y_top: float            # 图像顶部对应的数据坐标（如 time=0 ms）
    y_bottom: float         # 图像底部对应的数据坐标（如 time=2500 ms）
    pixel_width: int
    pixel_height: int
    slice_kind: str = "inline"   # inline | crossline | time
    slice_index: int | float | None = None  # 沿垂直方向该切片的位置

    def to_dict(self) -> dict:
        return {
            "axis_x_name": self.axis_x_name, "axis_y_name": self.axis_y_name,
            "x_min": self.x_min, "x_max": self.x_max,
            "y_top": self.y_top, "y_bottom": self.y_bottom,
            "pixel_width": self.pixel_width, "pixel_height": self.pixel_height,
            "slice_kind": self.slice_kind, "slice_index": self.slice_index,
        }


def pixel_to_data(bbox_pixel: list[float], geom: SliceGeometry) -> dict:
    """像素 [x1, y1, x2, y2] → 数据坐标 dict。"""
    x1, y1, x2, y2 = bbox_pixel
    dx = geom.x_max - geom.x_min
    dy = geom.y_bottom - geom.y_top
    d_x1 = geom.x_min + x1 / geom.pixel_width * dx
    d_x2 = geom.x_min + x2 / geom.pixel_width * dx
    d_y1 = geom.y_top + y1 / geom.pixel_height * dy
    d_y2 = geom.y_top + y2 / geom.pixel_height * dy
    return {
        f"{geom.axis_x_name}_min": min(d_x1, d_x2),
        f"{geom.axis_x_name}_max": max(d_x1, d_x2),
        f"{geom.axis_y_name}_top": min(d_y1, d_y2),
        f"{geom.axis_y_name}_bottom": max(d_y1, d_y2),
    }


def data_to_pixel(x_data: float, y_data: float,
                  geom: SliceGeometry) -> tuple[float, float]:
    """数据坐标 → 像素坐标（提供给 VLM prompt 使用）。"""
    dx = geom.x_max - geom.x_min
    dy = geom.y_bottom - geom.y_top
    px = (x_data - geom.x_min) / dx * geom.pixel_width if dx else 0
    py = (y_data - geom.y_top) / dy * geom.pixel_height if dy else 0
    return px, py
