"""SEG-Y I/O + 坐标几何 + 切片渲染。"""
from .geometry import SliceGeometry, pixel_to_data, data_to_pixel
from .render import render_slice, render_synthetic_seismic
from .segy import (
    SegyVolume, read_segy, write_attribute_segy, write_attribute_segy_like,
    extract_inline_slice, extract_xline_slice, extract_time_slice,
    synthetic_volume,
)

__all__ = [
    "SegyVolume", "read_segy", "write_attribute_segy",
    "write_attribute_segy_like", "synthetic_volume",
    "extract_inline_slice", "extract_xline_slice", "extract_time_slice",
    "SliceGeometry", "pixel_to_data", "data_to_pixel",
    "render_slice", "render_synthetic_seismic",
]
