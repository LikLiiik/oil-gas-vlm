"""SEG-Y 读取 + 切片提取 + 属性体写回。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SegyVolume:
    """3D 后叠加地震数据体的容器。

    cube.shape == (n_il, n_xl, n_samples)；对应 inline / crossline / time
    """
    cube: np.ndarray                    # 3D float32
    inlines: np.ndarray                 # 1D，每个 inline 的编号
    xlines: np.ndarray                  # 1D，每个 crossline 的编号
    sample_interval_ms: float           # 采样间隔（毫秒）
    n_samples: int
    source_path: str | None = None
    header: dict = field(default_factory=dict)

    @property
    def time_axis_ms(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.sample_interval_ms

    def to_meta(self) -> dict:
        return {
            "shape": list(self.cube.shape),
            "inlines": [int(self.inlines.min()), int(self.inlines.max())],
            "xlines":  [int(self.xlines.min()),  int(self.xlines.max())],
            "sample_interval_ms": self.sample_interval_ms,
            "n_samples": self.n_samples,
            "source_path": self.source_path,
        }


def read_segy(path: str, strict: bool = False) -> SegyVolume:
    """读取 SEG-Y 文件到 SegyVolume。strict=False 允许非标准头。"""
    import segyio

    with segyio.open(path, "r", strict=strict) as f:
        inlines = np.asarray(f.ilines, dtype=np.int32)
        xlines = np.asarray(f.xlines, dtype=np.int32)
        try:
            interval_us = int(f.bin[segyio.BinField.Interval])
        except Exception:
            interval_us = 4000
        n_samples = int(f.samples.size)
        cube = segyio.tools.cube(f).astype(np.float32)

    return SegyVolume(
        cube=cube, inlines=inlines, xlines=xlines,
        sample_interval_ms=interval_us / 1000.0,
        n_samples=n_samples,
        source_path=str(path),
    )


def write_attribute_segy(volume: SegyVolume, attribute: np.ndarray,
                          out_path: str, ref_segy: str | None = None) -> str:
    """把和 volume 同 shape 的属性体（如 fault probability）写成 SEG-Y。

    参数:
      ref_segy: 参考文件用于复制 headers。不提供时用 volume.source_path，
                都没有则用 default spec 从零构造。
    """
    import segyio

    if attribute.shape != volume.cube.shape:
        raise ValueError(
            f"attribute shape {attribute.shape} != volume {volume.cube.shape}")
    ref = ref_segy or volume.source_path
    attribute = attribute.astype(np.float32)

    if ref and Path(ref).exists():
        # 复制头文件结构，只替换 traces
        import shutil
        shutil.copyfile(ref, out_path)
        with segyio.open(out_path, "r+", strict=False) as f:
            for il_idx in range(attribute.shape[0]):
                f.iline[int(volume.inlines[il_idx])] = attribute[il_idx]
    else:
        # 从零构造（非标准工作流，仅用于合成数据/单测）
        spec = segyio.spec()
        spec.sorting = 2                # 2 = inline-sorted
        spec.format = 1                 # IBM float32
        spec.iline = 189
        spec.xline = 193
        spec.samples = np.arange(volume.n_samples) * volume.sample_interval_ms
        spec.ilines = list(volume.inlines)
        spec.xlines = list(volume.xlines)
        n_xl = attribute.shape[1]
        interval_us = int(volume.sample_interval_ms * 1000)
        with segyio.create(out_path, spec) as f:
            for il_idx in range(attribute.shape[0]):
                for xl_idx in range(n_xl):
                    trc = il_idx * n_xl + xl_idx
                    f.trace[trc] = attribute[il_idx, xl_idx]
                    f.header[trc] = {
                        segyio.TraceField.INLINE_3D: int(volume.inlines[il_idx]),
                        segyio.TraceField.CROSSLINE_3D: int(volume.xlines[xl_idx]),
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: interval_us,
                        segyio.TraceField.TRACE_SAMPLE_COUNT: volume.n_samples,
                    }
            f.bin[segyio.BinField.Interval] = interval_us
            f.bin[segyio.BinField.Samples] = volume.n_samples
    return out_path


def extract_inline_slice(vol: SegyVolume, il_idx: int) -> np.ndarray:
    """按 inline **数组下标**（不是 inline 号）取 (n_xl, n_samples) 切片。"""
    return vol.cube[il_idx, :, :].T  # (n_samples, n_xl)


def extract_xline_slice(vol: SegyVolume, xl_idx: int) -> np.ndarray:
    return vol.cube[:, xl_idx, :].T  # (n_samples, n_il)


def extract_time_slice(vol: SegyVolume, sample_idx: int) -> np.ndarray:
    return vol.cube[:, :, sample_idx]  # (n_il, n_xl)


def synthetic_volume(n_il: int = 30, n_xl: int = 60, n_samples: int = 200,
                      sample_interval_ms: float = 4.0,
                      seed: int = 0) -> SegyVolume:
    """生成一个小合成 SEG-Y volume（供单测/演示用）。"""
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((n_il, n_xl, n_samples)).astype(np.float32) * 0.1
    # 层位放在相对位置 (25%, 45%, 70%)，无论 n_samples 多小都不会越界
    for frac in (0.25, 0.45, 0.70):
        d = int(np.clip(frac * n_samples, 0, n_samples - 1))
        cube[:, :, d] += 1.0
    # 局部亮点 — 只在体足够大时插入
    if n_il > 15 and n_xl > 35 and n_samples > 65:
        cube[10:15, 25:35, 60:65] -= 2.0
    return SegyVolume(
        cube=cube,
        inlines=np.arange(100, 100 + n_il, dtype=np.int32),
        xlines=np.arange(200, 200 + n_xl, dtype=np.int32),
        sample_interval_ms=sample_interval_ms,
        n_samples=n_samples,
    )
