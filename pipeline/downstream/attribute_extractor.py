"""Seismic Attribute Extractor — 计算多种地震属性供 VLM 解释。

支持的属性类别：
  Instantaneous:  envelope, phase, frequency  (Hilbert 变换)
  Texture:        energy, contrast, homogeneity, correlation  (GLCM)
  Spectral:       spectral_10hz, spectral_20hz, ...  (带通滤波分频)
  Amplitude:      rms_amplitude, sweetness
"""
from __future__ import annotations

import numpy as np

from ._shared import image_to_array as _get_array


def _envelope(arr: np.ndarray) -> np.ndarray:
    """瞬时振幅包络 (Hilbert 变换的模)。"""
    from scipy.signal import hilbert
    analytic = hilbert(arr, axis=0)
    return np.abs(analytic).astype(np.float32)


def _instantaneous_phase(arr: np.ndarray) -> np.ndarray:
    """瞬时相位 [-π, π]。"""
    from scipy.signal import hilbert
    analytic = hilbert(arr, axis=0)
    return np.angle(analytic).astype(np.float32)


def _instantaneous_frequency(arr: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """瞬时频率 (Hz)，通过相位的时间导数估算。"""
    from scipy.signal import hilbert
    analytic = hilbert(arr, axis=0)
    phase = np.angle(analytic)
    # 沿时间轴(axis=0)差分 → 瞬时频率
    dphase = np.diff(np.unwrap(phase, axis=0), axis=0)
    # 补一行保持 shape
    dphase = np.vstack([dphase, dphase[-1:, :]])
    freq = dphase / (2 * np.pi * dt)
    return np.clip(np.abs(freq), 0, None).astype(np.float32)


def _rms_amplitude(arr: np.ndarray, win: int = 15) -> np.ndarray:
    """滑动窗口 RMS 振幅。"""
    from scipy.ndimage import uniform_filter1d
    squared = arr * arr
    rms = np.sqrt(uniform_filter1d(squared, size=win, axis=0, mode="reflect"))
    return rms.astype(np.float32)


def _sweetness(arr: np.ndarray, win: int = 15) -> np.ndarray:
    """甜点属性 = envelope / sqrt(avg_freq)，高值→储层。"""
    env = _envelope(arr)
    inst_freq = _instantaneous_frequency(arr) + 1e-8
    sweet = env / np.sqrt(inst_freq)
    return np.clip(sweet, 0, None).astype(np.float32)


def _glcm_attributes(arr: np.ndarray, levels: int = 16,
                     distances: tuple = (1, 2),
                     angles: tuple = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                     ) -> dict[str, np.ndarray]:
    """计算 GLCM 纹理属性：energy, contrast, homogeneity, correlation。

    对每个像素在其局部窗口内计算 GLCM，然后提取 Haralick 特征。
    返回 {attr_name: (H, W) float32 array}。
    """
    from skimage.feature import graycomatrix, graycoprops
    ny, nx = arr.shape
    # 量化到 [0, levels-1]
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        quant = np.zeros_like(arr, dtype=np.uint8)
    else:
        quant = ((arr - vmin) / (vmax - vmin) * (levels - 1)).astype(np.uint8)

    # 全局 GLCM（对整张图）
    glcm = graycomatrix(quant, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    props = {
        "energy": graycoprops(glcm, "energy").mean(),
        "contrast": graycoprops(glcm, "contrast").mean(),
        "homogeneity": graycoprops(glcm, "homogeneity").mean(),
        "correlation": graycoprops(glcm, "correlation").mean(),
    }
    # 返回标量统计（GLCM 是全图统计，不逐像素）
    return props


def _spectral_bands(arr: np.ndarray, dt: float = 1.0,
                    bands: list[tuple[float, float]] | None = None,
                    ) -> dict[str, np.ndarray]:
    """带通滤波分频：返回 {f"{low}-{high}hz": array}。

    bands: [(low, high), ...] Hz。默认分 3 个频带。
    """
    from scipy.signal import butter, filtfilt
    if bands is None:
        nyq = 0.5 / dt
        bands = [
            (0.0, nyq * 0.25),
            (nyq * 0.25, nyq * 0.5),
            (nyq * 0.5, nyq * 0.85),
        ]
    result: dict[str, np.ndarray] = {}
    nyq = 0.5 / dt
    for low, high in bands:
        key = f"{int(low)}-{int(high)}hz"
        if low <= 0:
            b, a = butter(4, high / nyq, btype="low")
        elif high >= nyq * 0.99:
            b, a = butter(4, low / nyq, btype="high")
        else:
            b, a = butter(4, [low / nyq, high / nyq], btype="band")
        result[key] = filtfilt(b, a, arr, axis=0).astype(np.float32)
    return result


# ── 下游模型 ────────────────────────────────────────────────────────────────

class AttributeExtractor:
    name = "attribute_extractor"
    description = (
        "地震属性提取。计算瞬时属性(包络/相位/频率)、GLCM纹理、谱分解、"
        "RMS振幅、甜点等，适合辅助沉积相分析、储层预测和裂缝检测"
    )
    required_fields = [
        "attributes (要计算的属性列表，可选值: envelope, phase, frequency, "
        "rms_amplitude, sweetness, glcm, spectral)",
        "spectral_bands? (谱分解频带: [[low,high], ...], Hz)",
        "regions_of_interest? (可选ROI: [{bbox_xyxy_norm:[x1,y1,x2,y2]},...])",
    ]
    output_shape = (
        "list[{id, attribute_name, statistics: {min, max, mean, std}, "
        "roi: {bbox_norm}, global_glcm?: {energy, contrast, ...}}]"
    )

    def detect(self, instruction: dict, image=None,
               context: dict | None = None) -> list[dict]:
        attr_names = instruction.get("attributes") or []
        if not attr_names:
            return []

        arr = _get_array(image, context)
        if arr is None:
            return []

        spec_bands = instruction.get("spectral_bands")
        rois_raw = instruction.get("regions_of_interest") or []
        ny, nx = arr.shape

        results: list[dict] = []
        n = 0

        def _add_roi(roi_idx: int | None, attr_name: str,
                     attr_map: np.ndarray, sub_arr: np.ndarray | None = None):
            nonlocal n
            if attr_map.ndim != 2:
                return
            mask = ~np.isnan(attr_map) & ~np.isinf(attr_map)
            if not mask.any():
                return
            vals = attr_map[mask]
            entry: dict = {
                "id": f"attr_{attr_name}_{n}",
                "attribute_name": attr_name,
                "statistics": {
                    "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4),
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                },
                "model": self.name,
            }
            if roi_idx is not None:
                entry["roi_index"] = roi_idx
            results.append(entry)
            n += 1

        for attr in attr_names:
            # ── ROI 处理 ──
            if rois_raw:
                for ri, roi in enumerate(rois_raw):
                    bn = roi.get("bbox_norm") or roi.get("bbox_xyxy_norm")
                    if not bn or len(bn) != 4:
                        continue
                    x1 = int(np.clip(bn[0] * nx, 0, nx - 1))
                    y1 = int(np.clip(bn[1] * ny, 0, ny - 1))
                    x2 = int(np.clip(bn[2] * nx, 0, nx - 1))
                    y2 = int(np.clip(bn[3] * ny, 0, ny - 1))
                    sub = arr[y1:y2 + 1, x1:x2 + 1]
                    for a_name, a_map in _compute_attrib(sub, attr, spec_bands):
                        _add_roi(ri, a_name, a_map)
            else:
                # 全图
                for a_name, a_map in _compute_attrib(arr, attr, spec_bands):
                    _add_roi(None, a_name, a_map)

        return results


def _compute_attrib(arr: np.ndarray, attr_name: str,
                    spec_bands) -> list[tuple[str, np.ndarray]]:
    """根据属性名计算一张或多张属性图。返回 [(sub_attr_name, map)]。"""
    out: list[tuple[str, np.ndarray]] = []
    try:
        if attr_name == "envelope":
            out.append(("envelope", _envelope(arr)))
        elif attr_name == "phase":
            out.append(("phase", _instantaneous_phase(arr)))
        elif attr_name == "frequency":
            out.append(("frequency", _instantaneous_frequency(arr)))
        elif attr_name == "rms_amplitude":
            out.append(("rms_amplitude", _rms_amplitude(arr)))
        elif attr_name == "sweetness":
            out.append(("sweetness", _sweetness(arr)))
        elif attr_name == "glcm":
            glcm = _glcm_attributes(arr)
            # GLCM 返回标量统计，生成同名属性占位图供 statistics 收集
            for gk, gv in glcm.items():
                out.append((f"glcm_{gk}",
                           np.full_like(arr, float(gv), dtype=np.float32)))
        elif attr_name == "spectral":
            bands = _spectral_bands(arr, bands=spec_bands)
            for bn, bm in bands.items():
                out.append((f"spectral_{bn}", bm))
    except Exception:
        pass
    return out
