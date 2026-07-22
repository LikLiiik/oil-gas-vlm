"""CIG-Bench 预训练模型 — Fault + Channel 检测。

CIG-Bench (douyimin/CIG-bench, pip install cig-bench):
  在合成+真实地震数据上训练的 HRNet 模型, 权重自动从 ModelScope 下载。
  - FaultPredictor: 3D 断层分割 (~40MB)
  - ChannelPredictor: 3D 河道分割 (~40MB)

首次调用自动下载权重，后续复用。
"""
from __future__ import annotations

import importlib.util

import numpy as np


class CigFaultDetector:
    name = "cig_fault"
    description = (
        "CIG-Bench FaultPredictor (HRNet)。在合成+真实地震数据上训练，"
        "3D断层概率体预测。适合高精度断层检测，自动下载权重(~40MB)"
    )
    required_fields = [
        "threshold? (概率阈值 0-1, 默认0.5)",
        "scale? (缩放系数, 默认1.0)",
    ]
    output_shape = (
        "list[{id, class_name, bbox_pixel, confidence, "
        "fault_prob_volume_shape}]"
    )
    TASK_NAME = "fault"

    def __init__(self):
        self._predictor = None
        self._available = None

    def runtime_status(self) -> tuple[bool, str]:
        if importlib.util.find_spec("cig_bench") is None:
            return False, "cig-bench is not installed"
        if importlib.util.find_spec("torch") is None:
            return False, "torch is not installed"
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA GPU is not available"
        return True, "ready"

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            import torch
            if not torch.cuda.is_available():
                print("[cig_fault] GPU not available, skip")
                self._available = False
                return False
            from cig_bench.predictor.fault import FaultPredictor
            print("[cig_fault] loading FaultPredictor (auto-download "
                  "weights from ModelScope) ...")
            self._predictor = FaultPredictor(device="cuda")
            self._available = True
            print("[cig_fault] loaded")
        except Exception as e:
            print(f"[cig_fault] unavailable: {e}")
            self._available = False
        return self._available

    def detect(self, instruction, image=None, context=None):
        if not self._load():
            return []

        arr = _get_volume(context)
        if arr is None:
            return []

        thr = float(instruction.get("threshold", 0.5))
        scale = float(instruction.get("scale", 1.0))

        # CIG-Bench FaultPredictor expects 3D volume (t, h, w)
        try:
            prob_vol, _ = self._predictor.predict(
                arr.astype(np.float32),
                threshold=thr,
                scale=scale,
            )
        except Exception as e:
            print(f"[cig_fault] predict failed: {e}")
            return []

        return self._volume_to_detections(prob_vol, thr)

    @staticmethod
    def _volume_to_detections(prob_vol: np.ndarray, thr: float,
                               class_name: str = "fault",
                               model_name: str = "cig_fault",
                               prefix: str = "cigfault",
                               ) -> list[dict]:
        """3D 概率体 → 2D bbox 检测列表 (逐切片连通域)。"""
        from scipy import ndimage
        results = []
        n_slices = prob_vol.shape[0]
        for i in range(n_slices):
            slc = prob_vol[i]
            if slc.max() < thr:
                continue
            binary = (slc >= thr).astype(np.uint8)
            labeled, n_regions = ndimage.label(binary)
            for rid in range(1, n_regions + 1):
                ys, xs = np.where(labeled == rid)
                if len(ys) < 20:
                    continue
                conf = float(slc[ys, xs].mean())
                results.append({
                    "id": f"{prefix}_{i}_{rid}",
                    "class_name": class_name,
                    "bbox_pixel": [
                        float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max()),
                    ],
                    "confidence": round(conf, 3),
                    "slice_index": i,
                    "area_pixels": int(len(ys)),
                    "model": model_name,
                })
        return results


class CigChannelDetector:
    name = "cig_channel"
    description = (
        "CIG-Bench ChannelPredictor (HRNet)。多尺度集成预测，"
        "3D河道/沉积体分割。适合河道、浊积水道检测，自动下载权重(~40MB)"
    )
    required_fields = [
        "threshold? (分数阈值, 默认2.0, sum模式下)",
        "scales? (多尺度列表, 默认[0.5,1.0])",
    ]
    output_shape = (
        "list[{id, class_name, bbox_pixel, confidence, "
        "channel_score_volume_shape}]"
    )
    TASK_NAME = "channel"

    def __init__(self):
        self._predictor = None
        self._available = None

    def runtime_status(self) -> tuple[bool, str]:
        if importlib.util.find_spec("cig_bench") is None:
            return False, "cig-bench is not installed"
        if importlib.util.find_spec("torch") is None:
            return False, "torch is not installed"
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA GPU is not available"
        return True, "ready"

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            import torch
            if not torch.cuda.is_available():
                print("[cig_channel] GPU not available, skip")
                self._available = False
                return False
            from cig_bench.predictor.channel import ChannelPredictor
            print("[cig_channel] loading ChannelPredictor (auto-download "
                  "weights from ModelScope) ...")
            self._predictor = ChannelPredictor(device="cuda")
            self._available = True
            print("[cig_channel] loaded")
        except Exception as e:
            print(f"[cig_channel] unavailable: {e}")
            self._available = False
        return self._available

    def detect(self, instruction, image=None, context=None):
        if not self._load():
            return []

        arr = _get_volume(context)
        if arr is None:
            return []

        scales_raw = instruction.get("scales", [0.5, 1.0])
        if isinstance(scales_raw, (int, float)):
            scales_raw = [float(scales_raw)]
        elif isinstance(scales_raw, str):
            try:
                scales_raw = [float(s.strip()) for s in scales_raw.split(",")
                              if s.strip()]
                if not scales_raw:
                    scales_raw = [1.0]
            except ValueError:
                scales_raw = [1.0]
        if not isinstance(scales_raw, list):
            scales_raw = [1.0]
        accumulate = instruction.get("accumulate", "sum")

        try:
            scores, _ = self._predictor.predict(
                arr.astype(np.float32),
                scales=scales_raw,
                accumulate=accumulate,
            )
        except Exception as e:
            print(f"[cig_channel] predict failed: {e}")
            return []

        thr = float(instruction.get("threshold", 2.0 if accumulate == "sum" else 0.5))
        return CigFaultDetector._volume_to_detections(
            scores, thr,
            class_name="channel",
            model_name="cig_channel",
            prefix="cigchannel",
        )


# ── 共享工具 ────────────────────────────────────────────────────────────────
def _get_volume(context: dict | None) -> np.ndarray | None:
    """从 context 获取 3D 地震数组。"""
    if context is None:
        return None
    arr = context.get("array")
    if arr is None:
        arr = context.get("volume")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        # 单切片 → 扩展为 3D (1, h, w)
        return arr[np.newaxis, ...]
    return None
