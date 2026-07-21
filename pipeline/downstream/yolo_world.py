"""YOLO-World 开放词汇检测。真实模型优先，失败自动 mock。"""
from __future__ import annotations

import os

import numpy as np

WEIGHTS_ENV = "YOLO_WORLD_PATH"
DEFAULT_WEIGHTS = "/data/yxjiang/oil-gas-llm/weights/yolov8s-world.pt"


def _bbox_overlaps(a: list[float], b: list[float]) -> bool:
    """两个 [x1,y1,x2,y2] bbox 有交集？"""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


class YoloWorld:
    name = "yolo_world"
    description = "开放词汇目标检测。适合断层/亮点/河道/盐丘等离散地质目标"
    required_fields = ["categories[].class_name",
                       "categories[].expected_cdp_range",
                       "categories[].expected_time_range_ms",
                       "categories[].confidence_threshold"]
    output_shape = "list[{id, class_name, bbox_pixel:[x1,y1,x2,y2], confidence, coordinate_system}]"

    def __init__(self, weights_path: str | None = None, allow_mock: bool = True):
        self.weights_path = weights_path or os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
        self.allow_mock = allow_mock
        self._model = None
        self._status = "unloaded"  # ready | unavailable | unloaded

    def _load(self):
        if self._status == "ready":
            return self._model
        if self._status == "unavailable":
            return None
        try:
            from ultralytics import YOLOWorld
            import torch
            ckpt = self.weights_path if os.path.exists(self.weights_path) else "yolov8s-world.pt"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = YOLOWorld(ckpt)
            self._model.to(device)
            self._status = "ready"
            print(f"  [yolo_world loaded from {ckpt} on {device}]")
            return self._model
        except Exception as e:
            self._status = "unavailable"
            print(f"  [yolo_world unavailable, mock={self.allow_mock}: {e}]")
            return None

    def detect(self, instruction, image=None, context=None):
        m = self._load()
        if m is None or image is None:
            return self._mock(instruction) if self.allow_mock else []
        cats = instruction.get("categories", [instruction])
        class_names = [c.get("class_name") for c in cats if c.get("class_name")]
        if not class_names:
            return []
        conf_thr = min(
            (c.get("confidence_threshold", 0.25) for c in cats), default=0.25,
        )
        m.set_classes(class_names)
        results = m.predict(image, conf=conf_thr, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        out = []
        for i in range(len(clses)):
            cls_idx = int(clses[i])
            cname = class_names[cls_idx] if cls_idx < len(class_names) else f"cls{cls_idx}"
            out.append({
                "id": f"yolo_{cname[:8].replace(' ', '_')}_{i}",
                "class_name": cname,
                "bbox_pixel": [float(v) for v in xyxy[i]],
                "confidence": round(float(confs[i]), 3),
                "model": self.name,
                "coordinate_system": "pixel",
            })
        return out

    def detect_open_vocab(self, image, class_prompts: list[str],
                           conf_threshold: float = 0.25,
                           roi_norm: list[float] | None = None) -> list[dict]:
        """给一批 class_prompts + 一张图 + 可选 ROI，直接跑开放词汇检测。

        roi_norm: 归一化 [x1,y1,x2,y2]，仅保留与之相交的检测。
        返回 [{class_name, bbox_pixel, bbox_norm, confidence, in_roi, model}]。
        """
        m = self._load()
        if m is None or image is None or not class_prompts:
            return []
        try:
            m.set_classes(class_prompts)
            results = m.predict(image, conf=conf_threshold, verbose=False)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            W, H = image.size
            out = []
            for i in range(len(clses)):
                cls_idx = int(clses[i])
                cname = (class_prompts[cls_idx]
                         if cls_idx < len(class_prompts) else f"cls{cls_idx}")
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                bbox_norm = [x1 / W, y1 / H, x2 / W, y2 / H]
                in_roi = True
                if roi_norm is not None:
                    in_roi = _bbox_overlaps(bbox_norm, roi_norm)
                out.append({
                    "id": f"yolo_{cname[:8].replace(' ', '_')}_{i}",
                    "class_name": cname,
                    "bbox_pixel": [x1, y1, x2, y2],
                    "bbox_norm": bbox_norm,
                    "confidence": round(float(confs[i]), 3),
                    "in_roi": bool(in_roi),
                    "model": self.name,
                })
            return out
        except RuntimeError as e:
            print(f"  [yolo_world detect_open_vocab failed: {e}]")
            return []

    def _mock(self, instruction):
        results = []
        for cat in instruction.get("categories", [instruction]):
            cdp = cat.get("expected_cdp_range", [0, 300])
            t = cat.get("expected_time_range_ms", [0, 2500])
            conf = np.random.uniform(0.4, 0.9)
            results.append({
                "id": f"yolo_{str(cat.get('class_name','det'))[:8]}_{np.random.randint(100)}",
                "class_name": cat.get("class_name"),
                "bbox": [cdp[0] + 5, cdp[1] - 5, t[0] + 30, t[1] - 30],
                "confidence": round(conf, 2),
                "model": self.name + "_mock",
            })
        return results
