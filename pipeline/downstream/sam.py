"""SAM — Segment Anything Model (Meta).

真实开源模型，通过 transformers 加载官方权重。
首次调用自动下载 vit-h SAM 权重 (~2.4GB)，后续调用复用。
GPU 不可用时自动退化为轻量级 Otsu 阈值分割。
"""
from __future__ import annotations

import numpy as np


class Sam:
    name = "sam"
    description = (
        "SAM (Meta Segment Anything Model)。bbox/point引导的像素级分割。"
        "适合层位/异常体/盐体的精确mask提取"
    )
    required_fields = [
        "prompt_type: point|bbox",
        "prompt_value: [x1,y1,x2,y2] for bbox, [x,y] for point",
        "label: 分割目标名称",
    ]
    output_shape = (
        "list[{id, label, mask_area_pixels, bbox_pixel:[x1,y1,x2,y2], "
        "centroid:[cx,cy]}]"
    )

    def __init__(self):
        self._model = None
        self._processor = None
        self._available = None  # None=未检测, True/False

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            import torch
            if not torch.cuda.is_available():
                self._available = False
                return False
            from transformers import SamModel, SamProcessor
            print("[sam] loading SAM vit-h from Meta (via transformers) ...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = SamModel.from_pretrained(
                "facebook/sam-vit-huge", torch_dtype=torch.float16,
            ).to(device)
            self._processor = SamProcessor.from_pretrained(
                "facebook/sam-vit-huge",
            )
            self._available = True
            print("[sam] loaded on GPU")
        except Exception as e:
            print(f"[sam] SAM unavailable, fallback to lightweight: {e}")
            self._available = False
        return self._available

    def detect(self, instruction, image=None, context=None):
        prompt_type = instruction.get("prompt_type", "bbox")
        prompt_value = instruction.get("prompt_value")
        label = instruction.get("label", "segment")

        if image is None or prompt_value is None:
            return self._empty(label)

        # 判断图像类型：seismic剖面(waveform)还是测井图/自然图
        use_real_sam = self._should_use_real_sam(image)

        if use_real_sam and self._load():
            result = self._detect_sam(instruction, image, label,
                                      prompt_type, prompt_value)
            # SAM可能返回极小的mask(模型不理解seismic) → 退化为轻量
            if result[0].get("mask_area_pixels", 0) < 10:
                return self._detect_lightweight(image, label, prompt_type,
                                                prompt_value, instruction)
            return result
        # 退化到轻量实现
        return self._detect_lightweight(image, label, prompt_type,
                                        prompt_value, instruction)

    @staticmethod
    def _should_use_real_sam(image) -> bool:
        """判断图像是否适合用真实 SAM。

        SAM 在自然图像上训练，对 seismic 波形剖面效果差。
        通过梯度密度判断：高梯度密度→图表/自然图→真实SAM；
        低梯度密度→seismic剖面→轻量实现。
        """
        import numpy as np
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        # 梯度密度：水平+垂直梯度的非零比例
        gx = np.abs(np.diff(gray, axis=1))
        gy = np.abs(np.diff(gray, axis=0))
        grad_density = (
            (gx > 15).mean() + (gy[:, :gx.shape[1]] > 15).mean()
        ) / 2
        return grad_density > 0.02

    # ── SAM 真实推理 ───────────────────────────────────────────────

    def _detect_sam(self, instruction, image, label, prompt_type,
                    prompt_value):
        import torch
        import numpy as np

        try:
            inputs = self._processor(image, return_tensors="pt").to(
                self._model.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(
                self._model.dtype)
        except Exception:
            inputs = self._processor(image, return_tensors="pt")

        if prompt_type == "bbox":
            x1, y1, x2, y2 = [int(v) for v in prompt_value[:4]]
            input_boxes = [[[x1, y1, x2, y2]]]
        elif prompt_type == "point":
            px, py = int(prompt_value[0]), int(prompt_value[1])
            input_points = [[[[px, py]]]]   # (1, 1, 1, 2)
            input_labels = [[[1]]]           # (1, 1, 1)
        else:
            return self._empty(label)

        with torch.no_grad():
            image_embeddings = self._model.get_image_embeddings(
                inputs["pixel_values"])
            dtype = image_embeddings.dtype
            device = image_embeddings.device
            if prompt_type == "bbox":
                inputs["input_boxes"] = torch.tensor(
                    input_boxes, dtype=dtype, device=device)
                outputs = self._model(
                    image_embeddings=image_embeddings,
                    input_boxes=inputs["input_boxes"],
                    multimask_output=False,
                )
            elif prompt_type == "point":
                inputs["input_points"] = torch.tensor(
                    input_points, dtype=dtype, device=device)
                inputs["input_labels"] = torch.tensor(
                    input_labels, dtype=torch.long, device=device)
                outputs = self._model(
                    image_embeddings=image_embeddings,
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    multimask_output=False,
                )

        try:
            masks = self._processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            mask = masks[0][0, 0].numpy() > 0.0
        except (AttributeError, KeyError, IndexError):
            mask = outputs.pred_masks[0, 0, 0].cpu().numpy() > 0.0
        return [self._mask_stats(mask, label)]

    # ── 轻量退化 ──────────────────────────────────────────────────

    def _detect_lightweight(self, image, label, prompt_type,
                            prompt_value, instruction):
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        if prompt_type == "bbox":
            if len(prompt_value) != 4:
                return self._empty(label)
            x1, y1, x2, y2 = [int(v) for v in prompt_value[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
            roi = gray[y1:y2 + 1, x1:x2 + 1]
            if roi.size == 0 or roi.std() < 1e-4:
                mask = np.zeros(gray.shape, dtype=bool)
                mask[y1:y2 + 1, x1:x2 + 1] = True
            else:
                thr = roi.mean() + roi.std() * 0.5
                roi_bin = roi > thr
                mask = np.zeros(gray.shape, dtype=bool)
                mask[y1:y2 + 1, x1:x2 + 1] = roi_bin
            return [self._mask_stats(mask, label)]

        elif prompt_type == "point":
            px, py = int(prompt_value[0]), int(prompt_value[1])
            tolerance = float(instruction.get("tolerance", 0.15))
            if px < 0 or px >= gray.shape[1] or py < 0 or py >= gray.shape[0]:
                return self._empty(label)
            seed = gray[py, px]
            lo, hi = seed - tolerance, seed + tolerance
            mask = np.zeros(gray.shape, dtype=bool)
            from collections import deque
            q = deque([(px, py)])
            visited = set()
            while q:
                x, y = q.popleft()
                if (x, y) in visited:
                    continue
                if x < 0 or x >= gray.shape[1] or y < 0 or y >= gray.shape[0]:
                    continue
                if not (lo <= gray[y, x] <= hi):
                    continue
                visited.add((x, y))
                mask[y, x] = True
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q.append((x + dx, y + dy))
            return [self._mask_stats(mask, label)]

        return self._empty(label)

    @staticmethod
    def _mask_stats(mask, label):
        mask = np.asarray(mask, dtype=bool)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return {"id": f"sam_{label}_empty", "label": label,
                    "mask_area_pixels": 0, "model": "sam",
                    "coordinate_space": "image",
                    "coordinate_shape": list(mask.shape),
                    "_mask_array": mask}
        return {
            "id": f"sam_{label}_{np.random.randint(10000)}",
            "label": label,
            "mask_area_pixels": int(len(ys)),
            "bbox_pixel": [float(xs.min()), float(ys.min()),
                           float(xs.max()), float(ys.max())],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "model": "sam",
            "coordinate_space": "image",
            "coordinate_shape": list(mask.shape),
            "_mask_array": mask,
        }

    @staticmethod
    def _empty(label):
        return [{"id": f"sam_{label}_noinput", "label": label,
                 "mask_area_pixels": 0, "model": "sam"}]
