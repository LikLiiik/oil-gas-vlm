"""DINOv2 Feature Extractor — Meta 自监督视觉模型。

通过 torch.hub 加载官方预训练权重 (ViT-S/14, 21M params, ~85MB)。
无需地震数据微调即可提取有区分力的地质纹理特征，
用于沉积相分类、异常检测等下游任务。
"""
from __future__ import annotations

import numpy as np


class DINOv2Extractor:
    name = "dinov2_extractor"
    description = (
        "DINOv2 (Meta) 自监督视觉特征提取。在1.42亿张图像上预训练，"
        "对地质纹理/构型有天然区分力，适合沉积相特征提取和异常检测"
    )
    required_fields = [
        "regions_of_interest? (可选ROI列表)",
        "patch_size? (特征图块大小，默认14)",
        "output_type? (features|embedding，默认features)",
    ]
    output_shape = (
        "list[{id, feature_map_shape, roi?, embedding_vector?:[]}]"
    )

    def __init__(self):
        self._model = None
        self._available = None

    def _load(self):
        if self._available is not None:
            return self._available
        try:
            import torch
            print("[dinov2] loading ViT-S/14 from Meta (via torch.hub) ...")
            self._model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14",
                pretrained=True, trust_repo=True,
            )
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._model.eval()
            self._available = True
            print("[dinov2] loaded")
        except Exception as e:
            print(f"[dinov2] unavailable: {e}")
            self._available = False
        return self._available

    def detect(self, instruction, image=None, context=None):
        if image is None:
            return []

        if not self._load():
            return [{"id": "dinov2_unavailable",
                     "result": "DINOv2 model failed to load",
                     "model": self.name}]

        import torch
        from torchvision import transforms

        # 预处理
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        img_tensor = transform(image).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        output_type = instruction.get("output_type", "features")

        with torch.no_grad():
            if output_type == "embedding":
                # 全图 CLS token embedding (384-d)
                out = self._model(img_tensor)
                emb = out.cpu().numpy()[0]
                return [{
                    "id": "dinov2_embed",
                    "embedding_dim": int(emb.shape[0]),
                    "embedding_vector": emb.tolist(),
                    "model": self.name,
                }]
            else:
                # 特征图: (1, 384, 37, 37) for 518×518 input
                out = self._model.forward_features(img_tensor)
                feats = out["x_norm_patchtokens"].cpu().numpy()
                h = w = int(np.sqrt(feats.shape[1]))
                feat_map = feats[0].reshape(h, w, -1)
                return [{
                    "id": "dinov2_features",
                    "feature_map_shape": list(feat_map.shape),
                    "feature_dim": int(feat_map.shape[-1]),
                    "model": self.name,
                }]
